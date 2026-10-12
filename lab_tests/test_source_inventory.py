from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from video_lab.database import Database
from video_lab.feasibility import FeasibilityService, StageDefinition, StageExecutionContext
from video_lab.feasibility_models import (
    FeasibilityEvidenceError,
    FeasibilityStage,
    StageOutcome,
    StageResult,
)
from video_lab.feasibility_repository import FeasibilityRepository
from video_lab.source_inventory import (
    ANIMEGEN_REQUIRED_FILES,
    ANIMEGEN_SOURCE_REPOS,
    ResolvedSource,
    SourceFile,
    SourceInventory,
    SourceInventoryClient,
    SourceInventoryError,
    SourceSpec,
    canonical_source_digest,
    inventory_from_evidence,
)


def test_canonical_source_digest_binds_repo_revision_and_sorted_verified_files() -> None:
    source = ResolvedSource(
        "animegen",
        "aidealab/AnimeGen-T2V",
        "a" * 40,
        "a" * 40,
        "apache-2.0",
        (
            SourceFile("z.bin", 2, upstream_sha256="2" * 64, local_sha256="2" * 64),
            SourceFile("a.json", 1, upstream_sha256="1" * 64, local_sha256="1" * 64),
        ),
    )

    digest = canonical_source_digest(source)
    reordered = source.__class__(
        source.component,
        source.repo_id,
        source.requested_revision,
        source.revision,
        source.license,
        tuple(reversed(source.files)),
    )

    assert len(digest) == 64
    assert canonical_source_digest(reordered) == digest
    assert (
        canonical_source_digest(
            source.__class__(
                source.component,
                source.repo_id,
                source.requested_revision,
                "b" * 40,
                source.license,
                source.files,
            )
        )
        != digest
    )


def test_canonical_source_digest_requires_verified_checksum() -> None:
    source = ResolvedSource(
        "animegen",
        "repo/model",
        "a" * 40,
        "a" * 40,
        "apache-2.0",
        (SourceFile("weights.bin", 1),),
    )

    with pytest.raises(SourceInventoryError, match="verified checksum"):
        canonical_source_digest(source)


def test_inventory_evidence_round_trips_without_operational_paths(tmp_path: Path) -> None:
    revision = "a" * 40
    spec = SourceSpec("animegen", "aidealab/AnimeGen-T2V", revision, ("config.json",))
    contents = {"aidealab/AnimeGen-T2V": {"config.json": b"{}"}}
    model_info, download = make_api(contents, revisions={"aidealab/AnimeGen-T2V": revision})
    client = SourceInventoryClient(tmp_path / "private-cache", model_info, download)
    inventory = client.resolve((spec,))

    evidence = inventory.export_evidence()
    restored = inventory_from_evidence(evidence, client)

    assert restored.export_evidence() == evidence
    serialized = json.dumps(evidence)
    assert "private-cache" not in serialized
    assert str(tmp_path) not in serialized
    assert (
        restored.sources[0].files[0].upstream_sha256
        == inventory.sources[0].files[0].upstream_sha256
    )


def test_verify_cached_rehashes_frozen_snapshot_without_metadata_or_download(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    cache = tmp_path / "cache"
    target = cache / "snapshot"
    target.mkdir(parents=True)
    (target / "config.json").write_bytes(b"{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    calls: list[dict[str, object]] = []

    def cached_snapshot(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(target)

    client = SourceInventoryClient(
        cache,
        lambda **_: pytest.fail("cached verification must not query metadata"),
        cached_snapshot,
    )
    inventory = SourceInventory(
        (
            ResolvedSource(
                "animegen",
                "repo/model",
                revision,
                revision,
                "apache-2.0",
                (SourceFile("config.json", 2, local_sha256=digest),),
            ),
        ),
        2,
        client,
    )

    verified = client.verify_cached(inventory)

    assert verified.sources[0].files[0].local_sha256 == digest
    assert calls == [
        {
            "repo_id": "repo/model",
            "revision": revision,
            "cache_dir": str(cache),
            "allow_patterns": ("config.json",),
            "local_files_only": True,
        }
    ]


def test_refresh_cached_adds_newly_required_file_from_frozen_snapshot(tmp_path: Path) -> None:
    revision = "a" * 40
    cache = tmp_path / "cache"
    target = cache / "snapshot"
    target.mkdir(parents=True)
    (target / "model_index.json").write_bytes(b"{}")
    (target / "transformer").mkdir()
    (target / "transformer/config.json").write_bytes(b'{"layers": 40}')
    client = SourceInventoryClient(
        cache,
        lambda **_: pytest.fail("refresh must not query metadata"),
        lambda **_: str(target),
    )
    inventory = SourceInventory(
        (
            ResolvedSource(
                "wan-base",
                "repo/model",
                revision,
                revision,
                "apache-2.0",
                (
                    SourceFile(
                        "model_index.json",
                        2,
                        local_sha256=hashlib.sha256(b"{}").hexdigest(),
                    ),
                ),
            ),
        ),
        2,
        client,
    )
    spec = SourceSpec(
        "wan-base",
        "repo/model",
        revision,
        ("model_index.json", "transformer/config.json"),
    )

    refreshed = client.refresh_cached(inventory, (spec,))

    assert refreshed.expected_bytes == 16
    assert [item.relative_path for item in refreshed.sources[0].files] == [
        "model_index.json",
        "transformer/config.json",
    ]
    assert (
        refreshed.sources[0].files[1].local_sha256 == hashlib.sha256(b'{"layers": 40}').hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["sources"][0].update(revision="main"),
        lambda value: value["sources"][0]["files"][0].update(relative_path="../escape"),
        lambda value: value["sources"][0]["files"][0].update(upstream_sha256="bad"),
    ],
)
def test_inventory_evidence_reconstruction_rejects_mutable_or_unsafe_data(
    tmp_path: Path, mutation: object
) -> None:
    client = SourceInventoryClient(tmp_path, lambda **_: object(), lambda **_: "")
    evidence = {
        "expected_bytes": 2,
        "sources": [
            {
                "component": "animegen",
                "repo_id": "aidealab/AnimeGen-T2V",
                "requested_revision": "a" * 40,
                "revision": "a" * 40,
                "license": "apache-2.0",
                "files": [
                    {
                        "relative_path": "config.json",
                        "size": 2,
                        "upstream_sha256": "1" * 64,
                    }
                ],
            }
        ],
    }
    mutation(evidence)  # type: ignore[operator]

    with pytest.raises(SourceInventoryError):
        inventory_from_evidence(evidence, client)


def test_preflight_creates_fresh_cache_but_rejects_file_and_symlink_roots(
    tmp_path: Path,
) -> None:
    inventory = SourceInventory(
        sources=(),
        expected_bytes=0,
        _client=SourceInventoryClient(tmp_path / "fresh", lambda **_: object(), lambda **_: ""),
    )
    inventory.preflight(lambda path: SimpleNamespace(free=100 * 1024**3))
    assert (tmp_path / "fresh").is_dir()

    cache_file = tmp_path / "file-cache"
    cache_file.write_text("not a directory")
    file_inventory = SourceInventory(
        (), 0, SourceInventoryClient(cache_file, lambda **_: object(), lambda **_: "")
    )
    with pytest.raises(SourceInventoryError, match="directory"):
        file_inventory.preflight()

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked-cache"
    symlink.symlink_to(target, target_is_directory=True)
    linked_inventory = SourceInventory(
        (), 0, SourceInventoryClient(symlink, lambda **_: object(), lambda **_: "")
    )
    with pytest.raises(SourceInventoryError, match="symlink"):
        linked_inventory.preflight()


def test_canonical_animegen_source_manifest_covers_all_assembly_inputs() -> None:
    assert ANIMEGEN_REQUIRED_FILES == {
        "animegen": ("high_noise.safetensors", "low_noise.safetensors", "LICENSE"),
        "wan-base": (
            "model_index.json",
            "transformer/config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/spiece.model",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00002-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "text_encoder/model.safetensors.index.json",
        ),
        "lightning-high": ("Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",),
        "lightning-low": ("Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",),
    }
    assert ANIMEGEN_SOURCE_REPOS["lightning-high"] == "lightx2v/Wan2.2-Lightning"
    assert ANIMEGEN_SOURCE_REPOS["lightning-low"] == "lightx2v/Wan2.2-Lightning"


COMMIT = "a" * 40


def sibling(name: str, content: bytes, *, sha256: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        rfilename=name,
        size=len(content),
        lfs=SimpleNamespace(sha256=sha256) if sha256 else None,
    )


def info(
    repo_id: str,
    files: list[SimpleNamespace],
    *,
    sha: str = COMMIT,
    license_name: str | None = "apache-2.0",
) -> SimpleNamespace:
    card_data = {} if license_name is None else {"license": license_name}
    return SimpleNamespace(id=repo_id, sha=sha, siblings=files, card_data=card_data)


@pytest.fixture
def source_contents() -> dict[str, dict[str, bytes]]:
    return {
        "aidealab/AnimeGen-T2V": {
            "config.json": b"anime-config",
            "high_noise.safetensors": b"anime-high-weights",
            "low_noise.safetensors": b"anime-low-weights",
            "LICENSE": b"apache",
        },
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {
            "model_index.json": b"wan-index",
            "transformer/config.json": b"transformer-config",
            "vae/config.json": b"vae-config",
            "vae/diffusion_pytorch_model.safetensors": b"vae",
            "scheduler/scheduler_config.json": b"scheduler",
            "tokenizer/tokenizer_config.json": b"tokenizer",
            "tokenizer/spiece.model": b"vocabulary",
            "text_encoder/config.json": b"text-encoder",
            "text_encoder/model-00001-of-00003.safetensors": b"text-encoder-weights-1",
            "text_encoder/model-00002-of-00003.safetensors": b"text-encoder-weights-2",
            "text_encoder/model-00003-of-00003.safetensors": b"text-encoder-weights-3",
            "text_encoder/model.safetensors.index.json": b"text-encoder-index",
        },
        "lightx2v/Wan2.2-Lightning": {
            "Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors": b"high",
            "Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors": b"low",
        },
    }


@pytest.fixture
def specs() -> tuple[SourceSpec, ...]:
    return (
        SourceSpec(
            component="animegen",
            repo_id=ANIMEGEN_SOURCE_REPOS["animegen"],
            revision="anime-v1",
            required_files=ANIMEGEN_REQUIRED_FILES["animegen"],
        ),
        SourceSpec(
            component="wan-base",
            repo_id=ANIMEGEN_SOURCE_REPOS["wan-base"],
            revision="wan-v1",
            required_files=(*ANIMEGEN_REQUIRED_FILES["wan-base"],),
        ),
        SourceSpec(
            component="lightning-high",
            repo_id=ANIMEGEN_SOURCE_REPOS["lightning-high"],
            revision="high-v1",
            required_files=ANIMEGEN_REQUIRED_FILES["lightning-high"],
        ),
        SourceSpec(
            component="lightning-low",
            repo_id=ANIMEGEN_SOURCE_REPOS["lightning-low"],
            revision="low-v1",
            required_files=ANIMEGEN_REQUIRED_FILES["lightning-low"],
        ),
    )


def make_api(contents: dict[str, dict[str, bytes]], *, revisions: dict[str, str] | None = None):
    calls: list[tuple[str, str, bool]] = []

    def model_info(repo_id: str, revision: str, *, files_metadata: bool) -> Any:
        calls.append((repo_id, revision, files_metadata))
        files = [
            sibling(name, content, sha256=hashlib.sha256(content).hexdigest())
            for name, content in contents[repo_id].items()
        ]
        return info(repo_id, files, sha=(revisions or {}).get(repo_id, COMMIT))

    return model_info, calls


def test_resolves_all_exact_sources_without_loading_weights(
    tmp_path: Path, specs: tuple[SourceSpec, ...], source_contents: dict[str, dict[str, bytes]]
) -> None:
    model_info, calls = make_api(source_contents)
    downloader = pytest.fail
    client = SourceInventoryClient(tmp_path / "private-cache", model_info, downloader)

    inventory = client.resolve(specs)

    assert [source.component for source in inventory.sources] == [spec.component for spec in specs]
    assert {source.revision for source in inventory.sources} == {COMMIT}
    assert all(source.license == "apache-2.0" for source in inventory.sources)
    assert inventory.expected_bytes == sum(
        len(source_contents[spec.repo_id][name]) for spec in specs for name in spec.required_files
    )
    assert calls == [(spec.repo_id, spec.revision, True) for spec in specs]
    evidence = inventory.export_evidence()
    assert str(tmp_path) not in repr(evidence)
    assert evidence["expected_bytes"] == inventory.expected_bytes


@pytest.mark.parametrize("revision", ["main", "master", "latest", "refs/heads/main"])
def test_rejects_mutable_final_revision(tmp_path: Path, revision: str) -> None:
    client = SourceInventoryClient(tmp_path, lambda **_: info("repo/model", []), pytest.fail)
    with pytest.raises(SourceInventoryError, match="immutable revision"):
        client.resolve((SourceSpec("component", "repo/model", revision, ("config.json",)),))


def test_rejects_missing_license(tmp_path: Path) -> None:
    response = info("repo/model", [sibling("config.json", b"{}")], license_name=None)
    client = SourceInventoryClient(tmp_path, lambda **_: response, pytest.fail)
    with pytest.raises(FeasibilityEvidenceError, match="license"):
        client.resolve((SourceSpec("component", "repo/model", "v1", ("config.json",)),))


@pytest.mark.parametrize(
    "filename", ["modeling_custom.py", "Modeling_Custom.PY", "../escape.bin", "/absolute.bin"]
)
def test_rejects_custom_code_and_unsafe_filenames(tmp_path: Path, filename: str) -> None:
    response = info("repo/model", [sibling("config.json", b"{}"), sibling(filename, b"x")])
    client = SourceInventoryClient(tmp_path, lambda **_: response, pytest.fail)
    with pytest.raises(SourceInventoryError, match="custom code|unsafe filename"):
        client.resolve((SourceSpec("component", "repo/model", "v1", ("config.json",)),))


def test_rejects_duplicate_components_and_missing_required_files(tmp_path: Path) -> None:
    response = info("repo/model", [sibling("config.json", b"{}")])
    client = SourceInventoryClient(tmp_path, lambda **_: response, pytest.fail)
    duplicate = SourceSpec("same", "repo/model", "v1", ("config.json",))
    with pytest.raises(SourceInventoryError, match="duplicate logical component"):
        client.resolve((duplicate, duplicate))
    with pytest.raises(SourceInventoryError, match="missing required file"):
        client.resolve((SourceSpec("one", "repo/model", "v1", ("weights.bin",)),))


def test_preflight_uses_twenty_gib_or_twenty_percent_margin(
    tmp_path: Path, specs: tuple[SourceSpec, ...], source_contents: dict[str, dict[str, bytes]]
) -> None:
    model_info, _ = make_api(source_contents)
    inventory = SourceInventoryClient(tmp_path, model_info, pytest.fail).resolve(specs)
    required = inventory.expected_bytes + 20 * 1024**3

    ok = inventory.preflight(lambda _: SimpleNamespace(free=required))
    assert ok.safety_margin_bytes == 20 * 1024**3
    assert ok.required_bytes == required
    assert "cache" not in repr(ok.export_evidence()).lower()

    with pytest.raises(SourceInventoryError, match="insufficient free space"):
        inventory.preflight(lambda _: SimpleNamespace(free=required - 1))


def test_download_is_pinned_resumable_and_verifies_inventory(
    tmp_path: Path, specs: tuple[SourceSpec, ...], source_contents: dict[str, dict[str, bytes]]
) -> None:
    model_info, _ = make_api(source_contents)
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        target = tmp_path / "cache" / str(kwargs["repo_id"]).replace("/", "--")
        for filename in kwargs["allow_patterns"]:  # type: ignore[union-attr]
            path = target / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_contents[str(kwargs["repo_id"])][filename])
        return str(target)

    inventory = SourceInventoryClient(tmp_path / "cache", model_info, download).resolve(specs)
    completed = inventory.download()

    assert len(calls) == 4
    assert all(call["revision"] == COMMIT for call in calls)
    assert all(call["cache_dir"] == str(tmp_path / "cache") for call in calls)
    assert all(call["local_files_only"] is False for call in calls)
    assert all(call["resume_download"] is True for call in calls)
    assert all(source.files[0].local_sha256 for source in completed.sources)
    assert str(tmp_path) not in repr(completed.export_evidence())


def test_download_rechecks_frozen_revision_before_downloading_any_bytes(
    tmp_path: Path, specs: tuple[SourceSpec, ...], source_contents: dict[str, dict[str, bytes]]
) -> None:
    revisions = {spec.repo_id: COMMIT for spec in specs}
    model_info, _ = make_api(source_contents, revisions=revisions)

    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        raise AssertionError("drift must fail before downloader")

    inventory = SourceInventoryClient(tmp_path / "cache", model_info, download).resolve(specs)
    revisions[specs[0].repo_id] = "b" * 40
    with pytest.raises(SourceInventoryError, match="resolved commit changed"):
        inventory.download()
    assert calls == []


def test_download_rejects_symlink_escape_with_stable_revision(
    tmp_path: Path, source_contents: dict[str, dict[str, bytes]]
) -> None:
    spec = SourceSpec("animegen", "aidealab/AnimeGen-T2V", "v1", ("config.json",))
    model_info, _ = make_api(source_contents)

    def download(**kwargs: object) -> str:
        target = tmp_path / "cache" / "download"
        target.mkdir(parents=True)
        os.symlink(tmp_path / "outside", target / "config.json")
        return str(target)

    inventory = SourceInventoryClient(tmp_path / "cache", model_info, download).resolve((spec,))
    with pytest.raises(SourceInventoryError, match="symlink escapes cache"):
        inventory.download()


def test_download_wraps_cyclic_symlink_resolution_error_with_cause(
    tmp_path: Path, source_contents: dict[str, dict[str, bytes]]
) -> None:
    spec = SourceSpec("animegen", "aidealab/AnimeGen-T2V", "v1", ("config.json",))
    model_info, _ = make_api(source_contents)

    def download(**_: object) -> str:
        target = tmp_path / "cache" / "download"
        target.mkdir(parents=True)
        os.symlink("config.json", target / "config.json")
        return str(target)

    inventory = SourceInventoryClient(tmp_path / "cache", model_info, download).resolve((spec,))
    with pytest.raises(SourceInventoryError, match="symlink resolution failed") as captured:
        inventory.download()
    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize("failure", ["missing", "size", "hash"])
def test_download_rejects_missing_size_and_hash_mismatches(
    tmp_path: Path, source_contents: dict[str, dict[str, bytes]], failure: str
) -> None:
    spec = SourceSpec("animegen", "aidealab/AnimeGen-T2V", "v1", ("config.json",))
    model_info, _ = make_api(source_contents)

    def download(**kwargs: object) -> str:
        target = tmp_path / "cache" / "download"
        target.mkdir(parents=True)
        if failure != "missing":
            content = b"x" if failure == "size" else b"wrong-content"
            if failure == "hash":
                content = b"anime-config"[:-1] + b"X"
            (target / "config.json").write_bytes(content)
        return str(target)

    inventory = SourceInventoryClient(tmp_path / "cache", model_info, download).resolve((spec,))
    expected = {"missing": "missing file", "size": "size mismatch", "hash": "hash mismatch"}
    with pytest.raises(SourceInventoryError, match=expected[failure]):
        inventory.download()


@pytest.mark.parametrize("sha", ["g" * 64, "A" * 64, "0" * 63, 42])
def test_inventory_rejects_non_lowercase_hex_upstream_sha256(tmp_path: Path, sha: object) -> None:
    response = info("repo/model", [sibling("config.json", b"{}")])
    response.siblings[0].lfs = SimpleNamespace(sha256=sha)
    client = SourceInventoryClient(tmp_path, lambda **_: response, pytest.fail)
    with pytest.raises(SourceInventoryError, match="invalid upstream hash"):
        client.resolve((SourceSpec("component", "repo/model", "v1", ("config.json",)),))


def test_filesystem_hash_failure_is_normalized_with_cause(
    tmp_path: Path, source_contents: dict[str, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_lab.source_inventory as module

    spec = SourceSpec("animegen", "aidealab/AnimeGen-T2V", "v1", ("config.json",))
    model_info, _ = make_api(source_contents)

    def download(**_: object) -> str:
        target = tmp_path / "cache" / "download"
        target.mkdir(parents=True)
        (target / "config.json").write_bytes(b"anime-config")
        return str(target)

    monkeypatch.setattr(module, "_sha256", lambda _: (_ for _ in ()).throw(OSError("denied")))
    inventory = SourceInventoryClient(tmp_path / "cache", model_info, download).resolve((spec,))
    with pytest.raises(SourceInventoryError, match="filesystem verification failed") as captured:
        inventory.download()
    assert isinstance(captured.value.__cause__, OSError)


def test_downloader_filesystem_failure_is_normalized_with_cause(
    tmp_path: Path, source_contents: dict[str, dict[str, bytes]]
) -> None:
    spec = SourceSpec("animegen", "aidealab/AnimeGen-T2V", "v1", ("config.json",))
    model_info, _ = make_api(source_contents)
    inventory = SourceInventoryClient(
        tmp_path / "cache",
        model_info,
        lambda **_: (_ for _ in ()).throw(OSError("disk denied")),
    ).resolve((spec,))
    with pytest.raises(SourceInventoryError, match="download filesystem access failed") as captured:
        inventory.download()
    assert isinstance(captured.value.__cause__, OSError)


def test_import_is_lazy_and_network_free() -> None:
    script = """
import builtins
import http.client
import socket
import urllib.request
real_import = builtins.__import__
def trapped(name, *args, **kwargs):
    if name.split('.')[0] in {'huggingface_hub', 'diffusers', 'torch'}:
        raise AssertionError(f'heavy import: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = trapped
def no_network(*args, **kwargs):
    raise AssertionError('network access')
socket.create_connection = no_network
http.client.HTTPConnection.connect = no_network
http.client.HTTPSConnection.connect = no_network
urllib.request.urlopen = no_network
import video_lab.source_inventory
from video_lab.adapter_registry import AdapterRegistry
from video_lab.operations import LaboratoryOperations
LaboratoryOperations.catalog_for(AdapterRegistry.with_defaults())
import video_lab.feasibility_models
from typer.testing import CliRunner
from video_lab.cli import app
result = CliRunner().invoke(app, ['--help'])
assert result.exit_code == 0, result.output
import video_lab.cli
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_only_resume_download_stage_invokes_downloader(
    database: Database, tmp_path: Path, source_contents: dict[str, dict[str, bytes]]
) -> None:
    spec = SourceSpec("animegen", "aidealab/AnimeGen-T2V", "v1", ("config.json",))
    model_info, _ = make_api(source_contents)
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        target = tmp_path / "cache" / "download"
        target.mkdir(parents=True, exist_ok=True)
        for filename in kwargs["allow_patterns"]:  # type: ignore[union-attr]
            path = target / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_contents[str(kwargs["repo_id"])][filename])
        return str(target)

    inventory = SourceInventoryClient(tmp_path / "cache", model_info, download).resolve((spec,))

    class InventoryExecutor:
        def execute(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult:
            if stage.stage is FeasibilityStage.DOWNLOAD_SOURCES:
                inventory.download()
            return StageResult.passed()

    service = FeasibilityService(FeasibilityRepository(database), InventoryExecutor())
    run = service.start(
        {
            "adapter_id": "diffusers.animegen",
            "protocol_version": "v1",
            "sources": {
                "model": {
                    "id": "animegen/model",
                    "revision": "a" * 40,
                    "checksum": f"sha256:{'1' * 64}",
                }
            },
            "runtime_versions": {"python": "3.12"},
            "hardware_fingerprint": "test",
            "os_fingerprint": "test",
            "manifest_profiles": {"minimum": {"frames": 1}, "standard": {"frames": 17}},
            "memory_strategy": "test",
        }
    )
    for expected_calls in (0, 0, 0, 1):
        service.resume(run.id)
        assert len(calls) == expected_calls
    assert service.status(run.id).attempts[-1].stage is FeasibilityStage.DOWNLOAD_SOURCES
    assert service.status(run.id).attempts[-1].outcome is StageOutcome.PASSED
