from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import video_lab.animegen_feasibility as animegen_feasibility_module
import video_lab.cli as cli_module
from lab_tests.test_animegen_adapter import Runtime, TorchStub
from video_lab.artifacts import ArtifactStore
from video_lab.cli import app
from video_lab.database import Database
from video_lab.domain import Manifest, ReviewDimension
from video_lab.feasibility_models import FeasibilityConfiguration, FeasibilityStage
from video_lab.feasibility_projection import FeasibilityEvidenceProjector
from video_lab.feasibility_repository import FeasibilityRepository
from video_lab.runtime_identity import animegen_runtime_versions, runtime_fingerprint
from video_lab.source_inventory import (
    ANIMEGEN_REQUIRED_FILES,
    ANIMEGEN_SOURCE_REPOS,
    ResolvedSource,
    SourceFile,
    SourceInventoryClient,
    canonical_source_digest,
)

runner = CliRunner()
ADAPTER = "diffusers.animegen-t2v.wan2.2-a14b.mps.v1"


def test_checked_animegen_example_is_the_bf16_guidance_contract() -> None:
    example = Path(__file__).parent / "fixtures/animegen-feasibility.json"
    configuration = FeasibilityConfiguration.model_validate_json(example.read_text())

    canonical = animegen_feasibility_module.canonical_animegen_configuration(configuration)
    minimum = Manifest.model_validate(canonical["manifest_profiles"]["minimum"])
    standard = Manifest.model_validate(canonical["manifest_profiles"]["standard"])

    assert minimum.quantization == "bf16"
    assert standard.quantization == "bf16"
    assert minimum.guidance_scale == 1.0
    assert standard.guidance_scale == 1.0
    assert minimum.seed == 7
    assert standard.seed == 7
    alternate = standard.model_copy(update={"guidance_scale": 3.5})
    changed = {
        key
        for key, value in standard.model_dump(mode="json").items()
        if alternate.model_dump(mode="json")[key] != value
    }
    assert changed == {"guidance_scale"}


def _configuration(path: Path) -> Path:
    revision = "a" * 40
    checksum = "1" * 64
    sources = {
        component: {"id": repo, "revision": revision, "checksum": f"sha256:{checksum}"}
        for component, repo in {
            "animegen": "aidealab/AnimeGen-T2V",
            "wan-base": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "lightning-high": "lightx2v/Wan2.2-Lightning",
            "lightning-low": "lightx2v/Wan2.2-Lightning",
        }.items()
    }

    runtime_versions = animegen_runtime_versions()

    def manifest(width: int, height: int) -> dict[str, object]:
        return {
            "model_id": sources["animegen"]["id"],
            "model_revision": revision,
            "model_checksum": checksum,
            "adapter_id": ADAPTER,
            "adapter_version": "1",
            "runtime_versions": runtime_versions,
            "source_checksums": {
                component: identity["checksum"].removeprefix("sha256:")
                for component, identity in sources.items()
            },
            "prompt": "anime character walking",
            "seed": 7,
            "width": width,
            "height": height,
            "num_frames": 17,
            "fps": 8,
            "inference_steps": 4,
            "guidance_scale": 1.0,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "bf16",
            "loras": [
                {
                    "model_id": sources[name]["id"],
                    "revision": revision,
                    "checksum": checksum,
                    "weight": weight,
                }
                for name, weight in (("lightning-high", 2.0), ("lightning-low", 1.0))
            ],
            "backend": "mps",
            "device": "mps",
            "memory_strategy": "model_cpu_offload",
            "hardware_fingerprint": "apple-silicon",
            "os_fingerprint": "macos-arm64",
        }

    value = {
        "adapter_id": ADAPTER,
        "protocol_version": "animegen-feasibility.v2",
        "sources": sources,
        "runtime_versions": runtime_versions,
        "hardware_fingerprint": "apple-silicon",
        "os_fingerprint": "macos-arm64",
        "manifest_profiles": {"minimum": manifest(128, 80), "standard": manifest(512, 288)},
        "memory_strategy": "model_cpu_offload",
    }
    path.write_text(json.dumps(value))
    return path


def _invoke(tmp_path: Path, *arguments: str):
    return runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "lab.db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "adapter",
            "feasibility",
            *arguments,
        ],
    )


def _invoke_as(tmp_path: Path, reviewer_id: str, *arguments: str):
    return runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "lab.db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--reviewer-id",
            reviewer_id,
            "adapter",
            "feasibility",
            *arguments,
        ],
    )


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {nested for item in value.values() for nested in _nested_keys(item)}
    if isinstance(value, list):
        return {nested for item in value for nested in _nested_keys(item)}
    return set()


def test_runtime_identity_command_reports_complete_fingerprinted_identity(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "runtime-identity", "--json")

    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)["data"]
    assert data["runtime_versions"] == animegen_runtime_versions()
    assert data["runtime_fingerprint"] == runtime_fingerprint(data["runtime_versions"])
    assert data["runtime_versions"]["video-lab"].startswith("source-sha256:")
    assert data["runtime_versions"]["dist:tokenizers"] == "0.22.2"
    assert "dist:pillow" in data["runtime_versions"]
    assert "dist:psutil" in data["runtime_versions"]


def test_start_rejects_configuration_from_a_different_runtime(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    value = json.loads(configuration.read_text())
    value["runtime_versions"]["dist:otherwise-uncurated"] = "drifted"
    for profile in value["manifest_profiles"].values():
        profile["runtime_versions"]["dist:otherwise-uncurated"] = "drifted"
    configuration.write_text(json.dumps(value))

    result = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")

    assert result.exit_code == 2
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "invalid_input"
    assert "runtime identity" in error["message"]


def _set_verified_digests(configuration: Path) -> dict[str, str]:
    configured = json.loads(configuration.read_text())
    digests = {
        component: canonical_source_digest(
            ResolvedSource(
                component,
                repo_id,
                "a" * 40,
                "a" * 40,
                "apache-2.0",
                tuple(
                    SourceFile(
                        name,
                        1,
                        upstream_sha256=hashlib.sha256(b"x").hexdigest(),
                        local_sha256=hashlib.sha256(b"x").hexdigest(),
                    )
                    for name in ANIMEGEN_REQUIRED_FILES[component]
                ),
            )
        )
        for component, repo_id in ANIMEGEN_SOURCE_REPOS.items()
    }
    for component, digest in digests.items():
        configured["sources"][component]["checksum"] = f"sha256:{digest}"
    for profile in configured["manifest_profiles"].values():
        profile["source_checksums"] = digests
        profile["model_checksum"] = digests["animegen"]
        profile["loras"][0]["checksum"] = digests["lightning-high"]
        profile["loras"][1]["checksum"] = digests["lightning-low"]
    configuration.write_text(json.dumps(configured))
    return digests


def test_start_status_and_resume_emit_compact_versioned_json(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    started = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")
    assert started.exit_code == 0
    assert "\n " not in started.stdout
    body = json.loads(started.stdout)
    assert body["schema_version"] == "feasibility-command.v1"
    assert body["ok"] is True
    assert body["data"]["run"]["verdict"] == "pending"
    assert body["data"]["attempts"] == []
    run_id = body["data"]["run"]["id"]

    status = _invoke(tmp_path, "status", run_id, "--json")
    assert status.exit_code == 0
    assert json.loads(status.stdout)["data"]["next_stage"] == "resolve_sources"

    resumed = _invoke(tmp_path, "resume", run_id, "--json")
    assert resumed.exit_code == 0
    data = json.loads(resumed.stdout)["data"]
    assert [attempt["stage"] for attempt in data["attempts"]] == ["resolve_sources"]
    assert data["completed_attempt"]["stage"] == "resolve_sources"
    assert data["next_stage"] == "inventory_sources"


def test_inventory_stage_imports_exact_path_neutral_evidence_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    started = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")
    run_id = json.loads(started.stdout)["data"]["run"]["id"]
    resolved = _invoke(tmp_path, "resume", run_id, "--json")
    assert resolved.exit_code == 0
    digest = hashlib.sha256(b"x").hexdigest()
    sources = []
    for component, repo_id in ANIMEGEN_SOURCE_REPOS.items():
        files = [
            {
                "relative_path": name,
                "size": 1,
                "upstream_sha256": digest,
            }
            for name in ANIMEGEN_REQUIRED_FILES[component]
        ]
        sources.append(
            {
                "component": component,
                "repo_id": repo_id,
                "requested_revision": "a" * 40,
                "revision": "a" * 40,
                "license": "apache-2.0",
                "files": files,
            }
        )
    evidence = tmp_path / "inventory.json"
    evidence.write_text(
        json.dumps(
            {
                "expected_bytes": sum(len(item["files"]) for item in sources),
                "sources": sources,
            }
        )
    )

    client = SourceInventoryClient(
        tmp_path / "cache",
        lambda **_: pytest.fail("inventory import must not call the Hub"),
        lambda **_: pytest.fail("inventory import must not download"),
    )
    monkeypatch.setattr(
        animegen_feasibility_module, "production_inventory_client", lambda _cache: client
    )

    imported = _invoke(
        tmp_path,
        "resume",
        run_id,
        "--cache",
        str(tmp_path / "cache"),
        "--inventory-evidence",
        str(evidence),
        "--json",
    )

    assert imported.exit_code == 0, imported.stdout
    attempt = json.loads(imported.stdout)["data"]["completed_attempt"]
    assert attempt["stage"] == "inventory_sources"
    assert attempt["outcome"] == "passed"
    assert "evidence" not in attempt

    exported = tmp_path / "exported.json"
    export_result = _invoke(
        tmp_path,
        "export-inventory",
        run_id,
        "--output",
        str(exported),
        "--json",
    )
    assert export_result.exit_code == 0, (export_result.stdout, export_result.stderr)
    assert json.loads(exported.read_text()) == json.loads(evidence.read_text())
    assert str(tmp_path) not in exported.read_text()


def test_inventory_import_rejects_source_identity_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    started = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")
    run_id = json.loads(started.stdout)["data"]["run"]["id"]
    _invoke(tmp_path, "resume", run_id, "--json")
    evidence = tmp_path / "inventory.json"
    evidence.write_text(
        json.dumps(
            {
                "expected_bytes": 1,
                "sources": [
                    {
                        "component": "animegen",
                        "repo_id": "attacker/model",
                        "requested_revision": "a" * 40,
                        "revision": "a" * 40,
                        "license": "apache-2.0",
                        "files": [
                            {
                                "relative_path": "high_noise.safetensors",
                                "size": 1,
                                "upstream_sha256": "1" * 64,
                            }
                        ],
                    }
                ],
            }
        )
    )
    client = SourceInventoryClient(tmp_path / "cache", lambda **_: object(), lambda **_: "")
    monkeypatch.setattr(
        animegen_feasibility_module, "production_inventory_client", lambda _cache: client
    )

    rejected = _invoke(
        tmp_path,
        "resume",
        run_id,
        "--cache",
        str(tmp_path / "cache"),
        "--inventory-evidence",
        str(evidence),
        "--json",
    )

    body = json.loads(rejected.stdout)
    assert rejected.exit_code == 0
    assert body["data"]["completed_attempt"]["outcome"] == "unexpected_failure"
    assert body["data"]["next_stage"] == "inventory_sources"


def test_start_rejects_unknown_adapter_and_mutable_revision(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    unknown = _invoke(tmp_path, "start", "unknown", "--input", str(configuration), "--json")
    assert unknown.exit_code == 3
    assert json.loads(unknown.stdout)["error"]["code"] == "adapter_not_found"

    value = json.loads(configuration.read_text())
    value["sources"]["animegen"]["revision"] = "main"
    configuration.write_text(json.dumps(value))
    mutable = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")
    assert mutable.exit_code == 2
    assert json.loads(mutable.stdout)["error"]["code"] == "invalid_input"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["manifest_profiles"].__setitem__(
            "minimum", {"width": 128, "height": 80, "frames": 17, "fps": 8}
        ),
        lambda value: value["manifest_profiles"]["standard"].__setitem__("dtype", "float16"),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("model_revision", "b" * 40),
        lambda value: value["manifest_profiles"]["standard"].__setitem__("width", 496),
        lambda value: value["manifest_profiles"]["standard"].__setitem__("inference_steps", 8),
        lambda value: value.__setitem__("protocol_version", "animegen-feasibility.v1"),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("width", 256),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("height", 144),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("num_frames", 33),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("fps", 16),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("inference_steps", 5),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("guidance_scale", 2.0),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("seed", 99),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__(
            "prompt", "a different prompt"
        ),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__(
            "negative_prompt", "watermark"
        ),
        lambda value: value["manifest_profiles"]["standard"].__setitem__("guidance_scale", 3.5),
    ],
)
def test_start_rejects_noncanonical_animegen_profiles(tmp_path: Path, mutate: object) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    value = json.loads(configuration.read_text())
    mutate(value)  # type: ignore[operator]
    configuration.write_text(json.dumps(value))

    result = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "invalid_input"


def test_unknown_transformer_storage_is_invalid_input_not_internal_error(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    value = json.loads(configuration.read_text())
    for profile in value["manifest_profiles"].values():
        profile["quantization"] = "unknown-storage"
    configuration.write_text(json.dumps(value))

    result = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")

    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "schema_version": "feasibility-command.v1",
        "ok": False,
        "data": None,
        "error": {
            "code": "invalid_input",
            "message": "minimum manifest uses unsupported transformer storage",
        },
    }


def test_unknown_run_and_missing_cache_have_stable_exit_codes(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")
    missing = _invoke(tmp_path, "status", "missing", "--json")
    assert missing.exit_code == 3
    assert json.loads(missing.stdout)["error"]["code"] == "feasibility_run_not_found"

    started = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")
    run_id = json.loads(started.stdout)["data"]["run"]["id"]
    assert _invoke(tmp_path, "resume", run_id, "--json").exit_code == 0
    blocked = _invoke(tmp_path, "resume", run_id, "--json")
    assert blocked.exit_code == 2
    assert json.loads(blocked.stdout)["error"]["code"] == "cache_required"


def test_status_does_not_create_a_missing_database(tmp_path: Path) -> None:
    database = tmp_path / "lab.db"

    result = _invoke(tmp_path, "status", "missing", "--json")

    assert result.exit_code == 4
    assert not database.exists()


def test_human_output_is_summary_and_help_does_not_import_inference_packages(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    result = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration))
    assert result.exit_code == 0
    assert "pending" in result.stdout
    assert ADAPTER in result.stdout

    before = set(sys.modules)
    help_result = runner.invoke(app, ["adapter", "feasibility", "--help"])
    assert help_result.exit_code == 0
    assert not ({"torch", "diffusers", "transformers"} - before) & set(sys.modules)


def test_start_and_status_are_zero_network_and_status_does_not_mutate_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    monkeypatch.setattr(
        cli_module,
        "production_inventory_client",
        lambda *_: pytest.fail("network composition must stay lazy"),
        raising=False,
    )
    started = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")
    assert started.exit_code == 0
    run_id = json.loads(started.stdout)["data"]["run"]["id"]
    database = tmp_path / "lab.db"
    with database.open("rb") as handle:
        assert handle.read(16) == b"SQLite format 3\x00"
    before = FeasibilityRepository(Database(database)).get_run(run_id)
    with Database(database).connect() as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before_bytes = database.read_bytes()
    before_mtime = database.stat().st_mtime_ns

    status = _invoke(tmp_path, "status", run_id, "--json")

    assert status.exit_code == 0
    assert database.read_bytes() == before_bytes
    assert database.stat().st_mtime_ns == before_mtime
    repository = FeasibilityRepository(Database(database))
    assert repository.get_run(run_id) == before
    assert repository.list_attempts(run_id) == ()


def test_json_projection_recursively_sanitizes_configuration(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "configuration.json")
    value = json.loads(configuration.read_text())
    value["hardware_fingerprint"] = "/Users/alice/.cache token=super-secret"
    for profile in value["manifest_profiles"].values():
        profile["hardware_fingerprint"] = value["hardware_fingerprint"]
    configuration.write_text(json.dumps(value))

    result = _invoke(tmp_path, "start", ADAPTER, "--input", str(configuration), "--json")

    assert result.exit_code == 0
    assert "/Users" not in result.stdout
    assert ".cache" not in result.stdout
    assert "super-secret" not in result.stdout
