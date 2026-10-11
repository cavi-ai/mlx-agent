"""Read-only qualification of explicit local Wan production authorities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import video_lab.host_canaries.wan_qualification as wan_qualification
from video_lab.host_canaries.wan_qualification import (
    WAN_PRODUCTION_AUTHORITY_NAMES,
    WanQualificationCandidateV1,
    WanQualificationManifestV1,
    WanQualificationRejectionV1,
    WanQualificationSearchV1,
    discover_wan_production_authorities,
)
from video_lab.local_authority import RetainedDirectoryAuthority, RetainedSqliteAuthority
from video_lab.trainers.wan_factory import root_manifest_digest

_REVISION = "a" * 40
_ADAPTER_PATHS = {
    "wan_snapshot": ".",
    "scheduler": "scheduler",
    "text_encoder": "text_encoder",
    "tokenizer": "tokenizer",
    "transformer_config": "transformer/config.json",
    "high_transformer": "weights/high.safetensors",
    "low_transformer": "weights/low.safetensors",
    "vae": "vae",
}
_RECIPE_AUTHORITY_NAMES = {
    "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
    "VIDEO_LAB_WAN_CACHE_REVISION",
    "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
    "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
    "VIDEO_LAB_WAN_MODEL_REVISION",
    "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
    "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
    "VIDEO_LAB_WAN_TOLERANCE_JSON",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_tree(root: Path) -> str:
    records: list[bytes] = []
    for path in sorted(root.rglob("*")):
        info = path.stat(follow_symlinks=False)
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            records.append(b"l\0" + relative + b"\0" + os.readlink(path).encode())
        elif path.is_dir():
            records.append(b"d\0" + relative + b"\0" + str(info.st_mode).encode())
        elif path.is_file():
            records.append(b"f\0" + relative + b"\0" + path.read_bytes())
        else:
            records.append(b"s\0" + relative + b"\0" + str(info.st_mode).encode())
    return hashlib.sha256(b"\n".join(records)).hexdigest()


def _write_model(root: Path) -> None:
    files = {
        "model_index.json": b"{}",
        "scheduler/scheduler_config.json": b"{}",
        "text_encoder/config.json": b"{}",
        "tokenizer/tokenizer.json": b"{}",
        "transformer/config.json": b"{}",
        "weights/high.safetensors": b"high",
        "weights/low.safetensors": b"low",
        "vae/config.json": b"{}",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _write_receipt_database(
    path: Path,
    artifact_root: Path,
    recipe_bindings: tuple[tuple[str, str], ...],
) -> tuple[str, str]:
    from video_lab.artifacts import ArtifactStore
    from video_lab.database import Database
    from video_lab.dataset_compiler import BackendOutput, DatasetCompilationService
    from video_lab.dataset_receipts import DatasetCompilationRequest, InputBinding

    class ReceiptBackend:
        compiler_id = "wan_curated_v1"

        def validate_recipe(self, recipe_bytes: bytes) -> None:
            if not recipe_bytes:
                raise ValueError

        def compile_batch(
            self, source_bytes: bytes, recipe_bytes: bytes
        ) -> tuple[BackendOutput, ...]:
            if not source_bytes or not recipe_bytes:
                raise ValueError
            return (
                BackendOutput("batch", b"compiled-batch", "application/vnd.safetensors"),
                BackendOutput("manifest", b"{}", "application/json"),
                BackendOutput("provenance", b'{"bound":true}', "application/json"),
            )

    database = Database(path)
    database.migrate()
    artifacts = ArtifactStore(database, artifact_root)
    source = artifacts.commit(
        artifacts.stage_bytes(b'{"source":1}', "application/json"), None
    )
    recipe_payload = _canonical(
        {
            "schema_version": "wan-production-qualification-recipe.v1",
            "authority_bindings": recipe_bindings,
        }
    ).encode()
    recipe = artifacts.commit(
        artifacts.stage_bytes(recipe_payload, "application/json"), None
    )
    request = DatasetCompilationRequest(
        operation_key="wan_production_qualification",
        compiler_id="wan_curated_v1",
        source=InputBinding(
            role="source_manifest",
            artifact_id=source.id,
            sha256=source.sha256,
            byte_size=source.byte_size,
            media_type=source.media_type,
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id=recipe.id,
            sha256=recipe.sha256,
            byte_size=recipe.byte_size,
            media_type=recipe.media_type,
        ),
        output_roles=("batch", "manifest", "provenance"),
    )
    receipt = DatasetCompilationService(
        database, artifacts, {"wan_curated_v1": ReceiptBackend()}
    ).compile(request)
    return receipt.receipt_id, receipt.operation_id


def _bindings(root: Path, *, model_name: str = "model") -> tuple[tuple[str, str], ...]:
    model = root / model_name
    cache = root / "cache"
    artifacts = root / "artifacts"
    database = root / "video-lab.sqlite3"
    _write_model(model)
    model_revision = hashlib.sha256(root_manifest_digest(model).encode()).hexdigest()[:40]
    (cache / "snapshots" / _REVISION).mkdir(parents=True, exist_ok=True)
    (cache / "refs").mkdir(exist_ok=True)
    (cache / "refs" / "main").write_text(_REVISION + "\n", encoding="utf-8")
    artifacts.mkdir(exist_ok=True)
    (artifacts / "authority.bin").write_bytes(b"authority")
    values = {
        "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON": _canonical(_ADAPTER_PATHS),
        "VIDEO_LAB_WAN_ARTIFACT_DATABASE": str(database),
        "VIDEO_LAB_WAN_ARTIFACT_ROOT": str(artifacts),
        "VIDEO_LAB_WAN_CACHE_REVISION": _REVISION,
        "VIDEO_LAB_WAN_CACHE_ROOT": str(cache),
        "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON": _canonical(
            [[name, importlib.metadata.version(name)] for name in ("diffusers", "peft", "safetensors", "torch", "transformers")]
        ),
        "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES": "8388608",
        "VIDEO_LAB_WAN_MODEL_REVISION": model_revision,
        "VIDEO_LAB_WAN_MODEL_ROOT": str(model),
        "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES": "9",
        "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON": _canonical(
            [
                {"copies": 1, "dtype": "bfloat16", "name": "activation", "role": "activation",
                 "schema_version": "wan-mps-tensor-envelope.v2", "shape": [1, 2, 2]},
                {"copies": 1, "dtype": "bfloat16", "name": "adapter", "role": "parameter",
                 "schema_version": "wan-mps-tensor-envelope.v2", "shape": [2, 2]},
                {"copies": 2, "dtype": "float32", "name": "optimizer", "role": "optimizer",
                 "schema_version": "wan-mps-tensor-envelope.v2", "shape": [2, 2]},
            ]
        ),
        "VIDEO_LAB_WAN_TOLERANCE_JSON": _canonical({"atol": 1e-5, "rtol": 1e-4}),
    }
    recipe_bindings = tuple(
        sorted((name, value) for name, value in values.items() if name in _RECIPE_AUTHORITY_NAMES)
    )
    receipt_id, operation_id = _write_receipt_database(
        database, artifacts, recipe_bindings
    )
    values["VIDEO_LAB_WAN_COMPILATION_OPERATION_ID"] = operation_id
    values["VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID"] = receipt_id
    return tuple(sorted(values.items()))


def _candidate_payload(bindings: tuple[tuple[str, str], ...]) -> dict[str, object]:
    evidence_names = {
        "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
        "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
        "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
        "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
        "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
        "VIDEO_LAB_WAN_MODEL_REVISION",
        "VIDEO_LAB_WAN_CACHE_REVISION",
        "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
        "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
        "VIDEO_LAB_WAN_TOLERANCE_JSON",
    }
    return {
        "schema_version": "wan-production-qualification-candidate.v1",
        "bindings": bindings,
        "accepted_compilation_evidence": tuple(
            item for item in bindings if item[0] in evidence_names
        ),
    }


def _make_local_model_inventory(root: Path) -> tuple[Path, tuple[tuple[str, str], ...]]:
    root.mkdir()
    bindings = _bindings(root)
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(bindings)), encoding="utf-8"
    )
    return root, bindings


def _request(root: Path) -> WanQualificationSearchV1:
    return WanQualificationSearchV1(
        schema_version="wan-production-qualification-search.v1",
        search_roots=(root,),
        maximum_directories=20_000,
        maximum_files=200_000,
    )


def test_discovery_is_read_only_and_import_free(tmp_path: Path) -> None:
    root, expected = _make_local_model_inventory(tmp_path / "scope")
    before = _digest_tree(tmp_path)
    code = f"""
import json, pathlib, sys
from video_lab.host_canaries.wan_qualification import (
    WanQualificationSearchV1, discover_wan_production_authorities,
)
r = discover_wan_production_authorities(WanQualificationSearchV1(
    schema_version='wan-production-qualification-search.v1',
    search_roots=(pathlib.Path({str(root)!r}),), maximum_directories=20000,
    maximum_files=200000))
assert r.status == 'qualified', r
assert dict(r.bindings) == json.loads({json.dumps(dict(expected))!r})
for name in ('torch', 'peft', 'diffusers', 'transformers'):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert _digest_tree(tmp_path) == before


def test_discovery_rejects_ambiguous_model_authority(tmp_path: Path) -> None:
    root, first = _make_local_model_inventory(tmp_path / "scope")
    second_model = root / "model-two"
    _write_model(second_model)
    second = tuple(
        (name, str(second_model) if name == "VIDEO_LAB_WAN_MODEL_ROOT" else value)
        for name, value in first
    )
    (root / "nested").mkdir()
    (root / "nested" / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(second)), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "ambiguous"
    assert result.bindings == ()
    assert result.ambiguous_roles == ("VIDEO_LAB_WAN_MODEL_ROOT",)


def test_missing_accepted_scientific_evidence_cannot_qualify(tmp_path: Path) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    payload = _candidate_payload(bindings)
    payload["accepted_compilation_evidence"] = tuple(
        item
        for item in payload["accepted_compilation_evidence"]  # type: ignore[union-attr]
        if item[0] != "VIDEO_LAB_WAN_TOLERANCE_JSON"
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(payload), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == ("VIDEO_LAB_WAN_TOLERANCE_JSON",)


def test_discovery_does_not_follow_candidate_symlinks_or_accept_special_files(
    tmp_path: Path,
) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    candidate = root / "wan-production-qualification-candidate.v1.json"
    candidate.unlink()
    outside, _ = _make_local_model_inventory(tmp_path / "outside")
    candidate.symlink_to(outside / candidate.name)
    fifo = root / "nested" / candidate.name
    fifo.parent.mkdir()
    os.mkfifo(fifo)

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


def test_search_and_manifest_models_are_strict_frozen_and_revalidated(tmp_path: Path) -> None:
    root = tmp_path / "scope"
    root.mkdir()
    request = _request(root)
    with pytest.raises(ValidationError):
        WanQualificationSearchV1.model_validate(
            {**request.model_dump(mode="python"), "search_roots": [root]}, strict=True
        )
    with pytest.raises(ValidationError):
        WanQualificationSearchV1(
            **request.model_dump(mode="python"), unexpected=True  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError) as captured:
        WanQualificationSearchV1(
            schema_version="wan-production-qualification-search.v1",
            search_roots=(Path("private-authority"),),
            maximum_directories=20_000,
            maximum_files=200_000,
        )
    assert "private-authority" not in str(captured.value)
    with pytest.raises(ValidationError):
        request.maximum_files = 1  # type: ignore[misc]

    invalid = WanQualificationManifestV1.model_construct(
        schema_version="wan-production-qualification-manifest.v1",
        status="qualified",
        bindings=(),
        missing_roles=(),
        ambiguous_roles=(),
        candidate_digests=(),
    )
    with pytest.raises(ValidationError):
        invalid.model_copy()


def test_candidate_rejects_noncanonical_or_unknown_authorities(tmp_path: Path) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    evidence = tuple(
        item
        for item in bindings
        if item[0]
        in {
            "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
            "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
            "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
            "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
            "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
            "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
            "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
            "VIDEO_LAB_WAN_TOLERANCE_JSON",
        }
    )
    candidate = WanQualificationCandidateV1(
        schema_version="wan-production-qualification-candidate.v1",
        bindings=bindings,
        accepted_compilation_evidence=evidence,
    )
    assert candidate.bindings == tuple(sorted(candidate.bindings))
    with pytest.raises(ValidationError):
        candidate.model_copy(update={"bindings": tuple(reversed(bindings))})
    with pytest.raises(ValidationError):
        WanQualificationCandidateV1(
            schema_version="wan-production-qualification-candidate.v1",
            bindings=bindings + (("VIDEO_LAB_WAN_UNKNOWN", "private"),),  # type: ignore[arg-type]
            accepted_compilation_evidence=evidence,
        )


def test_discovery_enforces_directory_and_file_bounds(tmp_path: Path) -> None:
    root, _ = _make_local_model_inventory(tmp_path / "scope")
    with pytest.raises(ValueError, match="wan_qualification_search_limit_exceeded"):
        discover_wan_production_authorities(
            _request(root).model_copy(update={"maximum_files": 1})
        )


def test_malformed_candidate_is_reported_without_private_values(tmp_path: Path) -> None:
    root = tmp_path / "scope"
    root.mkdir()
    (root / "wan-production-qualification-candidate.v1.json").write_bytes(b"{private")

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES
    assert result.candidate_digests == ()
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == "malformed"
    assert result.rejections[0].candidate_digest.startswith("sha256:")
    assert "private" not in result.model_dump_json()


def test_partial_candidates_covering_every_role_do_not_claim_all_absent(
    tmp_path: Path,
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    complete = _candidate_payload(bindings)
    evidence = tuple(complete["accepted_compilation_evidence"])  # type: ignore[arg-type]
    complete["accepted_compilation_evidence"] = tuple(
        item for item in evidence if item[0] != "VIDEO_LAB_WAN_TOLERANCE_JSON"
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(complete), encoding="utf-8"
    )
    nested = root / "nested"
    nested.mkdir()
    tolerance_only = _candidate_payload(bindings)
    tolerance_only["accepted_compilation_evidence"] = tuple(
        item
        for item in evidence
        if item[0]
        in {
            "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
            "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
            "VIDEO_LAB_WAN_TOLERANCE_JSON",
        }
    )
    (nested / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(tolerance_only), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.missing_roles == ()
    assert result.candidate_digests == ()
    assert tuple(item.reason for item in result.rejections) == (
        "incomplete",
        "incomplete",
    )


@pytest.mark.parametrize("status,roles", [("missing", WAN_PRODUCTION_AUTHORITY_NAMES),
                                           ("ambiguous", (WAN_PRODUCTION_AUTHORITY_NAMES[0],))])
def test_unqualified_manifest_rejects_candidate_binding_digests(
    status: str, roles: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError):
        WanQualificationManifestV1(
            schema_version="wan-production-qualification-manifest.v1",
            status=status,  # type: ignore[arg-type]
            bindings=(),
            missing_roles=roles if status == "missing" else (),  # type: ignore[arg-type]
            ambiguous_roles=roles if status == "ambiguous" else (),  # type: ignore[arg-type]
            candidate_digests=((WAN_PRODUCTION_AUTHORITY_NAMES[0], "sha256:" + "0" * 64),),
            rejections=(
                WanQualificationRejectionV1(
                    candidate_digest="sha256:" + "1" * 64,
                    reason="incomplete",
                ),
            ),
        )


def test_path_bindings_cannot_escape_an_approved_root_with_dot_dot(tmp_path: Path) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    (root / "nested").mkdir()
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    escaped = root / "nested" / ".." / ".." / outside.name
    bindings = tuple(
        (name, str(escaped) if name == "VIDEO_LAB_WAN_ARTIFACT_ROOT" else value)
        for name, value in bindings
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(bindings)), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


def test_model_revision_requires_local_revision_evidence(tmp_path: Path) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    bindings = tuple(
        (name, "b" * 40 if name == "VIDEO_LAB_WAN_MODEL_REVISION" else value)
        for name, value in bindings
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(bindings)), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == (
        "VIDEO_LAB_WAN_MODEL_REVISION",
        "VIDEO_LAB_WAN_MODEL_ROOT",
    )


def test_cache_revision_requires_snapshot_ref_or_component_evidence(tmp_path: Path) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    cache = root / "cache"
    for path in sorted((cache / "snapshots").rglob("*"), reverse=True):
        path.rmdir()
    (cache / "snapshots").rmdir()
    (cache / "refs" / "main").unlink()
    (cache / "refs").rmdir()

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == ("VIDEO_LAB_WAN_CACHE_ROOT",)


def test_memory_release_bound_must_be_positive(tmp_path: Path) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    bindings = tuple(
        (
            name,
            "0" if name == "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES" else value,
        )
        for name, value in bindings
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(bindings)), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == ("VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",)


def test_candidate_json_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    encoded = _canonical(_candidate_payload(bindings))
    duplicate = (
        encoded[:-1]
        + ',"schema_version":"wan-production-qualification-candidate.v1"}'
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        duplicate, encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


def test_database_binding_does_not_percent_decode_into_another_path(
    tmp_path: Path,
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    original = root / "video-lab.sqlite3"
    encoded = root / "encoded%2fvideo-lab.sqlite3"
    decoded = root / "encoded" / "video-lab.sqlite3"
    decoded.parent.mkdir()
    shutil.copyfile(original, decoded)
    encoded.write_bytes(b"not a sqlite database")
    bindings = tuple(
        (
            name,
            str(encoded) if name == "VIDEO_LAB_WAN_ARTIFACT_DATABASE" else value,
        )
        for name, value in bindings
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(bindings)), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


def test_database_replacement_before_retained_capture_cannot_escape_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    database = root / "video-lab.sqlite3"
    outside = tmp_path / "outside.sqlite3"
    shutil.copyfile(database, outside)
    original_capture = RetainedSqliteAuthority.capture
    replaced = False

    def replacing_capture(
        cls: type[RetainedSqliteAuthority], path: Path
    ) -> RetainedSqliteAuthority:
        nonlocal replaced
        if path == database and not replaced:
            replaced = True
            path.unlink()
            path.symlink_to(outside)
        return original_capture(path)

    monkeypatch.setattr(
        RetainedSqliteAuthority, "capture", classmethod(replacing_capture)
    )

    result = discover_wan_production_authorities(_request(root))

    assert replaced
    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


@pytest.mark.parametrize(
    ("authority_name", "replacement"),
    [
        (
            "VIDEO_LAB_WAN_TOLERANCE_JSON",
            _canonical({"atol": 0.25, "rtol": 0.5}),
        ),
        ("VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES", "16777216"),
    ],
)
def test_self_attested_scientific_values_do_not_create_provenance(
    tmp_path: Path, authority_name: str, replacement: str
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    bindings = tuple(
        (name, replacement if name == authority_name else value)
        for name, value in bindings
    )
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(bindings)), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == (authority_name,)


def test_tampered_retained_output_binding_invalidates_compilation_authority(
    tmp_path: Path,
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    database = Path(dict(bindings)["VIDEO_LAB_WAN_ARTIFACT_DATABASE"])
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER guard_dataset_compilation_output_update_v48")
        connection.execute(
            "UPDATE dataset_compilation_outputs SET sha256=? WHERE role='batch'",
            ("0" * 64,),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
    retained = RetainedSqliteAuthority.capture(database)
    try:
        with retained.connect_readonly() as reader:
            assert reader.execute(
                "SELECT sha256 FROM dataset_compilation_outputs WHERE role='batch'"
            ).fetchone()[0] == "0" * 64
    finally:
        retained.close()

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


@pytest.mark.parametrize(
    ("column", "replacement"),
    [("receipt_id", "tampered-receipt-id"), ("attempt", 2)],
)
def test_closed_database_output_identity_must_match_canonical_receipt(
    tmp_path: Path, column: str, replacement: str | int
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    database = Path(dict(bindings)["VIDEO_LAB_WAN_ARTIFACT_DATABASE"])
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER guard_dataset_compilation_output_update_v48")
        connection.execute(
            f"UPDATE dataset_compilation_outputs SET {column}=?", (replacement,)
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
    retained = RetainedSqliteAuthority.capture(database)
    try:
        with retained.connect_readonly() as reader:
            observed = reader.execute(
                f"SELECT DISTINCT {column} FROM dataset_compilation_outputs"
            ).fetchall()
            assert tuple(row[0] for row in observed) == (replacement,)
    finally:
        retained.close()

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("operation_key", "tampered_qualification"),
        ("compiler_id", "tampered_compiler_v1"),
    ],
)
def test_closed_database_durable_operation_identity_must_match_canonical_evidence(
    tmp_path: Path, column: str, replacement: str
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    database = Path(dict(bindings)["VIDEO_LAB_WAN_ARTIFACT_DATABASE"])
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER guard_dataset_compilation_operation_authority_v48")
        connection.execute(
            f"UPDATE dataset_compilation_operations SET {column}=?", (replacement,)
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
    retained = RetainedSqliteAuthority.capture(database)
    try:
        with retained.connect_readonly() as reader:
            assert reader.execute(
                f"SELECT {column} FROM dataset_compilation_operations"
            ).fetchone()[0] == replacement
    finally:
        retained.close()

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


def test_tampered_retained_artifact_bytes_invalidate_compilation_authority(
    tmp_path: Path,
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    database = Path(dict(bindings)["VIDEO_LAB_WAN_ARTIFACT_DATABASE"])
    artifact_root = Path(dict(bindings)["VIDEO_LAB_WAN_ARTIFACT_ROOT"])
    with sqlite3.connect(database) as connection:
        relative = connection.execute(
            "SELECT artifact.relative_path FROM artifacts artifact "
            "JOIN dataset_compilation_outputs output ON output.artifact_id=artifact.id "
            "WHERE output.role='batch'"
        ).fetchone()
    assert relative is not None
    (artifact_root / str(relative[0])).write_bytes(b"tampered-output")

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


def test_candidate_growth_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    candidate = root / "wan-production-qualification-candidate.v1.json"
    identity = (candidate.stat().st_dev, candidate.stat().st_ino)
    original_pread = os.pread
    mutated = False

    def growing_pread(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal mutated
        payload = original_pread(descriptor, size, offset)
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == identity and not mutated:
            mutated = True
            with candidate.open("ab") as stream:
                stream.write(b" ")
        return payload

    monkeypatch.setattr(wan_qualification.os, "pread", growing_pread)

    result = discover_wan_production_authorities(_request(root))

    assert mutated
    assert result.status == "missing"
    assert result.bindings == ()


def test_candidate_in_place_mutation_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    candidate = root / "wan-production-qualification-candidate.v1.json"
    identity = (candidate.stat().st_dev, candidate.stat().st_ino)
    original_pread = os.pread
    mutated = False

    def mutating_pread(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal mutated
        payload = original_pread(descriptor, size, offset)
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == identity and not mutated:
            mutated = True
            with candidate.open("r+b") as stream:
                stream.write(b"[")
                stream.flush()
                os.fsync(stream.fileno())
        return payload

    monkeypatch.setattr(wan_qualification.os, "pread", mutating_pread)

    result = discover_wan_production_authorities(_request(root))

    assert mutated
    assert result.status == "missing"
    assert result.bindings == ()


def test_candidate_fifo_swap_is_opened_nonblocking_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    candidate = root / "wan-production-qualification-candidate.v1.json"
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == candidate.name and dir_fd is not None and not swapped:
            swapped = True
            candidate.unlink()
            os.mkfifo(candidate)
            assert flags & getattr(os, "O_NONBLOCK", 0)
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(wan_qualification.os, "open", swapping_open)

    result = discover_wan_production_authorities(_request(root))

    assert swapped
    assert result.status == "missing"
    assert result.bindings == ()


def test_hardlinked_candidate_is_rejected(tmp_path: Path) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    candidate = root / "wan-production-qualification-candidate.v1.json"
    outside = tmp_path / candidate.name
    candidate.replace(outside)
    os.link(outside, candidate)

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "missing"
    assert result.bindings == ()


def test_search_root_with_symlinked_ancestor_is_rejected(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    root, _bindings_value = _make_local_model_inventory(real_parent / "scope")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="wan_qualification_search_root_invalid"):
        discover_wan_production_authorities(_request(alias / root.name))


def test_missing_search_root_is_rejected_without_creation(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ValueError, match="wan_qualification_search_root_invalid"):
        discover_wan_production_authorities(_request(missing))

    assert not missing.exists()


def test_search_root_inode_swap_after_candidate_read_cannot_qualify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    displaced = tmp_path / "displaced"
    original_read = wan_qualification._read_regular_file_at
    swapped = False

    def swapping_read(
        parent_descriptor: int, name: str, expected: os.stat_result
    ) -> bytes | None:
        nonlocal swapped
        payload = original_read(parent_descriptor, name, expected)
        if name == "wan-production-qualification-candidate.v1.json" and not swapped:
            swapped = True
            root.rename(displaced)
            shutil.copytree(displaced, root)
        return payload

    monkeypatch.setattr(wan_qualification, "_read_regular_file_at", swapping_read)

    result = discover_wan_production_authorities(_request(root))

    assert swapped
    assert result.status == "missing"
    assert result.bindings == ()
    assert result.missing_roles == WAN_PRODUCTION_AUTHORITY_NAMES


def test_search_root_deleted_at_capture_boundary_is_not_recreated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    original_open = os.open
    deleted = False

    def deleting_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal deleted
        if (
            not deleted
            and dir_fd is not None
            and path == root.name
            and flags & os.O_DIRECTORY
        ):
            deleted = True
            shutil.rmtree(root)
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(wan_qualification.os, "open", deleting_open)

    with pytest.raises(ValueError, match="wan_qualification_search_root_invalid"):
        discover_wan_production_authorities(_request(root))

    assert deleted
    assert not root.exists()


def test_duplicate_search_root_inode_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _bindings_value = _make_local_model_inventory(tmp_path / "first")
    second = tmp_path / "second"
    second.mkdir()
    original_capture = wan_qualification._capture_existing_directory_authority

    def duplicate_capture(
        path: Path,
        *,
        error_code: str = "wan_canary_directory_authority_changed",
    ) -> RetainedDirectoryAuthority:
        return original_capture(first if path == second else path, error_code=error_code)

    monkeypatch.setattr(
        wan_qualification, "_capture_existing_directory_authority", duplicate_capture
    )
    request = WanQualificationSearchV1(
        schema_version="wan-production-qualification-search.v1",
        search_roots=tuple(sorted((first, second), key=str)),
        maximum_directories=20_000,
        maximum_files=200_000,
    )

    with pytest.raises(ValueError, match="wan_qualification_search_root_invalid"):
        discover_wan_production_authorities(request)


def test_nested_search_roots_are_rejected(tmp_path: Path) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")
    nested = root / "nested"
    nested.mkdir()
    request = WanQualificationSearchV1(
        schema_version="wan-production-qualification-search.v1",
        search_roots=(root, nested),
        maximum_directories=20_000,
        maximum_files=200_000,
    )

    with pytest.raises(ValueError, match="wan_qualification_search_root_invalid"):
        discover_wan_production_authorities(request)


@pytest.mark.parametrize(
    ("maximum_directories", "maximum_files"),
    [(250_001, 200_000), (20_000, 2_000_001)],
)
def test_search_limits_have_hard_professional_ceilings(
    tmp_path: Path, maximum_directories: int, maximum_files: int
) -> None:
    root = tmp_path / "scope"
    root.mkdir()
    with pytest.raises(ValidationError):
        WanQualificationSearchV1(
            schema_version="wan-production-qualification-search.v1",
            search_roots=(root,),
            maximum_directories=maximum_directories,
            maximum_files=maximum_files,
        )


def test_model_and_cache_inventory_never_use_rglob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bindings_value = _make_local_model_inventory(tmp_path / "scope")

    def forbidden_rglob(self: Path, pattern: str) -> object:
        raise AssertionError(f"unbounded rglob forbidden: {self}:{pattern}")

    monkeypatch.setattr(Path, "rglob", forbidden_rglob)

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "qualified"


def test_nonmatching_cache_refs_do_not_suppress_component_revision(
    tmp_path: Path,
) -> None:
    root, bindings = _make_local_model_inventory(tmp_path / "scope")
    cache = root / "cache"
    for path in sorted((cache / "snapshots").rglob("*"), reverse=True):
        path.rmdir()
    (cache / "snapshots").rmdir()
    (cache / "refs" / "main").write_text("b" * 40 + "\n", encoding="utf-8")
    component_revision = hashlib.sha256(root_manifest_digest(cache).encode()).hexdigest()[:40]
    values = dict(bindings)
    values["VIDEO_LAB_WAN_CACHE_REVISION"] = component_revision
    database = Path(values["VIDEO_LAB_WAN_ARTIFACT_DATABASE"])
    database.unlink()
    recipe_bindings = tuple(
        sorted(
            (name, value)
            for name, value in values.items()
            if name in _RECIPE_AUTHORITY_NAMES
        )
    )
    receipt_id, operation_id = _write_receipt_database(
        database, Path(values["VIDEO_LAB_WAN_ARTIFACT_ROOT"]), recipe_bindings
    )
    values["VIDEO_LAB_WAN_COMPILATION_OPERATION_ID"] = operation_id
    values["VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID"] = receipt_id
    bindings = tuple(sorted(values.items()))
    (root / "wan-production-qualification-candidate.v1.json").write_text(
        _canonical(_candidate_payload(bindings)), encoding="utf-8"
    )

    result = discover_wan_production_authorities(_request(root))

    assert result.status == "qualified"
    assert dict(result.bindings)["VIDEO_LAB_WAN_CACHE_REVISION"] == component_revision
