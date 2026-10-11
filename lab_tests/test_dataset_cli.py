from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

from video_lab.artifacts import ArtifactIntegrityError, ArtifactStore
from video_lab.cli import app
from video_lab.curated_models import (
    CuratedDatasetManifestV1,
    CuratedFrameV1,
    CuratedSequenceV1,
    DatasetSplitPolicyV1,
    SequenceTransformRecipeV1,
    StructuredCaptionV1,
)
from video_lab.curated_source import DecodedFrame, decoded_pixel_digest
from video_lab.curated_validation import ValidationThresholdsV1
from video_lab.database import Database
from video_lab.dataset_cli import (
    DatasetApplicationService,
    DatasetValidationRequest,
    InspectRequest,
    ReproduceRequest,
    run_command,
)
from video_lab.dataset_compiler import (
    BackendOutput,
    DatasetCompilationError,
    DatasetCompilationService,
)
from video_lab.dataset_receipts import DatasetCompilationRequest, InputBinding
from video_lab.domain import canonical_json

runner = CliRunner()


def _pixel_digest(array: np.ndarray) -> str:
    frame = DecodedFrame(array.shape[1], array.shape[0], 3, 8, "RGB", "0" * 64, array.tobytes())
    return decoded_pixel_digest(frame)


def _write_frame(
    root: Path, name: str, array: np.ndarray, index: int
) -> tuple[CuratedFrameV1, dict[str, object]]:
    path = root / f"{name}.png"
    Image.fromarray(array, "RGB").save(path, format="PNG")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    frame = CuratedFrameV1(
        frame_id=f"frame_{name}",
        frame_index=index,
        timestamp_numerator=index,
        timestamp_denominator=8,
        source_path=path.name,
        source_digest=digest,
        pixel_digest=_pixel_digest(array),
        review_state="accepted",
        reviewer_evidence=("review_one",),
    )
    binding = {
        "frame_id": frame.frame_id,
        "artifact_id": f"artifact_{name}",
        "relative_path": path.name,
        "sha256": digest,
        "byte_size": len(raw),
        "media_type": "image/png",
    }
    return frame, binding


def _manifest(sequences: tuple[CuratedSequenceV1, ...]) -> CuratedDatasetManifestV1:
    return CuratedDatasetManifestV1(
        dataset_id="dataset_cli",
        sequences=sequences,
        split_policy=DatasetSplitPolicyV1(
            policy_version="policy_one",
            seed=7,
            train_numerator=8,
            validation_numerator=1,
            holdout_numerator=1,
            denominator=10,
        ),
        transform_recipe=SequenceTransformRecipeV1(
            recipe_id="recipe_one",
            output_width=8,
            output_height=8,
            output_frame_count=1,
            temporal_stride_numerator=1,
            temporal_stride_denominator=1,
            interpolation="nearest",
            padding_color=(0, 0, 0),
            seed=9,
        ),
    )


def _sequence(
    name: str, frames: tuple[CuratedFrameV1, ...], *, attributes: tuple[str, ...] = ("blue eyes",)
) -> CuratedSequenceV1:
    return CuratedSequenceV1(
        sequence_id=f"sequence_{name}",
        identity_id="hero",
        source_lineage=f"lineage_{name}",
        split="train",
        caption=StructuredCaptionV1(
            character_token="hero", immutable_attributes=attributes, action="walking"
        ),
        frames=frames,
    )


def _validation_request(root: Path, state: str) -> Path:
    rng = np.random.default_rng(42)
    left = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    first, first_binding = _write_frame(root, "one", left, 0)
    bindings = [first_binding]
    if state == "accepted":
        sequences = (_sequence("a", (first,)),)
    elif state == "rejected":
        second, second_binding = _write_frame(root, "two", left.copy(), 1)
        bindings.append(second_binding)
        sequences = (_sequence("a", (first, second)),)
    else:
        right = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        second, second_binding = _write_frame(root, "two", right, 0)
        bindings.append(second_binding)
        sequences = (
            _sequence("a", (first,)),
            _sequence("b", (second,), attributes=("green eyes",)),
        )
    request = {
        "schema_version": "dataset-validation-request.v1",
        "operation_key": f"validate_{state}",
        "manifest": _manifest(sequences).model_dump(mode="json"),
        "frame_bindings": bindings,
        "thresholds": ValidationThresholdsV1(
            perceptual_duplicate_hamming_max=0,
            blur_variance_min="0",
            scene_cut_histogram_max="1",
            temporal_motion_max="1",
        ).model_dump(mode="json"),
        "human_dispositions": {},
    }
    path = root / "request.json"
    path.write_text(canonical_json(request))
    return path


@pytest.mark.parametrize(
    ("state", "exit_code", "reason"),
    [
        ("accepted", 0, None),
        ("rejected", 65, "exact_duplicate"),
        ("needs_review", 66, "identity_inconsistency"),
    ],
)
def test_dataset_validate_runs_real_three_state_workflow(
    tmp_path: Path, state: str, exit_code: int, reason: str | None
) -> None:
    _validation_request(tmp_path, state)
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "validate",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--source-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == exit_code
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["data"]["status"] == state
    assert payload["data"]["validation_receipt"]["report"]["state"] == state
    if reason is not None:
        assert reason in payload["data"]["reason_codes"]


def test_dataset_validate_human_output_is_state(tmp_path: Path) -> None:
    _validation_request(tmp_path, "accepted")
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "validate",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--source-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert result.stdout == "accepted\n"


def test_validation_receipt_replays_conflicts_and_rejects_wrong_split(tmp_path: Path) -> None:
    request_path = _validation_request(tmp_path, "accepted")
    arguments = [
        "--database",
        str(tmp_path / "db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "dataset",
        "validate",
        "--request-root",
        str(tmp_path),
        "--request-file",
        "request.json",
        "--source-root",
        str(tmp_path),
        "--json",
    ]
    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)
    assert first.exit_code == second.exit_code == 0
    assert first.stdout == second.stdout

    document = json.loads(request_path.read_text())
    document["thresholds"]["blur_variance_min"] = "1.0000"
    request_path.write_text(canonical_json(document))
    conflict = runner.invoke(app, arguments)
    assert conflict.exit_code == 67

    document["operation_key"] = "validate_wrong_split"
    document["manifest"]["sequences"][0]["split"] = "holdout"
    request_path.write_text(canonical_json(document))
    wrong_split = runner.invoke(app, arguments)
    assert wrong_split.exit_code == 65


def test_validation_receipt_replays_byte_identically_across_processes(tmp_path: Path) -> None:
    _validation_request(tmp_path, "accepted")
    command = [
        str(Path(sys.executable).parent / "video-lab"),
        "--database",
        str(tmp_path / "db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "dataset",
        "validate",
        "--request-root",
        str(tmp_path),
        "--request-file",
        "request.json",
        "--source-root",
        str(tmp_path),
        "--json",
    ]
    first = subprocess.run(command, check=False, capture_output=True)
    second = subprocess.run(command, check=False, capture_output=True)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["data"]["validation_receipt"]["state"] == "accepted"


class _HashFailureInspectionApplication(DatasetApplicationService):
    def _inspect_authoritative(self, receipt_id: str) -> object:
        del receipt_id
        raise ArtifactIntegrityError("artifact verification cancelled or mismatched")


class _PostStreamCancellationApplication(DatasetApplicationService):
    def __init__(self, database: Database, artifacts: ArtifactStore, artifact_id: str) -> None:
        calls = 0

        def cancelled() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 2

        super().__init__(database=database, artifacts=artifacts, cancelled=cancelled)
        self.artifact_id = artifact_id

    def _inspect_authoritative(self, receipt_id: str) -> object:
        del receipt_id
        assert self.database is not None and self.artifacts is not None
        with self.database.connect() as connection:
            artifact = self.artifacts.get(self.artifact_id)
            with self.artifacts.open_verified_with_connection(
                connection,
                artifact.id,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
            ) as handle:
                assert len(handle.read(artifact.byte_size)) == artifact.byte_size
                self.artifacts.reverify_open_handle(
                    connection, handle, cancelled=self.cancelled
                )
        raise AssertionError("cancelled post-stream rehash must not publish success")


def test_inspection_translates_hash_cancellation_but_preserves_integrity() -> None:
    cancelled = _HashFailureInspectionApplication(cancelled=lambda: True)
    with pytest.raises(DatasetCompilationError, match="inspection cancelled"):
        cancelled.inspect("receipt_one")

    integrity = _HashFailureInspectionApplication(cancelled=lambda: False)
    with pytest.raises(ArtifactIntegrityError, match="mismatched"):
        integrity.inspect("receipt_one")


def test_inspection_translates_real_post_stream_hash_cancellation(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    artifact = artifacts.commit(
        artifacts.stage_bytes(b"x" * (2 * 1024 * 1024), "application/octet-stream"), None
    )
    application = _PostStreamCancellationApplication(database, artifacts, artifact.id)
    with pytest.raises(DatasetCompilationError, match="inspection cancelled"):
        application.inspect("receipt_one")


@pytest.mark.parametrize(
    ("cancelled", "exit_code", "error_code"),
    [(True, 69, "cancelled"), (False, 70, "integrity_failure")],
)
def test_inspection_hash_failure_cli_envelope_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cancelled: bool,
    exit_code: int,
    error_code: str,
) -> None:
    if cancelled:
        database = Database(tmp_path / "lab.sqlite3")
        database.migrate()
        artifacts = ArtifactStore(database, tmp_path / "stored")
        artifact = artifacts.commit(
            artifacts.stage_bytes(
                b"x" * (2 * 1024 * 1024), "application/octet-stream"
            ),
            None,
        )
        application: DatasetApplicationService = _PostStreamCancellationApplication(
            database, artifacts, artifact.id
        )
    else:
        application = _HashFailureInspectionApplication(cancelled=lambda: False)
    monkeypatch.setattr(
        "video_lab.dataset_cli.load_request", lambda *_: InspectRequest(receipt_id="receipt_one")
    )
    monkeypatch.setattr(
        "video_lab.dataset_cli.build_dataset_application", lambda **_: application
    )
    with pytest.raises(typer.Exit) as raised:
        run_command(
            command="inspect",
            database_path=tmp_path / "db",
            artifact_root=tmp_path / "artifacts",
            request_root=tmp_path,
            request_file="request.json",
            json_output=True,
        )
    assert raised.value.exit_code == exit_code
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["error"]["code"] == error_code


def test_validation_replay_rejects_tampered_report_artifact(tmp_path: Path) -> None:
    request_path = _validation_request(tmp_path, "accepted")
    request = DatasetValidationRequest.model_validate_json(request_path.read_bytes(), strict=True)
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    application = DatasetApplicationService(database=database, artifacts=artifacts)
    receipt = application.validate(request, source_root=tmp_path)
    artifacts.resolve(receipt.report_artifact.artifact_id).write_bytes(b"{}")
    with pytest.raises(Exception, match="integrity|mismatch|size|changed"):
        application.validate(request, source_root=tmp_path)


class _Backend:
    compiler_id = "test_compiler_v1"

    def validate_recipe(self, recipe_bytes: bytes) -> None:
        assert recipe_bytes

    def compile_batch(self, source_bytes: bytes, recipe_bytes: bytes) -> tuple[BackendOutput, ...]:
        return (
            BackendOutput("batch", b"compiled", "application/octet-stream"),
            BackendOutput(
                "manifest",
                canonical_json(
                    {
                        "inventory": [{"name": "latent"}],
                        "provenance": {"source": hashlib.sha256(source_bytes).hexdigest()},
                    }
                ).encode(),
                "application/json",
            ),
        )


def _stage(store: ArtifactStore, data: bytes, media_type: str):
    return store.commit(store.stage_bytes(data, media_type), None)


def test_compile_inspect_reproduce_application_integration(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    source = _stage(
        artifacts, canonical_json({"transform_receipt": {"seed": 9}}).encode(), "application/json"
    )
    recipe = _stage(artifacts, canonical_json({"recipe": "one"}).encode(), "application/json")
    request = DatasetCompilationRequest(
        operation_key="compile_cli",
        compiler_id="test_compiler_v1",
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
        output_roles=("batch", "manifest"),
    )
    compilation = DatasetCompilationService(database, artifacts, {_Backend.compiler_id: _Backend()})
    first = compilation.compile(request)
    application = DatasetApplicationService(compilation, database=database, artifacts=artifacts)

    with pytest.raises(DatasetCompilationError, match="unsupported inspection compiler"):
        application.inspect(first.receipt_id)
    reproduced = compilation.reproduce(first.receipt_id, operation_key="reproduce_cli")

    assert reproduced.reproduces_receipt_id == first.receipt_id


@pytest.mark.parametrize("command", ["validate", "compile", "inspect", "reproduce"])
def test_each_cli_handler_invokes_exactly_one_application_method(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    class Result:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"result": command}

    class Compilation:
        def compile(self, request: object) -> Result:
            calls.append("compile")
            return Result()

        def reproduce(self, receipt_id: str, *, operation_key: str) -> Result:
            calls.append("reproduce")
            return Result()

    class Application:
        compilation = Compilation()

        def validate(self, request: object, *, source_root: Path) -> object:
            calls.append("validate")
            report = SimpleNamespace(
                state="accepted",
                frame_findings=(),
                sequence_findings=(),
                model_dump=lambda **_: {"state": "accepted"},
            )
            return SimpleNamespace(
                state="accepted",
                report=report,
                model_dump=lambda **_: {"state": "accepted", "report": {"state": "accepted"}},
            )

        def inspect(self, receipt_id: str) -> Result:
            calls.append("inspect")
            return Result()

    parsed: object = (
        DatasetValidationRequest.model_construct()
        if command == "validate"
        else DatasetCompilationRequest.model_construct()
        if command == "compile"
        else InspectRequest(receipt_id="receipt_one")
        if command == "inspect"
        else ReproduceRequest(receipt_id="receipt_one", operation_key="reproduce_one")
    )
    monkeypatch.setattr("video_lab.dataset_cli.load_request", lambda *_: parsed)
    monkeypatch.setattr(
        "video_lab.dataset_cli.build_dataset_application", lambda **_: Application()
    )
    run_command(
        command=command,
        database_path=tmp_path / "db",
        artifact_root=tmp_path / "a",
        request_root=tmp_path,
        request_file="request.json",
        json_output=True,
        source_root=tmp_path if command == "validate" else None,
    )
    assert calls == [command]
    assert json.loads(capsys.readouterr().out)["ok"] is True
