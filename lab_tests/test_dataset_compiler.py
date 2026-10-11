from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_lab.artifacts import ArtifactStore
from video_lab.curated_models import SequenceTransformRecipeV1
from video_lab.curated_source import DecodedFrame, decoded_pixel_digest
from video_lab.database import Database
from video_lab.dataset_compiler import (
    BackendOutput,
    DatasetCompilationError,
    DatasetCompilationService,
    TransformInput,
    TransformLimits,
    temporal_sample_indices,
    transform_sequence,
)
from video_lab.dataset_receipts import (
    CompilationClaim,
    CompilationConflict,
    DatasetCompilationRequest,
    InputBinding,
)


def recipe(**updates: object) -> SequenceTransformRecipeV1:
    payload: dict[str, object] = {
        "recipe_id": "test_recipe",
        "output_width": 4,
        "output_height": 4,
        "output_frame_count": 2,
        "temporal_stride_numerator": 1,
        "temporal_stride_denominator": 1,
        "resize_mode": "fit_pad",
        "interpolation": "nearest",
        "padding_color": (1, 2, 3),
        "seed": 7,
    }
    payload.update(updates)
    return SequenceTransformRecipeV1.model_validate(payload)


def frame(value: int, frame_id: str) -> TransformInput:
    pixels = np.full((2, 4, 3), value, dtype=np.uint8).tobytes()
    provisional = DecodedFrame(4, 2, 3, 8, "RGB", "0" * 64, pixels)
    digest = decoded_pixel_digest(provisional)
    return TransformInput(
        frame_id=frame_id,
        decoded=DecodedFrame(4, 2, 3, 8, "RGB", digest, pixels),
    )


def test_temporal_sample_indices_are_the_single_task4_authority() -> None:
    configured = recipe(
        output_frame_count=3,
        temporal_stride_numerator=3,
        temporal_stride_denominator=2,
        seed=41,
    )
    assert temporal_sample_indices(6, configured) == (0, 1, 3)
    transformed = transform_sequence(
        tuple(frame(index, f"frame_{index}") for index in range(6)), configured
    )
    assert tuple(item.frame_id for item in transformed.receipt.inputs) == (
        "frame_0",
        "frame_1",
        "frame_3",
    )
    exact = recipe(
        output_frame_count=3,
        temporal_stride_numerator=1_000_000_000,
        temporal_stride_denominator=999_999_937,
    )
    assert temporal_sample_indices(3, exact) == (0, 1, 2)
    assert temporal_sample_indices(3, exact) == temporal_sample_indices(3, exact)
    boundary = recipe(
        output_frame_count=1000,
        temporal_stride_numerator=1_000_000_000,
        temporal_stride_denominator=1_000_000_000,
    )
    assert temporal_sample_indices(1000, boundary)[-1] == 999


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_count", 1_000_001),
        ("temporal_stride_numerator", 10**400),
        ("temporal_stride_denominator", 10**400),
        ("temporal_stride_numerator", True),
        ("temporal_stride_denominator", np.int64(1)),
        ("output_frame_count", True),
    ],
)
def test_temporal_sample_indices_reject_unbounded_or_non_builtin_authority(
    field: str, value: object
) -> None:
    payload = recipe().model_dump()
    if field != "frame_count":
        payload[field] = value
    configured = SequenceTransformRecipeV1.model_construct(**payload)
    frame_count = value if field == "frame_count" else 4
    with pytest.raises(
        DatasetCompilationError,
        match="temporal recipe authority exceeds bounded integer policy",
    ):
        temporal_sample_indices(frame_count, configured)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    ["stale_digest", "mutated_bytes", "width", "channels", "length"],
)
def test_transform_rejects_noncanonical_decoded_frame(mutation: str) -> None:
    authentic = frame(10, "frame_a")
    decoded = authentic.decoded
    values = {
        "width": decoded.width,
        "height": decoded.height,
        "channels": decoded.channels,
        "bit_depth": decoded.bit_depth,
        "color_space": decoded.color_space,
        "pixel_digest": decoded.pixel_digest,
        "pixels": decoded.pixels,
    }
    if mutation == "stale_digest":
        values["pixel_digest"] = "0" * 64
    elif mutation == "mutated_bytes":
        values["pixels"] = bytes([decoded.pixels[0] ^ 1]) + decoded.pixels[1:]
    elif mutation == "width":
        values["width"] = 5
    elif mutation == "channels":
        values["channels"] = 4
    else:
        values["pixels"] = decoded.pixels[:-1]
    with pytest.raises(DatasetCompilationError, match="canonical decoded frame"):
        transform_sequence(
            (TransformInput("frame_a", DecodedFrame(**values)),),
            recipe(output_frame_count=1),
        )


def test_transform_is_deterministic_and_binds_crop_shape_and_digest() -> None:
    inputs = (frame(10, "frame_a"), frame(20, "frame_b"), frame(30, "frame_c"))
    left = transform_sequence(inputs, recipe())
    right = transform_sequence(inputs, recipe())
    assert left.receipt == right.receipt
    assert np.array_equal(left.frames, right.frames)
    assert left.frames.shape == (2, 4, 4, 3)
    assert not left.frames.flags.writeable
    assert [binding.frame_id for binding in left.receipt.inputs] == ["frame_a", "frame_b"]
    assert all(binding.crop_rectangle == (0, 0, 4, 2) for binding in left.receipt.inputs)
    assert left.receipt.operation_version == "sequence-transform.v1"


def test_transform_rejects_budget_before_pillow_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("allocation attempted")

    monkeypatch.setattr(Image, "fromarray", forbidden)
    with pytest.raises(DatasetCompilationError, match="budget"):
        transform_sequence(
            (frame(1, "frame_a"),), recipe(output_width=5), limits=TransformLimits(max_width=4)
        )
    assert not called


class Backend:
    compiler_id = "test_compiler_v1"

    def validate_recipe(self, recipe_bytes: bytes) -> None:
        assert recipe_bytes

    def compile_batch(self, source_bytes: bytes, recipe_bytes: bytes) -> tuple[BackendOutput, ...]:
        return (
            BackendOutput("batch", b"compiled", "application/octet-stream"),
            BackendOutput("manifest", b"{}", "application/json"),
        )


def _stage(store: ArtifactStore, tmp_path: Path, name: str, data: bytes, media: str):
    path = tmp_path / name
    path.write_bytes(data)
    return store.commit(store.stage(path, media), None)


def test_service_compiles_replays_inspects_and_reproduces(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, tmp_path / "artifacts")
    source = _stage(store, tmp_path, "source.json", b'{"dataset":1}', "application/json")
    recipe_artifact = _stage(store, tmp_path, "recipe.json", b'{"recipe":1}', "application/json")
    request = DatasetCompilationRequest(
        operation_key="compile_fixture",
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
            artifact_id=recipe_artifact.id,
            sha256=recipe_artifact.sha256,
            byte_size=recipe_artifact.byte_size,
            media_type=recipe_artifact.media_type,
        ),
        output_roles=("batch", "manifest"),
    )
    service = DatasetCompilationService(database, store, {Backend.compiler_id: Backend()})
    first = service.compile(request)
    assert service.compile(request) == first
    assert service.inspect(first.receipt_id) == first
    reproduced = service.reproduce(first.receipt_id, operation_key="reproduce_fixture")
    assert reproduced.reproduces_receipt_id == first.receipt_id
    assert [output.sha256 for output in reproduced.outputs] == [
        output.sha256 for output in first.outputs
    ]


class BadBackend(Backend):
    compiler_id = "bad_compiler_v1"

    def compile_batch(self, source_bytes: bytes, recipe_bytes: bytes) -> tuple[BackendOutput, ...]:
        return (BackendOutput("batch", b"compiled", "application/octet-stream"),)


def test_service_rejects_incomplete_backend_roles_without_publication(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, tmp_path / "artifacts")
    source = _stage(store, tmp_path, "source.json", b"source", "application/json")
    recipe_artifact = _stage(store, tmp_path, "recipe.json", b"recipe", "application/json")
    request = DatasetCompilationRequest(
        operation_key="bad_fixture",
        compiler_id="bad_compiler_v1",
        source=InputBinding(
            role="source_manifest",
            artifact_id=source.id,
            sha256=source.sha256,
            byte_size=source.byte_size,
            media_type=source.media_type,
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id=recipe_artifact.id,
            sha256=recipe_artifact.sha256,
            byte_size=recipe_artifact.byte_size,
            media_type=recipe_artifact.media_type,
        ),
        output_roles=("batch", "manifest"),
    )
    service = DatasetCompilationService(database, store, {BadBackend.compiler_id: BadBackend()})
    with pytest.raises(DatasetCompilationError, match="output roles"):
        service.compile(request)
    with database.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM dataset_compilation_outputs").fetchone()[0]
            == 0
        )


def test_service_cancels_only_after_complete_backend_batch(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, tmp_path / "artifacts")
    source = _stage(store, tmp_path, "source.json", b"source", "application/json")
    recipe_artifact = _stage(store, tmp_path, "recipe.json", b"recipe", "application/json")
    request = DatasetCompilationRequest(
        operation_key="cancel_fixture",
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
            artifact_id=recipe_artifact.id,
            sha256=recipe_artifact.sha256,
            byte_size=recipe_artifact.byte_size,
            media_type=recipe_artifact.media_type,
        ),
        output_roles=("batch", "manifest"),
    )
    service = DatasetCompilationService(
        database, store, {Backend.compiler_id: Backend()}, cancelled=lambda: True
    )
    with pytest.raises(DatasetCompilationError, match="cancelled at batch boundary"):
        service.compile(request)
    assert service.repository.get("cancel_fixture").status == "cancelled"


def test_reproduction_mismatch_fails_without_receipt_or_link(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, tmp_path / "artifacts")
    source = _stage(store, tmp_path, "source.json", b"source", "application/json")
    recipe_artifact = _stage(store, tmp_path, "recipe.json", b"recipe", "application/json")
    backend = Backend()
    request = DatasetCompilationRequest(
        operation_key="original_fixture",
        compiler_id=backend.compiler_id,
        source=InputBinding(
            role="source_manifest",
            artifact_id=source.id,
            sha256=source.sha256,
            byte_size=source.byte_size,
            media_type=source.media_type,
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id=recipe_artifact.id,
            sha256=recipe_artifact.sha256,
            byte_size=recipe_artifact.byte_size,
            media_type=recipe_artifact.media_type,
        ),
        output_roles=("batch", "manifest"),
    )
    service = DatasetCompilationService(database, store, {backend.compiler_id: backend})
    original = service.compile(request)
    backend.compile_batch = lambda source_bytes, recipe_bytes: (
        BackendOutput("batch", b"changed", "application/octet-stream"),
        BackendOutput("manifest", b"{}", "application/json"),
    )
    with pytest.raises(DatasetCompilationError, match="reproduction output mismatch"):
        service.reproduce(original.receipt_id, operation_key="reproduce_bad")
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,receipt_id FROM dataset_compilation_operations WHERE operation_key=?",
            ("reproduce_bad",),
        ).fetchone()
        assert tuple(row) == ("failed", None)
        assert (
            connection.execute("SELECT count(*) FROM dataset_compilation_reproductions").fetchone()[
                0
            ]
            == 0
        )


def test_identical_bytes_with_distinct_media_have_distinct_artifacts(tmp_path: Path) -> None:
    class SameBytesBackend(Backend):
        compiler_id = "same_bytes_v1"

        def compile_batch(
            self, source_bytes: bytes, recipe_bytes: bytes
        ) -> tuple[BackendOutput, ...]:
            return (
                BackendOutput("batch", b"same", "application/octet-stream"),
                BackendOutput("manifest", b"same", "application/json"),
            )

    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, tmp_path / "artifacts")
    source = _stage(store, tmp_path, "source.json", b"source", "application/json")
    recipe_artifact = _stage(store, tmp_path, "recipe.json", b"recipe", "application/json")
    backend = SameBytesBackend()
    request = DatasetCompilationRequest(
        operation_key="same_bytes",
        compiler_id=backend.compiler_id,
        source=InputBinding(
            role="source_manifest",
            artifact_id=source.id,
            sha256=source.sha256,
            byte_size=source.byte_size,
            media_type=source.media_type,
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id=recipe_artifact.id,
            sha256=recipe_artifact.sha256,
            byte_size=recipe_artifact.byte_size,
            media_type=recipe_artifact.media_type,
        ),
        output_roles=("batch", "manifest"),
    )
    receipt = DatasetCompilationService(database, store, {backend.compiler_id: backend}).compile(
        request
    )
    assert len({output.artifact_id for output in receipt.outputs}) == 2


def test_cleanup_failure_attempts_every_stage_and_exact_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, tmp_path / "artifacts")
    source = _stage(store, tmp_path, "source.json", b"source", "application/json")
    recipe_artifact = _stage(store, tmp_path, "recipe.json", b"recipe", "application/json")
    request = DatasetCompilationRequest(
        operation_key="cleanup_retry",
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
            artifact_id=recipe_artifact.id,
            sha256=recipe_artifact.sha256,
            byte_size=recipe_artifact.byte_size,
            media_type=recipe_artifact.media_type,
        ),
        output_roles=("batch", "manifest"),
    )
    service = DatasetCompilationService(database, store, {Backend.compiler_id: Backend()})
    original_discard = store.discard
    discarded: list[object] = []

    raw_messages = [
        "/private/tmp/secret-a/11111111-1111-1111-1111-111111111111",
        "/Volumes/private/secret-b/22222222-2222-2222-2222-222222222222",
        "/private/tmp/secret-c/33333333-3333-3333-3333-333333333333",
        "/Volumes/private/secret-d/44444444-4444-4444-4444-444444444444",
    ]

    def broken_discard(item: object) -> None:
        discarded.append(item)
        message_index = len(discarded) - 1
        ordinal = message_index % 2
        if ordinal == 0:
            raise PermissionError(13, raw_messages[message_index])
        raise OSError(5, raw_messages[message_index])

    monkeypatch.setattr(
        store,
        "commit_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish failed")),
    )
    monkeypatch.setattr(store, "discard", broken_discard)
    with pytest.raises(DatasetCompilationError, match="cleanup failure"):
        service.compile(request)
    assert len(discarded) == 2
    operation = service.repository.get("cleanup_retry")
    assert operation.status == "retryable"
    assert len(operation.cleanup_failures) == 1
    first_digest = operation.cleanup_failures[0].failure_digest
    assert all(message not in operation.cleanup_failures[0].code for message in raw_messages)
    stale = CompilationClaim(
        operation_id=operation.operation_id,
        operation_key=operation.operation_key,
        request_digest=operation.request_digest,
        claimed=True,
        attempt=1,
        claimant_id="local_compiler",
        claim_token=operation.cleanup_failures[0].claim_token,
    )
    with pytest.raises(DatasetCompilationError, match="cleanup failure") as second_error:
        service.compile(request)
    second_operation = service.repository.get("cleanup_retry")
    assert second_operation.cleanup_failures[1].failure_digest == first_digest
    assert all(message not in str(second_error.value) for message in raw_messages)
    monkeypatch.undo()
    monkeypatch.setattr(store, "discard", original_discard)
    receipt = service.compile(request)
    assert receipt.attempt == 3
    with pytest.raises(CompilationConflict, match="claim mismatch"):
        service.repository.cancel(stale, evidence_digest="f" * 64)
