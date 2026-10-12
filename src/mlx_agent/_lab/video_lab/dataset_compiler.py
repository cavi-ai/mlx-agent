"""Deterministic transforms and trainer-neutral compilation service."""

from __future__ import annotations

import errno
import hashlib
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from video_lab.artifacts import Artifact, ArtifactError, ArtifactStore, StagedArtifact
from video_lab.curated_models import SequenceTransformRecipeV1
from video_lab.curated_source import CuratedSourceError, DecodedFrame, validate_decoded_frame
from video_lab.database import Database
from video_lab.dataset_receipts import (
    CompilationClaim,
    CompilationOutput,
    DatasetCompilationReceipt,
    DatasetCompilationRepository,
    DatasetCompilationRequest,
)
from video_lab.domain import canonical_json, utc_now


class DatasetCompilationError(RuntimeError):
    """Compilation failed before a complete receipt could be published."""


class _CompilationCancelled(DatasetCompilationError):
    pass


class _CleanupFailure(DatasetCompilationError):
    def __init__(self, failures: tuple[str, ...]) -> None:
        self.failures = failures
        super().__init__("compilation cleanup failure: " + "; ".join(failures))


@dataclass(frozen=True)
class BackendOutput:
    role: str
    data: bytes
    media_type: str


class DatasetCompilerBackend(Protocol):
    compiler_id: str

    def validate_recipe(self, recipe_bytes: bytes) -> None: ...

    def compile_batch(
        self, source_bytes: bytes, recipe_bytes: bytes
    ) -> tuple[BackendOutput, ...]: ...


@dataclass(frozen=True)
class TransformInput:
    frame_id: str
    decoded: DecodedFrame


@dataclass(frozen=True)
class TransformLimits:
    max_width: int = 4096
    max_height: int = 4096
    max_pixels_per_frame: int = 4096 * 4096
    max_frame_count: int = 256
    max_input_bytes: int = 4 * 1024 * 1024 * 1024
    max_output_bytes: int = 4 * 1024 * 1024 * 1024


DEFAULT_TRANSFORM_LIMITS = TransformLimits()
_MAX_TEMPORAL_FRAME_COUNT = 1_000_000
_MAX_TEMPORAL_FACTOR = 1_000_000_000
_MAX_TEMPORAL_PRODUCT = 1_000_000_000_000


class TransformInputReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    frame_id: str
    input_pixel_digest: str
    crop_rectangle: tuple[int, int, int, int]
    output_shape: tuple[int, int, int]
    output_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class TransformReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    operation_version: str = "sequence-transform.v1"
    recipe_id: str
    seed: int
    inputs: tuple[TransformInputReceipt, ...]
    decoded_output_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TransformedSequence:
    frames: np.ndarray
    receipt: TransformReceipt


_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def transformed_array_digest(array: np.ndarray) -> str:
    header = f"{array.shape}:{array.dtype.str}".encode()
    return hashlib.sha256(b"transformed-frame.v1\0" + header + b"\0" + array.tobytes()).hexdigest()


def validate_transformed_sequence(
    frames: np.ndarray,
    receipt: TransformReceipt,
    recipe: SequenceTransformRecipeV1,
) -> None:
    """Revalidate the exact Task 4 transformed array and receipt authority."""
    expected_shape = (
        recipe.output_frame_count,
        recipe.output_height,
        recipe.output_width,
        3,
    )
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.shape != expected_shape
        or receipt.operation_version != "sequence-transform.v1"
        or receipt.recipe_id != recipe.recipe_id
        or receipt.seed != recipe.seed
        or len(receipt.inputs) != recipe.output_frame_count
        or receipt.decoded_output_digest != transformed_array_digest(frames)
    ):
        raise DatasetCompilationError("transformed sequence receipt mismatch")
    frame_ids: set[str] = set()
    for index, binding in enumerate(receipt.inputs):
        frame = frames[index]
        left, top, right, bottom = binding.crop_rectangle
        if (
            binding.frame_id in frame_ids
            or not binding.frame_id
            or len(binding.input_pixel_digest) != 64
            or any(character not in "0123456789abcdef" for character in binding.input_pixel_digest)
            or not (0 <= left < right and 0 <= top < bottom)
            or binding.output_shape != frame.shape
            or binding.output_digest != transformed_array_digest(frame)
        ):
            raise DatasetCompilationError("transformed sequence frame binding mismatch")
        frame_ids.add(binding.frame_id)


def transform_sequence(
    frames: tuple[TransformInput, ...],
    recipe: SequenceTransformRecipeV1,
    *,
    limits: TransformLimits = DEFAULT_TRANSFORM_LIMITS,
) -> TransformedSequence:
    if not frames:
        raise DatasetCompilationError("transform requires accepted decoded frames")
    try:
        for item in frames:
            validate_decoded_frame(item.decoded)
    except CuratedSourceError:
        raise DatasetCompilationError("transform requires a canonical decoded frame") from None
    output_pixels = recipe.output_width * recipe.output_height
    input_bytes = sum(
        int(pixels.nbytes) if isinstance(pixels, np.ndarray) else len(pixels)
        for pixels in (frame.decoded.pixels for frame in frames)
    )
    output_bytes = output_pixels * 3 * recipe.output_frame_count
    if (
        recipe.output_width > limits.max_width
        or recipe.output_height > limits.max_height
        or output_pixels > limits.max_pixels_per_frame
        or recipe.output_frame_count > limits.max_frame_count
        or input_bytes > limits.max_input_bytes
        or output_bytes > limits.max_output_bytes
    ):
        raise DatasetCompilationError("transform budget exceeded")
    selected_indices = temporal_sample_indices(len(frames), recipe)
    transformed: list[np.ndarray] = []
    bindings: list[TransformInputReceipt] = []
    for index in selected_indices:
        item = frames[index]
        if isinstance(item.decoded.pixels, bytes):
            source = np.frombuffer(item.decoded.pixels, dtype=np.uint8).reshape(
                item.decoded.height,
                item.decoded.width,
                item.decoded.channels,
            )
        else:
            source = np.asarray(item.decoded.pixels)
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise DatasetCompilationError("transform input must be owned RGB uint8")
        height, width, _ = source.shape
        image = Image.fromarray(source, mode="RGB")
        if recipe.resize_mode == "center_crop":
            target_ratio = recipe.output_width / recipe.output_height
            source_ratio = width / height
            if source_ratio > target_ratio:
                crop_width = max(1, int(height * target_ratio))
                left = (width - crop_width) // 2
                box = (left, 0, left + crop_width, height)
            else:
                crop_height = max(1, int(width / target_ratio))
                top = (height - crop_height) // 2
                box = (0, top, width, top + crop_height)
            output_image = image.crop(box).resize(
                (recipe.output_width, recipe.output_height), _RESAMPLE[recipe.interpolation]
            )
        else:
            box = (0, 0, width, height)
            scale = min(recipe.output_width / width, recipe.output_height / height)
            resized = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                _RESAMPLE[recipe.interpolation],
            )
            output_image = Image.new(
                "RGB", (recipe.output_width, recipe.output_height), recipe.padding_color
            )
            output_image.paste(
                resized,
                (
                    (recipe.output_width - resized.width) // 2,
                    (recipe.output_height - resized.height) // 2,
                ),
            )
        output = np.array(output_image, dtype=np.uint8, copy=True)
        output.flags.writeable = False
        transformed.append(output)
        bindings.append(
            TransformInputReceipt(
                frame_id=item.frame_id,
                input_pixel_digest=item.decoded.pixel_digest,
                crop_rectangle=box,
                output_shape=output.shape,
                output_digest=transformed_array_digest(output),
            )
        )
    batch = np.stack(transformed)
    batch.flags.writeable = False
    receipt = TransformReceipt(
        recipe_id=recipe.recipe_id,
        seed=recipe.seed,
        inputs=tuple(bindings),
        decoded_output_digest=transformed_array_digest(batch),
    )
    return TransformedSequence(batch, receipt)


def temporal_sample_indices(
    frame_count: int, recipe: SequenceTransformRecipeV1
) -> tuple[int, ...]:
    """Return Task 4's exact deterministic temporal selection."""
    numerator = recipe.temporal_stride_numerator
    denominator = recipe.temporal_stride_denominator
    output_count = recipe.output_frame_count
    if (
        type(frame_count) is not int
        or type(numerator) is not int
        or type(denominator) is not int
        or type(output_count) is not int
        or not 0 < frame_count <= _MAX_TEMPORAL_FRAME_COUNT
        or not 0 < numerator <= _MAX_TEMPORAL_FACTOR
        or not 0 < denominator <= _MAX_TEMPORAL_FACTOR
        or not 0 < output_count <= _MAX_TEMPORAL_FRAME_COUNT
        or output_count * numerator > _MAX_TEMPORAL_PRODUCT
    ):
        raise DatasetCompilationError("temporal recipe authority exceeds bounded integer policy")
    selected = tuple((index * numerator) // denominator for index in range(output_count))
    if len(set(selected)) != len(selected) or max(selected) >= frame_count:
        raise DatasetCompilationError("temporal recipe exceeds available accepted frames")
    return selected


class DatasetCompilationService:
    def __init__(
        self,
        database: Database,
        artifacts: ArtifactStore,
        backends: Mapping[str, DatasetCompilerBackend],
        *,
        claimant_id: str = "local_compiler",
        lease_seconds: int = 300,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        self.database = database
        self.artifacts = artifacts
        self.backends = dict(backends)
        self.repository = DatasetCompilationRepository(database)
        self.claimant_id = claimant_id
        self.lease_seconds = lease_seconds
        self.cancelled = cancelled

    def validate(self, request: DatasetCompilationRequest) -> str:
        backend = self.backends.get(request.compiler_id)
        if backend is None or backend.compiler_id != request.compiler_id:
            raise DatasetCompilationError("compiler backend is unavailable")
        with self.database.connect() as connection, ExitStack() as stack:
            source = stack.enter_context(
                self.artifacts.open_verified_with_connection(
                    connection,
                    request.source.artifact_id,
                    sha256=request.source.sha256,
                    byte_size=request.source.byte_size,
                    media_type=request.source.media_type,
                )
            )
            recipe = stack.enter_context(
                self.artifacts.open_verified_with_connection(
                    connection,
                    request.recipe.artifact_id,
                    sha256=request.recipe.sha256,
                    byte_size=request.recipe.byte_size,
                    media_type=request.recipe.media_type,
                )
            )
            recipe_bytes = recipe.read()
            backend.validate_recipe(recipe_bytes)
            self.artifacts.reverify_open_handle(connection, source)
            self.artifacts.reverify_open_handle(connection, recipe)
        return request.digest

    def compile(self, request: DatasetCompilationRequest) -> DatasetCompilationReceipt:
        return self._compile(request, expected_outputs=None)

    def _compile(
        self,
        request: DatasetCompilationRequest,
        *,
        expected_outputs: tuple[tuple[str, str, int, str], ...] | None,
    ) -> DatasetCompilationReceipt:
        existing = self.repository.begin(request)
        if existing.status == "completed" and existing.receipt_id is not None:
            return self.inspect(existing.receipt_id)
        backend = self.backends.get(request.compiler_id)
        if backend is None or backend.compiler_id != request.compiler_id:
            raise DatasetCompilationError("compiler backend is unavailable")
        with self.database.connect() as connection, ExitStack() as stack:
            source = stack.enter_context(
                self.artifacts.open_verified_with_connection(
                    connection,
                    request.source.artifact_id,
                    sha256=request.source.sha256,
                    byte_size=request.source.byte_size,
                    media_type=request.source.media_type,
                )
            )
            recipe = stack.enter_context(
                self.artifacts.open_verified_with_connection(
                    connection,
                    request.recipe.artifact_id,
                    sha256=request.recipe.sha256,
                    byte_size=request.recipe.byte_size,
                    media_type=request.recipe.media_type,
                )
            )
            source_bytes, recipe_bytes = source.read(), recipe.read()
            backend.validate_recipe(recipe_bytes)
            claim = self.repository.claim(
                request.operation_key,
                claimant_id=self.claimant_id,
                lease_seconds=self.lease_seconds,
            )
            if not claim.claimed:
                raise DatasetCompilationError("compilation is already in progress")
            try:
                produced = backend.compile_batch(source_bytes, recipe_bytes)
                self._validate_outputs(request, produced)
                actual_outputs = tuple(
                    (
                        output.role,
                        hashlib.sha256(output.data).hexdigest(),
                        len(output.data),
                        output.media_type,
                    )
                    for output in sorted(produced, key=lambda item: item.role)
                )
                if expected_outputs is not None and actual_outputs != expected_outputs:
                    raise DatasetCompilationError("reproduction output mismatch")
                if self.cancelled():
                    raise _CompilationCancelled("compilation cancelled at batch boundary")
                self.artifacts.reverify_open_handle(connection, source)
                self.artifacts.reverify_open_handle(connection, recipe)
                return self._publish(request, claim, produced)
            except _CompilationCancelled as error:
                evidence = hashlib.sha256(str(error).encode()).hexdigest()
                self.repository.cancel(claim, evidence_digest=evidence)
                raise
            except _CleanupFailure as error:
                evidence = hashlib.sha256("\n".join(error.failures).encode()).hexdigest()
                self.repository.retryable_failure(
                    claim, code="cleanup_failed", failure_digest=evidence
                )
                raise
            except Exception as error:
                failure = hashlib.sha256(f"{type(error).__name__}:{error}".encode()).hexdigest()
                try:
                    self.repository.fail(claim, code="compilation_failed", failure_digest=failure)
                except Exception as cleanup_error:
                    raise DatasetCompilationError("compilation cleanup failure") from cleanup_error
                raise

    def inspect(self, receipt_id: str) -> DatasetCompilationReceipt:
        if self.cancelled():
            raise _CompilationCancelled("inspection cancelled before verification")
        receipt = self.repository.receipt(receipt_id)
        try:
            with self.database.connect() as connection:
                for output in receipt.outputs:
                    if self.cancelled():
                        raise _CompilationCancelled("inspection cancelled at artifact boundary")
                    with self.artifacts.open_verified_with_connection(
                        connection,
                        output.artifact_id,
                        sha256=output.sha256,
                        byte_size=output.byte_size,
                        media_type=output.media_type,
                        cancelled=self.cancelled,
                    ) as handle:
                        self.artifacts.reverify_open_handle(
                            connection, handle, cancelled=self.cancelled
                        )
        except ArtifactError as error:
            if self.cancelled():
                raise _CompilationCancelled("inspection cancelled during verification") from error
            raise
        return receipt

    def reproduce(self, receipt_id: str, *, operation_key: str) -> DatasetCompilationReceipt:
        original = self.inspect(receipt_id)
        request = DatasetCompilationRequest(
            operation_key=operation_key,
            compiler_id=original.compiler_id,
            source=original.source,
            recipe=original.recipe,
            output_roles=tuple(output.role for output in original.outputs),
            reproduces_receipt_id=original.receipt_id,
        )
        expected = tuple((o.role, o.sha256, o.byte_size, o.media_type) for o in original.outputs)
        return self._compile(request, expected_outputs=expected)

    @staticmethod
    def _validate_outputs(
        request: DatasetCompilationRequest, outputs: tuple[BackendOutput, ...]
    ) -> None:
        roles = tuple(sorted(output.role for output in outputs))
        if roles != request.output_roles or len(roles) != len(set(roles)):
            raise DatasetCompilationError(
                "backend output roles are incomplete, extra, or duplicate"
            )
        if any(
            not output.data or len(output.data) > 1024 * 1024 * 1024 or not output.media_type
            for output in outputs
        ):
            raise DatasetCompilationError("backend output is invalid")

    def _publish(
        self,
        request: DatasetCompilationRequest,
        claim: CompilationClaim,
        outputs: tuple[BackendOutput, ...],
    ) -> DatasetCompilationReceipt:
        staged_list = []
        try:
            for output in outputs:
                staged_list.append(self.artifacts.stage_bytes(output.data, output.media_type))
        except Exception:
            failures = self._discard_all(tuple(staged_list))
            if failures:
                raise _CleanupFailure(failures) from None
            raise
        staged = tuple(staged_list)
        by_role = dict(zip((output.role for output in outputs), staged, strict=True))
        now = utc_now().isoformat()

        def persist(
            connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]
        ) -> DatasetCompilationReceipt:
            artifact_by_content = {
                (artifact.sha256, artifact.media_type): artifact for artifact in artifacts
            }
            receipt_outputs = tuple(
                CompilationOutput(
                    role=role,
                    artifact_id=artifact_by_content[
                        (by_role[role].sha256, by_role[role].media_type)
                    ].id,
                    sha256=by_role[role].sha256,
                    byte_size=by_role[role].byte_size,
                    media_type=by_role[role].media_type,
                )
                for role in request.output_roles
            )
            receipt_identity = hashlib.sha256(
                (request.digest + str(claim.attempt)).encode()
            ).hexdigest()[:32]
            receipt = DatasetCompilationReceipt(
                receipt_id=f"dcr_{receipt_identity}",
                operation_id=claim.operation_id,
                operation_key=request.operation_key,
                request_digest=request.digest,
                compiler_id=request.compiler_id,
                attempt=claim.attempt,
                source=request.source,
                recipe=request.recipe,
                outputs=receipt_outputs,
                reproduces_receipt_id=request.reproduces_receipt_id,
                created_at=now,
            )
            payload = canonical_json(receipt.model_dump(mode="json"))
            self.repository.complete_with_connection(
                connection,
                claim,
                outputs={
                    output.role: (
                        output.artifact_id,
                        output.sha256,
                        output.byte_size,
                        output.media_type,
                    )
                    for output in receipt_outputs
                },
                receipt_digest=hashlib.sha256(payload.encode()).hexdigest(),
                receipt_json=payload,
            )
            return receipt

        try:
            return self.artifacts.commit_batch(staged, persist)
        except Exception:
            failures = self._discard_all(staged)
            if failures:
                raise _CleanupFailure(failures) from None
            raise

    def _discard_all(self, staged: tuple[StagedArtifact, ...]) -> tuple[str, ...]:
        failures = []
        for ordinal, item in enumerate(staged):
            try:
                self.artifacts.discard(item)
            except Exception as error:
                if isinstance(error, FileNotFoundError) or (
                    isinstance(error, OSError) and error.errno == errno.ENOENT
                ):
                    reason = "discard_missing"
                elif isinstance(error, PermissionError) or (
                    isinstance(error, OSError) and error.errno in {errno.EACCES, errno.EPERM}
                ):
                    reason = "discard_permission_denied"
                elif isinstance(error, OSError):
                    reason = "discard_io_error"
                else:
                    reason = "discard_unexpected"
                failures.append(f"{ordinal}:{reason}")
        return tuple(failures)
