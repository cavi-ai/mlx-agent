"""Strictly local source and capability boundary for Wan LoRA training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import ConfigDict, Field, StrictBool, StrictFloat, StrictInt, field_validator

from video_lab.domain import FrozenModel, canonical_json
from video_lab.trainer_protocol import (
    CalibrationRequest,
    CalibrationResult,
    CancellationCheck,
    CheckpointResult,
    LoRAConfiguration,
    ProbeTrainingRequest,
    TrainerCapabilityReport,
    TrainingCancelled,
    TrainingMaterialHandles,
    TrainingProgress,
    TrainingProgressCallback,
    TrainingRequest,
    TrainingResult,
    validate_cancelled_training_result,
    validate_training_result,
)
from video_lab.trainers.wan_checkpoint import (
    WanCheckpointCleanupError,
    WanCheckpointCodec,
    WanCheckpointState,
    WanCheckpointValidationContext,
    WanParentCheckpointDescriptor,
)
from video_lab.training_models import Sha256Digest
from video_lab.wan_pending import (
    TrainableParameterSpec,
    trainable_parameter_spec_records,
    validate_trainable_parameter_specs,
)

WAN_LORA_MPS_BACKEND_ID = "diffusers.wan2.2-animegen-lora.mps.v1"
_SAFE_BOUNDARIES = (1, 2, 3, 4)
_MAX_STEPS = 1_000_000
_EXECUTABLE_COMPONENTS = ("text_encoder", "transformer", "transformer_2", "vae")


@dataclass(frozen=True, slots=True)
class WanSourceDescriptor:
    """Factory-authorized identity for one required local source file."""

    name: str
    path: Callable[[], Path]
    expected_digest: Sha256Digest
    expected_size: int
    expected_device: int
    expected_inode: int


@dataclass(frozen=True, slots=True)
class WanSourceFile:
    name: str
    path: Path
    digest: Sha256Digest
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class WanTrainingSources:
    files: tuple[WanSourceFile, ...]
    base_model_digest: Sha256Digest
    source_digest: Sha256Digest


class WanMpsObservation(FrozenModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    runtime_digest: Sha256Digest
    built: StrictBool
    available: StrictBool
    bf16_supported: StrictBool
    component_placements: tuple[tuple[str, str], ...]
    disk_free_bytes: StrictInt = Field(ge=0)
    disk_required_bytes: StrictInt = Field(ge=0)
    memory_headroom_bytes: StrictInt = Field(ge=0)
    memory_required_bytes: StrictInt = Field(ge=0)

    @field_validator("component_placements", mode="before")
    @classmethod
    def canonical_placements(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("wan_component_placements_invalid")
        placements: list[tuple[str, str]] = []
        for item in value:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                or not item[0]
                or not item[1]
            ):
                raise ValueError("wan_component_placements_invalid")
            placements.append((item[0], item[1]))
        canonical = tuple(placements)
        if canonical != tuple(sorted(canonical)) or len({name for name, _ in canonical}) != len(
            canonical
        ):
            raise ValueError("wan_component_placements_invalid")
        return canonical


class WanParameterObservation(FrozenModel):
    """One independently observable prepared parameter identity."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    name: str = Field(min_length=1)
    module: str = Field(min_length=1)
    requires_grad: StrictBool
    digest: Sha256Digest
    device: Literal["mps"]
    dtype: Literal["bfloat16"]
    shape: tuple[StrictInt, ...]

    @field_validator("shape", mode="before")
    @classmethod
    def exact_positive_shape(cls, value: object) -> object:
        if (
            type(value) is not tuple
            or not value
            or any(type(item) is not int or item <= 0 for item in value)
        ):
            raise ValueError("wan_parameter_shape_invalid")
        return value


class WanCalibrationMeasurement(FrozenModel):
    """Measured forward/backward and endpoint resource telemetry."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    loss: StrictFloat
    gradient_norm: StrictFloat = Field(ge=0)
    wall_seconds: StrictFloat = Field(ge=0)
    rss_start_bytes: StrictFloat = Field(ge=0)
    rss_end_bytes: StrictFloat = Field(ge=0)
    mps_allocated_start_bytes: StrictFloat = Field(ge=0)
    mps_allocated_end_bytes: StrictFloat = Field(ge=0)
    mps_driver_start_bytes: StrictFloat = Field(ge=0)
    mps_driver_end_bytes: StrictFloat = Field(ge=0)

    @field_validator("*")
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("wan_calibration_measurement_invalid")
        return value


class WanTrainingStepMeasurement(FrozenModel):
    """One finite optimizer step observed through the injected runtime boundary."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    loss: StrictFloat
    gradient_norm: StrictFloat = Field(ge=0)

    @field_validator("*")
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("wan_training_step_invalid")
        return value


@dataclass(frozen=True, slots=True)
class WanRuntimeCheckpointState:
    """Runtime-owned trainable bytes and substantive continuation state."""

    weights: bytes
    optimizer: Mapping[str, object]
    trainer_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class WanRuntimeRestoreReceipt:
    """Canonical snapshot proving the runtime applied the exact parent state."""

    weights_digest: Sha256Digest
    optimizer_payload: bytes
    trainer_state_payload: bytes


@runtime_checkable
class WanPreparedCommonOperations(Protocol):
    def eligible_lora_modules(self) -> tuple[str, ...]: ...

    def inject_lora(
        self,
        *,
        targets: tuple[str, ...],
        configuration: LoRAConfiguration,
        seed: int,
        device: str,
        dtype: str,
    ) -> None: ...

    def parameter_inventory(self) -> tuple[object, ...]: ...



@runtime_checkable
class WanPreparedOperations(WanPreparedCommonOperations, Protocol):
    def measure_forward_backward(self) -> object: ...


@runtime_checkable
class WanPreparedTrainingOperations(WanPreparedCommonOperations, Protocol):
    def train_step(self, *, step: int, total_steps: int) -> object: ...

    def checkpoint_state(self, *, step: int) -> object: ...

    def restore_checkpoint(self, state: WanCheckpointState) -> None: ...

    def observe_restored_state(self) -> object: ...


class _WanPreparedBoundaryInvalid(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WanPreparedTraining:
    """Runtime-owned prepared state with a narrow, ML-free calibration boundary."""

    handle: object

    def eligible_lora_modules(self) -> tuple[str, ...]:
        operations = self._operations()
        value = operations.eligible_lora_modules()
        if type(value) is not tuple or any(type(item) is not str or not item for item in value):
            raise _WanPreparedBoundaryInvalid("targets")
        return value

    def inject_lora(
        self,
        *,
        targets: tuple[str, ...],
        configuration: LoRAConfiguration,
        seed: int,
    ) -> None:
        self._operations().inject_lora(
            targets=targets,
            configuration=configuration,
            seed=seed,
            device="mps",
            dtype="bfloat16",
        )

    def parameter_inventory(self) -> tuple[WanParameterObservation, ...]:
        raw = self._operations().parameter_inventory()
        if type(raw) is not tuple:
            raise _WanPreparedBoundaryInvalid("inventory")
        try:
            return tuple(WanParameterObservation.model_validate(item, strict=True) for item in raw)
        except Exception:
            raise _WanPreparedBoundaryInvalid("inventory") from None


    def measure_forward_backward(self) -> WanCalibrationMeasurement:
        raw = self._calibration_operations().measure_forward_backward()
        try:
            return WanCalibrationMeasurement.model_validate(raw, strict=True)
        except Exception:
            raise _WanPreparedBoundaryInvalid("measurement") from None

    def train_step(self, *, step: int, total_steps: int) -> WanTrainingStepMeasurement:
        raw = self._training_operations().train_step(step=step, total_steps=total_steps)
        try:
            return WanTrainingStepMeasurement.model_validate(raw, strict=True)
        except Exception:
            raise _WanPreparedBoundaryInvalid("training_step") from None

    def checkpoint_state(self, *, step: int) -> WanRuntimeCheckpointState:
        raw = self._training_operations().checkpoint_state(step=step)
        if (
            type(raw) is not WanRuntimeCheckpointState
            or type(raw.weights) is not bytes
            or not isinstance(raw.optimizer, Mapping)
            or not isinstance(raw.trainer_state, Mapping)
        ):
            raise _WanPreparedBoundaryInvalid("checkpoint_state")
        return raw

    def restore_checkpoint(self, state: WanCheckpointState) -> WanRuntimeRestoreReceipt:
        if type(state) is not WanCheckpointState:
            raise _WanPreparedBoundaryInvalid("restore_state")
        operations = self._training_operations()
        apply_restore = cast(
            Callable[[WanCheckpointState], object], operations.restore_checkpoint
        )
        result = apply_restore(state)
        if result is not None:
            raise _WanPreparedBoundaryInvalid("restore_receipt")
        raw = operations.observe_restored_state()
        if type(raw) is not WanRuntimeRestoreReceipt:
            raise _WanPreparedBoundaryInvalid("restore_receipt")
        expected_optimizer = canonical_json(dict(state.optimizer)).encode()
        expected_trainer = canonical_json(dict(state.trainer_state)).encode()
        expected_weights = "sha256:" + hashlib.sha256(state.weights).hexdigest()
        if (
            raw.weights_digest != expected_weights
            or type(raw.optimizer_payload) is not bytes
            or raw.optimizer_payload != expected_optimizer
            or type(raw.trainer_state_payload) is not bytes
            or raw.trainer_state_payload != expected_trainer
        ):
            raise _WanPreparedBoundaryInvalid("restore_receipt")
        return raw

    def _operations(self) -> WanPreparedCommonOperations:
        if not isinstance(self.handle, WanPreparedCommonOperations):
            raise _WanPreparedBoundaryInvalid("operations")
        return self.handle

    def _training_operations(self) -> WanPreparedTrainingOperations:
        if not isinstance(self.handle, WanPreparedTrainingOperations):
            raise _WanPreparedBoundaryInvalid("training_operations")
        return self.handle

    def _calibration_operations(self) -> WanPreparedOperations:
        if not isinstance(self.handle, WanPreparedOperations):
            raise _WanPreparedBoundaryInvalid("calibration_operations")
        return self.handle


@dataclass(frozen=True, slots=True)
class LoRATargetPolicy:
    """Fail-closed exact-module policy for isolated Wan LoRA injection."""

    declared_targets: tuple[str, ...]

    def __post_init__(self) -> None:
        targets = self.declared_targets
        if (
            type(targets) is not tuple
            or not targets
            or any(type(item) is not str or not item for item in targets)
            or targets != tuple(sorted(set(targets)))
        ):
            raise ValueError("wan_lora_targets_invalid")

    def require_exact_eligible(self, eligible: tuple[str, ...]) -> None:
        if eligible != self.declared_targets:
            raise ValueError("wan_lora_targets_invalid")

    def require_isolated_inventory(
        self,
        before: tuple[WanParameterObservation, ...],
        injected: tuple[WanParameterObservation, ...],
    ) -> None:
        before_by_name = _parameter_identities(before)
        injected_by_name = _parameter_identities(injected)
        if any(item.requires_grad for item in before):
            raise ValueError("wan_parameter_inventory_invalid")
        trainable = tuple(item for item in injected if item.requires_grad)
        frozen = {item.name: item.digest for item in injected if not item.requires_grad}
        if (
            not trainable
            or {item.module for item in trainable} != set(self.declared_targets)
            or any(item.module not in self.declared_targets for item in trainable)
            or frozen != before_by_name
            or len(injected_by_name) != len(injected)
        ):
            raise ValueError("wan_parameter_inventory_invalid")

    @staticmethod
    def require_frozen_unchanged(
        baseline: tuple[WanParameterObservation, ...],
        after: tuple[WanParameterObservation, ...],
    ) -> None:
        expected = {item.name: item.digest for item in baseline if not item.requires_grad}
        actual = {item.name: item.digest for item in after if not item.requires_grad}
        if expected != actual:
            raise ValueError("wan_frozen_parameters_changed")


def _trainable_specs_from_inventory(
    inventory: tuple[WanParameterObservation, ...],
    declared_targets: tuple[str, ...],
) -> tuple[TrainableParameterSpec, ...]:
    specs: list[TrainableParameterSpec] = []
    for item in inventory:
        if not item.requires_grad:
            continue
        role: Literal["A", "B"] | None = None
        logical_name: str | None = None
        for marker, candidate in ((".lora_A.", "A"), (".lora_B.", "B")):
            if marker in item.name:
                role = cast(Literal["A", "B"], candidate)
                logical_name = item.name.split(marker, 1)[0]
                break
        if role is None or logical_name is None or item.module != logical_name:
            raise ValueError("wan_parameter_inventory_invalid")
        specs.append(
            TrainableParameterSpec(
                physical_name=item.name,
                logical_target=item.module,
                role=role,
                shape=item.shape,
                dtype=item.dtype,
                device=item.device,
            )
        )
    try:
        return validate_trainable_parameter_specs(
            specs, declared_targets=declared_targets
        )
    except ValueError:
        raise ValueError("wan_parameter_inventory_invalid") from None


def _parameter_identities(
    inventory: tuple[WanParameterObservation, ...],
) -> dict[str, Sha256Digest]:
    identities = {item.name: item.digest for item in inventory}
    if not inventory or len(identities) != len(inventory):
        raise ValueError("wan_parameter_inventory_invalid")
    return identities


@runtime_checkable
class WanTrainingRuntime(Protocol):
    def observe_host(self) -> WanMpsObservation: ...

    def prepare(
        self,
        sources: WanTrainingSources,
        request: TrainingRequest | CalibrationRequest,
    ) -> WanPreparedTraining: ...

    def release(self, prepared: WanPreparedTraining | None) -> None: ...


@dataclass(frozen=True, slots=True)
class _LeasedSource:
    file: WanSourceFile
    descriptor: int


class WanTrainingSourceLease:
    """Backend-owned retained descriptors for one verified source snapshot."""

    def __init__(self, sources: WanTrainingSources, leased: tuple[_LeasedSource, ...]) -> None:
        self._sources = sources
        self._leased = leased
        self._closed = False

    @property
    def sources(self) -> WanTrainingSources:
        return self._sources

    def __enter__(self) -> WanTrainingSourceLease:
        if self._closed:
            raise ValueError("wan_source_lease_closed")
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: object | None,
    ) -> None:
        self.close()

    def reverify(self) -> None:
        if self._closed:
            raise ValueError("wan_source_lease_closed")
        try:
            for leased in self._leased:
                _verify_open_file(leased.descriptor, leased.file)
                current = _open_no_follow(leased.file.path)
                try:
                    _verify_open_file(current, leased.file)
                finally:
                    os.close(current)
        except (OSError, ValueError):
            raise ValueError("wan_source_invalid") from None

    def _verify_authority(self) -> None:
        for leased in self._leased:
            retained = os.fstat(leased.descriptor)
            if not _matches_source(retained, leased.file):
                raise ValueError("wan_source_invalid")
            current = _open_no_follow(leased.file.path)
            try:
                if _file_identity(os.fstat(current)) != _file_identity(retained):
                    raise ValueError("wan_source_invalid")
            finally:
                os.close(current)

    def copy_file_to(self, source: WanSourceFile, destination: int) -> tuple[str, int]:
        """Stream one retained source into an exclusive destination descriptor."""
        if self._closed:
            raise ValueError("wan_source_lease_closed")
        leased = next((item for item in self._leased if item.file == source), None)
        if leased is None:
            raise ValueError("wan_source_invalid")
        before = os.fstat(leased.descriptor)
        if not _matches_source(before, leased.file):
            raise ValueError("wan_source_invalid")
        os.lseek(leased.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while size < source.size_bytes:
            chunk = os.read(leased.descriptor, min(1024 * 1024, source.size_bytes - size))
            if not chunk:
                raise ValueError("wan_source_invalid")
            view = memoryview(chunk)
            while view:
                written = os.write(destination, view)
                if written <= 0:
                    raise OSError("wan snapshot short write")
                view = view[written:]
            digest.update(chunk)
            size += len(chunk)
        if size != source.size_bytes or "sha256:" + digest.hexdigest() != source.digest:
            raise ValueError("wan_source_invalid")
        os.fsync(destination)
        os.lseek(destination, 0, os.SEEK_SET)
        copied = hashlib.sha256()
        copied_size = 0
        while copied_size < size:
            chunk = os.read(destination, min(1024 * 1024, size - copied_size))
            if not chunk:
                raise ValueError("wan_source_invalid")
            copied.update(chunk)
            copied_size += len(chunk)
        if copied_size != size or copied.digest() != digest.digest():
            raise ValueError("wan_source_invalid")
        after = os.fstat(leased.descriptor)
        if (
            not _matches_source(after, leased.file)
            or _file_identity(before) != _file_identity(after)
        ):
            raise ValueError("wan_source_invalid")
        return source.digest, size

    def read_file_bytes(self, source: WanSourceFile) -> bytes:
        """Read exact bytes from the retained descriptor, never from its path."""

        if self._closed:
            raise ValueError("wan_source_lease_closed")
        leased = next((item for item in self._leased if item.file == source), None)
        if leased is None:
            raise ValueError("wan_source_invalid")
        try:
            before = os.fstat(leased.descriptor)
            if not _matches_source(before, source):
                raise ValueError
            payload = os.pread(leased.descriptor, source.size_bytes, 0)
            after = os.fstat(leased.descriptor)
            if (
                len(payload) != source.size_bytes
                or "sha256:" + hashlib.sha256(payload).hexdigest() != source.digest
                or _file_identity(before) != _file_identity(after)
                or not _matches_source(after, source)
            ):
                raise ValueError
            return payload
        except (OSError, ValueError):
            raise ValueError("wan_source_invalid") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for item in self._leased:
            try:
                os.close(item.descriptor)
            except OSError:
                pass


class WanTrainingSourceResolver:
    """Acquire retained, exact-byte leases for factory-authorized local artifacts."""

    def __init__(self, descriptors: tuple[WanSourceDescriptor, ...]) -> None:
        self._descriptors = descriptors

    def acquire(self) -> WanTrainingSourceLease:
        names = tuple(item.name for item in self._descriptors)
        if (
            not names
            or not all(names)
            or len(names) != len(set(names))
            or "base_model" not in names
        ):
            raise ValueError("wan_source_invalid")
        leased: list[_LeasedSource] = []
        try:
            for descriptor in sorted(self._descriptors, key=lambda item: item.name):
                leased.append(self._acquire_one(descriptor))
            files = tuple(item.file for item in leased)
            sources = _training_sources(files)
            lease = WanTrainingSourceLease(sources, tuple(leased))
            lease._verify_authority()
            return lease
        except Exception:
            for leased_source in leased:
                try:
                    os.close(leased_source.descriptor)
                except OSError:
                    pass
            raise ValueError("wan_source_invalid") from None

    @staticmethod
    def _acquire_one(descriptor: WanSourceDescriptor) -> _LeasedSource:
        try:
            path = Path(descriptor.path())
            expected_identity = (
                descriptor.expected_device,
                descriptor.expected_inode,
                descriptor.expected_size,
            )
            if descriptor.expected_size < 0:
                raise ValueError("wan_source_invalid")
            file_descriptor = _open_no_follow(path)
            try:
                info = os.fstat(file_descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or (info.st_dev, info.st_ino, info.st_size) != expected_identity
                    or _digest_open_file(file_descriptor, descriptor.expected_size)
                    != descriptor.expected_digest
                ):
                    raise ValueError("wan_source_invalid")
                after = os.fstat(file_descriptor)
                if _file_identity(info) != _file_identity(after):
                    raise ValueError("wan_source_invalid")
            except Exception:
                os.close(file_descriptor)
                raise
        except (OSError, ValueError, TypeError):
            raise ValueError("wan_source_invalid") from None
        return _LeasedSource(
            file=WanSourceFile(
                name=descriptor.name,
                path=path,
                digest=descriptor.expected_digest,
                size_bytes=descriptor.expected_size,
                device=info.st_dev,
                inode=info.st_ino,
            ),
            descriptor=file_descriptor,
        )


def _training_sources(files: tuple[WanSourceFile, ...]) -> WanTrainingSources:
    payload = {
        "files": [
            {
                "name": item.name,
                "sha256": item.digest,
                "size_bytes": item.size_bytes,
            }
            for item in files
        ]
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    source_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    base_model_digest = next(item.digest for item in files if item.name == "base_model")
    return WanTrainingSources(
        files=files, base_model_digest=base_model_digest, source_digest=source_digest
    )


def _open_no_follow(path: Path) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None or not path.is_absolute():
        raise ValueError("wan_source_invalid")
    parts = path.parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts[1:]):
        raise ValueError("wan_source_invalid")
    directory_flags = os.O_RDONLY | directory | no_follow | getattr(os, "O_CLOEXEC", 0)
    current = os.open(parts[0], directory_flags)
    try:
        for part in parts[1:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        file_flags = (
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        return os.open(parts[-1], file_flags, dir_fd=current)
    finally:
        os.close(current)


def _digest_open_file(file_descriptor: int, expected_size: int) -> str:
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while size <= expected_size:
        chunk = os.read(file_descriptor, min(1024 * 1024, expected_size + 1 - size))
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    if size != expected_size:
        raise ValueError("wan_source_invalid")
    return "sha256:" + digest.hexdigest()


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _matches_source(info: os.stat_result, source: WanSourceFile) -> bool:
    return stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino, info.st_size) == (
        source.device,
        source.inode,
        source.size_bytes,
    )


def _verify_open_file(file_descriptor: int, source: WanSourceFile) -> None:
    info = os.fstat(file_descriptor)
    if (
        not _matches_source(info, source)
        or _digest_open_file(file_descriptor, source.size_bytes) != source.digest
    ):
        raise ValueError("wan_source_invalid")


class WanBackendStageUnavailable(RuntimeError):
    def __init__(self, stage: Literal["calibrate", "train"]) -> None:
        self.stage = stage
        super().__init__("wan_backend_stage_unavailable")


class WanCalibrationRuntimeError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("wan_calibration_runtime_failed")


class WanTrainingRuntimeError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("wan_training_runtime_failed")


class WanTrainingDeadlineExceeded(RuntimeError):
    def __init__(self) -> None:
        super().__init__("wan_training_deadline_exceeded")


class WanLoRAMpsTrainer:
    backend_id = WAN_LORA_MPS_BACKEND_ID

    def __init__(
        self,
        *,
        backend_digest: Sha256Digest,
        material_digest: Sha256Digest,
        orchestration_source_digest: Sha256Digest | None = None,
        descriptor_source_digest: Sha256Digest | None = None,
        source_resolver: WanTrainingSourceResolver,
        runtime: WanTrainingRuntime,
        checkpoint_resolver: Callable[[TrainingRequest], WanParentCheckpointDescriptor]
        | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._backend_digest = backend_digest
        self._material_digest = material_digest
        self._orchestration_source_digest = orchestration_source_digest
        self._descriptor_source_digest = descriptor_source_digest
        self._source_resolver = source_resolver
        self._runtime = runtime
        self._checkpoint_resolver = checkpoint_resolver
        self._clock = clock
        self._closed = False
        self._active_operations = 0
        self._condition = threading.Condition()

    def __enter__(self) -> WanLoRAMpsTrainer:
        self._require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._active_operations:
                raise ValueError("wan_trainer_busy")
            close = getattr(self._runtime, "close", None)
            if callable(close):
                close()
            self._closed = True

    def _require_open(self) -> None:
        with self._condition:
            if self._closed:
                raise ValueError("wan_trainer_closed")

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._condition:
            if self._closed:
                raise ValueError("wan_trainer_closed")
            self._active_operations += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_operations -= 1
                self._condition.notify_all()

    def probe(self, request: ProbeTrainingRequest) -> TrainerCapabilityReport:
        with self._operation():
            return self._probe(request)

    def _probe(self, request: ProbeTrainingRequest) -> TrainerCapabilityReport:
        lease: WanTrainingSourceLease | None = None
        try:
            lease = self._source_resolver.acquire()
            observed = self._runtime.observe_host()
            observation = WanMpsObservation.model_validate(observed, strict=True)
            lease.reverify()
            return self._report(
                observation.runtime_digest,
                self._supports(request, lease.sources, observation),
            )
        except Exception:
            return self._report(request.materials.runtime_digest, False)
        finally:
            if lease is not None:
                lease.close()

    def calibrate(
        self,
        request: CalibrationRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> CalibrationResult:
        with self._operation():
            return self._calibrate(request, progress, cancelled)

    def _calibrate(
        self,
        request: CalibrationRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> CalibrationResult:
        lease: WanTrainingSourceLease | None = None
        prepared: WanPreparedTraining | None = None
        primary_error: BaseException | None = None
        try:
            lease = self._source_resolver.acquire()
            observed = self._runtime.observe_host()
            observation = WanMpsObservation.model_validate(observed, strict=True)
            if not self._supports_calibration(request, lease.sources, observation):
                raise ValueError("wan_calibration_unsupported")
            lease.reverify()
            prepared = self._runtime.prepare(lease.sources, request)
            if type(prepared) is not WanPreparedTraining:
                raise _WanPreparedBoundaryInvalid("prepared")
            lease.reverify()
            if cancelled():
                raise TrainingCancelled(0)

            policy = LoRATargetPolicy(request.trainable_targets)
            policy.require_exact_eligible(prepared.eligible_lora_modules())
            before = prepared.parameter_inventory()
            prepared.inject_lora(
                targets=policy.declared_targets,
                configuration=request.lora,
                seed=request.seed,
            )
            injected = prepared.parameter_inventory()
            policy.require_isolated_inventory(before, injected)
            if cancelled():
                raise TrainingCancelled(0)
            measurement = prepared.measure_forward_backward()
            after = prepared.parameter_inventory()
            policy.require_isolated_inventory(before, after)
            policy.require_frozen_unchanged(injected, after)
            if request.steps > 0:
                progress(
                    TrainingProgress(
                        step=request.steps,
                        total_steps=request.steps,
                        fraction=1.0,
                        loss=measurement.loss,
                        gradient_norm=measurement.gradient_norm,
                        checkpoint_boundary=True,
                    )
                )
            lease.reverify()
            calibration_digest = _calibration_digest(
                request=request,
                backend_digest=self._backend_digest,
                runtime_digest=observation.runtime_digest,
                before=before,
                injected=injected,
                after=after,
                measurement=measurement,
            )
            return CalibrationResult(
                request_id=request.request_id,
                run_id=request.run_id,
                backend_digest=self._backend_digest,
                runtime_digest=observation.runtime_digest,
                calibration_digest=calibration_digest,
                finite_gradients=True,
                frozen_parameters_unchanged=True,
                changed_targets=policy.declared_targets,
                measurements=measurement.model_dump(mode="python"),
            )
        except TrainingCancelled as error:
            primary_error = error
            raise
        except _WanPreparedBoundaryInvalid as error:
            primary_error = error
            if str(error) == "measurement":
                raise ValueError("wan_calibration_measurement_invalid") from None
            raise ValueError("wan_parameter_inventory_invalid") from None
        except ValueError as error:
            primary_error = error
            raise
        except BaseException as error:
            primary_error = error
            raise WanCalibrationRuntimeError() from None
        finally:
            cleanup_failed = False
            try:
                self._runtime.release(prepared)
            except BaseException:
                cleanup_failed = True
            if lease is not None:
                try:
                    lease.close()
                except BaseException:
                    cleanup_failed = True
            if cleanup_failed and primary_error is None:
                raise WanCalibrationRuntimeError() from None

    def train(
        self,
        request: TrainingRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> TrainingResult:
        with self._operation():
            return self._train(request, progress, cancelled)

    def _train(
        self,
        request: TrainingRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> TrainingResult:
        resume = request.start_step > 0
        if (not resume and request.telemetry_sequence_start != 1) or (
            resume and self._checkpoint_resolver is None
        ):
            raise ValueError("wan_training_resume_invalid")

        started = self._clock()
        lease: WanTrainingSourceLease | None = None
        prepared: WanPreparedTraining | None = None
        codec: WanCheckpointCodec | None = None
        checkpoints: list[CheckpointResult] = []
        restored: WanCheckpointState | None = None
        descriptor: WanParentCheckpointDescriptor | None = None
        primary_error: BaseException | None = None
        try:
            lease = self._source_resolver.acquire()
            self._training_observation(request, lease.sources)
            lease.reverify()
            codec = WanCheckpointCodec(request.scratch_root)
            if resume:
                assert self._checkpoint_resolver is not None
                descriptor = self._checkpoint_resolver(request)
                if (
                    descriptor.checkpoint_id != request.parent_checkpoint_id
                    or descriptor.checkpoint_digest != request.parent_checkpoint_digest
                    or descriptor.step != request.start_step
                    or descriptor.total_steps != request.total_steps
                ):
                    raise ValueError("wan_training_resume_invalid")
                optimizer_files = tuple(
                    item for item in descriptor.files if item.role == "optimizer"
                )
                if (
                    len(optimizer_files) != 1
                    or optimizer_files[0].digest != request.optimizer_state_digest
                ):
                    raise ValueError("wan_training_resume_invalid")
            prepared = self._runtime.prepare(lease.sources, request)
            if type(prepared) is not WanPreparedTraining:
                raise _WanPreparedBoundaryInvalid("prepared")
            lease.reverify()

            policy = LoRATargetPolicy(request.trainable_targets)
            eligible_modules = prepared.eligible_lora_modules()
            lease.reverify()
            policy.require_exact_eligible(eligible_modules)
            before = prepared.parameter_inventory()
            lease.reverify()
            prepared.inject_lora(
                targets=policy.declared_targets,
                configuration=request.lora,
                seed=request.seed,
            )
            lease.reverify()
            injected = prepared.parameter_inventory()
            lease.reverify()
            policy.require_isolated_inventory(before, injected)
            trainable_specs = _trainable_specs_from_inventory(
                injected, policy.declared_targets
            )
            telemetry: list[dict[str, object]] = []
            parent = request.parent_checkpoint_id

            if resume:
                assert descriptor is not None
                restored = codec.restore(
                    descriptor,
                    WanCheckpointValidationContext(
                        checkpoint_id=descriptor.checkpoint_id,
                        request_id=request.request_id,
                        run_id=request.run_id,
                        backend_id=request.backend_id,
                        seed=request.seed,
                        step=request.start_step,
                        total_steps=request.total_steps,
                        parent_checkpoint_id=descriptor.parent_checkpoint_id,
                        parent_optimizer_state_digest=descriptor.parent_optimizer_state_digest,
                        trainable_targets=request.trainable_targets,
                        trainable_parameter_specs=trainable_specs,
                        lora_rank=request.lora.rank,
                        lora_alpha=request.lora.alpha,
                        lora_dropout=request.lora.dropout,
                        optimizer_algorithm=request.optimizer.algorithm,
                        optimizer_config=tuple(
                            (item.name, item.value) for item in request.optimizer.parameters
                        ),
                    ),
                )
                codec.reverify_restored()
                prepared.restore_checkpoint(restored)
                codec.reverify_restored()
                if request.telemetry_sequence_start != request.start_step + 1:
                    raise ValueError("wan_training_resume_invalid")

            for step in range(request.start_step + 1, request.total_steps + 1):
                self._require_deadline(started, request.deadline_seconds)
                if resume and step == request.start_step + 1:
                    codec.reverify_restored()
                self._training_observation(request, lease.sources)
                lease.reverify()
                measurement = prepared.train_step(step=step, total_steps=request.total_steps)
                lease.reverify()
                after = prepared.parameter_inventory()
                lease.reverify()
                policy.require_isolated_inventory(before, after)
                policy.require_frozen_unchanged(injected, after)
                item = TrainingProgress(
                    step=step,
                    total_steps=request.total_steps,
                    fraction=step / request.total_steps,
                    loss=measurement.loss,
                    gradient_norm=measurement.gradient_norm,
                    checkpoint_boundary=step in request.checkpoint_steps,
                )
                progress(item)
                lease.reverify()
                telemetry.append(
                    {
                        "sequence": request.telemetry_sequence_start + len(telemetry),
                        "step": step,
                        "payload": item.model_dump(mode="json"),
                    }
                )
                self._require_deadline(started, request.deadline_seconds)
                if step not in request.checkpoint_steps:
                    continue

                runtime_state = prepared.checkpoint_state(step=step)
                lease.reverify()
                checkpoint_id = _checkpoint_id(request, step, parent)
                checkpoint = codec.write(
                    WanCheckpointState(
                        checkpoint_id=checkpoint_id,
                        step=step,
                        total_steps=request.total_steps,
                        parent_checkpoint_id=parent,
                        weights=runtime_state.weights,
                        optimizer=runtime_state.optimizer,
                        trainer_state=runtime_state.trainer_state,
                        telemetry=tuple(telemetry),
                        trainable_targets={
                            "schema_version": "trainer-trainable-targets.v2",
                            "request_id": request.request_id,
                            "run_id": request.run_id,
                            "step": step,
                            "targets": list(request.trainable_targets),
                            "physical_parameters": trainable_parameter_spec_records(
                                trainable_specs
                            ),
                        },
                        frozen_parameters={
                            "schema_version": "trainer-frozen-parameters.v1",
                            "request_id": request.request_id,
                            "run_id": request.run_id,
                            "step": step,
                            "trainable_targets": list(request.trainable_targets),
                            "unchanged": True,
                        },
                        finite_gradient_proof={
                            "schema_version": "trainer-finite-gradient-proof.v1",
                            "request_id": request.request_id,
                            "run_id": request.run_id,
                            "step": step,
                            "finite": True,
                            "loss": measurement.loss,
                            "gradient_norm": measurement.gradient_norm,
                        },
                    ),
                    WanCheckpointValidationContext(
                        checkpoint_id=checkpoint_id,
                        request_id=request.request_id,
                        run_id=request.run_id,
                        backend_id=request.backend_id,
                        seed=request.seed,
                        step=step,
                        total_steps=request.total_steps,
                        parent_checkpoint_id=parent,
                        parent_optimizer_state_digest=request.optimizer_state_digest,
                        trainable_targets=request.trainable_targets,
                        trainable_parameter_specs=trainable_specs,
                        lora_rank=request.lora.rank,
                        lora_alpha=request.lora.alpha,
                        lora_dropout=request.lora.dropout,
                        optimizer_algorithm=request.optimizer.algorithm,
                        optimizer_config=tuple(
                            (item.name, item.value) for item in request.optimizer.parameters
                        ),
                        start_step=request.start_step,
                        telemetry_sequence_start=request.telemetry_sequence_start,
                    ),
                )
                checkpoints.append(checkpoint)
                lease.reverify()
                parent = checkpoint.checkpoint_id
                result = self._training_result(
                    request=request,
                    checkpoints=tuple(checkpoints),
                    elapsed=self._elapsed(started),
                )
                self._require_deadline(started, request.deadline_seconds)
                validate_cancelled_training_result(request, result, step)
                lease.reverify()
                is_cancelled = cancelled()
                lease.reverify()
                if is_cancelled:
                    raise TrainingCancelled(step, result)

            result = self._training_result(
                request=request,
                checkpoints=tuple(checkpoints),
                elapsed=self._elapsed(started),
            )
            validate_training_result(request, result)
            lease.reverify()
            return result
        except TrainingCancelled as error:
            primary_error = error
            raise
        except WanCheckpointCleanupError as error:
            primary_error = error
            raise
        except _WanPreparedBoundaryInvalid as error:
            primary_error = error
            if str(error) in {"restore_state", "restore_receipt"}:
                raise ValueError("wan_training_restore_invalid") from None
            raise ValueError("wan_parameter_inventory_invalid") from None
        except (ValueError, WanTrainingDeadlineExceeded) as error:
            primary_error = error
            raise
        except BaseException:
            bounded_error = WanTrainingRuntimeError()
            primary_error = bounded_error
            raise bounded_error from None
        finally:
            cleanup_errors: list[BaseException] = []
            final_source_error: BaseException | None = None
            try:
                self._runtime.release(prepared)
            except BaseException as error:
                cleanup_errors.append(error)
            if lease is not None:
                try:
                    lease.reverify()
                except BaseException as error:
                    final_source_error = error
            discard_required = (
                primary_error is not None
                and not isinstance(primary_error, TrainingCancelled)
            ) or final_source_error is not None or bool(cleanup_errors)
            if codec is not None:
                if discard_required:
                    for checkpoint in reversed(checkpoints):
                        try:
                            codec.discard(checkpoint.checkpoint_id)
                        except WanCheckpointCleanupError as error:
                            cleanup_errors.extend(error.cleanup_errors)
                        except BaseException as error:
                            cleanup_errors.append(error)
                try:
                    codec.close()
                except BaseException as error:
                    cleanup_errors.append(error)
            if lease is not None:
                try:
                    lease.close()
                except BaseException as error:
                    cleanup_errors.append(error)
            if (
                isinstance(primary_error, TrainingCancelled)
                and not discard_required
                and cleanup_errors
            ):
                recovery_codec: WanCheckpointCodec | None = None
                try:
                    recovery_codec = WanCheckpointCodec(request.scratch_root)
                    for checkpoint in reversed(checkpoints):
                        try:
                            recovery_codec.discard(checkpoint.checkpoint_id)
                        except WanCheckpointCleanupError as error:
                            cleanup_errors.extend(error.cleanup_errors)
                        except BaseException as error:
                            cleanup_errors.append(error)
                except BaseException as error:
                    cleanup_errors.append(error)
                finally:
                    if recovery_codec is not None:
                        try:
                            recovery_codec.close()
                        except BaseException as error:
                            cleanup_errors.append(error)
            if isinstance(primary_error, WanCheckpointCleanupError):
                cleanup_errors[0:0] = primary_error.cleanup_errors
                primary_error = primary_error.primary_error
            if cleanup_errors:
                raise WanCheckpointCleanupError(
                    final_source_error or primary_error, tuple(cleanup_errors)
                ) from primary_error
            if final_source_error is not None:
                raise final_source_error from primary_error

    def _training_observation(
        self, request: TrainingRequest, sources: WanTrainingSources
    ) -> WanMpsObservation:
        observed = self._runtime.observe_host()
        observation = WanMpsObservation.model_validate(observed, strict=True)
        if not self._supports_common(
            backend_id=request.backend_id,
            materials=request.materials,
            sources=sources,
            observation=observation,
        ):
            raise ValueError("wan_training_placement_invalid")
        return observation

    def _elapsed(self, started: float) -> float:
        elapsed = self._clock() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise WanTrainingDeadlineExceeded()
        return elapsed

    def _require_deadline(self, started: float, deadline_seconds: int) -> None:
        if self._elapsed(started) > deadline_seconds:
            raise WanTrainingDeadlineExceeded()

    def _training_result(
        self,
        *,
        request: TrainingRequest,
        checkpoints: tuple[CheckpointResult, ...],
        elapsed: float,
    ) -> TrainingResult:
        if not checkpoints:
            raise ValueError("wan_training_checkpoint_missing")
        latest = checkpoints[-1]
        return TrainingResult(
            request_id=request.request_id,
            run_id=request.run_id,
            backend_id=self.backend_id,
            backend_digest=self._backend_digest,
            materials=request.materials,
            output_root=request.scratch_root,
            checkpoints=checkpoints,
            telemetry_digest=latest.telemetry_digest,
            changed_targets=request.trainable_targets,
            frozen_parameters_unchanged=True,
            finite_gradients=True,
            elapsed_seconds=elapsed,
        )

    def _report(self, runtime_digest: Sha256Digest, supported: bool) -> TrainerCapabilityReport:
        return TrainerCapabilityReport(
            backend_id=self.backend_id,
            backend_digest=self._backend_digest,
            runtime_digest=runtime_digest,
            supported=supported,
            max_steps=_MAX_STEPS,
            safe_boundaries=_SAFE_BOUNDARIES,
        )

    def _supports(
        self,
        request: ProbeTrainingRequest,
        sources: WanTrainingSources,
        observation: WanMpsObservation,
    ) -> bool:
        return request.expected_backend_digest == self._backend_digest and self._supports_common(
            backend_id=request.backend_id,
            materials=request.materials,
            sources=sources,
            observation=observation,
        )

    def _supports_calibration(
        self,
        request: CalibrationRequest,
        sources: WanTrainingSources,
        observation: WanMpsObservation,
    ) -> bool:
        return self._supports_common(
            backend_id=request.backend_id,
            materials=request.materials,
            sources=sources,
            observation=observation,
        )

    def _supports_common(
        self,
        *,
        backend_id: str,
        materials: TrainingMaterialHandles,
        sources: WanTrainingSources,
        observation: WanMpsObservation,
    ) -> bool:
        placements = observation.component_placements
        source_bindings_match = (
            materials.source_digest == sources.source_digest
            if self._orchestration_source_digest is None
            and self._descriptor_source_digest is None
            else materials.source_digest == self._orchestration_source_digest
            and sources.source_digest == self._descriptor_source_digest
        )
        return (
            backend_id == self.backend_id
            and materials.backend_digest == self._backend_digest
            and materials.material_digest == self._material_digest
            and materials.base_model_digest == sources.base_model_digest
            and source_bindings_match
            and materials.runtime_digest == observation.runtime_digest
            and observation.built
            and observation.available
            and observation.bf16_supported
            and tuple(name for name, _device in placements) == _EXECUTABLE_COMPONENTS
            and all(device == "mps" for _name, device in placements)
            and observation.disk_required_bytes > 0
            and observation.disk_free_bytes >= observation.disk_required_bytes
            and observation.memory_required_bytes > 0
            and observation.memory_headroom_bytes >= observation.memory_required_bytes
        )


def _calibration_digest(
    *,
    request: CalibrationRequest,
    backend_digest: Sha256Digest,
    runtime_digest: Sha256Digest,
    before: tuple[WanParameterObservation, ...],
    injected: tuple[WanParameterObservation, ...],
    after: tuple[WanParameterObservation, ...],
    measurement: WanCalibrationMeasurement,
) -> Sha256Digest:
    payload = {
        "request": request.model_dump(mode="json", exclude={"scratch_root"}),
        "backend_digest": backend_digest,
        "runtime_digest": runtime_digest,
        "before": [item.model_dump(mode="json") for item in before],
        "injected": [item.model_dump(mode="json") for item in injected],
        "after": [item.model_dump(mode="json") for item in after],
        "measurement": measurement.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _checkpoint_id(request: TrainingRequest, step: int, parent_checkpoint_id: str | None) -> str:
    payload = {
        "schema_version": "wan-checkpoint-identity.v1",
        "request_id": request.request_id,
        "run_id": request.run_id,
        "backend_id": request.backend_id,
        "backend_digest": request.materials.backend_digest,
        "material_digest": request.materials.material_digest,
        "step": step,
        "parent_checkpoint_id": parent_checkpoint_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "wan-" + hashlib.sha256(encoded).hexdigest()
