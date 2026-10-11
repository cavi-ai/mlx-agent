"""Dependency-free contracts for local character-training backends."""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import stat
import threading
import unicodedata
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from video_lab.domain import FrozenModel, canonical_json
from video_lab.wan_pending import (
    trainable_parameter_specs_from_records,
    validate_pending_gradients,
    validate_trainable_parameter_specs,
)

Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class TrainingMaterialHandles(FrozenModel):
    recipe_digest: Sha256Digest
    dataset_digest: Sha256Digest
    base_model_digest: Sha256Digest
    backend_digest: Sha256Digest
    source_digest: Sha256Digest
    runtime_digest: Sha256Digest
    material_digest: Sha256Digest


class OptimizerParameter(FrozenModel):
    name: str = Field(min_length=1)
    value: float | int | bool | str

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float | int | bool | str) -> float | int | bool | str:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("optimizer parameter must be finite")
        return value


class OptimizerConfiguration(FrozenModel):
    algorithm: str = Field(min_length=1)
    parameters: tuple[OptimizerParameter, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonical_parameters(self) -> OptimizerConfiguration:
        names = tuple(item.name for item in self.parameters)
        if names != tuple(sorted(set(names))):
            raise ValueError("optimizer parameters must be sorted and unique")
        return self


class ProbeTrainingRequest(FrozenModel):
    request_id: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    expected_backend_digest: Sha256Digest
    materials: TrainingMaterialHandles
    scratch_root: Path


class TrainerCapabilityReport(FrozenModel):
    backend_id: str = Field(min_length=1)
    backend_digest: Sha256Digest
    runtime_digest: Sha256Digest
    supported: bool
    max_steps: int = Field(gt=0)
    safe_boundaries: tuple[int, ...]

    @field_validator("safe_boundaries")
    @classmethod
    def validate_boundaries(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or tuple(sorted(set(value))) != value or value[0] <= 0:
            raise ValueError("safe boundaries must be positive and strictly monotonic")
        return value


class LoRAConfiguration(FrozenModel):
    """Exact low-rank adapter configuration bound to one trainer request."""

    rank: StrictInt = Field(gt=0)
    alpha: StrictInt = Field(gt=0)
    dropout: StrictFloat = Field(ge=0, lt=1)

    @field_validator("dropout")
    @classmethod
    def finite_dropout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("lora dropout must be finite")
        return value


class CalibrationRequest(FrozenModel):
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    materials: TrainingMaterialHandles
    scratch_root: Path
    steps: int = Field(ge=0)
    trainable_targets: tuple[str, ...] = Field(min_length=1)
    lora: LoRAConfiguration
    seed: int
    deadline_seconds: int = Field(gt=0)


class TrainingRequest(FrozenModel):
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    materials: TrainingMaterialHandles
    scratch_root: Path
    total_steps: int = Field(ge=4)
    checkpoint_steps: tuple[int, ...] = Field(min_length=1)
    trainable_targets: tuple[str, ...] = Field(min_length=1)
    lora: LoRAConfiguration
    seed: int
    deadline_seconds: int = Field(gt=0)
    optimizer: OptimizerConfiguration = OptimizerConfiguration(
        algorithm="adamw",
        parameters=(OptimizerParameter(name="learning_rate", value=0.001),),
    )
    start_step: int = Field(default=0, ge=0)
    telemetry_sequence_start: int = Field(default=1, gt=0)
    parent_checkpoint_id: str | None = None
    parent_checkpoint_digest: Sha256Digest | None = None
    optimizer_state_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_schedule(self) -> TrainingRequest:
        if (
            tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
            or self.checkpoint_steps[-1] != self.total_steps
            or self.checkpoint_steps[0] <= 0
        ):
            raise ValueError("checkpoint schedule is invalid")
        if not 0 <= self.start_step < self.checkpoint_steps[0]:
            raise ValueError("resume start step is invalid")
        resume_values = (
            self.parent_checkpoint_id,
            self.parent_checkpoint_digest,
            self.optimizer_state_digest,
        )
        if any(value is None for value in resume_values) != all(
            value is None for value in resume_values
        ):
            raise ValueError("resume identities must be complete")
        return self


class TrainingProgress(FrozenModel):
    step: int = Field(ge=0)
    total_steps: int = Field(gt=0)
    fraction: float = Field(ge=0, le=1)
    loss: float
    gradient_norm: float = Field(default=1.0, ge=0)
    checkpoint_boundary: bool = False

    @model_validator(mode="after")
    def validate_progress(self) -> TrainingProgress:
        if self.step > self.total_steps or not all(
            math.isfinite(value) for value in (self.fraction, self.loss, self.gradient_norm)
        ):
            raise ValueError("training progress must be bounded and finite")
        expected = self.step / self.total_steps
        if not math.isclose(self.fraction, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("training progress fraction does not match step")
        return self


class CalibrationResult(FrozenModel):
    schema_version: Literal["character-training-calibration-result.v1"] = (
        "character-training-calibration-result.v1"
    )
    request_id: str
    run_id: str
    backend_digest: Sha256Digest
    runtime_digest: Sha256Digest
    calibration_digest: Sha256Digest
    finite_gradients: bool
    frozen_parameters_unchanged: bool
    changed_targets: tuple[str, ...]
    measurements: Mapping[str, float]

    @field_validator("measurements")
    @classmethod
    def finite_measurements(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        if not value or any(not math.isfinite(item) for item in value.values()):
            raise ValueError("calibration measurements must be finite")
        return value


class OutputFile(FrozenModel):
    relative_path: str = Field(min_length=1)
    digest: Sha256Digest
    size_bytes: int = Field(gt=0)
    role: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def contained_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("output path escapes scratch root")
        return value


class CheckpointResult(FrozenModel):
    checkpoint_id: str
    step: int = Field(gt=0)
    fraction: float = Field(gt=0, le=1)
    parent_checkpoint_id: str | None
    files: tuple[OutputFile, ...]
    telemetry_digest: Sha256Digest
    frozen_parameters_digest: Sha256Digest
    trainable_targets_digest: Sha256Digest
    finite_gradient_proof_digest: Sha256Digest


class TrainingResult(FrozenModel):
    request_id: str
    run_id: str
    backend_id: str
    backend_digest: Sha256Digest
    materials: TrainingMaterialHandles
    output_root: Path
    checkpoints: tuple[CheckpointResult, ...]
    telemetry_digest: Sha256Digest
    changed_targets: tuple[str, ...]
    frozen_parameters_unchanged: bool
    finite_gradients: bool
    elapsed_seconds: float = Field(ge=0)
    fault: str | None = None


class TrainingPublicationReceipt(FrozenModel):
    schema_version: Literal["character-training-publication-receipt.v1"] = (
        "character-training-publication-receipt.v1"
    )
    request_id: str
    run_id: str
    lifecycle_state: Literal["evaluating", "cancelled"]
    checkpoint_ids: tuple[str, ...]
    telemetry_digest: Sha256Digest


TrainingProgressCallback = Callable[[TrainingProgress], None]
CancellationCheck = Callable[[], bool]


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()


class TrainingCancelled(RuntimeError):
    """Cooperative cancellation observed at one published checkpoint boundary."""

    def __init__(self, boundary_step: int, result: TrainingResult | None = None) -> None:
        self.boundary_step = boundary_step
        self.result = result
        super().__init__("training_cancelled")


@runtime_checkable
class TrainerBackend(Protocol):
    backend_id: str

    def probe(self, request: ProbeTrainingRequest) -> TrainerCapabilityReport: ...

    def calibrate(
        self,
        request: CalibrationRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> CalibrationResult: ...

    def train(
        self,
        request: TrainingRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> TrainingResult: ...


class TrainerRegistry:
    def __init__(self, *, reviewer_id: str) -> None:
        self.reviewer_id = reviewer_id
        self._backends: dict[str, TrainerBackend] = {}

    def register(self, backend: TrainerBackend, *, expected_id: str | None = None) -> None:
        if expected_id is not None and backend.backend_id != expected_id:
            raise ValueError("backend_id_mismatch")
        if backend.backend_id in self._backends:
            raise ValueError("backend_already_registered")
        self._backends[backend.backend_id] = backend

    def resolve(self, reviewer_id: str, backend_id: str) -> TrainerBackend:
        if reviewer_id != self.reviewer_id or backend_id not in self._backends:
            raise LookupError("backend_not_registered")
        backend = self._backends[backend_id]
        if backend.backend_id != backend_id:
            raise LookupError("backend_id_mismatch")
        return backend


class TrainerResultInvalid(ValueError):
    """A path-free rejection of untrusted backend output."""


def validate_training_result(request: TrainingRequest, result: TrainingResult) -> None:
    """Re-hash and structurally validate an untrusted backend result."""
    _validate_training_result(request, result, request.checkpoint_steps)


def validate_cancelled_training_result(
    request: TrainingRequest, result: TrainingResult, boundary_step: int
) -> None:
    """Validate a cooperative result as one exact non-empty checkpoint prefix."""
    if boundary_step not in request.checkpoint_steps:
        raise TrainerResultInvalid("trainer_cancellation_boundary_invalid")
    prefix = tuple(step for step in request.checkpoint_steps if step <= boundary_step)
    if not prefix or prefix[-1] != boundary_step:
        raise TrainerResultInvalid("trainer_cancellation_boundary_invalid")
    _validate_training_result(request, result, prefix)


def _validate_training_result(
    request: TrainingRequest, result: TrainingResult, expected_steps: tuple[int, ...]
) -> None:
    if (
        result.request_id != request.request_id
        or result.run_id != request.run_id
        or result.backend_id != request.backend_id
        or result.backend_digest != request.materials.backend_digest
        or result.materials != request.materials
        or result.output_root != request.scratch_root
        or result.changed_targets != request.trainable_targets
        or not result.frozen_parameters_unchanged
        or not result.finite_gradients
        or result.elapsed_seconds > request.deadline_seconds
        or result.fault is not None
    ):
        raise TrainerResultInvalid("trainer_result_identity_invalid")
    if len(result.checkpoints) != len(expected_steps):
        raise TrainerResultInvalid("trainer_checkpoint_schedule_invalid")
    try:
        root_info = result.output_root.lstat()
    except OSError:
        raise TrainerResultInvalid("trainer_output_missing") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise TrainerResultInvalid("trainer_output_unsafe")
    root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        root_flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(result.output_root, root_flags)
    except OSError:
        raise TrainerResultInvalid("trainer_output_unsafe") from None
    try:
        parent: str | None = request.parent_checkpoint_id
        all_paths: set[str] = set()
        for checkpoint, expected_step in zip(result.checkpoints, expected_steps, strict=True):
            if (
                checkpoint.step != expected_step
                or not math.isclose(
                    checkpoint.fraction,
                    expected_step / request.total_steps,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or checkpoint.parent_checkpoint_id != parent
            ):
                raise TrainerResultInvalid("trainer_checkpoint_lineage_invalid")
            roles = {item.role for item in checkpoint.files}
            if (
                roles
                != {
                    "weights",
                    "optimizer",
                    "trainer_state",
                    "telemetry",
                    "trainable_targets",
                    "frozen_parameters",
                    "finite_gradient_proof",
                }
                or len(checkpoint.files) != 7
            ):
                raise TrainerResultInvalid("trainer_checkpoint_inventory_invalid")
            role_digests: dict[str, str] = {}
            role_payloads: dict[str, bytes] = {}
            for item in checkpoint.files:
                normalized = unicodedata.normalize("NFC", item.relative_path)
                canonical = posixpath.normpath(normalized)
                if (
                    normalized != item.relative_path
                    or canonical != item.relative_path
                    or canonical in {"", "."}
                    or canonical in all_paths
                ):
                    raise TrainerResultInvalid("trainer_output_unsafe")
                all_paths.add(canonical)
                data = _read_output(root_fd, item.relative_path, item.size_bytes)
                digest = "sha256:" + hashlib.sha256(data).hexdigest()
                if digest != item.digest:
                    raise TrainerResultInvalid("trainer_output_digest_invalid")
                if item.role == "weights":
                    _validate_safetensors(data)
                elif item.role == "telemetry":
                    _validate_telemetry(data, request, checkpoint.step)
                else:
                    role_payloads[item.role] = data
                role_digests[item.role] = digest
            physical_names = _validate_trainable_targets_payload(
                role_payloads["trainable_targets"], request, checkpoint.step
            )
            for role, data in role_payloads.items():
                if role != "trainable_targets":
                    _validate_evidence_payload(
                        data,
                        request,
                        checkpoint.step,
                        role,
                        physical_names=physical_names,
                    )
            if (
                checkpoint.telemetry_digest != role_digests["telemetry"]
                or checkpoint.frozen_parameters_digest != role_digests["frozen_parameters"]
                or checkpoint.trainable_targets_digest != role_digests["trainable_targets"]
                or checkpoint.finite_gradient_proof_digest != role_digests["finite_gradient_proof"]
            ):
                raise TrainerResultInvalid("trainer_checkpoint_evidence_invalid")
            parent = checkpoint.checkpoint_id
        if result.telemetry_digest != result.checkpoints[-1].telemetry_digest:
            raise TrainerResultInvalid("trainer_telemetry_digest_invalid")
        final = os.fstat(root_fd)
        current = result.output_root.lstat()
        if (final.st_dev, final.st_ino) != (root_info.st_dev, root_info.st_ino) or (
            current.st_dev,
            current.st_ino,
        ) != (root_info.st_dev, root_info.st_ino):
            raise TrainerResultInvalid("trainer_output_replaced")
    except TrainerResultInvalid:
        raise
    except (OSError, ValueError, json.JSONDecodeError):
        raise TrainerResultInvalid("trainer_output_invalid") from None
    finally:
        os.close(root_fd)


def _read_output(root_fd: int, relative_path: str, expected_size: int) -> bytes:
    if relative_path.startswith("/") or ".." in relative_path.split("/"):
        raise TrainerResultInvalid("trainer_output_unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(relative_path, flags, dir_fd=root_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise TrainerResultInvalid("trainer_output_size_invalid")
        data = bytearray()
        while len(data) <= expected_size:
            chunk = os.read(descriptor, min(1024 * 1024, expected_size + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) != expected_size or os.fstat(descriptor).st_size != expected_size:
            raise TrainerResultInvalid("trainer_output_size_invalid")
        return bytes(data)
    finally:
        os.close(descriptor)


def _validate_safetensors(data: bytes) -> None:
    if len(data) < 9:
        raise TrainerResultInvalid("trainer_safetensors_invalid")
    header_size = int.from_bytes(data[:8], "little")
    if header_size <= 0 or header_size > len(data) - 8:
        raise TrainerResultInvalid("trainer_safetensors_invalid")
    raw_header = data[8 : 8 + header_size]
    header = json.loads(raw_header)
    if not isinstance(header, dict) or not header:
        raise TrainerResultInvalid("trainer_safetensors_invalid")
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    if canonical + b" " * (header_size - len(canonical)) != raw_header:
        raise TrainerResultInvalid("trainer_safetensors_invalid")
    payload_size = len(data) - 8 - header_size
    ranges: list[tuple[int, int]] = []
    widths = {"U8": 1, "I8": 1, "F16": 2, "BF16": 2, "F32": 4}
    for value in header.values():
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            raise TrainerResultInvalid("trainer_safetensors_invalid")
        offsets = value["data_offsets"]
        if (
            value["dtype"] not in {"U8", "I8", "F16", "BF16", "F32"}
            or not isinstance(value["shape"], list)
            or any(not isinstance(size, int) or size < 0 for size in value["shape"])
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
        ):
            raise TrainerResultInvalid("trainer_safetensors_invalid")
        if offsets[1] > payload_size or offsets[1] < offsets[0]:
            raise TrainerResultInvalid("trainer_safetensors_invalid")
        expected_bytes = math.prod(value["shape"]) * widths[value["dtype"]]
        if offsets[1] - offsets[0] != expected_bytes:
            raise TrainerResultInvalid("trainer_safetensors_invalid")
        ranges.append((offsets[0], offsets[1]))
    if sorted(ranges) != ranges or not ranges or ranges[0][0] != 0 or ranges[-1][1] != payload_size:
        raise TrainerResultInvalid("trainer_safetensors_invalid")
    if any(left[1] != right[0] for left, right in pairwise(ranges)):
        raise TrainerResultInvalid("trainer_safetensors_invalid")


def _validate_evidence_payload(
    data: bytes,
    request: TrainingRequest,
    step: int,
    role: str,
    *,
    physical_names: tuple[str, ...] = (),
) -> None:
    value = json.loads(data)
    shared = {"request_id": request.request_id, "run_id": request.run_id, "step": step}
    expected: dict[str, object]
    if role == "optimizer":
        _validate_optimizer_state(value, request, step, physical_names=physical_names)
        expected = value
    elif role == "trainer_state":
        _validate_trainer_state(value, request, step)
        expected = value
    elif role == "frozen_parameters":
        expected = {
            "schema_version": "trainer-frozen-parameters.v1",
            **shared,
            "trainable_targets": list(request.trainable_targets),
            "unchanged": True,
        }
    elif role == "finite_gradient_proof":
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema_version",
                "request_id",
                "run_id",
                "step",
                "finite",
                "loss",
                "gradient_norm",
            }
            or value.get("schema_version") != "trainer-finite-gradient-proof.v1"
            or any(value.get(key) != expected_value for key, expected_value in shared.items())
            or value.get("finite") is not True
            or not isinstance(value.get("loss"), (int, float))
            or not isinstance(value.get("gradient_norm"), (int, float))
            or not math.isfinite(float(value["loss"]))
            or not math.isfinite(float(value["gradient_norm"]))
            or float(value["gradient_norm"]) < 0
        ):
            raise TrainerResultInvalid("trainer_evidence_payload_invalid")
        expected = value
    else:
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    if canonical_json(value).encode() != data or value != expected:
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")


def _validate_trainable_targets_payload(
    data: bytes, request: TrainingRequest, step: int
) -> tuple[str, ...]:
    value = json.loads(data)
    shared = {"request_id": request.request_id, "run_id": request.run_id, "step": step}
    if request.backend_id != "diffusers.wan2.2-animegen-lora.mps.v1":
        expected = {
            "schema_version": "trainer-trainable-targets.v1",
            **shared,
            "targets": list(request.trainable_targets),
        }
        if value != expected or canonical_json(value).encode() != data:
            raise TrainerResultInvalid("trainer_evidence_payload_invalid")
        return ()
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "request_id", "run_id", "step", "targets", "physical_parameters"
    }:
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    try:
        specs = validate_trainable_parameter_specs(
            trainable_parameter_specs_from_records(value["physical_parameters"]),
            declared_targets=request.trainable_targets,
        )
    except (KeyError, TypeError, ValueError):
        raise TrainerResultInvalid("trainer_evidence_payload_invalid") from None
    if (
        value.get("schema_version") != "trainer-trainable-targets.v2"
        or any(value.get(key) != expected for key, expected in shared.items())
        or value.get("targets") != list(request.trainable_targets)
        or canonical_json(value).encode() != data
    ):
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    return tuple(item.physical_name for item in specs)


def _substantive_numeric_state(value: object) -> tuple[bool, bool]:
    if isinstance(value, bool):
        return True, False
    if isinstance(value, (int, float)):
        finite = math.isfinite(float(value))
        return finite, finite
    if isinstance(value, str):
        return bool(value), False
    if isinstance(value, list):
        children = tuple(_substantive_numeric_state(item) for item in value)
        return bool(children) and all(valid for valid, _ in children), any(
            substantive for _, substantive in children
        )
    if isinstance(value, dict):
        children = tuple(_substantive_numeric_state(item) for item in value.values())
        return (
            bool(children)
            and all(isinstance(key, str) and key for key in value)
            and all(valid for valid, _ in children),
            any(substantive for _, substantive in children),
        )
    return False, False


def _is_substantive_numeric_state(value: object) -> bool:
    valid, substantive = _substantive_numeric_state(value)
    return valid and substantive


def _is_bound_backend_state(value: object, backend_id: str) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {backend_id}
        and _is_substantive_numeric_state(value[backend_id])
    )


def _validate_optimizer_state(
    value: object,
    request: TrainingRequest,
    step: int,
    *,
    physical_names: tuple[str, ...] = (),
) -> None:
    config = {item.name: item.value for item in request.optimizer.parameters}
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "request_id",
        "run_id",
        "step",
        "algorithm",
        "config",
        "global_step",
        "parameter_groups",
        "parameter_state",
        "backend_state",
        "parent_optimizer_state_digest",
    }:
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    groups = value.get("parameter_groups")
    states = value.get("parameter_state")
    if (
        value.get("schema_version") != "trainer-optimizer-state.v2"
        or value.get("request_id") != request.request_id
        or value.get("run_id") != request.run_id
        or value.get("step") != step
        or value.get("global_step") != step
        or value.get("algorithm") != request.optimizer.algorithm
        or value.get("config") != config
        or value.get("parent_optimizer_state_digest") != request.optimizer_state_digest
        or not isinstance(groups, list)
        or not groups
        or not isinstance(states, dict)
        or set(states) != set(request.trainable_targets)
        or not _is_optimizer_backend_state(
            value.get("backend_state"), request, step, config, physical_names=physical_names
        )
    ):
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    covered: list[str] = []
    for group in groups:
        if (
            not isinstance(group, dict)
            or set(group) != {"group_id", "parameters", "hyperparameters"}
            or not isinstance(group.get("group_id"), str)
            or not isinstance(group.get("parameters"), list)
            or group.get("hyperparameters") != config
        ):
            raise TrainerResultInvalid("trainer_evidence_payload_invalid")
        covered.extend(group["parameters"])
    if tuple(covered) != request.trainable_targets:
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    for state in states.values():
        if (
            not isinstance(state, dict)
            or set(state) != {"step", "slots"}
            or state.get("step") != step
            or not _is_substantive_numeric_state(state.get("slots"))
        ):
            raise TrainerResultInvalid("trainer_evidence_payload_invalid")


def _is_optimizer_backend_state(
    value: object,
    request: TrainingRequest,
    step: int,
    config: Mapping[str, object],
    *,
    physical_names: tuple[str, ...] = (),
) -> bool:
    try:
        if request.backend_id != "diffusers.wan2.2-animegen-lora.mps.v1":
            return _is_bound_backend_state(value, request.backend_id)
        if not isinstance(value, dict) or set(value) != {request.backend_id}:
            return False
        state = value[request.backend_id]
        if not isinstance(state, dict):
            return False
        accumulation = config.get("gradient_accumulation_steps", 1)
        if type(accumulation) is not int or accumulation <= 0:
            return False
        residual = step % accumulation
        terminal_partial = step == request.total_steps and residual > 0
        position = 0 if residual == 0 or terminal_partial else residual
        expected_updates = step // accumulation + int(terminal_partial)
        if accumulation == 1:
            return state == {"optimizer_updates": expected_updates}
        if set(state) != {"optimizer_updates", "pending_gradients"}:
            return False
        if state["optimizer_updates"] != expected_updates:
            return False
        validate_pending_gradients(
            state["pending_gradients"],
            position,
            expected_parameters=physical_names,
        )
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _validate_trainer_state(value: object, request: TrainingRequest, step: int) -> None:
    common_fields = {
        "schema_version",
        "request_id",
        "run_id",
        "step",
        "backend_id",
        "seed",
        "total_steps",
        "global_step",
        "scheduler_state",
        "scaler_state",
        "backend_state",
    }
    if not isinstance(value, dict):
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    schema = value.get("schema_version")
    rng_field = "rng_states" if schema == "trainer-state.v3" else "rng_state"
    if set(value) != common_fields | {rng_field}:
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")
    if (
        schema not in {"trainer-state.v2", "trainer-state.v3"}
        or value.get("request_id") != request.request_id
        or value.get("run_id") != request.run_id
        or value.get("step") != step
        or value.get("global_step") != step
        or value.get("backend_id") != request.backend_id
        or value.get("seed") != request.seed
        or value.get("total_steps") != request.total_steps
        or (
            schema == "trainer-state.v2"
            and value.get("rng_state") != {"seed": request.seed, "counter": step}
        )
        or (schema == "trainer-state.v3" and not _valid_rng_states(value.get("rng_states")))
        or not _is_substantive_numeric_state(value.get("scheduler_state"))
        or not _is_substantive_numeric_state(value.get("scaler_state"))
        or not _is_bound_backend_state(value.get("backend_state"), request.backend_id)
    ):
        raise TrainerResultInvalid("trainer_evidence_payload_invalid")


def _valid_rng_states(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for key, item in value.items():
        if (
            type(key) is not str
            or not key
            or not isinstance(item, dict)
            or set(item) != {"algorithm", "device", "state_hex", "digest"}
            or type(item.get("algorithm")) is not str
            or not item.get("algorithm")
            or item.get("device") != key
        ):
            return False
        state_hex = item.get("state_hex")
        digest = item.get("digest")
        if (
            type(state_hex) is not str
            or not state_hex
            or state_hex != state_hex.lower()
            or len(state_hex) % 2
            or type(digest) is not str
        ):
            return False
        try:
            state = bytes.fromhex(state_hex)
        except ValueError:
            return False
        if not state or "sha256:" + hashlib.sha256(state).hexdigest() != digest:
            return False
    return True


def _validate_telemetry(data: bytes, request: TrainingRequest, checkpoint_step: int) -> None:
    value = json.loads(data)
    if canonical_json(value).encode() != data or not isinstance(value, list) or not value:
        raise TrainerResultInvalid("trainer_telemetry_payload_invalid")
    expected_sequence = request.telemetry_sequence_start
    expected_step = request.start_step + 1
    for record in value:
        if not isinstance(record, dict) or set(record) != {"sequence", "step", "payload"}:
            raise TrainerResultInvalid("trainer_telemetry_payload_invalid")
        if record["sequence"] != expected_sequence or record["step"] != expected_step:
            raise TrainerResultInvalid("trainer_telemetry_payload_invalid")
        try:
            progress = TrainingProgress.model_validate(record["payload"])
        except ValueError:
            raise TrainerResultInvalid("trainer_telemetry_payload_invalid") from None
        if progress.step != expected_step or progress.total_steps != request.total_steps:
            raise TrainerResultInvalid("trainer_telemetry_payload_invalid")
        expected_sequence += 1
        expected_step += 1
    if expected_step - 1 != checkpoint_step:
        raise TrainerResultInvalid("trainer_telemetry_payload_invalid")
