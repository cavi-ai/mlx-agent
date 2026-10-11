"""Deterministic trainer used to pressure-test orchestration without ML dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from video_lab.domain import canonical_json
from video_lab.trainer_protocol import (
    CalibrationRequest,
    CalibrationResult,
    CancellationCheck,
    CheckpointResult,
    OutputFile,
    ProbeTrainingRequest,
    TrainerCapabilityReport,
    TrainingCancelled,
    TrainingProgress,
    TrainingProgressCallback,
    TrainingRequest,
    TrainingResult,
)
from video_lab.training_models import checkpoint_steps


class FakeTrainerFault(StrEnum):
    NONFINITE_GRADIENT = "nonfinite_gradient"
    FROZEN_MUTATION = "frozen_mutation"
    CORRUPT_CHECKPOINT = "corrupt_checkpoint"
    MISSING_CHECKPOINT = "missing_checkpoint"
    EXTRA_CHECKPOINT = "extra_checkpoint"
    WRONG_STEP = "wrong_step"
    WRONG_FRACTION = "wrong_fraction"
    WRONG_PARENT = "wrong_parent"
    WRONG_TARGET = "wrong_target"
    DEADLINE = "deadline"
    CRASH = "crash"


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _identity(payload: object) -> str:
    return _sha(canonical_json(cast(Any, payload)).encode())


class FakeTrainerBackend:
    def __init__(
        self,
        *,
        backend_id: str = "fake",
        fault: FakeTrainerFault | None = None,
    ) -> None:
        self.backend_id = backend_id
        self.fault = fault

    @property
    def backend_digest(self) -> str:
        return _identity({"backend": self.backend_id, "version": 1})

    def probe(self, request: ProbeTrainingRequest) -> TrainerCapabilityReport:
        self._require_backend(request.backend_id)
        return TrainerCapabilityReport(
            backend_id=self.backend_id,
            backend_digest=request.expected_backend_digest,
            runtime_digest=request.materials.runtime_digest,
            supported=True,
            max_steps=1_000_000,
            safe_boundaries=(1, 2, 3, 4),
        )

    def calibrate(
        self,
        request: CalibrationRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> CalibrationResult:
        self._require_backend(request.backend_id)
        for step in range(1, request.steps + 1):
            item = TrainingProgress(
                step=step,
                total_steps=request.steps,
                fraction=step / request.steps,
                loss=1 / (step + 1),
                checkpoint_boundary=step == request.steps,
            )
            progress(item)
            if cancelled() and step == request.steps:
                raise TrainingCancelled(step)
        payload = {
            "request": request.model_dump(mode="json", exclude={"scratch_root"}),
            "backend": self.backend_digest,
        }
        return CalibrationResult(
            request_id=request.request_id,
            run_id=request.run_id,
            backend_digest=request.materials.backend_digest,
            runtime_digest=request.materials.runtime_digest,
            calibration_digest=_identity(payload),
            finite_gradients=self.fault != FakeTrainerFault.NONFINITE_GRADIENT,
            frozen_parameters_unchanged=self.fault != FakeTrainerFault.FROZEN_MUTATION,
            changed_targets=request.trainable_targets,
            measurements={"final_loss": 1 / (request.steps + 1), "gradient_norm": 1.0},
        )

    def train(
        self,
        request: TrainingRequest,
        progress: TrainingProgressCallback,
        cancelled: CancellationCheck,
    ) -> TrainingResult:
        self._require_backend(request.backend_id)
        root_fd, root_identity = self._open_scratch(request.scratch_root)
        try:
            if self.fault == FakeTrainerFault.CRASH:
                raise RuntimeError("fake_trainer_crash")
            checkpoints: list[CheckpointResult] = []
            parent: str | None = request.parent_checkpoint_id
            telemetry: list[dict[str, object]] = []
            for step in range(request.start_step + 1, request.total_steps + 1):
                loss = 1 / (step + request.seed + 1)
                if self.fault == FakeTrainerFault.NONFINITE_GRADIENT and step == 1:
                    loss = math.nan
                try:
                    item = TrainingProgress(
                        step=step,
                        total_steps=request.total_steps,
                        fraction=step / request.total_steps,
                        loss=loss,
                        checkpoint_boundary=step in request.checkpoint_steps,
                    )
                except ValueError:
                    item = TrainingProgress(
                        step=step,
                        total_steps=request.total_steps,
                        fraction=step / request.total_steps,
                        loss=0.0,
                        checkpoint_boundary=step in request.checkpoint_steps,
                    )
                progress(item)
                telemetry.append(item.model_dump(mode="json"))
                if step in request.checkpoint_steps:
                    ordinal = checkpoint_steps(request.total_steps).index(step)
                    checkpoint = self._checkpoint(
                        request, root_fd, ordinal, step, parent, telemetry
                    )
                    checkpoints.append(checkpoint)
                    parent = checkpoint.checkpoint_id
                    if cancelled():
                        self._reverify_root(request.scratch_root, root_fd, root_identity)
                        raise TrainingCancelled(
                            step,
                            TrainingResult(
                                request_id=request.request_id,
                                run_id=request.run_id,
                                backend_id=self.backend_id,
                                backend_digest=request.materials.backend_digest,
                                materials=request.materials,
                                output_root=request.scratch_root,
                                checkpoints=tuple(checkpoints),
                                telemetry_digest=checkpoint.telemetry_digest,
                                changed_targets=request.trainable_targets,
                                frozen_parameters_unchanged=True,
                                finite_gradients=True,
                                elapsed_seconds=0,
                            ),
                        )
            if self.fault == FakeTrainerFault.MISSING_CHECKPOINT:
                checkpoints.pop()
            elif self.fault == FakeTrainerFault.EXTRA_CHECKPOINT:
                checkpoints.append(checkpoints[-1].model_copy(update={"checkpoint_id": "extra"}))
            elif self.fault == FakeTrainerFault.WRONG_STEP:
                checkpoints[0] = checkpoints[0].model_copy(update={"step": 1})
            elif self.fault == FakeTrainerFault.WRONG_FRACTION:
                checkpoints[0] = checkpoints[0].model_copy(update={"fraction": 0.5})
            elif self.fault == FakeTrainerFault.WRONG_PARENT:
                checkpoints[1] = checkpoints[1].model_copy(update={"parent_checkpoint_id": "wrong"})
            self._reverify_root(request.scratch_root, root_fd, root_identity)
            telemetry_digest = checkpoints[-1].telemetry_digest
            return TrainingResult(
                request_id=request.request_id,
                run_id=request.run_id,
                backend_id=self.backend_id,
                backend_digest=request.materials.backend_digest,
                materials=request.materials,
                output_root=request.scratch_root,
                checkpoints=tuple(checkpoints),
                telemetry_digest=telemetry_digest,
                changed_targets=(
                    ("undeclared.target",)
                    if self.fault == FakeTrainerFault.WRONG_TARGET
                    else request.trainable_targets
                ),
                frozen_parameters_unchanged=self.fault != FakeTrainerFault.FROZEN_MUTATION,
                finite_gradients=self.fault != FakeTrainerFault.NONFINITE_GRADIENT,
                elapsed_seconds=(
                    request.deadline_seconds + 1
                    if self.fault == FakeTrainerFault.DEADLINE
                    else float(request.total_steps)
                ),
                fault=self.fault.value if self.fault else None,
            )
        finally:
            os.close(root_fd)

    def _checkpoint(
        self,
        request: TrainingRequest,
        root_fd: int,
        ordinal: int,
        step: int,
        parent: str | None,
        telemetry: list[dict[str, object]],
    ) -> CheckpointResult:
        checkpoint_id = f"checkpoint-{ordinal + 1}"
        prefix = f"checkpoint-{ordinal + 1}"
        roles = (
            "weights",
            "optimizer",
            "trainer_state",
            "telemetry",
            "trainable_targets",
            "frozen_parameters",
            "finite_gradient_proof",
        )
        files: list[OutputFile] = []
        for role in roles:
            name = f"{prefix}-{role}.safetensors" if role == "weights" else f"{prefix}-{role}.json"
            payload = self._file_bytes(request, step, role, telemetry)
            if (
                self.fault == FakeTrainerFault.CORRUPT_CHECKPOINT
                and ordinal == 0
                and role == "weights"
            ):
                payload = b"corrupt"
            self._write_file(root_fd, name, payload)
            files.append(
                OutputFile(
                    relative_path=name,
                    digest=_sha(payload),
                    size_bytes=len(payload),
                    role=role,
                )
            )
        by_role = {item.role: item.digest for item in files}
        return CheckpointResult(
            checkpoint_id=checkpoint_id,
            step=step,
            fraction=step / request.total_steps,
            parent_checkpoint_id=parent,
            files=tuple(files),
            telemetry_digest=by_role["telemetry"],
            frozen_parameters_digest=by_role["frozen_parameters"],
            trainable_targets_digest=by_role["trainable_targets"],
            finite_gradient_proof_digest=by_role["finite_gradient_proof"],
        )

    def _file_bytes(
        self, request: TrainingRequest, step: int, role: str, telemetry: list[dict[str, object]]
    ) -> bytes:
        if role == "telemetry":
            payload = canonical_json(
                cast(
                    Any,
                    [
                        {"sequence": index, "step": item["step"], "payload": item}
                        for index, item in enumerate(
                            telemetry, start=request.telemetry_sequence_start
                        )
                    ],
                )
            ).encode()
        elif role == "optimizer":
            config = {item.name: item.value for item in request.optimizer.parameters}
            payload = canonical_json(
                {
                    "schema_version": "trainer-optimizer-state.v2",
                    "request_id": request.request_id,
                    "run_id": request.run_id,
                    "step": step,
                    "algorithm": request.optimizer.algorithm,
                    "config": config,
                    "global_step": step,
                    "parameter_groups": [
                        {
                            "group_id": "default",
                            "parameters": list(request.trainable_targets),
                            "hyperparameters": config,
                        }
                    ],
                    "parameter_state": {
                        target: {
                            "step": step,
                            "slots": {"primary": step / 10, "secondary": step / 100},
                        }
                        for target in request.trainable_targets
                    },
                    "backend_state": {self.backend_id: {"restored_updates": step}},
                    "parent_optimizer_state_digest": request.optimizer_state_digest,
                }
            ).encode()
        elif role == "trainer_state":
            payload = canonical_json(
                {
                    "schema_version": "trainer-state.v2",
                    "request_id": request.request_id,
                    "run_id": request.run_id,
                    "step": step,
                    "backend_id": request.backend_id,
                    "seed": request.seed,
                    "total_steps": request.total_steps,
                    "global_step": step,
                    "rng_state": {"seed": request.seed, "counter": step},
                    "scheduler_state": {"step": step, "rate_scale": 1.0},
                    "scaler_state": {"enabled": False, "scale": 1.0},
                    "backend_state": {self.backend_id: {"checkpoint_step": step}},
                }
            ).encode()
        elif role == "trainable_targets":
            payload = canonical_json(
                {
                    "schema_version": "trainer-trainable-targets.v1",
                    "request_id": request.request_id,
                    "run_id": request.run_id,
                    "step": step,
                    "targets": list(request.trainable_targets),
                }
            ).encode()
        elif role == "frozen_parameters":
            payload = canonical_json(
                {
                    "schema_version": "trainer-frozen-parameters.v1",
                    "request_id": request.request_id,
                    "run_id": request.run_id,
                    "step": step,
                    "trainable_targets": list(request.trainable_targets),
                    "unchanged": True,
                }
            ).encode()
        elif role == "finite_gradient_proof":
            latest = telemetry[-1]
            payload = canonical_json(
                {
                    "schema_version": "trainer-finite-gradient-proof.v1",
                    "request_id": request.request_id,
                    "run_id": request.run_id,
                    "step": step,
                    "finite": True,
                    "loss": latest["loss"],
                    "gradient_norm": latest["gradient_norm"],
                }
            ).encode()
        elif role == "weights":
            payload = canonical_json(
                {
                    "request_id": request.request_id,
                    "run_id": request.run_id,
                    "step": step,
                    "trainable_targets": list(request.trainable_targets),
                }
            ).encode()
        else:
            payload = b""
        if role != "weights":
            return payload
        header = json.dumps(
            {"weight": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        padding = (-len(header)) % 8
        header += b" " * padding
        return len(header).to_bytes(8, "little") + header + payload

    @staticmethod
    def _write_file(root_fd: int, name: str, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(name, flags, 0o600, dir_fd=root_fd)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_scratch(root: Path) -> tuple[int, tuple[int, int]]:
        try:
            if root.exists() or root.is_symlink():
                info = root.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise ValueError("unsafe_scratch_root")
            else:
                root.mkdir(mode=0o700, parents=True)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(root, flags)
            info = os.fstat(descriptor)
            return descriptor, (info.st_dev, info.st_ino)
        except OSError:
            raise ValueError("unsafe_scratch_root") from None

    @staticmethod
    def _reverify_root(root: Path, descriptor: int, identity: tuple[int, int]) -> None:
        try:
            held = os.fstat(descriptor)
            current = root.lstat()
        except OSError:
            raise ValueError("unsafe_scratch_root") from None
        if (
            stat.S_ISLNK(current.st_mode)
            or (held.st_dev, held.st_ino) != identity
            or (
                current.st_dev,
                current.st_ino,
            )
            != identity
        ):
            raise ValueError("unsafe_scratch_root")

    def _require_backend(self, requested: str) -> None:
        if requested != self.backend_id:
            raise ValueError("backend_id_mismatch")
