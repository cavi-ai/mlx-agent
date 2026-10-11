"""Durable, dependency-free checkpoint publication for the Wan trainer."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import sys
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from video_lab.domain import canonical_json
from video_lab.trainer_protocol import CheckpointResult, OutputFile, TrainingProgress
from video_lab.wan_pending import (
    TrainableParameterSpec,
    trainable_parameter_specs_from_records,
    validate_pending_gradients,
    validate_trainable_parameter_specs,
)

_ROLES = (
    "weights",
    "optimizer",
    "trainer_state",
    "telemetry",
    "trainable_targets",
    "frozen_parameters",
    "finite_gradient_proof",
)


@dataclass(frozen=True, slots=True)
class WanCheckpointState:
    """Complete in-memory state for one independently provable boundary."""

    checkpoint_id: str
    step: int
    total_steps: int
    parent_checkpoint_id: str | None
    weights: bytes
    optimizer: Mapping[str, object]
    trainer_state: Mapping[str, object]
    telemetry: tuple[Mapping[str, object], ...]
    trainable_targets: Mapping[str, object]
    frozen_parameters: Mapping[str, object]
    finite_gradient_proof: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class WanBranchTensorSpecV2:
    """Path-free physical identity of one adapter tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class WanBranchCheckpointManifestV2:
    """Explicit v2 authority for production branch continuation."""

    schema_version: str
    branch: str
    selector: int
    step: int
    total_steps: int
    parent_checkpoint_id: str | None
    batch_digest: str
    recipe_digest: str
    component_inventory_digest: str
    adapter_tensors: tuple[WanBranchTensorSpecV2, ...]
    optimizer_parameter_names: tuple[str, ...]
    optimizer_slot_names: tuple[str, ...]
    cpu_rng_digest: str
    mps_rng_digest: str
    finite: bool


@dataclass(frozen=True, slots=True)
class WanBranchCheckpointStateV2:
    """Adapter-only bytes and continuation state validated against a v2 manifest."""

    manifest: WanBranchCheckpointManifestV2
    weights: bytes
    optimizer: Mapping[str, object]
    cpu_rng_state: bytes
    mps_rng_state: bytes


@dataclass(frozen=True, slots=True)
class WanBranchRestoreReceiptV2:
    """Observed equality proof published only after a fresh-runtime restore."""

    schema_version: str
    manifest_digest: str
    weights_digest: str
    optimizer_digest: str
    cpu_rng_digest: str
    mps_rng_digest: str


def validate_wan_branch_checkpoint_v2(
    state: WanBranchCheckpointStateV2,
    expected: WanBranchCheckpointManifestV2,
) -> WanBranchCheckpointStateV2:
    """Validate every identity before a caller is allowed to mutate a runtime."""

    try:
        manifest = state.manifest
        specs = manifest.adapter_tensors
        names = tuple(item.name for item in specs)
        tensors = _inspect_branch_safetensors_v2(state.weights)
        expected_tensors = {
            item.name: (item.dtype, item.shape)
            for item in specs
            if type(item) is WanBranchTensorSpecV2
        }
        parameter_state = state.optimizer["parameter_state"]
        if (
            type(state) is not WanBranchCheckpointStateV2
            or type(expected) is not WanBranchCheckpointManifestV2
            or manifest != expected
            or manifest.schema_version != "wan-branch-checkpoint.v2"
            or manifest.branch not in {"high_noise", "low_noise"}
            or manifest.selector != (0 if manifest.branch == "high_noise" else 1)
            or manifest.step != 1
            or manifest.total_steps != 1
            or not manifest.finite
            or not specs
            or names != tuple(sorted(set(names)))
            or tensors != expected_tensors
            or manifest.optimizer_parameter_names != names
            or not manifest.optimizer_slot_names
            or set(state.optimizer) != {"schema_version", "parameter_state"}
            or state.optimizer.get("schema_version") != "wan-branch-optimizer-state.v2"
            or not isinstance(parameter_state, Mapping)
            or set(parameter_state) != set(names)
            or any(
                not isinstance(parameter_state[name], Mapping)
                or set(cast(Mapping[str, object], parameter_state[name]))
                != set(manifest.optimizer_slot_names)
                for name in names
            )
            or not _finite_json(state.optimizer)
            or not state.cpu_rng_state
            or not state.mps_rng_state
            or "sha256:" + hashlib.sha256(state.cpu_rng_state).hexdigest()
            != manifest.cpu_rng_digest
            or "sha256:" + hashlib.sha256(state.mps_rng_state).hexdigest()
            != manifest.mps_rng_digest
            or any(
                not _is_sha256_digest(value)
                for value in (
                    manifest.batch_digest,
                    manifest.recipe_digest,
                    manifest.component_inventory_digest,
                    manifest.cpu_rng_digest,
                    manifest.mps_rng_digest,
                )
            )
        ):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("wan_branch_checkpoint_v2_invalid") from None
    return state


def _inspect_branch_safetensors_v2(data: bytes) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Inspect standard safetensors without imposing the legacy canonical-header encoding."""

    try:
        if len(data) < 9:
            raise ValueError
        header_size = int.from_bytes(data[:8], "little")
        if header_size <= 0 or header_size > len(data) - 8:
            raise ValueError
        header = json.loads(data[8 : 8 + header_size])
        payload_size = len(data) - 8 - header_size
        widths = {"F16": 2, "BF16": 2, "F32": 4}
        tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
        ranges: list[tuple[int, int]] = []
        if not isinstance(header, dict) or not header:
            raise ValueError
        for name, value in header.items():
            if name == "__metadata__":
                raise ValueError
            if (
                type(name) is not str
                or not name
                or not isinstance(value, dict)
                or set(value) != {"dtype", "shape", "data_offsets"}
            ):
                raise ValueError
            dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
            if (
                dtype not in widths
                or not isinstance(shape, list)
                or not shape
                or any(type(size) is not int or size <= 0 for size in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or offsets[0] < 0
                or offsets[1] - offsets[0] != math.prod(shape) * widths[dtype]
                or offsets[1] > payload_size
            ):
                raise ValueError
            tensors[name] = (dtype, tuple(shape))
            ranges.append((offsets[0], offsets[1]))
            raw = data[8 + header_size + offsets[0] : 8 + header_size + offsets[1]]
            width = widths[dtype]
            exponent_mask = (
                0x7C00
                if dtype == "F16"
                else 0x7F80
                if dtype == "BF16"
                else 0x7F800000
            )
            for index in range(0, len(raw), width):
                bits = int.from_bytes(raw[index : index + width], "little")
                if bits & exponent_mask == exponent_mask:
                    raise ValueError
        if (
            not tensors
            or sorted(ranges) != ranges
            or ranges[0][0] != 0
            or ranges[-1][1] != payload_size
            or any(left[1] != right[0] for left, right in pairwise(ranges))
        ):
            raise ValueError
        return tensors
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("wan_branch_checkpoint_v2_invalid") from None


OptimizerValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class WanCheckpointValidationContext:
    """Immutable request facts that checkpoint payloads cannot self-authorize."""

    checkpoint_id: str
    request_id: str
    run_id: str
    backend_id: str
    seed: int
    step: int
    total_steps: int
    parent_checkpoint_id: str | None
    parent_optimizer_state_digest: str | None
    trainable_targets: tuple[str, ...]
    trainable_parameter_specs: tuple[TrainableParameterSpec, ...]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    optimizer_algorithm: str
    optimizer_config: tuple[tuple[str, OptimizerValue], ...]
    start_step: int = 0
    telemetry_sequence_start: int = 1


@dataclass(frozen=True, slots=True)
class WanParentCheckpointFileDescriptor:
    """Factory-authorized identity for one immutable parent artifact."""

    role: str
    path: Path
    digest: str
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class WanParentCheckpointDescriptor:
    """Exact seven-file inventory selected by orchestration for resume."""

    checkpoint_id: str
    checkpoint_digest: str
    files: tuple[WanParentCheckpointFileDescriptor, ...]
    step: int = 0
    total_steps: int = 0
    parent_checkpoint_id: str | None = None
    parent_optimizer_state_digest: str | None = None


@dataclass(frozen=True, slots=True)
class _RestoredAncestorLease:
    name: str
    descriptor: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _RestoredFileLease:
    descriptor: WanParentCheckpointFileDescriptor
    file_descriptor: int
    path: Path
    ancestors: tuple[_RestoredAncestorLease, ...]


class WanCheckpointCleanupError(RuntimeError):
    """A checkpoint cleanup failed, optionally after a primary operation error."""

    def __init__(
        self,
        primary_error: BaseException | None,
        cleanup_errors: tuple[BaseException, ...],
    ) -> None:
        if not cleanup_errors:
            raise ValueError("wan_checkpoint_cleanup_errors_missing")
        self.primary_error = primary_error
        self.cleanup_errors = cleanup_errors
        super().__init__("wan_checkpoint_cleanup_failed")


class WanCheckpointCodec:
    """Publish complete checkpoint directories relative to one retained root."""

    def __init__(self, scratch_root: Path) -> None:
        self._root = scratch_root
        self._root_fd, self._root_identity = _open_scratch_root(scratch_root)
        self._closed = False
        self._restored: tuple[_RestoredFileLease, ...] = ()

    def __enter__(self) -> WanCheckpointCodec:
        if self._closed:
            raise ValueError("wan_checkpoint_codec_closed")
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: object | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for lease in self._restored:
                os.close(lease.file_descriptor)
                for ancestor in lease.ancestors:
                    os.close(ancestor.descriptor)
            self._restored = ()
            os.close(self._root_fd)

    def restore(
        self,
        descriptor: WanParentCheckpointDescriptor,
        context: WanCheckpointValidationContext,
    ) -> WanCheckpointState:
        """Stable-open, rehash, and decode one exact parent inventory."""
        if self._closed or self._restored:
            raise ValueError("wan_checkpoint_restore_invalid")
        if (
            type(descriptor) is not WanParentCheckpointDescriptor
            or descriptor.checkpoint_id != context.checkpoint_id
            or not _is_sha256_digest(descriptor.checkpoint_digest)
            or type(descriptor.files) is not tuple
            or len(descriptor.files) != len(_ROLES)
        ):
            raise ValueError("wan_checkpoint_restore_invalid")
        if any(
            type(item) is not WanParentCheckpointFileDescriptor
            or type(item.role) is not str
            or not isinstance(item.path, Path)
            or type(item.digest) is not str
            or type(item.size_bytes) is not int
            or type(item.device) is not int
            or type(item.inode) is not int
            for item in descriptor.files
        ):
            raise ValueError("wan_checkpoint_restore_invalid")
        by_role = {item.role: item for item in descriptor.files}
        if set(by_role) != set(_ROLES) or len(by_role) != len(descriptor.files):
            raise ValueError("wan_checkpoint_restore_invalid")
        leases: list[_RestoredFileLease] = []
        payloads: dict[str, bytes] = {}
        seen_paths: set[Path] = set()
        seen_identities: set[tuple[int, int]] = set()
        try:
            for role in _ROLES:
                item = by_role[role]
                if (
                    type(item) is not WanParentCheckpointFileDescriptor
                    or item.size_bytes <= 0
                    or not _is_sha256_digest(item.digest)
                ):
                    raise ValueError("wan_checkpoint_restore_invalid")
                path = item.path
                path = Path(*path.parts)
                identity = (item.device, item.inode)
                if path in seen_paths or identity in seen_identities:
                    raise ValueError("wan_checkpoint_restore_invalid")
                seen_paths.add(path)
                seen_identities.add(identity)
                lease = _open_exact_parent_file(path, item)
                leases.append(lease)
                payload = _read_open_regular(lease.file_descriptor, item.size_bytes)
                if "sha256:" + hashlib.sha256(payload).hexdigest() != item.digest:
                    raise ValueError("wan_checkpoint_restore_invalid")
                payloads[role] = payload
            state = _state_from_payloads(payloads, context)
            historical_context = _historical_validation_context(state, context)
            _validated_payloads(state, historical_context)
            self._restored = tuple(leases)
            self.reverify_restored()
            return state
        except BaseException:
            self._restored = ()
            for lease in leases:
                os.close(lease.file_descriptor)
                for ancestor in lease.ancestors:
                    os.close(ancestor.descriptor)
            raise ValueError("wan_checkpoint_restore_invalid") from None
    def reverify_restored(self) -> None:
        """Prove retained parent handles and their authorized paths still agree."""
        for lease in self._restored:
            info = os.fstat(lease.file_descriptor)
            try:
                _reverify_ancestor_chain(lease)
            except OSError:
                raise ValueError("wan_checkpoint_restore_invalid") from None
            if (
                not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino)
                != (lease.descriptor.device, lease.descriptor.inode)
                or info.st_size != lease.descriptor.size_bytes
            ):
                raise ValueError("wan_checkpoint_restore_invalid")

    def discard(self, checkpoint_id: str) -> None:
        """Remove a checkpoint that must not escape a failed training transaction."""
        if self._closed:
            return
        name = _safe_component(checkpoint_id)
        files = tuple(
            "weights.safetensors" if role == "weights" else f"{role}.json" for role in _ROLES
        )
        _remove_checkpoint(self._root_fd, name, files, -1)

    def write(
        self, state: WanCheckpointState, context: WanCheckpointValidationContext
    ) -> CheckpointResult:
        if self._closed:
            raise ValueError("wan_checkpoint_codec_closed")
        snapshot, payloads = _validated_payloads(state, context)
        checkpoint_name = _safe_component(snapshot.checkpoint_id)
        staging_name = f".{checkpoint_name}.staging"
        _reverify_root(self._root, self._root_fd, self._root_identity)
        staging_fd = -1
        staging_created = False
        published = False
        written: list[str] = []
        files: list[OutputFile] = []
        operation_error: BaseException | None = None
        close_errors: list[BaseException] = []
        try:
            os.mkdir(staging_name, 0o700, dir_fd=self._root_fd)
            staging_created = True
            staging_fd = _open_directory(staging_name, self._root_fd)
            for role, payload in payloads:
                name = "weights.safetensors" if role == "weights" else f"{role}.json"
                _write_exclusive(staging_fd, name, payload)
                written.append(name)
                reopened = _read_regular(staging_fd, name, len(payload))
                if reopened != payload:
                    raise OSError("checkpoint bytes changed after fsync")
                digest = "sha256:" + hashlib.sha256(reopened).hexdigest()
                files.append(
                    OutputFile(
                        relative_path=f"{checkpoint_name}/{name}",
                        digest=digest,
                        size_bytes=len(reopened),
                        role=role,
                    )
                )
            _fsync_directory(staging_fd)
            _reverify_root(self._root, self._root_fd, self._root_identity)
            _require_destination_absent(self._root_fd, checkpoint_name)
            _publish_directory_no_replace(self._root_fd, staging_name, checkpoint_name)
            published = True
            _fsync_directory(self._root_fd)
            _reverify_root(self._root, self._root_fd, self._root_identity)
        except BaseException as error:
            operation_error = error
        finally:
            if staging_fd >= 0:
                try:
                    _close_directory(staging_fd)
                except BaseException as error:
                    close_errors.append(error)
                staging_fd = -1
        if operation_error is not None or close_errors:
            cleanup_errors = list(close_errors)
            if staging_created:
                cleanup_name = checkpoint_name if published else staging_name
                try:
                    _remove_checkpoint(self._root_fd, cleanup_name, tuple(written), -1)
                except WanCheckpointCleanupError as cleanup_error:
                    cleanup_errors.extend(cleanup_error.cleanup_errors)
            if cleanup_errors:
                raise WanCheckpointCleanupError(
                    operation_error, tuple(cleanup_errors)
                ) from operation_error
            raise ValueError("wan_checkpoint_write_failed") from None
        by_role = {item.role: item.digest for item in files}
        return CheckpointResult(
            checkpoint_id=snapshot.checkpoint_id,
            step=snapshot.step,
            fraction=snapshot.step / snapshot.total_steps,
            parent_checkpoint_id=snapshot.parent_checkpoint_id,
            files=tuple(files),
            telemetry_digest=by_role["telemetry"],
            frozen_parameters_digest=by_role["frozen_parameters"],
            trainable_targets_digest=by_role["trainable_targets"],
            finite_gradient_proof_digest=by_role["finite_gradient_proof"],
        )


def _historical_validation_context(
    state: WanCheckpointState, context: WanCheckpointValidationContext
) -> WanCheckpointValidationContext:
    """Recover request-local fields from the digest-bound checkpoint payload."""
    request_id = state.trainable_targets.get("request_id")
    if type(request_id) is not str or not request_id or not state.telemetry:
        raise ValueError("wan_checkpoint_restore_invalid")
    first = state.telemetry[0]
    if not isinstance(first, Mapping):
        raise ValueError("wan_checkpoint_restore_invalid")
    first_step = first.get("step")
    first_sequence = first.get("sequence")
    if type(first_step) is not int or type(first_sequence) is not int:
        raise ValueError("wan_checkpoint_restore_invalid")
    return replace(
        context,
        request_id=request_id,
        start_step=first_step - 1,
        telemetry_sequence_start=first_sequence,
    )


def _validated_payloads(
    state: WanCheckpointState, context: WanCheckpointValidationContext
) -> tuple[WanCheckpointState, tuple[tuple[str, bytes], ...]]:
    if type(state) is not WanCheckpointState:
        raise ValueError("wan_checkpoint_state_invalid")
    checkpoint_id = state.checkpoint_id
    step = state.step
    total_steps = state.total_steps
    parent_checkpoint_id = state.parent_checkpoint_id
    weights = state.weights
    optimizer = state.optimizer
    trainer_state = state.trainer_state
    telemetry = state.telemetry
    trainable_targets = state.trainable_targets
    frozen_parameters = state.frozen_parameters
    finite_gradient_proof = state.finite_gradient_proof
    owned_context = _owned_validation_context(context)
    if (
        type(step) is not int
        or type(total_steps) is not int
        or step <= 0
        or total_steps < step
        or type(weights) is not bytes
        or not telemetry
    ):
        raise ValueError("wan_checkpoint_state_invalid")
    _validate_context(owned_context)
    _safe_component(checkpoint_id)
    if parent_checkpoint_id is not None:
        _safe_component(parent_checkpoint_id)
    if (
        checkpoint_id != owned_context.checkpoint_id
        or step != owned_context.step
        or total_steps != owned_context.total_steps
        or parent_checkpoint_id != owned_context.parent_checkpoint_id
    ):
        raise ValueError("wan_checkpoint_state_invalid")
    try:
        json_payloads = tuple(
            _snapshot_json_payload(value)
            for value in (
                optimizer,
                trainer_state,
                list(telemetry),
                trainable_targets,
                frozen_parameters,
                finite_gradient_proof,
            )
        )
        decoded = tuple(json.loads(payload) for payload in json_payloads)
        snapshot = WanCheckpointState(
            checkpoint_id=checkpoint_id,
            step=step,
            total_steps=total_steps,
            parent_checkpoint_id=parent_checkpoint_id,
            weights=weights,
            optimizer=cast(dict[str, object], decoded[0]),
            trainer_state=cast(dict[str, object], decoded[1]),
            telemetry=tuple(cast(list[dict[str, object]], decoded[2])),
            trainable_targets=cast(dict[str, object], decoded[3]),
            frozen_parameters=cast(dict[str, object], decoded[4]),
            finite_gradient_proof=cast(dict[str, object], decoded[5]),
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("wan_checkpoint_state_invalid") from None
    if any(not _finite_json(value) for value in decoded):
        raise ValueError("wan_checkpoint_state_invalid")
    _validate_role_schemas(snapshot, owned_context)
    tensors = _validate_safetensors(snapshot.weights)
    targets = snapshot.trainable_targets.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or any(type(target) is not str or not target for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("wan_checkpoint_state_invalid")
    _validate_lora_tensors(tensors, cast(list[str], targets), owned_context.lora_rank)
    payloads = tuple(zip(_ROLES, (weights, *json_payloads), strict=True))
    return snapshot, payloads


def _owned_validation_context(
    context: WanCheckpointValidationContext,
) -> WanCheckpointValidationContext:
    if type(context) is not WanCheckpointValidationContext:
        raise ValueError("wan_checkpoint_state_invalid")
    checkpoint_id = context.checkpoint_id
    request_id = context.request_id
    run_id = context.run_id
    backend_id = context.backend_id
    seed = context.seed
    step = context.step
    total_steps = context.total_steps
    parent_checkpoint_id = context.parent_checkpoint_id
    parent_optimizer_state_digest = context.parent_optimizer_state_digest
    trainable_targets = context.trainable_targets
    trainable_parameter_specs = context.trainable_parameter_specs
    lora_rank = context.lora_rank
    lora_alpha = context.lora_alpha
    lora_dropout = context.lora_dropout
    optimizer_algorithm = context.optimizer_algorithm
    optimizer_config = context.optimizer_config
    start_step = context.start_step
    telemetry_sequence_start = context.telemetry_sequence_start
    if (
        type(trainable_targets) is not tuple
        or type(trainable_parameter_specs) is not tuple
        or type(optimizer_config) is not tuple
    ):
        raise ValueError("wan_checkpoint_state_invalid")
    owned_targets: list[str] = []
    for target in trainable_targets:
        if type(target) is not str:
            raise ValueError("wan_checkpoint_state_invalid")
        owned_targets.append(target)
    owned_config: list[tuple[str, OptimizerValue]] = []
    for entry in optimizer_config:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("wan_checkpoint_state_invalid")
        name, value = entry
        if type(name) is not str or not _is_optimizer_value(value):
            raise ValueError("wan_checkpoint_state_invalid")
        owned_config.append((name, value))
    return WanCheckpointValidationContext(
        checkpoint_id=checkpoint_id,
        request_id=request_id,
        run_id=run_id,
        backend_id=backend_id,
        seed=seed,
        step=step,
        total_steps=total_steps,
        parent_checkpoint_id=parent_checkpoint_id,
        parent_optimizer_state_digest=parent_optimizer_state_digest,
        trainable_targets=tuple(owned_targets),
        trainable_parameter_specs=validate_trainable_parameter_specs(
            trainable_parameter_specs, declared_targets=owned_targets
        ),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        optimizer_algorithm=optimizer_algorithm,
        optimizer_config=tuple(owned_config),
        start_step=start_step,
        telemetry_sequence_start=telemetry_sequence_start,
    )


def _snapshot_json_payload(value: object) -> bytes:
    owned = _owned_json_value(value, set())
    if type(owned) not in {dict, list}:
        raise ValueError
    return canonical_json(cast(dict[str, Any], owned)).encode()


def _state_from_payloads(
    payloads: Mapping[str, bytes], context: WanCheckpointValidationContext
) -> WanCheckpointState:
    try:
        decoded = {
            role: json.loads(payloads[role])
            for role in _ROLES
            if role != "weights"
        }
        telemetry = decoded["telemetry"]
        if not isinstance(telemetry, list):
            raise ValueError
        return WanCheckpointState(
            checkpoint_id=context.checkpoint_id,
            step=context.step,
            total_steps=context.total_steps,
            parent_checkpoint_id=context.parent_checkpoint_id,
            weights=payloads["weights"],
            optimizer=cast(dict[str, object], decoded["optimizer"]),
            trainer_state=cast(dict[str, object], decoded["trainer_state"]),
            telemetry=tuple(cast(list[dict[str, object]], telemetry)),
            trainable_targets=cast(dict[str, object], decoded["trainable_targets"]),
            frozen_parameters=cast(dict[str, object], decoded["frozen_parameters"]),
            finite_gradient_proof=cast(
                dict[str, object], decoded["finite_gradient_proof"]
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("wan_checkpoint_restore_invalid") from None


def _owned_json_value(value: object, ancestors: set[int]) -> object:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if type(value) not in {dict, list}:
        raise ValueError
    identity = id(value)
    if identity in ancestors:
        raise ValueError
    ancestors.add(identity)
    try:
        if type(value) is dict:
            mapping = cast(dict[object, object], value)
            owned: dict[str, object] = {}
            for key, item in mapping.items():
                if type(key) is not str:
                    raise ValueError
                owned[key] = _owned_json_value(item, ancestors)
            return owned
        return [
            _owned_json_value(item, ancestors) for item in cast(list[object], value)
        ]
    finally:
        ancestors.remove(identity)


def _validate_context(context: WanCheckpointValidationContext) -> None:
    if (
        type(context) is not WanCheckpointValidationContext
        or any(
            type(value) is not str or not value
            for value in (
                context.checkpoint_id,
                context.request_id,
                context.run_id,
                context.backend_id,
                context.optimizer_algorithm,
            )
        )
        or type(context.seed) is not int
        or type(context.step) is not int
        or context.step <= 0
        or type(context.start_step) is not int
        or not 0 <= context.start_step < context.step
        or type(context.telemetry_sequence_start) is not int
        or context.telemetry_sequence_start <= 0
        or type(context.total_steps) is not int
        or context.total_steps < context.step
        or type(context.lora_rank) is not int
        or context.lora_rank <= 0
        or type(context.lora_alpha) is not int
        or context.lora_alpha <= 0
        or type(context.lora_dropout) is not float
        or not math.isfinite(context.lora_dropout)
        or not 0 <= context.lora_dropout < 1
        or not context.trainable_targets
        or any(type(target) is not str or not target for target in context.trainable_targets)
        or len(set(context.trainable_targets)) != len(context.trainable_targets)
        or validate_trainable_parameter_specs(
            context.trainable_parameter_specs,
            declared_targets=context.trainable_targets,
        )
        != context.trainable_parameter_specs
        or not context.optimizer_config
        or tuple(name for name, _value in context.optimizer_config)
        != tuple(sorted({name for name, _value in context.optimizer_config}))
        or any(
            type(name) is not str or not name or not _is_optimizer_value(value)
            for name, value in context.optimizer_config
        )
        or (
            context.parent_optimizer_state_digest is not None
            and not _is_sha256_digest(context.parent_optimizer_state_digest)
        )
    ):
        raise ValueError("wan_checkpoint_state_invalid")
    _safe_component(context.checkpoint_id)
    if context.parent_checkpoint_id is not None:
        _safe_component(context.parent_checkpoint_id)


def _validate_role_schemas(
    state: WanCheckpointState, context: WanCheckpointValidationContext
) -> None:
    try:
        trainable = _exact_mapping(
            state.trainable_targets,
            {
                "schema_version", "request_id", "run_id", "step", "targets",
                "physical_parameters",
            },
        )
        request_id = trainable.get("request_id")
        run_id = trainable.get("run_id")
        targets = trainable.get("targets")
        if (
            trainable.get("schema_version") != "trainer-trainable-targets.v2"
            or request_id != context.request_id
            or run_id != context.run_id
            or type(trainable.get("step")) is not int
            or trainable.get("step") != state.step
            or not isinstance(targets, list)
            or not targets
            or any(type(target) is not str or not target for target in targets)
            or len(set(targets)) != len(targets)
            or targets != list(context.trainable_targets)
            or validate_trainable_parameter_specs(
                trainable_parameter_specs_from_records(trainable.get("physical_parameters")),
                declared_targets=context.trainable_targets,
            )
            != context.trainable_parameter_specs
        ):
            raise ValueError
        shared = {
            "request_id": context.request_id,
            "run_id": context.run_id,
            "step": context.step,
        }
        _validate_trainer_role(state.trainer_state, state, shared, context)
        _validate_optimizer_role(state.optimizer, targets, shared, context)
        progress = _validate_telemetry_role(
            state.telemetry,
            state.step,
            state.total_steps,
            context.start_step,
            context.telemetry_sequence_start,
        )
        _validate_frozen_role(state.frozen_parameters, targets, shared)
        _validate_finite_proof_role(state.finite_gradient_proof, progress, shared)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("wan_checkpoint_state_invalid") from None


def _exact_mapping(value: object, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError
    return value


def _validate_trainer_role(
    value: object,
    state: WanCheckpointState,
    shared: Mapping[str, object],
    context: WanCheckpointValidationContext,
) -> None:
    trainer = _exact_mapping(
        value,
        {
            "schema_version",
            "request_id",
            "run_id",
            "step",
            "backend_id",
            "seed",
            "total_steps",
            "global_step",
            "rng_states",
            "scheduler_state",
            "scaler_state",
            "backend_state",
        },
    )
    backend_id = trainer.get("backend_id")
    seed = trainer.get("seed")
    if (
        trainer.get("schema_version") != "trainer-state.v3"
        or not _matches_shared(trainer, shared)
        or backend_id != context.backend_id
        or seed != context.seed
        or type(trainer.get("total_steps")) is not int
        or trainer.get("total_steps") != state.total_steps
        or type(trainer.get("global_step")) is not int
        or trainer.get("global_step") != state.step
        or not _matches_rng_states(trainer.get("rng_states"))
        or not _is_substantive_numeric_state(trainer.get("scheduler_state"))
        or not _is_substantive_numeric_state(trainer.get("scaler_state"))
        or trainer.get("backend_state")
        != {context.backend_id: {"data_cursor": context.step}}
    ):
        raise ValueError


def _validate_optimizer_role(
    value: object,
    targets: list[object],
    shared: Mapping[str, object],
    context: WanCheckpointValidationContext,
) -> None:
    optimizer = _exact_mapping(
        value,
        {
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
        },
    )
    config = optimizer.get("config")
    groups = optimizer.get("parameter_groups")
    parameter_state = optimizer.get("parameter_state")
    parent_digest = optimizer.get("parent_optimizer_state_digest")
    configured = dict(context.optimizer_config)
    accumulation = configured.get("gradient_accumulation_steps", 1)
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError
    residual = context.step % accumulation
    terminal_partial = context.step == context.total_steps and residual > 0
    expected_updates = context.step // accumulation + int(terminal_partial)
    expected_position = 0 if residual == 0 or terminal_partial else residual
    expected_backend: dict[str, object] = {"optimizer_updates": expected_updates}
    if accumulation > 1:
        expected_backend["pending_gradients"] = _validated_pending_gradients(
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], optimizer.get("backend_state"))[
                    context.backend_id
                ],
            ).get("pending_gradients"),
            expected_position,
            expected_parameters=tuple(
                item.physical_name for item in context.trainable_parameter_specs
            ),
            expected_specs={
                item.physical_name: (item.shape, item.dtype, item.device)
                for item in context.trainable_parameter_specs
            },
        )
    if (
        optimizer.get("schema_version") != "trainer-optimizer-state.v2"
        or not _matches_shared(optimizer, shared)
        or optimizer.get("algorithm") != context.optimizer_algorithm
        or not isinstance(config, Mapping)
        or not _matches_optimizer_config(config, context.optimizer_config)
        or type(optimizer.get("global_step")) is not int
        or optimizer.get("global_step") != context.step
        or not isinstance(groups, list)
        or not groups
        or not isinstance(parameter_state, Mapping)
        or set(parameter_state) != set(targets)
        or optimizer.get("backend_state") != {context.backend_id: expected_backend}
        or parent_digest != context.parent_optimizer_state_digest
    ):
        raise ValueError
    covered: list[object] = []
    group_ids: list[str] = []
    for value_group in groups:
        group = _exact_mapping(value_group, {"group_id", "parameters", "hyperparameters"})
        group_id = group.get("group_id")
        parameters = group.get("parameters")
        if (
            type(group_id) is not str
            or not group_id
            or not isinstance(parameters, list)
            or not parameters
            or any(type(parameter) is not str or not parameter for parameter in parameters)
            or group.get("hyperparameters") != config
        ):
            raise ValueError
        group_ids.append(group_id)
        covered.extend(parameters)
    if len(set(group_ids)) != len(group_ids) or covered != targets:
        raise ValueError
    for target in targets:
        parameter = _exact_mapping(parameter_state[target], {"step", "slots"})
        if (
            type(parameter.get("step")) is not int
            or parameter.get("step") != context.step
            or not _is_substantive_numeric_state(parameter.get("slots"))
        ):
            raise ValueError


def _validated_pending_gradients(
    value: object,
    position: int,
    *,
    expected_parameters: Collection[str],
    expected_specs: Mapping[str, tuple[tuple[int, ...], str, str]] | None = None,
) -> Mapping[str, object]:
    return validate_pending_gradients(
        value,
        position,
        expected_parameters=expected_parameters,
        expected_specs=expected_specs,
    )


def _validate_telemetry_role(
    value: tuple[Mapping[str, object], ...],
    step: int,
    total_steps: int,
    start_step: int,
    sequence_start: int,
) -> TrainingProgress:
    if len(value) != step - start_step:
        raise ValueError
    progress: TrainingProgress | None = None
    for offset, raw_record in enumerate(value):
        expected_step = start_step + offset + 1
        expected_sequence = sequence_start + offset
        record = _exact_mapping(raw_record, {"sequence", "step", "payload"})
        if (
            record.get("sequence") != expected_sequence
            or record.get("step") != expected_step
        ):
            raise ValueError
        progress = TrainingProgress.model_validate(record.get("payload"), strict=True)
        if progress.step != expected_step or progress.total_steps != total_steps:
            raise ValueError
    if progress is None or progress.step != step or not progress.checkpoint_boundary:
        raise ValueError
    return progress


def _validate_frozen_role(
    value: object, targets: list[object], shared: Mapping[str, object]
) -> None:
    frozen = _exact_mapping(
        value,
        {
            "schema_version",
            "request_id",
            "run_id",
            "step",
            "trainable_targets",
            "unchanged",
        },
    )
    if (
        frozen.get("schema_version") != "trainer-frozen-parameters.v1"
        or not _matches_shared(frozen, shared)
        or frozen.get("trainable_targets") != targets
        or frozen.get("unchanged") is not True
    ):
        raise ValueError


def _validate_finite_proof_role(
    value: object, progress: TrainingProgress, shared: Mapping[str, object]
) -> None:
    proof = _exact_mapping(
        value,
        {
            "schema_version",
            "request_id",
            "run_id",
            "step",
            "finite",
            "loss",
            "gradient_norm",
        },
    )
    if (
        proof.get("schema_version") != "trainer-finite-gradient-proof.v1"
        or not _matches_shared(proof, shared)
        or proof.get("finite") is not True
        or not _is_finite_number(proof.get("loss"))
        or not _is_finite_number(proof.get("gradient_norm"))
        or float(cast(int | float, proof["gradient_norm"])) < 0
        or proof.get("loss") != progress.loss
        or proof.get("gradient_norm") != progress.gradient_norm
    ):
        raise ValueError


def _is_finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(cast(int | float, value)))


def _is_optimizer_value(value: object) -> bool:
    return type(value) in {str, int, bool} or (
        type(value) is float and math.isfinite(value)
    )


def _matches_optimizer_config(
    value: Mapping[str, object], expected: tuple[tuple[str, OptimizerValue], ...]
) -> bool:
    if tuple(value) != tuple(name for name, _expected in expected):
        return False
    return all(
        type(value[name]) is type(expected_value) and value[name] == expected_value
        for name, expected_value in expected
    )


def _matches_shared(
    value: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    return (
        type(value.get("request_id")) is str
        and value.get("request_id") == expected["request_id"]
        and type(value.get("run_id")) is str
        and value.get("run_id") == expected["run_id"]
        and type(value.get("step")) is int
        and value.get("step") == expected["step"]
    )


def _matches_rng_states(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"cpu", "mps"}:
        return False
    for device in ("cpu", "mps"):
        item = value.get(device)
        if not isinstance(item, Mapping) or set(item) != {
            "algorithm",
            "device",
            "state_hex",
            "digest",
        }:
            return False
        algorithm = item.get("algorithm")
        state_hex = item.get("state_hex")
        digest = item.get("digest")
        if (
            type(algorithm) is not str
            or not algorithm
            or item.get("device") != device
            or type(state_hex) is not str
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


def _substantive_numeric_state(value: object) -> tuple[bool, bool]:
    if isinstance(value, bool):
        return True, False
    if type(value) in {int, float}:
        finite = math.isfinite(float(cast(int | float, value)))
        return finite, finite
    if isinstance(value, str):
        return bool(value), False
    if isinstance(value, list):
        children = tuple(_substantive_numeric_state(item) for item in value)
        return bool(children) and all(valid for valid, _ in children), any(
            substantive for _, substantive in children
        )
    if isinstance(value, Mapping):
        children = tuple(_substantive_numeric_state(item) for item in value.values())
        return (
            bool(children)
            and all(type(key) is str and bool(key) for key in value)
            and all(valid for valid, _ in children),
            any(substantive for _, substantive in children),
        )
    return False, False


def _is_substantive_numeric_state(value: object) -> bool:
    valid, substantive = _substantive_numeric_state(value)
    return valid and substantive


def _is_bound_backend_state(value: object, backend_id: str) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {backend_id}
        and _is_substantive_numeric_state(value[backend_id])
    )


def _is_sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _finite_json(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    if isinstance(value, list | tuple):
        return all(_finite_json(item) for item in value)
    if isinstance(value, Mapping):
        return all(type(key) is str and _finite_json(item) for key, item in value.items())
    return False


def _validate_safetensors(data: bytes) -> dict[str, tuple[str, tuple[int, ...]]]:
    try:
        if len(data) < 9:
            raise ValueError
        header_size = int.from_bytes(data[:8], "little")
        if header_size <= 0 or header_size > len(data) - 8:
            raise ValueError
        raw_header = data[8 : 8 + header_size]
        header = json.loads(raw_header)
        if not isinstance(header, dict) or not header:
            raise ValueError
        canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if canonical + b" " * (header_size - len(canonical)) != raw_header:
            raise ValueError
        payload_size = len(data) - 8 - header_size
        widths = {"F16": 2, "BF16": 2, "F32": 4}
        ranges: list[tuple[int, int]] = []
        tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
        for name, value in header.items():
            if (
                type(name) is not str
                or not name
                or not isinstance(value, dict)
                or set(value) != {"dtype", "shape", "data_offsets"}
                or value["dtype"] not in widths
            ):
                raise ValueError
            shape = value["shape"]
            offsets = value["data_offsets"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(size) is not int or size <= 0 for size in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] > payload_size
                or offsets[1] - offsets[0] != math.prod(shape) * widths[value["dtype"]]
            ):
                raise ValueError
            ranges.append((offsets[0], offsets[1]))
            tensors[name] = (value["dtype"], tuple(shape))
        if (
            sorted(ranges) != ranges
            or ranges[0][0] != 0
            or ranges[-1][1] != payload_size
            or any(left[1] != right[0] for left, right in pairwise(ranges))
        ):
            raise ValueError
        return tensors
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("wan_checkpoint_state_invalid") from None


def _validate_lora_tensors(
    tensors: Mapping[str, tuple[str, tuple[int, ...]]], targets: list[str], rank: int
) -> None:
    expected_names = {
        f"{target}.lora_{suffix}.weight" for target in targets for suffix in ("A", "B")
    }
    if set(tensors) != expected_names:
        raise ValueError("wan_checkpoint_state_invalid")
    for target in targets:
        dtype_a, shape_a = tensors[f"{target}.lora_A.weight"]
        dtype_b, shape_b = tensors[f"{target}.lora_B.weight"]
        if dtype_a != dtype_b or shape_a[0] != rank or shape_b[1] != rank:
            raise ValueError("wan_checkpoint_state_invalid")


def _safe_component(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError("wan_checkpoint_state_invalid")
    return value


def _open_scratch_root(root: Path) -> tuple[int, tuple[int, int]]:
    if not root.is_absolute() or any(part in {"", ".", ".."} for part in root.parts[1:]):
        raise ValueError("unsafe_scratch_root")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    current = os.open(root.parts[0], flags)
    try:
        for index, part in enumerate(root.parts[1:]):
            final = index == len(root.parts[1:]) - 1
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not final:
                    raise ValueError("unsafe_scratch_root") from None
                os.mkdir(part, 0o700, dir_fd=current)
                child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        info = os.fstat(current)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("unsafe_scratch_root")
        return current, (info.st_dev, info.st_ino)
    except Exception:
        os.close(current)
        raise ValueError("unsafe_scratch_root") from None


def _open_exact_parent_file(
    path: Path, expected: WanParentCheckpointFileDescriptor
) -> _RestoredFileLease:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError("wan_checkpoint_restore_invalid")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    current = os.open(path.parts[0], directory_flags)
    root_info = os.fstat(current)
    ancestors = [
        _RestoredAncestorLease(
            name=path.parts[0],
            descriptor=current,
            device=root_info.st_dev,
            inode=root_info.st_ino,
        )
    ]
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            current = child
            info = os.fstat(current)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("wan_checkpoint_restore_invalid")
            ancestors.append(
                _RestoredAncestorLease(
                    name=part,
                    descriptor=current,
                    device=info.st_dev,
                    inode=info.st_ino,
                )
            )
        descriptor = os.open(path.name, file_flags, dir_fd=current)
    except BaseException:
        for ancestor in ancestors:
            os.close(ancestor.descriptor)
        raise
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or (info.st_dev, info.st_ino) != (expected.device, expected.inode)
        or info.st_size != expected.size_bytes
    ):
        os.close(descriptor)
        for ancestor in ancestors:
            os.close(ancestor.descriptor)
        raise ValueError("wan_checkpoint_restore_invalid")
    return _RestoredFileLease(expected, descriptor, path, tuple(ancestors))


def _reverify_ancestor_chain(lease: _RestoredFileLease) -> None:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    current = os.dup(lease.ancestors[0].descriptor)
    try:
        root = os.fstat(current)
        expected_root = lease.ancestors[0]
        if (
            not stat.S_ISDIR(root.st_mode)
            or (root.st_dev, root.st_ino) != (expected_root.device, expected_root.inode)
        ):
            raise OSError("checkpoint root replaced")
        for expected in lease.ancestors[1:]:
            child = os.open(expected.name, directory_flags, dir_fd=current)
            os.close(current)
            current = child
            info = os.fstat(current)
            retained = os.fstat(expected.descriptor)
            identity = (expected.device, expected.inode)
            if (
                not stat.S_ISDIR(info.st_mode)
                or not stat.S_ISDIR(retained.st_mode)
                or (info.st_dev, info.st_ino) != identity
                or (retained.st_dev, retained.st_ino) != identity
            ):
                raise OSError("checkpoint ancestor replaced")
        leaf = os.open(lease.path.name, file_flags, dir_fd=current)
        try:
            info = os.fstat(leaf)
            if (
                not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino)
                != (lease.descriptor.device, lease.descriptor.inode)
                or info.st_size != lease.descriptor.size_bytes
            ):
                raise OSError("checkpoint file replaced")
        finally:
            os.close(leaf)
    finally:
        os.close(current)


def _read_open_regular(descriptor: int, expected_size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    data = bytearray()
    while len(data) <= expected_size:
        chunk = os.read(descriptor, min(1024 * 1024, expected_size + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    info = os.fstat(descriptor)
    if (
        len(data) != expected_size
        or info.st_size != expected_size
        or not stat.S_ISREG(info.st_mode)
    ):
        raise ValueError("wan_checkpoint_restore_invalid")
    return bytes(data)


def _open_directory(name: str, parent_fd: int) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent_fd)


def _require_destination_absent(root_fd: int, destination: str) -> None:
    try:
        os.stat(destination, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, "checkpoint destination exists", destination)


def _publish_directory_no_replace(root_fd: int, source: str, destination: str) -> None:
    """Atomically rename one directory without ever replacing a destination."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function_name = "renameatx_np"
        no_replace_flag = 0x00000004  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function_name = "renameat2"
        no_replace_flag = 0x00000001  # RENAME_NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename unsupported")
    try:
        rename = getattr(library, function_name)
    except AttributeError:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename unsupported") from None
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(
        root_fd,
        os.fsencode(source),
        root_fd,
        os.fsencode(destination),
        no_replace_flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(errno.EEXIST, "checkpoint destination exists", destination)
    unsupported_errors = {
        errno.ENOSYS,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }
    if error_number in unsupported_errors:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename unsupported")
    raise OSError(error_number, os.strerror(error_number))


def _write_exclusive(directory_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short checkpoint write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular(directory_fd: int, name: str, expected_size: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise OSError("checkpoint file replaced")
        data = bytearray()
        while len(data) <= expected_size:
            chunk = os.read(descriptor, min(1024 * 1024, expected_size + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) != expected_size or os.fstat(descriptor).st_size != expected_size:
            raise OSError("checkpoint file changed")
        return bytes(data)
    finally:
        os.close(descriptor)


def _reverify_root(root: Path, descriptor: int, identity: tuple[int, int]) -> None:
    try:
        held = os.fstat(descriptor)
        current = root.lstat()
    except OSError:
        raise ValueError("unsafe_scratch_root") from None
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (held.st_dev, held.st_ino) != identity
        or (current.st_dev, current.st_ino) != identity
    ):
        raise ValueError("unsafe_scratch_root")


def _remove_checkpoint(
    root_fd: int, directory_name: str, names: tuple[str, ...], open_fd: int
) -> None:
    cleanup_errors: list[BaseException] = []
    descriptor = open_fd
    owned_descriptor = False
    if descriptor < 0:
        try:
            descriptor = _open_directory(directory_name, root_fd)
            owned_descriptor = True
        except FileNotFoundError:
            descriptor = -1
        except OSError as error:
            cleanup_errors.append(error)
            descriptor = -1
    if descriptor >= 0:
        try:
            for name in names:
                try:
                    _unlink_checkpoint_entry(descriptor, name)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    cleanup_errors.append(error)
        finally:
            if owned_descriptor:
                try:
                    os.close(descriptor)
                except OSError as error:
                    cleanup_errors.append(error)
    try:
        _remove_checkpoint_directory(root_fd, directory_name)
    except FileNotFoundError:
        pass
    except OSError as error:
        cleanup_errors.append(error)
    try:
        _fsync_directory(root_fd)
    except OSError as error:
        cleanup_errors.append(error)
    if cleanup_errors:
        raise WanCheckpointCleanupError(None, tuple(cleanup_errors))


def _unlink_checkpoint_entry(directory_fd: int, name: str) -> None:
    os.unlink(name, dir_fd=directory_fd)


def _remove_checkpoint_directory(root_fd: int, directory_name: str) -> None:
    os.rmdir(directory_name, dir_fd=root_fd)


def _fsync_directory(descriptor: int) -> None:
    os.fsync(descriptor)


def _close_directory(descriptor: int) -> None:
    os.close(descriptor)
