"""Sequential, strictly local Wan production canary orchestration."""

from __future__ import annotations

import base64
import fcntl
import gc
import hashlib
import importlib.metadata
import os
import platform
import shutil
import stat
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import ValidationError

from video_lab.artifacts import ArtifactStore
from video_lab.database import Database
from video_lab.dataset_receipts import DatasetCompilationRepository
from video_lab.domain import canonical_json
from video_lab.local_authority import (
    RetainedDirectoryAuthority,
    RetainedSqliteAuthority,
)
from video_lab.trainer_protocol import LoRAConfiguration
from video_lab.trainers.wan_branch_loader import (
    WanBranchLoadRequest,
    WanMpsPreparedFactory,
    inventory_wan_branch,
)
from video_lab.trainers.wan_branch_runtime import (
    WanMpsBranchOperations,
    is_recognized_mps_oom,
)
from video_lab.trainers.wan_factory import _is_mps_device
from video_lab.trainers.wan_mps import WanTrainingSourceLease, WanTrainingSources
from video_lab.trainers.wan_production_models import (
    WanBranchCanaryResultV1,
    WanBranchNumericalProbeV2,
    WanBranchRecipeV1,
    WanCanaryPhase,
    WanCapacityResultV2,
    WanComponentInventoryV1,
    WanProductionCanaryReceiptV2,
    WanProductionCanaryRequestV2,
    WanProductionCanaryUnavailableV2,
    WanTransformerBranch,
    wan_branch_recipe_digest,
    wan_inventory_digest,
    wan_inventory_family_digest,
    wan_recipe_family_digest,
    wan_required_tensor_bytes,
)


class WanCanaryCancelled(RuntimeError):
    """Cancellation observed at a declared lifecycle boundary."""


class WanCapacityExceeded(RuntimeError):
    def __init__(self, result: WanCapacityResultV2) -> None:
        self.result = result
        super().__init__("wan_mps_capacity_exceeded")


@dataclass(frozen=True, slots=True)
class WanProductionCanaryExecutionV2:
    """Concrete local authorities required to execute, restore, and publish."""

    request: WanProductionCanaryRequestV2
    branch_load_request: WanBranchLoadRequest
    training_sources: WanTrainingSources
    training_source_lease: WanTrainingSourceLease
    compilation_receipt_id: str
    checkpoint_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        if (
            type(self.request) is not WanProductionCanaryRequestV2
            or type(self.branch_load_request) is not WanBranchLoadRequest
            or type(self.training_sources) is not WanTrainingSources
            or type(self.training_source_lease) is not WanTrainingSourceLease
            or self.training_source_lease.sources != self.training_sources
            or type(self.compilation_receipt_id) is not str
            or not self.compilation_receipt_id
        ):
            raise ValueError("wan_canary_execution_authority_invalid")
        for root in (self.checkpoint_root, self.output_root):
            if not root.is_absolute() or not root.is_dir() or root.is_symlink():
                raise ValueError("wan_canary_execution_authority_invalid")
        if self.checkpoint_root == self.output_root:
            raise ValueError("wan_canary_execution_authority_invalid")


def require_strict_wan_production_execution(
    execution: object,
) -> WanProductionCanaryExecutionV2:
    if type(execution) is not WanProductionCanaryExecutionV2:
        raise ValueError("wan_canary_execution_authority_invalid")
    try:
        request = WanProductionCanaryRequestV2.model_validate(
            execution.request.model_dump(mode="python"), strict=True
        )
    except (AttributeError, ValidationError, TypeError, ValueError):
        raise ValueError("wan_canary_execution_authority_invalid") from None
    if request != execution.request:
        raise ValueError("wan_canary_execution_authority_invalid")
    return replace(execution, request=request)


class _Preflight(Protocol):
    def __call__(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _WanCanaryHooks:
    preflight: _Preflight
    inventory: Callable[
        [WanTransformerBranch, WanComponentInventoryV1],
        WanComponentInventoryV1 | WanProductionCanaryUnavailableV2,
    ]
    device_preflight: Callable[[], None]
    load: Callable[[WanTransformerBranch, Literal["trained", "restored"]], object]
    step: Callable[[object, WanBranchRecipeV1], object]
    checkpoint: Callable[[object], object]
    restore: Callable[[object, WanBranchRecipeV1], None]
    continuity: Callable[
        [object, WanBranchRecipeV1],
        WanBranchCanaryResultV1 | _WanContinuityOutcome,
    ]
    close: Callable[[object], object]
    memory_released: Callable[[WanTransformerBranch, Literal["trained", "restored"], int], int]
    publish: Callable[[WanProductionCanaryReceiptV2], WanProductionCanaryReceiptV2]
    capacity_observation: Callable[[], tuple[int, int]]
    capacity_cleanup: Callable[[], None]


class _CloseOnce:
    def __init__(self, close: Callable[[object], object], handle: object) -> None:
        self._close, self._handle = close, handle

    def __call__(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            self._close(handle)


@dataclass(frozen=True, slots=True)
class _WanContinuityOutcome:
    result: WanBranchCanaryResultV1
    probe: WanBranchNumericalProbeV2


@dataclass(frozen=True, slots=True)
class _RetainedReadonlyDatabase:
    authority: RetainedSqliteAuthority

    @property
    def path(self) -> Path:
        return self.authority.path

    def connect(self) -> Any:
        return self.authority.connect_readonly()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_execution_roots(
    execution: WanProductionCanaryExecutionV2,
    output: RetainedDirectoryAuthority,
    checkpoint: RetainedDirectoryAuthority,
) -> None:
    model_root = execution.branch_load_request.model_root.path
    source_files = tuple(item.path for item in execution.training_source_lease.sources.files)
    protected = (model_root, *(item.parent for item in source_files), *source_files)
    if _paths_overlap(output.path, checkpoint.path) or any(
        _paths_overlap(root.path, source)
        for root in (output, checkpoint)
        for source in protected
    ):
        raise ValueError("wan_canary_root_overlap")
    try:
        source_roots = {item.parent for item in source_files}
        if any(
            os.stat(path, follow_symlinks=False).st_mode & 0o222
            for path in (*source_files, *source_roots, model_root)
        ):
            raise ValueError
    except (OSError, ValueError):
        raise ValueError("wan_canary_source_writable") from None
    execution.training_source_lease.reverify()
    output.reverify()
    checkpoint.reverify()


def _guard_hooks(
    hooks: _WanCanaryHooks,
    execution: WanProductionCanaryExecutionV2,
    output: RetainedDirectoryAuthority,
    checkpoint: RetainedDirectoryAuthority,
) -> _WanCanaryHooks:
    def guard(operation: Callable[..., Any]) -> Callable[..., Any]:
        def checked(*args: Any, **kwargs: Any) -> Any:
            _validate_execution_roots(execution, output, checkpoint)
            result = operation(*args, **kwargs)
            _validate_execution_roots(execution, output, checkpoint)
            return result

        return checked

    return _WanCanaryHooks(
        preflight=guard(hooks.preflight),
        inventory=guard(hooks.inventory),
        device_preflight=guard(hooks.device_preflight),
        load=guard(hooks.load),
        step=guard(hooks.step),
        checkpoint=guard(hooks.checkpoint),
        restore=guard(hooks.restore),
        continuity=guard(hooks.continuity),
        close=guard(hooks.close),
        memory_released=guard(hooks.memory_released),
        publish=guard(hooks.publish),
        capacity_observation=guard(hooks.capacity_observation),
        capacity_cleanup=guard(hooks.capacity_cleanup),
    )


def _selected(request: WanBranchLoadRequest, branch: WanTransformerBranch) -> WanBranchLoadRequest:
    selected = replace(request, branch=branch)
    object.__setattr__(selected, "_authority_signature", request._authority_signature)
    object.__setattr__(selected, "_snapshot_parent_identity", request._snapshot_parent_identity)
    return selected


def _numerical_probe(
    branch: WanTransformerBranch,
    recipe: WanBranchRecipeV1,
    mapping: object,
) -> WanBranchNumericalProbeV2:
    if type(mapping) is not dict or set(mapping) != {"probe.output"}:
        raise ValueError("wan_canary_probe_invalid")
    tensor = cast(dict[str, Any], mapping)["probe.output"]
    if not all(hasattr(tensor, attribute) for attribute in ("detach", "to", "reshape", "tolist")):
        raise ValueError("wan_canary_probe_invalid")
    observed = tensor.detach()
    dtype = str(observed.dtype).removeprefix("torch.")
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("wan_canary_probe_invalid")
    shape = tuple(int(size) for size in observed.shape)
    values = observed.to("cpu").reshape(-1).tolist()
    return WanBranchNumericalProbeV2(
        branch=branch,
        recipe_digest=wan_branch_recipe_digest(recipe),
        tensor_name="probe.output",
        shape=shape,
        dtype=cast(Literal["bfloat16", "float32"], dtype),
        values=tuple(float(value) for value in values),
    )


def _compare_numerical_probes(
    reference: WanBranchNumericalProbeV2,
    candidate: WanBranchNumericalProbeV2,
    recipe: WanBranchRecipeV1,
) -> tuple[float, float]:
    if (
        reference.branch != candidate.branch
        or reference.recipe_digest != candidate.recipe_digest
        or reference.tensor_name != candidate.tensor_name
        or reference.shape != candidate.shape
        or reference.dtype != candidate.dtype
        or reference.recipe_digest != wan_branch_recipe_digest(recipe)
    ):
        raise ValueError("wan_canary_continuity_mismatch")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for expected, observed in zip(reference.values, candidate.values, strict=True):
        absolute = abs(observed - expected)
        relative = absolute / max(abs(expected), recipe.tolerance.atol)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if absolute > recipe.tolerance.atol + recipe.tolerance.rtol * abs(expected):
            raise ValueError("wan_canary_continuity_mismatch")
    return maximum_absolute, maximum_relative


def _acquire_checkpoint_digest_lock(
    root_descriptor: int, final_name: str
) -> tuple[int, int]:
    """Retain a stable per-digest flock in the app-owned checkpoint root.

    Lock files intentionally outlive callers. Unlinking one could split current
    waiters on the old inode from a later publisher locking a newly created inode.
    """

    lock_directory = -1
    lock_descriptor = -1
    try:
        try:
            os.mkdir(".checkpoint-locks", mode=0o700, dir_fd=root_descriptor)
        except FileExistsError:
            pass
        lock_directory = os.open(
            ".checkpoint-locks",
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_descriptor,
        )
        if not stat.S_ISDIR(os.fstat(lock_directory).st_mode):
            raise OSError("checkpoint lock namespace is not a directory")
        lock_descriptor = os.open(
            f"{final_name}.lock",
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=lock_directory,
        )
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise OSError("checkpoint digest lock is not a regular file")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        return lock_directory, lock_descriptor
    except OSError:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        if lock_directory >= 0:
            os.close(lock_directory)
        raise ValueError("wan_canary_checkpoint_lock_failed") from None


def _publish_checkpoint(
    root: Path,
    branch: WanTransformerBranch,
    checkpoint: Any,
    *,
    root_authority: RetainedDirectoryAuthority,
) -> None:
    payload = canonical_json(
        {
            "manifest": asdict(checkpoint.manifest),
            "weights": base64.b64encode(checkpoint.weights).decode("ascii"),
            "optimizer": dict(checkpoint.optimizer),
            "cpu_rng_state": base64.b64encode(checkpoint.cpu_rng_state).decode("ascii"),
            "mps_rng_state": base64.b64encode(checkpoint.mps_rng_state).decode("ascii"),
        }
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    final_name = f"{branch}-{digest}.checkpoint.json"
    temporary_name = f".{final_name}.{uuid.uuid4().hex}.tmp"
    if root != root_authority.path:
        raise ValueError(root_authority.error_code)
    try:
        root_descriptor = os.dup(root_authority.descriptor)
    except OSError:
        raise ValueError(root_authority.error_code) from None
    lock_directory = -1
    lock_descriptor = -1
    descriptor = -1
    final_created = False
    temporary_created = False
    staged_identity: tuple[int, int] | None = None
    try:
        lock_directory, lock_descriptor = _acquire_checkpoint_digest_lock(
            root_descriptor, final_name
        )
        root_authority.reverify()
        try:
            descriptor = os.open(
                final_name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=root_descriptor,
            )
        except FileNotFoundError:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=root_descriptor,
            )
            temporary_created = True
            staged = os.fstat(descriptor)
            if not stat.S_ISREG(staged.st_mode):
                raise ValueError("wan_canary_checkpoint_integrity_failed") from None
            staged_identity = (staged.st_dev, staged.st_ino)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("wan_canary_checkpoint_write_failed") from None
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            try:
                os.link(
                    temporary_name,
                    final_name,
                    src_dir_fd=root_descriptor,
                    dst_dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                final_created = True
            except FileExistsError:
                descriptor = os.open(
                    final_name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=root_descriptor,
                )
                observed = os.pread(descriptor, len(payload) + 1, 0)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode) or observed != payload:
                    raise ValueError("wan_canary_checkpoint_integrity_failed") from None
            os.unlink(temporary_name, dir_fd=root_descriptor)
            temporary_created = False
            os.fsync(root_descriptor)
        else:
            observed = os.pread(descriptor, len(payload) + 1, 0)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode) or observed != payload:
                raise ValueError("wan_canary_checkpoint_integrity_failed")
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        root_authority.reverify()
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            descriptor = -1
        removed = False
        if final_created:
            try:
                observed_final = os.stat(
                    final_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if (observed_final.st_dev, observed_final.st_ino) != staged_identity:
                    raise OSError("checkpoint publication ownership changed")
                os.unlink(final_name, dir_fd=root_descriptor)
                removed = True
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=root_descriptor)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            else:
                removed = True
        if removed:
            try:
                os.fsync(root_descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            primary_message = str(primary)
            normalized_primary = (
                ValueError(primary_message)
                if isinstance(primary, ValueError)
                and primary_message.startswith("wan_canary_")
                and primary_message.replace("_", "").isalnum()
                else ValueError("wan_canary_checkpoint_publication_failed")
            )
            normalized_rollbacks = tuple(
                ValueError("wan_canary_checkpoint_rollback_operation_failed")
                for _error in cleanup_errors
            )
            raise ValueError("wan_canary_checkpoint_rollback_failed") from BaseExceptionGroup(
                "wan_canary_checkpoint_rollback_errors",
                (normalized_primary, *normalized_rollbacks),
            )
        raise
    finally:
        if lock_descriptor >= 0:
            try:
                os.close(lock_descriptor)
            except OSError:
                pass
        if lock_directory >= 0:
            try:
                os.close(lock_directory)
            except OSError:
                pass
        try:
            os.close(root_descriptor)
        except OSError:
            pass


def _production_hooks(
    execution: WanProductionCanaryExecutionV2,
    database: Database,
    artifacts: ArtifactStore,
    *,
    importer: Callable[[str], object] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    output_authority: RetainedDirectoryAuthority | None = None,
    checkpoint_authority: RetainedDirectoryAuthority | None = None,
    database_authority: RetainedSqliteAuthority | None = None,
) -> _WanCanaryHooks:
    state: dict[str, object] = {}
    library_importer = importer or importlib.import_module

    def preflight() -> None:
        if artifacts.root != execution.output_root.resolve():
            raise ValueError("wan_canary_output_authority_invalid")
        if platform.system() != "Darwin":
            raise ValueError("wan_canary_host_unsupported")
        if dict(execution.request.hardware_identity) != {
            "device": "mps", "machine": platform.machine(), "system": platform.system()
        }:
            raise ValueError("wan_canary_hardware_identity_stale")
        try:
            observed_versions = tuple(
                (name, importlib.metadata.version(name))
                for name, _ in execution.request.dependency_versions
            )
        except importlib.metadata.PackageNotFoundError:
            raise ValueError("wan_canary_dependency_unavailable") from None
        if observed_versions != execution.request.dependency_versions:
            raise ValueError("wan_canary_dependency_version_stale")
        if any(
            shutil.disk_usage(root).free < execution.request.required_artifact_bytes
            for root in (execution.checkpoint_root, execution.output_root)
        ):
            raise ValueError("wan_canary_artifact_capacity_insufficient")
        receipt_database = (
            database
            if database_authority is None
            else cast(Database, _RetainedReadonlyDatabase(database_authority))
        )
        receipt = DatasetCompilationRepository(receipt_database).receipt(
            execution.compilation_receipt_id
        )
        digest = "sha256:" + hashlib.sha256(
            canonical_json(receipt.model_dump(mode="json")).encode()
        ).hexdigest()
        if digest != execution.request.compilation_receipt_digest:
            raise ValueError("wan_canary_compilation_receipt_stale")
        batch_source = next(
            (
                item
                for item in execution.training_source_lease.sources.files
                if item.name == "batch_tensors"
            ),
            None,
        )
        if batch_source is None or batch_source.digest != execution.request.batch_digest:
            raise ValueError("wan_canary_batch_authority_stale")

    def device_preflight() -> None:
        torch = cast(Any, library_importer("torch"))
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise ValueError("wan_canary_mps_unavailable")
        probe: Any = None
        try:
            probe = torch.empty(1, device="mps", dtype=torch.bfloat16)
            if not _is_mps_device(probe.device) or str(probe.dtype) not in {
                "bfloat16", "torch.bfloat16"
            }:
                raise ValueError("wan_canary_bf16_unsupported")
        except ValueError:
            raise
        except BaseException:
            raise ValueError("wan_canary_bf16_unsupported") from None
        finally:
            probe = None
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except BaseException:
                raise ValueError("wan_canary_bf16_probe_cleanup_failed") from None
        state["torch"] = torch

    def inventory(
        branch: WanTransformerBranch, expected: WanComponentInventoryV1
    ) -> WanComponentInventoryV1 | WanProductionCanaryUnavailableV2:
        result = inventory_wan_branch(_selected(execution.branch_load_request, branch))
        if not result.available or result.inventory is None:
            return WanProductionCanaryUnavailableV2(
                schema_version="wan-mps-production-canary-unavailable.v2",
                branch=branch,
                missing_roles=result.missing_roles,
            )
        if result.inventory != expected:
            raise ValueError("wan_canary_inventory_stale")
        return result.inventory

    def load(branch: WanTransformerBranch, role: Literal["trained", "restored"]) -> object:
        recipe = execution.request.recipes[0 if branch == "high_noise" else 1]
        factory = WanMpsPreparedFactory(
            _selected(execution.branch_load_request, branch), importer=library_importer
        )
        prepared = factory.prepare_branch(
            execution.training_source_lease,
            object(),
            recipe,
            cancelled=cancelled,
        )
        operations = cast(WanMpsBranchOperations, prepared.handle)
        try:
            operations.inject_lora(
                targets=operations.eligible_lora_modules(),
                configuration=LoRAConfiguration(rank=2, alpha=recipe.alpha, dropout=0.0),
                seed=recipe.seed,
                device="mps",
                dtype="bfloat16",
            )
            return operations
        except BaseException as primary:
            try:
                operations.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "wan_branch_acquisition_cleanup_failed", (primary, cleanup)
                ) from None
            raise

    def step(handle: object, recipe: WanBranchRecipeV1) -> object:
        operations = cast(WanMpsBranchOperations, handle)
        result = operations.train_step(step=1, total_steps=1)
        state[f"{recipe.branch}.reference"] = _numerical_probe(
            recipe.branch, recipe, operations.continuity_probe()
        )
        return result

    def checkpoint(handle: object) -> object:
        operations = cast(WanMpsBranchOperations, handle)
        checkpoint = operations.checkpoint_state_v2(parent_checkpoint_id=None)
        authority = checkpoint_authority
        owns_authority = authority is None
        if authority is None:
            authority = RetainedDirectoryAuthority.capture(
                execution.checkpoint_root,
                error_code="wan_canary_checkpoint_root_changed",
            )
        try:
            _publish_checkpoint(
                execution.checkpoint_root,
                operations.recipe.branch,
                checkpoint,
                root_authority=authority,
            )
        finally:
            if owns_authority:
                authority.close()
        state[f"{operations.recipe.branch}.checkpoint"] = checkpoint
        return checkpoint

    def restore(handle: object, recipe: WanBranchRecipeV1) -> None:
        operations = cast(WanMpsBranchOperations, handle)
        checkpoint = cast(Any, state[f"{recipe.branch}.checkpoint"])
        operations.restore_checkpoint_v2(checkpoint, checkpoint.manifest)

    def continuity(
        handle: object, recipe: WanBranchRecipeV1
    ) -> _WanContinuityOutcome:
        operations = cast(WanMpsBranchOperations, handle)
        checkpoint = cast(Any, state[f"{recipe.branch}.checkpoint"])
        probe = _numerical_probe(
            recipe.branch, recipe, operations.continuity_probe()
        )
        maximum_absolute, maximum_relative = _compare_numerical_probes(
            cast(WanBranchNumericalProbeV2, state[f"{recipe.branch}.reference"]),
            probe,
            recipe,
        )
        return _WanContinuityOutcome(
            result=WanBranchCanaryResultV1(
                schema_version="wan-mps-branch-canary-result.v1",
                branch=recipe.branch,
                recipe_digest=wan_branch_recipe_digest(recipe),
                output_digest="sha256:" + hashlib.sha256(checkpoint.weights).hexdigest(),
                max_absolute_error=maximum_absolute,
                max_relative_error=maximum_relative,
            ),
            probe=probe,
        )

    def close(handle: object) -> object:
        cast(WanMpsBranchOperations, handle).close()
        return None

    def memory_released(
        branch: WanTransformerBranch, role: Literal["trained", "restored"], bound: int
    ) -> int:
        del branch, role
        torch = cast(Any, state["torch"])
        torch.mps.synchronize()
        gc.collect()
        torch.mps.empty_cache()
        residual = int(torch.mps.current_allocated_memory())
        if residual > bound:
            raise ValueError("wan_canary_memory_release_failed")
        return residual

    def publish(receipt: WanProductionCanaryReceiptV2) -> WanProductionCanaryReceiptV2:
        payload = canonical_json(receipt.model_dump(mode="json")).encode()
        if output_authority is not None and type(artifacts) is ArtifactStore:
            artifacts.commit_bytes_at_retained_root(
                payload,
                "application/vnd.video-lab.wan-canary+json",
                output_authority.descriptor,
                reverify=output_authority.reverify,
            )
            return receipt
        staged = artifacts.stage_bytes(payload, "application/vnd.video-lab.wan-canary+json")
        try:
            artifacts.commit(staged, None)
        except BaseException:
            artifacts.discard(staged)
            raise
        return receipt

    def capacity() -> tuple[int, int]:
        torch = cast(Any, state["torch"])
        return (
            int(torch.mps.driver_allocated_memory()),
            int(torch.mps.recommended_max_memory()),
        )

    def capacity_cleanup() -> None:
        torch = cast(Any, state["torch"])
        torch.mps.synchronize()
        gc.collect()
        torch.mps.empty_cache()

    return _WanCanaryHooks(
        preflight=preflight,
        inventory=inventory,
        device_preflight=device_preflight,
        load=load,
        step=step,
        checkpoint=checkpoint,
        restore=restore,
        continuity=continuity,
        close=close,
        memory_released=memory_released,
        publish=publish,
        capacity_observation=capacity,
        capacity_cleanup=capacity_cleanup,
    )


def _cancel(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise WanCanaryCancelled("wan_canary_cancelled")


def _capacity(
    request: WanProductionCanaryRequestV2,
    branch: WanTransformerBranch,
    phase: WanCanaryPhase,
    hooks: _WanCanaryHooks,
    before: int,
    after_failure: int,
    after_cleanup: int,
    recommended: int,
) -> WanCapacityResultV2:
    del hooks
    index = 0 if branch == "high_noise" else 1
    return WanCapacityResultV2(
        schema_version="wan-mps-capacity-result.v2", branch=branch, phase=phase,
        device="mps", dtype="bfloat16",
        requested_tensor_envelope=request.requested_tensor_envelope,
        driver_allocated_before_bytes=before,
        driver_allocated_after_failure_bytes=after_failure,
        driver_allocated_after_cleanup_bytes=after_cleanup,
        recommended_maximum_bytes=recommended,
        required_memory_bytes=wan_required_tensor_bytes(request.requested_tensor_envelope),
        recipe_digest=wan_branch_recipe_digest(request.recipes[index]),
        inventory_digest=wan_inventory_digest(request.inventories[index]), fits=False,
        classification="recognized_mps_allocation_failure",
    )


class _ObservedMpsOom(RuntimeError):
    def __init__(
        self,
        phase: WanCanaryPhase,
        before: int,
        after_failure: int,
        recommended: int,
    ) -> None:
        self.phase = phase
        self.before = before
        self.after_failure = after_failure
        self.recommended = recommended
        super().__init__("wan_capacity_observed_mps_oom")


def _observed_call(
    hooks: _WanCanaryHooks,
    phase: WanCanaryPhase,
    operation: Callable[[], Any],
) -> Any:
    try:
        before, recommended = hooks.capacity_observation()
    except BaseException:
        raise RuntimeError("wan_capacity_observation_unavailable") from None
    try:
        return operation()
    except BaseException as error:
        if not is_recognized_mps_oom(error):
            raise
        try:
            after_failure, observed_recommended = hooks.capacity_observation()
        except BaseException:
            raise RuntimeError("wan_capacity_observation_unavailable") from error
        if observed_recommended != recommended:
            raise RuntimeError("wan_capacity_observation_unavailable") from error
        raise _ObservedMpsOom(phase, before, after_failure, recommended) from None


def run_wan_curated_host_canary(
    *,
    execution: WanProductionCanaryExecutionV2,
    database: Database,
    artifacts: ArtifactStore,
    cancelled: Callable[[], bool] = lambda: False,
    _hooks: _WanCanaryHooks | None = None,
    _importer: Callable[[str], object] | None = None,
) -> WanProductionCanaryReceiptV2 | WanProductionCanaryUnavailableV2:
    """Run each trained and fresh-restored branch without overlapping residency."""

    execution = require_strict_wan_production_execution(execution)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("wan_mps_fallback_forbidden")
    if _hooks is not None:
        return _run_wan_curated_host_canary(
            execution=execution,
            hooks=_hooks,
            cancelled=cancelled,
            hooks_are_injected=True,
        )
    try:
        lease = execution.training_source_lease
    except AttributeError:
        raise ValueError("wan_canary_source_lease_required") from None
    if not hasattr(lease, "reverify") or not hasattr(lease, "sources"):
        raise ValueError("wan_canary_source_lease_required")
    with ExitStack() as authorities:
        output = RetainedDirectoryAuthority.capture(
            execution.output_root,
            error_code="wan_canary_output_root_changed",
        )
        authorities.callback(output.close)
        checkpoint = RetainedDirectoryAuthority.capture(
            execution.checkpoint_root,
            error_code="wan_canary_checkpoint_root_changed",
        )
        authorities.callback(checkpoint.close)
        database_authority = RetainedSqliteAuthority.capture(database.path)
        authorities.callback(database_authority.close)
        _validate_execution_roots(execution, output, checkpoint)
        hooks = _production_hooks(
            execution,
            database,
            artifacts,
            importer=_importer,
            cancelled=cancelled,
            output_authority=output,
            checkpoint_authority=checkpoint,
            database_authority=database_authority,
        )
        return _run_wan_curated_host_canary(
            execution=execution,
            hooks=_guard_hooks(hooks, execution, output, checkpoint),
            cancelled=cancelled,
            hooks_are_injected=False,
        )


def _run_wan_curated_host_canary(
    *,
    execution: WanProductionCanaryExecutionV2,
    hooks: _WanCanaryHooks,
    cancelled: Callable[[], bool],
    hooks_are_injected: bool,
) -> WanProductionCanaryReceiptV2 | WanProductionCanaryUnavailableV2:
    hooks.preflight()
    request = execution.request
    results: list[WanBranchCanaryResultV1] = []
    probes: list[WanBranchNumericalProbeV2] = []
    releases: list[tuple[str, int]] = []
    branches: tuple[WanTransformerBranch, WanTransformerBranch] = (
        "high_noise", "low_noise"
    )

    for index, branch in enumerate(branches):
        _cancel(cancelled)
        inventory = hooks.inventory(branch, request.inventories[index])
        if isinstance(inventory, WanProductionCanaryUnavailableV2):
            return inventory
        if inventory != request.inventories[index]:
            raise ValueError("wan_canary_inventory_stale")

    _cancel(cancelled)
    hooks.device_preflight()

    for index, branch in enumerate(branches):
        recipe = request.recipes[index]
        try:
            with ExitStack() as cleanup:
                _cancel(cancelled)
                trained = _observed_call(
                    hooks,
                    "load_trained",
                    partial(hooks.load, branch, "trained"),
                )
                close_trained = _CloseOnce(hooks.close, trained)
                cleanup.callback(close_trained)
                _cancel(cancelled)
                _observed_call(
                    hooks,
                    "step",
                    partial(hooks.step, trained, recipe),
                )
                _cancel(cancelled)
                _observed_call(
                    hooks,
                    "checkpoint",
                    partial(hooks.checkpoint, trained),
                )
                _cancel(cancelled)
                close_trained()
                _cancel(cancelled)
                residual = hooks.memory_released(
                    branch, "trained", request.memory_release_bound_bytes
                )
                releases.append((f"{branch}.trained", residual))
                _cancel(cancelled)
                restored = _observed_call(
                    hooks,
                    "load_fresh_restore",
                    partial(hooks.load, branch, "restored"),
                )
                close_restored = _CloseOnce(hooks.close, restored)
                cleanup.callback(close_restored)
                _cancel(cancelled)
                _observed_call(
                    hooks,
                    "restore",
                    partial(hooks.restore, restored, recipe),
                )
                _cancel(cancelled)
                outcome = _observed_call(
                    hooks,
                    "continuity",
                    partial(hooks.continuity, restored, recipe),
                )
                if type(outcome) is _WanContinuityOutcome:
                    result = outcome.result
                    probe = outcome.probe
                elif hooks_are_injected and type(outcome) is WanBranchCanaryResultV1:
                    result = outcome
                    probe = WanBranchNumericalProbeV2(
                        branch=recipe.branch,
                        recipe_digest=result.recipe_digest,
                        tensor_name="probe.output",
                        shape=(2,),
                        dtype="float32",
                        values=(result.max_absolute_error, result.max_relative_error),
                    )
                else:
                    raise ValueError("wan_canary_probe_invalid")
                if (
                    result.branch != branch
                    or result.recipe_digest != wan_branch_recipe_digest(recipe)
                ):
                    raise ValueError("wan_canary_continuity_mismatch")
                results.append(result)
                probes.append(probe)
                _cancel(cancelled)
                close_restored()
                _cancel(cancelled)
                residual = hooks.memory_released(
                    branch, "restored", request.memory_release_bound_bytes
                )
                releases.append((f"{branch}.restored", residual))
        except _ObservedMpsOom as error:
            try:
                hooks.capacity_cleanup()
                after_cleanup, after_recommended = hooks.capacity_observation()
            except BaseException:
                raise RuntimeError("wan_capacity_observation_unavailable") from error
            if after_recommended != error.recommended:
                raise RuntimeError("wan_capacity_observation_unavailable") from error
            raise WanCapacityExceeded(
                _capacity(
                    request,
                    branch,
                    error.phase,
                    hooks,
                    error.before,
                    error.after_failure,
                    after_cleanup,
                    error.recommended,
                )
            ) from None
    _cancel(cancelled)
    receipt = WanProductionCanaryReceiptV2(
        schema_version="wan-mps-production-canary-receipt.v2",
        branch_results=(results[0], results[1]),
        branch_probes=(probes[0], probes[1]),
        compilation_receipt_digest=request.compilation_receipt_digest,
        batch_digest=request.batch_digest,
        inventory_family_digest=wan_inventory_family_digest(request.inventories),
        recipe_family_digest=wan_recipe_family_digest(request.recipes),
        primary_run_id=request.primary_run_id,
        independent_run_id=request.independent_run_id,
        dependency_versions=request.dependency_versions,
        hardware_identity=request.hardware_identity,
        memory_release_bytes=(releases[0], releases[1], releases[2], releases[3]),
        cleanup_state="cleaned",
    )
    return hooks.publish(receipt)


__all__ = (
    "WanCanaryCancelled",
    "WanCapacityExceeded",
    "WanProductionCanaryExecutionV2",
    "WanProductionCanaryUnavailableV2",
    "require_strict_wan_production_execution",
    "run_wan_curated_host_canary",
)
