"""Inventory-first, local-only loading for one Wan transformer branch."""

from __future__ import annotations

import hashlib
import importlib
import os
import threading
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path, PureWindowsPath
from typing import Any, cast

from video_lab.trainers.wan_factory import (
    DEFAULT_WAN_SNAPSHOT_COPY_LIMITS,
    WanPinnedLocalRoot,
    WanRetainedRootFile,
    WanRetainedRootLease,
    WanSnapshotCopyLimits,
    remove_wan_snapshot_at,
    require_component_on_mps,
    retain_pinned_root,
    secure_snapshot_pinned_root_at,
)
from video_lab.trainers.wan_mps import WanPreparedTraining
from video_lab.trainers.wan_production_models import (
    WanBranchInventoryResultV1,
    WanBranchInventoryRole,
    WanBranchRecipeV1,
    WanComponentInventoryV1,
    WanComponentManifest,
    WanComponentName,
    WanTransformerBranch,
)

_BRANCH_LOAD_LOCK = threading.Lock()
@dataclass(frozen=True, slots=True)
class WanBranchLoadRequest:
    """Exact local authority and relative component locations for one branch."""

    model_root: WanPinnedLocalRoot
    snapshot_parent: Path
    branch: WanTransformerBranch
    scheduler: Path
    text_encoder: Path
    tokenizer: Path
    transformer_config: Path
    high_transformer: Path
    low_transformer: Path
    vae: Path
    _authority_signature: tuple[tuple[str, tuple[int, ...]], ...] = field(
        init=False, repr=False, compare=False
    )
    _snapshot_parent_identity: tuple[int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.model_root) is not WanPinnedLocalRoot:
            raise ValueError("wan_branch_request_invalid")
        if self.branch not in ("high_noise", "low_noise"):
            raise ValueError("wan_branch_request_invalid")
        parent = Path(self.snapshot_parent)
        if not parent.is_absolute() or not parent.is_dir() or parent.is_symlink():
            raise ValueError("wan_branch_request_invalid")
        object.__setattr__(self, "snapshot_parent", parent)
        parent_descriptor = _open_directory_descriptor(
            parent, error="wan_branch_request_invalid"
        )
        try:
            parent_info = os.fstat(parent_descriptor)
            object.__setattr__(
                self,
                "_snapshot_parent_identity",
                (parent_info.st_dev, parent_info.st_ino),
            )
        finally:
            os.close(parent_descriptor)
        for name in (
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer_config",
            "high_transformer",
            "low_transformer",
            "vae",
        ):
            value = _safe_relative_path(getattr(self, name))
            object.__setattr__(self, name, value)
        try:
            with retain_pinned_root(self.model_root) as lease:
                object.__setattr__(self, "_authority_signature", _lease_signature(lease))
        except ValueError:
            raise ValueError("wan_factory_snapshot_invalid") from None


class WanLoadedBranch:
    """Exclusive owner of one transformer and its retained local authorities."""

    def __init__(
        self,
        *,
        branch: WanTransformerBranch,
        inventory: WanComponentInventoryV1,
        source_lease: WanRetainedRootLease,
        snapshot: WanPinnedLocalRoot,
        snapshot_lease: WanRetainedRootLease,
        transformer: object,
        torch: Any,
        source_scope: AbstractContextManager[WanRetainedRootLease],
        snapshot_scope: AbstractContextManager[WanRetainedRootLease],
        snapshot_parent_descriptor: int,
        snapshot_root_descriptor: int,
        snapshot_root_identity: tuple[int, int],
    ) -> None:
        self.branch = branch
        self.inventory = inventory
        self.source_lease: WanRetainedRootLease | None = source_lease
        self.snapshot = snapshot
        self.snapshot_lease: WanRetainedRootLease | None = snapshot_lease
        self.transformer: object | None = transformer
        self._torch = torch
        self._source_scope: AbstractContextManager[WanRetainedRootLease] | None = source_scope
        self._snapshot_scope: AbstractContextManager[WanRetainedRootLease] | None = snapshot_scope
        self._snapshot_parent_descriptor: int | None = snapshot_parent_descriptor
        self._snapshot_root_descriptor: int | None = snapshot_root_descriptor
        self._snapshot_root_identity = snapshot_root_identity
        self._closed = False

    def close(self) -> None:
        """Drop all owned state and exhaust every cleanup boundary exactly once."""

        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        self.transformer = None
        torch, self._torch = self._torch, None
        try:
            torch.mps.synchronize()
        except BaseException:
            failures.append(RuntimeError("wan_branch_mps_synchronize_failed"))
        try:
            torch.mps.empty_cache()
        except BaseException:
            failures.append(RuntimeError("wan_branch_mps_cache_cleanup_failed"))
        parent_descriptor = self._snapshot_parent_descriptor
        root_descriptor = self._snapshot_root_descriptor
        self._snapshot_parent_descriptor = None
        self._snapshot_root_descriptor = None
        if parent_descriptor is not None and root_descriptor is not None:
            try:
                _remove_snapshot_at(
                    parent_descriptor=parent_descriptor,
                    root_descriptor=root_descriptor,
                    root_identity=self._snapshot_root_identity,
                )
            except BaseException:
                failures.append(RuntimeError("wan_branch_snapshot_cleanup_failed"))
        self.snapshot_lease = None
        snapshot_scope, self._snapshot_scope = self._snapshot_scope, None
        if snapshot_scope is not None:
            try:
                snapshot_scope.__exit__(None, None, None)
            except BaseException:
                failures.append(RuntimeError("wan_branch_snapshot_lease_close_failed"))
        self.source_lease = None
        source_scope, self._source_scope = self._source_scope, None
        if source_scope is not None:
            try:
                source_scope.__exit__(None, None, None)
            except BaseException:
                failures.append(RuntimeError("wan_branch_source_lease_close_failed"))
        for descriptor, label in (
            (root_descriptor, "wan_branch_snapshot_descriptor_close_failed"),
            (parent_descriptor, "wan_branch_snapshot_parent_close_failed"),
        ):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    failures.append(RuntimeError(label))
        try:
            _BRANCH_LOAD_LOCK.release()
        except BaseException:
            failures.append(RuntimeError("wan_branch_load_lease_release_failed"))
        if failures:
            raise BaseExceptionGroup("wan_branch_cleanup_failed", failures)


@dataclass(frozen=True, slots=True)
class WanMpsPreparedFactory:
    """Prepare the branch selected by an immutable production recipe."""

    request: WanBranchLoadRequest
    importer: Callable[[str], object] = importlib.import_module
    limits: WanSnapshotCopyLimits = DEFAULT_WAN_SNAPSHOT_COPY_LIMITS

    def prepare_branch(
        self,
        sources: object,
        request: object,
        branch_recipe: WanBranchRecipeV1,
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> WanPreparedTraining:
        from video_lab.trainers.wan_branch_runtime import WanMpsBranchOperations
        from video_lab.trainers.wan_mps import WanTrainingSourceLease, WanTrainingSources

        selected = replace(self.request, branch=branch_recipe.branch)
        object.__setattr__(
            selected, "_authority_signature", self.request._authority_signature
        )
        object.__setattr__(
            selected,
            "_snapshot_parent_identity",
            self.request._snapshot_parent_identity,
        )
        loaded = load_wan_branch(selected, limits=self.limits, importer=self.importer)
        # Preserve the Task 2 loader-only boundary for callers that have not
        # supplied the versioned training-source authority yet.
        if not isinstance(sources, (WanTrainingSourceLease, WanTrainingSources)):
            return WanPreparedTraining(handle=loaded)
        try:
            if type(sources) is WanTrainingSourceLease:
                operations = WanMpsBranchOperations.from_source_lease(
                    loaded_branch=loaded,
                    recipe=branch_recipe,
                    source_lease=sources,
                    importer=self.importer,
                    cancelled=cancelled,
                )
            else:
                assert isinstance(sources, WanTrainingSources)
                operations = WanMpsBranchOperations.from_sources(
                    loaded_branch=loaded,
                    recipe=branch_recipe,
                    sources=sources,
                    importer=self.importer,
                    cancelled=cancelled,
                )
        except BaseException:
            loaded.close()
            raise
        return WanPreparedTraining(
            handle=operations
        )

    def __call__(
        self,
        sources: object,
        request: object,
        branch_recipe: WanBranchRecipeV1,
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> WanPreparedTraining:
        return self.prepare_branch(
            sources, request, branch_recipe, cancelled=cancelled
        )


def inventory_wan_branch(
    request: WanBranchLoadRequest,
    *,
    limits: WanSnapshotCopyLimits = DEFAULT_WAN_SNAPSHOT_COPY_LIMITS,
) -> WanBranchInventoryResultV1:
    """Hash a bounded retained root without importing any ML dependency."""

    with retain_pinned_root(request.model_root, limits=limits) as lease:
        _require_request_authority(lease, request)
        lease.reverify()
        return _inventory_from_retained_files(lease, request)


def load_wan_branch(
    request: WanBranchLoadRequest,
    *,
    limits: WanSnapshotCopyLimits = DEFAULT_WAN_SNAPSHOT_COPY_LIMITS,
    importer: Callable[[str], object] = importlib.import_module,
) -> WanLoadedBranch:
    """Load exactly one local transformer after a successful inventory preflight."""

    if not _BRANCH_LOAD_LOCK.acquire(blocking=False):
        raise ValueError("wan_branch_concurrent_load")
    source_scope: AbstractContextManager[WanRetainedRootLease] | None = None
    snapshot_scope: AbstractContextManager[WanRetainedRootLease] | None = None
    snapshot: WanPinnedLocalRoot | None = None
    snapshot_parent_descriptor: int | None = None
    snapshot_root_descriptor: int | None = None
    snapshot_root_identity: tuple[int, int] | None = None
    snapshot_lease: WanRetainedRootLease | None = None
    torch: Any = None
    transformer: object | None = None
    try:
        snapshot_parent_descriptor = _open_snapshot_parent(request)
        source_scope = retain_pinned_root(request.model_root, limits=limits)
        source_lease = source_scope.__enter__()
        _require_request_authority(source_lease, request)
        inventory_result = _inventory_from_retained_files(source_lease, request)
        source_lease.reverify()
        if not inventory_result.available:
            raise ValueError("wan_branch_inventory_unavailable")
        inventory = inventory_result.inventory
        assert inventory is not None
        destination = request.snapshot_parent / f"wan-branch-{uuid.uuid4().hex}"
        snapshot = secure_snapshot_pinned_root_at(
            request.model_root,
            destination,
            destination_parent_descriptor=snapshot_parent_descriptor,
            limits=limits,
            lease=source_lease,
        )
        source_lease.reverify()
        snapshot_root_descriptor = os.open(
            destination.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=snapshot_parent_descriptor,
        )
        snapshot_info = os.fstat(snapshot_root_descriptor)
        snapshot_root_identity = (snapshot_info.st_dev, snapshot_info.st_ino)
        snapshot_scope = retain_pinned_root(snapshot, limits=limits)
        snapshot_lease = snapshot_scope.__enter__()
        retained_root_info = os.fstat(snapshot_lease.directory_descriptors[0])
        if (retained_root_info.st_dev, retained_root_info.st_ino) != snapshot_root_identity:
            raise ValueError("wan_factory_snapshot_invalid")
        snapshot_lease.reverify()

        torch = cast(Any, importer("torch"))
        diffusers = cast(Any, importer("diffusers"))
        safetensors = cast(Any, importer("safetensors.torch"))
        selected = (
            request.high_transformer
            if request.branch == "high_noise"
            else request.low_transformer
        )
        try:
            checkpoint = safetensors.load_file(
                str(_retained_file_path(snapshot_lease, selected))
            )
            if type(checkpoint) is not dict:
                raise ValueError
            transformer = diffusers.WanTransformer3DModel.from_single_file(
                checkpoint,
                config=str(
                    _retained_file_path(snapshot_lease, request.transformer_config)
                ),
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )
        except BaseException:
            raise RuntimeError("wan_branch_load_failed") from None
        cast(Any, transformer).to(device="mps", dtype=torch.bfloat16)
        require_component_on_mps(transformer)
        source_lease.reverify()
        snapshot_lease.reverify()
        return WanLoadedBranch(
            branch=request.branch,
            inventory=inventory,
            source_lease=source_lease,
            snapshot=snapshot,
            snapshot_lease=snapshot_lease,
            transformer=transformer,
            torch=torch,
            source_scope=source_scope,
            snapshot_scope=snapshot_scope,
            snapshot_parent_descriptor=snapshot_parent_descriptor,
            snapshot_root_descriptor=snapshot_root_descriptor,
            snapshot_root_identity=snapshot_root_identity,
        )
    except BaseException as primary:
        failures = _cleanup_failed_load(
            torch=torch,
            transformer=transformer,
            source_scope=source_scope,
            snapshot_scope=snapshot_scope,
            snapshot=snapshot,
            snapshot_parent_descriptor=snapshot_parent_descriptor,
            snapshot_root_descriptor=snapshot_root_descriptor,
            snapshot_root_identity=snapshot_root_identity,
        )
        if failures:
            raise BaseExceptionGroup(
                "wan_branch_load_and_cleanup_failed", (primary, *failures)
            ) from None
        raise


def _inventory_from_retained_files(
    lease: WanRetainedRootLease, request: WanBranchLoadRequest
) -> WanBranchInventoryResultV1:
    selected = (
        request.high_transformer
        if request.branch == "high_noise"
        else request.low_transformer
    )
    selected_role: WanBranchInventoryRole = (
        "high_transformer" if request.branch == "high_noise" else "low_transformer"
    )
    root_locations: dict[WanBranchInventoryRole, Path] = {
        "scheduler": request.scheduler,
        "text_encoder": request.text_encoder,
        "tokenizer": request.tokenizer,
        "vae": request.vae,
    }
    files = {item.relative: item for item in lease.files}
    role_files = {
        role: {
            relative
            for relative in files
            if relative == location or relative.is_relative_to(location)
        }
        for role, location in root_locations.items()
    }
    role_files["transformer_config"] = (
        {request.transformer_config} if request.transformer_config in files else set()
    )
    role_files[selected_role] = {selected} if selected in files else set()
    missing_roles = tuple(sorted(role for role, matches in role_files.items() if not matches))
    if missing_roles:
        return WanBranchInventoryResultV1(
            available=False,
            inventory=None,
            missing_roles=missing_roles,
        )

    component_files: dict[WanComponentName, set[Path]] = {
        "scheduler": role_files["scheduler"],
        "text_encoder": role_files["text_encoder"],
        "tokenizer": role_files["tokenizer"],
        "transformer": role_files["transformer_config"] | role_files[selected_role],
        "vae": role_files["vae"],
    }
    manifests: list[WanComponentManifest] = []
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        records = [
            relative.as_posix().encode() + b"\0" + files[relative].digest
            for relative in sorted(component_files[component])
        ]
        manifests.append(
            (component, "sha256:" + hashlib.sha256(b"\n".join(records)).hexdigest())
        )
    return WanBranchInventoryResultV1(
        available=True,
        inventory=WanComponentInventoryV1(
            schema_version="wan-mps-component-inventory.v1",
            component_manifests=tuple(manifests),
        ),
        missing_roles=(),
    )


def _safe_relative_path(value: object) -> Path:
    try:
        candidate = Path(cast(Path, value))
        windows = PureWindowsPath(str(value))
        if (
            candidate.is_absolute()
            or windows.anchor
            or windows.drive
            or not candidate.parts
            or "\\" in str(value)
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or any(PureWindowsPath(part).is_reserved() for part in candidate.parts)
        ):
            raise ValueError
        return candidate
    except (OSError, TypeError, ValueError):
        raise ValueError("wan_branch_request_invalid") from None


def _lease_signature(lease: WanRetainedRootLease) -> tuple[tuple[str, tuple[int, ...]], ...]:
    records: list[tuple[str, tuple[int, ...]]] = [
        (".", tuple(lease.directory_identities[0]))
    ]
    records.extend(
        (f"d:{relative.as_posix()}", tuple(identity))
        for relative, identity in zip(
            lease.directories, lease.directory_identities[1:], strict=True
        )
    )
    records.extend(
        (f"f:{item.relative.as_posix()}", tuple(item.identity)) for item in lease.files
    )
    return tuple(records)


def _require_request_authority(
    lease: WanRetainedRootLease, request: WanBranchLoadRequest
) -> None:
    if _lease_signature(lease) != request._authority_signature:
        raise ValueError("wan_factory_snapshot_invalid")


def _retained_file_path(lease: WanRetainedRootLease, relative: Path) -> Path:
    retained_file: WanRetainedRootFile | None = next(
        (item for item in lease.files if item.relative == relative), None
    )
    if retained_file is None:
        raise ValueError("wan_factory_snapshot_invalid")
    descriptor_path = Path("/dev/fd") / str(retained_file.descriptor)
    probe_descriptor: int | None = None
    try:
        probe_descriptor = os.open(
            descriptor_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        if not os.path.samestat(
            os.fstat(retained_file.descriptor), os.fstat(probe_descriptor)
        ):
            raise ValueError
    except (OSError, TypeError, ValueError):
        raise ValueError("wan_factory_snapshot_invalid") from None
    finally:
        if probe_descriptor is not None:
            os.close(probe_descriptor)
    return descriptor_path


def _open_directory_descriptor(path: Path, *, error: str) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        if not os.path.samestat(os.fstat(descriptor), os.stat(path, follow_symlinks=False)):
            raise ValueError
        return descriptor
    except (OSError, TypeError, ValueError):
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError(error) from None


def _open_snapshot_parent(request: WanBranchLoadRequest) -> int:
    descriptor = _open_directory_descriptor(
        request.snapshot_parent,
        error="wan_branch_snapshot_parent_invalid",
    )
    info = os.fstat(descriptor)
    if (info.st_dev, info.st_ino) != request._snapshot_parent_identity:
        os.close(descriptor)
        raise ValueError("wan_branch_snapshot_parent_invalid")
    return descriptor


def _cleanup_failed_load(
    *,
    torch: Any,
    transformer: object | None,
    source_scope: AbstractContextManager[WanRetainedRootLease] | None,
    snapshot_scope: AbstractContextManager[WanRetainedRootLease] | None,
    snapshot: WanPinnedLocalRoot | None,
    snapshot_parent_descriptor: int | None,
    snapshot_root_descriptor: int | None,
    snapshot_root_identity: tuple[int, int] | None,
) -> tuple[BaseException, ...]:
    del transformer, snapshot
    failures: list[BaseException] = []
    if torch is not None:
        try:
            torch.mps.synchronize()
        except BaseException:
            failures.append(RuntimeError("wan_branch_mps_synchronize_failed"))
        try:
            torch.mps.empty_cache()
        except BaseException:
            failures.append(RuntimeError("wan_branch_mps_cache_cleanup_failed"))
    if (
        snapshot_parent_descriptor is not None
        and snapshot_root_descriptor is not None
        and snapshot_root_identity is not None
    ):
        try:
            _remove_snapshot_at(
                parent_descriptor=snapshot_parent_descriptor,
                root_descriptor=snapshot_root_descriptor,
                root_identity=snapshot_root_identity,
            )
        except BaseException:
            failures.append(RuntimeError("wan_branch_snapshot_cleanup_failed"))
    for scope, label in (
        (snapshot_scope, "wan_branch_snapshot_lease_close_failed"),
        (source_scope, "wan_branch_source_lease_close_failed"),
    ):
        if scope is not None:
            try:
                scope.__exit__(None, None, None)
            except BaseException:
                failures.append(RuntimeError(label))
    for descriptor, label in (
        (snapshot_root_descriptor, "wan_branch_snapshot_descriptor_close_failed"),
        (snapshot_parent_descriptor, "wan_branch_snapshot_parent_close_failed"),
    ):
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                failures.append(RuntimeError(label))
    try:
        _BRANCH_LOAD_LOCK.release()
    except BaseException:
        failures.append(RuntimeError("wan_branch_load_lease_release_failed"))
    return tuple(failures)


_remove_snapshot_at = remove_wan_snapshot_at

__all__ = (
    "WanBranchLoadRequest",
    "WanLoadedBranch",
    "WanMpsPreparedFactory",
    "inventory_wan_branch",
    "load_wan_branch",
)
