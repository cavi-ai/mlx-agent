"""Explicit, dependency-free construction for the local Wan MPS trainer."""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from video_lab.domain import canonical_json
from video_lab.trainer_protocol import TrainerBackend, TrainingRequest
from video_lab.trainers.wan_checkpoint import WanParentCheckpointDescriptor
from video_lab.trainers.wan_diffusers_runtime import (
    WanDiffusersAdapterPaths,
    WanDiffusersTrainingRuntime,
)
from video_lab.trainers.wan_mps import (
    WanLoRAMpsTrainer,
    WanSourceDescriptor,
    WanTrainingRuntime,
    WanTrainingSourceLease,
    WanTrainingSourceResolver,
)
from video_lab.training_models import Sha256Digest

WAN_ANIMEGEN_ADAPTER_ID = "diffusers.wan2.2-animegen-precomputed-batch-lora.v1"


def _is_mps_device(device: object) -> bool:
    observed_type = getattr(device, "type", None)
    if observed_type is not None:
        return type(observed_type) is str and observed_type == "mps"
    text = str(device)
    return text == "mps" or text.startswith("mps:")


def is_bfloat16_dtype(value: object) -> bool:
    return str(value).removeprefix("torch.") == "bfloat16"


@dataclass(frozen=True, slots=True)
class WanPinnedLocalRoot:
    """A root whose revision is bound to a canonical complete file manifest."""

    path: Path
    revision: str
    manifest_digest: Sha256Digest | None = None


@dataclass(frozen=True, slots=True)
class WanSnapshotCopyLimits:
    max_files: int = 100_000
    max_bytes: int = 64 * 1024 * 1024 * 1024
    max_depth: int = 32


DEFAULT_WAN_SNAPSHOT_COPY_LIMITS = WanSnapshotCopyLimits()


@dataclass(frozen=True, slots=True)
class WanRetainedRootFile:
    relative: Path
    descriptor: int
    identity: tuple[int, int, int, int, int, int, int]
    digest: bytes


@dataclass(frozen=True, slots=True)
class WanRetainedRootLease:
    source: WanPinnedLocalRoot
    descriptors: tuple[int, ...]
    descriptor_identities: tuple[tuple[int, int], ...]
    directory_descriptors: tuple[int, ...]
    directory_identities: tuple[tuple[int, int, int, int, int, int, int], ...]
    directories: tuple[Path, ...]
    files: tuple[WanRetainedRootFile, ...]
    manifest_digest: Sha256Digest

    def reverify(self) -> None:
        if (
            tuple((os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in self.descriptors)
            != self.descriptor_identities
        ):
            raise ValueError("wan_factory_snapshot_invalid")
        if (
            tuple(_manifest_identity(os.fstat(fd)) for fd in self.directory_descriptors)
            != self.directory_identities
        ):
            raise ValueError("wan_factory_snapshot_invalid")
        for item in self.files:
            info = os.fstat(item.descriptor)
            if (
                _manifest_identity(info) != item.identity
                or hashlib.sha256(os.pread(item.descriptor, info.st_size, 0)).digest()
                != item.digest
            ):
                raise ValueError("wan_factory_snapshot_invalid")
        if root_manifest_digest(self.source.path) != self.manifest_digest:
            raise ValueError("wan_factory_snapshot_invalid")


@contextmanager
def retain_pinned_root(
    source: WanPinnedLocalRoot,
    *,
    limits: WanSnapshotCopyLimits = DEFAULT_WAN_SNAPSHOT_COPY_LIMITS,
) -> Iterator[WanRetainedRootLease]:
    """Retain every ancestor, directory, and file before snapshot copying starts."""

    descriptors: list[int] = []
    files: list[WanRetainedRootFile] = []
    directories: list[Path] = []
    directory_descriptors: list[int] = []
    records: list[bytes] = []
    byte_count = 0
    primary: BaseException | None = None
    lease: WanRetainedRootLease | None = None
    try:
        if type(source) is not WanPinnedLocalRoot or source.manifest_digest is None:
            raise ValueError
        else:
            descriptors.extend(_open_directory_chain(source.path))
            root_fd = descriptors[-1]
            directory_descriptors.append(root_fd)

            def retain_directory(directory_fd: int, relative: Path, depth: int) -> None:
                nonlocal byte_count
                if depth > limits.max_depth:
                    raise ValueError
                for name in sorted(os.listdir(directory_fd)):
                    if name in {"", ".", ".."} or "/" in name:
                        raise ValueError
                    child_relative = relative / name
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        child = os.open(
                            name,
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_CLOEXEC", 0),
                            dir_fd=directory_fd,
                        )
                        descriptors.append(child)
                        if _manifest_identity(os.fstat(child)) != _manifest_identity(info):
                            raise ValueError
                        directory_descriptors.append(child)
                        directories.append(child_relative)
                        retain_directory(child, child_relative, depth + 1)
                        continue
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink < 1:
                        raise ValueError
                    descriptor = _open_regular_file_at(directory_fd, name)
                    descriptors.append(descriptor)
                    identity = _manifest_identity(os.fstat(descriptor))
                    if identity != _manifest_identity(info):
                        raise ValueError
                    byte_count += info.st_size
                    if len(files) >= limits.max_files or byte_count > limits.max_bytes:
                        raise ValueError
                    digest = hashlib.sha256(os.pread(descriptor, info.st_size, 0)).digest()
                    if _manifest_identity(os.fstat(descriptor)) != identity:
                        raise ValueError
                    files.append(
                        WanRetainedRootFile(child_relative, descriptor, identity, digest)
                    )
                    records.append(
                        child_relative.as_posix().encode()
                        + b"\0"
                        + str(info.st_size).encode()
                        + b"\0"
                        + digest
                    )

            retain_directory(root_fd, Path(), 0)
            manifest = "sha256:" + hashlib.sha256(b"\n".join(records)).hexdigest()
            if manifest != source.manifest_digest:
                raise ValueError
            lease = WanRetainedRootLease(
                source,
                tuple(descriptors),
                tuple((os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in descriptors),
                tuple(directory_descriptors),
                tuple(_manifest_identity(os.fstat(fd)) for fd in directory_descriptors),
                tuple(directories),
                tuple(files),
                manifest,
            )
            lease.reverify()
    except (OSError, TypeError, ValueError):
        primary = ValueError("wan_factory_snapshot_invalid")
    if primary is None:
        assert lease is not None
        try:
            yield lease
        except BaseException as error:
            primary = error
    close_failures: list[Exception] = []
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            close_failures.append(RuntimeError("wan_retained_root_close_failed"))
    cleanup: BaseException | None = None
    if close_failures:
        cleanup = ExceptionGroup("wan_retained_root_close_failed", close_failures)
    if primary is not None and cleanup is not None:
        raise BaseExceptionGroup("wan_retained_root_primary_and_close_failed", (primary, cleanup))
    if cleanup is not None:
        raise cleanup
    if primary is not None:
        raise primary from None


@dataclass(frozen=True, slots=True)
class WanTrainingAdapterCapability:
    """Repository-owned declaration of the exact adapter contract available locally."""

    adapter_id: str
    supported: bool
    reason: str
    required_batch_format: str


class WanTrainingAdapterUnsupported(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"wan_training_adapter_unsupported:{reason}")


def local_wan_training_adapter_capability() -> WanTrainingAdapterCapability:
    """Report the honest local boundary without importing ML libraries."""

    return WanTrainingAdapterCapability(
        adapter_id=WAN_ANIMEGEN_ADAPTER_ID,
        supported=True,
        reason="supported",
        required_batch_format="wan-precomputed-training-batch.v1",
    )


def require_component_on_mps(
    component: object,
    *,
    parameterless_kind: str | None = None,
) -> None:
    """Reject hidden CPU/offload state; inspect every executable tensor."""

    device_map = getattr(component, "hf_device_map", None)
    if device_map is not None and (
        not isinstance(device_map, Mapping)
        or not device_map
        or any(not _is_mps_device(device) for device in device_map.values())
    ):
        raise ValueError("wan_runtime_placement_invalid")
    declared_device = getattr(component, "device", None)
    if declared_device is not None and not _is_mps_device(declared_device):
        raise ValueError("wan_runtime_placement_invalid")
    parameters = getattr(component, "parameters", None)
    buffers = getattr(component, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        raise ValueError("wan_runtime_placement_invalid")
    parameter_tensors = tuple(parameters())
    buffer_tensors = tuple(buffers())
    parameter_devices = tuple(getattr(item, "device", None) for item in parameter_tensors)
    buffer_devices = tuple(getattr(item, "device", None) for item in buffer_tensors)
    dtypes = tuple(getattr(item, "dtype", None) for item in parameter_tensors + buffer_tensors)
    if not parameter_tensors and not buffer_tensors:
        if (
            parameterless_kind not in {"tokenizer", "scheduler"}
            or not _is_mps_device(getattr(component, "device", None))
            or not is_bfloat16_dtype(getattr(component, "dtype", None))
        ):
            raise ValueError("wan_runtime_placement_invalid")
        return
    if any(not _is_mps_device(device) for device in parameter_devices + buffer_devices) or any(
        not is_bfloat16_dtype(dtype) for dtype in dtypes
    ):
        raise ValueError("wan_runtime_placement_invalid")


def create_wan_lora_mps_trainer(
    *,
    backend_digest: Sha256Digest,
    material_digest: Sha256Digest,
    orchestration_source_digest: Sha256Digest | None = None,
    descriptor_source_digest: Sha256Digest | None = None,
    source: WanPinnedLocalRoot,
    cache: WanPinnedLocalRoot,
    artifact_root: Path,
    source_descriptors: tuple[WanSourceDescriptor, ...],
    runtime: WanTrainingRuntime | None = None,
    adapter_paths: WanDiffusersAdapterPaths | None = None,
    runtime_digest: Sha256Digest | None = None,
    library_importer: Callable[[str], object] = importlib.import_module,
    checkpoint_resolver: Callable[[TrainingRequest], WanParentCheckpointDescriptor] | None = None,
) -> TrainerBackend:
    """Compose a tested runtime, or fail rather than claim absent production support."""

    source_root = _require_pinned_root(source)
    _require_pinned_root(cache)
    _require_local_directory(artifact_root, "wan_factory_artifact_root_invalid")
    if (
        not source_descriptors
        or orchestration_source_digest is None
        or descriptor_source_digest is None
    ):
        raise ValueError("wan_factory_source_invalid")
    try:
        for descriptor in source_descriptors:
            candidate = Path(descriptor.path())
            if not candidate.is_absolute() or not candidate.is_relative_to(source_root):
                raise ValueError("wan_factory_source_invalid")
    except (OSError, TypeError, ValueError):
        raise ValueError("wan_factory_source_invalid") from None
    snapshot_root: Path | None = None
    ownership_transferred = False
    try:
        try:
            with WanTrainingSourceResolver(source_descriptors).acquire() as lease:
                if lease.sources.source_digest != descriptor_source_digest:
                    raise ValueError("wan_factory_source_invalid")
                if runtime is None:
                    if adapter_paths is None or runtime_digest is None:
                        raise WanTrainingAdapterUnsupported(
                            "adapter_paths_and_runtime_digest_required"
                        )
                    snapshot_root = _snapshot_adapter_inputs(
                        source_root, adapter_paths, artifact_root, lease=lease
                    )
        except ValueError:
            raise ValueError("wan_factory_source_invalid") from None
        if runtime is None:
            assert adapter_paths is not None and runtime_digest is not None
            assert snapshot_root is not None
            runtime = WanDiffusersTrainingRuntime(
                root=snapshot_root,
                paths=adapter_paths,
                runtime_digest=runtime_digest,
                importer=library_importer,
                cleanup=lambda: _remove_snapshot(snapshot_root),
            )
        elif adapter_paths is not None or runtime_digest is not None:
            raise ValueError("wan_factory_runtime_invalid")
        trainer = WanLoRAMpsTrainer(
            backend_digest=backend_digest,
            material_digest=material_digest,
            orchestration_source_digest=orchestration_source_digest,
            descriptor_source_digest=descriptor_source_digest,
            source_resolver=WanTrainingSourceResolver(source_descriptors),
            runtime=runtime,
            checkpoint_resolver=checkpoint_resolver,
        )
        ownership_transferred = True
        return trainer
    finally:
        if snapshot_root is not None and not ownership_transferred:
            _remove_snapshot(snapshot_root)


def root_manifest_digest(
    root: Path,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
    max_depth: int | None = None,
) -> Sha256Digest:
    """Canonical content identity for every regular file below an exact local root."""
    base = _require_local_directory(root, "wan_factory_root_invalid")
    descriptors: list[int] = []
    try:
        descriptors = _open_directory_chain(base)
        ancestor_identities = tuple(
            (os.fstat(item).st_dev, os.fstat(item).st_ino) for item in descriptors[:-1]
        )
        root_identity = _manifest_identity(os.fstat(descriptors[-1]))
        root_descriptor = descriptors[-1]
        records: list[bytes] = []
        observed_bytes = 0
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(base)
            info = path.stat(follow_symlinks=False)
            if max_depth is not None and len(relative.parts) > max_depth:
                raise ValueError("wan_factory_root_invalid")
            if stat.S_ISREG(info.st_mode):
                observed_bytes += info.st_size
                if (max_files is not None and len(records) >= max_files) or (
                    max_bytes is not None and observed_bytes > max_bytes
                ):
                    raise ValueError("wan_factory_root_invalid")
            record = _manifest_file_record(root_descriptor, relative)
            if record is not None:
                records.append(record)
        if (
            tuple((os.fstat(item).st_dev, os.fstat(item).st_ino) for item in descriptors[:-1])
            != ancestor_identities
            or _manifest_identity(os.fstat(descriptors[-1])) != root_identity
        ):
            raise ValueError("wan_factory_root_invalid")
        return "sha256:" + hashlib.sha256(b"\n".join(records)).hexdigest()
    except (OSError, ValueError):
        raise ValueError("wan_factory_root_invalid") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def secure_snapshot_pinned_root(
    source: WanPinnedLocalRoot,
    destination: Path,
    *,
    limits: WanSnapshotCopyLimits = DEFAULT_WAN_SNAPSHOT_COPY_LIMITS,
    lease: WanRetainedRootLease | None = None,
    after_retain: Callable[[], None] | None = None,
) -> WanPinnedLocalRoot:
    """Copy one pinned root exclusively from pre-retained file descriptors."""
    if (
        type(source) is not WanPinnedLocalRoot
        or source.manifest_digest is None
        or limits.max_files <= 0
        or limits.max_bytes <= 0
        or limits.max_depth <= 0
    ):
        raise ValueError("wan_factory_snapshot_invalid")
    destination = Path(destination)
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError("wan_factory_snapshot_invalid")
    if lease is None:
        with retain_pinned_root(source, limits=limits) as retained:
            return secure_snapshot_pinned_root(
                source,
                destination,
                limits=limits,
                lease=retained,
                after_retain=after_retain,
            )
    retained = lease
    try:
        if retained.source != source or retained.manifest_digest != source.manifest_digest:
            raise ValueError
        retained.reverify()
        if after_retain is not None:
            after_retain()
        retained.reverify()
        destination.mkdir(mode=0o700)
        for relative in sorted(retained.directories, key=lambda item: len(item.parts)):
            (destination / relative).mkdir(mode=0o700)
        for retained_file in retained.files:
            retained.reverify()
            target_file = os.open(
                destination / retained_file.relative,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            try:
                _copy_retained_file_to_descriptor(retained_file, target_file)
            finally:
                os.close(target_file)
            retained.reverify()
        copied_digest = root_manifest_digest(
            destination,
            max_files=limits.max_files,
            max_bytes=limits.max_bytes,
            max_depth=limits.max_depth,
        )
        if copied_digest != source.manifest_digest:
            raise ValueError
        for snapshot_item in sorted(destination.rglob("*"), reverse=True):
            os.chmod(
                snapshot_item,
                0o500 if snapshot_item.is_dir() else 0o400,
                follow_symlinks=False,
            )
        os.chmod(destination, 0o500, follow_symlinks=False)
        return WanPinnedLocalRoot(destination, source.revision, copied_digest)
    except BaseException as error:
        primary: BaseException = (
            ValueError("wan_factory_snapshot_invalid")
            if isinstance(error, (OSError, TypeError, ValueError))
            else error
        )
        try:
            _remove_snapshot(destination)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "wan_snapshot_construction_and_cleanup_failed", (primary, cleanup)
            ) from None
        raise primary from None


def secure_snapshot_pinned_root_at(
    source: WanPinnedLocalRoot,
    destination: Path,
    *,
    destination_parent_descriptor: int,
    limits: WanSnapshotCopyLimits = DEFAULT_WAN_SNAPSHOT_COPY_LIMITS,
    lease: WanRetainedRootLease | None = None,
) -> WanPinnedLocalRoot:
    """Copy a retained root beneath one already-open destination parent."""

    if (
        type(source) is not WanPinnedLocalRoot
        or source.manifest_digest is None
        or limits.max_files <= 0
        or limits.max_bytes <= 0
        or limits.max_depth <= 0
    ):
        raise ValueError("wan_factory_snapshot_invalid")
    destination = Path(destination)
    name = destination.name
    if not name or name in {".", ".."} or Path(name).parts != (name,):
        raise ValueError("wan_factory_snapshot_invalid")
    try:
        parent_info = os.fstat(destination_parent_descriptor)
        if not stat.S_ISDIR(parent_info.st_mode):
            raise ValueError
        os.stat(name, dir_fd=destination_parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except (OSError, TypeError, ValueError):
        raise ValueError("wan_factory_snapshot_invalid") from None
    else:
        raise ValueError("wan_factory_snapshot_invalid")
    if lease is None:
        with retain_pinned_root(source, limits=limits) as retained:
            return secure_snapshot_pinned_root_at(
                source,
                destination,
                destination_parent_descriptor=destination_parent_descriptor,
                limits=limits,
                lease=retained,
            )

    retained = lease
    root_descriptor: int | None = None
    directory_descriptors: dict[Path, int] = {}
    root_identity: tuple[int, int] | None = None
    primary: BaseException | None = None
    try:
        if retained.source != source or retained.manifest_digest != source.manifest_digest:
            raise ValueError
        retained.reverify()
        os.mkdir(name, mode=0o700, dir_fd=destination_parent_descriptor)
        created_info = os.stat(
            name,
            dir_fd=destination_parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(created_info.st_mode):
            raise ValueError
        root_identity = (created_info.st_dev, created_info.st_ino)
        root_descriptor = os.open(
            name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=destination_parent_descriptor,
        )
        root_info = os.fstat(root_descriptor)
        if (root_info.st_dev, root_info.st_ino) != root_identity:
            raise ValueError
        directory_descriptors[Path()] = root_descriptor
        for relative in sorted(retained.directories, key=lambda item: len(item.parts)):
            parent_descriptor = directory_descriptors[relative.parent]
            os.mkdir(relative.name, mode=0o700, dir_fd=parent_descriptor)
            directory_descriptors[relative] = os.open(
                relative.name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        for retained_file in retained.files:
            retained.reverify()
            target_file = os.open(
                retained_file.relative.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o400,
                dir_fd=directory_descriptors[retained_file.relative.parent],
            )
            try:
                _copy_retained_file_to_descriptor(retained_file, target_file)
                os.fchmod(target_file, 0o400)
            finally:
                os.close(target_file)
            retained.reverify()
        for relative, descriptor in sorted(
            directory_descriptors.items(), key=lambda item: len(item[0].parts), reverse=True
        ):
            del relative
            os.fchmod(descriptor, 0o500)
            os.fsync(descriptor)
        retained.reverify()
        return WanPinnedLocalRoot(destination, source.revision, source.manifest_digest)
    except BaseException as error:
        primary = (
            ValueError("wan_factory_snapshot_invalid")
            if isinstance(error, (OSError, TypeError, ValueError))
            else error
        )
    finally:
        for relative, descriptor in sorted(
            directory_descriptors.items(), key=lambda item: len(item[0].parts), reverse=True
        ):
            if relative != Path():
                try:
                    os.close(descriptor)
                except OSError:
                    if primary is None:
                        primary = RuntimeError("wan_snapshot_descriptor_close_failed")
        if primary is None and root_descriptor is not None:
            os.close(root_descriptor)
    assert primary is not None
    cleanup: BaseException | None = None
    cleanup_descriptor = root_descriptor
    if cleanup_descriptor is None and root_identity is not None:
        try:
            cleanup_descriptor = _open_snapshot_root_by_identity(
                destination_parent_descriptor, root_identity
            )
        except BaseException as error:
            cleanup = error
    if cleanup_descriptor is not None and root_identity is not None:
        try:
            remove_wan_snapshot_at(
                parent_descriptor=destination_parent_descriptor,
                root_descriptor=cleanup_descriptor,
                root_identity=root_identity,
            )
        except BaseException as error:
            cleanup = error
        try:
            os.close(cleanup_descriptor)
        except OSError:
            close_failure = RuntimeError("wan_snapshot_descriptor_close_failed")
            cleanup = (
                close_failure
                if cleanup is None
                else BaseExceptionGroup(
                    "wan_snapshot_cleanup_and_close_failed", (cleanup, close_failure)
                )
            )
    if cleanup is not None:
        raise BaseExceptionGroup(
            "wan_snapshot_construction_and_cleanup_failed", (primary, cleanup)
        ) from None
    raise primary from None


def _open_snapshot_root_by_identity(
    parent_descriptor: int, root_identity: tuple[int, int]
) -> int:
    try:
        matching_names: list[str] = []
        for name in os.listdir(parent_descriptor):
            info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if (info.st_dev, info.st_ino) == root_identity:
                matching_names.append(name)
        if len(matching_names) != 1:
            raise ValueError
        descriptor = os.open(
            matching_names[0],
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != root_identity:
            os.close(descriptor)
            raise ValueError
        return descriptor
    except (OSError, TypeError, ValueError):
        raise ValueError("wan_snapshot_cleanup_authority_invalid") from None


def remove_wan_snapshot_at(
    *,
    parent_descriptor: int,
    root_descriptor: int,
    root_identity: tuple[int, int],
) -> None:
    """Clear only the retained snapshot inode and unlink that inode from its parent."""

    failures: list[Exception] = []
    try:
        root_info = os.fstat(root_descriptor)
        if (root_info.st_dev, root_info.st_ino) != root_identity:
            raise ValueError
        os.fchmod(root_descriptor, 0o700)
    except (OSError, TypeError, ValueError):
        raise ValueError("wan_snapshot_cleanup_authority_invalid") from None
    _clear_directory_descriptor(root_descriptor, failures)
    matching_names: list[str] = []
    try:
        for name in os.listdir(parent_descriptor):
            info = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (info.st_dev, info.st_ino) == root_identity:
                matching_names.append(name)
    except OSError:
        failures.append(RuntimeError("wan_snapshot_cleanup_inventory_failed"))
    if len(matching_names) == 1:
        try:
            os.rmdir(matching_names[0], dir_fd=parent_descriptor)
        except OSError:
            failures.append(RuntimeError("wan_snapshot_cleanup_root_failed"))
    else:
        failures.append(RuntimeError("wan_snapshot_cleanup_root_missing"))
    try:
        os.fsync(parent_descriptor)
    except OSError:
        failures.append(RuntimeError("wan_snapshot_cleanup_fsync_failed"))
    if failures:
        raise ExceptionGroup("wan_snapshot_cleanup_failed", failures)


def _copy_retained_file_to_descriptor(
    retained_file: WanRetainedRootFile, target_file: int
) -> None:
    offset = 0
    while offset < retained_file.identity[4]:
        chunk = os.pread(
            retained_file.descriptor,
            min(1 << 20, retained_file.identity[4] - offset),
            offset,
        )
        if not chunk or os.write(target_file, chunk) != len(chunk):
            raise ValueError
        offset += len(chunk)
    os.fsync(target_file)


def _clear_directory_descriptor(descriptor: int, failures: list[Exception]) -> None:
    try:
        names = tuple(os.listdir(descriptor))
    except OSError:
        failures.append(RuntimeError("wan_snapshot_cleanup_inventory_failed"))
        return
    for name in names:
        try:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    os.fchmod(child, 0o700)
                    _clear_directory_descriptor(child, failures)
                finally:
                    os.close(child)
                os.rmdir(name, dir_fd=descriptor)
            else:
                os.unlink(name, dir_fd=descriptor)
        except OSError:
            failures.append(RuntimeError("wan_snapshot_cleanup_remove_failed"))


def _open_directory_chain(path: Path) -> list[int]:
    flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors = [os.open(path.anchor, flags)]
    try:
        for part in path.parts[1:]:
            descriptors.append(os.open(part, flags, dir_fd=descriptors[-1]))
        return descriptors
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _manifest_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _manifest_file_record(root_descriptor: int, relative: Path) -> bytes | None:
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    opened: list[int] = []
    current = root_descriptor
    try:
        ancestor_identities: list[tuple[int, int, int, int, int, int, int]] = []
        for part in relative.parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            opened.append(current)
            ancestor_identities.append(_manifest_identity(os.fstat(current)))
        leaf_info = os.stat(relative.name, dir_fd=current, follow_symlinks=False)
        if stat.S_ISDIR(leaf_info.st_mode):
            return None
        descriptor = _open_regular_file_at(current, relative.name)
        opened.append(descriptor)
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        size = 0
        while size < before.st_size:
            chunk = os.read(descriptor, min(1 << 20, before.st_size - size))
            if not chunk:
                raise ValueError
            digest.update(chunk)
            size += len(chunk)
        if size != before.st_size or _manifest_identity(os.fstat(descriptor)) != _manifest_identity(
            before
        ):
            raise ValueError
        for ancestor, identity in zip(opened[:-1], ancestor_identities, strict=True):
            if _manifest_identity(os.fstat(ancestor)) != identity:
                raise ValueError
        relative_bytes = relative.as_posix().encode()
        return relative_bytes + b"\0" + str(size).encode() + b"\0" + digest.digest()
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _open_regular_file_at(directory_fd: int, name: str) -> int:
    """Open a leaf nonblocking/no-follow and reject it before reading unless regular."""
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink < 1:
            raise ValueError
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _require_pinned_root(root: WanPinnedLocalRoot) -> Path:
    if (
        type(root) is not WanPinnedLocalRoot
        or len(root.revision) != 40
        or any(character not in "0123456789abcdef" for character in root.revision)
        or root.manifest_digest is None
        or root.manifest_digest != root_manifest_digest(root.path)
    ):
        raise ValueError("wan_factory_root_invalid")
    return _require_local_directory(root.path, "wan_factory_root_invalid")


def _require_local_directory(path: Path, error: str) -> Path:
    try:
        candidate = Path(path)
        if not candidate.is_absolute() or not candidate.is_dir() or candidate.is_symlink():
            raise ValueError(error)
        if os.stat(candidate, follow_symlinks=False).st_nlink < 1:
            raise ValueError(error)
        return candidate
    except (OSError, TypeError, ValueError):
        raise ValueError(error) from None


def _snapshot_adapter_inputs(
    source_root: Path,
    paths: WanDiffusersAdapterPaths,
    artifact_root: Path,
    lease: WanTrainingSourceLease,
) -> Path:
    """Stable-open exact adapter inputs into a private immutable loader tree."""
    snapshot = Path(tempfile.mkdtemp(prefix="wan-adapter-", dir=artifact_root))
    try:
        _fsync_directory(artifact_root)
        relative_files: set[Path] = set()
        for relative in (
            paths.high_transformer,
            paths.low_transformer,
            paths.batch_manifest,
            paths.batch_tensors,
        ):
            relative_files.add(_safe_relative(relative))
        approved = {
            source_file.path.relative_to(source_root): source_file
            for source_file in lease.sources.files
        }
        relative_files.update(
            relative
            for relative in approved
            if relative.is_relative_to(_safe_relative(paths.wan_snapshot))
        )
        if relative_files != set(approved):
            raise ValueError("wan_factory_source_invalid")
        manifest_records: list[dict[str, object]] = []
        for relative in sorted(relative_files):
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                destination,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            try:
                digest, size = lease.copy_file_to(approved[relative], descriptor)
                manifest_records.append(
                    {"relative_path": relative.as_posix(), "digest": digest, "size": size}
                )
            finally:
                os.close(descriptor)
        manifest = snapshot / ".wan-snapshot-manifest.json"
        manifest.write_text(
            canonical_json(
                {
                    "schema_version": "wan-private-snapshot.v1",
                    "source_digest": lease.sources.source_digest,
                    "files": manifest_records,
                }
            )
        )
        with manifest.open("rb") as manifest_file:
            os.fsync(manifest_file.fileno())
        for directory in sorted(
            (item for item in snapshot.rglob("*") if item.is_dir()), reverse=True
        ):
            os.chmod(directory, 0o500, follow_symlinks=False)
        os.chmod(snapshot, 0o500, follow_symlinks=False)
        directories = [snapshot, *(item for item in snapshot.rglob("*") if item.is_dir())]
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _fsync_directory(directory)
        _fsync_directory(artifact_root)
        return snapshot
    except Exception:
        _remove_snapshot(snapshot)
        raise ValueError("wan_factory_source_invalid") from None


def _safe_relative(path: Path) -> Path:
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("wan_factory_source_invalid")
    return candidate


def _stable_read_relative(root: Path, relative: Path) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | no_follow | getattr(os, "O_CLOEXEC", 0)
    current = os.open(root, directory_flags)
    opened: list[int] = [current]
    try:
        for part in relative.parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            opened.append(current)
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current,
        )
        opened.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise ValueError("wan_factory_source_invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("wan_factory_source_invalid")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("wan_factory_source_invalid")
        return b"".join(chunks)
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _remove_snapshot(root: Path) -> None:
    if not root.exists():
        return
    failures: list[Exception] = []
    try:
        nested = tuple(root.rglob("*"))
    except OSError:
        nested = ()
        failures.append(RuntimeError("wan_snapshot_cleanup_inventory_failed"))
    for path in (root, *nested):
        try:
            os.chmod(path, 0o700 if path.is_dir() else 0o600, follow_symlinks=False)
        except OSError:
            failures.append(RuntimeError("wan_snapshot_cleanup_mode_failed"))
    for path in sorted(nested, reverse=True):
        try:
            path.rmdir() if path.is_dir() else path.unlink()
        except OSError:
            failures.append(RuntimeError("wan_snapshot_cleanup_remove_failed"))
    try:
        root.rmdir()
    except OSError:
        failures.append(RuntimeError("wan_snapshot_cleanup_root_failed"))
    try:
        _fsync_directory(root.parent)
    except OSError:
        failures.append(RuntimeError("wan_snapshot_cleanup_fsync_failed"))
    if failures:
        raise ExceptionGroup("wan_snapshot_cleanup_failed", failures)


def remove_wan_snapshot(root: Path) -> None:
    """Remove a private Wan snapshot with normalized path-free failures."""

    _remove_snapshot(root)


@contextmanager
def wan_snapshot_cleanup_scope(root: Path) -> Iterator[None]:
    """Always clean a private snapshot while preserving BaseException semantics."""

    primary: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary = error
    try:
        _remove_snapshot(root)
    except BaseException as cleanup:
        if primary is not None:
            raise BaseExceptionGroup(
                "wan_snapshot_primary_and_cleanup_failed", (primary, cleanup)
            ) from None
        raise
    if primary is not None:
        raise primary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
