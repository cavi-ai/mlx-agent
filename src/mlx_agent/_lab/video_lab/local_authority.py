"""Retained, pathname-aware authorities for local filesystems and SQLite."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import sqlite3
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

MAXIMUM_SQLITE_BYTES = 1024 * 1024 * 1024

_DIRECTORY_AUTHORITY_CHANGED = "wan_canary_directory_authority_changed"
_DATABASE_AUTHORITY_CHANGED = "wan_canary_database_authority_changed"

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_SAFETY_FLAGS = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_UNKNOWN_SQLITE_FUNCTION = re.compile(r"^unknown function: ([A-Za-z_][A-Za-z0-9_]*)\(\)$")
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_LINUX_AT_FDCWD = -100
_LINUX_AT_SYMLINK_FOLLOW = 0x400
_LINUX_AT_EMPTY_PATH = 0x1000
_LINKAT_CAPABILITY_ERRNOS = frozenset((errno.EACCES, errno.ENOENT, errno.EPERM))


@dataclass(slots=True)
class _AuthorityState:
    lock: RLock = field(default_factory=RLock)
    closed: bool = False


def _safe_child_name(name: str, *, error_code: str) -> str:
    if (
        type(name) is not str
        or name in {"", ".", ".."}
        or "\x00" in name
        or Path(name).name != name
        or os.path.isabs(name)
    ):
        raise ValueError(error_code)
    return name


def _directory_identity(descriptor: int) -> tuple[int, int]:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise OSError(errno.ENOTDIR, "retained authority is not a directory")
    return observed.st_dev, observed.st_ino


def _open_directory_chain(path: Path) -> list[int]:
    descriptors: list[int] = []
    try:
        current = os.open(path.anchor, _DIRECTORY_FLAGS)
        descriptors.append(current)
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise OSError(errno.EINVAL, "invalid directory authority")
            current = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            descriptors.append(current)
        return descriptors
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _create_directory_leaf(path: Path) -> None:
    parents = _open_directory_chain(path.parent)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parents[-1])
        os.fsync(parents[-1])
    finally:
        for descriptor in reversed(parents):
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class RetainedDirectoryAuthority:
    path: Path
    descriptor: int
    descriptors: tuple[int, ...]
    identities: tuple[tuple[int, int], ...]
    error_code: str = _DIRECTORY_AUTHORITY_CHANGED
    _state: _AuthorityState = field(
        default_factory=_AuthorityState, compare=False, repr=False
    )

    @classmethod
    def capture(
        cls,
        path: Path,
        *,
        error_code: str = _DIRECTORY_AUTHORITY_CHANGED,
    ) -> RetainedDirectoryAuthority:
        if not isinstance(path, Path) or not path.is_absolute() or not error_code:
            raise ValueError(error_code or _DIRECTORY_AUTHORITY_CHANGED)
        descriptors: list[int] = []
        try:
            try:
                descriptors = _open_directory_chain(path)
            except FileNotFoundError:
                _create_directory_leaf(path)
                descriptors = _open_directory_chain(path)
            identities = tuple(_directory_identity(item) for item in descriptors)
            authority = cls(
                path=path,
                descriptor=descriptors[-1],
                descriptors=tuple(descriptors),
                identities=identities,
                error_code=error_code,
            )
            authority.reverify()
            return authority
        except (OSError, ValueError) as error:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if isinstance(error, ValueError) and str(error) == error_code:
                raise
            raise ValueError(error_code) from None
        except BaseException:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise

    def _reverify_locked(self) -> None:
        if self._state.closed:
            raise ValueError(self.error_code)
        try:
            retained = tuple(_directory_identity(item) for item in self.descriptors)
            current = _open_directory_chain(self.path)
            try:
                observed = tuple(_directory_identity(item) for item in current)
            finally:
                for descriptor in reversed(current):
                    os.close(descriptor)
            if retained != self.identities or observed != self.identities:
                raise OSError(errno.ESTALE, "directory authority changed")
        except OSError:
            raise ValueError(self.error_code) from None

    def reverify(self) -> None:
        with self._state.lock:
            self._reverify_locked()

    def open_file(self, name: str, flags: int, *, mode: int = 0o600) -> int:
        name = _safe_child_name(name, error_code=self.error_code)
        with self._state.lock:
            self._reverify_locked()
            descriptor = os.open(
                name,
                flags | _FILE_SAFETY_FLAGS,
                mode,
                dir_fd=self.descriptor,
            )
            try:
                self._reverify_locked()
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor

    def open_exclusive(self, name: str, *, mode: int = 0o600) -> int:
        return self.open_file(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, mode=mode)

    def contains(self, name: str) -> bool:
        name = _safe_child_name(name, error_code=self.error_code)
        with self._state.lock:
            self._reverify_locked()
            try:
                os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            except FileNotFoundError:
                present = False
            else:
                present = True
            self._reverify_locked()
            return present

    def unlink(self, name: str, *, missing_ok: bool = False) -> None:
        name = _safe_child_name(name, error_code=self.error_code)
        with self._state.lock:
            self._reverify_locked()
            try:
                os.unlink(name, dir_fd=self.descriptor)
            except FileNotFoundError:
                if not missing_ok:
                    raise
            os.fsync(self.descriptor)
            self._reverify_locked()

    def close(self) -> None:
        with self._state.lock:
            if self._state.closed:
                return
            self._state.closed = True
            first_error: OSError | None = None
            for descriptor in reversed(self.descriptors):
                try:
                    os.close(descriptor)
                except OSError as error:
                    if error.errno != errno.EBADF and first_error is None:
                        first_error = error
            if first_error is not None:
                raise ValueError(self.error_code) from None


def _file_identity(observed: os.stat_result) -> tuple[int, int, int, int]:
    if not stat.S_ISREG(observed.st_mode):
        raise OSError(errno.EINVAL, "retained authority is not a regular file")
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    )


def _pread_bounded(descriptor: int, *, maximum_bytes: int) -> bytes:
    before = os.fstat(descriptor)
    identity = _file_identity(before)
    size = identity[2]
    if size < 0 or size > maximum_bytes:
        raise ValueError(_DATABASE_AUTHORITY_CHANGED)
    payload = bytearray()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise ValueError(_DATABASE_AUTHORITY_CHANGED)
        payload.extend(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise ValueError(_DATABASE_AUTHORITY_CHANGED)
    if _file_identity(os.fstat(descriptor)) != identity:
        raise ValueError(_DATABASE_AUTHORITY_CHANGED)
    return bytes(payload)


def _assert_no_sqlite_sidecars(directory_descriptor: int, name: str) -> None:
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        try:
            os.stat(
                name + suffix,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        raise OSError(errno.EBUSY, "SQLite sidecar state is active")


def _call_fclonefileat(
    source_descriptor: int,
    destination_descriptor: int,
    name: str,
) -> int:
    """Atomically clone, then immediately retain the published inode readonly.

    Darwin's public clone primitive returns status rather than a descriptor, so
    clone creation and descriptor capture are necessarily separate syscalls.
    """

    library = ctypes.CDLL(None, use_errno=True)
    clone = library.fclonefileat
    clone.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint32,
    )
    clone.restype = ctypes.c_int
    result = clone(
        source_descriptor,
        destination_descriptor,
        os.fsencode(name),
        0,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        raise OSError(observed_errno, os.strerror(observed_errno))
    return os.open(
        name,
        os.O_RDONLY | _FILE_SAFETY_FLAGS,
        dir_fd=destination_descriptor,
    )


def _call_linkat(
    source_descriptor: int,
    source_name: bytes,
    destination_descriptor: int,
    destination_name: str,
    flags: int,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    link = library.linkat
    link.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    link.restype = ctypes.c_int
    result = link(
        source_descriptor,
        source_name,
        destination_descriptor,
        os.fsencode(destination_name),
        flags,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        raise OSError(observed_errno, os.strerror(observed_errno))


def _link_anonymous_file(
    descriptor: int,
    destination_descriptor: int,
    name: str,
) -> None:
    """Link an ``O_TMPFILE`` descriptor without requiring a Linux capability."""

    try:
        _call_linkat(
            descriptor,
            b"",
            destination_descriptor,
            name,
            _LINUX_AT_EMPTY_PATH,
        )
        return
    except OSError as error:
        if error.errno not in _LINKAT_CAPABILITY_ERRNOS:
            raise
    _call_linkat(
        _LINUX_AT_FDCWD,
        os.fsencode(f"/proc/self/fd/{descriptor}"),
        destination_descriptor,
        name,
        _LINUX_AT_SYMLINK_FOLLOW,
    )


def _publish_atomic_clone(
    *,
    source_descriptor: int,
    destination_descriptor: int,
    name: str,
    payload: bytes,
) -> int:
    """Publish exact bytes without any cleanup-requiring visible stage."""

    if sys.platform == "darwin":
        return _call_fclonefileat(
            source_descriptor,
            destination_descriptor,
            name,
        )
    if sys.platform.startswith("linux") and hasattr(os, "O_TMPFILE"):
        descriptor = os.open(
            ".",
            os.O_RDWR | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=destination_descriptor,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError(errno.EIO, "database clone write failed")
                remaining = remaining[written:]
            os.fsync(descriptor)
            if _pread_bounded(descriptor, maximum_bytes=MAXIMUM_SQLITE_BYTES) != payload:
                raise OSError(errno.EIO, "database clone verification failed")
            _link_anonymous_file(descriptor, destination_descriptor, name)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise
    raise OSError(errno.ENOTSUP, "atomic retained clone publication is unavailable")


def _private_pager_payload(payload: bytes) -> bytes:
    """Make a closed WAL image readable by SQLite's anonymous private pager.

    SQLite persists WAL mode in header bytes 18 and 19 even after the final
    connection checkpoints and removes the WAL. ``deserialize`` cannot attach
    a WAL file to ``:memory:``, so normalize only those pager-mode bytes in the
    ephemeral image. The authority verifies that no WAL, shared-memory, or
    rollback-journal sidecar exists before this normalization. Authority
    digests and clone writes continue to use the exact retained payload.
    """

    if len(payload) >= 20 and payload[:16] == b"SQLite format 3\x00":
        read_version, write_version = payload[18], payload[19]
        if (read_version, write_version) == (2, 2):
            private = bytearray(payload)
            private[18:20] = b"\x01\x01"
            return bytes(private)
    return payload


def _validate_sqlite_integrity(connection: sqlite3.Connection) -> None:
    registered: set[str] = set()
    while True:
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.OperationalError as error:
            match = _UNKNOWN_SQLITE_FUNCTION.fullmatch(str(error))
            if match is None or match.group(1) in registered or len(registered) >= 128:
                raise
            name = match.group(1)
            registered.add(name)
            connection.create_function(name, -1, lambda *_arguments: 1, deterministic=True)
            continue
        if integrity is None or integrity[0] != "ok":
            raise sqlite3.DatabaseError("retained SQLite clone failed integrity check")
        return


@dataclass(frozen=True, slots=True)
class RetainedSqliteAuthority:
    path: Path
    descriptor: int
    parent_descriptor: int
    parent_device: int
    parent_inode: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    digest: str
    _state: _AuthorityState = field(
        default_factory=_AuthorityState, compare=False, repr=False
    )

    @classmethod
    def capture(cls, path: Path) -> RetainedSqliteAuthority:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(_DATABASE_AUTHORITY_CHANGED)
        descriptor = -1
        parent_descriptor = -1
        try:
            parent_descriptors = _open_directory_chain(path.parent)
            parent_descriptor = parent_descriptors.pop()
            for retained_parent in reversed(parent_descriptors):
                os.close(retained_parent)
            descriptor = os.open(
                path.name,
                os.O_RDONLY | _FILE_SAFETY_FLAGS,
                dir_fd=parent_descriptor,
            )
            _assert_no_sqlite_sidecars(parent_descriptor, path.name)
            payload = _pread_bounded(
                descriptor, maximum_bytes=MAXIMUM_SQLITE_BYTES
            )
            _assert_no_sqlite_sidecars(parent_descriptor, path.name)
            observed = os.fstat(descriptor)
            parent_observed = os.fstat(parent_descriptor)
            authority = cls(
                path=path,
                descriptor=descriptor,
                parent_descriptor=parent_descriptor,
                parent_device=parent_observed.st_dev,
                parent_inode=parent_observed.st_ino,
                device=observed.st_dev,
                inode=observed.st_ino,
                size=observed.st_size,
                mtime_ns=observed.st_mtime_ns,
                digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            )
            authority.reverify()
            return authority
        except (OSError, ValueError) as error:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if parent_descriptor >= 0:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass
            if isinstance(error, ValueError) and str(error) == _DATABASE_AUTHORITY_CHANGED:
                raise
            raise ValueError(_DATABASE_AUTHORITY_CHANGED) from None
        except BaseException:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if parent_descriptor >= 0:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass
            raise

    def _expected_identity(self) -> tuple[int, int, int, int]:
        return self.device, self.inode, self.size, self.mtime_ns

    def _verified_payload_locked(self) -> bytes:
        if self._state.closed:
            raise ValueError(_DATABASE_AUTHORITY_CHANGED)
        try:
            retained = _file_identity(os.fstat(self.descriptor))
            retained_parent = _directory_identity(self.parent_descriptor)
            current_parents = _open_directory_chain(self.path.parent)
            try:
                current_parent = _directory_identity(current_parents[-1])
            finally:
                for current_descriptor in reversed(current_parents):
                    os.close(current_descriptor)
            expected_parent = self.parent_device, self.parent_inode
            current = _file_identity(
                os.stat(
                    self.path.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            )
            if retained != self._expected_identity() or current != self._expected_identity():
                raise OSError(errno.ESTALE, "database authority changed")
            if retained_parent != expected_parent or current_parent != expected_parent:
                raise OSError(errno.ESTALE, "database parent authority changed")
            _assert_no_sqlite_sidecars(self.parent_descriptor, self.path.name)
            payload = _pread_bounded(
                self.descriptor, maximum_bytes=MAXIMUM_SQLITE_BYTES
            )
            if "sha256:" + hashlib.sha256(payload).hexdigest() != self.digest:
                raise OSError(errno.ESTALE, "database authority changed")
            if _file_identity(os.fstat(self.descriptor)) != self._expected_identity():
                raise OSError(errno.ESTALE, "database authority changed")
            _assert_no_sqlite_sidecars(self.parent_descriptor, self.path.name)
            return payload
        except (OSError, ValueError):
            raise ValueError(_DATABASE_AUTHORITY_CHANGED) from None

    def reverify(self) -> None:
        with self._state.lock:
            self._verified_payload_locked()

    @contextmanager
    def connect_readonly(self) -> Iterator[sqlite3.Connection]:
        with self._state.lock:
            payload = self._verified_payload_locked()
            connection = sqlite3.connect(":memory:", isolation_level=None)
            try:
                connection.deserialize(_private_pager_payload(payload))
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only = ON")
                self._verified_payload_locked()
                yield connection
                self._verified_payload_locked()
            finally:
                connection.close()

    def clone_to(
        self, destination: RetainedDirectoryAuthority, name: str
    ) -> RetainedSqliteAuthority:
        """Clone into an app-owned, mutation-exclusive destination namespace.

        Darwin exact publication relies on that ownership contract: public Darwin
        APIs cannot defend arbitrary, uncoordinated same-UID child-entry mutation
        between clone installation and retaining the published descriptor.
        """

        name = _safe_child_name(name, error_code=destination.error_code)
        source_descriptor = -1
        destination_descriptor = -1
        clone_descriptor = -1
        with self._state.lock:
            payload = self._verified_payload_locked()
            source_descriptor = os.dup(self.descriptor)
        try:
            with destination._state.lock:
                destination._reverify_locked()
                destination_descriptor = os.dup(destination.descriptor)
            connection = sqlite3.connect(":memory:", isolation_level=None)
            try:
                connection.deserialize(_private_pager_payload(payload))
                connection.execute("PRAGMA query_only = ON")
                _validate_sqlite_integrity(connection)
            finally:
                connection.close()
            if _file_identity(os.fstat(source_descriptor)) != self._expected_identity():
                raise OSError(errno.ESTALE, "database authority changed")
            if _pread_bounded(
                source_descriptor,
                maximum_bytes=MAXIMUM_SQLITE_BYTES,
            ) != payload:
                raise OSError(errno.ESTALE, "database authority changed")
            if _directory_identity(destination_descriptor) != destination.identities[-1]:
                raise OSError(errno.ESTALE, "destination authority changed")
            current_directories = _open_directory_chain(destination.path)
            try:
                current_identities = tuple(
                    _directory_identity(item) for item in current_directories
                )
            finally:
                for current_descriptor in reversed(current_directories):
                    os.close(current_descriptor)
            if current_identities != destination.identities:
                raise OSError(errno.ESTALE, "destination authority changed")
            clone_descriptor = _publish_atomic_clone(
                source_descriptor=source_descriptor,
                destination_descriptor=destination_descriptor,
                name=name,
                payload=payload,
            )
            observed = os.fstat(clone_descriptor)
            if _pread_bounded(
                clone_descriptor,
                maximum_bytes=MAXIMUM_SQLITE_BYTES,
            ) != payload:
                raise OSError(errno.EIO, "database clone verification failed")
            current_clone = _file_identity(
                os.stat(
                    name,
                    dir_fd=destination_descriptor,
                    follow_symlinks=False,
                )
            )
            if current_clone != _file_identity(observed):
                raise OSError(errno.ESTALE, "database clone publication changed")
            parent_observed = os.fstat(destination_descriptor)
            clone = RetainedSqliteAuthority(
                path=destination.path / name,
                descriptor=clone_descriptor,
                parent_descriptor=destination_descriptor,
                parent_device=parent_observed.st_dev,
                parent_inode=parent_observed.st_ino,
                device=observed.st_dev,
                inode=observed.st_ino,
                size=observed.st_size,
                mtime_ns=observed.st_mtime_ns,
                digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            )
            clone_descriptor = -1
            destination_descriptor = -1
            return clone
        except BaseException:
            if clone_descriptor >= 0:
                os.close(clone_descriptor)
            if destination_descriptor >= 0:
                os.close(destination_descriptor)
            raise
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)

    def close(self) -> None:
        with self._state.lock:
            if self._state.closed:
                return
            self._state.closed = True
            try:
                os.close(self.descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise ValueError(_DATABASE_AUTHORITY_CHANGED) from None
            try:
                os.close(self.parent_descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise ValueError(_DATABASE_AUTHORITY_CHANGED) from None
