from __future__ import annotations

import ctypes
import errno
import os
import sqlite3
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import video_lab.local_authority as local_authority
from video_lab.local_authority import (
    RetainedDirectoryAuthority,
    RetainedSqliteAuthority,
)


def _create_database(path: Path, *, marker: str) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE authority_probe(marker TEXT NOT NULL)")
        connection.execute("INSERT INTO authority_probe VALUES (?)", (marker,))
    return path


def _swap_path_to_attacker(path: Path, attacker: Path) -> Path:
    retained = path.with_name(f"retained-{path.name}")
    path.rename(retained)
    attacker.rename(path)
    return retained


def test_retained_directory_uses_descriptor_relative_operations(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    authority = RetainedDirectoryAuthority.capture(root)
    try:
        descriptor = authority.open_exclusive("run.sqlite", mode=0o600)
        try:
            assert os.write(descriptor, b"retained") == len(b"retained")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        assert authority.contains("run.sqlite")
        authority.unlink("run.sqlite")
        assert not authority.contains("run.sqlite")
    finally:
        authority.close()


def test_retained_directory_rejects_path_replacement(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    authority = RetainedDirectoryAuthority.capture(root)
    detached = tmp_path / "detached-runs"
    root.rename(detached)
    root.mkdir()
    try:
        with pytest.raises(ValueError, match="^wan_canary_directory_authority_changed$"):
            authority.reverify()
    finally:
        authority.close()


def test_retained_directory_rejects_unsafe_child_names(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    authority = RetainedDirectoryAuthority.capture(root)
    try:
        for name in ("", ".", "..", "nested/run.sqlite", "/run.sqlite"):
            with pytest.raises(
                ValueError, match="^wan_canary_directory_authority_changed$"
            ):
                authority.contains(name)
    finally:
        authority.close()


def test_retained_directory_close_exhausts_and_invalidates_descriptors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    authority = RetainedDirectoryAuthority.capture(root)
    descriptors = authority.descriptors

    authority.close()
    authority.close()

    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with pytest.raises(ValueError, match="^wan_canary_directory_authority_changed$"):
        authority.reverify()


def test_retained_authorities_are_frozen(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    database = _create_database(tmp_path / "source.sqlite", marker="expected")
    directory = RetainedDirectoryAuthority.capture(root)
    sqlite = RetainedSqliteAuthority.capture(database)
    try:
        with pytest.raises(FrozenInstanceError):
            directory.path = tmp_path  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            sqlite.path = tmp_path  # type: ignore[misc]
    finally:
        sqlite.close()
        directory.close()


def test_retained_sqlite_rejects_attacker_shared_pager_during_open(
    tmp_path: Path,
) -> None:
    expected = _create_database(tmp_path / "expected.sqlite", marker="expected")
    attacker = _create_database(tmp_path / "attacker.sqlite", marker="attacker")
    authority = RetainedSqliteAuthority.capture(expected)
    detached = _swap_path_to_attacker(expected, attacker)
    shared = sqlite3.connect(f"file:{expected}?mode=ro&cache=shared", uri=True)
    try:
        assert shared.execute("SELECT marker FROM authority_probe").fetchone() == (
            "attacker",
        )
        with pytest.raises(ValueError, match="^wan_canary_database_authority_changed$"):
            with authority.connect_readonly() as connection:
                connection.execute("SELECT marker FROM authority_probe").fetchone()
    finally:
        shared.close()
        authority.close()
        expected.unlink()
        detached.rename(expected)


def test_retained_sqlite_read_uses_private_pager_from_retained_bytes(
    tmp_path: Path,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    authority = RetainedSqliteAuthority.capture(source)
    try:
        with authority.connect_readonly() as connection:
            assert connection.execute(
                "SELECT marker FROM authority_probe"
            ).fetchone()[0] == "expected"
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("DELETE FROM authority_probe")
    finally:
        authority.close()


def test_retained_sqlite_reads_closed_wal_database_from_private_pager(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    connection = sqlite3.connect(source)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE authority_probe(marker TEXT NOT NULL)")
        connection.execute("INSERT INTO authority_probe VALUES ('expected')")
        connection.commit()
    finally:
        connection.close()
    retained_bytes = source.read_bytes()
    authority = RetainedSqliteAuthority.capture(source)
    try:
        with authority.connect_readonly() as connection:
            assert connection.execute(
                "SELECT marker FROM authority_probe"
            ).fetchone()[0] == "expected"
        assert source.read_bytes() == retained_bytes
        assert os.pread(authority.descriptor, authority.size, 0) == retained_bytes
    finally:
        authority.close()


def test_retained_sqlite_rejects_uncheckpointed_live_wal_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    connection = sqlite3.connect(source)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE authority_probe(marker TEXT NOT NULL)")
        connection.execute("INSERT INTO authority_probe VALUES ('old')")
        connection.commit()
    finally:
        connection.close()

    authority = RetainedSqliteAuthority.capture(source)
    writer = sqlite3.connect(source)
    try:
        writer.execute("UPDATE authority_probe SET marker = 'new'")
        writer.commit()
        assert writer.execute("SELECT marker FROM authority_probe").fetchone() == (
            "new",
        )
        assert source.with_name(source.name + "-wal").is_file()

        with pytest.raises(ValueError, match="^wan_canary_database_authority_changed$"):
            with authority.connect_readonly() as connection:
                connection.execute("SELECT marker FROM authority_probe").fetchone()
    finally:
        writer.close()
        authority.close()


def test_retained_sqlite_clone_uses_retained_bytes_during_path_swap(
    tmp_path: Path,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    attacker = _create_database(tmp_path / "attacker.sqlite", marker="attacker")
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(tmp_path / "runs")
    detached = _swap_path_to_attacker(authority.path, attacker)
    try:
        with pytest.raises(ValueError, match="^wan_canary_database_authority_changed$"):
            authority.clone_to(destination, "run.sqlite")
        assert not destination.contains("run.sqlite")
    finally:
        destination.close()
        authority.close()
        source.unlink()
        detached.rename(source)


def test_retained_sqlite_clone_rejects_embedded_nul_without_creating_file(
    tmp_path: Path,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    try:
        with pytest.raises(
            ValueError, match="^wan_canary_directory_authority_changed$"
        ):
            authority.clone_to(destination, "run.sqlite\0attacker")
        assert tuple(runs.iterdir()) == ()
    finally:
        destination.close()
        authority.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin clonefile regression")
def test_darwin_clone_retains_created_inode_after_public_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    real_clone = local_authority._call_fclonefileat
    replacement = os.pread(authority.descriptor, authority.size, 0)
    created_inode: int | None = None
    replacement_inode: int | None = None
    created_descriptor: int | None = None

    def clone_then_replace(
        source_descriptor: int,
        destination_descriptor: int,
        name: str,
    ) -> int:
        nonlocal created_descriptor, created_inode, replacement_inode
        descriptor = real_clone(source_descriptor, destination_descriptor, name)
        created_descriptor = descriptor
        created_inode = os.stat(
            name,
            dir_fd=destination_descriptor,
            follow_symlinks=False,
        ).st_ino
        os.unlink(name, dir_fd=destination_descriptor)
        replacement_descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_descriptor,
        )
        try:
            assert os.write(replacement_descriptor, replacement) == len(replacement)
            replacement_inode = os.fstat(replacement_descriptor).st_ino
        finally:
            os.close(replacement_descriptor)
        return descriptor

    monkeypatch.setattr(local_authority, "_call_fclonefileat", clone_then_replace)
    try:
        with pytest.raises(OSError, match="database clone publication changed"):
            authority.clone_to(destination, "run.sqlite")
        assert created_inode is not None
        assert replacement_inode is not None
        assert created_descriptor is not None
        assert created_inode != replacement_inode
        assert (runs / "run.sqlite").read_bytes() == replacement
        assert (runs / "run.sqlite").stat().st_ino == replacement_inode
        with pytest.raises(OSError):
            os.fstat(created_descriptor)
    finally:
        destination.close()
        authority.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin clonefile regression")
def test_darwin_clone_opens_mode_0400_publication_readonly(tmp_path: Path) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    source.chmod(0o400)
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    clone: RetainedSqliteAuthority | None = None
    try:
        clone = authority.clone_to(destination, "run.sqlite")
        assert os.stat(runs / "run.sqlite").st_mode & 0o777 == 0o400
        assert os.pread(clone.descriptor, clone.size, 0) == os.pread(
            authority.descriptor, authority.size, 0
        )
    finally:
        if clone is not None:
            clone.close()
        destination.close()
        authority.close()


def test_linkat_native_boundary_preserves_abi_and_errno(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinkat:
        argtypes: object = None
        restype: object = None
        call: tuple[int, bytes, int, bytes, int] | None = None

        def __call__(
            self,
            source_descriptor: int,
            source_name: bytes,
            destination_descriptor: int,
            destination_name: bytes,
            flags: int,
        ) -> int:
            self.call = (
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
                flags,
            )
            ctypes.set_errno(errno.EPERM)
            return -1

    linkat = FakeLinkat()

    class FakeLibrary:
        pass

    library = FakeLibrary()
    library.linkat = linkat  # type: ignore[attr-defined]
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: library)

    with pytest.raises(OSError) as captured:
        local_authority._call_linkat(11, b"", 22, "run.sqlite", 0x1000)

    assert captured.value.errno == errno.EPERM
    assert linkat.argtypes == (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    assert linkat.restype is ctypes.c_int
    assert linkat.call == (11, b"", 22, b"run.sqlite", 0x1000)


def test_linux_anonymous_link_falls_back_to_unprivileged_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bytes, int, str, int]] = []

    def call_linkat(
        source_descriptor: int,
        source_name: bytes,
        destination_descriptor: int,
        destination_name: str,
        flags: int,
    ) -> None:
        calls.append(
            (
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
                flags,
            )
        )
        if len(calls) == 1:
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT))

    monkeypatch.setattr(local_authority, "_call_linkat", call_linkat)

    local_authority._link_anonymous_file(11, 22, "run.sqlite")

    assert calls == [
        (11, b"", 22, "run.sqlite", 0x1000),
        (-100, b"/proc/self/fd/11", 22, "run.sqlite", 0x400),
    ]


def test_linux_anonymous_link_fails_closed_without_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bytes, int, str, int]] = []

    def call_linkat(
        source_descriptor: int,
        source_name: bytes,
        destination_descriptor: int,
        destination_name: str,
        flags: int,
    ) -> None:
        calls.append(
            (
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
                flags,
            )
        )
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT))

    monkeypatch.setattr(local_authority, "_call_linkat", call_linkat)

    with pytest.raises(OSError) as captured:
        local_authority._link_anonymous_file(11, 22, "run.sqlite")

    assert captured.value.errno == errno.ENOENT
    assert calls == [
        (11, b"", 22, "run.sqlite", 0x1000),
        (-100, b"/proc/self/fd/11", 22, "run.sqlite", 0x400),
    ]


def test_retained_sqlite_clone_is_exclusive_retained_and_validated(
    tmp_path: Path,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    clone = authority.clone_to(destination, "run.sqlite")
    try:
        assert clone.path == runs / "run.sqlite"
        assert clone.descriptor != authority.descriptor
        assert os.pread(clone.descriptor, clone.size, 0) == os.pread(
            authority.descriptor, authority.size, 0
        )
        with clone.connect_readonly() as connection:
            assert connection.execute(
                "SELECT marker FROM authority_probe"
            ).fetchone()[0] == "expected"
        with pytest.raises(FileExistsError):
            authority.clone_to(destination, "run.sqlite")
    finally:
        clone.close()
        destination.close()
        authority.close()


def test_retained_sqlite_clone_removes_invalid_database(tmp_path: Path) -> None:
    source = tmp_path / "invalid.sqlite"
    source.write_bytes(b"not a sqlite database")
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            authority.clone_to(destination, "run.sqlite")
        assert not destination.contains("run.sqlite")
    finally:
        destination.close()
        authority.close()


def test_retained_sqlite_clone_removes_failed_publication_after_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    runs = tmp_path / "runs"
    detached = tmp_path / "detached-runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)

    def fail_after_root_swap(_connection: sqlite3.Connection) -> None:
        runs.rename(detached)
        runs.mkdir()
        raise sqlite3.DatabaseError("forced clone validation failure")

    monkeypatch.setattr(
        local_authority,
        "_validate_sqlite_integrity",
        fail_after_root_swap,
    )
    try:
        with pytest.raises(
            sqlite3.DatabaseError, match="^forced clone validation failure$"
        ):
            authority.clone_to(destination, "run.sqlite")
        assert not (detached / "run.sqlite").exists()
        assert not (runs / "run.sqlite").exists()
    finally:
        destination.close()
        authority.close()


def test_retained_sqlite_clone_never_deletes_replaced_staging_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    real_unlink = os.unlink
    replacement = b"attacker staging replacement"
    injected_name: str | None = None

    def fail_validation(_connection: sqlite3.Connection) -> None:
        raise sqlite3.DatabaseError("forced clone validation failure")

    def replace_at_cleanup(
        path: str | bytes,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal injected_name
        if dir_fd != destination.descriptor or injected_name is not None:
            real_unlink(path, dir_fd=dir_fd)
            return
        assert isinstance(path, str)
        injected_name = path
        real_unlink(path, dir_fd=dir_fd)
        replacement_descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination.descriptor,
        )
        try:
            assert os.write(replacement_descriptor, replacement) == len(replacement)
        finally:
            os.close(replacement_descriptor)
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(local_authority, "_validate_sqlite_integrity", fail_validation)
    monkeypatch.setattr(os, "unlink", replace_at_cleanup)
    try:
        with pytest.raises(
            sqlite3.DatabaseError, match="^forced clone validation failure$"
        ):
            authority.clone_to(destination, "run.sqlite")
        if injected_name is None:
            assert tuple(runs.iterdir()) == ()
        else:
            replacement_path = runs / injected_name
            assert replacement_path.read_bytes() == replacement
            assert tuple(path.name for path in runs.iterdir()) == (injected_name,)
    finally:
        destination.close()
        authority.close()


def test_retained_sqlite_clone_survives_destination_close_and_descriptor_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    runs = tmp_path / "runs"
    attacker = tmp_path / "attacker"
    runs.mkdir()
    attacker.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    destination_descriptor = destination.descriptor
    real_validate = local_authority._validate_sqlite_integrity
    reused_descriptor = -1
    clone: RetainedSqliteAuthority | None = None

    def close_and_reuse_destination(connection: sqlite3.Connection) -> None:
        nonlocal reused_descriptor
        real_validate(connection)
        destination.close()
        attacker_descriptor = os.open(
            attacker,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            reused_descriptor = os.dup2(attacker_descriptor, destination_descriptor)
        finally:
            if attacker_descriptor != reused_descriptor:
                os.close(attacker_descriptor)

    monkeypatch.setattr(
        local_authority,
        "_validate_sqlite_integrity",
        close_and_reuse_destination,
    )
    try:
        clone = authority.clone_to(destination, "run.sqlite")
        assert reused_descriptor == destination_descriptor
        assert tuple(attacker.iterdir()) == ()
        with clone.connect_readonly() as connection:
            assert connection.execute(
                "SELECT marker FROM authority_probe"
            ).fetchone()[0] == "expected"
    finally:
        if clone is not None:
            clone.close()
        if reused_descriptor >= 0:
            os.close(reused_descriptor)
        destination.close()
        authority.close()


def test_retained_sqlite_clone_has_no_visible_post_create_identity_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _create_database(tmp_path / "source.sqlite", marker="expected")
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    real_open = os.open
    created_descriptor: int | None = None
    clone: RetainedSqliteAuthority | None = None

    def track_created_descriptor(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT and dir_fd == destination.descriptor:
            created_descriptor = descriptor
        return descriptor

    monkeypatch.setattr(os, "open", track_created_descriptor)
    try:
        clone = authority.clone_to(destination, "run.sqlite")
        assert created_descriptor is None
        assert tuple(path.name for path in runs.iterdir()) == ("run.sqlite",)
        with clone.connect_readonly() as connection:
            assert connection.execute(
                "SELECT marker FROM authority_probe"
            ).fetchone()[0] == "expected"
    finally:
        if clone is not None:
            clone.close()
        destination.close()
        authority.close()


def test_retained_sqlite_clone_validates_schema_with_application_functions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    connection = sqlite3.connect(source)
    try:
        connection.create_function("application_check", 1, lambda _value: 1)
        connection.execute(
            "CREATE TABLE authority_probe(marker TEXT CHECK(application_check(marker)))"
        )
        connection.execute("INSERT INTO authority_probe VALUES ('expected')")
        connection.commit()
    finally:
        connection.close()
    runs = tmp_path / "runs"
    runs.mkdir()
    authority = RetainedSqliteAuthority.capture(source)
    destination = RetainedDirectoryAuthority.capture(runs)
    clone: RetainedSqliteAuthority | None = None
    try:
        clone = authority.clone_to(destination, "run.sqlite")
        with clone.connect_readonly() as readonly:
            assert readonly.execute(
                "SELECT marker FROM authority_probe"
            ).fetchone()[0] == "expected"
    finally:
        if clone is not None:
            clone.close()
        destination.close()
        authority.close()
