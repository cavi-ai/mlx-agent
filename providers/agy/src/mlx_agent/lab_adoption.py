"""Explicit, reversible adoption of a quiescent legacy experiment store."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import uuid

from video_lab.database import Database, SCHEMA_VERSION
from video_lab.local_authority import RetainedDirectoryAuthority, RetainedSqliteAuthority


def _paths(source_database, source_artifacts, destination):
    paths = tuple(Path(value) for value in (source_database, source_artifacts, destination))
    if any(not path.is_absolute() for path in paths):
        raise ValueError("Adoption requires absolute paths.")
    database, artifacts, target = paths
    if not database.is_file() or not artifacts.is_dir() or not target.parent.is_dir():
        raise ValueError("The source store and destination parent must exist.")
    if target.exists() or target.is_symlink():
        raise ValueError("Adoption requires a new destination; existing stores are never replaced.")
    if target == artifacts or artifacts in target.parents:
        raise ValueError("The destination must be outside the source artifact store.")
    return paths


def _preflight(database_path, artifact_root, destination, stack):
    source = RetainedSqliteAuthority.capture(database_path)
    stack.callback(source.close)
    artifact_authority = RetainedDirectoryAuthority.capture(artifact_root)
    stack.callback(artifact_authority.close)
    with source.connect_readonly() as connection:
        if connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] > SCHEMA_VERSION:
            raise ValueError("The source schema is newer than this engine.")
        active = connection.execute(
            "SELECT count(*) FROM jobs WHERE state NOT IN ('queued','completed','failed','cancelled')"
        ).fetchone()[0]
        active += connection.execute(
            "SELECT count(*) FROM run_attempts WHERE status='active'"
        ).fetchone()[0]
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "training_runs" in tables:
            active += connection.execute("SELECT count(*) FROM training_runs WHERE state='running'").fetchone()[0]
        if active:
            raise ValueError("Stop or recover active generation and training jobs before adoption.")
        records = tuple(dict(row) for row in connection.execute("SELECT * FROM artifacts ORDER BY id"))
        for record in records:
            with _artifact(record, artifact_root):
                pass
    source.reverify()
    artifact_authority.reverify()
    material = {"schema": "lab-adoption/1", "source_database": str(database_path),
        "source_database_digest": source.digest, "source_artifacts": str(artifact_root),
        "destination": str(destination), "artifact_count": len(records),
        "artifact_bytes": sum(record["byte_size"] for record in records),
        "artifacts": [{key: record[key] for key in ("id","relative_path","sha256","byte_size","media_type")}
                      for record in records]}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    plan = {key: value for key, value in material.items() if key != "artifacts"}
    plan["preview_hash"] = hashlib.sha256(encoded).hexdigest()
    return plan, source, artifact_authority, records



def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


@contextmanager
def _artifact(record, root):
    relative = Path(record["relative_path"])
    if relative.is_absolute() or not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("Artifact paths must remain inside their store.")
    authority = RetainedDirectoryAuthority.capture(root / relative.parent)
    try:
        descriptor = authority.open_file(relative.name, os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as stream:
            identity = _identity(os.fstat(stream.fileno()))
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or identity[2] != record["byte_size"]:
                raise ValueError("Artifact size or file type changed.")
            def verify():
                digest = hashlib.sha256()
                stream.seek(0)
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                observed = os.stat(relative.name, dir_fd=authority.descriptor, follow_symlinks=False)
                if digest.hexdigest() != record["sha256"] or _identity(observed) != identity or _identity(os.fstat(stream.fileno())) != identity:
                    raise ValueError("Artifact identity changed.")
                authority.reverify()
            verify()
            stream.seek(0)
            yield stream
            verify()
    finally:
        authority.close()


def preview(source_database, source_artifacts, destination):
    paths = _paths(source_database, source_artifacts, destination)
    with ExitStack() as stack:
        plan, *_ = _preflight(*paths, stack)
        return plan


def _publish(parent, staged_name, name):
    parent.reverify()
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        flag = 4  # RENAME_EXCL: never replace a destination.
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        flag = 1  # RENAME_NOREPLACE
    else:
        raise ValueError("Atomic exclusive directory publication is unavailable on this host.")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(parent.descriptor, os.fsencode(staged_name), parent.descriptor, os.fsencode(name), flag):
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    os.fsync(parent.descriptor)
    parent.reverify()


def adopt(source_database, source_artifacts, destination, preview_hash):
    paths = _paths(source_database, source_artifacts, destination)
    source_path, source_root, target = paths
    with ExitStack() as stack:
        plan, source, artifacts, records = _preflight(*paths, stack)
        if preview_hash != plan["preview_hash"]:
            raise ValueError("Adoption preview changed; inspect a fresh preview.")
        parent = RetainedDirectoryAuthority.capture(target.parent)
        stack.callback(parent.close)
        staged_name = ".lab-adoption-" + uuid.uuid4().hex
        staging = target.parent / staged_name
        os.mkdir(staged_name, 0o700, dir_fd=parent.descriptor)
        retained = RetainedDirectoryAuthority.capture(staging)
        identity = staging.stat().st_dev, staging.stat().st_ino
        published = False
        try:
            backup = source.clone_to(retained, "before-adoption.sqlite3")
            backup.close()
            current = source.clone_to(retained, "experiments.sqlite3")
            current.close()
            (staging / "artifacts").mkdir(mode=0o700)
            for record in records:
                    with _artifact(record, source_root) as handle:
                        output = staging / "artifacts" / record["relative_path"]
                        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        authority = RetainedDirectoryAuthority.capture(output.parent)
                        try:
                            digest = hashlib.sha256()
                            with os.fdopen(authority.open_exclusive(output.name), "wb") as stream:
                                while chunk := handle.read(1024 * 1024):
                                    stream.write(chunk)
                                    digest.update(chunk)
                                stream.flush()
                                os.fsync(stream.fileno())
                            authority.reverify()
                            if digest.hexdigest() != record["sha256"]:
                                raise ValueError("An artifact changed during adoption.")
                        finally:
                            authority.close()
            Database(staging / "experiments.sqlite3").migrate()
            with Database(staging / "experiments.sqlite3", read_only=True).connect() as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("The adopted database failed its integrity check.")
            source.reverify()
            artifacts.reverify()
            receipt = {**plan, "backup": str(target / "before-adoption.sqlite3"),
                       "store_path": str(target), "status": "adopted"}
            with os.fdopen(retained.open_exclusive("adoption.json"), "w") as stream:
                json.dump(receipt, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            retained.reverify()
            retained.close()
            _publish(parent, staged_name, target.name)
            published = True
            return receipt
        finally:
            retained.close()
            if not published and staging.exists():
                parent.reverify()
                info = staging.lstat()
                if not staging.is_symlink() and (info.st_dev, info.st_ino) == identity:
                    shutil.rmtree(staging)
