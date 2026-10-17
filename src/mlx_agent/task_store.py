"""Private, transactional task history and immutable local artifacts."""

import hashlib
import json
import os
import sqlite3
import stat
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .task_contracts import (MAX_REQUEST_BYTES, RUN_SCHEMA, TERMINAL, TaskError, encoded,
                             task_id, validate_record, validate_request)


def default_root():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/mlx-agent/tasks"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "mlx-agent/tasks"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _private(path, directory=True):
    for item in (path, *path.parents):
        if item.is_symlink():
            raise TaskError("unsafe_task_storage", "Task storage cannot traverse symbolic links.")
    info = path.stat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise TaskError("unsafe_task_storage", "Task storage must be private and owned by the current user.")


class TaskStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_root()
        if not self.root.is_absolute():
            raise TaskError("invalid_state_dir", "Task state directory must be absolute.")
        self.database = self.root / "tasks.sqlite3"

    @contextmanager
    def _connection(self, write=False):
        connection = None
        try:
            if not self.root.exists() and not self.root.is_symlink() and not write:
                yield None
                return
            if write:
                # Check parents before creating anything through a caller-supplied path.
                for item in (self.root, *self.root.parents):
                    if item.is_symlink():
                        raise TaskError("unsafe_task_storage", "Task storage cannot traverse symbolic links.")
                self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
            _private(self.root)
            if not self.database.exists() and not self.database.is_symlink():
                if not write:
                    yield None
                    return
                try:
                    descriptor = os.open(self.database, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
                    os.close(descriptor)
                except FileExistsError:
                    pass
            _private(self.database, directory=False)
            uri = self.database.as_uri() + ("?mode=rw" if write else "?mode=ro")
            connection = sqlite3.connect(uri, uri=True, timeout=5)
            if write:
                connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0 and write:
                tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if tables:
                    raise TaskError("invalid_task_store", "An unversioned database cannot be adopted as task storage.")
                connection.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, record TEXT NOT NULL)")
                connection.execute("PRAGMA user_version = 1")
                # Initialization is independent of a later failed task write.
                connection.commit()
                connection.execute("BEGIN IMMEDIATE")
            elif version != 1:
                raise TaskError("unsupported_task_store", "Unsupported task database version.", "Preserve the database and use a compatible engine.")
            yield connection
            if write:
                connection.commit()
        except TaskError:
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            raise TaskError("task_storage_failed", "Cannot access task storage: {0}".format(error), "Check the task directory, free space and permissions; preserve existing records.") from error
        finally:
            if connection is not None:
                connection.close()

    def _load(self, connection, identifier):
        task_id(identifier)
        row = connection.execute("SELECT record FROM tasks WHERE id = ?", (identifier,)).fetchone() if connection else None
        if row is None:
            raise TaskError("task_not_found", "No task with that ID.", "Use mlx-agent task list with the same --state-dir.")
        try:
            value = validate_record(json.loads(row[0]))
            if value["id"] != identifier:
                raise TaskError("invalid_task_record", "Task row identity does not match its record.")
            return value
        except (ValueError, TypeError) as error:
            raise TaskError("invalid_task_record", "The stored task record is not readable JSON.") from error

    def _save(self, connection, record):
        validate_record(record)
        connection.execute("UPDATE tasks SET record = ? WHERE id = ?", (encoded(record).decode(), record["id"]))

    def get(self, identifier):
        with self._connection() as connection:
            return self._load(connection, identifier)

    def list(self, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise TaskError("invalid_limit", "Task history limit must be between 1 and 100.")
        with self._connection() as connection:
            if connection is None:
                return []
            ids = connection.execute("SELECT id FROM tasks ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
            return [self._load(connection, row[0]) for row in ids]

    def artifact_path(self, identifier, name):
        task_id(identifier)
        if not isinstance(name, str) or not name or name.startswith(".") or Path(name).name != name or "/" in name or "\\" in name:
            raise TaskError("invalid_artifact", "Artifact names must be plain file names.")
        try:
            for directory in (self.root, self.root / "artifacts", self.root / "artifacts" / identifier):
                _private(directory)
            path = self.root / "artifacts" / identifier / name
            if path.is_symlink():
                raise TaskError("unsafe_task_storage", "Task artifacts cannot be symbolic links.")
            if path.exists():
                _private(path, directory=False)
            return path
        except OSError as error:
            raise TaskError("task_storage_failed", "Cannot access the task artifact directory.") from error

    def _write_artifact(self, identifier, name, value):
        target = self.artifact_path(identifier, name)
        if target.exists():
            raise TaskError("artifact_exists", "Task evidence cannot be overwritten.")
        raw = encoded(value)
        temporary = target.parent / (".write-" + str(uuid4()))
        try:
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise TaskError("task_storage_failed", "Cannot save task evidence: {0}".format(error)) from error
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": str(target), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def create(self, request, identity):
        request = validate_request(request)
        identifier = str(uuid4())
        with self._connection(write=True) as connection:
            for directory in (self.root / "artifacts", self.root / "artifacts" / identifier):
                directory.mkdir(mode=0o700, exist_ok=True)
                _private(directory)
            reference = self._write_artifact(identifier, "request.json", request)
            record = {"schema": RUN_SCHEMA, "id": identifier, "kind": request["kind"], "state": "starting",
                      "model_path": request["model_path"], "created_at": _now(), "started_at": None, "completed_at": None,
                      "cancel_requested": False, "identity": identity, "request": reference, "result": None, "error": None,
                      "supervisor": None, "execution": None, "execution_unresolved": False}
            validate_record(record)
            connection.execute("INSERT INTO tasks VALUES (?, ?, ?)", (identifier, record["created_at"], encoded(record).decode()))
            return record

    def transition(self, identifier, expected, changes):
        mutable = {"state", "started_at", "completed_at", "result", "error", "supervisor", "execution", "execution_unresolved"}
        if not isinstance(changes, dict) or set(changes) - mutable:
            raise TaskError("invalid_transition", "Task identity and accepted inputs are immutable.")
        with self._connection(write=True) as connection:
            record = self._load(connection, identifier)
            if record["state"] in TERMINAL or record["state"] not in expected:
                raise TaskError("task_state_changed", "The task state has already changed.", "Read its current status.")
            if "supervisor" in changes and record["supervisor"] is not None:
                raise TaskError("task_state_changed", "The task already has a supervisor.")
            next_state = changes.get("state", record["state"])
            if next_state == "succeeded" and record["cancel_requested"]:
                raise TaskError("task_cancel_requested", "Cancellation was requested before result publication.")
            if record["state"] == "running" and next_state == "starting":
                raise TaskError("invalid_transition", "Running tasks cannot restart under the same identity.")
            record.update(changes)
            if next_state == "running" and record["started_at"] is None:
                record["started_at"] = _now()
            if next_state in TERMINAL:
                record["completed_at"] = _now()
            self._save(connection, record)
            return record

    def request_cancel(self, identifier):
        with self._connection(write=True) as connection:
            record = self._load(connection, identifier)
            if record["state"] not in TERMINAL:
                record["cancel_requested"] = True
                self._save(connection, record)
            return record

    def read_request(self, identifier):
        record = self.get(identifier)
        try:
            path = self.artifact_path(identifier, "request.json")
            if path.stat().st_size > MAX_REQUEST_BYTES:
                raise TaskError("request_changed", "The saved request exceeds its size limit.")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["request"].get("sha256"):
                raise TaskError("request_changed", "The accepted request snapshot changed.")
            return validate_request(json.loads(raw))
        except (OSError, ValueError) as error:
            raise TaskError("request_changed", "The accepted request snapshot is unavailable.") from error

    def write_result(self, identifier, value):
        with self._connection(write=True) as connection:
            record = self._load(connection, identifier)
            if record["state"] in TERMINAL:
                raise TaskError("task_state_changed", "A terminal task cannot acquire new result evidence.")
            return self._write_artifact(identifier, "result.json", value)
