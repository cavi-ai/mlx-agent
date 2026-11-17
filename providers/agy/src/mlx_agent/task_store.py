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
from typing import Dict, Optional, Sequence

from . import task_kinds
from .task_contracts import (MAX_REQUEST_BYTES, MEDIA_RUN_SCHEMA, MEDIA_SUFFIXES, RUN_SCHEMA, TERMINAL, TaskError, encoded,
                             _artifact_reference, task_id, validate_record, validate_request)


def default_root():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/mlx-agent/tasks"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "mlx-agent/tasks"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_private(path, directory=True):
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
            ensure_private(self.root)
            if not self.database.exists() and not self.database.is_symlink():
                if not write:
                    yield None
                    return
                try:
                    descriptor = os.open(self.database, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
                    os.close(descriptor)
                except FileExistsError:
                    pass
            ensure_private(self.database, directory=False)
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

    @contextmanager
    def updating(self, identifier):
        """Hold the write transaction over one record; changes to the yielded record are saved if the block succeeds."""
        with self._connection(write=True) as connection:
            record = self._load(connection, identifier)
            yield record
            self._save(connection, record)

    def _ensure_artifact_root(self):
        for item in (self.root, *self.root.parents):
            if item.is_symlink():
                raise TaskError("unsafe_task_storage", "Task storage cannot traverse symbolic links.")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        ensure_private(self.root)
        artifacts = self.root / "artifacts"
        artifacts.mkdir(mode=0o700, exist_ok=True)
        ensure_private(artifacts)
        return artifacts

    @staticmethod
    def fsync_directory(path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def stage_input(self, source, media_kind="audio", byte_limit=2 * 1024 * 1024 * 1024):
        """Copy a supported media input into a private, unaccepted staging directory."""
        source = Path(source).expanduser()
        suffix = source.suffix.lower()
        if media_kind not in MEDIA_SUFFIXES or not source.is_absolute() or suffix not in MEDIA_SUFFIXES[media_kind]:
            description = {"audio": "WAV, FLAC or MP3", "ltx_audio": "WAV, FLAC, MP3, M4A or AAC", "image": "PNG, JPEG or WebP",
                           "video": "MP4, MOV or M4V", "mesh": "GLB"}.get(media_kind, "supported media")
            raise TaskError("invalid_input", "Media input must be an absolute {0} path.".format(description))
        identifier = str(uuid4())
        artifacts = self._ensure_artifact_root()
        staging = artifacts / (".staging-" + identifier)
        try:
            staging.mkdir(mode=0o700)
            ensure_private(staging)
            if source.is_symlink():
                raise TaskError("invalid_input", "Media input must be a regular file, not a symbolic link.")
            source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            output_path = staging / ("input" + suffix)
            output_descriptor = None
            try:
                before = os.fstat(source_descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise TaskError("invalid_input", "Media input must be a readable regular file.")
                output_descriptor = os.open(output_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                                            getattr(os, "O_NOFOLLOW", 0), 0o600)
                digest, size = hashlib.sha256(), 0
                with os.fdopen(source_descriptor, "rb") as input_stream, os.fdopen(output_descriptor, "wb") as output_stream:
                    source_descriptor = output_descriptor = None
                    while True:
                        chunk = input_stream.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if byte_limit is not None and size > byte_limit:
                            raise TaskError("input_too_large", "Conditioning inputs exceed the per-job byte limit.")
                        output_stream.write(chunk)
                        digest.update(chunk)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                    after = os.fstat(input_stream.fileno())
                current = os.stat(source, follow_symlinks=False)
                identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
                if (identity(before) != identity(after) or identity(before) != identity(current)
                        or size != before.st_size):
                    raise TaskError("input_changed", "Transcription input changed while it was being copied.")
                ensure_private(output_path, directory=False)
                self.fsync_directory(staging)
                final = artifacts / identifier / output_path.name
                return {"id": identifier, "staging_path": str(output_path), "staging_directory": str(staging),
                        "reference": {"path": str(final), "sha256": digest.hexdigest(), "bytes": size}}
            finally:
                if source_descriptor is not None:
                    os.close(source_descriptor)
                if output_descriptor is not None:
                    os.close(output_descriptor)
        except (OSError, TaskError) as error:
            self.discard_staged_input({"id": identifier, "staging_directory": str(staging)})
            if isinstance(error, TaskError):
                raise
            raise TaskError("input_copy_failed", "Cannot safely copy the media input: {0}".format(error),
                            "Check that the source is readable and the task directory has free space.") from error

    def discard_staged_input(self, staged_input):
        """Remove only the private staging directory allocated for this submission."""
        identifier = task_id(staged_input.get("id"))
        expected = self.root / "artifacts" / (".staging-" + identifier)
        if Path(staged_input.get("staging_directory", "")) != expected or expected.is_symlink():
            raise TaskError("invalid_input", "The staged input does not belong to this submission.")
        if expected.exists():
            ensure_private(expected)
            for child in expected.iterdir():
                if child.is_symlink() or not child.is_file():
                    raise TaskError("unsafe_task_storage", "Staged input contains unexpected filesystem entries.")
                child.unlink()
            expected.rmdir()

    def stage_inputs(self, request):
        """Stage a named set of conditioning files in one task-owned directory."""
        from .task_inputs import MAX_INPUT_BYTES, MAX_INPUT_FILES, specifications
        entries = specifications(request)
        if len(entries) > MAX_INPUT_FILES:
            raise TaskError("too_many_inputs", "Conditioning inputs are limited to 16 files per job.")
        staged = []
        try:
            remaining = MAX_INPUT_BYTES
            for name, index, source, media in entries:
                item = self.stage_input(source, media, byte_limit=remaining)
                staged.append(item)
                remaining -= item["reference"]["bytes"]
            if staged:
                first = staged[0]
                identifier = first["id"]
                directory = Path(first["staging_directory"])
            else:
                identifier = str(uuid4())
                directory = self._ensure_artifact_root() / (".staging-" + identifier)
                directory.mkdir(mode=0o700)
                first = {"id": identifier, "staging_directory": str(directory)}
            references = {}
            paths = {}
            for number, ((name, index, _, _), item) in enumerate(zip(entries, staged)):
                target = directory / ("input-" + str(number) + Path(item["staging_path"]).suffix)
                os.replace(item["staging_path"], target)
                if item is not first:
                    Path(item["staging_directory"]).rmdir()
                reference = dict(item["reference"], path=str(self.root / "artifacts" / identifier / target.name))
                if index is None:
                    references[name], paths[name] = reference, str(target)
                else:
                    references.setdefault(name, []).append(reference)
                    paths.setdefault(name, []).append(str(target))
            self.fsync_directory(directory)
            first.update(references=references, paths=paths)
            return first
        except BaseException:
            for item in staged:
                self.discard_staged_input(item)
            raise

    def get(self, identifier):
        with self._connection() as connection:
            return self._load(connection, identifier)

    def list(self, limit=20):
        return self.list_page(limit)["tasks"]

    def list_page(self, limit: int = 20, kinds: Optional[Sequence[str]] = None) -> Dict:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise TaskError("invalid_limit", "Task history limit must be between 1 and 100.")
        if kinds is not None and (not isinstance(kinds, (tuple, list)) or not kinds
                                  or any(not isinstance(kind, str) or not kind for kind in kinds)):
            raise TaskError("invalid_kind", "Task history kinds must be a nonempty list of names.")
        with self._connection() as connection:
            if connection is None:
                return {"tasks": [], "has_more": False}
            cursor = connection.execute("SELECT id, record FROM tasks ORDER BY created_at DESC, id DESC")
            tasks = []
            for identifier, raw in cursor:
                try:
                    value = json.loads(raw)
                except (ValueError, TypeError) as error:
                    raise TaskError("invalid_task_record", "The stored task record is not readable JSON.") from error
                if kinds is not None and isinstance(value, dict) and value.get("kind") not in kinds:
                    continue
                record = validate_record(value)
                if record["id"] != identifier:
                    raise TaskError("invalid_task_record", "Task row identity does not match its record.")
                tasks.append(record)
                if len(tasks) > limit:
                    break
            return {"tasks": tasks[:limit], "has_more": len(tasks) > limit}

    def create_capture(self, task_id: str, request_ref: Dict, identity: Dict, preview_hash: str) -> Dict:
        """Index an accepted in-volume request without copying its body outside the volume."""
        from .coding_capture_contracts import KIND, RUN_SCHEMA as CAPTURE_RUN_SCHEMA
        record = {"schema": CAPTURE_RUN_SCHEMA, "id": task_id, "kind": KIND, "state": "starting",
                  "created_at": utc_now(), "started_at": None, "completed_at": None, "cancel_requested": False,
                  "request": request_ref, "preview_hash": preview_hash, "context": identity, "processes": {},
                  "client_outcome": "pending", "attribution": "pending", "result": None, "error": None,
                  "execution_unresolved": False, "artifacts_removed_at": None}
        record = json.loads(encoded(validate_record(record)))
        with self._connection(write=True) as connection:
            try:
                connection.execute("INSERT INTO tasks VALUES (?, ?, ?)", (task_id, record["created_at"], encoded(record).decode()))
            except sqlite3.IntegrityError as error:
                raise TaskError("task_exists", "A task with that ID already exists.") from error
        return record

    def artifact_path(self, identifier, name):
        task_id(identifier)
        if not isinstance(name, str) or not name or name.startswith(".") or Path(name).name != name or "/" in name or "\\" in name:
            raise TaskError("invalid_artifact", "Artifact names must be plain file names.")
        try:
            for directory in (self.root, self.root / "artifacts", self.root / "artifacts" / identifier):
                ensure_private(directory)
            path = self.root / "artifacts" / identifier / name
            if path.is_symlink():
                raise TaskError("unsafe_task_storage", "Task artifacts cannot be symbolic links.")
            if path.exists():
                ensure_private(path, directory=False)
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

    def create(self, request, identity, input_artifact=None, staged_input=None):
        request = validate_request(request)
        if staged_input is not None and "references" in staged_input:
            return self._create_named(request, identity, staged_input)
        media = request["schema"] == "task-request/2"
        if not media and (staged_input is not None or input_artifact is not None):
            raise TaskError("invalid_task", "Legacy tasks cannot carry a media input snapshot.")
        takes_input = task_kinds.get(request["kind"]).input_media is not None
        if takes_input and (staged_input is None or input_artifact is None):
            raise TaskError("invalid_task", "This task kind requires a retained input snapshot.")
        if not takes_input and (staged_input is not None or input_artifact is not None):
            raise TaskError("invalid_task", "This task kind does not accept an input snapshot.")
        identifier = staged_input["id"] if staged_input is not None else str(uuid4())
        if staged_input is not None:
            staged_directory = self.root / "artifacts" / (".staging-" + identifier)
            staged_path = Path(staged_input.get("staging_path", ""))
            if (Path(staged_input.get("staging_directory", "")) != staged_directory
                    or staged_path.parent != staged_directory or staged_path.is_symlink()):
                raise TaskError("invalid_input", "The staged input does not match this task submission.")
            ensure_private(staged_directory)
            ensure_private(staged_path, directory=False)
            digest, size = hashlib.sha256(), 0
            try:
                with staged_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
            except OSError as error:
                self.discard_staged_input(staged_input)
                raise TaskError("input_copy_failed", "The staged media input is unavailable.") from error
            if digest.hexdigest() != input_artifact.get("sha256") or size != input_artifact.get("bytes"):
                self.discard_staged_input(staged_input)
                raise TaskError("input_changed", "The staged transcription input changed before acceptance.")
        artifacts = self._ensure_artifact_root()
        final_directory = artifacts / identifier
        moved = False
        try:
            with self._connection(write=True) as connection:
                from .task_storage import check_submission
                check_submission(self, connection, input_artifact["bytes"] if input_artifact else 0, len(encoded(request)))
                if staged_input is not None:
                    expected_stage = artifacts / (".staging-" + identifier)
                    if (Path(staged_input.get("staging_directory", "")) != expected_stage
                            or Path(input_artifact.get("path", "")).parent != final_directory
                            or not expected_stage.is_dir() or final_directory.exists()):
                        raise TaskError("invalid_input", "The staged input does not match this task submission.")
                    os.replace(expected_stage, final_directory)
                    moved = True
                    self.fsync_directory(artifacts)
                    ensure_private(final_directory)
                else:
                    final_directory.mkdir(mode=0o700, exist_ok=True)
                    ensure_private(final_directory)
                reference = self._write_artifact(identifier, "request.json", request)
                record = {"schema": MEDIA_RUN_SCHEMA if media else RUN_SCHEMA, "id": identifier,
                          "kind": request["kind"], "state": "starting", "model_path": request["model_path"],
                          "created_at": utc_now(), "started_at": None, "completed_at": None,
                          "cancel_requested": False, "identity": identity, "request": reference, "result": None,
                          "error": None, "supervisor": None, "execution": None, "execution_unresolved": False}
                if media:
                    record["input_artifact"] = input_artifact
                validate_record(record)
                connection.execute("INSERT INTO tasks VALUES (?, ?, ?)", (identifier, record["created_at"], encoded(record).decode()))
                return record
        except (TaskError, OSError, sqlite3.Error, ValueError):
            if staged_input is not None:
                try:
                    self.discard_staged_input(staged_input)
                except TaskError:
                    pass
                if moved:
                    try:
                        self.get(identifier)
                    except TaskError as error:
                        if error.code == "task_not_found":
                            self._remove_unaccepted_directory(final_directory)
            raise

    def _create_named(self, request, identity, staged):
        from .task_inputs import specifications
        identifier = task_id(staged["id"])
        artifacts = self._ensure_artifact_root()
        directory = artifacts / (".staging-" + identifier)
        final = artifacts / identifier
        entries = specifications(request)
        if Path(staged.get("staging_directory", "")) != directory or final.exists():
            raise TaskError("invalid_input", "The staged inputs do not match this submission.")
        ensure_private(directory)
        refs, paths = staged["references"], staged["paths"]
        expected_keys = {name for name, _, _, _ in entries}
        if set(refs) != expected_keys or set(paths) != expected_keys:
            raise TaskError("invalid_input", "The staged input set does not match this submission.")
        view_count = sum(1 for name, _, _, _ in entries if name == "views")
        if view_count and (not isinstance(refs.get("views"), list) or not isinstance(paths.get("views"), list)
                           or len(refs["views"]) != view_count or len(paths["views"]) != view_count):
            raise TaskError("invalid_input", "The staged views do not match this submission.")
        expected_paths = {str(value) for item in paths.values() for value in (item if isinstance(item, list) else [item])}
        if {str(path) for path in directory.iterdir()} != expected_paths:
            raise TaskError("invalid_input", "The staging directory has unexpected files.")
        for number, (name, index, _, media) in enumerate(entries):
            try:
                ref = refs[name] if index is None else refs[name][index]
                path = Path(paths[name] if index is None else paths[name][index])
                _artifact_reference(ref)
            except (KeyError, IndexError, TypeError, TaskError) as error:
                raise TaskError("invalid_input", "A staged input reference is invalid.") from error
            expected_name = "input-" + str(number) + path.suffix
            if (path.parent != directory or path.name != expected_name or path.suffix not in MEDIA_SUFFIXES[media]
                    or ref["path"] != str(final / expected_name) or path.is_symlink()):
                raise TaskError("invalid_input", "A staged input is outside this submission.")
            ensure_private(path, directory=False)
            digest, size = hashlib.sha256(), 0
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != ref["sha256"] or size != ref["bytes"]:
                raise TaskError("input_changed", "A staged input changed before acceptance.")
        moved = False
        try:
            with self._connection(write=True) as connection:
                from .task_storage import check_submission
                check_submission(self, connection, sum(ref["bytes"] for value in refs.values()
                                 for ref in (value if isinstance(value, list) else [value])), len(encoded(request)))
                os.replace(directory, final)
                moved = True
                self.fsync_directory(artifacts)
                reference = self._write_artifact(identifier, "request.json", request)
                record = {"schema": "task-run/3", "id": identifier, "kind": request["kind"],
                          "state": "starting", "model_path": request["model_path"], "created_at": utc_now(),
                          "started_at": None, "completed_at": None, "cancel_requested": False,
                          "identity": identity, "request": reference, "input_artifacts": refs, "result": None,
                          "error": None, "supervisor": None, "execution": None, "execution_unresolved": False}
                validate_record(record)
                connection.execute("INSERT INTO tasks VALUES (?, ?, ?)", (identifier, record["created_at"], encoded(record).decode()))
                return record
        except (TaskError, OSError, sqlite3.Error, ValueError):
            if moved:
                try:
                    self.get(identifier)
                except TaskError as error:
                    if error.code == "task_not_found":
                        self._remove_unaccepted_directory(final)
            raise

    @staticmethod
    def _remove_unaccepted_directory(directory):
        if not directory.exists() or directory.is_symlink():
            return
        ensure_private(directory)
        for child in directory.iterdir():
            if child.is_symlink() or not child.is_file():
                raise TaskError("unsafe_task_storage", "Unaccepted task directory contains unexpected entries.")
            child.unlink()
        directory.rmdir()

    def verify_input_artifact(self, identifier, reference, media_kind="audio"):
        if media_kind not in MEDIA_SUFFIXES:
            raise TaskError("invalid_input", "The retained input media kind is unsupported.")
        names = tuple("input" + suffix for suffix in MEDIA_SUFFIXES[media_kind])
        return self.verify_artifact(identifier, reference, allowed_names=names,
                                    code="input_artifact")

    def verify_named_inputs(self, identifier, request, references):
        from .task_inputs import specifications
        entries = specifications(request)
        if set(references) != {name for name, _, _, _ in entries}:
            raise TaskError("input_artifact_changed", "The retained input set does not match the request.")
        paths = {}
        for number, (name, index, _, media) in enumerate(entries):
            try:
                reference = references[name] if index is None else references[name][index]
                suffix = Path(reference["path"]).suffix
                if suffix not in MEDIA_SUFFIXES[media]:
                    raise ValueError()
                owned = self.verify_artifact(identifier, reference, expected_name="input-" + str(number) + suffix,
                                             code="input_artifact")
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise TaskError("input_artifact_changed", "The retained input reference is invalid.") from error
            if index is None:
                paths[name] = str(owned)
            else:
                paths.setdefault(name, []).append(str(owned))
        if "views" in references and len(references["views"]) != len(paths.get("views", [])):
            raise TaskError("input_artifact_changed", "The retained views do not match the request.")
        return paths

    def verify_artifact(self, identifier, reference, expected_name=None, allowed_names=(), code="artifact"):
        task_id(identifier)
        try:
            _artifact_reference(reference)
        except TaskError as error:
            raise TaskError(code + "_changed", "The task artifact reference is invalid.") from error
        path = Path(reference["path"])
        names = set(allowed_names)
        if expected_name is not None:
            names.add(expected_name)
        if (path.parent.name != identifier or path.parent.parent != self.root / "artifacts"
                or path.name not in names):
            raise TaskError(code + "_changed", "The task artifact is outside its owned directory.")
        owned = self.artifact_path(identifier, path.name)
        digest, size = hashlib.sha256(), 0
        try:
            with owned.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as error:
            raise TaskError(code + "_unavailable", "The retained task artifact is unavailable.") from error
        if str(owned) != reference["path"] or digest.hexdigest() != reference["sha256"] or size != reference["bytes"]:
            raise TaskError(code + "_changed", "The retained task artifact changed.")
        return owned

    def transition(self, identifier, expected, changes):
        with self._connection(write=True) as connection:
            record = self._load(connection, identifier)
            capture = record["kind"] == "coding_capture"
            mutable = {"state", "started_at", "completed_at", "result", "error", "execution_unresolved"}
            mutable.update(("processes", "client_outcome", "attribution") if capture else ("supervisor", "execution"))
            if not isinstance(changes, dict) or set(changes) - mutable:
                raise TaskError("invalid_transition", "Task identity and accepted inputs are immutable.")
            if record["state"] in TERMINAL or record["state"] not in expected:
                raise TaskError("task_state_changed", "The task state has already changed.", "Read its current status.")
            if "supervisor" in changes and record["supervisor"] is not None:
                raise TaskError("task_state_changed", "The task already has a supervisor.")
            if capture and "processes" in changes and isinstance(changes["processes"], dict):
                if any(changes["processes"].get(role) != identity for role, identity in record["processes"].items()):
                    raise TaskError("task_state_changed", "Accepted capture process identities cannot be replaced.")
            next_state = changes.get("state", record["state"])
            if next_state == "succeeded" and record["cancel_requested"]:
                raise TaskError("task_cancel_requested", "Cancellation was requested before result publication.")
            if record["state"] == "running" and next_state == "starting":
                raise TaskError("invalid_transition", "Running tasks cannot restart under the same identity.")
            record.update(changes)
            if next_state == "running" and record["started_at"] is None:
                record["started_at"] = utc_now()
            if next_state in TERMINAL:
                record["completed_at"] = utc_now()
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
        if record["kind"] == "coding_capture":
            raise TaskError("capture_storage_required", "Read capture requests through their bounded owned volume.")
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
            if record["kind"] == "coding_capture":
                raise TaskError("capture_storage_required", "Publish capture results inside their bounded owned volume.")
            if record["state"] in TERMINAL:
                raise TaskError("task_state_changed", "A terminal task cannot acquire new result evidence.")
            return self._write_artifact(identifier, "result.json", value)
