"""Bounded, private evidence with explicit export and recoverable removal."""

import base64
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
from typing import Dict, Optional
from uuid import uuid4

from .capture_platform import CaptureVolume, capture_process_identity
from .coding_capture_contracts import (DEFAULT_LIMITS, KIND, METADATA_RESERVE, validate_capture_request,
                                       validate_capture_result, validate_limits, validate_reference)
from .task_contracts import MAX_REQUEST_BYTES, TERMINAL, TaskError, encoded, task_id as validate_task_id
from .task_store import TaskStore, ensure_private, utc_now


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _private_file(path):
    ensure_private(path, directory=False)
    if path.stat().st_nlink != 1:
        raise TaskError("unsafe_capture_storage", "Capture artifacts cannot share a hard link.")


def _failure(error):
    code = "capture_budget_exhausted" if getattr(error, "errno", None) in (errno.ENOSPC, errno.EDQUOT) else "capture_storage_failed"
    return TaskError(code, "Cannot safely access the owned capture data.", "Keep the capture for recovery; inspect free space and ownership.")


class CaptureStorage:
    def __init__(self, store: TaskStore, task_id: str):
        self.store = store
        self.identifier = validate_task_id(task_id)
        self.root = store.root / "captures" / self.identifier
        self.tombstone = self.root.parent / (self.identifier + ".removed.json")
        self._limits = None
        self._mounted = None
        self._readonly = True
        self._lock = threading.RLock()
        self._lock_descriptor = None

    @contextmanager
    def _exclusive(self):
        with self._lock:
            if self._lock_descriptor is not None:
                yield
                return
            ensure_private(self.root.parent)
            descriptor = os.open(self.root.parent / (self.identifier + ".lock"),
                                 os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                    raise TaskError("unsafe_capture_storage", "Capture operation lock ownership changed.")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise TaskError("capture_storage_busy", "Another operation owns this capture.") from error
                self._lock_descriptor = descriptor
                try:
                    yield
                finally:
                    self._lock_descriptor = None
            finally:
                os.close(descriptor)

    @classmethod
    def create(cls, store: TaskStore, task_id: str, limits: Dict):
        limits = validate_limits(limits)
        storage = cls(store, task_id)
        for path in (store.root, *store.root.parents):
            if path.is_symlink():
                raise TaskError("unsafe_capture_storage", "Capture storage cannot traverse a link.")
        store.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        ensure_private(store.root)
        storage.root.parent.mkdir(mode=0o700, exist_ok=True)
        ensure_private(storage.root.parent)
        if storage.tombstone.exists() or storage.tombstone.is_symlink():
            raise TaskError("task_exists", "Removed capture identities cannot be reused.")
        CaptureVolume.create(storage.root, task_id, limits["total_bytes"] - METADATA_RESERVE)
        storage._limits = limits
        with storage.active() as mounted:
            for name in ("artifacts", "profile", "tmp"):
                (mounted / name).mkdir(mode=0o700)
            storage._capacity(mounted, limits)
        return storage

    @contextmanager
    def active(self, readonly=False):
        """Internal scoped mount. Mounted paths never enter a CLI/MCP result."""
        with self._exclusive():
            ensure_private(self.root.parent)
            if self._mounted is not None:
                if self._readonly and not readonly:
                    raise TaskError("capture_readonly", "Capture is mounted for reading only.")
                yield self._mounted
                return
            try:
                with CaptureVolume.open_owned(self.root, readonly=readonly) as mounted:
                    self._mounted, self._readonly = mounted, readonly
                    try:
                        yield mounted
                    finally:
                        self._mounted = None
            except OSError as error:
                raise _failure(error) from error

    def _path(self, mounted, name):
        validate_reference({"name": name, "sha256": "0" * 64, "bytes": 0})
        directory = mounted / "artifacts"
        ensure_private(directory)
        path = directory / name
        if path.is_symlink():
            raise TaskError("unsafe_capture_storage", "Capture artifacts cannot be links.")
        return path

    def _read(self, mounted, name, expected=None, maximum=None):
        path = self._path(mounted, name)
        _private_file(path)
        maximum = maximum if maximum is not None else DEFAULT_LIMITS["artifact_bytes"]
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise TaskError("capture_artifact_limit", "Capture artifact exceeds its accepted size limit.")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                raw = stream.read(maximum + 1)
                after = os.fstat(stream.fileno())
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
            if len(raw) > maximum or identity(before) != identity(after) or identity(after) != identity(path.lstat()):
                raise TaskError("capture_artifact_changed", "Capture artifact changed during its read.")
            reference = {"name": name, "sha256": _hash(raw), "bytes": len(raw)}
            if expected is not None and expected != reference:
                raise TaskError("capture_artifact_changed", "Retained capture artifact no longer matches its accepted hash.")
            return raw, reference
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _request(self, mounted, record=None):
        record = record or self.store.get(self.identifier)
        raw, _ = self._read(mounted, "request.json", record["request"], MAX_REQUEST_BYTES)
        try:
            request = validate_capture_request(json.loads(raw))
        except (ValueError, TypeError) as error:
            raise TaskError("capture_request_changed", "Retained capture request is invalid.") from error
        context = record["context"]
        if (request["model_path"] != context["serve"]["model_path"]
                or request["serve_receipt"] != context["serve"]["receipt"]
                or request["client_path"] != context["client"]["path"]
                or request["project_path"] != context["project"]["path"]
                or request.get("workbench_context") != context.get("workbench_context")):
            raise TaskError("capture_context_changed", "Original capture request and accepted context disagree.")
        self._capacity(mounted, request["limits"])
        return request

    def _capacity(self, mounted, limits):
        info = os.statvfs(mounted)
        allocated = sum(item.lstat().st_blocks * 512 for item in (self.root / "capture.sparsebundle").rglob("*")
                        if not item.is_symlink())
        if info.f_blocks * info.f_frsize > limits["total_bytes"] - METADATA_RESERVE or allocated > limits["total_bytes"]:
            raise TaskError("capture_budget_exhausted", "Capture filesystem exceeds its accepted owned-data budget.")

    def _atomic(self, mounted, name, raw):
        target = self._path(mounted, name)
        if target.exists():
            raise TaskError("capture_artifact_exists", "Retained capture evidence cannot be overwritten.")
        temporary = target.parent / (".write-" + str(uuid4()))
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic publication refuses a destination created after the check.
            os.link(temporary, target, follow_symlinks=False)
            temporary.unlink()
            TaskStore.fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return {"name": name, "sha256": _hash(raw), "bytes": len(raw)}

    def accept_request(self, value):
        value = validate_capture_request(value)
        if self._limits != value["limits"]:
            raise TaskError("capture_limits_changed", "Capture request limits differ from its allocated volume.")
        with self.active() as mounted:
            return self._atomic(mounted, "request.json", encoded(value))

    def write_artifact(self, name, raw):
        if not isinstance(raw, bytes):
            raise TaskError("invalid_capture_artifact", "Capture artifact content must be bytes.")
        with self.active() as mounted:
            limits = self._request(mounted)["limits"]
            if name in ("request.json", "result.json"):
                raise TaskError("invalid_capture_artifact", "Capture request and result have dedicated publication boundaries.")
            if len(raw) > limits["artifact_bytes"]:
                raise TaskError("capture_artifact_limit", "Capture artifact exceeds its accepted size limit.")
            try:
                return self._atomic(mounted, name, raw)
            except OSError as error:
                raise _failure(error) from error

    def _verify(self, mounted, expected_digest=None):
        record = self.store.get(self.identifier)
        request = self._request(mounted, record)
        expected = record["result"]
        raw, reference = self._read(mounted, "result.json", expected, request["limits"]["artifact_bytes"])
        if expected_digest is not None and reference["sha256"] != expected_digest:
            raise TaskError("capture_result_changed", "Capture result digest changed.")
        try:
            manifest = validate_capture_result(json.loads(raw))
        except (ValueError, TypeError) as error:
            raise TaskError("invalid_capture_result", "Retained capture manifest is invalid.") from error
        if (raw != encoded(manifest) or manifest["task_id"] != self.identifier
                or manifest["request_sha256"] != record["request"]["sha256"]
                or manifest["preview_hash"] != record["preview_hash"] or manifest["context"] != record["context"]):
            raise TaskError("capture_context_changed", "Retained capture result is not bound to the accepted request and context.")
        references = []
        for artifact in manifest["artifacts"]:
            _, observed = self._read(mounted, artifact["name"], artifact, request["limits"]["artifact_bytes"])
            references.append(observed)
        return self._verified_reply(raw, reference, references)

    def _verified_reply(self, raw, reference, artifacts):
        reply = {"schema": "coding-capture-verified/1", "task_id": self.identifier, "sha256": reference["sha256"],
                 "manifest_utf8": raw.decode("utf-8"), "artifacts": artifacts}
        # Leave space for the shared result envelope within its 12,288-byte cap.
        if len(encoded(reply)) > 10240:
            raise TaskError("capture_result_limit", "Capture result exceeds the bounded verification response.",
                            "Retain the session artifacts; use a smaller request cohort for a verified result.")
        return reply

    @staticmethod
    def _cleanup_profile(mounted):
        for name in ("profile", "tmp"):
            directory = mounted / name
            if directory.exists() or directory.is_symlink():
                ensure_private(directory)
                for item in directory.rglob("*"):
                    if item.is_symlink():
                        raise TaskError("unsafe_capture_storage", "Capture profile cleanup cannot traverse a link.")
                shutil.rmtree(directory)
        TaskStore.fsync_directory(mounted)

    def publish(self, manifest: Dict) -> Dict:
        manifest = validate_capture_result(manifest)
        with self.active() as mounted:
            request = self._request(mounted)
            raw = encoded(manifest)
            if len(raw) > request["limits"]["artifact_bytes"]:
                raise TaskError("capture_artifact_limit", "Capture manifest exceeds its accepted artifact size limit.")
            try:
                path = self._path(mounted, "result.json")
                if not path.exists():
                    # Validate referenced evidence and immutable bindings before publication.
                    record = self.store.get(self.identifier)
                    if (manifest["task_id"] != self.identifier or manifest["request_sha256"] != record["request"]["sha256"]
                            or manifest["preview_hash"] != record["preview_hash"] or manifest["context"] != record["context"]):
                        raise TaskError("capture_context_changed", "Capture result does not match its accepted context.")
                    for artifact in manifest["artifacts"]:
                        self._read(mounted, artifact["name"], artifact, request["limits"]["artifact_bytes"])
                    self._verified_reply(raw, {"sha256": _hash(raw)}, manifest["artifacts"])
                    self._atomic(mounted, "result.json", raw)
                verified = self._verify(mounted, _hash(raw))
                if manifest["client_exit_code"] == 0 and manifest["attribution"] == "verified":
                    self._cleanup_profile(mounted)
                return {"name": "result.json", "sha256": verified["sha256"], "bytes": len(raw)}
            except OSError as error:
                raise _failure(error) from error

    def verify(self, expected_digest: Optional[str] = None) -> Dict:
        with self.active(readonly=True) as mounted:
            return self._verify(mounted, expected_digest)

    def _evidence(self, mounted, name):
        request = self._request(mounted)
        if self._path(mounted, "result.json").exists():
            verified = self._verify(mounted)
            references = {item["name"]: item for item in verified["artifacts"]}
            references["result.json"] = {"name": "result.json", "sha256": verified["sha256"],
                                         "bytes": len(verified["manifest_utf8"].encode("utf-8"))}
            references["request.json"] = self.store.get(self.identifier)["request"]
            if name not in references:
                raise TaskError("capture_artifact_unavailable", "Artifact is not part of the retained result.")
            return self._read(mounted, name, references[name], request["limits"]["artifact_bytes"])
        return self._read(mounted, name, maximum=request["limits"]["artifact_bytes"])

    def read_artifact(self, name: str, offset: int, length: int = 6144) -> Dict:
        if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 6144:
            raise TaskError("invalid_capture_read", "Artifact reads require a nonnegative offset and at most 6,144 bytes.")
        with self.active(readonly=True) as mounted:
            raw, reference = self._evidence(mounted, name)
            chunk = raw[offset:offset + length]
            return dict(reference, offset=offset, data_base64=base64.b64encode(chunk).decode("ascii"),
                        next_offset=offset + len(chunk) if offset + len(chunk) < len(raw) else None)

    def export_artifact(self, name: str, destination: Path, sanitize: bool) -> Dict:
        destination = Path(destination)
        if type(sanitize) is not bool or not destination.is_absolute() or ".." in destination.parts:
            raise TaskError("invalid_capture_export", "Choose a new absolute export file and explicit sharing mode.")
        for path in (destination, *destination.parents):
            if path.is_symlink():
                raise TaskError("unsafe_capture_export", "Export cannot traverse a symbolic link.")
        if destination.exists() or self.store.root == destination or self.store.root in destination.parents:
            raise TaskError("unsafe_capture_export", "Export must be a new file outside engine task storage.")
        with self.active(readonly=True) as mounted:
            raw, source = self._evidence(mounted, name)
            if sanitize:
                record = self.store.get(self.identifier)
                raw = encoded({"schema": "coding-capture-sharing/1", "task_id": self.identifier,
                               "state": record["state"], "client_outcome": record["client_outcome"],
                               "attribution": record["attribution"], "content_removed": True,
                               "omitted": ["prompts", "code", "tool output", "credentials", "project paths"]})
            temporary = destination.parent / (".capture-export-" + str(uuid4()))
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, destination, follow_symlinks=False)
                temporary.unlink()
                TaskStore.fsync_directory(destination.parent)
            except OSError as error:
                raise TaskError("capture_export_failed", "Cannot publish the chosen export file.") from error
            finally:
                temporary.unlink(missing_ok=True)
            return {"source": source, "destination": str(destination), "sha256": _hash(raw), "bytes": len(raw),
                    "content_removed": sanitize}

    def _removable(self, record):
        if record["kind"] != KIND or record["state"] not in TERMINAL or record["execution_unresolved"]:
            raise TaskError("capture_removal_unavailable", "Only terminal captures with resolved owned execution can be removed.")
        for process in record["processes"].values():
            try:
                os.kill(process["pid"], 0)
            except ProcessLookupError:
                continue
            except OSError as error:
                raise TaskError("capture_removal_unavailable", "Cannot establish that owned execution has stopped.") from error
            if capture_process_identity(process["pid"]) == process:
                raise TaskError("capture_removal_unavailable", "A capture-owned process is still running.")

    def _removal_preview(self, record):
        self._removable(record)
        if record["artifacts_removed_at"] is not None:
            raise TaskError("capture_artifacts_removed", "Capture artifacts were already explicitly removed.")
        with self.active(readonly=True) as mounted:
            request = self._request(mounted, record)
            if self._path(mounted, "result.json").exists():
                self._verify(mounted)
            tree = hashlib.sha256()
            profile_bytes, owned_bytes, count = 0, 0, 0
            artifacts = []
            for path in sorted(mounted.rglob("*")):
                if path.is_symlink():
                    raise TaskError("unsafe_capture_storage", "Capture removal cannot traverse a symbolic link.")
                ensure_private(path, directory=path.is_dir())
                if path.is_file():
                    _private_file(path)
                    info = path.lstat()
                    relative = str(path.relative_to(mounted))
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(65536), b""):
                            digest.update(chunk)
                    # The volume's device number changes with each attach; the inode identifies a file within it.
                    tree.update(encoded([relative, info.st_ino, info.st_size, digest.hexdigest()]))
                    count += 1
                    owned_bytes += info.st_size
                    if path.relative_to(mounted).parts[0] in ("profile", "tmp"):
                        profile_bytes += info.st_size
                    elif path.parent == mounted / "artifacts" and not path.name.startswith("."):
                        artifacts.append({"name": path.name, "sha256": digest.hexdigest(), "bytes": info.st_size})
            marker = self.root / "volume.json"
            _private_file(marker)
            binding = {"task_id": self.identifier, "request": record["request"], "result": record["result"],
                       "completed_at": record["completed_at"], "tree_sha256": tree.hexdigest(),
                       "volume_sha256": _hash(marker.read_bytes())}
            return {"schema": "coding-capture-remove-preview/1", "task_id": self.identifier,
                    "preview_hash": _hash(encoded(binding)), "artifacts": artifacts[:20],
                    "artifact_count": len(artifacts), "file_count": count, "profile_bytes": profile_bytes,
                    "owned_bytes": owned_bytes, "limits": request["limits"], "has_more_artifacts": len(artifacts) > 20}

    def remove_preview(self) -> Dict:
        return self._removal_preview(self.store.get(self.identifier))

    def remove(self, preview_hash: str) -> Dict:
        with self._exclusive():
            return self._remove_locked(preview_hash)

    def _remove_locked(self, preview_hash):
        # The database transaction fences lifecycle changes and competing removals.
        # The bounded tombstone bridges volume deletion and the row commit.
        with self.store.updating(self.identifier) as record:
            self._removable(record)
            if not self.root.exists():
                _private_file(self.tombstone)
                if self.tombstone.stat().st_size > 2048:
                    raise TaskError("unsafe_capture_storage", "Removal recovery metadata exceeds its limit.")
                saved = json.loads(self.tombstone.read_bytes())
                if (saved.get("task_id") != self.identifier or saved.get("preview_hash") != preview_hash
                        or saved.get("request_sha256") != record["request"]["sha256"]):
                    raise TaskError("capture_removal_changed", "Removal recovery identity changed.")
            else:
                preview = self._removal_preview(record)
                if preview["preview_hash"] != preview_hash:
                    raise TaskError("capture_removal_changed", "Capture contents changed after removal preview.")
                saved = {"schema": "coding-capture-removal/1", "task_id": self.identifier,
                         "preview_hash": preview_hash, "request_sha256": record["request"]["sha256"],
                         "removed_at": utc_now()}
                raw = encoded(saved)
                if len(raw) > 2048:
                    raise TaskError("capture_metadata_limit", "Removal metadata exceeds its limit.")
                if self.tombstone.exists() or self.tombstone.is_symlink():
                    _private_file(self.tombstone)
                    if self.tombstone.stat().st_size > 2048:
                        raise TaskError("unsafe_capture_storage", "Removal recovery metadata exceeds its limit.")
                    previous = json.loads(self.tombstone.read_bytes())
                    if previous != saved and any(previous.get(key) != saved[key] for key in ("schema", "task_id", "preview_hash", "request_sha256")):
                        raise TaskError("capture_removal_changed", "An earlier removal must be recovered first.")
                    saved = previous
                else:
                    descriptor = os.open(self.tombstone, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    TaskStore.fsync_directory(self.root.parent)
                CaptureVolume.remove(self.root)
            record["artifacts_removed_at"] = saved["removed_at"]
            return {"task_id": self.identifier, "artifacts_removed_at": saved["removed_at"]}
