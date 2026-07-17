"""Durable, reversible configuration transactions for Wire.

The transaction root is opened with ``O_NOFOLLOW`` where macOS/Python exposes
it.  Stage, backup, receipt, and replace operations then use that directory
file descriptor.  A hostile actor can still race an already-open target leaf
between validation and replacement, but ``os.replace`` replaces that leaf and
never follows it; ancestors are verified before opening their directory fd.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import stat
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .wiring import ConfigAdapter, redact_endpoint, redact_secrets


RECEIPT_SCHEMA_VERSION = "2.0"
MAX_PREVIEW_CHARS = 12000
_STATUSES = {"pending", "applied", "failed", "rolled_back", "rollback_failed", "recovery_required"}
_HASH = __import__("re").compile(r"^[0-9a-f]{64}$")


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _absolute(path):
    value = Path(path)
    return Path(os.path.abspath(str(value)))


def _has_parent_reference(path):
    return ".." in Path(path).parts


def _assert_no_symlink_ancestors(path, leaf_may_be_missing=False):
    """Reject symlinks in every existing component without resolving them."""
    value = _absolute(path)
    if _has_parent_reference(path):
        raise ValueError("path traversal is not allowed: {0}".format(path))
    current = Path(value.anchor)
    parts = value.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            item = os.lstat(str(current))
        except FileNotFoundError:
            if leaf_may_be_missing and index == len(parts) - 1:
                return value
            raise ValueError("path component does not exist: {0}".format(current))
        # macOS exposes /var as a system-owned compatibility alias to /private/var;
        # accept only that exact root alias so temporary directories remain usable.
        if stat.S_ISLNK(item.st_mode) and not (str(current) == "/var" and os.readlink(str(current)) == "private/var"):
            raise ValueError("refusing symlink path component: {0}".format(current))
    return value


def _assert_safe_directory(path, create=False):
    value = _absolute(path)
    if create and not value.exists():
        parent = _assert_safe_directory(value.parent, create=True)
        os.mkdir(str(value), 0o700)
        _fsync_directory(parent)
    _assert_no_symlink_ancestors(value)
    if not value.is_dir():
        raise ValueError("safe directory required: {0}".format(value))
    return value


def _assert_safe_target(path):
    value = _absolute(path)
    _assert_no_symlink_ancestors(value, leaf_may_be_missing=True)
    _assert_safe_directory(value.parent)
    if value.exists():
        item = os.lstat(str(value))
        if not stat.S_ISREG(item.st_mode):
            raise ValueError("target is not a regular file: {0}".format(value))
    return value


def _open_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("not a directory: {0}".format(path))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fsync_directory(path_or_fd):
    descriptor = path_or_fd if isinstance(path_or_fd, int) else _open_directory(path_or_fd)
    close = not isinstance(path_or_fd, int)
    try:
        os.fsync(descriptor)
    finally:
        if close:
            os.close(descriptor)


def _write_all(descriptor, content):
    offset = 0
    while offset < len(content):
        offset += os.write(descriptor, content[offset:])


def _atomic_in_directory(directory, name, content, mode, validator=None):
    """Write and fsync a stage file before atomically replacing ``name``."""
    directory = _assert_safe_directory(directory)
    dir_fd = _open_directory(directory)
    stage_name = ".mlx-agent-stage-{0}".format(uuid.uuid4().hex)
    stage_fd = None
    try:
        stage_fd = os.open(stage_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=dir_fd)
        _write_all(stage_fd, content)
        os.fsync(stage_fd)
        os.close(stage_fd)
        stage_fd = None
        _fsync_directory(dir_fd)
        if validator is not None:
            read_fd = os.open(stage_name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd)
            try:
                data = b""
                while True:
                    chunk = os.read(read_fd, 65536)
                    if not chunk:
                        break
                    data += chunk
            finally:
                os.close(read_fd)
            validator(data.decode("utf-8"))
        os.chmod(stage_name, mode, dir_fd=dir_fd, follow_symlinks=False)
        os.replace(stage_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        _fsync_directory(dir_fd)
    except BaseException:
        if stage_fd is not None:
            os.close(stage_fd)
        try:
            os.unlink(stage_name, dir_fd=dir_fd)
            _fsync_directory(dir_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(dir_fd)


def _read_regular(path):
    value = _assert_safe_target(path)
    if not value.exists():
        return b""
    return value.read_bytes()


@dataclass
class Receipt:
    schema_version: str
    transaction_id: str
    adapter_version: str
    timestamp: str
    transaction_root: str
    targets: list
    target_roots: dict
    target_modes: dict
    before_hashes: dict
    after_hashes: dict
    backup_paths: dict
    validations: dict
    status: str
    preview: str
    preview_hash: str
    receipt_path: str = field(default="", repr=False, compare=False)

    def to_dict(self):
        value = asdict(self)
        value.pop("receipt_path", None)
        return value

    @classmethod
    def from_dict(cls, value, receipt_path=""):
        required = {
            "schema_version", "transaction_id", "adapter_version", "timestamp", "transaction_root",
            "targets", "target_roots", "target_modes", "before_hashes", "after_hashes", "backup_paths",
            "validations", "status", "preview", "preview_hash",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("receipt fields are malformed or untrusted")
        if value["schema_version"] != RECEIPT_SCHEMA_VERSION or value["status"] not in _STATUSES:
            raise ValueError("receipt version or status is unsupported")
        try:
            uuid.UUID(value["transaction_id"])
            parsed_timestamp = datetime.fromisoformat(value["timestamp"].replace("Z", "+00:00"))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("receipt ID or timestamp is malformed")
        if parsed_timestamp.tzinfo is None:
            raise ValueError("receipt timestamp must include a timezone")
        if not isinstance(value["adapter_version"], str) or not value["adapter_version"]:
            raise ValueError("receipt adapter version is malformed")
        root = _assert_safe_directory(value["transaction_root"])
        if Path(value["transaction_root"]) != root:
            raise ValueError("receipt transaction root must be normalized")
        if receipt_path:
            location = _assert_safe_target(receipt_path)
            if location.parent != root or location.name != "receipt.json":
                raise ValueError("receipt is outside its transaction layout")
        targets = value["targets"]
        maps = (value["target_roots"], value["target_modes"], value["before_hashes"], value["after_hashes"], value["backup_paths"])
        if not isinstance(targets, list) or not targets or not all(isinstance(item, str) for item in targets) or len(targets) != len(set(targets)) or not all(isinstance(item, dict) for item in maps):
            raise ValueError("receipt targets are malformed")
        if set(targets) != set(value["target_roots"]) or set(targets) != set(value["target_modes"]) or set(targets) != set(value["before_hashes"]) or set(targets) != set(value["after_hashes"]) or set(targets) != set(value["backup_paths"]):
            raise ValueError("receipt target maps do not match")
        for index, target_name in enumerate(targets):
            target = _assert_safe_target(target_name)
            if str(target) != target_name or value["target_roots"][target_name] != str(target.parent):
                raise ValueError("receipt target is not rooted safely")
            mode = value["target_modes"][target_name]
            if mode is not None and (not isinstance(mode, int) or mode < 0 or mode > 0o777):
                raise ValueError("receipt mode is malformed")
            for hashes in (value["before_hashes"], value["after_hashes"]):
                if not isinstance(hashes[target_name], str) or not _HASH.fullmatch(hashes[target_name]):
                    raise ValueError("receipt hash is malformed")
            backup = value["backup_paths"][target_name]
            expected_backup = root / "backup-{0}.bin".format(index)
            if backup is None:
                if value["before_hashes"][target_name] != _sha256(b""):
                    raise ValueError("missing backup has a non-empty hash")
            elif not isinstance(backup, str) or Path(backup) != expected_backup:
                raise ValueError("receipt backup is outside its transaction layout")
        validations_redacted = False
        if isinstance(value["validations"], dict):
            try:
                validations_redacted = json.loads(redact_secrets(json.dumps(value["validations"]))) == value["validations"]
            except (TypeError, ValueError, json.JSONDecodeError):
                validations_redacted = False
        if not isinstance(value["validations"], dict) or not validations_redacted or not isinstance(value["preview"], str) or len(value["preview"]) > MAX_PREVIEW_CHARS + 32 or redact_secrets(value["preview"]) != value["preview"] or not isinstance(value["preview_hash"], str) or not _HASH.fullmatch(value["preview_hash"]):
            raise ValueError("receipt validation or preview is malformed")
        return cls(receipt_path=str(receipt_path), **value)


class Transaction:
    """Create a crash-recoverable journal before any configuration mutation."""

    def __init__(self, receipts_dir=None, health_checker=None, fault_injector=None, receipt_writer=None):
        self.receipts_dir = Path(receipts_dir) if receipts_dir else None
        self.health_checker = health_checker
        self.fault_injector = fault_injector
        self.receipt_writer = receipt_writer
        self._changes = []
        self._preview = ""
        self._preview_hash = ""

    def preview(self, changes):
        if not isinstance(changes, (list, tuple)) or not changes:
            raise ValueError("changes must be a non-empty list")
        prepared, diffs, binding = [], [], []
        seen = set()
        for change in changes:
            if not isinstance(change, dict) or "path" not in change or "content" not in change:
                raise ValueError("each change requires path and content")
            path = _assert_safe_target(change["path"])
            if str(path) in seen:
                raise ValueError("each target may appear only once")
            seen.add(str(path))
            content = change["content"]
            if not isinstance(content, str):
                raise TypeError("change content must be text")
            adapter = change.get("adapter") or ConfigAdapter.detect(path, runtime=change.get("runtime"))
            adapter.validate(content)
            before = _read_regular(path)
            after = content.encode("utf-8")
            before_text = before.decode("utf-8")
            diffs.append("".join(difflib.unified_diff(
                redact_secrets(before_text).splitlines(True), redact_secrets(content).splitlines(True),
                fromfile=str(path), tofile=str(path), lineterm="",
            )))
            binding.append({"path": str(path), "before": _sha256(before), "after": _sha256(after)})
            prepared.append({"path": path, "content": content, "adapter": adapter, "endpoint": change.get("endpoint"), "before_hash": _sha256(before), "mode": (path.stat().st_mode & 0o777) if path.exists() else None})
        self._changes = prepared
        self._preview = self._bounded("\n".join(diffs))
        self._preview_hash = _sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return {"changes": len(prepared), "diff": self._preview, "preview_hash": self._preview_hash, "requires_confirmation": True}

    def apply(self, confirmation):
        if confirmation is not True and confirmation != self._preview_hash:
            raise PermissionError("explicit confirmation for this preview is required")
        if not self._changes:
            raise ValueError("preview changes before applying them")
        for change in self._changes:
            current = _sha256(_read_regular(change["path"]))
            if current != change["before_hash"]:
                raise ValueError("preview is stale; target changed after preview")
        receipt = self._prepare_journal()
        journal_written = False
        mutation_started = False
        try:
            self._persist(receipt)
            journal_written = True
            self._fault("after_pending_receipt")
            for index, change in enumerate(self._changes):
                path = change["path"]
                mutation_started = True
                self._replace_target(path, change["content"].encode("utf-8"), change["adapter"], change["mode"])
                self._fault("after_replace:{0}".format(index))
                change["adapter"].validate(_read_regular(path).decode("utf-8"))
                receipt.after_hashes[str(path)] = _sha256(_read_regular(path))
                receipt.validations[str(path)] = {"pre": True, "post": True, "passed": True}
                self._persist(receipt)
            receipt.validations["health_check"] = self._run_health_checks()
            if receipt.validations["health_check"]["passed"]:
                receipt.status = "applied"
                self._persist(receipt)
            else:
                self._finish_restore(receipt)
        except Exception as error:
            receipt.validations["error"] = {"passed": False, "message": redact_secrets(str(error))}
            if not journal_written and not mutation_started:
                receipt.status = "failed"
                return receipt
            self._finish_restore(receipt)
        return receipt

    def rollback(self, receipt_path):
        return rollback(receipt_path)

    def _prepare_journal(self):
        receipts_dir = _assert_safe_directory(self.receipts_dir or self._changes[0]["path"].parent / ".mlx-agent-receipts", create=True)
        transaction_id = str(uuid.uuid4())
        root = receipts_dir / transaction_id
        os.mkdir(str(root), 0o700)
        _fsync_directory(receipts_dir)
        targets = [str(change["path"]) for change in self._changes]
        receipt = Receipt(
            schema_version=RECEIPT_SCHEMA_VERSION, transaction_id=transaction_id, adapter_version=ConfigAdapter.version,
            timestamp=_timestamp(), transaction_root=str(root), targets=targets,
            target_roots={str(change["path"]): str(change["path"].parent) for change in self._changes},
            target_modes={str(change["path"]): change["mode"] for change in self._changes},
            before_hashes={}, after_hashes={}, backup_paths={}, validations={}, status="pending",
            preview=self._preview, preview_hash=self._preview_hash, receipt_path=str(root / "receipt.json"),
        )
        for index, change in enumerate(self._changes):
            path = change["path"]
            before = _read_regular(path)
            target = str(path)
            receipt.before_hashes[target] = _sha256(before)
            receipt.after_hashes[target] = _sha256(before)
            if change["mode"] is not None:
                backup = root / "backup-{0}.bin".format(index)
                _atomic_in_directory(root, backup.name, before, change["mode"] or 0o600)
                receipt.backup_paths[target] = str(backup)
            else:
                receipt.backup_paths[target] = None
        return receipt

    def _replace_target(self, path, content, adapter, mode):
        _assert_safe_target(path)
        _atomic_in_directory(path.parent, path.name, content, mode if mode is not None else 0o600, adapter.validate)

    def _persist(self, receipt):
        root = _assert_safe_directory(receipt.transaction_root)
        if self.receipt_writer is not None:
            self.receipt_writer(Path(receipt.receipt_path), receipt.to_dict())
            return
        _atomic_in_directory(root, "receipt.json", (json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8"), 0o600)

    def _finish_restore(self, receipt):
        if _restore_receipt(receipt):
            receipt.status = "rolled_back"
        else:
            receipt.status = "recovery_required"
        try:
            self._persist(receipt)
        except Exception as error:
            receipt.status = "recovery_required"
            receipt.validations["receipt_write"] = {"passed": False, "message": redact_secrets(str(error))}

    def _run_health_checks(self):
        checks = []
        for change in self._changes:
            endpoint = change.get("endpoint")
            if not endpoint:
                continue
            passed = bool(self.health_checker(endpoint)) if self.health_checker else ConfigAdapter.health_check(endpoint)
            checks.append({"endpoint": redact_endpoint(endpoint), "passed": passed})
        return {"passed": all(item["passed"] for item in checks), "endpoints": checks}

    def _fault(self, point):
        if self.fault_injector is not None:
            self.fault_injector(point)

    @staticmethod
    def _bounded(value):
        return value if len(value) <= MAX_PREVIEW_CHARS else value[:MAX_PREVIEW_CHARS] + "\n... [preview truncated]"


def _restore_receipt(receipt):
    """Preflight every backup, then restore and prove byte-for-byte hashes."""
    planned = []
    try:
        root = _assert_safe_directory(receipt.transaction_root)
        for index, target_name in enumerate(receipt.targets):
            target = _assert_safe_target(target_name)
            expected = receipt.before_hashes[target_name]
            backup_name = receipt.backup_paths[target_name]
            if backup_name is None:
                if expected != _sha256(b""):
                    raise ValueError("missing backup has a non-empty hash")
                planned.append((target, None, receipt.target_modes[target_name], expected))
                continue
            backup = _assert_safe_target(backup_name)
            if backup.parent != root or backup.name != "backup-{0}.bin".format(index) or not backup.exists():
                raise ValueError("backup is missing or outside transaction root")
            data = backup.read_bytes()
            if _sha256(data) != expected:
                raise ValueError("backup hash does not match receipt")
            planned.append((target, data, receipt.target_modes[target_name], expected))
        for target, data, mode, expected in planned:
            if data is None:
                if target.exists():
                    _remove_target(target)
            else:
                _atomic_in_directory(target.parent, target.name, data, mode if mode is not None else 0o600)
            actual = _sha256(_read_regular(target))
            if actual != expected:
                raise ValueError("restore hash does not match receipt")
            receipt.after_hashes[str(target)] = actual
        receipt.validations["rollback"] = {"passed": True, "targets": [str(item[0]) for item in planned]}
        return True
    except Exception as error:
        receipt.validations["rollback"] = {"passed": False, "message": redact_secrets(str(error))}
        return False


def _remove_target(path):
    _assert_safe_target(path)
    if not path.exists():
        return
    directory = _open_directory(path.parent)
    try:
        item = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(item.st_mode):
            raise ValueError("refusing non-regular target removal")
        os.unlink(path.name, dir_fd=directory)
        _fsync_directory(directory)
    finally:
        os.close(directory)


def rollback(receipt_path):
    location = _assert_safe_target(receipt_path)
    receipt = Receipt.from_dict(json.loads(location.read_text(encoding="utf-8")), str(location))
    if receipt.status == "rolled_back":
        return receipt
    if _restore_receipt(receipt):
        receipt.status = "rolled_back"
    else:
        receipt.status = "rollback_failed"
    try:
        _atomic_in_directory(Path(receipt.transaction_root), "receipt.json", (json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8"), 0o600)
    except Exception as error:
        receipt.status = "recovery_required"
        receipt.validations["receipt_write"] = {"passed": False, "message": redact_secrets(str(error))}
    return receipt
