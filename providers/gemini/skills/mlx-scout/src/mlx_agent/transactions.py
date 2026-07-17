"""Durable, reversible configuration transactions for Wire.

The transaction root is opened with ``O_NOFOLLOW`` where macOS/Python exposes
it.  Stage, backup, receipt, and replace operations then use that directory
file descriptor.  A hostile actor can still race an already-open target leaf
between validation and replacement, but ``os.replace`` replaces that leaf and
never follows it; ancestors are verified before opening their directory fd.
"""

from __future__ import annotations

import difflib
import errno
import fcntl
import hashlib
import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager, nullcontext

from .wiring import ConfigAdapter, redact_endpoint, redact_secrets


RECEIPT_SCHEMA_VERSION = "2.0"
MAX_PREVIEW_CHARS = 12000
_STATUSES = {"pending", "applied", "failed", "rolled_back", "rollback_failed", "recovery_required"}
_HASH = __import__("re").compile(r"^[0-9a-f]{64}$")
COOPERATIVE_CONCURRENCY_NOTE = "Advisory lock protects accidental/cooperative writers; a malicious process ignoring it can still race the final rename."
LEGACY_LOCK_MIGRATION = "legacy-target-locks-v1"
_LEGACY_LOCK_MARKER = "legacy-lock-migration-v1.json"


class ConcurrentTransactionError(ValueError):
    """A cooperative writer holds the transaction's advisory lock."""


class LegacyLockError(ConcurrentTransactionError):
    """A legacy target-adjacent lock prevents a safe scoped-lock upgrade."""


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _absolute(path):
    value = Path(path)
    if _has_parent_reference(value):
        raise ValueError("path traversal is not allowed: {0}".format(path))
    return Path(os.path.abspath(str(value)))


def _physical_absolute(path):
    result = _absolute(path)
    # /var is a fixed macOS compatibility alias. Normalize before descriptor
    # traversal rather than following a user-controlled symlink component.
    if str(result) == "/var" or str(result).startswith("/var/"):
        if not os.path.islink("/var") or os.readlink("/var") != "private/var":
            raise ValueError("untrusted /var compatibility alias")
        result = Path("/private/var") / result.relative_to("/var")
    return result


def _has_parent_reference(path):
    return ".." in Path(path).parts


def _walk_directory(path, create=False, component_hook=None):
    """Open each absolute component from `/` using dir-fd + O_NOFOLLOW only."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks required no-follow directory traversal")
    logical = _absolute(path)
    value = _physical_absolute(logical)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in value.parts[1:]:
            if component_hook is not None:
                component_hook(descriptor, component)
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise ValueError("path component does not exist: {0}".format(component))
                os.mkdir(component, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError("refusing unsafe directory component {0}: {1}".format(component, error))
            os.close(descriptor)
            descriptor = child
        return logical, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_safe_directory(path, create=False):
    value, descriptor = _walk_directory(path, create=create)
    os.close(descriptor)
    return value


def _target_lock_name(target):
    return ".mlx-agent-wire-{0}.lock".format(_target_lock_digest(target))


def _target_lock_digest(target):
    """Identity shared by legacy locks, scoped locks, and migration state."""
    return _sha256(str(_physical_absolute(target)).encode("utf-8"))


def _legacy_lock_entries(targets):
    """Open existing legacy locks without creating target parents or lock files."""
    held = []
    for target in sorted({_physical_absolute(item) for item in targets}, key=str):
        try:
            _parent, directory_fd = _walk_directory(target.parent)
        except ValueError as error:
            if str(error).startswith("path component does not exist:"):
                continue
            raise
        descriptor = None
        name = _target_lock_name(target)
        try:
            try:
                descriptor = os.open(name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory_fd)
            except FileNotFoundError:
                os.close(directory_fd)
                continue
            except OSError as error:
                raise ValueError("could not safely open legacy target transaction lock: {0}".format(error))
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("legacy target transaction lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    raise LegacyLockError("legacy_lock_busy: stop older mlx-agent processes before upgrading scoped locks")
                raise ValueError("could not safely acquire legacy target transaction lock: {0}".format(error))
            held.append((descriptor, directory_fd, name, str(target.parent / name), _target_lock_digest(target)))
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory_fd)
            for held_descriptor, held_directory, _held_name, _held_path, _held_digest in reversed(held):
                try:
                    fcntl.flock(held_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(held_descriptor)
                    os.close(held_directory)
            raise
    return held


def _read_legacy_migration_state(lock_root):
    """Read validated target-specific scoped legacy-lock migration state."""
    try:
        _root, directory_fd = _walk_directory(lock_root)
    except ValueError as error:
        if str(error).startswith("path component does not exist:"):
            return {"version": LEGACY_LOCK_MIGRATION, "targets": {}}
        raise
    descriptor = None
    try:
        try:
            descriptor = os.open(_LEGACY_LOCK_MARKER, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        except FileNotFoundError:
            return {"version": LEGACY_LOCK_MIGRATION, "targets": {}}
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("legacy lock migration marker is not a regular file")
        value = json.loads(os.read(descriptor, 1048576).decode("utf-8"))
        # The prior release-candidate marker had no target map. Treat it as
        # an empty target-specific map: it cannot prove which targets migrated.
        if value == {"version": LEGACY_LOCK_MIGRATION}:
            return {"version": LEGACY_LOCK_MIGRATION, "targets": {}}
        if not isinstance(value, dict) or set(value) != {"version", "targets"} or value["version"] != LEGACY_LOCK_MIGRATION or not isinstance(value["targets"], dict):
            raise ValueError("legacy lock migration marker is malformed")
        for digest, entry in value["targets"].items():
            if not isinstance(digest, str) or not _HASH.fullmatch(digest) or not isinstance(entry, dict) or set(entry) != {"migrated_at"} or not isinstance(entry["migrated_at"], str):
                raise ValueError("legacy lock migration marker is malformed")
            try:
                if datetime.fromisoformat(entry["migrated_at"].replace("Z", "+00:00")).tzinfo is None:
                    raise ValueError("missing timezone")
            except ValueError:
                raise ValueError("legacy lock migration marker is malformed")
        return value
    except OSError as error:
        raise ValueError("could not safely read legacy lock migration marker: {0}".format(error))
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _write_legacy_migration_state(lock_root, state):
    _atomic_in_directory(
        lock_root,
        _LEGACY_LOCK_MARKER,
        (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        0o600,
    )


@contextmanager
def _legacy_migration_window(lock_root):
    """Serialize target-map upgrades so multi-target writes cannot lose entries."""
    _root, directory_fd = _walk_directory(lock_root, create=True)
    descriptor = None
    try:
        descriptor = os.open(".legacy-lock-migration-v1.state.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("legacy lock migration state lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise LegacyLockError("legacy_lock_busy: another scoped legacy-lock migration is active")
            raise ValueError("could not safely acquire legacy lock migration state lock: {0}".format(error))
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(directory_fd)


@contextmanager
def _migrate_legacy_target_locks(targets, lock_root):
    """Upgrade target-adjacent legacy locks before entering scoped lock storage."""
    held = _legacy_lock_entries(targets)
    try:
        canonical = sorted({_physical_absolute(item) for item in targets}, key=str)
        with _legacy_migration_window(lock_root):
            state = _read_legacy_migration_state(lock_root)
            migrated = state["targets"]
            if any(digest in migrated for _descriptor, _directory_fd, _name, _path, digest in held):
                raise LegacyLockError("legacy_lock_recreated: stop older mlx-agent processes and remove recreated legacy locks before continuing")
            for _descriptor, directory_fd, name, _path, _digest in held:
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            changed = False
            for target in canonical:
                digest = _target_lock_digest(target)
                if digest not in migrated:
                    migrated[digest] = {"migrated_at": _timestamp()}
                    changed = True
            _write_legacy_migration_state(lock_root, state)
            yield
    finally:
        for descriptor, directory_fd, _name, _path, _digest in reversed(held):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                os.close(directory_fd)


def legacy_lock_problem(targets, lock_root):
    """Describe legacy-lock state for installer doctor without mutating it."""
    migrated = _read_legacy_migration_state(lock_root)["targets"]
    paths = {"legacy_lock_recreated": [], "legacy_lock_migration_required": []}
    for target in sorted({_physical_absolute(item) for item in targets}, key=str):
        try:
            _parent, directory_fd = _walk_directory(target.parent)
        except ValueError as error:
            if str(error).startswith("path component does not exist:"):
                continue
            raise
        try:
            name = _target_lock_name(target)
            try:
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise ValueError("could not safely open legacy target transaction lock: {0}".format(error))
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError("legacy target transaction lock is not a regular file")
                code = "legacy_lock_recreated" if _target_lock_digest(target) in migrated else "legacy_lock_migration_required"
                paths[code].append(str(target.parent / name))
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_fd)
    return [
        {
            "code": code,
            "paths": affected,
            "remediation": "stop older mlx-agent processes; after target-specific scoped-lock migration, do not run an older binary and remove only recreated legacy locks.",
        }
        for code, affected in paths.items() if affected
    ]


@contextmanager
def _target_locks(targets, create_parents=False, lock_root=None):
    """Acquire target-adjacent physical locks in canonical order to avoid deadlocks."""
    canonical = sorted({_physical_absolute(target) for target in targets}, key=str)
    held = []
    try:
        for target in canonical:
            if lock_root is None:
                _parent, directory_fd = _walk_directory(target.parent, create=create_parents)
            else:
                _target_parent, target_parent_fd = _walk_directory(target.parent, create=create_parents)
                os.close(target_parent_fd)
                _parent, directory_fd = _walk_directory(lock_root, create=True)
            lock_fd = None
            try:
                lock_fd = os.open(
                    _target_lock_name(target),
                    os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
                if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                    raise ValueError("target transaction lock is not a regular file")
                os.fsync(directory_fd)
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append((lock_fd, directory_fd))
            except OSError as error:
                if lock_fd is not None:
                    os.close(lock_fd)
                os.close(directory_fd)
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    raise ConcurrentTransactionError("another cooperative Wire transaction is active for a target")
                raise ValueError("could not safely acquire target transaction lock: {0}".format(error))
            except BaseException:
                if lock_fd is not None:
                    os.close(lock_fd)
                os.close(directory_fd)
                raise
        yield canonical
    finally:
        for lock_fd, directory_fd in reversed(held):
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
                os.close(directory_fd)


def _assert_safe_target(path):
    value = _absolute(path)
    _read_target(value)
    return value


def _open_directory(path):
    _value, descriptor = _walk_directory(path)
    return descriptor


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


def _read_target(path, component_hook=None):
    """Read a regular leaf through its opened parent directory descriptor."""
    value = _absolute(path)
    _parent, directory = _walk_directory(value.parent, component_hook=component_hook)
    try:
        try:
            descriptor = os.open(value.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except FileNotFoundError:
            return b"", False, None
        try:
            item = os.fstat(descriptor)
            if not stat.S_ISREG(item.st_mode):
                raise ValueError("target is not a regular file: {0}".format(value))
            parts = []
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                parts.append(chunk)
            return b"".join(parts), True, item.st_mode & 0o777
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError("refusing unsafe target leaf {0}: {1}".format(value, error))
    finally:
        os.close(directory)


def _read_regular(path):
    return _read_target(path)[0]


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
    lock_root: str = None
    lock_migration: str = None
    receipt_path: str = field(default="", repr=False, compare=False)

    def to_dict(self):
        value = asdict(self)
        value.pop("receipt_path", None)
        if value.get("lock_root") is None:
            value.pop("lock_root")
        if value.get("lock_migration") is None:
            value.pop("lock_migration")
        return value

    @classmethod
    def from_dict(cls, value, receipt_path=""):
        required = {
            "schema_version", "transaction_id", "adapter_version", "timestamp", "transaction_root",
            "targets", "target_roots", "target_modes", "before_hashes", "after_hashes", "backup_paths",
            "validations", "status", "preview", "preview_hash",
        }
        if not isinstance(value, dict) or not required.issubset(value) or set(value) - (required | {"lock_root", "lock_migration"}):
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
        lock_root = value.get("lock_root")
        if lock_root is not None:
            if not isinstance(lock_root, str) or not lock_root:
                raise ValueError("receipt lock root is malformed")
            safe_lock_root = _assert_safe_directory(lock_root)
            if str(safe_lock_root) != lock_root:
                raise ValueError("receipt lock root must be normalized")
        migration = value.get("lock_migration")
        if migration is not None and (lock_root is None or migration != LEGACY_LOCK_MIGRATION):
            raise ValueError("receipt lock migration is malformed")
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

    def __init__(self, receipts_dir=None, health_checker=None, fault_injector=None, receipt_writer=None, path_race_hook=None, create_target_parents=False, transaction_id=None, lock_root=None):
        self.receipts_dir = Path(receipts_dir) if receipts_dir else None
        self.lock_root = Path(lock_root) if lock_root is not None else None
        self.health_checker = health_checker
        self.fault_injector = fault_injector
        self.receipt_writer = receipt_writer
        self.path_race_hook = path_race_hook
        self.create_target_parents = bool(create_target_parents)
        if transaction_id is not None:
            try:
                self.transaction_id = str(uuid.UUID(str(transaction_id)))
            except (TypeError, ValueError, AttributeError):
                raise ValueError("transaction_id must be a UUID")
        else:
            self.transaction_id = None
        self._changes = []
        self._preview = ""
        self._preview_hash = ""
        self._expected_current = None

    def preview(self, changes, expected_current=None):
        if not isinstance(changes, (list, tuple)) or not changes:
            raise ValueError("changes must be a non-empty list")
        prepared, diffs, binding = [], [], []
        seen = set()
        for change in changes:
            if not isinstance(change, dict) or "path" not in change or "content" not in change:
                raise ValueError("each change requires path and content")
            path = _absolute(change["path"])
            if str(path) in seen:
                raise ValueError("each target may appear only once")
            seen.add(str(path))
            content = change["content"]
            if not isinstance(content, str):
                raise TypeError("change content must be text")
            adapter = change.get("adapter") or ConfigAdapter.detect(path, runtime=change.get("runtime"))
            adapter.validate(content)
            try:
                before, existed, mode = _read_target(path, self.path_race_hook)
            except ValueError as error:
                if not self.create_target_parents or not str(error).startswith("path component does not exist:"):
                    raise
                before, existed, mode = b"", False, None
            after = content.encode("utf-8")
            before_text = before.decode("utf-8")
            diffs.append("".join(difflib.unified_diff(
                redact_secrets(before_text).splitlines(True), redact_secrets(content).splitlines(True),
                fromfile=str(path), tofile=str(path), lineterm="",
            )))
            binding.append({"path": str(path), "before": _sha256(before), "after": _sha256(after), "exists": existed, "mode": mode, "endpoint": _sha256(str(change.get("endpoint") or "").encode("utf-8"))})
            prepared.append({"path": path, "content": content, "adapter": adapter, "endpoint": change.get("endpoint"), "before_hash": _sha256(before), "existed": existed, "mode": mode})
        self._changes = prepared
        self._expected_current = self._normalize_expected_current(expected_current)
        self._preview = self._bounded("\n".join(diffs))
        self._preview_hash = _sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return {"changes": len(prepared), "diff": self._preview, "preview_hash": self._preview_hash, "requires_confirmation": True}

    def apply(self, confirmation, expected_current=None):
        if confirmation is not True and confirmation != self._preview_hash:
            raise PermissionError("explicit confirmation for this preview is required")
        if not self._changes:
            raise ValueError("preview changes before applying them")
        if expected_current is not None:
            expected_current = self._normalize_expected_current(expected_current)
            if self._expected_current is not None and expected_current != self._expected_current:
                raise ValueError("apply expected_current does not match preview")
            self._expected_current = expected_current
        with self._advisory_lock():
            return self._apply_locked()

    @contextmanager
    def _advisory_lock(self):
        lock_root = self._safe_lock_root() if self.lock_root is not None else None
        if lock_root is None:
            with _target_locks(self._lock_targets(), create_parents=self.create_target_parents):
                yield
            return
        with _migrate_legacy_target_locks(self._lock_targets(), lock_root):
            with _target_locks(self._lock_targets(), create_parents=self.create_target_parents, lock_root=lock_root):
                yield

    def _safe_lock_root(self):
        root, descriptor = _walk_directory(self.lock_root, create=True)
        os.close(descriptor)
        return root

    def _lock_targets(self):
        return [str(path) for path in sorted({change["path"] for change in self._changes}, key=str)]

    def _apply_locked(self):
        if self._expected_current is not None:
            for change in self._changes:
                current, exists, mode = _read_target(change["path"], self.path_race_hook)
                expected = self._expected_current[str(change["path"])]
                if _sha256(current) != expected["hash"] or exists != expected["exists"] or mode != expected["mode"]:
                    raise ValueError("expected current target state changed before mutation")
        for change in self._changes:
            current, exists, mode = _read_target(change["path"], self.path_race_hook)
            if _sha256(current) != change["before_hash"] or exists != change["existed"] or mode != change["mode"]:
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
                self._fault("before_replace:{0}".format(index))
                current, exists, mode = _read_target(path, self.path_race_hook)
                if _sha256(current) != receipt.before_hashes[str(path)] or exists != change["existed"] or mode != receipt.target_modes[str(path)]:
                    raise ValueError("preview is stale; target changed before replacement")
                mutation_started = True
                self._replace_target(path, change["content"].encode("utf-8"), change["adapter"], receipt.target_modes[str(path)])
                self._fault("after_replace:{0}".format(index))
                change["adapter"].validate(_read_regular(path).decode("utf-8"))
                after, after_exists, after_mode = _read_target(path)
                if not after_exists or (receipt.target_modes[str(path)] is not None and after_mode != receipt.target_modes[str(path)]):
                    raise ValueError("replacement mode does not match the reviewed target mode")
                receipt.after_hashes[str(path)] = _sha256(after)
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

    def _normalize_expected_current(self, expected_current):
        if expected_current is None:
            return None
        if not isinstance(expected_current, dict):
            raise ValueError("expected_current must map every target to its current state")
        targets = {str(change["path"]) for change in self._changes}
        if set(expected_current) != targets:
            raise ValueError("expected_current targets do not match the preview")
        normalized = {}
        for target in sorted(targets):
            state = expected_current[target]
            if not isinstance(state, dict) or set(state) != {"hash", "exists", "mode"}:
                raise ValueError("expected_current state must include hash, exists, and mode")
            if not isinstance(state["hash"], str) or not _HASH.match(state["hash"]):
                raise ValueError("expected_current hash is invalid")
            if not isinstance(state["exists"], bool) or (state["mode"] is not None and not isinstance(state["mode"], int)):
                raise ValueError("expected_current existence or mode is invalid")
            normalized[target] = {"hash": state["hash"], "exists": state["exists"], "mode": state["mode"]}
        return normalized

    def rollback(self, receipt_path):
        return rollback(receipt_path)

    def _prepare_journal(self):
        self._fault("before_journal_capture")
        captured = []
        for change in self._changes:
            before, exists, mode = _read_target(change["path"], self.path_race_hook)
            if _sha256(before) != change["before_hash"] or exists != change["existed"] or mode != change["mode"]:
                raise ValueError("preview is stale; target changed before journal capture")
            captured.append((before, exists, mode))
        self._fault("after_journal_capture")
        for index, change in enumerate(self._changes):
            before, exists, mode = _read_target(change["path"], self.path_race_hook)
            if _sha256(before) != change["before_hash"] or exists != change["existed"] or mode != change["mode"]:
                raise ValueError("preview is stale; target changed after journal capture")
            captured[index] = (before, exists, mode)
        receipts_dir, receipts_fd = _walk_directory(self.receipts_dir or self._changes[0]["path"].parent / ".mlx-agent-receipts", create=True)
        transaction_id = self.transaction_id or str(uuid.uuid4())
        root = receipts_dir / transaction_id
        try:
            self._fault("before_transaction_root_create")
            os.mkdir(transaction_id, 0o700, dir_fd=receipts_fd)
            os.fsync(receipts_fd)
            root_fd = os.open(transaction_id, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW, dir_fd=receipts_fd)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
        finally:
            os.close(receipts_fd)
        targets = [str(change["path"]) for change in self._changes]
        receipt = Receipt(
            schema_version=RECEIPT_SCHEMA_VERSION, transaction_id=transaction_id, adapter_version=ConfigAdapter.version,
            timestamp=_timestamp(), transaction_root=str(root), targets=targets,
            target_roots={str(change["path"]): str(change["path"].parent) for change in self._changes},
            target_modes={str(change["path"]): captured[index][2] for index, change in enumerate(self._changes)},
            before_hashes={}, after_hashes={}, backup_paths={}, validations={
                "concurrency": {"scope": "advisory_cooperative_target_scoped", "note": COOPERATIVE_CONCURRENCY_NOTE, "passed": True},
            }, status="pending",
            preview=self._preview, preview_hash=self._preview_hash,
            lock_root=str(self._safe_lock_root()) if self.lock_root is not None else None,
            lock_migration=LEGACY_LOCK_MIGRATION if self.lock_root is not None else None,
            receipt_path=str(root / "receipt.json"),
        )
        for index, change in enumerate(self._changes):
            path = change["path"]
            before, exists, mode = captured[index]
            target = str(path)
            receipt.before_hashes[target] = _sha256(before)
            receipt.after_hashes[target] = _sha256(before)
            if exists:
                backup = root / "backup-{0}.bin".format(index)
                _atomic_in_directory(root, backup.name, before, mode or 0o600)
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
            backup_data, backup_exists, _backup_mode = _read_target(backup)
            if backup.parent != root or backup.name != "backup-{0}.bin".format(index) or not backup_exists:
                raise ValueError("backup is missing or outside transaction root")
            if _sha256(backup_data) != expected:
                raise ValueError("backup hash does not match receipt")
            planned.append((target, backup_data, receipt.target_modes[target_name], expected))
        for target, data, mode, expected in planned:
            if data is None:
                _current, exists, _current_mode = _read_target(target)
                if exists:
                    _remove_target(target)
            else:
                _atomic_in_directory(target.parent, target.name, data, mode if mode is not None else 0o600)
            actual_content, actual_exists, actual_mode = _read_target(target)
            actual = _sha256(actual_content)
            if actual != expected or actual_exists != (mode is not None) or actual_mode != mode:
                raise ValueError("restore hash does not match receipt")
            receipt.after_hashes[str(target)] = actual
        receipt.validations["rollback"] = {"passed": True, "targets": [str(item[0]) for item in planned]}
        return True
    except Exception as error:
        receipt.validations["rollback"] = {"passed": False, "message": redact_secrets(str(error))}
        return False


def _remove_target(path):
    _assert_safe_target(path)
    _content, exists, _mode = _read_target(path)
    if not exists:
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


def rollback(receipt_path, expected_after_hashes=None):
    """Restore a receipt, optionally proving reviewed current hashes under its lock."""
    location = _assert_safe_target(receipt_path)
    initial = Receipt.from_dict(json.loads(_read_regular(location).decode("utf-8")), str(location))
    lock_context = (
        _migrate_legacy_target_locks(initial.targets, initial.lock_root)
        if initial.lock_root is not None else nullcontext()
    )
    with lock_context:
        with _target_locks(initial.targets, lock_root=initial.lock_root):
            receipt = Receipt.from_dict(json.loads(_read_regular(location).decode("utf-8")), str(location))
            if expected_after_hashes is not None:
                if not isinstance(expected_after_hashes, dict) or set(expected_after_hashes) != set(receipt.targets):
                    raise ValueError("reviewed rollback hashes do not match receipt targets")
                for target_name in receipt.targets:
                    current, exists, _mode = _read_target(target_name)
                    if not exists or _sha256(current) != expected_after_hashes[target_name]:
                        raise ValueError("receipt target changed after reviewed rollback preview")
            if receipt.status == "rolled_back":
                current_matches = True
                for target_name in receipt.targets:
                    current, exists, mode = _read_target(target_name)
                    expected_absent = receipt.backup_paths[target_name] is None
                    if _sha256(current) != receipt.before_hashes[target_name] or exists == expected_absent or mode != receipt.target_modes[target_name]:
                        current_matches = False
                        break
                if current_matches:
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
