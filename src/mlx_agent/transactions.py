"""Atomic, reversible configuration transactions used by Wire and installers."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .wiring import ConfigAdapter, redact_secrets


RECEIPT_SCHEMA_VERSION = "1.0"
MAX_PREVIEW_CHARS = 12000


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Receipt:
    schema_version: str
    transaction_id: str
    adapter_version: str
    timestamp: str
    targets: list
    before_hashes: dict
    after_hashes: dict
    backup_paths: dict
    validations: dict
    status: str
    preview: str = ""
    receipt_path: str = field(default="", repr=False, compare=False)

    def to_dict(self):
        value = asdict(self)
        value.pop("receipt_path", None)
        return value

    @classmethod
    def from_dict(cls, value, receipt_path=""):
        required = {
            "schema_version", "transaction_id", "adapter_version", "timestamp", "targets",
            "before_hashes", "after_hashes", "backup_paths", "validations", "status",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError("receipt is missing fields: {0}".format(", ".join(sorted(missing))))
        if value["schema_version"] != RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported receipt schema version")
        return cls(receipt_path=receipt_path, **{key: value[key] for key in required}, preview=value.get("preview", ""))


class Transaction:
    """Preview, atomically apply, and precisely restore a set of text config changes."""

    def __init__(self, receipts_dir=None, health_checker=None):
        self.receipts_dir = Path(receipts_dir) if receipts_dir else None
        self.health_checker = health_checker
        self._changes = []
        self._preview = ""

    def preview(self, changes):
        """Validate a proposed transaction and return a redacted unified diff."""
        if not isinstance(changes, (list, tuple)) or not changes:
            raise ValueError("changes must be a non-empty list")
        prepared = []
        diffs = []
        for change in changes:
            if not isinstance(change, dict):
                raise TypeError("each change must be an object")
            if "path" not in change or "content" not in change:
                raise ValueError("each change requires path and content")
            path = Path(change["path"])
            if path.is_symlink():
                raise ValueError("refusing symlink target: {0}".format(path))
            if path.exists() and not path.is_file():
                raise ValueError("target is not a regular file: {0}".format(path))
            if not path.parent.exists() or path.parent.is_symlink():
                raise ValueError("target directory is unavailable or a symlink: {0}".format(path.parent))
            content = change["content"]
            if not isinstance(content, str):
                raise TypeError("change content must be text")
            adapter = change.get("adapter") or ConfigAdapter.detect(path, runtime=change.get("runtime"))
            adapter.validate(content)  # pre-stage parse validation
            before = path.read_text(encoding="utf-8") if path.exists() else ""
            diff = "".join(difflib.unified_diff(
                redact_secrets(before).splitlines(True),
                redact_secrets(content).splitlines(True),
                fromfile=str(path), tofile=str(path), lineterm="",
            ))
            diffs.append(diff)
            prepared.append({
                "path": path, "content": content, "adapter": adapter,
                "endpoint": change.get("endpoint"), "before": before,
            })
        self._changes = prepared
        self._preview = self._bounded("\n".join(diffs))
        return {"changes": len(prepared), "diff": self._preview, "requires_confirmation": True}

    def apply(self, confirmation):
        """Apply a previewed change set only after explicit boolean confirmation."""
        if confirmation is not True:
            raise PermissionError("explicit confirmation is required before mutation")
        if not self._changes:
            raise ValueError("preview changes before applying them")
        receipt = self._new_receipt()
        applied = []
        try:
            for change in self._changes:
                path = change["path"]
                if path.is_symlink():
                    raise ValueError("refusing symlink target: {0}".format(path))
                before = path.read_bytes() if path.exists() else b""
                receipt.before_hashes[str(path)] = _sha256(before)
                backup = self._backup(path, before)
                receipt.backup_paths[str(path)] = str(backup) if backup else None
                applied.append(change)
                self._atomic_replace(path, change["content"].encode("utf-8"), change["adapter"])
                change["adapter"].validate(path.read_text(encoding="utf-8"))  # post-replace validation
                after = path.read_bytes()
                receipt.after_hashes[str(path)] = _sha256(after)
                receipt.validations[str(path)] = {"pre": True, "post": True, "passed": True}
            health = self._run_health_checks()
            receipt.validations["health_check"] = health
            if not health["passed"]:
                self._restore(applied, receipt)
                receipt.status = "rolled_back"
            else:
                receipt.status = "applied"
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            receipt.validations["error"] = {"passed": False, "message": str(error)}
            self._restore(applied, receipt)
            receipt.status = "rolled_back" if applied else "failed"
        self._write_receipt(receipt)
        return receipt

    def rollback(self, receipt_path):
        return rollback(receipt_path)

    def _new_receipt(self):
        return Receipt(
            schema_version=RECEIPT_SCHEMA_VERSION,
            transaction_id=str(uuid.uuid4()),
            adapter_version=ConfigAdapter.version,
            timestamp=_timestamp(),
            targets=[str(item["path"]) for item in self._changes],
            before_hashes={}, after_hashes={}, backup_paths={}, validations={}, status="pending",
            preview=self._preview,
        )

    def _backup(self, path, before):
        if not path.exists():
            return None
        descriptor, temporary = tempfile.mkstemp(prefix=".mlx-agent-backup-", dir=str(path.parent))
        backup = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(before)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(str(backup), path.stat().st_mode & 0o777)
            return backup
        except BaseException:
            try:
                backup.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_replace(path, content, adapter=None):
        original_mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
        descriptor, temporary = tempfile.mkstemp(prefix=".mlx-agent-stage-", dir=str(path.parent))
        stage = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if adapter is not None:
                adapter.validate(stage.read_text(encoding="utf-8"))
            os.chmod(str(stage), original_mode)
            os.replace(str(stage), str(path))
        except BaseException:
            try:
                stage.unlink()
            except OSError:
                pass
            raise

    def _run_health_checks(self):
        endpoints = [item["endpoint"] for item in self._changes if item.get("endpoint")]
        if not endpoints:
            return {"passed": True, "endpoints": []}
        checks = []
        for endpoint in endpoints:
            passed = bool(self.health_checker(endpoint)) if self.health_checker else ConfigAdapter.health_check(endpoint)
            checks.append({"endpoint": redact_secrets(endpoint), "passed": passed})
        return {"passed": all(item["passed"] for item in checks), "endpoints": checks}

    def _restore(self, changes, receipt):
        for change in reversed(changes):
            path = change["path"]
            backup = receipt.backup_paths.get(str(path))
            if backup:
                source = Path(backup)
                self._atomic_replace(path, source.read_bytes())
                os.chmod(str(path), source.stat().st_mode & 0o777)
            elif path.exists():
                path.unlink()
            receipt.after_hashes[str(path)] = _sha256(path.read_bytes()) if path.exists() else _sha256(b"")

    def _write_receipt(self, receipt):
        directory = self.receipts_dir or self._changes[0]["path"].parent / ".mlx-agent-receipts"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / "{0}.json".format(receipt.transaction_id)
        _atomic_json_write(path, receipt.to_dict())
        receipt.receipt_path = str(path)

    @staticmethod
    def _bounded(value):
        if len(value) <= MAX_PREVIEW_CHARS:
            return value
        return value[:MAX_PREVIEW_CHARS] + "\n... [preview truncated]"


def _atomic_json_write(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".mlx-agent-receipt-", dir=str(path.parent))
    stage = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(stage), 0o600)
        os.replace(str(stage), str(path))
    except BaseException:
        try:
            stage.unlink()
        except OSError:
            pass
        raise


def rollback(receipt_path):
    """Restore exact backup bytes from a receipt and update it atomically."""
    location = Path(receipt_path)
    if location.is_symlink():
        raise ValueError("refusing symlink receipt: {0}".format(location))
    receipt = Receipt.from_dict(json.loads(location.read_text(encoding="utf-8")), str(location))
    if receipt.status == "rolled_back":
        return receipt
    restored = []
    try:
        for target_name in receipt.targets:
            target = Path(target_name)
            if target.is_symlink() or target.parent.is_symlink():
                raise ValueError("refusing symlink rollback target: {0}".format(target))
            backup_name = receipt.backup_paths.get(target_name)
            expected = receipt.before_hashes.get(target_name)
            if backup_name is None:
                if target.exists():
                    target.unlink()
                actual = _sha256(b"")
            else:
                backup = Path(backup_name)
                if backup.is_symlink():
                    raise ValueError("refusing symlink backup: {0}".format(backup))
                content = backup.read_bytes()
                if _sha256(content) != expected:
                    raise ValueError("backup hash does not match receipt for {0}".format(target))
                Transaction._atomic_replace(target, content)
                os.chmod(str(target), backup.stat().st_mode & 0o777)
                actual = _sha256(target.read_bytes())
            if actual != expected:
                raise ValueError("rollback hash does not match receipt for {0}".format(target))
            receipt.after_hashes[target_name] = actual
            restored.append(target_name)
        receipt.status = "rolled_back"
        receipt.validations["rollback"] = {"passed": True, "targets": restored}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        receipt.status = "rollback_failed"
        receipt.validations["rollback"] = {"passed": False, "message": str(error), "targets": restored}
    _atomic_json_write(location, receipt.to_dict())
    return receipt
