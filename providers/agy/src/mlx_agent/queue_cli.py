"""Versioned, process-serialized conversion queue operations. Stdlib only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from mlx_agent.compute import compute_root
from mlx_agent.convert import load_receipts, status_convert
from mlx_agent.convert_queue import ConvertQueue, QueueStore, QueueOperationError, QueuePersistenceError, _validated_items
from mlx_agent import queue_bridge
from mlx_agent.transactions import _assert_safe_directory, _assert_safe_target, _open_directory

MAXIMUM_BYTES = 8 * 1024 * 1024
_FIELDS = {
    "snapshot": set(), "run": set(), "clear": set(),
    "cancel": {"id"}, "retry": {"id"}, "move": {"id", "direction"},
    "enqueue": {"kind", "preview_hash", "q_bits", "out", "path", "repo", "hf_cache", "label"},
    "adopt-preview": {"source"}, "adopt": {"source", "preview_hash"},
}


def default_path():
    return compute_root().parent / "queues/convert-queue.json"


def _read(path):
    location = _assert_safe_target(path)
    descriptor = os.open(location, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAXIMUM_BYTES:
            raise ValueError("Queue input must be a bounded regular file.")
        chunks, size = [], 0
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            size += len(chunk)
            if size > MAXIMUM_BYTES:
                raise ValueError("Queue input exceeded its limit.")
            chunks.append(chunk)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino,
            os.fstat(descriptor).st_size, os.fstat(descriptor).st_mtime_ns):
            raise ValueError("Queue input changed while reading.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@contextmanager
def _guard(path):
    _assert_safe_directory(path.parent, create=True)
    directory = _open_directory(path.parent)
    descriptor = None
    try:
        descriptor = os.open(path.name + ".lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError("Queue lock must be an owned regular file.")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _assert_safe_target(path)
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _adoption(source, target, receipts_dir):
    if target.exists() or target.is_symlink():
        raise ValueError("Queue adoption never replaces an existing queue.")
    source = Path(source).expanduser()
    payload = _read(source)
    items = _validated_items(json.loads(payload))
    if any(item["state"] == "starting" for item in items):
        raise ValueError("Reconcile starting conversions before transferring queue ownership.")
    if any(item["state"] in ("running", "unknown") for item in status_convert(receipts_dir)):
        raise ValueError("Stop active conversions before transferring queue ownership.")
    digest = hashlib.sha256(payload).hexdigest()
    plan = {"source": str(source), "queue_path": str(target), "source_digest": digest,
            "item_count": len(items), "preview_hash": hashlib.sha256(
                (str(source) + "\\0" + str(target) + "\\0" + digest).encode()).hexdigest()}
    return plan, payload, items


def execute(request, path=None, receipts_dir=None):
    if not isinstance(request, dict) or request.get("operation") not in _FIELDS:
        raise ValueError("Unknown conversion queue operation.")
    operation = request["operation"]
    if set(request) - (_FIELDS[operation] | {"operation"}):
        raise ValueError("Unexpected conversion queue fields.")
    required = {"enqueue": {"kind", "preview_hash", "q_bits"}, "cancel": {"id"},
                "retry": {"id"}, "move": {"id", "direction"},
                "adopt-preview": {"source"}, "adopt": {"source", "preview_hash"}}.get(operation, set())
    if not required <= set(request):
        raise ValueError("Missing conversion queue fields.")
    location = Path(path).expanduser() if path is not None else default_path()
    with _guard(location):
        if operation in ("adopt-preview", "adopt"):
            plan, original, items = _adoption(request["source"], location, receipts_dir)
            if operation == "adopt-preview":
                return plan
            if request["preview_hash"] != plan["preview_hash"]:
                raise ValueError("Queue adoption preview changed.")
            backup = location.with_name(location.name + "." + plan["source_digest"] + ".backup")
            if backup.exists():
                if _read(backup) != original:
                    raise ValueError("Queue backup does not match the source.")
            else:
                _write_backup(backup, original)
            try:
                QueueStore(location).save(items)
                if _read(Path(request["source"])) != original:
                    location.unlink()
                    raise ValueError("Source queue changed during adoption; retry from a fresh preview.")
            except BaseException:
                if location.exists():
                    location.unlink()
                raise
            return {**plan, "status": "adopted", "backup": str(backup),
                    "conversion_queue": {"schema_version": "1.1", "items": items}}
        if location.exists() and location.stat().st_size > MAXIMUM_BYTES:
            raise ValueError("Queue state exceeds the supported size.")
        queue = ConvertQueue(location)
        if queue.load_error:
            raise QueuePersistenceError(queue.load_error["code"], queue.load_error["message"],
                                        queue.load_error["remediation"], location,
                                        queue.load_error.get("preserved_path"))
        result = {}
        if operation == "enqueue":
            result["item"] = queue.enqueue(**{key: value for key, value in request.items() if key != "operation"})
        elif operation == "cancel":
            result["changed"] = queue.cancel(request["id"])
        elif operation == "retry":
            result["item"] = queue.retry(request["id"])
        elif operation == "move":
            result["changed"] = queue.move(request["id"], request["direction"])
        elif operation == "clear":
            result["removed_count"] = queue.clear()
        elif operation == "run":
            queue_bridge._receipts_directory = receipts_dir
            receipts = load_receipts(receipts_dir)
            for item in queue.snapshot():
                if item["state"] == "queued" and any(queue._receipt_matches(item, receipt) for receipt in receipts):
                    queue.cancel(item["id"])
            result = queue.try_start_next(str(Path(__file__).resolve().parents[2])) or {"status": "waiting"}
        return {**result, "queue_path": str(location),
                "conversion_queue": {"schema_version": "1.1", "items": queue.snapshot()}}


def _write_backup(backup, original):
    descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database")
    parser.add_argument("--artifacts")
    parser.add_argument("--path", type=Path)
    parser.add_argument("--receipts-dir")
    parser.add_argument("queue", choices=["queue"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        data = execute(json.loads(_read(arguments.request)), arguments.path, arguments.receipts_dir)
        print(json.dumps({"schema_version": "queue.v1", "ok": True, "data": data}))
        return 0
    except (ValueError, TypeError, KeyError, OSError, QueueOperationError, QueuePersistenceError, queue_bridge.BridgeError) as error:
        detail = error.to_dict() if hasattr(error, "to_dict") else {
            "code": "queue_rejected", "message": str(error), "remediation": "Inspect the queue request and retained state."}
        print(json.dumps({"schema_version": "queue.v1", "ok": False, "error": detail}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
