"""Per-source admission for immutable managed downloads.

The index locates receipts; it is never evidence that model files are complete.
Every reuse revalidates the original receipt and its current source identities.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import unicodedata

from .atomicio import write_text_atomic
from .source_evidence import completed_fetch, read_json, safe_path
from .serve import _pid_alive


class StorageError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def default_root():
    state = os.environ.get("XDG_STATE_HOME")
    base = Path(state).expanduser() if state else Path.home() / ".local/state"
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute.")
    return base / "mlx-agent/model-storage"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(plan):
    # Companions and fetch recipes belong to a selection, rather than creating
    # another copy of the same primary file under a different key.
    return {"repo": plan["repo"].casefold(), "revision": plan["revision"].lower(),
            "file": plan["file"], "subfolder": plan.get("subfolder")}


def selection(plan):
    return {key: plan.get(key) or None for key in (
        "additional_files", "allow_patterns", "ignore_patterns", "base", "requires", "downloads")}


def _root(value, create=False):
    root = safe_path(value)
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.exists():
        info = root.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError("The storage index must be an owned directory without shared write access.")
    return root


def _entry(root, key):
    path = root / (key + ".json")
    try:
        return read_json(path)
    except FileNotFoundError:
        # A dangling link is an invalid index, never a missing registration.
        if path.is_symlink():
            raise ValueError("The storage index cannot contain symbolic links.")
        return None


def _sources(entry):
    if entry is None:
        return {}
    if entry.get("schema") != 1 or not isinstance(entry.get("sources"), dict):
        raise ValueError("The storage index is incomplete or invalid.")
    return entry["sources"]


def _overlap(source, files, other):
    previous = other["source"]
    if (source["repo"], source["revision"]) != (previous["repo"], previous["revision"]):
        return False
    if source["file"] and previous["file"]:
        return bool({source["file"], *(files["additional_files"] or [])} &
                    {previous["file"], *(other["selection"]["additional_files"] or [])})
    # A snapshot or checkpoint selection can include any file below its prefix.
    prefix = (source.get("subfolder") or "") + "/"
    prior = (previous.get("subfolder") or "") + "/"
    if source["file"]:
        return any(not previous.get("subfolder") or name.startswith(prior)
                   for name in [source["file"], *(files["additional_files"] or [])])
    if previous["file"]:
        return any(not source.get("subfolder") or name.startswith(prefix)
                   for name in [previous["file"], *(other["selection"]["additional_files"] or [])])
    return prefix.startswith(prior) or prior.startswith(prefix) or prefix == "/" or prior == "/"


def inspect(plan, root):
    root = _root(root)
    source, files = identity(plan), selection(plan)
    key, source_key = "sources", _digest(source)
    index = _entry(root, key)
    sources = _sources(index)
    result = {"root": str(root), "key": key, "source_key": source_key, "generation": _digest(index), "mode": "download"}
    entry = sources.get(source_key)
    if entry is None:
        destination = str(safe_path(plan["local_dir"]))
        for other in sources.values():
            occupied = str(safe_path(other["directory"]))
            left, right = (unicodedata.normalize("NFC", value).casefold().rstrip("/") for value in (destination, occupied))
            if left == right or left.startswith(right + "/") or right.startswith(left + "/"):
                raise StorageError("storage_destination_conflict", "The destination overlaps another registered source; choose a separate model directory.")
        if any(_overlap(source, files, other) for other in sources.values()):
            raise StorageError("storage_selection_conflict", "An overlapping file or snapshot selection is already registered at this revision; use its existing source.")
        return result
    if entry.get("source") != source or not isinstance(entry.get("receipt"), str):
        raise ValueError("The storage index entry is incomplete or invalid.")
    if entry.get("selection") != files:
        raise StorageError("storage_selection_conflict", "This primary source is already registered with a different companion or fetch selection.")
    receipt_path = safe_path(entry["receipt"])
    receipt = read_json(receipt_path)
    if identity(receipt) != source or selection(receipt) != files:
        raise ValueError("The storage index does not match its source receipt.")
    try:
        completed_fetch(receipt_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        destination = safe_path(receipt["local_dir"])
        marker_path = receipt_path.with_name(receipt_path.stem + ".done.json")
        if not destination.exists() and receipt.get("marker") == str(marker_path):
            marker = read_json(marker_path)
            bound = ("repo", "revision", "file", "subfolder", "additional_files", "local_dir", "preview_hash", "proof_token", "started_at", "downloads_dir", "downloads")
            pid = receipt.get("pid")
            if (receipt.get("proof_token") and marker.get("exit_status") in ("done", "failed")
                    and all(marker.get(key) == receipt.get(key) for key in bound)
                    and isinstance(pid, int) and pid > 0 and not _pid_alive(pid)):
                # Explicit removal/quarantine of the old source permits a retry,
                # always at its original location, after the worker has stopped.
                result["local_dir"] = str(destination)
                return result
        raise StorageError("storage_source_changed", "The registered source is incomplete, running, missing or changed; inspect its original receipt before downloading again.") from error
    result.update(mode="reuse", receipt=str(receipt_path), local_dir=receipt["local_dir"], fetch_preview_hash=receipt["preview_hash"])
    return result


@contextmanager
def admission(plan):
    reviewed = plan["storage"]
    root = _root(reviewed["root"], create=True)
    lock = safe_path(root / ".admission.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError("The storage admission lock must be a private owned regular file.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StorageError("storage_busy", "Another process is admitting this source; wait for its receipt before retrying.") from error
        current = _entry(root, reviewed["key"])
        if _digest(current) != reviewed["generation"]:
            raise StorageError("storage_changed", "Source admission changed after preview; review the current source before continuing.")
        yield root / (reviewed["key"] + ".json")
    finally:
        os.close(descriptor)


def register(path, plan, receipt):
    sources = _sources(_entry(path.parent, path.stem))
    sources[plan["storage"]["source_key"]] = {"source": identity(plan), "selection": selection(plan),
                                             "directory": str(safe_path(plan["local_dir"])), "receipt": str(receipt)}
    write_text_atomic(path, json.dumps({"schema": 1, "sources": sources}, sort_keys=True))


def reused_receipt(plan):
    current = inspect(plan, plan["storage"]["root"])
    if current != plan["storage"]:
        raise StorageError("storage_changed", "The registered source changed after preview.")
    return read_json(current["receipt"])


def reuse_outcome(plan):
    receipt = reused_receipt(plan)
    path = plan["storage"]["receipt"]
    marker = read_json(receipt["marker"])
    job = {key: receipt.get(key) for key in (
        "repo", "revision", "file", "subfolder", "additional_files", "local_dir", "preview_hash", "log_path", "started_at")}
    job.update(receipt=path, state="done", path=receipt["local_dir"],
               completed_at=marker.get("finished_at"), files=marker["files"])
    return {"status": "reused", "receipt": receipt, "reused_fetch": job}
