"""Bounded local source identities and completed managed-fetch provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

MAX_FILES = 10000
MAX_JSON = 2 * 1024 * 1024
_COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def immutable_revision(value):
    return isinstance(value, str) and _COMMIT.fullmatch(value) is not None


def safe_path(value):
    path = Path(value).expanduser().absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            if str(component) in ("/var", "/tmp", "/etc") and os.readlink(component) == "private" + str(component):
                continue
            raise ValueError("Source evidence cannot traverse symbolic links.")
    return path.resolve()


def read_json(path):
    location = safe_path(path)
    descriptor = os.open(location, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON
                or before.st_uid != os.getuid() or before.st_mode & 0o022):
            raise ValueError("Source evidence must be a bounded regular JSON file.")
        raw = os.read(descriptor, MAX_JSON + 1)
        after = os.fstat(descriptor)
        if len(raw) > MAX_JSON or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Source evidence changed while reading.")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Source evidence must be an object.")
        return value
    finally:
        os.close(descriptor)


def file_identity(path, root=None):
    location = safe_path(path)
    before = location.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("A source identity requires a regular file.")
    digest = hashlib.sha256()
    digest.update(json.dumps([before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns]).encode())
    with location.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if before.st_size > 2 * 1024 * 1024:
            stream.seek(-1024 * 1024, os.SEEK_END)
            digest.update(stream.read(1024 * 1024))
    after = location.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Source files changed while reading.")
    name = str(location.relative_to(safe_path(root))) if root is not None else str(location)
    return {"path": name, "bytes": before.st_size, "signature": digest.hexdigest()}


def identities(path, exclude=()):
    root = safe_path(path)
    if root.is_file():
        return [file_identity(root)]
    if not root.is_dir():
        raise ValueError("The source directory is missing.")
    result = []
    def failed(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        dirs[:] = sorted(name for name in dirs if name != ".cache")
        for name in dirs:
            safe_path(Path(directory) / name)
        for name in sorted(files):
            item = Path(directory) / name
            if str(item.relative_to(root)) in exclude:
                continue
            result.append(file_identity(item, root))
            if len(result) > MAX_FILES:
                raise ValueError("The source exceeds the bounded file identity limit.")
    return sorted(result, key=lambda item: item["path"])


def fingerprint(path, exclude=()):
    return hashlib.sha256(json.dumps(identities(path, exclude), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def completed_fetch(receipt_path, selected_path=None, bundle_root=False):
    receipt_path = safe_path(receipt_path)
    receipt = read_json(receipt_path)
    if receipt.get("kind") != "fetch" or not immutable_revision(receipt.get("revision")) or not receipt.get("fresh_destination"):
        raise ValueError("Conversion needs a completed immutable managed fetch receipt.")
    marker_path = receipt_path.with_name(receipt_path.stem + ".done.json")
    if safe_path(receipt.get("marker", "")) != marker_path:
        raise ValueError("The fetch marker is outside its recorded receipt.")
    marker = read_json(marker_path)
    keys = ("repo", "revision", "file", "subfolder", "additional_files", "local_dir", "preview_hash", "proof_token", "started_at", "downloads_dir", "downloads")
    if marker.get("exit_status") != "done" or not receipt.get("proof_token") or any(marker.get(key) != receipt.get(key) for key in keys):
        raise ValueError("The completed fetch does not match its source receipt.")
    root = safe_path(receipt["local_dir"])
    if safe_path(marker.get("path", "")) != root or marker.get("files") != identities(root):
        raise ValueError("The downloaded source files changed after completion.")
    suffix = receipt.get("file") or receipt.get("subfolder")
    selected = root if bundle_root and receipt.get("file") else (root / suffix if suffix else root)
    selected = safe_path(selected)
    if selected_path is not None and safe_path(selected_path) != selected:
        raise ValueError("The conversion source differs from the completed download selection.")
    origin = {key: receipt.get(key) for key in ("repo", "revision", "file", "subfolder", "additional_files")}
    evidence = {"origin": origin, "path": str(selected), "fingerprint": fingerprint(selected),
                "fetch_receipt": str(receipt_path)}
    if receipt.get("downloads"):
        support = safe_path(receipt["downloads_dir"])
        if support == root or not support.is_relative_to(root):
            raise ValueError("Managed support files are outside the reviewed download.")
        files = []
        for item in receipt["downloads"]:
            target = support / item["sha256"] / item["url"].rsplit("/", 1)[-1]
            files.append(file_identity(target, root))
        evidence["support"] = {"downloads_dir": str(support), "downloads": receipt["downloads"], "files": files}
    return evidence
