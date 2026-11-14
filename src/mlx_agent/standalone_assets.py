"""Read-only inventory and explicit byte comparison of standalone model files."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path

_CHUNK = 1024 * 1024
_SHARD = re.compile(r"(?:^|[-_.])\d+-of-\d+(?:\.|$)", re.IGNORECASE)
_SKIP = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", ".quarantine", ".Trash"}


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)


def _model_marker(names):
    return any(name in {"config.json", "model_index.json"} or name.endswith(".index.json") for name in names)


def _provider_path(path):
    return any(part.startswith("models--") for part in path.parts)


def _exists(path):
    try:
        path.stat()
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


def _excluded_ancestor(path):
    for parent in (path, *path.parents):
        if _provider_path(parent):
            return True
        markers = ("config.json", "model_index.json", "model.safetensors.index.json",
                   "pytorch_model.bin.index.json", "diffusion_pytorch_model.safetensors.index.json")
        if any(_exists(parent / marker) for marker in markers):
            return True
        if _exists(parent / "blobs") and (_exists(parent / "manifests") or _exists(parent / "manifests-v2")):
            return True
    return False


def _open_parent(path, root, root_identity):
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptors = [root_fd]
    try:
        opened_root = os.fstat(root_fd)
        if (opened_root.st_dev, opened_root.st_ino) != root_identity:
            raise OSError("Scan root changed")
        relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
        parent_fd = root_fd
        for component in relative.parts[:-1]:
            parent_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            descriptors.append(parent_fd)
        return parent_fd, relative.name, descriptors
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _stat_beneath(path, root, root_identity):
    parent, name, descriptors = _open_parent(path, root, root_identity)
    try:
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def hash_file(path, expected, root, root_identity):
    """Hash a stable regular file through no-follow descriptors beneath its root."""
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    parent_fd, name, descriptors = _open_parent(path, root, root_identity)
    fd = None
    try:
        fd = os.open(name, file_flags, dir_fd=parent_fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or _identity(before) != expected:
            raise OSError("File changed before comparison")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, _CHUNK)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        current = _stat_beneath(path, root, root_identity)
        current_root = os.lstat(root)
        if (not stat.S_ISREG(current.st_mode) or _identity(after) != expected or _identity(current) != expected
                or (current_root.st_dev, current_root.st_ino) != root_identity):
            raise OSError("File or scan root changed during comparison")
        return digest.hexdigest()
    finally:
        if fd is not None:
            os.close(fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def scan(roots, owned_items, limit, compare=False):
    """Return storage items, full-digest groups and explicit coverage details."""
    owned = {file["id"] for item in owned_items for file in item["files"]}
    items, errors, identities = [], [], {}
    seen = set()
    limited = False
    visited = 0
    visit_limit = max(1024, limit * 32)

    def error(path, message):
        record = {"path": str(path), "message": str(message)}
        if record not in errors:
            errors.append(record)

    for root in dict.fromkeys(str(Path(value).expanduser()) for value in roots):
        base = Path(root)
        try:
            st_root = base.lstat()
            if not stat.S_ISDIR(st_root.st_mode) or _excluded_ancestor(Path(os.path.abspath(base))):
                continue
        except OSError as exc:
            error(base, exc)
            continue
        for folder, dirs, files in os.walk(base, followlinks=False, onerror=lambda exc: error(exc.filename or base, exc)):
            here = Path(folder)
            visited += 1 + len(dirs) + len(files)
            if visited > visit_limit:
                limited = True
                break
            if _provider_path(here) or _model_marker(files) or ("blobs" in dirs and ({"manifests", "manifests-v2"} & set(dirs))):
                dirs[:] = []
                continue
            dirs[:] = sorted(name for name in dirs if name not in _SKIP and not name.startswith(".")
                             and not name.startswith("models--") and not (here / name).is_symlink())
            for name in sorted(files):
                path = here / name
                if path.suffix.lower() not in {".safetensors", ".ckpt"} or _SHARD.search(name):
                    continue
                key = str(path)
                unique_path = os.path.abspath(key)
                if unique_path in seen:
                    continue
                seen.add(unique_path)
                try:
                    st = path.lstat()
                except OSError as exc:
                    error(path, exc)
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                try:
                    checked_stat = _stat_beneath(key, root, (st_root.st_dev, st_root.st_ino))
                    if not stat.S_ISREG(checked_stat.st_mode) or _identity(st) != _identity(checked_stat):
                        raise OSError("File changed during inventory")
                except OSError as exc:
                    error(path, exc)
                    continue
                disk_id = "{0}:{1}".format(st.st_dev, st.st_ino)
                if disk_id in owned:
                    continue
                if len(items) >= limit:
                    limited = True
                    break
                items.append({"path": key, "kind": "asset", "format": path.suffix[1:].lower(), "name": name,
                              "model_key": key, "shape": None, "files": [{"id": disk_id, "path": key, "bytes": st.st_size}],
                              "asset": {"sha256": None, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns, "nlink": st.st_nlink}})
                identities[key] = (_identity(st), root, (st_root.st_dev, st_root.st_ino))
            if limited:
                break
        if limited:
            break
    if limited:
        error(base, "Standalone model-file scan reached a file or traversal limit; results are incomplete.")
    items.sort(key=lambda item: item["path"])
    sizes = {}
    for item in items:
        identity = identities[item["path"]][0]
        if identity[-1] == 1:
            sizes.setdefault(identity[2], []).append(item)
    candidates = [item for group in sizes.values() if len(group) > 1 for item in group] if compare else []
    checked = 0
    digests = {}
    for item in candidates:
        try:
            if _excluded_ancestor(Path(os.path.abspath(item["path"])).parent):
                raise OSError("Model-file ownership changed before comparison")
            digest = hash_file(item["path"], *identities[item["path"]])
            if _excluded_ancestor(Path(os.path.abspath(item["path"])).parent):
                raise OSError("Model-file ownership changed during comparison")
        except OSError as exc:
            error(item["path"], exc)
            continue
        checked += 1
        item["asset"]["sha256"] = digest
        digests.setdefault(digest, []).append(item)
    groups = []
    for digest, members in sorted(digests.items()):
        stable = []
        for item in members:
            try:
                identity, root, root_identity = identities[item["path"]]
                if _excluded_ancestor(Path(os.path.abspath(item["path"])).parent):
                    raise OSError("Model-file ownership changed after comparison")
                current = _stat_beneath(item["path"], root, root_identity)
                if not stat.S_ISREG(current.st_mode) or _identity(current) != identities[item["path"]][0]:
                    raise OSError("File changed after comparison")
                stable.append(item)
            except OSError as exc:
                error(item["path"], exc)
                item["asset"]["sha256"] = None
                checked -= 1
        members = stable
        if len(members) < 2:
            continue
        members.sort(key=lambda item: item["path"])
        identity = "asset:sha256:" + digest
        groups.append({"kind": "exact", "verification": "full_sha256", "identity": identity,
                       "model_key": members[0]["name"], "keep": members[0]["path"],
                       "redundant": [item["path"] for item in members[1:]],
                       "reclaimable_bytes": sum(item["files"][0]["bytes"] for item in members[1:])})
    return items, groups, {"compared": compare, "candidates": len(candidates), "checked": checked, "errors": errors}
