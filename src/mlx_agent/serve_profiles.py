"""Named local Strata settings; profiles contain no model paths or commands."""

import json
import os
from pathlib import Path
import re
import stat

from .jit_serve import validate_memory_policy
from .serve import receipts_root
from .strata_runtime import StrataError, _validate_context
from .transactions import _atomic_in_directory, _target_locks, _walk_directory

_FIELDS = {"memory_budget_gb", "min_headroom_gb", "strata_context_tokens", "strata_decoding",
           "idle_timeout", "keep_loaded", "unload_under_pressure"}


def _validate(settings):
    if not isinstance(settings, dict) or set(settings) != _FIELDS:
        raise ValueError("A serving profile requires only memory, headroom, context, decoding and idle settings.")
    if settings["memory_budget_gb"] is None or settings["min_headroom_gb"] is None:
        raise ValueError("Serving profiles require an explicit model budget and headroom reserve.")
    validate_memory_policy({"model_memory_budget_gb": settings["memory_budget_gb"],
        "minimum_headroom_gb": settings["min_headroom_gb"], "idle_timeout_seconds": settings["idle_timeout"],
        "keep_loaded": settings["keep_loaded"], "unload_under_pressure": settings["unload_under_pressure"]})
    try:
        _validate_context(settings["strata_context_tokens"])
    except StrataError as error:
        raise ValueError(str(error)) from error
    if settings["strata_decoding"] not in ("plain", "lookup", "mtp"):
        raise ValueError("Profile decoding must be plain, lookup or mtp.")
    return dict(settings)


def _path(name, root):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
        raise ValueError("Profile names must start with a lowercase letter and contain up to 64 letters, digits, _ or -.")
    return receipts_root(root) / "profiles" / (name + ".json")


def _read(directory, name):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 8192:
                raise ValueError("Serving profile must be a regular file of at most 8 KiB.")
            return _validate(json.loads(handle.read(8193)))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("Cannot read serving profile: " + str(error)) from error


def load_profile(name, root=None):
    path = _path(name, root)
    _, fd = _walk_directory(path.parent)
    try:
        return _read(fd, path.name)
    finally:
        os.close(fd)


def save_profile(name, settings, root=None, replace=False):
    path, settings = _path(name, root), _validate(settings)
    if type(replace) is not bool:
        raise ValueError("Replace must be a boolean.")
    with _target_locks([path.parent / ".profiles"], create_parents=True):
        try:
            existing = path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None and (not stat.S_ISREG(existing.st_mode) or not replace):
            raise ValueError("Profile already exists or is a symbolic link; use --replace to update a regular profile.")
        if not path.exists() and len(list_profiles(root)) >= 100:
            raise ValueError("Too many serving profiles (maximum 100).")
        _atomic_in_directory(path.parent, path.name, (json.dumps(settings, sort_keys=True) + "\n").encode(), 0o600)
    return settings


def list_profiles(root=None):
    directory = _path("unused", root).parent
    if not directory.exists() and not directory.is_symlink():
        return {}
    _, fd = _walk_directory(directory)
    try:
        names = sorted(n for n in os.listdir(fd) if n.endswith(".json"))
        if len(names) > 100:
            raise ValueError("Too many serving profiles (maximum 100).")
        return {Path(n).stem: _read(fd, _path(Path(n).stem, root).name) for n in names}
    finally:
        os.close(fd)


def delete_profile(name, root=None):
    path = _path(name, root)
    with _target_locks([path.parent / ".profiles"]):
        _, fd = _walk_directory(path.parent)
        try:
            _read(fd, path.name)
            os.unlink(path.name, dir_fd=fd)
            os.fsync(fd)
        finally:
            os.close(fd)
