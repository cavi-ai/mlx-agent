"""One model object retained only inside an engine-owned resident worker."""
from pathlib import Path
import time

_enabled = False
_entry = None
_loads = 0
_cache_hits = 0
_load_seconds = 0.0


def load_model(loader, path, **options):
    global _entry, _loads, _cache_hits, _load_seconds
    if not _enabled:
        return loader(path, **options)
    local = Path(path)
    if not local.is_absolute() or not local.is_dir():
        raise ValueError("Resident inference requires existing absolute local model files.")
    key = (loader.__module__, loader.__qualname__, str(local.resolve()), repr(sorted(options.items())))
    if _entry is not None and _entry[0] != key:
        raise ValueError("A resident worker cannot switch models or loaders.")
    if _entry is None:
        started = time.monotonic()
        _entry = key, loader(path, **options)
        _load_seconds += time.monotonic() - started
        _loads += 1
    else:
        _cache_hits += 1
    return _entry[1]


def loaded():
    return _entry is not None


def enabled():
    return _enabled


def statistics():
    return {'loads':_loads,'cache_hits':_cache_hits,'load_seconds':_load_seconds}
