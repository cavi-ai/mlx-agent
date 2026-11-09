"""One model object retained only inside an engine-owned resident worker."""
from pathlib import Path

_enabled = False
_entry = None


def load_model(loader, path, **options):
    global _entry
    if not _enabled:
        return loader(path, **options)
    local = Path(path)
    if not local.is_absolute() or not local.is_dir():
        raise ValueError("Resident inference requires existing absolute local model files.")
    key = (loader.__module__, loader.__qualname__, str(local.resolve()), repr(sorted(options.items())))
    if _entry is not None and _entry[0] != key:
        raise ValueError("A resident worker cannot switch models or loaders.")
    if _entry is None:
        _entry = key, loader(path, **options)
    return _entry[1]


def loaded():
    return _entry is not None


def enabled():
    return _enabled
