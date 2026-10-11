"""Atomic queue publication uses the agent's retained-directory writer."""
from pathlib import Path
from .transactions import _assert_safe_directory, _atomic_in_directory

def write_text_atomic(path, text, encoding="utf-8"):
    location = Path(path)
    if location.is_symlink():
        raise OSError("Refusing to write through a symbolic link.")
    _assert_safe_directory(location.parent, create=True)
    _atomic_in_directory(location.parent, location.name, text.encode(encoding), 0o600)
    return location
