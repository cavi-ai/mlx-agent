"""No-shell executor for Gemini custom-command argument files."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

from .gemini_args import GeminiArgumentError, parse_gemini_arguments


MAX_ARGS_BYTES = 4096
_ARGS_DIRECTORY = "mlx-agent-gemini-command-args"


class GeminiCommandError(ValueError):
    """A custom-command argument file is unsafe or invalid."""


def command_args_root():
    """Return the private OS-temporary root reserved for Gemini argument files."""
    root = Path(tempfile.gettempdir()) / _ARGS_DIRECTORY
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        details = os.lstat(str(root))
    except OSError as error:
        raise GeminiCommandError("private argument storage is unavailable") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode) or details.st_mode & 0o077:
        raise GeminiCommandError("private argument storage is unsafe")
    return root


def _safe_args_path(value):
    root = command_args_root()
    path = Path(value)
    if not path.is_absolute() or path.parent != root or ".." in path.parts or not path.name:
        raise GeminiCommandError("argument file path is outside private storage")
    return path


def _discard_if_safe(path, opened=None):
    """Remove only the named regular file we opened, or a rejected symlink."""
    try:
        current = os.lstat(str(path))
        if stat.S_ISLNK(current.st_mode):
            os.unlink(str(path))
        elif stat.S_ISREG(current.st_mode) and (opened is None or (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)):
            os.unlink(str(path))
    except OSError:
        pass


def _read_args_file(value):
    path = _safe_args_path(value)
    opened = None
    descriptor = None
    try:
        try:
            before = os.lstat(str(path))
        except OSError as error:
            raise GeminiCommandError("argument file is unavailable") from error
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise GeminiCommandError("argument file must be a regular file")
        if before.st_size > MAX_ARGS_BYTES:
            raise GeminiCommandError("argument file exceeds the maximum size")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(str(path), flags)
            opened = os.fstat(descriptor)
        except OSError as error:
            raise GeminiCommandError("argument file cannot be opened safely") from error
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise GeminiCommandError("argument file changed before it could be read")
        data = os.read(descriptor, MAX_ARGS_BYTES + 1)
        if len(data) > MAX_ARGS_BYTES:
            raise GeminiCommandError("argument file exceeds the maximum size")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GeminiCommandError("argument file must be UTF-8") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        _discard_if_safe(path, opened)


def execute_gemini_command(capability, args_file, core=None):
    """Read, validate, execute, and discard one trusted Gemini argument file."""
    try:
        raw = _read_args_file(args_file)
        argv = parse_gemini_arguments(capability, raw)
    except (GeminiArgumentError, GeminiCommandError) as error:
        raise GeminiCommandError("Gemini command arguments were rejected") from error
    if core is None:
        from .cli import main as core
    try:
        result = core(argv)
    except Exception as error:
        raise GeminiCommandError("Gemini command execution failed") from error
    exit_code = 0 if result is None else int(result)
    return {"status": "ok", "capability": capability, "exit_code": exit_code}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", required=True, choices=("scout", "adopt", "wire"))
    parser.add_argument("--args-file", required=True)
    arguments = parser.parse_args(argv)
    try:
        print(json.dumps(execute_gemini_command(arguments.capability, arguments.args_file), sort_keys=True))
    except GeminiCommandError:
        parser.error("Gemini command arguments were rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
