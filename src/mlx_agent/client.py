"""Run one ``mlx-agent convert`` verb and return its JSON result data."""

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

STDOUT_LIMIT = 16 * 1024 * 1024
STDERR_LIMIT = 8 * 1024
DIAGNOSTIC_LIMIT = 2000


class ConvertError(RuntimeError):
    """A failed convert call, carrying the envelope's error fields."""

    def __init__(self, operation, code, message, remediation=None):
        self.operation = operation
        self.code = code
        self.message = message
        self.remediation = remediation
        text = "mlx-agent {0}: {1}: {2}".format(operation, code, message)
        if remediation:
            text += " ({0})".format(remediation)
        super().__init__(text)


def resolve_cli(cli=None):
    """Return the CLI argv: an argv sequence, an executable, a script, or ``mlx-agent`` on PATH."""
    if cli is not None and not isinstance(cli, (str, os.PathLike)):
        return [str(part) for part in cli]
    value = str(cli) if cli is not None else shutil.which("mlx-agent")
    if not value or not Path(value).is_file():
        raise ConvertError(
            "convert",
            "cli_not_found",
            "mlx-agent CLI not found: {0}".format(value or "not on PATH"),
            "Pass the path to the mlx-agent executable or script, or put mlx-agent on PATH.",
        )
    return [value] if os.access(value, os.X_OK) else [sys.executable, value]


def _drain(stream, limit, sink):
    kept = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        if kept < limit:
            sink.append(chunk[: limit - kept])
            kept += len(chunk)


def _run(argv, timeout):
    process = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = [], []
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, STDOUT_LIMIT, out), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, STDERR_LIMIT, err), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return None, b"".join(err)
    finally:
        for reader in readers:
            reader.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
    return (process.returncode, b"".join(out)), b"".join(err)


def convert(verb, arguments, timeout, cli=None, grace=60):
    """Run ``convert <verb> <arguments> --timeout N --json`` and return the envelope ``data``."""
    argv = [*resolve_cli(cli), "convert", verb, *map(str, arguments), "--timeout", str(timeout), "--json"]
    finished, stderr = _run(argv, timeout + grace)
    diagnostics = stderr.decode("utf-8", "replace").strip()[-DIAGNOSTIC_LIMIT:]
    if finished is None:
        raise ConvertError(
            verb, "timeout", "no result within {0}s. {1}".format(timeout + grace, diagnostics).strip(),
            "Retry with a larger timeout.",
        )
    returncode, stdout = finished
    try:
        envelope = json.loads(stdout)
        if not isinstance(envelope, dict):
            raise ValueError("not an object")
    except ValueError:
        raise ConvertError(
            verb, "no_json",
            "no JSON result (exit {0}): {1}".format(returncode, diagnostics),
            "Check the mlx-agent installation and the convert arguments.",
        ) from None
    if returncode or envelope.get("status") != "ok":
        error = envelope.get("error") or {}
        raise ConvertError(
            verb,
            error.get("code") or "unknown",
            error.get("message") or "exit {0}".format(returncode),
            error.get("remediation"),
        )
    return envelope["data"]
