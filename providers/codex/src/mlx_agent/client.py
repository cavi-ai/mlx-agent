"""Run one ``mlx-agent convert`` verb and return its JSON result data."""

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

STDOUT_LIMIT = 16 * 1024 * 1024
STDERR_LIMIT = 8 * 1024
DIAGNOSTIC_LIMIT = 2000
_BACKEND_TIMEOUT_VERBS = frozenset({
    "transcribe", "separate", "decide", "generate", "music", "speak", "describe", "mesh", "video",
})
_CLI_REMEDIATION = "Pass the path to the mlx-agent executable or script, or put mlx-agent on PATH."


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
        argv = [str(part) for part in cli]
        if argv and argv[0]:
            return argv
        value = None
    else:
        value = str(cli) if cli is not None else shutil.which("mlx-agent")
    if not value or not Path(value).is_file():
        raise ConvertError(
            "convert",
            "cli_not_found",
            "mlx-agent CLI not found: {0}".format(value or "not on PATH"),
            _CLI_REMEDIATION,
        )
    executable = str(Path(value).absolute())
    return [executable] if os.access(executable, os.X_OK) else [sys.executable, executable]


def _run(verb, argv, timeout):
    stdout, stderr = bytearray(), bytearray()
    with selectors.DefaultSelector() as selector:
        try:
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, ValueError) as error:
            code = "cli_not_found" if isinstance(error, FileNotFoundError) else "cli_launch_failed"
            raise ConvertError(verb, code, str(error), _CLI_REMEDIATION) from error

        with process:
            deadline = time.monotonic() + timeout
            try:
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stderr, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None, stderr
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        elif key.fileobj is process.stdout:
                            stdout.extend(chunk[:STDOUT_LIMIT - len(stdout)])
                        else:
                            stderr.extend(chunk)
                            del stderr[:-STDERR_LIMIT]
                returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
                return (returncode, stdout), stderr
            except subprocess.TimeoutExpired:
                return None, stderr
            finally:
                if process.returncode is None:
                    # The unreaped child still owns this process group ID.
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)


def convert(verb, arguments, timeout, cli=None, grace=60):
    """Run a JSON convert command with a deadline and return its result or confirmation preview."""
    argv = [*resolve_cli(cli), "convert", verb, *map(str, arguments)]
    if verb in _BACKEND_TIMEOUT_VERBS:
        argv.extend(("--timeout", str(timeout)))
    argv.append("--json")
    finished, stderr = _run(verb, argv, timeout + grace)
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
    if envelope.get("status") == "ok":
        if returncode == 0:
            return envelope["data"]
        data = envelope.get("data")
        if (verb == "start" and returncode == 2 and isinstance(data, dict)
                and data.get("requires_confirmation") is True):
            return data
    error = envelope.get("error")
    if not isinstance(error, dict):
        error = {}
    raise ConvertError(
        verb,
        error.get("code") or "unknown",
        error.get("message") or "exit {0}".format(returncode),
        error.get("remediation"),
    )
