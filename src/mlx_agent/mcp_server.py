"""Bounded stdio MCP tools for discovery and validated MLX-agent execution."""

from __future__ import annotations

import codecs
import io
import json
import math
import os
from pathlib import Path
import select
import selectors
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout

from . import __version__
from .command_args import CommandArgumentError, parse_command_arguments


MAX_ARGUMENT_BYTES = 4096
MAX_OUTPUT_CHARS = 16384
MAX_REQUEST_BYTES = 65536
MAX_PENDING_TOOLS = 8
TOOL_TIMEOUT_SECONDS = 660
TOOL_NAME = "mlx_agent_execute"
CAPABILITIES_TOOL_NAME = "mlx_agent_capabilities"
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")


class _BoundedWriter(io.TextIOBase):
    def __init__(self, limit=MAX_OUTPUT_CHARS):
        self.limit = limit
        self.parts = []
        self.length = 0
        self.truncated = False

    def writable(self):
        return True

    def write(self, value):
        text = str(value)
        remaining = max(0, self.limit - self.length)
        if remaining and text:
            retained = text[:remaining]
            self.parts.append(retained)
            self.length += len(retained)
        if len(text) > remaining:
            self.truncated = True
        return len(text)

    def value(self):
        suffix = "\n... [output truncated]" if self.truncated else ""
        return "".join(self.parts) + suffix


def _tool_definition():
    return {
        "name": TOOL_NAME,
        "description": "Run one validated model lifecycle or experiment action without a shell. Actions may change local files, start jobs, or contact model registries.",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["capability", "arguments"],
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "Capability whose validated command grammar is used; task capabilities --json lists supported inference tasks without running a model.",
                    "enum": ["scout", "adopt", "wire", "bench", "doctor", "watch", "fleet", "lab", "task"],
                },
                "arguments": {
                    "type": "string",
                    "description": "Arguments for the chosen capability, at most 4096 UTF-8 bytes. Shell operators are refused. Example for task: capabilities --json.",
                    "maxLength": MAX_ARGUMENT_BYTES,
                },
            },
        },
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "exit_code", "stdout", "stderr", "stdout_truncated", "stderr_truncated"],
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]},
                "exit_code": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "stdout_truncated": {"type": "boolean"},
                "stderr_truncated": {"type": "boolean"},
                "error": {"type": "string"},
            },
        },
    }


def _tool_result(arguments, core=None, execute=None):
    if not isinstance(arguments, dict) or set(arguments) != {"capability", "arguments"}:
        raise CommandArgumentError("tool arguments must contain capability and arguments")
    capability = arguments["capability"]
    raw = arguments["arguments"]
    if not isinstance(raw, str):
        raise CommandArgumentError("tool arguments must be UTF-8 text")
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CommandArgumentError("tool arguments must be UTF-8 text") from error
    if len(encoded) > MAX_ARGUMENT_BYTES:
        raise CommandArgumentError("tool arguments exceed the bounded UTF-8 input size")
    argv = parse_command_arguments(capability, raw)
    if execute is not None:
        return execute(argv)
    if core is None:
        from .cli import main as core
    stdout = _BoundedWriter()
    stderr = _BoundedWriter()
    execution_error = None
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            returned = core(argv)
        exit_code = 0 if returned is None else int(returned)
    except SystemExit as error:
        if error.code is None:
            exit_code = 0
        elif isinstance(error.code, int):
            exit_code = int(error.code)
        else:
            exit_code = 1
    except Exception:
        exit_code = 1
        execution_error = "MLX-agent execution failed"
    return _execution_result(exit_code, stdout, stderr, execution_error)


def _execution_result(exit_code, stdout, stderr, execution_error=None):
    payload = {
        "status": "ok" if exit_code == 0 else "error",
        "exit_code": exit_code,
        "stdout": stdout.value(),
        "stderr": stderr.value(),
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
    }
    if execution_error is not None:
        payload["error"] = execution_error
    return {
        "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
        "structuredContent": payload,
        "isError": exit_code != 0,
    }


def _run_process(command, cancelled, timeout, env):
    stdout, stderr = _BoundedWriter(), _BoundedWriter()
    if cancelled():
        return _execution_result(130, stdout, stderr, "MCP tool call cancelled.")
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
    except OSError:
        return _execution_result(1, stdout, stderr, "Could not start the MLX-agent command.")

    writers = (stdout, stderr)
    decoders = [codecs.getincrementaldecoder("utf-8")(errors="replace") for _ in writers]
    try:
        with selectors.DefaultSelector() as selector:
            for index, pipe in enumerate((process.stdout, process.stderr)):
                selector.register(pipe, selectors.EVENT_READ, index)
            while True:
                if cancelled():
                    return _execution_result(130, stdout, stderr, "MCP tool call cancelled.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return _execution_result(124, stdout, stderr, "MCP tool call timed out.")
                events = selector.select(min(0.1, remaining))
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    index = key.data
                    writers[index].write(decoders[index].decode(chunk))
                if (not events or not selector.get_map()) and process.poll() is not None:
                    break
        for writer, decoder in zip(writers, decoders):
            writer.write(decoder.decode(b"", final=True))
        return _execution_result(process.returncode, stdout, stderr)
    except OSError:
        return _execution_result(1, stdout, stderr, "Could not read the MLX-agent command output.")
    finally:
        try:
            if process.returncode is None:
                # An unreaped direct child still owns this process group ID.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
        finally:
            process.stdout.close()
            process.stderr.close()


def _execute_cli(argv, cancelled):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    if sys.pycache_prefix is not None:
        env["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
    return _run_process([sys.executable, "-P", "-m", "mlx_agent", *argv], cancelled, TOOL_TIMEOUT_SECONDS, env)


def _capabilities_definition():
    return {
        "name": CAPABILITIES_TOOL_NAME,
        "description": "Describe supported local inference tasks without running models, accessing task state, or contacting registries. Takes no arguments.",
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _tool_definition()["outputSchema"],
    }


def _valid_id(value):
    return value is None or isinstance(value, str) or (
        type(value) in (int, float) and (not isinstance(value, float) or math.isfinite(value))
    )


def handle_request(request, core=None, execute=None):
    """Handle one JSON-RPC request. Notifications intentionally return None."""
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    request_id = request.get("id")
    if (
        not _valid_id(request_id)
        or not isinstance(request.get("method"), str)
        or ("params" in request and not isinstance(request["params"], (dict, list)))
    ):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    if "id" not in request:
        return None
    method = request.get("method")
    try:
        if method == "initialize":
            params = request.get("params") or {}
            protocol = params.get("protocolVersion") if isinstance(params, dict) else None
            if protocol not in SUPPORTED_PROTOCOL_VERSIONS:
                protocol = SUPPORTED_PROTOCOL_VERSIONS[-1]
            result = {
                "protocolVersion": protocol,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mlx-agent", "version": __version__},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [_tool_definition(), _capabilities_definition()]}
        elif method == "tools/call":
            params = request.get("params")
            if not isinstance(params, dict):
                raise CommandArgumentError("unknown tool")
            if params.get("name") == CAPABILITIES_TOOL_NAME:
                if params.get("arguments", {}) != {}:
                    raise CommandArgumentError("capabilities takes no arguments")
                result = _tool_result({"capability": "task", "arguments": "capabilities --json"},
                                      core=core, execute=execute)
            elif params.get("name") == TOOL_NAME:
                result = _tool_result(params.get("arguments"), core=core, execute=execute)
            else:
                raise CommandArgumentError("unknown tool")
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except CommandArgumentError:
        return {"jsonrpc": "2.0", "id": request_id, "error": {
            "code": -32602, "message": "Invalid params", "data": {"retryable": False}}}


def _finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def _requests():
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            # Discard the entire rejected frame without buffering its remainder.
            while line and not line.endswith(b"\n"):
                line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            yield None, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        else:
            try:
                request = json.loads(line.decode("utf-8"), parse_float=_finite_number, parse_constant=_finite_number)
            except (ValueError, RecursionError):
                yield None, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            else:
                yield request, None


def main():
    pending = {}
    pending_lock = threading.Lock()
    reply_lock = threading.Lock()
    output = select.poll()
    try:
        output.register(sys.stdout.fileno(), select.POLLHUP | select.POLLERR)
    except (AttributeError, OSError):
        pass

    def send(response):
        with reply_lock:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    def execute(request, cancelled):
        try:
            try:
                response = handle_request(request, execute=lambda argv: _execute_cli(
                    argv, lambda: cancelled.is_set() or bool(output.poll(0))))
            except (OSError, subprocess.SubprocessError):
                response = {"jsonrpc": "2.0", "id": request["id"],
                            "error": {"code": -32603, "message": "Could not complete the MLX-agent command."}}
            if not cancelled.is_set() and not output.poll(0):
                send(response)
        except BrokenPipeError:
            pass
        finally:
            with pending_lock:
                pending.pop(request["id"], None)

    if threading.current_thread() is not threading.main_thread():
        return _run_session(pending, pending_lock, send, execute)

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return _run_session(pending, pending_lock, send, execute)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _run_session(pending, pending_lock, send, execute):
    tools = ThreadPoolExecutor(max_workers=1)
    try:
        for request, error in _requests():
            if error is not None:
                send(error)
                continue
            if isinstance(request, dict) and request.get("jsonrpc") == "2.0":
                method = request.get("method")
                if method == "notifications/cancelled" and "id" not in request:
                    params = request.get("params")
                    if isinstance(params, dict) and "requestId" in params and _valid_id(params["requestId"]):
                        with pending_lock:
                            event = pending.get(params["requestId"])
                            if event is not None:
                                event.set()
                    continue
                if method == "tools/call" and "id" in request and _valid_id(request["id"]):
                    identifier = request["id"]
                    with pending_lock:
                        if identifier in pending:
                            error = {"code": -32600, "message": "Duplicate request ID"}
                        elif len(pending) >= MAX_PENDING_TOOLS:
                            error = {"code": -32000, "message": "MCP tool queue is full"}
                        else:
                            event = pending[identifier] = threading.Event()
                    if error is not None:
                        send({"jsonrpc": "2.0", "id": identifier, "error": error})
                    else:
                        tools.submit(execute, request, event)
                    continue
            response = handle_request(request)
            if response is not None:
                send(response)
        tools.shutdown()
    except BaseException:
        with pending_lock:
            for event in pending.values():
                event.set()
        tools.shutdown(cancel_futures=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
