"""Small stdio MCP server exposing one validated MLX-agent execution tool."""

from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stderr, redirect_stdout

from . import __version__
from .command_args import CommandArgumentError, parse_command_arguments


MAX_ARGUMENT_BYTES = 4096
MAX_OUTPUT_CHARS = 16384
MAX_REQUEST_BYTES = 65536
TOOL_NAME = "mlx_agent_execute"
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
        if len(text) > remaining:
            self.parts.append(text[:remaining])
            self.length += remaining
            self.truncated = True
        else:
            self.parts.append(text)
            self.length += len(text)
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


def _tool_result(arguments, core=None):
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


def handle_request(request, core=None):
    """Handle one JSON-RPC request. Notifications intentionally return None."""
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    request_id = request.get("id")
    valid_id = request_id is None or isinstance(request_id, str) or (
        type(request_id) in (int, float) and (
            not isinstance(request_id, float) or math.isfinite(request_id)
        )
    )
    if (
        not valid_id
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
            result = {"tools": [_tool_definition()]}
        elif method == "tools/call":
            params = request.get("params")
            if not isinstance(params, dict) or params.get("name") != TOOL_NAME:
                raise CommandArgumentError("unknown tool")
            result = _tool_result(params.get("arguments"), core=core)
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except CommandArgumentError:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "Invalid params"}}


def _finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def main():
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return 0
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            # Discard the entire rejected frame without buffering its remainder.
            while line and not line.endswith(b"\n"):
                line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        else:
            try:
                request = json.loads(line.decode("utf-8"), parse_float=_finite_number, parse_constant=_finite_number)
            except (ValueError, RecursionError):
                response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            else:
                response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
