"""Small stdio MCP server exposing one validated MLX-agent execution tool."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout

from . import __version__
from .gemini_args import GeminiArgumentError, parse_gemini_arguments


MAX_ARGUMENT_BYTES = 4096
MAX_OUTPUT_CHARS = 16384
MAX_REQUEST_BYTES = 65536
TOOL_NAME = "mlx_agent_execute"


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
        "description": "Run one allowlisted Scout, Adopt, or Wire action without a shell.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["capability", "arguments"],
            "properties": {
                "capability": {"enum": ["scout", "adopt", "wire"]},
                "arguments": {"type": "string", "maxLength": MAX_ARGUMENT_BYTES},
            },
        },
    }


def _tool_result(arguments, core=None):
    if not isinstance(arguments, dict) or set(arguments) != {"capability", "arguments"}:
        raise GeminiArgumentError("tool arguments must contain capability and arguments")
    capability = arguments["capability"]
    raw = arguments["arguments"]
    if not isinstance(raw, str):
        raise GeminiArgumentError("tool arguments must be UTF-8 text")
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GeminiArgumentError("tool arguments must be UTF-8 text") from error
    if len(encoded) > MAX_ARGUMENT_BYTES:
        raise GeminiArgumentError("tool arguments exceed the bounded UTF-8 input size")
    argv = parse_gemini_arguments(capability, raw)
    if core is None:
        from .cli import main as core
    stdout = _BoundedWriter()
    stderr = _BoundedWriter()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            returned = core(argv)
        exit_code = 0 if returned is None else int(returned)
        execution_error = None
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
        "isError": exit_code != 0,
    }


def handle_request(request, core=None):
    """Handle one JSON-RPC request. Notifications intentionally return None."""
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    request_id = request.get("id")
    if "id" not in request:
        return None
    method = request.get("method")
    try:
        if method == "initialize":
            params = request.get("params") or {}
            protocol = params.get("protocolVersion", "2024-11-05") if isinstance(params, dict) else "2024-11-05"
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
                raise GeminiArgumentError("unknown tool")
            result = _tool_result(params.get("arguments"), core=core)
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except GeminiArgumentError:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "Invalid params"}}


def main():
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return 0
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        else:
            try:
                response = handle_request(json.loads(line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
