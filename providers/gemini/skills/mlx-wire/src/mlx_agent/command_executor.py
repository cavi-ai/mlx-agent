"""Bounded stdin executor for native provider custom tools."""

from __future__ import annotations

import argparse
import io
import json
import sys

from .gemini_args import GeminiArgumentError, parse_gemini_arguments


MAX_ARGUMENT_BYTES = 4096
_PROVIDERS = ("opencode",)
_CAPABILITIES = ("scout", "adopt", "wire")


class CommandExecutorError(ValueError):
    """Custom-tool input or execution failed without exposing raw arguments."""


def read_command_arguments(stream):
    """Read one small UTF-8 argument payload from a byte stream."""
    if not hasattr(stream, "read"):
        raise CommandExecutorError("custom tool input was rejected")
    try:
        value = stream.read(MAX_ARGUMENT_BYTES + 1)
    except (OSError, ValueError, TypeError) as error:
        raise CommandExecutorError("custom tool input was rejected") from error
    if not isinstance(value, bytes) or len(value) > MAX_ARGUMENT_BYTES:
        raise CommandExecutorError("custom tool input was rejected")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CommandExecutorError("custom tool input was rejected") from error


def execute_command(provider, capability, stream, core=None):
    """Validate stdin and execute only the capability's allowlisted core argv."""
    if provider not in _PROVIDERS or capability not in _CAPABILITIES:
        raise CommandExecutorError("custom tool request was rejected")
    try:
        arguments = parse_gemini_arguments(capability, read_command_arguments(stream))
    except (GeminiArgumentError, CommandExecutorError) as error:
        raise CommandExecutorError("custom tool request was rejected") from error
    if core is None:
        from .cli import main as core
    try:
        result = core(arguments)
    except Exception as error:
        raise CommandExecutorError("custom tool execution failed") from error
    try:
        exit_code = 0 if result is None else int(result)
    except (TypeError, ValueError) as error:
        raise CommandExecutorError("custom tool execution failed") from error
    return {"status": "ok", "provider": provider, "capability": capability, "exit_code": exit_code}


def main(argv=None, stdin=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=_PROVIDERS)
    parser.add_argument("--capability", required=True, choices=_CAPABILITIES)
    arguments = parser.parse_args(argv)
    try:
        result = execute_command(arguments.provider, arguments.capability, stdin or sys.stdin.buffer)
    except CommandExecutorError:
        print(json.dumps({"status": "error", "provider": arguments.provider, "capability": arguments.capability, "error": "command rejected"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
