"""Deterministic contract simulation for Gemini custom-command TOML transport.

This module simulates documented ``{{args}}`` prompt substitution for tests and
local adapters. It is not a claim about Gemini model or tool behavior.
"""

from __future__ import annotations

import os
import tempfile

from .gemini_executor import MAX_ARGS_BYTES, GeminiCommandError, command_args_root, execute_gemini_command


_OPEN = "<mlx-agent-untrusted-args>\n"
_CLOSE = "\n</mlx-agent-untrusted-args>"


class GeminiTransportError(ValueError):
    """The simulated TOML transport could not safely carry command data."""


def substitute_toml_args(prompt, raw_args):
    """Simulate one documented TOML placeholder substitution without parsing it."""
    if not isinstance(prompt, str) or not isinstance(raw_args, str) or prompt.count("{{args}}") != 1:
        raise GeminiTransportError("TOML transport template is invalid")
    return prompt.replace("{{args}}", raw_args)


def extract_untrusted_payload(rendered_prompt):
    """Extract the delimited payload exactly, preserving embedded delimiter text."""
    if not isinstance(rendered_prompt, str):
        raise GeminiTransportError("TOML transport payload is invalid")
    start = rendered_prompt.find(_OPEN)
    end = rendered_prompt.rfind(_CLOSE)
    if start < 0 or end < start + len(_OPEN):
        raise GeminiTransportError("TOML transport delimiters are invalid")
    return rendered_prompt[start + len(_OPEN):end]


def write_private_args_file(payload):
    """Write opaque transport data to a 0600 private file without a shell."""
    if not isinstance(payload, str):
        raise GeminiTransportError("TOML transport payload is invalid")
    encoded = payload.encode("utf-8")
    if len(encoded) > MAX_ARGS_BYTES:
        raise GeminiTransportError("TOML transport payload exceeds the maximum size")
    root = command_args_root()
    descriptor = None
    name = None
    complete = False
    try:
        descriptor, name = tempfile.mkstemp(prefix="transport-", suffix=".args", dir=str(root))
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, encoded)
        complete = True
        return name
    except OSError as error:
        raise GeminiTransportError("private transport storage is unavailable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if name is not None and not complete:
            try:
                os.unlink(name)
            except OSError:
                pass


def simulate_toml_transport(prompt, raw_args, capability, core=None):
    """Simulate TOML prompt transport and dispatch only through the executor."""
    rendered = substitute_toml_args(prompt, raw_args)
    payload = extract_untrusted_payload(rendered)
    args_file = write_private_args_file(payload)
    try:
        return execute_gemini_command(capability, args_file, core=core)
    except GeminiCommandError as error:
        raise GeminiTransportError("TOML transport payload was rejected") from error
