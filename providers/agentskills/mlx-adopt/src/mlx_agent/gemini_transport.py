"""Deterministic contract simulation for Gemini custom-command TOML transport.

This module simulates documented ``{{args}}`` prompt substitution for tests and
local adapters. It is not a claim about Gemini model or tool behavior.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

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


def build_structured_executor_call(skill_dir, capability, args_file):
    """Build the only permitted non-shell Gemini executor process shape."""
    skill_path = Path(skill_dir)
    argument_path = Path(args_file)
    root = command_args_root()
    if (
        capability not in {"scout", "adopt", "wire"}
        or not skill_path.is_absolute()
        or not argument_path.is_absolute()
        or argument_path.parent != root
        or not argument_path.name
    ):
        raise GeminiTransportError("structured executor fields are invalid")
    return {
        "program": "python3",
        "argv": ["-m", "mlx_agent.gemini_executor", "--capability", capability, "--args-file", str(argument_path)],
        "env": {"PYTHONPATH": str(skill_path / "src")},
        "shell": False,
    }


def run_structured_executor_call(call, extra_env=None):
    """Run a validated structured executor fixture through subprocess without a shell."""
    if not isinstance(call, dict) or set(call) != {"program", "argv", "env", "shell"}:
        raise GeminiTransportError("structured executor fields are invalid")
    program, argv, env, shell = call["program"], call["argv"], call["env"], call["shell"]
    if (
        program != "python3"
        or not isinstance(argv, list)
        or len(argv) != 6
        or argv[:3] != ["-m", "mlx_agent.gemini_executor", "--capability"]
        or argv[3] not in {"scout", "adopt", "wire"}
        or argv[4] != "--args-file"
        or not isinstance(argv[5], str)
        or not isinstance(env, dict)
        or set(env) != {"PYTHONPATH"}
        or not isinstance(env["PYTHONPATH"], str)
    ):
        raise GeminiTransportError("structured executor fields are invalid")
    if shell is not False or not env["PYTHONPATH"]:
        raise GeminiTransportError("structured executor fields are invalid")
    environment = dict(os.environ)
    if extra_env is not None:
        environment.update(extra_env)
    environment.update(env)
    return subprocess.run([program] + argv, env=environment, shell=False, text=True, capture_output=True, check=False)


def simulate_toml_transport(prompt, raw_args, capability, core=None):
    """Simulate TOML prompt transport and dispatch only through the executor."""
    rendered = substitute_toml_args(prompt, raw_args)
    payload = extract_untrusted_payload(rendered)
    args_file = write_private_args_file(payload)
    try:
        return execute_gemini_command(capability, args_file, core=core)
    except GeminiCommandError as error:
        raise GeminiTransportError("TOML transport payload was rejected") from error
