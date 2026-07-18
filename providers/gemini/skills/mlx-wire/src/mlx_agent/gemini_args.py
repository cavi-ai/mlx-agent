"""Strict argument grammar for Gemini custom-command input.

Gemini TOML commands pass user text to the model, not directly to a process.
This module converts that opaque text into an allowlisted MLX-agent argv list
before a bundled skill may execute the structured CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import urllib.parse
from pathlib import Path

from .models import ROLES


class GeminiArgumentError(ValueError):
    """Gemini command input is outside the documented capability grammar."""


_RUNTIMES = {"ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"}
_ROLES = {role for role, _keywords, _label in ROLES}
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
_SAFE_QUANTIZATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_METACHARACTERS = set(";|&`$<>")


def _tokens(raw):
    if not isinstance(raw, str):
        raise GeminiArgumentError("arguments must be text")
    if len(raw) > 4096 or any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise GeminiArgumentError("arguments must not contain control characters")
    if any(character in _METACHARACTERS for character in raw):
        raise GeminiArgumentError("shell metacharacters are not allowed")
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError as error:
        raise GeminiArgumentError("arguments must use balanced shell-style quotes: {0}".format(error))
    if len(tokens) > 32 or any(not token or len(token) > 512 for token in tokens):
        raise GeminiArgumentError("arguments exceed the supported command grammar")
    return tokens


def _value(tokens, index, flag):
    if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
        raise GeminiArgumentError("{0} requires a value".format(flag))
    return tokens[index + 1]


def _path(value, label):
    if not isinstance(value, str) or not value or value.startswith("~") or "\\" in value:
        raise GeminiArgumentError("{0} must be a portable non-empty path".format(label))
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise GeminiArgumentError("{0} must not traverse parent directories".format(label))
    return value


def _model(value):
    if not _MODEL.fullmatch(value):
        raise GeminiArgumentError("model must be a publisher/model identifier")
    return value


def _integer(value, label, minimum, maximum):
    if not re.fullmatch(r"[0-9]+", value):
        raise GeminiArgumentError("{0} must be an integer".format(label))
    number = int(value)
    if number < minimum or number > maximum:
        raise GeminiArgumentError("{0} must be between {1} and {2}".format(label, minimum, maximum))
    return str(number)


def _decimal(value, label, minimum, maximum):
    if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", value):
        raise GeminiArgumentError("{0} must be a decimal number".format(label))
    number = float(value)
    if number < minimum or number > maximum:
        raise GeminiArgumentError("{0} must be between {1} and {2}".format(label, minimum, maximum))
    return value


def _endpoint(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
    except (UnicodeError, ValueError) as error:
        raise GeminiArgumentError("endpoint is not a valid local HTTP URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or hostname not in {"127.0.0.1", "localhost", "::1"}
        or username is not None
        or password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GeminiArgumentError("endpoint must be a credential-free local HTTP URL")
    if port is not None and not 1 <= port <= 65535:
        raise GeminiArgumentError("endpoint port must be between 1 and 65535")
    return value


def _parse_scout(tokens):
    argv, index = ["discover"], 0
    seen, cache = set(), None
    values = {
        "--role": (lambda value: value if value in _ROLES else None, "role"),
        "--limit": (lambda value: _integer(value, "limit", 1, 50), "limit"),
        "--memory-gb": (lambda value: _decimal(value, "memory-gb", 0.25, 1024), "memory-gb"),
        "--quantization": (lambda value: value if _SAFE_QUANTIZATION.fullmatch(value) else None, "quantization"),
        "--runtime": (lambda value: value if value in _RUNTIMES else None, "runtime"),
        "--state-dir": (lambda value: _path(value, "state-dir"), "state-dir"),
        "--wire": (lambda value: _model(value), "wire"),
        "--target": (lambda value: value if value in _RUNTIMES else None, "target"),
        "--port": (lambda value: _integer(value, "port", 1, 65535), "port"),
    }
    repeatable = {"--license", "--publisher"}
    booleans = {"--exclude-gated", "--include-gated", "--new", "--fast", "--json"}
    while index < len(tokens):
        flag = tokens[index]
        if flag in booleans:
            if flag in seen:
                raise GeminiArgumentError("duplicate flag: {0}".format(flag))
            if flag in {"--exclude-gated", "--include-gated"} and {"--exclude-gated", "--include-gated"} & seen:
                raise GeminiArgumentError("gated-repository flags are mutually exclusive")
            seen.add(flag)
            argv.append(flag)
            index += 1
        elif flag in {"--refresh", "--offline"}:
            if cache is not None:
                raise GeminiArgumentError("--refresh and --offline are mutually exclusive")
            cache = flag
            argv.append(flag)
            index += 1
        elif flag in repeatable:
            value = _value(tokens, index, flag)
            if not _SAFE_NAME.fullmatch(value):
                raise GeminiArgumentError("{0} must be a safe identifier".format(flag))
            argv.extend([flag, value])
            index += 2
        elif flag in values:
            if flag in seen:
                raise GeminiArgumentError("duplicate flag: {0}".format(flag))
            value = values[flag][0](_value(tokens, index, flag))
            if value is None:
                raise GeminiArgumentError("invalid {0} value".format(values[flag][1]))
            seen.add(flag)
            argv.extend([flag, value])
            index += 2
        else:
            raise GeminiArgumentError("unsupported Scout argument: {0}".format(flag))
    return argv


def _parse_adopt(tokens):
    if not tokens or tokens[0] not in {"start", "resume", "status"}:
        raise GeminiArgumentError("Adopt requires start, resume, or status")
    action, argv, index = tokens[0], ["adopt", tokens[0]], 1
    state = None
    allowed_booleans = {"--json"} if action != "start" else {"--offline", "--refresh", "--fast", "--no-network", "--json"}
    while index < len(tokens):
        flag = tokens[index]
        if flag == "--state":
            if state is not None:
                raise GeminiArgumentError("duplicate flag: --state")
            state = _path(_value(tokens, index, flag), "state")
            argv.extend([flag, state])
            index += 2
        elif action == "start" and flag == "--role":
            role = _value(tokens, index, flag)
            if role not in _ROLES:
                raise GeminiArgumentError("unknown adoption role")
            argv.extend([flag, role])
            index += 2
        elif action == "start" and flag == "--shortlist-limit":
            argv.extend([flag, _integer(_value(tokens, index, flag), "shortlist-limit", 1, 20)])
            index += 2
        elif flag in allowed_booleans and flag not in argv:
            argv.append(flag)
            index += 1
        else:
            raise GeminiArgumentError("unsupported Adopt argument: {0}".format(flag))
    if state is None:
        raise GeminiArgumentError("Adopt requires an explicit --state path")
    return argv


def _parse_wire(tokens):
    if not tokens or tokens[0] not in {"render", "apply", "status", "rollback"}:
        raise GeminiArgumentError("Wire requires render, apply, status, or rollback")
    action, argv, index = tokens[0], ["wire", tokens[0]], 1
    if action in {"status", "rollback"}:
        if index >= len(tokens):
            raise GeminiArgumentError("Wire {0} requires a receipt path".format(action))
        argv.append(_path(tokens[index], "receipt"))
        index += 1
        allowed = (
            {"--json"}
            if action == "status"
            else {"--confirm", "--preview-hash", "--json"}
        )
    else:
        if index >= len(tokens):
            raise GeminiArgumentError("Wire {0} requires a model".format(action))
        argv.append(_model(tokens[index]))
        index += 1
        allowed = {"--target", "--path", "--json"}
        if action == "apply":
            allowed.update({"--confirm", "--preview-hash", "--receipts-dir", "--endpoint"})
    path_seen = False
    while index < len(tokens):
        flag = tokens[index]
        if flag not in allowed:
            raise GeminiArgumentError("unsupported Wire argument: {0}".format(flag))
        if flag in argv:
            raise GeminiArgumentError("duplicate flag: {0}".format(flag))
        if flag in {"--json", "--confirm"}:
            argv.append(flag)
            index += 1
        else:
            value = _value(tokens, index, flag)
            if flag == "--target":
                if value not in _RUNTIMES:
                    raise GeminiArgumentError("unsupported Wire target")
            elif flag in {"--path", "--receipts-dir"}:
                value = _path(value, flag[2:])
                path_seen = path_seen or flag == "--path"
            elif flag == "--preview-hash":
                if not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise GeminiArgumentError("preview-hash must be a SHA-256 digest")
            elif flag == "--endpoint":
                value = _endpoint(value)
            argv.extend([flag, value])
            index += 2
    if action in {"render", "apply"} and not path_seen:
        raise GeminiArgumentError("Wire {0} requires --path".format(action))
    return argv


def parse_gemini_arguments(capability, raw):
    """Return a validated argv list; this function never runs a process."""
    tokens = _tokens(raw)
    if capability == "scout":
        return _parse_scout(tokens)
    if capability == "adopt":
        return _parse_adopt(tokens)
    if capability == "wire":
        return _parse_wire(tokens)
    raise GeminiArgumentError("unknown Gemini capability")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capability", choices=("scout", "adopt", "wire"))
    parser.add_argument("raw", help="one opaque custom-command argument string")
    arguments = parser.parse_args(argv)
    try:
        print(json.dumps({"argv": parse_gemini_arguments(arguments.capability, arguments.raw)}))
    except GeminiArgumentError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
