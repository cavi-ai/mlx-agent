"""Focused standard-library validation for the canonical MLX agent contracts."""

import json
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = "1.0"
PLUGIN_VERSION = "0.2.0"
CAPABILITIES = ("scout", "adopt", "wire")
CAPABILITY_ACTIONS = {
    "scout": ("discover",),
    "adopt": ("start", "resume", "status"),
    "wire": ("render", "apply", "status", "rollback"),
}
NATIVE_PROVIDERS = ("claude", "codex", "gemini", "opencode")
PROVIDERS = NATIVE_PROVIDERS + ("agentskills",)
PROVIDER_INVOCATIONS = {
    "claude": {"kind": "command", "prefix": "/"},
    "codex": {"kind": "skill", "prefix": "$"},
    "gemini": {"kind": "command", "prefix": "/"},
    "opencode": {"kind": "command", "prefix": "/"},
    "agentskills": {"kind": "skill", "prefix": ""},
}
PROVIDER_COMMANDS = {
    "claude": ["mlx-scout", "mlx-adopt", "mlx-wire"],
    "codex": ["mlx-agent:mlx-scout", "mlx-agent:mlx-adopt", "mlx-agent:mlx-wire"],
    "gemini": ["mlx-scout", "mlx-adopt", "mlx-wire"],
    "opencode": ["mlx-scout", "mlx-adopt", "mlx-wire"],
    "agentskills": [],
}
DATE_TIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
SEMVER_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _unexpected_keys(
    value: Dict[str, Any], allowed: Iterable[str], prefix: str, errors: List[str]
) -> None:
    unexpected = sorted(set(value) - set(allowed))
    if unexpected:
        errors.append("{0} has unexpected keys: {1}".format(prefix, unexpected))


def _require_keys(
    value: Dict[str, Any], keys: Iterable[str], prefix: str, errors: List[str]
) -> None:
    for key in keys:
        if key not in value:
            errors.append("{0}.{1} is required".format(prefix, key))


def _require_string(
    value: Dict[str, Any], key: str, prefix: str, errors: List[str]
) -> None:
    item = value.get(key)
    if not isinstance(item, str):
        errors.append("{0}.{1} must be a string".format(prefix, key))
    elif not item:
        errors.append("{0}.{1} must not be empty".format(prefix, key))


def _validate_date_time(value: Any, prefix: str, errors: List[str]) -> None:
    if not isinstance(value, str) or not DATE_TIME_PATTERN.match(value):
        errors.append("{0} must be an ISO-8601 date-time".format(prefix))
        return
    normalized = "{0}+00:00".format(value[:-1]) if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        errors.append("{0} must be an ISO-8601 date-time".format(prefix))
        return
    if parsed.tzinfo is None:
        errors.append("{0} must be an ISO-8601 date-time".format(prefix))


def _validate_warning(value: Any, prefix: str, errors: List[str]) -> None:
    if not _is_dict(value) or not all(isinstance(item, str) for item in value.values()):
        errors.append("{0} must be an object of strings".format(prefix))


def validate_result(value: dict) -> List[str]:
    """Return contract errors for a serialized ResultEnvelope, if any."""
    if not _is_dict(value):
        return ["result must be an object"]

    errors: List[str] = []
    required = ("schema_version", "generated_at", "operation", "status", "data", "warnings")
    _require_keys(value, required, "result", errors)
    _unexpected_keys(value, required + ("error",), "result", errors)
    _require_string(value, "schema_version", "result", errors)
    _require_string(value, "operation", "result", errors)
    _validate_date_time(value.get("generated_at"), "result.generated_at", errors)

    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must equal '1.0'")
    if value.get("status") not in ("ok", "error"):
        errors.append("status must be one of ['ok', 'error']")
    if "data" in value and not _is_dict(value["data"]):
        errors.append("data must be an object")
    if "warnings" in value and not isinstance(value["warnings"], list):
        errors.append("warnings must be an array")
    elif isinstance(value.get("warnings"), list):
        for index, warning in enumerate(value["warnings"]):
            _validate_warning(warning, "warnings[{0}]".format(index), errors)

    if value.get("status") == "error":
        error = value.get("error")
        if not _is_dict(error):
            errors.append("error must be an object")
        else:
            error_keys = ("code", "message", "remediation", "retryable")
            _require_keys(error, error_keys, "error", errors)
            _unexpected_keys(error, error_keys, "error", errors)
            for key in ("code", "message", "remediation"):
                _require_string(error, key, "error", errors)
            if not isinstance(error.get("retryable"), bool):
                errors.append("error.retryable must be a boolean")
    elif "error" in value:
        errors.append("error is only allowed when status is 'error'")

    return errors


def _validate_argument(value: Any, prefix: str, errors: List[str]) -> None:
    if not _is_dict(value):
        errors.append("{0} must be an object".format(prefix))
        return
    required_keys = ("name", "kind", "type", "required")
    allowed_keys = required_keys + ("flag", "repeatable")
    _require_keys(value, required_keys, prefix, errors)
    _unexpected_keys(value, allowed_keys, prefix, errors)
    _require_string(value, "name", prefix, errors)
    if value.get("type") not in ("string", "integer", "number", "boolean"):
        errors.append("{0}.type is invalid".format(prefix))
    if value.get("kind") not in ("positional", "option"):
        errors.append("{0}.kind is invalid".format(prefix))
    if value.get("kind") == "option":
        if not isinstance(value.get("flag"), str) or not re.fullmatch(r"--[a-z][a-z0-9-]*", value["flag"]):
            errors.append("{0}.flag must be a long option".format(prefix))
    elif "flag" in value:
        errors.append("{0}.flag is not allowed for positional arguments".format(prefix))
    if not isinstance(value.get("required"), bool):
        errors.append("{0}.required must be a boolean".format(prefix))
    if "repeatable" in value and not isinstance(value["repeatable"], bool):
        errors.append("{0}.repeatable must be a boolean".format(prefix))


def _validate_capability(value: Any, name: str, errors: List[str]) -> None:
    prefix = "capabilities.{0}".format(name)
    if not _is_dict(value):
        errors.append("{0} must be an object".format(prefix))
        return
    keys = ("command", "description", "actions")
    _require_keys(value, keys, prefix, errors)
    _unexpected_keys(value, keys, prefix, errors)
    if value.get("command") != "mlx-{0}".format(name):
        errors.append("{0}.command must equal 'mlx-{1}'".format(prefix, name))
    _require_string(value, "description", prefix, errors)
    actions = value.get("actions")
    if not isinstance(actions, dict):
        errors.append("{0}.actions must be an object".format(prefix))
        return
    if tuple(actions) != CAPABILITY_ACTIONS[name]:
        errors.append("{0}.actions must equal {1}".format(prefix, list(CAPABILITY_ACTIONS[name])))
    for action_name, action in actions.items():
        action_prefix = "{0}.actions.{1}".format(prefix, action_name)
        if not isinstance(action, dict) or set(action) != {"arguments"} or not isinstance(action.get("arguments"), list):
            errors.append("{0} must contain only an arguments array".format(action_prefix))
            continue
        for index, argument in enumerate(action["arguments"]):
            _validate_argument(argument, "{0}.arguments[{1}]".format(action_prefix, index), errors)


def _validate_provider(value: Any, name: str, errors: List[str]) -> None:
    prefix = "providers.{0}".format(name)
    if not _is_dict(value):
        errors.append("{0} must be an object".format(prefix))
        return
    keys = (
        "native", "capabilities", "commands", "detect_commands", "user_root",
        "project_root", "invocation", "minimum_version", "last_tested_version",
        "version_probe", "install_mode", "artifacts", "config_paths",
    )
    _require_keys(value, keys, prefix, errors)
    _unexpected_keys(value, keys, prefix, errors)
    if value.get("capabilities") != list(CAPABILITIES):
        errors.append("{0}.capabilities must equal ['scout', 'adopt', 'wire']".format(prefix))
    commands = value.get("detect_commands")
    if not isinstance(commands, list) or not all(isinstance(item, str) and item for item in commands):
        errors.append("{0}.detect_commands must be an array of non-empty strings".format(prefix))
    for key in ("user_root", "project_root"):
        if not isinstance(value.get(key), str) or not value[key]:
            errors.append("{0}.{1} must be a non-empty string".format(prefix, key))
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("{0}.artifacts must be a non-empty array".format(prefix))
    elif not all(
        isinstance(item, dict)
        and set(item) in ({"source", "destination"}, {"source", "destination", "scope"}, {"source", "destination", "project_destination"}, {"source", "destination", "project_destination", "scope"})
        and all(isinstance(item[field], str) and item[field] for field in item)
        and ("scope" not in item or item["scope"] in ("user", "project"))
        for item in artifacts
    ):
        errors.append("{0}.artifacts must contain source/destination strings and optional project_destination/user-or-project scope".format(prefix))
    if not isinstance(value.get("config_paths"), list) or not all(isinstance(item, str) for item in value["config_paths"]):
        errors.append("{0}.config_paths must be an array of strings".format(prefix))
    if value.get("invocation") != PROVIDER_INVOCATIONS[name]:
        errors.append("{0}.invocation must equal {1}".format(prefix, PROVIDER_INVOCATIONS[name]))
    for key in ("minimum_version", "last_tested_version"):
        version = value.get(key)
        if version is not None and (not isinstance(version, str) or not SEMVER_PATTERN.fullmatch(version)):
            errors.append("{0}.{1} must be a SemVer core or null".format(prefix, key))
    probe = value.get("version_probe")
    if probe is not None and (
        not isinstance(probe, dict)
        or set(probe) != {"args", "timeout_seconds"}
        or not isinstance(probe["args"], list)
        or not 1 <= len(probe["args"]) <= 4
        or not all(isinstance(item, str) and item for item in probe["args"])
        or not isinstance(probe["timeout_seconds"], (int, float))
        or isinstance(probe["timeout_seconds"], bool)
        or not 0 < probe["timeout_seconds"] <= 5
    ):
        errors.append("{0}.version_probe is invalid".format(prefix))
    if value.get("install_mode") not in ("direct", "staged", "portable"):
        errors.append("{0}.install_mode is invalid".format(prefix))
    if name in NATIVE_PROVIDERS:
        if value.get("native") is not True:
            errors.append("{0}.native must be true".format(prefix))
        if value.get("commands") != PROVIDER_COMMANDS[name]:
            errors.append("{0}.commands must equal {1}".format(prefix, PROVIDER_COMMANDS[name]))
    else:
        if value.get("native") is not False:
            errors.append("{0}.native must be false".format(prefix))
        if value.get("commands") != PROVIDER_COMMANDS[name]:
            errors.append("{0}.commands must equal []".format(prefix))


def _subparsers(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _parser_argument_contract(parser):
    result = []
    for action in parser._actions:
        if action.dest == "help" or isinstance(action, argparse._SubParsersAction):
            continue
        if action.option_strings:
            flag = next((item for item in action.option_strings if item.startswith("--")), action.option_strings[0])
            kind = "option"
            name = flag[2:].replace("-", "_")
            required = bool(action.required)
        else:
            flag = None
            kind = "positional"
            name = action.dest
            required = action.nargs not in ("?", "*")
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            value_type = "boolean"
        elif action.type is int:
            value_type = "integer"
        elif action.type is float:
            value_type = "number"
        else:
            value_type = "string"
        item = {
            "name": name,
            "kind": kind,
            "type": value_type,
            "required": required,
        }
        if flag is not None:
            item["flag"] = flag
        if isinstance(action, argparse._AppendAction):
            item["repeatable"] = True
        result.append(item)
    return result


def _validate_cli_parity(value, errors):
    try:
        from mlx_agent.cli import build_parser
        root = _subparsers(build_parser())
        parser_actions = {
            "scout": {"discover": root["discover"]},
            "adopt": _subparsers(root["adopt"]),
            "wire": _subparsers(root["wire"]),
        }
    except Exception as error:
        errors.append("could not inspect CLI parser: {0}".format(error))
        return
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, dict):
        return
    for capability, actions in parser_actions.items():
        manifest_actions = capabilities.get(capability, {}).get("actions")
        if not isinstance(manifest_actions, dict):
            continue
        for action_name, parser in actions.items():
            if action_name not in manifest_actions:
                continue
            expected = manifest_actions[action_name].get("arguments")
            actual = _parser_argument_contract(parser)
            if expected != actual:
                errors.append(
                    "capabilities.{0}.actions.{1}.arguments do not match the CLI parser: expected {2}, actual {3}".format(
                        capability, action_name, expected, actual
                    )
                )


def _validate_version_parity(value, errors):
    try:
        from mlx_agent import __version__
    except Exception as error:
        errors.append("could not inspect core version: {0}".format(error))
        return
    if value.get("version") != __version__:
        errors.append("manifest version does not match mlx_agent.__version__")
    matrix_path = Path(__file__).resolve().parents[1] / "compatibility" / "providers.json"
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append("could not inspect compatibility version: {0}".format(error))
        return
    if matrix.get("plugin_version") != value.get("version"):
        errors.append("compatibility plugin_version does not match manifest version")


def validate_manifest(path: Path) -> List[str]:
    """Return canonical-manifest errors without requiring a JSON Schema package."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return ["could not read manifest: {0}".format(error)]
    if not _is_dict(value):
        return ["manifest must be an object"]

    errors: List[str] = []
    manifest_keys = (
        "$schema",
        "schema_version",
        "version",
        "identity",
        "scopes",
        "requirements",
        "safety",
        "capabilities",
        "providers",
    )
    _require_keys(value, manifest_keys, "manifest", errors)
    _unexpected_keys(value, manifest_keys, "manifest", errors)
    _require_string(value, "$schema", "manifest", errors)
    _require_string(value, "schema_version", "manifest", errors)
    _require_string(value, "version", "manifest", errors)
    _require_string(value, "identity", "manifest", errors)
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must equal '1.0'")
    if value.get("identity") != "mlx-agent":
        errors.append("identity must equal 'mlx-agent'")
    if value.get("version") != PLUGIN_VERSION:
        errors.append("version must equal '{0}'".format(PLUGIN_VERSION))

    scopes = value.get("scopes")
    if not isinstance(scopes, list):
        errors.append("scopes must be an array")
    elif len(scopes) != 2 or not all(
        scope in ("user", "project") for scope in scopes
    ) or scopes[0] == scopes[1]:
        errors.append("scopes must equal ['user', 'project']")

    requirements = value.get("requirements")
    if not _is_dict(requirements):
        errors.append("requirements must be an object")
    else:
        _require_keys(requirements, ("python3",), "requirements", errors)
        _unexpected_keys(requirements, ("python3",), "requirements", errors)
        if requirements.get("python3") != ">=3.9":
            errors.append("requirements.python3 must equal '>=3.9'")

    safety = value.get("safety")
    safety_keys = ("auto_install_provider_cli", "auto_download_model", "persist_secrets")
    if not _is_dict(safety):
        errors.append("safety must be an object")
    else:
        _require_keys(safety, safety_keys, "safety", errors)
        _unexpected_keys(safety, safety_keys, "safety", errors)
        for key in safety_keys:
            if safety.get(key) is not False:
                errors.append("safety.{0} must be false".format(key))

    capabilities = value.get("capabilities")
    if not _is_dict(capabilities):
        errors.append("capabilities must be an object")
    else:
        if set(capabilities) != set(CAPABILITIES):
            errors.append("capabilities must equal ['scout', 'adopt', 'wire']")
        for capability in CAPABILITIES:
            if capability in capabilities:
                _validate_capability(capabilities[capability], capability, errors)

    providers = value.get("providers")
    if not _is_dict(providers):
        errors.append("providers must be an object")
    else:
        if set(providers) != set(PROVIDERS):
            errors.append(
                "providers must equal ['claude', 'codex', 'gemini', 'opencode', 'agentskills']"
            )
        for provider in PROVIDERS:
            if provider in providers:
                _validate_provider(providers[provider], provider, errors)

    _validate_cli_parity(value, errors)
    _validate_version_parity(value, errors)

    return errors


if __name__ == "__main__":
    import sys

    manifest_path = Path(__file__).resolve().parents[1] / "plugin.json"
    validation_errors = validate_manifest(manifest_path)
    for validation_error in validation_errors:
        print(validation_error)
    sys.exit(1 if validation_errors else 0)
