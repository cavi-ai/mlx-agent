"""Focused standard-library validation for the canonical MLX agent contracts."""

import json
from pathlib import Path
from typing import Any, Dict, List


SCHEMA_VERSION = "1.0"
CAPABILITIES = ("scout", "adopt", "wire")
NATIVE_PROVIDERS = ("claude", "codex", "gemini", "opencode")
PROVIDERS = NATIVE_PROVIDERS + ("agentskills",)


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _require_mapping_values(
    value: Dict[str, Any],
    keys: List[str],
    prefix: str,
    errors: List[str],
) -> None:
    for key in keys:
        if key not in value:
            errors.append("{0}.{1} is required".format(prefix, key))


def _require_string(value: Dict[str, Any], key: str, prefix: str, errors: List[str]) -> None:
    if key not in value:
        errors.append("{0}.{1} must be a string".format(prefix, key))
    elif not isinstance(value[key], str):
        errors.append("{0}.{1} must be a string".format(prefix, key))


def validate_result(value: dict) -> List[str]:
    """Return contract errors for a serialized ResultEnvelope, if any."""
    errors: List[str] = []
    if not _is_dict(value):
        return ["result must be an object"]

    required = ["schema_version", "generated_at", "operation", "status", "data", "warnings"]
    _require_mapping_values(value, required, "result", errors)
    for key in ("schema_version", "generated_at", "operation"):
        _require_string(value, key, "result", errors)

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
            if not _is_dict(warning) or not all(
                isinstance(item, str) for item in warning.values()
            ):
                errors.append("warnings[{0}] must be an object of strings".format(index))

    if value.get("status") == "error":
        error = value.get("error")
        if not _is_dict(error):
            errors.append("error must be an object")
        else:
            for key in ("code", "message", "remediation"):
                _require_string(error, key, "error", errors)
            if "retryable" not in error:
                errors.append("error.retryable is required")
            elif not isinstance(error["retryable"], bool):
                errors.append("error.retryable must be a boolean")
    elif "error" in value:
        errors.append("error is only allowed when status is 'error'")

    return errors


def _validate_argument(value: Any, prefix: str, errors: List[str]) -> None:
    if not _is_dict(value):
        errors.append("{0} must be an object".format(prefix))
        return
    _require_string(value, "name", prefix, errors)
    if value.get("type") not in ("string", "integer", "boolean"):
        errors.append("{0}.type must be one of ['string', 'integer', 'boolean']".format(prefix))
    if not isinstance(value.get("required"), bool):
        errors.append("{0}.required must be a boolean".format(prefix))


def validate_manifest(path: Path) -> List[str]:
    """Return canonical-manifest errors without requiring a JSON Schema package."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return ["could not read manifest: {0}".format(error)]

    errors: List[str] = []
    if not _is_dict(value):
        return ["manifest must be an object"]

    _require_mapping_values(
        value,
        ["$schema", "schema_version", "identity", "scopes", "requirements", "safety", "capabilities", "providers"],
        "manifest",
        errors,
    )
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must equal '1.0'")
    if value.get("identity") != "mlx-agent":
        errors.append("identity must equal 'mlx-agent'")
    if set(value.get("scopes", [])) != {"user", "project"}:
        errors.append("scopes must equal ['user', 'project']")

    requirements = value.get("requirements")
    if not _is_dict(requirements):
        errors.append("requirements must be an object")
    elif requirements.get("python3") != ">=3.9":
        errors.append("requirements.python3 must equal '>=3.9'")

    safety = value.get("safety")
    if not _is_dict(safety):
        errors.append("safety must be an object")
    else:
        for key in ("auto_install_provider_cli", "auto_download_model", "persist_secrets"):
            if safety.get(key) is not False:
                errors.append("safety.{0} must be false".format(key))

    capabilities = value.get("capabilities")
    if not _is_dict(capabilities):
        errors.append("capabilities must be an object")
    elif set(capabilities) != set(CAPABILITIES):
        errors.append("capabilities must equal ['scout', 'adopt', 'wire']")
    else:
        for capability in CAPABILITIES:
            definition = capabilities[capability]
            prefix = "capabilities.{0}".format(capability)
            if not _is_dict(definition):
                errors.append("{0} must be an object".format(prefix))
                continue
            if definition.get("command") != "mlx-{0}".format(capability):
                errors.append("{0}.command must equal 'mlx-{1}'".format(prefix, capability))
            _require_string(definition, "description", prefix, errors)
            arguments = definition.get("arguments")
            if not isinstance(arguments, list):
                errors.append("{0}.arguments must be an array".format(prefix))
            else:
                for index, argument in enumerate(arguments):
                    _validate_argument(argument, "{0}.arguments[{1}]".format(prefix, index), errors)

    expected_commands = ["mlx-{0}".format(capability) for capability in CAPABILITIES]
    providers = value.get("providers")
    if not _is_dict(providers):
        errors.append("providers must be an object")
    elif set(providers) != set(PROVIDERS):
        errors.append("providers must equal ['claude', 'codex', 'gemini', 'opencode', 'agentskills']")
    else:
        for provider in PROVIDERS:
            definition = providers[provider]
            prefix = "providers.{0}".format(provider)
            if not _is_dict(definition):
                errors.append("{0} must be an object".format(prefix))
                continue
            if definition.get("capabilities") != list(CAPABILITIES):
                errors.append("{0}.capabilities must equal ['scout', 'adopt', 'wire']".format(prefix))
            if provider in NATIVE_PROVIDERS:
                if definition.get("native") is not True:
                    errors.append("{0}.native must be true".format(prefix))
                if definition.get("commands") != expected_commands:
                    errors.append("{0}.commands must equal {1}".format(prefix, expected_commands))
            else:
                if definition.get("native") is not False:
                    errors.append("{0}.native must be false".format(prefix))
                if definition.get("commands") != []:
                    errors.append("{0}.commands must equal []".format(prefix))

    return errors


if __name__ == "__main__":
    import sys

    manifest_path = Path(__file__).resolve().parents[1] / "plugin.json"
    validation_errors = validate_manifest(manifest_path)
    for validation_error in validation_errors:
        print(validation_error)
    sys.exit(1 if validation_errors else 0)
