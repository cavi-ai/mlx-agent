"""One-shot per-role routing configuration backed by Wire transactions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .wiring import _MODEL, require_secret_free_config


FLEET_RUNTIMES = ("mlx_lm", "mlx-vlm")
ROLE_DEFAULT_RUNTIME = {
    "general": "mlx_lm",
    "coding": "mlx_lm",
    "reasoning": "mlx_lm",
    "embedding": "mlx_lm",
    "tool-use": "mlx_lm",
    "vision": "mlx-vlm",
}
RUNTIME_PORTS = {"mlx_lm": 8080, "mlx-vlm": 8083}
_ROLE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
MAX_ASSIGNMENTS = 6

_HEADER = "# MLX_AGENT_WIRE v1"


class FleetError(ValueError):
    """Classified fleet failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def parse_assignments(values):
    """Parse repeatable role=repo assignments into an ordered mapping."""
    assignments = {}
    for value in values or []:
        if not isinstance(value, str) or "=" not in value:
            raise FleetError(
                "invalid_assignment",
                "Assignments must use role=repo form: {0}".format(value),
                "Pass --assign coding=publisher/model.",
            )
        role, repo = value.split("=", 1)
        role = role.strip()
        repo = repo.strip()
        if role not in ROLE_DEFAULT_RUNTIME:
            raise FleetError(
                "invalid_role",
                "Unknown fleet role: {0}".format(role),
                "Use one of: {0}.".format(", ".join(sorted(ROLE_DEFAULT_RUNTIME))),
            )
        if role in assignments:
            raise FleetError(
                "duplicate_role",
                "Role {0} is assigned twice.".format(role),
                "Assign each role at most once.",
            )
        if not _MODEL.fullmatch(repo):
            raise FleetError(
                "invalid_repo",
                "Model must be a safe publisher/model identifier: {0}".format(repo),
                "Pass the Hugging Face publisher/model form.",
            )
        assignments[role] = repo
    if not assignments:
        raise FleetError(
            "missing_assignments",
            "Fleet requires at least one role assignment.",
            "Pass --assign role=repo or --from-adoption <state-path>.",
        )
    if len(assignments) > MAX_ASSIGNMENTS:
        raise FleetError(
            "invalid_assignment",
            "Fleet supports at most {0} role assignments.".format(MAX_ASSIGNMENTS),
            "Reduce the number of assigned roles.",
        )
    return assignments


def assignments_from_adoption(state_path):
    """Read role -> repo recommendations from a completed adopt handoff."""
    try:
        value = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise FleetError(
            "invalid_state",
            "The adoption handoff could not be read: {0}".format(error),
            "Pass a readable adopt state file produced by adopt start/resume.",
        )
    recommendations = value.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        raise FleetError(
            "empty_recommendations",
            "The adoption handoff contains no recommendations.",
            "Complete the adoption workflow first, then rerun fleet.",
        )
    assignments = {}
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        repo = item.get("repo")
        if role in ROLE_DEFAULT_RUNTIME and isinstance(repo, str) and role not in assignments:
            assignments[role] = repo
    return parse_assignments(["{0}={1}".format(role, repo) for role, repo in assignments.items()])


def parse_runtime_map(values):
    """Parse repeatable role=runtime overrides."""
    runtime_map = {}
    for value in values or []:
        if not isinstance(value, str) or "=" not in value:
            raise FleetError(
                "invalid_runtime_map",
                "Runtime overrides must use role=runtime form: {0}".format(value),
                "Pass --runtime-map vision=mlx-vlm.",
            )
        role, runtime = value.split("=", 1)
        role = role.strip()
        runtime = runtime.strip()
        if role not in ROLE_DEFAULT_RUNTIME:
            raise FleetError(
                "invalid_role",
                "Unknown fleet role: {0}".format(role),
                "Use one of: {0}.".format(", ".join(sorted(ROLE_DEFAULT_RUNTIME))),
            )
        if runtime not in FLEET_RUNTIMES:
            raise FleetError(
                "unsupported_runtime",
                "Fleet routes only through: {0}.".format(", ".join(FLEET_RUNTIMES)),
                "Ollama and LM Studio manage their own routing; use mlx_lm or mlx-vlm.",
            )
        runtime_map[role] = runtime
    return runtime_map


class FleetConfigAdapter:
    """Validate the exact bounded multi-model router subset fleet renders."""

    version = "1.0"
    runtime = "litellm-fleet"

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None

    def render(self, assignments, runtime_map=None, existing=""):
        runtime_map = runtime_map or {}
        if not assignments:
            raise FleetError(
                "missing_assignments",
                "Fleet requires at least one role assignment.",
                "Pass --assign role=repo or --from-adoption <state-path>.",
            )
        if not isinstance(existing, str):
            raise TypeError("existing config content must be text")
        require_secret_free_config(existing)
        if existing.strip() and not existing.startswith(_HEADER):
            raise FleetError(
                "existing_config_unmanaged",
                "The target file exists and is not fleet-managed.",
                "Pick a new --path or remove the unmanaged file yourself.",
            )
        lines = [_HEADER, "model_list:"]
        for role in sorted(assignments):
            runtime = runtime_map.get(role, ROLE_DEFAULT_RUNTIME[role])
            if runtime not in FLEET_RUNTIMES:
                raise FleetError(
                    "unsupported_runtime",
                    "Fleet routes only through: {0}.".format(", ".join(FLEET_RUNTIMES)),
                    "Use mlx_lm or mlx-vlm for every role.",
                )
            port = RUNTIME_PORTS[runtime]
            lines.extend([
                "  - model_name: {0}".format(role),
                "    litellm_params:",
                "      model: openai/{0}".format(assignments[role]),
                "      api_base: http://127.0.0.1:{0}/v1".format(port),
                "      api_key: os.environ/MLX_AGENT_LOCAL_API_KEY",
            ])
        content = "\n".join(lines) + "\n"
        self.validate(content)
        return content

    def validate(self, content):
        if not isinstance(content, str):
            raise TypeError("configuration content must be text")
        require_secret_free_config(content)
        lines = content.splitlines()
        if len(lines) < 7 or (len(lines) - 2) % 5 != 0:
            raise ValueError("fleet configuration must contain whole five-line entries")
        if lines[0] != _HEADER or lines[1] != "model_list:":
            raise ValueError("fleet configuration must start with the managed header")
        roles = []
        for offset in range(2, len(lines), 5):
            header, params, model, api_base, api_key = lines[offset:offset + 5]
            role = re.fullmatch(r"  - model_name: ([a-z][a-z0-9-]{0,31})", header)
            if role is None or not _ROLE.fullmatch(role.group(1)):
                raise ValueError("fleet entry has an invalid model_name")
            if params != "    litellm_params:":
                raise ValueError("fleet entry is missing litellm_params")
            model_match = re.fullmatch(r"      model: openai/(" + _MODEL.pattern[1:-1] + r")", model)
            if model_match is None:
                raise ValueError("fleet entry has an invalid model identifier")
            if api_base not in (
                "      api_base: http://127.0.0.1:8080/v1",
                "      api_base: http://127.0.0.1:8083/v1",
            ):
                raise ValueError("fleet entry api_base must be a bounded loopback port")
            if api_key != "      api_key: os.environ/MLX_AGENT_LOCAL_API_KEY":
                raise ValueError("fleet entry must reference the managed API key env var")
            roles.append(role.group(1))
        if len(roles) != len(set(roles)):
            raise ValueError("fleet configuration contains duplicate roles")
        return True
