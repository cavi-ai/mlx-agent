"""Manifest-backed provider definitions and non-mutating provider detection."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderArtifact:
    source: Path
    destination: Path


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    detect_commands: tuple
    user_root: Path
    project_root: Path
    artifacts: tuple
    config_paths: tuple

    def destination(self, scope, project_root=None):
        if scope == "user":
            return self.user_root
        if scope != "project":
            raise ValueError("scope must be 'user' or 'project'")
        if project_root is None:
            raise ValueError("project scope requires a project root")
        return Path(project_root).resolve() / self.project_root


@dataclass(frozen=True)
class ProviderDetection:
    id: str
    available: bool
    command: str = ""
    command_path: str = None

    def to_dict(self):
        return {
            "id": self.id,
            "available": self.available,
            "command": self.command or None,
            "command_path": self.command_path,
        }


def _safe_relative(value, field):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("{0} must be a safe relative path".format(field))
    return path


class ProviderRegistry:
    """Load immutable installation data from the canonical plugin manifest."""

    def __init__(self, manifest_path, home=None, config_root=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.home = Path(home).expanduser().resolve() if home else Path.home().resolve()
        self.config_root = (
            Path(config_root).expanduser().resolve() if config_root else self.home
        )

    def definitions(self):
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("provider manifest could not be read: {0}".format(error))
        providers = manifest.get("providers")
        if not isinstance(providers, dict):
            raise ValueError("provider manifest has no providers object")
        definitions = {}
        for provider_id, value in providers.items():
            definitions[provider_id] = self._definition(provider_id, value)
        return definitions

    def _definition(self, provider_id, value):
        if not isinstance(value, dict):
            raise ValueError("provider {0} must be an object".format(provider_id))
        required = ("detect_commands", "user_root", "project_root", "artifacts", "config_paths")
        if any(key not in value for key in required):
            raise ValueError("provider {0} has no installer definition".format(provider_id))
        commands = value["detect_commands"]
        if not isinstance(commands, list) or not all(isinstance(item, str) and item for item in commands):
            raise ValueError("provider {0}.detect_commands is invalid".format(provider_id))
        user_root = self._user_root(value["user_root"], provider_id)
        project_root = self._project_root(value["project_root"], provider_id)
        artifacts = []
        if not isinstance(value["artifacts"], list) or not value["artifacts"]:
            raise ValueError("provider {0}.artifacts is invalid".format(provider_id))
        for index, artifact in enumerate(value["artifacts"]):
            if not isinstance(artifact, dict) or set(artifact) != {"source", "destination"}:
                raise ValueError("provider {0}.artifacts[{1}] is invalid".format(provider_id, index))
            source = _safe_relative(artifact["source"], "artifact source")
            location = (self.manifest_path.parent / source).resolve()
            if self.manifest_path.parent not in location.parents or not location.is_file():
                raise ValueError("provider artifact source is outside the plugin or missing: {0}".format(source))
            artifacts.append(ProviderArtifact(location, _safe_relative(artifact["destination"], "artifact destination")))
        config_paths = value["config_paths"]
        if not isinstance(config_paths, list) or not all(isinstance(item, str) for item in config_paths):
            raise ValueError("provider {0}.config_paths is invalid".format(provider_id))
        return ProviderDefinition(
            id=provider_id,
            detect_commands=tuple(commands),
            user_root=user_root,
            project_root=project_root,
            artifacts=tuple(artifacts),
            config_paths=tuple(config_paths),
        )

    def _user_root(self, template, provider_id):
        if not isinstance(template, str) or "{project}" in template:
            raise ValueError("provider {0}.user_root is invalid".format(provider_id))
        try:
            path = Path(template.format(home=str(self.home), config_root=str(self.config_root))).resolve()
        except (KeyError, ValueError) as error:
            raise ValueError("provider {0}.user_root is invalid: {1}".format(provider_id, error))
        allowed_roots = (self.home, self.config_root)
        if not any(root == path or root in path.parents for root in allowed_roots):
            raise ValueError("provider {0}.user_root must be under home or config_root".format(provider_id))
        return path

    def _project_root(self, template, provider_id):
        if not isinstance(template, str) or "{project}" not in template:
            raise ValueError("provider {0}.project_root is invalid".format(provider_id))
        marker = Path("/__mlx_agent_project_marker__")
        try:
            rendered = Path(template.format(project=str(marker), home=str(self.home), config_root=str(self.config_root))).resolve()
            return rendered.relative_to(marker)
        except (KeyError, ValueError):
            raise ValueError("provider {0}.project_root is invalid".format(provider_id))


def detect_providers(definitions, env=None, path=None, executable_lookup=None):
    """Report installed provider CLIs only; this function never installs anything."""
    environment = dict(os.environ if env is None else env)
    search_path = path if path is not None else environment.get("PATH", "")
    lookup = executable_lookup or shutil.which
    detections = []
    for definition in definitions:
        command_path = None
        command = ""
        for candidate in definition.detect_commands:
            found = lookup(candidate, path=search_path)
            if found:
                command, command_path = candidate, str(found)
                break
        detections.append(ProviderDetection(definition.id, bool(command_path), command, command_path))
    return detections
