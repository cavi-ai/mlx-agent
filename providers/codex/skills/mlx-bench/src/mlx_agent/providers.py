"""Manifest-backed provider definitions and non-mutating provider detection."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


_GEMINI_EXTENSION_SUFFIX = Path(".gemini") / "extensions" / "mlx-agent"


@dataclass(frozen=True)
class ProviderArtifact:
    source: Path
    destination: Path
    project_destination: Path = None
    scopes: tuple = ("user", "project")


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    detect_commands: tuple
    user_root: Path
    project_root: Path
    invocation_kind: str
    invocation_prefix: str
    minimum_version: str
    last_tested_version: str
    version_probe_args: tuple
    version_probe_timeout: float
    install_mode: str
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

    def artifact_destination(self, scope, project_root, artifact):
        """Return the receipt-owned target for one declared artifact."""
        if scope == "project" and artifact.project_destination is not None:
            return Path(project_root).resolve() / artifact.project_destination
        return self.destination(scope, project_root) / artifact.destination

    @staticmethod
    def applies_to(scope, artifact):
        return scope in artifact.scopes


@dataclass(frozen=True)
class ProviderDetection:
    id: str
    available: bool
    command: str = ""
    command_path: str = None
    state: str = "absent"
    version: str = None
    minimum_version: str = None
    last_tested_version: str = None
    error: str = None
    install_mode: str = "direct"

    def to_dict(self):
        return {
            "id": self.id,
            "available": self.available,
            "command": self.command or None,
            "command_path": self.command_path,
            "state": self.state,
            "version": self.version,
            "minimum_version": self.minimum_version,
            "last_tested_version": self.last_tested_version,
            "error": self.error,
            "install_mode": self.install_mode,
        }


def _safe_relative(value, field, allow_current=False):
    path = Path(value)
    if allow_current and str(value) == ".":
        return path
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("{0} must be a safe relative path".format(field))
    return path


class ProviderRegistry:
    """Load immutable installation data from the canonical plugin manifest."""

    def __init__(self, manifest_path, home=None, config_root=None, xdg_config_home=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.home = Path(home).expanduser().resolve() if home else Path.home().resolve()
        self.config_root = (
            Path(config_root).expanduser().resolve() if config_root else self.home
        )
        self.xdg_config_home = (
            Path(xdg_config_home).expanduser().resolve()
            if xdg_config_home else (self.home / ".config").resolve()
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
        required = (
            "detect_commands", "user_root", "project_root", "artifacts", "config_paths",
            "invocation", "minimum_version", "last_tested_version", "version_probe",
            "install_mode",
        )
        if any(key not in value for key in required):
            raise ValueError("provider {0} has no installer definition".format(provider_id))
        commands = value["detect_commands"]
        if not isinstance(commands, list) or not all(isinstance(item, str) and item for item in commands):
            raise ValueError("provider {0}.detect_commands is invalid".format(provider_id))
        user_root = self._user_root(value["user_root"], provider_id)
        project_root = self._project_root(value["project_root"], provider_id)
        invocation = value["invocation"]
        if not isinstance(invocation, dict) or set(invocation) != {"kind", "prefix"}:
            raise ValueError("provider {0}.invocation is invalid".format(provider_id))
        invocation_kind, invocation_prefix = invocation["kind"], invocation["prefix"]
        if invocation_kind not in {"command", "skill"} or not isinstance(invocation_prefix, str):
            raise ValueError("provider {0}.invocation is invalid".format(provider_id))
        if invocation_kind == "command" and invocation_prefix != "/":
            raise ValueError("provider {0}.invocation is invalid".format(provider_id))
        if invocation_kind == "skill" and invocation_prefix not in {"", "$"}:
            raise ValueError("provider {0}.invocation is invalid".format(provider_id))
        minimum_version = value["minimum_version"]
        last_tested_version = value["last_tested_version"]
        for label, version in (
            ("minimum_version", minimum_version),
            ("last_tested_version", last_tested_version),
        ):
            if version is not None and not _SEMVER.fullmatch(version):
                raise ValueError("provider {0}.{1} is invalid".format(provider_id, label))
        probe = value["version_probe"]
        if probe is None:
            version_probe_args, version_probe_timeout = (), 0.0
        elif (
            not isinstance(probe, dict)
            or set(probe) != {"args", "timeout_seconds"}
            or not isinstance(probe["args"], list)
            or not 1 <= len(probe["args"]) <= 4
            or not all(isinstance(item, str) and item and len(item) <= 64 for item in probe["args"])
            or not isinstance(probe["timeout_seconds"], (int, float))
            or isinstance(probe["timeout_seconds"], bool)
            or not 0 < probe["timeout_seconds"] <= 5
        ):
            raise ValueError("provider {0}.version_probe is invalid".format(provider_id))
        else:
            version_probe_args = tuple(probe["args"])
            version_probe_timeout = float(probe["timeout_seconds"])
        install_mode = value["install_mode"]
        if install_mode not in {"direct", "staged", "portable"}:
            raise ValueError("provider {0}.install_mode is invalid".format(provider_id))
        if install_mode == "portable" and (commands or probe is not None or minimum_version is not None or last_tested_version is not None):
            raise ValueError("portable providers must not declare CLI compatibility probes")
        if install_mode != "portable" and (not commands or probe is None or minimum_version is None):
            raise ValueError("native providers require a minimum version and bounded probe")
        artifacts = []
        if not isinstance(value["artifacts"], list) or not value["artifacts"]:
            raise ValueError("provider {0}.artifacts is invalid".format(provider_id))
        for index, artifact in enumerate(value["artifacts"]):
            if not isinstance(artifact, dict) or set(artifact) not in ({"source", "destination"}, {"source", "destination", "scope"}, {"source", "destination", "project_destination"}, {"source", "destination", "project_destination", "scope"}):
                raise ValueError("provider {0}.artifacts[{1}] is invalid".format(provider_id, index))
            source = _safe_relative(artifact["source"], "artifact source")
            unresolved = self.manifest_path.parent / source
            cursor = self.manifest_path.parent
            for component in source.parts:
                cursor = cursor / component
                if cursor.is_symlink():
                    raise ValueError(
                        "provider artifact source must not contain symlinks: {0}".format(source)
                    )
            location = unresolved.resolve()
            if self.manifest_path.parent not in location.parents or not (location.is_file() or location.is_dir()):
                raise ValueError("provider artifact source is outside the plugin or missing: {0}".format(source))
            destination = _safe_relative(
                artifact["destination"], "artifact destination", allow_current=True
            )
            project_destination = (
                _safe_relative(artifact["project_destination"], "artifact project_destination")
                if "project_destination" in artifact else None
            )
            scopes = (artifact.get("scope", "user"),) if "scope" in artifact else ("user", "project")
            if scopes not in (("user",), ("project",)) and scopes != ("user", "project"):
                raise ValueError("provider {0}.artifacts[{1}].scope is invalid".format(provider_id, index))
            if location.is_file():
                artifacts.append(ProviderArtifact(location, destination, project_destination, scopes))
            else:
                for child in sorted(location.rglob("*")):
                    if child.is_symlink():
                        raise ValueError(
                            "provider artifact source must not contain symlinks: {0}".format(child)
                        )
                    resolved = child.resolve()
                    relative = child.relative_to(location)
                    if "__pycache__" in relative.parts or child.suffix == ".pyc":
                        continue
                    if child.is_file() and location in resolved.parents:
                        project_child = project_destination / child.relative_to(location) if project_destination else None
                        artifacts.append(ProviderArtifact(resolved, destination / child.relative_to(location), project_child, scopes))
        config_paths = value["config_paths"]
        if not isinstance(config_paths, list) or not all(isinstance(item, str) for item in config_paths):
            raise ValueError("provider {0}.config_paths is invalid".format(provider_id))
        if provider_id == "gemini":
            self._validate_gemini_extension_layout(user_root, project_root, artifacts)
        if provider_id == "opencode":
            self._validate_opencode_layout(user_root, project_root, artifacts)
        return ProviderDefinition(
            id=provider_id,
            detect_commands=tuple(commands),
            user_root=user_root,
            project_root=project_root,
            invocation_kind=invocation_kind,
            invocation_prefix=invocation_prefix,
            minimum_version=minimum_version,
            last_tested_version=last_tested_version,
            version_probe_args=version_probe_args,
            version_probe_timeout=version_probe_timeout,
            install_mode=install_mode,
            artifacts=tuple(artifacts),
            config_paths=tuple(config_paths),
        )

    def _validate_gemini_extension_layout(self, user_root, project_root, artifacts):
        """Keep Gemini's install roots aligned with its extension discovery layout."""
        if user_root != (self.home / _GEMINI_EXTENSION_SUFFIX).resolve():
            raise ValueError("provider gemini.user_root must target home/.gemini/extensions/mlx-agent")
        if project_root != _GEMINI_EXTENSION_SUFFIX:
            raise ValueError("provider gemini.project_root must target .gemini/extensions/mlx-agent")
        if not any(item.destination == Path("gemini-extension.json") for item in artifacts):
            raise ValueError("provider gemini must install gemini-extension.json")
        for item in artifacts:
            if item.project_destination is not None and item.project_destination.parts[:1] != (".gemini",):
                raise ValueError("provider gemini project artifacts must stay under .gemini")

    def _validate_opencode_layout(self, user_root, project_root, artifacts):
        """Keep OpenCode's documented global and project discovery locations exact."""
        if user_root != (self.xdg_config_home / "opencode").resolve():
            raise ValueError("provider opencode.user_root must target the OpenCode XDG config directory")
        if project_root != Path(".opencode"):
            raise ValueError("provider opencode.project_root must target .opencode")
        if not any(item.destination == Path("commands/mlx-scout.md") for item in artifacts):
            raise ValueError("provider opencode must install native command files")
        if not any(item.destination == Path("agents/mlx-advisor.md") for item in artifacts):
            raise ValueError("provider opencode must install its advisor agent")
        if not any(item.destination == Path("skills/mlx-scout/SKILL.md") for item in artifacts):
            raise ValueError("provider opencode must install native skills")
        if any(item.destination == Path("opencode.json") for item in artifacts):
            raise ValueError("provider opencode must not mutate global or project config")
        if not any(item.destination == Path("plugins/mlx-agent-command.ts") for item in artifacts):
            raise ValueError("provider opencode must install its native custom tool")
        if not any(item.destination == Path("src/mlx_agent/command_executor.py") for item in artifacts):
            raise ValueError("provider opencode must install its custom-tool runtime")

    def _user_root(self, template, provider_id):
        if not isinstance(template, str) or "{project}" in template:
            raise ValueError("provider {0}.user_root is invalid".format(provider_id))
        try:
            path = Path(template.format(
                home=str(self.home), config_root=str(self.config_root),
                xdg_config_home=str(self.xdg_config_home),
            )).resolve()
        except (KeyError, ValueError) as error:
            raise ValueError("provider {0}.user_root is invalid: {1}".format(provider_id, error))
        allowed_roots = (self.home, self.config_root, self.xdg_config_home)
        if not any(root == path or root in path.parents for root in allowed_roots):
            raise ValueError("provider {0}.user_root must be under an approved user configuration root".format(provider_id))
        return path

    def _project_root(self, template, provider_id):
        if not isinstance(template, str) or "{project}" not in template:
            raise ValueError("provider {0}.project_root is invalid".format(provider_id))
        marker = Path("/__mlx_agent_project_marker__")
        try:
            rendered = Path(template.format(
                project=str(marker), home=str(self.home), config_root=str(self.config_root),
                xdg_config_home=str(self.xdg_config_home),
            )).resolve()
            return rendered.relative_to(marker)
        except (KeyError, ValueError):
            raise ValueError("provider {0}.project_root is invalid".format(provider_id))


_SEMVER = re.compile(r"(?<![0-9])((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))(?![0-9])")
_MAX_VERSION_OUTPUT = 8192


def _version_tuple(value):
    return tuple(int(part) for part in value.split("."))


def _run_bounded_probe(argv, *, timeout, env):
    """Run one no-shell probe while bounding time and captured output bytes."""
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        env=env,
    )
    selector = selectors.DefaultSelector()
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            ready = selector.select(remaining)
            if not ready:
                if process.poll() is not None:
                    break
                raise subprocess.TimeoutExpired(argv, timeout)
            chunk = os.read(
                process.stdout.fileno(),
                min(4096, _MAX_VERSION_OUTPUT + 1 - len(output)),
            )
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > _MAX_VERSION_OUTPUT:
                raise ValueError("version probe output exceeded the bounded limit")
        returncode = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        return subprocess.CompletedProcess(argv, returncode, bytes(output))
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()


def detect_providers(definitions, env=None, path=None, executable_lookup=None, probe_runner=None):
    """Report installed provider CLIs only; this function never installs anything."""
    environment = dict(os.environ if env is None else env)
    search_path = path if path is not None else environment.get("PATH", "")
    lookup = executable_lookup or shutil.which
    detections = []
    for definition in definitions:
        if definition.install_mode == "portable":
            detections.append(ProviderDetection(
                definition.id, True, state="portable", install_mode="portable"
            ))
            continue
        command_path = None
        command = ""
        for candidate in definition.detect_commands:
            found = lookup(candidate, path=search_path)
            if found:
                command, command_path = candidate, str(found)
                break
        if not command_path:
            detections.append(ProviderDetection(
                definition.id, False, command, command_path, state="absent",
                minimum_version=definition.minimum_version,
                last_tested_version=definition.last_tested_version,
                error="provider executable was not found",
                install_mode=definition.install_mode,
            ))
            continue
        try:
            argv = [command_path] + list(definition.version_probe_args)
            if probe_runner is None:
                completed = _run_bounded_probe(
                    argv,
                    timeout=definition.version_probe_timeout,
                    env=environment,
                )
            else:
                completed = probe_runner(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=definition.version_probe_timeout,
                    check=False,
                    shell=False,
                    env=environment,
                )
            raw = completed.stdout or b""
            if isinstance(raw, bytes):
                output = raw[:_MAX_VERSION_OUTPUT].decode("utf-8", errors="replace")
            else:
                output = str(raw)[:_MAX_VERSION_OUTPUT]
            match = _SEMVER.search(output)
            if completed.returncode != 0 or match is None:
                raise ValueError("version probe did not return a SemVer core")
            version = match.group(1)
            supported = definition.last_tested_version is not None and (
                _version_tuple(definition.minimum_version)
                <= _version_tuple(version)
                <= _version_tuple(definition.last_tested_version)
            )
            state = "native-visible" if supported else "unsupported"
            detections.append(ProviderDetection(
                definition.id, supported, command, command_path, state=state,
                version=version, minimum_version=definition.minimum_version,
                last_tested_version=definition.last_tested_version,
                error=None if supported else "provider version is outside the validated range",
                install_mode=definition.install_mode,
            ))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            detections.append(ProviderDetection(
                definition.id, False, command, command_path, state="unsupported",
                minimum_version=definition.minimum_version,
                last_tested_version=definition.last_tested_version,
                error=str(error), install_mode=definition.install_mode,
            ))
    return detections
