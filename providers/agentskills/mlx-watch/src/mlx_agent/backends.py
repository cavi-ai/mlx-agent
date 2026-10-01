"""Declared MLX converter backends: manifests, model-type registries, isolated envs."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import zipfile
import sys
from pathlib import Path

BACKENDS_DIR = Path(__file__).resolve().parent / "resources" / "backends"
MANIFEST_SCHEMA = "backend/1"
INSTALL_MARKER = ".mlx-agent-backend.json"
CATEGORIES = ("text_llm", "vision_language", "speech_to_text", "text_to_speech")
MAX_PROBE_FILES = 4000
MAX_PROBE_FILE_BYTES = 1024 * 1024
TYPE_PREFERENCE = {
    "speech_to_text": ("mlx-audio",),
    "text_to_speech": ("mlx-audio",),
    "vision_language": ("mlx-vlm", "mlx-lm"),
    "text_llm": ("mlx-lm", "mlx-vlm"),
}
DEFAULT_PREFERENCE = ("mlx-lm", "mlx-vlm", "mlx-audio")
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
_STRIP_SUFFIXES = ("_encoder", "_decoder", "_text", "_vision", "_audio", "_model")
_VISION_FILE = re.compile(r"(vision|visual|image|siglip|clip)", re.IGNORECASE)
_ENV_ALLOWLIST = (
    "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "HF_HOME", "HF_HUB_CACHE",
    "HF_HUB_OFFLINE", "XDG_CACHE_HOME", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
)


class BackendError(RuntimeError):
    """Classified backend failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def normalize_type(value):
    return str(value).strip().lower().replace("-", "_")


def squash_type(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def load_manifests(directory=BACKENDS_DIR):
    manifests = {}
    for path in sorted(Path(directory).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        _validate_manifest(value, path)
        manifests[value["id"]] = value
    return manifests


def _invalid(path, detail):
    return BackendError(
        "manifest_invalid",
        "{0}: {1}".format(Path(path).name, detail),
        "Fix the backend manifest under resources/backends.",
    )


def _validate_manifest(value, path):
    if not isinstance(value, dict):
        raise _invalid(path, "manifest must be an object")
    required = {
        "schema": str, "id": str, "version": str, "builtin": bool, "package": str,
        "convert": str, "categories": dict, "remap_files": dict, "registry": dict,
    }
    for key, kind in required.items():
        if not isinstance(value.get(key), kind):
            raise _invalid(path, "field {0} must be {1}".format(key, kind.__name__))
    if value["schema"] != MANIFEST_SCHEMA or not _ID.fullmatch(value["id"]) or Path(path).stem != value["id"]:
        raise _invalid(path, "schema or id mismatch")
    if not set(value["categories"]) <= set(CATEGORIES) or not value["categories"]:
        raise _invalid(path, "categories must be a non-empty subset of {0}".format(list(CATEGORIES)))
    if not set(value["remap_files"]) <= set(value["categories"]):
        raise _invalid(path, "remap_files keys must be declared categories")
    if not value["builtin"] and not isinstance(value.get("lock"), str):
        raise _invalid(path, "an optional backend needs a lock file")


def backends_root(env=None):
    env = os.environ if env is None else env
    base = env.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / "mlx-workbench" / "backends"


def backend_target(manifest, root=None):
    return Path(root if root is not None else backends_root()) / manifest["id"]


def backend_python(manifest, root=None):
    if manifest["builtin"]:
        return Path(sys.executable)
    return backend_target(manifest, root) / "bin" / "python"


def read_install_marker(manifest, root=None):
    location = backend_target(manifest, root) / INSTALL_MARKER
    try:
        value = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("id") != manifest["id"] or value.get("version") != manifest["version"]:
        return None
    return value


def is_installed(manifest, root=None, find_spec=importlib.util.find_spec):
    if manifest["builtin"]:
        try:
            return find_spec(manifest["package"]) is not None
        except (ImportError, ValueError):
            return False
    return read_install_marker(manifest, root) is not None and backend_python(manifest, root).exists()


def package_dir(manifest, root=None, find_spec=importlib.util.find_spec):
    if manifest["builtin"]:
        try:
            spec = find_spec(manifest["package"])
        except (ImportError, ValueError):
            return None
        if spec is None or not spec.submodule_search_locations:
            return None
        return Path(list(spec.submodule_search_locations)[0])
    lib = backend_target(manifest, root) / "lib"
    for candidate in sorted(lib.glob("python3.*/site-packages/{0}".format(manifest["package"]))):
        if candidate.is_dir():
            return candidate
    return None


def _module_name(relative, prefix):
    if not relative.startswith(prefix):
        return None
    parts = relative[len(prefix):].split("/")
    if len(parts) == 2 and parts[1] == "__init__.py":
        name = parts[0]
    elif len(parts) == 1 and parts[0].endswith(".py"):
        name = parts[0][:-3]
    else:
        return None
    if not name or name.startswith("_"):
        return None
    return name


def _vision_marker(relative, prefix):
    """The module a `<prefix><module>/<file>.py` vision/image file belongs to."""
    if not relative.startswith(prefix):
        return None
    parts = relative[len(prefix):].split("/")
    if len(parts) != 2 or not parts[1].endswith(".py") or parts[1] == "__init__.py":
        return None
    return parts[0] if _VISION_FILE.search(parts[1][:-3]) else None


def _category_prefixes(manifest):
    return [directory.strip("/") + "/" for directory in manifest["categories"].values()]


def _remap_files(manifest):
    return {relative for files in manifest["remap_files"].values() for relative in files}


def probe_sources_from_wheel(wheel_path, manifest):
    """Read only the files the registry probe needs from a wheel."""
    package_prefix = manifest["package"] + "/"
    prefixes = _category_prefixes(manifest)
    remap = _remap_files(manifest)
    sources = {}
    with zipfile.ZipFile(wheel_path) as archive:
        for info in archive.infolist():
            if not info.filename.startswith(package_prefix) or info.file_size > MAX_PROBE_FILE_BYTES:
                continue
            relative = info.filename[len(package_prefix):]
            if relative in remap or any(_module_name(relative, prefix) for prefix in prefixes):
                sources[relative] = archive.read(info).decode("utf-8", errors="replace")
            elif any(_vision_marker(relative, prefix) for prefix in prefixes):
                sources[relative] = ""
            if len(sources) >= MAX_PROBE_FILES:
                break
    return sources


def probe_sources_from_directory(directory, manifest):
    """Read the same files from an installed package directory."""
    directory = Path(directory)
    sources = {}
    for relative in sorted(_remap_files(manifest)):
        _add_source(sources, directory, relative)
    for prefix in _category_prefixes(manifest):
        base = directory / prefix
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if len(sources) >= MAX_PROBE_FILES:
                break
            if entry.is_dir():
                _add_source(sources, directory, "{0}{1}/__init__.py".format(prefix, entry.name))
                for child in sorted(entry.iterdir()):
                    marker = "{0}{1}/{2}".format(prefix, entry.name, child.name)
                    if child.is_file() and _vision_marker(marker, prefix):
                        sources[marker] = ""
            elif entry.suffix == ".py":
                _add_source(sources, directory, "{0}{1}".format(prefix, entry.name))
    return {key: sources[key] for key in sorted(sources) if any(
        _module_name(key, prefix) or _vision_marker(key, prefix)
        for prefix in _category_prefixes(manifest)
    ) or key in _remap_files(manifest)}


def _add_source(sources, directory, relative):
    location = directory / relative
    try:
        if not location.is_file() or location.stat().st_size > MAX_PROBE_FILE_BYTES:
            return
        sources[relative] = location.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return


def _binds_model(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return True
        if isinstance(node, ast.ImportFrom) and any(
            (alias.asname or alias.name) == "Model" for alias in node.names
        ):
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "Model" for target in node.targets
        ):
            return True
    return False


def _remapping(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "MODEL_REMAPPING" for target in targets):
            continue
        try:
            literal = ast.literal_eval(value)
        except ValueError:
            return {}
        if isinstance(literal, dict):
            return {
                normalize_type(key): normalize_type(item)
                for key, item in literal.items()
                if isinstance(key, str) and isinstance(item, str)
            }
    return {}


def registry_from_sources(sources, manifest):
    """Model types and MODEL_REMAPPING per category from probe sources."""
    registry = {}
    for category, directory in sorted(manifest["categories"].items()):
        prefix = directory.strip("/") + "/"
        names = set()
        for relative, text in sources.items():
            module = _module_name(relative, prefix)
            if module and _binds_model(text):
                names.add(normalize_type(module))
        remapping = {}
        for relative in manifest["remap_files"].get(category, ()):
            remapping.update(_remapping(sources.get(relative, "")))
        registry[category] = {"model_types": sorted(names), "remapping": dict(sorted(remapping.items()))}
        if category == "vision_language":
            vision = {normalize_type(_vision_marker(relative, prefix)) for relative in sources
                      if _vision_marker(relative, prefix)}
            registry[category]["vision_types"] = sorted(vision & names)
    return registry


def load_registries(manifests, root=None, find_spec=importlib.util.find_spec):
    """Installed registries win over bundled snapshots when they are non-empty."""
    registries = {}
    for backend_id, manifest in manifests.items():
        installed = is_installed(manifest, root, find_spec)
        registry, source = manifest["registry"], "snapshot"
        if installed:
            directory = package_dir(manifest, root, find_spec)
            if directory is not None:
                live = registry_from_sources(probe_sources_from_directory(directory, manifest), manifest)
                if any(entry["model_types"] for entry in live.values()):
                    registry, source = live, "installed"
        registries[backend_id] = {"registry": registry, "source": source, "installed": installed}
    return registries


def _module_path(manifest, category, module):
    return "{0}.{1}.{2}".format(
        manifest["package"], manifest["categories"][category].strip("/").replace("/", "."), module
    )


def _hit(backend_id, manifest, category, module, match):
    return {
        "backend": backend_id,
        "category": category,
        "match": match,
        "module": _module_path(manifest, category, module),
    }


def _registry(backend_id, manifest, registries):
    return registries.get(backend_id, {}).get("registry", manifest["registry"])


def lookup(model_type, manifests, registries):
    if not model_type:
        return []
    wanted = normalize_type(model_type)
    hits = []
    for backend_id in sorted(manifests):
        manifest = manifests[backend_id]
        registry = _registry(backend_id, manifest, registries)
        for category in sorted(registry):
            entry = registry[category]
            types = set(entry.get("model_types", ()))
            remapping = entry.get("remapping", {})
            if wanted in types:
                hits.append(_hit(backend_id, manifest, category, wanted, "exact"))
            elif remapping.get(wanted) in types:
                hits.append(_hit(backend_id, manifest, category, remapping[wanted], "remap"))
    return hits


def lookup_squashed(architecture, manifests, registries):
    wanted = squash_type(architecture or "")
    if not wanted:
        return []
    hits = []
    for backend_id in sorted(manifests):
        manifest = manifests[backend_id]
        registry = _registry(backend_id, manifest, registries)
        for category in sorted(registry):
            for module in registry[category].get("model_types", ()):
                if squash_type(module) == wanted:
                    hits.append(_hit(backend_id, manifest, category, module, "squashed"))
                    break
    return hits


def is_vision_type(model_type, manifests, registries):
    """True when a vision-language backend's module for this type ships vision files."""
    if not model_type:
        return False
    wanted = normalize_type(model_type)
    for backend_id in sorted(manifests):
        entry = _registry(backend_id, manifests[backend_id], registries).get("vision_language")
        if not entry:
            continue
        module = entry.get("remapping", {}).get(wanted, wanted)
        if module in entry.get("vision_types", ()):
            return True
    return False


def component_matches(model_type, manifests, registries):
    hits = lookup(model_type, manifests, registries)
    if hits or not model_type:
        return hits
    normalized = normalize_type(model_type)
    for suffix in _STRIP_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            stripped = lookup(normalized[: -len(suffix)], manifests, registries)
            return [dict(hit, match="stripped") for hit in stripped]
    return []


def choose_backend(hits, task_type, has_vision_config=False):
    preference = TYPE_PREFERENCE.get(task_type, DEFAULT_PREFERENCE)
    if has_vision_config and task_type == "text_llm":
        preference = ("mlx-vlm", "mlx-lm")
    ranked = sorted(
        (hit for hit in hits if hit["backend"] in preference),
        key=lambda hit: (hit["match"] == "remap", preference.index(hit["backend"])),
    )
    return ranked[0]["backend"] if ranked else None


def backend_environment(env=None):
    env = os.environ if env is None else env
    return {key: env[key] for key in _ENV_ALLOWLIST if key in env}


def spawn_with_env(argv, log_path, env):
    handle = open(log_path, "ab", buffering=0)
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    finally:
        handle.close()
    return process.pid
