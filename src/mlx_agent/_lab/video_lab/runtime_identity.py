from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Iterable, Mapping
from importlib.metadata import distributions
from pathlib import Path

_NORMALIZE_DISTRIBUTION_NAME = re.compile(r"[-_.]+")
_VALID_NORMALIZED_DISTRIBUTION_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class RuntimeIdentityError(ValueError):
    pass


def source_tree_digest(package_root: Path | None = None) -> str:
    root = Path(__file__).resolve().parent if package_root is None else Path(package_root).resolve()
    digest = hashlib.sha256(b"video-lab-source-tree.v1\0")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and path.name != ".DS_Store"
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"source-sha256:{digest.hexdigest()}"


def animegen_runtime_versions(
    *,
    package_root: Path | None = None,
    distribution_entries: Iterable[tuple[str, str]] | None = None,
    python_version: str | None = None,
) -> dict[str, str]:
    entries = (
        _installed_distribution_entries() if distribution_entries is None else distribution_entries
    )
    return {
        "python": platform.python_version() if python_version is None else python_version,
        **_normalized_distribution_versions(entries),
        "video-lab": source_tree_digest(package_root),
    }


def _installed_distribution_entries() -> Iterable[tuple[str, str]]:
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeIdentityError("installed distribution metadata is incomplete")
        yield name, version


def _normalized_distribution_versions(
    entries: Iterable[tuple[str, str]],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_name, raw_version in entries:
        name = _NORMALIZE_DISTRIBUTION_NAME.sub("-", raw_name.strip()).lower()
        version = raw_version.strip()
        if not _VALID_NORMALIZED_DISTRIBUTION_NAME.fullmatch(name) or not version:
            raise RuntimeIdentityError("installed distribution metadata is invalid")
        existing = normalized.get(name)
        if existing is not None and existing != version:
            raise RuntimeIdentityError(
                f"conflicting installed versions for normalized distribution {name}"
            )
        normalized[name] = version
    return {f"dist:{name}": normalized[name] for name in sorted(normalized)}


def runtime_fingerprint(runtime_versions: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(runtime_versions), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()
