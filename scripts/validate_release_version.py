#!/usr/bin/env python3
"""Assert every committed copy of the plugin version agrees.

The version is embedded in the runtime, the plugin manifest, both marketplace
manifests, the compatibility matrix, the release evidence, the site catalog,
the contract validator, and the documentation artifact. A release that ships
with any one of them stale produces a package that lies about itself, so this
runs as its own gate rather than as a side effect of another check.

With ``--tag vX.Y.Z`` the tag under release must match as well.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _runtime_version():
    text = (ROOT / "src" / "mlx_agent" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def _validator_version():
    text = (ROOT / "scripts" / "validate_contracts.py").read_text(encoding="utf-8")
    match = re.search(r'^PLUGIN_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def _marketplace_plugin_version():
    plugins = _json(".claude-plugin/marketplace.json").get("plugins")
    if not isinstance(plugins, list) or not plugins:
        return None
    return plugins[0].get("version")


def _docs_artifact_version(version):
    """The artifact directory is version-named, so read the one being shipped."""
    manifest = ROOT / "docs" / "mlx-agent" / "v{0}".format(version) / "manifest.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("version")


def collect(version=None):
    """Return every declared version keyed by the file that declares it."""
    declared = {
        "src/mlx_agent/__init__.py": _runtime_version(),
        "plugin.json": _json("plugin.json").get("version"),
        ".claude-plugin/plugin.json": _json(".claude-plugin/plugin.json").get("version"),
        ".claude-plugin/marketplace.json": _marketplace_plugin_version(),
        "compatibility/providers.json": _json("compatibility/providers.json").get("plugin_version"),
        "compatibility/release-evidence.json": _json("compatibility/release-evidence.json").get("plugin_version"),
        "site/catalog.json": (_json("site/catalog.json").get("release") or {}).get("version"),
        "scripts/validate_contracts.py": _validator_version(),
    }
    reference = version or declared["src/mlx_agent/__init__.py"]
    if reference:
        artifact = _docs_artifact_version(reference)
        if artifact is not None:
            declared["docs/mlx-agent/v{0}/manifest.json".format(reference)] = artifact
    return declared


def validate(tag=None):
    declared = collect()
    errors = []
    missing = sorted(name for name, value in declared.items() if not value)
    if missing:
        errors.append("no version found in: {0}".format(", ".join(missing)))
    present = {name: value for name, value in declared.items() if value}
    distinct = sorted(set(present.values()))
    if len(distinct) > 1:
        for name, value in sorted(present.items()):
            errors.append("{0} declares {1}".format(name, value))
        errors.append("every declared version must agree; found {0}".format(distinct))
    version = distinct[0] if len(distinct) == 1 else None
    if version and not SEMVER.fullmatch(version):
        errors.append("{0} is not a SemVer core version".format(version))
    if tag is not None:
        expected = tag[1:] if tag.startswith("v") else tag
        if version and expected != version:
            errors.append("tag {0} does not match the declared version {1}".format(tag, version))
    if version:
        artifact = ROOT / "docs" / "mlx-agent" / "v{0}".format(version)
        if not artifact.is_dir():
            errors.append("documentation artifact {0} has not been built".format(artifact.relative_to(ROOT)))
    return version, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="release tag that must match, for example v0.5.0")
    parser.add_argument("--print", action="store_true", dest="print_version")
    arguments = parser.parse_args(argv)
    version, errors = validate(arguments.tag)
    if errors:
        for error in errors:
            print("version drift: {0}".format(error), file=sys.stderr)
        return 1
    if arguments.print_version:
        print(version)
    else:
        print("every declared version is {0}".format(version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
