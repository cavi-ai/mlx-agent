#!/usr/bin/env python3
"""Build the immutable mlx-agent documentation artifact (bobby-browser pattern).

Copies docs/mlx-agent/source into docs/mlx-agent/v<version>/ and writes a
manifest.json whose contentSha256 matches the consumer's verifier exactly:
sha256 over every file except manifest.json, lexical order, each entry as
``path \\0 bytes \\0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "mlx-agent" / "source"
sys.path.insert(0, str(ROOT / "src"))
from mlx_agent import __version__  # noqa: E402

PACKAGE = "mlx-agent"


def _artifact_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def _content_sha256(root, files):
    digest = hashlib.sha256()
    for relative in files:
        if relative == "manifest.json":
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build(version, check=False):
    if not SOURCE.is_dir():
        raise ValueError("documentation source is missing: {0}".format(SOURCE))
    destination = ROOT / "docs" / "mlx-agent" / "v{0}".format(version)
    navigation = json.loads((SOURCE / "navigation.json").read_text(encoding="utf-8"))
    if navigation.get("version") != version:
        raise ValueError(
            "navigation version {0} does not match package version {1}".format(
                navigation.get("version"), version
            )
        )
    if check:
        if not destination.is_dir():
            raise ValueError("documentation artifact is missing: {0}".format(destination))
        built_files = _artifact_files(destination)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        source_files = ["navigation.json"] + [
            page.relative_to(SOURCE / "pages").as_posix()
            for page in sorted((SOURCE / "pages").rglob("*"))
            if page.is_file()
        ]
        if sorted(item for item in built_files if item != "manifest.json") != sorted(source_files):
            raise ValueError("documentation artifact files differ from the source")
        expected = _content_sha256(destination, built_files)
        if manifest.get("contentSha256") != expected:
            raise ValueError("documentation content digest mismatch")
        return destination

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(SOURCE / "navigation.json", destination / "navigation.json")
    for page in (SOURCE / "pages").rglob("*"):
        if page.is_file():
            target = destination / page.relative_to(SOURCE / "pages")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(page, target)
    files = _artifact_files(destination)
    manifest = {
        "package": PACKAGE,
        "product": PACKAGE,
        "version": version,
        "contentSha256": _content_sha256(destination, files),
        "publicBasePath": "/docs/mlx-agent/v{0}".format(version),
        "stableAlias": "/docs/mlx-agent",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed artifact instead of rebuilding")
    arguments = parser.parse_args(argv)
    try:
        destination = build(__version__, check=arguments.check)
    except ValueError as error:
        print("docs artifact: {0}".format(error), file=sys.stderr)
        return 2
    print(("verified" if arguments.check else "built") + " {0}".format(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
