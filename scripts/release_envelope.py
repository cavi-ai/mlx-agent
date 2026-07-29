#!/usr/bin/env python3
"""Render the two documents a release hands to consumers.

``manifest`` writes the ``cavi-release.json`` that travels inside the artifact.
``dispatch`` writes the ``repository_dispatch`` body that points at it. Both
describe the same release, so they are generated from one place: a consumer
rejects the artifact when the two disagree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SLUG = "mlx-agent"
KIND = "product-docs"
REPOSITORY = "cavi-ai/mlx-agent"
EVENT_TYPE = "cavi-oss-release"
SCHEMA_VERSION = 1
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class EnvelopeError(ValueError):
    """The release identity is not one a consumer will accept."""


def _validate(version, tag, commit):
    if not _SEMVER.fullmatch(version):
        raise EnvelopeError("version must be a SemVer core version: {0}".format(version))
    if tag != "v{0}".format(version):
        raise EnvelopeError("tag {0} does not match version {1}".format(tag, version))
    if not _COMMIT.fullmatch(commit):
        raise EnvelopeError("commit must be a full lowercase SHA: {0}".format(commit))


def release_manifest(version, tag, commit):
    """The identity document carried inside the artifact."""
    _validate(version, tag, commit)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "slug": SLUG,
        "kind": KIND,
        "version": version,
        "tag": tag,
        "repository": REPOSITORY,
        "commit": commit,
    }


def dispatch_payload(version, tag, commit, url, sha256):
    """The repository_dispatch body that points a consumer at the artifact."""
    _validate(version, tag, commit)
    if not _SHA256.fullmatch(sha256):
        raise EnvelopeError("artifact sha256 must be lowercase hexadecimal")
    if not url.startswith("https://") or "?" in url or "#" in url or "@" in url:
        raise EnvelopeError("artifact URL must be a plain HTTPS location: {0}".format(url))
    envelope = release_manifest(version, tag, commit)
    envelope["artifact"] = {"url": url, "sha256": sha256, "format": "tar.gz"}
    return {"event_type": EVENT_TYPE, "client_payload": envelope}


def _write(value, out):
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if out in (None, "-"):
        sys.stdout.write(text)
    else:
        Path(out).write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    manifest = actions.add_parser("manifest", help="write cavi-release.json")
    dispatch = actions.add_parser("dispatch", help="write the repository_dispatch body")
    for action in (manifest, dispatch):
        action.add_argument("--version", required=True)
        action.add_argument("--tag", required=True)
        action.add_argument("--commit", required=True)
        action.add_argument("--out", default="-")
    dispatch.add_argument("--url", required=True)
    dispatch.add_argument("--sha256", required=True)

    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "manifest":
            value = release_manifest(arguments.version, arguments.tag, arguments.commit)
        else:
            value = dispatch_payload(
                arguments.version, arguments.tag, arguments.commit,
                arguments.url, arguments.sha256,
            )
    except EnvelopeError as error:
        print("release envelope: {0}".format(error), file=sys.stderr)
        return 1
    _write(value, arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
