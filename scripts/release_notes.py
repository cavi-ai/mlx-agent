#!/usr/bin/env python3
"""Emit the CHANGELOG section for one version as release notes.

The changelog is the single record of what shipped, so a release body is an
extract of it rather than a second, drifting account.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
_HEADING = re.compile(r"^##\s+(.+?)\s*$")


def sections(text):
    """Split the changelog into ordered (heading, body) pairs."""
    found = []
    heading = None
    body = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is None:
            if heading is not None:
                body.append(line)
            continue
        if heading is not None:
            found.append((heading, "\n".join(body).strip()))
        heading, body = match.group(1), []
    if heading is not None:
        found.append((heading, "\n".join(body).strip()))
    return found


def notes_for(version, text):
    """Return the body released under ``version``, or None."""
    for heading, body in sections(text):
        if heading == version or heading.startswith("{0} ".format(version)):
            return body
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--changelog", default=str(CHANGELOG))
    arguments = parser.parse_args(argv)
    try:
        text = Path(arguments.changelog).read_text(encoding="utf-8")
    except OSError as error:
        print("changelog is unreadable: {0}".format(error), file=sys.stderr)
        return 1
    body = notes_for(arguments.version, text)
    if body is None:
        print(
            "no changelog section for {0}; add one before releasing".format(arguments.version),
            file=sys.stderr,
        )
        return 1
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
