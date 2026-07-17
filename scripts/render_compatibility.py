#!/usr/bin/env python3
"""Render the marked README compatibility block from the committed matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- compatibility:begin -->"
END = "<!-- compatibility:end -->"
EVIDENCE_FIELDS = ("schema", "install_round_trip", "native_discovery", "bundle_execution", "model_backed_invocation")


def _status(status):
    if status in {"blocked", "not-run"}:
        return "{0} — not supported".format(status)
    return status


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _load_matrix(path):
    try:
        matrix = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("could not read compatibility matrix: {0}".format(error))
    if not isinstance(matrix, dict) or not isinstance(matrix.get("providers"), dict):
        raise ValueError("compatibility matrix must contain providers")
    statuses = matrix.get("allowed_evidence_statuses")
    if not isinstance(statuses, list) or not all(isinstance(item, str) for item in statuses):
        raise ValueError("compatibility matrix must declare allowed evidence statuses")
    for provider_id, entry in matrix["providers"].items():
        if not isinstance(entry, dict) or entry.get("id") != provider_id:
            raise ValueError("provider {0} has an invalid identity".format(provider_id))
        if not isinstance(entry.get("scopes"), list) or not entry["scopes"]:
            raise ValueError("provider {0} has invalid scopes".format(provider_id))
        if not isinstance(entry.get("config_paths"), list) or not entry["config_paths"]:
            raise ValueError("provider {0} has invalid config paths".format(provider_id))
        capabilities = entry.get("capabilities")
        if not isinstance(capabilities, dict) or set(capabilities) != {"scout", "adopt", "wire"}:
            raise ValueError("provider {0} has invalid capabilities".format(provider_id))
        smoke = entry.get("last_smoke_test")
        if not isinstance(smoke, dict) or set(smoke) != {"status", "date", "summary"} or smoke["status"] not in statuses:
            raise ValueError("provider {0} has invalid smoke evidence".format(provider_id))
        try:
            date.fromisoformat(smoke["date"])
        except (TypeError, ValueError):
            raise ValueError("provider {0} has a non-ISO smoke date".format(provider_id))
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_FIELDS):
            raise ValueError("provider {0} has invalid evidence fields".format(provider_id))
        for field in EVIDENCE_FIELDS:
            value = evidence[field]
            if not isinstance(value, dict) or value.get("status") not in statuses:
                raise ValueError("provider {0} has invalid {1} evidence".format(provider_id, field))
    return matrix


def render(matrix):
    lines = [
        BEGIN,
        "## Compatibility evidence",
        "",
        "This block is generated from [`compatibility/providers.json`](compatibility/providers.json). `not-run` and `blocked` mean **not supported by current evidence**.",
        "",
        "| Provider | Scopes | Config paths | Native capability invocation | Latest smoke | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in matrix["providers"].values():
        capabilities = "<br>".join(
            "{0}: `{1}`".format(name, entry["capabilities"][name]["invocation"])
            for name in ("scout", "adopt", "wire")
        )
        smoke = entry["last_smoke_test"]
        latest = "{0} ({1}): {2}".format(_status(smoke["status"]), smoke["date"], smoke["summary"])
        evidence = "<br>".join(
            "{0}: {1}".format(field.replace("_", " "), _status(entry["evidence"][field]["status"]))
            for field in EVIDENCE_FIELDS
        )
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} |".format(
            _cell(entry["display_name"]),
            _cell(", ".join(entry["scopes"])),
            _cell("<br>".join(entry["config_paths"])),
            _cell(capabilities),
            _cell(latest),
            _cell(evidence),
        ))
    lines.extend([END, ""])
    return "\n".join(lines)


def replace_block(readme, block):
    start = readme.find(BEGIN)
    end = readme.find(END)
    if start < 0 or end < 0 or end < start:
        raise ValueError("README compatibility block markers are missing or malformed")
    end += len(END)
    return readme[:start] + block.rstrip("\n") + readme[end:]


def _atomic_write(path, content):
    destination = Path(path)
    mode = destination.stat().st_mode & 0o777
    descriptor, temporary = tempfile.mkstemp(prefix=".mlx-agent-compatibility-", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="render mlx-agent compatibility evidence into README")
    parser.add_argument("--matrix", default=str(ROOT / "compatibility" / "providers.json"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="fail when the README block differs from the matrix")
    action.add_argument("--write", action="store_true", help="explicitly update the marked README block atomically")
    arguments = parser.parse_args(argv)
    try:
        matrix = _load_matrix(arguments.matrix)
        path = Path(arguments.readme)
        expected = replace_block(path.read_text(encoding="utf-8"), render(matrix))
        current = path.read_text(encoding="utf-8")
        if current == expected:
            return 0
        if arguments.write:
            _atomic_write(path, expected)
            return 0
        print("README compatibility block is out of date; run scripts/render_compatibility.py --write", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print("compatibility block check failed: {0}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
