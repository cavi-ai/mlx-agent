"""Create one isolated backend venv from a hash-locked requirements file."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MARKER = ".mlx-agent-backend.json"


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _discard(target, trash, backend_id):
    if not target.exists():
        return
    trash.mkdir(parents=True, exist_ok=True)
    destination = trash / "{0}-failed-{1}".format(backend_id, _stamp())
    shutil.move(str(target), str(destination))
    print("moved partial install to {0}".format(destination), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--id", "--version", "--package", "--lock", "--target", "--python", "--trash", "--preview-hash"):
        parser.add_argument(name, required=True)
    arguments = parser.parse_args(argv)
    target = Path(arguments.target)
    if target.exists() or target.is_symlink():
        print("target already exists: {0}".format(target), flush=True)
        return 3
    venv_python = str(target / "bin" / "python")
    steps = [
        [arguments.python, "-m", "venv", str(target)],
        [venv_python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
         "--require-hashes", "--no-deps", "--only-binary=:all:", "-r", arguments.lock],
        [venv_python, "-c", "import {0}".format(arguments.package)],
    ]
    for step in steps:
        print("$ " + " ".join(step), flush=True)
        completed = subprocess.run(step, stdin=subprocess.DEVNULL, check=False)
        if completed.returncode != 0:
            print("step failed with exit status {0}".format(completed.returncode), flush=True)
            _discard(target, Path(arguments.trash), arguments.id)
            return completed.returncode
    marker = {
        "id": arguments.id, "version": arguments.version,
        "preview_hash": arguments.preview_hash,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    (target / MARKER).write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    print("installed {0} {1} at {2}".format(arguments.id, arguments.version, target), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
