"""Fail a pull request that adds or moves a pinned dependency or changes how consumers are pinned.

Consumers pin mlx-agent main through the pin-consumers job. A pull request never adds or moves a
submodule or a git dependency in pyproject.toml, uv.lock, package.json or a requirements file.

Usage: pin_guard.py BASE HEAD, in a checkout that holds both commits.
"""

import json
import re
import subprocess
import sys
import tomllib

GUARD = "scripts/pin_guard.py"
MECHANISM = (GUARD,)
JOBS = {".github/workflows/test.yml": ("pin-guard", "pin-consumers")}
GIT_PACKAGE = re.compile(r"^(git\+|git@|git:|github:|ssh://|https?://)|\.git(#|$)")
GIT_REQUIREMENT = re.compile(r"git\+|\.git[@#]")
REQUIREMENTS = re.compile(r"(^|/)requirements[^/]*\.txt$")


def git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else None


def show(rev, path):
    return git("show", f"{rev}:{path}")


def job(text, name):
    """The lines of one top-level workflow job, or None when it is absent."""
    lines = (text or "").splitlines()
    if f"  {name}:" not in lines:
        return None
    start = lines.index(f"  {name}:")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^ {0,2}\S", lines[i])),
               len(lines))
    return lines[start:end]


def git_pins(rev):
    """Git-sourced dependencies declared or resolved at one commit."""
    project = tomllib.loads(show(rev, "pyproject.toml") or "")
    lock = tomllib.loads(show(rev, "uv.lock") or "")
    npm = json.loads(show(rev, "package.json") or "{}")
    pins = {
        "pyproject.toml": project.get("tool", {}).get("uv", {}).get("sources", {}),
        "uv.lock": {p["name"]: p["source"]["git"] for p in lock.get("package", [])
                    if "git" in p.get("source", {})},
        "package.json": {name: value for group in ("dependencies", "devDependencies")
                         for name, value in npm.get(group, {}).items()
                         if GIT_PACKAGE.search(value)},
    }
    for path in (git("ls-tree", "-r", "--name-only", rev) or "").splitlines():
        if REQUIREMENTS.search(path):
            pins[path] = sorted(line.strip() for line in (show(rev, path) or "").splitlines()
                                if GIT_REQUIREMENT.search(line))
    return pins


def violations(base, head):
    found = []
    for path in MECHANISM:
        before = show(base, path)
        if before is not None and before != show(head, path):
            found.append((path, "changes how pins move"))
    for path, names in JOBS.items():
        for name in names:
            before = job(show(base, path), name)
            if before is not None and before != job(show(head, path), name):
                found.append((path, f"changes the {name} job"))
    if show(base, ".gitmodules") != show(head, ".gitmodules"):
        found.append((".gitmodules", "adds or moves a submodule"))
    for line in (git("diff", "--raw", "--no-renames", base, head) or "").splitlines():
        meta, path = line.split("\t", 1)
        if "160000" in meta.split()[:2]:
            found.append((path, "adds or moves a submodule"))
    before, after = git_pins(base), git_pins(head)
    found += [(path, "adds or moves a git dependency pin") for path in sorted({*before, *after})
              if before.get(path, []) != after.get(path, [])]
    return found


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    base = (git("merge-base", argv[0], argv[1]) or "").strip()
    if not base:
        print(f"::error::No merge base for {argv[0]} and {argv[1]}")
        return 1
    found = violations(base, argv[1])
    for path, message in found:
        print(f"::error file={path}::{path} {message}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
