"""Workflows run pinned actions with read-only tokens unless a job needs more."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)(.*)$")


def _jobs(text):
    """Return each job's lines, keyed by job id."""
    lines = text.splitlines()
    jobs, current = {}, None
    for line in lines[lines.index("jobs:") + 1:]:
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if match:
            current = jobs.setdefault(match.group(1), [])
        elif current is not None:
            current.append(line)
    return jobs


def _grants(block):
    """Return the job-level permission grants in one job's lines."""
    grants, inside = {}, False
    for line in block:
        if line == "    permissions:":
            inside = True
        elif inside and re.match(r"^      [a-z-]+: (?:read|write|none)$", line):
            scope, level = line.strip().split(": ")
            grants[scope] = level
        else:
            inside = False
    return grants


class WorkflowSupplyChainTests(unittest.TestCase):
    def test_every_action_is_pinned_to_a_commit_with_its_release(self):
        self.assertTrue(WORKFLOWS)
        for path in WORKFLOWS:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                match = USES.match(line)
                if match is None or match.group(1).startswith("./"):
                    continue
                with self.subTest(workflow=path.name, line=number):
                    self.assertRegex(match.group(1), r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
                    self.assertRegex(match.group(2), r"^\s+# v\d+\.\d+\.\d+$")

    def test_default_token_is_read_only(self):
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                self.assertIn("\npermissions:\n  contents: read\n\njobs:\n", path.read_text(encoding="utf-8"))

    def test_only_publishing_jobs_receive_write_scopes(self):
        expected = {("release.yml", "release"): {"contents": "write"},
                    ("release.yml", "pypi"): {"id-token": "write"}}
        for path in WORKFLOWS:
            for job, block in _jobs(path.read_text(encoding="utf-8")).items():
                with self.subTest(workflow=path.name, job=job):
                    self.assertEqual(expected.get((path.name, job), {}), _grants(block))


if __name__ == "__main__":
    unittest.main()
