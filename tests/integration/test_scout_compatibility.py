import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"
LEGACY = ROOT / "skills" / "mlx-scout" / "scripts" / "scout.py"


def repo_ids(report):
    return [
        item["repo"]
        for items in report["roles"].values()
        for item in items
    ]


class ScoutCompatibilityTests(unittest.TestCase):
    def test_legacy_script_and_new_cli_report_same_repositories(self):
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "MLX_AGENT_FIXTURE": str(FIXTURE),
        }
        old = subprocess.run(
            [sys.executable, str(LEGACY), "--json", "--limit", "2"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        new = subprocess.run(
            [sys.executable, "-m", "mlx_agent", "discover", "--json", "--limit", "2"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(repo_ids(json.loads(old.stdout)), repo_ids(json.loads(new.stdout)["data"]))

    def test_legacy_json_shape_is_preserved(self):
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "MLX_AGENT_FIXTURE": str(FIXTURE),
        }
        old = subprocess.run(
            [sys.executable, str(LEGACY), "--json", "--fast", "--limit", "2"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(set(json.loads(old.stdout)), {"host", "fast", "roles"})
