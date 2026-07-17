import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"
LEGACY = ROOT / "skills" / "mlx-scout" / "scripts" / "scout.py"


class ScoutCompatibilityTests(unittest.TestCase):
    def _environment(self, fixture=FIXTURE):
        return {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "MLX_AGENT_FIXTURE": str(fixture),
        }

    def _run_legacy(self, *arguments, fixture=FIXTURE, check=True):
        return subprocess.run(
            [sys.executable, str(LEGACY), *arguments],
            env=self._environment(fixture),
            text=True,
            capture_output=True,
            check=check,
        )

    def _run_new(self, *arguments, fixture=FIXTURE, check=True):
        return subprocess.run(
            [sys.executable, "-m", "mlx_agent", "discover", *arguments],
            env=self._environment(fixture),
            text=True,
            capture_output=True,
            check=check,
        )

    def test_json_flag_preserves_role_buckets_and_relevant_fields_for_each_discovery_surface(self):
        expected_buckets = {
            "coding": ["lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-Q8"],
            "general": ["mlx-community/Qwen3-8B-Instruct-4bit"],
            "reasoning": ["community/Thinking-7B-4bit"],
        }
        cases = [
            (("--role", "coding", "--limit", "1"), {"coding": expected_buckets["coding"]}, False),
            (("--new", "--limit", "2"), expected_buckets, False),
            (("--fast", "--limit", "2"), expected_buckets, True),
        ]
        for arguments, expected_roles, expected_fast in cases:
            with self.subTest(arguments=arguments):
                old = self._run_legacy("--json", *arguments)
                new = self._run_new("--json", *arguments)
                legacy_report = json.loads(old.stdout)
                envelope = json.loads(new.stdout)

                self.assertEqual(envelope["data"], legacy_report)
                self.assertEqual(
                    {role: [item["repo"] for item in items] for role, items in legacy_report["roles"].items()},
                    expected_roles,
                )
                self.assertEqual(legacy_report["fast"], expected_fast)
                self.assertTrue(all("wiring" in item and "reason_src" in item for items in legacy_report["roles"].values() for item in items))
                self.assertEqual(envelope["warnings"], [{"code": "synthetic_fixture", "message": "Fixture-backed discovery; this is not live Hugging Face evidence."}])
                self.assertIn("synthetic fixture", old.stderr.lower())
                self.assertIn("synthetic fixture", new.stderr.lower())

    def test_human_readable_output_matches_and_retains_rendered_bucket(self):
        old = self._run_legacy("--role", "coding", "--limit", "1")
        new = self._run_new("--role", "coding", "--limit", "1")

        self.assertEqual(new.stdout, old.stdout)
        self.assertIn("# mlx-scout report", old.stdout)
        self.assertIn("## Coding", old.stdout)
        self.assertIn("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-Q8", old.stdout)
        self.assertNotIn("## General", old.stdout)

    def test_wire_output_matches_for_every_target_and_port(self):
        repo = "mlx-community/Qwen3-8B-Instruct-4bit"
        expected_fragments = {
            "ollama": "ollama pull hf.co/",
            "lmstudio": "lms get mlx-community/Qwen3-8B-Instruct-4bit",
            "mlx_lm": "--port 9001",
            "mlx-vlm": "--port 9001",
            "litellm": "http://127.0.0.1:9001/v1",
        }
        for target, expected_fragment in expected_fragments.items():
            with self.subTest(target=target):
                old = self._run_legacy("--wire", repo, "--target", target, "--port", "9001")
                new = self._run_new("--wire", repo, "--target", target, "--port", "9001")
                self.assertEqual(new.stdout, old.stdout)
                self.assertIn(expected_fragment, old.stdout)

    def test_missing_or_malformed_fixture_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            malformed = Path(directory) / "malformed.json"
            malformed.write_text(json.dumps({"host": "not-an-object"}))
            for fixture in (missing, malformed):
                with self.subTest(fixture=fixture.name):
                    completed = self._run_new("--json", fixture=fixture, check=False)
                    self.assertEqual(completed.returncode, 2)
                    value = json.loads(completed.stdout)
                    self.assertEqual(value["status"], "error")
                    self.assertEqual(value["error"]["code"], "invalid_fixture")
                    self.assertIn("unset MLX_AGENT_FIXTURE", value["error"]["remediation"])

    def test_malformed_model_entries_return_structured_fixture_errors(self):
        payload = json.loads(FIXTURE.read_text())
        cases = {
            "non-object": ["not-a-model-object"],
            "non-string-id": [{"id": 7}],
            "non-string-model-id": [{"modelId": 7}],
            "missing-id-and-model-id": [{"downloads": 1, "likes": 0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, models in cases.items():
                with self.subTest(name=name):
                    fixture = Path(directory) / (name + ".json")
                    malformed = dict(payload)
                    malformed["models"] = models
                    fixture.write_text(json.dumps(malformed))
                    completed = self._run_new("--json", fixture=fixture, check=False)
                    self.assertEqual(completed.returncode, 2)
                    value = json.loads(completed.stdout)
                    self.assertEqual(value["status"], "error")
                    self.assertEqual(value["error"]["code"], "invalid_fixture")
                    self.assertIn("fixture.models", value["error"]["message"])
