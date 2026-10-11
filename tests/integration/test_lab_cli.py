"""The agent owns experiment entrypoints without eagerly importing inference."""
import json
import importlib.util
import shutil
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

@unittest.skipUnless(sys.version_info >= (3, 12) and importlib.util.find_spec("typer") is not None
    and importlib.util.find_spec("pydantic") is not None, "Optional Python 3.12 lab runtime is not installed.")
class LabIntegrationTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "scripts/mlx-agent"), *arguments],
                              capture_output=True, text=True, timeout=30)

    def test_existing_agent_help_does_not_import_optional_engine(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lab", result.stdout)

    def test_experiments_share_an_explicit_durable_store(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["lab", "--database", str(Path(directory) / "lab.sqlite3"),
                    "--artifacts", str(Path(directory) / "artifacts"),
                    "experiment", "list", "--json"]
            first = self.invoke(*args)
            self.assertEqual(first.returncode, 0, first.stderr)
            payload = json.loads(first.stdout)
            self.assertEqual(payload["status"], "ok", payload)
            self.assertEqual(payload["data"]["experiments"], [])
            second = self.invoke(*args)
            self.assertEqual(json.loads(second.stdout)["data"], payload["data"])

    def test_invalid_command_is_structured_and_does_not_create_state(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "lab.sqlite3"
            result = self.invoke("lab", "--database", str(database), "unknown", "--json")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertFalse(database.exists())

    def test_shell_syntax_is_rejected_by_lab_mcp_grammar(self):
        sys.path.insert(0, str(ROOT / "src"))
        from mlx_agent.command_args import parse_command_arguments, CommandArgumentError
        with self.assertRaises(CommandArgumentError):
            parse_command_arguments("lab", "experiment list ; whoami")
        self.assertEqual(parse_command_arguments("lab", "experiment list --json"),
                         ["lab", "experiment", "list", "--json"])
