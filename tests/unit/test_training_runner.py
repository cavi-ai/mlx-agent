import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainingRunnerBootstrapTests(unittest.TestCase):
    def test_checkout_runner_records_exit_without_pythonpath(self):
        runner = Path(__file__).resolve().parents[2] / "src/mlx_agent/training_runner.py"
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run([sys.executable, str(runner), "--result", str(output), "--",
                sys.executable, "-c", "raise SystemExit(7)"], env=environment,
                cwd=directory, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(json.loads(output.read_text())["exit_code"], 7)

    def test_timeout_records_exit_124_and_error(self):
        runner = Path(__file__).resolve().parents[2] / "src/mlx_agent/training_runner.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run(
                [sys.executable, str(runner), "--result", str(output), "--timeout", "0.2", "--",
                 sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=directory, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 124, result.stderr)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["exit_code"], 124)
            self.assertIn("timed out", payload["error"])
