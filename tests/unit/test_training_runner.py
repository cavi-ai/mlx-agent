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
