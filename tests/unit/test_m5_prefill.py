"""Keep MLX's process-wide resource limits out of portable core tests."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

from mlx_agent.prefill_runtime import VERSIONS, identity


class M5PrefillProcessTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin" and identity()["versions"] == VERSIONS,
                         "Requires the supported MLX runtime")
    def test_numerical_checks_run_in_an_owned_process(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(root / "tests/gpu_m5_prefill.py"), "-v"],
            cwd=str(root), env=dict(os.environ, PYTHONPATH=str(root / "src")),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 77:
            self.skipTest("Requires the supported M5/MLX runtime")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
