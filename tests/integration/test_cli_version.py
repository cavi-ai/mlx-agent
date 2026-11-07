from pathlib import Path
import subprocess
import sys
import unittest

from mlx_agent import __version__


class CLIVersionTests(unittest.TestCase):
    def test_launcher_reports_the_package_version(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, str(root / "scripts/mlx-agent"), "--version"],
                                cwd=root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "mlx-agent " + __version__)
        self.assertEqual(result.stderr, "")
