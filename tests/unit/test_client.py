import os
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_agent.client import ConvertError, convert, resolve_cli

SRC = str(Path(__file__).resolve().parents[2] / "src")
CLI = [sys.executable, "-m", "mlx_agent"]


class ConvertClientTests(unittest.TestCase):
    def setUp(self):
        previous = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = SRC
        self.addCleanup(
            lambda: os.environ.__setitem__("PYTHONPATH", previous)
            if previous is not None
            else os.environ.pop("PYTHONPATH", None)
        )

    def test_missing_model_raises_envelope_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ConvertError) as raised:
                convert(
                    "mesh",
                    ["--path", root + "/none", "--image", root + "/i.png", "--out", root + "/o.glb"],
                    5,
                    cli=CLI,
                )
        error = raised.exception
        self.assertEqual(error.code, "model_unreadable")
        self.assertEqual(error.message, "No readable converted model config.")
        self.assertEqual(error.remediation, "Pass --path to the converted model directory.")
        self.assertIn("model_unreadable", str(error))

    def test_non_json_output_includes_stderr(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "fake-cli"
            script.write_text("import sys\nsys.stderr.write('boom ' * 5000)\nsys.exit(3)\n")
            with self.assertRaises(ConvertError) as raised:
                convert("status", [], 5, cli=str(script))
        error = raised.exception
        self.assertEqual(error.code, "no_json")
        self.assertIn("boom", str(error))
        self.assertLess(len(str(error)), 5000)

    def test_timeout_raises(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "slow-cli"
            script.write_text("import time\ntime.sleep(30)\n")
            with self.assertRaises(ConvertError) as raised:
                convert("status", [], 0, cli=str(script), grace=1)
        self.assertEqual(raised.exception.code, "timeout")

    def test_unresolvable_cli(self):
        with self.assertRaises(ConvertError) as raised:
            resolve_cli("/nonexistent/mlx-agent")
        self.assertEqual(raised.exception.code, "cli_not_found")


if __name__ == "__main__":
    unittest.main()
