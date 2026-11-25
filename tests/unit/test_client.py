import os
import sys
import tempfile
import time
import unittest
from contextlib import chdir
from pathlib import Path

from mlx_agent.client import ConvertError, convert, resolve_cli

from .test_gguf import write_gguf

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

    def test_status_reads_empty_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            data = convert("status", ["--receipts-dir", root], 5, cli=CLI)
        self.assertEqual(data, {"jobs": []})

    def test_scan_reads_gguf_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            model = write_gguf(Path(root) / "sample.gguf")
            data = convert("scan", ["--gguf-root", root, "--receipts-dir", root], 5, cli=CLI)
        self.assertEqual([item["path"] for item in data["models"]], [str(model)])

    def test_start_returns_confirmation_preview(self):
        with tempfile.TemporaryDirectory() as root:
            model = write_gguf(Path(root) / "sample.gguf")
            output = Path(root) / "converted"
            data = convert("start", ["--gguf", model, "--out", output], 5, cli=CLI)
            self.assertFalse(output.exists())
        self.assertIs(data["requires_confirmation"], True)
        self.assertEqual(data["plan"]["source"]["path"], str(model))
        self.assertRegex(data["plan"]["preview_hash"], r"^[0-9a-f]{64}$")

    def test_start_propagates_envelope_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ConvertError) as raised:
                convert("start", ["--gguf", Path(root) / "absent.gguf"], 5, cli=CLI)
        self.assertEqual(raised.exception.code, "source_not_found")

    def test_non_json_output_includes_stderr(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "diagnostic-cli"
            script.write_text(
                "import sys\nsys.stderr.write('noise ' * 5000 + 'final error')\nsys.exit(3)\n"
            )
            with self.assertRaises(ConvertError) as raised:
                convert("status", [], 5, cli=str(script))
        error = raised.exception
        self.assertEqual(error.code, "no_json")
        self.assertTrue(error.message.endswith("final error"))
        self.assertLess(len(str(error)), 5000)

    def test_non_object_error_raises_convert_error(self):
        launcher = [
            sys.executable, "-c",
            'print(\'{"status":"error","error":["failed"]}\')\nraise SystemExit(2)',
        ]
        with self.assertRaises(ConvertError) as raised:
            convert("status", [], 5, cli=launcher)
        self.assertEqual(raised.exception.code, "unknown")
        self.assertEqual(raised.exception.message, "exit 2")

    def test_timeout_raises(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "slow-cli"
            script.write_text("import time\ntime.sleep(30)\n")
            with self.assertRaises(ConvertError) as raised:
                convert("status", [], 0, cli=str(script), grace=1)
        self.assertEqual(raised.exception.code, "timeout")

    def test_timeout_does_not_wait_for_inherited_pipes(self):
        launcher = ["/bin/sh", "-c", "sleep 4 & printf 'child ready\\n' >&2; wait"]
        started = time.monotonic()
        with self.assertRaises(ConvertError) as raised:
            convert("status", [], 0.5, cli=launcher, grace=0)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(raised.exception.code, "timeout")
        self.assertIn("child ready", raised.exception.message)

    def test_bare_executable_name_runs_the_file_in_current_directory(self):
        with tempfile.TemporaryDirectory() as root:
            launcher = Path(root) / "mlx-agent-client-test-launcher"
            launcher.write_text('#!/bin/sh\nprintf \'{"status":"ok","data":"local executable"}\\n\'\n')
            launcher.chmod(0o700)
            with chdir(root):
                data = convert("status", [], 5, cli=launcher.name)
        self.assertEqual(data, "local executable")

    def test_missing_argv_executable_raises_convert_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ConvertError) as raised:
                convert("status", [], 5, cli=[Path(root) / "absent"])
        self.assertEqual(raised.exception.code, "cli_not_found")
        self.assertEqual(raised.exception.operation, "status")

    def test_unresolvable_cli(self):
        for cli in ("/nonexistent/mlx-agent", [], [""]):
            with self.subTest(cli=cli):
                with self.assertRaises(ConvertError) as raised:
                    resolve_cli(cli)
                self.assertEqual(raised.exception.code, "cli_not_found")


if __name__ == "__main__":
    unittest.main()
