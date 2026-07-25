import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class ServeCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_start_preview_requires_confirmation(self):
        code, output = self._run([
            "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
        ])
        self.assertEqual(code, 2)
        self.assertIn("preview_hash", output)
        self.assertIn("Confirmation required", output)

    def test_start_preview_json(self):
        code, output = self._run([
            "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm", "--json",
        ])
        self.assertEqual(code, 2)
        payload = json.loads(output)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["data"]["requires_confirmation"])
        self.assertEqual(payload["data"]["plan"]["port"], 8080)

    def test_confirm_without_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
                "--confirm", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "preview_hash_required")

    def test_status_empty(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "serve", "status", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["servers"], [])

    def test_stop_without_receipt(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "serve", "stop", "--port", "8080",
                "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "receipt_not_found")

    def test_unsupported_runtime_choices_are_rejected(self):
        with self.assertRaises(SystemExit):
            self._run(["serve", "start", "--repo", "pub/model", "--runtime", "ollama"])


if __name__ == "__main__":
    unittest.main()
