import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class ConvertCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_start_preview_requires_confirmation(self):
        code, output = self._run([
            "convert", "start", "--repo", "pub/model", "--q-bits", "4",
        ])
        self.assertEqual(code, 2)
        self.assertIn("preview_hash", output)
        self.assertIn("Confirmation required", output)

    def test_start_preview_json(self):
        code, output = self._run([
            "convert", "start", "--repo", "pub/model", "--json",
        ])
        self.assertEqual(code, 2)
        payload = json.loads(output)
        self.assertTrue(payload["data"]["requires_confirmation"])
        self.assertEqual(payload["data"]["plan"]["q_bits"], 4)

    def test_confirm_without_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "convert", "start", "--repo", "pub/model",
                "--out", str(Path(directory) / "out"),
                "--confirm", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "preview_hash_required")

    def test_status_empty(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "convert", "status", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["jobs"], [])

    def test_invalid_repo(self):
        code, output = self._run(["convert", "start", "--repo", "bad", "--json"])
        self.assertEqual(code, 2)
        payload = json.loads(output)
        self.assertEqual(payload["error"]["code"], "invalid_repo")


if __name__ == "__main__":
    unittest.main()
