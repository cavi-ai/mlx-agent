import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class LoraCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def _dataset(self, root):
        path = Path(root) / "train.jsonl"
        path.write_text('{"text": "hello"}\n')

    def test_start_preview_requires_confirmation(self):
        with TemporaryDirectory() as directory:
            self._dataset(directory)
            code, output = self._run([
                "lora", "start", "--repo", "pub/model", "--data", directory,
            ])
            self.assertEqual(code, 2)
            self.assertIn("preview_hash", output)
            self.assertIn("Confirmation required", output)

    def test_start_preview_json(self):
        with TemporaryDirectory() as directory:
            self._dataset(directory)
            code, output = self._run([
                "lora", "start", "--repo", "pub/model", "--data", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertTrue(payload["data"]["requires_confirmation"])
            self.assertEqual(payload["data"]["plan"]["dataset"]["train_lines"], 1)

    def test_invalid_dataset_fails_before_preview(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "lora", "start", "--repo", "pub/model", "--data", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "dataset_invalid")

    def test_confirm_without_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            self._dataset(directory)
            code, output = self._run([
                "lora", "start", "--repo", "pub/model", "--data", directory,
                "--out", str(Path(directory) / "adapter"),
                "--confirm", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "preview_hash_required")

    def test_status_empty(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "lora", "status", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["jobs"], [])


if __name__ == "__main__":
    unittest.main()
