import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class FuseCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def _adapter(self, root):
        path = Path(root) / "adapter"
        path.mkdir()
        (path / "adapter_config.json").write_text("{}")

    def test_start_preview_requires_confirmation(self):
        with TemporaryDirectory() as directory:
            self._adapter(directory)
            code, output = self._run([
                "fuse", "start", "--repo", "pub/model",
                "--adapter", str(Path(directory) / "adapter"),
            ])
            self.assertEqual(code, 2)
            self.assertIn("preview_hash", output)
            self.assertIn("Confirmation required", output)

    def test_invalid_adapter_fails_before_preview(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "fuse", "start", "--repo", "pub/model",
                "--adapter", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "adapter_invalid")

    def test_confirm_without_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            self._adapter(directory)
            code, output = self._run([
                "fuse", "start", "--repo", "pub/model",
                "--adapter", str(Path(directory) / "adapter"),
                "--out", str(Path(directory) / "fused"),
                "--confirm", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "preview_hash_required")

    def test_status_empty(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "fuse", "status", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["jobs"], [])


if __name__ == "__main__":
    unittest.main()
