import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class ServeLaunchdCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_launchd_preview_requires_confirmation(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "hf"
            (cache / "models--pub--model" / "snapshots" / "r1").mkdir(parents=True)
            (cache / "models--pub--model" / "refs").mkdir()
            (cache / "models--pub--model" / "refs" / "main").write_text("r1")
            code, output = self._run([
                "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
                "--launchd", "--launchd-dir", directory,
                "--hf-cache", str(cache), "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertTrue(payload["data"]["requires_confirmation"])
            plist = Path(directory) / "com.mlx-agent.serve.8080.plist"
            self.assertFalse(plist.exists())

    def test_launchd_apply_installs_plist(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "hf"
            (cache / "models--pub--model" / "snapshots" / "r1").mkdir(parents=True)
            (cache / "models--pub--model" / "refs").mkdir()
            (cache / "models--pub--model" / "refs" / "main").write_text("r1")
            argv = [
                "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
                "--launchd", "--launchd-dir", directory,
                "--hf-cache", str(cache), "--receipts-dir", directory, "--json",
            ]
            code, output = self._run(argv)
            preview = json.loads(output)["data"]["preview"]
            code, output = self._run(argv + [
                "--confirm", "--preview-hash", preview["preview_hash"],
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["receipt"]["status"], "applied")
            plist = Path(directory) / "com.mlx-agent.serve.8080.plist"
            self.assertTrue(plist.is_file())
            self.assertIn("pub/model", plist.read_text())

    def test_launchd_refuses_missing_model(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
                "--launchd", "--launchd-dir", directory,
                "--hf-cache", str(Path(directory) / "empty"), "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "model_not_local")

    def test_launchd_refuses_existing_plist(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "com.mlx-agent.serve.8080.plist").write_text("x")
            code, output = self._run([
                "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
                "--launchd", "--launchd-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "output_exists")


if __name__ == "__main__":
    unittest.main()
