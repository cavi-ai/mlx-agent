import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.cli import main


class ConvertCliTests(unittest.TestCase):
    def test_explicit_mlx_root_does_not_expand_to_default_gguf_folders(self):
        with TemporaryDirectory() as folder:
            selected = Path(folder) / 'selected'; selected.mkdir()
            excluded = Path(folder) / 'excluded'; excluded.mkdir()
            with patch('mlx_agent.gguf.default_gguf_roots', return_value=[str(excluded)]):
                code, output = self._run(['convert', 'scan', '--mlx-root', str(selected), '--json'])
            self.assertEqual(code, 0)
            roots = json.loads(output)['data']['roots']
            self.assertEqual(roots['gguf'], [])
            self.assertEqual(roots['mlx'], [str(selected)])


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

    def test_downloaded_directory_is_handed_to_converter_without_repo_resolution(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "config.json").write_text('{"model_type":"qwen2"}')
            code, output = self._run(["convert", "start", "--repo", "pub/model", "--source-path", str(source), "--json"])
            self.assertEqual(code, 2)
            plan = json.loads(output)["data"]["plan"]
            self.assertEqual(plan["argv"][2], str(source))
            self.assertEqual(plan["source"]["path"], str(source))

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
