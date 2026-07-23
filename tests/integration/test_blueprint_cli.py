import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class BlueprintCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_blueprint_writes_pack_json(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "blueprint",
                "--goal", "On-device legal OCR assistant",
                "--modality", "document-vision",
                "--project", project,
                "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["status"], "ok")
            pack = payload["data"]["pack"]
            self.assertEqual(pack["brief"]["goal"], "On-device legal OCR assistant")
            self.assertEqual(pack["brief"]["modalities"], ["document-vision"])
            path = Path(payload["data"]["path"])
            self.assertTrue(path.exists())
            self.assertEqual(path.parent.name, "mlx-blueprints")
            markdown = path.read_text()
            self.assertIn("# MLX Project Design Pack:", markdown)
            self.assertIn("## Quantization ideas", markdown)
            self.assertIn("mlx-vlm", markdown)
            self.assertTrue(path.with_suffix(".json").is_file())

    def test_blueprint_requires_goal(self):
        code, output = self._run(["blueprint", "--json"])
        self.assertEqual(code, 2)
        self.assertIn("goal_required", output)

    def test_blueprint_no_write(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "blueprint",
                "--goal", "ASR notes",
                "--modality", "audio",
                "--project", project,
                "--no-write",
            ])
            self.assertEqual(code, 0)
            self.assertIn("# MLX Project Design Pack: ASR notes", output)
            self.assertFalse((Path(project) / "mlx-blueprints").exists())


if __name__ == "__main__":
    unittest.main()
