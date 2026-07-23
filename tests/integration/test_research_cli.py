import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"


class ResearchCliTests(unittest.TestCase):
    def setUp(self):
        os.environ["MLX_AGENT_FIXTURE"] = str(FIXTURE)
        self.addCleanup(lambda: os.environ.pop("MLX_AGENT_FIXTURE", None))

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_research_writes_pack_json(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "research", "--domain", "legal contract review",
                "--role", "vision", "--keyword", "ocr",
                "--project", project, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["status"], "ok")
            self.assertIn("pack", payload["data"])
            written = Path(payload["data"]["path"])
            self.assertTrue(written.exists())
            self.assertEqual(written.parent, (Path(project) / "mlx-research").resolve())
            self.assertIn("# MLX Research Pack: legal contract review", written.read_text())
            sidecar = Path(payload["data"]["json_path"])
            self.assertEqual(sidecar, written.with_suffix(".json"))
            self.assertTrue(sidecar.exists())
            self.assertIn("intent", json.loads(sidecar.read_text()))

    def test_research_requires_domain(self):
        code, output = self._run(["research", "--json"])
        self.assertEqual(code, 2)
        self.assertIn("domain_required", output)

    def test_research_human_output_prints_path(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "research", "--domain", "audio", "--role", "vision",
                "--project", project,
            ])
            self.assertEqual(code, 0)
            self.assertIn("mlx-research", output)

    def test_research_no_write_renders_markdown_only(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "research", "--domain", "legal contract review",
                "--role", "vision", "--project", project, "--no-write",
            ])
            self.assertEqual(code, 0)
            self.assertIn("# MLX Research Pack: legal contract review", output)
            self.assertNotIn("Research pack written to", output)
            self.assertFalse((Path(project) / "mlx-research").exists())

    def test_research_invalid_intent(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "research", "--domain", "audio", "--memory-gb", "-1",
                "--project", project, "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("invalid_intent", output)


if __name__ == "__main__":
    unittest.main()
