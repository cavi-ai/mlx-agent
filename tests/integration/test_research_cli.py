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
            markdown = written.read_text()
            self.assertIn("# MLX Research Pack: legal contract review", markdown)
            self.assertIn("## Modality foundations", markdown)
            self.assertIn("document-vision", markdown)
            self.assertIn("## Adapters / LoRAs", markdown)
            self.assertIn("## Datasets", markdown)
            self.assertIn("## Dataset blueprint", markdown)
            sidecar = written.with_suffix(".json")
            self.assertTrue(sidecar.is_file())
            pack = json.loads(sidecar.read_text())
            self.assertIn("adapters", pack)
            self.assertIn("datasets", pack)
            self.assertIsNotNone(pack["dataset_blueprint"])
            self.assertEqual(pack["intent"]["modalities"], ["document-vision"])

    def test_research_fixture_mode_stays_offline_for_catalog(self):
        """Empty fixture adapters/datasets must not require network."""
        with TemporaryDirectory() as project:
            code, output = self._run([
                "research", "--domain", "offline catalog",
                "--role", "vision", "--keyword", "ocr",
                "--project", project, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            pack = payload["data"]["pack"]
            self.assertEqual(pack["adapters"], [])
            self.assertEqual(pack["datasets"], [])
            self.assertIsNotNone(pack["dataset_blueprint"])
            self.assertEqual(payload["status"], "ok")

    def test_research_requires_domain(self):
        code, output = self._run(["research", "--json"])
        self.assertEqual(code, 2)
        self.assertIn("domain_required", output)

    def test_research_requires_modality_when_undetected(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "research", "--domain", "billing helper",
                "--project", project, "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("modality_required", output)

    def test_research_modality_flag_seeds_pack(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "research", "--domain", "billing helper",
                "--modality", "audio", "--facet", "asr",
                "--project", project, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            pack = payload["data"]["pack"]
            self.assertEqual(pack["intent"]["modalities"], ["audio"])
            self.assertEqual(pack["intent"]["facets"], ["asr"])
            self.assertIn("general", pack["intent"]["roles"])
            markdown = Path(payload["data"]["path"]).read_text()
            self.assertIn("## Modality foundations", markdown)
            self.assertIn("`audio`", markdown)

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
                "--role", "vision", "--modality", "document-vision",
                "--project", project, "--no-write",
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
