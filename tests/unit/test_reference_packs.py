import tempfile
import unittest
from pathlib import Path

from tests.contracts.test_generated_adapters import load_generator


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "src" / "mlx_agent" / "resources" / "references"
PACKS = ("quantization.md", "model-families.md", "troubleshooting.md")


class ReferencePackTests(unittest.TestCase):
    def test_packs_exist_and_are_substantive(self):
        for name in PACKS:
            path = REFERENCE_DIR / name
            self.assertTrue(path.is_file(), name)
            content = path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("# "), name)
            self.assertGreater(len(content), 1000, name)

    def test_packs_ship_once_per_package_that_installs_the_whole_plugin(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            generated = generator.generate(("agentskills", "claude", "codex", "agy", "opencode"), Path(directory))
            bundled = {
                path.relative_to(directory)
                for path in generated
                if "references" in path.parts and path.name in PACKS
            }
        expected = {
            Path("providers", provider, "src/mlx_agent/resources/references", name)
            for provider in ("codex", "agy")
            for name in PACKS
        }
        self.assertEqual(expected, bundled)

    def test_scout_skills_point_at_the_packs(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            generated = generator.generate(("agentskills",), Path(directory))
            skill = (
                Path(directory)
                / "providers/agentskills/mlx-scout/SKILL.md"
            ).read_text(encoding="utf-8")
            self.assertIn("references/quantization.md", skill)
            self.assertIn("references/model-families.md", skill)
            self.assertIn("references/troubleshooting.md", skill)
            self.assertIn("<skill-dir>/src/mlx_agent/resources/references/quantization.md", skill)


if __name__ == "__main__":
    unittest.main()
