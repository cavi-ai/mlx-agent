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

    def test_packs_are_bundled_into_every_generated_skill(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            generated = generator.generate(("agentskills", "claude"), Path(directory))
            bundled = {
                path.relative_to(directory)
                for path in generated
                if "references" in path.parts and path.name in PACKS
            }
            for skill in ("mlx-scout", "mlx-adopt", "mlx-wire", "mlx-bench"):
                for name in PACKS:
                    expected = Path(
                        "providers/agentskills", skill,
                        "src/mlx_agent/resources/references", name,
                    )
                    self.assertIn(expected, bundled, str(expected))
            claude = {
                path for path in bundled
                if path.parts[:2] == ("providers", "claude")
            }
            self.assertEqual(len(claude), 3)

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


if __name__ == "__main__":
    unittest.main()
