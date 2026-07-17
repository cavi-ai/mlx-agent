import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "generate_adapters.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_adapters", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratedAdapterTests(unittest.TestCase):
    def test_generation_matches_committed_adapters_byte_for_byte(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generated = generator.generate(("claude", "agentskills"), output_root)
            relative_paths = [path.relative_to(output_root) for path in generated]
            self.assertEqual(relative_paths, sorted(relative_paths, key=str))
            for relative_path in relative_paths:
                self.assertEqual(
                    (output_root / relative_path).read_bytes(),
                    (ROOT / relative_path).read_bytes(),
                    str(relative_path),
                )

    def test_prompts_are_capability_parity_wrappers_over_the_structured_cli(self):
        generator = load_generator()
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generated = generator.generate(("claude", "agentskills"), output_root)
            prompts = [
                path for path in generated
                if path.suffix == ".md" and ("commands" in path.parts or "agents" in path.parts or "agentskills" in path.parts)
            ]
            self.assertEqual(len(prompts), 11)
            for prompt in prompts:
                content = prompt.read_text(encoding="utf-8")
                self.assertTrue(content.endswith("\n"))
                self.assertNotIn("\r", content)
                self.assertNotRegex(content.lower(), r"edit (the )?(user )?config|insert .*config")
                self.assertNotIn("model-ranking", content.lower())
            for capability in manifest["capabilities"]:
                self.assertTrue(
                    any("canonical capability ID: {0}.{1}".format(manifest["identity"], capability) in prompt.read_text(encoding="utf-8") for prompt in prompts),
                    capability,
                )
            claude_commands = [path for path in prompts if "commands" in path.parts]
            for prompt in claude_commands:
                self.assertIn("${CLAUDE_PLUGIN_ROOT}", prompt.read_text(encoding="utf-8"))
            generic_skills = [path for path in prompts if "agentskills" in path.parts]
            self.assertEqual(len(generic_skills), 3)
            for skill in generic_skills:
                content = skill.read_text(encoding="utf-8")
                self.assertNotIn("CLAUDE_", content)
                self.assertIn("../../../scripts/mlx-agent", content)
                self.assertNotIn("../../../scripts/mlx-agent/scripts/mlx-agent", content)

    def test_claude_adopt_workflow_only_delegates_to_durable_adoption_state(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("claude",), output_root)
            workflow = (output_root / "scripts" / "mlx-adopt.workflow.mjs").read_text(encoding="utf-8")
        self.assertIn("adopt start", workflow)
        self.assertIn("adopt resume", workflow)
        self.assertIn("--state", workflow)
        self.assertNotIn("reasoning-confirmed", workflow)
        self.assertNotIn("/api/tags", workflow)
        self.assertNotIn("model card", workflow.lower())
        self.assertIn("const shellQuote", workflow)
        self.assertIn("const allowedRoles", workflow)
