import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_agent.installer import Installer
from mlx_agent.providers import ProviderRegistry


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

    def test_wire_prompts_require_render_then_reviewed_apply_preview_then_confirmation(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            generated = generator.generate(("claude", "agentskills"), Path(directory))
            wire_prompts = [
                path for path in generated
                if path.name in {"mlx-wire.md", "mlx-advisor.md"}
                or (path.name == "SKILL.md" and "mlx-wire" in path.parts)
            ]
            self.assertEqual(5, len(wire_prompts))
            for prompt in wire_prompts:
                content = prompt.read_text(encoding="utf-8")
                render = content.index("wire render <model>")
                preview = content.index("wire apply <model> --target <target> --path <config-path> --json")
                confirmation = content.index("explicitly confirms that exact preview")
                apply = content.index("wire apply <model> --target <target> --path <config-path> --confirm --preview-hash <preview-hash> --json")
                self.assertLess(render, preview, str(prompt))
                self.assertLess(preview, confirmation, str(prompt))
                self.assertLess(confirmation, apply, str(prompt))

    def test_provider_and_every_installed_generic_skill_run_from_an_unrelated_working_directory(self):
        generator = load_generator()
        fixture = ROOT / "tests" / "fixtures" / "scout_responses.json"
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "package"
            generator.generate(("claude", "agentskills"), output_root)
            unrelated = Path(directory) / "unrelated"
            unrelated.mkdir()
            environment = dict(os.environ, MLX_AGENT_FIXTURE=str(fixture))
            executables = [output_root / "providers" / "claude" / "scripts" / "mlx-agent"]
            config_root = Path(directory) / "config"
            project_root = Path(directory) / "project"
            project_root.mkdir()
            installer = Installer(ProviderRegistry(ROOT / "plugin.json", home=Path(directory) / "home", config_root=config_root), project_root=project_root)
            plan = installer.plan("install", ["agentskills"], "user", project_root)
            installer.execute(plan, confirmed=plan.preview["preview_hash"])
            executables.extend(config_root / ".agents" / "skills" / "mlx-{0}".format(capability) / "scripts" / "mlx-agent" for capability in ("scout", "adopt", "wire"))
            for executable in executables:
                result = subprocess.run(
                    [sys.executable, str(executable), "discover", "--limit", "1", "--json"],
                    cwd=str(unrelated), env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual("discover", json.loads(result.stdout)["operation"])

    def test_generation_removes_only_stale_inventoried_files_and_check_reports_them(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("claude", "agentskills"), output_root)
            surface = output_root / "providers" / "agentskills"
            inventory_path = surface / ".mlx-agent-generated-files.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["files"].append("mlx-scout/obsolete-generated.md")
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            stale = surface / "mlx-scout" / "obsolete-generated.md"
            stale.write_text("generated stale\n", encoding="utf-8")
            handwritten = surface / "handwritten.md"
            handwritten.write_text("preserve me\n", encoding="utf-8")
            drift = generator._check(("claude", "agentskills"), output_root)
            self.assertIn(Path("providers/agentskills/mlx-scout/obsolete-generated.md"), drift)
            generator.generate(("claude", "agentskills"), output_root)
            self.assertFalse(stale.exists())
            self.assertEqual("preserve me\n", handwritten.read_text(encoding="utf-8"))

    def test_manifest_uses_three_portable_agentskills_artifacts(self):
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        artifacts = manifest["providers"]["agentskills"]["artifacts"]
        for capability in ("scout", "adopt", "wire"):
            self.assertIn(
                {"source": "providers/agentskills/mlx-{0}".format(capability), "destination": "skills/mlx-{0}".format(capability)},
                artifacts,
            )

    def test_manifest_descriptions_are_safe_yaml_scalars(self):
        generator = load_generator()
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        adversarial = copy.deepcopy(manifest)
        adversarial["capabilities"]["scout"]["description"] = 'colon: "quoted" value'
        prompt = generator._command_markdown(adversarial, "scout", "${CLAUDE_PLUGIN_ROOT}")
        self.assertIn('description: "colon: \\"quoted\\" value"', prompt)
        self.assertEqual(1, [line for line in prompt.splitlines() if line.startswith("description:")].__len__())
        adversarial["capabilities"]["scout"]["description"] = "safe\nfrontmatter: injected"
        with self.assertRaises(ValueError):
            generator._command_markdown(adversarial, "scout", "${CLAUDE_PLUGIN_ROOT}")

    def test_unittest_discovery_enumerates_contract_tests(self):
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_manifest.py", "-v"],
            cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Ran 4 tests", result.stderr)

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
                self.assertIn("absolute directory containing this SKILL.md", content)
                self.assertIn("<skill-dir>/scripts/mlx-agent", content)

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
