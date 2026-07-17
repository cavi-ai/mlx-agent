"""Contract tests for the native Codex plugin surface.

Official contract sources, checked 2026-07-17:
* https://github.com/openai/plugins
* https://github.com/openai/codex/blob/main/codex-rs/skills/src/assets/samples/plugin-creator/references/plugin-json-spec.md
* https://github.com/openai/codex/issues/11817

Codex plugins use `.codex-plugin/plugin.json` plus `skills/`.  Custom slash
commands are unsupported; users invoke these installed skills with `$`.
"""

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
CAPABILITIES = ("scout", "adopt", "wire")
OFFICIAL_PLUGIN_SPEC = (
    "https://github.com/openai/codex/blob/main/codex-rs/skills/src/assets/"
    "samples/plugin-creator/references/plugin-json-spec.md"
)
OFFICIAL_SLASH_STATUS = "https://github.com/openai/codex/issues/11817"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_adapters", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CodexAdapterContractTests(unittest.TestCase):
    def test_smoke_script_isolates_codex_home_and_never_claims_slash_command_support(self):
        smoke = ROOT / "tests" / "smoke" / "codex.sh"
        content = smoke.read_text(encoding="utf-8")
        self.assertIn("CODEX_HOME", content)
        self.assertIn("SKIP: Codex CLI unavailable", content)
        self.assertIn("$mlx-scout", content)
        self.assertNotIn("commands/mlx-scout", content)
        self.assertIn("plugin marketplace add", content)
        self.assertIn("plugin remove", content)

    def test_manifest_declares_codex_skills_not_unsupported_slash_commands(self):
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        codex = manifest["providers"]["codex"]
        self.assertEqual({"kind": "skill", "prefix": "$"}, codex["invocation"])
        self.assertEqual(["mlx-scout", "mlx-adopt", "mlx-wire"], codex["commands"])

    def test_generated_plugin_has_the_official_manifest_and_three_skill_entries(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("codex",), output_root)
            plugin_root = output_root / "providers" / "codex"
            metadata = json.loads(
                (plugin_root / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
            )
            self.assertEqual("mlx-agent", metadata["name"])
            self.assertEqual("0.1.0", metadata["version"])
            self.assertEqual("./skills/", metadata["skills"])
            self.assertEqual("Sasan Sotoodehfar", metadata["author"]["name"])
            self.assertIn("interface", metadata)
            self.assertFalse((plugin_root / "commands").exists())
            self.assertEqual([], generator._check(("codex",), output_root))
            for capability in CAPABILITIES:
                skill = plugin_root / "skills" / "mlx-{0}".format(capability) / "SKILL.md"
                content = skill.read_text(encoding="utf-8")
                self.assertIn('name: "mlx-{0}"'.format(capability), content)
                self.assertIn("canonical capability ID: mlx-agent.{0}".format(capability), content)
                self.assertIn("$mlx-{0}".format(capability), content)
                self.assertTrue((skill.parent / "scripts" / "mlx-agent").is_file())

    def test_codex_installer_roots_match_the_official_plugin_marketplace_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = ProviderRegistry(
                ROOT / "plugin.json", home=root / "home", config_root=root / "config"
            ).definitions()["codex"]
            self.assertEqual((root / "home" / "plugins" / "mlx-agent").resolve(), definition.user_root)
            self.assertEqual(Path(".agents/plugins/mlx-agent"), definition.project_root)
            self.assertEqual(
                (root / "project" / ".agents" / "plugins" / "mlx-agent").resolve(),
                definition.destination("project", root / "project"),
            )
            self.assertEqual("skill", definition.invocation_kind)
            self.assertEqual("$", definition.invocation_prefix)

    def test_installed_codex_runtime_runs_from_an_unrelated_working_directory(self):
        fixture = ROOT / "tests" / "fixtures" / "scout_responses.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            installer = Installer(
                ProviderRegistry(ROOT / "plugin.json", home=root / "home", config_root=root / "config"),
                project_root=project,
            )
            plan = installer.plan("install", ["codex"], "user", project)
            installer.execute(plan, confirmed=plan.preview["preview_hash"])
            unrelated = root / "unrelated"
            unrelated.mkdir()
            executable = root / "home" / "plugins" / "mlx-agent" / "skills" / "mlx-scout" / "scripts" / "mlx-agent"
            result = subprocess.run(
                [sys.executable, str(executable), "discover", "--limit", "1", "--json"],
                cwd=str(unrelated),
                env=dict(os.environ, MLX_AGENT_FIXTURE=str(fixture)),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("discover", json.loads(result.stdout)["operation"])


if __name__ == "__main__":
    unittest.main()
