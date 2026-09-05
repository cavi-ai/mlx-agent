"""Contract tests for the native Agy plugin surface."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_agent.command_args import CommandArgumentError, parse_command_arguments
from mlx_agent.installer import Installer
from mlx_agent.providers import ProviderRegistry


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "generate_adapters.py"
CAPABILITIES = ("scout", "adopt", "wire", "bench", "doctor", "watch", "fleet")


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_adapters", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AgyAdapterContractTests(unittest.TestCase):
    def test_manifest_has_native_agy_skill_mapping(self):
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        agy = manifest["providers"]["agy"]
        self.assertEqual({"kind": "skill", "prefix": ""}, agy["invocation"])
        self.assertEqual(["mlx-{}".format(name) for name in CAPABILITIES], agy["commands"])
        self.assertEqual(["agy"], agy["detect_commands"])

    def test_generated_package_has_a_plugin_root_and_all_native_skills(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator.generate(("agy",), root)
            package = root / "providers" / "agy"
            metadata = json.loads((package / "plugin.json").read_text(encoding="utf-8"))
            self.assertEqual("mlx-agent", metadata["name"])
            self.assertEqual("CAVI AI", metadata["author"]["organization"])
            self.assertEqual([], generator._check(("agy",), root))
            for capability in CAPABILITIES:
                skill = package / "skills" / "mlx-{}".format(capability)
                self.assertTrue((skill / "SKILL.md").is_file())
                self.assertTrue((skill / "scripts" / "mlx-agent").is_file())
                self.assertNotIn(str(ROOT), (skill / "SKILL.md").read_text(encoding="utf-8"))

    def test_installer_copies_and_removes_agy_plugin_in_both_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            installer = Installer(
                ProviderRegistry(ROOT / "plugin.json", home=root / "home", config_root=root / "config"),
                project_root=project,
            )
            expected = {
                "user": root / "home" / ".gemini" / "config" / "plugins" / "mlx-agent",
                "project": project / ".agents" / "plugins" / "mlx-agent",
            }
            for scope, package in expected.items():
                plan = installer.plan("install", ["agy"], scope, project)
                self.assertEqual("applied", installer.execute(plan, confirmed=plan.preview["preview_hash"]).status)
                self.assertTrue((package / "plugin.json").is_file())
                self.assertTrue((package / "skills" / "mlx-scout" / "SKILL.md").is_file())
                remove = installer.plan("uninstall", ["agy"], scope, project)
                self.assertEqual("rolled_back", installer.execute(remove, confirmed=remove.preview["preview_hash"]).status)
                self.assertFalse((package / "plugin.json").exists())

    def test_installed_agy_skill_runs_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            installer = Installer(
                ProviderRegistry(ROOT / "plugin.json", home=root / "home", config_root=root / "config"),
                project_root=project,
            )
            plan = installer.plan("install", ["agy"], "project", project)
            installer.execute(plan, confirmed=plan.preview["preview_hash"])
            launcher = project / ".agents" / "plugins" / "mlx-agent" / "skills" / "mlx-scout" / "scripts" / "mlx-agent"
            result = subprocess.run(
                [sys.executable, str(launcher), "discover", "--limit", "1", "--json"],
                cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=dict(os.environ, MLX_AGENT_FIXTURE=str(ROOT / "tests" / "fixtures" / "scout_responses.json")),
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("discover", json.loads(result.stdout)["operation"])

    def test_shared_command_parser_rejects_hostile_input_without_host_specific_names(self):
        self.assertEqual(
            ["discover", "--role", "coding", "--limit", "2", "--offline", "--json"],
            parse_command_arguments("scout", "--role coding --limit 2 --offline --json"),
        )
        for capability, raw in (
            ("scout", "--role coding; touch owned"),
            ("adopt", "start --state ../outside.json --role coding"),
            ("wire", "render model/id --path config.json --target mlx_lm; whoami"),
        ):
            with self.subTest(capability=capability, raw=raw):
                with self.assertRaises(CommandArgumentError):
                    parse_command_arguments(capability, raw)


if __name__ == "__main__":
    unittest.main()
