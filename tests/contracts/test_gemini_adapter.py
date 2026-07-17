"""Contract tests for the native Gemini CLI extension surface.

Official contract sources, checked 2026-07-17:
* https://geminicli.com/docs/extensions/reference/
* https://geminicli.com/docs/cli/commands/
* https://geminicli.com/docs/cli/skills/
"""

import importlib.util
import json
import os
import re
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


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_adapters", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_command_toml_fallback(content):
    """Parse only the generated two-string TOML subset on Python 3.9/3.10."""
    values = {}
    for key, quoted, block in re.findall(
        r'^([a-z_]+)\s*=\s*(?:"((?:\\.|[^"\\])*)"|"""([\s\S]*?)""")\s*$', content, re.M
    ):
        values[key] = block if block else json.loads('"' + quoted + '"')
    if set(values) != {"description", "prompt"}:
        raise ValueError("unsupported Gemini command TOML")
    return values


def parse_command_toml(content):
    """Parse Gemini's small command TOML subset on every supported Python."""
    try:
        import tomllib
    except ImportError:
        return parse_command_toml_fallback(content)
    return tomllib.loads(content)


class GeminiAdapterContractTests(unittest.TestCase):
    def test_manifest_has_native_gemini_slash_command_mapping(self):
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        gemini = manifest["providers"]["gemini"]
        self.assertEqual({"kind": "command", "prefix": "/"}, gemini["invocation"])
        self.assertEqual(["mlx-scout", "mlx-adopt", "mlx-wire"], gemini["commands"])

    def test_generated_extension_has_required_manifest_commands_and_skills(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("gemini",), output_root)
            extension_root = output_root / "providers" / "gemini"
            manifest = json.loads((extension_root / "gemini-extension.json").read_text(encoding="utf-8"))
            self.assertEqual("mlx-agent", manifest["name"])
            self.assertEqual("0.1.0", manifest["version"])
            self.assertIsInstance(manifest["description"], str)
            self.assertEqual([], generator._check(("gemini",), output_root))
            for capability in CAPABILITIES:
                command = parse_command_toml(
                    (extension_root / "commands" / "mlx-{}.toml".format(capability)).read_text(encoding="utf-8")
                )
                fallback = parse_command_toml_fallback(
                    (extension_root / "commands" / "mlx-{}.toml".format(capability)).read_text(encoding="utf-8")
                )
                self.assertEqual(command, fallback)
                self.assertEqual(
                    "Activate and follow the bundled mlx-{} skill.".format(capability), command["prompt"].splitlines()[0]
                )
                skill = extension_root / "skills" / "mlx-{}".format(capability) / "SKILL.md"
                self.assertTrue(skill.is_file())
                self.assertIn("canonical capability ID: mlx-agent.{}".format(capability), skill.read_text(encoding="utf-8"))
                self.assertTrue((skill.parent / "scripts" / "mlx-agent").is_file())

    def test_commands_are_exactly_the_manifest_capabilities_and_do_not_embed_absolute_paths(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            extension_root = Path(directory) / "providers" / "gemini"
            generator.generate(("gemini",), Path(directory))
            command_names = sorted(path.stem for path in (extension_root / "commands").glob("*.toml"))
            self.assertEqual(["mlx-adopt", "mlx-scout", "mlx-wire"], command_names)
            for path in extension_root.rglob("*"):
                if path.is_file() and path.name != ".mlx-agent-generated-files.json":
                    self.assertNotIn(str(ROOT), path.read_text(encoding="utf-8", errors="replace"), str(path))

    def test_gemini_installer_copies_extension_for_user_and_project_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            installer = Installer(
                ProviderRegistry(ROOT / "plugin.json", home=root / "home", config_root=root / "config"),
                project_root=project,
            )
            for scope, extension_root in (
                ("user", root / "config" / ".gemini" / "extensions" / "mlx-agent"),
                ("project", project / ".gemini" / "extensions" / "mlx-agent"),
            ):
                plan = installer.plan("install", ["gemini"], scope, project)
                installer.execute(plan, confirmed=plan.preview["preview_hash"])
                self.assertTrue((extension_root / "gemini-extension.json").is_file())
                for capability in CAPABILITIES:
                    self.assertTrue((extension_root / "commands" / "mlx-{}.toml".format(capability)).is_file())

    def test_recursive_provider_artifacts_exclude_runtime_bytecode(self):
        registry = ProviderRegistry(ROOT / "plugin.json")
        for provider in ("codex", "gemini"):
            definition = registry.definitions()[provider]
            with self.subTest(provider=provider):
                self.assertFalse(any("__pycache__" in item.source.parts for item in definition.artifacts))
                self.assertFalse(any(item.source.suffix == ".pyc" for item in definition.artifacts))

    def test_smoke_script_is_isolated_and_checks_extension_command_and_bundle_proof(self):
        smoke = (ROOT / "tests" / "smoke" / "gemini.sh").read_text(encoding="utf-8")
        self.assertIn("SKIP: Gemini CLI unavailable", smoke)
        self.assertIn("gemini extensions install", smoke)
        self.assertIn("gemini extensions list", smoke)
        self.assertIn("gemini skills list", smoke)
        self.assertIn("/mlx-scout", smoke)
        self.assertIn("gemini extensions uninstall mlx-agent", smoke)
        self.assertIn("separate self-contained bundle proof", smoke)
        self.assertIn("MLX_AGENT_FIXTURE", smoke)


if __name__ == "__main__":
    unittest.main()
