"""Contract tests for the native OpenCode package.

Official contract sources, checked 2026-07-17:
* https://opencode.ai/docs/commands/
* https://opencode.ai/docs/agents/
* https://opencode.ai/docs/skills/
* https://opencode.ai/docs/plugins/
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


def frontmatter(content):
    match = re.match(r"\A---\n([\s\S]*?)\n---\n", content)
    if match is None:
        raise ValueError("missing YAML frontmatter")
    values = {}
    for line in match.group(1).splitlines():
        if not line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        values[key] = value.strip()
    return values


class OpenCodeAdapterContractTests(unittest.TestCase):
    def test_manifest_has_native_opencode_slash_command_mapping_and_native_artifacts(self):
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        opencode = manifest["providers"]["opencode"]
        self.assertEqual({"kind": "command", "prefix": "/"}, opencode["invocation"])
        self.assertEqual(["mlx-scout", "mlx-adopt", "mlx-wire"], opencode["commands"])
        self.assertEqual(["opencode"], opencode["detect_commands"])
        sources = {item["source"] for item in opencode["artifacts"]}
        self.assertIn("providers/opencode/opencode.json", sources)
        self.assertIn("providers/opencode/commands", sources)
        self.assertIn("providers/opencode/agents/mlx-advisor.md", sources)
        for capability in CAPABILITIES:
            self.assertIn("providers/opencode/skills/mlx-{0}".format(capability), sources)

    def test_generated_package_has_valid_config_and_exact_native_command_parity(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("opencode",), output_root)
            package_root = output_root / "providers" / "opencode"
            config = json.loads((package_root / "opencode.json").read_text(encoding="utf-8"))
            self.assertEqual("https://opencode.ai/config.json", config["$schema"])
            self.assertEqual([], generator._check(("opencode",), output_root))
            command_paths = sorted(path.name for path in (package_root / "commands").glob("*.md"))
            self.assertEqual(["mlx-adopt.md", "mlx-scout.md", "mlx-wire.md"], command_paths)
            for capability in CAPABILITIES:
                command = package_root / "commands" / "mlx-{0}.md".format(capability)
                content = command.read_text(encoding="utf-8")
                values = frontmatter(content)
                self.assertEqual("\"{0}\"".format(
                    json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))["capabilities"][capability]["description"]
                ), values["description"])
                self.assertIn("canonical capability ID: mlx-agent.{0}".format(capability), content)
                self.assertIn("<mlx-agent-untrusted-args>", content)
                self.assertIn("validated non-shell", content)
                self.assertIn("$ARGUMENTS", content)
                self.assertNotIn("!`", content)
                skill = package_root / "skills" / "mlx-{0}".format(capability) / "SKILL.md"
                self.assertTrue(skill.is_file())
                skill_values = frontmatter(skill.read_text(encoding="utf-8"))
                self.assertEqual("\"mlx-{0}\"".format(capability), skill_values["name"])
                self.assertIn("compatibility", skill_values)
                skill_text = skill.read_text(encoding="utf-8")
                self.assertIn("structured executor", skill_text)
                self.assertIn("shell: false", skill_text)

    def test_adopt_subtask_is_bounded_and_advisor_requires_confirmation_for_mutations(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("opencode",), output_root)
            package_root = output_root / "providers" / "opencode"
            adopt = (package_root / "commands" / "mlx-adopt.md").read_text(encoding="utf-8")
            self.assertEqual("true", frontmatter(adopt)["subtask"])
            self.assertIn("independent verification record", adopt)
            self.assertIn("one bounded", adopt)
            advisor = (package_root / "agents" / "mlx-advisor.md").read_text(encoding="utf-8")
            values = frontmatter(advisor)
            self.assertEqual("subagent", values["mode"])
            self.assertIn("ask", advisor)
            self.assertRegex(advisor, r"(?m)^\s+edit: ask$")
            self.assertRegex(advisor, r"(?m)^\s+bash: ask$")
            self.assertNotIn("edit: allow", advisor)
            self.assertNotIn("bash: allow", advisor)
            wire = (package_root / "commands" / "mlx-wire.md").read_text(encoding="utf-8")
            self.assertIn("transaction CLI", wire)
            self.assertIn("--confirm --preview-hash", wire)
            self.assertIn("Do not edit configuration directly", wire)

    def test_installer_copies_opencode_package_to_both_documented_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            installer = Installer(
                ProviderRegistry(ROOT / "plugin.json", home=root / "home", config_root=root / "config"),
                project_root=project,
            )
            for scope, package_root in (
                ("user", root / "home" / ".config" / "opencode"),
                ("project", project / ".opencode"),
            ):
                plan = installer.plan("install", ["opencode"], scope, project)
                installer.execute(plan, confirmed=plan.preview["preview_hash"])
                config_path = package_root / "opencode.json" if scope == "user" else project / "opencode.json"
                self.assertTrue(config_path.is_file())
                self.assertTrue((package_root / "agents" / "mlx-advisor.md").is_file())
                for capability in CAPABILITIES:
                    self.assertTrue((package_root / "commands" / "mlx-{}.md".format(capability)).is_file())
                    self.assertTrue((package_root / "skills" / "mlx-{}".format(capability) / "SKILL.md").is_file())

    def test_smoke_script_is_isolated_and_honest_about_unavailable_auth(self):
        smoke = (ROOT / "tests" / "smoke" / "opencode.sh").read_text(encoding="utf-8")
        self.assertIn("SKIP: OpenCode CLI unavailable", smoke)
        self.assertIn("XDG_CONFIG_HOME", smoke)
        self.assertIn("mlx-scout", smoke)
        self.assertIn("mlx-adopt", smoke)
        self.assertIn("mlx-wire", smoke)
        self.assertIn("uninstall opencode", smoke)
        self.assertIn("MLX_AGENT_OPENCODE_LIVE_COMMAND_DISCOVERY", smoke)
        self.assertIn("MLX_AGENT_FIXTURE", smoke)
        self.assertIn("never claim a model response if no auth", smoke.lower())

    def test_smoke_skips_successfully_when_opencode_is_unavailable(self):
        smoke = ROOT / "tests" / "smoke" / "opencode.sh"
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment["PATH"] = directory
            result = subprocess.run(
                ["/bin/bash", str(smoke)], cwd=str(ROOT), env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("SKIP: OpenCode CLI unavailable", result.stdout)


if __name__ == "__main__":
    unittest.main()
