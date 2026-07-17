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
        self.assertIn("providers/opencode/plugins", sources)
        self.assertIn("providers/opencode/src", sources)
        self.assertNotIn("providers/opencode/opencode.json", sources)
        self.assertIn("providers/opencode/commands", sources)
        self.assertIn("providers/opencode/agents/mlx-advisor.md", sources)
        for capability in CAPABILITIES:
            self.assertIn("providers/opencode/skills/mlx-{0}".format(capability), sources)

    def test_generated_package_has_native_plugin_and_exact_command_parity(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("opencode",), output_root)
            package_root = output_root / "providers" / "opencode"
            self.assertEqual([], generator._check(("opencode",), output_root))
            self.assertFalse((package_root / "opencode.json").exists())
            self.assertFalse((package_root / "src" / "mlx_agent" / "gemini_executor.py").exists())
            self.assertFalse((package_root / "src" / "mlx_agent" / "gemini_transport.py").exists())
            plugin = package_root / "plugins" / "mlx-agent-command.ts"
            plugin_text = plugin.read_text(encoding="utf-8")
            self.assertIn('tool: {', plugin_text)
            self.assertIn('mlx_agent_command', plugin_text)
            self.assertIn('tool.schema.enum(["scout", "adopt", "wire"])', plugin_text)
            self.assertIn('tool.schema.string().max(MAX_ARGUMENT_BYTES)', plugin_text)
            self.assertIn('Bun.spawn({', plugin_text)
            self.assertIn('cmd: ["python3", "-m", "mlx_agent.command_executor"', plugin_text)
            self.assertIn('"--provider", "opencode", "--capability", args.capability', plugin_text)
            self.assertIn('stdin: "pipe"', plugin_text)
            self.assertNotIn('Bun.$', plugin_text)
            self.assertNotIn('shell:', plugin_text)
            self.assertNotIn('gemini', plugin_text.lower())
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
                self.assertIn("mlx_agent_command", content)
                self.assertIn("exact raw argument string", content)
                self.assertIn("$ARGUMENTS", content)
                self.assertNotIn("!`", content)
                skill = package_root / "skills" / "mlx-{0}".format(capability) / "SKILL.md"
                self.assertTrue(skill.is_file())
                skill_values = frontmatter(skill.read_text(encoding="utf-8"))
                self.assertEqual("\"mlx-{0}\"".format(capability), skill_values["name"])
                self.assertIn("compatibility", skill_values)
                skill_text = skill.read_text(encoding="utf-8")
                self.assertIn("mlx_agent_command", skill_text)
                self.assertNotIn("gemini", skill_text.lower())

    def test_adopt_subtask_is_bounded_and_advisor_requires_confirmation_for_mutations(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("opencode",), output_root)
            package_root = output_root / "providers" / "opencode"
            adopt = (package_root / "commands" / "mlx-adopt.md").read_text(encoding="utf-8")
            scout = (package_root / "commands" / "mlx-scout.md").read_text(encoding="utf-8")
            wire = (package_root / "commands" / "mlx-wire.md").read_text(encoding="utf-8")
            self.assertNotIn("agent:", frontmatter(scout))
            self.assertNotIn("subtask:", frontmatter(scout))
            self.assertNotIn("agent:", frontmatter(wire))
            self.assertNotIn("subtask:", frontmatter(wire))
            self.assertEqual("mlx-advisor", frontmatter(adopt)["agent"])
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
                self.assertFalse((package_root / "opencode.json").exists())
                self.assertFalse((project / "opencode.json").exists())
                self.assertTrue((package_root / "agents" / "mlx-advisor.md").is_file())
                self.assertTrue((package_root / "plugins" / "mlx-agent-command.ts").is_file())
                for capability in CAPABILITIES:
                    self.assertTrue((package_root / "commands" / "mlx-{}.md".format(capability)).is_file())
                    self.assertTrue((package_root / "skills" / "mlx-{}".format(capability) / "SKILL.md").is_file())

    def test_installer_preserves_unowned_opencode_config_and_uninstalls_only_owned_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            config = project / "opencode.json"
            original = b'{"model":"unowned"}\n'
            config.write_bytes(original)
            installer = Installer(
                ProviderRegistry(ROOT / "plugin.json", home=root / "home", config_root=root / "config"),
                project_root=project,
            )
            plan = installer.plan("install", ["opencode"], "project", project)
            installer.execute(plan, confirmed=plan.preview["preview_hash"])
            self.assertEqual(original, config.read_bytes())
            self.assertTrue((project / ".opencode" / "plugins" / "mlx-agent-command.ts").is_file())
            remove = installer.plan("uninstall", ["opencode"], "project", project)
            installer.execute(remove, confirmed=remove.preview["preview_hash"])
            self.assertEqual(original, config.read_bytes())
            self.assertFalse((project / ".opencode" / "plugins" / "mlx-agent-command.ts").exists())

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
        self.assertIn("opencode_tool_transport.mjs", smoke)
        self.assertIn("no model", smoke.lower())
        self.assertIn("response is claimed when they are absent", smoke.lower())

    def test_smoke_skips_successfully_when_opencode_is_unavailable(self):
        smoke = ROOT / "tests" / "smoke" / "opencode.sh"
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment["PATH"] = "/usr/bin:/bin"
            result = subprocess.run(
                ["/bin/bash", str(smoke)], cwd=str(ROOT), env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("SKIP: OpenCode CLI unavailable", result.stdout)


if __name__ == "__main__":
    unittest.main()
