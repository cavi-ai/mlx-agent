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
    def test_smoke_script_proves_codex_skill_discovery_and_noninteractive_execution(self):
        smoke = ROOT / "tests" / "smoke" / "codex.sh"
        content = smoke.read_text(encoding="utf-8")
        self.assertIn("CODEX_HOME", content)
        self.assertIn("SKIP: Codex CLI unavailable", content)
        self.assertIn("codex debug prompt-input", content)
        self.assertIn("codex exec", content)
        self.assertIn("MLX_AGENT_SMOKE_EXECUTION_MARKER", content)
        self.assertIn("MLX_AGENT_FIXTURE", content)
        self.assertIn("$mlx-agent:mlx-scout", content)
        self.assertIn("$mlx-agent:mlx-adopt", content)
        self.assertIn("$mlx-agent:mlx-wire", content)
        self.assertNotIn("commands/mlx-scout", content)
        self.assertIn("plugin marketplace add", content)
        self.assertIn("plugin remove", content)

    def test_smoke_script_executes_the_isolated_codex_lifecycle(self):
        """The shell smoke copies auth into its ephemeral home before `exec`.

        A harmless stand-in CLI records the public Codex calls and refuses the
        noninteractive invocation unless the temporary CODEX_HOME has an auth
        file. This exercises the script's lifecycle rather than merely
        inspecting its text.
        """
        smoke = ROOT / "tests" / "smoke" / "codex.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            home = root / "caller-home"
            (home / ".codex").mkdir(parents=True)
            (home / ".codex" / "auth.json").write_text("{}\n", encoding="utf-8")
            log = root / "codex-calls.log"
            fake_codex = bin_dir / "codex"
            fake_codex.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' \"$*\" >> "$FAKE_CODEX_LOG"
case "${1:-} ${2:-}" in
  "plugin marketplace") exit 0 ;;
  "plugin list") printf '{"plugins":[{"name":"mlx-agent"}]}\\n' ;;
  "plugin add") exit 0 ;;
  "plugin remove") exit 0 ;;
  "debug prompt-input")
    printf '{"developer":["mlx-agent:mlx-scout","mlx-agent:mlx-adopt","mlx-agent:mlx-wire"]}\\n'
    ;;
  "exec "*)
    test -f "$CODEX_HOME/auth.json"
    output=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--output-last-message" ]; then
        shift
        output="$1"
      fi
      shift
    done
    printf '{"operation":"discover","fixture_model":"lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-Q8","marker":"MLX_AGENT_SMOKE_EXECUTION_MARKER"}\\n' > "$output"
    ;;
  *) printf 'unexpected fake Codex call: %s\\n' "$*" >&2; exit 1 ;;
esac
""",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "HOME": str(home),
                    "PATH": str(bin_dir) + os.pathsep + environment["PATH"],
                    "FAKE_CODEX_LOG": str(log),
                }
            )
            environment.pop("CODEX_HOME", None)
            result = subprocess.run(
                ["bash", str(smoke)],
                cwd=str(ROOT),
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("PASS: Codex plugin installed", result.stdout)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("plugin marketplace add", calls)
            self.assertIn("debug prompt-input", calls)
            self.assertIn("exec --ephemeral", calls)
            self.assertIn("plugin remove", calls)

    def test_smoke_removes_plugin_when_fixture_only_exec_fails(self):
        smoke = ROOT / "tests" / "smoke" / "codex.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            home = root / "caller-home"
            (home / ".codex").mkdir(parents=True)
            (home / ".codex" / "auth.json").write_text("{}\n", encoding="utf-8")
            log = root / "codex-calls.log"
            fake_codex = bin_dir / "codex"
            fake_codex.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_CODEX_LOG"
case "${1:-} ${2:-}" in
  "plugin marketplace"|"plugin add"|"plugin remove") exit 0 ;;
  "plugin list") printf '{"plugins":[{"name":"mlx-agent"}]}\\n' ;;
  "debug prompt-input") printf '{"developer":["mlx-agent:mlx-scout","mlx-agent:mlx-adopt","mlx-agent:mlx-wire"]}\\n' ;;
  "exec "*) exit 9 ;;
  *) exit 1 ;;
esac
""",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            environment = dict(os.environ, HOME=str(home), PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], FAKE_CODEX_LOG=str(log))
            environment.pop("CODEX_HOME", None)
            result = subprocess.run(["bash", str(smoke)], cwd=str(ROOT), env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(9, result.returncode)
            self.assertIn("plugin remove", log.read_text(encoding="utf-8"))

    def test_public_docs_record_the_codex_skill_invocation_correction(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        install_guide = (ROOT / "docs" / "install" / "codex.md").read_text(encoding="utf-8")
        for content in (readme, install_guide):
            normalized = " ".join(content.split())
            self.assertIn("$mlx-agent:mlx-scout", normalized)
            self.assertIn("$mlx-agent:mlx-adopt", normalized)
            self.assertIn("$mlx-agent:mlx-wire", normalized)
            self.assertRegex(normalized, r"(?:does not support|not) custom slash commands")

    def test_manifest_declares_codex_skills_not_unsupported_slash_commands(self):
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        codex = manifest["providers"]["codex"]
        self.assertEqual({"kind": "skill", "prefix": "$"}, codex["invocation"])
        self.assertEqual(
            ["mlx-agent:mlx-scout", "mlx-agent:mlx-adopt", "mlx-agent:mlx-wire"],
            codex["commands"],
        )

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
            self.assertEqual("0.2.0", metadata["version"])
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
                self.assertIn("$mlx-agent:mlx-{0}".format(capability), content)
                self.assertTrue((skill.parent / "scripts" / "mlx-agent").is_file())

    def test_codex_installer_roots_match_the_official_plugin_marketplace_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = ProviderRegistry(
                ROOT / "plugin.json", home=root / "home", config_root=root / "config"
            ).definitions()["codex"]
            self.assertEqual((root / "home" / "plugins" / "mlx-agent").resolve(), definition.user_root)
            self.assertEqual(Path("plugins/mlx-agent"), definition.project_root)
            self.assertEqual(
                (root / "project" / "plugins" / "mlx-agent").resolve(),
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
