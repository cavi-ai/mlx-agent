"""Contract tests for the native Gemini CLI extension surface.

Official contract sources, checked 2026-07-17:
* https://geminicli.com/docs/extensions/reference/
* https://geminicli.com/docs/cli/commands/
* https://geminicli.com/docs/cli/skills/
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_agent.installer import Installer
from mlx_agent.gemini_args import GeminiArgumentError, parse_gemini_arguments
from mlx_agent.gemini_executor import GeminiCommandError, command_args_root, execute_gemini_command
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
    def _args_file(self, content, binary=False):
        root = command_args_root()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="test-", suffix=".args", dir=str(root))
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content if binary else content.encode("utf-8"))
        return Path(name)

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
                self.assertIn("{{args}}", command["prompt"])
                self.assertIn("untrusted opaque command data", command["prompt"].lower())
                skill = extension_root / "skills" / "mlx-{}".format(capability) / "SKILL.md"
                self.assertTrue(skill.is_file())
                skill_text = skill.read_text(encoding="utf-8")
                self.assertIn("canonical capability ID: mlx-agent.{}".format(capability), skill_text)
                self.assertIn("mlx_agent.gemini_executor", skill_text)
                self.assertIn("non-shell file-writing tool/API", skill_text)
                self.assertIn("never interpolate raw", skill_text.lower())
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
                ("user", root / "home" / ".gemini" / "extensions" / "mlx-agent"),
                ("project", project / ".gemini" / "extensions" / "mlx-agent"),
            ):
                plan = installer.plan("install", ["gemini"], scope, project)
                installer.execute(plan, confirmed=plan.preview["preview_hash"])
                self.assertTrue((extension_root / "gemini-extension.json").is_file())
                for capability in CAPABILITIES:
                    self.assertTrue((extension_root / "commands" / "mlx-{}.toml".format(capability)).is_file())
            for capability in CAPABILITIES:
                self.assertTrue((project / ".gemini" / "commands" / "mlx-{}.toml".format(capability)).is_file())
                self.assertTrue((project / ".gemini" / "skills" / "mlx-{}".format(capability) / "SKILL.md").is_file())
            project_uninstall = installer.plan("uninstall", ["gemini"], "project", project)
            installer.execute(project_uninstall, confirmed=project_uninstall.preview["preview_hash"])
            self.assertFalse((project / ".gemini" / "extensions" / "mlx-agent" / "gemini-extension.json").exists())
            for capability in CAPABILITIES:
                self.assertFalse((project / ".gemini" / "commands" / "mlx-{}.toml".format(capability)).exists())
                self.assertFalse((project / ".gemini" / "skills" / "mlx-{}".format(capability) / "SKILL.md").exists())

    def test_gemini_user_scope_uses_home_not_config_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = ProviderRegistry(
                ROOT / "plugin.json", home=root / "home", config_root=root / "config"
            ).definitions()["gemini"]
            self.assertEqual((root / "home" / ".gemini" / "extensions" / "mlx-agent").resolve(), definition.user_root)
            self.assertEqual(Path(".gemini/extensions/mlx-agent"), definition.project_root)

    def test_gemini_argument_parser_builds_only_allowlisted_argv(self):
        self.assertEqual(
            ["discover", "--role", "coding", "--limit", "2", "--offline", "--json"],
            parse_gemini_arguments("scout", "--role coding --limit 2 --offline --json"),
        )
        self.assertEqual(
            ["adopt", "start", "--state", "state/adopt.json", "--role", "coding", "--offline", "--json"],
            parse_gemini_arguments("adopt", "start --state state/adopt.json --role coding --offline --json"),
        )
        self.assertEqual(
            ["wire", "render", "mlx-community/Qwen3-8B-4bit", "--target", "mlx_lm", "--path", "config/providers.json", "--json"],
            parse_gemini_arguments("wire", "render mlx-community/Qwen3-8B-4bit --target mlx_lm --path config/providers.json --json"),
        )

    def test_gemini_argument_parser_rejects_hostile_or_unknown_input_without_execution(self):
        hostile = (
            ("scout", "--role coding; touch owned"),
            ("scout", "--unknown value"),
            ("adopt", "start --state ../outside.json --role coding"),
            ("adopt", "resume --state $(touch owned)"),
            ("wire", "render mlx-community/Qwen3-8B-4bit --path config.json --target mlx_lm;whoami"),
            ("wire", "apply bad/model --path config.json --endpoint http://user:pass@127.0.0.1:8080"),
            ("wire", "render mlx-community/Qwen3-8B-4bit --path config\nnext.json"),
        )
        for capability, raw in hostile:
            with self.subTest(capability=capability, raw=raw):
                with self.assertRaises(GeminiArgumentError):
                    parse_gemini_arguments(capability, raw)

    def test_gemini_toml_arguments_flow_through_private_file_executor_to_fixture_scout(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            generator.generate(("gemini",), Path(directory))
            command = parse_command_toml(
                (Path(directory) / "providers" / "gemini" / "commands" / "mlx-scout.toml").read_text(encoding="utf-8")
            )
            self.assertIn("{{args}}", command["prompt"])
        args_file = self._args_file("--limit 1 --json")
        output = io.StringIO()
        previous_fixture = os.environ.get("MLX_AGENT_FIXTURE")
        os.environ["MLX_AGENT_FIXTURE"] = str(ROOT / "tests" / "fixtures" / "scout_responses.json")
        try:
            with contextlib.redirect_stdout(output):
                result = execute_gemini_command("scout", args_file)
        finally:
            if previous_fixture is None:
                os.environ.pop("MLX_AGENT_FIXTURE", None)
            else:
                os.environ["MLX_AGENT_FIXTURE"] = previous_fixture
        self.assertEqual({"status": "ok", "capability": "scout", "exit_code": 0}, result)
        self.assertFalse(args_file.exists())
        self.assertIn('"operation": "discover"', output.getvalue())

    def test_gemini_executor_rejects_hostile_file_input_without_invoking_core(self):
        hostile = ("--role 'coding'; touch owned", "--role $(touch owned)", "--role coding\n--json")
        for raw in hostile:
            with self.subTest(raw=raw):
                args_file = self._args_file(raw)
                calls = []
                with self.assertRaises(GeminiCommandError):
                    execute_gemini_command("scout", args_file, core=lambda argv: calls.append(argv))
                self.assertEqual([], calls)
                self.assertFalse(args_file.exists())

    def test_gemini_executor_rejects_unsafe_argument_files_without_invoking_core(self):
        calls = []
        outside = Path(tempfile.mkstemp(prefix="mlx-agent-outside-")[1])
        outside.write_text("--limit 1", encoding="utf-8")
        link = command_args_root() / "symlink.args"
        link.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        link.symlink_to(outside)
        oversized = self._args_file(b"x" * 4097, binary=True)
        invalid_utf8 = self._args_file(b"\xff", binary=True)
        try:
            for args_file in (link, oversized, invalid_utf8, outside):
                with self.subTest(path=args_file.name):
                    with self.assertRaises(GeminiCommandError):
                        execute_gemini_command("scout", args_file, core=lambda argv: calls.append(argv))
            self.assertEqual([], calls)
            self.assertTrue(outside.exists())
            self.assertFalse(link.exists())
            self.assertFalse(oversized.exists())
            self.assertFalse(invalid_utf8.exists())
        finally:
            if outside.exists():
                outside.unlink()

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
        self.assertIn("/commands list", smoke)
        self.assertIn("gemini skills list", smoke)
        self.assertIn("/mlx-scout", smoke)
        self.assertIn("gemini extensions uninstall mlx-agent", smoke)
        self.assertIn("separate self-contained bundle proof", smoke)
        self.assertIn("project scope", smoke.lower())
        self.assertIn("MLX_AGENT_GEMINI_LIVE_COMMAND_DISCOVERY", smoke)
        self.assertNotIn("not-a-secret", smoke)
        self.assertIn("MLX_AGENT_FIXTURE", smoke)


if __name__ == "__main__":
    unittest.main()
