import copy
import hashlib
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
            config_root = Path(directory) / "config"
            project_root = Path(directory) / "project"
            project_root.mkdir()
            home_root = Path(directory) / "home"
            installer = Installer(ProviderRegistry(ROOT / "plugin.json", home=home_root, config_root=config_root), project_root=project_root)
            plan = installer.plan("install", ["agentskills"], "user", project_root)
            installer.execute(plan, confirmed=plan.preview["preview_hash"])
            executions = [
                (
                    output_root / "providers" / "claude" / "scripts" / "mlx-agent",
                    ["discover", "--limit", "1", "--json"],
                    "discover",
                ),
                (
                    home_root / ".agents" / "skills" / "mlx-scout" / "scripts" / "mlx-agent",
                    ["discover", "--limit", "1", "--json"],
                    "discover",
                ),
                (
                    home_root / ".agents" / "skills" / "mlx-adopt" / "scripts" / "mlx-agent",
                    [
                        "adopt", "start", "--state", str(unrelated / "adoption.json"),
                        "--shortlist-limit", "1", "--fast", "--no-network", "--json",
                    ],
                    "adopt-start",
                ),
                (
                    home_root / ".agents" / "skills" / "mlx-wire" / "scripts" / "mlx-agent",
                    [
                        "wire", "render", "mlx-community/Test-4bit", "--target", "mlx_lm",
                        "--path", str(unrelated / "mlx-lm.json"), "--json",
                    ],
                    "wire-render",
                ),
            ]
            for executable, arguments, operation in executions:
                result = subprocess.run(
                    [sys.executable, str(executable)] + arguments,
                    cwd=str(unrelated), env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual(operation, json.loads(result.stdout)["operation"])

    def test_generated_provider_mcp_transport_runs_from_an_isolated_bundle(self):
        generator = load_generator()
        fixture = ROOT / "tests" / "fixtures" / "scout_responses.json"
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "package"
            generator.generate(("claude", "gemini"), output_root)
            unrelated = Path(directory) / "unrelated"
            unrelated.mkdir()
            environment = dict(os.environ, MLX_AGENT_FIXTURE=str(fixture))
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "mlx_agent_execute",
                    "arguments": {"capability": "scout", "arguments": "--limit 1 --json"},
                },
            }
            for provider in ("claude", "gemini"):
                executable = output_root / "providers" / provider / "scripts" / "mlx-agent-mcp"
                result = subprocess.run(
                    [sys.executable, str(executable)], input=json.dumps(request) + "\n",
                    cwd=str(unrelated), env=environment, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                response = json.loads(result.stdout)
                self.assertEqual(1, response["id"])
                self.assertFalse(response["result"]["isError"])
                payload = json.loads(response["result"]["content"][0]["text"])
                self.assertEqual("ok", payload["status"])
                self.assertEqual("discover", json.loads(payload["stdout"])["operation"])

    def test_generation_removes_only_hash_matched_stale_inventoried_files(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("claude", "agentskills"), output_root)
            surface = output_root / "providers" / "agentskills"
            inventory_path = surface / ".mlx-agent-generated-files.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            stale = surface / "mlx-scout" / "SKILL.md"
            handwritten = surface / "handwritten.md"
            handwritten.write_text("preserve me\n", encoding="utf-8")
            rendered = generator._render(json.loads((ROOT / "plugin.json").read_text(encoding="utf-8")), ("claude", "agentskills"))
            del rendered[Path("providers/agentskills/mlx-scout/SKILL.md")]
            generator._remove_stale_inventoried_files(output_root, rendered)
            self.assertFalse(stale.exists())
            self.assertEqual("preserve me\n", handwritten.read_text(encoding="utf-8"))

    def test_stale_cleanup_fails_closed_for_modified_generated_file(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("claude", "agentskills"), output_root)
            stale = output_root / "providers" / "agentskills" / "mlx-scout" / "SKILL.md"
            stale.write_text("user edit\n", encoding="utf-8")
            rendered = generator._render(json.loads((ROOT / "plugin.json").read_text(encoding="utf-8")), ("claude", "agentskills"))
            del rendered[Path("providers/agentskills/mlx-scout/SKILL.md")]
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                generator._remove_stale_inventoried_files(output_root, rendered)
            self.assertEqual("user edit\n", stale.read_text(encoding="utf-8"))

    def test_stale_cleanup_refuses_symlinked_surface_ancestor_without_touching_external_files(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            generator.generate(("claude", "agentskills"), output_root)
            providers = output_root / "providers"
            installed_surface = providers / "agentskills"
            external_surface = Path(directory) / "external-agentskills"
            installed_surface.replace(external_surface)
            os.symlink(str(external_surface), str(installed_surface))
            stale = external_surface / "mlx-scout" / "SKILL.md"
            original = stale.read_bytes()
            original_render = generator._render

            def render_without_stale_skill(manifest, provider_ids):
                rendered = original_render(manifest, provider_ids)
                del rendered[Path("providers/agentskills/mlx-scout/SKILL.md")]
                return rendered

            generator._render = render_without_stale_skill
            try:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    generator.generate(("claude", "agentskills"), output_root)
            finally:
                generator._render = original_render
            self.assertEqual(original, stale.read_bytes())

    def test_generation_refuses_surface_swap_after_preflight_without_external_write_or_chmod(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            generator.generate(("claude", "agentskills"), output_root)
            surface = output_root / "providers" / "agentskills"
            moved_surface = Path(directory) / "moved-agentskills"
            external = Path(directory) / "external"
            external_skill = external / "mlx-scout" / "SKILL.md"
            external_skill.parent.mkdir(parents=True)
            external_skill.write_text("external sentinel\n", encoding="utf-8")
            external_skill.chmod(0o600)
            original = external_skill.read_bytes()
            original_mode = external_skill.stat().st_mode & 0o777

            def swap_surface(_parent_fd, component):
                if component == "agentskills":
                    surface.replace(moved_surface)
                    os.symlink(str(external), str(surface))

            with self.assertRaisesRegex(ValueError, "unsafe|symlink"):
                generator.generate(("claude", "agentskills"), output_root, path_race_hook=swap_surface)
            self.assertEqual(original, external_skill.read_bytes())
            self.assertEqual(original_mode, external_skill.stat().st_mode & 0o777)

    def test_check_refuses_leaf_and_ancestor_symlinks_even_when_external_bytes_match(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            generator.generate(("claude", "agentskills"), output_root)
            leaf_relative = Path("providers/claude/commands/mlx-scout.md")
            leaf = output_root / leaf_relative
            external_leaf = Path(directory) / "external-leaf.md"
            external_leaf.write_bytes(leaf.read_bytes())
            leaf.unlink()
            os.symlink(str(external_leaf), str(leaf))
            self.assertIn(leaf_relative, generator._check(("claude", "agentskills"), output_root))

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            generator.generate(("claude", "agentskills"), output_root)
            provider = output_root / "providers" / "claude"
            external_provider = Path(directory) / "external-claude"
            provider.replace(external_provider)
            os.symlink(str(external_provider), str(provider))
            artifact = Path("providers/claude/commands/mlx-scout.md")
            self.assertIn(artifact, generator._check(("claude", "agentskills"), output_root))

    def test_check_refuses_surface_swap_during_descriptor_descent(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            generator.generate(("claude", "agentskills"), output_root)
            provider = output_root / "providers" / "claude"
            external_provider = Path(directory) / "external-claude"
            swapped = [False]

            def swap_surface(_parent_fd, component):
                if component == "claude" and not swapped[0]:
                    provider.replace(external_provider)
                    os.symlink(str(external_provider), str(provider))
                    swapped[0] = True

            artifact = Path("providers/claude/commands/mlx-scout.md")
            drift = generator._check(("claude", "agentskills"), output_root, path_race_hook=swap_surface)
            self.assertTrue(swapped[0])
            self.assertIn(artifact, drift)

    def test_tampered_inventory_rejects_cross_surface_traversal_duplicates_and_root_files(self):
        generator = load_generator()
        cases = ("README.md", "../outside.md", "/absolute.md", "providers/claude/commands/mlx-scout.md")
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("claude", "agentskills"), output_root)
            readme = output_root / "README.md"
            readme.write_text("handwritten\n", encoding="utf-8")
            inventory_path = output_root / ".mlx-agent-generated-files.json"
            original = json.loads(inventory_path.read_text(encoding="utf-8"))
            for bad_path in cases:
                with self.subTest(path=bad_path):
                    tampered = copy.deepcopy(original)
                    tampered["files"].append({"path": bad_path, "sha256": hashlib.sha256(b"tampered").hexdigest()})
                    inventory_path.write_text(json.dumps(tampered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "invalid generated inventory"):
                        generator.generate(("claude", "agentskills"), output_root)
                    self.assertEqual("handwritten\n", readme.read_text(encoding="utf-8"))
            duplicate = copy.deepcopy(original)
            duplicate["files"].append(dict(duplicate["files"][0]))
            inventory_path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid generated inventory"):
                generator.generate(("claude", "agentskills"), output_root)
            malformed = copy.deepcopy(original)
            malformed["surface"] = "agentskills-package"
            malformed["files"].append({"path": "commands/mlx-scout.md", "sha256": "not-a-hash"})
            inventory_path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid generated inventory"):
                generator.generate(("claude", "agentskills"), output_root)

    def test_runtime_bundle_contract_is_recursive_and_copies_nested_resources(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "mlx_agent"
            (source_root / "resources" / "nested").mkdir(parents=True)
            (source_root / "__init__.py").write_text("\n", encoding="utf-8")
            (source_root / "resources" / "nested" / "fixture.txt").write_text("fixture\n", encoding="utf-8")
            (source_root / "resources" / "nested" / "fixture.bin").write_bytes(b"\x00\xffresource")
            (source_root / "__pycache__").mkdir()
            (source_root / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
            bundle = generator._runtime_bundle(Path("package"), source_root=source_root)
            self.assertIn(Path("package/src/mlx_agent/resources/nested/fixture.txt"), bundle)
            self.assertEqual(b"\x00\xffresource", bundle[Path("package/src/mlx_agent/resources/nested/fixture.bin")])
            self.assertNotIn(Path("package/src/mlx_agent/__pycache__/ignored.pyc"), bundle)

    def test_check_detects_missing_nested_non_python_runtime_resource(self):
        generator = load_generator()
        resource = ROOT / "src" / "mlx_agent" / "resources" / "adapter-runtime.json"
        self.assertTrue(resource.is_file())
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("claude", "agentskills"), output_root)
            relative = Path("src/mlx_agent/resources/adapter-runtime.json")
            copies = [
                output_root / "providers" / "claude" / relative,
                *(output_root / "providers" / "agentskills" / "mlx-{0}".format(capability) / relative for capability in ("scout", "adopt", "wire")),
            ]
            for copy in copies:
                self.assertTrue(copy.is_file(), str(copy))
            copies[0].unlink()
            self.assertIn(Path("providers/claude") / relative, generator._check(("claude", "agentskills"), output_root))

    def test_check_detects_uninventoried_stale_file_in_allowed_generated_surface(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generator.generate(("opencode",), output_root)
            stale_relative = Path("providers/opencode/opencode.json")
            stale = output_root / stale_relative
            inventory = json.loads(
                (
                    output_root
                    / "providers"
                    / "opencode"
                    / ".mlx-agent-generated-files.json"
                ).read_text(encoding="utf-8")
            )
            self.assertNotIn(
                "opencode.json",
                {entry["path"] for entry in inventory["files"]},
            )
            stale.write_text('{"stale": true}\n', encoding="utf-8")
            self.assertIn(
                stale_relative,
                generator._check(("opencode",), output_root),
            )
            generator.generate(("opencode",), output_root)
            self.assertEqual('{"stale": true}\n', stale.read_text(encoding="utf-8"))
            self.assertIn(
                stale_relative,
                generator._check(("opencode",), output_root),
            )

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
        self.assertIn("Ran 5 tests", result.stderr)

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
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
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
        encoded_roles = workflow.split("const allowedRoles = new Set(", 1)[1].split(")", 1)[0]
        expected_roles = [role["id"] for role in manifest["roles"]]
        self.assertEqual(expected_roles, json.loads(encoded_roles))
        self.assertEqual(6, len(json.loads(encoded_roles)))
        self.assertIn("tool-use", json.loads(encoded_roles))
        self.assertNotIn(
            "new Set(['general', 'coding', 'reasoning', 'vision', 'embedding'])",
            workflow,
        )

    def test_claude_workflow_role_allowlist_is_derived_from_manifest(self):
        generator = load_generator()
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        manifest["roles"] = [{"id": "manifest-role"}]
        workflow = generator._workflow(manifest)
        encoded_roles = workflow.split("const allowedRoles = new Set(", 1)[1].split(")", 1)[0]
        self.assertEqual(["manifest-role"], json.loads(encoded_roles))
