"""Tool-use parity contracts for every generated provider distribution."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "generate_adapters.py"
PROVIDERS = ("claude", "codex", "gemini", "opencode", "agentskills")


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_adapters", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory_paths(surface):
    inventory = json.loads(
        (surface / ".mlx-agent-generated-files.json").read_text(encoding="utf-8")
    )
    return tuple(Path(entry["path"]) for entry in inventory["files"])


def assert_tool_use_guidance(test_case, path, canonical_guidance):
    content = path.read_text(encoding="utf-8")
    test_case.assertEqual(1, content.count(canonical_guidance), str(path))


class ToolUseGeneratedParityTests(unittest.TestCase):
    def test_targeted_scout_adopt_and_advisor_artifacts_carry_tool_use_policy(self):
        generator = load_generator()
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        canonical_guidance = generator._tool_use_guidance(manifest)
        tool_use = next(role for role in manifest["roles"] if role["id"] == "tool-use")
        self.assertIn(tool_use["description"], canonical_guidance)
        self.assertIn(
            "membership is {0}".format(tool_use["membership"]),
            canonical_guidance,
        )
        self.assertIn(
            "recommendation minimum is {0}".format(
                tool_use["recommendation_minimum"]
            ),
            canonical_guidance,
        )
        self.assertIn("automatic model downloads are disabled", canonical_guidance)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator.generate(PROVIDERS, root)
            root_targets = (
                Path("commands/mlx-scout.md"),
                Path("commands/mlx-adopt.md"),
                Path("agents/mlx-advisor.md"),
            )
            root_inventory = set(inventory_paths(root))
            for relative in root_targets:
                with self.subTest(provider="root-claude-compat", path=str(relative)):
                    self.assertIn(relative, root_inventory)
                    assert_tool_use_guidance(
                        self, root / relative, canonical_guidance
                    )
            targets = {
                "claude": (
                    "commands/mlx-scout.md",
                    "commands/mlx-adopt.md",
                    "agents/mlx-advisor.md",
                ),
                "codex": (
                    "skills/mlx-scout/SKILL.md",
                    "skills/mlx-adopt/SKILL.md",
                ),
                "gemini": (
                    "commands/mlx-scout.toml",
                    "commands/mlx-adopt.toml",
                    "skills/mlx-scout/SKILL.md",
                    "skills/mlx-adopt/SKILL.md",
                ),
                "opencode": (
                    "commands/mlx-scout.md",
                    "commands/mlx-adopt.md",
                    "skills/mlx-scout/SKILL.md",
                    "skills/mlx-adopt/SKILL.md",
                    "agents/mlx-advisor.md",
                ),
                "agentskills": (
                    "mlx-scout/SKILL.md",
                    "mlx-adopt/SKILL.md",
                ),
            }
            for provider, relevant_paths in targets.items():
                surface = root / "providers" / provider
                inventoried = set(inventory_paths(surface))
                for relative in relevant_paths:
                    with self.subTest(provider=provider, path=relative):
                        self.assertIn(Path(relative), inventoried)
                        assert_tool_use_guidance(
                            self, surface / relative, canonical_guidance
                        )

    def test_canonical_guidance_is_rendered_from_manifest_role_metadata(self):
        generator = load_generator()
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        modified = json.loads(json.dumps(manifest))
        tool_use = next(
            role for role in modified["roles"] if role["id"] == "tool-use"
        )
        tool_use["description"] = "Manifest-derived tool-use description."
        tool_use["membership"] = "manifest-membership"
        tool_use["recommendation_minimum"] = "manifest-minimum"
        guidance = generator._tool_use_guidance(modified)
        self.assertIn(tool_use["description"], guidance)
        self.assertIn("membership is manifest-membership", guidance)
        self.assertIn("recommendation minimum is manifest-minimum", guidance)

    def test_bundled_adoption_and_cli_parser_match_canonical_tool_use_support(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator.generate(PROVIDERS, root)
            canonical_adoption = (ROOT / "src" / "mlx_agent" / "adoption.py").read_bytes()
            for provider in PROVIDERS:
                surface = root / "providers" / provider
                cli_paths = [
                    path
                    for path in inventory_paths(surface)
                    if path.parts[-2:] == ("mlx_agent", "cli.py")
                ]
                self.assertTrue(cli_paths, provider)
                for cli_path in cli_paths:
                    src_index = cli_path.parts.index("src")
                    bundle_root = surface.joinpath(*cli_path.parts[:src_index])
                    with self.subTest(provider=provider, bundle=str(bundle_root)):
                        adoption = bundle_root / "src" / "mlx_agent" / "adoption.py"
                        self.assertEqual(canonical_adoption, adoption.read_bytes())
                        self.assertIn("tool-use", adoption.read_text(encoding="utf-8"))
                        completed = subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                (
                                    "from mlx_agent.cli import build_parser; "
                                    "value = build_parser().parse_args("
                                    "['adopt', 'start', '--state', 'state.json', "
                                    "'--role', 'tool-use']); "
                                    "assert value.roles == ['tool-use']"
                                ),
                            ],
                            cwd=str(bundle_root),
                            env=dict(
                                os.environ,
                                PYTHONPATH=str(bundle_root / "src"),
                            ),
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_drift_checker_detects_stale_tool_use_artifact(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator.generate(PROVIDERS, root)
            stale = root / "providers" / "codex" / "skills" / "mlx-adopt" / "SKILL.md"
            stale.write_text("stale\n", encoding="utf-8")
            self.assertIn(
                stale.relative_to(root),
                generator._check(PROVIDERS, root),
            )


if __name__ == "__main__":
    unittest.main()
