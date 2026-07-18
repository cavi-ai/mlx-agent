import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_agent.providers import ProviderRegistry, _run_bounded_probe, detect_providers


ROOT = Path(__file__).resolve().parents[2]


class ProviderRegistryTests(unittest.TestCase):
    def test_manifest_definitions_resolve_user_and_project_roots_from_injected_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProviderRegistry(
                ROOT / "plugin.json", home=root / "home", config_root=root / "config"
            )
            definitions = registry.definitions()

            self.assertEqual(
                {"claude", "codex", "gemini", "opencode", "agentskills"},
                set(definitions),
            )
            for provider_id, definition in definitions.items():
                with self.subTest(provider=provider_id):
                    if provider_id in {"claude", "codex", "agentskills"}:
                        expected_root = root / "home/plugins/mlx-agent"
                        if provider_id == "claude":
                            expected_root = root / "home/.claude/plugins/mlx-agent"
                        elif provider_id == "agentskills":
                            expected_root = root / "home/.agents"
                    elif provider_id in {"gemini", "opencode"}:
                        expected_root = root / "home"
                    else:
                        expected_root = root / "config"
                    self.assertTrue(str(definition.user_root).startswith(str(expected_root.resolve())))
                    self.assertEqual(
                        (root / "project" / definition.project_root).resolve(),
                        definition.destination("project", root / "project"),
                    )
                    self.assertTrue(definition.artifacts)
                    self.assertTrue(all(item.source.is_file() for item in definition.artifacts))

    def test_detection_uses_injected_executable_lookup_without_installing_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ProviderRegistry(ROOT / "plugin.json", config_root=Path(directory))
            detections = detect_providers(
                registry.definitions().values(),
                env={"PATH": "/fake/bin"},
                executable_lookup=lambda command, path=None: "/fake/bin/" + command
                if command in {"claude", "codex"}
                else None,
                probe_runner=lambda argv, **kwargs: subprocess.CompletedProcess(
                    argv, 0, b"2.1.198\n" if argv[0].endswith("claude") else b"codex-cli 0.137.0\n"
                ),
            )

        available = {item.id for item in detections if item.available}
        self.assertEqual({"claude", "codex", "agentskills"}, available)
        self.assertTrue(all(item.command_path is None or item.command_path.startswith("/fake/bin/") for item in detections))
        states = {item.id: item.state for item in detections}
        self.assertEqual("native-visible", states["claude"])
        self.assertEqual("portable", states["agentskills"])
        self.assertEqual("absent", states["gemini"])

    def test_detection_rejects_out_of_range_versions_with_a_bounded_no_shell_probe(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, b"gemini 99.0.0\n")

        registry = ProviderRegistry(ROOT / "plugin.json")
        definition = registry.definitions()["gemini"]
        detection = detect_providers(
            [definition],
            env={"PATH": "/fake/bin"},
            executable_lookup=lambda command, path=None: "/fake/bin/gemini",
            probe_runner=runner,
        )[0]
        self.assertFalse(detection.available)
        self.assertEqual("unsupported", detection.state)
        self.assertEqual("99.0.0", detection.version)
        self.assertEqual(["/fake/bin/gemini", "--version"], calls[0][0])
        self.assertFalse(calls[0][1]["shell"])
        self.assertLessEqual(calls[0][1]["timeout"], 5)

    def test_default_version_probe_caps_captured_output_before_process_exit(self):
        completed = _run_bounded_probe(
            [sys.executable, "-c", "print('provider 1.2.3')"],
            timeout=2,
            env=dict(os.environ),
        )
        self.assertEqual(0, completed.returncode)
        self.assertIn(b"1.2.3", completed.stdout)
        with self.assertRaisesRegex(ValueError, "output exceeded"):
            _run_bounded_probe(
                [sys.executable, "-c", "import sys; sys.stdout.write('x' * 9000)"],
                timeout=2,
                env=dict(os.environ),
            )

    def test_invalid_provider_root_template_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads((ROOT / "plugin.json").read_text())
            manifest["providers"]["claude"]["user_root"] = "{project}/escape"
            path = root / "plugin.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                ProviderRegistry(path, config_root=root).definitions()

    def test_provider_artifact_sources_never_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.md"
            outside.write_text("fixture\n")
            linked = root / "linked.md"
            linked.symlink_to(outside)
            manifest = json.loads((ROOT / "plugin.json").read_text())
            manifest["providers"] = {"agentskills": manifest["providers"]["agentskills"]}
            manifest["providers"]["agentskills"]["artifacts"] = [
                {"source": "linked.md", "destination": "skills/linked.md"}
            ]
            path = root / "plugin.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "symlink"):
                ProviderRegistry(path, home=root / "home", config_root=root / "config").definitions()
