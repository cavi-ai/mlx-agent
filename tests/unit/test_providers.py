import json
import tempfile
import unittest
from pathlib import Path

from mlx_agent.providers import ProviderRegistry, detect_providers


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
                    expected_root = root / ("home/plugins/mlx-agent" if provider_id == "codex" else "config")
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
            )

        available = {item.id for item in detections if item.available}
        self.assertEqual({"claude", "codex"}, available)
        self.assertTrue(all(item.command_path is None or item.command_path.startswith("/fake/bin/") for item in detections))

    def test_invalid_provider_root_template_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads((ROOT / "plugin.json").read_text())
            manifest["providers"]["claude"]["user_root"] = "{project}/escape"
            path = root / "plugin.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                ProviderRegistry(path, config_root=root).definitions()
