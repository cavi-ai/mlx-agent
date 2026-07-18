import json
import tempfile
import unittest
from pathlib import Path

from mlx_agent import __version__
from scripts.validate_contracts import validate_manifest


ROOT = Path(__file__).resolve().parents[2]


class ManifestTests(unittest.TestCase):
    def test_manifest_has_three_capabilities_and_five_native_providers(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        self.assertEqual(set(manifest["capabilities"]), {"scout", "adopt", "wire"})
        self.assertEqual(
            set(manifest["providers"]),
            {"claude", "codex", "gemini", "opencode", "agentskills"},
        )
        self.assertEqual(validate_manifest(ROOT / "plugin.json"), [])
        for provider in ("claude", "gemini", "opencode"):
            self.assertEqual(
                manifest["providers"][provider]["commands"],
                ["mlx-scout", "mlx-adopt", "mlx-wire"],
            )
        self.assertEqual(
            ["mlx-agent:mlx-scout", "mlx-agent:mlx-adopt", "mlx-agent:mlx-wire"],
            manifest["providers"]["codex"]["commands"],
        )

    def test_canonical_version_matches_every_native_provider_manifest(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        version = manifest["version"]
        self.assertEqual(__version__, version)
        self.assertEqual(
            version,
            json.loads((ROOT / "compatibility" / "providers.json").read_text())["plugin_version"],
        )
        self.assertEqual(
            version,
            json.loads((ROOT / "compatibility" / "release-evidence.json").read_text())["plugin_version"],
        )
        native_manifests = (
            ROOT / ".claude-plugin" / "plugin.json",
            ROOT / "providers" / "claude" / ".claude-plugin" / "plugin.json",
            ROOT / "providers" / "codex" / ".codex-plugin" / "plugin.json",
            ROOT / "providers" / "gemini" / "gemini-extension.json",
        )
        for path in native_manifests:
            with self.subTest(path=path):
                self.assertEqual(version, json.loads(path.read_text())["version"])
        marketplaces = (
            ROOT / ".claude-plugin" / "marketplace.json",
            ROOT / "providers" / "claude" / ".claude-plugin" / "marketplace.json",
        )
        for path in marketplaces:
            with self.subTest(path=path):
                self.assertEqual(version, json.loads(path.read_text())["plugins"][0]["version"])

    def test_manifest_validator_reports_native_command_drift(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        manifest["providers"]["codex"]["commands"] = ["mlx-agent:mlx-scout"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugin.json"
            path.write_text(json.dumps(manifest))
            errors = validate_manifest(path)
        self.assertIn(
            "providers.codex.commands must equal ['mlx-agent:mlx-scout', 'mlx-agent:mlx-adopt', 'mlx-agent:mlx-wire']",
            errors,
        )

    def test_manifest_validator_returns_errors_for_wrong_shaped_sections(self):
        original = json.loads((ROOT / "plugin.json").read_text())
        for section in ("scopes", "capabilities", "providers"):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                manifest = dict(original)
                manifest[section] = 1
                path = Path(directory) / "plugin.json"
                path.write_text(json.dumps(manifest))
                errors = validate_manifest(path)
                self.assertTrue(errors)
                self.assertIn("{0} must be an".format(section), "\n".join(errors))

    def test_manifest_validator_enforces_schema_only_constraints(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        manifest["$schema"] = 1
        manifest["unexpected"] = True
        manifest["requirements"]["unexpected"] = True
        manifest["safety"]["unexpected"] = True
        manifest["capabilities"]["scout"]["description"] = ""
        manifest["capabilities"]["scout"]["unexpected"] = True
        argument = manifest["capabilities"]["scout"]["actions"]["discover"]["arguments"][0]
        argument["name"] = ""
        argument["unexpected"] = True
        manifest["providers"]["claude"]["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugin.json"
            path.write_text(json.dumps(manifest))
            errors = validate_manifest(path)
        self.assertIn("manifest.$schema must be a string", errors)
        self.assertIn("manifest has unexpected keys: ['unexpected']", errors)
        self.assertIn("requirements has unexpected keys: ['unexpected']", errors)
        self.assertIn("safety has unexpected keys: ['unexpected']", errors)
        self.assertIn("capabilities.scout.description must not be empty", errors)
        self.assertIn("capabilities.scout has unexpected keys: ['unexpected']", errors)
        self.assertIn("capabilities.scout.actions.discover.arguments[0].name must not be empty", errors)
        self.assertIn(
            "capabilities.scout.actions.discover.arguments[0] has unexpected keys: ['unexpected']",
            errors,
        )
        self.assertIn("providers.claude has unexpected keys: ['unexpected']", errors)
