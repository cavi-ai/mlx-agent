import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_contracts import validate_manifest


ROOT = Path(__file__).resolve().parents[2]


class ManifestTests(unittest.TestCase):
    def test_manifest_has_three_capabilities_and_four_native_providers(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        self.assertEqual(set(manifest["capabilities"]), {"scout", "adopt", "wire"})
        self.assertEqual(
            set(manifest["providers"]),
            {"claude", "codex", "gemini", "opencode", "agentskills"},
        )
        self.assertEqual(validate_manifest(ROOT / "plugin.json"), [])
        for provider in ("claude", "codex", "gemini", "opencode"):
            self.assertEqual(
                manifest["providers"][provider]["commands"],
                ["mlx-scout", "mlx-adopt", "mlx-wire"],
            )

    def test_manifest_validator_reports_native_command_drift(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        manifest["providers"]["codex"]["commands"] = ["mlx-scout"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugin.json"
            path.write_text(json.dumps(manifest))
            errors = validate_manifest(path)
        self.assertIn(
            "providers.codex.commands must equal ['mlx-scout', 'mlx-adopt', 'mlx-wire']",
            errors,
        )
