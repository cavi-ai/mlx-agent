import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class SiteCatalogTests(unittest.TestCase):
    def test_site_catalog_matches_plugin_and_provider_contracts(self):
        catalog = json.loads((ROOT / "site/catalog.json").read_text())
        plugin = json.loads((ROOT / "plugin.json").read_text())
        compatibility = json.loads((ROOT / "compatibility/providers.json").read_text())

        self.assertEqual(catalog["product"]["slug"], plugin["name"])
        self.assertEqual(catalog["release"]["version"], plugin["version"])
        self.assertEqual(set(catalog["providers"]), set(compatibility["providers"]))
        self.assertEqual(catalog["scopes"], plugin["scopes"])
        self.assertEqual(
            {capability["slug"]: capability["command"] for capability in catalog["capabilities"]},
            {slug: detail["command"] for slug, detail in plugin["capabilities"].items()},
        )
        self.assertEqual(catalog["verification_evidence"]["core"], compatibility["core_evidence"])

        for provider, details in compatibility["providers"].items():
            self.assertEqual(catalog["providers"][provider]["scopes"], details["scopes"])
            self.assertEqual(catalog["providers"][provider]["commands"], details["commands"])
            self.assertEqual(
                catalog["verification_evidence"]["provider_contracts"][provider],
                {
                    key: details[key]
                    for key in ("schema_validation", "cli_smoke", "last_tested")
                },
            )
            self.assertTrue((ROOT / catalog["providers"][provider]["install_guide"]).is_file())

    def test_site_catalog_has_schema_required_fields_and_lifecycle_commands(self):
        catalog = json.loads((ROOT / "site/catalog.json").read_text())
        schema = json.loads((ROOT / "schemas/site-catalog.schema.json").read_text())

        self.assertEqual(catalog["$schema"], "../schemas/site-catalog.schema.json")
        self.assert_required_fields(catalog, schema, "catalog")
        for section in ("product", "release", "links", "requirements", "lifecycle_commands", "verification_evidence"):
            self.assert_required_fields(catalog[section], schema["properties"][section], section)
        self.assert_required_fields(catalog["providers"], schema["properties"]["providers"], "providers")

        expected_lifecycle_keys = {"install", "verify", "update", "uninstall"}
        self.assertEqual(set(catalog["lifecycle_commands"]), expected_lifecycle_keys)
        install_docs = (ROOT / "docs/install/index.md").read_text()
        for command in catalog["lifecycle_commands"].values():
            self.assertIn(command, install_docs)

    def assert_required_fields(self, value, definition, name):
        missing = set(definition.get("required", [])) - set(value)
        self.assertFalse(missing, f"{name} is missing schema-required fields: {sorted(missing)}")
