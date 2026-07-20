import json
import unittest
from copy import deepcopy
from pathlib import Path

from tests.unit.site_catalog_schema import SchemaValidationError, validate_site_catalog

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PRODUCT_SLUG = "mlx-agent"
CANONICAL_PRODUCT_NAME = "MLX Agent"
CANONICAL_RELEASE_VERSION = "0.2.0"
CANONICAL_REPOSITORY_URL = "https://github.com/cavi-ai/mlx-agent"


class SiteCatalogTests(unittest.TestCase):
    def test_site_catalog_matches_plugin_and_provider_contracts(self):
        catalog = json.loads((ROOT / "site/catalog.json").read_text())
        plugin = json.loads((ROOT / "plugin.json").read_text())
        compatibility = json.loads((ROOT / "compatibility/providers.json").read_text())

        self.assertEqual(catalog["product"]["slug"], CANONICAL_PRODUCT_SLUG)
        self.assertEqual(plugin["name"], CANONICAL_PRODUCT_SLUG)
        self.assertEqual(catalog["release"]["version"], CANONICAL_RELEASE_VERSION)
        self.assertEqual(plugin["version"], CANONICAL_RELEASE_VERSION)
        self.assertEqual(set(catalog["providers"]), set(compatibility["providers"]))
        self.assertEqual(catalog["scopes"], plugin["scopes"])
        catalog_capabilities = catalog["capabilities"]
        catalog_slugs = [capability["slug"] for capability in catalog_capabilities]
        self.assertEqual(len(catalog_slugs), len(set(catalog_slugs)))
        self.assertEqual(
            sorted((capability["slug"], capability["command"]) for capability in catalog_capabilities),
            sorted((slug, detail["command"]) for slug, detail in plugin["capabilities"].items()),
        )
        self.assertEqual(catalog["verification_evidence"]["core"], compatibility["core_evidence"])

        for provider, details in compatibility["providers"].items():
            self.assertEqual(catalog["providers"][provider]["minimum_version"], details["minimum_version"])
            self.assertEqual(catalog["providers"][provider]["last_tested_version"], details["last_tested_version"])
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

    def test_site_catalog_schema_is_canonical_and_lifecycle_commands_are_documented(self):
        catalog = json.loads((ROOT / "site/catalog.json").read_text())
        schema = json.loads((ROOT / "schemas/site-catalog.schema.json").read_text())

        self.assertEqual(catalog["$schema"], "../schemas/site-catalog.schema.json")
        self.assertEqual(schema["$id"], f"{CANONICAL_REPOSITORY_URL}/schemas/site-catalog.schema.json")
        self.assertEqual(schema["properties"]["product"]["properties"]["slug"]["const"], CANONICAL_PRODUCT_SLUG)
        self.assertEqual(schema["properties"]["product"]["properties"]["name"]["const"], CANONICAL_PRODUCT_NAME)
        self.assertEqual(schema["properties"]["release"]["properties"]["version"]["const"], CANONICAL_RELEASE_VERSION)
        self.assertEqual(schema["properties"]["links"]["properties"]["repository"]["const"], CANONICAL_REPOSITORY_URL)
        self.assertEqual(catalog["links"]["repository"], CANONICAL_REPOSITORY_URL)

        expected_lifecycle_keys = {"install", "verify", "update", "uninstall"}
        self.assertEqual(set(catalog["lifecycle_commands"]), expected_lifecycle_keys)
        install_docs = (ROOT / "docs/install/index.md").read_text()
        for command in catalog["lifecycle_commands"].values():
            self.assertIn(command, install_docs)

    def test_site_catalog_schema_validation_rejects_invalid_semantics(self):
        catalog = json.loads((ROOT / "site/catalog.json").read_text())
        schema = json.loads((ROOT / "schemas/site-catalog.schema.json").read_text())

        validate_site_catalog(catalog, schema)

        invalid_catalogs = {
            "invalid type": self.with_value(catalog, ("product", "description"), 42),
            "invalid enum": self.with_value(catalog, ("scopes", 0), "workspace"),
            "invalid const": self.with_value(catalog, ("release", "channel"), "stable"),
            "additional property": self.with_value(catalog, ("unexpected",), True),
        }
        for name, invalid_catalog in invalid_catalogs.items():
            with self.subTest(name=name):
                with self.assertRaises(SchemaValidationError):
                    validate_site_catalog(invalid_catalog, schema)

    def test_site_catalog_schema_rejects_duplicate_capability_slugs(self):
        catalog = json.loads((ROOT / "site/catalog.json").read_text())
        schema = json.loads((ROOT / "schemas/site-catalog.schema.json").read_text())
        duplicate_catalog = self.with_value(catalog, ("capabilities", 2, "slug"), "scout")

        slugs = [capability["slug"] for capability in duplicate_catalog["capabilities"]]
        self.assertNotEqual(len(slugs), len(set(slugs)))
        with self.assertRaises(SchemaValidationError):
            validate_site_catalog(duplicate_catalog, schema)

    def with_value(self, catalog, path, value):
        result = deepcopy(catalog)
        target = result
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        return result
