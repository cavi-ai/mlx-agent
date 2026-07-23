"""Fixture HTTP stubs keep research catalog enrichment offline."""

import unittest

from mlx_agent.cli import _fixture_http_get


class FixtureCatalogHttpTests(unittest.TestCase):
    def test_datasets_api_returns_fixture_datasets(self):
        getter = _fixture_http_get({
            "models": [{"id": "m/x"}],
            "adapters": [{"id": "a/lora"}],
            "datasets": [{"id": "d/legal"}],
            "trees": {},
            "details": {},
            "host": {},
        })
        rows = getter("https://huggingface.co/api/datasets?search=ocr&limit=20")
        self.assertEqual(rows, [{"id": "d/legal"}])

    def test_peft_filter_returns_fixture_adapters(self):
        getter = _fixture_http_get({
            "models": [{"id": "m/x"}],
            "adapters": [{"id": "a/lora"}],
            "datasets": [],
            "trees": {},
            "details": {},
            "host": {},
        })
        rows = getter(
            "https://huggingface.co/api/models?filter=peft&search=ocr&limit=20"
        )
        self.assertEqual(rows, [{"id": "a/lora"}])

    def test_missing_catalog_keys_default_to_empty(self):
        getter = _fixture_http_get({
            "models": [{"id": "m/x"}],
            "trees": {},
            "details": {},
            "host": {},
        })
        self.assertEqual(
            getter("https://huggingface.co/api/datasets?limit=5"),
            [],
        )
        self.assertEqual(
            getter("https://huggingface.co/api/models?filter=peft&limit=5"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
