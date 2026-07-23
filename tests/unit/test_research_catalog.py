"""Phase 3: adapters, datasets, and dataset blueprint in research packs."""

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.blueprint import build_dataset_blueprint
from mlx_agent.contracts import ResultEnvelope
from mlx_agent.interview import DomainIntent
from mlx_agent.research import (
    CatalogItem,
    ResearchPack,
    generate_pack,
    hub_row_metadata,
    render_pack,
    write_pack,
)


def _candidate(repo, role, downloads=100, license="mit", est_ram=4.0):
    return {
        "repo": repo,
        "role": role,
        "roles": [role],
        "downloads": downloads,
        "likes": 3,
        "est_ram_gb": est_ram,
        "license": license,
        "wiring": "`mlx_lm.server --model {0}`".format(repo),
        "facts": {"repository": repo, "downloads": downloads, "likes": 3},
        "tags": [],
    }


class FakeDiscovery:
    def __init__(self, buckets):
        self._buckets = buckets

    def discover(self, request):
        data = {
            "host": {"chip": "M3", "ram_gb": 32, "ollama": True, "lmstudio": True},
            "fast": False,
            "roles": {request.role: self._buckets.get(request.role, [])},
        }
        return ResultEnvelope.ok("discover", data)


class FakeCatalogHF:
    def __init__(self, cards=None, adapters=None, datasets=None, dataset_cards=None):
        self._cards = cards or {}
        self._adapters = adapters if adapters is not None else []
        self._datasets = datasets if datasets is not None else []
        self._dataset_cards = dataset_cards or {}
        self.adapter_searches = []
        self.dataset_searches = []

    def fetch_model_card(self, repo, timeout=8):
        return self._cards.get(repo)

    def fetch_dataset_card(self, repo, timeout=8):
        return self._dataset_cards.get(repo)

    def list_adapters(self, search="", limit_fetch=20, timeout=10):
        self.adapter_searches.append(search)
        return list(self._adapters)[:limit_fetch]

    def list_datasets(self, search="", limit_fetch=20, timeout=10):
        self.dataset_searches.append(search)
        return list(self._datasets)[:limit_fetch]


def _hub_row(repo_id, downloads=50, tags=None, license="mit"):
    return {
        "id": repo_id,
        "downloads": downloads,
        "likes": 2,
        "tags": tags or [],
        "license": license,
    }


class HubRowMetadataTests(unittest.TestCase):
    def test_maps_hub_row_with_intent_roles_and_no_memory(self):
        intent = DomainIntent(domain="legal", roles=("vision", "general"))
        metadata = hub_row_metadata(_hub_row("org/adapter", tags=["peft", "lora"]), intent)
        self.assertEqual(metadata["roles"], ["vision", "general"])
        self.assertIsNone(metadata["est_ram_gb"])
        self.assertEqual(metadata["downloads"], 50)
        self.assertIn("peft", metadata["tags"])


class CatalogEnrichmentTests(unittest.TestCase):
    def _intent(self):
        return DomainIntent(
            domain="legal",
            roles=("vision",),
            keywords=("ocr", "contract"),
            license_allow=("mit",),
        )

    def test_ranks_adapters_and_datasets(self):
        adapters = [
            _hub_row("low/adapter", downloads=1, tags=["peft"]),
            _hub_row("high/adapter", downloads=100000, tags=["peft", "ocr"]),
        ]
        datasets = [
            _hub_row("low/data", downloads=2, tags=["dataset"]),
            _hub_row("high/data", downloads=90000, tags=["ocr", "contract"]),
        ]
        pack = generate_pack(
            self._intent(),
            FakeDiscovery({"vision": [_candidate("base/model", "vision")]}),
            FakeCatalogHF(
                cards={"base/model": "OCR model. Usage."},
                adapters=adapters,
                datasets=datasets,
                dataset_cards={
                    "high/data": "Legal OCR contracts dataset. Usage examples.",
                    "low/data": "misc",
                },
            ),
            now=datetime(2026, 7, 23, tzinfo=timezone.utc),
            catalog_limit=1,
        )
        self.assertEqual(len(pack.adapters), 1)
        self.assertEqual(pack.adapters[0].repo, "high/adapter")
        self.assertEqual(pack.adapters[0].kind, "adapter")
        self.assertEqual(len(pack.datasets), 1)
        self.assertEqual(pack.datasets[0].repo, "high/data")
        self.assertEqual(pack.datasets[0].kind, "dataset")
        self.assertIsNone(pack.dataset_blueprint)

    def test_empty_datasets_yields_blueprint(self):
        pack = generate_pack(
            self._intent(),
            FakeDiscovery({"vision": []}),
            FakeCatalogHF(adapters=[_hub_row("a/lora", tags=["peft"])], datasets=[]),
            now=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )
        self.assertEqual(pack.datasets, ())
        self.assertIsNotNone(pack.dataset_blueprint)
        self.assertIn("legal", pack.dataset_blueprint.goal.lower())

    def test_nonempty_datasets_omit_blueprint(self):
        pack = generate_pack(
            self._intent(),
            FakeDiscovery({"vision": []}),
            FakeCatalogHF(
                datasets=[_hub_row("org/legal-ocr", tags=["ocr"])],
                dataset_cards={"org/legal-ocr": "OCR contracts"},
            ),
            now=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )
        self.assertEqual(len(pack.datasets), 1)
        self.assertIsNone(pack.dataset_blueprint)

    def test_search_uses_intent_keywords(self):
        hf = FakeCatalogHF()
        generate_pack(
            self._intent(),
            FakeDiscovery({"vision": []}),
            hf,
            now=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )
        self.assertTrue(hf.adapter_searches)
        self.assertTrue(hf.dataset_searches)
        self.assertIn("ocr", hf.adapter_searches[0])
        self.assertIn("contract", hf.dataset_searches[0])


class BlueprintTests(unittest.TestCase):
    def test_deterministic_sections(self):
        intent = DomainIntent(
            domain="Legal OCR",
            roles=("vision",),
            keywords=("ocr", "contract"),
            license_allow=("apache-2.0",),
            notes="PII redaction required",
        )
        first = build_dataset_blueprint(intent)
        second = build_dataset_blueprint(intent)
        self.assertEqual(first.to_dict(), second.to_dict())
        text = first.to_markdown()
        for heading in (
            "Goal",
            "Suggested schema",
            "Labeling notes",
            "Train / val split",
            "License and privacy",
            "MLX fine-tune next steps",
        ):
            self.assertIn(heading, text)
        self.assertIn("Legal OCR", text)
        self.assertIn("ocr", text)
        self.assertIn("apache-2.0", text)
        self.assertIn("PII", text)


class RenderCatalogTests(unittest.TestCase):
    def test_headings_for_adapters_datasets_blueprint(self):
        pack = generate_pack(
            DomainIntent(domain="Legal", roles=("vision",), keywords=("ocr",)),
            FakeDiscovery({"vision": []}),
            FakeCatalogHF(
                adapters=[_hub_row("org/lora", tags=["peft", "ocr"])],
                datasets=[],
            ),
            now=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )
        markdown = render_pack(pack)
        self.assertIn("## Adapters / LoRAs", markdown)
        self.assertIn("## Datasets", markdown)
        self.assertIn("## Dataset blueprint", markdown)
        self.assertIn("`org/lora`", markdown)

    def test_blueprint_section_absent_when_datasets_found(self):
        pack = generate_pack(
            DomainIntent(domain="Legal", roles=("vision",), keywords=("ocr",)),
            FakeDiscovery({"vision": []}),
            FakeCatalogHF(
                datasets=[_hub_row("org/data", tags=["ocr"])],
                dataset_cards={"org/data": "OCR data"},
            ),
            now=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )
        markdown = render_pack(pack)
        self.assertIn("## Datasets", markdown)
        self.assertNotIn("## Dataset blueprint", markdown)


class WritePackSidecarTests(unittest.TestCase):
    def test_writes_json_sidecar_with_catalog_fields(self):
        intent = DomainIntent(domain="Legal", roles=("vision",), keywords=("ocr",))
        pack = generate_pack(
            intent,
            FakeDiscovery({"vision": [_candidate("a/m", "vision")]}),
            FakeCatalogHF(
                cards={"a/m": "OCR. Usage."},
                adapters=[_hub_row("a/lora", tags=["peft"])],
                datasets=[],
            ),
            now=datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc),
        )
        with TemporaryDirectory() as root:
            path = write_pack(
                render_pack(pack),
                intent,
                root=root,
                now=datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc),
                pack=pack,
            )
            sidecar = path.with_suffix(".json")
            self.assertTrue(sidecar.is_file())
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertIn("intent", payload)
            self.assertIn("candidates", payload)
            self.assertIn("adapters", payload)
            self.assertIn("datasets", payload)
            self.assertIn("dataset_blueprint", payload)
            self.assertIsInstance(payload["adapters"], list)
            self.assertIsNotNone(payload["dataset_blueprint"])


class CatalogItemShapeTests(unittest.TestCase):
    def test_catalog_item_to_dict(self):
        item = CatalogItem(
            repo="org/x",
            kind="adapter",
            score=42.5,
            signals=(),
            provenance=(),
            card_excerpt="hello",
            record={"id": "org/x"},
        )
        payload = item.to_dict()
        self.assertEqual(payload["kind"], "adapter")
        self.assertEqual(payload["repo"], "org/x")
        self.assertEqual(payload["record"]["id"], "org/x")


if __name__ == "__main__":
    unittest.main()
