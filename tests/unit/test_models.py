import json
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from mlx_agent.discovery import DiscoveryRequest, DiscoveryService
from mlx_agent.host import HostInventory
from mlx_agent.huggingface import HuggingFaceClient


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"


def fixture_http_get(url, timeout=10.0):
    del timeout
    payload = json.loads(FIXTURE.read_text())
    if "/tree/main" in url:
        repo = url.split("/api/models/", 1)[1].split("/tree/main", 1)[0]
        return payload["trees"].get(repo, [])
    if "/api/models/" in url:
        repo = url.split("/api/models/", 1)[1]
        return payload["details"].get(repo, {})
    return payload["models"]


class DiscoveryModelTests(unittest.TestCase):
    def setUp(self):
        host = HostInventory(**json.loads(FIXTURE.read_text())["host"])
        self.service = DiscoveryService(
            host=host,
            huggingface=HuggingFaceClient(http_get=fixture_http_get),
        )

    def test_quant_variants_keep_highest_ranked_candidate_per_logical_model(self):
        result = self.service.discover(DiscoveryRequest(limit=4)).to_dict()

        coding = result["data"]["roles"]["coding"]
        self.assertEqual([item["repo"] for item in coding], [
            "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-Q8"
        ])

    def test_reasoning_detection_prefers_chat_template_over_tags_and_name(self):
        result = self.service.discover(DiscoveryRequest(role="general", limit=2)).to_dict()

        model = result["data"]["roles"]["general"][0]
        self.assertTrue(model["reasoning"])
        self.assertEqual(model["reason_src"], "chat_template")

    def test_weight_size_uses_actual_tree_bytes_when_available(self):
        result = self.service.discover(DiscoveryRequest(role="general", limit=2)).to_dict()

        model = result["data"]["roles"]["general"][0]
        self.assertEqual(model["est_ram_gb"], 4.2)
        self.assertEqual(model["ram_src"], "estimated_from_weight_bytes")
        self.assertEqual(model["facts"]["weight_bytes"], 4200000000)

    def test_discovery_data_retains_legacy_report_keys(self):
        result = self.service.discover(DiscoveryRequest(limit=2, fast=True)).to_dict()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["data"]), {"host", "fast", "roles"})

    def test_list_provenance_records_the_exact_filtered_ranked_request_url(self):
        requested = []

        def capture_http_get(url, timeout=10.0):
            del timeout
            requested.append(url)
            return [{"id": "allowed/Open-Coder-7B-4bit", "downloads": 3, "likes": 2}]

        service = DiscoveryService(
            host=HostInventory(ram_gb=32, chip="Apple Test"),
            huggingface=HuggingFaceClient(http_get=capture_http_get),
        )
        candidate = service.discover(
            DiscoveryRequest(role="coding", fast=True, new=True, limit=1)
        ).to_dict()["data"]["roles"]["coding"][0]
        source = next(
            record for record in candidate["provenance"]
            if record["source"] == "huggingface_model_list"
        )
        self.assertEqual(requested[0], source["url"])
        self.assertEqual(
            {
                "filter": ["mlx"],
                "sort": ["lastModified"],
                "direction": ["-1"],
                "limit": ["300"],
            },
            parse_qs(urlsplit(source["url"]).query),
        )
