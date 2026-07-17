import unittest

from mlx_agent.discovery import DiscoveryRequest, DiscoveryService
from mlx_agent.host import HostInventory


class StubHub:
    def list_models(self, sort="trendingScore"):
        return [
            {"id": "allowed/Open-Coder-7B-4bit", "downloads": 20, "likes": 2},
            {"id": "allowed/Gated-Coder-7B-4bit", "downloads": 19, "likes": 2},
            {"id": "allowed/Large-Coder-30B-4bit", "downloads": 18, "likes": 2},
            {"id": "blocked/Open-Coder-7B-4bit", "downloads": 17, "likes": 2},
            {"id": "allowed/Open-Coder-7B-Q8", "downloads": 16, "likes": 2},
            {"id": "allowed/Open-Vision-7B-4bit", "downloads": 15, "likes": 2},
        ]

    def inspect_model(self, repo):
        return {
            "weight_bytes": None,
            "params_total": None,
            "gated": repo.endswith("Gated-Coder-7B-4bit"),
            "license": "apache-2.0" if "Open" in repo else "mit",
            "reasoning": False,
            "reason_src": "checked",
            "tags": ["mlx"],
        }


class DiscoveryFilterTests(unittest.TestCase):
    def setUp(self):
        self.service = DiscoveryService(
            host=HostInventory(ram_gb=32, chip="Apple Test", lmstudio=True),
            huggingface=StubHub(),
        )

    def _repos(self, **request):
        result = self.service.discover(DiscoveryRequest(role="coding", **request)).to_dict()
        self.assertEqual(result["status"], "ok")
        return [candidate["repo"] for candidate in result["data"]["roles"].get("coding", [])]

    def test_excludes_gated_models_unless_requested(self):
        repos = self._repos(include_gated=False)
        self.assertNotIn("allowed/Gated-Coder-7B-4bit", repos)

    def test_filters_by_license_memory_publisher_and_quantization(self):
        repos = self._repos(
            licenses=("apache-2.0",),
            memory_gb=8,
            publishers=("allowed",),
            quantization="4bit",
        )
        self.assertEqual(repos, ["allowed/Open-Coder-7B-4bit"])

    def test_filters_models_incompatible_with_requested_runtime(self):
        result = self.service.discover(
            DiscoveryRequest(role="vision", runtime="mlx_lm")
        ).to_dict()
        self.assertEqual(result["data"]["roles"], {})

    def test_candidate_explains_evidence_ranking_and_rejections(self):
        result = self.service.discover(
            DiscoveryRequest(role="coding", include_gated=False, limit=1)
        ).to_dict()
        candidate = result["data"]["roles"]["coding"][0]
        self.assertIn("downloads", candidate["facts"])
        self.assertIn("ram_gb", candidate["estimates"])
        self.assertIn("role", candidate["heuristics"])
        self.assertTrue(candidate["provenance"])
        self.assertIsInstance(candidate["rank_score"], (int, float))
        self.assertTrue(candidate["selection_reasons"])
        self.assertEqual(candidate["rejection_reasons"], [])
        self.assertIn("gated", result["data"]["rejected"]["allowed/Gated-Coder-7B-4bit"]["rejection_reasons"])
