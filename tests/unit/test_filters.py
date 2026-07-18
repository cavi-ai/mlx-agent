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
            "metadata_available": True,
            "tree_available": False,
            "gated": repo.endswith("Gated-Coder-7B-4bit"),
            "license": "apache-2.0" if "Open" in repo else "mit",
            "reasoning": False,
            "reason_src": "checked",
            "tags": ["mlx"],
        }


class UnknownMetadataHub(StubHub):
    def inspect_model(self, repo):
        value = super().inspect_model(repo)
        value.update({"metadata_available": False, "gated": None, "license": None})
        return value


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

    def test_license_allow_list_is_an_independent_filter(self):
        repos = self._repos(licenses=("apache-2.0",))
        self.assertNotIn("allowed/Gated-Coder-7B-4bit", repos)
        self.assertTrue(repos)

    def test_memory_budget_is_an_independent_filter(self):
        self.assertNotIn("allowed/Large-Coder-30B-4bit", self._repos(memory_gb=8))

    def test_publisher_allow_list_is_an_independent_filter(self):
        self.assertTrue(all(repo.startswith("allowed/") for repo in self._repos(publishers=("allowed",))))

    def test_quantization_is_an_independent_filter(self):
        self.assertTrue(all("4bit" in repo for repo in self._repos(quantization="q4")))

    def test_filters_models_incompatible_with_requested_runtime(self):
        result = self.service.discover(
            DiscoveryRequest(role="vision", runtime="mlx_lm")
        ).to_dict()
        self.assertEqual(result["data"]["roles"], {})

    def test_mlx_vlm_rejects_non_vision_candidates_to_match_wiring(self):
        result = self.service.discover(
            DiscoveryRequest(role="coding", runtime="mlx-vlm")
        ).to_dict()
        self.assertEqual(result["data"]["roles"], {})
        rejected = result["data"]["rejected"]
        self.assertIn("runtime", rejected["allowed/Open-Coder-7B-4bit"]["rejection_reasons"])

    def test_unknown_gated_status_is_rejected_when_gated_models_are_excluded(self):
        service = DiscoveryService(
            host=HostInventory(ram_gb=32, chip="Apple Test"),
            huggingface=UnknownMetadataHub(),
        )
        result = service.discover(
            DiscoveryRequest(role="coding", include_gated=False)
        ).to_dict()
        candidate = result["data"]["rejected"]["allowed/Open-Coder-7B-4bit"]
        self.assertEqual(candidate["facts"]["gated"], "unknown")
        self.assertIn("gated_status_unknown", candidate["rejection_reasons"])

    def test_provenance_cites_only_evidence_that_was_available(self):
        class ProvenanceHub(StubHub):
            def inspect_model(self, repo):
                value = super().inspect_model(repo)
                value.update({"weight_bytes": 4_000_000_000, "tree_available": True})
                return value

        full = DiscoveryService(host=HostInventory(ram_gb=32), huggingface=ProvenanceHub()).discover(
            DiscoveryRequest(role="coding", limit=1)
        ).to_dict()["data"]["roles"]["coding"][0]
        facts = {field: record for record in full["provenance"] for field in record["fields"]}
        self.assertIn("/tree/main?recursive=true", facts["weight_bytes"]["url"])
        self.assertIn("/api/models/{0}".format(full["repo"].replace("/", "%2F")), facts["license"]["url"])
        self.assertEqual(facts["role"]["source"], "local_name_derivation")

        fast = self.service.discover(DiscoveryRequest(role="coding", fast=True, limit=1)).to_dict()["data"]["roles"]["coding"][0]
        self.assertFalse(any("/api/models/allowed%2F" in record.get("url", "") for record in fast["provenance"]))

    def test_tied_evidence_uses_repository_id_as_a_deterministic_final_sort_key(self):
        class TieHub(StubHub):
            def list_models(self, sort="trendingScore"):
                return [
                    {"id": "same/Zed-Coder-7B-4bit", "downloads": 1, "likes": 1},
                    {"id": "same/Alpha-Coder-7B-4bit", "downloads": 1, "likes": 1},
                ]

        service = DiscoveryService(host=HostInventory(ram_gb=32), huggingface=TieHub())
        result = service.discover(DiscoveryRequest(role="coding", limit=2)).to_dict()
        self.assertEqual([item["repo"] for item in result["data"]["roles"]["coding"]], [
            "same/Zed-Coder-7B-4bit", "same/Alpha-Coder-7B-4bit",
        ])

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
