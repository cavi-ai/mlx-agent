import unittest

from mlx_agent.verification import (
    EvidenceStrength,
    LMStudioRuntimeClient,
    OllamaRuntimeClient,
    Verifier,
)


class FakeRuntimeClient:
    name = "fake-runtime"

    def __init__(self, installed, responses=None, failures=None):
        self.installed = list(installed)
        self.responses = responses or {}
        self.failures = failures or {}
        self.generated = []
        self.downloads = []

    def list_models(self):
        return list(self.installed)

    def generate(self, model, prompt, max_tokens):
        self.generated.append((model, prompt, max_tokens))
        if model in self.failures:
            raise RuntimeError(self.failures[model])
        return self.responses.get(model, {"message": {"content": "ready"}})

    def download(self, model):
        self.downloads.append(model)
        raise AssertionError("verification must never download a model")


class FakeMetadataClient:
    def __init__(self, records=None, failures=None):
        self.records = records or {}
        self.failures = failures or {}
        self.inspected = []

    def inspect_model(self, repo):
        self.inspected.append(repo)
        if repo in self.failures:
            raise RuntimeError(self.failures[repo])
        return self.records.get(repo, {"metadata_available": True, "tags": []})


class VerificationTests(unittest.TestCase):
    def test_builtin_runtime_clients_use_only_inventory_and_generation_endpoints(self):
        gets = []
        posts = []

        def http_get(url, timeout=3.0):
            gets.append((url, timeout))
            if url.endswith("/api/tags"):
                return {"models": [{"name": "local/ollama-model"}]}
            return {"data": [{"id": "local/lmstudio-model"}]}

        def http_post(url, payload, timeout=10.0):
            posts.append((url, payload, timeout))
            return {"message": {"content": "ready"}}

        ollama = OllamaRuntimeClient(http_get=http_get, http_post=http_post)
        lmstudio = LMStudioRuntimeClient(http_get=http_get, http_post=http_post)

        self.assertEqual(ollama.list_models()["models"][0]["name"], "local/ollama-model")
        self.assertEqual(lmstudio.list_models()["data"][0]["id"], "local/lmstudio-model")
        ollama.generate("local/ollama-model", "ready?", max_tokens=12)
        lmstudio.generate("local/lmstudio-model", "ready?", max_tokens=12)

        called_urls = [item[0] for item in gets + posts]
        self.assertEqual(called_urls, [
            "http://127.0.0.1:11434/api/tags",
            "http://127.0.0.1:1234/v1/models",
            "http://127.0.0.1:11434/api/chat",
            "http://127.0.0.1:1234/v1/chat/completions",
        ])
        self.assertFalse(any("pull" in url or "download" in url for url in called_urls))
        self.assertEqual(posts[0][1]["options"]["num_predict"], 12)
        self.assertEqual(posts[1][1]["max_tokens"], 12)

    def test_evidence_strength_exposes_exact_contract_names(self):
        self.assertEqual(EvidenceStrength.runtime_tested.value, "runtime_tested")
        self.assertEqual(EvidenceStrength.runtime_inventory.value, "runtime_inventory")
        self.assertEqual(EvidenceStrength.metadata_only.value, "metadata_only")
        self.assertEqual(EvidenceStrength.heuristic_only.value, "heuristic_only")

    def test_only_installed_models_are_generated_and_missing_models_use_metadata(self):
        runtime = FakeRuntimeClient(
            ["local/normal-7b"],
            responses={"local/normal-7b": {"message": {"content": "pong"}}},
        )
        metadata = FakeMetadataClient(
            {"remote/reasoner-7b": {"metadata_available": True, "tags": ["reasoning"]}}
        )
        verifier = Verifier(runtime_clients=[runtime], metadata_client=metadata)

        installed = verifier.verify(
            {"repo": "local/normal-7b", "role": "general", "reasoning": False},
            {"ram_gb": 32},
        )
        missing = verifier.verify(
            {"repo": "remote/reasoner-7b", "role": "reasoning", "reasoning": True},
            {"ram_gb": 32},
        )

        self.assertEqual(installed.strength, EvidenceStrength.RUNTIME_TESTED)
        self.assertFalse(installed.reasoning_confirmed)
        self.assertEqual(missing.strength, EvidenceStrength.METADATA_ONLY)
        self.assertFalse(missing.available_locally)
        self.assertEqual([call[0] for call in runtime.generated], ["local/normal-7b"])
        self.assertEqual(metadata.inspected, ["remote/reasoner-7b"])
        self.assertEqual(runtime.downloads, [])

    def test_hidden_reasoning_with_empty_content_is_runtime_confirmed(self):
        runtime = FakeRuntimeClient(
            ["local/thinking-7b"],
            responses={
                "local/thinking-7b": {
                    "choices": [
                        {"message": {"content": "", "reasoning_content": "working"}}
                    ]
                }
            },
        )
        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/thinking-7b", "role": "general"},
            {"ram_gb": 16},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_TESTED)
        self.assertTrue(evidence.reasoning_confirmed)
        self.assertTrue(evidence.loads)

    def test_runtime_and_metadata_failures_become_bounded_evidence(self):
        runtime = FakeRuntimeClient(
            ["local/broken-7b"], failures={"local/broken-7b": "generation failed"}
        )
        metadata = FakeMetadataClient(failures={"remote/broken-7b": "metadata failed"})
        verifier = Verifier(runtime_clients=[runtime], metadata_client=metadata)

        local = verifier.verify(
            {"repo": "local/broken-7b", "role": "coding"}, {"ram_gb": 64}
        )
        remote = verifier.verify(
            {"repo": "remote/broken-7b", "role": "coding", "reasoning": False},
            {"ram_gb": 64},
        )

        self.assertEqual(local.strength, EvidenceStrength.RUNTIME_INVENTORY)
        self.assertFalse(local.loads)
        self.assertIn("generation failed", local.note)
        self.assertEqual(remote.strength, EvidenceStrength.HEURISTIC_ONLY)
        self.assertIn("metadata failed", remote.note)
        self.assertLessEqual(len(local.note), 300)
        self.assertLessEqual(len(remote.note), 300)

    def test_offline_missing_model_uses_heuristics_without_metadata_request(self):
        runtime = FakeRuntimeClient([])
        metadata = FakeMetadataClient()
        evidence = Verifier(runtime_client=runtime, metadata_client=metadata).verify(
            {"repo": "remote/model-7b", "role": "general", "reasoning": False},
            {"ram_gb": 8},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.HEURISTIC_ONLY)
        self.assertEqual(metadata.inspected, [])
        self.assertEqual(runtime.generated, [])


if __name__ == "__main__":
    unittest.main()
