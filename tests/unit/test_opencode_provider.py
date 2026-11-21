import unittest

from mlx_agent.inference_runner import InferenceError
from mlx_agent.opencode_provider import provider_models


def service(port, path, runtime="splash", alive=True, argv_match=True):
    return {"port": port, "path": path, "runtime": runtime, "alive": alive, "argv_match": argv_match}


class ProviderModelsTests(unittest.TestCase):
    def test_lists_each_uniquely_named_live_chat_endpoint_with_its_key(self):
        services = [
            service(8095, "/models/Qwen3.8-Flash-Next-Strata-Q2_0"),
            service(8092, "/models/MiMo-V2.6-Distill-Qwen-9B", runtime="mlx-vlm"),
            service(8091, "/models/Qwen3-Embedding-4B", runtime="tasks"),
            service(8097, "/models/Stopped-Model", alive=False),
            service(8098, "/models/Twice"),
            service(8099, "/models/Twice", runtime="mlx_lm"),
            service(8100, "/models/Locked"),
            service(8101, "/models/Open", runtime="mlx_lm"),
        ]
        keys = {8095: "k-8095", 8092: "k-8092", 8101: None}

        def key(entry):
            if entry["port"] not in keys:
                raise InferenceError("endpoint_key_unavailable", "unreadable", "run as owner")
            return keys[entry["port"]]

        models = provider_models(discover=lambda: {"services": services}, key=key)

        self.assertEqual([
            {"id": "MiMo-V2.6-Distill-Qwen-9B", "runtime": "mlx-vlm",
             "settings": {"baseURL": "http://127.0.0.1:8092/v1", "apiKey": "k-8092"}},
            {"id": "Open", "runtime": "mlx_lm", "settings": {"baseURL": "http://127.0.0.1:8101/v1"}},
            {"id": "Qwen3.8-Flash-Next-Strata-Q2_0", "runtime": "splash",
             "settings": {"baseURL": "http://127.0.0.1:8095/v1", "apiKey": "k-8095"}},
        ], models)


if __name__ == "__main__":
    unittest.main()
