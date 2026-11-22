import hashlib
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

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


class FrontDoorProviderTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "chat", "object": "model", "owned_by": "mlx-agent"},
            {"id": "vision", "object": "model", "owned_by": "mlx-agent"},
            {"id": "mesh", "object": "model", "owned_by": "mlx-agent", "task_kinds": ["mesh_generation"]},
        ]
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status = 200 if self.headers.get("Authorization") == "Bearer local-test-key" else 401
                raw = json.dumps({"object": "list", "data": owner.rows}).encode()
                self.send_response(status)
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%s/v1" % self.server.server_port
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_authenticated_front_door_filters_tasks_and_keeps_vision(self):
        services = [dict(service(1234, "/models/vision", runtime="mlx-vlm"), jit=True,
                         api_key_sha256=hashlib.sha256(b"local-test-key").hexdigest())]
        environment = {"MLX_AGENT_FRONT_DOOR_URL": self.url, "MLX_AGENT_FRONT_DOOR_KEY": "local-test-key"}
        with patch.dict("os.environ", environment, clear=True):
            rows = provider_models(discover=lambda: {"services": services})
        self.assertEqual([(row["id"], row["runtime"]) for row in rows],
                         [("chat", "front_door"), ("vision", "mlx-vlm")])
        self.assertTrue(all(row["settings"]["baseURL"] == self.url for row in rows))

    def test_invalid_or_unauthenticated_opt_in_never_falls_back(self):
        cases = [(self.url, "wrong"), ("http://localhost:42/v1", "local-test-key"),
                 (self.url, ""), ("", "local-test-key")]
        for url, key in cases:
            environment = {"MLX_AGENT_FRONT_DOOR_URL": url, "MLX_AGENT_FRONT_DOOR_KEY": key}
            with self.subTest(url=url), patch.dict("os.environ", environment, clear=True):
                rows = provider_models(discover=lambda: {"services": [service(1234, "/models/direct")]})
                self.assertEqual(rows, [])

    def test_duplicate_names_are_omitted(self):
        self.rows = [
            {"id": "same", "object": "model", "owned_by": "mlx-agent"},
            {"id": "same", "object": "model", "owned_by": "mlx-agent"},
            {"id": "bad", "object": "model"},
        ]
        environment = {"MLX_AGENT_FRONT_DOOR_URL": self.url, "MLX_AGENT_FRONT_DOOR_KEY": "local-test-key"}
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(provider_models(discover=lambda: {"services": []}), [])


if __name__ == "__main__":
    unittest.main()
