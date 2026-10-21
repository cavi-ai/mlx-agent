import threading
import time
import unittest

from mlx_agent.discovery import DiscoveryRequest, DiscoveryService
from mlx_agent.host import HostInventory
from mlx_agent.huggingface import HuggingFaceClient


REPOS = ["mlx-community/Coder-{0}-7B-4bit".format(index) for index in range(6)]


class DiscoveryConcurrencyTests(unittest.TestCase):
    def test_listed_models_are_inspected_concurrently(self):
        lock, state = threading.Lock(), {"active": 0, "peak": 0}

        def http_get(url, timeout=10.0):
            if url.split("?", 1)[0].endswith("/api/models"):
                return [{"id": repo, "downloads": 100 - index, "likes": 1} for index, repo in enumerate(REPOS)]
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.2)
            with lock:
                state["active"] -= 1
            return [] if "/tree/" in url else {"tags": ["mlx"], "config": {}}

        service = DiscoveryService(host=HostInventory(ram_gb=64), huggingface=HuggingFaceClient(http_get=http_get))
        started = time.monotonic()
        result = service.discover(DiscoveryRequest(role="coding", limit=6)).to_dict()
        elapsed = time.monotonic() - started
        self.assertEqual("ok", result["status"])
        self.assertEqual(set(REPOS), {item["repo"] for item in result["data"]["roles"]["coding"]})
        self.assertGreater(state["peak"], 1)
        # Twelve 0.2 s requests take 2.4 s one after another.
        self.assertLess(elapsed, 1.5)


if __name__ == "__main__":
    unittest.main()
