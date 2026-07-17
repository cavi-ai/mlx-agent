import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_agent.discovery import DiscoveryRequest, DiscoveryService
from mlx_agent.host import HostInventory


class CountingHub:
    def __init__(self):
        self.calls = 0

    def list_models(self, sort="trendingScore"):
        self.calls += 1
        return [{"id": "mlx-community/Cache-Coder-7B-4bit", "downloads": 1, "likes": 1}]

    def inspect_model(self, repo):
        return {"weight_bytes": None, "params_total": None, "gated": False, "license": "apache-2.0", "reasoning": False, "reason_src": "checked", "tags": ["mlx"]}


class DiscoveryCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.hub = CountingHub()
        self.host = HostInventory(ram_gb=32, chip="Apple Test")

    def _service(self, hub=None):
        return DiscoveryService(host=self.host, huggingface=hub or self.hub, state_dir=self.directory.name)

    def test_fresh_cache_is_reused_and_refresh_bypasses_it(self):
        request = DiscoveryRequest(role="coding")
        first = self._service().discover(request).to_dict()
        second = self._service().discover(request).to_dict()
        refreshed = self._service().discover(DiscoveryRequest(role="coding", refresh=True)).to_dict()
        cached = self._service().discover(DiscoveryRequest(role="coding", offline=True)).to_dict()
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertEqual(self.hub.calls, 2)
        self.assertEqual(cached["data"]["cache"]["status"], "fresh")
        self.assertEqual(refreshed["data"]["cache"]["status"], "refreshed")

    def test_offline_cache_miss_has_stable_error_code(self):
        result = self._service().discover(DiscoveryRequest(role="coding", offline=True)).to_dict()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "offline_cache_missing")

    def test_offline_stale_cache_is_returned_with_warning(self):
        request = DiscoveryRequest(role="coding")
        self._service().discover(request)
        cache_path = next(Path(self.directory.name).glob("*.json"))
        payload = json.loads(cache_path.read_text())
        payload["fetched_at"] = "2000-01-01T00:00:00+00:00"
        cache_path.write_text(json.dumps(payload))
        result = self._service().discover(DiscoveryRequest(role="coding", offline=True)).to_dict()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["warnings"][0]["code"], "stale_cache")

    def test_equivalent_quantization_policies_share_one_cache_entry(self):
        self._service().discover(DiscoveryRequest(role="coding", quantization="q8"))
        result = self._service().discover(DiscoveryRequest(role="coding", quantization="8bit")).to_dict()
        self.assertEqual(self.hub.calls, 1)
        self.assertEqual(result["data"]["cache"]["status"], "fresh")

    def test_refresh_replaces_cache_atomically(self):
        request = DiscoveryRequest(role="coding")
        self._service().discover(request)
        cache_path = next(Path(self.directory.name).glob("*.json"))
        initial = cache_path.read_text()
        with patch("mlx_agent.discovery.os.replace", wraps=os.replace) as replace:
            self._service().discover(DiscoveryRequest(role="coding", refresh=True))
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(Path(replace.call_args[0][1]), cache_path)
        self.assertNotEqual(cache_path.read_text(), initial)
        self.assertEqual(list(Path(self.directory.name).glob("*.tmp")), [])
