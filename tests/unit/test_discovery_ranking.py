import threading
import unittest
import urllib.parse

from mlx_agent.discovery import DiscoveryRequest, DiscoveryService
from mlx_agent.host import HostInventory
from mlx_agent.huggingface import HuggingFaceClient


TOOL_TEMPLATE = "{% if tools %}<tool_call>{% endif %}"


class RecordingHub:
    """A Hugging Face transport that records every request that leaves."""

    def __init__(self, listing, tool_repos=()):
        self.listing = listing
        self.tool_repos = set(tool_repos)
        self.urls = []
        self._lock = threading.Lock()

    def __call__(self, url, timeout=10.0):
        with self._lock:
            self.urls.append(url)
        path = urllib.parse.urlsplit(url).path
        if path.endswith("/api/models"):
            return self.listing
        if "/tree/" in path:
            return [{"path": "model.safetensors", "size": 4_000_000_000}]
        repo = urllib.parse.unquote(path.split("/api/models/", 1)[1])
        template = TOOL_TEMPLATE if repo in self.tool_repos else ""
        return {"tags": ["mlx"], "config": {"tokenizer_config": {"chat_template": template}}}

    def inspected(self):
        return {
            urllib.parse.unquote(urllib.parse.urlsplit(url).path.split("/api/models/", 1)[1])
            for url in self.urls
            if "/api/models/" in url and "/tree/" not in url
        }

    def list_query(self):
        listed = [url for url in self.urls if urllib.parse.urlsplit(url).path.endswith("/api/models")]
        return urllib.parse.parse_qs(urllib.parse.urlsplit(listed[0]).query)


def discover(hub, **request):
    service = DiscoveryService(host=HostInventory(ram_gb=64), huggingface=HuggingFaceClient(http_get=hub))
    result = service.discover(DiscoveryRequest(**request)).to_dict()
    assert result["status"] == "ok", result
    return result["data"]


def repos(data, role):
    return [item["repo"] for item in data["roles"].get(role, [])]


def mixed_listing():
    names = []
    for index in range(40):
        names.append("mlx-community/Coder-{0}-7B-4bit".format(index))
        names.append("someone/Chat-{0}-8B-4bit".format(index))
        names.append("mlx-community/Vision-VL-{0}-2B-4bit".format(index))
    return [{"id": name, "downloads": 10_000 - rank, "likes": 1} for rank, name in enumerate(names)]


class LazyInspectionTests(unittest.TestCase):
    def test_a_role_inspects_only_enough_repositories_to_fill_its_list(self):
        listing = [{"id": "mlx-community/Coder-{0}-7B-4bit".format(index), "downloads": 1000 - index, "likes": 1}
                   for index in range(120)]
        hub = RecordingHub(listing)

        data = discover(hub, role="coding", limit=3)

        self.assertEqual(["mlx-community/Coder-0-7B-4bit", "mlx-community/Coder-1-7B-4bit", "mlx-community/Coder-2-7B-4bit"],
                         repos(data, "coding"))
        self.assertLessEqual(len(hub.inspected()), 6)

    def test_every_role_matches_a_full_inspection_of_the_listing(self):
        listing = mixed_listing()
        # Low-ranked repositories whose chat template alone makes them tool-use candidates.
        tool_repos = ["someone/Chat-30-8B-4bit", "mlx-community/Coder-35-7B-4bit", "mlx-community/Vision-VL-39-2B-4bit"]
        full_hub, lazy_hub = RecordingHub(listing, tool_repos), RecordingHub(listing, tool_repos)

        full = discover(full_hub, limit=len(listing))
        lazy = discover(lazy_hub, limit=2)

        for role in set(full["roles"]) | set(lazy["roles"]):
            self.assertEqual(repos(full, role)[:2], repos(lazy, role), role)
        self.assertEqual(["mlx-community/Coder-35-7B-4bit", "mlx-community/Vision-VL-39-2B-4bit"], repos(lazy, "tool-use"))
        self.assertLess(len(lazy_hub.inspected()), len(full_hub.inspected()))


class LensTests(unittest.TestCase):
    def test_downloads_lens_keeps_the_hugging_face_order_over_publisher_trust(self):
        listing = [
            {"id": "someone/Qwen3-27B-4bit", "downloads": 3_000_000, "likes": 50},
            {"id": "mlx-community/Qwen3-1B-4bit", "downloads": 900, "likes": 2},
        ]
        legacy = discover(RecordingHub(listing), limit=5)
        hub = RecordingHub(listing)

        data = discover(hub, limit=5, sort="downloads")

        self.assertEqual(["mlx-community/Qwen3-1B-4bit", "someone/Qwen3-27B-4bit"], repos(legacy, "general"))
        self.assertEqual(["someone/Qwen3-27B-4bit", "mlx-community/Qwen3-1B-4bit"], repos(data, "general"))
        self.assertEqual([0, 1], [item["list_rank"] for item in data["roles"]["general"]])
        self.assertEqual(["downloads"], hub.list_query()["sort"])

    def test_newest_lens_lists_new_uploads_from_reputable_publishers(self):
        listing = [
            {"id": "someone/Fresh-7B-4bit", "downloads": 0, "likes": 0},
            {"id": "mlx-community/Fresh-8B-4bit", "downloads": 3, "likes": 0},
        ]
        hub = RecordingHub(listing)

        data = discover(hub, limit=5, sort="newest")

        self.assertEqual(["mlx-community/Fresh-8B-4bit"], repos(data, "general"))
        self.assertEqual(["createdAt"], hub.list_query()["sort"])

    def test_gguf_only_repositories_are_left_out_of_mlx_discovery(self):
        listing = [
            {"id": "someone/gemma-4-E4B-it-i1-GGUF", "downloads": 9000, "likes": 3, "tags": ["gguf", "mlx"]},
            {"id": "mlx-community/gemma-4-E4B-it-8bit", "downloads": 50, "likes": 1, "tags": ["mlx", "safetensors", "gguf"]},
        ]

        data = discover(RecordingHub(listing), limit=5, sort="downloads")

        self.assertEqual(["mlx-community/gemma-4-E4B-it-8bit"], repos(data, "general"))

    def test_search_text_is_sent_to_hugging_face(self):
        hub = RecordingHub([{"id": "mlx-community/Qwen3-8B-4bit", "downloads": 5, "likes": 1}])

        data = discover(hub, limit=5, sort="downloads", search="  qwen3   8b ")

        self.assertEqual(["qwen3 8b"], hub.list_query()["search"])
        self.assertEqual("qwen3 8b", data["request"]["search"])

    def test_blank_search_sends_no_search(self):
        hub = RecordingHub([{"id": "mlx-community/Qwen3-8B-4bit", "downloads": 5, "likes": 1}])

        discover(hub, limit=5, search="   ")

        self.assertNotIn("search", hub.list_query())


if __name__ == "__main__":
    unittest.main()
