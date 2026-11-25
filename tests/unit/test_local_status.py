import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent import gguf, local_status
from mlx_agent.cli import main
from mlx_agent.discovery import DiscoveryRequest, DiscoveryService
from mlx_agent.host import HostInventory
from mlx_agent.huggingface import HuggingFaceClient
from tests.unit.test_discovery_ranking import RecordingHub
from tests.unit.test_gguf import write_mlx_model, write_ollama_model


QWEN_4BIT = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
OTHER = "mlx-community/Llama-Other-8B-4bit"


class LocalLibrary(unittest.TestCase):
    """A home folder with LM Studio downloads, a Hugging Face cache, a converter output and an Ollama store."""

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        home = patch.dict(os.environ, {"HOME": str(self.home)})
        home.start()
        self.addCleanup(home.stop)
        self.library = self.home / "llm"
        self.lmstudio = self.library / "lmstudio"
        (self.home / ".lmstudio").mkdir()
        (self.home / ".lmstudio" / "settings.json").write_text(json.dumps({"downloadsFolder": str(self.lmstudio)}))
        self.eight = write_mlx_model(self.lmstudio / "lmstudio-community" / "Qwen3.8-27B-MLX-8bit", bits=8, weights=b"8" * 8)
        self.six = write_mlx_model(self.lmstudio / "lmstudio-community" / "Qwen3.8-27B-MLX-6bit", bits=6, weights=b"6" * 6)
        self.hub = self.home / ".cache" / "huggingface" / "hub"
        self.cached = write_mlx_model(self.hub / "models--mlx-community--Gemma-Nine-4bit" / "snapshots" / "abc", hidden_size=2048)
        self.converted = write_mlx_model(self.library / "converted" / "Phi-Mini-MLX-4bit", hidden_size=512)
        (self.converted / "mlx-converter.json").write_text(json.dumps({"source": {"kind": "hf", "repo": "microsoft/Phi-Mini"}}))
        config = {"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28}
        self.ollama = write_ollama_model(
            self.library / "ollama", "qwen3.8", "27b-mlx",
            [("json", json.dumps(config).encode("utf-8"), "config.json"), ("tensor", b"tensor", "lm_head.weight")],
            {"model_format": "safetensors", "file_type": "nvfp4"},
        )

    def records(self):
        return local_status.scan([str(self.library), str(self.hub)], signature=False)["local_models"]


class LocalModelsTests(LocalLibrary):
    def test_each_local_model_reports_where_it_lives_how_it_arrived_and_its_task(self):
        records = {record["path"]: record for record in self.records()}
        fields = ("store", "repo", "acquired", "converted_from", "model_key", "quantization")
        self.assertEqual(
            {path: tuple(records[str(path)][field] for field in fields) + (records[str(path)]["task"]["type"],)
             for path in (self.eight, self.cached, self.converted, self.ollama)},
            {
                self.eight: ("lmstudio", "lmstudio-community/Qwen3.8-27B-MLX-8bit", "downloaded", None,
                             "qwen3-8-27b", "8-bit", "text_llm"),
                self.cached: ("huggingface", "mlx-community/Gemma-Nine-4bit", "cached", None,
                              "gemma-nine", "4-bit", "text_llm"),
                self.converted: ("folder", None, "converted", "microsoft/Phi-Mini", "phi-mini", "4-bit", "text_llm"),
                self.ollama: ("ollama", None, "downloaded", None, "qwen3-8-27b", "nvfp4", "text_llm"),
            },
        )
        variant = records[str(self.eight)]["duplicate_groups"]
        self.assertEqual(len(variant), 1)
        self.assertEqual([path for path, record in sorted(records.items()) if variant[0] in record["duplicate_groups"]],
                         sorted([str(self.ollama), str(self.six), str(self.eight)]))

    def test_default_roots_include_the_lm_studio_downloads_folder(self):
        self.assertIn(str(self.lmstudio), gguf.default_gguf_roots())

    def test_a_repository_is_exact_variant_or_absent(self):
        records = self.records()
        statuses = {
            repo: (found["status"], found["basis"], sorted(copy["path"] for copy in found["copies"]))
            for repo, found in (
                ("lmstudio-community/Qwen3.8-27B-MLX-8bit", local_status.match(records, "lmstudio-community/Qwen3.8-27B-MLX-8bit")),
                ("mlx-community/Qwen3.8-27B-8bit", local_status.match(records, "mlx-community/Qwen3.8-27B-8bit", weight_bytes=8)),
                (QWEN_4BIT, local_status.match(records, QWEN_4BIT)),
                (OTHER, local_status.match(records, OTHER)),
            )
        }
        copies = sorted([str(self.eight), str(self.six), str(self.ollama)])
        self.assertEqual(statuses, {
            "lmstudio-community/Qwen3.8-27B-MLX-8bit": ("exact", "repository", copies),
            "mlx-community/Qwen3.8-27B-8bit": ("exact", "weight_bytes", copies),
            QWEN_4BIT: ("variant", "model_key", copies),
            OTHER: ("absent", None, []),
        })


def discover(hub, **request):
    service = DiscoveryService(host=HostInventory(ram_gb=64), huggingface=HuggingFaceClient(http_get=hub))
    result = service.discover(DiscoveryRequest(**request)).to_dict()
    assert result["status"] == "ok", result
    return result["data"]


def listing():
    return [{"id": QWEN_4BIT, "downloads": 9000, "likes": 1}, {"id": OTHER, "downloads": 100, "likes": 1}]


class DiscoveryLocalTests(LocalLibrary):
    def test_candidates_carry_their_local_status(self):
        data = discover(RecordingHub(listing()), role="general", limit=2, local_models=self.records())
        self.assertEqual({item["repo"]: item["local"]["status"] for item in data["roles"]["general"]},
                         {QWEN_4BIT: "variant", OTHER: "absent"})

    def test_exclude_local_leaves_local_models_out_before_ranking(self):
        hub = RecordingHub(listing())
        data = discover(hub, role="general", limit=1, local_models=self.records(), exclude_local=True)
        self.assertEqual([item["repo"] for item in data["roles"]["general"]], [OTHER])
        self.assertEqual([(item["repo"], item["status"]) for item in data["local"]["excluded"]], [(QWEN_4BIT, "variant")])
        self.assertNotIn(QWEN_4BIT, hub.inspected())

    def test_cached_exclusions_change_when_local_models_become_drafters(self):
        hub = RecordingHub(listing())
        service = DiscoveryService(host=HostInventory(ram_gb=64), huggingface=HuggingFaceClient(http_get=hub),
                                   state_dir=self.home / "state")
        records = self.records()
        request = DiscoveryRequest(role="general", limit=1, local_models=records, exclude_local=True)
        first = service.discover(request).to_dict()["data"]
        self.assertEqual(first["roles"]["general"][0]["repo"], OTHER)

        for record in records:
            if record["model_key"] == gguf.name_key(QWEN_4BIT):
                record["task"] = {"type": "speculative_draft"}
        second = service.discover(request).to_dict()["data"]
        self.assertEqual(second["roles"]["general"][0]["repo"], QWEN_4BIT)

    def test_discover_command_reads_local_folders_and_excludes_them(self):
        fixture = self.home / "fixture.json"
        fixture.write_text(json.dumps({
            "host": {"ram_gb": 64, "chip": None, "ollama": False, "lmstudio": True},
            "models": listing(), "details": {}, "trees": {},
        }))
        out = io.StringIO()
        with patch.dict(os.environ, {"MLX_AGENT_FIXTURE": str(fixture)}), redirect_stdout(out):
            code = main(["discover", "--role", "general", "--limit", "1", "--exclude-local",
                         "--gguf-root", str(self.library), "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())["data"]
        self.assertEqual([item["repo"] for item in data["roles"]["general"]], [OTHER])
        self.assertEqual(data["roles"]["general"][0]["local"]["status"], "absent")
        self.assertEqual([item["repo"] for item in data["local"]["excluded"]], [QWEN_4BIT])

    def test_discover_command_shows_local_status_in_markdown(self):
        fixture = self.home / "fixture.json"
        fixture.write_text(json.dumps({
            "host": {"ram_gb": 64, "chip": None, "ollama": False, "lmstudio": True},
            "models": listing(), "details": {}, "trees": {},
        }))
        out = io.StringIO()
        with patch.dict(os.environ, {"MLX_AGENT_FIXTURE": str(fixture)}), redirect_stdout(out):
            code = main(["discover", "--role", "general", "--limit", "2", "--local-status",
                         "--gguf-root", str(self.library)])
        self.assertEqual(code, 0)
        rows = out.getvalue().splitlines()
        self.assertTrue(next(row for row in rows if QWEN_4BIT in row).endswith("| variant |"))
        self.assertTrue(next(row for row in rows if OTHER in row).endswith("| absent |"))


class EmptyLocalLibraryTests(unittest.TestCase):
    def test_first_discovery_with_an_empty_home_reports_models_as_absent(self):
        with TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.json"
            fixture.write_text(json.dumps({
                "host": {"ram_gb": 64, "chip": None, "ollama": False, "lmstudio": False},
                "models": listing(), "details": {}, "trees": {},
            }))
            out = io.StringIO()
            with patch.dict(os.environ, {"HOME": directory, "MLX_AGENT_FIXTURE": str(fixture)}), redirect_stdout(out):
                code = main(["discover", "--role", "general", "--limit", "1", "--exclude-local", "--json"])
            response = json.loads(out.getvalue())
            self.assertEqual(code, 0, response)
            self.assertEqual(response["data"]["local"]["models"], 0)
            self.assertEqual(response["data"]["roles"]["general"][0]["local"]["status"], "absent")


class TaskRoleTests(unittest.TestCase):
    def test_each_task_role_lists_hugging_face_by_its_pipeline_tag(self):
        cases = {
            "speech-to-text": ("automatic-speech-recognition", "mlx-community/whisper-large-v3-turbo"),
            "text-to-speech": ("text-to-speech", "mlx-community/Kokoro-82M-bf16"),
            "image-generation": ("text-to-image", "mlx-community/FLUX.2-klein-4bit"),
            "video-generation": ("text-to-video", "mlx-community/LTX-Video-2B-bf16"),
            "mesh-generation": ("image-to-3d", "mlx-community/TripoSG-MLX"),
            "music-generation": ("text-to-audio", "mlx-community/musicgen-small"),
        }
        for role, (tag, repo) in cases.items():
            with self.subTest(role=role):
                hub = RecordingHub([{"id": repo, "pipeline_tag": tag, "downloads": 10, "likes": 1}])
                data = discover(hub, role=role, limit=2)
                self.assertEqual(hub.list_query()["pipeline_tag"], [tag])
                candidates = data["roles"].get(role, [])
                self.assertEqual([item["repo"] for item in candidates], [repo])
                self.assertIn("mlx-agent task capabilities", candidates[0]["wiring"])

    def test_an_unfiltered_listing_places_models_by_pipeline_tag(self):
        rows = [
            ("mlx-community/Kokoro-82M-bf16", "text-to-speech"),
            ("mlx-community/musicgen-small", "text-to-audio"),
            ("mlx-community/FLUX.2-klein-4bit", "text-to-image"),
            ("mlx-community/openai-privacy-filter-bf16", "token-classification"),
            ("mlx-community/Chat-8B-4bit", "text-generation"),
        ]
        hub = RecordingHub([{"id": repo, "pipeline_tag": tag, "downloads": 10, "likes": 1} for repo, tag in rows])
        data = discover(hub, limit=3)
        self.assertEqual({role: [item["repo"] for item in items] for role, items in data["roles"].items()}, {
            "text-to-speech": ["mlx-community/Kokoro-82M-bf16"],
            "music-generation": ["mlx-community/musicgen-small"],
            "image-generation": ["mlx-community/FLUX.2-klein-4bit"],
            "general": ["mlx-community/Chat-8B-4bit"],
        })


if __name__ == "__main__":
    unittest.main()
