import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.task_contracts import TaskError, capabilities


class MediaSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = Path(temporary.name).resolve()

    def describe(self, config):
        (self.model / "config.json").write_text(json.dumps(config))
        return capabilities(model_path=str(self.model))

    def test_model_capabilities_bind_config_without_loading_weights(self):
        result = self.describe({"model_type": "triposg"})
        self.assertEqual(result["model"], {
            "path": str(self.model),
            "config_sha256": hashlib.sha256((self.model / "config.json").read_bytes()).hexdigest(),
        })
        tasks = {item["kind"]: item for item in result["tasks"]}
        cache = tasks["mesh_generation"]["denoiser_cache"]
        self.assertEqual(cache["modes"], ["off", "adaptive"])
        self.assertEqual(cache["default"], "off")
        self.assertTrue(cache["approximate"])
        self.assertEqual(cache["cache_memory_mib"], {"minimum": 8, "maximum": 4096, "default": 256})
        self.assertEqual(tasks["video_generation"]["denoiser_cache"]["modes"], ["off"])
        self.assertNotIn("ready", result)
        self.assertEqual({path.name for path in self.model.iterdir()}, {"config.json"})

    def test_only_single_model_wan21_and_triposg_offer_adaptive_reuse(self):
        for config, supported in [
            ({"model_type": "t2v", "model_version": "2.1"}, True),
            ({"model_type": "t2v", "model_version": "2.1", "dual_model": True}, False),
            ({"model_type": "t2v", "model_version": "2.2"}, False),
            ({"model_type": "i2v", "model_version": "2.1"}, False),
            ({"model_type": "t2v"}, False),
            ({"model_type": "pixal3d"}, False),
        ]:
            with self.subTest(config=config):
                tasks = {item["kind"]: item for item in self.describe(config)["tasks"]}
                descriptor = tasks["video_generation"]["denoiser_cache"]
                self.assertEqual("adaptive" in descriptor["modes"], supported)
                if not supported:
                    self.assertTrue(descriptor["unavailable_reason"])
                self.assertEqual(tasks["mesh_generation"]["denoiser_cache"]["modes"], ["off"])

    def test_unscoped_capabilities_remain_unchanged_and_config_changes_are_visible(self):
        self.assertNotIn("model", capabilities())
        self.assertTrue(all("denoiser_cache" not in item for item in capabilities()["tasks"]))
        before = self.describe({"model_type": "triposg"})
        after = self.describe({"model_type": "pixal3d"})
        self.assertNotEqual(before["model"], after["model"])

    def test_config_links_are_read_as_metadata_and_keep_the_model_binding(self):
        source = self.model / "source.json"
        source.write_text('{"model_type":"triposg"}')
        (self.model / "config.json").symlink_to(source)
        result = capabilities(model_path=str(self.model))
        self.assertEqual(result["model"]["path"], str(self.model))
        self.assertEqual(result["model"]["config_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_bad_or_unbounded_configs_fail_without_opening_special_files(self):
        for content in ("[]", "invalid", " " * (1024 * 1024 + 1)):
            (self.model / "config.json").write_text(content)
            with self.subTest(content=content[:8]), self.assertRaises(TaskError):
                capabilities(model_path=str(self.model))
        (self.model / "config.json").unlink()
        os.mkfifo(self.model / "config.json")
        with self.assertRaises(TaskError):
            capabilities(model_path=str(self.model))
        with self.assertRaises(TaskError):
            capabilities(model_path="relative")
