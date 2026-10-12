"""Converted Wan training inputs: local validation and real MLX loader checks."""
import importlib.util
import json
import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

PORT = Path(__file__).resolve().parents[2] / "src/mlx_agent/resources/ports/mlx-video"
sys.path.insert(0, str(PORT))
HAS_MLX_VIDEO = importlib.util.find_spec("mlx_video") is not None


class WanTrainingInputTests(unittest.TestCase):
    def setUp(self):
        import importlib
        self.inputs = importlib.import_module("t2v.training_inputs")
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "tokenizer").mkdir()
        (self.root / "tokenizer/tokenizer_config.json").write_text("{}")
        self.config = {"model_type": "t2v", "model_version": "2.1", "dual_model": False,
                       "in_dim": 16, "out_dim": 16, "vae_z_dim": 16}
        self.write_config()
        for name in ("model.safetensors", "t5_encoder.safetensors", "vae.safetensors"):
            (self.root / name).write_bytes(b"placeholder")

    def write_config(self):
        (self.root / "config.json").write_text(json.dumps(self.config))

    def test_complete_local_sources_have_content_identities(self):
        config, identities = self.inputs.training_sources(self.root)
        self.assertEqual(config, self.config)
        self.assertEqual(set(identities), {"config.json", "model.safetensors",
                         "t5_encoder.safetensors", "vae.safetensors",
                         "tokenizer/tokenizer_config.json"})
        old = identities["model.safetensors"]
        (self.root / "model.safetensors").write_bytes(b"different")
        self.assertNotEqual(self.inputs.training_sources(self.root)[1]["model.safetensors"], old)

    def test_unsupported_models_are_rejected_before_loading(self):
        for key, value in (("dual_model", True), ("model_version", "2.2"),
                           ("model_type", "i2v"), ("out_dim", 48), ("vae_z_dim", 48)):
            original = self.config[key]
            self.config[key] = value
            self.write_config()
            with self.assertRaises(ValueError):
                self.inputs.training_sources(self.root)
            self.config[key] = original

    def test_converted_large_tokenizer_is_accepted(self):
        (self.root / "tokenizer/tokenizer.json").write_bytes(b"x" * (5 * 1024 * 1024))
        self.assertIn("tokenizer/tokenizer.json", self.inputs.training_sources(self.root)[1])

    def test_missing_and_linked_sources_are_rejected(self):
        target = self.root / "model.safetensors"
        target.unlink()
        with self.assertRaises(ValueError):
            self.inputs.training_sources(self.root)
        target.symlink_to(self.root / "vae.safetensors")
        with self.assertRaises(ValueError):
            self.inputs.training_sources(self.root)

    def test_tokenizer_is_required_and_links_are_rejected(self):
        (self.root / "tokenizer/tokenizer_config.json").unlink()
        with self.assertRaises(ValueError):
            self.inputs.training_sources(self.root)
        (self.root / "tokenizer/tokenizer_config.json").symlink_to(self.root / "config.json")
        with self.assertRaises(ValueError):
            self.inputs.training_sources(self.root)

    def test_media_shape_is_never_silently_cropped_or_resized(self):
        for shape in ((0, 64, 64, 3), (6, 64, 64, 3), (17, 63, 64, 3),
                      (17, 64, 64, 4), (245, 64, 64, 3), (17, 2048, 2048, 3)):
            with self.assertRaises(ValueError):
                self.inputs.validate_video_shape(shape, {"patch_size": [1, 2, 2],
                                                       "vae_stride": [4, 8, 8]})
        self.inputs.validate_video_shape((17, 64, 64, 3),
                                         {"patch_size": [1, 2, 2], "vae_stride": [4, 8, 8]})

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Requires numpy for decoded frames")
    def test_incomplete_video_decode_is_rejected(self):
        import numpy as np
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frames = iter([frame] * 5)
        metadata = {1: 17, 2: 64, 3: 64}
        reader = SimpleNamespace(isOpened=lambda: True, get=lambda key: metadata[key],
                    read=lambda: (True, next(frames)) if frames.__length_hint__() else (False, None),
                    release=lambda: None)
        decoder = SimpleNamespace(VideoCapture=lambda path: reader, CAP_PROP_FRAME_COUNT=1,
                    CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3, COLOR_BGR2RGB=4,
                    cvtColor=lambda value, code: value)
        with mock.patch.dict(sys.modules, {"cv2": decoder}):
            with self.assertRaises(ValueError):
                self.inputs._decode_video(self.root / "clip.mp4", {})

    def test_invalid_caption_seed_or_sigma_rejected_before_runtime_imports(self):
        for caption, seed, sigma in (("", 42, .5), ("ok", True, .5),
                                     ("ok", -1, .5), ("ok", 42, float("nan")),
                                     ("ok", 42, 1.1)):
            with self.assertRaises(ValueError):
                self.inputs.prepare_training_batch(self.root, self.root / "absent.mp4",
                                                    caption, seed=seed, sigma=sigma)


@unittest.skipUnless(HAS_MLX_VIDEO, "Requires the optional MLX video backend")
class RealWanLoaderTests(WanTrainingInputTests):
    def converted_backbone(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_video.models.wan.config import WanModelConfig
        from mlx_video.models.wan.model import WanModel
        config = WanModelConfig(model_version="2.1", dual_model=False, dim=128,
                  ffn_dim=256, num_heads=2, num_layers=1, text_dim=32,
                  text_len=8, freq_dim=32)
        self.config = asdict(config)
        self.config["quantization"] = {"bits": 4, "group_size": 64}
        self.write_config()
        mx.random.seed(7)
        model = WanModel(config)
        from mlx_video.convert_wan import _quantize_predicate
        nn.quantize(model, bits=4, group_size=64,
                    class_predicate=lambda path, layer: _quantize_predicate(path, layer))
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(self.root / "model.safetensors", weights)
        return weights

    def test_loaded_converted_backbone_can_optimize_and_restore(self):
        import mlx.core as mx
        self.converted_backbone()
        session = self.inputs.load_training_session(self.root,
                  targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3)
        batch = (mx.zeros((1, 16, 1, 4, 4)),
                 mx.random.normal((1, 16, 1, 4, 4), key=mx.random.key(11)),
                 mx.zeros((1, 4, 32)), mx.array([.5]))
        self.assertGreater(session.train_step(*batch)["gradient_norm"], 0)
        checkpoint = session.checkpoint_bytes()
        expected = session.train_step(*batch)
        restored = self.inputs.load_training_session(self.root,
                  targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3)
        restored.restore_checkpoint(checkpoint)
        self.assertAlmostEqual(restored.train_step(*batch)["loss"], expected["loss"], places=6)

    def test_missing_backbone_tensor_never_uses_random_initialization(self):
        import mlx.core as mx
        weights = self.converted_backbone()
        weights.pop(next(iter(weights)))
        mx.save_safetensors(self.root / "model.safetensors", weights)
        with self.assertRaises(ValueError):
            self.inputs.load_training_session(self.root,
                  targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3)
