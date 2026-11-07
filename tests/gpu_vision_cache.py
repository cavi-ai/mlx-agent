"""Real MLX-VLM embeddings in an isolated native process."""
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx_vlm.models.qwen2_5_vl.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen2_5_vl.qwen2_5_vl import Model
from mlx_agent._vision_cache import apply


class VisionEmbeddingTests(unittest.TestCase):
    def model(self):
        mx.random.seed(23)
        config = ModelConfig(model_type="qwen2_5_vl", image_token_id=11, video_token_id=13,
            vision_start_token_id=10, vision_end_token_id=12,
            text_config=TextConfig(model_type="qwen2_5_vl", hidden_size=128, num_hidden_layers=1,
                intermediate_size=256, num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=64,
                rope_scaling={"type": "mrope", "mrope_section": [4, 6, 6]}),
            vision_config=VisionConfig(depth=1, hidden_size=64, intermediate_size=128,
                out_hidden_size=128, num_heads=4, fullatt_block_indexes=[0], window_size=56))
        model = Model(config)
        model.eval()
        mx.eval(model.parameters())
        return model

    def inputs(self):
        return mx.array([[3, 10, 11, 11, 11, 11, 12, 4]]), mx.random.normal((16, 1176)), mx.array([[1, 4, 4]])

    def equal(self, expected, actual):
        for key in ("inputs_embeds", "position_ids", "rope_deltas"):
            self.assertTrue(mx.array_equal(getattr(expected, key), getattr(actual, key)).item(), key)

    def test_repeat_reuses_encoder_and_rebuilds_current_token_and_rope_outputs(self):
        model = self.model()
        ids, pixels, grid = self.inputs()
        expected = model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        changed_ids = mx.array([[5, 10, 11, 11, 11, 11, 12, 6]])
        changed_expected = model.get_input_embeddings(changed_ids, pixels, image_grid_thw=grid)
        original = type(model.vision_tower).__call__
        calls = []
        def counted(owner, *args, **kwargs):
            calls.append(1)
            return original(owner, *args, **kwargs)
        self.assertTrue(apply(model))
        with patch.object(type(model.vision_tower), "__call__", counted):
            self.equal(expected, model.get_input_embeddings(ids, pixels, image_grid_thw=grid))
            self.equal(changed_expected, model.get_input_embeddings(changed_ids, pixels, image_grid_thw=grid))
        self.assertEqual(len(calls), 1)
        self.assertFalse(apply(model))

    def test_changed_pixels_and_grid_do_not_reuse_stale_image_features(self):
        model = self.model()
        ids, pixels, grid = self.inputs()
        reference = model.get_input_embeddings
        cases = ((pixels, grid), (pixels+1, grid), (pixels, mx.array([[1, 2, 8]])))
        expected = [reference(ids, p, image_grid_thw=g) for p, g in cases]
        self.assertTrue(apply(model))
        for (p, g), value in zip(cases, expected):
            self.equal(value, model.get_input_embeddings(ids, p, image_grid_thw=g))
        self.assertEqual(model._mlx_agent_vision_cache.misses, 3)

    def test_oversized_features_are_not_retained_and_text_stays_native(self):
        model = self.model()
        ids, pixels, grid = self.inputs()
        expected = model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        text = mx.array([[3, 4, 5]])
        text_expected = model.get_input_embeddings(text)
        self.assertTrue(apply(model, max_bytes=1))
        self.equal(expected, model.get_input_embeddings(ids, pixels, image_grid_thw=grid))
        self.equal(expected, model.get_input_embeddings(ids, pixels, image_grid_thw=grid))
        self.equal(text_expected, model.get_input_embeddings(text))
        self.assertEqual(model._mlx_agent_vision_cache.bytes, 0)

    def test_training_bypasses_and_discards_inference_features(self):
        model = self.model()
        ids, pixels, grid = self.inputs()
        self.assertTrue(apply(model))
        model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        self.assertGreater(model._mlx_agent_vision_cache.bytes, 0)
        model.train()
        model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        self.assertEqual(model._mlx_agent_vision_cache.bytes, 0)
        model.eval()
        model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        self.assertEqual(model._mlx_agent_vision_cache.misses, 2)

    def test_image_order_and_encoder_errors_keep_native_outputs(self):
        model = self.model()
        _, pixels, grid = self.inputs()
        ids = mx.array([[3, 10, 11, 11, 11, 11, 12, 4, 10, 11, 11, 11, 11, 12, 5]])
        grid = mx.concatenate([grid, grid])
        a, b = mx.concatenate([pixels, pixels+1]), mx.concatenate([pixels+1, pixels])
        first = model.get_input_embeddings(ids, a, image_grid_thw=grid)
        second = model.get_input_embeddings(ids, b, image_grid_thw=grid)
        self.assertTrue(apply(model))
        with patch.object(type(model.vision_tower), "__call__", side_effect=RuntimeError("encoder failed")):
            with self.assertRaises(RuntimeError):
                model.get_input_embeddings(ids, a, image_grid_thw=grid)
        self.assertEqual(model._mlx_agent_vision_cache.bytes, 0)
        self.equal(first, model.get_input_embeddings(ids, a, image_grid_thw=grid))
        self.equal(second, model.get_input_embeddings(ids, b, image_grid_thw=grid))
        self.assertEqual(model._mlx_agent_vision_cache.misses, 3)


if __name__ == "__main__":
    unittest.main()
