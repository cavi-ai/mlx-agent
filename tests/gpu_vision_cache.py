"""Real MLX-VLM embeddings in an isolated native process."""
import unittest
from unittest.mock import patch
import importlib

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_agent._vision_cache import apply


class VisionEmbeddingTests(unittest.TestCase):
    family = "qwen3_vl"

    def model(self):
        mx.random.seed(23)
        config_module = importlib.import_module("mlx_vlm.models."+self.family+".config")
        Model = importlib.import_module("mlx_vlm.models."+self.family+"."+self.family).Model
        text = dict(model_type=self.family, hidden_size=128, num_hidden_layers=4,
            intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            rms_norm_eps=1e-6, vocab_size=64, rope_theta=1000000., max_position_embeddings=1280,
            rope_scaling={"type": "mrope", "mrope_section": [4, 6, 6]})
        if self.family == "qwen3_vl_moe":
            text.update(num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
                        mlp_only_layers=[], moe_intermediate_size=64)
        config = config_module.ModelConfig(model_type=self.family, image_token_id=11, video_token_id=13,
            vision_start_token_id=10, vision_end_token_id=12,
            text_config=config_module.TextConfig(**text),
            vision_config=config_module.VisionConfig(depth=3, hidden_size=64, intermediate_size=128,
                out_hidden_size=128, num_heads=4, patch_size=16, spatial_patch_size=16,
                num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2]))
        model = Model(config)
        model.eval()
        mx.eval(model.parameters())
        return model

    def inputs(self):
        return mx.array([[3, 10, 11, 11, 11, 11, 12, 4]]), mx.random.normal((16, 1536)), mx.array([[1, 4, 4]])

    def equal(self, expected, actual):
        def equal(a, b):
            if a is None or b is None:
                self.assertIs(a, b)
            elif isinstance(a, (tuple, list)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b): equal(x, y)
            else:
                self.assertTrue(mx.array_equal(a, b).item())
        for key in ("inputs_embeds", "visual_pos_masks", "deepstack_visual_embeds", "position_ids", "rope_deltas"):
            equal(getattr(expected, key), getattr(actual, key))
        self.assertIsNotNone(actual.deepstack_visual_embeds)

    def test_repeat_reuses_encoder_and_rebuilds_current_token_and_rope_outputs(self):
        model = self.model()
        ids, pixels, grid = self.inputs()
        expected = model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        changed_ids = mx.array([[5, 10, 11, 11, 11, 11, 12, 6]])
        changed_expected = model.get_input_embeddings(changed_ids, pixels, image_grid_thw=grid)
        original = type(model.vision_tower).__call__
        tower_class = type(model.vision_tower)
        calls = []
        def counted(owner, *args, **kwargs):
            calls.append(1)
            return original(owner, *args, **kwargs)
        self.assertTrue(apply(model))
        with patch.object(tower_class, "__call__", counted):
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
        actual = model.get_input_embeddings(text)
        self.assertTrue(mx.array_equal(text_expected.inputs_embeds, actual.inputs_embeds).item())
        self.assertIsNone(actual.deepstack_visual_embeds)
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
        tower_class = type(model.vision_tower)
        self.assertTrue(apply(model))
        with patch.object(tower_class, "__call__", side_effect=RuntimeError("encoder failed")):
            with self.assertRaises(RuntimeError):
                model.get_input_embeddings(ids, a, image_grid_thw=grid)
        self.assertEqual(model._mlx_agent_vision_cache.bytes, 0)
        self.equal(first, model.get_input_embeddings(ids, a, image_grid_thw=grid))
        self.equal(second, model.get_input_embeddings(ids, b, image_grid_thw=grid))
        self.assertEqual(model._mlx_agent_vision_cache.misses, 3)

    def test_weight_paths_and_storage_are_preserved_and_all_outputs_are_budgeted(self):
        model = self.model()
        ids, pixels, grid = self.inputs()
        weights = dict(tree_flatten(model.parameters()))
        outputs = model.vision_tower(pixels, grid)
        size = sum(v.nbytes for _, v in tree_flatten(outputs))
        self.assertGreater(size, outputs[0].nbytes)
        self.assertTrue(apply(model))
        adapted = dict(tree_flatten(model.parameters()))
        self.assertEqual(weights.keys(), adapted.keys())
        for name in weights: self.assertIs(weights[name], adapted[name])
        model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        self.assertEqual(model._mlx_agent_vision_cache.bytes, size)

    def test_bfloat16_encoder_outputs_are_reused_without_losing_deepstack(self):
        model = self.model()
        from mlx.utils import tree_map
        model.update(tree_map(lambda x: x.astype(mx.bfloat16), model.parameters()))
        ids, pixels, grid = self.inputs()
        expected = model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        self.assertTrue(apply(model))
        self.equal(expected, model.get_input_embeddings(ids, pixels, image_grid_thw=grid))
        self.equal(expected, model.get_input_embeddings(ids, pixels, image_grid_thw=grid))
        self.assertEqual(model._mlx_agent_vision_cache.encodings, 1)


class MoEVisionEmbeddingTests(VisionEmbeddingTests):
    family = "qwen3_vl_moe"


if __name__ == "__main__":
    unittest.main()
