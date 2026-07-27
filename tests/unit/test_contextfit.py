import unittest

from mlx_agent.contextfit import (
    extract_architecture,
    kv_bytes_per_token,
    kv_cache_bytes,
    max_context_tokens,
)


QWEN3_32B_CONFIG = {
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 5120,
    "max_position_embeddings": 40960,
}


class ExtractArchitectureTests(unittest.TestCase):
    def test_gqa_config(self):
        arch = extract_architecture(QWEN3_32B_CONFIG)
        self.assertEqual(arch["layers"], 64)
        self.assertEqual(arch["kv_heads"], 8)
        self.assertEqual(arch["head_dim"], 80)
        self.assertEqual(arch["max_position_embeddings"], 40960)

    def test_mha_fallback_uses_attention_heads(self):
        config = dict(QWEN3_32B_CONFIG)
        del config["num_key_value_heads"]
        arch = extract_architecture(config)
        self.assertEqual(arch["kv_heads"], 64)

    def test_explicit_head_dim_wins(self):
        config = dict(QWEN3_32B_CONFIG, head_dim=128)
        arch = extract_architecture(config)
        self.assertEqual(arch["head_dim"], 128)

    def test_missing_layers_returns_none(self):
        config = dict(QWEN3_32B_CONFIG)
        del config["num_hidden_layers"]
        self.assertIsNone(extract_architecture(config))

    def test_non_dict_returns_none(self):
        self.assertIsNone(extract_architecture("nope"))
        self.assertIsNone(extract_architecture(None))

    def test_out_of_range_values_return_none(self):
        config = dict(QWEN3_32B_CONFIG, num_hidden_layers=-3)
        self.assertIsNone(extract_architecture(config))


class KvMathTests(unittest.TestCase):
    def test_kv_bytes_per_token_gqa(self):
        arch = extract_architecture(QWEN3_32B_CONFIG)
        self.assertEqual(kv_bytes_per_token(arch), 2 * 64 * 8 * 80 * 2)

    def test_kv_cache_bytes_scales_with_context(self):
        arch = extract_architecture(QWEN3_32B_CONFIG)
        per_token = kv_bytes_per_token(arch)
        self.assertEqual(kv_cache_bytes(arch, 32768), per_token * 32768)

    def test_context_bounds(self):
        arch = extract_architecture(QWEN3_32B_CONFIG)
        with self.assertRaises(ValueError):
            kv_cache_bytes(arch, 8)
        with self.assertRaises(ValueError):
            kv_cache_bytes(arch, 2 ** 30)

    def test_max_context_respects_budget(self):
        arch = extract_architecture(QWEN3_32B_CONFIG)
        weights = 20e9
        budget = 25e9
        maximum = max_context_tokens(arch, weights, budget)
        per_token = kv_bytes_per_token(arch)
        self.assertEqual(maximum, min(int(5e9 // per_token), 40960))

    def test_max_context_zero_when_weights_exceed_budget(self):
        arch = extract_architecture(QWEN3_32B_CONFIG)
        self.assertEqual(max_context_tokens(arch, 30e9, 25e9), 0)

    def test_max_context_unknown_without_arch(self):
        self.assertIsNone(max_context_tokens(None, 1e9, 2e9))


if __name__ == "__main__":
    unittest.main()
