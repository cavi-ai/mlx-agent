"""The bundled DeepSeek-V4 DSpark port; runs where gguf, numpy and mlx are importable."""

import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import PORTS_DIR

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("gguf", "numpy", "mlx"))
STAGES, HIDDEN, HEADS, HEAD_DIM, Q_LORA, O_GROUPS, O_LORA = 2, 64, 2, 64, 64, 2, 64
EXPERTS, MOE, VOCAB, RANK, HC = 4, 64, 96, 64, 4


def load_port():
    sys.path.insert(0, str(PORTS_DIR / "mlx-lm"))
    try:
        return importlib.import_module("deepseek_v4_dspark.convert")
    finally:
        sys.path.pop(0)


def layer_shapes():
    """Numpy-order shapes and GGUF storage of one stage's tensors (MXFP4 experts listed apart)."""
    return {
        "hc_attn_fn.weight": ((2 + HC) * HC, HC * HIDDEN, "F32"),
        "hc_attn_base.weight": ((2 + HC) * HC, None, "F32"),
        "hc_attn_scale.weight": (3, None, "F32"),
        "hc_ffn_fn.weight": ((2 + HC) * HC, HC * HIDDEN, "F32"),
        "hc_ffn_base.weight": ((2 + HC) * HC, None, "F32"),
        "hc_ffn_scale.weight": (3, None, "F32"),
        "attn_sinks.weight": (HEADS, None, "F32"),
        "attn_q_a.weight": (Q_LORA, HIDDEN, "Q8_0"),
        "attn_q_b.weight": (HEADS * HEAD_DIM, Q_LORA, "Q8_0"),
        "attn_q_a_norm.weight": (Q_LORA, None, "F32"),
        "attn_kv.weight": (HEAD_DIM, HIDDEN, "Q8_0"),
        "attn_kv_a_norm.weight": (HEAD_DIM, None, "F32"),
        "attn_output_a.weight": (O_GROUPS * O_LORA, HEADS * HEAD_DIM // O_GROUPS, "Q8_0"),
        "attn_output_b.weight": (HIDDEN, O_GROUPS * O_LORA, "Q8_0"),
        "attn_norm.weight": (HIDDEN, None, "F32"),
        "ffn_norm.weight": (HIDDEN, None, "F32"),
        "ffn_gate_inp.weight": (EXPERTS, HIDDEN, "BF16"),
        "exp_probs_b.bias": (EXPERTS, None, "F32"),
        "ffn_gate_shexp.weight": (MOE, HIDDEN, "Q8_0"),
        "ffn_down_shexp.weight": (HIDDEN, MOE, "Q8_0"),
        "ffn_up_shexp.weight": (MOE, HIDDEN, "Q8_0"),
    }


ROOT_SHAPES = {
    "fc.weight": (HIDDEN, 3 * HIDDEN, "Q8_0"),
    "enc.output_norm.weight": (HIDDEN, None, "F32"),
    "output_norm.weight": (HIDDEN, None, "F32"),
    "output_hc_fn.weight": (HC, HC * HIDDEN, "F32"),
    "output_hc_base.weight": (HC, None, "F32"),
    "output_hc_scale.weight": (1, None, "F32"),
    "markov_w1.weight": (VOCAB, RANK, "BF16"),
    "markov_w2.weight": (VOCAB, RANK, "BF16"),
    "conf_proj.weight": (1, HIDDEN + RANK, "BF16"),
}


def mxfp4_blocks(rng, rows, columns):
    """Random GGML MXFP4 bytes: an E8M0 exponent near 2^0, then 16 bytes of E2M1 nibbles per 32 values."""
    import numpy as np

    blocks = rng.integers(0, 256, size=(rows, columns // 32, 17), dtype=np.uint8)
    blocks[..., 0] = rng.integers(120, 135, size=(rows, columns // 32), dtype=np.uint8)
    return blocks.reshape(rows, -1)


def write_dspark(path, rng, drop=(), extra_tensor=None, hyper_connections=True):
    """A small dflash GGUF with the DSpark tensor layout; returns {name: dequantized source values}."""
    import gguf
    import numpy as np
    from gguf import quants

    writer = gguf.GGUFWriter(str(path), "dflash")
    writer.add_name("DeepSeek-V4-Flash-0731")
    key = "dflash."
    writer.add_uint32(key + "block_count", STAGES)
    writer.add_uint32(key + "context_length", 1048576)
    writer.add_uint32(key + "embedding_length", HIDDEN)
    writer.add_uint32(key + "attention.head_count", HEADS)
    writer.add_uint32(key + "attention.head_count_kv", 1)
    writer.add_string(key + "rope.scaling.type", "yarn")
    writer.add_float32(key + "rope.scaling.factor", 16.0)
    writer.add_uint32(key + "rope.scaling.original_context_length", 65536)
    writer.add_float32(key + "rope.scaling.yarn_beta_fast", 32.0)
    writer.add_float32(key + "rope.scaling.yarn_beta_slow", 1.0)
    writer.add_float32(key + "rope.freq_base", 10000.0)
    writer.add_float32(key + "attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_uint32(key + "expert_count", EXPERTS)
    writer.add_uint32(key + "expert_used_count", 2)
    writer.add_uint32(key + "expert_gating_func", 4)
    writer.add_uint32(key + "attention.key_length", HEAD_DIM)
    writer.add_uint32(key + "attention.value_length", HEAD_DIM)
    writer.add_uint32(key + "rope.dimension_count", 16)
    writer.add_uint32(key + "attention.q_lora_rank", Q_LORA)
    writer.add_uint32(key + "attention.sliding_window", 128)
    writer.add_uint32(key + "expert_feed_forward_length", MOE)
    writer.add_uint32(key + "expert_shared_count", 1)
    writer.add_float32(key + "expert_weights_scale", 1.5)
    writer.add_bool(key + "expert_weights_norm", True)
    writer.add_array(key + "swiglu_clamp_exp", [10.0] * STAGES)
    writer.add_array(key + "swiglu_clamp_shexp", [10.0] * STAGES)
    writer.add_uint32(key + "attention.indexer.head_count", 4)
    writer.add_uint32(key + "attention.indexer.key_length", 16)
    writer.add_uint32(key + "attention.indexer.top_k", 8)
    writer.add_uint32(key + "attention.output_group_count", O_GROUPS)
    writer.add_uint32(key + "attention.output_lora_rank", O_LORA)
    writer.add_array(key + "attention.compress_ratios", [0] * STAGES)
    writer.add_float32(key + "attention.compress_rope_freq_base", 160000.0)
    if hyper_connections:
        writer.add_uint32(key + "hyper_connection.count", HC)
    writer.add_uint32(key + "hyper_connection.sinkhorn_iterations", 20)
    writer.add_float32(key + "hyper_connection.epsilon", 1e-6)
    writer.add_uint32(key + "block_size", 5)
    writer.add_array(key + "target_layers", [41, 42])
    writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 1)
    writer.add_uint32("tokenizer.ggml.mask_token_id", 90)

    tensors = {}
    for stage in range(STAGES):
        for name, shape in layer_shapes().items():
            tensors["blk.{0}.{1}".format(stage, name)] = shape
    tensors.update(ROOT_SHAPES)
    sources = {}
    for name, (rows, columns, storage) in tensors.items():
        if name in drop:
            continue
        shape = (rows,) if columns is None else (rows, columns)
        values = rng.standard_normal(shape).astype(np.float32)
        if storage == "F32":
            writer.add_tensor(name, values)
            sources[name] = values
            continue
        qtype = getattr(gguf.GGMLQuantizationType, storage)
        data = quants.quantize(values, qtype)
        writer.add_tensor(name, data, raw_dtype=qtype)
        sources[name] = quants.dequantize(data, qtype)
    for stage in range(STAGES):
        for name, (rows, columns) in (("ffn_gate_exps", (MOE, HIDDEN)), ("ffn_up_exps", (MOE, HIDDEN)),
                                      ("ffn_down_exps", (HIDDEN, MOE))):
            full = "blk.{0}.{1}.weight".format(stage, name)
            if full in drop:
                continue
            data = np.stack([mxfp4_blocks(rng, rows, columns) for _ in range(EXPERTS)])
            writer.add_tensor(full, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
            sources[full] = quants.dequantize(data, gguf.GGMLQuantizationType.MXFP4)
    if extra_tensor:
        writer.add_tensor(extra_tensor, np.zeros((4,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sources


@unittest.skipUnless(HAS_RUNTIME, "gguf, numpy and mlx are not all installed in this interpreter")
class DSparkPortTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.port = load_port()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rng = np.random.default_rng(7)

    def convert(self, q_bits=8, **kwargs):
        source = self.root / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
        sources = write_dspark(source, self.rng, **kwargs)
        out = self.root / "out"
        out.mkdir()
        summary = self.port.convert(source, out, q_bits, log=lambda message: None)
        return sources, out, summary

    def load(self, out):
        import mlx.core as mx

        index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weights = {}
        for shard in sorted(set(index["weight_map"].values())):
            weights.update(mx.load(str(out / shard)))
        self.assertEqual(set(weights), set(index["weight_map"]))
        return weights

    def test_repack_mxfp4_is_exact(self):
        import gguf
        import mlx.core as mx
        from gguf import quants

        blocks = mxfp4_blocks(self.rng, 8, 128)
        expected = quants.dequantize(blocks, gguf.GGMLQuantizationType.MXFP4)
        weight, scales = self.port.repack_mxfp4(blocks)
        self.assertEqual((weight.shape, scales.shape), ((8, 16), (8, 4)))
        actual = mx.dequantize(mx.array(weight), mx.array(scales), group_size=32, bits=4, mode="mxfp4")
        self.assertTrue(self.np.array_equal(self.np.array(actual.astype(mx.float32)), expected))

    def test_names_mirror_deepseeks_checkpoint(self):
        _, out, summary = self.convert()
        weights = self.load(out)
        self.assertIn("stages.0.main_proj.weight", weights)
        self.assertIn("stages.0.main_norm.weight", weights)
        for name in ("norm.weight", "hc_head_fn", "markov_head.markov_w1.weight", "confidence_head.proj.weight"):
            self.assertIn("stages.1." + name, weights)
        for name in ("attn.attn_sink", "attn.wkv.scales", "attn.kv_norm.weight", "ffn.gate.bias", "hc_ffn_fn",
                     "ffn.shared_experts.w2.biases", "ffn.experts.3.w2.scales"):
            self.assertIn("stages.0." + name, weights)
        self.assertFalse([name for name in weights if name.startswith("blk.") or "exps" in name])
        self.assertEqual(summary["tensors"], len(weights))

    def test_config_carries_the_drafter_contract(self):
        _, out, _ = self.convert(q_bits=4)
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["model_type"], "deepseek_v4_dspark")
        self.assertEqual(config["dspark_target_layer_ids"], [40, 41])
        self.assertEqual((config["dspark_block_size"], config["dspark_noise_token_id"]), (5, 90))
        self.assertEqual((config["dspark_markov_rank"], config["n_mtp_layers"]), (RANK, STAGES))
        self.assertEqual(config["dspark_target_name"], "DeepSeek-V4-Flash-0731")
        text = config["text_config"]
        self.assertEqual((text["model_type"], text["vocab_size"], text["hidden_size"]), ("deepseek_v4", VOCAB, HIDDEN))
        self.assertEqual((text["rms_norm_eps"], text["scoring_func"], text["swiglu_limit"]), (1e-06, "sqrtsoftplus", 10.0))
        self.assertEqual(text["rope_scaling"], {
            "type": "yarn", "factor": 16.0, "original_max_position_embeddings": 65536, "beta_fast": 32.0, "beta_slow": 1.0,
        })
        quantization = config["quantization"]
        self.assertEqual(quantization, config["quantization_config"])
        self.assertEqual({key: quantization[key] for key in ("group_size", "bits", "mode")},
                         {"group_size": 64, "bits": 4, "mode": "affine"})
        self.assertEqual(quantization["stages.1.ffn.experts.0.w3"], {"group_size": 32, "bits": 4, "mode": "mxfp4"})
        self.assertEqual(len([key for key in quantization if ".experts." in key]), STAGES * 3 * EXPERTS)

    def test_weights_match_the_gguf(self):
        import mlx.core as mx

        sources, out, _ = self.convert(q_bits=8)
        weights = self.load(out)
        np = self.np

        def as_numpy(array):
            return np.array(array.astype(mx.float32))

        self.assertTrue(np.array_equal(as_numpy(weights["stages.0.hc_attn_fn"]), sources["blk.0.hc_attn_fn.weight"]))
        self.assertEqual(weights["stages.0.hc_attn_fn"].dtype, mx.float32)
        self.assertTrue(np.array_equal(as_numpy(weights["stages.1.markov_head.markov_w2.weight"]),
                                       sources["markov_w2.weight"]))
        self.assertEqual(weights["stages.0.ffn.gate.weight"].dtype, mx.bfloat16)
        for stage, expert in ((0, 0), (1, 3)):
            base = "stages.{0}.ffn.experts.{1}.w2".format(stage, expert)
            actual = mx.dequantize(weights[base + ".weight"], weights[base + ".scales"], group_size=32, bits=4, mode="mxfp4")
            self.assertTrue(np.array_equal(as_numpy(actual), sources["blk.{0}.ffn_down_exps.weight".format(stage)][expert]))
        base = "stages.1.attn.wq_b"
        actual = as_numpy(mx.dequantize(weights[base + ".weight"], weights[base + ".scales"], weights[base + ".biases"],
                                        group_size=64, bits=8))
        source = sources["blk.1.attn_q_b.weight"]
        self.assertEqual(actual.shape, source.shape)
        self.assertLess(float(np.abs(actual - source).max()), 0.02 * float(np.abs(source).max()))

    def test_a_missing_tensor_is_named(self):
        with self.assertRaises(ValueError) as caught:
            self.convert(drop=("blk.1.attn_kv.weight",))
        self.assertIn("missing blk.1.attn_kv.weight", str(caught.exception))

    def test_an_unexpected_tensor_is_named(self):
        with self.assertRaises(ValueError) as caught:
            self.convert(extra_tensor="token_embd.weight")
        self.assertIn("unexpected token_embd.weight", str(caught.exception))

    def test_other_dflash_drafters_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.convert(hyper_connections=False)
        self.assertIn("not a DeepSeek-V4 DSpark drafter", str(caught.exception))

    def test_shards_split_at_the_limit(self):
        source = self.root / "dspark.gguf"
        write_dspark(source, self.rng)
        out = self.root / "sharded"
        out.mkdir()
        self.port.convert(source, out, 4, log=lambda message: None, shard_bytes=64 * 1024)
        shards = sorted(path.name for path in out.glob("*.safetensors"))
        self.assertGreater(len(shards), 1)
        self.assertEqual(shards[-1], "model-{0:05d}-of-{0:05d}.safetensors".format(len(shards)))
        self.assertFalse(list(out.glob("*.partial.safetensors")))
        self.load(out)


if __name__ == "__main__":
    unittest.main()
