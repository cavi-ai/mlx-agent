"""The bundled Qwen3.5 GGUF port, run through the GGUF runner; needs gguf, numpy and mlx in this interpreter."""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import gguf_runner

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("gguf", "numpy", "mlx"))
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HIDDEN, FF, HEADS, KV_HEADS, HEAD_DIM, ROTARY = 128, 256, 2, 1, 64, 16
KEY_HEADS, KEY_DIM, VALUE_HEADS, VALUE_DIM, KERNEL = 2, 64, 4, 64, 4
LAYERS, VOCAB = 4, 64
CONV = 2 * KEY_HEADS * KEY_DIM + VALUE_HEADS * VALUE_DIM
NAMED = ["h", "e", "l", "o", "he", "ll", "hell", "hello", "<|endoftext|>", "<|im_start|>", "<|im_end|>",
         "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<think>"]
TOKENS = NAMED + ["[PAD{0}]".format(index) for index in range(len(NAMED), VOCAB)]
TYPES = [1] * 8 + [3] * 7 + [4] + [5] * (VOCAB - len(NAMED))
DEPTH, VISION, VISION_FF, VISION_HEADS, PATCH, CHANNELS, POSITIONS, MERGE = 2, 64, 128, 2, 4, 3, 16, 2
MERGES = ["h e", "l l", "he ll", "hell o"]
TEMPLATE = "{% for message in messages %}<|im_start|>{{ message.content }}<|im_end|>{% endfor %}"


def checkpoint(rng):
    """Random Qwen3.5 text weights in the Hugging Face checkpoint layout, keyed by checkpoint name."""
    import numpy as np

    def tensor(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    weights = {"model.embed_tokens.weight": tensor(VOCAB, HIDDEN), "model.norm.weight": tensor(HIDDEN),
               "lm_head.weight": tensor(VOCAB, HIDDEN)}
    for layer in range(LAYERS):
        base = "model.layers.{0}.".format(layer)
        weights.update({
            base + "input_layernorm.weight": tensor(HIDDEN),
            base + "post_attention_layernorm.weight": tensor(HIDDEN),
            base + "mlp.gate_proj.weight": tensor(FF, HIDDEN),
            base + "mlp.up_proj.weight": tensor(FF, HIDDEN),
            base + "mlp.down_proj.weight": tensor(HIDDEN, FF),
        })
        if (layer + 1) % 4 == 0:
            weights.update({
                base + "self_attn.q_proj.weight": tensor(2 * HEADS * HEAD_DIM, HIDDEN),
                base + "self_attn.k_proj.weight": tensor(KV_HEADS * HEAD_DIM, HIDDEN),
                base + "self_attn.v_proj.weight": tensor(KV_HEADS * HEAD_DIM, HIDDEN),
                base + "self_attn.o_proj.weight": tensor(HIDDEN, HEADS * HEAD_DIM),
                base + "self_attn.q_norm.weight": tensor(HEAD_DIM),
                base + "self_attn.k_norm.weight": tensor(HEAD_DIM),
            })
            continue
        weights.update({
            base + "linear_attn.in_proj_qkv.weight": tensor(CONV, HIDDEN),
            base + "linear_attn.in_proj_z.weight": tensor(VALUE_HEADS * VALUE_DIM, HIDDEN),
            base + "linear_attn.in_proj_b.weight": tensor(VALUE_HEADS, HIDDEN),
            base + "linear_attn.in_proj_a.weight": tensor(VALUE_HEADS, HIDDEN),
            base + "linear_attn.out_proj.weight": tensor(HIDDEN, VALUE_HEADS * VALUE_DIM),
            base + "linear_attn.norm.weight": tensor(VALUE_DIM),
            base + "linear_attn.conv1d.weight": tensor(CONV, 1, KERNEL),
            base + "linear_attn.A_log": tensor(VALUE_HEADS),
            base + "linear_attn.dt_bias": tensor(VALUE_HEADS),
        })
    return weights


def tiled(values, axis, head_dim):
    """llama.cpp's ``_LinearAttentionVReorderBase._reorder_v_heads``: grouped-by-key-head to tiled order."""
    import numpy as np

    shape = list(values.shape)
    grouped = values.reshape(shape[:axis] + [KEY_HEADS, VALUE_HEADS // KEY_HEADS, head_dim] + shape[axis + 1:])
    return np.ascontiguousarray(np.swapaxes(grouped, axis, axis + 1)).reshape(shape)


def llama_cpp_tensors(weights):
    """What llama.cpp's Qwen3_5TextModel writes for ``weights``, keyed by GGUF name (conversion/qwen.py)."""
    import numpy as np

    key_dim = KEY_HEADS * KEY_DIM
    gguf_names = {
        "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight", "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight", "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
        "linear_attn.in_proj_z.weight": "attn_gate.weight", "linear_attn.in_proj_b.weight": "ssm_beta.weight",
        "linear_attn.in_proj_a.weight": "ssm_alpha.weight", "linear_attn.out_proj.weight": "ssm_out.weight",
        "linear_attn.norm.weight": "ssm_norm.weight", "linear_attn.conv1d.weight": "ssm_conv1d.weight",
        "linear_attn.A_log": "ssm_a", "linear_attn.dt_bias": "ssm_dt.bias",
    }
    root = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
            "lm_head.weight": "output.weight"}
    written = {}
    for name, values in weights.items():
        if name in root:
            written[root[name]] = values + 1 if name == "model.norm.weight" else values
            continue
        layer, suffix = name.split(".", 3)[2:]
        if suffix == "linear_attn.in_proj_qkv.weight":
            values = np.concatenate([values[: 2 * key_dim], tiled(values[2 * key_dim:], 0, VALUE_DIM)])
        elif suffix == "linear_attn.in_proj_z.weight":
            values = tiled(values, 0, VALUE_DIM)
        elif suffix in ("linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight", "linear_attn.dt_bias"):
            values = tiled(values, 0, 1)
        elif suffix == "linear_attn.A_log":
            values = -np.exp(tiled(values, 0, 1))
        elif suffix == "linear_attn.conv1d.weight":
            values = values.squeeze()
            values = np.concatenate([values[: 2 * key_dim], tiled(values[2 * key_dim:], 0, VALUE_DIM)])
        elif suffix == "linear_attn.out_proj.weight":
            values = tiled(values, 1, VALUE_DIM)
        elif suffix.endswith("norm.weight") and suffix != "linear_attn.norm.weight":
            values = values + 1
        written["blk.{0}.{1}".format(layer, gguf_names[suffix])] = values
    return written


def write_qwen35(path, tensors, prediction=True, tokenizer_pre="qwen35"):
    """A small qwen35 GGUF holding ``tensors`` (projections as Q8_0) plus one multi-token-prediction block."""
    import gguf
    import numpy as np
    from gguf import quants

    writer = gguf.GGUFWriter(str(path), "qwen35")
    writer.add_name("Qwen3.5 Tiny")
    key = "qwen35."
    writer.add_uint32(key + "block_count", LAYERS + 1 if prediction else LAYERS)
    writer.add_uint32(key + "nextn_predict_layers", 1 if prediction else 0)
    writer.add_uint32(key + "context_length", 4096)
    writer.add_uint32(key + "embedding_length", HIDDEN)
    writer.add_uint32(key + "feed_forward_length", FF)
    writer.add_uint32(key + "attention.head_count", HEADS)
    writer.add_uint32(key + "attention.head_count_kv", KV_HEADS)
    writer.add_uint32(key + "attention.key_length", HEAD_DIM)
    writer.add_uint32(key + "attention.value_length", HEAD_DIM)
    writer.add_float32(key + "attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32(key + "rope.freq_base", 10000000.0)
    writer.add_uint32(key + "rope.dimension_count", ROTARY)
    writer.add_array(key + "rope.dimension_sections", [11, 11, 10, 0])
    writer.add_uint32(key + "ssm.conv_kernel", KERNEL)
    writer.add_uint32(key + "ssm.state_size", KEY_DIM)
    writer.add_uint32(key + "ssm.group_count", KEY_HEADS)
    writer.add_uint32(key + "ssm.time_step_rank", VALUE_HEADS)
    writer.add_uint32(key + "ssm.inner_size", VALUE_HEADS * VALUE_DIM)
    writer.add_uint32(key + "full_attention_interval", 4)
    writer.add_float32("general.sampling.temp", 0.6)
    writer.add_int32("general.sampling.top_k", 20)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(tokenizer_pre)
    writer.add_token_list(TOKENS)
    writer.add_token_types(TYPES)
    writer.add_token_merges(MERGES)
    writer.add_bos_token_id(8)
    writer.add_eos_token_id(10)
    writer.add_pad_token_id(8)
    writer.add_chat_template(TEMPLATE)
    stored = {}
    rng = np.random.default_rng(3)
    if prediction:
        tensors = dict(tensors)
        for name, shape in (("attn_q.weight", (2 * HEADS * HEAD_DIM, HIDDEN)), ("nextn.eh_proj.weight", (HIDDEN, 2 * HIDDEN)),
                            ("nextn.enorm.weight", (HIDDEN,))):
            tensors["blk.{0}.{1}".format(LAYERS, name)] = rng.standard_normal(shape).astype(np.float32)
    for name, values in tensors.items():
        if values.ndim == 2 and name.endswith(("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")):
            data = quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            stored[name] = quants.dequantize(data, gguf.GGMLQuantizationType.Q8_0)
            continue
        writer.add_tensor(name, values)
        stored[name] = values
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return stored


def vision_checkpoint(rng):
    """Random Qwen3.5 vision-tower weights in the Hugging Face checkpoint layout (``model.visual.*``)."""
    import numpy as np

    def tensor(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    merged = VISION * MERGE * MERGE
    weights = {
        "patch_embed.proj.weight": tensor(VISION, CHANNELS, 2, PATCH, PATCH), "patch_embed.proj.bias": tensor(VISION),
        "pos_embed.weight": tensor(POSITIONS, VISION),
        "merger.norm.weight": tensor(VISION), "merger.norm.bias": tensor(VISION),
        "merger.linear_fc1.weight": tensor(merged, merged), "merger.linear_fc1.bias": tensor(merged),
        "merger.linear_fc2.weight": tensor(HIDDEN, merged), "merger.linear_fc2.bias": tensor(HIDDEN),
    }
    for block in range(DEPTH):
        for name, shape in (("attn.qkv", (3 * VISION, VISION)), ("attn.proj", (VISION, VISION)),
                            ("mlp.linear_fc1", (VISION_FF, VISION)), ("mlp.linear_fc2", (VISION, VISION_FF)),
                            ("norm1", (VISION,)), ("norm2", (VISION,))):
            weights["blocks.{0}.{1}.weight".format(block, name)] = tensor(*shape)
            weights["blocks.{0}.{1}.bias".format(block, name)] = tensor(shape[0])
    return weights


def write_projector(path, weights, projection=HIDDEN, projector_type="qwen3vl_merger"):
    """What llama.cpp's Qwen3VLVisionModel writes for ``weights`` (conversion/qwen3vl.py), as a clip mmproj GGUF."""
    import gguf

    names = {"attn.qkv": "attn_qkv", "attn.proj": "attn_out", "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down",
             "norm1": "ln1", "norm2": "ln2"}
    root = {"merger.linear_fc1": "mm.0", "merger.linear_fc2": "mm.2", "merger.norm": "v.post_ln",
            "pos_embed": "v.position_embd", "patch_embed.proj": "v.patch_embd"}
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_string("clip.projector_type", projector_type)
    writer.add_uint32("clip.vision.projection_dim", projection)
    writer.add_uint32("clip.vision.block_count", DEPTH)
    writer.add_uint32("clip.vision.embedding_length", VISION)
    writer.add_uint32("clip.vision.feed_forward_length", VISION_FF)
    writer.add_uint32("clip.vision.attention.head_count", VISION_HEADS)
    writer.add_uint32("clip.vision.patch_size", PATCH)
    writer.add_uint32("clip.vision.spatial_merge_size", MERGE)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.is_deepstack_layers", [False] * DEPTH)
    for name, values in weights.items():
        module, kind = name.rsplit(".", 1)
        if name == "patch_embed.proj.weight":
            writer.add_tensor("v.patch_embd.weight", values[:, :, 0].copy())
            writer.add_tensor("v.patch_embd.weight.1", values[:, :, 1].copy())
        elif module.startswith("blocks."):
            _, block, rest = module.split(".", 2)
            writer.add_tensor("v.blk.{0}.{1}.{2}".format(block, names[rest], kind), values)
        else:
            writer.add_tensor("{0}.{1}".format(root[module], kind), values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(HAS_RUNTIME, "gguf, numpy and mlx are not all installed in this interpreter")
class Qwen35PortTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.weights = checkpoint(np.random.default_rng(11))
        self.source = self.root / "Qwen3.5-Tiny-Q8_0.gguf"
        self.out = self.root / "Qwen3.5-Tiny-MLX-8bit"

    def run_runner(self, projector=None, **kwargs):
        self.stored = write_qwen35(self.source, kwargs.pop("tensors", llama_cpp_tensors(self.weights)), **kwargs)
        argv = ["--gguf", str(self.source), "--out", str(self.out), "--q-bits", "8", "--port", "qwen35"]
        return gguf_runner.main(argv + (["--projector", str(projector)] if projector else []))

    def test_a_projector_adds_the_vision_tower(self):
        import mlx.core as mx

        vision = vision_checkpoint(self.np.random.default_rng(5))
        projector = self.root / "mmproj-Qwen3.5-Tiny-F32.gguf"
        write_projector(projector, vision)
        self.assertEqual(self.run_runner(projector=projector), 0)
        weights = self.load()
        tower = {name[len("vision_tower."):]: value for name, value in weights.items() if name.startswith("vision_tower.")}
        self.assertEqual(set(tower), set(vision))
        for name, source in vision.items():
            expected = source.transpose(0, 2, 3, 4, 1) if name == "patch_embed.proj.weight" else source
            self.assertEqual(tower[name].dtype, mx.bfloat16, name)
            self.assertTrue(self.np.array_equal(self.as_numpy(tower[name]), self.bfloat16(expected)), name)
        config = json.loads((self.out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["model_type"], config["text_config"]["model_type"]), ("qwen3_5", "qwen3_5_text"))
        self.assertEqual({key: config["vision_config"][key] for key in (
            "depth", "hidden_size", "intermediate_size", "num_heads", "in_channels", "patch_size", "spatial_merge_size",
            "temporal_patch_size", "num_position_embeddings", "out_hidden_size", "deepstack_visual_indexes")},
            {"depth": DEPTH, "hidden_size": VISION, "intermediate_size": VISION_FF, "num_heads": VISION_HEADS,
             "in_channels": CHANNELS, "patch_size": PATCH, "spatial_merge_size": MERGE, "temporal_patch_size": 2,
             "num_position_embeddings": POSITIONS, "out_hidden_size": HIDDEN, "deepstack_visual_indexes": []})
        self.assertEqual((config["image_token_id"], config["video_token_id"], config["vision_start_token_id"],
                          config["vision_end_token_id"]), (13, 14, 11, 12))
        self.assertEqual(config["quantization"], {"group_size": 64, "bits": 8, "mode": "affine"})
        processor = json.loads((self.out / "processor_config.json").read_text(encoding="utf-8"))
        self.assertEqual((processor["processor_class"], processor["image_processor"]["patch_size"],
                          processor["image_processor"]["merge_size"], processor["image_processor"]["image_mean"]),
                         ("Qwen3VLProcessor", PATCH, MERGE, [0.5, 0.5, 0.5]))
        self.assertTrue((self.out / "preprocessor_config.json").is_file())
        self.assertTrue((self.out / "video_preprocessor_config.json").is_file())
        marker = json.loads((self.out / gguf_runner.PROVENANCE_NAME).read_text(encoding="utf-8"))
        self.assertEqual(marker["projector"]["name"], projector.name)

    def test_a_projector_for_another_width_is_refused(self):
        projector = self.root / "mmproj-other.gguf"
        write_projector(projector, vision_checkpoint(self.np.random.default_rng(5)), projection=2 * HIDDEN)
        self.assertEqual(self.run_runner(projector=projector), 1)
        self.assertFalse(self.out.exists())

    def load(self):
        import mlx.core as mx

        index = json.loads((self.out / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weights = {}
        for shard in sorted(set(index["weight_map"].values())):
            weights.update(mx.load(str(self.out / shard)))
        self.assertEqual(set(weights), set(index["weight_map"]))
        return weights

    def as_numpy(self, array):
        import mlx.core as mx

        return self.np.array(array.astype(mx.float32))

    def bfloat16(self, values):
        import mlx.core as mx

        return self.as_numpy(mx.array(values).astype(mx.bfloat16))

    def test_weights_restore_the_checkpoint_in_mlx_vlm_layout(self):
        import mlx.core as mx

        self.assertEqual(self.run_runner(), 0)
        weights = self.load()
        np = self.np
        self.assertEqual({name.rsplit(".", 1)[0] for name in weights if name.endswith((".scales", ".biases"))},
                         {"language_model." + name[: -len(".weight")] for name, values in self.weights.items() if values.ndim == 2})
        self.assertFalse([name for name in weights if ".layers.{0}.".format(LAYERS) in name or "nextn" in name])
        for name, source in self.weights.items():
            target = "language_model." + name
            base = target[: -len(".weight")]
            if base + ".scales" in weights:
                actual = self.as_numpy(mx.dequantize(weights[base + ".weight"], weights[base + ".scales"],
                                                     weights[base + ".biases"], group_size=64, bits=8))
                self.assertEqual(actual.shape, source.shape, name)
                self.assertLess(float(np.abs(actual - source).max()), 0.03 * float(np.abs(source).max()), name)
                continue
            if name.endswith("A_log"):
                self.assertEqual(weights[target].dtype, mx.float32)
                self.assertTrue(np.allclose(self.as_numpy(weights[target]), source, rtol=1e-5, atol=1e-6), name)
                continue
            if name.endswith("conv1d.weight"):
                expected = self.bfloat16(np.moveaxis(source, 2, 1))
            elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
                expected = self.bfloat16(source + 1)
            else:
                expected = self.bfloat16(source)
            self.assertEqual(weights[target].dtype, mx.bfloat16, name)
            self.assertTrue(np.array_equal(self.as_numpy(weights[target]), expected), name)

    def test_config_is_qwen3_5_text(self):
        self.assertEqual(self.run_runner(), 0)
        config = json.loads((self.out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["model_type"], config["num_hidden_layers"], config["vocab_size"]), ("qwen3_5_text", LAYERS, VOCAB))
        self.assertEqual(config["layer_types"], ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual((config["linear_num_key_heads"], config["linear_num_value_heads"], config["linear_value_head_dim"]),
                         (KEY_HEADS, VALUE_HEADS, VALUE_DIM))
        self.assertEqual(config["rope_parameters"], {
            "rope_type": "default", "mrope_section": [11, 11, 10], "mrope_interleaved": True,
            "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
        })
        self.assertEqual((config["rms_norm_eps"], config["tie_word_embeddings"], config["eos_token_id"]), (1e-06, False, 10))
        self.assertEqual(config["quantization"], {"group_size": 64, "bits": 8, "mode": "affine"})
        generation = json.loads((self.out / "generation_config.json").read_text(encoding="utf-8"))
        self.assertEqual(generation, {"bos_token_id": 8, "eos_token_id": 10, "pad_token_id": 8, "temperature": 0.6,
                                      "top_k": 20, "do_sample": True})
        marker = json.loads((self.out / gguf_runner.PROVENANCE_NAME).read_text(encoding="utf-8"))
        self.assertEqual((marker["port"], marker["q_bits"]), ("qwen35", 8))

    def test_tokenizer_is_rebuilt_from_the_vocabulary(self):
        self.assertEqual(self.run_runner(), 0)
        tokenizer = json.loads((self.out / "tokenizer.json").read_text(encoding="utf-8"))
        self.assertEqual(tokenizer["model"]["vocab"], {token: index for index, token in enumerate(NAMED[:8])})
        self.assertEqual(tokenizer["model"]["merges"], [merge.split(" ") for merge in MERGES])
        self.assertEqual([(entry["id"], entry["content"], entry["special"]) for entry in tokenizer["added_tokens"]],
                         [(index, NAMED[index], index != 15) for index in range(8, 16)])
        self.assertIn(r"[\p{L}\p{M}]+", tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"])
        config = json.loads((self.out / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["tokenizer_class"], config["eos_token"], config["pad_token"], config["bos_token"]),
                         ("Qwen3_5Tokenizer", "<|im_end|>", "<|endoftext|>", None))
        self.assertEqual(config["chat_template"], TEMPLATE)
        self.assertEqual((self.out / "chat_template.jinja").read_text(encoding="utf-8"), TEMPLATE)
        if HAS_TOKENIZERS:
            from tokenizers import Tokenizer

            encoded = Tokenizer.from_file(str(self.out / "tokenizer.json")).encode("<|im_start|>hello<think>")
            self.assertEqual(encoded.ids, [9, 7, 15])

    def test_a_prediction_drafter_without_embeddings_is_refused(self):
        tensors = {name: values for name, values in llama_cpp_tensors(self.weights).items() if not name.startswith("token_embd")}
        self.assertEqual(self.run_runner(tensors=tensors), 1)
        self.assertFalse(self.out.exists())

    def test_a_missing_tensor_leaves_no_output(self):
        tensors = {name: values for name, values in llama_cpp_tensors(self.weights).items() if name != "blk.1.ssm_out.weight"}
        self.assertEqual(self.run_runner(tensors=tensors), 1)
        self.assertFalse(self.out.exists())

    def test_an_unknown_pre_tokenizer_is_refused(self):
        self.assertEqual(self.run_runner(tokenizer_pre="llama-bpe"), 1)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
