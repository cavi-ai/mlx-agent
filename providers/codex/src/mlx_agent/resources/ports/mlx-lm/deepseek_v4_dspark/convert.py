"""Convert a DeepSeek-V4 DSpark drafter from llama.cpp's ``dflash`` GGUF to an MLX sidecar.

The drafter is DeepSeek-V4's DSpark stages: no token embeddings and no output
head (it borrows the target's), fed by the target's hidden states, so it only
runs beside its target model. Tensors keep the names of DeepSeek's checkpoint
with ``mtp.N`` written ``stages.N``. The projections DeepSeek ships in FP8 are
affine-quantized to ``q_bits``; the routed experts, MXFP4 in the source, are
repacked into MLX ``mxfp4`` without rounding; every other tensor keeps its
stored dtype.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARCHITECTURE = "dflash"
MODEL_TYPE = "deepseek_v4_dspark"
TARGET_MODEL_TYPE = "deepseek_v4"
GROUP_SIZE = 64
MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
MXFP4_BLOCK_BYTES = 17
SHARD_BYTES = 5 << 30
_KEY = ARCHITECTURE + "."

# llama.cpp's DeepseekV4DSparkModel tensor map (conversion/deepseek.py), reversed.
LAYER_TENSORS = {
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
}
EXPERT_TENSORS = {"ffn_gate_exps.weight": "w1", "ffn_down_exps.weight": "w2", "ffn_up_exps.weight": "w3"}
# llama.cpp writes these without a stage; DeepSeek keeps them on the first and the last stage.
FIRST_STAGE_TENSORS = {"fc.weight": "main_proj.weight", "enc.output_norm.weight": "main_norm.weight"}
LAST_STAGE_TENSORS = {
    "output_norm.weight": "norm.weight",
    "output_hc_fn.weight": "hc_head_fn",
    "output_hc_base.weight": "hc_head_base",
    "output_hc_scale.weight": "hc_head_scale",
    "markov_w1.weight": "markov_head.markov_w1.weight",
    "markov_w2.weight": "markov_head.markov_w2.weight",
    "conf_proj.weight": "confidence_head.proj.weight",
}
# DeepSeek ships these in FP8 (llama.cpp stores them Q8_0 or BF16).
QUANTIZED = frozenset((
    "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight",
    "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight",
    "main_proj.weight",
))
# llama_expert_gating_func_type values DeepSeek-V4 uses.
SCORING_FUNCS = {4: "sqrtsoftplus"}


def read_metadata(reader) -> dict:
    """Header fields, without the tokenizer's vocabulary arrays."""
    return {
        name: field.contents() for name, field in reader.fields.items()
        if not name.startswith("tokenizer.ggml.") or name.endswith("_token_id")
    }


def _float(value) -> float:
    """The shortest decimal that round-trips the stored float32 (1e-06, not 9.99999997e-07)."""
    return float(np.format_float_positional(np.float32(value), unique=True, trim="-"))


def stage_count(metadata: dict, name: str) -> int:
    if metadata.get("general.architecture") != ARCHITECTURE or _KEY + "hyper_connection.count" not in metadata:
        raise ValueError("{0} is not a DeepSeek-V4 DSpark drafter (dflash with hyper-connections).".format(name))
    return int(metadata[_KEY + "block_count"])


def tensor_names(stages: int) -> dict:
    """GGUF name to MLX name for every tensor but the routed experts."""
    names = {}
    for stage in range(stages):
        for source, target in LAYER_TENSORS.items():
            names["blk.{0}.{1}".format(stage, source)] = "stages.{0}.{1}".format(stage, target)
    for source, target in FIRST_STAGE_TENSORS.items():
        names[source] = "stages.0." + target
    for source, target in LAST_STAGE_TENSORS.items():
        names[source] = "stages.{0}.{1}".format(stages - 1, target)
    return names


def expert_tensors(stages: int) -> dict:
    """GGUF name to (stage, projection) for the stacked routed experts."""
    return {
        "blk.{0}.{1}".format(stage, source): (stage, projection)
        for stage in range(stages) for source, projection in EXPERT_TENSORS.items()
    }


def check_tensors(present, stages: int, name: str) -> None:
    expected = set(tensor_names(stages)) | set(expert_tensors(stages))
    missing, unexpected = sorted(expected - set(present)), sorted(set(present) - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing[:5]))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected[:5]))
        raise ValueError("{0} does not match the DeepSeek-V4 DSpark tensor layout: {1}.".format(name, "; ".join(detail)))


def repack_mxfp4(blocks: np.ndarray):
    """GGML MXFP4 rows (``n_blocks * 17`` bytes each) as MLX ``mxfp4`` (uint32 codes, uint8 scales).

    Both formats hold E2M1 codes with one E8M0 exponent per 32 values. GGML
    stores value ``j`` and ``j + 16`` in the low and high nibble of byte ``j``;
    MLX packs eight consecutive codes per uint32, low nibble first. The
    exponent byte carries over unchanged, so the repack is exact.
    """
    blocks = np.asarray(blocks, dtype=np.uint8)
    blocks = blocks.reshape(blocks.shape[:-1] + (-1, MXFP4_BLOCK_BYTES))
    scales = np.ascontiguousarray(blocks[..., 0])
    nibbles = blocks[..., 1:]
    codes = np.concatenate((nibbles & 0x0F, nibbles >> 4), axis=-1)
    codes = codes.reshape(codes.shape[:-2] + (-1, 8)).astype(np.uint32)
    packed = np.zeros(codes.shape[:-1], dtype=np.uint32)
    for index in range(8):
        packed |= codes[..., index] << np.uint32(4 * index)
    return packed, scales


def drafter_config(metadata: dict, stages: int, vocab_size: int, markov_rank: int, quantization: dict) -> dict:
    """``config.json``: the DSpark fields at the top, the drafter's DeepSeek-V4 parameters in ``text_config``."""
    gating = int(metadata[_KEY + "expert_gating_func"])
    if gating not in SCORING_FUNCS:
        raise ValueError("Unexpected expert gating function {0}; DeepSeek-V4 uses sqrtsoftplus (4).".format(gating))
    clamps = {_float(value) for value in metadata[_KEY + "swiglu_clamp_exp"]}
    if len(clamps) != 1:
        raise ValueError("The SwiGLU clamp differs between stages: {0}.".format(sorted(clamps)))
    text = {
        "model_type": TARGET_MODEL_TYPE,
        "vocab_size": vocab_size,
        "hidden_size": int(metadata[_KEY + "embedding_length"]),
        "num_attention_heads": int(metadata[_KEY + "attention.head_count"]),
        "num_key_value_heads": int(metadata[_KEY + "attention.head_count_kv"]),
        "head_dim": int(metadata[_KEY + "attention.key_length"]),
        "qk_rope_head_dim": int(metadata[_KEY + "rope.dimension_count"]),
        "q_lora_rank": int(metadata[_KEY + "attention.q_lora_rank"]),
        "o_lora_rank": int(metadata[_KEY + "attention.output_lora_rank"]),
        "o_groups": int(metadata[_KEY + "attention.output_group_count"]),
        "n_routed_experts": int(metadata[_KEY + "expert_count"]),
        "num_experts_per_tok": int(metadata[_KEY + "expert_used_count"]),
        "n_shared_experts": int(metadata[_KEY + "expert_shared_count"]),
        "moe_intermediate_size": int(metadata[_KEY + "expert_feed_forward_length"]),
        "routed_scaling_factor": _float(metadata[_KEY + "expert_weights_scale"]),
        "norm_topk_prob": bool(metadata[_KEY + "expert_weights_norm"]),
        "scoring_func": SCORING_FUNCS[gating],
        "swiglu_limit": clamps.pop(),
        "rms_norm_eps": _float(metadata[_KEY + "attention.layer_norm_rms_epsilon"]),
        "rope_theta": _float(metadata[_KEY + "rope.freq_base"]),
        "rope_scaling": {
            "type": metadata[_KEY + "rope.scaling.type"],
            "factor": _float(metadata[_KEY + "rope.scaling.factor"]),
            "original_max_position_embeddings": int(metadata[_KEY + "rope.scaling.original_context_length"]),
            "beta_fast": _float(metadata[_KEY + "rope.scaling.yarn_beta_fast"]),
            "beta_slow": _float(metadata[_KEY + "rope.scaling.yarn_beta_slow"]),
        },
        "max_position_embeddings": int(metadata[_KEY + "context_length"]),
        "sliding_window": int(metadata[_KEY + "attention.sliding_window"]),
        "compress_ratios": [int(value) for value in metadata[_KEY + "attention.compress_ratios"]],
        "compress_rope_theta": _float(metadata[_KEY + "attention.compress_rope_freq_base"]),
        "index_n_heads": int(metadata[_KEY + "attention.indexer.head_count"]),
        "index_head_dim": int(metadata[_KEY + "attention.indexer.key_length"]),
        "index_topk": int(metadata[_KEY + "attention.indexer.top_k"]),
        "hc_mult": int(metadata[_KEY + "hyper_connection.count"]),
        "hc_sinkhorn_iters": int(metadata[_KEY + "hyper_connection.sinkhorn_iterations"]),
        "hc_eps": _float(metadata[_KEY + "hyper_connection.epsilon"]),
    }
    for key, field in (("bos_token_id", "tokenizer.ggml.bos_token_id"), ("eos_token_id", "tokenizer.ggml.eos_token_id")):
        if field in metadata:
            text[key] = int(metadata[field])
    return {
        "model_type": MODEL_TYPE,
        "n_mtp_layers": stages,
        "dspark_block_size": int(metadata[_KEY + "block_size"]),
        "dspark_noise_token_id": int(metadata["tokenizer.ggml.mask_token_id"]),
        # llama.cpp stores each id + 1: the input of layer i is layer i - 1's output.
        "dspark_target_layer_ids": [int(value) - 1 for value in metadata[_KEY + "target_layers"]],
        "dspark_markov_rank": markov_rank,
        "dspark_target_name": metadata.get("general.name"),
        "quantization": quantization,
        "quantization_config": quantization,
        "text_config": text,
    }


class ShardWriter:
    """Writes ``model-0000k-of-0000n.safetensors`` shards of at most ``limit`` bytes, then the index."""

    def __init__(self, directory: Path, limit: int = SHARD_BYTES):
        self.directory = Path(directory)
        self.limit = limit
        self.pending = {}
        self.pending_bytes = 0
        self.shards = []
        self.weight_map = {}
        self.total = 0

    def add(self, name: str, array) -> None:
        if self.pending and self.pending_bytes + array.nbytes > self.limit:
            self._flush()
        self.pending[name] = array
        self.pending_bytes += array.nbytes
        self.total += array.nbytes

    def _flush(self) -> None:
        import mlx.core as mx

        path = self.directory / "model-{0:05d}.partial.safetensors".format(len(self.shards) + 1)
        mx.save_safetensors(str(path), self.pending, metadata={"format": "mlx"})
        self.shards.append((path, sorted(self.pending)))
        self.pending, self.pending_bytes = {}, 0

    def finish(self) -> int:
        if self.pending:
            self._flush()
        count = len(self.shards)
        for index, (path, names) in enumerate(self.shards, start=1):
            final = "model-{0:05d}-of-{1:05d}.safetensors".format(index, count)
            path.rename(self.directory / final)
            self.weight_map.update({name: final for name in names})
        index = {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))}
        (self.directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return count


def convert(gguf_path, out_dir, q_bits: int, log=print, shard_bytes: int = SHARD_BYTES) -> dict:
    """Write the MLX drafter into ``out_dir`` (which must exist and be empty); ``config.json`` lands last."""
    import gguf
    import mlx.core as mx
    from gguf import quants

    source = Path(gguf_path)
    reader = gguf.GGUFReader(str(source))
    metadata = read_metadata(reader)
    stages = stage_count(metadata, source.name)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    check_tensors(tensors, stages, source.name)
    names, experts = tensor_names(stages), expert_tensors(stages)
    dtypes = {
        gguf.GGMLQuantizationType.F32: mx.float32,
        gguf.GGMLQuantizationType.F16: mx.float16,
        gguf.GGMLQuantizationType.BF16: mx.bfloat16,
    }
    expert_count = int(metadata[_KEY + "expert_count"])
    quantization = {"group_size": GROUP_SIZE, "bits": q_bits, "mode": "affine"}
    writer = ShardWriter(out_dir, shard_bytes)
    for tensor in reader.tensors:
        if tensor.name in experts:
            stage, projection = experts[tensor.name]
            if tensor.tensor_type != gguf.GGMLQuantizationType.MXFP4 or tensor.data.shape[0] != expert_count:
                raise ValueError("{0} must hold {1} MXFP4 experts; found {2} {3}.".format(
                    tensor.name, expert_count, tensor.data.shape[0], tensor.tensor_type.name))
            log("repacking {0} ({1} experts, mxfp4)".format(tensor.name, expert_count))
            for expert in range(expert_count):
                weight, scales = repack_mxfp4(tensor.data[expert])
                base = "stages.{0}.ffn.experts.{1}.{2}".format(stage, expert, projection)
                writer.add(base + ".weight", mx.array(weight))
                writer.add(base + ".scales", mx.array(scales))
                quantization[base] = dict(MXFP4)
            continue
        target = names[tensor.name]
        values = quants.dequantize(tensor.data, tensor.tensor_type)
        if target.split(".", 2)[2] in QUANTIZED:
            log("quantizing {0} -> {1} ({2}-bit)".format(tensor.name, target, q_bits))
            weight, scales, biases = mx.quantize(mx.array(values).astype(mx.bfloat16), group_size=GROUP_SIZE, bits=q_bits)
            base = target[: -len(".weight")]
            writer.add(base + ".weight", weight)
            writer.add(base + ".scales", scales)
            writer.add(base + ".biases", biases)
            continue
        dtype = dtypes.get(tensor.tensor_type)
        if dtype is None:
            raise ValueError("{0} is stored as {1}; expected F32, F16, or BF16.".format(tensor.name, tensor.tensor_type.name))
        writer.add(target, mx.array(values).astype(dtype))
    shards = writer.finish()
    # GGUF shapes list the innermost dimension first: markov_w1 is [rank, vocab].
    markov_rank, vocab_size = (int(size) for size in tensors["markov_w1.weight"].shape)
    config = drafter_config(metadata, stages, vocab_size, markov_rank, quantization)
    (Path(out_dir) / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log("wrote {0} tensors in {1} shard(s), {2} bytes".format(len(writer.weight_map), shards, writer.total))
    return {
        "port": MODEL_TYPE,
        "target": config["dspark_target_name"],
        "tensors": len(writer.weight_map),
        "bytes": writer.total,
    }
