"""Convert a llama.cpp Qwen3.5 GGUF (``qwen35``, also Qwen3.6 and Qwen3.8 checkpoints) to MLX-VLM's ``qwen3_5_text``.

llama.cpp's ``Qwen3_5TextModel`` converter (conversion/qwen.py) reorders the
gated-delta value heads from grouped-by-key-head to tiled order, stores
``-exp(A_log)``, squeezes the conv1d kernel, writes the ``1 + w`` RMSNorm
weights already shifted, and appends the multi-token-prediction block as the
last ``blk``. This port restores the grouped head order and ``A_log``, keeps
the shifted norms and a ``(channels, kernel, 1)`` kernel (the layout MLX-VLM
loads without re-sanitizing), drops the prediction block, and affine-quantizes
every projection and both vocabulary tables to ``q_bits``. The tokenizer is
rebuilt from the GGUF vocabulary. Given the model's Qwen3-VL projector GGUF
(``mmproj``), the vision tower joins it as MLX-VLM's ``qwen3_5``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARCHITECTURE = "qwen35"
MODEL_TYPE = "qwen3_5_text"
GROUP_SIZE = 64
_KEY = ARCHITECTURE + "."

# llama.cpp's Qwen3.5 tensor map (gguf-py tensor_mapping), reversed.
LAYER_TENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
LINEAR_ATTENTION_TENSORS = {
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
}
FULL_ATTENTION_TENSORS = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}
ROOT_TENSORS = {
    "token_embd.weight": "language_model.model.embed_tokens.weight",
    "output_norm.weight": "language_model.model.norm.weight",
    "output.weight": "language_model.lm_head.weight",
}
# llama.cpp's Qwen3VLVisionModel projector map (conversion/qwen3vl.py), reversed.
PROJECTOR_TYPE = "qwen3vl_merger"
VISION_LAYER_TENSORS = {
    "attn_qkv": "attn.qkv", "attn_out": "attn.proj", "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2", "ln1": "norm1", "ln2": "norm2",
}
VISION_ROOT_TENSORS = {
    "mm.0.weight": "merger.linear_fc1.weight", "mm.0.bias": "merger.linear_fc1.bias",
    "mm.2.weight": "merger.linear_fc2.weight", "mm.2.bias": "merger.linear_fc2.bias",
    "v.post_ln.weight": "merger.norm.weight", "v.post_ln.bias": "merger.norm.bias",
    "v.patch_embd.bias": "patch_embed.proj.bias", "v.position_embd.weight": "pos_embed.weight",
}
# The Conv3d patch kernel, split by llama.cpp into one Conv2d kernel per temporal step.
PATCH_KERNELS = ("v.patch_embd.weight", "v.patch_embd.weight.1")
# Qwen3.5's published processor limits; the projector header carries only mean, std, patch and merge sizes.
IMAGE_PIXELS = (65536, 16777216)
VIDEO_PIXELS = (4096, 25165824)
VIDEO_FRAMES = (4, 768)
VIDEO_FPS = 2.0
# tokenizer.ggml.pre to the tokenizer.json split regex (llama.cpp llama-vocab.cpp) and its transformers class.
PRE_TOKENIZERS = {
    "qwen35": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        "Qwen3_5Tokenizer",
    ),
    "qwen2": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        "Qwen2Tokenizer",
    ),
}
# llama_token_type values in tokenizer.ggml.token_type.
NORMAL, CONTROL, USER_DEFINED = 1, 3, 4


def read_metadata(reader) -> dict:
    """Header fields, without the tokenizer's vocabulary arrays."""
    return {
        name: field.contents() for name, field in reader.fields.items()
        if name not in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.scores")
    }


def _float(value) -> float:
    """The shortest decimal that round-trips the stored float32 (1e-06, not 9.99999997e-07)."""
    return float(np.format_float_positional(np.float32(value), unique=True, trim="-"))


def _heads(value, name: str) -> int:
    """A head count llama.cpp may store per layer; Qwen3.5 uses one value."""
    counts = {int(count) for count in (value if isinstance(value, list) else [value]) if int(count) > 0}
    if len(counts) != 1:
        raise ValueError("{0} varies across layers: {1}.".format(name, sorted(counts)))
    return counts.pop()


def dimensions(metadata: dict, name: str) -> dict:
    """The text-model parameters, from the qwen35 header."""
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise ValueError("{0} is a {1} GGUF, not Qwen3.5 (qwen35).".format(name, metadata.get("general.architecture")))
    blocks = int(metadata[_KEY + "block_count"])
    predict = int(metadata.get(_KEY + "nextn_predict_layers", 0))
    value_heads = int(metadata[_KEY + "ssm.time_step_rank"])
    inner = int(metadata[_KEY + "ssm.inner_size"])
    head_dim = int(metadata[_KEY + "attention.key_length"])
    if inner % value_heads:
        raise ValueError("{0}: ssm.inner_size {1} is not a multiple of {2} value heads.".format(name, inner, value_heads))
    return {
        "layers": blocks - predict,
        "predict_layers": predict,
        "hidden": int(metadata[_KEY + "embedding_length"]),
        "intermediate": int(metadata[_KEY + "feed_forward_length"]),
        "heads": _heads(metadata[_KEY + "attention.head_count"], "attention.head_count"),
        "kv_heads": _heads(metadata[_KEY + "attention.head_count_kv"], "attention.head_count_kv"),
        "head_dim": head_dim,
        "key_heads": int(metadata[_KEY + "ssm.group_count"]),
        "key_head_dim": int(metadata[_KEY + "ssm.state_size"]),
        "value_heads": value_heads,
        "value_head_dim": inner // value_heads,
        "conv_kernel": int(metadata[_KEY + "ssm.conv_kernel"]),
        "interval": int(metadata.get(_KEY + "full_attention_interval", 4)),
        "rotary_dim": int(metadata[_KEY + "rope.dimension_count"]),
    }


def is_full_attention(layer: int, interval: int) -> bool:
    return (layer + 1) % interval == 0


def tensor_names(dims: dict, tied: bool) -> dict:
    """GGUF name to MLX name for every tensor the text model keeps."""
    names = {source: target for source, target in ROOT_TENSORS.items() if not (tied and source == "output.weight")}
    for layer in range(dims["layers"]):
        kind = FULL_ATTENTION_TENSORS if is_full_attention(layer, dims["interval"]) else LINEAR_ATTENTION_TENSORS
        for source, target in {**LAYER_TENSORS, **kind}.items():
            names["blk.{0}.{1}".format(layer, source)] = "language_model.model.layers.{0}.{1}".format(layer, target)
    return names


def check_tensors(present, dims: dict, tied: bool, name: str) -> None:
    expected = set(tensor_names(dims, tied))
    prediction = {
        tensor for tensor in present
        if tensor.startswith("blk.") and int(tensor.split(".", 2)[1]) >= dims["layers"]
    }
    missing = sorted(expected - set(present))
    unexpected = sorted(set(present) - expected - prediction)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing[:5]))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected[:5]))
        raise ValueError("{0} does not match the Qwen3.5 tensor layout: {1}.".format(name, "; ".join(detail)))


def grouped_heads(values: np.ndarray, axis: int, key_heads: int, value_heads: int, head_dim: int) -> np.ndarray:
    """Undo llama.cpp's ``_reorder_v_heads``: tiled ``[v0 of every key head, v1 ...]`` back to grouped by key head."""
    shape = list(values.shape)
    axis %= values.ndim
    tiled = values.reshape(shape[:axis] + [value_heads // key_heads, key_heads, head_dim] + shape[axis + 1:])
    return np.ascontiguousarray(np.swapaxes(tiled, axis, axis + 1)).reshape(shape)


def restore(suffix: str, values: np.ndarray, dims: dict) -> np.ndarray:
    """The checkpoint tensor llama.cpp wrote as ``values`` (the dequantized GGUF tensor)."""
    def regroup(array, axis, head_dim):
        return grouped_heads(array, axis, dims["key_heads"], dims["value_heads"], head_dim)

    key_dim = dims["key_heads"] * dims["key_head_dim"]
    if suffix == "attn_qkv.weight":
        return np.concatenate([values[: 2 * key_dim], regroup(values[2 * key_dim:], 0, dims["value_head_dim"])])
    if suffix == "attn_gate.weight":
        return regroup(values, 0, dims["value_head_dim"])
    if suffix in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias"):
        return regroup(values, 0, 1)
    if suffix == "ssm_a":
        return regroup(np.log(-values), 0, 1)
    if suffix == "ssm_conv1d.weight":
        kernel = np.concatenate([values[: 2 * key_dim], regroup(values[2 * key_dim:], 0, dims["value_head_dim"])])
        return kernel.reshape(kernel.shape + (1,))
    if suffix == "ssm_out.weight":
        return regroup(values, 1, dims["value_head_dim"])
    return values


def text_config(metadata: dict, dims: dict, vocab_size: int, tied: bool, quantization: dict) -> dict:
    """``config.json`` in the flat ``qwen3_5_text`` layout MLX-VLM reads."""
    sections = [int(value) for value in metadata.get(_KEY + "rope.dimension_sections", [11, 11, 10, 0])]
    while sections and sections[-1] == 0:
        sections.pop()
    partial = dims["rotary_dim"] / dims["head_dim"]
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": MODEL_TYPE,
        "vocab_size": vocab_size,
        "hidden_size": dims["hidden"],
        "intermediate_size": dims["intermediate"],
        "num_hidden_layers": dims["layers"],
        "num_attention_heads": dims["heads"],
        "num_key_value_heads": dims["kv_heads"],
        "head_dim": dims["head_dim"],
        "attn_output_gate": True,
        "linear_num_key_heads": dims["key_heads"],
        "linear_key_head_dim": dims["key_head_dim"],
        "linear_num_value_heads": dims["value_heads"],
        "linear_value_head_dim": dims["value_head_dim"],
        "linear_conv_kernel_dim": dims["conv_kernel"],
        "full_attention_interval": dims["interval"],
        "layer_types": [
            "full_attention" if is_full_attention(layer, dims["interval"]) else "linear_attention"
            for layer in range(dims["layers"])
        ],
        "rms_norm_eps": _float(metadata[_KEY + "attention.layer_norm_rms_epsilon"]),
        "max_position_embeddings": int(metadata[_KEY + "context_length"]),
        "partial_rotary_factor": partial,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_section": sections,
            "mrope_interleaved": True,
            "rope_theta": _float(metadata[_KEY + "rope.freq_base"]),
            "partial_rotary_factor": partial,
        },
        "attention_bias": False,
        "tie_word_embeddings": tied,
        "quantization": quantization,
        "quantization_config": quantization,
    }
    for key, field in (("bos_token_id", "tokenizer.ggml.bos_token_id"), ("eos_token_id", "tokenizer.ggml.eos_token_id"),
                       ("pad_token_id", "tokenizer.ggml.padding_token_id")):
        if field in metadata:
            config[key] = int(metadata[field])
    return config


def vision_names(blocks: int) -> dict:
    """Projector GGUF name to MLX name for every vision tensor but the split patch kernel."""
    names = {source: "vision_tower." + target for source, target in VISION_ROOT_TENSORS.items()}
    for block in range(blocks):
        for source, target in VISION_LAYER_TENSORS.items():
            for kind in ("weight", "bias"):
                names["v.blk.{0}.{1}.{2}".format(block, source, kind)] = "vision_tower.blocks.{0}.{1}.{2}".format(block, target, kind)
    return names


def vision_config(metadata: dict, tensors: dict, dims: dict, name: str) -> dict:
    """``vision_config`` for MLX-VLM's Qwen3.5 tower, read from a ``qwen3vl_merger`` projector GGUF."""
    if metadata.get("general.architecture") != "clip" or metadata.get("clip.projector_type") != PROJECTOR_TYPE:
        raise ValueError("{0} is not a Qwen3-VL vision projector (found {1}).".format(name, metadata.get("clip.projector_type")))
    if any(metadata.get("clip.vision.is_deepstack_layers") or ()):
        raise ValueError("{0} has deepstack layers, which Qwen3.5 does not use.".format(name))
    projection = int(metadata["clip.vision.projection_dim"])
    if projection != dims["hidden"]:
        raise ValueError("{0} projects to {1} dimensions; the model's embeddings have {2}.".format(name, projection, dims["hidden"]))
    blocks = int(metadata["clip.vision.block_count"])
    expected = set(vision_names(blocks)) | set(PATCH_KERNELS)
    missing, unexpected = sorted(expected - set(tensors)), sorted(set(tensors) - expected)
    if missing or unexpected:
        detail = (["missing " + ", ".join(missing[:5])] if missing else []) + (
            ["unexpected " + ", ".join(unexpected[:5])] if unexpected else [])
        raise ValueError("{0} does not match the Qwen3-VL projector layout: {1}.".format(name, "; ".join(detail)))
    return {
        "model_type": "qwen3_5",
        "depth": blocks,
        "hidden_size": int(metadata["clip.vision.embedding_length"]),
        "intermediate_size": int(metadata["clip.vision.feed_forward_length"]),
        "num_heads": int(metadata["clip.vision.attention.head_count"]),
        "hidden_act": "gelu_pytorch_tanh",
        # GGUF shapes list the innermost dimension first: a patch kernel is [width, height, channels, out].
        "in_channels": int(tensors["v.patch_embd.weight"].shape[2]),
        "patch_size": int(metadata["clip.vision.patch_size"]),
        "spatial_merge_size": int(metadata["clip.vision.spatial_merge_size"]),
        "temporal_patch_size": len(PATCH_KERNELS),
        "num_position_embeddings": int(tensors["v.position_embd.weight"].shape[1]),
        "out_hidden_size": projection,
        "layer_norm_eps": _float(metadata.get("clip.vision.attention.layer_norm_epsilon", 1e-6)),
        "deepstack_visual_indexes": [],
    }


def processor_files(metadata: dict, vision: dict) -> dict:
    """The Qwen3-VL image and video processor settings for the projector's normalization and patching."""
    shared = {
        "patch_size": vision["patch_size"],
        "temporal_patch_size": vision["temporal_patch_size"],
        "merge_size": vision["spatial_merge_size"],
        "image_mean": [_float(value) for value in metadata["clip.vision.image_mean"]],
        "image_std": [_float(value) for value in metadata["clip.vision.image_std"]],
    }
    flags = {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "rescale_factor": 1 / 255}
    return {
        "preprocessor_config.json": dict(shared, size={"shortest_edge": IMAGE_PIXELS[0], "longest_edge": IMAGE_PIXELS[1]},
                                         processor_class="Qwen3VLProcessor", image_processor_type="Qwen2VLImageProcessorFast"),
        "video_preprocessor_config.json": dict(shared, size={"shortest_edge": VIDEO_PIXELS[0], "longest_edge": VIDEO_PIXELS[1]},
                                               processor_class="Qwen3VLProcessor", video_processor_type="Qwen3VLVideoProcessor"),
        "processor_config.json": {
            "processor_class": "Qwen3VLProcessor",
            "image_processor": dict(shared, **flags, image_processor_type="Qwen3VLImageProcessor",
                                    min_pixels=IMAGE_PIXELS[0], max_pixels=IMAGE_PIXELS[1]),
            "video_processor": dict(shared, **flags, video_processor_type="Qwen3VLVideoProcessor", fps=VIDEO_FPS,
                                    min_frames=VIDEO_FRAMES[0], max_frames=VIDEO_FRAMES[1],
                                    min_pixels=VIDEO_PIXELS[0], max_pixels=VIDEO_PIXELS[1]),
        },
    }


def vision_language_config(text: dict, vision: dict, tokens: list) -> dict:
    """``config.json`` for MLX-VLM's ``qwen3_5``: the text model nested beside the vision tower."""
    special = {"image_token_id": "<|image_pad|>", "video_token_id": "<|video_pad|>",
               "vision_start_token_id": "<|vision_start|>", "vision_end_token_id": "<|vision_end|>"}
    missing = sorted(token for token in special.values() if token not in tokens)
    if missing:
        raise ValueError("The vocabulary lacks the vision tokens {0}.".format(", ".join(missing)))
    quantization = text["quantization"]
    nested = {key: value for key, value in text.items() if key not in ("architectures", "quantization", "quantization_config")}
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": nested,
        "vision_config": vision,
        "tie_word_embeddings": text["tie_word_embeddings"],
        "quantization": quantization,
        "quantization_config": quantization,
    }
    config.update({key: tokens.index(token) for key, token in special.items()})
    if "eos_token_id" in text:
        config["eos_token_id"] = text["eos_token_id"]
    return config


def generation_config(metadata: dict) -> dict:
    config = {}
    for key, field in (("bos_token_id", "tokenizer.ggml.bos_token_id"), ("eos_token_id", "tokenizer.ggml.eos_token_id"),
                       ("pad_token_id", "tokenizer.ggml.padding_token_id")):
        if field in metadata:
            config[key] = int(metadata[field])
    for key, field in (("temperature", "general.sampling.temp"), ("top_p", "general.sampling.top_p")):
        if field in metadata:
            config[key] = _float(metadata[field])
    if "general.sampling.top_k" in metadata:
        config["top_k"] = int(metadata["general.sampling.top_k"])
    if "temperature" in config:
        config["do_sample"] = config["temperature"] > 0
    return config


def tokenizer_files(reader, metadata: dict, tokens: list, name: str) -> dict:
    """``tokenizer.json``, ``tokenizer_config.json`` and the chat template, rebuilt from the GGUF vocabulary."""
    model, pre = metadata.get("tokenizer.ggml.model"), metadata.get("tokenizer.ggml.pre")
    if model != "gpt2" or pre not in PRE_TOKENIZERS:
        raise ValueError("{0} has a {1}/{2} tokenizer; the qwen35 port reads gpt2 BPE with a qwen35 or qwen2 pre-tokenizer.".format(
            name, model, pre))
    pattern, tokenizer_class = PRE_TOKENIZERS[pre]
    types = reader.fields["tokenizer.ggml.token_type"].contents()
    merges = [merge.split(" ") for merge in reader.fields["tokenizer.ggml.merges"].contents()]
    if any(len(pair) != 2 for pair in merges):
        raise ValueError("{0} has a merge that is not one pair of byte-level tokens.".format(name))
    added = [
        {"id": index, "content": token, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": kind == CONTROL}
        for index, (token, kind) in enumerate(zip(tokens, types)) if kind in (CONTROL, USER_DEFINED)
    ]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": {token: index for index, (token, kind) in enumerate(zip(tokens, types)) if kind == NORMAL},
            "merges": merges,
        },
    }
    config = {
        "tokenizer_class": tokenizer_class,
        "add_prefix_space": False,
        "bos_token": None,
        "unk_token": None,
        "errors": "replace",
        "clean_up_tokenization_spaces": False,
        "split_special_tokens": False,
        "model_max_length": int(metadata[_KEY + "context_length"]),
        "added_tokens_decoder": {
            str(entry["id"]): {key: value for key, value in entry.items() if key != "id"} for entry in added
        },
    }
    for key, field in (("eos_token", "tokenizer.ggml.eos_token_id"), ("pad_token", "tokenizer.ggml.padding_token_id")):
        if field in metadata:
            config[key] = tokens[int(metadata[field])]
    files = {"tokenizer.json": tokenizer, "tokenizer_config.json": config}
    template = metadata.get("tokenizer.chat_template")
    if template:
        config["chat_template"] = template
        files["chat_template.jinja"] = template
    return files


def convert(gguf_path, out_dir, q_bits: int, writer, log=print, projector=None) -> dict:
    """Write the MLX model into ``out_dir`` (which must exist and be empty) through ``writer``; ``config.json`` lands last.

    With ``projector`` (the model's Qwen3-VL ``mmproj`` GGUF) the output is MLX-VLM's
    vision-language ``qwen3_5`` with the vision tower kept in bfloat16; without it, ``qwen3_5_text``.
    """
    import gguf
    import mlx.core as mx
    from gguf import quants

    source, out_dir = Path(gguf_path), Path(out_dir)
    reader = gguf.GGUFReader(str(source))
    metadata = read_metadata(reader)
    dims = dimensions(metadata, source.name)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if "token_embd.weight" not in tensors:
        raise ValueError("{0} holds no token embeddings: it is a multi-token-prediction drafter, not a model.".format(source.name))
    tied = "output.weight" not in tensors
    check_tensors(tensors, dims, tied, source.name)
    names = tensor_names(dims, tied)
    query_rows = int(tensors["blk.{0}.attn_q.weight".format(dims["interval"] - 1)].shape[1])
    if query_rows != 2 * dims["heads"] * dims["head_dim"]:
        raise ValueError("{0}: attn_q has {1} rows; Qwen3.5 gates attention with {2}.".format(
            source.name, query_rows, 2 * dims["heads"] * dims["head_dim"]))
    vision = None
    if projector is not None:
        projector = Path(projector)
        vision_reader = gguf.GGUFReader(str(projector))
        vision_metadata = read_metadata(vision_reader)
        vision_tensors = {tensor.name: tensor for tensor in vision_reader.tensors}
        vision = vision_config(vision_metadata, vision_tensors, dims, projector.name)
    quantization = {"group_size": GROUP_SIZE, "bits": q_bits, "mode": "affine"}
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    files = tokenizer_files(reader, metadata, tokens, source.name)
    for tensor in reader.tensors:
        target = names.get(tensor.name)
        if target is None:
            continue
        suffix = tensor.name.split(".", 2)[2] if tensor.name.startswith("blk.") else tensor.name
        values = restore(suffix, quants.dequantize(tensor.data, tensor.tensor_type), dims)
        if values.ndim == 2 and values.shape[-1] % GROUP_SIZE == 0:
            log("quantizing {0} -> {1} ({2}-bit)".format(tensor.name, target, q_bits))
            weight, scales, biases = mx.quantize(mx.array(values).astype(mx.bfloat16), group_size=GROUP_SIZE, bits=q_bits)
            mx.eval(weight, scales, biases)
            base = target[: -len(".weight")]
            writer.add(base + ".weight", weight)
            writer.add(base + ".scales", scales)
            writer.add(base + ".biases", biases)
            continue
        # A_log stays float32 (MLX-VLM's cast predicate keeps it so); every other small tensor is bfloat16.
        array = mx.array(values).astype(mx.float32 if suffix == "ssm_a" else mx.bfloat16)
        mx.eval(array)
        writer.add(target, array)
    if vision is not None:
        log("adding the vision tower from {0}".format(projector.name))
        for source_name, target in vision_names(vision["depth"]).items():
            tensor = vision_tensors[source_name]
            array = mx.array(quants.dequantize(tensor.data, tensor.tensor_type)).astype(mx.bfloat16)
            mx.eval(array)
            writer.add(target, array)
        # Conv2d kernels [out, channels, height, width] per temporal step -> MLX Conv3d [out, time, height, width, channels].
        kernels = [quants.dequantize(vision_tensors[name].data, vision_tensors[name].tensor_type) for name in PATCH_KERNELS]
        kernel = mx.array(np.stack(kernels, axis=1).transpose(0, 1, 3, 4, 2)).astype(mx.bfloat16)
        mx.eval(kernel)
        writer.add("vision_tower.patch_embed.proj.weight", kernel)
        files.update(processor_files(vision_metadata, vision))
    shards = writer.finish()
    vocab_size = int(tensors["token_embd.weight"].shape[1])
    for filename, value in files.items():
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        (out_dir / filename).write_text(text, encoding="utf-8")
    (out_dir / "generation_config.json").write_text(json.dumps(generation_config(metadata), indent=2) + "\n", encoding="utf-8")
    config = text_config(metadata, dims, vocab_size, tied, quantization)
    if vision is not None:
        config = vision_language_config(config, vision, tokens)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log("wrote {0} tensors in {1} shard(s), {2} bytes; dropped {3} prediction block(s)".format(
        len(writer.weight_map), shards, writer.total, dims["predict_layers"]))
    return {
        "port": ARCHITECTURE,
        "model_type": config["model_type"],
        "name": metadata.get("general.name"),
        "tensors": len(writer.weight_map),
        "bytes": writer.total,
    }
