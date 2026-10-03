"""Qwen-Image 2.1 (Qwen/Qwen-Image-2.1 diffusers layout) through mflux's implementation.

mflux 0.20 loads no LoRA for this model, so a LoRA is merged into the loaded
transformer weights before quantization: W + scale * B @ A in float32, rounded
to the weight's dtype, which reproduces a checkpoint that ships the merge.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

MFLUX_MODEL = "qwen-image-2.1"


def _node(blocks, module: str):
    parts = module.split(".")
    if len(parts) < 3 or parts[0] != "transformer_blocks" or not parts[1].isdigit():
        raise ValueError("LoRA module {0!r} is not a transformer block layer".format(module))
    node = blocks[int(parts[1])]
    for part in parts[2:]:
        node = node[int(part)] if part.isdigit() else node[part]
    return node


def merge_lora(transformer: dict, lora: dict, scale: float) -> int:
    """Fold PEFT ``lora_A``/``lora_B`` pairs into ``transformer`` (mflux's loaded weight tree); returns the count."""
    blocks = transformer["transformer_blocks"]
    merged = 0
    for name in sorted(lora):
        if ".lora_A." not in name:
            continue
        module, suffix = name.split(".lora_A.")
        partner = module + ".lora_B." + suffix
        if partner not in lora:
            raise ValueError("LoRA {0} has no lora_B".format(module))
        node = _node(blocks, module)
        weight = node["weight"]
        a, b = lora[name].astype(mx.float32), lora[partner].astype(mx.float32)
        if (b.shape[0], a.shape[1]) != tuple(weight.shape):
            raise ValueError("LoRA {0} is {1}x{2}, the weight is {3}".format(module, b.shape[0], a.shape[1], tuple(weight.shape)))
        node["weight"] = (weight.astype(mx.float32) + scale * (b @ a)).astype(weight.dtype)
        merged += 1
    if not merged:
        raise ValueError("the LoRA file has no lora_A/lora_B pairs")
    return merged


def load(path):
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    return QwenImage21(model_path=str(path), model_config=ModelConfig.qwen_image_21())


def generate(model, prompt: str, out: str, width: int = 1024, height: int = 1024, steps: int = 40, seed: int = 42,
             guidance: float = 1.0) -> dict:
    """Render one image to ``out`` (PNG); returns its dimensions, timing, and pixel spread."""
    started = time.time()
    image = model.generate_image(
        seed=seed, prompt=prompt, num_inference_steps=steps, width=width, height=height, guidance=guidance,
    )
    seconds = time.time() - started
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(path=out)
    pixels = np.asarray(image.image).astype(np.float32)
    return {
        "path": str(out), "width": int(pixels.shape[1]), "height": int(pixels.shape[0]), "steps": steps, "seed": seed,
        "seconds": round(seconds, 3), "pixel_std": round(float(pixels.std()), 3),
    }
