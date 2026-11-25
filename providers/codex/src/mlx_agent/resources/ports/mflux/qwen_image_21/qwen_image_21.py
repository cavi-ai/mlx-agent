"""Qwen-Image 2.1 (Qwen/Qwen-Image-2.1 diffusers layout) through mflux's implementation.

A checkpoint whose config.json sets ``max_references`` keeps the Qwen3-VL vision
tower and loads as mflux's ``QwenImage21Edit``, which renders from text alone or
from up to ten reference images. An older text-only checkpoint loads as ``QwenImage21``.

A LoRA is merged into the loaded transformer weights before quantization:
W + scale * B @ A in float32, rounded to the weight's dtype, which reproduces a
checkpoint that ships the merge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

MFLUX_MODEL = "qwen-image-2.1"
MAX_REFERENCES = 10


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

    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    if config.get("max_references"):
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

        return QwenImage21Edit(model_path=str(path), model_config=ModelConfig.qwen_image_21())
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    return QwenImage21(model_path=str(path), model_config=ModelConfig.qwen_image_21())


def prepare_references(images):
    """Decode references before model loading and keep source paths out of mflux metadata."""
    from PIL import Image

    if len(images) > MAX_REFERENCES:
        raise ValueError("Qwen-Image 2.1 takes at most {0} reference images.".format(MAX_REFERENCES))
    references = []
    for reference in images:
        if isinstance(reference, Image.Image):
            references.append(reference)
            continue
        try:
            with Image.open(reference, formats=("PNG", "JPEG", "WEBP")) as image:
                references.append(image.copy())
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            raise ValueError("--image {0} is not a readable PNG, JPEG or WebP file.".format(reference)) from error
    return references


def generate(model, prompt: str, out: str, width: int = 1024, height: int = 1024, steps: int = 40, seed: int = 42,
             guidance: float = 1.0, images=()) -> dict:
    """Render one image to ``out`` (PNG), conditioned on ``images`` when given; returns its dimensions, timing, and pixel spread."""
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

    images = list(images)
    references = prepare_references(images)
    paths = [str(path) for path in images if isinstance(path, (str, Path))]
    render_width, render_height = width, height
    if isinstance(model, QwenImage21Edit):
        # The edit backend needs multiples of 32; the CLI accepts multiples of 16.
        render_width = ((width + 31) // 32) * 32
        render_height = ((height + 31) // 32) * 32
    started = time.time()
    image = model.generate_image(
        seed=seed, prompt=prompt, num_inference_steps=steps, width=render_width, height=render_height, guidance=guidance,
        **({"image_paths": references} if references else {}),
    )
    seconds = time.time() - started
    if (render_width, render_height) != (width, height):
        left, top = (render_width - width) // 2, (render_height - height) // 2
        image.image = image.image.crop((left, top, left + width, top + height))
        image.width, image.height = width, height
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(path=out)
    pixels = np.asarray(image.image).astype(np.float32)
    return {
        "path": str(out), "width": int(pixels.shape[1]), "height": int(pixels.shape[0]), "steps": steps, "seed": seed,
        "seconds": round(seconds, 3), "pixel_std": round(float(pixels.std()), 3), "images": paths,
    }
