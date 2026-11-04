"""DINOv2 image features using the Hugging Face checkpoint layout."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


def preprocess(path, config):
    with Image.open(path) as image:
        image.load()
        # Composite alpha onto white; inputs should already have a clean background.
        rgba = image.convert("RGBA")
        image = Image.new("RGBA", rgba.size, "white")
        image.alpha_composite(rgba)
        image = image.convert("RGB")
        if config.get("do_resize", True):
            size = config["size"]["shortest_edge"]
            width, height = image.size
            new_size = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
            image = image.resize(new_size, resample=Image.Resampling(config.get("resample", 3)))
        if config.get("do_center_crop", True):
            crop = config["crop_size"]
            left, top = (image.width - crop["width"]) // 2, (image.height - crop["height"]) // 2
            image = image.crop((left, top, left + crop["width"], top + crop["height"]))
        pixels = np.asarray(image, dtype=np.float32)
    if config.get("do_rescale", True):
        pixels = pixels * config.get("rescale_factor", 1 / 255)
    if config.get("do_normalize", True):
        pixels = (pixels - np.array(config["image_mean"], dtype=np.float32)) / np.array(config["image_std"], dtype=np.float32)
    return mx.array(pixels[None])


def interpolate_positions(positions, height, width):
    """PyTorch bicubic interpolation with align_corners=False."""
    grid = math.isqrt(positions.shape[1] - 1)
    if height == width == grid:
        return positions
    patch = positions[:, 1:].reshape(grid, grid, -1)

    def resize_axis(values, size, axis):
        coordinate = (np.arange(size, dtype=np.float32) + .5) * grid / size - .5
        base = np.floor(coordinate).astype(np.int32)
        indices = base[:, None] + np.arange(-1, 3, dtype=np.int32)
        distance = np.abs(coordinate[:, None] - indices)
        a = -.75
        coeff = np.where(distance <= 1, (a + 2) * distance ** 3 - (a + 3) * distance ** 2 + 1,
                         np.where(distance < 2, a * distance ** 3 - 5 * a * distance ** 2 + 8 * a * distance - 4 * a, 0))
        selected = mx.take(values, mx.array(np.clip(indices, 0, grid - 1)), axis=axis)
        shape = [1] * selected.ndim
        shape[axis:axis + 2] = [size, 4]
        return mx.sum(selected * mx.array(coeff.astype(np.float32)).reshape(shape), axis=axis + 1)

    patch = resize_axis(resize_axis(patch, height, 0), width, 1)
    return mx.concatenate([positions[:, :1], patch.reshape(1, height * width, -1)], axis=1)


def encode(w, config, pixels):
    patch = config["patch_size"]
    projection = w["embeddings.patch_embeddings.projection.weight"].transpose(0, 2, 3, 1)
    x = mx.conv2d(pixels, projection, stride=patch) + w["embeddings.patch_embeddings.projection.bias"]
    batch, height, width, channels = x.shape
    x = x.reshape(batch, -1, channels)
    cls = mx.broadcast_to(w["embeddings.cls_token"], (batch, 1, channels))
    x = mx.concatenate([cls, x], axis=1) + interpolate_positions(w["embeddings.position_embeddings"], height, width)
    eps, heads = config["layer_norm_eps"], config["num_attention_heads"]
    for index in range(config["num_hidden_layers"]):
        name = "encoder.layer." + str(index)
        normalized = w.norm(name + ".norm1", x, eps)
        q, k, v = [w.linear(name + ".attention.attention." + key, normalized)
                   .reshape(batch, -1, heads, channels // heads).transpose(0, 2, 1, 3)
                   for key in ("query", "key", "value")]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=(channels // heads) ** -.5)
        y = y.transpose(0, 2, 1, 3).reshape(batch, -1, channels)
        x = x + w.linear(name + ".attention.output.dense", y) * w[name + ".layer_scale1.lambda1"]
        y = w.linear(name + ".mlp.fc2", nn.gelu(w.linear(name + ".mlp.fc1", w.norm(name + ".norm2", x, eps))))
        x = x + y * w[name + ".layer_scale2.lambda1"]
        mx.eval(x)
    return w.norm("layernorm", x, eps)
