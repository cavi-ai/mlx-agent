"""Image preprocessing and DINOv3 ViT features in the Hugging Face checkpoint layout."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_INPUT_SIZE = 1024
ALPHA_THRESHOLD = 0.8 * 255


def crop_object(image, margin=1.0):
    """Square crop around the opaque object, ``margin`` times its larger side, RGB premultiplied by alpha;
    an opaque image keeps its full frame."""
    image = image.convert("RGBA")
    scale = min(1, MAX_INPUT_SIZE / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    pixels = np.asarray(image)
    found = np.argwhere(pixels[:, :, 3] > ALPHA_THRESHOLD)
    if not len(found):
        raise ValueError("The input image has no opaque object.")
    left, top = found[:, 1].min(), found[:, 0].min()
    right, bottom = found[:, 1].max(), found[:, 0].max()
    center = (left + right) / 2, (top + bottom) / 2
    size = int(max(right - left, bottom - top) * margin)
    box = (center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2)
    rgba = np.asarray(image.crop(box)).astype(np.float32) / 255
    return Image.fromarray((rgba[:, :, :3] * rgba[:, :, 3:4] * 255).astype(np.uint8))


def pixels(image, size):
    """ImageNet-normalized NHWC pixels of a square image resized to ``size``."""
    resized = image.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
    values = np.asarray(resized, dtype=np.float32) / 255
    return mx.array(((values - IMAGE_MEAN) / IMAGE_STD)[None])


def rope_tables(height, width, head_dim, theta):
    """Cosine and sine tables for the patch tokens: (y, x) patch centers in [-1, 1], halves rotated."""
    inv_freq = 1 / theta ** np.arange(0, 1, 4 / head_dim, dtype=np.float32)
    rows = (np.arange(0.5, height, dtype=np.float32) / height) * 2 - 1
    cols = (np.arange(0.5, width, dtype=np.float32) / width) * 2 - 1
    centers = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1).reshape(-1, 2)
    angles = (2 * math.pi * centers[:, :, None] * inv_freq[None, None]).reshape(len(centers), -1)
    angles = np.tile(angles, 2)
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


def _with_rope(x, cos, sin):
    prefix = x.shape[2] - cos.shape[0]
    return mx.concatenate([x[:, :, :prefix], _rotate(x[:, :, prefix:], cos, sin)], axis=2)


def encode(w, config, image_pixels):
    """Hidden states of every layer's output, layer-normalized without affine (CLS, registers, patches)."""
    patch = config["patch_size"]
    kernel = w["embeddings.patch_embeddings.weight"].transpose(0, 2, 3, 1)
    x = mx.conv2d(image_pixels, kernel, stride=patch) + w["embeddings.patch_embeddings.bias"]
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height * width, channels)
    prefix = [mx.broadcast_to(w["embeddings.cls_token"], (batch, 1, channels)),
              mx.broadcast_to(w["embeddings.register_tokens"], (batch, config["num_register_tokens"], channels))]
    x = mx.concatenate(prefix + [x], axis=1)
    heads = config["num_attention_heads"]
    head_dim = channels // heads
    cos, sin = rope_tables(height, width, head_dim, config["rope_theta"])
    eps = config["layer_norm_eps"]
    act = nn.gelu if config.get("hidden_act", "gelu") == "gelu" else getattr(nn, config["hidden_act"])
    for index in range(config["num_hidden_layers"]):
        name = "layer.{0}".format(index)
        h = w.norm(name + ".norm1", x, eps)
        q, k, v = [w.linear(name + ".attention." + key, h).reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)
                   for key in ("q_proj", "k_proj", "v_proj")]
        q, k = _with_rope(q, cos, sin), _with_rope(k, cos, sin)
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim ** -0.5)
        h = w.linear(name + ".attention.o_proj", h.transpose(0, 2, 1, 3).reshape(batch, -1, channels))
        x = x + h * w[name + ".layer_scale1.lambda1"]
        h = w.norm(name + ".norm2", x, eps)
        if config.get("use_gated_mlp"):
            h = act(w.linear(name + ".mlp.gate_proj", h)) * w.linear(name + ".mlp.up_proj", h)
        else:
            h = act(w.linear(name + ".mlp.up_proj", h))
        x = x + w.linear(name + ".mlp.down_proj", h) * w[name + ".layer_scale2.lambda1"]
        mx.eval(x)
    return mx.fast.layer_norm(x, None, None, 1e-5)
