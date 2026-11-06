"""NAF: image-guided upsampling of patch features by neighborhood attention over the patch grid."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ops

DIM = 256
ATTENTION_HEADS = 4
ROPE_HEADS = 4
KERNEL = 9
GROUPS = 8


def _encoder(w, name, x):
    """A stem convolution, then two non-residual GroupNorm-SiLU-conv blocks, all with reflect padding."""
    x = ops.conv2d(w, name + ".0", x, "reflect")
    for index in (1, 2):
        block = "{0}.{1}".format(name, index)
        x = ops.conv2d(w, block + ".conv1", nn.silu(ops.group_norm(w, block + ".norm1", x, GROUPS)), "reflect")
        x = ops.conv2d(w, block + ".conv2", nn.silu(ops.group_norm(w, block + ".norm2", x, GROUPS)), "reflect")
    return x


def rope(x, periods, heads=ROPE_HEADS):
    """Axial rotary encoding of (B, H, W, C) on (row, column) coordinates in [-1, 1], halves rotated per head."""
    batch, height, width, channels = x.shape
    rows = np.arange(0.5, height, dtype=np.float32) / height * 2 - 1
    cols = np.arange(0.5, width, dtype=np.float32) / width * 2 - 1
    coords = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1).reshape(-1, 2)
    angles = 2 * math.pi * coords[:, :, None] / np.asarray(periods, dtype=np.float32)[None, None]
    angles = np.tile(angles.reshape(len(coords), -1), 2)
    cos, sin = mx.array(np.cos(angles)), mx.array(np.sin(angles))
    x = x.reshape(batch, height * width, heads, channels // heads)
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    x = x * cos[:, None] + rotated * sin[:, None]
    return x.reshape(batch, height, width, channels)


def guidance(w, image, size):
    """Query features (B, h, w, 256) of an RGB image in [0, 1] (B, H, W, 3) at the output ``size``."""
    height, width = image.shape[1:3]
    if height > 4 * size[0] or width > 4 * size[1]:
        image = ops.resize(image, (min(height, 4 * size[0], 4 * size[1]), min(width, 4 * size[1], 4 * size[0])))
    x = mx.concatenate([_encoder(w, "image_encoder.encoder", image),
                        _encoder(w, "image_encoder.sem_encoder", image)], axis=-1)
    return rope(ops.adaptive_avg_pool(x, size), w["image_encoder.rope.periods"])


def window_starts(cells, kernel=KERNEL):
    """First patch of each patch's window: centered, then shifted inside the grid at its borders."""
    return np.clip(np.arange(cells) - kernel // 2, 0, cells - kernel)


def attend(queries, keys, values, kernel=KERNEL, heads=ATTENTION_HEADS):
    """Each output pixel attends to the ``kernel`` x ``kernel`` patches around its own patch.

    ``queries`` (B, H, W, D) at pixel resolution, ``keys`` (B, h, w, D) and ``values`` (B, h, w, C) at patch
    resolution, with H and W whole multiples of h and w. Pixels of one patch share that patch's window.
    """
    batch, height, width, dim = queries.shape
    cells_y, cells_x = keys.shape[1:3]
    step_y, step_x = height // cells_y, width // cells_x
    if step_y * cells_y != height or step_x * cells_x != width or min(cells_y, cells_x) < kernel:
        raise ValueError("NAF needs an output size that is a whole multiple of a patch grid of at least 9.")
    rows = mx.array((window_starts(cells_y, kernel)[:, None] + np.arange(kernel)).astype(np.int32))
    cols = mx.array((window_starts(cells_x, kernel)[:, None] + np.arange(kernel)).astype(np.int32))
    head_dim, value_dim = dim // heads, values.shape[-1] // heads
    q = queries.reshape(batch, cells_y, step_y, cells_x, step_x, heads, head_dim)
    q = q.transpose(0, 1, 3, 5, 2, 4, 6).reshape(batch, cells_y, cells_x, heads, step_y * step_x, head_dim)
    out = []
    for row in range(cells_y):
        k = keys[:, rows[row]][:, :, cols].reshape(batch, kernel, cells_x, kernel, heads, head_dim)
        k = k.transpose(0, 2, 4, 1, 3, 5).reshape(batch, cells_x, heads, kernel * kernel, head_dim)
        v = values[:, rows[row]][:, :, cols].reshape(batch, kernel, cells_x, kernel, heads, value_dim)
        v = v.transpose(0, 2, 4, 1, 3, 5).reshape(batch, cells_x, heads, kernel * kernel, value_dim)
        scores = (q[:, row] @ k.transpose(0, 1, 2, 4, 3)) * head_dim ** -0.5
        part = mx.softmax(scores, axis=-1) @ v
        out.append(part)
        mx.eval(part)
    out = mx.stack(out, axis=1).reshape(batch, cells_y, cells_x, heads, step_y, step_x, value_dim)
    return out.transpose(0, 1, 4, 2, 5, 3, 6).reshape(batch, height, width, heads * value_dim)


def upsample(w, image, features, size):
    """Patch features (B, h, w, C) upsampled to ``size`` under the guidance of ``image`` (B, H, W, 3) in [0, 1]."""
    queries = guidance(w, image, size)
    keys = ops.adaptive_avg_pool(queries, features.shape[1:3])
    return attend(queries, keys, features)
