"""Dense 3D convolutional decoder from the sparse-structure latent to an occupancy grid (channels-last)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _conv(w, name, x):
    kernel = w[name + ".weight"].transpose(0, 2, 3, 4, 1)
    return mx.conv3d(x, kernel, padding=kernel.shape[1] // 2) + w[name + ".bias"]


def _norm(w, name, x):
    return w.norm(name, x, 1e-5)


def _res_block(w, name, x):
    h = _conv(w, name + ".conv1", nn.silu(_norm(w, name + ".norm1", x)))
    h = _conv(w, name + ".conv2", nn.silu(_norm(w, name + ".norm2", h)))
    return h + x


def pixel_shuffle(x, factor=2):
    """Channel ``c * f**3 + i * f**2 + j * f + k`` becomes voxel offset (i, j, k) of output channel ``c``."""
    batch, d1, d2, d3, channels = x.shape
    out = channels // factor ** 3
    x = x.reshape(batch, d1, d2, d3, out, factor, factor, factor)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(batch, d1 * factor, d2 * factor, d3 * factor, out)


def decode(w, config, latent):
    """Occupancy logits (B, R, R, R) for a latent (B, r, r, r, C)."""
    h = _conv(w, "input_layer", latent)
    for index in range(config["num_res_blocks_middle"]):
        h = _res_block(w, "middle_block.{0}".format(index), h)
    index = 0
    for level in range(len(config["channels"])):
        for _ in range(config["num_res_blocks"]):
            h = _res_block(w, "blocks.{0}".format(index), h)
            index += 1
        if level < len(config["channels"]) - 1:
            h = pixel_shuffle(_conv(w, "blocks.{0}.conv".format(index), h))
            index += 1
        mx.eval(h)
    h = _conv(w, "out_layer.2", nn.silu(_norm(w, "out_layer.0", h)))
    return h[..., 0]


def occupied(logits, resolution):
    """Integer (x, y, z) coordinates of occupied voxels at ``resolution``, pooled from the decoder grid."""
    grid = np.array(logits[0] > 0)
    ratio = grid.shape[0] // resolution
    if ratio > 1:
        grid = grid.reshape(resolution, ratio, resolution, ratio, resolution, ratio).any(axis=(1, 3, 5))
    return np.argwhere(grid).astype(np.int32)
