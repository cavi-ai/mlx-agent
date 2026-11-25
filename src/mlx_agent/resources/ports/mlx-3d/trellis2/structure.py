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


def fit(grid, box):
    """Fit occupied bounds to a centered box, pooling on shrinking axes to retain thin geometry."""
    found = np.argwhere(grid)
    if not len(found):
        return grid
    low, high = found.min(axis=0), found.max(axis=0) + 1
    size = grid.shape[0]
    box = np.asarray(box, dtype=np.float64)
    spans = np.maximum(np.rint(box / box.max() * size), 1).astype(np.int64)
    starts = (size - spans) // 2
    occupied = grid[tuple(slice(start, stop) for start, stop in zip(low, high))]
    for axis, span in enumerate(spans):
        width = occupied.shape[axis]
        if span < width:
            # Partition the full source axis into nonempty bins, including both ends.
            edges = (np.arange(span) * width + span // 2) // span
            occupied = np.logical_or.reduceat(occupied, edges, axis=axis)
        elif span > width:
            picks = ((np.arange(span) + 0.5) * width / span).astype(np.intp)
            occupied = np.take(occupied, picks, axis=axis)
    fitted = np.zeros_like(grid)
    fitted[tuple(slice(start, start + span) for start, span in zip(starts, spans))] = occupied
    return fitted


def occupied(logits, resolution, box=None):
    """Integer (x, y, z) coordinates of occupied voxels at ``resolution``, pooled from the decoder grid after
    fitting it to ``box`` proportions when given."""
    grid = np.array(logits[0] > 0)
    if box is not None:
        grid = fit(grid, box)
    ratio = grid.shape[0] // resolution
    if ratio > 1:
        grid = grid.reshape(resolution, ratio, resolution, ratio, resolution, ratio).any(axis=(1, 3, 5))
    return np.argwhere(grid).astype(np.int32)
