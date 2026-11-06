"""Sparse voxel decoder: submanifold 3x3x3 convolution, ConvNeXt blocks and channel-to-space subdivision."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GATHER_ELEMENTS = 1 << 27
OFFSETS = np.array([(v // 9 - 1, v // 3 % 3 - 1, v % 3 - 1) for v in range(27)], dtype=np.int64)


def neighbor_map(coords):
    """(27, N) source rows for each kernel offset of each voxel; N marks an empty neighbor."""
    coords = np.asarray(coords, dtype=np.int64)
    count = len(coords)
    span = int(coords.max()) + 3 if count else 3
    keys = ((coords[:, 0] + 1) * span + coords[:, 1] + 1) * span + coords[:, 2] + 1
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    rows = np.empty((27, count), dtype=np.int32)
    for index, (dx, dy, dz) in enumerate(OFFSETS):
        query = keys + (dx * span + dy) * span + dz
        position = np.minimum(np.searchsorted(ordered, query), max(count - 1, 0))
        hit = ordered[position] == query if count else np.zeros(0, dtype=bool)
        rows[index] = np.where(hit, order[position], count)
    return mx.array(rows)


class Voxels:
    """Features (N, C) at integer coordinates (N, 3) of one level, with that level's neighbor map."""

    def __init__(self, feats, coords, neighbors=None):
        self.feats, self.coords, self._neighbors = feats, np.asarray(coords, dtype=np.int32), neighbors

    @property
    def neighbors(self):
        if self._neighbors is None:
            self._neighbors = neighbor_map(self.coords)
        return self._neighbors

    def replace(self, feats):
        return Voxels(feats, self.coords, self._neighbors)


def conv_kernel(weight):
    """(Co, 3, 3, 3, Ci) checkpoint layout to (27, Ci, Co); offset v = 9 * dx + 3 * dy + dz (each +1)."""
    co, ci = weight.shape[0], weight.shape[-1]
    return weight.reshape(co, 27, ci).transpose(1, 2, 0)


def sparse_conv(w, name, voxels):
    kernel = w.values.get(name + ".kernel")
    if kernel is None:
        kernel = conv_kernel(w[name + ".weight"])
    feats, rows = voxels.feats, voxels.neighbors
    count, channels = feats.shape
    padded = mx.concatenate([feats, mx.zeros((1, channels), dtype=feats.dtype)], axis=0)
    flat = kernel.reshape(27 * channels, -1)
    chunk = max(1, GATHER_ELEMENTS // (27 * channels))
    outputs = []
    for start in range(0, count, chunk):
        gathered = padded[rows[:, start:start + chunk].T].reshape(-1, 27 * channels)
        part = gathered @ flat + w[name + ".bias"]
        if count > chunk:
            mx.eval(part)
        outputs.append(part)
    return voxels.replace(outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0))


def _norm(w, name, x, eps=1e-6):
    return w.norm(name, x, eps)


def convnext_block(w, name, voxels):
    h = sparse_conv(w, name + ".conv", voxels).feats
    h = _norm(w, name + ".norm", h)
    h = w.linear(name + ".mlp.2", nn.silu(w.linear(name + ".mlp.0", h)))
    return voxels.replace(h + voxels.feats)


def subdivide(voxels, feats, keep):
    """Split each voxel into its kept octants; octant bit i offsets axis i. Returns the child features of ``feats``."""
    keep = np.asarray(keep, dtype=bool)
    parent, child = np.nonzero(keep)
    bits = np.stack([child % 2, child // 2 % 2, child // 4 % 2], axis=1).astype(np.int32)
    coords = voxels.coords[parent] * 2 + bits
    rows = mx.array((parent * 8 + child).astype(np.int32))
    split = [value.reshape(value.shape[0] * 8, -1)[rows] for value in feats]
    return coords, split


def up_block(w, name, voxels, out_channels):
    """Channel-to-space residual block: predicts which octants exist and moves channels into them."""
    subdiv = w.linear(name + ".to_subdiv", voxels.feats)
    h = nn.silu(_norm(w, name + ".norm1", voxels.feats))
    h = sparse_conv(w, name + ".conv1", voxels.replace(h)).feats
    coords, (h, skip) = subdivide(voxels, (h, voxels.feats), np.array(subdiv > 0))
    child = Voxels(h, coords)
    h = nn.silu(mx.fast.layer_norm(h, None, None, 1e-6))
    h = sparse_conv(w, name + ".conv2", child.replace(h)).feats
    repeat = out_channels // skip.shape[1]
    skip = mx.repeat(skip, repeat, axis=1) if repeat > 1 else skip
    return child.replace(h + skip)


def _levels(config):
    channels, blocks = config["model_channels"], config["num_blocks"]
    return [(level, channels[level], blocks[level]) for level in range(len(blocks))]


def run(w, config, feats, coords, stop_level=None):
    """Decoder trunk: returns the last level's voxels, or the coordinates reached at ``stop_level``."""
    voxels = Voxels(w.linear("from_latent", feats), coords)
    channels = config["model_channels"]
    for level, width, count in _levels(config):
        if level == stop_level:
            return voxels.coords
        for index in range(count):
            voxels = convnext_block(w, "blocks.{0}.{1}".format(level, index), voxels)
            mx.eval(voxels.feats)
        if level < len(channels) - 1:
            voxels = up_block(w, "blocks.{0}.{1}".format(level, count), voxels, channels[level + 1])
            mx.eval(voxels.feats)
    return voxels


def decode(w, config, feats, coords):
    """Per-voxel dual vertex (3), edge crossing flags (3) and quad split weight (1) at the output level."""
    voxels = run(w, config, feats, coords)
    h = w.linear("output_layer", mx.fast.layer_norm(voxels.feats, None, None, 1e-5))
    margin = 0.5
    vertices = (1 + 2 * margin) * mx.sigmoid(h[:, 0:3]) - margin
    crossed = h[:, 3:6] > 0
    split = nn.softplus(h[:, 6:7])
    mx.eval(vertices, crossed, split)
    return voxels.coords, np.array(vertices), np.array(crossed), np.array(split)
