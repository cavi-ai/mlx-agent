"""Channels-last image operations with PyTorch's padding, resampling and normalization conventions."""

import math

import mlx.core as mx
import numpy as np


def _padded_index(length, pad, mode):
    index = np.arange(-pad, length + pad)
    if mode == "reflect":
        index = np.abs(index)
        index = np.where(index > length - 1, 2 * (length - 1) - index, index)
    else:
        index = np.clip(index, 0, length - 1)
    return mx.array(index.astype(np.int32))


def pad(x, amount, mode):
    """Pad height and width of (B, H, W, C) by ``amount``: ``zeros``, ``reflect`` or ``replicate``."""
    if amount == 0:
        return x
    if mode == "zeros":
        return mx.pad(x, [(0, 0), (amount, amount), (amount, amount), (0, 0)])
    x = mx.take(x, _padded_index(x.shape[1], amount, mode), axis=1)
    return mx.take(x, _padded_index(x.shape[2], amount, mode), axis=2)


def conv2d(w, name, x, padding_mode="zeros", stride=1):
    """A PyTorch ``Conv2d`` (weight O, I, kh, kw) with same padding on channels-last input."""
    kernel = w[name + ".weight"].transpose(0, 2, 3, 1)
    out = mx.conv2d(pad(x, kernel.shape[1] // 2, padding_mode), kernel, stride=stride)
    bias = w.values.get(name + ".bias")
    return out if bias is None else out + bias


def group_norm(w, name, x, groups, eps=1e-5):
    batch, height, width, channels = x.shape
    grouped = x.reshape(batch, height * width, groups, channels // groups)
    mean = grouped.mean(axis=(1, 3), keepdims=True)
    var = ((grouped - mean) ** 2).mean(axis=(1, 3), keepdims=True)
    x = ((grouped - mean) / mx.sqrt(var + eps)).reshape(x.shape)
    return x * w[name + ".weight"] + w[name + ".bias"]


def linear_weights(source, target, scale=None, antialias=False):
    """(target, source) bilinear weights, ``align_corners=False``; antialiased when shrinking, as PyTorch does."""
    scale = source / target if scale is None else scale
    weights = np.zeros((target, source), dtype=np.float64)
    if antialias and scale > 1:
        support = scale
        for out in range(target):
            center = scale * (out + 0.5)
            low, high = max(int(center - support + 0.5), 0), min(int(center + support + 0.5), source)
            taps = np.maximum(0, 1 - np.abs((np.arange(low, high) - center + 0.5) / scale))
            weights[out, low:high] = taps / taps.sum()
        return weights
    for out in range(target):
        position = max((out + 0.5) * scale - 0.5, 0.0)
        low = min(int(math.floor(position)), source - 1)
        high = min(low + 1, source - 1)
        frac = position - low
        weights[out, low] += 1 - frac
        weights[out, high] += frac
    return weights


def resize(x, size, antialias=False):
    """Bilinear resize of (B, H, W, C) to ``size`` (H, W) like ``F.interpolate(mode="bilinear", align_corners=False)``."""
    rows = mx.array(linear_weights(x.shape[1], size[0], antialias=antialias).astype(np.float32))
    cols = mx.array(linear_weights(x.shape[2], size[1], antialias=antialias).astype(np.float32))
    x = mx.einsum("ih,bhwc->biwc", rows, x)
    return mx.einsum("jw,biwc->bijc", cols, x)


def _pool_weights(source, target):
    weights = np.zeros((target, source), dtype=np.float32)
    for out in range(target):
        low, high = (out * source) // target, -((-(out + 1) * source) // target)
        weights[out, low:high] = 1 / (high - low)
    return mx.array(weights)


def adaptive_avg_pool(x, size):
    """``F.adaptive_avg_pool2d`` of (B, H, W, C) to ``size`` (H, W)."""
    if (x.shape[1], x.shape[2]) == tuple(size):
        return x
    x = mx.einsum("ih,bhwc->biwc", _pool_weights(x.shape[1], size[0]), x)
    return mx.einsum("jw,biwc->bijc", _pool_weights(x.shape[2], size[1]), x)


def _cubic(distance, a=-0.75):
    distance = np.abs(distance)
    near = ((a + 2) * distance - (a + 3)) * distance * distance + 1
    far = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
    return np.where(distance <= 1, near, np.where(distance < 2, far, 0.0))


def _cubic_weights(source, target, scale):
    weights = np.zeros((target, source), dtype=np.float64)
    for out in range(target):
        position = (out + 0.5) * scale - 0.5
        base = math.floor(position)
        for tap in range(-1, 3):
            weights[out, min(max(base + tap, 0), source - 1)] += _cubic(position - (base + tap))
    return weights


def bicubic_scale(grid, scale_factor):
    """``F.interpolate(mode="bicubic", scale_factor=..., align_corners=False)`` of an (H, W, C) NumPy grid."""
    rows = _cubic_weights(grid.shape[0], int(math.floor(grid.shape[0] * scale_factor[0])), 1 / scale_factor[0])
    cols = _cubic_weights(grid.shape[1], int(math.floor(grid.shape[1] * scale_factor[1])), 1 / scale_factor[1])
    return np.einsum("ih,jw,hwc->ijc", rows, cols, grid.astype(np.float64)).astype(np.float32)
