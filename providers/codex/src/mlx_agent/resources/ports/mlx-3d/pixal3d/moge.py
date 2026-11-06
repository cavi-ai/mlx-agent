"""MoGe-2 camera field of view: a DINOv2 ViT-L/14 encoder, convolutional neck and point and mask heads."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ops

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 14
HEADS = 16
EPS = 1e-6
POS_OFFSET = 0.1
LEVELS = 5
FIT_SIZE = 64
RESOLUTION_LEVEL = 9


def token_grid(height, width, num_tokens):
    aspect = width / height
    return round((num_tokens / aspect) ** 0.5), round((num_tokens * aspect) ** 0.5)


def view_plane_uv(width, height, aspect):
    """(height, width, 2) image-plane coordinates spanning the half diagonal, pixel centers, x first."""
    span_x, span_y = aspect / (1 + aspect ** 2) ** 0.5, 1 / (1 + aspect ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=np.float32)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=np.float32)
    return np.stack(np.meshgrid(u, v, indexing="xy"), axis=-1)


def _position_embedding(w, rows, cols):
    table = np.asarray(w["encoder.backbone.pos_embed"], dtype=np.float32)[0]
    side = int(round(math.sqrt(table.shape[0] - 1)))
    grid = table[1:].reshape(side, side, -1)
    grid = ops.bicubic_scale(grid, ((rows + POS_OFFSET) / side, (cols + POS_OFFSET) / side))
    if grid.shape[:2] != (rows, cols):
        raise ValueError("MoGe position embedding interpolation produced the wrong grid.")
    return mx.array(np.concatenate([table[:1], grid.reshape(rows * cols, -1)]))


def encode(w, config, image, rows, cols):
    """Summed 1x1 projections of the normalized intermediate ViT layers, (1, rows, cols, dim_out)."""
    x = ops.resize(image, (rows * PATCH, cols * PATCH), antialias=True)
    x = (x - mx.array(IMAGE_MEAN)) / mx.array(IMAGE_STD)
    kernel = w["encoder.backbone.patch_embed.proj.weight"].transpose(0, 2, 3, 1)
    x = mx.conv2d(x, kernel, stride=PATCH) + w["encoder.backbone.patch_embed.proj.bias"]
    channels = x.shape[-1]
    x = x.reshape(1, rows * cols, channels)
    x = mx.concatenate([w["encoder.backbone.cls_token"], x], axis=1) + _position_embedding(w, rows, cols)
    layers = config["encoder"]["intermediate_layers"]
    head_dim = channels // HEADS
    projected = None
    for index in range(max(layers) + 1):
        name = "encoder.backbone.blocks.{0}".format(index)
        h = w.norm(name + ".norm1", x, EPS)
        qkv = w.linear(name + ".attn.qkv", h).reshape(1, -1, 3, HEADS, head_dim).transpose(2, 0, 3, 1, 4)
        h = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=head_dim ** -0.5)
        h = w.linear(name + ".attn.proj", h.transpose(0, 2, 1, 3).reshape(1, -1, channels))
        x = x + h * w[name + ".ls1.gamma"]
        h = w.linear(name + ".mlp.fc2", nn.gelu(w.linear(name + ".mlp.fc1", w.norm(name + ".norm2", x, EPS))))
        x = x + h * w[name + ".ls2.gamma"]
        if index in layers:
            feature = w.norm("encoder.backbone.norm", x, EPS)[:, 1:].reshape(1, rows, cols, channels)
            feature = ops.conv2d(w, "encoder.output_projections.{0}".format(layers.index(index)), feature)
            projected = feature if projected is None else projected + feature
        mx.eval(x if projected is None else [x, projected])
    return projected


def conv_transpose2x(w, name, x):
    """``ConvTranspose2d(kernel_size=2, stride=2)`` (weight in, out, 2, 2) on channels-last input."""
    batch, height, width, _ = x.shape
    weight = w[name + ".weight"]
    y = mx.einsum("bhwc,coij->bhiwjo", x, weight).reshape(batch, 2 * height, 2 * width, weight.shape[1])
    return y + w[name + ".bias"]


def _resample(w, name, x, kind):
    if kind == "conv_transpose":
        x = conv_transpose2x(w, name + ".0", x)
    elif kind == "bilinear":
        x = ops.resize(x, (2 * x.shape[1], 2 * x.shape[2]))
    else:
        raise ValueError("MoGe resampler {0} is not supported.".format(kind))
    return ops.conv2d(w, name + ".1", x, "replicate")


def conv_stack(w, name, config, inputs):
    """A MoGe ``ConvStack``: per-level 1x1 inputs, residual blocks without norms, 2x resamplers, 1x1 outputs."""
    outputs, x = [], None
    for level in range(len(config["dim_res_blocks"])):
        feature = inputs[level]
        if config["dim_in"][level] is not None:
            feature = ops.conv2d(w, "{0}.input_blocks.{1}".format(name, level), feature)
        x = feature if x is None else x + feature
        for block in range(config["num_res_blocks"][level]):
            layers = "{0}.res_blocks.{1}.{2}.layers".format(name, level, block)
            h = ops.conv2d(w, layers + ".2", nn.relu(x), "replicate")
            x = x + ops.conv2d(w, layers + ".5", nn.relu(h), "replicate")
        out = x
        if config["dim_out"] is not None and config["dim_out"][level] is not None:
            out = ops.conv2d(w, "{0}.output_blocks.{1}".format(name, level), x)
        outputs.append(out)
        if level < len(config["dim_res_blocks"]) - 1:
            x = _resample(w, "{0}.resamplers.{1}".format(name, level), x, config["resamplers"][level])
        mx.eval(x)
    return outputs


def fit_indices(size, samples=FIT_SIZE):
    """Rows (or columns) that ``F.interpolate(mode="nearest")`` keeps when shrinking ``size`` to ``samples``."""
    scale = np.float32(size) / np.float32(samples)
    return np.minimum(np.floor(np.arange(samples, dtype=np.float32) * scale).astype(np.int64), size - 1)


def solve_focal(uv, points):
    """Focal length relative to the half diagonal that best maps ``points`` (N, 3) onto ``uv`` (N, 2)."""
    from scipy.optimize import least_squares

    xy, z = points[:, :2].astype(np.float64), points[:, 2].astype(np.float64)
    uv = uv.astype(np.float64)

    def residual(shift):
        projected = xy / (z + shift)[:, None]
        focal = (projected * uv).sum() / np.square(projected).sum()
        return (focal * projected - uv).ravel()

    shift = np.float32(least_squares(residual, x0=0, ftol=1e-3, method="lm")["x"].squeeze())
    projected = xy / (z + shift)[:, None]
    return float((projected * uv).sum() / np.square(projected).sum())


def camera_angle_x(w, config, picture):
    """Horizontal field of view in radians of an RGB image in [0, 1] (1, H, W, 3)."""
    height, width = picture.shape[1:3]
    aspect = width / height
    low, high = config["num_tokens_range"]
    rows, cols = token_grid(height, width, int(low + RESOLUTION_LEVEL / 9 * (high - low)))
    features = [mx.concatenate([encode(w, config, picture, rows, cols),
                                mx.array(view_plane_uv(cols, rows, aspect)[None])], axis=-1)]
    for level in range(1, LEVELS):
        features.append(mx.array(view_plane_uv(cols * 2 ** level, rows * 2 ** level, aspect)[None]))
    neck = conv_stack(w, "neck", config["neck"], features)
    points = conv_stack(w, "points_head", config["points_head"], neck)[-1][0]
    mask = conv_stack(w, "mask_head", config["mask_head"], neck)[-1][0]
    keep_rows, keep_cols = fit_indices(height), fit_indices(width)
    rows_weight = mx.array(ops.linear_weights(points.shape[0], height)[keep_rows].astype(np.float32))
    cols_weight = mx.array(ops.linear_weights(points.shape[1], width)[keep_cols].astype(np.float32))
    sample = lambda value: np.asarray(mx.einsum("ih,jw,hwc->ijc", rows_weight, cols_weight, value))
    points, mask = sample(points), sample(mask)[..., 0] > 0
    depth = np.exp(points[..., 2:])
    points = np.concatenate([points[..., :2] * depth, depth], axis=-1)
    uv = view_plane_uv(width, height, aspect)[keep_rows][:, keep_cols]
    focal = 1.0 if mask.sum() < 2 else solve_focal(uv[mask], points[mask])
    fx = focal / 2 * (1 + aspect ** 2) ** 0.5 / aspect
    return 2 * math.atan(1 / (2 * fx))
