"""TRELLIS.2 flow transformers (sparse structure and structured latent) and the guidance-interval Euler sampler."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

FREQUENCY_SIZE = 256
SIGMA_MIN = 1e-5


def timestep_embedding(t, dim=FREQUENCY_SIZE, max_period=10000):
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def rope_tables(positions, head_dim, theta=10000.0):
    """Per-token rotation of (even, odd) channel pairs from integer (x, y, z) positions; unused pairs stay fixed."""
    positions = np.asarray(positions, dtype=np.float32)
    freq_dim = head_dim // 2 // positions.shape[1]
    freqs = 1.0 / theta ** (np.arange(freq_dim, dtype=np.float32) / freq_dim)
    angles = (positions[:, :, None] * freqs[None, None]).reshape(len(positions), -1)
    angles = np.pad(angles, ((0, 0), (0, head_dim // 2 - angles.shape[1])))
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


def grid_positions(resolution):
    axes = np.arange(resolution)
    return np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _rotate(x, cos, sin):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    return mx.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1).reshape(x.shape)


def _head_norm(x, gamma):
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.maximum(norm, 1e-12) * gamma[:, None, :] * math.sqrt(x.shape[-1])


def _layer_norm(x, eps=1e-6):
    return mx.fast.layer_norm(x, None, None, eps)


def _block(w, name, x, mod, context, cos, sin, heads, proj=None):
    """One shared-modulation block; with ``proj``, per-token image features add to the cross-attention output."""
    batch, tokens, channels = x.shape
    cross = name + (".cross_attn" if proj is None else ".cross_attn.cross_attn_block")
    head_dim = channels // heads
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
        w[name + ".modulation"] + mod, 6, axis=-1)
    h = _layer_norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    q, k, v = mx.split(w.linear(name + ".self_attn.to_qkv", h).reshape(batch, tokens, 3, heads, head_dim), 3, axis=2)
    q, k, v = (value.squeeze(2).transpose(0, 2, 1, 3) for value in (q, k, v))
    q = _rotate(_head_norm(q, w[name + ".self_attn.q_rms_norm.gamma"]), cos, sin)
    k = _rotate(_head_norm(k, w[name + ".self_attn.k_rms_norm.gamma"]), cos, sin)
    h = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim ** -0.5)
    h = w.linear(name + ".self_attn.to_out", h.transpose(0, 2, 1, 3).reshape(batch, tokens, channels))
    x = x + h * gate_msa[:, None]
    h = w.norm(name + ".norm2", x, 1e-6)
    q = w.linear(cross + ".to_q", h).reshape(batch, tokens, heads, head_dim).transpose(0, 2, 1, 3)
    kv = w.linear(cross + ".to_kv", context).reshape(batch, -1, 2, heads, head_dim)
    k, v = (value.squeeze(2).transpose(0, 2, 1, 3) for value in mx.split(kv, 2, axis=2))
    q = _head_norm(q, w[cross + ".q_rms_norm.gamma"])
    k = _head_norm(k, w[cross + ".k_rms_norm.gamma"])
    h = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim ** -0.5)
    h = w.linear(cross + ".to_out", h.transpose(0, 2, 1, 3).reshape(batch, tokens, channels))
    if proj is not None:
        h = h + w.linear(name + ".cross_attn.proj_linear", proj)
    x = x + h
    h = _layer_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    h = w.linear(name + ".mlp.mlp.2", nn.gelu_approx(w.linear(name + ".mlp.mlp.0", h)))
    return x + h * gate_mlp[:, None]


def velocity(w, config, x, t, context, rope, proj=None):
    """Predicted flow velocity for tokens ``x`` (B, N, in) at timesteps ``t`` (B,) given image tokens ``context``
    and, for projection-conditioned models, image features ``proj`` (B, N, P) aligned with the tokens."""
    cos, sin = rope
    h = w.linear("input_layer", x)
    mod = w.linear("t_embedder.mlp.0", timestep_embedding(t))
    mod = w.linear("t_embedder.mlp.2", nn.silu(mod))
    mod = w.linear("adaLN_modulation.1", nn.silu(mod))
    for index in range(config["num_blocks"]):
        h = _block(w, "blocks.{0}".format(index), h, mod, context, cos, sin, config["num_heads"], proj)
    return w.linear("out_layer", _layer_norm(h, 1e-5))


def schedule(steps, rescale_t):
    t = np.linspace(1, 0, steps + 1)
    t = rescale_t * t / (1 + (rescale_t - 1) * t)
    return [(float(t[i]), float(t[i + 1])) for i in range(steps)]


def _guided(x_t, t, positive, negative, strength, rescale):
    pred = strength * positive + (1 - strength) * negative
    if rescale <= 0:
        return pred
    scale = SIGMA_MIN + (1 - SIGMA_MIN) * t
    x0_pos = (1 - SIGMA_MIN) * x_t - scale * positive
    x0_cfg = (1 - SIGMA_MIN) * x_t - scale * pred
    ratio = mx.sqrt(mx.var(x0_pos)) / mx.sqrt(mx.var(x0_cfg))
    x0 = rescale * x0_cfg * ratio + (1 - rescale) * x0_cfg
    return ((1 - SIGMA_MIN) * x_t - x0) / scale


def sample(w, config, noise, cond, neg_cond, rope, steps, guidance_strength, guidance_rescale,
           guidance_interval, rescale_t, extra=None, proj=None):
    """Euler integration from t=1 to 0; classifier-free guidance only inside ``guidance_interval``.
    The unconditional branch of a projection-conditioned model sees zero projected features."""
    x = noise
    both_proj = None if proj is None else mx.concatenate([proj, mx.zeros_like(proj)], axis=0)
    for t, t_prev in schedule(steps, rescale_t):
        guided = guidance_strength != 1 and guidance_interval[0] <= t <= guidance_interval[1]
        inputs = x if extra is None else mx.concatenate([x, extra], axis=-1)
        if guided:
            both = velocity(w, config, mx.concatenate([inputs, inputs], axis=0),
                            mx.array([1000 * t, 1000 * t], dtype=mx.float32),
                            mx.concatenate([cond, neg_cond], axis=0), rope, both_proj)
            v = _guided(x, t, both[:1], both[1:], guidance_strength, guidance_rescale)
        else:
            v = velocity(w, config, inputs, mx.array([1000 * t], dtype=mx.float32), cond, rope, proj)
        x = x - (t - t_prev) * v
        mx.eval(x)
    return x
