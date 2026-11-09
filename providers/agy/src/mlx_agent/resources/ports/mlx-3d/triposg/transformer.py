"""TripoSG denoising transformer and VAE decoder, evaluated with MLX."""

import math

import mlx.core as mx
import mlx.nn as nn


def _cross_keys_values(w, name, context, heads, dim, qk_norm, cross_norm):
    if cross_norm:
        context = w.norm(name + ".norm_cross", context)
    k, v = (w.linear(name + ".to_" + key, context) for key in ("k", "v"))
    batch = context.shape[0]
    k, v = mx.split(mx.concatenate([k, v], axis=-1).reshape(batch, -1, heads, 2 * dim), 2, axis=-1)
    k, v = (a.reshape(batch, -1, heads, dim).transpose(0, 2, 1, 3) for a in (k, v))
    return (w.rms(name + ".norm_k", k) if qk_norm else k), v


class _CrossAttentionCache:
    """Fixed conditioning for one generation or point decoder, never a resident model."""

    def __init__(self, weights, context):
        self.weights, self.context = weights, context
        self.values = {}

    def get(self, w, name, context, heads, dim, qk_norm, cross_norm):
        if w is not self.weights or context is not self.context:
            raise ValueError("Cross-attention cache belongs to another conditioning input.")
        key = name, heads, dim, qk_norm, cross_norm
        if key not in self.values:
            k, v = _cross_keys_values(w, name, context, heads, dim, qk_norm, cross_norm)
            mx.eval(k, v)
            self.values[key] = k, v
        return self.values[key]


def attention(w, name, x, heads, context=None, qk_norm=False, cross_norm=False, context_cache=None):
    cross = context is not None
    context = x if context is None else context
    q = w.linear(name + ".to_q", x)
    batch, length, width = q.shape
    dim = width // heads
    # The released checkpoint concatenates projections, then splits within each
    # head. Splitting the projection into ordinary heads changes the model.
    if cross:
        get = _cross_keys_values if context_cache is None else context_cache.get
        k, v = get(w, name, context, heads, dim, qk_norm, cross_norm)
        q = q.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)
        if qk_norm:
            q = w.rms(name + ".norm_q", q)
    else:
        if cross_norm:
            context = w.norm(name + ".norm_cross", context)
        k, v = (w.linear(name + ".to_" + key, context) for key in ("k", "v"))
        q, k, v = mx.split(mx.concatenate([q, k, v], axis=-1).reshape(batch, length, heads, 3 * dim), 3, axis=-1)
        q, k, v = [a.reshape(batch, -1, heads, dim).transpose(0, 2, 1, 3) for a in (q, k, v)]
        if qk_norm:
            q, k = w.rms(name + ".norm_q", q), w.rms(name + ".norm_k", k)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim ** -0.5)
    return w.linear(name + ".to_out.0", y.transpose(0, 2, 1, 3).reshape(batch, length, width))


def block(w, name, x, heads, context=None, self_attention=True, qk_norm=False, skip=None, context_cache=None):
    if skip is not None:
        x = w.norm(name + ".skip_norm", w.linear(name + ".skip_linear", mx.concatenate([skip, x], axis=-1)))
    if self_attention:
        x = x + attention(w, name + ".attn1", w.norm(name + ".norm1", x), heads, qk_norm=qk_norm)
    if context is not None:
        x = x + attention(w, name + ".attn2", w.norm(name + ".norm2", x), heads,
                          context=context, qk_norm=qk_norm, cross_norm=not qk_norm, context_cache=context_cache)
    return x + w.feed_forward(name + ".ff", w.norm(name + ".norm3", x))


def denoise(w, config, latents, timestep, image_features, context_cache=None):
    width, layers, heads = config["width"], config["num_layers"], config["num_attention_heads"]
    half = width // 2
    frequencies = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
    angles = mx.array(timestep, dtype=mx.float32).reshape(-1, 1) * frequencies
    embedding = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1).astype(latents.dtype)
    embedding = w.linear("time_proj.linear_2", nn.gelu(w.linear("time_proj.linear_1", embedding)))
    x = mx.concatenate([embedding[:, None, :], w.linear("proj_in", latents)], axis=1)
    skips = []
    for index in range(layers):
        x = block(w, "blocks." + str(index), x, heads, context=image_features,
                  qk_norm=True, skip=skips.pop() if index > layers // 2 else None,
                  context_cache=context_cache)
        if index < layers // 2:
            skips.append(x)
        mx.eval(x)
    return w.linear("proj_out", w.norm("norm_out", x)[:, 1:])


def position_embedding(points, frequency, include_pi=False):
    frequencies = 2.0 ** mx.arange(frequency, dtype=mx.float32)
    if include_pi:
        frequencies = frequencies * math.pi
    angles = (points[..., None] * frequencies).reshape(*points.shape[:-1], -1)
    return mx.concatenate([points, mx.sin(angles), mx.cos(angles)], axis=-1) if frequency else points


class Decoder:
    def __init__(self, weights, config, latents):
        self.weights, self.config = weights, config
        self.context = weights.linear("post_quant", latents)
        for index in range(config["num_layers_decoder"]):
            self.context = block(weights, "decoder.blocks." + str(index), self.context,
                                 config["num_attention_heads"])
            mx.eval(self.context)
        self.context_cache = _CrossAttentionCache(weights, self.context)

    def __call__(self, points):
        w, config = self.weights, self.config
        query = position_embedding(points, config["embed_frequency"], config.get("embed_include_pi", False))
        query = w.linear("decoder.proj_query", query)
        query = block(w, "decoder.blocks." + str(config["num_layers_decoder"]), query,
                      config["num_attention_heads"], context=self.context, self_attention=False,
                      context_cache=self.context_cache)
        return -w.linear("decoder.proj_out", w.norm("decoder.norm_out", query))
