"""Frame-local causal prefix reuse for the pinned MiniMax music backend."""

import mlx.core as mx
import mlx.nn as nn


class _DepthCache(nn.Module):
    """Append one codebook position; reset for the next frame's two-position prefix."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        object.__setattr__(self, "_prefix", None)

    @property
    def projection(self):
        return self.decoder.projection

    @property
    def audio_embeddings(self):
        return self.decoder.audio_embeddings

    @property
    def audio_heads(self):
        return self.decoder.audio_heads

    def __call__(self, embeddings):
        batch, length, _ = embeddings.shape
        previous = self._prefix
        if (length == 2 or previous is None or length != previous[0] + 1
                or batch != previous[1] or embeddings.dtype != previous[2]):
            offset, past, outputs = 0, [None] * len(self.decoder.layers), None
        else:
            offset, past, outputs = previous[0], previous[3], previous[4]
        hidden = embeddings[:, offset:] + self.decoder.pos_embedding(mx.arange(offset, length))
        keys_values = []
        for layer, cached in zip(self.decoder.layers, past):
            normalized = layer.input_layernorm(hidden)
            attention = layer.attn
            heads, dim, count = attention.heads, attention.head_dim, hidden.shape[1]
            q, k, v = (project(normalized).reshape(batch, count, heads, dim).transpose(0, 2, 1, 3)
                       for project in (attention.to_q, attention.to_k, attention.to_v))
            if cached is not None:
                q, k, v = (mx.concatenate([old, new], axis=2) for old, new in zip(cached, (q, k, v)))
            keys_values.append((q, k, v))
            # Preserve prefix-shaped attention: its GPU rounding is closer to the
            # original than single-query SDPA. Projections/FFN only visit new rows.
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim ** -0.5, mask="causal")
            attended = attended[:, :, -count:]
            hidden = hidden + attention.to_out(attended.transpose(0, 2, 1, 3).reshape(batch, count, -1))
            normalized = layer.post_attention_layernorm(hidden)
            hidden = hidden + layer.down_proj(nn.silu(layer.gate_proj(normalized)) * layer.up_proj(normalized))
        hidden = self.decoder.norm(hidden)
        if outputs is not None:
            hidden = mx.concatenate([outputs, hidden], axis=1)
        object.__setattr__(self, "_prefix", (length, batch, embeddings.dtype, keys_values, hidden))
        return hidden


def configure_depth_cache(model):
    """Wrap a supported decoder once; never change installed backend code or weights."""
    if model.model_type != "minimax_music3":
        return False
    decoder = model.rvq_depth_decoder
    if isinstance(decoder, _DepthCache):
        return True
    from mlx_audio.music.models.minimax_music3.depth import RVQDepthDecoder
    if type(decoder) is not RVQDepthDecoder:
        return False
    model.rvq_depth_decoder = _DepthCache(decoder)
    return True
