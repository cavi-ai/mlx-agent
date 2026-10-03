"""Bounded KV-cache arithmetic for context-aware memory fit."""

from __future__ import annotations


_MAX_INT = 2 ** 63 - 1
MAX_ARCH_VALUE = 1000000
DEFAULT_DTYPE_BYTES = 2
CONTEXT_MIN, CONTEXT_MAX = 1024, 1048576


def extract_architecture(config):
    """Return bounded architecture facts from a Hugging Face config object."""
    if not isinstance(config, dict):
        return None

    def _positive_int(key):
        value = config.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 < value <= MAX_ARCH_VALUE
        ):
            return None
        return value

    layers = _positive_int("num_hidden_layers")
    attention_heads = _positive_int("num_attention_heads")
    kv_heads = _positive_int("num_key_value_heads") or attention_heads
    head_dim = _positive_int("head_dim")
    if head_dim is None:
        hidden_size = _positive_int("hidden_size")
        if hidden_size is not None and attention_heads:
            if hidden_size % attention_heads != 0:
                return None
            head_dim = hidden_size // attention_heads
    if layers is None or kv_heads is None or head_dim is None:
        return None
    return {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": _positive_int("max_position_embeddings"),
    }


def kv_bytes_per_token(architecture, dtype_bytes=DEFAULT_DTYPE_BYTES):
    if not isinstance(architecture, dict):
        return None
    layers = architecture.get("layers")
    kv_heads = architecture.get("kv_heads")
    head_dim = architecture.get("head_dim")
    if not all(isinstance(value, int) and value > 0 for value in (layers, kv_heads, head_dim)):
        return None
    return min(2 * layers * kv_heads * head_dim * dtype_bytes, _MAX_INT)


def kv_cache_bytes(architecture, context_tokens, dtype_bytes=DEFAULT_DTYPE_BYTES):
    per_token = kv_bytes_per_token(architecture, dtype_bytes)
    if per_token is None:
        return None
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool):
        raise TypeError("context_tokens must be an integer")
    if not CONTEXT_MIN <= context_tokens <= CONTEXT_MAX:
        raise ValueError(
            "context_tokens must be between {0} and {1}".format(CONTEXT_MIN, CONTEXT_MAX)
        )
    return min(per_token * context_tokens, _MAX_INT)


def max_context_tokens(architecture, weight_bytes, budget_bytes, dtype_bytes=DEFAULT_DTYPE_BYTES):
    """Largest context whose weights + KV fit the budget; None when unknown."""
    per_token = kv_bytes_per_token(architecture, dtype_bytes)
    if per_token is None:
        return None
    if not isinstance(weight_bytes, (int, float)) or not isinstance(budget_bytes, (int, float)):
        return None
    if weight_bytes < 0 or budget_bytes <= 0:
        return None
    headroom = budget_bytes - weight_bytes
    if headroom <= 0:
        return 0
    maximum = int(headroom // per_token)
    cap = architecture.get("max_position_embeddings")
    if isinstance(cap, int) and cap > 0:
        maximum = min(maximum, cap)
    return min(maximum, CONTEXT_MAX)
