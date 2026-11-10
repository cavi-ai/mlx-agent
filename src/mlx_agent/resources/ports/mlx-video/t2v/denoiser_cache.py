"""Bounded, first-block-gated residual forecasting for the Wan2.1 denoiser.

The policy has no framework dependency. MLX modules are imported only when a
session is attached; retained activations never enter a model parameter tree.
"""

import math
from collections import Counter


class DenoiserCachePolicy:
    def __init__(self, steps, threshold=0.0125, max_reuses=3, warmup=3, final_steps=2,
                 memory_limit_bytes=256 * 1024 ** 2):
        for value in (steps, max_reuses, warmup, final_steps, memory_limit_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("cache counts and memory limit must be positive integers")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("cache threshold must be finite and nonnegative")
        self.steps, self.threshold, self.max_reuses = steps, threshold, max_reuses
        self.warmup, self.final_steps = warmup, final_steps
        self.memory_limit_bytes = memory_limit_bytes
        self.calls = self.reused = self.streak = self.age = self.peak_bytes = 0
        self.accumulated = 0.0
        self.signature = None
        self.reasons = Counter()

    def decide(self, error, signature, cache_bytes):
        """Return whether this call may reuse an established residual."""
        index = self.calls
        self.calls += 1
        if signature != self.signature:
            self.signature, self.age, self.streak, self.accumulated = signature, 0, 0, 0.0
        self.age += 1
        if cache_bytes > self.memory_limit_bytes or cache_bytes <= 0:
            reason = "memory_limit"
        elif not math.isfinite(error) or error < 0:
            reason = "invalid_indicator"
            self.age = 0
        else:
            self.peak_bytes = max(self.peak_bytes, cache_bytes)
            self.accumulated += error
            if self.age <= self.warmup:
                reason = "warmup"
            elif index >= self.steps - self.final_steps:
                reason = "final_refresh"
            elif self.streak >= self.max_reuses:
                reason = "reuse_limit"
            elif self.threshold == 0 or self.accumulated >= self.threshold:
                reason = "threshold"
            else:
                self.reused += 1
                self.streak += 1
                return True
        self.streak, self.accumulated = 0, 0.0
        self.reasons[reason] += 1
        return False

    def stats(self):
        return {"mode": "adaptive", "threshold": self.threshold,
                "max_reuses": self.max_reuses,
                "full_steps": self.calls - self.reused, "reused_steps": self.reused,
                "peak_cache_bytes": self.peak_bytes, "memory_limit_bytes": self.memory_limit_bytes,
                "refresh_reasons": dict(self.reasons)}


def required_cache_bytes(config, width, height, frames):
    """Six float32 feature buffers, conservatively reserving batched CFG."""
    stride = config.get("vae_stride", (4, 8, 8))
    patch = config.get("patch_size", (1, 2, 2))
    length = ((frames - 1) // stride[0] + 1) * (height // stride[1]) * (width // stride[2])
    length = math.ceil(length / (patch[1] * patch[2]))
    return 6 * 2 * length * config["dim"] * 4


class WanDenoiserCache:
    """One generation's state; close restores blocks and releases all features."""

    def __init__(self, steps, memory_limit_bytes, **profile):
        profile.setdefault("max_reuses", 1)
        self.policy = DenoiserCachePolicy(steps, memory_limit_bytes=memory_limit_bytes, **profile)
        self.model = None
        self.original_blocks = None
        self.anchors = []
        self.current_probe = self.residual = self.first_output = None
        self.current_time = 0.0
        self.signature = None
        self.reuse = False

    def attach(self, model):
        import mlx.core as mx
        import mlx.nn as nn

        if self.model is not None or len(model.blocks) < 2:
            raise ValueError("cache requires one Wan model with at least two blocks")
        session = self
        originals = model.blocks
        cached = getattr(model, "_mlx_agent_compiled_blocks", None)
        identities = tuple(id(block) for block in originals)
        if cached is None or cached[0] != identities:
            cached = (identities, tuple(mx.compile(block) for block in originals))
            object.__setattr__(model, "_mlx_agent_compiled_blocks", cached)

        class Block(nn.Module):
            def __init__(self, index, block, compiled):
                super().__init__()
                self.block = block
                object.__setattr__(self, "index", index)
                object.__setattr__(self, "compiled", compiled)

            @property
            def cross_attn(self):
                return self.block.cross_attn

            def __call__(self, x, **kwargs):
                if self.index == 0:
                    output = self.compiled(x, **kwargs)
                    signature = (id(kwargs["context"]), id(kwargs.get("cross_kv_cache")),
                                 id(kwargs.get("rope_cos_sin")), x.shape, str(x.dtype),
                                 tuple(kwargs["seq_lens"]), tuple(kwargs["grid_sizes"]))
                    probe = output - x
                    same = signature == session.signature
                    if not same:
                        session.anchors.clear()
                    session.residual = None
                    error = 0.0
                    factor = 0.0
                    if len(session.anchors) == 2:
                        earlier, latest = session.anchors
                        gap = latest[0] - earlier[0]
                        factor = (session.current_time - latest[0]) / gap if gap else 0.0
                        predicted = latest[1] + factor * (latest[1] - earlier[1])
                        error = (mx.mean(mx.abs(probe.astype(mx.float32) - predicted.astype(mx.float32))) /
                                 mx.maximum(mx.mean(mx.abs(probe.astype(mx.float32))), 1e-6)).item()
                        if not gap or not math.isfinite(factor) or not 0 < factor <= 2:
                            error = math.inf
                    size = 6 * probe.nbytes
                    session.reuse = session.policy.decide(error, signature, size)
                    session.reuse = session.reuse and same and len(session.anchors) == 2
                    session.signature = signature
                    if session.reuse:
                        session.residual = latest[2] + factor * (latest[2] - earlier[2])
                    elif size <= session.policy.memory_limit_bytes:
                        mx.async_eval(probe)
                        session.current_probe, session.first_output = probe, output
                    else:
                        session.anchors.clear()
                        session.current_probe = session.first_output = None
                    return output
                if session.reuse:
                    return x + session.residual if self.index == len(originals) - 1 else x
                output = self.compiled(x, **kwargs)
                if self.index == len(originals) - 1 and session.first_output is not None:
                    residual = output - session.first_output
                    mx.async_eval(residual)
                    session.anchors = (session.anchors + [(session.current_time, session.current_probe, residual)])[-2:]
                    session.current_probe = session.first_output = None
                return output

        class Forward:
            # Override any retained monolithic graph, while keeping native helpers.
            def __getattr__(self, name):
                return getattr(model, name)

            def _compiled(self, *args, **kwargs):
                t = kwargs.get("t")
                session.current_time = float(t.reshape(-1)[0].item()) if t is not None else float(session.policy.calls)
                return model(*args, **kwargs)

        wrappers = [Block(i, block, cached[1][i]) for i, block in enumerate(originals)]
        self.model, self.original_blocks = model, originals
        model.blocks = wrappers
        return Forward()

    def close(self):
        if self.model is not None:
            self.model.blocks = self.original_blocks
        self.model = self.original_blocks = None
        self.anchors.clear()
        self.current_probe = self.residual = self.first_output = self.signature = None

    def stats(self):
        return self.policy.stats()
