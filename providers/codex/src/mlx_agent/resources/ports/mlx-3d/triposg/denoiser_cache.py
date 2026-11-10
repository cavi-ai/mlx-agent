"""Per-generation, first-block-gated residual forecasts for TripoSG."""
import math
from collections import Counter


class DenoiserCache:
    def __init__(self, steps, memory_limit_bytes=256 * 1024 ** 2, threshold=.0125):
        if type(steps) is not int or steps < 1 or type(memory_limit_bytes) is not int or memory_limit_bytes < 1:
            raise ValueError('Cache steps and byte limit must be positive integers.')
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('Cache threshold must be finite and nonnegative.')
        self.steps, self.memory_limit_bytes, self.threshold = steps, memory_limit_bytes, threshold
        self.calls = self.reused = self.age = self.streak = self.peak_bytes = 0
        self.accumulated = 0.0
        self.signature = None
        self.owners = None
        self.anchors = []
        self.current_probe = self.first_output = None
        self.current_time = self.factor = 0.0
        self.reuse = False
        self.reasons = Counter()

    def decide(self, error, signature, cache_bytes):
        index = self.calls
        self.calls += 1
        if signature != self.signature:
            self.close()
            self.signature, self.age, self.streak, self.accumulated = signature, 0, 0, 0.0
        self.age += 1
        if cache_bytes <= 0 or cache_bytes > self.memory_limit_bytes:
            reason = 'memory_limit'
        elif not math.isfinite(error) or error < 0:
            reason = 'invalid_indicator'
            self.age = 0
        else:
            self.peak_bytes = max(self.peak_bytes, cache_bytes)
            self.accumulated += error
            if self.age <= 3:
                reason = 'warmup'
            elif index >= self.steps - 2:
                reason = 'final_refresh'
            elif self.streak >= 1:
                reason = 'reuse_limit'
            elif self.threshold == 0 or self.accumulated >= self.threshold:
                reason = 'threshold'
            else:
                self.streak += 1
                self.reused += 1
                return True
        self.streak, self.accumulated = 0, 0.0
        self.reasons[reason] += 1
        return False

    def begin(self, weights, features, first_input, first_output, timestep):
        import mlx.core as mx

        signature = (id(weights), id(features), first_output.shape, str(first_output.dtype))
        if signature != self.signature:
            self.anchors.clear()
        probe = first_output - first_input
        error, factor = 0.0, 0.0
        if len(self.anchors) == 2:
            earlier, latest = self.anchors
            gap = latest[0] - earlier[0]
            factor = (timestep - latest[0]) / gap if gap else 0.0
            predicted = latest[1] + factor * (latest[1] - earlier[1])
            error = (mx.mean(mx.abs(probe.astype(mx.float32) - predicted.astype(mx.float32))) /
                     mx.maximum(mx.mean(mx.abs(probe.astype(mx.float32))), 1e-6)).item()
            if not math.isfinite(factor) or not 0 < factor <= 2:
                error = math.inf
        if not math.isfinite(timestep):
            error = math.inf
        if len(self.anchors) < 2 and signature == self.signature and self.age >= 3:
            error = math.inf
        size = 6 * probe.nbytes
        self.reuse = self.decide(error, signature, size) and len(self.anchors) == 2
        self.current_time, self.factor = timestep, factor
        self.owners = weights, features
        if not self.reuse and size <= self.memory_limit_bytes:
            mx.async_eval(probe)
            self.current_probe, self.first_output = probe, first_output
        elif size > self.memory_limit_bytes:
            self.anchors.clear()
            self.current_probe = self.first_output = None
        return self.reuse

    def predict(self, first_output):
        earlier, latest = self.anchors
        return first_output + (latest[2] + self.factor * (latest[2] - earlier[2]))

    def record(self, output):
        import mlx.core as mx

        if self.first_output is not None:
            residual = output - self.first_output
            mx.async_eval(residual)
            self.anchors = (self.anchors + [(self.current_time, self.current_probe, residual)])[-2:]
            self.current_probe = self.first_output = None

    def close(self):
        self.anchors.clear()
        self.current_probe = self.first_output = self.owners = None
        self.signature = None
        self.reuse = False

    def stats(self):
        return {'mode':'adaptive', 'threshold':self.threshold, 'max_reuses':1,
                'full_steps':self.calls-self.reused, 'reused_steps':self.reused,
                'peak_cache_bytes':self.peak_bytes, 'memory_limit_bytes':self.memory_limit_bytes,
                'refresh_reasons':dict(self.reasons)}
