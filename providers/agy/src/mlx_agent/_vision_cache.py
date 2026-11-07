"""Bounded, model-owned image features; GPU imports occur only when applied."""
from collections import OrderedDict
import hashlib
import json
import logging

MAX_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 20


def content_key(parts):
    digest = hashlib.sha256()
    for dtype, shape, buffer in parts:
        header = json.dumps([dtype, list(shape), len(buffer)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(buffer)
    return digest.digest()


class FeatureCache:
    def __init__(self, max_bytes=MAX_BYTES, max_entries=MAX_ENTRIES):
        if type(max_bytes) is not int or type(max_entries) is not int or min(max_bytes, max_entries) <= 0:
            raise ValueError("Vision cache bounds must be positive integers.")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.encodings = 0

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key, value, size):
        if type(size) is not int or not 0 < size <= self.max_bytes:
            return False
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
            self.bytes -= self.entries.popitem(last=False)[1][1]
        self.entries[key] = (value, size)
        self.bytes += size
        return True

    def clear(self):
        self.entries.clear()
        self.bytes = 0


def apply(model, max_bytes=MAX_BYTES, max_entries=MAX_ENTRIES):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm.models.qwen2_5_vl.qwen2_5_vl import Model as Qwen25Model
    from mlx_vlm.models.qwen2_5_vl.vision import VisionModel as Qwen25VisionModel
    from mlx_vlm.models.qwen3_vl.qwen3_vl import Model
    from mlx_vlm.models.qwen3_vl.vision import VisionModel
    from mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe import Model as MoEModel
    from mlx_vlm.models.qwen3_vl_moe.vision import VisionModel as MoEVisionModel

    towers = {Qwen25Model: Qwen25VisionModel, Model: VisionModel, MoEModel: MoEVisionModel}
    if (type(model) not in towers or model.training or hasattr(model, "_mlx_agent_vision_cache")
            or type(model.vision_tower) is not towers[type(model)]):
        return False
    cache = FeatureCache(max_bytes, max_entries)
    original = model.vision_tower
    base = type(original)

    class CachedVisionTower(base):
        def __init__(self):
            # Preserve native parameter paths and share their storage without reinitializing weights.
            nn.Module.__init__(self)
            dict.update(self, original)
            self.__dict__.update(original.__dict__)

        def __call__(self, pixels, grid, **kwargs):
            if model.training or self.training:
                cache.clear()
                return base.__call__(self, pixels, grid, **kwargs)
            native_kwargs = not kwargs or (base is Qwen25VisionModel and kwargs == {"output_hidden_states": False})
            if (not native_kwargs or not isinstance(pixels, mx.array)
                    or pixels.dtype not in (mx.float16, mx.bfloat16, mx.float32)
                    or not isinstance(grid, mx.array) or grid.ndim != 2 or grid.shape[-1] != 3):
                return base.__call__(self, pixels, grid, **kwargs)
            mx.eval(pixels, grid)
            parts = []
            for value in (pixels, grid):
                view = memoryview(value)
                buffer = view.cast("B") if view.c_contiguous else view.tobytes(order="C")
                parts.append((str(value.dtype), value.shape, buffer))
            key = content_key(parts)
            features = cache.get(key)
            hit = features is not None
            if features is None:
                features = base.__call__(self, pixels, grid, **kwargs)
                mx.eval(features)
                cache.encodings += 1
                cache.put(key, features, sum(v.nbytes for _, v in tree_flatten(features)))
            logging.getLogger(__name__).info("Vision feature cache: hit=%s entries=%d bytes=%d encodings=%d",
                                             hit, len(cache.entries), cache.bytes, cache.encodings)
            # Reuse all DeepStack tensors; native embedding assembly still handles masks and MRoPE.
            return tree_map(lambda value: value, features)

    model.vision_tower = CachedVisionTower()
    object.__setattr__(model, "_mlx_agent_vision_cache", cache)
    return True
