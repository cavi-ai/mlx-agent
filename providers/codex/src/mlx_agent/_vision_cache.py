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
    import numpy as np
    from mlx_vlm.models.qwen2_5_vl.qwen2_5_vl import Model

    if (type(model) is not Model or model.training or hasattr(model, "_mlx_agent_vision_cache")):
        return False
    cache = FeatureCache(max_bytes, max_entries)
    original = model.get_input_embeddings

    def embeddings(input_ids=None, pixel_values=None, **kwargs):
        if model.training or model.vision_tower.training:
            cache.clear()
            return original(input_ids, pixel_values, **kwargs)
        grid = kwargs.get("image_grid_thw")
        if (pixel_values is None or not isinstance(pixel_values, mx.array)
                or pixel_values.dtype not in (mx.float16, mx.float32)
                or not isinstance(grid, mx.array) or grid.ndim != 2 or grid.shape[-1] != 3
                or kwargs.get("video_grid_thw") is not None or kwargs.get("pixel_values_videos") is not None
                or kwargs.get("cached_image_features") is not None):
            return original(input_ids, pixel_values, **kwargs)
        arrays = [np.ascontiguousarray(value) for value in (pixel_values, grid)]
        key = content_key((str(a.dtype), a.shape, memoryview(a).cast("B")) for a in arrays)
        features = cache.get(key)
        hit = features is not None
        if features is None:
            pixels = pixel_values.astype(model.vision_tower.patch_embed.proj.weight.dtype)
            features = model.vision_tower(pixels, grid, output_hidden_states=False)
            mx.eval(features)
            cache.encodings += 1
            cache.put(key, features, features.nbytes)
        logging.getLogger(__name__).info("Vision feature cache: hit=%s entries=%d bytes=%d encodings=%d",
                                         hit, len(cache.entries), cache.bytes, cache.encodings)
        return original(input_ids, pixel_values, **dict(kwargs, cached_image_features=features))

    object.__setattr__(model, "_mlx_agent_vision_cache", cache)
    object.__setattr__(model, "get_input_embeddings", embeddings)
    return True
