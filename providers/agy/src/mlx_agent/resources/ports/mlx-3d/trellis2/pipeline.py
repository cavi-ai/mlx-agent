"""Offline image-to-mesh generation for TRELLIS.2; model evaluation runs entirely on MLX."""

import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import mlx.core as mx
import numpy as np
from PIL import Image

from . import closed, components, dinov3, dual_grid, flow, sparse, structure
from .convert import CASCADE_FLOOR, MAX_TOKENS
from .guide import fit, fit_size, place, surface_coords
from ..triposg.geometry import decimate, write_glb
from ..triposg.weights import Weights

COMPONENTS = ("image_encoder", "ss_flow", "ss_decoder", "shape_flow_512", "shape_flow_1024", "shape_decoder")
TEXTURE_COMPONENTS = components.TEXTURE
DECODERS = ("shape_decoder", "tex_decoder")
RESOLUTIONS = (512, 1024, 1536)
# DINOv3 image size that conditions the high-resolution flows.
HIGH_CONDITION_PIXELS = 1024
LOW_RESOLUTION = 512


def gltf_axes(vertices):
    """z-up mesh vertices in glTF axes: y up, with the -y side facing +z, glTF's front."""
    return np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1)


def _kernels(values):
    """Sparse-conv weights in the gather layout once, instead of per call."""
    out = {}
    for name, value in values.items():
        if name.endswith(".weight") and value.ndim == 5:
            out[name[:-len("weight")] + "kernel"] = sparse.conv_kernel(value)
        else:
            out[name] = value
    return out


def textured(config):
    """Whether the converted model carries the texture stage."""
    return components.textured(config)


def srgb_to_linear(color):
    color = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
    return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4).astype(np.float32)


def vertex_materials(attributes):
    """Linear base color and glTF occlusion-roughness-metallic per vertex from decoded base color, metallic,
    roughness (channels 0-4); occlusion is 1."""
    attributes = np.asarray(attributes, dtype=np.float32)
    channels = np.clip(attributes[:, [4, 3]], 0.0, 1.0)
    occlusion = np.ones((len(attributes), 1), dtype=np.float32)
    return srgb_to_linear(attributes[:, :3]), np.concatenate([occlusion, channels], axis=1)


def voxel_colors(points, coords, colors, grid, reach=3):
    """The color of the voxel holding each point (P, 3) in the unit cube centered on the origin, else of the closest
    occupied voxel within ``reach`` cells, else the mean color."""
    coords = np.asarray(coords, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.float32)
    cells = np.clip(np.floor((np.asarray(points, dtype=np.float64) + 0.5) * grid).astype(np.int64), 0, grid - 1)
    span = grid + 2 * reach
    keys = ((coords[:, 0] + reach) * span + coords[:, 1] + reach) * span + coords[:, 2] + reach
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    out = np.tile(colors.mean(axis=0), (len(cells), 1))
    pending = np.arange(len(cells))
    side = np.arange(-reach, reach + 1)
    offsets = np.stack(np.meshgrid(side, side, side, indexing="ij"), axis=-1).reshape(-1, 3)
    distance = np.abs(offsets).max(axis=1)
    for radius in range(reach + 1):
        ring = offsets[distance == radius]
        ring = ring[np.argsort((ring ** 2).sum(axis=1), kind="stable")]
        for offset in ring:
            if not len(pending):
                return out
            cell = cells[pending] + reach + offset
            query = (cell[:, 0] * span + cell[:, 1]) * span + cell[:, 2]
            position = np.minimum(np.searchsorted(ordered, query), len(ordered) - 1)
            hit = ordered[position] == query
            out[pending[hit]] = colors[order[position[hit]]]
            pending = pending[~hit]
    return out


class Model(NamedTuple):
    config: dict
    weights: object

    def component_statistics(self):
        return self.weights.statistics()


class ComponentWeights(dict):
    def __init__(self, path, config):
        super().__init__()
        self.path, self.config = Path(path), config
        self.loads, self.seconds, self.bytes = {}, {}, {}

    def statistics(self):
        return {'schema': 'resident-components/1', 'resident': sorted(self),
                'loads': dict(self.loads), 'load_seconds': dict(self.seconds),
                'checkpoint_bytes': dict(self.bytes)}


def load(path):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != 'trellis2':
        raise ValueError('Pass a converted TRELLIS.2 model directory.')
    return Model(config, ComponentWeights(path, config))


def prepare(model, resolution):
    config, weights = model
    needed = components.required_components(config, resolution)
    missing = [name for name in needed if name not in weights]
    sizes = components.checkpoint_bytes(weights.path, missing)
    pending, times = {}, {}
    for name in missing:
        started = time.monotonic()
        values = mx.load(str(weights.path / (name + '.safetensors')))
        component = config['components'].get(name, {})
        if 'dtype' in component:
            values = flow.block_values(values, flow.compute_dtype(component))
        if name in DECODERS:
            values = _kernels(sparse.block_values(values, sparse.compute_dtype(component)))
        prepared = Weights(values, **config.get('quantization', {}))
        mx.eval(*prepared.values.values())
        pending[name] = prepared
        times[name] = time.monotonic() - started
    weights.update(pending)
    for name in missing:
        weights.loads[name] = weights.loads.get(name, 0) + 1
        weights.seconds[name] = weights.seconds.get(name, 0) + times[name]
        weights.bytes[name] = sizes[name]
    return model


def _noise(rng, shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def _denormalize(latent, normalization):
    return latent * mx.array(normalization["std"], dtype=mx.float32) + mx.array(normalization["mean"], dtype=mx.float32)


def _sampler(params, steps, guidance):
    return dict(steps=steps, guidance_strength=guidance, guidance_rescale=params["guidance_rescale"],
                guidance_interval=params["guidance_interval"], rescale_t=params["rescale_t"])


def cascade_settings(config):
    """The converted model's ``mesh`` token cap and cascade floor, else the conversion defaults."""
    mesh = config.get("mesh") if isinstance(config.get("mesh"), dict) else {}
    return mesh.get("max_tokens", MAX_TOKENS), mesh.get("cascade_floor", CASCADE_FLOOR)


def cascade_coords(coords, low, high, max_tokens=MAX_TOKENS, floor=CASCADE_FLOOR):
    """High-resolution latent coordinates from upsampled low-resolution voxels, lowering ``high`` by 128 to fit the
    token cap while it stays at or above ``floor``."""
    coords = np.asarray(coords, dtype=np.float64)
    while True:
        quantized = np.unique(((coords + 0.5) / low * (high // 16)).astype(np.int32), axis=0)
        if len(quantized) <= max_tokens or high - 128 < floor:
            return quantized, high
        high -= 128


def surface(voxels, vertices, crossed, split, resolution):
    """Closed glTF-axis mesh from decoded dual-grid voxels, with the seconds of its ``mesh`` and ``closed_surface``
    stages. These CPU stages share unified memory with MLX, so the buffer cache the GPU stages filled is freed first."""
    mx.clear_cache()
    started = time.perf_counter()
    vertices, faces = dual_grid.to_mesh(voxels, vertices, crossed, split, resolution)
    meshed = time.perf_counter()
    vertices, faces = closed.closed_surface(vertices, faces, resolution)
    return gltf_axes(vertices), faces, {"mesh": meshed - started, "closed_surface": time.perf_counter() - meshed}


def _write_mesh(out, vertices, faces, colors, materials, bbox):
    if bbox is None:
        write_glb(out, vertices, faces, colors, materials)
        return
    with TemporaryDirectory(prefix=".trellis2-", dir=Path(out).parent) as directory:
        staged = Path(directory) / "mesh.glb"
        write_glb(staged, vertices, faces, colors, materials)
        # TRELLIS (x, y, z) extents become glTF (x, z, y) extents.
        fit(staged, (bbox[0], bbox[2], bbox[1]))
        # Publish the complete file without replacing an output claimed by another job.
        os.link(staged, out, follow_symlinks=False)


def generate(model, image, out, steps=12, seed=42, resolution=1024, guidance=7.5, max_faces=500_000, views=(),
             guide=None, bbox=None):
    """``views`` are more images of the object, one per sampler step in turn; a ``guide`` GLB's surface voxels
    replace the sampled sparse structure. ``bbox`` sizes (x, y, z, z up) fit the sampled sparse structure to their
    proportions and the mesh bounds to those sizes, centered on the origin."""
    config, weights = model
    if resolution not in RESOLUTIONS:
        raise ValueError("TRELLIS.2 resolution must be 512, 1024 or 1536.")
    if bbox is not None:
        if guide is not None:
            raise ValueError("TRELLIS.2 bbox takes three positive sizes and no guide.")
        bbox = fit_size(bbox).tolist()
    prepare(model, resolution)
    settings = config["pipeline"]
    started = time.perf_counter()
    stages = {}
    mx.reset_peak_memory()
    rng = np.random.default_rng(seed)
    pictures = []
    for path in (image, *views):
        with Image.open(path) as source:
            source.load()
            pictures.append(dinov3.crop_object(source))
    encoder = config["components"]["image_encoder"]
    cond_low = [dinov3.encode(weights["image_encoder"], encoder, dinov3.pixels(picture, LOW_RESOLUTION))
                for picture in pictures]
    cond_high = None
    if resolution > LOW_RESOLUTION:
        cond_high = [dinov3.encode(weights["image_encoder"], encoder, dinov3.pixels(picture, HIGH_CONDITION_PIXELS))
                     for picture in pictures]
    cursor = time.perf_counter()
    stages["conditioning"] = cursor - started

    if guide is None:
        ss = config["components"]["ss_flow"]
        grid = ss["resolution"]
        noise = _noise(rng, (1, ss["in_channels"], grid ** 3)).transpose(0, 2, 1)
        rope = flow.rope_tables(flow.grid_positions(grid), ss["model_channels"] // ss["num_heads"])
        latent = flow.sample(weights["ss_flow"], ss, noise, cond_low, mx.zeros_like(cond_low[0]), rope,
                             **_sampler(settings["sparse_structure_sampler"], steps, guidance))
        logits = structure.decode(weights["ss_decoder"], config["components"]["ss_decoder"],
                                  latent.reshape(1, grid, grid, grid, -1))
        coords = structure.occupied(logits, LOW_RESOLUTION // 16, box=bbox)
    else:
        coords, guide_center, guide_extent = surface_coords(guide, LOW_RESOLUTION // 16)
    previous, cursor = cursor, time.perf_counter()
    stages["sparse_structure" if guide is None else "guide"] = cursor - previous

    shape_params = _sampler(settings["shape_slat_sampler"], steps, guidance)
    normalization = settings["shape_slat_normalization"]
    low = config["components"]["shape_flow_512"]
    head_dim = low["model_channels"] // low["num_heads"]
    slat = flow.sample(weights["shape_flow_512"], low, _noise(rng, (1, len(coords), low["in_channels"])),
                       cond_low, mx.zeros_like(cond_low[0]), flow.rope_tables(coords, head_dim), **shape_params)
    slat = _denormalize(slat[0], normalization)
    previous, cursor = cursor, time.perf_counter()
    stages["shape_latent_512"] = cursor - previous
    decoder = config["components"]["shape_decoder"]
    level_resolution = LOW_RESOLUTION
    if resolution > LOW_RESOLUTION:
        upsampled = sparse.run(weights["shape_decoder"], decoder, slat, coords, stop_level=4)
        coords, level_resolution = cascade_coords(upsampled, LOW_RESOLUTION, resolution, *cascade_settings(config))
        high = config["components"]["shape_flow_1024"]
        slat = flow.sample(weights["shape_flow_1024"], high, _noise(rng, (1, len(coords), high["in_channels"])),
                           cond_high, mx.zeros_like(cond_high[0]), flow.rope_tables(coords, head_dim), **shape_params)
        slat = _denormalize(slat[0], normalization)
        previous, cursor = cursor, time.perf_counter()
        stages["shape_latent_high"] = cursor - previous
    paint = textured(config)
    texture_latent = None
    if paint:
        name = "tex_flow_512" if resolution == LOW_RESOLUTION else "tex_flow_1024"
        tex, cond = config["components"][name], cond_low if cond_high is None else cond_high
        shape = (slat - mx.array(normalization["mean"], dtype=mx.float32)) / mx.array(normalization["std"],
                                                                                    dtype=mx.float32)
        params = settings["tex_slat_sampler"]
        noise = _noise(rng, (1, len(coords), tex["in_channels"] - shape.shape[-1]))
        texture_latent = flow.sample(weights[name], tex, noise, cond, mx.zeros_like(cond[0]),
                                     flow.rope_tables(coords, head_dim), extra=shape[None],
                                     **_sampler(params, steps, params["guidance_strength"]))
        texture_latent = _denormalize(texture_latent[0], settings["tex_slat_normalization"])
        previous, cursor = cursor, time.perf_counter()
        stages["texture_latent"] = cursor - previous
    subdivisions = []
    voxels, vertices, crossed, split = sparse.decode(weights["shape_decoder"], decoder, slat, coords, subdivisions)
    previous, cursor = cursor, time.perf_counter()
    stages["decode"] = cursor - previous
    attributes = None
    if paint:
        _, attributes = sparse.decode_attributes(weights["tex_decoder"], config["components"]["tex_decoder"],
                                                 texture_latent, coords, subdivisions)
        previous, cursor = cursor, time.perf_counter()
        stages["texture_decode"] = cursor - previous
    vertices, faces, seconds = surface(voxels, vertices, crossed, split, level_resolution)
    stages.update(seconds)
    cursor = time.perf_counter()
    vertices, faces = closed.without_fins(*decimate(vertices, faces, max_faces, aggression=3))
    previous, cursor = cursor, time.perf_counter()
    stages["decimation"] = cursor - previous
    colors = materials = None
    if paint:
        upright = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1)
        colors, materials = vertex_materials(voxel_colors(upright, voxels, attributes[:, :5], level_resolution))
    _write_mesh(out, vertices, faces, colors, materials, bbox)
    stages["export"] = time.perf_counter() - cursor
    if guide is not None:
        # The guided mesh replaces the guide in place, ready to bake onto it.
        place(out, guide_center, guide_extent)
    return {"path": str(out), "seconds": round(time.perf_counter() - started, 3), "steps": steps,
            "views": len(pictures), "guided": guide is not None, "bbox": bbox,
            "seed": seed, "resolution": level_resolution, "vertices": len(vertices), "faces": len(faces),
            "voxels": len(voxels), "vertex_colors": paint, "pbr": paint,
            "stage_seconds": {name: round(value, 3) for name, value in stages.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3)}
