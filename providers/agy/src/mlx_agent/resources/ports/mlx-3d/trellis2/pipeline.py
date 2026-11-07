"""Offline image-to-mesh generation for TRELLIS.2; model evaluation runs entirely on MLX."""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from . import dinov3, dual_grid, flow, sparse, structure
from ..triposg.geometry import decimate, write_glb
from ..triposg.weights import Weights

COMPONENTS = ("image_encoder", "ss_flow", "ss_decoder", "shape_flow_512", "shape_flow_1024", "shape_decoder")
RESOLUTIONS = (512, 1024, 1536)
MAX_TOKENS = 49152
CASCADE_FLOOR = 1024
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


def load(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "trellis2":
        raise ValueError("Pass a converted TRELLIS.2 model directory.")
    quantization = config.get("quantization", {})
    weights = {}
    for name in COMPONENTS:
        values = mx.load(str(path / (name + ".safetensors")))
        weights[name] = Weights(_kernels(values) if name == "shape_decoder" else values, **quantization)
    mx.eval(*[value for component in weights.values() for value in component.values.values()])
    return config, weights


def _noise(rng, shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def _denormalize(latent, normalization):
    return latent * mx.array(normalization["std"], dtype=mx.float32) + mx.array(normalization["mean"], dtype=mx.float32)


def _sampler(params, steps, guidance):
    return dict(steps=steps, guidance_strength=guidance, guidance_rescale=params["guidance_rescale"],
                guidance_interval=params["guidance_interval"], rescale_t=params["rescale_t"])


def cascade_coords(coords, low, high, max_tokens=MAX_TOKENS):
    """High-resolution latent coordinates from upsampled low-resolution voxels, lowering ``high`` to fit the token cap."""
    coords = np.asarray(coords, dtype=np.float64)
    while True:
        quantized = np.unique(((coords + 0.5) / low * (high // 16)).astype(np.int32), axis=0)
        if len(quantized) < max_tokens or high == CASCADE_FLOOR:
            return quantized, high
        high -= 128


def generate(model, image, out, steps=12, seed=42, resolution=1024, guidance=7.5, max_faces=500_000):
    config, weights = model
    if resolution not in RESOLUTIONS:
        raise ValueError("TRELLIS.2 resolution must be 512, 1024 or 1536.")
    settings = config["pipeline"]
    started = time.perf_counter()
    stages = {}
    mx.reset_peak_memory()
    rng = np.random.default_rng(seed)
    with Image.open(image) as source:
        source.load()
        picture = dinov3.crop_object(source)
    encoder = config["components"]["image_encoder"]
    cond_low = dinov3.encode(weights["image_encoder"], encoder, dinov3.pixels(picture, LOW_RESOLUTION))
    cond_high = None
    if resolution > LOW_RESOLUTION:
        cond_high = dinov3.encode(weights["image_encoder"], encoder, dinov3.pixels(picture, CASCADE_FLOOR))
    cursor = time.perf_counter()
    stages["conditioning"] = cursor - started

    ss = config["components"]["ss_flow"]
    grid = ss["resolution"]
    noise = _noise(rng, (1, ss["in_channels"], grid ** 3)).transpose(0, 2, 1)
    rope = flow.rope_tables(flow.grid_positions(grid), ss["model_channels"] // ss["num_heads"])
    latent = flow.sample(weights["ss_flow"], ss, noise, cond_low, mx.zeros_like(cond_low), rope,
                         **_sampler(settings["sparse_structure_sampler"], steps, guidance))
    logits = structure.decode(weights["ss_decoder"], config["components"]["ss_decoder"],
                              latent.reshape(1, grid, grid, grid, -1))
    coords = structure.occupied(logits, LOW_RESOLUTION // 16)
    previous, cursor = cursor, time.perf_counter()
    stages["sparse_structure"] = cursor - previous

    shape_params = _sampler(settings["shape_slat_sampler"], steps, guidance)
    normalization = settings["shape_slat_normalization"]
    low = config["components"]["shape_flow_512"]
    head_dim = low["model_channels"] // low["num_heads"]
    slat = flow.sample(weights["shape_flow_512"], low, _noise(rng, (1, len(coords), low["in_channels"])),
                       cond_low, mx.zeros_like(cond_low), flow.rope_tables(coords, head_dim), **shape_params)
    slat = _denormalize(slat[0], normalization)
    previous, cursor = cursor, time.perf_counter()
    stages["shape_latent_512"] = cursor - previous
    decoder = config["components"]["shape_decoder"]
    level_resolution = LOW_RESOLUTION
    if resolution > LOW_RESOLUTION:
        upsampled = sparse.run(weights["shape_decoder"], decoder, slat, coords, stop_level=4)
        coords, level_resolution = cascade_coords(upsampled, LOW_RESOLUTION, resolution)
        high = config["components"]["shape_flow_1024"]
        slat = flow.sample(weights["shape_flow_1024"], high, _noise(rng, (1, len(coords), high["in_channels"])),
                           cond_high, mx.zeros_like(cond_high), flow.rope_tables(coords, head_dim), **shape_params)
        slat = _denormalize(slat[0], normalization)
        previous, cursor = cursor, time.perf_counter()
        stages["shape_latent_high"] = cursor - previous
    voxels, vertices, crossed, split = sparse.decode(weights["shape_decoder"], decoder, slat, coords)
    previous, cursor = cursor, time.perf_counter()
    stages["decode"] = cursor - previous
    vertices, faces = dual_grid.to_mesh(voxels, vertices, crossed, split, level_resolution)
    vertices = gltf_axes(vertices)
    previous, cursor = cursor, time.perf_counter()
    stages["mesh"] = cursor - previous
    vertices, faces = decimate(vertices, faces, max_faces)
    previous, cursor = cursor, time.perf_counter()
    stages["decimation"] = cursor - previous
    write_glb(out, vertices, faces)
    stages["export"] = time.perf_counter() - cursor
    return {"path": str(out), "seconds": round(time.perf_counter() - started, 3), "steps": steps,
            "seed": seed, "resolution": level_resolution, "vertices": len(vertices), "faces": len(faces),
            "voxels": len(voxels), "stage_seconds": {name: round(value, 3) for name, value in stages.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3)}
