"""Offline image-to-mesh generation for Pixal3D; model evaluation runs entirely on MLX."""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from . import moge, naf, projection
from ..trellis2 import dinov3, dual_grid, flow, sparse, structure
from ..trellis2.pipeline import _denormalize, _kernels, _noise, _sampler, gltf_axes
from ..triposg.geometry import decimate, write_glb
from ..triposg.weights import Weights

COMPONENTS = ("image_encoder", "ss_flow", "ss_decoder", "shape_flow_512", "shape_flow_1024", "shape_decoder",
              "moge", "naf")
RESOLUTIONS = (1024, 1536)
MAX_TOKENS = 49152
CASCADE_FLOOR = 1024
LOW_RESOLUTION = 512
CROP_MARGIN = 1.1
NAF_SIZE = (512, 512)


def gltf_vertices(vertices):
    """Projection-grid vertices in glTF axes, through the z-up world the input camera sees from -y,
    so the side facing the camera is the glTF front (+z)."""
    return gltf_axes(vertices @ projection.AXES.T)


def load(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "pixal3d":
        raise ValueError("Pass a converted Pixal3D model directory.")
    quantization = config.get("quantization", {})
    weights = {}
    for name in COMPONENTS:
        values = mx.load(str(path / (name + ".safetensors")))
        weights[name] = Weights(_kernels(values) if name == "shape_decoder" else values, **quantization)
    mx.eval(*[value for component in weights.values() for value in component.values.values()])
    return config, weights


def cascade_coords(coords, low, high, max_tokens=MAX_TOKENS):
    """Rounded high-resolution latent coordinates, lowering ``high`` by 128 until they fit the token cap."""
    coords = np.asarray(coords, dtype=np.float64)
    while True:
        quantized = np.unique(np.round((coords + 0.5) / low * (high // 16 - 1)).astype(np.int32), axis=0)
        if len(quantized) < max_tokens or high == CASCADE_FLOOR:
            return quantized, high
        high -= 128


class Condition:
    """DINOv3 tokens of one image size, with its NAF-upsampled patch map when the stage uses one."""

    def __init__(self, weights, config, picture, size, upsample):
        self.size = size
        resized = picture.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
        guide = mx.array((np.asarray(resized, dtype=np.float32) / 255)[None])
        encoder = config["components"]["image_encoder"]
        tokens = dinov3.encode(weights["image_encoder"], encoder,
                               (guide - mx.array(dinov3.IMAGE_MEAN)) / mx.array(dinov3.IMAGE_STD))
        cells, start = size // encoder["patch_size"], 1 + encoder["num_register_tokens"]
        self.globals = tokens[:, :start]
        self.patches = tokens[0, start:].reshape(cells, cells, -1)
        self.upsampled = None
        if upsample:
            self.upsampled = naf.upsample(weights["naf"], guide, self.patches[None], NAF_SIZE)[0]
        mx.eval(self.globals, self.patches, self.upsampled if upsample else self.patches)

    def project(self, coords, resolution, camera_angle_x, camera_distance):
        """(1, N, C) features at the projections of voxel ``coords`` on a ``resolution`` grid."""
        positions = projection.pixels(projection.grid_points(coords, resolution), camera_angle_x,
                                      camera_distance, self.size)
        parts = [projection.sample(self.patches, positions, self.size)]
        if self.upsampled is not None:
            parts.append(projection.sample(self.upsampled, positions, self.size))
        return mx.concatenate(parts, axis=-1)[None]


def generate(model, image, out, steps=12, seed=42, resolution=1536, guidance=7.5, max_faces=500_000):
    config, weights = model
    if resolution not in RESOLUTIONS:
        raise ValueError("Pixal3D resolution must be 1024 or 1536.")
    settings = config["pipeline"]
    started = time.perf_counter()
    stages = {}
    mx.reset_peak_memory()
    rng = np.random.default_rng(seed)
    with Image.open(image) as source:
        source.load()
        picture = dinov3.crop_object(source, CROP_MARGIN)
    rgb = mx.array((np.asarray(picture.convert("RGB"), dtype=np.float32) / 255)[None])
    angle = moge.camera_angle_x(weights["moge"], config["components"]["moge"], rgb)
    camera = (angle, projection.distance(angle))
    cursor = time.perf_counter()
    stages["camera"] = cursor - started

    low_cond = Condition(weights, config, picture, LOW_RESOLUTION, upsample=False)
    ss = config["components"]["ss_flow"]
    grid = ss["resolution"]
    all_cells = flow.grid_positions(grid)
    proj = low_cond.project(all_cells, grid, *camera)
    noise = _noise(rng, (1, ss["in_channels"], grid ** 3)).transpose(0, 2, 1)
    rope = flow.rope_tables(all_cells, ss["model_channels"] // ss["num_heads"])
    latent = flow.sample(weights["ss_flow"], ss, noise, low_cond.globals, mx.zeros_like(low_cond.globals), rope,
                         **_sampler(settings["sparse_structure_sampler"], steps, guidance), proj=proj)
    logits = structure.decode(weights["ss_decoder"], config["components"]["ss_decoder"],
                              latent.reshape(1, grid, grid, grid, -1))
    coords = structure.occupied(logits, LOW_RESOLUTION // 16)
    del proj, low_cond
    previous, cursor = cursor, time.perf_counter()
    stages["sparse_structure"] = cursor - previous

    shape_params = _sampler(settings["shape_slat_sampler"], steps, guidance)
    normalization = settings["shape_slat_normalization"]
    cond = Condition(weights, config, picture, LOW_RESOLUTION, upsample=True)
    low = config["components"]["shape_flow_512"]
    head_dim = low["model_channels"] // low["num_heads"]
    slat = flow.sample(weights["shape_flow_512"], low, _noise(rng, (1, len(coords), low["in_channels"])),
                       cond.globals, mx.zeros_like(cond.globals), flow.rope_tables(coords, head_dim),
                       **shape_params, proj=cond.project(coords, LOW_RESOLUTION // 16, *camera))
    slat = _denormalize(slat[0], normalization)
    decoder = config["components"]["shape_decoder"]
    upsampled = sparse.run(weights["shape_decoder"], decoder, slat, coords, stop_level=4)
    coords, level_resolution = cascade_coords(upsampled, LOW_RESOLUTION, resolution)
    del cond, slat, upsampled
    previous, cursor = cursor, time.perf_counter()
    stages["shape_latent_512"] = cursor - previous

    cond = Condition(weights, config, picture, CASCADE_FLOOR, upsample=True)
    high = config["components"]["shape_flow_1024"]
    slat = flow.sample(weights["shape_flow_1024"], high, _noise(rng, (1, len(coords), high["in_channels"])),
                       cond.globals, mx.zeros_like(cond.globals),
                       flow.rope_tables(coords, high["model_channels"] // high["num_heads"]),
                       **shape_params, proj=cond.project(coords, level_resolution // 16, *camera))
    slat = _denormalize(slat[0], normalization)
    del cond
    previous, cursor = cursor, time.perf_counter()
    stages["shape_latent_high"] = cursor - previous

    voxels, vertices, crossed, split = sparse.decode(weights["shape_decoder"], decoder, slat, coords)
    previous, cursor = cursor, time.perf_counter()
    stages["decode"] = cursor - previous
    vertices, faces = dual_grid.to_mesh(voxels, vertices, crossed, split, level_resolution)
    vertices = gltf_vertices(vertices)
    previous, cursor = cursor, time.perf_counter()
    stages["mesh"] = cursor - previous
    vertices, faces = decimate(vertices, faces, max_faces)
    previous, cursor = cursor, time.perf_counter()
    stages["decimation"] = cursor - previous
    write_glb(out, vertices, faces)
    stages["export"] = time.perf_counter() - cursor
    return {"path": str(out), "seconds": round(time.perf_counter() - started, 3), "steps": steps,
            "seed": seed, "resolution": level_resolution, "camera_angle_x": round(angle, 6),
            "vertices": len(vertices), "faces": len(faces), "voxels": len(voxels),
            "stage_seconds": {name: round(value, 3) for name, value in stages.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3)}
