"""Offline image-to-mesh generation; model evaluation runs entirely on MLX."""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .dinov2 import encode, preprocess
from .geometry import decimate, extract_mesh, write_glb
from .transformer import Decoder, denoise
from .weights import Weights


NUM_TOKENS = 2048
QUERY_CHUNK = 32768


class FieldQuery:
    """The VAE decoder as a point-query function: float16 points (N, 3) in, float32 logits (N,) out."""

    def __init__(self, decoder, chunk=QUERY_CHUNK):
        self.decoder, self.chunk = decoder, chunk

    def __call__(self, points):
        outputs, pending = [], None
        for start in range(0, len(points), self.chunk):
            logits = self.decoder(mx.array(points[start:start + self.chunk])[None]).reshape(-1)
            mx.async_eval(logits)
            if pending is not None:
                outputs.append(np.array(pending))
            pending = logits
        if pending is None:
            return np.empty(0, dtype=np.float32)
        outputs.append(np.array(pending))
        return np.concatenate(outputs)


def initial_latents(seed, channels):
    """Float32 standard-normal noise from NumPy's default generator, which reproduces the reference noise of a seed."""
    return mx.array(np.random.default_rng(seed).standard_normal((1, NUM_TOKENS, channels)).astype(np.float32))


def flow_schedule(config, steps):
    """Use the checkpoint's scheduler, including TripoSG's positive flow direction."""
    train_steps = config.get("num_train_timesteps", 1000)
    kind = config.get("_class_name")
    if kind == "RectifiedFlowScheduler":
        sigmas = [1 - index / steps for index in range(steps)] + [0.0]
        direction = -1
    elif kind == "FlowMatchEulerDiscreteScheduler":
        sigmas = [1 - i * (1 - 1 / train_steps) / max(steps - 1, 1) for i in range(steps)] + [0.0]
        direction = 1
    else:
        raise ValueError("Unsupported TripoSG flow scheduler: " + str(kind))
    return [(sigmas[i] * train_steps, direction * (sigmas[i + 1] - sigmas[i])) for i in range(steps)]


def load(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "triposg":
        raise ValueError("Pass a converted TripoSG model directory.")
    quantization = config.get("quantization", {})
    weights = {name: Weights(mx.load(str(path / (name + ".safetensors"))), **quantization)
               for name in ("transformer", "vae", "image_encoder_dinov2")}
    mx.eval(*[value for component in weights.values() for value in component.values.values()])
    return config, weights


def generate(model, image, out, steps=50, seed=42, resolution=512, guidance=7.0, max_faces=500_000):
    config, weights = model
    started = time.perf_counter()
    stages = {}
    mx.reset_peak_memory()
    pixels = preprocess(image, config["preprocessor"])
    features = encode(weights["image_encoder_dinov2"], config["components"]["image_encoder_dinov2"], pixels)
    features = mx.concatenate([mx.zeros_like(features), features], axis=0) if guidance > 1 else features
    mx.eval(features)
    cursor = time.perf_counter()
    stages["conditioning"] = cursor - started
    transformer = config["components"]["transformer"]
    latents = initial_latents(seed, transformer["in_channels"])
    for timestep, delta in flow_schedule(config["scheduler"], steps):
        inputs = mx.concatenate([latents, latents], axis=0) if guidance > 1 else latents
        prediction = denoise(weights["transformer"], transformer, inputs,
                             [timestep] * inputs.shape[0], features)
        if guidance > 1:
            unconditioned, conditioned = mx.split(prediction, 2, axis=0)
            prediction = unconditioned + guidance * (conditioned - unconditioned)
        latents = latents + delta * prediction
        mx.eval(latents)
    previous, cursor = cursor, time.perf_counter()
    stages["sampling"] = cursor - previous
    decoder = Decoder(weights["vae"], config["components"]["vae"], latents)
    context = time.perf_counter() - cursor
    decode = {}
    vertices, faces, queries = extract_mesh(FieldQuery(decoder), resolution.bit_length() - 1, seconds=decode)
    stages["decode_dense"] = decode["decode_dense"] + context
    stages["decode_refine"] = decode["decode_refine"]
    stages["marching_cubes"] = decode["marching_cubes"]
    cursor = time.perf_counter()
    vertices, faces = decimate(vertices, faces, max_faces)
    previous, cursor = cursor, time.perf_counter()
    stages["decimation"] = cursor - previous
    write_glb(out, vertices, faces)
    stages["export"] = time.perf_counter() - cursor
    return {"path": str(out), "seconds": round(time.perf_counter() - started, 3), "steps": steps,
            "seed": seed, "resolution": resolution, "vertices": len(vertices), "faces": len(faces),
            "queries": queries, "stage_seconds": {name: round(value, 3) for name, value in stages.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3)}
