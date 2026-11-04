"""Offline image-to-mesh generation; model evaluation runs entirely on MLX."""

import json
import time
from pathlib import Path

import mlx.core as mx

from .dinov2 import encode, preprocess
from .geometry import extract_mesh, write_glb
from .transformer import Decoder, denoise
from .weights import Weights


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


def generate(model, image, out, steps=50, seed=42, resolution=128, guidance=7.0):
    config, weights = model
    started = time.monotonic()
    mx.reset_peak_memory()
    mx.random.seed(seed)
    pixels = preprocess(image, config["preprocessor"])
    features = encode(weights["image_encoder_dinov2"], config["components"]["image_encoder_dinov2"], pixels)
    features = mx.concatenate([mx.zeros_like(features), features], axis=0) if guidance > 1 else features
    transformer = config["components"]["transformer"]
    latents = mx.random.normal((1, 2048, transformer["in_channels"]))
    for timestep, delta in flow_schedule(config["scheduler"], steps):
        inputs = mx.concatenate([latents, latents], axis=0) if guidance > 1 else latents
        prediction = denoise(weights["transformer"], transformer, inputs,
                             [timestep] * inputs.shape[0], features)
        if guidance > 1:
            unconditioned, conditioned = mx.split(prediction, 2, axis=0)
            prediction = unconditioned + guidance * (conditioned - unconditioned)
        latents = latents + delta * prediction
        mx.eval(latents)
    decoder = Decoder(weights["vae"], config["components"]["vae"], latents)
    vertices, faces = extract_mesh(decoder, resolution)
    write_glb(out, vertices, faces)
    return {"path": str(out), "seconds": round(time.monotonic() - started, 3), "steps": steps,
            "seed": seed, "resolution": resolution, "vertices": len(vertices), "faces": len(faces),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3)}
