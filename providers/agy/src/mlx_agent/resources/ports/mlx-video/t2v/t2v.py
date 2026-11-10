"""Render one prompt to an MP4 with a converted Wan text-to-video directory, through mlx-video's pipeline.

The package's ``generate_video`` loads every component itself, so the port times its loaders to split load
time from generation time, redirects its tokenizer to the copy ``convert`` saved beside the weights (no hub
access), and replaces its video writer to measure the frames and encode the MP4 with OpenCV.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path
from unittest import mock

TOKENIZER_REPO = "google/umt5-xxl"
TOKENIZER_DIR = "tokenizer"
LOADERS = ("load_t5_encoder", "load_wan_model", "load_vae_decoder")


class Model:
    """A converted Wan directory: its path and parsed config.json."""

    def __init__(self, path: Path, config: dict):
        self.path = path
        self.config = config
        self.retain_weights = False
        self.components = {}


def load(path) -> Model:
    directory = Path(path).expanduser()
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("no readable config.json in {0}: {1}".format(directory, error)) from error
    if not isinstance(config, dict):
        raise ValueError("config.json in {0} is not an object".format(directory))
    transformer = (
        ("low_noise_model.safetensors", "high_noise_model.safetensors") if config.get("dual_model")
        else ("model.safetensors",)
    )
    missing = [name for name in ("t5_encoder.safetensors", "vae.safetensors") + transformer if not (directory / name).is_file()]
    if missing:
        raise ValueError("{0} is not a complete converted Wan model; missing {1}".format(directory, ", ".join(missing)))
    return Model(directory, config)


def _encode(frames, out: str, fps: int) -> None:
    import cv2

    height, width = frames.shape[1], frames.shape[2]
    for code in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*code), float(fps), (width, height))
        if writer.isOpened():
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            if Path(out).is_file() and Path(out).stat().st_size > 0:
                return
        else:
            writer.release()
        if Path(out).exists():
            Path(out).unlink()
    raise RuntimeError("OpenCV could not encode an MP4 (tried avc1 and mp4v)")


def _pixel_std(frames) -> float:
    import numpy as np

    total = squares = 0.0
    for frame in frames:
        values = frame.astype(np.float64)
        total += float(values.sum())
        squares += float((values * values).sum())
    count = float(frames.size)
    mean = total / count
    return max(squares / count - mean * mean, 0.0) ** 0.5


def generate(model: Model, prompt: str, out: str, width: int = 832, height: int = 480, frames: int = 81,
             fps: int = 16, steps: int = 50, seed: int = 42, denoiser_cache: str = "off",
             cache_memory_mib: int = 256) -> dict:
    """Render one video to ``out`` (MP4); returns its dimensions, timing, memory, and pixel spread."""
    import mlx.core as mx
    import mlx_video.generate_wan as wan
    from transformers import AutoTokenizer

    if denoiser_cache not in ("off", "adaptive"):
        raise ValueError("denoiser_cache must be off or adaptive")
    if isinstance(cache_memory_mib, bool) or not isinstance(cache_memory_mib, int) or not 8 <= cache_memory_mib <= 4096:
        raise ValueError("cache_memory_mib must be 8..4096")
    cache = None
    cache_result = None
    if denoiser_cache == "adaptive":
        from .denoiser_cache import WanDenoiserCache, required_cache_bytes

        if model.config.get("model_version") != "2.1" or model.config.get("dual_model", False) or model.config.get("model_type") != "t2v":
            raise ValueError("adaptive denoiser caching requires single-model Wan2.1 T2V")
        limit = cache_memory_mib * 1024 ** 2
        if required_cache_bytes(model.config, width, height, frames) > limit:
            cache_result = {"mode": "adaptive", "full_steps": steps, "reused_steps": 0,
                            "peak_cache_bytes": 0, "memory_limit_bytes": limit, "fallback_reason": "memory_limit"}
        else:
            cache = WanDenoiserCache(steps, limit)

    spent = {"load": 0.0}
    written = {}
    original_loaders = {name: getattr(wan, name) for name in LOADERS}
    from_pretrained = AutoTokenizer.from_pretrained
    local_tokenizer = model.path / TOKENIZER_DIR

    def timed(function):
        def wrapper(*arguments, **keywords):
            started = time.time()
            try:
                key = repr((arguments, sorted(keywords.items())))
                retained = model.components.get(function.__name__) if model.retain_weights else None
                if function.__name__ == "load_vae_decoder" and cache is not None:
                    cache.close()
                if retained is not None and retained[0] == key:
                    value = retained[1]
                else:
                    value = function(*arguments, **keywords)
                    if model.retain_weights:
                        model.components[function.__name__] = (key, value)
                if function.__name__ == "load_wan_model" and cache is not None:
                    return cache.attach(value)
                return value
            finally:
                spent["load"] += time.time() - started
        return wrapper

    def redirected(name, *arguments, **keywords):
        started = time.time()
        try:
            if name == TOKENIZER_REPO and local_tokenizer.is_dir():
                name = str(local_tokenizer)
            keywords.update(local_files_only=True, trust_remote_code=False)
            key = repr((name, arguments, sorted(keywords.items())))
            retained = model.components.get("tokenizer") if model.retain_weights else None
            if retained is not None and retained[0] == key:
                return retained[1]
            value = from_pretrained(name, *arguments, **keywords)
            if model.retain_weights:
                model.components["tokenizer"] = (key, value)
            return value
        finally:
            spent["load"] += time.time() - started

    def capture(video, output_path, fps=None):
        written["frames"] = int(video.shape[0])
        written["height"], written["width"] = int(video.shape[1]), int(video.shape[2])
        written["pixel_std"] = _pixel_std(video)
        _encode(video, output_path, playback)

    playback = fps
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mx.reset_peak_memory()
    started = time.time()
    patches = [mock.patch.object(wan, name, timed(function)) for name, function in original_loaders.items()]
    patches += [
        mock.patch.object(wan, "save_video", capture),
        mock.patch.object(AutoTokenizer, "from_pretrained", redirected),
    ]
    with contextlib.ExitStack() as stack:
        if cache is not None:
            stack.callback(cache.close)
        for patch in patches:
            stack.enter_context(patch)
        with contextlib.redirect_stdout(sys.stderr):
            wan.generate_video(
                model_dir=str(model.path), prompt=prompt, width=width, height=height, num_frames=frames,
                steps=steps, seed=seed, output_path=out,
                **({"no_compile": True} if cache is not None else {}),
            )
    total = time.time() - started
    peak_memory_gb = mx.get_peak_memory() / 1e9
    if "frames" not in written:
        raise RuntimeError("the pipeline produced no video")
    seconds = max(total - spent["load"], 0.0)
    result = {
        "path": str(out), "width": written["width"], "height": written["height"], "frames": written["frames"],
        "fps": fps, "duration_seconds": round(written["frames"] / fps, 3), "steps": steps, "seed": seed,
        "seconds": round(seconds, 3), "load_seconds": round(spent["load"], 3),
        "seconds_per_frame": round(seconds / written["frames"], 3),
        "peak_memory_gb": round(peak_memory_gb, 3), "pixel_std": round(written["pixel_std"], 3),
    }
    if denoiser_cache == "adaptive":
        result["denoiser_cache"] = cache.stats() if cache is not None else cache_result
    return result
