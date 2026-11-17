"""Render one prompt to an MP4 with a converted LTX-2.3 directory through mlx-video's distilled pipeline.

An optional first-frame image conditions the video (image-to-video) and an optional audio file drives
it (audio-to-video: the audio latents stay fixed, so mouths and motion follow the sound, and the MP4
carries that audio). The port passes the Gemma attention mask as booleans: the package's additive
bfloat16 mask turns fully padded rows into NaN at 1024 tokens on MLX 0.32, which decodes every frame
black.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

REQUIRED = ("transformer", "text_projections", "vae/encoder", "vae/decoder", "audio_vae", "vocoder", "text_encoder",
            "tokenizer")


class Model:
    """A converted LTX-2.3 directory: its path and parsed config.json."""

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
    if not isinstance(config, dict) or config.get("model_type") != "ltx2":
        raise ValueError("{0} is not a converted LTX-2.3 directory".format(directory))
    missing = [name for name in REQUIRED if not (directory / name).is_dir()]
    if not (directory / str(config.get("spatial_upscaler", ""))).is_file():
        missing.append("spatial upscaler")
    if missing:
        raise ValueError("{0} is not a complete converted LTX-2.3 model; missing {1}".format(directory, ", ".join(missing)))
    return Model(directory, config)


def _boolean_mask(self, seq_len, attention_mask, dtype):
    import mlx.core as mx

    causal = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
    if attention_mask is None:
        return causal[None, None, :, :]
    return (causal[None, :, :] & attention_mask.astype(mx.bool_)[:, None, :])[:, None, :, :]


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


def generate(model: Model, prompt: str, out: str, width: int = 768, height: int = 512, frames: int = 97,
             fps: int = 24, steps: int = 8, seed: int = 42, image: str | None = None, audio: str | None = None,
             audio_start: float = 0.0) -> dict:
    """Render one video to ``out`` (MP4); returns its dimensions, timing, memory, and pixel spread."""
    import mlx.core as mx
    from mlx_video.models.ltx_2 import generate as ltx
    from mlx_video.models.ltx_2.text_encoder import LanguageModel

    muxed = {}
    mux = ltx.mux_video_audio

    def checked_mux(video_path, audio_path, output_path):
        muxed["ok"] = bool(mux(video_path, audio_path, output_path))
        return muxed["ok"]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mx.reset_peak_memory()
    started = time.time()
    with contextlib.ExitStack() as stack:
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ltx2-")))
        stack.enter_context(mock.patch.object(LanguageModel, "_create_causal_mask_with_padding", _boolean_mask))
        stack.enter_context(mock.patch.object(ltx, "mux_video_audio", checked_mux))
        stack.enter_context(contextlib.redirect_stdout(sys.stderr))
        result = ltx.generate_video(
            model_repo=str(model.path), text_encoder_repo=str(model.path / "text_encoder"), prompt=prompt,
            pipeline=ltx.PipelineType.DISTILLED, height=height, width=width, num_frames=frames,
            num_inference_steps=steps, seed=seed, fps=fps, output_path=out, verbose=False,
            image=image, audio_file=audio, audio_start_time=audio_start,
            output_audio_path=str(scratch / "audio.wav") if audio else None,
            spatial_upscaler=model.config["spatial_upscaler"],
        )
    total = time.time() - started
    video = result[0] if isinstance(result, tuple) else result
    if video is None or len(video) == 0:
        raise RuntimeError("the pipeline produced no video")
    if audio and not muxed.get("ok"):
        Path(out).unlink(missing_ok=True)
        raise RuntimeError("ffmpeg could not add the audio to the video")
    count = int(video.shape[0])
    return {
        "path": str(out), "width": int(video.shape[2]), "height": int(video.shape[1]), "frames": count,
        "fps": fps, "duration_seconds": round(count / fps, 3), "steps": steps, "seed": seed,
        "seconds": round(total, 3), "load_seconds": 0.0, "seconds_per_frame": round(total / count, 3),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3), "pixel_std": round(_pixel_std(video), 3),
    }
