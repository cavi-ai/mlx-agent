"""Render one prompt with a converted text-to-video model through its backend (writes only the MP4)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .backends import (
    BackendError,
    backend_environment,
    backend_python,
    is_installed,
    load_manifests,
    load_registries,
    lookup,
    sync_ports,
)
from .wiring import redact_secrets

VIDEO_SCHEMA = "video/1"
VIDEO_RUNNER = Path(__file__).resolve().with_name("video_runner.py")
DEFAULT_TIMEOUT_SECONDS = 3600
MAX_PROMPT_CHARS = 2000
MAX_FAILURE_CHARS = 800
MAX_SIDE, MIN_SIDE = 1920, 64
MAX_FRAMES = 241
MAX_FPS = 60
DEFAULT_PATCH, DEFAULT_SPATIAL_STRIDE, DEFAULT_TEMPORAL_STRIDE = 2, 8, 4
DEFAULT_STEPS, DEFAULT_FPS = 50, 16
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FRAMES = 832, 480, 81


class VideoError(RuntimeError):
    """Classified video-generation failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _invalid(message, remediation):
    return VideoError("invalid_arguments", message, remediation)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _stride(config, key, index, default):
    value = config.get(key)
    if isinstance(value, (list, tuple)) and len(value) == 3 and _is_int(value[index]) and value[index] > 0:
        return value[index]
    return default


def _config_int(config, key, default):
    value = config.get(key)
    return value if _is_int(value) and value > 0 else default


def plan_video(model_path, prompt, out, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, frames=DEFAULT_FRAMES,
               fps=None, steps=None, seed=42, manifests=None, registries=None, root=None):
    """Pick the video backend for a converted model directory; pure apart from reading config.json.

    The size alignment, the frame rule, the area cap, and the step and rate defaults come from the
    model's own config.json (its VAE and patch strides), so a model with other strides is held to its own.
    """
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VideoError(
            "model_unreadable", "No readable config.json in {0}: {1}".format(model, error),
            "Pass the converted model directory.",
        ) from error
    if not isinstance(config, dict):
        raise VideoError("model_unreadable", "config.json in {0} is not an object.".format(model), "Pass the converted model directory.")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS or any(
        ord(character) < 32 and character not in "\n\t" for character in prompt
    ):
        raise _invalid("The prompt must be 1..{0} characters of text.".format(MAX_PROMPT_CHARS), "Pass --prompt with plain text.")
    target = Path(out).expanduser()
    if not target.is_absolute() or target.suffix.lower() != ".mp4" or target.exists():
        raise _invalid("--out must be a new absolute .mp4 path.", "Pass a path such as /Users/me/Movies/clip.mp4.")
    if not target.parent.is_dir():
        raise _invalid("The directory of --out does not exist.", "Create the directory or pass a path inside an existing one.")
    alignment = _stride(config, "patch_size", 1, DEFAULT_PATCH) * _stride(config, "vae_stride", 1, DEFAULT_SPATIAL_STRIDE)
    temporal = _stride(config, "vae_stride", 0, DEFAULT_TEMPORAL_STRIDE)
    area_cap = config.get("max_area") if _is_int(config.get("max_area")) and config.get("max_area") > 0 else None
    for name, value in (("width", width), ("height", height)):
        if not _is_int(value) or not MIN_SIDE <= value <= MAX_SIDE or value % alignment:
            raise _invalid(
                "--{0} must be a multiple of {1} from {2} to {3}.".format(name, alignment, MIN_SIDE, MAX_SIDE),
                "Pass --{0} {1}, for example.".format(name, 480 if name == "height" else 832),
            )
    if area_cap and width * height > area_cap:
        raise _invalid("This model renders at most {0} pixels per frame.".format(area_cap), "Pass a smaller --width and --height.")
    if not _is_int(frames) or not 1 + temporal <= frames <= MAX_FRAMES or (frames - 1) % temporal:
        raise _invalid(
            "--frames must be {0}n+1 from {1} to {2}.".format(temporal, 1 + temporal, MAX_FRAMES),
            "Pass --frames 17, for example.",
        )
    fps = _config_int(config, "sample_fps", DEFAULT_FPS) if fps is None else fps
    if not _is_int(fps) or not 1 <= fps <= MAX_FPS:
        raise _invalid("--fps must be 1..{0}.".format(MAX_FPS), "Pass --fps 16, for example.")
    steps = _config_int(config, "sample_steps", DEFAULT_STEPS) if steps is None else steps
    if not _is_int(steps) or not 1 <= steps <= 100:
        raise _invalid("--steps must be 1..100.", "Pass --steps 30, for example.")
    if not _is_int(seed) or not 0 <= seed < 2 ** 32:
        raise _invalid("--seed must be 0..4294967295.", "Pass --seed 42, for example.")
    model_type = config.get("model_type")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = sorted(
        (hit for hit in lookup(model_type, manifests, registries) if hit["category"] == "video_generation"),
        key=lambda hit: hit["backend"],
    )
    if not hits:
        raise VideoError(
            "not_video_model", "No video-generation backend implements model_type {0!r}.".format(model_type),
            "Video generation applies to converted text-to-video models such as Wan2.1 T2V.",
        )
    manifest = manifests[hits[0]["backend"]]
    if not is_installed(manifest, root):
        raise VideoError(
            "backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
            "Install it first: mlx-agent backend install {0}.".format(manifest["id"]),
        )
    argv = [
        str(backend_python(manifest, root)), str(VIDEO_RUNNER), "--module", hits[0]["module"], "--model", str(model),
        "--prompt=" + prompt, "--out", str(target), "--width", str(width), "--height", str(height),
        "--frames", str(frames), "--fps", str(fps), "--steps", str(steps), "--seed", str(seed),
    ]
    return {
        "backend": manifest["id"], "model": str(model), "out": str(target),
        "width": width, "height": height, "frames": frames, "fps": fps, "steps": steps, "seed": seed, "argv": argv,
    }


def _failure(plan, completed=None, reason=None):
    tail = reason
    if tail is None:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:]) or "no output"
        tail = redact_secrets(tail)[-MAX_FAILURE_CHARS:]
    return VideoError(
        "video_failed", "The {0} backend could not render the prompt: {1}".format(plan["backend"], tail),
        "Check that the directory is a complete converted text-to-video model, or use fewer frames, steps, or a smaller size.",
    )


def run_video(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    manifests = load_manifests() if manifests is None else manifests
    try:
        sync_ports(manifests[plan["backend"]], root)
    except BackendError as error:
        raise VideoError(error.code, str(error), error.remediation) from error
    try:
        completed = runner(
            plan["argv"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, env=backend_environment(), check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise _failure(plan, reason="video generation exceeded {0} s; use fewer frames or steps, or a larger --timeout".format(timeout)) from error
    result = None
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            result = value
            break
    if completed.returncode != 0 or result is None:
        raise _failure(plan, completed)
    keys = (
        "path", "width", "height", "frames", "fps", "duration_seconds", "steps", "seed", "seconds", "load_seconds",
        "seconds_per_frame", "peak_memory_gb", "pixel_std",
    )
    return dict({"schema": VIDEO_SCHEMA, "backend": plan["backend"], "model": plan["model"]},
                **{key: result.get(key) for key in keys})
