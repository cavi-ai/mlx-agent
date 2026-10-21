"""Answer one question about an image or a video with a converted vision-language model through its backend (read-only)."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from .backends import (
    backend_python,
    is_installed,
    is_vision_type,
    load_manifests,
    load_registries,
    lookup,
)
from .inference_runner import InferenceError, RunnerContract, invoke

DESCRIBE_SCHEMA = "describe/1"
DESCRIBE_RUNNER = Path(__file__).resolve().with_name("describe_runner.py")
DEFAULT_TIMEOUT_SECONDS = 900
MAX_PROMPT_CHARS = 4000
MAX_TOKENS_RANGE = range(1, 4097)
MAX_TEMPERATURE = 2.0
MIN_FPS, MAX_FPS = 0.1, 8.0
DEFAULT_FPS = 1.0
MAX_PIXELS_LIMIT = 100_000_000
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")


class DescribeError(InferenceError):
    """Classified vision-language failure safe to surface in a result envelope."""


def _invalid(message, remediation):
    return DescribeError("invalid_arguments", message, remediation)


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _media(kind, value, suffixes):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if path.suffix.lower() not in suffixes:
        raise _invalid(
            "--{0} must be a {1} file.".format(kind, "/".join(suffix[1:] for suffix in suffixes)),
            "Pass --{0} with one of: {1}.".format(kind, ", ".join(suffixes)),
        )
    if not path.is_file():
        raise _invalid("No {0} file at {1}.".format(kind, path), "Pass the path of an existing {0} file.".format(kind))
    return path


def plan_describe(model_path, prompt, image=None, video=None, max_tokens=256, temperature=0.0, fps=None,
                  max_pixels=None, manifests=None, registries=None, root=None):
    """Pick the vision-language backend for a converted model directory; pure apart from reading files."""
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DescribeError(
            "model_unreadable", "No readable config.json in {0}: {1}".format(model, error),
            "Pass the converted model directory.",
        ) from error
    if not isinstance(config, dict):
        raise DescribeError("model_unreadable", "config.json in {0} is not an object.".format(model), "Pass the converted model directory.")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS or any(
        ord(character) < 32 and character not in "\n\t" for character in prompt
    ):
        raise _invalid("The prompt must be 1..{0} characters of plain text.".format(MAX_PROMPT_CHARS), "Pass --prompt with a question about the media.")
    if (image is None) == (video is None):
        raise _invalid("Pass exactly one of --image or --video.", "Pass --image FILE for a picture or --video FILE for a clip.")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens not in MAX_TOKENS_RANGE:
        raise _invalid("--max-tokens must be 1..4096.", "Pass --max-tokens 256, for example.")
    if not _number(temperature) or not 0 <= temperature <= MAX_TEMPERATURE:
        raise _invalid("--temperature must be 0..{0}.".format(MAX_TEMPERATURE), "Pass --temperature 0 for a repeatable answer.")
    if fps is not None and video is None:
        raise _invalid("--fps applies to --video only.", "Drop --fps when describing an image.")
    if fps is not None and (not _number(fps) or not MIN_FPS <= fps <= MAX_FPS):
        raise _invalid("--fps must be {0}..{1}.".format(MIN_FPS, MAX_FPS), "Pass --fps 1.0, for example.")
    if max_pixels is not None and (not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or not 1 <= max_pixels <= MAX_PIXELS_LIMIT):
        raise _invalid("--max-pixels must be 1..{0}.".format(MAX_PIXELS_LIMIT), "Pass --max-pixels 1000000, for example.")
    kind = "image" if image is not None else "video"
    media = _media(kind, image if image is not None else video, IMAGE_SUFFIXES if image is not None else VIDEO_SUFFIXES)
    model_type = config.get("model_type")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = sorted(
        (hit for hit in lookup(model_type, manifests, registries) if hit["category"] == "vision_language"),
        key=lambda hit: hit["backend"],
    )
    if not hits or not is_vision_type(model_type, manifests, registries):
        raise DescribeError(
            "not_vision_model", "No vision-language backend implements model_type {0!r} with a vision tower.".format(model_type),
            "Describing media applies to converted vision-language models such as Qwen2.5-VL.",
        )
    manifest = manifests[hits[0]["backend"]]
    if not is_installed(manifest, root):
        raise DescribeError(
            "backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
            "Install it first: mlx-agent backend install {0}.".format(manifest["id"]),
        )
    used_fps = (DEFAULT_FPS if fps is None else float(fps)) if kind == "video" else None
    argv = [
        str(backend_python(manifest, root)), str(DESCRIBE_RUNNER), "--model", str(model), "--prompt=" + prompt,
        "--{0}".format(kind), str(media), "--max-tokens", str(max_tokens), "--temperature", repr(float(temperature)),
    ]
    if used_fps is not None:
        argv += ["--fps", repr(used_fps)]
    if max_pixels is not None:
        argv += ["--max-pixels", str(max_pixels)]
    return {
        "backend": manifest["id"], "model": str(model), "model_type": model_type,
        "input": {"kind": kind, "path": str(media)}, "argv": argv,
    }


def _video_unsupported(plan, detail):
    return ("The {0} backend cannot take video for model_type {1!r}: {2}".format(plan["backend"], plan["model_type"], detail),
            "Describe a still image with --image, or use a model type that reads video.")


def _not_vision_model(plan, detail):
    return ("The {0} backend has no image path for model_type {1!r}.".format(plan["backend"], plan["model_type"]),
            "Describing media applies to converted vision-language models such as Qwen2.5-VL.")


CONTRACT = RunnerContract(
    verb="describe", error=DescribeError, accept=lambda value, plan: isinstance(value.get("text"), str),
    action="describe the {input[kind]}",
    remediation="Check that the directory is a complete converted vision-language model and that the media file decodes.",
    activity="Description", timeout_remediation="Use a shorter clip, fewer tokens, or a larger --timeout.",
    refusals={"video_unsupported": _video_unsupported, "not_vision_model": _not_vision_model},
)


def run_describe(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    keys = ("text", "prompt_tokens", "generation_tokens", "prompt_tps", "generation_tps",
            "peak_memory_gb", "seconds", "load_seconds")
    return dict({"schema": DESCRIBE_SCHEMA, "backend": plan["backend"], "model": plan["model"]},
                **{key: result.get(key) for key in keys}, input=dict(plan["input"]))
