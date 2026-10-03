"""Answer one question about an image or a video with a converted vision-language model through its backend (read-only)."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from .backends import (
    BackendError,
    backend_environment,
    backend_python,
    is_installed,
    is_vision_type,
    load_manifests,
    load_registries,
    lookup,
    sync_ports,
)
from .wiring import redact_secrets

DESCRIBE_SCHEMA = "describe/1"
DESCRIBE_RUNNER = Path(__file__).resolve().with_name("describe_runner.py")
DEFAULT_TIMEOUT_SECONDS = 900
MAX_PROMPT_CHARS = 4000
MAX_TOKENS_RANGE = range(1, 4097)
MAX_TEMPERATURE = 2.0
MIN_FPS, MAX_FPS = 0.1, 8.0
DEFAULT_FPS = 1.0
MAX_PIXELS_LIMIT = 100_000_000
MAX_FAILURE_CHARS = 800
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")
REFUSALS = ("not_vision_model", "video_unsupported")


class DescribeError(RuntimeError):
    """Classified vision-language failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


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
        str(backend_python(manifest, root)), str(DESCRIBE_RUNNER), "--model", str(model), "--prompt", prompt,
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


def _failure(plan, completed=None, reason=None):
    tail = reason
    if tail is None:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:]) or "no output"
        tail = redact_secrets(tail)[-MAX_FAILURE_CHARS:]
    return DescribeError(
        "describe_failed", "The {0} backend could not describe the {1}: {2}".format(plan["backend"], plan["input"]["kind"], tail),
        "Check that the directory is a complete converted vision-language model and that the media file decodes.",
    )


def run_describe(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    manifests = load_manifests() if manifests is None else manifests
    try:
        sync_ports(manifests[plan["backend"]], root)
    except BackendError as error:
        raise DescribeError(error.code, str(error), error.remediation) from error
    try:
        completed = runner(
            plan["argv"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, env=backend_environment(), check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise _failure(plan, reason="description exceeded {0} s; use a shorter clip, fewer tokens, or a larger --timeout".format(timeout)) from error
    result = None
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and (isinstance(value.get("text"), str) or value.get("error") in REFUSALS):
            result = value
            break
    if result is not None and result.get("error") == "video_unsupported":
        raise DescribeError(
            "video_unsupported", "The {0} backend cannot take video for model_type {1!r}: {2}".format(
                plan["backend"], plan["model_type"], redact_secrets(str(result.get("detail") or "no detail"))[-MAX_FAILURE_CHARS:]),
            "Describe a still image with --image, or use a model type that reads video.",
        )
    if result is not None and result.get("error") == "not_vision_model":
        raise DescribeError(
            "not_vision_model", "The {0} backend has no image path for model_type {1!r}.".format(plan["backend"], plan["model_type"]),
            "Describing media applies to converted vision-language models such as Qwen2.5-VL.",
        )
    if completed.returncode != 0 or result is None:
        raise _failure(plan, completed)
    keys = ("text", "prompt_tokens", "generation_tokens", "prompt_tps", "generation_tps",
            "peak_memory_gb", "seconds", "load_seconds")
    return dict({"schema": DESCRIBE_SCHEMA, "backend": plan["backend"], "model": plan["model"]},
                **{key: result.get(key) for key in keys}, input=dict(plan["input"]))
