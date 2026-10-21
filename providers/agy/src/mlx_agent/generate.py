"""Render one prompt with a converted image-generation model through its backend (writes only the PNG)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .backends import (
    backend_python,
    is_installed,
    load_manifests,
    load_registries,
    lookup,
)
from .inference_runner import InferenceError, RunnerContract, invoke

GENERATE_SCHEMA = "generate/1"
GENERATE_RUNNER = Path(__file__).resolve().with_name("generate_runner.py")
DEFAULT_TIMEOUT_SECONDS = 1800
MAX_PROMPT_CHARS = 2000
SIDES = range(256, 2049)


class GenerateError(InferenceError):
    """Classified image-generation failure safe to surface in a result envelope."""


def _invalid(message, remediation):
    return GenerateError("invalid_arguments", message, remediation)


def plan_generate(model_path, prompt, out, width=1024, height=1024, steps=40, seed=42,
                  manifests=None, registries=None, root=None):
    """Pick the image backend for a converted model directory; pure apart from reading config.json."""
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GenerateError(
            "model_unreadable", "No readable config.json in {0}: {1}".format(model, error),
            "Pass the converted model directory.",
        ) from error
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS or any(
        ord(character) < 32 and character not in "\n\t" for character in prompt
    ):
        raise _invalid("The prompt must be 1..{0} characters of text.".format(MAX_PROMPT_CHARS), "Pass --prompt with plain text.")
    target = Path(out).expanduser()
    if not target.is_absolute() or target.suffix.lower() != ".png" or target.exists():
        raise _invalid("--out must be a new absolute .png path.", "Pass a path such as /Users/me/Pictures/render.png.")
    for name, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) or value not in SIDES or value % 16:
            raise _invalid("--{0} must be a multiple of 16 from 256 to 2048.".format(name), "Pass --{0} 1024, for example.".format(name))
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 100:
        raise _invalid("--steps must be 1..100.", "Pass --steps 40, for example.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 32:
        raise _invalid("--seed must be 0..4294967295.", "Pass --seed 42, for example.")
    model_type = config.get("model_type") if isinstance(config, dict) else None
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = sorted(
        (hit for hit in lookup(model_type, manifests, registries) if hit["category"] == "image_generation"),
        key=lambda hit: hit["backend"],
    )
    if not hits:
        raise GenerateError(
            "not_image_model", "No image-generation backend implements model_type {0!r}.".format(model_type),
            "Image generation applies to converted image models such as Qwen-Image 2.1.",
        )
    manifest = manifests[hits[0]["backend"]]
    if not is_installed(manifest, root):
        raise GenerateError(
            "backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
            "Install it first: mlx-agent backend install {0}.".format(manifest["id"]),
        )
    argv = [
        str(backend_python(manifest, root)), str(GENERATE_RUNNER), "--module", hits[0]["module"], "--model", str(model),
        "--prompt", prompt, "--out", str(target), "--width", str(width), "--height", str(height),
        "--steps", str(steps), "--seed", str(seed),
    ]
    return {"backend": manifest["id"], "model": str(model), "out": str(target), "argv": argv}


CONTRACT = RunnerContract(
    verb="generate", error=GenerateError, accept=lambda value, plan: isinstance(value.get("path"), str),
    action="render the prompt", remediation="Check that the directory is a complete converted image model.",
    activity="Image generation", timeout_remediation="Use fewer steps, a smaller size, or a larger --timeout.",
)


def run_generate(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    keys = ("path", "width", "height", "steps", "seed", "seconds", "load_seconds", "pixel_std")
    return dict({"schema": GENERATE_SCHEMA, "backend": plan["backend"], "model": plan["model"]},
                **{key: result.get(key) for key in keys})
