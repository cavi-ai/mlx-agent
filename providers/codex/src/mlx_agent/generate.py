"""Render one prompt with a converted image-generation model through its backend (writes only the PNG)."""

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

GENERATE_SCHEMA = "generate/1"
GENERATE_RUNNER = Path(__file__).resolve().with_name("generate_runner.py")
DEFAULT_TIMEOUT_SECONDS = 1800
MAX_PROMPT_CHARS = 2000
SIDES = range(256, 2049)


class GenerateError(RuntimeError):
    """Classified image-generation failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


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


def run_generate(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    manifests = load_manifests() if manifests is None else manifests
    try:
        sync_ports(manifests[plan["backend"]], root)
    except BackendError as error:
        raise GenerateError(error.code, str(error), error.remediation) from error
    try:
        completed = runner(
            plan["argv"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, env=backend_environment(), check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise GenerateError(
            "generate_timeout", "Image generation exceeded {0} s.".format(timeout),
            "Use fewer steps, a smaller size, or a larger --timeout.",
        ) from error
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
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:])
        raise GenerateError(
            "generate_failed", "The {0} backend could not render the prompt: {1}".format(plan["backend"], tail or "no output"),
            "Check that the directory is a complete converted image model.",
        )
    keys = ("path", "width", "height", "steps", "seed", "seconds", "load_seconds", "pixel_std")
    return dict({"schema": GENERATE_SCHEMA, "backend": plan["backend"], "model": plan["model"]},
                **{key: result.get(key) for key in keys})
