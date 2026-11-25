"""Split one audio mix into a vocal stem and the rest with a source-separation model through its backend."""

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

SEPARATE_SCHEMA = "separate/1"
SEPARATE_RUNNER = Path(__file__).resolve().with_name("separate_runner.py")
DEFAULT_TIMEOUT_SECONDS = 1800


class SeparateError(InferenceError):
    """Classified separation failure safe to surface in a result envelope."""


def _invalid(message, remediation):
    return SeparateError("invalid_arguments", message, remediation)


def _new_wav(value, flag):
    target = Path(value).expanduser()
    if not target.is_absolute() or target.suffix.lower() != ".wav" or target.exists() or target.is_symlink():
        raise _invalid("{0} must be a new absolute .wav path.".format(flag), "Pass a path such as /Users/me/Music/vocals.wav.")
    if not target.parent.is_dir():
        raise _invalid("The directory of {0} does not exist.".format(flag), "Create the directory or pass a path inside an existing one.")
    return target


def plan_separate(model_path, audio_path, out, rest=None, manifests=None, registries=None, root=None):
    """Pick the separation backend and port for a model directory; pure apart from reading config.json."""
    model = Path(model_path).expanduser().resolve()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SeparateError(
            "model_unreadable", "No readable config.json in {0}: {1}".format(model, error),
            "Pass the separation model directory.",
        ) from error
    model_type = config.get("model_type") if isinstance(config, dict) else None
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise SeparateError("audio_not_found", "No audio file at {0}.".format(audio), "Pass a WAV, FLAC, or MP3 file.")
    target = _new_wav(out, "--out")
    remainder = None if rest is None else _new_wav(rest, "--rest")
    if remainder == target:
        raise _invalid("--rest must differ from --out.", "Pass two different output paths.")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = sorted((hit for hit in lookup(model_type, manifests, registries) if hit["category"] == "source_separation"),
                  key=lambda hit: hit["backend"])
    if not hits:
        raise SeparateError(
            "not_separation_model", "No source-separation backend implements model_type {0!r}.".format(model_type),
            "Separation applies to vocal separation models such as Mel-Band-RoFormer.",
        )
    manifest = manifests[hits[0]["backend"]]
    if not is_installed(manifest, root):
        raise SeparateError(
            "backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
            "Install it first: mlx-agent backend install {0}.".format(manifest["id"]),
        )
    argv = [str(backend_python(manifest, root)), str(SEPARATE_RUNNER), "--module", hits[0]["module"],
            "--model", str(model), "--audio", str(audio), "--out", str(target)]
    if remainder is not None:
        argv += ["--rest", str(remainder)]
    return {"backend": manifest["id"], "model": str(model), "audio": str(audio), "out": str(target),
            "rest": None if remainder is None else str(remainder), "argv": argv}


CONTRACT = RunnerContract(
    verb="separate", error=SeparateError,
    accept=lambda value, plan: value.get("path") == plan["out"] and value.get("rest") == plan["rest"],
    action="separate the mix", remediation="Check that the directory is a complete source-separation model.",
    activity="Source separation", timeout_remediation="Use a shorter mix or a larger --timeout.",
)


def run_separate(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    return {
        "schema": SEPARATE_SCHEMA, "backend": plan["backend"], "model": plan["model"], "audio": plan["audio"],
        "path": result["path"], "rest": result.get("rest"), "sample_rate": result.get("sample_rate"),
        "audio_seconds": result.get("audio_seconds"), "seconds": result.get("seconds"),
        "load_seconds": result.get("load_seconds"),
    }
