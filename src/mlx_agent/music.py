"""Generate music from existing local weights in the isolated audio backend."""

import json
import math
import subprocess
from pathlib import Path

from .backends import backend_environment, backend_python, is_installed, load_manifests, load_registries, lookup
from .wiring import redact_secrets
from .music_runner import tokenizer_directory

MUSIC_RUNNER = Path(__file__).with_name("music_runner.py")


class MusicError(RuntimeError):
    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code, self.remediation = code, remediation


def plan_music(model_path, caption, lyrics, out, duration=15.0, steps=30, seed=42,
               manifests=None, registries=None, root=None):
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        if not model.is_absolute() or not isinstance(config, dict):
            raise ValueError("An absolute local model directory is required")
    except (OSError, ValueError) as error:
        raise MusicError("model_unreadable", str(error), "Pass an existing converted music model directory.") from error
    for name, text, limit in (("caption", caption, 2000), ("lyrics", lyrics, 10000)):
        if not isinstance(text, str) or not text.strip() or len(text) > limit or any(ord(c) < 32 and c not in "\n\t" for c in text):
            raise MusicError("invalid_arguments", "Invalid {0}.".format(name), "Provide a caption and lyrics; use [instrumental] for instrumental music.")
    if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or not 0 < duration <= 360:
        raise MusicError("invalid_arguments", "Duration must be greater than 0 and at most 360 seconds.", "Use --duration 15.")
    if type(steps) is not int or not 1 <= steps <= 30 or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise MusicError("invalid_arguments", "Steps or seed is out of range.", "Use 1..30 steps and a 32-bit unsigned seed.")
    target = Path(out).expanduser()
    if not target.is_absolute() or target.suffix.lower() != ".wav" or target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise MusicError("invalid_arguments", "Output must be a new absolute WAV path.", "Choose a new .wav file in an existing directory.")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = [hit for hit in lookup(config.get("model_type", ""), manifests, registries) if hit["category"] == "music_generation"]
    if not hits:
        raise MusicError("not_music_model", "No music backend implements this model type.", "Use a supported converted music generation model.")
    if config.get("model_type") == "minimax_music3":
        try:
            tokenizer_directory(model)
        except ValueError as error:
            raise MusicError("model_unreadable", str(error), "Restore the checkpoint's tokenizer files before generating music.") from error
    manifest = manifests[sorted(hits, key=lambda hit: hit["backend"])[0]["backend"]]
    if not is_installed(manifest, root):
        raise MusicError("backend_not_installed", "The audio backend is not installed.", "Install mlx-audio first.")
    return {"backend": manifest["id"], "model": str(model), "out": str(target),
            "duration": duration, "steps": steps, "seed": seed,
            "argv": [str(backend_python(manifest, root)), str(MUSIC_RUNNER), "--model", str(model),
                     "--caption=" + caption, "--lyrics=" + lyrics, "--out", str(target),
                     "--duration", str(duration), "--steps", str(steps), "--seed", str(seed)]}


def run_music(plan, timeout=3600, runner=subprocess.run):
    environment = backend_environment()
    # Generation never resolves or downloads missing weights or tokenizer files.
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        completed = runner(plan["argv"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=timeout, env=environment, check=False)
    except subprocess.TimeoutExpired as error:
        raise MusicError("music_failed", "Music generation timed out.", "Use a shorter duration or a larger timeout.") from error
    result = None
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("path") == plan["out"]:
            result = value
            break
    if completed.returncode != 0 or result is None or not Path(plan["out"]).is_file() or Path(plan["out"]).is_symlink():
        reason = redact_secrets(completed.stderr or "no audio output")[-800:]
        raise MusicError("music_failed", reason, "Check local model completeness and the audio backend.")
    keys = ("path", "sample_rate", "audio_seconds", "seconds", "load_seconds", "real_time_factor", "peak_memory_gb")
    return dict({"schema": "music/1", "backend": plan["backend"], "model": plan["model"],
                 "duration": plan["duration"], "steps": plan["steps"], "seed": plan["seed"]},
                **{key: result.get(key) for key in keys})
