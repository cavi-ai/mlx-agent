"""Synthesize one text with a converted text-to-speech model through its backend (writes only the WAV)."""

from __future__ import annotations

import json
import math
import re
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

SPEAK_SCHEMA = "speak/1"
SPEAK_RUNNER = Path(__file__).resolve().with_name("speak_runner.py")
DEFAULT_TIMEOUT_SECONDS = 600
MAX_TEXT_CHARS = 2000
MIN_SPEED, MAX_SPEED = 0.5, 2.0
MAX_FAILURE_CHARS = 800
_VOICE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
_LANG_CODE = re.compile(r"[A-Za-z]{1,8}(?:[-_][A-Za-z0-9]{1,8})?")


class SpeakError(RuntimeError):
    """Classified speech-synthesis failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _invalid(message, remediation):
    return SpeakError("invalid_arguments", message, remediation)


def _type_candidates(config, model):
    """Model types to try: the config's own, or (when it names none) the first token of the directory name."""
    declared = [value for value in (config.get("model_type"), config.get("architecture")) if isinstance(value, str) and value]
    if declared:
        return declared
    name = model.name
    if model.parent.name == "snapshots":
        name = model.parent.parent.name.split("--")[-1]
    token = next((part for part in re.split(r"[^a-z0-9]+", name.lower()) if part), None)
    return [token] if token else []


def plan_speak(model_path, text, out, voice=None, speed=1.0, lang_code=None,
               manifests=None, registries=None, root=None):
    """Pick the speech backend for a converted model directory; pure apart from reading config.json."""
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SpeakError(
            "model_unreadable", "No readable config.json in {0}: {1}".format(model, error),
            "Pass the converted model directory.",
        ) from error
    if not isinstance(config, dict):
        raise SpeakError("model_unreadable", "config.json in {0} is not an object.".format(model), "Pass the converted model directory.")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS or any(
        ord(character) < 32 and character not in "\n\t" for character in text
    ):
        raise _invalid("The text must be 1..{0} characters of plain text.".format(MAX_TEXT_CHARS), "Pass --text with a sentence to speak.")
    target = Path(out).expanduser()
    if not target.is_absolute() or target.suffix.lower() != ".wav" or target.exists():
        raise _invalid("--out must be a new absolute .wav path.", "Pass a path such as /Users/me/Music/speech.wav.")
    if not target.parent.is_dir():
        raise _invalid("The directory of --out does not exist.", "Create the directory or pass a path inside an existing one.")
    if voice is not None and (not isinstance(voice, str) or not _VOICE.fullmatch(voice)):
        raise _invalid("--voice must be a short voice name such as af_heart.", "Pass --voice with a name the model lists.")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED:
        raise _invalid("--speed must be {0}..{1}.".format(MIN_SPEED, MAX_SPEED), "Pass --speed 1.0, for example.")
    if lang_code is not None and (not isinstance(lang_code, str) or not _LANG_CODE.fullmatch(lang_code)):
        raise _invalid("--lang-code must be a short language code such as a or en.", "Pass --lang-code en, for example.")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    candidates = _type_candidates(config, model)
    hits = []
    for candidate in candidates:
        hits = sorted(
            (hit for hit in lookup(candidate, manifests, registries) if hit["category"] == "text_to_speech"),
            key=lambda hit: hit["backend"],
        )
        if hits:
            break
    if not hits:
        raise SpeakError(
            "not_tts_model", "No text-to-speech backend implements model_type {0!r}.".format(candidates[0] if candidates else None),
            "Speech synthesis applies to converted text-to-speech models such as Kokoro.",
        )
    manifest = manifests[hits[0]["backend"]]
    if not is_installed(manifest, root):
        raise SpeakError(
            "backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
            "Install it first: mlx-agent backend install {0}.".format(manifest["id"]),
        )
    model_type = hits[0]["module"].rsplit(".", 1)[-1]
    argv = [
        str(backend_python(manifest, root)), str(SPEAK_RUNNER), "--model", str(model), "--model-type", model_type,
        "--text", text, "--out", str(target), "--speed", repr(float(speed)),
    ]
    if voice is not None:
        argv += ["--voice", voice]
    if lang_code is not None:
        argv += ["--lang-code", lang_code]
    return {
        "backend": manifest["id"], "model": str(model), "out": str(target), "voice": voice,
        "speed": float(speed), "lang_code": lang_code, "argv": argv,
    }


def _failure(plan, completed=None, reason=None):
    tail = reason
    if tail is None:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:]) or "no output"
        tail = redact_secrets(tail)[-MAX_FAILURE_CHARS:]
    return SpeakError(
        "speak_failed", "The {0} backend could not synthesize the text: {1}".format(plan["backend"], tail),
        "Check that the directory is a complete converted text-to-speech model and that --voice and --lang-code suit it.",
    )


def run_speak(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    manifests = load_manifests() if manifests is None else manifests
    try:
        sync_ports(manifests[plan["backend"]], root)
    except BackendError as error:
        raise SpeakError(error.code, str(error), error.remediation) from error
    try:
        completed = runner(
            plan["argv"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, env=backend_environment(), check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise _failure(plan, reason="speech synthesis exceeded {0} s; use shorter text or a larger --timeout".format(timeout)) from error
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
    keys = ("path", "sample_rate", "audio_seconds", "seconds", "load_seconds", "real_time_factor", "peak_memory_gb")
    return dict({"schema": SPEAK_SCHEMA, "backend": plan["backend"], "model": plan["model"],
                 "voice": plan["voice"], "speed": plan["speed"], "lang_code": plan["lang_code"]},
                **{key: result.get(key) for key in keys})
