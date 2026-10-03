"""Transcribe one audio file with a converted speech model through its backend (read-only)."""

from __future__ import annotations

import json
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

TRANSCRIBE_SCHEMA = "transcribe/1"
TRANSCRIBE_RUNNER = Path(__file__).resolve().with_name("transcribe_runner.py")
DEFAULT_TIMEOUT_SECONDS = 300
_LANGUAGE = re.compile(r"[a-z]{2,8}")


class TranscribeError(RuntimeError):
    """Classified transcription failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def plan_transcribe(model_path, audio_path, language=None, manifests=None, registries=None, root=None):
    """Pick the speech backend for a converted model directory; pure apart from reading config.json."""
    model = Path(model_path).expanduser().resolve()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TranscribeError(
            "model_unreadable", "No readable config.json in {0}: {1}".format(model, error),
            "Pass the converted model directory.",
        ) from error
    model_type = config.get("model_type") if isinstance(config, dict) else None
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise TranscribeError("audio_not_found", "No audio file at {0}.".format(audio), "Pass a WAV, FLAC, or MP3 file.")
    if language is not None and not _LANGUAGE.fullmatch(language):
        raise TranscribeError("invalid_arguments", "language must be a short lowercase code.", "Pass --language en, for example.")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    backends = sorted({hit["backend"] for hit in lookup(model_type, manifests, registries) if hit["category"] == "speech_to_text"})
    if not backends:
        raise TranscribeError(
            "not_speech_model", "No speech-to-text backend implements model_type {0!r}.".format(model_type),
            "Transcription applies to converted speech-to-text models.",
        )
    manifest = manifests[backends[0]]
    if not is_installed(manifest, root):
        raise TranscribeError(
            "backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
            "Install it first: mlx-agent backend install {0}.".format(manifest["id"]),
        )
    argv = [str(backend_python(manifest, root)), str(TRANSCRIBE_RUNNER), "--model", str(model), "--audio", str(audio)]
    if language:
        argv += ["--language", language]
    return {"backend": manifest["id"], "model": str(model), "audio": str(audio), "language": language, "argv": argv}


def run_transcribe(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    manifests = load_manifests() if manifests is None else manifests
    try:
        sync_ports(manifests[plan["backend"]], root)
    except BackendError as error:
        raise TranscribeError(error.code, str(error), error.remediation) from error
    try:
        completed = runner(
            plan["argv"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, env=backend_environment(), check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TranscribeError(
            "transcribe_timeout", "Transcription exceeded {0} s.".format(timeout),
            "Use a shorter clip or a larger --timeout.",
        ) from error
    result = None
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            result = value
            break
    if completed.returncode != 0 or result is None:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:])
        raise TranscribeError(
            "transcribe_failed", "The {0} backend could not transcribe the clip: {1}".format(plan["backend"], tail or "no output"),
            "Check that the directory is a complete converted speech model.",
        )
    return {
        "schema": TRANSCRIBE_SCHEMA, "backend": plan["backend"], "model": plan["model"], "audio": plan["audio"],
        "language": plan["language"], "text": result["text"],
        "seconds": result.get("seconds"), "audio_seconds": result.get("audio_seconds"),
    }
