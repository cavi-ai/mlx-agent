"""Durable task kinds: each entry owns its parameters, plan, run, media input and output artifact."""

from __future__ import annotations

import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from .backends import backend_environment
from .decide import plan_decide, run_decide, validate_request as validate_decision
from .describe import plan_describe, run_describe
from .generate import plan_generate, run_generate
from .inference_runner import InferenceError
from .music import plan_music, run_music, validate_parameters as validate_music
from .speak import plan_speak, run_speak
from .task_contracts import (MEDIA_REQUEST_SCHEMA, MEDIA_RUN_SCHEMA, REQUEST_SCHEMA, RUN_SCHEMA, TaskError,
                             absolute_path, encoded, finite_number)
from .transcribe import plan_transcribe, run_transcribe
from .video import plan_video, run_video


@dataclass(frozen=True)
class TaskKind:
    """One task kind. `plan` takes (model_path, parameters, output, media, directory)."""

    name: str
    result_schema: str
    legacy: bool
    validate: Callable[[dict], dict]
    planner: Callable[..., dict]
    runner: Callable[[dict], dict]
    input_field: Optional[str] = None
    input_media: Optional[str] = None
    output: Optional[str] = None
    output_check: Optional[Callable[[Path, dict], None]] = None

    @property
    def request_schema(self):
        return REQUEST_SCHEMA if self.legacy else MEDIA_REQUEST_SCHEMA

    @property
    def run_schema(self):
        return RUN_SCHEMA if self.legacy else MEDIA_RUN_SCHEMA

    def plan(self, request, directory, input_path=None):
        """Plan one run; output lands in `directory`, media comes from `input_path` or the request."""
        parameters = request["parameters"]
        media = input_path or (parameters[self.input_field] if self.input_field else None)
        output = str(Path(directory) / self.output) if self.output else None
        return self.planner(request["model_path"], parameters, output, media, Path(directory))

    def run(self, plan):
        return self.runner(plan)

    def check_output(self, path, plan):
        """Fail closed unless the run left a usable output artifact at `path`."""
        if self.output_check is not None:
            self.output_check(path, plan)

    def capability(self):
        entry = {"kind": self.name, "result_schema": self.result_schema}
        if not self.legacy:
            entry.update(request_schema=MEDIA_REQUEST_SCHEMA, run_schema=MEDIA_RUN_SCHEMA)
        entry.update(cancellation=True, requires_local_model=True, model_verification=True)
        return entry


def write_json(path, value):
    """Write one private JSON file that must not already exist."""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def _options(parameters, keys):
    return {key: parameters[key] for key in keys if key in parameters}


def _invalid(message):
    return TaskError("invalid_request", message)


def _require_keys(parameters, required, optional, message):
    if set(parameters) - set(required) - set(optional) or not set(required) <= set(parameters):
        raise _invalid(message)


def _require_string(parameters, *keys):
    for key in keys:
        if key in parameters and not isinstance(parameters[key], str):
            raise _invalid(key + " must be a string.")


def _require_integer(parameters, *keys):
    for key in keys:
        if key in parameters and type(parameters[key]) is not int:
            raise _invalid(key + " must be an integer.")


def _require_finite(parameters, *keys):
    for key in keys:
        if key in parameters and not finite_number(parameters[key]):
            raise _invalid(key + " must be a finite number.")


def _require_absolute(parameters, key):
    if not absolute_path(parameters[key]):
        raise _invalid(key + " must be an absolute local path.")


def _validate_classification(parameters):
    if set(parameters) != {"state", "questions"}:
        raise _invalid("Classification parameters need state and questions.")
    validate_decision(parameters)
    return parameters


def _validate_music(parameters):
    if "caption" not in parameters or set(parameters) - {"caption", "lyrics", "duration", "steps", "seed"}:
        raise _invalid("Music parameters need a caption and supported generation settings.")
    return validate_music(parameters["caption"], parameters.get("lyrics", "[instrumental]"),
                          parameters.get("duration", 15.0), parameters.get("steps", 30), parameters.get("seed", 42))


def _validate_transcribe(parameters):
    _require_keys(parameters, ("audio_path",), ("language",), "Transcription parameters need audio_path and optional language.")
    _require_absolute(parameters, "audio_path")
    _require_string(parameters, "language")
    return parameters


def _validate_speak(parameters):
    _require_keys(parameters, ("text",), ("voice", "speed", "lang_code"), "Speech parameters need text and supported synthesis settings.")
    _require_string(parameters, "text", "voice")
    if "speed" in parameters and (isinstance(parameters["speed"], bool) or not isinstance(parameters["speed"], (int, float))
                                   or not math.isfinite(parameters["speed"])):
        raise _invalid("speed must be a finite number.")
    _require_string(parameters, "lang_code")
    return parameters


def _validate_image_generation(parameters):
    _require_keys(parameters, ("prompt",), ("width", "height", "steps", "seed"),
                  "Image-generation parameters need prompt and optional size, steps and seed.")
    _require_string(parameters, "prompt")
    _require_integer(parameters, "width", "height", "steps", "seed")
    return parameters


def _validate_image_understanding(parameters):
    _require_keys(parameters, ("image_path", "prompt"), ("max_tokens", "temperature", "max_pixels"),
                  "Image-understanding parameters need image_path, prompt and supported options.")
    _require_absolute(parameters, "image_path")
    _require_string(parameters, "prompt")
    _require_integer(parameters, "max_tokens", "max_pixels")
    _require_finite(parameters, "temperature")
    return parameters


def _validate_video_generation(parameters):
    _require_keys(parameters, ("prompt",), ("width", "height", "frames", "steps", "seed", "fps"),
                  "Video-generation parameters need prompt and supported options.")
    _require_string(parameters, "prompt")
    _require_integer(parameters, "width", "height", "frames", "steps", "seed")
    _require_finite(parameters, "fps")
    return parameters


def _validate_video_understanding(parameters):
    _require_keys(parameters, ("video_path", "prompt"), ("fps", "temperature", "max_tokens", "max_pixels"),
                  "Video-understanding parameters need video_path, prompt and supported options.")
    _require_absolute(parameters, "video_path")
    _require_string(parameters, "prompt")
    _require_integer(parameters, "max_tokens", "max_pixels")
    _require_finite(parameters, "fps", "temperature")
    return parameters


def _plan_classification(model, parameters, output, media, directory):
    path = directory / "decision.json"
    write_json(path, parameters)
    return plan_decide(model, str(path))


def _plan_music(model, parameters, output, media, directory):
    return plan_music(model, parameters["caption"], parameters["lyrics"], output,
                      duration=parameters["duration"], steps=parameters["steps"], seed=parameters["seed"])


def _plan_transcribe(model, parameters, output, media, directory):
    return plan_transcribe(model, media, **_options(parameters, ("language",)))


def _plan_speak(model, parameters, output, media, directory):
    return plan_speak(model, parameters["text"], output, **_options(parameters, ("voice", "speed", "lang_code")))


def _plan_image_generation(model, parameters, output, media, directory):
    return plan_generate(model, parameters["prompt"], output, **_options(parameters, ("width", "height", "steps", "seed")))


def _plan_image_understanding(model, parameters, output, media, directory):
    return plan_describe(model, parameters["prompt"], image=media,
                         **_options(parameters, ("max_tokens", "temperature", "max_pixels")))


def _plan_video_generation(model, parameters, output, media, directory):
    return plan_video(model, parameters["prompt"], output,
                      **_options(parameters, ("width", "height", "frames", "fps", "steps", "seed")))


def _plan_video_understanding(model, parameters, output, media, directory):
    return plan_describe(model, parameters["prompt"], video=media,
                         **_options(parameters, ("fps", "max_tokens", "temperature", "max_pixels")))


def _wav_check(activity):
    def check(path, plan):
        try:
            with wave.open(str(path), "rb") as audio:
                frames = audio.getnframes()
                channels = audio.getnchannels()
                sample_width = audio.getsampwidth()
                if frames <= 0 or channels <= 0 or sample_width <= 0 or audio.getframerate() <= 0:
                    raise TaskError("invalid_audio_output", "{0} did not produce usable WAV audio.".format(activity))
                frame_bytes = channels * sample_width
                chunk_frames = max(1, 65536 // frame_bytes)
                received = 0
                remaining = frames
                while remaining:
                    count = min(remaining, chunk_frames)
                    received += len(audio.readframes(count))
                    remaining -= count
                if received != frames * frame_bytes:
                    raise TaskError("invalid_audio_output", "{0} produced a truncated WAV payload.".format(activity))
        except (OSError, EOFError, wave.Error) as error:
            raise TaskError("invalid_audio_output", "{0} did not produce a readable WAV file.".format(activity)) from error
    return check


def _backend_validator(script, code, unchecked, undecodable, remediation):
    def check(path, plan, runner=subprocess.run):
        validator = Path(__file__).with_name(script)
        try:
            completed = runner([plan["argv"][0], str(validator), str(path)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
                               env=backend_environment(), check=False)
        except (OSError, subprocess.SubprocessError) as error:
            raise TaskError(code, unchecked, "Check the installed {0} and retry the task.".format(remediation)) from error
        if completed.returncode != 0:
            raise TaskError(code, undecodable, "Check the {0} output and retry the task.".format(remediation))
    return check


check_png = _backend_validator("task_png_validator.py", "invalid_image_output",
                               "Image generation could not validate its PNG output.",
                               "Image generation did not produce a fully decodable PNG.", "image backend")
check_mp4 = _backend_validator("task_mp4_validator.py", "invalid_video_output",
                               "Video generation could not validate its MP4 output.",
                               "Video generation did not produce a decodable MP4 video stream.", "video backend")


KINDS: Dict[str, TaskKind] = {kind.name: kind for kind in (
    TaskKind("classification", "decide/1", True, _validate_classification, _plan_classification,
             lambda plan: run_decide(plan)),
    TaskKind("music_generation", "music/1", True, _validate_music, _plan_music, lambda plan: run_music(plan),
             output="output.wav", output_check=_wav_check("Music generation")),
    TaskKind("transcribe", "transcribe/1", False, _validate_transcribe, _plan_transcribe,
             lambda plan: run_transcribe(plan), input_field="audio_path", input_media="audio"),
    TaskKind("speak", "speak/1", False, _validate_speak, _plan_speak, lambda plan: run_speak(plan),
             output="output.wav", output_check=_wav_check("Speech synthesis")),
    TaskKind("image_generation", "generate/1", False, _validate_image_generation, _plan_image_generation,
             lambda plan: run_generate(plan), output="output.png", output_check=lambda path, plan: check_png(path, plan)),
    TaskKind("image_understanding", "describe/1", False, _validate_image_understanding, _plan_image_understanding,
             lambda plan: run_describe(plan), input_field="image_path", input_media="image"),
    TaskKind("video_generation", "video/1", False, _validate_video_generation, _plan_video_generation,
             lambda plan: run_video(plan), output="output.mp4", output_check=lambda path, plan: check_mp4(path, plan)),
    TaskKind("video_understanding", "describe/1", False, _validate_video_understanding, _plan_video_understanding,
             lambda plan: run_describe(plan), input_field="video_path", input_media="video"),
)}


def find(name):
    """The kind named `name`, or None for anything that is not a known kind name."""
    return KINDS.get(name) if isinstance(name, str) else None


def get(name):
    kind = find(name)
    if kind is None:
        raise TaskError("unsupported_task", "Unsupported task schema or kind.", "Run mlx-agent task capabilities --json.")
    return kind


def validate_parameters(kind, parameters):
    """Normalized parameters for `kind`; classified inference failures become task errors."""
    try:
        return kind.validate(parameters)
    except InferenceError as error:
        raise TaskError(error.code, str(error), error.remediation) from error
