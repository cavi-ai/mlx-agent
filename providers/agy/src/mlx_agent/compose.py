"""Plan a song with the ACE-Step planner beside a music model (read-only).

The planner rewrites a caption into the detailed caption ACE-Step was trained on, structures the lyrics, and
chooses the tempo, key, length, meter and language a take leaves open. Nothing is rendered.
"""

import json
import math
import subprocess
from pathlib import Path

from .backends import backend_python, is_installed, load_manifests, load_registries, lookup
from .inference_runner import RunnerContract, invoke
from .music import MusicError, _acestep_planner, validate_parameters

COMPOSE_SCHEMA = "compose/1"
COMPOSE_RUNNER = Path(__file__).with_name("acestep_compose_runner.py")
DEFAULT_TIMEOUT_SECONDS = 900
MAX_TEMPERATURE = 2.0
RESULT_KEYS = ("caption", "lyrics", "bpm", "duration", "key", "time_signature", "language", "seconds", "load_seconds")


def plan_compose(model_path, caption, lyrics, bpm=None, duration=None, key=None, time_signature=None, language=None,
                 temperature=0.85, manifests=None, registries=None, root=None):
    """The planner run for an ACE-Step checkpoint; bpm, duration, key, time signature and language bind the plan."""
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        if not model.is_absolute() or not isinstance(config, dict):
            raise ValueError("An absolute local model directory is required")
    except (OSError, ValueError) as error:
        raise MusicError("model_unreadable", str(error), "Pass an existing ACE-Step checkpoint directory.") from error
    checked = validate_parameters(caption, lyrics, 15.0 if duration is None else duration, bpm=bpm,
                                  time_signature=time_signature, key=key)
    if language is not None and (not isinstance(language, str) or not language.isascii() or not language.isalpha()
                                 or not 2 <= len(language) <= 8):
        raise MusicError("invalid_arguments", "The language must be a language code such as en.", "Use --language en.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) \
            or not 0 <= temperature <= MAX_TEMPERATURE:
        raise MusicError("invalid_arguments", "Temperature must be 0..{0}.".format(MAX_TEMPERATURE), "Use --temperature 0.85.")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = [hit for hit in lookup(config.get("model_type", ""), manifests, registries)
            if hit["category"] == "music_generation" and hit["backend"] == "ace-step"]
    if not hits:
        raise MusicError("not_music_model", "Only ACE-Step checkpoints have a song planner.", "Pass an ACE-Step 1.5 checkpoint.")
    planner = _acestep_planner(model)
    if planner is None:
        raise MusicError("model_unreadable", "No ACE-Step planner (acestep-5Hz-lm-*) is beside {0}.".format(model),
                         "Keep acestep-5Hz-lm-1.7B in the ACE-Step checkpoints folder.")
    manifest = manifests["ace-step"]
    if not is_installed(manifest, root):
        raise MusicError("backend_not_installed", "The ace-step backend is not installed.",
                         "Install it first: mlx-agent backend install ace-step.")
    argv = [str(backend_python(manifest, root)), str(COMPOSE_RUNNER), "--planner", str(model.parent / planner),
            "--caption=" + caption, "--lyrics=" + lyrics, "--temperature", repr(float(temperature))]
    bound = {"bpm": bpm, "duration": duration, "key": checked.get("key"), "time_signature": time_signature,
             "language": language}
    for name, value in bound.items():
        if value is not None:
            argv += ["--" + name.replace("_", "-") + "=" + str(value)]
    return {"backend": "ace-step", "model": str(model), "planner": planner, "temperature": float(temperature),
            "bound": {name: value for name, value in bound.items() if value is not None}, "argv": argv}


CONTRACT = RunnerContract(
    verb="compose", error=MusicError, action="plan the song",
    accept=lambda value, plan: isinstance(value.get("caption"), str) and bool(value["caption"].strip())
    and isinstance(value.get("lyrics"), str),
    remediation="Check that the ACE-Step planner beside the checkpoint is complete.",
    activity="Song planning", timeout_remediation="Use a larger --timeout.",
    environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
)


def run_compose(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    return dict({"schema": COMPOSE_SCHEMA, "backend": plan["backend"], "model": plan["model"], "planner": plan["planner"],
                 "temperature": plan["temperature"], "bound": dict(plan["bound"])},
                **{key: result.get(key) for key in RESULT_KEYS})
