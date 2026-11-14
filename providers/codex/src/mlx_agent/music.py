"""Generate music from existing local weights in the isolated audio backend."""

import json
import math
import subprocess
from pathlib import Path

from .backends import backend_python, is_installed, load_manifests, load_registries, lookup
from .inference_runner import InferenceError, RunnerContract, invoke
from .music_runner import tokenizer_directory

MUSIC_RUNNER = Path(__file__).with_name("music_runner.py")
ACESTEP_RUNNER = Path(__file__).with_name("acestep_runner.py")
MAX_STEPS = 64
STEP_LIMITS = {"mlx-audio": 30}
# ACE-Step 1.5 loads its VAE and text encoder from folders beside the DiT checkpoint.
ACESTEP_SIBLINGS = ("vae", "Qwen3-Embedding-0.6B")
ACESTEP_PLANNER = "acestep-5Hz-lm-1.7B"
ACESTEP_TURBO_STEPS = 20
TIME_SIGNATURES = (2, 3, 4, 6)
REFERENCE_FORMATS = (".wav", ".flac", ".mp3", ".m4a", ".aif", ".aiff", ".ogg")


class MusicError(InferenceError):
    """Classified music-generation failure safe to surface in a result envelope."""


def validate_parameters(caption, lyrics, duration=15.0, steps=30, seed=42, bpm=None, time_signature=None, reference=None):
    """Validate generation settings without accessing weights or a backend."""
    for name, text, limit in (("caption", caption, 2000), ("lyrics", lyrics, 10000)):
        if not isinstance(text, str) or not text.strip() or len(text) > limit or any(ord(c) < 32 and c not in "\n\t" for c in text):
            raise MusicError("invalid_arguments", "Invalid {0}.".format(name), "Provide a caption and lyrics; use [instrumental] for instrumental music.")
    if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or not 0 < duration <= 360:
        raise MusicError("invalid_arguments", "Duration must be greater than 0 and at most 360 seconds.", "Use --duration 15.")
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise MusicError("invalid_arguments", "Steps or seed is out of range.", "Use 1..{0} steps and a 32-bit unsigned seed.".format(MAX_STEPS))
    if bpm is not None and (type(bpm) is not int or not 30 <= bpm <= 300):
        raise MusicError("invalid_arguments", "BPM must be a whole number from 30 to 300.", "Use --bpm 120.")
    if time_signature is not None and (type(time_signature) is not int or time_signature not in TIME_SIGNATURES):
        raise MusicError("invalid_arguments", "Time signature must be 2, 3, 4 or 6 beats per bar.", "Use --time-signature 4.")
    if reference is not None:
        path = Path(reference).expanduser() if isinstance(reference, str) else None
        if path is None or not path.is_absolute() or not path.is_file() or path.suffix.lower() not in REFERENCE_FORMATS:
            raise MusicError("invalid_arguments", "The reference must be an existing absolute audio file.",
                             "Pass --reference as a {0} file.".format(", ".join(REFERENCE_FORMATS)))
    settings = {"bpm": bpm, "time_signature": time_signature, "reference": reference}
    return dict({"caption": caption, "lyrics": lyrics, "duration": duration, "steps": steps, "seed": seed},
                **{key: value for key, value in settings.items() if value is not None})


def _acestep_planner(model):
    """The planner beside an ACE-Step checkpoint, after refusing a checkpoint missing what it loads."""
    missing = [name for name in ACESTEP_SIBLINGS if not (model.parent / name / "config.json").is_file()]
    if not ((model / "model.safetensors").is_file() or (model / "model.safetensors.index.json").is_file()):
        missing.append(model.name + "/model.safetensors")
    if not (model / "silence_latent.pt").is_file():
        missing.append(model.name + "/silence_latent.pt")
    if missing:
        raise MusicError("model_unreadable", "The ACE-Step checkpoint is missing {0}.".format(", ".join(missing)),
                         "Keep the DiT folder beside vae and Qwen3-Embedding-0.6B in the ACE-Step checkpoints folder.")
    planners = sorted(path.name for path in model.parent.glob("acestep-5Hz-lm-*") if (path / "config.json").is_file())
    return ACESTEP_PLANNER if ACESTEP_PLANNER in planners else (planners[0] if planners else None)


def plan_music(model_path, caption, lyrics, out, duration=15.0, steps=30, seed=42, bpm=None, time_signature=None,
               reference=None, manifests=None, registries=None, root=None):
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        if not model.is_absolute() or not isinstance(config, dict):
            raise ValueError("An absolute local model directory is required")
    except (OSError, ValueError) as error:
        raise MusicError("model_unreadable", str(error), "Pass an existing converted music model directory.") from error
    validate_parameters(caption, lyrics, duration, steps, seed, bpm, time_signature, reference)
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
    acestep = manifest["id"] == "ace-step"
    if not acestep and (bpm, time_signature, reference) != (None, None, None):
        raise MusicError("invalid_arguments", "BPM, time signature and reference apply to ACE-Step models only.",
                         "Drop --bpm, --time-signature and --reference for this model.")
    turbo = acestep and config.get("is_turbo") is True
    limit = ACESTEP_TURBO_STEPS if turbo else STEP_LIMITS.get(manifest["id"], MAX_STEPS)
    if steps > limit:
        raise MusicError("invalid_arguments", "This model takes 1..{0} steps.".format(limit),
                         "Use --steps 8." if turbo else "Use --steps {0} or fewer.".format(limit))
    planner = _acestep_planner(model) if acestep else None
    if not is_installed(manifest, root):
        raise MusicError("backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
                         "Install it first: mlx-agent backend install {0}.".format(manifest["id"]))
    plan = {"backend": manifest["id"], "model": str(model), "out": str(target),
            "duration": duration, "steps": steps, "seed": seed,
            "argv": [str(backend_python(manifest, root)), str(ACESTEP_RUNNER if acestep else MUSIC_RUNNER),
                     "--model", str(model), "--caption=" + caption, "--lyrics=" + lyrics, "--out", str(target),
                     "--duration", str(duration), "--steps", str(steps), "--seed", str(seed)]}
    if acestep:
        # Turbo checkpoints are distilled for shift 3 and ignore guidance; the others use ACE-Step's defaults.
        plan.update(bpm=bpm, time_signature=time_signature, reference=reference, planner=planner,
                    shift=3.0 if turbo else 1.0, guidance=7.0)
        plan["argv"] += ["--shift", str(plan["shift"]), "--guidance", str(plan["guidance"])]
        for flag, value in (("--bpm", bpm), ("--time-signature", time_signature), ("--reference", reference), ("--planner", planner)):
            if value is not None:
                plan["argv"] += [flag, str(value)]
    return plan


def _wrote_output(value, plan):
    target = Path(plan["out"])
    return value.get("path") == plan["out"] and target.is_file() and not target.is_symlink()


def _unknown_checkpoint(plan, detail):
    return ("ACE-Step does not know the checkpoint folder {0}.".format(detail),
            "Keep the checkpoint under its published name, such as acestep-v15-turbo or acestep-v15-xl-sft.")


CONTRACT = RunnerContract(
    verb="music", error=MusicError, accept=_wrote_output, action="generate the music",
    remediation="Check local model completeness and the audio backend.",
    activity="Music generation", timeout_remediation="Use a shorter duration or a larger timeout.",
    refusals={"unknown_checkpoint": _unknown_checkpoint},
    # Generation never resolves or downloads missing weights or tokenizer files.
    environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
)


def run_music(plan, manifests=None, root=None, timeout=3600, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    keys = ("path", "sample_rate", "audio_seconds", "seconds", "load_seconds", "real_time_factor", "peak_memory_gb")
    settings = {key: plan[key] for key in ("bpm", "time_signature", "reference", "planner", "shift", "guidance") if key in plan}
    return dict({"schema": "music/1", "backend": plan["backend"], "model": plan["model"],
                 "duration": plan["duration"], "steps": plan["steps"], "seed": plan["seed"]},
                **settings, **{key: result.get(key) for key in keys})
