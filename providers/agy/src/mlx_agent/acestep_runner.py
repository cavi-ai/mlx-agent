"""Run inside the isolated ACE-Step backend; write a new WAV and measured metrics.

ACE-Step copies its model code into the checkpoint folder it loads, so it loads from a private
folder of links to the checkpoint's weights and configs and never writes into the model folder.
The network is closed: every file comes from the folders beside the checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import tempfile
import time

SIBLINGS = ("vae", "Qwen3-Embedding-0.6B")


def offline(*args, **kwargs):
    raise OSError("network access is disabled for music generation")


def link_folder(source, target):
    """Link every file of a checkpoint folder except Python code into a new folder."""
    target.mkdir()
    for entry in sorted(Path(source).iterdir()):
        if entry.suffix == ".py" or entry.name == "__pycache__":
            continue
        (target / entry.name).symlink_to(entry.resolve())


def stage(model, planner, directory):
    """A checkpoints folder that holds links to the DiT, its VAE, text encoder and planner."""
    checkpoints = Path(directory) / "checkpoints"
    checkpoints.mkdir()
    for name in (model.name, *SIBLINGS, *([planner] if planner else [])):
        link_folder(model.parent / name, checkpoints / name)
    return checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "caption", "lyrics", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shift", type=float, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--time-signature", type=int)
    parser.add_argument("--reference")
    parser.add_argument("--planner")
    args = parser.parse_args(argv)
    model, out = Path(args.model), Path(args.out)
    from acestep.model_downloader import _CHECKPOINT_TO_VARIANT
    if model.name not in _CHECKPOINT_TO_VARIANT:
        print(json.dumps({"error": "unknown_checkpoint", "detail": model.name}), flush=True)
        return 2
    socket.socket.connect = offline
    socket.create_connection = offline
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    import mlx.core as mx
    import soundfile
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    class LocalHandler(AceStepHandler):
        """Never download: the staged folders hold every file the DiT, VAE and text encoder read."""

        def _ensure_models_present(self, *, checkpoint_path, config_path, prefer_source, vae_variant=None):
            return None

    instrumental = args.lyrics.strip().lower() == "[instrumental]"
    # The take is written beside the output, so linking it to the output name stays on one volume.
    with tempfile.TemporaryDirectory(prefix="mlx-agent-acestep-") as directory, \
            tempfile.TemporaryDirectory(prefix=".acestep-", dir=out.parent) as scratch:
        checkpoints = stage(model, args.planner, directory)
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)
        started = time.perf_counter()
        handler = LocalHandler()
        status, ok = handler.initialize_service(
            project_root=directory, config_path=model.name, device="mps", use_flash_attention=False,
            compile_model=False, offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None, use_mlx_dit=True)
        if not ok:
            raise RuntimeError("ACE-Step did not load: {0}".format(status))
        planner = None
        if args.planner:
            planner = LLMHandler()
            status, ok = planner.initialize(checkpoint_dir=str(checkpoints), lm_model_path=args.planner,
                                            backend="mlx", device="mps")
            if not ok:
                raise RuntimeError("The ACE-Step planner did not load: {0}".format(status))
        load_seconds = time.perf_counter() - started
        mx.reset_peak_memory()
        params = GenerationParams(
            task_type="text2music", caption=args.caption, lyrics="[Instrumental]" if instrumental else args.lyrics,
            instrumental=instrumental, bpm=args.bpm,
            timesignature=str(args.time_signature) if args.time_signature else "",
            duration=args.duration, inference_steps=args.steps, guidance_scale=args.guidance, shift=args.shift,
            seed=args.seed, thinking=planner is not None, use_cot_metas=planner is not None,
            use_cot_caption=planner is not None, use_cot_language=planner is not None,
            reference_audio=args.reference)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[args.seed], audio_format="wav")
        started = time.perf_counter()
        result = generate_music(dit_handler=handler, llm_handler=planner, params=params, config=config,
                                save_dir=scratch)
        seconds = time.perf_counter() - started
        if not result.success or not result.audios:
            raise RuntimeError("ACE-Step generated nothing: {0}".format(result.error))
        written = Path(result.audios[0]["path"])
        info = soundfile.info(str(written))
        # A hard link claims the new output name or fails, so an existing file is never replaced.
        os.link(written, out)
    audio_seconds = info.frames / info.samplerate
    print(json.dumps({"path": str(out), "sample_rate": info.samplerate, "audio_seconds": audio_seconds,
                      "seconds": seconds, "load_seconds": load_seconds, "real_time_factor": seconds / audio_seconds,
                      "peak_memory_gb": mx.get_peak_memory() / 1e9}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
