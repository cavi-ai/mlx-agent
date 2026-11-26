"""Run inside the isolated ACE-Step backend; write a new WAV and measured metrics.

ACE-Step copies its model code into the checkpoint folder it loads and keeps a progress cache in
its project root, so both live in a private folder of links to the checkpoint's weights and configs;
nothing is written into the model folder or the working directory.
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
VAE_HOP_FRAMES = 1920  # ACE-Step's 25 Hz latents at 48 kHz.


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


def adapt_quantizer(checkpoint):
    """Adapt the trusted model code copied into the private staging folder."""
    files = list(Path(checkpoint).glob("modeling_acestep_v15_*.py"))
    if len(files) != 1 or files[0].is_symlink():
        raise RuntimeError("ACE-Step staged model code is missing or ambiguous")
    path = files[0]
    source = path.read_text(encoding="utf-8")
    original = "from vector_quantize_pytorch import ResidualFSQ"
    replacement = "from acestep_fsq import ResidualFSQ"
    if source.count(original) != 1 or replacement in source:
        raise RuntimeError("ACE-Step staged quantizer import is unexpected")
    path.write_text(source.replace(original, replacement), encoding="utf-8")


# ACE-Step planner metadata names and the names a result reports them under.
PLANNED = {"bpm": "bpm", "keyscale": "key", "duration": "duration", "timesignature": "time_signature",
           "language": "language", "vocal_language": "language"}


def planned(metadata, caption, keep_caption):
    """The tempo, key, length, meter and language the planner chose, and the caption the DiT was given."""
    metadata = metadata if isinstance(metadata, dict) else {}
    found = {}
    for source, name in PLANNED.items():
        if name not in found and metadata.get(source) not in (None, "", "N/A"):
            found[name] = metadata[source]
    # The planner writes numbers as text at times, and a meter as "4" or "4/4".
    for name, kind in (("bpm", int), ("duration", float), ("time_signature", int)):
        if name in found:
            try:
                found[name] = kind(float(str(found[name]).split("/")[0]))
            except ValueError:
                del found[name]
    found["caption"] = caption if keep_caption else str(metadata.get("caption") or caption)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "caption", "lyrics", "out"):
        parser.add_argument("--" + name, required=True)
    # Cover, repaint and complete keep the source's length, so they take no duration.
    parser.add_argument("--duration", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shift", type=float, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--time-signature", type=int)
    parser.add_argument("--reference")
    parser.add_argument("--key")
    parser.add_argument("--keep-caption", action="store_true")
    parser.add_argument("--planner")
    parser.add_argument("--task", choices=("text2music", "cover", "repaint", "complete"), default="text2music")
    parser.add_argument("--source")
    parser.add_argument("--track", action="append")
    parser.add_argument("--cover-strength", type=float, default=1.0)
    parser.add_argument("--repaint-start", type=float, default=0.0)
    parser.add_argument("--repaint-end", type=float, default=-1.0)
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

        repaint_source = None

        def process_src_audio(self, audio_file):
            if self.repaint_source is not None and audio_file == args.source:
                return self.repaint_source
            return super().process_src_audio(audio_file)

        def _ensure_models_present(self, *, checkpoint_path, config_path, prefer_source, vae_variant=None):
            return None

        def _sync_model_code_if_needed(self, config_path, checkpoint_path):
            super()._sync_model_code_if_needed(config_path, checkpoint_path)
            adapt_quantizer(Path(checkpoint_path) / config_path)

    class LocalPlanner(LLMHandler):
        def has_all_metas(self, user_metadata):
            # The caller can bind every musical setting while the caption or vocal language still needs planning.
            return False

    instrumental = args.lyrics.strip().lower() == "[instrumental]"
    # The take is written beside the output, so linking it to the output name stays on one volume.
    with tempfile.TemporaryDirectory(prefix="mlx-agent-acestep-") as directory, \
            tempfile.TemporaryDirectory(prefix=".acestep-", dir=out.parent) as scratch:
        checkpoints = stage(model, args.planner, directory)
        # ACE-Step keeps its progress cache under its project root, the working directory by default.
        os.environ.update(ACESTEP_CHECKPOINTS_DIR=str(checkpoints), ACESTEP_PROJECT_ROOT=directory)
        started = time.perf_counter()
        handler = LocalHandler()
        repaint_frames = None
        if args.task == "repaint":
            source = handler.process_src_audio(args.source)
            if source is None or source.shape[-1] == 0:
                detail = "The repaint source has no readable audio."
                print(json.dumps({"error": "invalid_arguments", "detail": detail}), flush=True)
                return 2
            repaint_frames = source.shape[-1]
            source_seconds = repaint_frames / handler.sample_rate
            if args.repaint_end > source_seconds:
                detail = "Repaint ends at {0:g} s, beyond the {1:.9g} s source.".format(args.repaint_end, source_seconds)
                print(json.dumps({"error": "invalid_arguments", "detail": detail}), flush=True)
                return 2
            # The VAE drops a partial final frame. Pad the decoded source once and reuse it for generation.
            padding = -repaint_frames % VAE_HOP_FRAMES
            if padding:
                from torch.nn.functional import pad

                source = pad(source, (0, padding))
            handler.repaint_source = source
        status, ok = handler.initialize_service(
            project_root=directory, config_path=model.name, device="mps", use_flash_attention=False,
            compile_model=False, offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None, use_mlx_dit=True)
        if not ok:
            raise RuntimeError("ACE-Step did not load: {0}".format(status))
        planner = None
        if args.planner:
            planner = LocalPlanner()
            status, ok = planner.initialize(checkpoint_dir=str(checkpoints), lm_model_path=args.planner,
                                            backend="mlx", device="mps")
            if not ok:
                raise RuntimeError("The ACE-Step planner did not load: {0}".format(status))
        load_seconds = time.perf_counter() - started
        mx.reset_peak_memory()
        task = {"task_type": "text2music"} if args.task == "text2music" else {
            "task_type": args.task,
            "instruction": handler.generate_instruction(task_type=args.task, complete_track_classes=args.track),
            "src_audio": args.source, "audio_cover_strength": args.cover_strength,
            "repainting_start": args.repaint_start, "repainting_end": args.repaint_end}
        params = GenerationParams(
            caption=args.caption, lyrics="[Instrumental]" if instrumental else args.lyrics,
            instrumental=instrumental, bpm=args.bpm, keyscale=args.key or "",
            timesignature=str(args.time_signature) if args.time_signature else "",
            duration=args.duration, inference_steps=args.steps, guidance_scale=args.guidance, shift=args.shift,
            seed=args.seed, thinking=planner is not None, use_cot_metas=planner is not None,
            # A kept caption reaches the DiT as written; the planner still chooses what the take leaves open.
            use_cot_caption=planner is not None and not args.keep_caption, use_cot_language=planner is not None,
            reference_audio=args.reference, enable_normalization=args.task != "repaint", **task)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[args.seed], audio_format="wav")
        started = time.perf_counter()
        result = generate_music(dit_handler=handler, llm_handler=planner, params=params, config=config,
                                save_dir=scratch)
        seconds = time.perf_counter() - started
        if not result.success or not result.audios:
            raise RuntimeError("ACE-Step generated nothing: {0}".format(result.error))
        written = Path(result.audios[0]["path"])
        info = soundfile.info(str(written))
        if repaint_frames is not None:
            if info.samplerate != handler.sample_rate or info.frames < repaint_frames:
                raise RuntimeError("ACE-Step repaint did not render the full source at the expected sample rate.")
            if info.frames > repaint_frames:
                with soundfile.SoundFile(str(written), "r+") as wav:
                    wav.truncate(repaint_frames)
                info = soundfile.info(str(written))
        # A hard link claims the new output name or fails, so an existing file is never replaced.
        os.link(written, out)
    audio_seconds = info.frames / info.samplerate
    report = {"path": str(out), "sample_rate": info.samplerate, "audio_seconds": audio_seconds,
              "seconds": seconds, "load_seconds": load_seconds, "real_time_factor": seconds / audio_seconds,
              "peak_memory_gb": mx.get_peak_memory() / 1e9}
    if planner is not None:
        report["planned"] = planned((result.extra_outputs or {}).get("lm_metadata"), args.caption, args.keep_caption)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
