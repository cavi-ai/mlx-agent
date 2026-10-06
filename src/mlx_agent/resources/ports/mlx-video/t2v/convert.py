"""Convert a Wan text-to-video checkpoint (original layout, already in the cache) to an mlx-video directory.

Nothing is downloaded: ``--hf-path`` is a local checkpoint directory or a repo id already in the cache.
The converter writes bfloat16 transformer and T5 weights, a float32 VAE, ``config.json`` (carrying
``model_type`` and, with ``--quantize``, ``quantization``), and the tokenizer files the pipeline loads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

MODEL_TYPE = "t2v"
TOKENIZER_SOURCE = ("google", "umt5-xxl")
TOKENIZER_DIR = "tokenizer"
TOKENIZER_FILES = ("special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json")
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
VAE_FILES = ("Wan2.1_VAE.pth", "Wan2.2_VAE.pth")


def convert(hf_path: str, mlx_path: str, quantize: bool = False, q_bits: int = 4) -> Path:
    base, out = Path(hf_path).expanduser(), Path(mlx_path).expanduser()
    if not base.is_dir():
        from huggingface_hub import snapshot_download

        try:
            base = Path(snapshot_download(hf_path, local_files_only=True))
        except (OSError, ValueError):
            raise SystemExit("{0} is not a checkpoint directory or a repository in the cache.".format(hf_path))
    if out.exists() and any(out.iterdir()):
        raise SystemExit("{0} already exists and is not empty.".format(out))
    try:
        source = json.loads((base / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("{0} is not a Wan checkpoint (no readable config.json).".format(base))
    if not isinstance(source, dict) or source.get("model_type") != MODEL_TYPE:
        raise SystemExit("{0} is a {1!r} checkpoint, not a {2!r} text-to-video checkpoint.".format(
            base, source.get("model_type") if isinstance(source, dict) else None, MODEL_TYPE))
    missing = []
    if not (base / T5_FILE).is_file():
        missing.append(T5_FILE)
    if not any((base / name).is_file() for name in VAE_FILES):
        missing.append(VAE_FILES[0])
    if not (any(base.glob("diffusion_pytorch_model*.safetensors")) or (base / "low_noise_model").is_dir()):
        missing.append("diffusion_pytorch_model*.safetensors")
    if missing:
        raise SystemExit("{0} is missing Wan checkpoint files: {1}.".format(base, ", ".join(missing)))
    tokenizer = base.joinpath(*TOKENIZER_SOURCE)
    absent = [name for name in TOKENIZER_FILES if not (tokenizer / name).is_file()]
    if absent:
        raise SystemExit("{0} is missing the tokenizer files {1}.".format(tokenizer, ", ".join(absent)))
    from mlx_video.convert_wan import convert_wan_checkpoint

    print("[INFO] Converting {0}".format(base), flush=True)
    convert_wan_checkpoint(
        str(base), str(out), dtype="bfloat16", model_version="auto", quantize=quantize, bits=q_bits, group_size=64,
    )
    destination = out / TOKENIZER_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        shutil.copyfile(tokenizer / name, destination / name)
    try:
        written = json.loads((out / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("the converter wrote no readable config.json in {0}.".format(out))
    if written.get("model_type") != MODEL_TYPE or (quantize and not written.get("quantization")):
        raise SystemExit("the converted config.json in {0} lacks model_type {1!r} or its quantization.".format(out, MODEL_TYPE))
    print("[INFO] Conversion complete", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Wan checkpoint directory, or a repo id already in the cache")
    parser.add_argument("--mlx-path", required=True, help="destination directory")
    parser.add_argument("-q", "--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=4, choices=(4, 8))
    arguments = parser.parse_args(argv)
    convert(arguments.hf_path, arguments.mlx_path, arguments.quantize, arguments.q_bits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
