"""Assemble an mlx-video LTX-2.3 directory from an MLX LTX-2.3 snapshot and its Gemma 3 12B text encoder.

Nothing is downloaded and no weights are rewritten: every weight file is cloned (copied where the
volume cannot clone) into ``--mlx-path``. The converter writes ``config.json`` (``model_type`` ``ltx2``), puts the
text encoder under ``text_encoder/`` and its tokenizer under ``tokenizer/``, and rewrites the VAE
encoder's ``encoder_blocks`` from the encoder weights, since published LTX-2.3 snapshots declare
blocks the weights do not have.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import sys
from pathlib import Path

MODEL_TYPE = "ltx2"
TEXT_ENCODER = "mlx-community/gemma-3-12b-it-bf16"
SIGNATURE = ("transformer/config.json", "text_projections/model.safetensors", "vae/encoder/config.json",
             "audio_vae/encoder/config.json", "vocoder/config.json")
COMPONENTS = ("transformer", "text_projections", "vae", "audio_vae", "vocoder")
UPSCALER = "*spatial-upscaler-x2*.safetensors"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.json")
STRIDES = {"compress_space_res": 4, "compress_time_res": 2, "compress_all_res": 8}
RESIDUAL = re.compile(r"down_blocks\.(\d+)\.res_blocks\.(\d+)\.")
# The planner aligns sizes to patch_size x vae_stride: stage 1 renders at half size, so 64 pixels.
CONFIG = {"model_type": MODEL_TYPE, "model_version": "2.3", "pipeline": "distilled", "vae_stride": [8, 32, 32],
          "patch_size": [1, 2, 2], "sample_fps": 24, "sample_steps": 8}


def _clone(source: Path, target: Path) -> None:
    # A hard link would change the source's link count and change time, which fetch evidence binds.
    target.parent.mkdir(parents=True, exist_ok=True)
    clonefile = getattr(ctypes.CDLL(None, use_errno=True), "clonefile", None)
    if clonefile is None or clonefile(os.fsencode(source), os.fsencode(target), 0) != 0:
        shutil.copy2(source, target)


def _shapes(directory: Path) -> dict:
    from safetensors import safe_open

    shapes = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="numpy") as handle:
            for name in handle.keys():
                shapes[name] = tuple(handle.get_slice(name).get_shape())
    return shapes


def encoder_blocks(blocks: list, shapes: dict) -> list:
    """The encoder blocks the weights hold: residual counts and compression multipliers read from them."""
    residuals = {}
    for name in shapes:
        match = RESIDUAL.match(name)
        if match:
            index, layer = int(match.group(1)), int(match.group(2))
            residuals[index] = max(residuals.get(index, 0), layer + 1)
    channels = shapes["conv_in.conv.weight"][0]
    fixed = []
    for index, (name, options) in enumerate(blocks):
        options = dict(options)
        if name == "res_x":
            if index not in residuals:
                raise SystemExit("the VAE encoder weights hold no residual blocks for block {0}.".format(index))
            options["num_layers"] = residuals[index]
        elif name in STRIDES:
            weight = shapes.get("down_blocks.{0}.conv.conv.weight".format(index))
            if weight is None:
                raise SystemExit("the VAE encoder weights hold no convolution for block {0}.".format(index))
            out = weight[0] * STRIDES[name]
            if out % channels:
                raise SystemExit("block {0} maps {1} to {2} channels, not a whole multiple.".format(index, channels, out))
            options["multiplier"] = out // channels
            channels = out
        else:
            raise SystemExit("unknown VAE encoder block {0!r}.".format(name))
        fixed.append([name, options])
    return fixed


def convert(hf_path: str, mlx_path: str, sources: dict | None = None) -> Path:
    base, out = Path(hf_path).expanduser(), Path(mlx_path).expanduser()
    if not base.is_dir():
        from huggingface_hub import snapshot_download

        try:
            base = Path(snapshot_download(hf_path, local_files_only=True))
        except (OSError, ValueError):
            raise SystemExit("{0} is not a snapshot directory or a repository in the cache.".format(hf_path))
    if out.exists() and any(out.iterdir()):
        raise SystemExit("{0} already exists and is not empty.".format(out))
    missing = [name for name in SIGNATURE if not (base / name).is_file()]
    upscalers = sorted(base.glob(UPSCALER))
    if not upscalers:
        missing.append(UPSCALER)
    if missing:
        raise SystemExit("{0} is not an MLX LTX-2.3 snapshot; missing {1}.".format(base, ", ".join(missing)))
    encoder = Path((sources or {}).get(TEXT_ENCODER, ""))
    if not (encoder / "config.json").is_file() or not any(encoder.glob("*.safetensors")):
        raise SystemExit("pass --requires {0}=<snapshot> with its config.json and weights.".format(TEXT_ENCODER))
    absent = [name for name in TOKENIZER_FILES[:3] if not (encoder / name).is_file()]
    if absent:
        raise SystemExit("{0} is missing the tokenizer files {1}.".format(encoder, ", ".join(absent)))
    try:
        text_config = json.loads((encoder / "config.json").read_text(encoding="utf-8"))["text_config"]
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit("{0}/config.json has no text_config; it is not a Gemma 3 checkpoint.".format(encoder))
    if text_config.get("hidden_size") != 3840 or text_config.get("num_hidden_layers") != 48:
        raise SystemExit("LTX-2.3 reads Gemma 3 12B hidden states (48 layers of 3840); {0} is another model.".format(encoder))
    blocks_path = base / "vae" / "encoder" / "config.json"
    encoder_config = json.loads(blocks_path.read_text(encoding="utf-8"))
    encoder_config["encoder_blocks"] = encoder_blocks(encoder_config["encoder_blocks"], _shapes(base / "vae" / "encoder"))

    print("[INFO] Cloning {0}".format(base), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for component in COMPONENTS:
        for source in sorted((base / component).rglob("*")):
            relative = source.relative_to(base)
            if source.is_file() and relative != Path("vae/encoder/config.json"):
                _clone(source, out / relative)
    for upscaler in upscalers:
        _clone(upscaler, out / upscaler.name)
    (out / "vae" / "encoder" / "config.json").write_text(json.dumps(encoder_config, indent=2) + "\n", encoding="utf-8")
    for source in sorted(encoder.iterdir()):
        if source.is_file() and (source.suffix == ".safetensors" or source.name in ("config.json", "model.safetensors.index.json")):
            _clone(source, out / "text_encoder" / source.name)
        if source.is_file() and source.name in TOKENIZER_FILES:
            _clone(source, out / "tokenizer" / source.name)
    config = dict(CONFIG, spatial_upscaler=upscalers[-1].name)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("[INFO] Conversion complete", flush=True)
    return out


def _source(text):
    repo, separator, path = text.partition("=")
    if not separator or repo.count("/") != 1 or not path:
        raise argparse.ArgumentTypeError("Use --requires <org>/<repo>=<local snapshot>.")
    return repo, path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="MLX LTX-2.3 snapshot directory, or a repo id already in the cache")
    parser.add_argument("--mlx-path", required=True, help="destination directory")
    parser.add_argument("--requires", type=_source, action="append", default=[])
    arguments = parser.parse_args(argv)
    convert(arguments.hf_path, arguments.mlx_path, dict(arguments.requires))
    return 0


if __name__ == "__main__":
    sys.exit(main())
