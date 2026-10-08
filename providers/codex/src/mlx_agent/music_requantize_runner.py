"""Requantize an exported local MLX music checkpoint without fetching weights.

The upstream raw MiniMax converter expects component checkpoints. Exported
MLX weights instead go through the music loader and its quantization policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def copy_supporting_files(source, destination):
    """Keep tokenizer/scheduler assets; never duplicate checkpoint weights/code."""
    extensions = {".json", ".txt", ".jinja", ".model", ".tiktoken", ".yaml", ".yml"}
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if len(relative.parts) > 1 and relative.parts[0] not in {"tokenizer", "scheduler"}:
            continue
        if (item.suffix not in extensions or item.name in {"config.json", "model.safetensors.index.json"}
                and len(relative.parts) == 1):
            continue
        if item.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, type=Path)
    parser.add_argument("--mlx-path", required=True, type=Path)
    parser.add_argument("--quantize", required=True, action="store_true")
    parser.add_argument("--q-bits", required=True, type=int, choices=(4, 8))
    args = parser.parse_args(argv)
    source = args.hf_path.expanduser().resolve(strict=True)
    destination = args.mlx_path.expanduser().absolute()
    if destination.exists() or destination.is_symlink() or destination.resolve().is_relative_to(source):
        raise ValueError("Choose a new output directory outside the source checkpoint.")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "minimax_music3" or not isinstance(config.get("quantization"), dict):
        raise ValueError("Expected a quantized local MiniMax Music 3 MLX checkpoint.")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_audio.music import load
    from mlx_audio.convert import build_quant_predicate
    from mlx_audio.lm.convert import dequantize_model, quantize_model, save_config, save_model

    model = dequantize_model(load(str(source)))
    mx.eval(model.parameters())
    mx.clear_cache()
    # Remove the source policy after dequantization. Keeping it makes the
    # backend record per-layer overrides under the old global bit width.
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    model, config = quantize_model(model, config, 64, args.q_bits,
                                   quant_predicate=build_quant_predicate(model))
    mx.eval(model.parameters())
    destination.mkdir(parents=True, exist_ok=False)
    save_model(destination, model, donate_model=True)
    copy_supporting_files(source, destination)
    save_config(config, destination / "config.json")


if __name__ == "__main__":
    main()
