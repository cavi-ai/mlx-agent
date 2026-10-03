"""Convert a Laya checkpoint (repository root: encoder config, decision settings, tokenizer) to MLX."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .laya import TOKENIZER_FILES, Model, ModelArgs

SOURCE_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
SETTINGS = ("head_layers", "max_len", "head_max_len", "act_costs", "temperature", "temperature_by_options", "model_name")
QUANTIZE_INCLUDE = ("encoder.layers.", "head.layers.")
GROUP_SIZE = 32


def source_directory(hf_path: str, revision: str = None) -> Path:
    path = Path(hf_path).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_path, revision=revision, allow_patterns=list(SOURCE_FILES)))


def converted_config(settings: dict, encoder: dict) -> dict:
    config = {"model_type": "laya", "encoder": encoder}
    config.update({key: settings[key] for key in SETTINGS if key in settings})
    return config


def quantizes(path: str, module: nn.Module) -> bool:
    """Linear layers of the encoder and decision head; embeddings, norms, and the scorers stay as shipped."""
    return (
        isinstance(module, nn.Linear)
        and path.startswith(QUANTIZE_INCLUDE)
        and module.weight.shape[-1] % GROUP_SIZE == 0
    )


def convert(hf_path: str, mlx_path: str, quantize: bool = False, q_bits: int = 4, revision: str = None) -> Path:
    out = Path(mlx_path).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("{0} already exists and is not empty.".format(out))
    print("[INFO] Loading {0}".format(hf_path), flush=True)
    source = source_directory(hf_path, revision)
    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit("{0} is not a Laya checkpoint; missing {1}.".format(hf_path, ", ".join(missing)))
    settings = json.loads((source / "rl_agent_config.json").read_text(encoding="utf-8"))
    encoder = json.loads((source / "encoder" / "config.json").read_text(encoding="utf-8"))
    config = converted_config(settings, encoder)
    model = Model(ModelArgs.from_dict(config))
    model.load_weights(list(Model.sanitize(mx.load(str(source / "model.safetensors"))).items()), strict=True)
    if quantize:
        print("[INFO] Quantizing to {0} bits".format(q_bits), flush=True)
        nn.quantize(model, group_size=GROUP_SIZE, bits=q_bits, class_predicate=quantizes)
        config["quantization"] = {"group_size": GROUP_SIZE, "bits": q_bits}
    out.mkdir(parents=True, exist_ok=True)
    print("[INFO] Writing {0}".format(out), flush=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})
    for name in TOKENIZER_FILES:
        shutil.copyfile(source / "tokenizer" / name, out / name)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("[INFO] Conversion complete", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Hugging Face repo id or local checkpoint directory")
    parser.add_argument("--mlx-path", required=True, help="destination directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("-q", "--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=4, choices=(4, 8))
    arguments = parser.parse_args(argv)
    convert(arguments.hf_path, arguments.mlx_path, arguments.quantize, arguments.q_bits, arguments.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
