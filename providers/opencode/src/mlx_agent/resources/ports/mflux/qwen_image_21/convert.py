"""Convert Qwen-Image 2.1 (diffusers layout, already in the cache) to an mflux MLX directory.

``--lora`` merges a PEFT LoRA into the transformer first (``--lora-scale``).
Nothing is downloaded: ``--hf-path`` and ``--lora`` are local paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE = "QwenImage21Pipeline"


def convert(hf_path: str, mlx_path: str, quantize: bool = False, q_bits: int = 8, lora: str = None,
            lora_scale: float = 1.0) -> Path:
    import mlx.core as mx
    from mlx import nn
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    from .qwen_image_21 import MFLUX_MODEL, merge_lora

    base, out = Path(hf_path).expanduser(), Path(mlx_path).expanduser()
    if not base.is_dir():
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(hf_path, local_files_only=True))
    if out.exists() and any(out.iterdir()):
        raise SystemExit("{0} already exists and is not empty.".format(out))
    try:
        index = json.loads((base / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("{0} is not a diffusers checkpoint (no readable model_index.json).".format(base))
    if index.get("_class_name") != PIPELINE:
        raise SystemExit("{0} is a {1}, not a {2}.".format(base, index.get("_class_name"), PIPELINE))
    print("[INFO] Loading {0}".format(base), flush=True)
    model = QwenImage21.__new__(QwenImage21)
    nn.Module.__init__(model)
    Qwen21Initializer._init_config(model, ModelConfig.qwen_image_21())
    weights = Qwen21Initializer._load_weights(str(base))
    if lora:
        count = merge_lora(weights.components["transformer"], mx.load(str(Path(lora).expanduser())), lora_scale)
        print("[INFO] Merged {0} LoRA layers at scale {1}".format(count, lora_scale), flush=True)
    Qwen21Initializer._init_tokenizers(model, str(base))
    Qwen21Initializer._init_models(model)
    bits = q_bits if quantize else None
    if bits:
        print("[INFO] Quantizing to {0} bits".format(bits), flush=True)
    Qwen21Initializer._apply_weights(model, weights, bits)
    print("[INFO] Writing {0}".format(out), flush=True)
    model.save_model(str(out))
    config = {
        "model_type": "qwen_image_21", "mflux_model": MFLUX_MODEL,
        "components": ["transformer", "text_encoder", "vae"],
        "quantization": {"bits": bits} if bits else None,
        "lora": {"file": Path(lora).name, "scale": lora_scale} if lora else None,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("[INFO] Conversion complete", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="diffusers checkpoint directory, or a repo id already in the cache")
    parser.add_argument("--mlx-path", required=True, help="destination directory")
    parser.add_argument("-q", "--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=8, choices=(4, 8))
    parser.add_argument("--lora", default=None, help="local PEFT LoRA .safetensors merged into the transformer")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    convert(arguments.hf_path, arguments.mlx_path, arguments.quantize, arguments.q_bits, arguments.lora, arguments.lora_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
