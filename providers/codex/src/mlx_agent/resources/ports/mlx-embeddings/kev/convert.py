"""Convert a Kev checkpoint (LoRA adapter and pointer head over a Qwen3 base) to MLX.

The LoRA is folded into the base weights in float32, then the decoder is cast to bfloat16
and optionally quantized. The pointer head stays float32.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .kev import TOKENIZER_FILES, Model, ModelArgs

CHECKPOINT_FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt")
BASE_FILES = ("config.json",) + TOKENIZER_FILES
QUANTIZE_INCLUDE = ("backbone.embed_tokens", "backbone.layers.")
GROUP_SIZE = 64
LORA_PREFIX = "base_model.model."
STORAGES = {"FloatStorage": np.float32, "HalfStorage": np.float16}


class HeadUnpickler(pickle.Unpickler):
    """Reads a torch zip checkpoint of plain tensors and settings; any other global is refused."""

    def __init__(self, data: bytes, archive: zipfile.ZipFile, prefix: str):
        super().__init__(io.BytesIO(data))
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return self.tensor
        if module == "torch" and name in STORAGES:
            return name
        raise pickle.UnpicklingError("head.pt references {0}.{1}".format(module, name))

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage" and pid[1] in STORAGES):
            raise pickle.UnpicklingError("head.pt holds an unsupported storage")
        raw = self.archive.read("{0}data/{1}".format(self.prefix, pid[2]))
        return np.frombuffer(raw, dtype=STORAGES[pid[1]])

    @staticmethod
    def tensor(storage, offset, size, stride, *_):
        count = math.prod(size)
        contiguous = tuple(math.prod(size[index + 1:]) for index in range(len(size)))
        if tuple(stride) != contiguous or offset + count > storage.size:
            raise pickle.UnpicklingError("head.pt holds a non-contiguous tensor")
        return storage[offset:offset + count].reshape(size)


def read_head(path: Path) -> dict:
    """The head checkpoint's settings and weights (numpy arrays) without importing torch."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            pickles = [name for name in names if name.endswith("/data.pkl")]
            if len(pickles) != 1:
                raise SystemExit("{0} is not a torch zip checkpoint.".format(path))
            prefix = pickles[0][: -len("data.pkl")]
            if prefix + "byteorder" in names and archive.read(prefix + "byteorder").strip() != b"little":
                raise SystemExit("{0} is not little-endian.".format(path))
            meta = HeadUnpickler(archive.read(pickles[0]), archive, prefix).load()
    except (zipfile.BadZipFile, pickle.UnpicklingError, KeyError, ValueError) as error:
        raise SystemExit("{0} is not a readable Kev head: {1}".format(path, error)) from error
    if not isinstance(meta, dict) or not isinstance(meta.get("head"), dict):
        raise SystemExit("{0} has no head weights.".format(path))
    return meta


def lora_scaling(settings: dict, scale: float) -> float:
    if (settings.get("peft_type") != "LORA" or settings.get("use_dora") or settings.get("trainable_token_indices")
            or settings.get("modules_to_save") or settings.get("rank_pattern") or settings.get("alpha_pattern")):
        raise SystemExit("Only a plain LoRA adapter with one rank folds into the base weights.")
    rank, alpha = settings["r"], settings["lora_alpha"]
    return scale * (alpha / math.sqrt(rank) if settings.get("use_rslora") else alpha / rank)


def merged_weights(base: Path, adapter: Path, scaling: float) -> dict:
    """The base decoder under backbone names with the LoRA folded in: a float32 sum, stored as bfloat16."""
    weights = {}
    for shard in sorted(base.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    lora = mx.load(str(adapter))
    suffix_a, suffix_b = ".lora_A.weight", ".lora_B.weight"
    targets = sorted(name[len(LORA_PREFIX):-len(suffix_a)] for name in lora if name.endswith(suffix_a))
    if not targets or set(lora) != {LORA_PREFIX + target + suffix for target in targets for suffix in (suffix_a, suffix_b)}:
        raise SystemExit("{0} holds tensors other than LoRA A/B pairs for the decoder.".format(adapter))
    for target in targets:
        name = "model.{0}.weight".format(target)
        if name not in weights:
            raise SystemExit("The adapter targets {0}, which the base does not have.".format(name))
        delta = lora[LORA_PREFIX + target + suffix_b].astype(mx.float32) @ lora[LORA_PREFIX + target + suffix_a].astype(mx.float32)
        weights[name] = (weights[name].astype(mx.float32) + scaling * delta).astype(mx.bfloat16)
        mx.eval(weights[name])
    result = {}
    for name, value in weights.items():
        if name.startswith("lm_head."):
            continue
        if not name.startswith("model."):
            raise SystemExit("The base holds {0}, outside the decoder.".format(name))
        result["backbone." + name[len("model."):]] = value.astype(mx.bfloat16)
    return result


def quantizes(path: str, module: nn.Module) -> bool:
    """Decoder linear layers and the token embedding; norms and the pointer head stay as converted."""
    return (
        isinstance(module, (nn.Linear, nn.Embedding))
        and path.startswith(QUANTIZE_INCLUDE)
        and module.weight.shape[-1] % GROUP_SIZE == 0
    )


def convert(hf_path: str, mlx_path: str, quantize: bool = False, q_bits: int = 8, lora: str = None,
            lora_scale: float = 1.0) -> Path:
    out = Path(mlx_path).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("{0} already exists and is not empty.".format(out))
    if lora is None:
        raise SystemExit("Pass the checkpoint's adapter with --lora; --hf-path is its Qwen3 base snapshot.")
    base, adapter = Path(hf_path).expanduser(), Path(lora).expanduser()
    checkpoint = adapter.parent
    missing = [str(checkpoint / name) for name in CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    missing += [str(base / name) for name in BASE_FILES if not (base / name).is_file()]
    if missing:
        raise SystemExit("Not a Kev checkpoint and Qwen3 base; missing {0}.".format(", ".join(missing)))
    print("[INFO] Loading {0} over {1}".format(checkpoint, base), flush=True)
    meta = read_head(checkpoint / "head.pt")
    if meta.get("option_isolation") or meta.get("special_embeddings"):
        raise SystemExit("Option isolation and trained delimiter embeddings are not ported.")
    backbone = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if backbone.get("model_type") != "qwen3":
        raise SystemExit("The base is {0!r}; the Kev port runs a qwen3 decoder.".format(backbone.get("model_type")))
    config = {"model_type": "kev", "backbone": backbone, "head_dim": int(meta.get("head_dim", 256))}
    config.update({key: meta[key] for key in ("base", "base_revision") if isinstance(meta.get(key), str)})
    settings = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    model = Model(ModelArgs.from_dict(config))
    weights = merged_weights(base, adapter, lora_scaling(settings, lora_scale))
    weights.update({"head." + name: mx.array(value) for name, value in meta["head"].items()})
    model.load_weights(list(weights.items()), strict=True)
    del weights
    if quantize:
        print("[INFO] Quantizing to {0} bits".format(q_bits), flush=True)
        nn.quantize(model, group_size=GROUP_SIZE, bits=q_bits, class_predicate=quantizes)
        config["quantization"] = {"group_size": GROUP_SIZE, "bits": q_bits}
    out.mkdir(parents=True, exist_ok=True)
    print("[INFO] Writing {0}".format(out), flush=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})
    for name in TOKENIZER_FILES:
        shutil.copyfile(base / name, out / name)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("[INFO] Conversion complete", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="the Qwen3 base snapshot directory")
    parser.add_argument("--mlx-path", required=True, help="destination directory")
    parser.add_argument("--lora", required=True, help="the checkpoint's adapter_model.safetensors (head.pt beside it)")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("-q", "--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=8, choices=(4, 5, 6, 8))
    arguments = parser.parse_args(argv)
    convert(arguments.hf_path, arguments.mlx_path, arguments.quantize, arguments.q_bits, arguments.lora, arguments.lora_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
