"""Convert a local TripoSG Diffusers checkpoint to native MLX weights."""

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROVENANCE_NAME = "mlx-converter.json"
UNQUANTIZED_BITS = 32
FILES = {
    "transformer": "diffusion_pytorch_model.safetensors",
    "vae": "diffusion_pytorch_model.safetensors",
    "image_encoder_dinov2": "model.safetensors",
}


def convert(hf_path, mlx_path, quantize=False, q_bits=8):
    base, out = Path(hf_path).expanduser(), Path(mlx_path).expanduser()
    if out.exists() or out.is_symlink():
        raise ValueError("The output directory already exists.")
    try:
        index = json.loads((base / "model_index.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Pass a local TripoSGPipeline checkpoint with model_index.json.") from error
    if index.get("_class_name") != "TripoSGPipeline":
        raise ValueError("Only TripoSGPipeline image-to-3D checkpoints are supported.")
    required = [component + "/" + name for component, name in FILES.items()]
    required += [component + "/config.json" for component in FILES]
    required += ["feature_extractor_dinov2/preprocessor_config.json", "scheduler/scheduler_config.json"]
    missing = [name for name in required if not (base / name).is_file()]
    if missing:
        raise ValueError("The checkpoint is missing: " + ", ".join(missing))
    configs = {name: json.loads((base / name / "config.json").read_text()) for name in FILES}
    scheduler = json.loads((base / "scheduler/scheduler_config.json").read_text())
    if configs["transformer"].get("use_cross_attention_2") or configs["image_encoder_dinov2"].get("use_swiglu_ffn"):
        raise ValueError("This TripoSG variant uses an unsupported transformer or image encoder.")
    if scheduler.get("_class_name") not in ("RectifiedFlowScheduler", "FlowMatchEulerDiscreteScheduler") or scheduler.get("use_dynamic_shifting") or scheduler.get("shift", 1) != 1:
        raise ValueError("Only the TripoSG unshifted rectified-flow scheduler is supported.")
    if configs["vae"].get("embedding_type", "frequency") != "frequency":
        raise ValueError("Only the frequency positional VAE is supported.")
    if quantize and q_bits != 8:
        raise ValueError("TripoSG conversion supports 8-bit quantization only.")
    import mlx.core as mx

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".triposg-", dir=out.parent))
    try:
        for component, filename in FILES.items():
            values = mx.load(str(base / component / filename))
            converted = {}
            for name, value in values.items():
                # Preserve all tensors, including the VAE encoder. Inference uses
                # only its decoder; no source Python code is imported or copied.
                value = value.astype(mx.float32)
                if quantize and name.endswith(".weight") and value.ndim == 2 and value.shape[-1] % 64 == 0:
                    packed, scales, biases = mx.quantize(value, group_size=64, bits=q_bits)
                    mx.eval(packed, scales, biases)
                    converted.update({name: packed, name[:-6] + "scales": scales, name[:-6] + "biases": biases})
                else:
                    converted[name] = value
                    mx.eval(value)
            mx.save_safetensors(str(staging / (component + ".safetensors")), converted)
            del values, converted
        config = {"model_type": "triposg", "components": configs, "scheduler": scheduler,
                  "preprocessor": json.loads((base / required[-2]).read_text())}
        if quantize:
            config["quantization"] = {"bits": q_bits, "group_size": 64}
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        # The marker lets scans list unquantized outputs, whose config carries no quantization block.
        provenance = {"schema_version": "1.0", "tool": "mlx-agent.triposg",
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "q_bits": q_bits if quantize else UNQUANTIZED_BITS,
                      "source": {"kind": "diffusers", "path": str(base), "pipeline": "TripoSGPipeline"}}
        (staging / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        # mkdir is the no-replace claim; rename alone can replace an empty folder.
        out.mkdir()
        # Publish the config last, so scans cannot treat a partial copy as ready.
        for child in sorted(staging.iterdir(), key=lambda item: item.name == "config.json"):
            child.rename(out / child.name)
    finally:
        shutil.rmtree(staging)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--q-bits", type=int, default=8, choices=(8,))
    args = parser.parse_args(argv)
    convert(args.hf_path, args.mlx_path, args.quantize, args.q_bits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
