"""Convert a local TRELLIS.2 image-to-3D checkpoint, with its image encoder and structure decoder, to MLX weights."""

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROVENANCE_NAME = "mlx-converter.json"
UNQUANTIZED_BITS = 32
PIPELINE_NAME = "Trellis2ImageTo3DPipeline"
MODELS = {
    "ss_flow": ("sparse_structure_flow_model", "SparseStructureFlowModel"),
    "ss_decoder": ("sparse_structure_decoder", "SparseStructureDecoder"),
    "shape_flow_512": ("shape_slat_flow_model_512", "SLatFlowModel"),
    "shape_flow_1024": ("shape_slat_flow_model_1024", "SLatFlowModel"),
    "shape_decoder": ("shape_slat_decoder", "FlexiDualGridVaeDecoder"),
}
QUANTIZED = ("image_encoder", "ss_flow", "shape_flow_512", "shape_flow_1024")
FLOW_FLAGS = {"pe_mode": "rope", "share_mod": True, "qk_rms_norm": True, "qk_rms_norm_cross": True}
SAMPLERS = ("sparse_structure_sampler", "shape_slat_sampler")
ENCODER_TYPE = "dinov3_vit"
RESOLUTIONS = (512, 1024, 1536)
DEFAULT_RESOLUTION = 1024


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Unreadable checkpoint file: {0}".format(path)) from error


def _locate(base, sources, reference):
    """A checkpoint reference is ``ckpts/<name>`` in this repository or ``<org>/<repo>/<path>`` in a source."""
    parts = reference.split("/")
    if parts[0] == "ckpts":
        return base / reference
    repo = "/".join(parts[:2])
    if repo not in sources:
        raise ValueError("Pass --requires {0}=<local snapshot> for {1}.".format(repo, reference))
    return Path(sources[repo]) / "/".join(parts[2:])


def _check_flow(name, args):
    wrong = [key for key, value in FLOW_FLAGS.items() if args.get(key) != value]
    if args.get("image_attn_mode", "cross") != "cross":
        wrong.append("image_attn_mode")
    if wrong:
        raise ValueError("{0} uses an unsupported variant: {1}.".format(name, ", ".join(wrong)))


def _plan(base, sources):
    pipeline = _read(base / "pipeline.json")
    if pipeline.get("name") != PIPELINE_NAME:
        raise ValueError("Only {0} checkpoints are supported.".format(PIPELINE_NAME))
    args = pipeline.get("args") or {}
    models = args.get("models") or {}
    files, configs = {}, {}
    for component, (key, kind) in MODELS.items():
        if key not in models:
            raise ValueError("pipeline.json names no {0}.".format(key))
        stem = _locate(base, sources, models[key])
        described = _read(str(stem) + ".json")
        if described.get("name") != kind:
            raise ValueError("{0} must be a {1}.".format(key, kind))
        configs[component] = described.get("args") or {}
        files[component] = Path(str(stem) + ".safetensors")
        if kind.endswith("FlowModel"):
            _check_flow(key, configs[component])
    encoder = (args.get("image_cond_model") or {})
    if encoder.get("name") != "DinoV3FeatureExtractor":
        raise ValueError("Only the DINOv3 image condition model is supported.")
    repo = (encoder.get("args") or {}).get("model_name", "")
    if repo not in sources:
        raise ValueError("Pass --requires {0}=<local snapshot> for the image encoder.".format(repo))
    configs["image_encoder"] = _read(Path(sources[repo]) / "config.json")
    if configs["image_encoder"].get("model_type") != ENCODER_TYPE:
        raise ValueError("The image encoder must be a DINOv3 ViT checkpoint.")
    files["image_encoder"] = Path(sources[repo]) / "model.safetensors"
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError("The checkpoint is missing: " + ", ".join(missing))
    settings = {name: args[name]["params"] for name in SAMPLERS if name in args}
    if len(settings) != len(SAMPLERS) or "shape_slat_normalization" not in args:
        raise ValueError("pipeline.json lacks the sampler or normalization settings.")
    settings["shape_slat_normalization"] = args["shape_slat_normalization"]
    return files, configs, settings


def _convert_tensors(path, quantize, q_bits):
    import mlx.core as mx

    converted = {}
    for name, value in mx.load(str(path)).items():
        value = value.astype(mx.float32)
        if quantize and name.endswith(".weight") and value.ndim == 2 and value.shape[-1] % 64 == 0:
            packed, scales, biases = mx.quantize(value, group_size=64, bits=q_bits)
            mx.eval(packed, scales, biases)
            converted.update({name: packed, name[:-6] + "scales": scales, name[:-6] + "biases": biases})
        else:
            mx.eval(value)
            converted[name] = value
    return converted


def convert(hf_path, mlx_path, sources=None, quantize=False, q_bits=8):
    base, out = Path(hf_path).expanduser(), Path(mlx_path).expanduser()
    sources = {repo: Path(path).expanduser() for repo, path in (sources or {}).items()}
    if out.exists() or out.is_symlink():
        raise ValueError("The output directory already exists.")
    if quantize and q_bits != 8:
        raise ValueError("TRELLIS.2 conversion supports 8-bit quantization only.")
    files, configs, settings = _plan(base, sources)
    import mlx.core as mx

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trellis2-", dir=out.parent))
    try:
        for component, path in files.items():
            converted = _convert_tensors(path, quantize and component in QUANTIZED, q_bits)
            mx.save_safetensors(str(staging / (component + ".safetensors")), converted)
            del converted
        sampler = settings["shape_slat_sampler"]
        mesh = {"resolutions": list(RESOLUTIONS), "resolution": DEFAULT_RESOLUTION, "steps": sampler["steps"],
                "guidance": sampler["guidance_strength"]}
        config = {"model_type": "trellis2", "components": configs, "pipeline": settings, "mesh": mesh}
        if quantize:
            config["quantization"] = {"bits": q_bits, "group_size": 64}
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        provenance = {"schema_version": "1.0", "tool": "mlx-agent.trellis2",
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "q_bits": q_bits if quantize else UNQUANTIZED_BITS,
                      "source": {"kind": "trellis2", "path": str(base), "pipeline": PIPELINE_NAME,
                                 "requires": {repo: str(path) for repo, path in sorted(sources.items())}}}
        (staging / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        out.mkdir()
        for child in sorted(staging.iterdir(), key=lambda item: item.name == "config.json"):
            child.rename(out / child.name)
    finally:
        shutil.rmtree(staging)
    return out


def _source(text):
    repo, separator, path = text.partition("=")
    if not separator or repo.count("/") != 1 or not path:
        raise argparse.ArgumentTypeError("Use --requires <org>/<repo>=<local snapshot>.")
    return repo, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--requires", type=_source, action="append", default=[])
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--q-bits", type=int, default=8, choices=(8,))
    args = parser.parse_args(argv)
    convert(args.hf_path, args.mlx_path, dict(args.requires), args.quantize, args.q_bits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
