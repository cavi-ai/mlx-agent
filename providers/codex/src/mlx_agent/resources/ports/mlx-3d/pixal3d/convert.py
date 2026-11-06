"""Convert a local Pixal3D image-to-3D checkpoint, with its image encoder, camera estimator and feature upsampler,
to MLX weights."""

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..trellis2.convert import (
    ENCODER_TYPE, FLOW_FLAGS, PIPELINE_NAME, PROVENANCE_NAME, SAMPLERS, UNQUANTIZED_BITS, _convert_tensors, _read,
    _source,
)

MODELS = {
    "ss_flow": ("sparse_structure_flow_model", "SparseStructureFlowModel", 1),
    "ss_decoder": ("sparse_structure_decoder", "SparseStructureDecoder", None),
    "shape_flow_512": ("shape_slat_flow_model_512", "ElasticSLatFlowModel", 2),
    "shape_flow_1024": ("shape_slat_flow_model_1024", "ElasticSLatFlowModel", 2),
    "shape_decoder": ("shape_slat_decoder", "FlexiDualGridVaeDecoder", None),
}
QUANTIZED = ("image_encoder", "ss_flow", "shape_flow_512", "shape_flow_1024")
MOGE_REPO = "Ruicheng/moge-2-vitl"
MOGE_FILE = "model.pt"
MOGE_PARTS = ("encoder.backbone.", "encoder.output_projections.", "neck.", "points_head.", "mask_head.")
NAF_FILE = "naf_release.pth"
NAF_KEYS = ("image_encoder.encoder.0.weight", "image_encoder.sem_encoder.0.weight", "image_encoder.rope.periods")
RESOLUTIONS = (1024, 1536)
PIPELINE_TYPES = {"1024_cascade": 1024, "1536_cascade": 1536}


def _check_flow(name, args, encoder_channels, proj_factor):
    wrong = [key for key, value in FLOW_FLAGS.items() if args.get(key) != value]
    if args.get("image_attn_mode") != "proj":
        wrong.append("image_attn_mode")
    if (args.get("proj_in_channels") or encoder_channels) != proj_factor * encoder_channels:
        wrong.append("proj_in_channels")
    if wrong:
        raise ValueError("{0} uses an unsupported variant: {1}.".format(name, ", ".join(wrong)))


def _check_moge(config):
    encoder = config.get("encoder") or {}
    heads = [config.get(name) or {} for name in ("neck", "points_head", "mask_head")]
    supported = (
        encoder.get("backbone") == "dinov2_vitl14" and config.get("remap_output") == "exp"
        and all(head.get("res_block_in_norm") == "none" and head.get("res_block_hidden_norm") == "none"
                and set(head.get("resamplers") or ()) <= {"conv_transpose", "bilinear"} for head in heads)
        and all(heads[1:])
    )
    if not supported:
        raise ValueError("Only the MoGe-2 ViT-L camera model is supported.")


def _plan(base, sources, downloads):
    pipeline = _read(base / "pipeline.json")
    if pipeline.get("name") != PIPELINE_NAME:
        raise ValueError("Only Pixal3D checkpoints with a {0} pipeline are supported.".format(PIPELINE_NAME))
    args = pipeline.get("args") or {}
    models = args.get("models") or {}
    encoder = args.get("image_cond_model") or {}
    if encoder.get("name") != "DinoV3FeatureExtractor":
        raise ValueError("Only the DINOv3 image condition model is supported.")
    repo = (encoder.get("args") or {}).get("model_name", "")
    for needed in (repo, MOGE_REPO):
        if needed not in sources:
            raise ValueError("Pass --requires {0}=<local snapshot>.".format(needed))
    if NAF_FILE not in downloads:
        raise ValueError("Pass --download {0}=<local file>.".format(NAF_FILE))
    files, configs = {}, {"image_encoder": _read(Path(sources[repo]) / "config.json")}
    if configs["image_encoder"].get("model_type") != ENCODER_TYPE:
        raise ValueError("The image encoder must be a DINOv3 ViT checkpoint.")
    files["image_encoder"] = Path(sources[repo]) / "model.safetensors"
    channels = configs["image_encoder"].get("hidden_size")
    for component, (key, kind, proj_factor) in MODELS.items():
        reference = models.get(key, "")
        if not reference.startswith("ckpts/"):
            raise ValueError("pipeline.json names no local {0}.".format(key))
        described = _read(base / (reference + ".json"))
        if described.get("name") != kind:
            raise ValueError("{0} must be a {1}.".format(key, kind))
        configs[component] = described.get("args") or {}
        files[component] = base / (reference + ".safetensors")
        if proj_factor:
            _check_flow(key, configs[component], channels, proj_factor)
    files["moge"] = Path(sources[MOGE_REPO]) / MOGE_FILE
    files["naf"] = Path(downloads[NAF_FILE])
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError("The checkpoint is missing: " + ", ".join(missing))
    settings = {name: args[name]["params"] for name in SAMPLERS if name in args}
    if len(settings) != len(SAMPLERS) or "shape_slat_normalization" not in args:
        raise ValueError("pipeline.json lacks the sampler or normalization settings.")
    settings["shape_slat_normalization"] = args["shape_slat_normalization"]
    default = PIPELINE_TYPES.get(args.get("default_pipeline_type", "1536_cascade"))
    if default is None:
        raise ValueError("pipeline.json names an unsupported default pipeline type.")
    return files, configs, settings, default


def _torch_tensors(path, keep):
    import mlx.core as mx
    from . import checkpoint

    state = checkpoint.load(path)
    converted = {name: mx.array(value.astype("float32")) for name, value in state.items() if keep(name)}
    mx.eval(list(converted.values()))
    return converted


def _moge(path):
    import mlx.core as mx
    from . import checkpoint

    loaded = checkpoint.load(path)
    if not isinstance(loaded, dict) or "model_config" not in loaded or "model" not in loaded:
        raise ValueError("The MoGe checkpoint holds no model and configuration.")
    config = loaded["model_config"]
    _check_moge(config)
    converted = {name: mx.array(value.astype("float32")) for name, value in loaded["model"].items()
                 if name.startswith(MOGE_PARTS)}
    mx.eval(list(converted.values()))
    return converted, config


def convert(hf_path, mlx_path, sources=None, downloads=None, quantize=False, q_bits=8):
    base, out = Path(hf_path).expanduser(), Path(mlx_path).expanduser()
    sources = {repo: Path(path).expanduser() for repo, path in (sources or {}).items()}
    downloads = {name: Path(path).expanduser() for name, path in (downloads or {}).items()}
    if out.exists() or out.is_symlink():
        raise ValueError("The output directory already exists.")
    if quantize and q_bits != 8:
        raise ValueError("Pixal3D conversion supports 8-bit quantization only.")
    files, configs, settings, default = _plan(base, sources, downloads)
    import mlx.core as mx

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pixal3d-", dir=out.parent))
    try:
        for component, path in files.items():
            if component == "moge":
                converted, configs["moge"] = _moge(path)
            elif component == "naf":
                converted = _torch_tensors(path, lambda name: name.startswith("image_encoder."))
                if not all(name in converted for name in NAF_KEYS):
                    raise ValueError("The NAF checkpoint lacks its image encoder.")
            else:
                converted = _convert_tensors(path, quantize and component in QUANTIZED, q_bits)
            mx.save_safetensors(str(staging / (component + ".safetensors")), converted)
            del converted
        sampler = settings["shape_slat_sampler"]
        mesh = {"resolutions": list(RESOLUTIONS), "resolution": default, "steps": sampler["steps"],
                "guidance": sampler["guidance_strength"]}
        config = {"model_type": "pixal3d", "components": configs, "pipeline": settings, "mesh": mesh}
        if quantize:
            config["quantization"] = {"bits": q_bits, "group_size": 64}
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        provenance = {"schema_version": "1.0", "tool": "mlx-agent.pixal3d",
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "q_bits": q_bits if quantize else UNQUANTIZED_BITS,
                      "source": {"kind": "pixal3d", "path": str(base), "pipeline": PIPELINE_NAME,
                                 "requires": {repo: str(path) for repo, path in sorted(sources.items())},
                                 "downloads": {name: str(path) for name, path in sorted(downloads.items())}}}
        (staging / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        out.mkdir()
        for child in sorted(staging.iterdir(), key=lambda item: item.name == "config.json"):
            child.rename(out / child.name)
    finally:
        shutil.rmtree(staging)
    return out


def _download(text):
    name, separator, path = text.partition("=")
    if not separator or not name or "/" in name or not path:
        raise argparse.ArgumentTypeError("Use --download <file name>=<local file>.")
    return name, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--requires", type=_source, action="append", default=[])
    parser.add_argument("--download", type=_download, action="append", default=[])
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--q-bits", type=int, default=8, choices=(8,))
    args = parser.parse_args(argv)
    convert(args.hf_path, args.mlx_path, dict(args.requires), dict(args.download), args.quantize, args.q_bits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
