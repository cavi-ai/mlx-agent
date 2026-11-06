"""Run a local image-to-mesh model through its isolated MLX backend."""

import json
import math
import struct
import subprocess
from pathlib import Path

from .backends import backend_python, is_installed, load_manifests, load_registries, lookup
from .inference_runner import InferenceError, RunnerContract, invoke

MESH_SCHEMA = "mesh/1"
MESH_RUNNER = Path(__file__).with_name("mesh_runner.py")


class MeshError(InferenceError):
    """A mesh failure safe to surface in a result envelope."""


RESOLUTIONS = (32, 64, 128, 256, 512)
DEFAULTS = {"steps": 50, "resolution": 512, "guidance": 7.0}
MAX_FACES_RANGE = (1000, 10_000_000)
RESOLUTION_RANGE = (16, 4096)


def _settings(config):
    """A converted model's own mesh resolutions and defaults (its config ``mesh`` block), else the octree ones."""
    mesh = config.get("mesh") if isinstance(config.get("mesh"), dict) else {}
    resolutions = mesh.get("resolutions", RESOLUTIONS)
    if not isinstance(resolutions, (list, tuple)) or not resolutions or not all(
        type(value) is int and RESOLUTION_RANGE[0] <= value <= RESOLUTION_RANGE[1] for value in resolutions
    ):
        raise MeshError("model_unreadable", "The model config lists invalid mesh resolutions.", "Convert the model again.")
    return tuple(resolutions), {key: mesh.get(key, value) for key, value in DEFAULTS.items()}


def _choices(values):
    values = [str(value) for value in values]
    return values[0] if len(values) == 1 else "{0} or {1}".format(", ".join(values[:-1]), values[-1])


def plan_mesh(model_path, image, out, steps=None, seed=42, resolution=None, guidance=None, max_faces=500_000,
              manifests=None, registries=None, root=None):
    """The backend run for one image; unset steps, resolution and guidance take the model's own defaults."""
    model, image, target = (Path(value).expanduser().absolute() for value in (model_path, image, out))
    try:
        config = json.loads((model / "config.json").read_text())
    except (OSError, ValueError) as error:
        raise MeshError("model_unreadable", "No readable converted model config.", "Pass --path to the converted model directory.") from error
    if not isinstance(config, dict):
        raise MeshError("model_unreadable", "Model config must be an object.", "Select a converted model.")
    resolutions, defaults = _settings(config)
    steps = defaults["steps"] if steps is None else steps
    resolution = defaults["resolution"] if resolution is None else resolution
    guidance = defaults["guidance"] if guidance is None else guidance
    if not image.is_file() or image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise MeshError("invalid_arguments", "--image must be an existing PNG, JPEG or WebP file.", "Select an input image.")
    if target.suffix.lower() != ".glb" or target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise MeshError("invalid_arguments", "--out must be a new .glb path in an existing directory.", "Choose a new output path.")
    if type(steps) is not int or not 1 <= steps <= 100 or type(seed) is not int or not 0 <= seed < 2**32:
        raise MeshError("invalid_arguments", "Steps must be 1..100 and seed 0..4294967295.", "Use --steps 50 --seed 42.")
    if type(resolution) is not int or resolution not in resolutions:
        raise MeshError("invalid_arguments", "Resolution must be {0}.".format(_choices(resolutions)),
                        "Use --resolution {0}.".format(defaults["resolution"]))
    if type(max_faces) is not int or not MAX_FACES_RANGE[0] <= max_faces <= MAX_FACES_RANGE[1]:
        raise MeshError("invalid_arguments", "Max faces must be 1000..10000000.", "Use --max-faces 500000.")
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or not 1 <= guidance <= 20:
        raise MeshError("invalid_arguments", "Guidance must be finite and between 1 and 20.", "Use --guidance 7.")
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = [hit for hit in lookup(config.get("model_type"), manifests, registries) if hit["category"] == "mesh_generation"]
    if not hits:
        raise MeshError("not_mesh_model", "No mesh-generation backend implements this model.", "Convert a supported image-to-3D model first.")
    manifest = manifests[hits[0]["backend"]]
    if not is_installed(manifest, root):
        raise MeshError("backend_not_installed", "The mesh backend is not installed.", "Install it with mlx-agent backend install " + manifest["id"] + ".")
    argv = [str(backend_python(manifest, root)), str(MESH_RUNNER), "--module", hits[0]["module"],
            "--model", str(model), "--image", str(image), "--out", str(target), "--steps", str(steps),
            "--seed", str(seed), "--resolution", str(resolution), "--guidance", str(guidance),
            "--max-faces", str(max_faces)]
    return {"backend": manifest["id"], "model": str(model), "out": str(target), "argv": argv}


CONTRACT = RunnerContract(
    verb="mesh", error=MeshError, accept=lambda value, plan: value.get("path") == plan["out"],
    action="generate the mesh", remediation="Check the model and input image, or use a lower resolution.",
    activity="Mesh generation", timeout_remediation="Use fewer steps or a lower resolution.",
)


def run_mesh(plan, manifests=None, root=None, timeout=3600, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    output = Path(plan["out"])
    try:
        with output.open("rb") as stream:
            header = stream.read(12)
        magic, version, length = struct.unpack("<III", header)
        if output.is_symlink() or magic != 0x46546C67 or version != 2 or length < 28 or length != output.stat().st_size:
            raise ValueError("Invalid GLB header")
    except (OSError, ValueError, struct.error) as error:
        raise MeshError("mesh_failed", "The backend did not write a valid GLB file.", "Retry with a new output path.") from error
    return dict(result, schema=MESH_SCHEMA, backend=plan["backend"], model=plan["model"])
