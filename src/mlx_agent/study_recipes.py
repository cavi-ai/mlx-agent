"""Offline recipes for the registered Wan Diffusers study adapter."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import stat

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REPOSITORY = "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
ADAPTER_ID = "diffusers.wan2.1-t2v-1.3b.mps.v1"
_REVISION = re.compile(r"[0-9a-f]{40}")
_COMPONENTS = {
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}
_PACKAGES = ("torch", "diffusers", "transformers", "safetensors", "huggingface-hub",
             "accelerate", "tokenizers", "sentencepiece")


def cache_root():
    # Use exactly the cache resolver used by the generation adapter.
    from video_lab.adapters.wan_mps import WanMpsAdapter
    return WanMpsAdapter()._effective_cache_root()


def runtime_versions():
    versions = {"python": platform.python_version()}
    missing = []
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise ValueError("Install video runtime in Studies → Runtime setup. Missing: " + ", ".join(missing))
    from video_lab.runtime_identity import source_tree_digest
    versions["video-lab"] = source_tree_digest()
    versions["study-recipes"] = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return versions


def catalog(root=None):
    root = Path(root) if root is not None else cache_root()
    snapshots = root / REPOSITORY / "snapshots"
    revisions = sorted(path.name for path in snapshots.glob("*")
                       if _REVISION.fullmatch(path.name) and path.is_dir() and not path.is_symlink())
    reason = None
    try:
        runtime_versions()
    except ValueError as error:
        reason = str(error)
    if not revisions:
        reason = "No cached revision of " + MODEL_ID + ". Add its Diffusers weights before capturing a recipe."
    return [{"model_id": MODEL_ID, "adapter_id": ADAPTER_ID,
             "display_name": "Wan 2.1 T2V 1.3B (MPS)", "revisions": revisions,
             "unavailable_reason": reason}]


def validate_settings(value):
    fields = {"revision", "prompt", "negative_prompt", "seed", "width", "height", "num_frames",
              "fps", "inference_steps", "guidance_scale"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Recipe settings must contain exactly the supported generation fields.")
    if not isinstance(value["revision"], str) or not _REVISION.fullmatch(value["revision"]):
        raise ValueError("Choose a pinned cached model revision.")
    for name in ("prompt", "negative_prompt"):
        if not isinstance(value[name], str) or len(value[name]) > 8000 or (name == "prompt" and not value[name].strip()):
            raise ValueError("Use a nonempty prompt of at most 8,000 characters and an optional negative prompt.")
    for name, low, high, multiple in (("seed", 0, 2**63 - 1, 1), ("width", 16, 4096, 16),
                                    ("height", 16, 4096, 16), ("fps", 1, 120, 1),
                                    ("inference_steps", 1, 1000, 1), ("num_frames", 17, 129, 1)):
        item = value[name]
        if type(item) is not int or not low <= item <= high or item % multiple:
            raise ValueError("Invalid " + name + ": expected " + str(low) + ".." + str(high) + " in multiples of " + str(multiple) + ".")
    if (value["num_frames"] - 1) % 16:
        raise ValueError("Frame count must equal 16n+1.")
    guidance = value["guidance_scale"]
    if type(guidance) not in (int, float) or not 0 <= guidance <= 100:
        raise ValueError("Guidance must be between 0 and 100.")
    return dict(value)


def _read(path, repository, limit=None):
    target = path.resolve(strict=True)
    if not target.is_relative_to(repository):
        raise ValueError("A recipe source resolves outside its model cache.")
    with target.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or (limit is not None and before.st_size > limit):
            raise ValueError("Recipe sources must be nonempty regular files within the supported size.")
        digest = hashlib.sha256()
        content = bytearray() if limit is not None else None
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if content is not None:
                content.extend(chunk)
                if len(content) > limit:
                    raise ValueError("Recipe source exceeded the supported size during capture.")
        after = os.fstat(stream.fileno())
        current = path.stat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(current) or path.resolve(strict=True) != target:
            raise ValueError("Recipe source changed during capture.")
    return digest.hexdigest(), before.st_size, bytes(content) if content is not None else None


def _signature(path):
    observed = path.stat()
    return path.resolve(strict=True), observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns


def _snapshot(revision, root):
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError("A pinned cached model revision is required.")
    repository = (Path(root) / REPOSITORY).resolve()
    folder = repository / "snapshots" / revision
    if not folder.is_dir() or folder.is_symlink() or not folder.resolve().is_relative_to(repository):
        raise ValueError("The selected cached revision is unavailable.")
    paths = [folder / "model_index.json"]
    directory_signatures = {folder: _signature(folder)}
    for component in _COMPONENTS:
        directory = folder / component
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Cached model is incomplete: missing " + component + ".")
        directory_signatures[directory] = _signature(directory)
        entries = list(directory.iterdir())
        if any(path.is_dir() for path in entries):
            raise ValueError("Unexpected nested model component directory.")
        paths.extend(entries)
    checksums, sizes, documents, signatures = {}, {}, {}, {}
    for path in sorted(paths):
        name = path.relative_to(folder).as_posix()
        signatures[path] = _signature(path)
        digest, size, content = _read(path, repository, 32 * 1024 * 1024 if path.suffix == ".json" else None)
        sizes[name] = size
        if path.suffix == ".json":
            try:
                documents[name] = json.loads(content)
            except (ValueError, UnicodeError) as error:
                raise ValueError("Invalid cached JSON: " + name) from error
        checksums[name] = digest
    index = documents.get("model_index.json", {})
    if not isinstance(index, dict) or index.get("_class_name") != "WanPipeline" or any(index.get(name) != value for name, value in _COMPONENTS.items()):
        raise ValueError("Cached model does not declare the supported Wan pipeline.")
    for name in ("transformer/config.json", "vae/config.json", "text_encoder/config.json",
                 "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json", "scheduler/scheduler_config.json"):
        if not isinstance(documents.get(name), dict):
            raise ValueError("Cached model is incomplete: missing " + name + ".")
    for component, basename in (("transformer", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model"), ("text_encoder", "model")):
        index_name = component + "/" + basename + ".safetensors.index.json"
        if index_name in documents:
            document = documents[index_name]
            weight_map = document.get("weight_map") if isinstance(document, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("Cached weight index is incomplete.")
            for name in weight_map.values():
                if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") or component + "/" + name not in checksums:
                    raise ValueError("Cached weight shard is missing or invalid.")
        elif component + "/" + basename + ".safetensors" not in checksums:
            raise ValueError("Cached model is incomplete: missing " + component + " weights.")
    weights = hashlib.sha256()
    for name in sorted(checksums):
        if name.endswith(".safetensors"):
            weights.update(name.encode() + b"\0" + checksums[name].encode())
    # Keep the weight digest compatible with WanMpsAdapter's pinned weight check.
    assets = {name: digest for name, digest in checksums.items() if not name.endswith(".safetensors")}
    for path, signature in {**directory_signatures, **signatures}.items():
        if _signature(path) != signature:
            raise ValueError("Cached recipe source changed during capture.")
    return "sha256:" + weights.hexdigest(), assets, sizes


def prepare(settings, root=None):
    from video_lab.domain import Manifest
    value = validate_settings(settings)
    versions = runtime_versions()
    digest, assets, _ = _snapshot(value.pop("revision"), Path(root) if root is not None else cache_root())
    manifest = Manifest(model_id=MODEL_ID, model_revision=settings["revision"], model_checksum=digest,
        adapter_id=ADAPTER_ID, adapter_version="1", runtime_versions=versions, source_checksums=assets,
        scheduler="unipc", dtype="bfloat16", quantization="none", backend="mps", device="mps",
        memory_strategy="resident", hardware_fingerprint=platform.machine(), os_fingerprint=platform.platform(), **value)
    return manifest.model_dump(mode="json")


def verify(manifest, root=None):
    """Recheck a captured recipe before and after model execution."""
    value = manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else manifest
    if value["model_id"] != MODEL_ID or value["adapter_id"] != ADAPTER_ID or value["adapter_version"] != "1":
        raise ValueError("Captured recipe adapter identity changed.")
    if value["runtime_versions"] != runtime_versions():
        raise ValueError("Captured recipe runtime changed. Capture a new recipe.")
    if value["hardware_fingerprint"] != platform.machine() or value["os_fingerprint"] != platform.platform():
        raise ValueError("Captured recipe platform changed. Capture a new recipe.")
    digest, assets, sizes = _snapshot(value["model_revision"], Path(root) if root is not None else cache_root())
    if digest != value["model_checksum"] or assets != value["source_checksums"]:
        raise ValueError("Captured model weights or assets changed. Capture a new recipe.")
    return sizes
