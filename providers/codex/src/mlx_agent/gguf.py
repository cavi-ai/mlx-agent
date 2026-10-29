"""Read-only GGUF inventory: header parsing, duplicate grouping, conversion pairing.

Nothing here downloads, deletes, moves, or converts. It reads bounded GGUF
headers, groups files that describe the same model, and reports which of them
already have an MLX output. Mutating conversion lives in :mod:`mlx_agent.convert`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path


MAX_SCAN_FILES = 4000
MAX_METADATA_PAIRS = 512
MAX_STRING_BYTES = 8192
MAX_ARRAY_ITEMS = 1024
MAX_CONFIG_BYTES = 1024 * 1024
SIGNATURE_CHUNK_BYTES = 1024 * 1024
PROVENANCE_NAME = "mlx-converter.json"
GGUF_MAGIC = b"GGUF"
GGUF_SUFFIX = ".gguf"
SUPPORTED_GGUF_VERSIONS = (2, 3)

_SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".quarantine", ".Trash",
}
_SHARD = re.compile(r"^(?P<stem>.+?)-(?P<index>\d{5})-of-(?P<total>\d{5})$")
_QUANT_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(IQ[1-4](?:_[A-Z]+)*|Q[2-8](?:_[0-9KMSL]+)*|MXFP4|TQ[12]_0|BF16|F16|F32)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Vision projectors and other companion files ship beside a base model, carry
# its general.name, and are never converted on their own.
_COMPANION_NAMES = ("mmproj", "projector")
_COMPANION_ARCHITECTURES = ("clip", "mmproj")
# llama.cpp speculative-decoding drafters: they borrow the target's embeddings
# and head, share its general.name, and run only beside it.
DRAFT_ARCHITECTURES = ("dflash", "eagle3")
_DRAFT_SUFFIXES = (".block_size", ".hyper_connection.count", ".nextn_predict_layers")
# Drafter families a bundled port converts, keyed by what marks them in the header.
_DRAFT_PORTS = (("deepseek_v4_dspark", "dflash", "dflash.hyper_connection.count"),)
# Model architectures transformers cannot load from GGUF, converted by a bundled port of the same name.
ARCHITECTURE_PORTS = {"qwen35": "qwen35"}
# The llama.cpp vision projector type each port attaches from the model's folder.
PORT_PROJECTORS = {"qwen35": "qwen3vl_merger"}
_MIN_NAME_MATCH = 6
# Trailing MLX quantization labels in folder names: Foo-MLX-4bit, Foo-4.4bpw, Foo-bf16, Foo-mxfp8.
_MLX_QUANT_SUFFIX = re.compile(
    r"[-_.](?:mlx|vq|\d+(?:\.\d+)?-?bits?|\d+(?:\.\d+)?bpw|bf16|fp16|f16|fp32|f32|mxfp[48]|nvfp4|q[2-8](?:_[0-9a-z]+)*)$",
    re.IGNORECASE,
)
# A trailing parameter count (Foo-27B, foo:27b); the architecture check tells sizes apart.
_SIZE_SUFFIX = re.compile(r"[-_.]\d+(?:\.\d+)?[bm]$", re.IGNORECASE)
# Ollama model stores: manifests (current and v2 trees) naming content-addressed blobs.
_OLLAMA_MANIFEST_TREES = ("manifests-v2", "manifests")
_OLLAMA_MEDIA = "application/vnd.ollama.image."
_OLLAMA_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MLX_STRUCTURE_KEYS = (
    "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
)
_INTERESTING_KEYS = (
    "general.architecture",
    "general.name",
    "general.basename",
    "general.size_label",
    "general.file_type",
    "general.quantization_version",
    "general.parameter_count",
    "clip.projector_type",
    "clip.vision.projection_dim",
)
_STRUCTURE_SUFFIXES = (
    ".block_count",
    ".embedding_length",
    ".feed_forward_length",
    ".attention.head_count",
    ".context_length",
)

# llama.cpp LLAMA_FTYPE values that appear as general.file_type in the header.
_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

# Metadata value type tags, GGUF spec v2/v3.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12
_SCALARS = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1), _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4), _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}


_DEFAULT_ROOTS = (
    "~/.cache/huggingface/hub",
    "~/.cache/lm-studio/models",
    "~/.lmstudio/models",
    "~/models",
    "~/Models",
)


def default_gguf_roots():
    """Well-known local weight directories that exist on this host."""
    roots = []
    for candidate in _DEFAULT_ROOTS:
        expanded = Path(candidate).expanduser()
        if expanded.is_dir():
            roots.append(str(expanded))
    return roots


class GGUFError(RuntimeError):
    """Classified GGUF inventory failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


class _Truncated(ValueError):
    """The header ended before a declared field was complete."""


def _read_exactly(handle, count):
    if count < 0 or count > MAX_STRING_BYTES * MAX_ARRAY_ITEMS:
        raise _Truncated("declared length is out of bounds")
    chunk = handle.read(count)
    if len(chunk) != count:
        raise _Truncated("header ended early")
    return chunk


def _read_scalar(handle, value_type):
    layout, size = _SCALARS[value_type]
    return struct.unpack(layout, _read_exactly(handle, size))[0]


def _read_string(handle):
    length = struct.unpack("<Q", _read_exactly(handle, 8))[0]
    if length > MAX_STRING_BYTES:
        raise _Truncated("string exceeds the bounded header budget")
    return _read_exactly(handle, length).decode("utf-8", "replace")


def _read_value(handle, value_type, depth=0):
    if value_type in _SCALARS:
        return _read_scalar(handle, value_type)
    if value_type == _STRING:
        return _read_string(handle)
    if value_type == _ARRAY and depth == 0:
        item_type = struct.unpack("<I", _read_exactly(handle, 4))[0]
        count = struct.unpack("<Q", _read_exactly(handle, 8))[0]
        if item_type == _STRING:
            # Tokenizer vocabularies are huge and never inventory-relevant.
            return {"array": "string", "count": count, "skipped": True}
        if item_type not in _SCALARS:
            raise _Truncated("unsupported nested array element")
        _layout, size = _SCALARS[item_type]
        if count > MAX_ARRAY_ITEMS:
            handle.seek(count * size, os.SEEK_CUR)
            return {"array": item_type, "count": count, "skipped": True}
        return [_read_scalar(handle, item_type) for _ in range(count)]
    raise _Truncated("unsupported metadata value type")


def _skip_value(handle, value_type):
    """Consume a value we do not keep, without materializing it."""
    if value_type in _SCALARS:
        handle.seek(_SCALARS[value_type][1], os.SEEK_CUR)
        return
    if value_type == _STRING:
        length = struct.unpack("<Q", _read_exactly(handle, 8))[0]
        if length > MAX_STRING_BYTES * MAX_ARRAY_ITEMS:
            raise _Truncated("string exceeds the bounded header budget")
        handle.seek(length, os.SEEK_CUR)
        return
    if value_type != _ARRAY:
        raise _Truncated("unsupported metadata value type")
    item_type = struct.unpack("<I", _read_exactly(handle, 4))[0]
    count = struct.unpack("<Q", _read_exactly(handle, 8))[0]
    if item_type in _SCALARS:
        handle.seek(_SCALARS[item_type][1] * count, os.SEEK_CUR)
        return
    if item_type != _STRING:
        raise _Truncated("unsupported nested array element")
    for _ in range(count):
        length = struct.unpack("<Q", _read_exactly(handle, 8))[0]
        handle.seek(length, os.SEEK_CUR)


def _keep(key):
    if key in _INTERESTING_KEYS:
        return True
    return any(key.endswith(suffix) for suffix in _STRUCTURE_SUFFIXES + _DRAFT_SUFFIXES)


def read_gguf_header(path, include_tensors=False):
    """Parse the bounded metadata header of one GGUF file.

    With include_tensors, also read bounded tensor descriptors and their data
    offset without reading tensor payloads.

    Returns a dict with ``version``, ``tensor_count`` and the inventory-relevant
    ``metadata`` subset, or raises :class:`GGUFError` for anything unreadable.
    """
    location = Path(path)
    try:
        with location.open("rb") as handle:
            if _read_exactly(handle, 4) != GGUF_MAGIC:
                raise GGUFError(
                    "not_gguf",
                    "{0} does not start with the GGUF magic bytes.".format(location.name),
                    "Point the scan at real GGUF weights; the file may be a partial download.",
                )
            version = struct.unpack("<I", _read_exactly(handle, 4))[0]
            if version not in SUPPORTED_GGUF_VERSIONS:
                raise GGUFError(
                    "unsupported_gguf_version",
                    "GGUF version {0} is not supported.".format(version),
                    "Only GGUF v2 and v3 headers are parsed; re-export the model with a current llama.cpp.",
                )
            tensor_count = struct.unpack("<Q", _read_exactly(handle, 8))[0]
            pair_count = struct.unpack("<Q", _read_exactly(handle, 8))[0]
            metadata = {}
            for _ in range(min(pair_count, MAX_METADATA_PAIRS)):
                key = _read_string(handle)
                value_type = struct.unpack("<I", _read_exactly(handle, 4))[0]
                if _keep(key) or (include_tensors and key == "general.alignment"):
                    metadata[key] = _read_value(handle, value_type)
                else:
                    _skip_value(handle, value_type)
            tensors = []
            if include_tensors:
                if pair_count > MAX_METADATA_PAIRS or not 0 < tensor_count <= 100000:
                    raise _Truncated("unbounded tensor header")
                for _ in range(tensor_count):
                    name = _read_string(handle)
                    rank = struct.unpack("<I", _read_exactly(handle, 4))[0]
                    if not 1 <= rank <= 4:
                        raise _Truncated("invalid tensor rank")
                    shape = [struct.unpack("<Q", _read_exactly(handle, 8))[0] for _ in range(rank)]
                    if any(not 0 < dimension <= 2**31 for dimension in shape):
                        raise _Truncated("invalid tensor dimensions")
                    kind, offset = struct.unpack("<IQ", _read_exactly(handle, 12))
                    tensors.append({"name": name, "shape": shape, "type": kind, "offset": offset})
                    if handle.tell() > 64 * 1024 * 1024:
                        raise _Truncated("oversized tensor header")
                alignment = metadata.get("general.alignment", 32)
                if type(alignment) is not int or not 0 < alignment <= 4096 or alignment & (alignment - 1):
                    raise _Truncated("invalid tensor alignment")
                tensor_data_offset = (handle.tell() + alignment - 1) // alignment * alignment
    except (_Truncated, struct.error) as error:
        raise GGUFError(
            "gguf_header_unreadable",
            "{0} has a truncated or malformed GGUF header.".format(location.name),
            "Re-download the file; a truncated shard cannot be converted.",
        ) from error
    except OSError as error:
        raise GGUFError(
            "gguf_unreadable",
            "{0} could not be read.".format(location.name),
            "Check the path and its permissions, then rerun the scan.",
        ) from error
    result = {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_pairs": pair_count,
        "metadata": metadata,
    }
    if include_tensors:
        result.update(tensors=tensors, tensor_data_offset=tensor_data_offset)
    return result


def file_signature(path, chunk_bytes=SIGNATURE_CHUNK_BYTES):
    """Cheap content signature: size plus the head and tail of the file.

    Full digests of multi-gigabyte weights are not worth their I/O. Two files
    that agree on size, head and tail are treated as the same bytes.
    """
    location = Path(path)
    size = location.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with location.open("rb") as handle:
        digest.update(handle.read(chunk_bytes))
        if size > chunk_bytes * 2:
            handle.seek(-chunk_bytes, os.SEEK_END)
            digest.update(handle.read(chunk_bytes))
    return digest.hexdigest()


def _normalize_stem(stem):
    shard = _SHARD.match(stem)
    if shard:
        stem = shard.group("stem")
    without_quant = _QUANT_IN_NAME.sub("", stem)
    cleaned = re.sub(r"[^a-z0-9]+", "-", without_quant.lower()).strip("-")
    return cleaned or stem.lower()


def _quantization(metadata, stem):
    file_type = metadata.get("general.file_type")
    if isinstance(file_type, int) and file_type in _FILE_TYPES:
        return _FILE_TYPES[file_type]
    match = _QUANT_IN_NAME.search(stem)
    if match:
        return match.group(1).upper()
    return "unknown"


def _structure(metadata, tensor_count):
    architecture = metadata.get("general.architecture")
    fields = [str(architecture), str(tensor_count)]
    for suffix in _STRUCTURE_SUFFIXES:
        key = "{0}{1}".format(architecture, suffix)
        fields.append(str(metadata.get(key)))
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:16]


def _prediction_only(metadata, architecture, tensor_count):
    """A multi-token-prediction export: llama.cpp's nextn block without the layers it predicts from."""
    layers = metadata.get("{0}.nextn_predict_layers".format(architecture))
    blocks = metadata.get("{0}.block_count".format(architecture))
    counts = [value for value in (layers, blocks) if isinstance(value, int) and not isinstance(value, bool)]
    return len(counts) == 2 and layers > 0 and tensor_count < blocks


def _draft(metadata, architecture, tensor_count):
    """Drafter facts for a speculative-decoding GGUF, else None; ``port`` is set when one converts it."""
    if str(architecture).lower() not in DRAFT_ARCHITECTURES and not _prediction_only(metadata, architecture, tensor_count):
        return None
    port = next((name for name, arch, key in _DRAFT_PORTS if arch == str(architecture).lower() and key in metadata), None)
    block_size = metadata.get("{0}.block_size".format(architecture))
    return {
        "port": port,
        "target": metadata.get("general.name") or metadata.get("general.basename"),
        "block_size": block_size if isinstance(block_size, int) and not isinstance(block_size, bool) else None,
    }


def describe_gguf(path, signature=True):
    """Build one inventory entry for a GGUF file. Never raises for bad headers."""
    location = Path(path)
    try:
        stat = location.stat()
    except OSError as error:
        raise GGUFError(
            "gguf_unreadable",
            "{0} could not be read.".format(location.name),
            "Check the path and its permissions, then rerun the scan.",
        ) from error
    stem = location.stem
    shard = _SHARD.match(stem)
    entry = {
        "path": str(location),
        "name": location.name,
        "bytes": stat.st_size,
        "modified_at": int(stat.st_mtime),
        "shard": None if shard is None else {
            "index": int(shard.group("index")),
            "total": int(shard.group("total")),
        },
        "model_key": _normalize_stem(stem),
        "architecture": None,
        "quantization": "unknown",
        "parameters": None,
        "structure": None,
        "signature": None,
        "companion": any(token in stem.lower() for token in _COMPANION_NAMES),
        "draft": None,
        "port": None,
        "readable": True,
        "error": None,
    }
    try:
        header = read_gguf_header(location)
    except GGUFError as error:
        entry["readable"] = False
        entry["error"] = {"code": error.code, "message": str(error)}
        return entry
    metadata = header["metadata"]
    name = metadata.get("general.name") or metadata.get("general.basename")
    entry["architecture"] = metadata.get("general.architecture")
    entry["companion"] = (
        any(token in stem.lower() for token in _COMPANION_NAMES)
        or str(entry["architecture"]).lower() in _COMPANION_ARCHITECTURES
    )
    entry["quantization"] = _quantization(metadata, stem)
    entry["structure"] = _structure(metadata, header["tensor_count"])
    entry["tensor_count"] = header["tensor_count"]
    parameters = metadata.get("general.parameter_count")
    if isinstance(parameters, int) and not isinstance(parameters, bool):
        entry["parameters"] = parameters
    if isinstance(name, str) and name:
        entry["model_key"] = _normalize_stem(name)
    entry["draft"] = _draft(metadata, entry["architecture"], header["tensor_count"])
    if entry["draft"] is not None:
        # A drafter carries its target's name; keep it out of the target's duplicate group.
        entry["model_key"] += "-draft"
    elif not entry["companion"]:
        entry["port"] = ARCHITECTURE_PORTS.get(str(entry["architecture"]).lower())
    if signature:
        try:
            entry["signature"] = file_signature(location)
        except OSError:
            entry["signature"] = None
    return entry


def find_projector(path, port):
    """The single vision projector beside ``path`` that ``port`` attaches, else None.

    It must be the projector type the port reads and project to the model's
    embedding width; none or several candidates leave the model text-only.
    """
    projector_type = PORT_PROJECTORS.get(port)
    location = Path(path)
    if projector_type is None:
        return None
    try:
        metadata = read_gguf_header(location)["metadata"]
        siblings = sorted(location.parent.iterdir())
    except (GGUFError, OSError):
        return None
    width = metadata.get("{0}.embedding_length".format(metadata.get("general.architecture")))
    matches = []
    for sibling in siblings:
        if sibling == location or sibling.suffix.lower() != GGUF_SUFFIX or not sibling.is_file():
            continue
        try:
            candidate = read_gguf_header(sibling)["metadata"]
        except (GGUFError, OSError):
            continue
        if (candidate.get("general.architecture") == "clip" and candidate.get("clip.projector_type") == projector_type
                and candidate.get("clip.vision.projection_dim") == width):
            matches.append(sibling)
    if len(matches) != 1:
        return None
    return {"path": str(matches[0]), "name": matches[0].name, "bytes": matches[0].stat().st_size}


def _iter_files(roots, suffix, limit=MAX_SCAN_FILES):
    seen = set()
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            if suffix == ".gguf" and "strata-model.json" in filenames:
                from .strata_runtime import StrataError, read_model
                try:
                    read_model(Path(dirpath))
                except StrataError:
                    pass
                else:
                    dirnames[:] = []
                    continue
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in _SKIP_DIRECTORIES and not name.startswith(".")
            )
            for filename in sorted(filenames):
                if not filename.lower().endswith(suffix):
                    continue
                location = Path(dirpath) / filename
                try:
                    resolved = location.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield location
                if len(seen) >= limit:
                    return


def scan_gguf(roots, signature=True, limit=MAX_SCAN_FILES):
    """Inventory every GGUF file beneath the configured roots."""
    entries = []
    for location in _iter_files(roots, GGUF_SUFFIX, limit):
        try:
            entries.append(describe_gguf(location, signature=signature))
        except GGUFError:
            continue
    entries.sort(key=lambda entry: entry["path"])
    return entries


def read_provenance(directory):
    """Read an MLX output's converter provenance marker, if it has one."""
    location = Path(directory) / PROVENANCE_NAME
    try:
        value = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("source"), dict):
        return None
    return value


def _mlx_config(directory):
    """Return the MLX marker for a model directory, or None if it is not one.

    A directory only counts as an MLX output when its ``config.json`` carries a
    ``quantization`` block (what ``mlx_lm.convert -q`` writes) or when this tool
    left a provenance marker. Plain PyTorch checkpoints in the Hugging Face
    cache also hold ``config.json`` and safetensors, and must not be mistaken
    for conversions.
    """
    location = Path(directory) / "config.json"
    try:
        if location.stat().st_size > MAX_CONFIG_BYTES:
            return None
        value = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    quantization = value.get("quantization")
    if not isinstance(quantization, dict):
        return None
    return {
        "bits": quantization.get("bits"),
        "group_size": quantization.get("group_size"),
        "model_type": value.get("model_type"),
    }


def _mlx_community_snapshot(directory):
    """A Hugging Face cache snapshot of an mlx-community repository: MLX weights even when unquantized (bf16)."""
    parts = Path(directory).parts
    return len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--mlx-community--")


def _component_weights(directory):
    """A converter that saves per-component folders (mflux) lists them in config.json ``components``."""
    try:
        value = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    names = value.get("components") if isinstance(value, dict) else None
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name and "/" not in name and name not in (".", "..") for name in names
    ):
        return False
    return all(any(Path(directory, name).glob("*.safetensors")) for name in names)


def scan_mlx_outputs(roots, limit=MAX_SCAN_FILES):
    """Inventory MLX model directories beneath the configured roots."""
    outputs = []
    seen = set()
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in _SKIP_DIRECTORIES and not name.startswith(".")
            )
            if "config.json" not in filenames:
                continue
            directory = Path(dirpath)
            if "strata-model.json" in filenames:
                from .strata_runtime import StrataError, read_model
                try:
                    profile = read_model(directory)
                except StrataError:
                    continue
                resolved = directory.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    outputs.append({"path": str(directory), "name": directory.name,
                                    "model_key": _normalize_stem(directory.name), "quantization": None,
                                    "provenance": None, "runtime": "strata", "format": "gguf",
                                    "weight_bytes": profile["weight_bytes"]})
                dirnames[:] = []
                if len(seen) >= limit:
                    return sorted(outputs, key=lambda item: item["path"])
                continue
            # is_file follows cache symlinks: a snapshot whose weight blobs are missing is not a model.
            if not any(name.endswith(".safetensors") and (directory / name).is_file() for name in filenames) \
                    and not _component_weights(directory):
                continue
            provenance = read_provenance(directory)
            marker = _mlx_config(directory)
            if provenance is None and marker is None and not _mlx_community_snapshot(directory):
                continue
            try:
                resolved = directory.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            dirnames[:] = []
            outputs.append({
                "path": str(directory),
                "name": directory.name,
                "model_key": _normalize_stem(directory.name),
                "quantization": marker,
                "provenance": provenance,
            })
            if len(seen) >= limit:
                return sorted(outputs, key=lambda item: item["path"])
    return sorted(outputs, key=lambda item: item["path"])


def _output_matches(entry, output):
    provenance = output.get("provenance") or {}
    source = provenance.get("source") or {}
    if entry.get("signature") and source.get("signature") == entry["signature"]:
        return "provenance"
    if source.get("path") == entry["path"]:
        return "provenance"
    key = entry["model_key"]
    if len(key) < _MIN_NAME_MATCH:
        return None
    candidate = output["model_key"]
    if candidate == key or candidate.startswith(key + "-"):
        return "name"
    return None


def pair_conversions(entries, outputs, receipts=()):
    """Mark each GGUF as converted or pending, by provenance then by name."""
    receipt_sources = {}
    for receipt in receipts:
        source = receipt.get("source") or {}
        for key in ("signature", "path"):
            value = source.get(key)
            if isinstance(value, str) and value:
                receipt_sources.setdefault(value, receipt)
    paired = []
    for entry in entries:
        shard = entry.get("shard")
        if shard is not None and shard["index"] != 1:
            paired.append(dict(entry, status="shard", outputs=[], evidence="shard"))
            continue
        if entry.get("companion"):
            paired.append(dict(entry, status="companion", outputs=[], evidence="companion"))
            continue
        matches = []
        evidence = None
        for output in outputs:
            match = _output_matches(entry, output)
            if match is None:
                continue
            matches.append(output["path"])
            evidence = "provenance" if match == "provenance" else evidence or "name"
        receipt = receipt_sources.get(entry.get("signature")) or receipt_sources.get(entry["path"])
        if receipt is not None and not matches:
            out = receipt.get("out")
            if isinstance(out, str) and Path(out).is_dir():
                matches.append(out)
                evidence = "receipt"
        paired.append(dict(
            entry,
            status="converted" if matches else "pending",
            outputs=sorted(set(matches)),
            evidence=evidence,
        ))
    return paired


def group_duplicates(entries):
    """Group GGUFs that describe the same model.

    ``exact`` groups are byte-identical or same-structure-same-quantization
    files: every member but the keeper is redundant. ``variant`` groups are the
    same model at different quantization levels — reported, never recommended
    for removal, because the choice between them is the user's.
    """
    exact = {}
    variant = {}
    for entry in entries:
        if entry.get("shard") is not None and entry["shard"]["index"] != 1:
            continue
        if not entry.get("readable") or entry.get("companion"):
            continue
        structure = entry.get("structure")
        identity = entry.get("signature") or (
            "{0}:{1}:{2}".format(entry["model_key"], structure, entry["quantization"])
        )
        exact.setdefault(identity, []).append(entry)
        if structure:
            variant.setdefault("{0}:{1}".format(entry["model_key"], structure), []).append(entry)

    def _keeper(members):
        # Keep the already-converted copy, then an Ollama copy (only Ollama removes its blobs), else the oldest path.
        converted = [item for item in members if item.get("status") == "converted"]
        pool = converted or [item for item in members if item.get("status") == "ollama"] or members
        return sorted(pool, key=lambda item: (item["modified_at"], item["path"]))[0]

    groups = []
    for identity, members in sorted(exact.items()):
        if len(members) < 2:
            continue
        keep = _keeper(members)
        groups.append({
            "kind": "exact",
            "identity": identity,
            "model_key": keep["model_key"],
            "quantization": keep["quantization"],
            "keep": keep["path"],
            "redundant": sorted(item["path"] for item in members if item["path"] != keep["path"]),
            "reclaimable_bytes": sum(
                item["bytes"] for item in members if item["path"] != keep["path"]
            ),
        })
    for identity, members in sorted(variant.items()):
        quantizations = {item["quantization"] for item in members}
        if len(members) < 2 or len(quantizations) < 2:
            continue
        groups.append({
            "kind": "variant",
            "identity": identity,
            "model_key": members[0]["model_key"],
            "quantizations": sorted(quantizations),
            "members": sorted(item["path"] for item in members),
            "redundant": [],
            "reclaimable_bytes": 0,
        })
    return groups


def _hf_snapshot(directory):
    parts = Path(directory).parts
    return len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--")


def _mlx_name(directory):
    """Folder name of an MLX model; a cache snapshot uses its repository name."""
    return Path(directory).parts[-3].split("--")[-1] if _hf_snapshot(directory) else Path(directory).name


def _mlx_name_key(name, sizes=False):
    """A model name without trailing MLX quantization labels, and without its parameter count when ``sizes``."""
    stem = name
    while True:
        trimmed = _MLX_QUANT_SUFFIX.sub("", stem)
        if sizes:
            trimmed = _SIZE_SUFFIX.sub("", trimmed)
        if trimmed == stem or not trimmed:
            break
        stem = trimmed
    return _normalize_stem(stem)


def _mlx_model_key(directory):
    return _mlx_name_key(_mlx_name(directory))


def _read_json_object(location, limit):
    try:
        if Path(location).stat().st_size > limit:
            return None
        value = json.loads(Path(location).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _mlx_structure(directory):
    return _config_structure(_read_json_object(Path(directory) / "config.json", MAX_CONFIG_BYTES))


def _config_structure(value):
    """Architecture dimensions from a config.json object (or its text_config), or None when it names none."""
    if not isinstance(value, dict):
        return None
    nested = value.get("text_config") if isinstance(value.get("text_config"), dict) else {}
    fields = [str(value.get(key, nested.get(key))) for key in _MLX_STRUCTURE_KEYS]
    if all(field == "None" for field in fields[1:]):
        return None
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:16]


def _mlx_files(directory):
    """(relative path, size, file identity) of every file in an MLX folder, cache symlinks followed.

    None when a file cannot be read, such as a cache symlink whose blob is gone.
    """
    base = Path(directory)
    files = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRECTORIES and not name.startswith("."))
        for filename in filenames:
            location = Path(dirpath) / filename
            try:
                stat = location.stat()
            except OSError:
                return None
            files.append((location.relative_to(base).as_posix(), stat.st_size, (stat.st_dev, stat.st_ino)))
    return sorted(files)


def _mlx_quantization(output):
    marker = output.get("quantization")
    bits = marker.get("bits") if isinstance(marker, dict) else None
    return "{0}-bit".format(bits) if isinstance(bits, int) and not isinstance(bits, bool) else "unknown"


def _ollama_stores(roots):
    """Ollama model stores at or below the roots: folders holding ``blobs`` beside a manifests tree."""
    stores = []
    seen = set()
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for dirpath, dirnames, _filenames in os.walk(base, followlinks=False):
            if "blobs" in dirnames and any(tree in dirnames for tree in _OLLAMA_MANIFEST_TREES):
                resolved = Path(dirpath).resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    stores.append(Path(dirpath))
                dirnames[:] = []
                continue
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in _SKIP_DIRECTORIES and not name.startswith(".")
            )
    return stores


def _ollama_blob(store, digest):
    if not isinstance(digest, str) or not _OLLAMA_DIGEST.match(digest):
        return None
    location = store / "blobs" / digest.replace(":", "-")
    return location if location.is_file() else None


def _describe_ollama(store, tree, manifest, signature):
    """One Ollama model and the set of its layer digests, or None when it is not a local model."""
    parts = manifest.relative_to(tree).parts
    value = _read_json_object(manifest, MAX_MANIFEST_BYTES)
    layers = value.get("layers") if value else None
    if len(parts) < 3 or not isinstance(layers, list) or not layers:
        return None
    described = []
    for layer in layers:
        media = layer.get("mediaType") if isinstance(layer, dict) else None
        blob = _ollama_blob(store, layer.get("digest")) if isinstance(media, str) else None
        if blob is None:
            return None
        described.append((media[len(_OLLAMA_MEDIA):] if media.startswith(_OLLAMA_MEDIA) else media, layer, blob))
    config = value.get("config") if isinstance(value.get("config"), dict) else {}
    config_blob = _ollama_blob(store, config.get("digest"))
    config = (_read_json_object(config_blob, MAX_CONFIG_BYTES) if config_blob else None) or {}
    model, tag = parts[-2], parts[-1]
    name = "{0}:{1}".format(model if parts[1:-2] == ("library",) else "/".join(parts[:-1]), tag)
    sizes = {layer["digest"]: layer.get("size") for _kind, layer, _blob in described}
    record = {
        "name": name,
        "path": str(manifest),
        "blob": None,
        "bytes": sum(size for size in sizes.values() if isinstance(size, int) and not isinstance(size, bool)),
        "modified_at": int(manifest.stat().st_mtime),
    }
    gguf = next((blob for kind, _layer, blob in described if kind == "model"), None)
    if gguf is not None:
        try:
            entry = describe_gguf(gguf, signature=signature)
        except GGUFError:
            return None
        record.update(
            format="gguf", blob=str(gguf), architecture=entry["architecture"], model_key=entry["model_key"],
            quantization=entry["quantization"], structure=entry["structure"], signature=entry["signature"],
            readable=entry["readable"], draft=entry["draft"],
        )
    elif any(kind == "tensor" for kind, _layer, _blob in described):
        settings = next((_read_json_object(blob, MAX_CONFIG_BYTES) for kind, layer, blob in described
                         if kind == "json" and layer.get("name") == "config.json"), None) or {}
        marker = settings.get("quantization") if isinstance(settings.get("quantization"), dict) else {}
        label = _mlx_quantization({"quantization": marker})
        if isinstance(config.get("file_type"), str) and config["file_type"]:
            label = config["file_type"]
        record.update(
            format="safetensors", architecture=settings.get("model_type"),
            model_key=_mlx_name_key("{0}-{1}".format(model, tag)), quantization=label,
            structure=_config_structure(settings), signature=None, readable=True, draft=None,
        )
    else:
        return None
    return record, frozenset(layer["digest"] for _kind, layer, _blob in described)


def scan_ollama(roots, signature=True, limit=MAX_SCAN_FILES):
    """Inventory the models in Ollama stores at or below the roots. Read-only.

    A GGUF model is described from its model blob, a safetensors (MLX) model
    from its ``config.json`` layer. A manifest naming a missing blob is skipped.
    A model listed in both manifest trees, by name or by identical layers,
    counts once, from the current (v2) tree.
    """
    models = []
    seen = set()
    for store in _ollama_stores(roots):
        for name in _OLLAMA_MANIFEST_TREES:
            tree = store / name
            for dirpath, dirnames, filenames in os.walk(tree, followlinks=False):
                dirnames[:] = sorted(item for item in dirnames if not item.startswith("."))
                for filename in sorted(item for item in filenames if not item.startswith(".")):
                    found = _describe_ollama(store, tree, Path(dirpath) / filename, signature)
                    if found is None:
                        continue
                    keys = {(store, found[0]["name"]), (store, found[1])}
                    if keys & seen:
                        continue
                    seen.update(keys)
                    models.append(found[0])
                    if len(models) >= limit:
                        return sorted(models, key=lambda item: item["name"])
    return sorted(models, key=lambda item: item["name"])


def _ollama_gguf_entries(models):
    """Ollama GGUF models as duplicate-grouping entries; models sharing one blob count once."""
    entries = {}
    for model in models:
        if model["format"] != "gguf" or model["blob"] in entries:
            continue
        try:
            size = Path(model["blob"]).stat().st_size
        except OSError:
            continue
        entries[model["blob"]] = dict(model, bytes=size, shard=None, companion=False, status="ollama")
    return list(entries.values())


def group_mlx_duplicates(outputs, signature=True, ollama=()):
    """Group MLX folders that hold the same model, with the same rules as :func:`group_duplicates`.

    ``exact`` folders hold identical weights: the same weight files at the same
    sizes, and with ``signature`` the same head and tail bytes; without it, the
    same model name and bit width too. Folders whose weights are the same files
    on disk (cache revisions sharing blobs) count once. ``variant`` folders and
    Ollama safetensors models share a model name (without its parameter count)
    and architecture at different bit widths.
    """
    described = []
    for output in outputs:
        if output.get("runtime") == "strata":
            continue
        files = _mlx_files(output["path"])
        if files is None:
            continue
        weights = [item for item in files if item[0].endswith(".safetensors")]
        if not weights:
            continue
        try:
            modified = (Path(output["path"]) / "config.json").stat().st_mtime
        except OSError:
            continue
        described.append(dict(
            output,
            model_key=_mlx_model_key(output["path"]),
            variant_key=_mlx_name_key(_mlx_name(output["path"]), sizes=True),
            label=_mlx_quantization(output),
            structure=_mlx_structure(output["path"]),
            weights=tuple((name, size) for name, size, _identity in weights),
            identities=frozenset(identity for _name, _size, identity in weights),
            bytes=sum(size for _name, size, _identity in files),
            modified_at=modified,
        ))

    candidates = {}
    for item in described:
        key = item["weights"] if signature else (item["model_key"], item["label"], item["weights"])
        candidates.setdefault(key, {}).setdefault(item["identities"], item)
    exact = {}
    for key, distinct in candidates.items():
        if len(distinct) < 2:
            continue
        for item in distinct.values():
            if not signature:
                exact.setdefault(hashlib.sha256(repr(key).encode("utf-8")).hexdigest(), []).append(item)
                continue
            try:
                digest = hashlib.sha256("|".join(
                    "{0}:{1}".format(name, file_signature(Path(item["path"]) / name))
                    for name, _size in item["weights"]
                ).encode("utf-8")).hexdigest()
            except OSError:
                continue
            exact.setdefault(digest, []).append(item)

    def _keeper(members):
        # Keep a converter output, then a shared Hugging Face cache snapshot, then the oldest copy.
        return sorted(members, key=lambda item: (
            item.get("provenance") is None, not _hf_snapshot(item["path"]), item["modified_at"], item["path"],
        ))[0]

    groups = []
    for identity, members in sorted(exact.items(), key=lambda pair: sorted(item["path"] for item in pair[1])):
        if len(members) < 2:
            continue
        keep = _keeper(members)
        groups.append({
            "kind": "exact",
            "format": "mlx",
            "identity": "mlx:" + identity,
            "model_key": keep["model_key"],
            "quantization": keep["label"],
            "keep": keep["path"],
            "redundant": sorted(item["path"] for item in members if item["path"] != keep["path"]),
            "reclaimable_bytes": sum(item["bytes"] for item in members if item["path"] != keep["path"]),
        })
    stored = [
        dict(model, label=model["quantization"], status="ollama",
             variant_key=_mlx_name_key(model["name"].rsplit("/", 1)[-1].replace(":", "-"), sizes=True))
        for model in ollama if model["format"] == "safetensors"
    ]
    variant = {}
    for item in described + stored:
        if item["structure"] and len(item["variant_key"]) >= _MIN_NAME_MATCH:
            variant.setdefault("{0}:{1}".format(item["variant_key"], item["structure"]), []).append(item)
    for identity, members in sorted(variant.items()):
        labels = {item["label"] for item in members}
        if len(members) < 2 or len(labels) < 2:
            continue
        named = sorted(members, key=lambda item: (item.get("status") == "ollama", item["path"]))[0]
        groups.append({
            "kind": "variant",
            "format": "mlx",
            "identity": "mlx:" + identity,
            "model_key": named["model_key"],
            "quantizations": sorted(labels),
            "members": sorted(item["path"] for item in members),
            "redundant": [],
            "reclaimable_bytes": 0,
        })
    return groups


def inventory(gguf_roots, mlx_roots=(), receipts=(), signature=True, limit=MAX_SCAN_FILES):
    """One read-only pass: what exists, what is converted, what is redundant."""
    entries = scan_gguf(gguf_roots, signature=signature, limit=limit)
    outputs = scan_mlx_outputs(mlx_roots or gguf_roots, limit=limit)
    ollama = scan_ollama(list(gguf_roots) + list(mlx_roots), signature=signature, limit=limit)
    paired = pair_conversions(entries, outputs, receipts)
    duplicates = group_duplicates(paired + _ollama_gguf_entries(ollama)) + group_mlx_duplicates(
        outputs, signature=signature, ollama=ollama)
    for group in duplicates:
        group["group_id"] = "{0}:{1}".format(group["kind"], group["identity"])
    pending = [item for item in paired if item["status"] == "pending"]
    return {
        "roots": {
            "gguf": [str(Path(root).expanduser()) for root in gguf_roots],
            "mlx": [str(Path(root).expanduser()) for root in (mlx_roots or gguf_roots)],
        },
        "models": paired,
        "outputs": outputs,
        "ollama": ollama,
        "pending": [item["path"] for item in pending],
        "duplicates": duplicates,
        "totals": {
            "gguf": len(paired),
            "pending": len(pending),
            "converted": len([item for item in paired if item["status"] == "converted"]),
            "unreadable": len([item for item in paired if not item["readable"]]),
            "bytes": sum(item["bytes"] for item in paired),
            "reclaimable_bytes": sum(group["reclaimable_bytes"] for group in duplicates),
        },
    }
