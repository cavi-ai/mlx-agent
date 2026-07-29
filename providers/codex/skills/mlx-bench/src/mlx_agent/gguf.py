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
_MIN_NAME_MATCH = 6
_INTERESTING_KEYS = (
    "general.architecture",
    "general.name",
    "general.basename",
    "general.size_label",
    "general.file_type",
    "general.quantization_version",
    "general.parameter_count",
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
    return any(key.endswith(suffix) for suffix in _STRUCTURE_SUFFIXES)


def read_gguf_header(path):
    """Parse the bounded metadata header of one GGUF file.

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
                if _keep(key):
                    metadata[key] = _read_value(handle, value_type)
                else:
                    _skip_value(handle, value_type)
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
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_pairs": pair_count,
        "metadata": metadata,
    }


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
    if signature:
        try:
            entry["signature"] = file_signature(location)
        except OSError:
            entry["signature"] = None
    return entry


def _iter_files(roots, suffix, limit=MAX_SCAN_FILES):
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
            if not any(name.endswith(".safetensors") for name in filenames):
                continue
            directory = Path(dirpath)
            provenance = read_provenance(directory)
            marker = _mlx_config(directory)
            if provenance is None and marker is None:
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
        # Keep the already-converted copy, else the oldest stable path.
        converted = [item for item in members if item.get("status") == "converted"]
        pool = converted or members
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


def inventory(gguf_roots, mlx_roots=(), receipts=(), signature=True, limit=MAX_SCAN_FILES):
    """One read-only pass: what exists, what is converted, what is redundant."""
    entries = scan_gguf(gguf_roots, signature=signature, limit=limit)
    outputs = scan_mlx_outputs(mlx_roots or gguf_roots, limit=limit)
    paired = pair_conversions(entries, outputs, receipts)
    duplicates = group_duplicates(paired)
    pending = [item for item in paired if item["status"] == "pending"]
    return {
        "roots": {
            "gguf": [str(Path(root).expanduser()) for root in gguf_roots],
            "mlx": [str(Path(root).expanduser()) for root in (mlx_roots or gguf_roots)],
        },
        "models": paired,
        "outputs": outputs,
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
