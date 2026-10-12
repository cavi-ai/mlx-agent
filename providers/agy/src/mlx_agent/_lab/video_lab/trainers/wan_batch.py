"""Shared bounded authority for canonical Wan precomputed batch artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from collections.abc import Mapping
from typing import Any, cast

from video_lab.domain import canonical_json

WAN_BATCH_SCHEMA = "wan-precomputed-training-batch.v1"
WAN_BATCH_TENSORS = frozenset(
    {
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "clean_latents",
        "noise_latents",
        "target",
        "transformer_selector",
    }
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_RANK = 32
_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024 * 1024


def validated_safetensors_header(
    payload: bytes, *, expected_names: set[str] | frozenset[str], batch_size: int
) -> dict[str, dict[str, object]]:
    """Parse a bounded, gapless safetensors envelope without allocating tensors."""
    try:
        if len(payload) < 9 or len(payload) > _MAX_PAYLOAD_BYTES:
            raise ValueError
        header_size = struct.unpack("<Q", payload[:8])[0]
        if header_size <= 1 or header_size > _MAX_HEADER_BYTES or 8 + header_size > len(payload):
            raise ValueError
        return validated_safetensors_header_prefix(
            payload[: 8 + header_size],
            total_size=len(payload),
            expected_names=expected_names,
            batch_size=batch_size,
        )
    except (ValueError, struct.error):
        raise ValueError("wan_training_batch_invalid") from None


def validated_safetensors_header_prefix(
    prefix: bytes,
    *,
    total_size: int,
    expected_names: set[str] | frozenset[str],
    batch_size: int,
) -> dict[str, dict[str, object]]:
    """Validate an exact bounded header prefix against the declared full artifact size."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        if (
            len(prefix) < 9
            or total_size > _MAX_PAYLOAD_BYTES
            or type(batch_size) is not int
            or batch_size <= 0
        ):
            raise ValueError
        header_size = struct.unpack("<Q", prefix[:8])[0]
        if header_size <= 1 or header_size > _MAX_HEADER_BYTES or len(prefix) != 8 + header_size:
            raise ValueError
        parsed = json.loads(
            prefix[8:].decode().rstrip(" "), object_pairs_hook=unique_object
        )
        if not isinstance(parsed, dict):
            raise ValueError
        metadata = parsed.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(
                type(key) is not str or type(value) is not str for key, value in metadata.items()
            )
        ):
            raise ValueError
        if set(parsed) != set(expected_names) or len(parsed) > 100_000:
            raise ValueError
        data_size = total_size - 8 - header_size
        spans: list[tuple[int, int]] = []
        for item in parsed.values():
            if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
                raise ValueError
            dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
            if (
                type(dtype) is not str
                or dtype not in _DTYPE_BYTES
                or not isinstance(shape, list)
                or not shape
                or len(shape) > _MAX_RANK
                or any(type(dim) is not int or dim <= 0 or dim > 2**31 for dim in shape)
                or shape[0] != batch_size
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)
            ):
                raise ValueError
            count = math.prod(shape)
            start, end = offsets
            if count > data_size or end - start != count * _DTYPE_BYTES[dtype] or end > data_size:
                raise ValueError
            spans.append((start, end))
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError
            cursor = end
        if cursor != data_size:
            raise ValueError
        return cast(dict[str, dict[str, object]], parsed)
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, struct.error):
        raise ValueError("wan_training_batch_invalid") from None


def build_wan_batch_manifest(
    payload: bytes,
    *,
    batch_size: int,
    timestep_min: int,
    timestep_max: int,
    timestep_boundary: int,
) -> bytes:
    header = validated_safetensors_header(
        payload, expected_names=WAN_BATCH_TENSORS, batch_size=batch_size
    )
    header_size = struct.unpack("<Q", payload[:8])[0]
    tensors: list[dict[str, object]] = []
    for name in sorted(WAN_BATCH_TENSORS):
        item = header[name]
        shape = cast(list[int], item["shape"])
        offsets = cast(list[int], item["data_offsets"])
        start, end = 8 + header_size + offsets[0], 8 + header_size + offsets[1]
        tensors.append(
            {
                "name": name,
                "dtype": item["dtype"],
                "rank": len(shape),
                "shape": shape,
                "byte_offsets": offsets,
                "digest": "sha256:" + hashlib.sha256(payload[start:end]).hexdigest(),
                "finite": True,
            }
        )
    by_name = {cast(str, item["name"]): item for item in tensors}
    manifest = {
        "schema_version": WAN_BATCH_SCHEMA,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "batch_size": batch_size,
        "tensors": tensors,
        "objective": {
            "target_kind": "flow_noise_minus_clean",
            "formula": "target=noise_latents-clean_latents",
            "clean_latents_digest": by_name["clean_latents"]["digest"],
            "noise_latents_digest": by_name["noise_latents"]["digest"],
            "target_digest": by_name["target"]["digest"],
        },
        "noise_schedule": {
            "timestep_min": timestep_min,
            "timestep_max": timestep_max,
            "timestep_boundary": timestep_boundary,
            "selector_rule": "high_if_timestep_gte_boundary_else_low",
        },
    }
    return canonical_json(manifest).encode()


def validate_loaded_wan_batch(
    tensors: Mapping[str, Any],
    *,
    batch_size: int,
    timestep_min: int,
    timestep_max: int,
    boundary: int,
    in_channels: int,
) -> None:
    try:
        if set(tensors) != WAN_BATCH_TENSORS or not 0 <= timestep_min < boundary <= timestep_max:
            raise ValueError
        if any(
            getattr(tensor, "ndim", 0) < 1 or int(tensor.shape[0]) != batch_size
            for tensor in tensors.values()
        ):
            raise ValueError
        hidden, clean, noise = (
            tensors["hidden_states"],
            tensors["clean_latents"],
            tensors["noise_latents"],
        )
        target, timestep = tensors["target"], tensors["timestep"]
        selector, encoder = tensors["transformer_selector"], tensors["encoder_hidden_states"]
        sigma = (timestep / float(timestep_max)).to(dtype=hidden.dtype)
        while sigma.ndim < hidden.ndim:
            sigma = sigma.unsqueeze(-1)
        if (
            tuple(hidden.shape) != tuple(clean.shape)
            or tuple(hidden.shape) != tuple(noise.shape)
            or tuple(hidden.shape) != tuple(target.shape)
            or hidden.ndim < 3
            or int(hidden.shape[1]) != in_channels
            or encoder.ndim != 3
            or tuple(timestep.shape) != (batch_size,)
            or str(timestep.dtype) != "torch.float32"
            or tuple(selector.shape) != (batch_size,)
            or str(selector.dtype) != "torch.int64"
            or not bool((timestep >= timestep_min).all().item())
            or not bool((timestep <= timestep_max).all().item())
            or not bool(((selector == 0) | (selector == 1)).all().item())
            or not bool((selector == (timestep < boundary).to(dtype=selector.dtype)).all().item())
            or not bool((hidden == sigma * noise + (1.0 - sigma) * clean).all().item())
            or not bool((target == noise - clean).all().item())
            or any(
                not bool(tensor.isfinite().all().item())
                for name, tensor in tensors.items()
                if name != "transformer_selector"
            )
        ):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("wan_training_batch_invalid") from None


def validate_wan_batch_bytes(
    manifest_bytes: bytes, payload: bytes, *, torch_module: Any | None = None
) -> dict[str, Any]:
    """Independently validate exact serialized bytes, then decode and verify arithmetic."""
    try:
        parsed_manifest = json.loads(manifest_bytes)
        if canonical_json(parsed_manifest).encode() != manifest_bytes:
            raise ValueError
        manifest = validate_wan_batch_envelope(manifest_bytes, payload)
        batch_size = cast(int, manifest["batch_size"])
        torch_module = importlib.import_module("torch") if torch_module is None else torch_module
        safetensors = importlib.import_module("safetensors.torch")
        tensors = cast(dict[str, Any], safetensors.load(payload))
        schedule = cast(dict[str, int], manifest["noise_schedule"])
        validate_loaded_wan_batch(
            tensors,
            batch_size=batch_size,
            timestep_min=schedule["timestep_min"],
            timestep_max=schedule["timestep_max"],
            boundary=schedule["timestep_boundary"],
            in_channels=int(tensors["clean_latents"].shape[1]),
        )
        return tensors
    except (ImportError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("wan_training_batch_invalid") from None


def validate_wan_batch_envelope(manifest_bytes: bytes, payload: bytes) -> dict[str, object]:
    """Validate canonical manifest and tensor byte inventory without ML imports."""
    try:
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "sha256",
            "batch_size",
            "tensors",
            "objective",
            "noise_schedule",
        }:
            raise ValueError
        batch_size = manifest["batch_size"]
        schedule = manifest["noise_schedule"]
        if (
            type(batch_size) is not int
            or manifest["schema_version"] != WAN_BATCH_SCHEMA
            or not isinstance(schedule, dict)
            or set(schedule)
            != {"timestep_min", "timestep_max", "timestep_boundary", "selector_rule"}
        ):
            raise ValueError
        rebuilt = build_wan_batch_manifest(
            payload,
            batch_size=batch_size,
            timestep_min=schedule["timestep_min"],
            timestep_max=schedule["timestep_max"],
            timestep_boundary=schedule["timestep_boundary"],
        )
        if json.loads(rebuilt) != manifest:
            raise ValueError
        return cast(dict[str, object], manifest)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("wan_training_batch_invalid") from None
