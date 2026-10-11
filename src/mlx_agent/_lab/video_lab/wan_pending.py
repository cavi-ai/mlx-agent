"""Canonical validation for durable Wan pending-gradient records."""

from __future__ import annotations

import base64
import hashlib
import math
import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal, cast

_DTYPE_WIDTHS = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
_DEVICES = {"cpu", "mps"}


@dataclass(frozen=True, slots=True)
class TrainableParameterSpec:
    physical_name: str
    logical_target: str
    role: Literal["A", "B"]
    shape: tuple[int, ...]
    dtype: str
    device: str


def validate_trainable_parameter_specs(
    specs: Collection[TrainableParameterSpec], *, declared_targets: Collection[str]
) -> tuple[TrainableParameterSpec, ...]:
    owned = tuple(specs)
    declared = tuple(declared_targets)
    if (
        not owned
        or not declared
        or len(set(declared)) != len(declared)
        or any(type(target) is not str or not target for target in declared)
        or any(type(item) is not TrainableParameterSpec for item in owned)
        or len({item.physical_name for item in owned}) != len(owned)
        or {item.logical_target for item in owned} != set(declared)
    ):
        raise ValueError
    for item in owned:
        if (
            not item.physical_name
            or item.role not in {"A", "B"}
            or not item.shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in item.shape)
            or item.dtype not in _DTYPE_WIDTHS
            or item.device not in _DEVICES
        ):
            raise ValueError
    for target in declared:
        bound = tuple(item for item in owned if item.logical_target == target)
        if len(bound) != 2 or {item.role for item in bound} != {"A", "B"}:
            raise ValueError
    return tuple(sorted(owned, key=lambda item: item.physical_name))


def trainable_parameter_spec_records(
    specs: Collection[TrainableParameterSpec],
) -> list[dict[str, object]]:
    return [
        {
            "physical_name": item.physical_name,
            "logical_target": item.logical_target,
            "role": item.role,
            "shape": list(item.shape),
            "dtype": item.dtype,
            "device": item.device,
        }
        for item in sorted(specs, key=lambda value: value.physical_name)
    ]


def trainable_parameter_specs_from_records(
    records: object,
) -> tuple[TrainableParameterSpec, ...]:
    if not isinstance(records, list):
        raise ValueError
    specs: list[TrainableParameterSpec] = []
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "physical_name", "logical_target", "role", "shape", "dtype", "device"
        }:
            raise ValueError
        shape = raw.get("shape")
        role = raw.get("role")
        if (
            type(raw.get("physical_name")) is not str
            or type(raw.get("logical_target")) is not str
            or role not in {"A", "B"}
            or not isinstance(shape, list)
            or any(type(item) is not int for item in shape)
            or type(raw.get("dtype")) is not str
            or type(raw.get("device")) is not str
        ):
            raise ValueError
        specs.append(
            TrainableParameterSpec(
                physical_name=cast(str, raw["physical_name"]),
                logical_target=cast(str, raw["logical_target"]),
                role=cast(Literal["A", "B"], role),
                shape=tuple(cast(list[int], shape)),
                dtype=cast(str, raw["dtype"]),
                device=cast(str, raw["device"]),
            )
        )
    return tuple(specs)


def validate_pending_gradients(
    value: object,
    position: int,
    *,
    expected_parameters: Collection[str],
    expected_specs: Mapping[str, tuple[tuple[int, ...], str, str]] | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "accumulation_position",
        "parameters",
    }:
        raise ValueError
    parameters = value.get("parameters")
    expected = set(expected_parameters)
    if (
        type(position) is not int
        or position < 0
        or value.get("accumulation_position") != position
        or not isinstance(parameters, Mapping)
        or set(parameters) != (expected if position else set())
        or (expected_specs is not None and set(expected_specs) != expected)
    ):
        raise ValueError
    for name in expected if position else ():
        raw = parameters[name]
        if not isinstance(raw, Mapping) or set(raw) != {
            "digest",
            "serialized_b64",
            "shape",
            "dtype",
            "device",
        }:
            raise ValueError
        encoded = raw.get("serialized_b64")
        if type(encoded) is not str:
            raise ValueError
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError):
            raise ValueError from None
        shape_value = raw.get("shape")
        dtype = raw.get("dtype")
        device = raw.get("device")
        if (
            not isinstance(shape_value, list)
            or not shape_value
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape_value)
            or type(dtype) is not str
            or dtype not in _DTYPE_WIDTHS
            or type(device) is not str
            or device not in _DEVICES
        ):
            raise ValueError
        shape = tuple(cast(list[int], shape_value))
        elements = math.prod(shape)
        if (
            elements > sys.maxsize // _DTYPE_WIDTHS[dtype]
            or len(payload) != elements * _DTYPE_WIDTHS[dtype]
            or raw.get("digest") != "sha256:" + hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError
        if expected_specs is not None and (shape, dtype, device) != expected_specs[name]:
            raise ValueError
    return value
