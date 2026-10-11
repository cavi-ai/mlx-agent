"""Path-free, preregistered continuity evidence for Wan branch state."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, StrictBool, StrictFloat, field_validator

from video_lab.domain import FrozenModel, canonical_json
from video_lab.trainers.wan_production_models import (
    WanBranchRecipeV1,
    wan_branch_recipe_digest,
)
from video_lab.training_models import Sha256Digest


class WanContinuityComparisonV1(FrozenModel):
    """Redacted numerical evidence for an uninterrupted/restored comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["wan-branch-continuity-comparison.v1"] = (
        "wan-branch-continuity-comparison.v1"
    )
    recipe_digest: Sha256Digest
    tolerance_digest: Sha256Digest
    accepted: StrictBool
    max_absolute_error: StrictFloat = Field(ge=0)
    max_relative_error: StrictFloat = Field(ge=0)
    failing_tensor_names: tuple[str, ...]

    @field_validator("max_absolute_error", "max_relative_error")
    @classmethod
    def finite_errors(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("wan_continuity_nonfinite")
        return value

    @field_validator("failing_tensor_names")
    @classmethod
    def canonical_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))) or any(not name for name in value):
            raise ValueError("wan_continuity_structure_invalid")
        return value


def _tensor(value: object) -> tuple[tuple[int, ...], str, tuple[float, ...]]:
    candidate: Any = value
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
        dtype = str(candidate.dtype)
        shape = tuple(int(size) for size in candidate.shape)
    if hasattr(candidate, "to"):
        candidate = candidate.to("cpu")
    if shape is not None and hasattr(candidate, "reshape") and hasattr(candidate, "tolist"):
        candidate = candidate.reshape(-1).tolist()
    if not isinstance(candidate, Iterable) or isinstance(candidate, (str, bytes, Mapping)):
        raise ValueError("wan_continuity_structure_invalid")
    if shape is None:
        def inspect(sequence: object) -> tuple[tuple[int, ...], str, list[float]]:
            if type(sequence) in (int, float):
                number = float(cast(int | float, sequence))
                if not math.isfinite(number):
                    raise ValueError("wan_continuity_nonfinite")
                return (), "float64" if type(sequence) is float else "int64", [number]
            if not isinstance(sequence, (tuple, list)) or not sequence:
                raise ValueError("wan_continuity_structure_invalid")
            children = [inspect(item) for item in sequence]
            if any(child[:2] != children[0][:2] for child in children[1:]):
                raise ValueError("wan_continuity_structure_invalid")
            return (len(children), *children[0][0]), children[0][1], [
                item for child in children for item in child[2]
            ]

        shape, dtype, result = inspect(candidate)
    else:
        result = []
        for item in candidate:
            if type(item) not in (int, float):
                raise ValueError("wan_continuity_structure_invalid")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("wan_continuity_nonfinite")
            result.append(number)
    if not result:
        raise ValueError("wan_continuity_structure_invalid")
    assert dtype is not None
    return shape, dtype, tuple(result)


def compare_wan_branch_continuity(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    tolerance: WanBranchRecipeV1,
) -> WanContinuityComparisonV1:
    """Compare exact state names/shapes with the recipe's immutable tolerance."""

    if type(tolerance) is not WanBranchRecipeV1 or set(reference) != set(candidate):
        raise ValueError("wan_continuity_structure_invalid")
    if not reference or any(type(name) is not str or not name for name in reference):
        raise ValueError("wan_continuity_structure_invalid")
    atol, rtol = tolerance.tolerance.atol, tolerance.tolerance.rtol
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failed: list[str] = []
    for name in sorted(reference):
        expected_shape, expected_dtype, expected = _tensor(reference[name])
        observed_shape, observed_dtype, observed = _tensor(candidate[name])
        if (expected_shape, expected_dtype) != (observed_shape, observed_dtype):
            raise ValueError("wan_continuity_structure_invalid")
        tensor_failed = False
        for expected_value, observed_value in zip(expected, observed, strict=True):
            absolute = abs(observed_value - expected_value)
            relative = absolute / max(abs(expected_value), atol)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            if absolute > atol + rtol * abs(expected_value):
                tensor_failed = True
        if tensor_failed:
            failed.append(name)
    recipe_digest = wan_branch_recipe_digest(tolerance)
    tolerance_digest = "sha256:" + hashlib.sha256(
        canonical_json(
            {
                "domain": "wan-branch-continuity-tolerance.v1",
                "recipe_digest": recipe_digest,
                "atol": atol,
                "rtol": rtol,
            }
        ).encode()
    ).hexdigest()
    return WanContinuityComparisonV1(
        recipe_digest=recipe_digest,
        tolerance_digest=tolerance_digest,
        accepted=not failed,
        max_absolute_error=maximum_absolute,
        max_relative_error=maximum_relative,
        failing_tensor_names=tuple(failed),
    )


__all__ = ("WanContinuityComparisonV1", "compare_wan_branch_continuity")
