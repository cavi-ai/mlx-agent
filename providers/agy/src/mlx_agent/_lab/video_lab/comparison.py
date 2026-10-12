from typing import Any

from pydantic import ValidationError

from video_lab.domain import ExperimentDraft, FrozenModel, Manifest

_MISSING = object()


class ComparisonError(ValueError):
    pass


class ControlledPair(FrozenModel):
    variant_a: Manifest
    variant_b: Manifest
    variable_path: str
    changed_paths: tuple[str, ...]


def changed_paths(left: Any, right: Any, prefix: str = "") -> tuple[str, ...]:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        return tuple(
            path
            for key in keys
            for path in changed_paths(
                left[key] if key in left else _MISSING,
                right[key] if key in right else _MISSING,
                f"{prefix}.{key}".strip("."),
            )
        )
    if isinstance(left, list) and isinstance(right, list):
        paths: list[str] = []
        for index in range(max(len(left), len(right))):
            a = left[index] if index < len(left) else _MISSING
            b = right[index] if index < len(right) else _MISSING
            paths.extend(changed_paths(a, b, f"{prefix}.{index}"))
        return tuple(paths)
    return () if left == right else (prefix,)


def _replace_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: Any = payload
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            raise ComparisonError(f"unknown variable path: {path}")
        cursor = cursor[part]
    if not isinstance(cursor, dict) or parts[-1] not in cursor:
        raise ComparisonError(f"unknown variable path: {path}")
    cursor[parts[-1]] = value


def build_controlled_pair(draft: ExperimentDraft) -> ControlledPair:
    payload = draft.baseline.model_dump(mode="json")
    _replace_path(payload, draft.variable_path, draft.alternate_value)
    try:
        alternate = Manifest.model_validate(payload)
    except ValidationError as error:
        raise ComparisonError(str(error)) from error
    paths = changed_paths(draft.baseline.model_dump(mode="json"), alternate.model_dump(mode="json"))
    if paths != (draft.variable_path,):
        raise ComparisonError("alternate value must change exactly one field")
    return ControlledPair(
        variant_a=draft.baseline,
        variant_b=alternate,
        variable_path=draft.variable_path,
        changed_paths=paths,
    )
