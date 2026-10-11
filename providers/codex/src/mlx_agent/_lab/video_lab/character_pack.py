from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from typing import Final, Literal

from video_lab.character_images import (
    DUPLICATE_HASH_ALGORITHM_VERSION,
    hamming_distance,
    is_near_duplicate,
)
from video_lab.character_models import (
    BodyVisibility,
    CharacterPackDraft,
    CharacterReferenceInput,
    CuratorDisposition,
    DuplicateWarning,
    Expression,
    Framing,
    HeadAngle,
    Lighting,
    Occlusion,
    WarningResolution,
)
from video_lab.domain import FrozenModel, canonical_json

MIN_REFERENCE_COUNT = 12
MAX_REFERENCE_COUNT = 20
NEAR_DUPLICATE_THRESHOLD: Final[Literal[5]] = 5


class CharacterCoverage(FrozenModel):
    ready: bool
    missing: tuple[str, ...]
    duplicate_warnings: tuple[DuplicateWarning, ...]


class CharacterPack(FrozenModel):
    pack_id: str
    character_id: str
    duplicate_hash_algorithm_version: Literal["dhash64-exif-transpose-lanczos.v1"]
    near_duplicate_threshold_inclusive: Literal[5]
    references: tuple[CharacterReferenceInput, ...]
    duplicate_warnings: tuple[DuplicateWarning, ...]
    supersedes_pack_id: str | None


class CharacterPackNotReady(ValueError):
    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        self.reason_codes = reason_codes
        super().__init__(",".join(reason_codes))


def evaluate_pack(draft: CharacterPackDraft) -> CharacterCoverage:
    active_references = tuple(
        sorted(
            (
                reference
                for reference in draft.references
                if reference.disposition is CuratorDisposition.ACTIVE
            ),
            key=lambda reference: reference.reference_id,
        )
    )
    warnings = _duplicate_warnings(active_references, draft.duplicate_warnings)
    missing = _missing_coverage(active_references, warnings)
    return CharacterCoverage(
        ready=not missing,
        missing=tuple(missing),
        duplicate_warnings=warnings,
    )


def character_pack_id(draft: CharacterPackDraft) -> str:
    coverage = evaluate_pack(draft)
    payload = CharacterPack(
        pack_id="",
        character_id=draft.character_id,
        duplicate_hash_algorithm_version=DUPLICATE_HASH_ALGORITHM_VERSION,
        near_duplicate_threshold_inclusive=NEAR_DUPLICATE_THRESHOLD,
        references=tuple(sorted(draft.references, key=lambda reference: reference.reference_id)),
        duplicate_warnings=coverage.duplicate_warnings,
        supersedes_pack_id=draft.supersedes_pack_id,
    ).model_dump(mode="json", exclude={"pack_id"})
    digest = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"character_pack_{digest}"


def freeze_pack(draft: CharacterPackDraft) -> CharacterPack:
    coverage = evaluate_pack(draft)
    if not coverage.ready:
        raise CharacterPackNotReady(coverage.missing)
    return CharacterPack(
        pack_id=character_pack_id(draft),
        character_id=draft.character_id,
        duplicate_hash_algorithm_version=DUPLICATE_HASH_ALGORITHM_VERSION,
        near_duplicate_threshold_inclusive=NEAR_DUPLICATE_THRESHOLD,
        references=tuple(sorted(draft.references, key=lambda reference: reference.reference_id)),
        duplicate_warnings=coverage.duplicate_warnings,
        supersedes_pack_id=draft.supersedes_pack_id,
    )


def _duplicate_warnings(
    references: tuple[CharacterReferenceInput, ...],
    supplied_warnings: tuple[DuplicateWarning, ...],
) -> tuple[DuplicateWarning, ...]:
    resolutions = {
        _warning_key(warning.left_reference_id, warning.right_reference_id): warning.resolution
        for warning in supplied_warnings
    }
    warnings: list[DuplicateWarning] = []
    for left, right in combinations(references, 2):
        distance = hamming_distance(int(left.dhash_hex, 16), int(right.dhash_hex, 16))
        if is_near_duplicate(
            int(left.dhash_hex, 16), int(right.dhash_hex, 16), threshold=NEAR_DUPLICATE_THRESHOLD
        ):
            left_reference_id, right_reference_id = _warning_key(
                left.reference_id, right.reference_id
            )
            warnings.append(
                DuplicateWarning(
                    left_reference_id=left_reference_id,
                    right_reference_id=right_reference_id,
                    distance=distance,
                    resolution=resolutions.get(
                        (left_reference_id, right_reference_id), WarningResolution.UNRESOLVED
                    ),
                )
            )
    return tuple(warnings)


def _missing_coverage(
    references: tuple[CharacterReferenceInput, ...], warnings: tuple[DuplicateWarning, ...]
) -> list[str]:
    missing: list[str] = []
    if len(references) < MIN_REFERENCE_COUNT:
        missing.append(f"reference_count>={MIN_REFERENCE_COUNT}")
    if len(references) > MAX_REFERENCE_COUNT:
        missing.append(f"reference_count<={MAX_REFERENCE_COUNT}")
    for head_angle in HeadAngle:
        if not any(reference.annotation.head_angle is head_angle for reference in references):
            missing.append(f"head_angle:{head_angle.value}")
    if sum(reference.annotation.framing is Framing.PORTRAIT for reference in references) < 3:
        missing.append("framing:portrait>=3")
    if sum(reference.annotation.framing is Framing.FULL_BODY for reference in references) < 3:
        missing.append("framing:full_body>=3")
    if not any(reference.annotation.expression is Expression.NEUTRAL for reference in references):
        missing.append("expression:neutral")
    non_neutral_expressions = {
        reference.annotation.expression
        for reference in references
        if reference.annotation.expression is not Expression.NEUTRAL
    }
    if len(non_neutral_expressions) < 2:
        missing.append("expression:non_neutral>=2")
    for lighting in Lighting:
        if not any(reference.annotation.lighting is lighting for reference in references):
            missing.append(f"lighting:{lighting.value}")
    clear_faces = sum(
        reference.annotation.body_visibility is BodyVisibility.CLEAR_FACE
        and reference.annotation.occlusion is Occlusion.NONE
        for reference in references
    )
    if clear_faces < 2:
        missing.append("clear_faces>=2")
    if any(not reference.annotation.identity_confirmed for reference in references):
        missing.append("identity_confirmation")
    if any(warning.resolution is WarningResolution.UNRESOLVED for warning in warnings):
        missing.append("duplicate_warning:unresolved")
    return missing


def _warning_key(left_reference_id: str, right_reference_id: str) -> tuple[str, str]:
    first, second = sorted((left_reference_id, right_reference_id))
    return first, second
