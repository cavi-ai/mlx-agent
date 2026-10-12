from __future__ import annotations

from hashlib import sha256

import pytest
from pydantic import ValidationError

from video_lab.character_models import (
    BodyVisibility,
    CharacterAnnotation,
    CharacterPackDraft,
    CharacterReferenceInput,
    CuratorDisposition,
    Expression,
    Framing,
    HeadAngle,
    Lighting,
    Occlusion,
    WarningResolution,
)
from video_lab.character_pack import (
    CharacterPackNotReady,
    DuplicateWarning,
    character_pack_id,
    evaluate_pack,
    freeze_pack,
)
from video_lab.domain import canonical_json


def reference(
    identifier: str,
    *,
    framing: Framing,
    head_angle: HeadAngle,
    expression: Expression,
    lighting: Lighting = Lighting.ORDINARY,
    disposition: CuratorDisposition = CuratorDisposition.ACTIVE,
    identity_confirmed: bool = True,
    dhash_hex: str | None = None,
    media_type: str = "image/jpeg",
) -> CharacterReferenceInput:
    return CharacterReferenceInput(
        reference_id=identifier,
        image_sha256=f"sha256:{identifier}",
        media_type=media_type,
        dhash_hex=dhash_hex or D_HASHES[int(identifier.removeprefix("ref_")) - 1],
        disposition=disposition,
        annotation=CharacterAnnotation(
            framing=framing,
            head_angle=head_angle,
            body_visibility=(
                BodyVisibility.CLEAR_FACE
                if framing is Framing.PORTRAIT
                else BodyVisibility.FULL_BODY
            ),
            expression=expression,
            lighting=lighting,
            occlusion=Occlusion.NONE,
            identity_confirmed=identity_confirmed,
        ),
    )


D_HASHES = (
    "f2a74de452e6b438",
    "6513270e269e0d37",
    "0c5c7fd0a6a3a450",
    "d23f0824128b2f33",
    "1818e811892f902b",
    "9531985d5d9dc9f8",
    "e8e25d940ed90475",
    "36f675cc81e74ef5",
    "1600a35a099950d8",
    "6b0d549b6f03675a",
    "3d9c172411e20b8f",
    "8d116ece1738f7d9",
    "0f21ddb66cad4a26",
    "90c192cfd3ac94af",
    "f28c105d1fb17c23",
    "a170b33839263059",
    "953f48f1a09f76b5",
    "0fd630f1f29d0da9",
    "95e60af593bd04cf",
    "0cb1e29c658cda14",
    "3898d190f9ebdacc",
    "8e81973e0becd7b0",
)


def complete_references(count: int) -> tuple[CharacterReferenceInput, ...]:
    return tuple(
        reference(
            f"ref_{index + 1}",
            framing=Framing.PORTRAIT if index % 2 == 0 else Framing.FULL_BODY,
            head_angle=(HeadAngle.FRONT, HeadAngle.THREE_QUARTER, HeadAngle.PROFILE)[index % 3],
            expression=(Expression.NEUTRAL, Expression.SMILE, Expression.SERIOUS)[index % 3],
            lighting=Lighting.DIRECTIONAL if index % 4 == 0 else Lighting.ORDINARY,
        )
        for index in range(count)
    )


def ready_draft(
    *, references: tuple[CharacterReferenceInput, ...] | None = None
) -> CharacterPackDraft:
    return CharacterPackDraft(
        character_id="char_lyra",
        references=references or complete_references(12),
    )


def test_pack_requires_the_complete_v1_coverage_profile() -> None:
    draft = ready_draft(
        references=(
            reference(
                "ref_1",
                framing=Framing.PORTRAIT,
                head_angle=HeadAngle.FRONT,
                expression=Expression.NEUTRAL,
            ),
        )
    )

    coverage = evaluate_pack(draft)

    assert not coverage.ready
    assert "head_angle:profile" in coverage.missing
    assert "framing:full_body>=3" in coverage.missing
    assert "reference_count>=12" in coverage.missing


def test_pack_id_is_content_addressed_and_order_independent() -> None:
    draft_ab = ready_draft()
    draft_ba = ready_draft(references=tuple(reversed(draft_ab.references)))

    assert character_pack_id(draft_ab) == character_pack_id(draft_ba)
    assert freeze_pack(draft_ab) == freeze_pack(draft_ab)
    assert tuple(reference.reference_id for reference in freeze_pack(draft_ba).references) == (
        *sorted(f"ref_{index}" for index in range(1, 13)),
    )


def test_frozen_pack_binds_duplicate_detection_provenance_into_identity() -> None:
    frozen = freeze_pack(ready_draft())
    manifest = frozen.model_dump(mode="json")

    assert manifest.get("duplicate_hash_algorithm_version") == ("dhash64-exif-transpose-lanczos.v1")
    assert manifest.get("near_duplicate_threshold_inclusive") == 5

    legacy_payload = dict(manifest)
    legacy_payload.pop("pack_id")
    legacy_payload.pop("duplicate_hash_algorithm_version")
    legacy_payload.pop("near_duplicate_threshold_inclusive")
    legacy_id = (
        "character_pack_" + sha256(canonical_json(legacy_payload).encode("utf-8")).hexdigest()
    )
    assert frozen.pack_id != legacy_id


def test_pack_enforces_count_bounds_and_excludes_rejected_references() -> None:
    draft = ready_draft(
        references=ready_draft().references
        + (
            reference(
                "ref_22",
                framing=Framing.PORTRAIT,
                head_angle=HeadAngle.FRONT,
                expression=Expression.NEUTRAL,
                disposition=CuratorDisposition.REJECTED,
            ),
        )
    )

    assert evaluate_pack(draft).ready

    eleven = ready_draft(references=complete_references(11))
    twenty = ready_draft(references=complete_references(20))
    twenty_one = ready_draft(references=complete_references(21))

    assert "reference_count>=12" in evaluate_pack(eleven).missing
    assert evaluate_pack(twenty).ready
    assert "reference_count<=20" in evaluate_pack(twenty_one).missing


def test_pack_requires_two_clear_faces_and_confirmed_active_references() -> None:
    references = list(ready_draft().references)
    references[0] = references[0].model_copy(
        update={
            "annotation": references[0].annotation.model_copy(update={"identity_confirmed": False})
        }
    )
    for index, reference_input in enumerate(references):
        if index > 0 and reference_input.annotation.body_visibility is BodyVisibility.CLEAR_FACE:
            references[index] = reference_input.model_copy(
                update={
                    "annotation": reference_input.annotation.model_copy(
                        update={"occlusion": Occlusion.PARTIAL}
                    )
                }
            )

    coverage = evaluate_pack(ready_draft(references=tuple(references)))

    assert "identity_confirmation" in coverage.missing
    assert "clear_faces>=2" in coverage.missing


def test_pack_blocks_unresolved_near_duplicate_warnings() -> None:
    references = list(ready_draft().references)
    references[0] = references[0].model_copy(update={"dhash_hex": "0000000000000000"})
    references[1] = references[1].model_copy(update={"dhash_hex": "000000000000001f"})
    draft = ready_draft(references=tuple(references))

    coverage = evaluate_pack(draft)

    assert coverage.duplicate_warnings == (
        DuplicateWarning(
            left_reference_id="ref_1",
            right_reference_id="ref_2",
            distance=5,
            resolution=WarningResolution.UNRESOLVED,
        ),
    )
    assert "duplicate_warning:unresolved" in coverage.missing
    with pytest.raises(CharacterPackNotReady, match="duplicate_warning:unresolved"):
        freeze_pack(draft)


def test_resolved_near_duplicate_warning_allows_freezing() -> None:
    references = list(ready_draft().references)
    references[0] = references[0].model_copy(update={"dhash_hex": "0000000000000000"})
    references[1] = references[1].model_copy(update={"dhash_hex": "000000000000001f"})
    draft = ready_draft(
        references=tuple(references),
    ).model_copy(
        update={
            "duplicate_warnings": (
                DuplicateWarning(
                    left_reference_id="ref_1",
                    right_reference_id="ref_2",
                    distance=5,
                    resolution=WarningResolution.ACCEPTED,
                ),
            )
        }
    )

    assert freeze_pack(draft).pack_id == character_pack_id(draft)


def test_successor_identity_changes_the_content_addressed_pack_id() -> None:
    original = ready_draft()
    successor = original.model_copy(update={"supersedes_pack_id": character_pack_id(original)})

    assert character_pack_id(successor) != character_pack_id(original)
    assert freeze_pack(successor).supersedes_pack_id == character_pack_id(original)


def test_character_reference_requires_supported_media_type() -> None:
    supported = reference(
        "ref_1",
        framing=Framing.PORTRAIT,
        head_angle=HeadAngle.FRONT,
        expression=Expression.NEUTRAL,
        media_type="image/png",
    )

    assert supported.media_type == "image/png"
    with pytest.raises(ValidationError, match="image/jpeg"):
        reference(
            "ref_1",
            framing=Framing.PORTRAIT,
            head_angle=HeadAngle.FRONT,
            expression=Expression.NEUTRAL,
            media_type="image/gif",
        )


@pytest.mark.parametrize(
    "dhash_hex",
    (
        "+000000000000000",
        "_000000000000000",
        " 000000000000000",
        "0x00000000000000",
        "f" * 15,
        "f" * 17,
    ),
)
def test_character_reference_rejects_noncanonical_dhash_hex(dhash_hex: str) -> None:
    with pytest.raises(ValidationError):
        reference(
            "ref_1",
            framing=Framing.PORTRAIT,
            head_angle=HeadAngle.FRONT,
            expression=Expression.NEUTRAL,
            dhash_hex=dhash_hex,
        )


def test_character_reference_normalizes_uppercase_dhash_before_validation() -> None:
    normalized = reference(
        "ref_1",
        framing=Framing.PORTRAIT,
        head_angle=HeadAngle.FRONT,
        expression=Expression.NEUTRAL,
        dhash_hex="ABCDEF0123456789",
    )

    assert normalized.dhash_hex == "abcdef0123456789"


def test_pack_id_normalizes_warning_endpoint_order_and_discards_stale_warnings() -> None:
    references = list(complete_references(12))
    references[0] = references[0].model_copy(update={"dhash_hex": "0000000000000000"})
    references[1] = references[1].model_copy(update={"dhash_hex": "000000000000001f"})
    base = ready_draft(references=tuple(references))
    forward = base.model_copy(
        update={
            "duplicate_warnings": (
                DuplicateWarning(
                    left_reference_id="ref_1",
                    right_reference_id="ref_2",
                    distance=5,
                    resolution=WarningResolution.ACCEPTED,
                ),
            )
        }
    )
    reversed_endpoints = base.model_copy(
        update={
            "duplicate_warnings": (
                DuplicateWarning(
                    left_reference_id="ref_2",
                    right_reference_id="ref_1",
                    distance=5,
                    resolution=WarningResolution.ACCEPTED,
                ),
            )
        }
    )
    stale = forward.model_copy(
        update={
            "duplicate_warnings": forward.duplicate_warnings
            + (
                DuplicateWarning(
                    left_reference_id="ref_3",
                    right_reference_id="ref_4",
                    distance=0,
                    resolution=WarningResolution.DISMISSED,
                ),
            )
        }
    )

    assert character_pack_id(forward) == character_pack_id(reversed_endpoints)
    assert character_pack_id(forward) == character_pack_id(stale)
    assert freeze_pack(forward).pack_id == freeze_pack(stale).pack_id


def test_character_models_are_immutable_and_reject_extra_fields() -> None:
    draft = ready_draft()

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CharacterPackDraft(character_id="char_lyra", references=(), ignored=True)
    with pytest.raises(ValidationError, match="Instance is frozen"):
        draft.character_id = "char_other"
