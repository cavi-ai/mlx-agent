from __future__ import annotations

import pytest
from pydantic import ValidationError

import video_lab
from video_lab.curated_ids import curated_dataset_digest, structured_caption_digest
from video_lab.curated_models import (
    CuratedDatasetManifestV1,
    CuratedFrameV1,
    CuratedSequenceV1,
    DatasetSplitPolicyV1,
    SequenceTransformRecipeV1,
    StructuredCaptionV1,
    render_caption_v1,
)

SHA256 = "a" * 64


def frame(*, frame_id: str = "frame_0001", index: int = 0) -> CuratedFrameV1:
    return CuratedFrameV1(
        frame_id=frame_id,
        frame_index=index,
        timestamp_numerator=index,
        timestamp_denominator=24,
        source_path=f"frames/{frame_id}.png",
        source_digest=SHA256,
        pixel_digest=SHA256,
        review_state="accepted",
        reviewer_evidence=("review_001",),
    )


def sequence(
    *, split: str = "train", frames: tuple[CuratedFrameV1, ...] | None = None
) -> CuratedSequenceV1:
    return CuratedSequenceV1(
        sequence_id="sequence_001",
        identity_id="mirza_char",
        source_lineage="capture_001",
        split=split,
        caption=StructuredCaptionV1(
            character_token="mirza_char",
            immutable_attributes=("blue eyes", "black hair"),
            action="walking",
        ),
        frames=frames if frames is not None else (frame(),),
    )


def manifest(*, sequences: tuple[CuratedSequenceV1, ...] | None = None) -> CuratedDatasetManifestV1:
    return CuratedDatasetManifestV1(
        dataset_id="dataset_001",
        sequences=sequences if sequences is not None else (sequence(),),
        split_policy=DatasetSplitPolicyV1(
            policy_version="split_policy_v1",
            seed=7,
            train_numerator=8,
            validation_numerator=1,
            holdout_numerator=1,
            denominator=10,
        ),
        transform_recipe=SequenceTransformRecipeV1(
            recipe_id="recipe_001",
            output_width=512,
            output_height=512,
            output_frame_count=16,
            temporal_stride_numerator=1,
            temporal_stride_denominator=1,
            interpolation="bicubic",
            padding_color=(0, 0, 0),
            seed=7,
        ),
    )


def invalid_duplicate_manifest() -> dict[str, object]:
    first = sequence(frames=(frame(),))
    second = CuratedSequenceV1(
        sequence_id="sequence_002",
        identity_id="mirza_char",
        source_lineage="capture_002",
        split="validation",
        caption=first.caption,
        frames=(frame(frame_id="frame_0001", index=0),),
    )
    return manifest(sequences=(first, second)).model_dump(mode="python")


def test_caption_rendering_is_order_independent_and_identity_explicit() -> None:
    left = StructuredCaptionV1(
        character_token="mirza_char",
        immutable_attributes=("blue eyes", "black hair"),
        action="walking",
    )
    right = StructuredCaptionV1(
        character_token="mirza_char",
        immutable_attributes=("black hair", "blue eyes"),
        action="walking",
    )

    assert render_caption_v1(left) == render_caption_v1(right)
    assert structured_caption_digest(left) == structured_caption_digest(right)
    assert "token: mirza_char" in render_caption_v1(left)
    assert left.immutable_attributes == ("black hair", "blue eyes")


def test_manifest_rejects_duplicate_frame_identity_and_split_values() -> None:
    with pytest.raises(ValidationError):
        CuratedDatasetManifestV1.model_validate(invalid_duplicate_manifest())


def test_models_reject_extra_keys_and_invalid_content() -> None:
    with pytest.raises(ValidationError):
        StructuredCaptionV1.model_validate(
            {
                "character_token": "mirza_char",
                "immutable_attributes": ("blue eyes",),
                "action": "walking",
                "unknown": "nope",
            }
        )
    with pytest.raises(ValidationError):
        CuratedFrameV1.model_validate({**frame().model_dump(), "source_digest": "not-a-digest"})
    with pytest.raises(ValidationError):
        CuratedFrameV1.model_validate({**frame().model_dump(), "reviewer_evidence": ()})
    with pytest.raises(ValidationError):
        CuratedSequenceV1.model_validate({**sequence().model_dump(), "split": "test"})


def test_caption_normalizes_unicode_and_duplicate_set_like_fields() -> None:
    caption = StructuredCaptionV1(
        character_token="mirza_char",
        immutable_attributes=("caf\u00e9", "cafe\u0301", "blue eyes"),
        clothing=("coat", "coat"),
        action="walking",
        omissions=("hat", "hat"),
    )

    assert caption.immutable_attributes == ("blue eyes", "caf\u00e9")
    assert caption.clothing == ("coat",)
    assert caption.omissions == ("hat",)


def test_sequence_rejects_out_of_order_rational_frames() -> None:
    with pytest.raises(ValidationError):
        sequence(frames=(frame(frame_id="frame_0002", index=1), frame(index=0)))


def test_manifest_digest_rejects_permutation_conflicts() -> None:
    first = sequence()
    second = CuratedSequenceV1(
        sequence_id="sequence_002",
        identity_id="mirza_char",
        source_lineage="capture_002",
        split="validation",
        caption=first.caption,
        frames=(frame(frame_id="frame_0002"),),
    )
    left = manifest(sequences=(first, second))

    assert len(curated_dataset_digest(left)) == 64
    with pytest.raises(ValidationError):
        manifest(sequences=(second, first))


def test_model_copy_revalidates_updates_and_preserves_valid_copies() -> None:
    original = frame()

    copied = original.model_copy(update={"frame_id": "frame_0002"})

    assert copied.frame_id == "frame_0002"
    assert copied.source_digest == SHA256
    with pytest.raises(ValidationError):
        original.model_copy(update={"source_digest": "invalid"})
    with pytest.raises(ValidationError):
        original.model_copy(update={"reviewer_evidence": ()})


def test_frozen_models_reject_attribute_mutation() -> None:
    candidate = frame()

    with pytest.raises(ValidationError):
        candidate.frame_id = "INVALID"


def test_sequence_rejects_caption_token_that_differs_from_identity() -> None:
    with pytest.raises(ValidationError):
        CuratedSequenceV1(
            sequence_id="sequence_001",
            identity_id="mirza_char",
            source_lineage="capture_001",
            split="train",
            caption=StructuredCaptionV1(
                character_token="other_char",
                immutable_attributes=("blue eyes",),
                action="walking",
            ),
            frames=(frame(),),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_index", True),
        ("timestamp_numerator", "0"),
        ("reviewer_evidence", ["review_001"]),
    ],
)
def test_models_reject_non_strict_input_types(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        CuratedFrameV1.model_validate({**frame().model_dump(), field: value})


def test_caption_renderer_uses_all_declared_sections_in_fixed_order() -> None:
    caption = StructuredCaptionV1(
        character_token="mirza_char",
        immutable_attributes=("blue eyes",),
        clothing=("black coat", "silver ring"),
        action="walking",
        pose="full body",
        expression="focused",
        environment="city street",
        lighting="soft daylight",
        camera="medium shot",
        style="cinematic",
        omissions=("hat",),
    )

    assert render_caption_v1(caption) == (
        "token: mirza_char; immutable identity: blue eyes; "
        "clothing: black coat, silver ring; action/pose: walking; full body; "
        "expression: focused; environment: city street; lighting: soft daylight; "
        "camera: medium shot; style: cinematic; omissions: hat"
    )


def test_caption_renderer_omits_explicitly_empty_optional_sections() -> None:
    caption = StructuredCaptionV1(
        character_token="mirza_char",
        immutable_attributes=("blue eyes",),
        action="walking",
    )

    assert render_caption_v1(caption) == (
        "token: mirza_char; immutable identity: blue eyes; action/pose: walking"
    )


def test_caption_rejects_unknown_accessories_field_without_ambiguous_rendering() -> None:
    payload = StructuredCaptionV1(
        character_token="mirza_char",
        immutable_attributes=("blue eyes",),
        clothing=("black coat", "silver ring"),
        action="walking",
    ).model_dump()

    with pytest.raises(ValidationError):
        StructuredCaptionV1.model_validate({**payload, "accessories": ("silver ring",)})


def test_clothing_normalizes_duplicate_accessory_descriptors_once() -> None:
    caption = StructuredCaptionV1(
        character_token="mirza_char",
        immutable_attributes=("blue eyes",),
        clothing=("silver ring", "black coat", "silver ring"),
        action="walking",
    )

    assert caption.clothing == ("black coat", "silver ring")
    assert render_caption_v1(caption) == (
        "token: mirza_char; immutable identity: blue eyes; "
        "clothing: black coat, silver ring; action/pose: walking"
    )


def test_public_package_exports_curated_contracts() -> None:
    assert video_lab.StructuredCaptionV1 is StructuredCaptionV1
    assert video_lab.CuratedFrameV1 is CuratedFrameV1
    assert video_lab.CuratedSequenceV1 is CuratedSequenceV1
    assert video_lab.CuratedDatasetManifestV1 is CuratedDatasetManifestV1
    assert video_lab.DatasetSplitPolicyV1 is DatasetSplitPolicyV1
    assert video_lab.SequenceTransformRecipeV1 is SequenceTransformRecipeV1
    assert video_lab.curated_dataset_digest is curated_dataset_digest
    assert video_lab.structured_caption_digest is structured_caption_digest
    assert video_lab.render_caption_v1 is render_caption_v1
