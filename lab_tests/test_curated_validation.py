from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from video_lab.curated_models import (
    CuratedDatasetManifestV1,
    CuratedFrameV1,
    CuratedSequenceV1,
    DatasetSplitPolicyV1,
    SequenceTransformRecipeV1,
    StructuredCaptionV1,
)
from video_lab.curated_source import DecodedFrame, decoded_pixel_digest
from video_lab.curated_splits import assign_identity_aware_splits
from video_lab.curated_validation import ValidationThresholdsV1, validate_curated_dataset
from video_lab.training_models import CuratedEvidenceReferenceV1


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decoded(array: np.ndarray) -> DecodedFrame:
    pixels = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    frame = DecodedFrame(
        width=array.shape[1],
        height=array.shape[0],
        channels=3,
        bit_depth=8,
        color_space="RGB",
        pixel_digest="0" * 64,
        pixels=pixels,
    )
    return replace(frame, pixel_digest=decoded_pixel_digest(frame))


def _frame(index: int, decoded: DecodedFrame) -> CuratedFrameV1:
    return CuratedFrameV1(
        frame_id=f"frame_{index}",
        frame_index=index,
        timestamp_numerator=index,
        timestamp_denominator=8,
        source_path=f"sequence/{index:04d}.png",
        source_digest=_digest(f"source-{index}".encode()),
        pixel_digest=decoded.pixel_digest,
        review_state="accepted",
        reviewer_evidence=("review_one",),
    )


def _manifest(
    frames: tuple[CuratedFrameV1, ...], *, immutable: tuple[str, ...] = ("blue eyes",)
) -> CuratedDatasetManifestV1:
    caption = StructuredCaptionV1(
        character_token="hero", immutable_attributes=immutable, action="walking"
    )
    sequence = CuratedSequenceV1(
        sequence_id="sequence_a",
        identity_id="hero",
        source_lineage="lineage_a",
        split="train",
        caption=caption,
        frames=frames,
    )
    return CuratedDatasetManifestV1(
        dataset_id="dataset_a",
        sequences=(sequence,),
        split_policy=DatasetSplitPolicyV1(
            policy_version="policy_one",
            seed=7,
            train_numerator=8,
            validation_numerator=1,
            holdout_numerator=1,
            denominator=10,
        ),
        transform_recipe=SequenceTransformRecipeV1(
            recipe_id="recipe_one",
            output_width=8,
            output_height=8,
            output_frame_count=2,
            temporal_stride_numerator=1,
            temporal_stride_denominator=8,
            interpolation="nearest",
            padding_color=(0, 0, 0),
            seed=9,
        ),
    )


def test_validation_reports_stable_quantized_quality_evidence() -> None:
    sharp = np.zeros((8, 8, 3), dtype=np.uint8)
    sharp[::2, ::2] = 255
    duplicate = sharp.copy()
    decoded = (_decoded(sharp), _decoded(duplicate))
    manifest = _manifest(tuple(_frame(i, value) for i, value in enumerate(decoded)))

    report = validate_curated_dataset(
        manifest,
        {f"frame_{i}": value for i, value in enumerate(decoded)},
        ValidationThresholdsV1(
            perceptual_duplicate_hamming_max=1,
            blur_variance_min="1.0000",
            scene_cut_histogram_max="0.9000",
            temporal_motion_max="0.9000",
        ),
    )

    finding = next(item for item in report.frame_findings if item.reason_code == "exact_duplicate")
    assert finding.state == "rejected"
    assert finding.observed_value == "0.0000"
    assert finding.threshold == "0.0000"
    assert finding.frame_id == "frame_1"
    assert finding.source_digest == manifest.sequences[0].frames[1].source_digest
    assert finding.pixel_digest == decoded[1].pixel_digest
    assert report.state == "rejected"


def test_validation_report_produces_verified_split_receipt() -> None:
    decoded = _decoded(np.random.default_rng(9).integers(0, 256, (8, 8, 3), dtype=np.uint8))
    manifest = _manifest((_frame(0, decoded),))
    thresholds = ValidationThresholdsV1(
        blur_variance_min="0",
        scene_cut_histogram_max="1",
        temporal_motion_max="1",
    )

    report = validate_curated_dataset(manifest, {"frame_0": decoded}, thresholds)

    assert report.split_assignment_evidence is not None
    assigned = assign_identity_aware_splits(
        manifest,
        manifest.split_policy,
        decoded_frames={"frame_0": decoded},
        evidence=report.split_assignment_evidence,
        enforce_declared=False,
    )
    assert assigned.sequences[0].sequence_id == "sequence_a"


def test_validation_emits_blur_scene_motion_and_perceptual_reason_codes() -> None:
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    almost_black = black.copy()
    almost_black[0, 0] = 1
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    decoded = (_decoded(black), _decoded(almost_black), _decoded(white))
    manifest = _manifest(tuple(_frame(i, value) for i, value in enumerate(decoded)))

    report = validate_curated_dataset(
        manifest,
        {f"frame_{i}": value for i, value in enumerate(decoded)},
        ValidationThresholdsV1(
            perceptual_duplicate_hamming_max=4,
            blur_variance_min="1",
            scene_cut_histogram_max="0.2",
            temporal_motion_max="0.2",
        ),
    )

    codes = {item.reason_code for item in report.frame_findings}
    assert {
        "perceptual_duplicate",
        "blur_below_minimum",
        "scene_cut",
        "temporal_discontinuity",
    } <= codes
    assert all(len(item.observed_value.partition(".")[2]) == 4 for item in report.frame_findings)


def test_borderline_identity_evidence_requires_human_disposition() -> None:
    generator = np.random.default_rng(17)
    left = _decoded(generator.integers(0, 256, (8, 8, 3), dtype=np.uint8))
    right = _decoded(generator.integers(0, 256, (8, 8, 3), dtype=np.uint8))
    first = _manifest((_frame(0, left),)).sequences[0]
    second = CuratedSequenceV1(
        sequence_id="sequence_b",
        identity_id="hero",
        source_lineage="lineage_b",
        split="validation",
        caption=StructuredCaptionV1(
            character_token="hero", immutable_attributes=("green eyes",), action="walking"
        ),
        frames=(_frame(1, right).model_copy(update={"frame_index": 0, "timestamp_numerator": 0}),),
    )
    manifest = _manifest((_frame(0, left),)).model_copy(update={"sequences": (first, second)})

    permissive = ValidationThresholdsV1(
        perceptual_duplicate_hamming_max=0,
        blur_variance_min="0",
        scene_cut_histogram_max="1",
        temporal_motion_max="1",
    )
    pending = validate_curated_dataset(manifest, {"frame_0": left, "frame_1": right}, permissive)
    identity = next(
        item for item in pending.sequence_findings if item.reason_code == "identity_inconsistency"
    )
    assert identity.state == "needs_review"
    assert pending.state == "needs_review"

    reviewed = validate_curated_dataset(
        manifest,
        {"frame_0": left, "frame_1": right},
        permissive,
        human_dispositions={identity.finding_id: "accepted"},
    )
    assert (
        next(
            item for item in reviewed.sequence_findings if item.finding_id == identity.finding_id
        ).state
        == "accepted"
    )


def test_validation_rejects_near_duplicate_cross_split_leakage() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4] = 255
    left, right = _decoded(image), _decoded(image.copy())
    first = _manifest((_frame(0, left),)).sequences[0]
    second_frame = _frame(1, right).model_copy(update={"frame_index": 0, "timestamp_numerator": 0})
    second = first.model_copy(
        update={
            "sequence_id": "sequence_b",
            "source_lineage": "lineage_b",
            "split": "holdout",
            "frames": (second_frame,),
        }
    )
    manifest = _manifest((_frame(0, left),)).model_copy(update={"sequences": (first, second)})

    report = validate_curated_dataset(manifest, {"frame_0": left, "frame_1": right})

    leakage = next(
        item
        for item in report.sequence_findings
        if item.reason_code == "near_duplicate_split_leakage"
    )
    assert leakage.state == "rejected"
    assert leakage.related_sequence_ids == ("sequence_a", "sequence_b")


def test_curated_training_evidence_reference_is_accepted_only_and_versioned() -> None:
    reference = CuratedEvidenceReferenceV1(
        curated_dataset_digest="sha256:" + "1" * 64,
        validation_report_digest="sha256:" + "2" * 64,
        sequence_id="sequence_a",
        validation_state="accepted",
    )

    assert reference.schema_version == "curated-training-evidence-reference.v1"
    with pytest.raises(ValidationError):
        CuratedEvidenceReferenceV1(
            curated_dataset_digest="sha256:" + "1" * 64,
            validation_report_digest="sha256:" + "2" * 64,
            sequence_id="sequence_a",
            validation_state="needs_review",
        )


def test_finding_identity_binds_threshold_source_artifact_and_observed_content() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    decoded = _decoded(image)
    manifest = _manifest((_frame(0, decoded),))
    first = validate_curated_dataset(
        manifest,
        {"frame_0": decoded},
        ValidationThresholdsV1(blur_variance_min="1"),
    )
    finding = next(
        item for item in first.frame_findings if item.reason_code == "blur_below_minimum"
    )
    changed_threshold = validate_curated_dataset(
        manifest,
        {"frame_0": decoded},
        ValidationThresholdsV1(blur_variance_min="2"),
    )
    changed_frame = (
        manifest.sequences[0]
        .frames[0]
        .model_copy(update={"source_digest": _digest(b"different-source")})
    )
    changed_manifest = manifest.model_copy(
        update={
            "sequences": (manifest.sequences[0].model_copy(update={"frames": (changed_frame,)}),)
        }
    )
    changed_artifact = validate_curated_dataset(
        changed_manifest,
        {"frame_0": decoded},
        ValidationThresholdsV1(blur_variance_min="1"),
    )

    threshold_id = next(
        item.finding_id
        for item in changed_threshold.frame_findings
        if item.reason_code == "blur_below_minimum"
    )
    artifact_finding = next(
        item for item in changed_artifact.frame_findings if item.reason_code == "blur_below_minimum"
    )
    assert finding.finding_id != threshold_id
    assert finding.finding_id != artifact_finding.finding_id
    assert finding.source_digest == manifest.sequences[0].frames[0].source_digest
    assert artifact_finding.source_digest == changed_frame.source_digest


def test_human_disposition_replay_rejects_changed_threshold_content_or_artifact() -> None:
    generator = np.random.default_rng(71)
    left = _decoded(generator.integers(0, 256, (8, 8, 3), dtype=np.uint8))
    right = _decoded(generator.integers(0, 256, (8, 8, 3), dtype=np.uint8))
    first = _manifest((_frame(0, left),)).sequences[0]
    second = first.model_copy(
        update={
            "sequence_id": "sequence_b",
            "source_lineage": "lineage_b",
            "split": "validation",
            "caption": StructuredCaptionV1(
                character_token="hero",
                immutable_attributes=("green eyes",),
                action="walking",
            ),
            "frames": (
                _frame(1, right).model_copy(update={"frame_index": 0, "timestamp_numerator": 0}),
            ),
        }
    )
    manifest = _manifest((_frame(0, left),)).model_copy(update={"sequences": (first, second)})
    permissive = ValidationThresholdsV1(
        perceptual_duplicate_hamming_max=0,
        blur_variance_min="0",
        scene_cut_histogram_max="1",
        temporal_motion_max="1",
    )
    report = validate_curated_dataset(manifest, {"frame_0": left, "frame_1": right}, permissive)
    finding = next(
        item for item in report.sequence_findings if item.reason_code == "identity_inconsistency"
    )

    changed_threshold = permissive.model_copy(update={"blur_variance_min": "0.1000"})
    with pytest.raises(ValueError, match="immutable finding evidence"):
        validate_curated_dataset(
            manifest,
            {"frame_0": left, "frame_1": right},
            changed_threshold,
            human_dispositions={finding.finding_id: "accepted"},
        )
    changed_source = second.frames[0].model_copy(
        update={"source_digest": _digest(b"changed-artifact")}
    )
    changed_manifest = manifest.model_copy(
        update={"sequences": (first, second.model_copy(update={"frames": (changed_source,)}))}
    )
    with pytest.raises(ValueError, match="immutable finding evidence"):
        validate_curated_dataset(
            changed_manifest,
            {"frame_0": left, "frame_1": right},
            permissive,
            human_dispositions={finding.finding_id: "accepted"},
        )
    changed_caption = second.caption.model_copy(update={"immutable_attributes": ("hazel eyes",)})
    changed_content_manifest = manifest.model_copy(
        update={
            "sequences": (
                first,
                second.model_copy(update={"caption": changed_caption}),
            )
        }
    )
    with pytest.raises(ValueError, match="immutable finding evidence"):
        validate_curated_dataset(
            changed_content_manifest,
            {"frame_0": left, "frame_1": right},
            permissive,
            human_dispositions={finding.finding_id: "accepted"},
        )


def test_validation_comparison_budget_accepts_boundary_and_rejects_one_over() -> None:
    arrays = (
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 2, dtype=np.uint8),
    )
    decoded = tuple(_decoded(array) for array in arrays)
    two_manifest = _manifest(tuple(_frame(i, value) for i, value in enumerate(decoded[:2])))
    boundary = validate_curated_dataset(
        two_manifest,
        {f"frame_{i}": value for i, value in enumerate(decoded[:2])},
        ValidationThresholdsV1(max_comparisons=1),
    )
    assert all(
        item.reason_code != "validation_comparison_budget_exceeded"
        for item in boundary.sequence_findings
    )

    three_manifest = _manifest(tuple(_frame(i, value) for i, value in enumerate(decoded)))
    over = validate_curated_dataset(
        three_manifest,
        {f"frame_{i}": value for i, value in enumerate(decoded)},
        ValidationThresholdsV1(max_comparisons=2),
    )
    assert over.sequence_findings[0].reason_code == "validation_comparison_budget_exceeded"


def test_validation_budget_fails_before_pairwise_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _decoded(np.zeros((8, 8, 3), dtype=np.uint8))
    right = _decoded(np.ones((8, 8, 3), dtype=np.uint8))
    manifest = _manifest((_frame(0, left), _frame(1, right)))
    monkeypatch.setattr(
        "video_lab.curated_validation.hamming_distance64",
        lambda *_: pytest.fail("pairwise work must not start beyond budget"),
    )

    report = validate_curated_dataset(
        manifest,
        {"frame_0": left, "frame_1": right},
        ValidationThresholdsV1(max_frames=1, max_comparisons=0),
    )

    assert report.state == "rejected"
    assert report.sequence_findings[0].reason_code == "validation_frame_budget_exceeded"
    with pytest.raises(ValueError, match="immutable finding evidence"):
        validate_curated_dataset(
            manifest,
            {"frame_0": left, "frame_1": right},
            ValidationThresholdsV1(max_frames=1, max_comparisons=0),
            human_dispositions={"finding_stale": "accepted"},
        )


def test_decoded_digest_mismatch_explains_expected_and_observed_content() -> None:
    decoded = _decoded(np.zeros((8, 8, 3), dtype=np.uint8))
    frame = _frame(0, decoded)
    changed = replace(decoded, pixels=b"\x01" + decoded.pixels[1:])
    tampered = replace(changed, pixel_digest=decoded_pixel_digest(changed))
    manifest = _manifest((frame,))

    report = validate_curated_dataset(manifest, {"frame_0": tampered})

    mismatch = next(
        item for item in report.frame_findings if item.reason_code == "decoded_digest_mismatch"
    )
    assert mismatch.source_digest == frame.source_digest
    assert mismatch.pixel_digest == frame.pixel_digest
    assert mismatch.observed_pixel_digest == tampered.pixel_digest


@pytest.mark.parametrize(
    "update",
    [
        {"width": 7},
        {"height": 7},
        {"channels": 4},
        {"bit_depth": 16},
        {"bit_depth": 8.0},
        {"bit_depth": np.int64(8)},
        {"bit_depth": True},
        {"color_space": "RGBA"},
        {"pixels": b"short"},
    ],
)
def test_validation_rejects_forged_decoded_metadata_or_length(update: dict[str, object]) -> None:
    decoded = _decoded(np.zeros((8, 8, 3), dtype=np.uint8))
    manifest = _manifest((_frame(0, decoded),))
    forged = replace(decoded, **update)

    with pytest.raises(ValueError, match="decoded frame"):
        validate_curated_dataset(manifest, {"frame_0": forged})


def test_validation_rejects_changed_pixels_with_retained_digest() -> None:
    decoded = _decoded(np.zeros((8, 8, 3), dtype=np.uint8))
    manifest = _manifest((_frame(0, decoded),))
    forged = replace(decoded, pixels=b"\x01" + decoded.pixels[1:])

    with pytest.raises(ValueError, match="decoded frame"):
        validate_curated_dataset(manifest, {"frame_0": forged})


def test_exact_duplicate_finding_distinguishes_source_artifacts() -> None:
    decoded = _decoded(np.zeros((8, 8, 3), dtype=np.uint8))
    manifest = _manifest((_frame(0, decoded), _frame(1, decoded)))

    report = validate_curated_dataset(manifest, {"frame_0": decoded, "frame_1": decoded})

    duplicate = next(
        item for item in report.frame_findings if item.reason_code == "exact_duplicate"
    )
    assert duplicate.source_digest == manifest.sequences[0].frames[1].source_digest
    assert duplicate.related_source_digest == manifest.sequences[0].frames[0].source_digest
