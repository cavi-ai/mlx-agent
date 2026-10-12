from __future__ import annotations

import hashlib

import numpy as np
import pytest

from video_lab.curated_models import (
    CuratedDatasetManifestV1,
    CuratedFrameV1,
    CuratedSequenceV1,
    DatasetSplitPolicyV1,
    SequenceTransformRecipeV1,
    StructuredCaptionV1,
)
from video_lab.curated_source import DecodedFrame, decoded_pixel_digest
from video_lab.curated_splits import (
    CuratedSplitError,
    assign_identity_aware_splits,
    produce_split_assignment_evidence,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pixels(index: int) -> bytes:
    return np.random.default_rng(index).integers(0, 256, (8, 8, 3), dtype=np.uint8).tobytes()


def _pixel_digest(index: int) -> str:
    frame = DecodedFrame(
        width=8,
        height=8,
        channels=3,
        bit_depth=8,
        color_space="RGB",
        pixel_digest="0" * 64,
        pixels=_pixels(index),
    )
    return decoded_pixel_digest(frame)


def _sequence(index: int, lineage: str, digest: str, split: str = "train") -> CuratedSequenceV1:
    frame = CuratedFrameV1(
        frame_id=f"frame_{index}",
        frame_index=0,
        timestamp_numerator=0,
        timestamp_denominator=1,
        source_path=f"s{index}/0.png",
        source_digest=_sha(f"s{index}"),
        pixel_digest=digest,
        review_state="accepted",
        reviewer_evidence=("review_one",),
    )
    return CuratedSequenceV1(
        sequence_id=f"sequence_{index}",
        identity_id="hero",
        source_lineage=lineage,
        split=split,
        caption=StructuredCaptionV1(
            character_token="hero", immutable_attributes=("blue eyes",), action="walking"
        ),
        frames=(frame,),
    )


def _manifest(
    sequences: tuple[CuratedSequenceV1, ...], policy: DatasetSplitPolicyV1
) -> CuratedDatasetManifestV1:
    return CuratedDatasetManifestV1(
        dataset_id="dataset_a",
        sequences=sequences,
        split_policy=policy,
        transform_recipe=SequenceTransformRecipeV1(
            recipe_id="recipe_one",
            output_width=8,
            output_height=8,
            output_frame_count=1,
            temporal_stride_numerator=1,
            temporal_stride_denominator=1,
            interpolation="nearest",
            padding_color=(0, 0, 0),
            seed=1,
        ),
    )


def _decoded(manifest: CuratedDatasetManifestV1) -> dict[str, DecodedFrame]:
    result: dict[str, DecodedFrame] = {}
    known_pixels = {_pixel_digest(index): _pixels(index) for index in range(128)}
    for frame in (frame for sequence in manifest.sequences for frame in sequence.frames):
        pixels = known_pixels[frame.pixel_digest]
        result[frame.frame_id] = DecodedFrame(
            width=8,
            height=8,
            channels=3,
            bit_depth=8,
            color_space="RGB",
            pixel_digest=frame.pixel_digest,
            pixels=pixels,
        )
    return result


def _assignment_inputs(
    manifest: CuratedDatasetManifestV1, threshold: int = 4
) -> tuple[dict[str, DecodedFrame], object]:
    decoded = _decoded(manifest)
    evidence = produce_split_assignment_evidence(
        manifest,
        decoded,
        perceptual_hamming_threshold=threshold,
        max_sequences=len(manifest.sequences),
        max_frames=len(decoded),
        max_comparisons=len(decoded) * (len(decoded) - 1) // 2,
    )
    return decoded, evidence


def test_split_evidence_and_reproduction_honor_cancellation() -> None:
    policy = DatasetSplitPolicyV1(
        policy_version="policy_cancel",
        seed=1,
        train_numerator=1,
        validation_numerator=1,
        holdout_numerator=1,
        denominator=3,
    )
    manifest = _manifest((_sequence(1, "lineage", _pixel_digest(1)),), policy)
    decoded = _decoded(manifest)
    with pytest.raises(CuratedSplitError, match="split_cancelled"):
        produce_split_assignment_evidence(
            manifest,
            decoded,
            perceptual_hamming_threshold=4,
            max_sequences=1,
            max_frames=1,
            max_comparisons=0,
            cancelled=lambda: True,
        )
    evidence = produce_split_assignment_evidence(
        manifest,
        decoded,
        perceptual_hamming_threshold=4,
        max_sequences=1,
        max_frames=1,
        max_comparisons=0,
    )
    with pytest.raises(CuratedSplitError, match="split_cancelled"):
        assign_identity_aware_splits(
            manifest,
            policy,
            decoded_frames=decoded,
            evidence=evidence,
            cancelled=lambda: True,
        )


def test_split_assignment_is_deterministic_and_keeps_lineage_and_duplicates_together() -> None:
    policy = DatasetSplitPolicyV1(
        policy_version="policy_one",
        seed=42,
        train_numerator=1,
        validation_numerator=1,
        holdout_numerator=1,
        denominator=3,
    )
    duplicate = _pixel_digest(2)
    sequences = (
        _sequence(0, "shared", _pixel_digest(0)),
        _sequence(1, "shared", _pixel_digest(1)),
        _sequence(2, "other", duplicate),
        _sequence(3, "third", duplicate),
        _sequence(4, "fourth", _pixel_digest(4)),
    )
    manifest = _manifest(sequences, policy)

    decoded, evidence = _assignment_inputs(manifest)
    assigned = assign_identity_aware_splits(
        manifest, policy, decoded_frames=decoded, evidence=evidence, enforce_declared=False
    )
    repeated = assign_identity_aware_splits(
        manifest, policy, decoded_frames=decoded, evidence=evidence, enforce_declared=False
    )

    assert assigned == repeated
    by_id = {sequence.sequence_id: sequence.split for sequence in assigned.sequences}
    assert by_id["sequence_0"] == by_id["sequence_1"]
    assert by_id["sequence_2"] == by_id["sequence_3"]


def test_split_assignment_rejects_declared_conflict() -> None:
    policy = DatasetSplitPolicyV1(
        policy_version="policy_one",
        seed=42,
        train_numerator=1,
        validation_numerator=1,
        holdout_numerator=1,
        denominator=3,
    )
    manifest = _manifest((_sequence(0, "one", _pixel_digest(0), "holdout"),), policy)

    decoded, evidence = _assignment_inputs(manifest)
    assigned = assign_identity_aware_splits(
        manifest, policy, decoded_frames=decoded, evidence=evidence, enforce_declared=False
    )
    conflicting = manifest.model_copy(
        update={
            "sequences": (
                manifest.sequences[0].model_copy(
                    update={
                        "split": "train" if assigned.sequences[0].split != "train" else "holdout"
                    }
                ),
            )
        }
    )
    with pytest.raises(CuratedSplitError, match="declared_split_conflict"):
        conflicting_evidence = produce_split_assignment_evidence(
            conflicting,
            decoded,
            perceptual_hamming_threshold=4,
            max_sequences=1,
            max_frames=1,
            max_comparisons=0,
        )
        assign_identity_aware_splits(
            conflicting, policy, decoded_frames=decoded, evidence=conflicting_evidence
        )


def test_split_assignment_co_clusters_distinct_perceptual_near_duplicates() -> None:
    policy = DatasetSplitPolicyV1(
        policy_version="policy_one",
        seed=42,
        train_numerator=1,
        validation_numerator=1,
        holdout_numerator=1,
        denominator=3,
    )
    sequences = (
        _sequence(0, "one", _pixel_digest(0)),
        _sequence(1, "two", _pixel_digest(1)),
    )
    manifest = _manifest(sequences, policy)
    decoded = _decoded(manifest)
    evidence = produce_split_assignment_evidence(
        manifest,
        decoded,
        perceptual_hamming_threshold=64,
        max_sequences=2,
        max_frames=2,
        max_comparisons=1,
    )

    assigned = assign_identity_aware_splits(
        manifest,
        policy,
        decoded_frames=decoded,
        evidence=evidence,
        enforce_declared=False,
    )

    assert assigned.sequences[0].split == assigned.sequences[1].split


def test_split_assignment_rejects_missing_or_incomplete_evidence() -> None:
    policy = DatasetSplitPolicyV1(
        policy_version="policy_one",
        seed=42,
        train_numerator=1,
        validation_numerator=1,
        holdout_numerator=1,
        denominator=3,
    )
    manifest = _manifest(
        (_sequence(0, "one", _pixel_digest(0)), _sequence(1, "two", _pixel_digest(1))),
        policy,
    )
    decoded = _decoded(manifest)
    with pytest.raises(CuratedSplitError, match="split_evidence_required"):
        assign_identity_aware_splits(manifest, policy, decoded_frames=decoded)

    evidence = produce_split_assignment_evidence(
        manifest,
        decoded,
        perceptual_hamming_threshold=4,
        max_sequences=2,
        max_frames=2,
        max_comparisons=1,
    )
    incomplete = evidence.model_copy(update={"comparisons": ()})
    with pytest.raises(CuratedSplitError, match="split_evidence_verification_failed"):
        assign_identity_aware_splits(manifest, policy, decoded_frames=decoded, evidence=incomplete)


def test_split_assignment_rejects_false_distance_policy_or_threshold() -> None:
    policy = DatasetSplitPolicyV1(
        policy_version="policy_one",
        seed=42,
        train_numerator=1,
        validation_numerator=1,
        holdout_numerator=1,
        denominator=3,
    )
    manifest = _manifest(
        (_sequence(0, "one", _pixel_digest(0)), _sequence(1, "two", _pixel_digest(1))),
        policy,
    )
    decoded, evidence = _assignment_inputs(manifest)
    comparison = evidence.comparisons[0]
    mutations = (
        {"metric_policy_digest": _sha("false-policy")},
        {"perceptual_hamming_threshold": evidence.perceptual_hamming_threshold + 1},
        {
            "comparisons": (
                comparison.model_copy(
                    update={"hamming_distance": (comparison.hamming_distance + 1) % 65}
                ),
            )
        },
    )
    for update in mutations:
        with pytest.raises(CuratedSplitError, match="split_evidence_verification_failed"):
            assign_identity_aware_splits(
                manifest,
                policy,
                decoded_frames=decoded,
                evidence=evidence.model_copy(update=update),
            )


@pytest.mark.parametrize("bit_depth", [8.0, np.int64(8), True])
def test_split_producer_rejects_non_builtin_bit_depth(bit_depth: object) -> None:
    policy = DatasetSplitPolicyV1(
        policy_version="policy_one",
        seed=42,
        train_numerator=1,
        validation_numerator=1,
        holdout_numerator=1,
        denominator=3,
    )
    manifest = _manifest((_sequence(0, "one", _pixel_digest(0)),), policy)
    decoded = _decoded(manifest)
    forged = DecodedFrame(
        width=8,
        height=8,
        channels=3,
        bit_depth=bit_depth,  # type: ignore[arg-type]
        color_space="RGB",
        pixel_digest=decoded["frame_0"].pixel_digest,
        pixels=decoded["frame_0"].pixels,
    )

    with pytest.raises(CuratedSplitError, match="split_decoded_frame_invalid"):
        produce_split_assignment_evidence(
            manifest,
            {"frame_0": forged},
            perceptual_hamming_threshold=4,
            max_sequences=1,
            max_frames=1,
            max_comparisons=0,
        )
