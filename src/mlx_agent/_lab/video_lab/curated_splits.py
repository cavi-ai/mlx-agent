"""Verified deterministic identity, lineage, and perceptual-aware split assignment."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from video_lab.curated_ids import curated_dataset_digest
from video_lab.curated_metrics import hamming_distance64, metric_policy_digest, perceptual_hash64
from video_lab.curated_models import CuratedDatasetManifestV1, DatasetSplitPolicyV1
from video_lab.curated_source import CuratedSourceError, DecodedFrame, validate_decoded_frame
from video_lab.domain import canonical_json


class CuratedSplitError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _SplitModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class PerceptualFrameEvidenceV1(_SplitModel):
    schema_version: Literal["perceptual-frame-evidence.v1"] = "perceptual-frame-evidence.v1"
    frame_id: str
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    pixel_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    perceptual_hash64: int = Field(ge=0, le=(1 << 64) - 1)


class PerceptualComparisonEvidenceV1(_SplitModel):
    schema_version: Literal["perceptual-comparison-evidence.v1"] = (
        "perceptual-comparison-evidence.v1"
    )
    left_frame_id: str
    right_frame_id: str
    hamming_distance: int = Field(ge=0, le=64)

    @model_validator(mode="after")
    def validate_order(self) -> PerceptualComparisonEvidenceV1:
        if self.left_frame_id >= self.right_frame_id:
            raise ValueError("comparison frame IDs must be canonically ordered")
        return self


class SplitAssignmentEvidenceV1(_SplitModel):
    schema_version: Literal["split-assignment-evidence.v2"] = "split-assignment-evidence.v2"
    producer_id: Literal["curated-validator.v1"] = "curated-validator.v1"
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_policy_version: Literal["curated-validation-thresholds.v1"] = (
        "curated-validation-thresholds.v1"
    )
    perceptual_hamming_threshold: int = Field(ge=0, le=64)
    max_sequences: int = Field(gt=0)
    max_frames: int = Field(gt=0)
    max_comparisons: int = Field(ge=0)
    comparison_universe_count: int = Field(ge=0)
    frames: tuple[PerceptualFrameEvidenceV1, ...]
    comparisons: tuple[PerceptualComparisonEvidenceV1, ...]
    producer_receipt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_canonical_universe(self) -> SplitAssignmentEvidenceV1:
        frame_ids = tuple(frame.frame_id for frame in self.frames)
        if frame_ids != tuple(sorted(frame_ids)) or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("perceptual frames must be unique and canonically ordered")
        keys = tuple(
            (comparison.left_frame_id, comparison.right_frame_id) for comparison in self.comparisons
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("comparisons must be unique and canonically ordered")
        expected_count = len(frame_ids) * (len(frame_ids) - 1) // 2
        if self.comparison_universe_count != expected_count or len(keys) != expected_count:
            raise ValueError("comparison evidence must cover the complete universe")
        if expected_count > self.max_comparisons:
            raise ValueError("comparison evidence exceeds its declared budget")
        return self


def produce_split_assignment_evidence(
    manifest: CuratedDatasetManifestV1,
    decoded_frames: Mapping[str, DecodedFrame],
    *,
    perceptual_hamming_threshold: int,
    max_sequences: int,
    max_frames: int,
    max_comparisons: int,
    cancelled: Callable[[], bool] = lambda: False,
) -> SplitAssignmentEvidenceV1:
    """Produce a complete receipt from descriptor-verified decoded frame evidence."""
    if not 0 <= perceptual_hamming_threshold <= 64:
        raise CuratedSplitError("split_threshold_invalid")
    sequence_count = len(manifest.sequences)
    manifest_frames = tuple(frame for sequence in manifest.sequences for frame in sequence.frames)
    frame_count = len(manifest_frames)
    comparison_count = frame_count * (frame_count - 1) // 2
    if sequence_count > max_sequences:
        raise CuratedSplitError("split_sequence_budget_exceeded")
    if frame_count > max_frames:
        raise CuratedSplitError("split_frame_budget_exceeded")
    if comparison_count > max_comparisons:
        raise CuratedSplitError("split_comparison_budget_exceeded")
    if set(decoded_frames) != {frame.frame_id for frame in manifest_frames}:
        raise CuratedSplitError("split_decoded_frame_set_mismatch")
    try:
        for frame_id in sorted(decoded_frames):
            if cancelled():
                raise CuratedSplitError("split_cancelled")
            validate_decoded_frame(decoded_frames[frame_id])
    except CuratedSourceError:
        raise CuratedSplitError("split_decoded_frame_invalid") from None

    frame_evidence: list[PerceptualFrameEvidenceV1] = []
    for frame in sorted(manifest_frames, key=lambda value: value.frame_id):
        if cancelled():
            raise CuratedSplitError("split_cancelled")
        decoded = decoded_frames[frame.frame_id]
        if decoded.pixel_digest != frame.pixel_digest:
            raise CuratedSplitError("split_decoded_digest_mismatch")
        frame_evidence.append(
            PerceptualFrameEvidenceV1(
                frame_id=frame.frame_id,
                source_digest=frame.source_digest,
                pixel_digest=frame.pixel_digest,
                perceptual_hash64=perceptual_hash64(decoded),
            )
        )
    comparisons: list[PerceptualComparisonEvidenceV1] = []
    for left, right in combinations(frame_evidence, 2):
        if cancelled():
            raise CuratedSplitError("split_cancelled")
        comparisons.append(PerceptualComparisonEvidenceV1(
            left_frame_id=left.frame_id,
            right_frame_id=right.frame_id,
            hamming_distance=hamming_distance64(left.perceptual_hash64, right.perceptual_hash64),
        ))
    comparisons_evidence = tuple(comparisons)
    payload: dict[str, object] = {
        "schema_version": "split-assignment-evidence.v2",
        "producer_id": "curated-validator.v1",
        "manifest_digest": curated_dataset_digest(manifest),
        "metric_policy_digest": metric_policy_digest(),
        "validation_policy_version": "curated-validation-thresholds.v1",
        "perceptual_hamming_threshold": perceptual_hamming_threshold,
        "max_sequences": max_sequences,
        "max_frames": max_frames,
        "max_comparisons": max_comparisons,
        "comparison_universe_count": comparison_count,
        "frames": tuple(frame.model_dump(mode="json") for frame in frame_evidence),
        "comparisons": tuple(
            comparison.model_dump(mode="json") for comparison in comparisons_evidence
        ),
    }
    receipt = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return SplitAssignmentEvidenceV1(
        manifest_digest=curated_dataset_digest(manifest),
        metric_policy_digest=metric_policy_digest(),
        perceptual_hamming_threshold=perceptual_hamming_threshold,
        max_sequences=max_sequences,
        max_frames=max_frames,
        max_comparisons=max_comparisons,
        comparison_universe_count=comparison_count,
        frames=tuple(frame_evidence),
        comparisons=comparisons_evidence,
        producer_receipt_digest=receipt,
    )


class _UnionFind:
    def __init__(self, values: tuple[str, ...]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def assign_identity_aware_splits(
    manifest: CuratedDatasetManifestV1,
    policy: DatasetSplitPolicyV1,
    *,
    decoded_frames: Mapping[str, DecodedFrame] | None = None,
    evidence: SplitAssignmentEvidenceV1 | None = None,
    enforce_declared: bool = True,
    cancelled: Callable[[], bool] = lambda: False,
) -> CuratedDatasetManifestV1:
    """Assign splits only after independently reproducing the complete pHash receipt."""
    if manifest.split_policy != policy:
        raise CuratedSplitError("split_policy_conflict")
    if evidence is None or decoded_frames is None:
        raise CuratedSplitError("split_evidence_required")
    try:
        reproduced = produce_split_assignment_evidence(
            manifest,
            decoded_frames,
            perceptual_hamming_threshold=evidence.perceptual_hamming_threshold,
            max_sequences=evidence.max_sequences,
            max_frames=evidence.max_frames,
            max_comparisons=evidence.max_comparisons,
            cancelled=cancelled,
        )
    except CuratedSplitError:
        raise
    if reproduced != evidence:
        raise CuratedSplitError("split_evidence_verification_failed")

    sequence_ids = tuple(sequence.sequence_id for sequence in manifest.sequences)
    groups = _UnionFind(sequence_ids)
    lineage_owner: dict[tuple[str, str], str] = {}
    digest_owner: dict[str, str] = {}
    frame_owner: dict[str, str] = {}
    for sequence in manifest.sequences:
        if cancelled():
            raise CuratedSplitError("split_cancelled")
        lineage_key = (sequence.identity_id, sequence.source_lineage)
        owner = lineage_owner.setdefault(lineage_key, sequence.sequence_id)
        groups.union(owner, sequence.sequence_id)
        for frame in sequence.frames:
            frame_owner[frame.frame_id] = sequence.sequence_id
            owner = digest_owner.setdefault(frame.pixel_digest, sequence.sequence_id)
            groups.union(owner, sequence.sequence_id)
    for comparison in evidence.comparisons:
        if cancelled():
            raise CuratedSplitError("split_cancelled")
        if comparison.hamming_distance <= evidence.perceptual_hamming_threshold:
            groups.union(
                frame_owner[comparison.left_frame_id],
                frame_owner[comparison.right_frame_id],
            )

    assignments: dict[str, str] = {}
    components: dict[str, list[str]] = {}
    for sequence_id in sequence_ids:
        if cancelled():
            raise CuratedSplitError("split_cancelled")
        components.setdefault(groups.find(sequence_id), []).append(sequence_id)
    by_id = {sequence.sequence_id: sequence for sequence in manifest.sequences}
    for members in components.values():
        if cancelled():
            raise CuratedSplitError("split_cancelled")
        representative = min(members)
        sequence = by_id[representative]
        material = "\x00".join(
            (policy.policy_version, str(policy.seed), sequence.identity_id, sequence.source_lineage)
        ).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(material).digest(), "big") % policy.denominator
        split = (
            "train"
            if bucket < policy.train_numerator
            else "validation"
            if bucket < policy.train_numerator + policy.validation_numerator
            else "holdout"
        )
        for member in members:
            assignments[member] = split

    if enforce_declared and any(
        sequence.split != assignments[sequence.sequence_id] for sequence in manifest.sequences
    ):
        raise CuratedSplitError("declared_split_conflict")
    sequences = tuple(
        sequence.model_copy(update={"split": assignments[sequence.sequence_id]})
        for sequence in manifest.sequences
    )
    return manifest.model_copy(update={"sequences": sequences})
