"""Pure deterministic validation and normalized evidence for curated datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from video_lab.curated_ids import curated_dataset_digest
from video_lab.curated_metrics import (
    blur_variance,
    hamming_distance64,
    histogram_discontinuity,
    metric_policy_digest,
    perceptual_hash64,
    quantize_metric,
    temporal_motion,
)
from video_lab.curated_models import CuratedDatasetManifestV1, CuratedFrameV1
from video_lab.curated_source import CuratedSourceError, DecodedFrame, validate_decoded_frame
from video_lab.curated_splits import (
    SplitAssignmentEvidenceV1,
    produce_split_assignment_evidence,
)
from video_lab.domain import canonical_json

FindingState = Literal["accepted", "rejected", "needs_review"]


class _ValidationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ValidationThresholdsV1(_ValidationModel):
    schema_version: Literal["curated-validation-thresholds.v1"] = "curated-validation-thresholds.v1"
    perceptual_duplicate_hamming_max: int = Field(default=4, ge=0, le=64)
    blur_variance_min: str = "20.0000"
    scene_cut_histogram_max: str = "0.6500"
    temporal_motion_max: str = "0.6500"
    max_sequences: int = Field(default=512, gt=0)
    max_frames: int = Field(default=4096, gt=0)
    max_comparisons: int = Field(default=1_000_000, ge=0)

    @field_validator("blur_variance_min", "scene_cut_histogram_max", "temporal_motion_max")
    @classmethod
    def validate_decimal(cls, value: str) -> str:
        decimal = Decimal(value)
        if not decimal.is_finite() or decimal < 0:
            raise ValueError("threshold must be a finite nonnegative decimal")
        return quantize_metric(decimal)


class FrameFinding(_ValidationModel):
    finding_id: str
    state: FindingState
    reason_code: str
    frame_id: str
    sequence_id: str
    source_digest: str
    pixel_digest: str
    observed_pixel_digest: str | None = None
    observed_value: str
    threshold: str
    related_frame_id: str | None = None
    related_source_digest: str | None = None
    related_pixel_digest: str | None = None
    human_disposition: Literal["accepted", "rejected"] | None = None


class ImplicatedFrameEvidence(_ValidationModel):
    frame_id: str
    source_digest: str
    pixel_digest: str


class SequenceFinding(_ValidationModel):
    finding_id: str
    state: FindingState
    reason_code: str
    sequence_id: str
    related_sequence_ids: tuple[str, ...] = ()
    implicated_frames: tuple[ImplicatedFrameEvidence, ...] = ()
    observed_value: str
    threshold: str
    human_disposition: Literal["accepted", "rejected"] | None = None


class CuratedValidationReport(_ValidationModel):
    schema_version: Literal["curated-validation-report.v1"] = "curated-validation-report.v1"
    dataset_digest: str
    thresholds: ValidationThresholdsV1
    metric_policy_digest: str
    split_assignment_evidence: SplitAssignmentEvidenceV1 | None = None
    state: FindingState
    frame_findings: tuple[FrameFinding, ...]
    sequence_findings: tuple[SequenceFinding, ...]


def _finding_id(payload: Mapping[str, object]) -> str:
    return "finding_" + hashlib.sha256(canonical_json(dict(payload)).encode()).hexdigest()


def _validation_policy_digest(thresholds: ValidationThresholdsV1) -> str:
    payload = {
        "metric_policy_digest": metric_policy_digest(),
        "thresholds": thresholds.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _state_with_disposition(
    default: FindingState, finding_id: str, dispositions: Mapping[str, str]
) -> tuple[FindingState, Literal["accepted", "rejected"] | None]:
    disposition = dispositions.get(finding_id)
    if disposition not in (None, "accepted", "rejected"):
        raise ValueError("human disposition must be accepted or rejected")
    if disposition is not None and default != "needs_review":
        raise ValueError("human dispositions apply only to review findings")
    return (disposition if disposition is not None else default), disposition  # type: ignore[return-value]


def validate_curated_dataset(
    manifest: CuratedDatasetManifestV1,
    decoded_frames: Mapping[str, DecodedFrame],
    thresholds: ValidationThresholdsV1 | None = None,
    *,
    human_dispositions: Mapping[str, str] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> CuratedValidationReport:
    configured = thresholds or ValidationThresholdsV1()
    dispositions = human_dispositions or {}
    dataset_digest = curated_dataset_digest(manifest)
    policy_digest = _validation_policy_digest(configured)
    sequence_count = len(manifest.sequences)
    frame_count = sum(len(sequence.frames) for sequence in manifest.sequences)
    comparison_count = frame_count * (frame_count - 1) // 2
    budget_code = (
        "validation_sequence_budget_exceeded"
        if sequence_count > configured.max_sequences
        else "validation_frame_budget_exceeded"
        if frame_count > configured.max_frames
        else "validation_comparison_budget_exceeded"
        if comparison_count > configured.max_comparisons
        else None
    )
    if budget_code is not None:
        if dispositions:
            raise ValueError("human disposition does not bind current immutable finding evidence")
        finding = _sequence_finding(
            budget_code,
            "rejected",
            manifest.sequences[0].sequence_id,
            (),
            (),
            quantize_metric(
                sequence_count
                if "sequence" in budget_code
                else frame_count
                if "frame" in budget_code
                else comparison_count
            ),
            quantize_metric(
                configured.max_sequences
                if "sequence" in budget_code
                else configured.max_frames
                if "frame" in budget_code
                else configured.max_comparisons
            ),
            dispositions,
            dataset_digest,
            policy_digest,
        )
        return CuratedValidationReport(
            dataset_digest=dataset_digest,
            thresholds=configured,
            metric_policy_digest=metric_policy_digest(),
            state="rejected",
            frame_findings=(),
            sequence_findings=(finding,),
        )
    expected_ids = {frame.frame_id for sequence in manifest.sequences for frame in sequence.frames}
    if set(decoded_frames) != expected_ids:
        raise ValueError("decoded frame identities must exactly match the manifest")
    try:
        for frame_id in sorted(decoded_frames):
            validate_decoded_frame(decoded_frames[frame_id])
    except CuratedSourceError:
        raise ValueError("decoded frame evidence is not canonical") from None
    frame_findings: list[FrameFinding] = []
    sequence_findings: list[SequenceFinding] = []
    seen_exact: dict[str, tuple[str, str]] = {}
    seen_hash: list[tuple[int, str, str, str]] = []
    frame_models = {
        frame.frame_id: frame for sequence in manifest.sequences for frame in sequence.frames
    }

    for sequence in manifest.sequences:
        if cancelled():
            raise ValueError("curated_validation_cancelled")
        previous: tuple[str, DecodedFrame] | None = None
        expected_shape: tuple[int, int, int] | None = None
        for frame in sequence.frames:
            if cancelled():
                raise ValueError("curated_validation_cancelled")
            decoded = decoded_frames[frame.frame_id]
            shape = (decoded.width, decoded.height, decoded.channels)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                frame_findings.append(
                    _frame_finding(
                        "dimension_inconsistency",
                        "rejected",
                        sequence.sequence_id,
                        frame,
                        "1.0000",
                        "0.0000",
                        frame_models[previous[0]] if previous is not None else None,
                        dataset_digest,
                        policy_digest,
                    )
                )
            if decoded.pixel_digest != frame.pixel_digest:
                frame_findings.append(
                    _frame_finding(
                        "decoded_digest_mismatch",
                        "rejected",
                        sequence.sequence_id,
                        frame,
                        "1.0000",
                        "0.0000",
                        None,
                        dataset_digest,
                        policy_digest,
                        observed_pixel_digest=decoded.pixel_digest,
                    )
                )
            exact = seen_exact.get(decoded.pixel_digest)
            if exact is not None:
                frame_findings.append(
                    _frame_finding(
                        "exact_duplicate",
                        "rejected",
                        sequence.sequence_id,
                        frame,
                        "0.0000",
                        "0.0000",
                        frame_models[exact[1]],
                        dataset_digest,
                        policy_digest,
                    )
                )
            else:
                seen_exact[decoded.pixel_digest] = (sequence.sequence_id, frame.frame_id)
            phash = perceptual_hash64(decoded)
            for other_hash, other_sequence, other_frame, other_digest in seen_hash:
                if cancelled():
                    raise ValueError("curated_validation_cancelled")
                distance = hamming_distance64(phash, other_hash)
                if (
                    distance <= configured.perceptual_duplicate_hamming_max
                    and other_digest != decoded.pixel_digest
                ):
                    frame_findings.append(
                        _frame_finding(
                            "perceptual_duplicate",
                            "rejected",
                            sequence.sequence_id,
                            frame,
                            quantize_metric(distance),
                            quantize_metric(configured.perceptual_duplicate_hamming_max),
                            frame_models[other_frame],
                            dataset_digest,
                            policy_digest,
                        )
                    )
                if (
                    distance <= configured.perceptual_duplicate_hamming_max
                    and other_sequence != sequence.sequence_id
                ):
                    left_split = next(
                        item.split
                        for item in manifest.sequences
                        if item.sequence_id == other_sequence
                    )
                    if left_split != sequence.split:
                        related = tuple(sorted((other_sequence, sequence.sequence_id)))
                        sequence_findings.append(
                            _sequence_finding(
                                "near_duplicate_split_leakage",
                                "rejected",
                                sequence.sequence_id,
                                related,
                                tuple(
                                    sorted(
                                        (
                                            _implicated(frame),
                                            _implicated(frame_models[other_frame]),
                                        ),
                                        key=lambda item: item.frame_id,
                                    )
                                ),
                                quantize_metric(distance),
                                quantize_metric(configured.perceptual_duplicate_hamming_max),
                                dispositions,
                                dataset_digest,
                                policy_digest,
                            )
                        )
            seen_hash.append((phash, sequence.sequence_id, frame.frame_id, decoded.pixel_digest))
            blur = blur_variance(decoded)
            if Decimal(str(blur)) < Decimal(configured.blur_variance_min):
                frame_findings.append(
                    _frame_finding(
                        "blur_below_minimum",
                        "rejected",
                        sequence.sequence_id,
                        frame,
                        quantize_metric(blur),
                        configured.blur_variance_min,
                        None,
                        dataset_digest,
                        policy_digest,
                    )
                )
            if previous is not None:
                previous_id, previous_decoded = previous
                scene = histogram_discontinuity(previous_decoded, decoded)
                if Decimal(str(scene)) > Decimal(configured.scene_cut_histogram_max):
                    frame_findings.append(
                        _frame_finding(
                            "scene_cut",
                            "rejected",
                            sequence.sequence_id,
                            frame,
                            quantize_metric(scene),
                            configured.scene_cut_histogram_max,
                            frame_models[previous_id],
                            dataset_digest,
                            policy_digest,
                        )
                    )
                motion = temporal_motion(previous_decoded, decoded)
                if Decimal(str(motion)) > Decimal(configured.temporal_motion_max):
                    frame_findings.append(
                        _frame_finding(
                            "temporal_discontinuity",
                            "rejected",
                            sequence.sequence_id,
                            frame,
                            quantize_metric(motion),
                            configured.temporal_motion_max,
                            frame_models[previous_id],
                            dataset_digest,
                            policy_digest,
                        )
                    )
            previous = (frame.frame_id, decoded)

    identities: dict[str, tuple[tuple[str, ...], str]] = {}
    sequences_by_id = {sequence.sequence_id: sequence for sequence in manifest.sequences}
    for sequence in manifest.sequences:
        if cancelled():
            raise ValueError("curated_validation_cancelled")
        prior = identities.setdefault(
            sequence.identity_id, (sequence.caption.immutable_attributes, sequence.sequence_id)
        )
        if prior[0] != sequence.caption.immutable_attributes:
            implicated = tuple(
                sorted(
                    (
                        _implicated(frame)
                        for implicated_sequence in (
                            sequences_by_id[prior[1]],
                            sequence,
                        )
                        for frame in implicated_sequence.frames
                    ),
                    key=lambda item: item.frame_id,
                )
            )
            sequence_findings.append(
                _sequence_finding(
                    "identity_inconsistency",
                    "needs_review",
                    sequence.sequence_id,
                    tuple(sorted((prior[1], sequence.sequence_id))),
                    implicated,
                    "1.0000",
                    "0.0000",
                    dispositions,
                    dataset_digest,
                    policy_digest,
                )
            )
            sequence_findings.append(
                _sequence_finding(
                    "caption_identity_mismatch",
                    "needs_review",
                    sequence.sequence_id,
                    tuple(sorted((prior[1], sequence.sequence_id))),
                    implicated,
                    "1.0000",
                    "0.0000",
                    dispositions,
                    dataset_digest,
                    policy_digest,
                )
            )

    frame_findings.sort(
        key=lambda item: (item.sequence_id, item.frame_id, item.reason_code, item.finding_id)
    )
    unique_sequence = {item.finding_id: item for item in sequence_findings}
    ordered_sequence = tuple(
        sorted(
            unique_sequence.values(),
            key=lambda item: (item.sequence_id, item.reason_code, item.finding_id),
        )
    )
    states = [item.state for item in frame_findings] + [item.state for item in ordered_sequence]
    review_ids = {
        item.finding_id
        for item in frame_findings
        if item.human_disposition is not None or item.state == "needs_review"
    } | {
        item.finding_id
        for item in ordered_sequence
        if item.human_disposition is not None or item.state == "needs_review"
    }
    unknown_dispositions = set(dispositions) - review_ids
    if unknown_dispositions:
        raise ValueError("human disposition does not bind current immutable finding evidence")
    state: FindingState = (
        "rejected"
        if "rejected" in states
        else "needs_review"
        if "needs_review" in states
        else "accepted"
    )
    if cancelled():
        raise ValueError("curated_validation_cancelled")
    split_evidence = (
        None
        if any(item.reason_code == "decoded_digest_mismatch" for item in frame_findings)
        else produce_split_assignment_evidence(
            manifest,
            decoded_frames,
            perceptual_hamming_threshold=configured.perceptual_duplicate_hamming_max,
            max_sequences=configured.max_sequences,
            max_frames=configured.max_frames,
            max_comparisons=configured.max_comparisons,
        )
    )
    return CuratedValidationReport(
        dataset_digest=dataset_digest,
        thresholds=configured,
        metric_policy_digest=metric_policy_digest(),
        split_assignment_evidence=split_evidence,
        state=state,
        frame_findings=tuple(frame_findings),
        sequence_findings=ordered_sequence,
    )


def _frame_finding(
    code: str,
    state: FindingState,
    sequence_id: str,
    frame: CuratedFrameV1,
    observed: str,
    threshold: str,
    related_frame: CuratedFrameV1 | None,
    dataset_digest: str,
    policy_digest: str,
    observed_pixel_digest: str | None = None,
) -> FrameFinding:
    payload = {
        "finding_schema": "frame-finding.v1",
        "dataset_digest": dataset_digest,
        "validation_policy_digest": policy_digest,
        "reason_code": code,
        "sequence_id": sequence_id,
        "frame_id": frame.frame_id,
        "source_digest": frame.source_digest,
        "pixel_digest": frame.pixel_digest,
        "observed_pixel_digest": observed_pixel_digest,
        "observed_value": observed,
        "threshold": threshold,
        "related_frame_id": related_frame.frame_id if related_frame else None,
        "related_source_digest": related_frame.source_digest if related_frame else None,
        "related_pixel_digest": related_frame.pixel_digest if related_frame else None,
    }
    return FrameFinding(
        finding_id=_finding_id(payload),
        state=state,
        reason_code=code,
        frame_id=frame.frame_id,
        sequence_id=sequence_id,
        source_digest=frame.source_digest,
        pixel_digest=frame.pixel_digest,
        observed_pixel_digest=observed_pixel_digest,
        observed_value=observed,
        threshold=threshold,
        related_frame_id=related_frame.frame_id if related_frame else None,
        related_source_digest=related_frame.source_digest if related_frame else None,
        related_pixel_digest=related_frame.pixel_digest if related_frame else None,
    )


def _sequence_finding(
    code: str,
    state: FindingState,
    sequence_id: str,
    related: tuple[str, ...],
    implicated_frames: tuple[ImplicatedFrameEvidence, ...],
    observed: str,
    threshold: str,
    dispositions: Mapping[str, str],
    dataset_digest: str,
    policy_digest: str,
) -> SequenceFinding:
    payload = {
        "finding_schema": "sequence-finding.v1",
        "dataset_digest": dataset_digest,
        "validation_policy_digest": policy_digest,
        "reason_code": code,
        "sequence_id": sequence_id,
        "related_sequence_ids": related,
        "implicated_frames": tuple(frame.model_dump(mode="json") for frame in implicated_frames),
        "observed_value": observed,
        "threshold": threshold,
    }
    finding_id = _finding_id(payload)
    final_state, disposition = _state_with_disposition(state, finding_id, dispositions)
    return SequenceFinding(
        finding_id=finding_id,
        state=final_state,
        reason_code=code,
        sequence_id=sequence_id,
        related_sequence_ids=related,
        implicated_frames=implicated_frames,
        observed_value=observed,
        threshold=threshold,
        human_disposition=disposition,
    )


def _implicated(frame: CuratedFrameV1) -> ImplicatedFrameEvidence:
    return ImplicatedFrameEvidence(
        frame_id=frame.frame_id,
        source_digest=frame.source_digest,
        pixel_digest=frame.pixel_digest,
    )
