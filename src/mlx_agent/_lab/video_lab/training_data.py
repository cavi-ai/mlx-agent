"""Content-addressed training data with caller-selected readiness policy."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from fractions import Fraction
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from video_lab.domain import FrozenModel, canonical_json

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class DatasetMedia(FrozenModel):
    """Media identity and metadata; source paths are not persisted."""

    schema_version: Literal["training-media.v1"] = "training-media.v1"
    sha256: Digest
    media_type: Literal["image/jpeg", "image/png", "image/webp", "video/mp4"]
    byte_size: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    frame_count: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_kind_metadata(self) -> DatasetMedia:
        values = (self.duration_seconds, self.frame_count, self.fps)
        if self.media_type == "video/mp4":
            if any(value is None for value in values):
                raise ValueError("MP4 metadata requires duration, frame_count, and fps")
        elif any(value is not None for value in values):
            raise ValueError("still metadata must not contain video fields")
        return self


class DatasetSample(FrozenModel):
    sample_id: str = Field(min_length=1)
    source_lineage: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    split: Literal["train", "validation", "sealed_holdout"]
    media: DatasetMedia
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    synthetic: bool = False

    @property
    def sample_digest(self) -> str:
        return self.media.sha256


class DatasetPolicy(FrozenModel):
    """Optional composition limits; no character or campaign requirements."""

    min_samples: int = Field(default=1, ge=1)
    min_clip_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_clip_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_synthetic_share: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_duration_range(self) -> DatasetPolicy:
        if (
            self.min_clip_seconds is not None
            and self.max_clip_seconds is not None
            and self.min_clip_seconds > self.max_clip_seconds
        ):
            raise ValueError("min_clip_seconds must not exceed max_clip_seconds")
        return self


class TrainingDataset(FrozenModel):
    schema_version: Literal["training-dataset.v1"] = "training-dataset.v1"
    samples: tuple[DatasetSample, ...]
    policy: DatasetPolicy = Field(default_factory=DatasetPolicy)


class DatasetNotReady(ValueError):
    def __init__(self, reasons: Iterable[str]) -> None:
        self.reason_codes = tuple(sorted(set(reasons)))
        super().__init__(
            "training dataset is not ready: " + ", ".join(self.reason_codes)
        )


class _SampleIdentity(Protocol):
    sample_id: str
    sample_digest: str
    source_lineage: str
    split: str


def dataset_integrity_reasons(samples: Iterable[_SampleIdentity]) -> set[str]:
    """Shared integrity checks independent of any training policy or media backend."""
    items = tuple(samples)
    reasons: set[str] = set()
    if len({sample.sample_id for sample in items}) != len(items):
        reasons.add("duplicate_sample_id")
    if len({sample.sample_digest for sample in items}) != len(items):
        reasons.add("duplicate_content")
    lineages: dict[str, set[str]] = defaultdict(set)
    for sample in items:
        lineages[sample.source_lineage].add(str(sample.split))
    if any(len(splits) > 1 for splits in lineages.values()):
        reasons.add("lineage_split_leakage")
    return reasons


def _readiness_reasons(dataset: TrainingDataset) -> set[str]:
    reasons = dataset_integrity_reasons(dataset.samples)
    if len(dataset.samples) < dataset.policy.min_samples:
        reasons.add("sample_count")
    if not any(sample.split == "train" for sample in dataset.samples):
        reasons.add("training_split_empty")
    for sample in dataset.samples:
        duration = sample.media.duration_seconds
        if duration is None:
            continue
        if (
            dataset.policy.min_clip_seconds is not None
            and duration < dataset.policy.min_clip_seconds
        ) or (
            dataset.policy.max_clip_seconds is not None
            and duration > dataset.policy.max_clip_seconds
        ):
            reasons.add("clip_duration")
    cap = dataset.policy.max_synthetic_share
    if cap is not None and dataset.samples:
        total = sum(
            (Fraction(str(sample.weight)) for sample in dataset.samples), Fraction(0)
        )
        synthetic = sum(
            (
                Fraction(str(sample.weight))
                for sample in dataset.samples
                if sample.synthetic
            ),
            Fraction(0),
        )
        if synthetic > Fraction(str(cap)) * total:
            reasons.add("synthetic_share")
    return reasons


def dataset_digest(dataset: TrainingDataset) -> str:
    payload = dataset.model_dump(mode="json", exclude={"dataset_digest"})
    payload["samples"] = sorted(
        payload["samples"], key=lambda sample: sample["sample_id"]
    )
    return (
        "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    )


class FrozenDataset(TrainingDataset):
    dataset_digest: Digest

    @model_validator(mode="after")
    def validate_snapshot(self) -> FrozenDataset:
        reasons = _readiness_reasons(self)
        if reasons:
            raise DatasetNotReady(reasons)
        if dataset_digest(self) != self.dataset_digest:
            raise ValueError("dataset_digest does not match dataset content")
        return self


def freeze_dataset(dataset: TrainingDataset) -> FrozenDataset:
    """Validate and freeze the media, captions, splits, weights, and readiness policy."""
    reasons = _readiness_reasons(dataset)
    if reasons:
        raise DatasetNotReady(reasons)
    return FrozenDataset(
        **dataset.model_dump(mode="python"),
        dataset_digest=dataset_digest(dataset),
    )
