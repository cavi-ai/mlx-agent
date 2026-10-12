"""Frozen, trainer-neutral contracts for curated video-frame sequences."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from math import gcd
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_STABLE_ID = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RGB_CHANNEL = Annotated[int, Field(ge=0, le=255)]


class _CuratedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        dumped = self.model_dump(round_trip=True)
        payload = deepcopy(dumped) if deep else dumped
        if update is not None:
            payload.update(update)
        return self.__class__.model_validate(payload)


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized != normalized.strip():
        raise ValueError("text must be nonempty and have no surrounding whitespace")
    return normalized


def _stable_id(value: str) -> str:
    normalized = _normalized_text(value)
    if _STABLE_ID.fullmatch(normalized) is None:
        raise ValueError("must be a lowercase stable identifier")
    return normalized


def _sha256(value: str) -> str:
    normalized = _normalized_text(value)
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError("must be a lowercase SHA-256 hex digest")
    return normalized


def _caption_items(value: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(sorted({_normalized_text(item) for item in value}))


class StructuredCaptionV1(_CuratedModel):
    schema_version: Literal["structured-caption.v1"] = "structured-caption.v1"
    character_token: str
    immutable_attributes: tuple[str, ...] = Field(min_length=1)
    clothing: tuple[str, ...] = ()
    action: str
    pose: str | None = None
    expression: str | None = None
    environment: str | None = None
    lighting: str | None = None
    camera: str | None = None
    style: str | None = None
    omissions: tuple[str, ...] = ()

    @field_validator("character_token")
    @classmethod
    def validate_character_token(cls, value: str) -> str:
        return _stable_id(value)

    @field_validator("immutable_attributes", "clothing", "omissions")
    @classmethod
    def normalize_caption_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _caption_items(value)

    @field_validator(
        "action",
        "pose",
        "expression",
        "environment",
        "lighting",
        "camera",
        "style",
    )
    @classmethod
    def normalize_caption_text(cls, value: str | None) -> str | None:
        return None if value is None else _normalized_text(value)


class CuratedFrameV1(_CuratedModel):
    schema_version: Literal["curated-frame.v1"] = "curated-frame.v1"
    frame_id: str
    frame_index: int = Field(ge=0)
    timestamp_numerator: int = Field(ge=0)
    timestamp_denominator: int = Field(gt=0)
    source_path: str
    source_digest: str
    pixel_digest: str
    review_state: Literal["accepted"]
    reviewer_evidence: tuple[str, ...] = Field(min_length=1)

    @field_validator("frame_id", "reviewer_evidence")
    @classmethod
    def validate_stable_ids(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, tuple):
            return tuple(_stable_id(item) for item in value)
        return _stable_id(value)

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        normalized = _normalized_text(value)
        components = normalized.split("/")
        has_ambiguous_component = any(component in {"", ".", ".."} for component in components)
        if normalized.startswith("/") or has_ambiguous_component:
            raise ValueError("must be a relative, component-wise source path")
        return normalized

    @field_validator("source_digest", "pixel_digest")
    @classmethod
    def validate_digests(cls, value: str) -> str:
        return _sha256(value)


class CuratedSequenceV1(_CuratedModel):
    schema_version: Literal["curated-sequence.v1"] = "curated-sequence.v1"
    sequence_id: str
    identity_id: str
    source_lineage: str
    split: Literal["train", "validation", "holdout"]
    caption: StructuredCaptionV1
    frames: tuple[CuratedFrameV1, ...] = Field(min_length=1)

    @field_validator("sequence_id", "identity_id", "source_lineage")
    @classmethod
    def validate_stable_ids(cls, value: str) -> str:
        return _stable_id(value)

    @model_validator(mode="after")
    def validate_frames(self) -> CuratedSequenceV1:
        if self.identity_id != self.caption.character_token:
            raise ValueError("identity ID must exactly match the caption character token")
        frame_ids = tuple(frame.frame_id for frame in self.frames)
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame IDs must be unique within a sequence")
        if tuple(frame.frame_index for frame in self.frames) != tuple(range(len(self.frames))):
            raise ValueError("frames must use contiguous, ordered frame indices")

        previous_numerator = -1
        previous_denominator = 1
        for frame in self.frames:
            current_time = frame.timestamp_numerator * previous_denominator
            previous_time = previous_numerator * frame.timestamp_denominator
            if current_time <= previous_time:
                raise ValueError("frame timestamps must be strictly increasing rational values")
            previous_numerator = frame.timestamp_numerator
            previous_denominator = frame.timestamp_denominator
        return self


class DatasetSplitPolicyV1(_CuratedModel):
    schema_version: Literal["dataset-split-policy.v1"] = "dataset-split-policy.v1"
    policy_version: str
    seed: int = Field(ge=0)
    train_numerator: int = Field(gt=0)
    validation_numerator: int = Field(gt=0)
    holdout_numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        return _stable_id(value)

    @model_validator(mode="after")
    def validate_partition(self) -> DatasetSplitPolicyV1:
        total = self.train_numerator + self.validation_numerator + self.holdout_numerator
        if total != self.denominator:
            raise ValueError("split numerators must exactly sum to the denominator")
        ratio_divisor = gcd(
            self.train_numerator,
            self.validation_numerator,
            self.holdout_numerator,
            self.denominator,
        )
        if ratio_divisor != 1:
            raise ValueError("split policy ratios must be reduced")
        return self


class SequenceTransformRecipeV1(_CuratedModel):
    schema_version: Literal["sequence-transform-recipe.v1"] = "sequence-transform-recipe.v1"
    recipe_id: str
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)
    output_frame_count: int = Field(gt=0)
    temporal_stride_numerator: int = Field(gt=0)
    temporal_stride_denominator: int = Field(gt=0)
    resize_mode: Literal["center_crop", "fit_pad"] = "center_crop"
    interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"]
    padding_color: tuple[_RGB_CHANNEL, _RGB_CHANNEL, _RGB_CHANNEL]
    seed: int = Field(ge=0)

    @field_validator("recipe_id")
    @classmethod
    def validate_recipe_id(cls, value: str) -> str:
        return _stable_id(value)


class CuratedDatasetManifestV1(_CuratedModel):
    schema_version: Literal["curated-dataset-manifest.v1"] = "curated-dataset-manifest.v1"
    dataset_id: str
    sequences: tuple[CuratedSequenceV1, ...] = Field(min_length=1)
    split_policy: DatasetSplitPolicyV1
    transform_recipe: SequenceTransformRecipeV1

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _stable_id(value)

    @model_validator(mode="after")
    def validate_sequences(self) -> CuratedDatasetManifestV1:
        sequence_ids = tuple(sequence.sequence_id for sequence in self.sequences)
        if sequence_ids != tuple(sorted(sequence_ids)):
            raise ValueError("sequences must be ordered by sequence ID")
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("sequence IDs must be unique")

        frame_ids = tuple(
            frame.frame_id for sequence in self.sequences for frame in sequence.frames
        )
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame IDs must be globally unique")
        return self


def render_caption_v1(caption: StructuredCaptionV1) -> str:
    sections: list[tuple[str, str]] = [
        ("token", caption.character_token),
        ("immutable identity", ", ".join(caption.immutable_attributes)),
    ]
    if caption.clothing:
        sections.append(("clothing", ", ".join(caption.clothing)))

    action_pose = caption.action if caption.pose is None else f"{caption.action}; {caption.pose}"
    sections.append(("action/pose", action_pose))
    for label, value in (
        ("expression", caption.expression),
        ("environment", caption.environment),
        ("lighting", caption.lighting),
        ("camera", caption.camera),
        ("style", caption.style),
    ):
        if value is not None:
            sections.append((label, value))
    if caption.omissions:
        sections.append(("omissions", ", ".join(caption.omissions)))
    return "; ".join(f"{label}: {value}" for label, value in sections)
