from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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


class LoraBinding(FrozenModel):
    model_id: str
    revision: str
    checksum: str
    weight: float


class Manifest(FrozenModel):
    schema_version: str = "manifest.v1"
    protocol_version: str = "ab.v1"
    model_id: str
    model_revision: str
    model_checksum: str
    adapter_id: str
    adapter_version: str
    runtime_versions: Mapping[str, str]
    source_checksums: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}))
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    seed: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    num_frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    inference_steps: int = Field(gt=0)
    guidance_scale: float = Field(ge=0)
    scheduler: str
    dtype: str
    quantization: str
    loras: tuple[LoraBinding, ...] = ()
    backend: str
    device: str
    memory_strategy: str
    hardware_fingerprint: str
    os_fingerprint: str

    @field_validator("runtime_versions", "source_checksums")
    @classmethod
    def freeze_runtime_versions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("runtime_versions", "source_checksums")
    def serialize_runtime_versions(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class ExperimentDraft(FrozenModel):
    baseline: Manifest
    variable_path: str
    alternate_value: Any


class Variant(FrozenModel):
    id: str
    experiment_id: str
    label: str
    manifest: Manifest


class ReviewDimension(StrEnum):
    PROMPT_ADHERENCE = "prompt_adherence"
    VISUAL_QUALITY = "visual_quality"
    MOTION_QUALITY = "motion_quality"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    CHARACTER_CONSISTENCY = "character_consistency"


class DisplayPreference(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    TIE = "tie"


class ReviewSession(FrozenModel):
    id: str
    experiment_id: str
    display_positions: tuple[Literal["left", "right"], Literal["left", "right"]] = (
        "left",
        "right",
    )
    created_at: datetime

    @field_validator("display_positions")
    @classmethod
    def validate_display_positions(
        cls, value: tuple[Literal["left", "right"], Literal["left", "right"]]
    ) -> tuple[Literal["left", "right"], Literal["left", "right"]]:
        if value != ("left", "right"):
            raise ValueError("display positions must be left and right")
        return value

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class RubricScore(FrozenModel):
    display_position: Literal["left", "right"]
    dimension: ReviewDimension
    score: int = Field(ge=1, le=5)


class ReviewRevision(FrozenModel):
    id: str
    session_id: str
    preferred_variant_id: str | None
    display_preference: DisplayPreference
    scores: tuple[RubricScore, ...]
    notes: str | None
    created_at: datetime
    revision: int = Field(gt=0)
    supersedes_review_id: str | None = None

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class JobState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    GENERATING = "generating"
    ENCODING = "encoding"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(FrozenModel):
    id: str
    variant_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime


class RunAttemptStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class RunAttempt(FrozenModel):
    id: str
    job_id: str
    number: int = Field(gt=0)
    worker_id: str
    status: RunAttemptStatus
    started_at: datetime
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifact_id: str | None = None

    @field_validator("started_at", "finished_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value)


class RunEvent(FrozenModel):
    id: int
    job_id: str
    attempt_id: str
    kind: str
    payload: dict[str, object]
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class TelemetrySample(FrozenModel):
    id: int | None = None
    attempt_id: str
    recorded_at: datetime
    elapsed_seconds: float = Field(ge=0)
    process_rss_bytes: int = Field(ge=0)
    system_used_bytes: int = Field(ge=0)
    system_available_bytes: int = Field(ge=0)
    swap_used_bytes: int = Field(ge=0)
    thermal_pressure: str | None = None

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class ClaimedJob(FrozenModel):
    job: Job
    attempt: RunAttempt


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_json(value: BaseModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
