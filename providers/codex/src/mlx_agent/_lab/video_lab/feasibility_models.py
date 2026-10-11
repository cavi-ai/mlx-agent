from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any

from pydantic import Field, field_serializer, field_validator, model_validator

from video_lab.domain import FrozenModel, _aware_utc

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Mapping[str, JsonValue] | tuple[JsonValue, ...]


class FeasibilityEvidenceError(ValueError):
    """Base error for evidence that cannot be frozen or verified."""


class FeasibilityStage(StrEnum):
    RESOLVE_SOURCES = "resolve_sources"
    INVENTORY_SOURCES = "inventory_sources"
    STORAGE_PREFLIGHT = "storage_preflight"
    DOWNLOAD_SOURCES = "download_sources"
    LOAD_TEXT_STACK = "load_text_stack"
    LOAD_VAE = "load_vae"
    LOAD_HIGH_TRANSFORMER = "load_high_transformer"
    LOAD_LOW_TRANSFORMER = "load_low_transformer"
    ASSEMBLE_PIPELINE = "assemble_pipeline"
    GENERATE_MINIMUM_1 = "generate_minimum_1"
    GENERATE_MINIMUM_2 = "generate_minimum_2"
    QUALIFY_STANDARD = "qualify_standard"
    VALIDATE_CONTROLLED_PAIR = "validate_controlled_pair"
    PUBLISH_VERDICT = "publish_verdict"


class StageOutcome(StrEnum):
    PASSED = "passed"
    INCOMPATIBLE = "incompatible"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INTERRUPTED = "interrupted"
    INVALID_EVIDENCE = "invalid_evidence"
    UNEXPECTED_FAILURE = "unexpected_failure"


class SupportVerdict(StrEnum):
    PENDING = "pending"
    INCOMPATIBLE = "incompatible"
    RESOURCE_LIMITED = "resource_limited"
    EXPERIMENTAL_LOCAL_HYBRID = "experimental_local_hybrid"
    STANDARD_LAB_QUALIFIED = "standard_lab_qualified"


def _freeze_json(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class StageEvidence(FrozenModel):
    measurements: Mapping[str, float] = Field(default_factory=lambda: MappingProxyType({}))
    details: Mapping[str, JsonValue] = Field(default_factory=lambda: MappingProxyType({}))

    @field_validator("measurements")
    @classmethod
    def freeze_measurements(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        copied = {str(key): float(measurement) for key, measurement in value.items()}
        if any(measurement < 0 or not isfinite(measurement) for measurement in copied.values()):
            raise ValueError("measurements must be finite and non-negative")
        return MappingProxyType(copied)

    @field_validator("details")
    @classmethod
    def freeze_details(cls, value: object) -> Mapping[str, JsonValue]:
        frozen = _freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise ValueError("details must be a mapping")
        return frozen

    @field_serializer("measurements")
    def serialize_measurements(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    @field_serializer("details")
    def serialize_details(self, value: Mapping[str, JsonValue]) -> dict[str, object]:
        return {key: _thaw_json(item) for key, item in value.items()}


class StageResult(FrozenModel):
    outcome: StageOutcome
    evidence: StageEvidence = Field(default_factory=StageEvidence)
    diagnostic_code: str | None = Field(default=None, min_length=1, max_length=64)
    diagnostic_message: str | None = Field(default=None, min_length=1, max_length=1000)

    @classmethod
    def passed(cls, evidence: StageEvidence | None = None) -> StageResult:
        return cls(outcome=StageOutcome.PASSED, evidence=evidence or StageEvidence())


class FeasibilitySourceIdentity(FrozenModel):
    id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("id", "revision", "checksum", mode="before")
    @classmethod
    def strip_identity(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class FeasibilityManifestProfiles(FrozenModel):
    minimum: Mapping[str, JsonValue]
    standard: Mapping[str, JsonValue]

    @field_validator("minimum", "standard", mode="before")
    @classmethod
    def freeze_profile(cls, value: object) -> Mapping[str, JsonValue]:
        frozen = _freeze_json(_normalize_identity_json(value))
        if not isinstance(frozen, Mapping) or not frozen:
            raise ValueError("manifest profiles must be non-empty mappings")
        return frozen

    @field_serializer("minimum", "standard")
    def serialize_profile(self, value: Mapping[str, JsonValue]) -> dict[str, object]:
        return {key: _thaw_json(item) for key, item in value.items()}


class FeasibilityConfiguration(FrozenModel):
    adapter_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    sources: Mapping[str, FeasibilitySourceIdentity]
    runtime_versions: Mapping[str, str]
    hardware_fingerprint: str = Field(min_length=1)
    os_fingerprint: str = Field(min_length=1)
    manifest_profiles: FeasibilityManifestProfiles
    memory_strategy: str = Field(min_length=1)

    @field_validator(
        "adapter_id",
        "protocol_version",
        "hardware_fingerprint",
        "os_fingerprint",
        "memory_strategy",
        mode="before",
    )
    @classmethod
    def strip_identity(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("adapter_id")
    @classmethod
    def canonicalize_adapter_id(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_source_keys(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = {_strip_required_string(key): item for key, item in value.items()}
        if not normalized:
            raise ValueError("sources must contain named exact identities")
        return normalized

    @field_validator("sources")
    @classmethod
    def freeze_sources(
        cls, value: Mapping[str, FeasibilitySourceIdentity]
    ) -> Mapping[str, FeasibilitySourceIdentity]:
        return MappingProxyType(dict(value))

    @field_validator("runtime_versions", mode="before")
    @classmethod
    def freeze_runtime_versions(cls, value: object) -> Mapping[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError("runtime_versions must be a mapping")
        copied = {
            _strip_required_string(key): _strip_required_string(version)
            for key, version in value.items()
        }
        if not copied:
            raise ValueError("runtime_versions must contain named versions")
        return MappingProxyType(copied)

    @field_serializer("sources", "runtime_versions")
    def serialize_mapping(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)


def _strip_required_string(value: object) -> str:
    if not isinstance(value, str) or not (stripped := value.strip()):
        raise ValueError("identity strings must not be blank")
    return stripped


def _normalize_identity_json(value: object) -> object:
    if isinstance(value, str):
        return _strip_required_string(value)
    if isinstance(value, Mapping):
        return {
            _strip_required_string(key): (
                ""
                if _strip_required_string(key) == "negative_prompt" and item == ""
                else _normalize_identity_json(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_identity_json(item) for item in value]
    return value


class FeasibilityRun(FrozenModel):
    id: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: Mapping[str, JsonValue]
    verdict: SupportVerdict = SupportVerdict.PENDING
    created_at: datetime
    updated_at: datetime

    @field_validator("configuration")
    @classmethod
    def freeze_configuration(cls, value: object) -> Mapping[str, JsonValue]:
        frozen = _freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise ValueError("configuration must be a mapping")
        return frozen

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @field_serializer("configuration")
    def serialize_configuration(self, value: Mapping[str, JsonValue]) -> dict[str, object]:
        return {key: _thaw_json(item) for key, item in value.items()}

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> FeasibilityRun:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class StageAttempt(FrozenModel):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    stage: FeasibilityStage
    ordinal: int = Field(gt=0)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: StageOutcome | None = None
    started_at: datetime
    finished_at: datetime | None = None
    evidence: StageEvidence | None = None
    diagnostic_code: str | None = Field(default=None, min_length=1, max_length=64)
    diagnostic_message: str | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("started_at", "finished_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> StageAttempt:
        terminal_values = (
            self.outcome,
            self.finished_at,
            self.evidence,
            self.diagnostic_code,
            self.diagnostic_message,
        )
        if self.outcome is None and any(value is not None for value in terminal_values[1:]):
            raise ValueError("active attempts cannot contain terminal evidence")
        if self.outcome is not None and self.finished_at is None:
            raise ValueError("finished attempts require finished_at")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self
