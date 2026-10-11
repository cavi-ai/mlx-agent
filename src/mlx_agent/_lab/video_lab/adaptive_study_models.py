from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from video_lab.domain import FrozenModel, Manifest, _aware_utc, canonical_json
from video_lab.preference_optimizer import (
    AcquisitionScore,
    CandidatePair,
    OptimizerConfig,
    OptimizerDecision,
    PosteriorSummary,
    PreferenceObservation,
    PreferenceOutcome,
    SearchDisposition,
    StoppingDecision,
    StopReason,
)

ADAPTIVE_STUDY_SCHEMA_VERSION: Literal["adaptive-study.v1"] = "adaptive-study.v1"
ADAPTIVE_STUDY_INPUT_SCHEMA_VERSION: Literal["adaptive-study-input.v1"] = "adaptive-study-input.v1"
ADAPTIVE_STUDY_EVENT_SCHEMA_VERSION: Literal["adaptive-study-event.v1"] = "adaptive-study-event.v1"
ADAPTIVE_STUDY_OWNER_BINDING_SCHEMA_VERSION: Literal["adaptive-study-owner-binding.v1"] = (
    "adaptive-study-owner-binding.v1"
)
ADAPTIVE_OBSERVATION_SCHEMA_VERSION: Literal["adaptive-observation.v1"] = "adaptive-observation.v1"
ADAPTIVE_OBSERVATION_PROVENANCE_SCHEMA_VERSION: Literal["adaptive-observation-provenance.v1"] = (
    "adaptive-observation-provenance.v1"
)
ADAPTIVE_OPTIMIZER_STEP_SCHEMA_VERSION: Literal["adaptive-optimizer-step.v1"] = (
    "adaptive-optimizer-step.v1"
)

Digest = str
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Mapping[str, JsonValue] | tuple[JsonValue, ...]


def _sha256(value: dict[str, Any]) -> Digest:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def adaptive_observation_provenance_digest(
    *,
    observation: Mapping[str, Any],
    optimizer_output_digest: Digest,
    experiment: Mapping[str, Any],
    review: Mapping[str, Any],
    display_mapping: Mapping[str, Any],
) -> Digest:
    return _sha256(
        {
            "schema_version": ADAPTIVE_OBSERVATION_PROVENANCE_SCHEMA_VERSION,
            "observation": dict(observation),
            "optimizer_output_digest": optimizer_output_digest,
            "experiment": dict(experiment),
            "review": dict(review),
            "display_mapping": dict(display_mapping),
        }
    )


def adaptive_study_input_fingerprint(
    base_manifest: Manifest, optimizer_config: OptimizerConfig
) -> Digest:
    return _sha256(
        {
            "schema_version": ADAPTIVE_STUDY_INPUT_SCHEMA_VERSION,
            "prompt": base_manifest.prompt,
            "negative_prompt": base_manifest.negative_prompt or None,
            "seed": base_manifest.seed,
            "base_manifest": base_manifest.model_dump(mode="json"),
            "optimizer_config": optimizer_config.model_dump(mode="json"),
        }
    )


def adaptive_optimizer_input_digest(
    optimizer_config: OptimizerConfig,
    observations: tuple[PreferenceObservation, ...],
) -> Digest:
    return _sha256(
        {
            "schema_version": "adaptive-optimizer-input.v1",
            "optimizer_config": optimizer_config.model_dump(mode="json"),
            "observations": [item.model_dump(mode="json") for item in observations],
        }
    )


def adaptive_optimizer_output_digest(decision: OptimizerDecision) -> Digest:
    return _sha256(
        {
            "schema_version": "adaptive-optimizer-output.v1",
            "posterior": decision.posterior.model_dump(mode="json"),
            "acquisition_ledger": [
                item.model_dump(mode="json") for item in decision.candidate_scores
            ],
            "selected_pair": (
                None
                if decision.selected_pair is None
                else decision.selected_pair.model_dump(mode="json")
            ),
            "stopping": decision.stopping.model_dump(mode="json"),
        }
    )


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


class AdaptiveStudyState(StrEnum):
    DRAFT = "draft"
    SEARCH_READY = "search_ready"
    GENERATING_PAIR = "generating_pair"
    AWAITING_PREFERENCE = "awaiting_preference"
    UPDATING_POSTERIOR = "updating_posterior"
    SEARCH_CONVERGED = "search_converged"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    VALIDATING = "validating"
    COMPLETED = "completed"
    PAUSED = "paused"
    OPTIMIZER_FAILED = "optimizer_failed"


ADAPTIVE_STUDY_TRANSITIONS: Mapping[AdaptiveStudyState, frozenset[AdaptiveStudyState]] = (
    MappingProxyType(
        {
            AdaptiveStudyState.DRAFT: frozenset({AdaptiveStudyState.SEARCH_READY}),
            AdaptiveStudyState.SEARCH_READY: frozenset(
                {
                    AdaptiveStudyState.GENERATING_PAIR,
                    AdaptiveStudyState.PAUSED,
                    AdaptiveStudyState.OPTIMIZER_FAILED,
                }
            ),
            AdaptiveStudyState.GENERATING_PAIR: frozenset(
                {AdaptiveStudyState.AWAITING_PREFERENCE, AdaptiveStudyState.PAUSED}
            ),
            AdaptiveStudyState.AWAITING_PREFERENCE: frozenset(
                {AdaptiveStudyState.UPDATING_POSTERIOR, AdaptiveStudyState.PAUSED}
            ),
            AdaptiveStudyState.UPDATING_POSTERIOR: frozenset(
                {
                    AdaptiveStudyState.GENERATING_PAIR,
                    AdaptiveStudyState.SEARCH_CONVERGED,
                    AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED,
                    AdaptiveStudyState.PAUSED,
                    AdaptiveStudyState.OPTIMIZER_FAILED,
                }
            ),
            AdaptiveStudyState.SEARCH_CONVERGED: frozenset({AdaptiveStudyState.VALIDATING}),
            AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED: frozenset({AdaptiveStudyState.VALIDATING}),
            AdaptiveStudyState.VALIDATING: frozenset(
                {AdaptiveStudyState.COMPLETED, AdaptiveStudyState.PAUSED}
            ),
            AdaptiveStudyState.COMPLETED: frozenset(),
            AdaptiveStudyState.PAUSED: frozenset(
                {
                    AdaptiveStudyState.SEARCH_READY,
                    AdaptiveStudyState.GENERATING_PAIR,
                    AdaptiveStudyState.AWAITING_PREFERENCE,
                    AdaptiveStudyState.UPDATING_POSTERIOR,
                    AdaptiveStudyState.VALIDATING,
                }
            ),
            AdaptiveStudyState.OPTIMIZER_FAILED: frozenset(
                {
                    AdaptiveStudyState.SEARCH_READY,
                    AdaptiveStudyState.UPDATING_POSTERIOR,
                    AdaptiveStudyState.PAUSED,
                }
            ),
        }
    )
)


class AdaptiveStudyDefinition(FrozenModel):
    id: str = Field(min_length=1)
    schema_version: Literal["adaptive-study.v1"] = ADAPTIVE_STUDY_SCHEMA_VERSION
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    seed: int
    base_manifest: Manifest
    optimizer_config: OptimizerConfig
    input_fingerprint: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime

    @field_validator("negative_prompt")
    @classmethod
    def normalize_negative_prompt(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def validate_frozen_inputs(self) -> Self:
        if self.prompt != self.base_manifest.prompt:
            raise ValueError("study prompt must match the base manifest")
        if (self.negative_prompt or "") != self.base_manifest.negative_prompt:
            raise ValueError("study negative prompt must match the base manifest")
        if self.seed != self.base_manifest.seed:
            raise ValueError("study seed must match the base manifest")
        expected = adaptive_study_input_fingerprint(self.base_manifest, self.optimizer_config)
        if self.input_fingerprint != expected:
            raise ValueError("study input fingerprint does not match its immutable inputs")
        return self


class AdaptiveStudyEvent(FrozenModel):
    study_id: str = Field(min_length=1)
    schema_version: Literal["adaptive-study-event.v1"] = ADAPTIVE_STUDY_EVENT_SCHEMA_VERSION
    sequence: int = Field(gt=0)
    previous_state: AdaptiveStudyState
    new_state: AdaptiveStudyState
    payload: Mapping[str, JsonValue] = Field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime

    @field_validator("payload")
    @classmethod
    def freeze_payload(cls, value: object) -> Mapping[str, JsonValue]:
        frozen = _freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise ValueError("event payload must be a mapping")
        return frozen

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, JsonValue]) -> dict[str, object]:
        return {key: _thaw_json(item) for key, item in value.items()}

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def state_must_change(self) -> Self:
        if self.previous_state is self.new_state:
            raise ValueError("study event must change state")
        return self


class AdaptiveObservation(FrozenModel):
    study_id: str = Field(min_length=1)
    schema_version: Literal["adaptive-observation.v1"] = ADAPTIVE_OBSERVATION_SCHEMA_VERSION
    ordinal: int = Field(gt=0)
    optimizer_step_ordinal: int = Field(gt=0)
    pair: CandidatePair
    experiment_id: str = Field(min_length=1)
    review_id: str = Field(min_length=1)
    display_mapping_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: PreferenceOutcome
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def require_canonical_pair(self) -> Self:
        if self.pair.value_a >= self.pair.value_b:
            raise ValueError("observation pair must be in ascending canonical order")
        if self.optimizer_step_ordinal != self.ordinal:
            raise ValueError("observation must consume the optimizer step at the same ordinal")
        return self

    @property
    def preference_observation(self) -> PreferenceObservation:
        return PreferenceObservation(
            value_a=self.pair.value_a,
            value_b=self.pair.value_b,
            outcome=self.outcome,
        )


class AdaptiveObservationRequest(FrozenModel):
    study_id: str = Field(min_length=1)
    ordinal: int = Field(gt=0)
    review_id: str = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class AdaptiveOptimizerStep(FrozenModel):
    study_id: str = Field(min_length=1)
    schema_version: Literal["adaptive-optimizer-step.v1"] = ADAPTIVE_OPTIMIZER_STEP_SCHEMA_VERSION
    ordinal: int = Field(gt=0)
    input_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    output_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    posterior: PosteriorSummary
    acquisition_ledger: tuple[AcquisitionScore, ...]
    selected_pair: CandidatePair | None
    stop_reason: StopReason
    disposition: SearchDisposition | None = None
    comparison_count: int = Field(ge=0)
    maximum_information_bits: float = Field(ge=0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def validate_complete_decision(self) -> Self:
        if self.ordinal != self.comparison_count + 1:
            raise ValueError("optimizer step ordinal must be one greater than comparison count")
        if not isfinite(self.maximum_information_bits):
            raise ValueError("maximum information must be finite")
        posterior = self.posterior
        cardinality = len(posterior.grid)
        if (
            cardinality == 0
            or len(posterior.mean) != cardinality
            or len(posterior.covariance) != cardinality
            or any(len(row) != cardinality for row in posterior.covariance)
            or len(posterior.argmax_probability) != cardinality
        ):
            raise ValueError("posterior arrays must match the non-empty guidance grid")
        posterior_values = (
            *posterior.grid,
            *posterior.mean,
            *(value for row in posterior.covariance for value in row),
            *posterior.argmax_probability,
            posterior.recommendation,
            *posterior.concentration_interval,
            posterior.concentration_probability,
        )
        if any(not isfinite(value) for value in posterior_values):
            raise ValueError("posterior values must be finite")
        if any(score.pair.value_a >= score.pair.value_b for score in self.acquisition_ledger):
            raise ValueError("acquisition ledger pairs must be canonical")
        if any(not isfinite(score.expected_information_bits) for score in self.acquisition_ledger):
            raise ValueError("acquisition scores must be finite")
        ledger_pairs = tuple(
            (score.pair.value_a, score.pair.value_b) for score in self.acquisition_ledger
        )
        if len(set(ledger_pairs)) != len(ledger_pairs):
            raise ValueError("acquisition ledger pairs must be unique")
        posterior_digest = self.posterior.content_digest
        if not (
            posterior_digest.startswith("sha256:")
            and len(posterior_digest) == 71
            and set(posterior_digest[7:]) <= set("0123456789abcdef")
        ):
            raise ValueError("posterior content digest must use the sha256:<hex> form")
        if (
            self.selected_pair is not None
            and self.selected_pair.value_a >= self.selected_pair.value_b
        ):
            raise ValueError("selected pair must be in ascending canonical order")
        if self.stop_reason is StopReason.CONTINUE and self.selected_pair is None:
            raise ValueError("continuing optimizer step must select a pair")
        if self.stop_reason is not StopReason.CONTINUE and self.selected_pair is not None:
            raise ValueError("stopped optimizer step must not select a pair")
        if self.stop_reason is StopReason.SEARCH_BUDGET_EXHAUSTED:
            if self.disposition is not SearchDisposition.INCONCLUSIVE_WITHIN_BUDGET:
                raise ValueError("budget exhaustion requires inconclusive disposition")
        elif self.disposition is not None:
            raise ValueError("only budget exhaustion has a search disposition")
        if self.output_digest != adaptive_optimizer_output_digest(self.decision):
            raise ValueError("optimizer output digest does not match the full decision")
        return self

    @property
    def decision(self) -> OptimizerDecision:
        return OptimizerDecision(
            posterior=self.posterior,
            candidate_scores=self.acquisition_ledger,
            selected_pair=self.selected_pair,
            stopping=StoppingDecision(
                stop=self.stop_reason is not StopReason.CONTINUE,
                reason=self.stop_reason,
                disposition=self.disposition,
                comparison_count=self.comparison_count,
                maximum_information_bits=self.maximum_information_bits,
            ),
        )
