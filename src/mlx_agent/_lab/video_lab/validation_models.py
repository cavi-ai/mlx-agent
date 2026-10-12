from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from video_lab.domain import FrozenModel, Manifest, _aware_utc, canonical_json

VALIDATION_CAMPAIGN_SCHEMA_VERSION: Literal["validation-campaign.v1"] = "validation-campaign.v1"
VALIDATION_PANEL_SCHEMA_VERSION: Literal["validation-panel.v1"] = "validation-panel.v1"
VALIDATION_PROTOCOL_VERSION: Literal["balanced-validation.v1"] = "balanced-validation.v1"
VALIDATION_ANALYSIS_VERSION: Literal["dirichlet-validation.v1"] = "dirichlet-validation.v1"


class ValidationIneligible(ValueError):
    """Raised when Stage 1 evidence cannot enter the fixed validation protocol."""


class ValidationCampaignState(StrEnum):
    DRAFT = "draft"
    PREREGISTERED = "preregistered"
    GENERATING = "generating"
    READY = "ready"
    JUDGING = "judging"
    ANALYZING = "analyzing"
    QUALIFIED = "qualified"
    NOT_QUALIFIED = "not_qualified"
    INCONCLUSIVE = "inconclusive"


class ValidationArchetype(StrEnum):
    PORTRAIT_MOTION = "portrait_motion"
    FULL_BODY_ACTION = "full_body_action"
    CAMERA_MOTION = "camera_motion"
    ENVIRONMENT_INTERACTION = "environment_interaction"


class ValidationChoice(StrEnum):
    A = "a"
    B = "b"
    TIE = "tie"


class ValidationOutcome(StrEnum):
    WIN = "win"
    TIE = "tie"
    LOSS = "loss"


class ValidationDisposition(StrEnum):
    QUALIFIED = "qualified"
    NOT_QUALIFIED = "not_qualified"
    INCONCLUSIVE = "inconclusive"


class ValidationReason(StrEnum):
    ALL_GATES_PASSED = "all_gates_passed"
    CHALLENGER_REJECTION = "challenger_rejection"
    ARCHETYPE_REVERSAL = "archetype_reversal"
    REPEAT_RELIABILITY_FAILED = "repeat_reliability_failed"
    INCOMPLETE_PRIMARY_EVIDENCE = "incomplete_primary_evidence"
    INSUFFICIENT_PRIMARY_EVIDENCE = "insufficient_primary_evidence"


class ValidationCandidates(FrozenModel):
    winner: float
    lower: float
    higher: float
    distant: float

    @model_validator(mode="after")
    def require_three_distinct_challengers(self) -> Self:
        values = (self.winner, self.lower, self.higher, self.distant)
        if len(set(values)) != len(values):
            raise ValueError("validation candidates must be distinct")
        return self

    @property
    def challengers(self) -> tuple[float, float, float]:
        return (self.lower, self.higher, self.distant)


class ValidationPrompt(FrozenModel):
    archetype: ValidationArchetype
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    seeds: tuple[int, int, int]

    @field_validator("negative_prompt")
    @classmethod
    def normalize_negative_prompt(cls, value: str | None) -> str | None:
        return value or None

    @model_validator(mode="after")
    def require_distinct_seeds(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("validation prompt seeds must be distinct")
        return self


class ValidationPanelInput(FrozenModel):
    prompts: tuple[ValidationPrompt, ValidationPrompt, ValidationPrompt, ValidationPrompt]
    schedule_seed: int

    @model_validator(mode="after")
    def require_each_archetype_once(self) -> Self:
        if set(prompt.archetype for prompt in self.prompts) != set(ValidationArchetype):
            raise ValueError("validation panel input must contain every archetype exactly once")
        return self


class ValidationCampaignDefinition(FrozenModel):
    id: str = Field(min_length=1)
    schema_version: Literal["validation-campaign.v1"] = VALIDATION_CAMPAIGN_SCHEMA_VERSION
    stage_1_study_id: str = Field(min_length=1)
    stage_1_evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_manifest: Manifest
    candidates: ValidationCandidates
    panel_input: ValidationPanelInput
    protocol_version: Literal["balanced-validation.v1"] = VALIDATION_PROTOCOL_VERSION
    analysis_version: Literal["dirichlet-validation.v1"] = VALIDATION_ANALYSIS_VERSION
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class ValidationTrial(FrozenModel):
    id: str = Field(min_length=1)
    ordinal: int = Field(gt=0)
    archetype: ValidationArchetype
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    seed: int
    challenger: float
    hidden_repeat: bool = False
    reuses_trial_id: str | None = None
    winner_display_position: Literal["left", "right"]

    @model_validator(mode="after")
    def require_repeat_provenance(self) -> Self:
        if self.hidden_repeat != (self.reuses_trial_id is not None):
            raise ValueError("only hidden repeats may reuse a primary trial")
        if self.reuses_trial_id == self.id:
            raise ValueError("a repeat trial cannot reuse itself")
        return self


class ValidationPanel(FrozenModel):
    campaign_id: str = Field(min_length=1)
    schema_version: Literal["validation-panel.v1"] = VALIDATION_PANEL_SCHEMA_VERSION
    trials: tuple[ValidationTrial, ...]
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @property
    def challengers(self) -> tuple[float, ...]:
        return tuple(sorted({trial.challenger for trial in self.trials}))


class ValidationEvent(FrozenModel):
    campaign_id: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    previous_state: ValidationCampaignState
    new_state: ValidationCampaignState
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def require_state_transition(self) -> Self:
        if self.previous_state is self.new_state:
            raise ValueError("validation event must change state")
        return self


class ValidationTrialBinding(FrozenModel):
    campaign_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    experiment_id: str | None = None
    review_id: str | None = None
    reused_from_trial_id: str | None = None


class ValidationExperimentBinding(FrozenModel):
    campaign_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)


class ValidationJudgment(FrozenModel):
    campaign_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    ordinal: int = Field(gt=0)
    archetype: ValidationArchetype
    challenger: float
    seed: int
    hidden_repeat: bool
    winner_display_position: Literal["left", "right"]
    choice: ValidationChoice
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @property
    def outcome(self) -> ValidationOutcome:
        if self.choice is ValidationChoice.TIE:
            return ValidationOutcome.TIE
        winner_is_left = self.winner_display_position == "left"
        if (self.choice is ValidationChoice.A) == winner_is_left:
            return ValidationOutcome.WIN
        return ValidationOutcome.LOSS


class OutcomeCounts(FrozenModel):
    wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.losses


class ChallengerPosterior(FrozenModel):
    challenger: float | None = None
    counts: OutcomeCounts
    superiority_probability: float = Field(ge=0, le=1)
    rejection_probability: float = Field(ge=0, le=1)


class ArchetypeReversal(FrozenModel):
    archetype: ValidationArchetype
    counts: OutcomeCounts
    reversal_probability: float = Field(ge=0, le=1)


class ValidationResult(FrozenModel):
    campaign_id: str = Field(min_length=1)
    disposition: ValidationDisposition
    reason: ValidationReason
    challengers: tuple[ChallengerPosterior, ...]
    archetype_reversals: tuple[ArchetypeReversal, ...]
    repeat_agreements: int = Field(ge=0, le=4)


def validation_panel_digest(campaign_id: str, trials: tuple[ValidationTrial, ...]) -> str:
    payload = {
        "schema_version": VALIDATION_PANEL_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "trials": [trial.model_dump(mode="json") for trial in trials],
    }
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"
