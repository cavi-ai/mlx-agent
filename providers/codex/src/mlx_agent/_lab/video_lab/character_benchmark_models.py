from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from video_lab.domain import FrozenModel

CHARACTER_BENCHMARK_SCHEMA_VERSION: Literal["character-benchmark.v1"] = "character-benchmark.v1"
CHARACTER_BENCHMARK_PROTOCOL_VERSION: Literal["character-panel.v1"] = "character-panel.v1"
CHARACTER_BENCHMARK_ANALYSIS_VERSION: Literal["character-dirichlet.v1"] = "character-dirichlet.v1"
CHARACTER_POSTERIOR_DRAWS = 65_536
CHARACTER_POSTERIOR_SEED = 73_021

Candidate = Literal["a", "b"]
CandidatePosition = Literal["left", "right"]


class CharacterContextName(StrEnum):
    NEUTRAL_FACE_PORTRAIT = "neutral_face_portrait"
    THREE_QUARTER_PORTRAIT = "three_quarter_portrait"
    PROFILE_VIEW = "profile_view"
    FULL_BODY_NEUTRAL_POSE = "full_body_neutral_pose"
    FULL_BODY_ACTION = "full_body_action"
    STRONG_EXPRESSION = "strong_expression"
    CAMERA_MOTION = "camera_motion"
    CHARACTER_OBJECT_INTERACTION = "character_object_interaction"
    STRONG_DIRECTIONAL_LIGHTING = "strong_directional_lighting"
    PARTIAL_FACIAL_OCCLUSION = "partial_facial_occlusion"
    FORESHORTENED_OR_DIFFICULT_ANGLE = "foreshortened_or_difficult_angle"
    BUSY_ENVIRONMENT_WITH_COMPETING_SUBJECTS = "busy_environment_with_competing_subjects"


class CharacterContextGroup(StrEnum):
    CORE = "core"
    STRESS = "stress"


class IdentityChoice(StrEnum):
    A = "a"
    B = "b"
    BOTH = "both"
    NEITHER = "neither"


class SecondaryChoice(StrEnum):
    A = "a"
    B = "b"
    TIE = "tie"


class SecondaryDimension(StrEnum):
    PROMPT_ADHERENCE = "prompt_adherence"
    MOTION_QUALITY = "motion_quality"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    VISUAL_QUALITY = "visual_quality"


class BaselineDisposition(StrEnum):
    BASELINE_ESTABLISHED = "baseline_established"
    EVALUATOR_UNRELIABLE = "evaluator_unreliable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class IdentityDisposition(StrEnum):
    IDENTITY_RECIPE_QUALIFIED = "identity_recipe_qualified"
    CONTROL_QUALIFIED = "control_qualified"
    INCONCLUSIVE = "inconclusive"


class CharacterBenchmarkContext(FrozenModel):
    ordinal: int = Field(ge=1, le=12)
    name: CharacterContextName
    group: CharacterContextGroup
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None


class CharacterBenchmarkPanelInput(FrozenModel):
    contexts: tuple[CharacterBenchmarkContext, ...]
    seeds: tuple[int, int, int]
    schedule_seed: int

    @model_validator(mode="after")
    def require_frozen_context_matrix(self) -> Self:
        expected_names = tuple(CharacterContextName)
        if len(self.contexts) != len(expected_names):
            raise ValueError("character panel must contain exactly twelve contexts")
        if tuple(context.ordinal for context in self.contexts) != tuple(range(1, 13)):
            raise ValueError("character panel contexts must retain ordinals one through twelve")
        if tuple(context.name for context in self.contexts) != expected_names:
            raise ValueError("character panel contexts must retain the frozen order and names")
        expected_groups = (CharacterContextGroup.CORE,) * 8 + (CharacterContextGroup.STRESS,) * 4
        if tuple(context.group for context in self.contexts) != expected_groups:
            raise ValueError(
                "character panel contexts one through eight are core and nine through twelve stress"
            )
        if len(set(self.seeds)) != 3:
            raise ValueError("character panel seeds must be distinct")
        return self


class CharacterRecipePair(FrozenModel):
    control_digest: str = Field(min_length=1)
    identity_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_recipes(self) -> Self:
        if self.control_digest == self.identity_digest:
            raise ValueError("control and identity recipes must be distinct")
        return self


class CharacterBenchmarkDefinition(FrozenModel):
    id: str = Field(pattern=r"^character_benchmark_[0-9a-f]{64}$")
    schema_version: Literal["character-benchmark.v1"] = CHARACTER_BENCHMARK_SCHEMA_VERSION
    protocol_version: Literal["character-panel.v1"] = CHARACTER_BENCHMARK_PROTOCOL_VERSION
    analysis_version: Literal["character-dirichlet.v1"] = CHARACTER_BENCHMARK_ANALYSIS_VERSION
    character_pack_digest: str = Field(min_length=1)
    stage_2_manifest_digest: str = Field(min_length=1)
    stage_2_evidence_digest: str = Field(min_length=1)
    recipes: CharacterRecipePair
    panel_input: CharacterBenchmarkPanelInput


class CharacterGenerationSpec(FrozenModel):
    id: str = Field(pattern=r"^character_generation_[0-9a-f]{64}$")
    benchmark_id: str = Field(pattern=r"^character_benchmark_[0-9a-f]{64}$")
    context: CharacterBenchmarkContext
    seed: int
    candidate: Candidate
    recipe_digest: str = Field(min_length=1)


class CharacterTrial(FrozenModel):
    id: str = Field(pattern=r"^character_trial_[0-9a-f]{64}$")
    benchmark_id: str = Field(pattern=r"^character_benchmark_[0-9a-f]{64}$")
    ordinal: int = Field(ge=1, le=40)
    context: CharacterBenchmarkContext
    seed: int
    hidden_repeat: bool = False
    reuses_trial_id: str | None = None
    generation_specs: tuple[CharacterGenerationSpec, ...]
    candidate_a_position: CandidatePosition

    @model_validator(mode="after")
    def require_repeat_media_reuse(self) -> Self:
        if self.hidden_repeat:
            if self.reuses_trial_id is None:
                raise ValueError("hidden repeats must identify their primary source trial")
            if self.generation_specs:
                raise ValueError("hidden repeats reuse media and cannot create generation specs")
            if self.reuses_trial_id == self.id:
                raise ValueError("a hidden repeat cannot reuse itself")
            return self
        if self.reuses_trial_id is not None:
            raise ValueError("primary trials cannot reuse another trial")
        if tuple(spec.candidate for spec in self.generation_specs) != ("a", "b"):
            raise ValueError("primary trials require sealed candidate a and b generation specs")
        if any(
            spec.context != self.context or spec.seed != self.seed for spec in self.generation_specs
        ):
            raise ValueError("generation specs must match their trial context and seed")
        return self


class IdentityObservation(FrozenModel):
    benchmark_id: str = Field(pattern=r"^character_benchmark_[0-9a-f]{64}$")
    trial_id: str = Field(pattern=r"^character_trial_[0-9a-f]{64}$")
    choice: IdentityChoice


class SecondaryObservation(FrozenModel):
    benchmark_id: str = Field(pattern=r"^character_benchmark_[0-9a-f]{64}$")
    trial_id: str = Field(pattern=r"^character_trial_[0-9a-f]{64}$")
    dimension: SecondaryDimension
    choice: SecondaryChoice


class CharacterBenchmarkEvidence(FrozenModel):
    definition: CharacterBenchmarkDefinition
    trials: tuple[CharacterTrial, ...]
    identity_observations: tuple[IdentityObservation, ...]
    secondary_observations: tuple[SecondaryObservation, ...] = ()
    media_integrity_complete: bool = True
    provenance_complete: bool = True


class IdentityCounts(FrozenModel):
    a: int = Field(ge=0)
    b: int = Field(ge=0)
    both: int = Field(ge=0)
    neither: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.a + self.b + self.both + self.neither


class IdentityPosterior(FrozenModel):
    counts: IdentityCounts
    control_superiority_probability: float = Field(ge=0, le=1)
    identity_recipe_superiority_probability: float = Field(ge=0, le=1)


class ContextIdentityResult(FrozenModel):
    context: CharacterBenchmarkContext
    posterior: IdentityPosterior
    both_rate: float = Field(ge=0, le=1)
    neither_rate: float = Field(ge=0, le=1)


class GroupIdentityResult(FrozenModel):
    group: CharacterContextGroup
    posterior: IdentityPosterior
    both_rate: float = Field(ge=0, le=1)
    neither_rate: float = Field(ge=0, le=1)


class ContextReversal(FrozenModel):
    context: CharacterBenchmarkContext
    disfavored_recipe: Literal["control", "identity"]
    superiority_probability: float = Field(ge=0, le=1)


class SecondaryCounts(FrozenModel):
    a: int = Field(ge=0)
    b: int = Field(ge=0)
    tie: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.a + self.b + self.tie


class SecondaryDimensionResult(FrozenModel):
    counts: SecondaryCounts
    observation_count: int = Field(ge=0)
    control_superiority_probability: float = Field(ge=0, le=1)
    identity_recipe_superiority_probability: float = Field(ge=0, le=1)


class SecondarySummary(FrozenModel):
    prompt_adherence: SecondaryDimensionResult
    motion_quality: SecondaryDimensionResult
    temporal_consistency: SecondaryDimensionResult
    visual_quality: SecondaryDimensionResult


class CharacterBenchmarkResult(FrozenModel):
    benchmark_id: str = Field(pattern=r"^character_benchmark_[0-9a-f]{64}$")
    analysis_version: Literal["character-dirichlet.v1"] = CHARACTER_BENCHMARK_ANALYSIS_VERSION
    baseline_disposition: BaselineDisposition
    identity_disposition: IdentityDisposition
    posterior: IdentityPosterior
    both_rate: float = Field(ge=0, le=1)
    neither_rate: float = Field(ge=0, le=1)
    context_results: tuple[ContextIdentityResult, ...]
    group_results: tuple[GroupIdentityResult, ...]
    context_reversals: tuple[ContextReversal, ...]
    secondary: SecondarySummary
    repeat_agreements: int = Field(ge=0, le=4)
    identity_break_contexts: tuple[CharacterContextName, ...]
