"""Trainer-neutral, immutable contracts for character-training inputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence, Set
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal, cast

from pydantic import (
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from video_lab.domain import FrozenModel, canonical_json
from video_lab.trainer_protocol import LoRAConfiguration
from video_lab.training_data import DatasetMedia

TRAINING_DATASET_SCHEMA_VERSION = "character-training-dataset.v1"
TRAINING_RECIPE_SCHEMA_VERSION = "character-training-recipe.v1"
TRAINING_MEDIA_SCHEMA_VERSION = "character-training-media.v1"
TRAINING_RECIPE_VALIDATION_SCHEMA_VERSION = "character-training-recipe-validation.v1"
TRAINING_RUN_SCHEMA_VERSION = "character-training-run.v1"
TRAINING_IMPORT_SCHEMA_VERSION = "character-training-import.v1"
TRAINING_CONFIRMATION_SCHEMA_VERSION = "character-training-confirmation.v1"
TRAINING_ORCHESTRATION_OPERATION_SCHEMA_VERSION = "character-training-orchestration-operation.v1"

Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class TrainingOperationKind(StrEnum):
    PROBE = "probe"
    CALIBRATION = "calibration"
    START = "start"
    RESUME = "resume"
    CANCEL = "cancel"


class TrainingOperationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingOperationIdentity(FrozenModel):
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    recipe_digest: Sha256Digest
    dataset_digest: Sha256Digest
    backend_digest: Sha256Digest
    runtime_digest: Sha256Digest
    material_digest: Sha256Digest
    source_digest: Sha256Digest


class ProbeOperationRequest(TrainingOperationIdentity):
    pass


class CalibrationOperationRequest(TrainingOperationIdentity):
    probe_operation_id: str = Field(min_length=1)
    capability_digest: Sha256Digest
    steps: int = Field(ge=0)
    trainable_targets: tuple[str, ...] = Field(min_length=1)
    lora: LoRAConfiguration
    seed: int


class StartOperationRequest(TrainingOperationIdentity):
    preregistration_id: str = Field(min_length=1)
    preregistration_request_digest: Sha256Digest
    checkpoint_steps: tuple[int, ...] = Field(min_length=1)
    checkpoint_fractions: tuple[Literal[0.25, 0.5, 0.75, 1.0], ...] = Field(min_length=1)  # type: ignore[valid-type]
    material_artifact_ids: Mapping[str, str]


class ResumeInventoryArtifact(FrozenModel):
    role: Literal[
        "weights",
        "optimizer",
        "trainer_state",
        "telemetry",
        "trainable_targets",
        "frozen_parameters",
        "finite_gradient_proof",
    ]
    artifact_id: str = Field(min_length=1)
    digest: Sha256Digest
    size_bytes: int = Field(gt=0)


class ResumeOperationRequest(TrainingOperationIdentity):
    parent_checkpoint_id: str = Field(min_length=1)
    parent_checkpoint_digest: Sha256Digest
    parent_inventory: tuple[ResumeInventoryArtifact, ...] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def validate_parent_inventory(self) -> ResumeOperationRequest:
        roles = tuple(item.role for item in self.parent_inventory)
        if len(set(roles)) != 7:
            raise ValueError("resume parent inventory must bind every exact role")
        return self


class CancelOperationRequest(TrainingOperationIdentity):
    cancellation_request_id: str = Field(min_length=1)
    request_digest: Sha256Digest
    requested_step: int = Field(ge=0)


class OperationEvidenceEntry(FrozenModel):
    evidence_id: str = Field(min_length=1)
    digest: Sha256Digest


class OperationEventEntry(OperationEvidenceEntry):
    sequence: int = Field(gt=0)


class OperationTelemetryEntry(OperationEvidenceEntry):
    sequence: int = Field(gt=0)
    step: int = Field(ge=0)


class OperationCheckpointEntry(OperationEvidenceEntry):
    step: int = Field(ge=0)


class OperationInventoryEntry(OperationEvidenceEntry):
    checkpoint_id: str = Field(min_length=1)
    role: str = Field(min_length=1)


class OperationArtifactEntry(OperationEvidenceEntry):
    size_bytes: int = Field(gt=0)


class EvidenceOperationReceipt(FrozenModel):
    run_id: str = Field(min_length=1)
    lifecycle_version: int = Field(ge=0)
    lifecycle_state: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    evidence_digest: Sha256Digest


class CollectionOperationReceipt(FrozenModel):
    run_id: str = Field(min_length=1)
    lifecycle_version: int = Field(ge=0)
    lifecycle_state: str = Field(min_length=1)
    events: tuple[OperationEventEntry, ...]
    telemetry: tuple[OperationTelemetryEntry, ...]
    checkpoints: tuple[OperationCheckpointEntry, ...]
    artifacts: tuple[OperationArtifactEntry, ...]
    inventory: tuple[OperationInventoryEntry, ...]
    evaluation: OperationEvidenceEntry | None = None

    @model_validator(mode="after")
    def validate_unique_receipt_ids(self) -> CollectionOperationReceipt:
        for entries in (
            self.events,
            self.telemetry,
            self.checkpoints,
            self.artifacts,
            self.inventory,
        ):
            identities = tuple(entry.evidence_id for entry in entries)
            if len(set(identities)) != len(identities):
                raise ValueError("receipt evidence identities must be unique per kind")
        return self


class StartOperationReceipt(CollectionOperationReceipt):
    pass


class ResumeOperationReceipt(CollectionOperationReceipt):
    resume: OperationEvidenceEntry


class CancelOperationReceipt(FrozenModel):
    run_id: str = Field(min_length=1)
    lifecycle_version: int = Field(ge=0)
    lifecycle_state: Literal["cancelled"]
    cancellation: OperationEvidenceEntry
    terminal_event: OperationEventEntry
    boundary_checkpoint: OperationCheckpointEntry
    cancellation_request_id: str = Field(min_length=1)
    request_digest: Sha256Digest
    requested_step: int = Field(ge=0)
    boundary_step: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_boundary_checkpoint(self) -> CancelOperationReceipt:
        if self.boundary_checkpoint.step != self.boundary_step:
            raise ValueError("boundary checkpoint step must equal cancellation boundary")
        return self


OperationRequest = Annotated[
    ProbeOperationRequest
    | CalibrationOperationRequest
    | StartOperationRequest
    | ResumeOperationRequest
    | CancelOperationRequest,
    Field(union_mode="left_to_right"),
]


def training_operation_payload_digest(payload: Mapping[str, object]) -> str:
    serialized = canonical_json(cast(Any, _json_value(dict(payload))))
    return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"


class TrainingOrchestrationOperation(FrozenModel):
    """Immutable request identity plus the latest persisted execution receipt."""

    schema_version: Literal["character-training-orchestration-operation.v1"] = (
        "character-training-orchestration-operation.v1"
    )
    operation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    recipe_id: str = Field(min_length=1)
    operation_kind: TrainingOperationKind
    idempotency_key: str = Field(min_length=1)
    request_digest: Sha256Digest
    request_payload: Mapping[str, object]
    lifecycle_version: int = Field(ge=0)
    lifecycle_state: Literal[
        "draft",
        "probed",
        "calibrated",
        "preregistered",
        "training",
        "evaluating",
        "qualified",
        "rejected",
        "failed",
        "cancelled",
    ]
    status: TrainingOperationStatus
    attempt: int = Field(ge=0)
    claimant_id: str | None = Field(default=None, min_length=1)
    claim_token: str | None = Field(default=None, min_length=1)
    lease_expires_at: datetime | None = None
    receipt_digest: Sha256Digest | None = None
    receipt_payload: Mapping[str, object] | None = None
    result_digest: Sha256Digest | None = None
    result_payload: Mapping[str, object] | None = None
    failure_digest: Sha256Digest | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("request_payload", "receipt_payload", "result_payload")
    @classmethod
    def freeze_operation_payload(
        cls, value: Mapping[str, object] | None
    ) -> Mapping[str, object] | None:
        if value is None:
            return None
        return cast(Mapping[str, object], _freeze_mapping(cast(Mapping[object, object], value)))

    @field_serializer("request_payload", "receipt_payload", "result_payload")
    def serialize_operation_payload(
        self, value: Mapping[str, object] | None
    ) -> dict[str, object] | None:
        return (
            None if value is None else {str(key): _json_value(item) for key, item in value.items()}
        )

    @model_validator(mode="after")
    def validate_operation_receipt(self) -> TrainingOrchestrationOperation:
        creation_states = {
            TrainingOperationKind.PROBE: "draft",
            TrainingOperationKind.CALIBRATION: "probed",
            TrainingOperationKind.START: "preregistered",
            TrainingOperationKind.CANCEL: "training",
        }
        valid_creation_states = (
            {"training", "cancelled"}
            if self.operation_kind is TrainingOperationKind.RESUME
            else {creation_states[self.operation_kind]}
        )
        if self.lifecycle_state not in valid_creation_states:
            raise ValueError("operation creation lifecycle does not match operation kind")
        if self.request_digest != training_operation_payload_digest(self.request_payload):
            raise ValueError("request digest does not match canonical request payload")
        running = self.status == TrainingOperationStatus.RUNNING
        if running != (
            self.claimant_id is not None
            and self.claim_token is not None
            and self.lease_expires_at is not None
        ):
            raise ValueError("running operation requires claimant, claim token, and lease")
        completed = self.status == TrainingOperationStatus.COMPLETED
        if completed != (self.receipt_digest is not None and self.receipt_payload is not None):
            raise ValueError("completed operation requires an exact receipt")
        if completed and self.receipt_digest != training_operation_payload_digest(
            cast(Mapping[str, object], self.receipt_payload)
        ):
            raise ValueError("receipt digest does not match canonical receipt payload")
        if completed != (self.result_digest is not None and self.result_payload is not None):
            raise ValueError("completed operation requires an exact durable result")
        if completed and self.result_digest != training_operation_payload_digest(
            cast(Mapping[str, object], self.result_payload)
        ):
            raise ValueError("result digest does not match canonical result payload")
        if (self.status == TrainingOperationStatus.FAILED) != (self.failure_digest is not None):
            raise ValueError("failed operation requires a failure digest")
        return self


class TrainingSourceTier(StrEnum):
    GOLD_STILL = "gold_still"
    GOLD_CLIP = "gold_clip"
    SYNTHETIC_CLIP = "synthetic_clip"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    SEALED_HOLDOUT = "sealed_holdout"


class TrainingMedia(DatasetMedia):
    """Versioned media metadata for persisted character-training records."""

    schema_version: Literal["character-training-media.v1"] = "character-training-media.v1"


class TrainingAnnotation(FrozenModel):
    """Curated coverage labels, intentionally independent from trainer internals."""

    canonical_outfit_id: str = Field(min_length=1)
    head_angle: Literal["front", "three_quarter", "profile"]
    framing: Literal["portrait", "full_body"]
    motion_speed: Literal["slow", "fast"]
    camera_motion: Literal["static", "moving"]
    expression_change: bool
    partial_occlusion: bool
    lighting: Literal["ordinary", "directional"]
    visible_outfit_details: bool


class TrainingCurationSuggestions(FrozenModel):
    """Untrusted machine-authored values; absence never implies confirmation."""

    source_tier: TrainingSourceTier | None = None
    source_lineage: str | None = Field(default=None, min_length=1)
    caption: str | None = Field(default=None, min_length=1)
    split: DatasetSplit | None = None
    annotation: TrainingAnnotation | None = None


class ImportedTrainingSample(FrozenModel):
    schema_version: Literal["character-training-import.v1"] = "character-training-import.v1"
    import_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    character_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    artifact_digest: Sha256Digest
    artifact_byte_size: int = Field(gt=0)
    media: TrainingMedia
    source_lineage: str = Field(min_length=1)
    suggestions: TrainingCurationSuggestions | None = None
    imported_at: datetime

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> ImportedTrainingSample:
        if self.artifact_digest != self.media.sha256:
            raise ValueError("artifact_digest must match media sha256")
        if self.artifact_byte_size != self.media.byte_size:
            raise ValueError("artifact_byte_size must match media byte_size")
        if self.import_id != imported_training_sample_id(self):
            raise ValueError("import_id does not match imported sample content")
        return self


class CurationDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class TrainingCurationConfirmation(FrozenModel):
    schema_version: Literal["character-training-confirmation.v1"] = (
        "character-training-confirmation.v1"
    )
    confirmation_id: str = Field(min_length=1)
    import_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    decision: CurationDecision
    confirmed_at: datetime
    version: int = Field(gt=0)
    sample_id: str | None = Field(default=None, min_length=1)
    source_tier: TrainingSourceTier | None = None
    source_lineage: str | None = Field(default=None, min_length=1)
    caption: str | None = Field(default=None, min_length=1)
    split: DatasetSplit | None = None
    annotation: TrainingAnnotation | None = None

    @model_validator(mode="after")
    def validate_decision_fields(self) -> TrainingCurationConfirmation:
        curated = (
            self.sample_id,
            self.source_tier,
            self.source_lineage,
            self.caption,
            self.split,
            self.annotation,
        )
        if self.decision == CurationDecision.ACCEPT and any(value is None for value in curated):
            raise ValueError("accepted confirmation requires every curated sample field")
        if self.decision == CurationDecision.REJECT and any(value is not None for value in curated):
            raise ValueError("rejected confirmation must not contain curated sample fields")
        if self.confirmation_id != training_confirmation_id(self):
            raise ValueError("confirmation_id does not match confirmation content")
        return self

    def accepted_sample(self, imported: ImportedTrainingSample) -> TrainingSample:
        if (
            self.decision != CurationDecision.ACCEPT
            or self.import_id != imported.import_id
            or self.reviewer_id != imported.reviewer_id
        ):
            raise ValueError("confirmation does not accept this imported sample")
        return TrainingSample(
            sample_id=cast(str, self.sample_id),
            source_tier=cast(TrainingSourceTier, self.source_tier),
            sample_digest=imported.artifact_digest,
            source_lineage=cast(str, self.source_lineage),
            caption=cast(str, self.caption),
            split=cast(DatasetSplit, self.split),
            media=imported.media,
            annotation=cast(TrainingAnnotation, self.annotation),
        )


class CuratedEvidenceReferenceV1(FrozenModel):
    """Accepted, content-addressed curated validation evidence for one sequence."""

    schema_version: Literal["curated-training-evidence-reference.v1"] = (
        "curated-training-evidence-reference.v1"
    )
    curated_dataset_digest: Sha256Digest
    validation_report_digest: Sha256Digest
    sequence_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    validation_state: Literal["accepted"]


class TrainingSample(FrozenModel):
    """A content-addressed accepted item. Raw source paths are never retained."""

    sample_id: str = Field(min_length=1)
    source_tier: TrainingSourceTier
    sample_digest: Sha256Digest
    source_lineage: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    split: DatasetSplit
    media: TrainingMedia
    annotation: TrainingAnnotation
    accepted: bool = True
    curated_evidence: CuratedEvidenceReferenceV1 | None = None

    @model_validator(mode="after")
    def validate_digest_matches_media(self) -> TrainingSample:
        if self.sample_digest != self.media.sha256:
            raise ValueError("sample_digest must match media sha256")
        if (
            self.source_tier == TrainingSourceTier.GOLD_STILL
            and self.media.media_type == "video/mp4"
        ):
            raise ValueError("gold_still samples must be still images")
        if (
            self.source_tier != TrainingSourceTier.GOLD_STILL
            and self.media.media_type != "video/mp4"
        ):
            raise ValueError("clip samples must be MP4 videos")
        return self


class DatasetCoverage(FrozenModel):
    accepted_gold_stills: int = Field(ge=0)
    accepted_gold_clips: int = Field(ge=0)
    accepted_synthetic_clips: int = Field(ge=0)
    accepted_clips: int = Field(ge=0)
    synthetic_effective_weight: float = Field(ge=0)
    frontal: bool
    three_quarter: bool
    profile: bool
    portrait: bool
    full_body: bool
    slow_motion: bool
    fast_motion: bool
    static_camera: bool
    moving_camera: bool
    expression_change: bool
    partial_occlusion: bool
    ordinary_lighting: bool
    directional_lighting: bool
    visible_outfit_details: bool


class TrainingDatasetDraft(FrozenModel):
    schema_version: Literal["character-training-dataset.v1"] = "character-training-dataset.v1"
    character_id: str = Field(min_length=1)
    character_pack_digest: Sha256Digest
    canonical_outfit_id: str = Field(min_length=1)
    reserved_token: str = Field(pattern=r"^<[^<>\s]+>$")
    source_tier_weights: Mapping[TrainingSourceTier, float] = Field(
        default_factory=lambda: MappingProxyType(
            {
                TrainingSourceTier.GOLD_STILL: 1.0,
                TrainingSourceTier.GOLD_CLIP: 1.0,
                TrainingSourceTier.SYNTHETIC_CLIP: 1.0,
            }
        )
    )
    samples: tuple[TrainingSample, ...]
    imported_sample_ids: tuple[str, ...] = ()
    curation_confirmation_ids: tuple[str, ...] = ()

    @field_validator("imported_sample_ids", "curation_confirmation_ids")
    @classmethod
    def normalize_ledger_ids(cls, value: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        field_name = info.field_name
        prefix = (
            "training_import_" if field_name == "imported_sample_ids" else "training_confirmation_"
        )
        pattern = re.compile(rf"^{prefix}[0-9a-f]{{64}}$")
        if any(pattern.fullmatch(item) is None for item in value):
            raise ValueError("dataset ledger contains malformed content identity")
        if len(set(value)) != len(value):
            raise ValueError("dataset ledger contains duplicate content identity")
        return tuple(sorted(value))

    @field_validator("source_tier_weights")
    @classmethod
    def freeze_source_tier_weights(
        cls, value: Mapping[TrainingSourceTier, float]
    ) -> Mapping[TrainingSourceTier, float]:
        if set(value) != set(TrainingSourceTier):
            raise ValueError("source_tier_weights must include every source tier")
        if any(weight <= 0 or not math.isfinite(weight) for weight in value.values()):
            raise ValueError("source tier weights must be finite and positive")
        return MappingProxyType(dict(value))

    @field_serializer("source_tier_weights")
    def serialize_source_tier_weights(
        self, value: Mapping[TrainingSourceTier, float]
    ) -> dict[str, float]:
        return {tier.value: weight for tier, weight in value.items()}


class FrozenTrainingDataset(TrainingDatasetDraft):
    dataset_digest: Sha256Digest
    coverage: DatasetCoverage


class CheckpointPlan(FrozenModel):
    fractions: tuple[Literal[0.25, 0.5, 0.75, 1.0], ...] = (  # type: ignore[valid-type]
        0.25,
        0.5,
        0.75,
        1.0,
    )

    @field_validator("fractions")
    @classmethod
    def require_complete_checkpoint_plan(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if value != (0.25, 0.5, 0.75, 1.0):
            raise ValueError("checkpoint fractions must be 0.25, 0.5, 0.75, and 1.0")
        return value


class TrainingRecipe(FrozenModel):
    schema_version: Literal["character-training-recipe.v1"]
    dataset_digest: Sha256Digest
    character_pack_digest: Sha256Digest
    stage_2_manifest_digest: Sha256Digest
    stage_2_evidence_digest: Sha256Digest
    stage_3_evidence_digest: Sha256Digest
    base_model_digest: Sha256Digest
    lightning_adapter_digests: Mapping[str, Sha256Digest]
    source_code_digest: Sha256Digest
    runtime_identity: Mapping[str, str]
    backend_id: str
    backend_config: Mapping[str, object]
    trainable_targets: tuple[str, ...]
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0, lt=1)
    optimizer: str
    learning_rate: float = Field(gt=0)
    total_steps: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    source_tier_weights: Mapping[TrainingSourceTier, float]
    checkpoint_plan: CheckpointPlan
    calibration_steps: int = Field(ge=0)
    wall_clock_limit_seconds: Literal[28800, 57600]
    seed: int

    @field_validator(
        "lightning_adapter_digests",
        "runtime_identity",
        "backend_config",
        "source_tier_weights",
    )
    @classmethod
    def freeze_mappings(cls, value: Mapping[object, object]) -> Mapping[object, object]:
        return _freeze_mapping(value)

    @field_serializer(
        "lightning_adapter_digests",
        "runtime_identity",
        "backend_config",
        "source_tier_weights",
    )
    def serialize_mappings(self, value: Mapping[object, object]) -> dict[str, object]:
        return {str(key): _json_value(item) for key, item in value.items()}

    @model_validator(mode="after")
    def validate_tier_weights(self) -> TrainingRecipe:
        if set(self.source_tier_weights) != set(TrainingSourceTier):
            raise ValueError("source_tier_weights must include every source tier")
        if any(
            weight <= 0 or not math.isfinite(weight) for weight in self.source_tier_weights.values()
        ):
            raise ValueError("source tier weights must be finite and positive")
        if not self.trainable_targets or any(not target for target in self.trainable_targets):
            raise ValueError("trainable_targets must not be empty")
        checkpoint_steps(self.total_steps)
        return self


class TrainingRecipeValidationEvidence(FrozenModel):
    """Composition-derived evidence binding an operative recipe to frozen data."""

    schema_version: Literal["character-training-recipe-validation.v1"] = (
        "character-training-recipe-validation.v1"
    )
    recipe_digest: Sha256Digest
    dataset_digest: Sha256Digest
    character_pack_digest: Sha256Digest
    accepted_gold_clip_count: int = Field(ge=0)
    accepted_synthetic_clip_count: int = Field(ge=0)
    gold_clip_weight: float = Field(gt=0)
    synthetic_clip_weight: float = Field(gt=0)
    synthetic_effective_share: float = Field(ge=0, le=0.5)


class TrainingRun(FrozenModel):
    schema_version: Literal["character-training-run.v1"] = "character-training-run.v1"
    run_id: str = Field(min_length=1)
    recipe_id: str = Field(min_length=1)
    dataset_digest: Sha256Digest
    total_steps: int = Field(gt=0)


class TrainingRunEvent(FrozenModel):
    sequence: int = Field(gt=0)
    previous_state: Literal[
        "draft",
        "probed",
        "calibrated",
        "preregistered",
        "training",
        "evaluating",
        "qualified",
        "rejected",
        "failed",
        "cancelled",
    ]
    new_state: Literal[
        "probed",
        "calibrated",
        "preregistered",
        "training",
        "evaluating",
        "qualified",
        "rejected",
        "failed",
        "cancelled",
    ]
    payload: Mapping[str, object]

    @field_validator("payload")
    @classmethod
    def freeze_payload(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return cast(Mapping[str, object], _freeze_mapping(cast(Mapping[object, object], value)))

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, object]) -> dict[str, object]:
        return {str(key): _json_value(item) for key, item in value.items()}

    @model_validator(mode="after")
    def validate_edge(self) -> TrainingRunEvent:
        legal = {
            ("draft", "probed"),
            ("probed", "calibrated"),
            ("calibrated", "preregistered"),
            ("preregistered", "training"),
            ("cancelled", "training"),
            ("training", "evaluating"),
            ("evaluating", "qualified"),
            ("evaluating", "rejected"),
        }
        failure_edge = self.new_state == "failed" and self.previous_state in {
            "draft",
            "probed",
            "calibrated",
            "preregistered",
            "training",
            "evaluating",
        }
        cancellation_edge = self.new_state == "cancelled" and self.previous_state == "training"
        if (self.previous_state, self.new_state) not in legal and not (
            failure_edge or cancellation_edge
        ):
            raise ValueError("invalid scientific training lifecycle edge")
        return self


class TrainingProbeEvidence(FrozenModel):
    evidence_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    recipe_digest: Sha256Digest
    dataset_digest: Sha256Digest
    backend_digest: Sha256Digest
    runtime_digest: Sha256Digest
    material_digest: Sha256Digest
    source_digest: Sha256Digest
    capability_digest: Sha256Digest
    succeeded: bool


class TrainingCalibrationEvidence(FrozenModel):
    evidence_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    probe_evidence_id: str = Field(min_length=1)
    recipe_digest: Sha256Digest
    dataset_digest: Sha256Digest
    backend_digest: Sha256Digest
    runtime_digest: Sha256Digest
    material_digest: Sha256Digest
    source_digest: Sha256Digest
    calibration_digest: Sha256Digest
    succeeded: bool


class TrainingCancellationEvidence(FrozenModel):
    cancellation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    cancellation_request_id: str = Field(min_length=1)
    request_digest: Sha256Digest
    requested_step: int = Field(ge=0)
    boundary_step: int = Field(ge=0)
    checkpoint_digest: Sha256Digest

    @model_validator(mode="after")
    def boundary_follows_request(self) -> TrainingCancellationEvidence:
        if self.boundary_step < self.requested_step:
            raise ValueError("cancellation boundary precedes request")
        return self


class TrainingCheckpointArtifact(FrozenModel):
    """One immutable file in a checkpoint's complete artifact inventory."""

    relative_path: str = Field(min_length=1)
    digest: Sha256Digest
    size_bytes: int = Field(gt=0)
    role: Literal[
        "weights",
        "optimizer",
        "trainer_state",
        "telemetry",
        "trainable_targets",
        "frozen_parameters",
        "finite_gradient_proof",
    ]

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("checkpoint inventory path must be relative and contained")
        return value


class TrainingCheckpointEvidence(FrozenModel):
    schema_version: Literal["character-training-checkpoint-evidence.v1"] = (
        "character-training-checkpoint-evidence.v1"
    )
    run_id: str = Field(min_length=1)
    recipe_digest: Sha256Digest
    dataset_digest: Sha256Digest
    backend_digest: Sha256Digest
    runtime_digest: Sha256Digest
    material_digest: Sha256Digest
    source_digest: Sha256Digest
    step: int = Field(gt=0)
    fraction: Literal[0.25, 0.5, 0.75, 1.0]  # type: ignore[valid-type]
    parent_checkpoint_digest: Sha256Digest | None
    safetensors_digest: Sha256Digest
    safetensors_size_bytes: int = Field(gt=0)
    optimizer_state_digest: Sha256Digest
    trainer_state_digest: Sha256Digest
    telemetry_digest: Sha256Digest
    telemetry_sequence_start: int = Field(gt=0)
    telemetry_sequence_end: int = Field(gt=0)
    trainable_targets_digest: Sha256Digest
    frozen_parameters_digest: Sha256Digest
    finite_gradient_proof_digest: Sha256Digest
    cancellation_boundary: bool
    inventory: tuple[TrainingCheckpointArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_checkpoint_evidence(self) -> TrainingCheckpointEvidence:
        if self.telemetry_sequence_end < self.telemetry_sequence_start:
            raise ValueError("checkpoint telemetry range is reversed")
        paths = tuple(item.relative_path for item in self.inventory)
        if len(paths) != len(set(paths)):
            raise ValueError("checkpoint inventory paths must be unique")
        expected_digests = {
            "weights": self.safetensors_digest,
            "optimizer": self.optimizer_state_digest,
            "trainer_state": self.trainer_state_digest,
            "telemetry": self.telemetry_digest,
            "trainable_targets": self.trainable_targets_digest,
            "frozen_parameters": self.frozen_parameters_digest,
            "finite_gradient_proof": self.finite_gradient_proof_digest,
        }
        by_role = {item.role: item for item in self.inventory}
        if len(self.inventory) != len(expected_digests):
            raise ValueError("checkpoint inventory is incomplete")
        if set(by_role) != set(expected_digests) or len(by_role) != len(self.inventory):
            raise ValueError("checkpoint inventory roles are not exact")
        if any(item.digest != expected_digests[item.role] for item in self.inventory):
            raise ValueError("checkpoint inventory role binding mismatch")
        # The weights file has a separately declared byte size. Other role files have no
        # independent size claim in v1; their canonical semantics are a strictly positive,
        # inventory-owned byte count (enforced by TrainingCheckpointArtifact).
        if by_role["weights"].size_bytes != self.safetensors_size_bytes:
            raise ValueError("checkpoint inventory role binding size mismatch")
        return self


class TrainingResumeEvidence(FrozenModel):
    resume_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    parent_checkpoint_id: str = Field(min_length=1)
    parent_checkpoint_digest: Sha256Digest
    parent_artifact_digest: Sha256Digest
    recipe_digest: Sha256Digest
    dataset_digest: Sha256Digest
    backend_digest: Sha256Digest
    runtime_digest: Sha256Digest
    material_digest: Sha256Digest
    source_digest: Sha256Digest
    optimizer_state_digest: Sha256Digest


class TrainingEvaluationEvidence(FrozenModel):
    evidence_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    disposition: Literal["qualified", "rejected"]
    evidence_digest: Sha256Digest


class TrainingTelemetry(FrozenModel):
    sequence: int = Field(gt=0)
    step: int = Field(ge=0)
    payload: Mapping[str, object]

    @field_validator("payload")
    @classmethod
    def freeze_payload(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return cast(Mapping[str, object], _freeze_mapping(cast(Mapping[object, object], value)))

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, object]) -> dict[str, object]:
        return {str(key): _json_value(item) for key, item in value.items()}


class TrainingCheckpoint(FrozenModel):
    checkpoint_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step: int = Field(gt=0)
    fraction: Literal[0.25, 0.5, 0.75, 1.0]  # type: ignore[valid-type]
    parent_checkpoint_id: str | None
    artifact_digest: Sha256Digest
    manifest_digest: Sha256Digest
    manifest: Mapping[str, object]

    @field_validator("manifest")
    @classmethod
    def freeze_manifest(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        required = {
            "schema_version",
            "run_id",
            "reviewer_id",
            "recipe_id",
            "dataset_digest",
            "character_id",
            "step",
            "fraction",
            "parent_checkpoint_id",
            "artifact_digest",
            "safetensors_size_bytes",
            "backend_digest",
            "runtime_digest",
            "material_digest",
            "source_digest",
            "optimizer_state_digest",
            "trainer_state_digest",
            "telemetry_digest",
            "telemetry_sequence_start",
            "telemetry_sequence_end",
            "trainable_targets_digest",
            "frozen_parameters_digest",
            "finite_gradient_proof_digest",
            "cancellation_boundary",
            "inventory",
        }
        if set(value) != required:
            raise ValueError("checkpoint manifest is incomplete")
        return cast(Mapping[str, object], _freeze_mapping(cast(Mapping[object, object], value)))

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> TrainingCheckpoint:
        if (
            self.manifest["run_id"] != self.run_id
            or self.manifest["step"] != self.step
            or self.manifest["fraction"] != self.fraction
            or self.manifest["parent_checkpoint_id"] != self.parent_checkpoint_id
            or self.manifest["artifact_digest"] != self.artifact_digest
            or _content_digest(dict(self.manifest)) != self.manifest_digest
        ):
            raise ValueError("checkpoint manifest identity or digest mismatch")
        return self

    @field_serializer("manifest")
    def serialize_manifest(self, value: Mapping[str, object]) -> dict[str, object]:
        return {str(key): _json_value(item) for key, item in value.items()}


class TrainingFailure(FrozenModel):
    failure_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


def checkpoint_steps(total_steps: int) -> tuple[int, int, int, int]:
    """Return collision-free 25/50/75/100% integer milestones."""
    if total_steps < 4:
        raise ValueError("total_steps must represent four checkpoint milestones")
    steps = tuple((total_steps * numerator + 3) // 4 for numerator in (1, 2, 3, 4))
    if len(set(steps)) != 4 or steps[-1] != total_steps:
        raise ValueError("checkpoint milestones collide")
    return cast(tuple[int, int, int, int], steps)


def training_recipe_id(recipe: TrainingRecipe) -> str:
    """Return the versioned canonical SHA-256 identity for a training recipe."""
    payload = recipe.model_dump(mode="json")
    payload["trainable_targets"] = sorted(set(recipe.trainable_targets))
    return _content_digest(payload)


def training_dataset_id(draft: TrainingDatasetDraft) -> str:
    """Order-independent identity that binds every accepted-sample contract field."""
    payload = draft.model_dump(mode="json")
    for sample in payload["samples"]:
        if sample["curated_evidence"] is None:
            # Preserve the v1 identity of legacy samples while binding opt-in evidence.
            del sample["curated_evidence"]
    payload["samples"] = sorted(payload["samples"], key=_sample_order)
    payload["imported_sample_ids"] = sorted(set(payload["imported_sample_ids"]))
    payload["curation_confirmation_ids"] = sorted(set(payload["curation_confirmation_ids"]))
    return _content_digest(payload)


def imported_training_sample_id(sample: ImportedTrainingSample) -> str:
    payload = sample.model_dump(mode="json", exclude={"import_id"})
    return "training_import_" + _content_digest(payload).removeprefix("sha256:")


def training_confirmation_id(confirmation: TrainingCurationConfirmation) -> str:
    payload = confirmation.model_dump(mode="json", exclude={"confirmation_id"})
    return "training_confirmation_" + _content_digest(payload).removeprefix("sha256:")


def _sample_order(sample: object) -> str:
    return json.dumps(sample, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_digest(payload: object) -> str:
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _freeze_mapping(value: Mapping[object, object]) -> Mapping[object, object]:
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def _freeze_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("unsupported non-finite JSON number")
    if value is None or isinstance(value, str | int | float | bool | StrEnum):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str | StrEnum) for key in value):
            raise TypeError("mapping keys must be strings")
        return _freeze_mapping(value)
    if isinstance(value, Set):
        return tuple(sorted((_freeze_json(item) for item in value), key=_sample_order))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
