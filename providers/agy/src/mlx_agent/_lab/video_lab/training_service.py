"""Fail-closed application boundary for character-training evidence."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import sqlite3
import stat
import sys
import tempfile
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, ParamSpec, Protocol, TypeVar, cast

from pydantic import Field

from video_lab.artifacts import (
    Artifact,
    ArtifactError,
    ArtifactStore,
    StagedArtifact,
    VerifiedArtifactHandle,
)
from video_lab.domain import FrozenModel, canonical_json
from video_lab.trainer_protocol import (
    CalibrationRequest,
    CalibrationResult,
    LoRAConfiguration,
    OptimizerConfiguration,
    OptimizerParameter,
    ProbeTrainingRequest,
    TrainerBackend,
    TrainerCapabilityReport,
    TrainerRegistry,
    TrainerResultInvalid,
    TrainingCancelled,
    TrainingMaterialHandles,
    TrainingProgress,
    TrainingPublicationReceipt,
    TrainingRequest,
    TrainingResult,
    validate_cancelled_training_result,
    validate_training_result,
)
from video_lab.training_dataset import (
    evaluate_training_dataset,
    freeze_training_dataset,
    validate_training_recipe_dataset,
)
from video_lab.training_media import (
    TrainingMediaInspectionError,
    open_training_media_collection_snapshot,
)
from video_lab.training_models import (
    CalibrationOperationRequest,
    CancelOperationRequest,
    CheckpointPlan,
    CurationDecision,
    DatasetCoverage,
    DatasetSplit,
    FrozenTrainingDataset,
    ImportedTrainingSample,
    ProbeOperationRequest,
    ResumeInventoryArtifact,
    ResumeOperationRequest,
    StartOperationRequest,
    TrainingAnnotation,
    TrainingCalibrationEvidence,
    TrainingCancellationEvidence,
    TrainingCheckpoint,
    TrainingCheckpointArtifact,
    TrainingCheckpointEvidence,
    TrainingCurationConfirmation,
    TrainingCurationSuggestions,
    TrainingDatasetDraft,
    TrainingEvaluationEvidence,
    TrainingFailure,
    TrainingMedia,
    TrainingOperationKind,
    TrainingOperationStatus,
    TrainingOrchestrationOperation,
    TrainingProbeEvidence,
    TrainingRecipe,
    TrainingResumeEvidence,
    TrainingRun,
    TrainingRunEvent,
    TrainingSourceTier,
    TrainingTelemetry,
    checkpoint_steps,
    imported_training_sample_id,
    training_confirmation_id,
    training_operation_payload_digest,
    training_recipe_id,
)
from video_lab.training_repository import (
    DatasetLineageRecord,
    RecipeMaterialBinding,
    RecipeRecord,
    TrainingConflict,
    TrainingNotFound,
    TrainingRepository,
)

_SUPPORTED_MEDIA = {".jpg", ".jpeg", ".png", ".webp", ".mp4"}
_SOURCE_LIMIT = 512 * 1024 * 1024


class TrainingServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _public_guard(function: Callable[_P, _R]) -> Callable[_P, _R]:  # noqa: UP047
    """Ensure the application boundary never exposes local implementation details."""

    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except TrainingServiceError as error:
            raise TrainingServiceError(error.code) from None
        except Exception:
            raise TrainingServiceError("training_service_failed") from None

    return guarded


class RuntimeIdentityProvider(Protocol):
    def identity(self) -> Mapping[str, str]: ...


class LocalRuntimeIdentity:
    def identity(self) -> Mapping[str, str]:
        return {
            "machine": platform.machine(),
            "platform": platform.system(),
            "python": platform.python_version(),
            "implementation": sys.implementation.name,
        }


class DatasetCreateInput(FrozenModel):
    character_pack_id: str = Field(min_length=1)
    canonical_outfit_id: str = Field(min_length=1)
    reserved_token: str = Field(pattern=r"^<[^<>\s]+>$")


class CurationInput(FrozenModel):
    decision: CurationDecision
    sample_id: str | None = Field(default=None, min_length=1)
    source_tier: TrainingSourceTier | None = None
    source_lineage: str | None = Field(default=None, min_length=1)
    caption: str | None = Field(default=None, min_length=1)
    split: DatasetSplit | None = None
    annotation: TrainingAnnotation | None = None


@dataclass(frozen=True, slots=True)
class _PreparedTrainingExecution:
    record: RecipeRecord
    backend: TrainerBackend
    request: TrainingRequest
    resume_evidence: TrainingResumeEvidence | None
    telemetry_offset: int


class RecipeInput(FrozenModel):
    stage_3_campaign_id: str = Field(min_length=1)
    stage_3_result_id: str = Field(min_length=1)
    base_model_artifact_id: str = Field(min_length=1)
    lightning_adapter_artifact_ids: Mapping[str, str]
    backend_id: str = Field(min_length=1)
    backend_config: Mapping[str, object]
    trainable_targets: tuple[str, ...] = Field(min_length=1)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0, lt=1)
    optimizer: str = Field(min_length=1)
    learning_rate: float = Field(gt=0)
    total_steps: int = Field(ge=4)
    gradient_accumulation_steps: int = Field(gt=0)
    source_tier_weights: Mapping[TrainingSourceTier, float]
    checkpoint_plan: CheckpointPlan = CheckpointPlan()
    calibration_steps: int = Field(ge=0)
    wall_clock_limit_seconds: Literal[28800, 57600]
    seed: int


class TrainingDatasetView(FrozenModel):
    lineage_id: str
    dataset_id: str
    head_version: int
    state: Literal["draft", "sealed"]
    character_id: str
    imported: tuple[ImportedTrainingSample, ...]
    coverage: DatasetCoverage


class TrainingRecipeView(FrozenModel):
    recipe_id: str
    state: Literal["preview", "draft", "preregistered"]
    recipe: TrainingRecipe


class TrainingRecipePreview(FrozenModel):
    state: Literal["preview"] = "preview"
    preview_token: str
    source_snapshot_digest: str
    source_snapshot_byte_size: int
    source_snapshot_artifact_id: None = None
    base_model_digest: str
    lightning_adapter_digests: Mapping[str, str]


def discover_training_media(source: Path) -> tuple[Path, ...]:
    try:
        root = source.lstat()
        if stat.S_ISLNK(root.st_mode) or not stat.S_ISDIR(root.st_mode):
            raise TrainingServiceError("unsafe_path")
        result: list[Path] = []
        for current, directories, files in os.walk(source, followlinks=False):
            for name in directories:
                if (Path(current) / name).is_symlink():
                    raise TrainingServiceError("unsafe_path")
            for name in files:
                candidate = Path(current) / name
                mode = candidate.lstat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise TrainingServiceError("unsafe_path")
                if candidate.suffix.lower() not in _SUPPORTED_MEDIA:
                    raise TrainingServiceError("unsupported_entry")
                result.append(candidate)
        if not result:
            raise TrainingServiceError("empty_media_folder")
        return tuple(sorted(result, key=lambda item: _sort_key(item.relative_to(source))))
    except TrainingServiceError:
        raise
    except OSError:
        raise TrainingServiceError("unsafe_path") from None


class TrainingService:
    def __init__(
        self,
        repository: TrainingRepository,
        artifacts: ArtifactStore,
        character_packs: object,
        *,
        reviewer_id: str,
        source_root: Path,
        runtime: RuntimeIdentityProvider | None = None,
        trainer_registry: TrainerRegistry | None = None,
    ) -> None:
        if repository.reviewer_id != reviewer_id:
            raise ValueError("reviewer identity mismatch")
        self.repository = repository
        self.artifacts = artifacts
        self.character_packs = character_packs
        self.reviewer_id = reviewer_id
        self.source_root = source_root
        self.runtime = runtime or LocalRuntimeIdentity()
        self.trainer_registry = trainer_registry or TrainerRegistry(reviewer_id=reviewer_id)

    @_public_guard
    def probe_backend(
        self,
        recipe_id: str,
        *,
        run_id: str,
        request_id: str,
        scratch_root: Path,
    ) -> TrainerCapabilityReport:
        record = self.repository.get_recipe(recipe_id)
        if record.state != "preregistered":
            raise TrainingServiceError("recipe_not_preregistered")
        recipe = record.recipe
        materials = self._training_materials(record)
        expected_backend_digest = _json_digest(
            recipe.model_dump(mode="json")["backend_config"]
        )
        request_payload = ProbeOperationRequest(
            request_id=request_id,
            run_id=run_id,
            recipe_digest=materials.recipe_digest,
            dataset_digest=materials.dataset_digest,
            backend_digest=materials.backend_digest,
            runtime_digest=materials.runtime_digest,
            material_digest=materials.material_digest,
            source_digest=materials.source_digest,
        ).model_dump(mode="json")
        dataset = self.repository.get_dataset(recipe.dataset_digest)

        def begin(
            connection: sqlite3.Connection,
        ) -> tuple[TrainingOrchestrationOperation, TrainingOrchestrationOperation | None]:
            try:
                self.repository.get_run_with_connection(connection, run_id)
            except TrainingNotFound:
                self.repository.create_run_with_connection(
                    connection,
                    TrainingRun(
                        run_id=run_id,
                        recipe_id=record.recipe_id,
                        dataset_digest=recipe.dataset_digest,
                        total_steps=recipe.total_steps,
                    ),
                    record.storage_id,
                    dataset.storage_id,
                )
            operation = self.repository.begin_operation_with_connection(
                connection,
                operation_id=f"operation_probe_{_json_digest(request_payload)[7:]}",
                run_id=run_id,
                operation_kind=TrainingOperationKind.PROBE,
                idempotency_key=request_id,
                request_payload=request_payload,
            )
            if operation.status is TrainingOperationStatus.COMPLETED:
                return operation, None
            claim = self.repository.claim_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.PROBE,
                idempotency_key=request_id,
                claimant_id=f"training-service:{self.reviewer_id}",
                lease_seconds=3600,
            )
            if not claim.claimed:
                raise TrainingServiceError("operation_in_progress")
            return operation, claim.operation

        operation, claim = cast(
            tuple[TrainingOrchestrationOperation, TrainingOrchestrationOperation | None],
            self.repository.atomic(begin),
        )
        if claim is None:
            payload = operation.result_payload
            if not isinstance(payload, Mapping) or not isinstance(payload.get("report"), Mapping):
                raise TrainingServiceError("operation_receipt_invalid")
            return TrainerCapabilityReport.model_validate(payload["report"])
        backend = self.trainer_registry.resolve(self.reviewer_id, recipe.backend_id)
        report = backend.probe(
            ProbeTrainingRequest(
                request_id=request_id,
                backend_id=recipe.backend_id,
                expected_backend_digest=expected_backend_digest,
                materials=materials,
                scratch_root=scratch_root,
            )
        )
        if (
            report.backend_id != recipe.backend_id
            or report.backend_digest != expected_backend_digest
            or report.runtime_digest != materials.runtime_digest
            or not report.supported
            or report.max_steps < recipe.total_steps
        ):
            raise TrainingServiceError("backend_probe_invalid")
        capability_digest = _json_digest(report.model_dump(mode="json"))
        evidence = TrainingProbeEvidence(
            evidence_id=f"probe_{capability_digest[7:]}",
            run_id=run_id,
            recipe_digest=training_recipe_id(recipe),
            dataset_digest=recipe.dataset_digest,
            backend_digest=expected_backend_digest,
            runtime_digest=materials.runtime_digest,
            material_digest=materials.material_digest,
            source_digest=materials.source_digest,
            capability_digest=capability_digest,
            succeeded=True,
        )

        def publish(connection: sqlite3.Connection) -> None:
            self.repository.record_probe_with_connection(connection, evidence)
            self.repository.append_event_with_connection(
                connection,
                run_id,
                TrainingRunEvent(
                    sequence=1,
                    previous_state="draft",
                    new_state="probed",
                    payload={"probe_evidence_id": evidence.evidence_id},
                ),
                expected_version=0,
            )
            evidence_digest = str(
                connection.execute(
                    "SELECT training_operation_evidence_digest(?,?,?)",
                    (operation.operation_id, "probe", evidence.evidence_id),
                ).fetchone()[0]
            )
            self.repository.link_operation_evidence_with_connection(
                connection,
                operation_id=operation.operation_id,
                attempt=claim.attempt,
                claim_token=cast(str, claim.claim_token),
                evidence_kind="probe",
                evidence_id=evidence.evidence_id,
                evidence_digest=evidence_digest,
            )
            self.repository.complete_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.PROBE,
                idempotency_key=request_id,
                claimant_id=cast(str, claim.claimant_id),
                claim_token=cast(str, claim.claim_token),
                attempt=claim.attempt,
                receipt_payload={
                    "run_id": run_id,
                    "lifecycle_version": 1,
                    "lifecycle_state": "probed",
                    "evidence_id": evidence.evidence_id,
                    "evidence_digest": evidence_digest,
                },
                result_payload={
                    "schema_version": "trainer-capability-report.v1",
                    "report": report.model_dump(mode="json"),
                },
            )

        self.repository.atomic(publish)
        return report

    @_public_guard
    def start_calibration(
        self, run_id: str, *, request_id: str, scratch_root: Path
    ) -> CalibrationResult:
        with self.repository.database.connect() as connection:
            try:
                replay = self.repository.reopen_operation_with_connection(
                    connection, TrainingOperationKind.CALIBRATION, request_id
                )
            except TrainingNotFound:
                replay = None
        if replay is not None:
            if replay.run_id != run_id:
                raise TrainingServiceError("operation_request_conflict")
        run = self.repository.get_run(run_id)
        if replay is None and run.state != "probed":
            raise TrainingServiceError("calibration_state_invalid")
        record = self.repository.get_recipe(run.run.recipe_id)
        recipe = record.recipe
        materials = self._training_materials(record)
        with self.repository.database.connect() as connection:
            probe = connection.execute(
                "SELECT evidence.id,evidence.capability_digest,operation.id operation_id "
                "FROM training_probe_evidence evidence "
                "JOIN training_orchestration_operations operation "
                "ON operation.run_id=evidence.run_id "
                "AND operation.operation_kind='probe' AND operation.status='completed' "
                "WHERE evidence.run_id=? AND evidence.reviewer_id=?",
                (run_id, self.reviewer_id),
            ).fetchone()
        if probe is None:
            raise TrainingServiceError("probe_evidence_missing")
        request_payload = CalibrationOperationRequest(
            request_id=request_id,
            run_id=run_id,
            recipe_digest=materials.recipe_digest,
            dataset_digest=materials.dataset_digest,
            backend_digest=materials.backend_digest,
            runtime_digest=materials.runtime_digest,
            material_digest=materials.material_digest,
            source_digest=materials.source_digest,
            probe_operation_id=str(probe["operation_id"]),
            capability_digest=str(probe["capability_digest"]),
            steps=recipe.calibration_steps,
            trainable_targets=recipe.trainable_targets,
            lora=LoRAConfiguration(
                rank=recipe.lora_rank,
                alpha=recipe.lora_alpha,
                dropout=recipe.lora_dropout,
            ),
            seed=recipe.seed,
        ).model_dump(mode="json")
        if replay is not None:
            if not _calibration_replay_is_equivalent(replay, request_payload):
                raise TrainingServiceError("operation_request_conflict")
            if replay.status is not TrainingOperationStatus.COMPLETED:
                raise TrainingServiceError("operation_in_progress")
            payload = replay.result_payload
            if not isinstance(payload, Mapping) or not isinstance(payload.get("result"), Mapping):
                raise TrainingServiceError("operation_receipt_invalid")
            return CalibrationResult.model_validate(payload["result"])

        def begin(connection: sqlite3.Connection) -> TrainingOrchestrationOperation:
            operation = self.repository.begin_operation_with_connection(
                connection,
                operation_id=f"operation_calibration_{_json_digest(request_payload)[7:]}",
                run_id=run_id,
                operation_kind=TrainingOperationKind.CALIBRATION,
                idempotency_key=request_id,
                request_payload=request_payload,
            )
            if operation.status is TrainingOperationStatus.COMPLETED:
                return operation
            claim = self.repository.claim_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.CALIBRATION,
                idempotency_key=request_id,
                claimant_id=f"training-service:{self.reviewer_id}",
                lease_seconds=3600,
            )
            if not claim.claimed:
                raise TrainingServiceError("operation_in_progress")
            return claim.operation

        operation = cast(TrainingOrchestrationOperation, self.repository.atomic(begin))
        if operation.status is TrainingOperationStatus.COMPLETED:
            payload = operation.result_payload
            if not isinstance(payload, Mapping) or not isinstance(payload.get("result"), Mapping):
                raise TrainingServiceError("operation_receipt_invalid")
            return CalibrationResult.model_validate(payload["result"])
        backend = self.trainer_registry.resolve(self.reviewer_id, recipe.backend_id)
        result = backend.calibrate(
            CalibrationRequest(
                request_id=request_id,
                run_id=run_id,
                backend_id=recipe.backend_id,
                materials=materials,
                scratch_root=scratch_root,
                steps=recipe.calibration_steps,
                trainable_targets=recipe.trainable_targets,
                lora=LoRAConfiguration(
                    rank=recipe.lora_rank,
                    alpha=recipe.lora_alpha,
                    dropout=recipe.lora_dropout,
                ),
                seed=recipe.seed,
                deadline_seconds=recipe.wall_clock_limit_seconds,
            ),
            lambda progress: None,
            lambda: False,
        )
        if (
            result.request_id != request_id
            or result.run_id != run_id
            or result.backend_digest != materials.backend_digest
            or result.runtime_digest != materials.runtime_digest
            or result.changed_targets != recipe.trainable_targets
            or not result.finite_gradients
            or not result.frozen_parameters_unchanged
        ):
            raise TrainingServiceError("calibration_invalid")
        return result

    @_public_guard
    def complete_calibration(self, run_id: str, result: CalibrationResult) -> None:
        run = self.repository.get_run(run_id)
        if run.state != "probed" or result.run_id != run_id:
            raise TrainingServiceError("calibration_state_invalid")
        record = self.repository.get_recipe(run.run.recipe_id)
        recipe = record.recipe
        materials = self._training_materials(record)
        with self.repository.database.connect() as connection:
            probe = connection.execute(
                "SELECT id FROM training_probe_evidence WHERE run_id=? AND reviewer_id=? "
                "AND succeeded=1 ORDER BY created_at DESC LIMIT 1",
                (run_id, self.reviewer_id),
            ).fetchone()
        if probe is None:
            raise TrainingServiceError("probe_evidence_missing")
        with self.repository.database.connect() as connection:
            operation = self.repository.reopen_operation_with_connection(
                connection, TrainingOperationKind.CALIBRATION, result.request_id
            )
        if (
            operation.status is not TrainingOperationStatus.RUNNING
            or operation.request_payload.get("run_id") != run_id
            or operation.request_payload.get("recipe_digest") != materials.recipe_digest
            or operation.request_payload.get("dataset_digest") != materials.dataset_digest
            or operation.request_payload.get("backend_digest") != result.backend_digest
            or operation.request_payload.get("runtime_digest") != result.runtime_digest
            or tuple(cast(tuple[str, ...], operation.request_payload.get("trainable_targets", ())))
            != result.changed_targets
            or not result.finite_gradients
            or not result.frozen_parameters_unchanged
        ):
            raise TrainingServiceError("calibration_invalid")
        evidence = TrainingCalibrationEvidence(
            evidence_id=f"calibration_{result.calibration_digest[7:]}",
            run_id=run_id,
            probe_evidence_id=str(probe[0]),
            recipe_digest=training_recipe_id(recipe),
            dataset_digest=recipe.dataset_digest,
            backend_digest=_json_digest(recipe.model_dump(mode="json")["backend_config"]),
            runtime_digest=materials.runtime_digest,
            material_digest=materials.material_digest,
            source_digest=materials.source_digest,
            calibration_digest=result.calibration_digest,
            succeeded=True,
        )

        def publish(connection: sqlite3.Connection) -> None:
            self.repository.record_calibration_with_connection(connection, evidence)
            self.repository.append_event_with_connection(
                connection,
                run_id,
                TrainingRunEvent(
                    sequence=2,
                    previous_state="probed",
                    new_state="calibrated",
                    payload={"calibration_evidence_id": evidence.evidence_id},
                ),
                expected_version=1,
            )
            evidence_digest = str(
                connection.execute(
                    "SELECT training_operation_evidence_digest(?,?,?)",
                    (operation.operation_id, "calibration", evidence.evidence_id),
                ).fetchone()[0]
            )
            self.repository.link_operation_evidence_with_connection(
                connection,
                operation_id=operation.operation_id,
                attempt=operation.attempt,
                claim_token=cast(str, operation.claim_token),
                evidence_kind="calibration",
                evidence_id=evidence.evidence_id,
                evidence_digest=evidence_digest,
            )
            self.repository.complete_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.CALIBRATION,
                idempotency_key=result.request_id,
                claimant_id=cast(str, operation.claimant_id),
                claim_token=cast(str, operation.claim_token),
                attempt=operation.attempt,
                receipt_payload={
                    "run_id": run_id,
                    "lifecycle_version": 2,
                    "lifecycle_state": "calibrated",
                    "evidence_id": evidence.evidence_id,
                    "evidence_digest": evidence_digest,
                },
                result_payload={
                    "schema_version": "trainer-calibration-result.v1",
                    "result": result.model_dump(mode="json"),
                },
            )

        self.repository.atomic(publish)

    @_public_guard
    def preregister_run(self, run_id: str, *, idempotency_key: str) -> str:
        run = self.repository.get_run(run_id)
        if run.state != "calibrated" or not idempotency_key:
            raise TrainingServiceError("run_preregistration_invalid")
        record = self.repository.get_recipe(run.run.recipe_id)
        materials = self._training_materials(record)
        request_digest = _json_digest(
            {
                "schema_version": "character-training-request.v1",
                "run_id": run_id,
                "recipe_id": record.recipe_id,
                "materials": materials.model_dump(mode="json"),
                "idempotency_key": idempotency_key,
            }
        )

        def publish(connection: sqlite3.Connection) -> str:
            probe = connection.execute(
                "SELECT id FROM training_probe_evidence WHERE run_id=? AND reviewer_id=? "
                "AND succeeded=1 ORDER BY created_at DESC LIMIT 1",
                (run_id, self.reviewer_id),
            ).fetchone()
            calibration = connection.execute(
                "SELECT id FROM training_calibration_evidence WHERE run_id=? AND reviewer_id=? "
                "AND succeeded=1 ORDER BY created_at DESC LIMIT 1",
                (run_id, self.reviewer_id),
            ).fetchone()
            if probe is None or calibration is None:
                raise TrainingServiceError("run_prerequisite_missing")
            rebound = self.repository.bind_idempotency_with_connection(
                connection,
                run_id=run_id,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                probe_evidence_id=str(probe[0]),
                calibration_evidence_id=str(calibration[0]),
            )
            if rebound != run_id:
                return rebound
            self.repository.append_event_with_connection(
                connection,
                run_id,
                TrainingRunEvent(
                    sequence=3,
                    previous_state="calibrated",
                    new_state="preregistered",
                    payload={
                        "idempotency_key": idempotency_key,
                        "request_digest": request_digest,
                        "probe_evidence_id": str(probe[0]),
                        "calibration_evidence_id": str(calibration[0]),
                    },
                ),
                expected_version=2,
            )
            return run_id

        return str(self.repository.atomic(publish))

    @_public_guard
    def start_training(
        self,
        run_id: str,
        *,
        request_id: str,
        scratch_root: Path,
        progress: Callable[[TrainingProgress], None] | None = None,
    ) -> TrainingPublicationReceipt:
        with self.repository.database.connect() as connection:
            try:
                replay = self.repository.reopen_operation_with_connection(
                    connection, TrainingOperationKind.START, request_id
                )
            except TrainingNotFound:
                replay = None
        if replay is not None:
            if replay.run_id != run_id:
                raise TrainingServiceError("operation_request_conflict")
            if replay.status is TrainingOperationStatus.COMPLETED:
                return TrainingPublicationReceipt.model_validate(replay.result_payload)
            raise TrainingServiceError("operation_in_progress")
        run = self.repository.get_run(run_id)
        if run.state != "preregistered":
            raise TrainingServiceError("training_state_invalid")
        prepared = self._prepare_training_execution(
            run_id, request_id=request_id, scratch_root=scratch_root
        )
        record = prepared.record
        materials = prepared.request.materials
        with self.repository.database.connect() as connection:
            preregistration = connection.execute(
                "SELECT idempotency_key,request_digest FROM training_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        if preregistration is None:
            raise TrainingServiceError("run_preregistration_invalid")
        artifact_ids = {
            "base_model": record.binding.base_model_artifact_id,
            "source_snapshot": record.binding.source_snapshot_artifact_id,
            **{
                f"lightning:{name}": artifact_id
                for name, artifact_id in record.binding.lightning_adapter_artifact_ids.items()
            },
        }
        request_payload = StartOperationRequest(
            request_id=request_id,
            run_id=run_id,
            recipe_digest=materials.recipe_digest,
            dataset_digest=materials.dataset_digest,
            backend_digest=materials.backend_digest,
            runtime_digest=materials.runtime_digest,
            material_digest=materials.material_digest,
            source_digest=materials.source_digest,
            preregistration_id=str(preregistration["idempotency_key"]),
            preregistration_request_digest=str(preregistration["request_digest"]),
            checkpoint_steps=checkpoint_steps(record.recipe.total_steps),
            checkpoint_fractions=(0.25, 0.5, 0.75, 1.0),
            material_artifact_ids=artifact_ids,
        ).model_dump(mode="json")

        def begin(connection: sqlite3.Connection) -> TrainingOrchestrationOperation:
            operation = self.repository.begin_operation_with_connection(
                connection,
                operation_id=f"operation_start_{_json_digest(request_payload)[7:]}",
                run_id=run_id,
                operation_kind=TrainingOperationKind.START,
                idempotency_key=request_id,
                request_payload=request_payload,
            )
            if operation.status is TrainingOperationStatus.COMPLETED:
                return operation
            claim = self.repository.claim_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.START,
                idempotency_key=request_id,
                claimant_id=f"training-service:{self.reviewer_id}",
                lease_seconds=57600,
            )
            if not claim.claimed:
                raise TrainingServiceError("operation_in_progress")
            return claim.operation

        operation = cast(TrainingOrchestrationOperation, self.repository.atomic(begin))
        if operation.status is TrainingOperationStatus.COMPLETED:
            return TrainingPublicationReceipt.model_validate(operation.result_payload)
        self.repository.append_event(
            run_id,
            TrainingRunEvent(
                sequence=4,
                previous_state="preregistered",
                new_state="training",
                payload={"request_digest": _json_digest({"request_id": request_id})},
            ),
            expected_version=3,
        )
        return self._execute_training(
            prepared,
            progress=progress,
            operation=operation,
        )

    @_public_guard
    def resume_training(
        self,
        run_id: str,
        *,
        request_id: str,
        scratch_root: Path,
        progress: Callable[[TrainingProgress], None] | None = None,
    ) -> TrainingPublicationReceipt:
        with self.repository.database.connect() as connection:
            try:
                replay = self.repository.reopen_operation_with_connection(
                    connection, TrainingOperationKind.RESUME, request_id
                )
            except TrainingNotFound:
                replay = None
        if replay is not None:
            if replay.run_id != run_id:
                raise TrainingServiceError("operation_request_conflict")
            if replay.status is TrainingOperationStatus.COMPLETED:
                return TrainingPublicationReceipt.model_validate(replay.result_payload)
            if not (
                replay.status is TrainingOperationStatus.RUNNING
                and replay.lease_expires_at is not None
                and replay.lease_expires_at <= datetime.now(UTC)
            ):
                raise TrainingServiceError("operation_in_progress")
        run = self.repository.get_run(run_id)
        checkpoints = self.repository.list_checkpoints(run_id)
        if run.state not in {"training", "cancelled"} or not checkpoints:
            raise TrainingServiceError("resume_state_invalid")
        prepared = self._prepare_training_execution(
            run_id, request_id=request_id, scratch_root=scratch_root
        )
        parent_checkpoint_id = prepared.request.parent_checkpoint_id
        parent_checkpoint_digest = prepared.request.parent_checkpoint_digest
        if parent_checkpoint_id is None or parent_checkpoint_digest is None:
            raise TrainingServiceError("resume_state_invalid")
        with self.repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT role,artifact_id,artifact_digest,size_bytes "
                "FROM training_checkpoint_inventory_artifacts WHERE checkpoint_id=? "
                "ORDER BY ordinal",
                (parent_checkpoint_id,),
            ).fetchall()
        inventory = tuple(
            ResumeInventoryArtifact(
                role=cast(Any, str(row["role"])),
                artifact_id=str(row["artifact_id"]),
                digest=str(row["artifact_digest"]),
                size_bytes=int(row["size_bytes"]),
            )
            for row in rows
        )
        request_payload = ResumeOperationRequest(
            request_id=request_id,
            run_id=run_id,
            recipe_digest=prepared.request.materials.recipe_digest,
            dataset_digest=prepared.request.materials.dataset_digest,
            backend_digest=prepared.request.materials.backend_digest,
            runtime_digest=prepared.request.materials.runtime_digest,
            material_digest=prepared.request.materials.material_digest,
            source_digest=prepared.request.materials.source_digest,
            parent_checkpoint_id=parent_checkpoint_id,
            parent_checkpoint_digest=parent_checkpoint_digest,
            parent_inventory=inventory,
        ).model_dump(mode="json")

        def begin(connection: sqlite3.Connection) -> TrainingOrchestrationOperation:
            operation = self.repository.begin_operation_with_connection(
                connection,
                operation_id=f"operation_resume_{_json_digest(request_payload)[7:]}",
                run_id=run_id,
                operation_kind=TrainingOperationKind.RESUME,
                idempotency_key=request_id,
                request_payload=request_payload,
            )
            if operation.status is TrainingOperationStatus.COMPLETED:
                return operation
            claim = self.repository.claim_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.RESUME,
                idempotency_key=request_id,
                claimant_id=f"training-service:{self.reviewer_id}",
                lease_seconds=57600,
            )
            if not claim.claimed:
                raise TrainingServiceError("operation_in_progress")
            return claim.operation

        operation = cast(TrainingOrchestrationOperation, self.repository.atomic(begin))
        if operation.status is TrainingOperationStatus.COMPLETED:
            return TrainingPublicationReceipt.model_validate(operation.result_payload)
        parent_stack = ExitStack()
        primary_error: BaseException | None = None
        try:
            if run.state == "cancelled":
                with self.repository.database.connect() as connection:
                    lifecycle_row = connection.execute(
                        "SELECT lifecycle_version FROM training_runs WHERE id=?",
                        (run_id,),
                    ).fetchone()
                if lifecycle_row is None:
                    raise TrainingServiceError("resume_state_invalid")
                lifecycle_version = int(lifecycle_row["lifecycle_version"])
                self.repository.append_event(
                    run_id,
                    TrainingRunEvent(
                        sequence=lifecycle_version + 1,
                        previous_state="cancelled",
                        new_state="training",
                        payload={
                            "request_digest": _json_digest({"request_id": request_id}),
                            "parent_checkpoint_id": parent_checkpoint_id,
                            "parent_checkpoint_digest": parent_checkpoint_digest,
                        },
                    ),
                    expected_version=lifecycle_version,
                )
            with self.repository.database.connect() as connection:
                handles = tuple(
                    parent_stack.enter_context(
                        self.artifacts.open_verified_with_connection(
                            connection,
                            item.artifact_id,
                            sha256=item.digest[7:],
                            byte_size=item.size_bytes,
                            media_type="application/octet-stream",
                        )
                    )
                    for item in inventory
                )
                return self._execute_training(
                    prepared,
                    progress=progress,
                    operation=operation,
                    parent_handles=handles,
                    close_parent_handles=parent_stack.close,
                )
        except TrainingServiceError as error:
            primary_error = error
            raise
        except Exception as error:
            primary_error = error
            if self._compensate_resume_failure(operation, "training_backend_failed"):
                raise TrainingServiceError("training_backend_failed") from None
            raise
        finally:
            try:
                parent_stack.close()
            except Exception:
                if primary_error is None:
                    raise

    def _prepare_training_execution(
        self,
        run_id: str,
        *,
        request_id: str,
        scratch_root: Path,
    ) -> _PreparedTrainingExecution:
        run = self.repository.get_run(run_id)
        record = self.repository.get_recipe(run.run.recipe_id)
        recipe = record.recipe
        materials = self._training_materials(record)
        backend = self.trainer_registry.resolve(self.reviewer_id, recipe.backend_id)
        prior = self.repository.list_checkpoints(run_id)
        parent = prior[-1].checkpoint if prior else None
        parent_evidence_digest = (
            self._checkpoint_evidence_digest(parent.checkpoint_id) if parent is not None else None
        )
        telemetry_offset = len(self.repository.list_telemetry(run_id))
        schedule = checkpoint_steps(recipe.total_steps)
        remaining = tuple(step for step in schedule if parent is None or step > parent.step)
        if not remaining:
            raise TrainingServiceError("resume_complete")

        raw_optimizer_parameters = recipe.backend_config.get("optimizer_parameters", {})
        if not isinstance(raw_optimizer_parameters, Mapping) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, (bool, int, float, str))
            for name, value in raw_optimizer_parameters.items()
        ):
            raise TrainingServiceError("optimizer_parameters_invalid")
        optimizer_parameters: dict[str, bool | int | float | str] = {
            str(name): cast(bool | int | float | str, value)
            for name, value in raw_optimizer_parameters.items()
        }
        recipe_owned_optimizer_parameters = {
            "gradient_accumulation_steps",
            "learning_rate",
        }
        if recipe_owned_optimizer_parameters.intersection(optimizer_parameters):
            raise TrainingServiceError("optimizer_parameters_reserved")
        request = TrainingRequest(
            request_id=request_id,
            run_id=run_id,
            backend_id=recipe.backend_id,
            materials=materials,
            scratch_root=scratch_root,
            total_steps=recipe.total_steps,
            checkpoint_steps=remaining,
            trainable_targets=recipe.trainable_targets,
            lora=LoRAConfiguration(
                rank=recipe.lora_rank,
                alpha=recipe.lora_alpha,
                dropout=recipe.lora_dropout,
            ),
            seed=recipe.seed,
            deadline_seconds=recipe.wall_clock_limit_seconds,
            optimizer=OptimizerConfiguration(
                algorithm=recipe.optimizer,
                parameters=tuple(
                    OptimizerParameter(name=name, value=value)
                    for name, value in sorted(
                        {
                            "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
                            "learning_rate": recipe.learning_rate,
                            **optimizer_parameters,
                        }.items()
                    )
                ),
            ),
            start_step=parent.step if parent is not None else 0,
            telemetry_sequence_start=telemetry_offset + 1,
            parent_checkpoint_id=parent.checkpoint_id if parent is not None else None,
            parent_checkpoint_digest=parent_evidence_digest,
            optimizer_state_digest=(
                str(parent.manifest["optimizer_state_digest"]) if parent is not None else None
            ),
        )
        resume_evidence = (
            TrainingResumeEvidence(
                resume_id=f"resume_{_json_digest({'request_id': request_id})[7:]}",
                run_id=run_id,
                parent_checkpoint_id=parent.checkpoint_id,
                parent_checkpoint_digest=cast(str, parent_evidence_digest),
                parent_artifact_digest=parent.artifact_digest,
                recipe_digest=record.recipe_id,
                dataset_digest=recipe.dataset_digest,
                backend_digest=materials.backend_digest,
                runtime_digest=materials.runtime_digest,
                material_digest=materials.material_digest,
                source_digest=materials.source_digest,
                optimizer_state_digest=str(parent.manifest["optimizer_state_digest"]),
            )
            if parent is not None
            else None
        )
        return _PreparedTrainingExecution(
            record=record,
            backend=backend,
            request=request,
            resume_evidence=resume_evidence,
            telemetry_offset=telemetry_offset,
        )

    def _execute_training(
        self,
        prepared: _PreparedTrainingExecution,
        *,
        progress: Callable[[TrainingProgress], None] | None,
        operation: TrainingOrchestrationOperation | None = None,
        parent_handles: tuple[VerifiedArtifactHandle, ...] = (),
        close_parent_handles: Callable[[], None] | None = None,
    ) -> TrainingPublicationReceipt:
        record = prepared.record
        recipe = record.recipe
        backend = prepared.backend
        request = prepared.request
        resume_evidence = prepared.resume_evidence
        telemetry_offset = prepared.telemetry_offset
        run_id = request.run_id
        request_id = request.request_id
        observed: list[TrainingProgress] = []

        def observe(item: TrainingProgress) -> None:
            if observed and item.step <= observed[-1].step:
                raise TrainingServiceError("training_progress_invalid")
            observed.append(item)
            if progress is not None:
                progress(item)

        def cancelled() -> bool:
            with self.repository.database.connect() as connection:
                return (
                    connection.execute(
                        "SELECT 1 FROM training_cancellation_requests request "
                        "WHERE request.run_id=? AND request.reviewer_id=? "
                        "AND NOT EXISTS(SELECT 1 FROM training_cancellation_evidence evidence "
                        "WHERE evidence.cancellation_request_id=request.id) LIMIT 1",
                        (run_id, self.reviewer_id),
                    ).fetchone()
                    is not None
                )

        try:
            with self.repository.database.connect() as connection, ExitStack() as stack:
                handles = self._verify_bound_artifacts(connection, recipe, record.binding, stack)
                self._provenance_with_connection(connection, recipe, record.binding)
                result = backend.train(request, observe, cancelled)
                for handle in handles:
                    self.artifacts.reverify_open_handle(connection, handle)
            validate_training_result(request, result)
            self._publish_training_result(
                record,
                result,
                observed,
                telemetry_offset=telemetry_offset,
                resume_evidence=resume_evidence,
                operation=operation,
                parent_handles=parent_handles,
                close_parent_handles=close_parent_handles,
            )
            if operation is None:
                return self._publication_receipt(result, "evaluating", request_id)
            with self.repository.database.connect() as connection:
                completed = self.repository.reopen_operation_with_connection(
                    connection, operation.operation_kind, operation.idempotency_key
                )
            return TrainingPublicationReceipt.model_validate(completed.result_payload)
        except TrainingCancelled as cancellation:
            try:
                self._publish_cancellation(
                    request,
                    cancellation,
                    observed,
                    parent_handles=parent_handles,
                )
            except ArtifactError:
                self._fail_training(run_id, "training_result_invalid")
                raise TrainingServiceError("training_result_invalid") from None
            if operation is not None and operation.claimant_id and operation.claim_token:
                self.repository.atomic(
                    lambda connection: self.repository.fail_operation_with_connection(
                        connection,
                        operation_kind=operation.operation_kind,
                        idempotency_key=operation.idempotency_key,
                        claimant_id=cast(str, operation.claimant_id),
                        claim_token=cast(str, operation.claim_token),
                        attempt=operation.attempt,
                        failure_digest=_json_digest({"reason": "training_cancelled"}),
                    )
                )
            raise TrainingServiceError("training_cancelled") from None
        except TrainingServiceError:
            raise
        except (ArtifactError, TrainerResultInvalid, TrainingConflict, ValueError):
            if self._compensate_resume_failure(operation, "training_result_invalid"):
                raise TrainingServiceError("training_result_invalid") from None
            self._fail_training(run_id, "training_result_invalid")
            raise TrainingServiceError("training_result_invalid") from None
        except Exception:
            if self._compensate_resume_failure(operation, "training_backend_failed"):
                raise TrainingServiceError("training_backend_failed") from None
            self._fail_training(run_id, "training_backend_failed")
            raise TrainingServiceError("training_backend_failed") from None

    def _compensate_resume_failure(
        self,
        operation: TrainingOrchestrationOperation | None,
        reason: str,
    ) -> bool:
        if (
            operation is None
            or operation.operation_kind is not TrainingOperationKind.RESUME
            or operation.lifecycle_state != "cancelled"
            or operation.claim_token is None
        ):
            return False
        failure_digest = _json_digest({"reason": reason})
        created_at = datetime.now(UTC).isoformat()

        def compensate(connection: sqlite3.Connection) -> None:
            connection.execute(
                "INSERT INTO training_resume_attempt_failures "
                "(operation_id,attempt,claim_token,failure_digest,created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    operation.operation_id,
                    operation.attempt,
                    operation.claim_token,
                    failure_digest,
                    created_at,
                ),
            )
            connection.execute(
                "UPDATE training_orchestration_operations SET lease_expires_at=?,updated_at=? "
                "WHERE id=? AND status='running' AND attempt=? AND claim_token=?",
                (
                    "1970-01-01T00:00:00+00:00",
                    created_at,
                    operation.operation_id,
                    operation.attempt,
                    operation.claim_token,
                ),
            )
            row = connection.execute(
                "SELECT lifecycle_version FROM training_runs WHERE id=?",
                (operation.run_id,),
            ).fetchone()
            if row is None:
                raise TrainingConflict("training run missing")
            version = int(row["lifecycle_version"])
            self.repository.append_event_with_connection(
                connection,
                operation.run_id,
                TrainingRunEvent(
                    sequence=version + 1,
                    previous_state="training",
                    new_state="cancelled",
                    payload={
                        "resume_operation_id": operation.operation_id,
                        "resume_attempt": operation.attempt,
                        "resume_failure_digest": failure_digest,
                    },
                ),
                expected_version=version,
            )

        self.repository.atomic(compensate)
        return True

    def _publish_training_result(
        self,
        record: RecipeRecord,
        result: TrainingResult,
        observed: list[TrainingProgress],
        cancellation: TrainingCancellationEvidence | None = None,
        telemetry_offset: int = 0,
        resume_evidence: TrainingResumeEvidence | None = None,
        operation: TrainingOrchestrationOperation | None = None,
        parent_handles: tuple[VerifiedArtifactHandle, ...] = (),
        close_parent_handles: Callable[[], None] | None = None,
    ) -> None:
        files = tuple(item for checkpoint in result.checkpoints for item in checkpoint.files)
        staged: list[StagedArtifact] = []
        try:
            for item in files:
                staged.append(
                    self.artifacts.stage(
                        result.output_root / item.relative_path,
                        "application/octet-stream",
                    )
                )
            verified_inputs = tuple(
                self.artifacts.get(identifier)
                for identifier in (
                    record.binding.base_model_artifact_id,
                    record.binding.source_snapshot_artifact_id,
                    *record.binding.lightning_adapter_artifact_ids.values(),
                )
            )

            def publish(
                connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]
            ) -> None:
                self._provenance_with_connection(connection, record.recipe, record.binding)
                if resume_evidence is not None:
                    self.repository.record_resume_with_connection(connection, resume_evidence)
                for sequence, item in enumerate(observed, start=telemetry_offset + 1):
                    self.repository.append_telemetry_with_connection(
                        connection,
                        result.run_id,
                        TrainingTelemetry(
                            sequence=sequence,
                            step=item.step,
                            payload=item.model_dump(mode="json"),
                        ),
                    )
                self._publish_checkpoints(
                    connection,
                    record,
                    result,
                    cancelled=cancellation is not None,
                    telemetry_offset=telemetry_offset,
                )
                artifact_offset = 0
                for checkpoint in result.checkpoints:
                    checkpoint_artifacts = artifacts[
                        artifact_offset : artifact_offset + len(checkpoint.files)
                    ]
                    self.repository.bind_checkpoint_inventory_artifacts_with_connection(
                        connection,
                        checkpoint_id=checkpoint.checkpoint_id,
                        artifact_ids_by_role={
                            item.role: artifact.id
                            for item, artifact in zip(
                                checkpoint.files, checkpoint_artifacts, strict=True
                            )
                        },
                    )
                    artifact_offset += len(checkpoint.files)
                lifecycle_row = connection.execute(
                    "SELECT lifecycle_version FROM training_runs WHERE id=?",
                    (result.run_id,),
                ).fetchone()
                if lifecycle_row is None:
                    raise TrainingConflict("training run missing")
                lifecycle_version = int(lifecycle_row["lifecycle_version"])
                if cancellation is not None:
                    self.repository.record_cancellation_with_connection(
                        connection, cancellation
                    )
                    event = TrainingRunEvent(
                        sequence=lifecycle_version + 1,
                        previous_state="training",
                        new_state="cancelled",
                        payload={
                            "cancellation_id": cancellation.cancellation_id,
                            "cancellation_request_id": (
                                cancellation.cancellation_request_id
                            ),
                            "request_digest": cancellation.request_digest,
                            "checkpoint_digest": cancellation.checkpoint_digest,
                            "boundary_step": cancellation.boundary_step,
                        },
                    )
                else:
                    event = TrainingRunEvent(
                        sequence=lifecycle_version + 1,
                        previous_state="training",
                        new_state="evaluating",
                        payload={"telemetry_digest": result.telemetry_digest},
                    )
                self.repository.append_event_with_connection(
                    connection,
                    result.run_id,
                    event,
                    expected_version=lifecycle_version,
                )
                if operation is not None:
                    for handle in parent_handles:
                        self.artifacts.reverify_open_handle(connection, handle)
                    if close_parent_handles is not None:
                        close_parent_handles()
                    if operation.operation_kind is TrainingOperationKind.CANCEL:
                        if cancellation is None:
                            raise TrainingServiceError("cancellation_evidence_missing")
                        self._complete_cancel_operation(
                            connection, operation, cancellation
                        )
                    else:
                        self._complete_collection_operation(
                            connection,
                            operation,
                            result,
                            lifecycle_state=cast(
                                Literal["evaluating", "cancelled"], event.new_state
                            ),
                        )

            self.artifacts.commit_batch(tuple(staged), publish, verify_artifacts=verified_inputs)
        except Exception:
            for staged_item in staged:
                try:
                    self.artifacts.discard(staged_item)
                except ArtifactError:
                    pass
            raise

    @staticmethod
    def _publication_receipt(
        result: TrainingResult,
        lifecycle_state: Literal["evaluating", "cancelled"],
        request_id: str,
    ) -> TrainingPublicationReceipt:
        return TrainingPublicationReceipt(
            request_id=request_id,
            run_id=result.run_id,
            lifecycle_state=lifecycle_state,
            checkpoint_ids=tuple(item.checkpoint_id for item in result.checkpoints),
            telemetry_digest=result.telemetry_digest,
        )

    def _complete_collection_operation(
        self,
        connection: sqlite3.Connection,
        operation: TrainingOrchestrationOperation,
        result: TrainingResult,
        *,
        lifecycle_state: Literal["evaluating", "cancelled"],
    ) -> None:
        if operation.claim_token is None or operation.claimant_id is None:
            raise TrainingServiceError("operation_claim_missing")
        evidence: dict[str, list[dict[str, object]]] = {
            "events": [],
            "telemetry": [],
            "checkpoints": [],
            "artifacts": [],
            "inventory": [],
        }
        identifiers: list[tuple[str, str, dict[str, object]]] = []
        operation_started_at = cast(datetime, operation.updated_at).isoformat()
        resume_entry: dict[str, object] | None = None
        for row in connection.execute(
            "SELECT sequence FROM training_run_events WHERE run_id=? "
            "AND julianday(created_at)>=julianday(?) ORDER BY sequence",
            (result.run_id, operation_started_at),
        ):
            value = str(row["sequence"])
            identifiers.append(("event", value, {"sequence": int(value)}))
        for row in connection.execute(
            "SELECT sequence,step FROM training_telemetry WHERE run_id=? "
            "AND julianday(created_at)>=julianday(?) ORDER BY sequence",
            (result.run_id, operation_started_at),
        ):
            identifiers.append(
                (
                    "telemetry",
                    str(row["sequence"]),
                    {"sequence": int(row["sequence"]), "step": int(row["step"])},
                )
            )
        for checkpoint in result.checkpoints:
            identifiers.append(("checkpoint", checkpoint.checkpoint_id, {"step": checkpoint.step}))
        if operation.operation_kind is TrainingOperationKind.RESUME:
            row = connection.execute(
                "SELECT id FROM training_resume_evidence WHERE run_id=? "
                "AND julianday(created_at)>=julianday(?) ORDER BY created_at DESC LIMIT 1",
                (result.run_id, operation_started_at),
            ).fetchone()
            if row is None:
                raise TrainingServiceError("resume_evidence_missing")
            identifiers.append(("resume", str(row["id"]), {}))
        bindings = connection.execute(
            "SELECT binding.checkpoint_id,binding.role,binding.artifact_id,artifact.byte_size "
            "FROM training_checkpoint_inventory_artifacts binding JOIN artifacts artifact "
            "ON artifact.id=binding.artifact_id JOIN training_checkpoints checkpoint "
            "ON checkpoint.id=binding.checkpoint_id WHERE checkpoint.run_id=? "
            "AND julianday(checkpoint.created_at)>=julianday(?) "
            "ORDER BY checkpoint.step,binding.ordinal",
            (result.run_id, operation_started_at),
        ).fetchall()
        for row in bindings:
            identifiers.append(
                (
                    "inventory",
                    f"{row['checkpoint_id']}:{row['role']}",
                    {"checkpoint_id": str(row["checkpoint_id"]), "role": str(row["role"])},
                )
            )
        for artifact_id in sorted({str(row["artifact_id"]) for row in bindings}):
            size = next(
                int(row["byte_size"]) for row in bindings if row["artifact_id"] == artifact_id
            )
            identifiers.append(("artifact", artifact_id, {"size_bytes": size}))
        names = {
            "event": "events",
            "telemetry": "telemetry",
            "checkpoint": "checkpoints",
            "artifact": "artifacts",
            "inventory": "inventory",
            "resume": "resume",
        }
        for kind, identifier, metadata in identifiers:
            digest = str(
                connection.execute(
                    "SELECT training_operation_evidence_digest(?,?,?)",
                    (operation.operation_id, kind, identifier),
                ).fetchone()[0]
            )
            self.repository.link_operation_evidence_with_connection(
                connection,
                operation_id=operation.operation_id,
                attempt=operation.attempt,
                claim_token=operation.claim_token,
                evidence_kind=kind,
                evidence_id=identifier,
                evidence_digest=digest,
            )
            if kind == "resume":
                resume_entry = {"evidence_id": identifier, "digest": digest}
            else:
                evidence[names[kind]].append(
                    {"evidence_id": identifier, "digest": digest, **metadata}
                )
        lifecycle_version = int(
            connection.execute(
                "SELECT lifecycle_version FROM training_runs WHERE id=?", (result.run_id,)
            ).fetchone()[0]
        )
        receipt: dict[str, object] = {
            "run_id": result.run_id,
            "lifecycle_version": lifecycle_version,
            "lifecycle_state": lifecycle_state,
            **evidence,
            "evaluation": None,
        }
        if operation.operation_kind is TrainingOperationKind.RESUME:
            if resume_entry is None:
                raise TrainingServiceError("resume_evidence_missing")
            receipt["resume"] = resume_entry
        durable = self._publication_receipt(
            result, lifecycle_state, str(operation.request_payload["request_id"])
        )
        self.repository.complete_operation_with_connection(
            connection,
            operation_kind=operation.operation_kind,
            idempotency_key=operation.idempotency_key,
            claimant_id=operation.claimant_id,
            claim_token=operation.claim_token,
            attempt=operation.attempt,
            receipt_payload=receipt,
            result_payload=durable.model_dump(mode="json"),
        )

    def _complete_cancel_operation(
        self,
        connection: sqlite3.Connection,
        operation: TrainingOrchestrationOperation,
        cancellation: TrainingCancellationEvidence,
    ) -> None:
        if operation.claim_token is None or operation.claimant_id is None:
            raise TrainingServiceError("operation_claim_missing")
        event = connection.execute(
            "SELECT sequence FROM training_run_events WHERE run_id=? AND new_state='cancelled' "
            "ORDER BY sequence DESC LIMIT 1",
            (operation.run_id,),
        ).fetchone()
        checkpoint = connection.execute(
            "SELECT id,step FROM training_checkpoints WHERE run_id=? AND step=? "
            "AND publication_state='finalized'",
            (operation.run_id, cancellation.boundary_step),
        ).fetchone()
        if event is None or checkpoint is None:
            raise TrainingServiceError("cancellation_evidence_missing")
        entries: dict[str, dict[str, object]] = {}
        for kind, identifier, metadata in (
            ("cancel", cancellation.cancellation_id, {}),
            ("event", str(event["sequence"]), {"sequence": int(event["sequence"])}),
            (
                "checkpoint",
                str(checkpoint["id"]),
                {"step": int(checkpoint["step"])},
            ),
        ):
            digest = str(
                connection.execute(
                    "SELECT training_operation_evidence_digest(?,?,?)",
                    (operation.operation_id, kind, identifier),
                ).fetchone()[0]
            )
            self.repository.link_operation_evidence_with_connection(
                connection,
                operation_id=operation.operation_id,
                attempt=operation.attempt,
                claim_token=operation.claim_token,
                evidence_kind=kind,
                evidence_id=identifier,
                evidence_digest=digest,
            )
            entries[kind] = {"evidence_id": identifier, "digest": digest, **metadata}
        receipt = {
            "run_id": operation.run_id,
            "lifecycle_version": 5,
            "lifecycle_state": "cancelled",
            "cancellation": entries["cancel"],
            "terminal_event": entries["event"],
            "boundary_checkpoint": entries["checkpoint"],
            "cancellation_request_id": cancellation.cancellation_request_id,
            "request_digest": cancellation.request_digest,
            "requested_step": cancellation.requested_step,
            "boundary_step": cancellation.boundary_step,
        }
        self.repository.complete_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.CANCEL,
            idempotency_key=operation.idempotency_key,
            claimant_id=operation.claimant_id,
            claim_token=operation.claim_token,
            attempt=operation.attempt,
            receipt_payload=receipt,
            result_payload=receipt,
        )

    def _publish_checkpoints(
        self,
        connection: sqlite3.Connection,
        record: RecipeRecord,
        result: TrainingResult,
        *,
        cancelled: bool = False,
        telemetry_offset: int = 0,
    ) -> None:
        character_id = self.repository.get_dataset(record.recipe.dataset_digest).character_id
        prior = self.repository.list_checkpoints(result.run_id)
        parent_digest: str | None = (
            self._checkpoint_evidence_digest(prior[-1].checkpoint.checkpoint_id) if prior else None
        )
        for checkpoint in result.checkpoints:
            by_role = {item.role: item for item in checkpoint.files}
            inventory = tuple(
                TrainingCheckpointArtifact.model_validate(item.model_dump(mode="python"))
                for item in checkpoint.files
            )
            evidence = TrainingCheckpointEvidence(
                run_id=result.run_id,
                recipe_digest=record.recipe_id,
                dataset_digest=record.recipe.dataset_digest,
                backend_digest=result.materials.backend_digest,
                runtime_digest=result.materials.runtime_digest,
                material_digest=result.materials.material_digest,
                source_digest=result.materials.source_digest,
                step=checkpoint.step,
                fraction=checkpoint.fraction,
                parent_checkpoint_digest=parent_digest,
                safetensors_digest=by_role["weights"].digest,
                safetensors_size_bytes=by_role["weights"].size_bytes,
                optimizer_state_digest=by_role["optimizer"].digest,
                trainer_state_digest=by_role["trainer_state"].digest,
                telemetry_digest=checkpoint.telemetry_digest,
                telemetry_sequence_start=telemetry_offset + 1,
                telemetry_sequence_end=telemetry_offset
                + checkpoint.step
                - (prior[-1].checkpoint.step if prior else 0),
                trainable_targets_digest=checkpoint.trainable_targets_digest,
                frozen_parameters_digest=checkpoint.frozen_parameters_digest,
                finite_gradient_proof_digest=checkpoint.finite_gradient_proof_digest,
                cancellation_boundary=cancelled and checkpoint is result.checkpoints[-1],
                inventory=inventory,
            )
            manifest = {
                "schema_version": "character-training-checkpoint.v1",
                "run_id": result.run_id,
                "reviewer_id": self.reviewer_id,
                "recipe_id": record.recipe_id,
                "dataset_digest": record.recipe.dataset_digest,
                "character_id": character_id,
                "step": checkpoint.step,
                "fraction": checkpoint.fraction,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "artifact_digest": by_role["weights"].digest,
                "safetensors_size_bytes": by_role["weights"].size_bytes,
                "backend_digest": result.materials.backend_digest,
                "runtime_digest": result.materials.runtime_digest,
                "material_digest": result.materials.material_digest,
                "source_digest": result.materials.source_digest,
                "optimizer_state_digest": by_role["optimizer"].digest,
                "trainer_state_digest": by_role["trainer_state"].digest,
                "telemetry_digest": checkpoint.telemetry_digest,
                "telemetry_sequence_start": telemetry_offset + 1,
                "telemetry_sequence_end": telemetry_offset
                + checkpoint.step
                - (prior[-1].checkpoint.step if prior else 0),
                "trainable_targets_digest": checkpoint.trainable_targets_digest,
                "frozen_parameters_digest": checkpoint.frozen_parameters_digest,
                "finite_gradient_proof_digest": checkpoint.finite_gradient_proof_digest,
                "cancellation_boundary": cancelled and checkpoint is result.checkpoints[-1],
                "inventory": [item.model_dump(mode="json") for item in inventory],
            }
            persisted = TrainingCheckpoint(
                checkpoint_id=checkpoint.checkpoint_id,
                run_id=result.run_id,
                step=checkpoint.step,
                fraction=checkpoint.fraction,
                parent_checkpoint_id=checkpoint.parent_checkpoint_id,
                artifact_digest=by_role["weights"].digest,
                manifest_digest=_json_digest(manifest),
                manifest=manifest,
            )
            self.repository.add_checkpoint_with_connection(connection, persisted, evidence)
            parent_digest = _json_digest(evidence.model_dump(mode="json"))

    def _checkpoint_evidence_digest(self, checkpoint_id: str) -> str:
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT evidence_digest FROM training_checkpoints "
                "WHERE id=? AND reviewer_id=? AND publication_state='finalized'",
                (checkpoint_id, self.reviewer_id),
            ).fetchone()
        if row is None:
            raise TrainingServiceError("checkpoint_evidence_missing")
        return str(row["evidence_digest"])

    @_public_guard
    def request_cancellation(
        self, run_id: str, *, request_id: str, requested_step: int
    ) -> TrainingOrchestrationOperation:
        digest = _json_digest(
            {"request_id": request_id, "run_id": run_id, "requested_step": requested_step}
        )
        with self.repository.database.connect() as connection:
            try:
                existing_operation = self.repository.reopen_operation_with_connection(
                    connection, TrainingOperationKind.CANCEL, request_id
                )
            except TrainingNotFound:
                existing_operation = None
        if existing_operation is not None:
            if (
                existing_operation.run_id != run_id
                or existing_operation.request_payload.get("request_digest") != digest
                or existing_operation.request_payload.get("requested_step") != requested_step
            ):
                raise TrainingServiceError("operation_request_conflict")
            return existing_operation
        run = self.repository.get_run(run_id)
        if run.state != "training":
            raise TrainingServiceError("cancellation_state_invalid")
        record = self.repository.get_recipe(run.run.recipe_id)
        materials = self._training_materials(record)
        request_payload = CancelOperationRequest(
            request_id=request_id,
            run_id=run_id,
            recipe_digest=materials.recipe_digest,
            dataset_digest=materials.dataset_digest,
            backend_digest=materials.backend_digest,
            runtime_digest=materials.runtime_digest,
            material_digest=materials.material_digest,
            source_digest=materials.source_digest,
            cancellation_request_id=request_id,
            request_digest=digest,
            requested_step=requested_step,
        ).model_dump(mode="json")

        def create(connection: sqlite3.Connection) -> TrainingOrchestrationOperation:
            existing = connection.execute(
                "SELECT 1 FROM training_cancellation_requests WHERE id=?",
                (request_id,),
            ).fetchone()
            if existing is None:
                self.repository.request_cancellation_with_connection(
                    connection,
                    request_id=request_id,
                    run_id=run_id,
                    requested_step=requested_step,
                    request_digest=digest,
                )
            operation = self.repository.begin_operation_with_connection(
                connection,
                operation_id=f"operation_cancel_{_json_digest(request_payload)[7:]}",
                run_id=run_id,
                operation_kind=TrainingOperationKind.CANCEL,
                idempotency_key=request_id,
                request_payload=request_payload,
            )
            if operation.status is not TrainingOperationStatus.PENDING:
                return operation
            claim = self.repository.claim_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.CANCEL,
                idempotency_key=request_id,
                claimant_id=f"training-service:{self.reviewer_id}",
                lease_seconds=57600,
            )
            if not claim.claimed:
                raise TrainingServiceError("operation_in_progress")
            return claim.operation

        return cast(TrainingOrchestrationOperation, self.repository.atomic(create))

    def _publish_cancellation(
        self,
        request_model: TrainingRequest,
        cancellation: TrainingCancelled,
        observed: list[TrainingProgress],
        *,
        parent_handles: tuple[VerifiedArtifactHandle, ...] = (),
    ) -> None:
        run_id = request_model.run_id
        if (
            cancellation.result is None
            or not observed
            or observed[-1].step != cancellation.boundary_step
            or not observed[-1].checkpoint_boundary
        ):
            self._fail_training(run_id, "cancellation_evidence_missing")
            return
        validate_cancelled_training_result(
            request_model, cancellation.result, cancellation.boundary_step
        )
        with self.repository.database.connect() as connection:
            request = connection.execute(
                "SELECT id,requested_step,request_digest FROM training_cancellation_requests "
                "WHERE run_id=? AND reviewer_id=? ORDER BY created_at,id LIMIT 1",
                (run_id, self.reviewer_id),
            ).fetchone()
        if request is None:
            self._fail_training(run_id, "cancellation_evidence_missing")
            return
        result = cancellation.result
        digest = next(
            item.digest for item in result.checkpoints[-1].files if item.role == "weights"
        )
        evidence = TrainingCancellationEvidence(
            cancellation_id=f"cancel_{digest[7:]}",
            run_id=run_id,
            cancellation_request_id=str(request["id"]),
            request_digest=str(request["request_digest"]),
            requested_step=int(request["requested_step"]),
            boundary_step=cancellation.boundary_step,
            checkpoint_digest=digest,
        )
        record = self.repository.get_recipe(self.repository.get_run(run_id).run.recipe_id)
        with self.repository.database.connect() as connection:
            operation = self.repository.reopen_operation_with_connection(
                connection, TrainingOperationKind.CANCEL, str(request["id"])
            )
        telemetry_offset = len(self.repository.list_telemetry(run_id))
        self._publish_training_result(
            record,
            result,
            observed,
            cancellation=evidence,
            telemetry_offset=telemetry_offset,
            operation=operation,
            parent_handles=parent_handles,
        )

    @_public_guard
    def begin_evaluation(self, run_id: str) -> None:
        if self.repository.get_run(run_id).state != "training":
            raise TrainingServiceError("evaluation_state_invalid")
        sequence = len(self.repository.list_events(run_id)) + 1
        self.repository.append_event(
            run_id,
            TrainingRunEvent(
                sequence=sequence,
                previous_state="training",
                new_state="evaluating",
                payload={"checkpoint_count": len(self.repository.list_checkpoints(run_id))},
            ),
            expected_version=sequence - 1,
        )

    @_public_guard
    def complete_evaluation(self, run_id: str, *, qualified: bool, evidence: object) -> str:
        if self.repository.get_run(run_id).state != "evaluating":
            raise TrainingServiceError("evaluation_state_invalid")
        disposition: Literal["qualified", "rejected"] = "qualified" if qualified else "rejected"
        evidence_digest = _json_digest({"evidence": evidence})
        item = TrainingEvaluationEvidence(
            evidence_id=f"evaluation_{evidence_digest[7:]}",
            run_id=run_id,
            disposition=disposition,
            evidence_digest=evidence_digest,
        )
        sequence = len(self.repository.list_events(run_id)) + 1

        def publish(connection: sqlite3.Connection) -> None:
            self.repository.record_evaluation_with_connection(connection, item)
            self.repository.append_event_with_connection(
                connection,
                run_id,
                TrainingRunEvent(
                    sequence=sequence,
                    previous_state="evaluating",
                    new_state=disposition,
                    payload={
                        "evaluation_evidence_id": item.evidence_id,
                        "evaluation_evidence_digest": evidence_digest,
                    },
                ),
                expected_version=sequence - 1,
            )

        self.repository.atomic(publish)
        return disposition

    @_public_guard
    def export_terminal_evidence(self, run_id: str) -> bytes:
        run = self.repository.get_run(run_id)
        if run.state not in {"qualified", "rejected", "failed", "cancelled"}:
            raise TrainingServiceError("terminal_evidence_incomplete")
        with self.repository.database.connect() as connection, ExitStack() as stack:
            operation_rows = connection.execute(
                "SELECT operation_kind,idempotency_key,request_digest,request_payload_json,status,"
                "attempt,receipt_digest,receipt_payload_json,result_digest,result_payload_json,"
                "failure_digest FROM training_orchestration_operations "
                "WHERE run_id=? AND reviewer_id=? ORDER BY created_at,id",
                (run_id, self.reviewer_id),
            ).fetchall()
            evidence_tables = (
                "training_probe_evidence",
                "training_calibration_evidence",
                "training_resume_evidence",
                "training_cancellation_requests",
                "training_cancellation_evidence",
                "training_evaluation_evidence",
            )

            def normalized_rows(table: str) -> list[dict[str, object]]:
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE run_id=? AND reviewer_id=? "  # noqa: S608
                    "ORDER BY created_at,rowid",
                    (run_id, self.reviewer_id),
                ).fetchall()
                return [
                    {
                        key: (
                            json.loads(str(row[key]))
                            if key.endswith("_json")
                            else row[key]
                        )
                        for key in row.keys()
                        if key not in {"created_at", "reviewer_id"}
                    }
                    for row in rows
                ]

            telemetry = normalized_rows("training_telemetry")
            inventory = connection.execute(
                "SELECT inventory.checkpoint_id,inventory.ordinal,inventory.role,"
                "inventory.artifact_digest,inventory.size_bytes,binding.artifact_id "
                "FROM training_checkpoint_inventory inventory "
                "JOIN training_checkpoints checkpoint ON checkpoint.id=inventory.checkpoint_id "
                "JOIN training_checkpoint_inventory_artifacts binding "
                "ON binding.checkpoint_id=inventory.checkpoint_id "
                "AND binding.ordinal=inventory.ordinal WHERE checkpoint.run_id=? "
                "AND checkpoint.reviewer_id=? ORDER BY checkpoint.step,inventory.ordinal",
                (run_id, self.reviewer_id),
            ).fetchall()
            material_rows = connection.execute(
                "SELECT value artifact_id FROM training_orchestration_operations operation,"
                "json_each(operation.request_payload_json,'$.material_artifact_ids') "
                "WHERE operation.run_id=? AND operation.reviewer_id=? "
                "AND operation.operation_kind='start' ORDER BY key",
                (run_id, self.reviewer_id),
            ).fetchall()
            artifact_ids = {
                *(str(row["artifact_id"]) for row in inventory),
                *(str(row["artifact_id"]) for row in material_rows),
            }
            artifacts: list[dict[str, object]] = []
            for artifact_id in sorted(artifact_ids):
                artifact = self.artifacts.get(artifact_id)
                handle = stack.enter_context(
                    self.artifacts.open_verified_with_connection(
                        connection,
                        artifact.id,
                        sha256=artifact.sha256,
                        byte_size=artifact.byte_size,
                        media_type=artifact.media_type,
                    )
                )
                self.artifacts.reverify_open_handle(connection, handle)
                artifacts.append(
                    {
                        "artifact_id": artifact.id,
                        "digest": f"sha256:{artifact.sha256}",
                        "size_bytes": artifact.byte_size,
                        "media_type": artifact.media_type,
                    }
                )
            operations = [
                {
                    key: (
                        json.loads(str(row[key]))
                        if key.endswith("_json") and row[key] is not None
                        else row[key]
                    )
                    for key in row.keys()
                }
                for row in operation_rows
            ]
            evidence = {
                table.removeprefix("training_"): normalized_rows(table)
                for table in evidence_tables
            }
        payload = {
            "schema_version": "character-training-terminal-evidence.v1",
            "run": run.run.model_dump(mode="json"),
            "state": run.state,
            "events": [
                item.event.model_dump(mode="json") for item in self.repository.list_events(run_id)
            ],
            "checkpoints": [
                item.checkpoint.model_dump(mode="json")
                for item in self.repository.list_checkpoints(run_id)
            ],
            "failures": [
                item.failure.model_dump(mode="json")
                for item in self.repository.list_failures(run_id)
            ],
            "operations": operations,
            "evidence": evidence,
            "telemetry": telemetry,
            "checkpoint_inventory": [dict(row) for row in inventory],
            "artifacts": artifacts,
        }
        return canonical_json(payload).encode()

    def _fail_training(self, run_id: str, code: str) -> None:
        run = self.repository.get_run(run_id)
        if run.state in {"qualified", "rejected", "failed", "cancelled"}:
            return
        sequence = len(self.repository.list_events(run_id)) + 1
        failure = TrainingFailure(
            failure_id=f"failure_{uuid.uuid4().hex}", run_id=run_id, code=code, message=code
        )

        def publish(connection: sqlite3.Connection) -> None:
            self.repository.record_failure_with_connection(connection, failure)
            self.repository.append_event_with_connection(
                connection,
                run_id,
                TrainingRunEvent(
                    sequence=sequence,
                    previous_state=run.state,
                    new_state="failed",
                    payload={"failure_id": failure.failure_id, "code": code},
                ),
                expected_version=sequence - 1,
            )

        self.repository.atomic(publish)

    def _training_materials(self, record: RecipeRecord) -> TrainingMaterialHandles:
        recipe = record.recipe
        return TrainingMaterialHandles(
            recipe_digest=training_recipe_id(recipe),
            dataset_digest=recipe.dataset_digest,
            base_model_digest=recipe.base_model_digest,
            backend_digest=_json_digest(recipe.model_dump(mode="json")["backend_config"]),
            source_digest=recipe.source_code_digest,
            runtime_digest=record.binding.runtime_digest,
            material_digest=_json_digest(record.binding.model_dump(mode="json")),
        )

    @_public_guard
    def create_dataset(self, request: DatasetCreateInput) -> TrainingDatasetView:
        try:
            exported = self.character_packs.export_pack(request.character_pack_id)  # type: ignore[attr-defined]
            if exported.pack.pack_id != request.character_pack_id:
                raise ValueError
            draft = TrainingDatasetDraft(
                character_id=exported.pack.character_id,
                character_pack_digest=_json_digest(exported.pack.model_dump(mode="json")),
                canonical_outfit_id=request.canonical_outfit_id,
                reserved_token=request.reserved_token,
                samples=(),
            )
            lineage = f"training_lineage_{uuid.uuid4().hex}"
            self.repository.create_dataset_for_lineage(lineage, draft.character_id, draft)
            return self.dataset_status(lineage)
        except (AttributeError, LookupError, PermissionError, TrainingConflict, ValueError):
            raise TrainingServiceError("dataset_creation_failed") from None

    @_public_guard
    def import_media(
        self,
        dataset_id: str,
        source: Path,
        *,
        suggestions: TrainingCurationSuggestions | None = None,
    ) -> TrainingDatasetView:
        head = self._head(dataset_id)
        current = self.repository.get_dataset(head.current_dataset_id)
        prepared: list[tuple[Path, TrainingMedia, StagedArtifact]] = []
        try:
            existing = {
                x.artifact_digest for x in self.repository.list_imported_samples(head.lineage_id)
            }
            with open_training_media_collection_snapshot(source) as collection:
                for relative, snapshot in collection.entries:
                    path = Path(relative)
                    media = snapshot.media
                    staged = self.artifacts.stage(snapshot.path, media.media_type)
                    if (f"sha256:{staged.sha256}", staged.byte_size) != (
                        media.sha256,
                        media.byte_size,
                    ):
                        self.artifacts.discard(staged)
                        raise TrainingServiceError("source_changed")
                    if media.sha256 in existing or any(
                        x[1].sha256 == media.sha256 for x in prepared
                    ):
                        self.artifacts.discard(staged)
                        raise TrainingServiceError("duplicate_content")
                    prepared.append((path, media, staged))

            def persist(connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]) -> None:
                ids = list(current.draft.imported_sample_ids)
                for (path, media, _), artifact in zip(prepared, artifacts, strict=True):
                    self.artifacts.resolve_with_connection(
                        connection,
                        artifact.id,
                        sha256=media.sha256.removeprefix("sha256:"),
                        byte_size=media.byte_size,
                        media_type=media.media_type,
                    )
                    imported = _imported(
                        reviewer_id=self.reviewer_id,
                        character_id=head.character_id,
                        lineage_id=head.lineage_id,
                        artifact=artifact,
                        media=media,
                        source_lineage=unicodedata.normalize("NFC", path.stem),
                        suggestions=suggestions,
                    )
                    self.repository.persist_imported_sample_with_connection(connection, imported)
                    ids.append(imported.import_id)
                changed = current.draft.model_copy(update={"imported_sample_ids": tuple(ids)})
                self.repository.publish_dataset_head_with_connection(
                    connection,
                    lineage_id=head.lineage_id,
                    expected_head_version=head.head_version,
                    draft=changed,
                )

            self.artifacts.commit_batch(tuple(x[2] for x in prepared), persist)
            return self.dataset_status(head.lineage_id)
        except TrainingServiceError:
            self._discard(prepared)
            raise
        except (ArtifactError, OSError, TrainingConflict, TrainingMediaInspectionError, ValueError):
            self._discard(prepared)
            raise TrainingServiceError("media_import_failed") from None

    @_public_guard
    def curate_sample(
        self, dataset_id: str, import_id: str, decision: CurationInput
    ) -> TrainingDatasetView:
        head = self._head(dataset_id)
        current = self.repository.get_dataset(head.current_dataset_id)
        imported = next(
            (
                x
                for x in self.repository.list_imported_samples(head.lineage_id)
                if x.import_id == import_id
            ),
            None,
        )
        if imported is None:
            raise TrainingServiceError("sample_not_found")
        if self.repository.list_confirmations(import_id):
            raise TrainingServiceError("stale_sample")
        confirmation = _confirmation(self.reviewer_id, imported.import_id, decision)
        samples = current.draft.samples
        if confirmation.decision is CurationDecision.ACCEPT:
            sample = confirmation.accepted_sample(imported)
            if current.draft.reserved_token not in sample.caption:
                raise TrainingServiceError("reserved_token_missing")
            if sample.annotation.canonical_outfit_id != current.draft.canonical_outfit_id:
                raise TrainingServiceError("canonical_outfit_mismatch")
            samples += (sample,)
        changed = current.draft.model_copy(
            update={
                "samples": samples,
                "curation_confirmation_ids": current.draft.curation_confirmation_ids
                + (confirmation.confirmation_id,),
            }
        )
        try:
            with self.repository.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self.repository.persist_confirmation_with_connection(connection, confirmation)
                self.repository.publish_dataset_head_with_connection(
                    connection,
                    lineage_id=head.lineage_id,
                    expected_head_version=head.head_version,
                    draft=changed,
                )
                connection.commit()
            return self.dataset_status(head.lineage_id)
        except (TrainingConflict, ValueError):
            raise TrainingServiceError("curation_failed") from None

    @_public_guard
    def dataset_status(self, dataset_id: str) -> TrainingDatasetView:
        head = self._head(dataset_id, allow_sealed=True)
        draft = self.repository.get_dataset(head.current_dataset_id).draft
        return TrainingDatasetView(
            lineage_id=head.lineage_id,
            dataset_id=head.current_dataset_id,
            head_version=head.head_version,
            state=head.state,
            character_id=head.character_id,
            imported=self.repository.list_imported_samples(head.lineage_id),
            coverage=evaluate_training_dataset(draft),
        )

    @_public_guard
    def seal_dataset(self, dataset_id: str) -> FrozenTrainingDataset:
        head = self._head(dataset_id, allow_sealed=True)
        record = self.repository.get_dataset(head.current_dataset_id)
        if head.state == "sealed":
            if record.frozen_dataset is None:
                raise TrainingServiceError("sealed_dataset_invalid")
            return record.frozen_dataset
        frozen = freeze_training_dataset(record.draft)
        with self.repository.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.repository.publish_dataset_head_with_connection(
                connection,
                lineage_id=head.lineage_id,
                expected_head_version=head.head_version,
                draft=frozen,
                seal=True,
            )
            connection.commit()
        return frozen

    @_public_guard
    def preview_recipe(self, dataset_id: str, request: RecipeInput) -> TrainingRecipePreview:
        self._frozen(dataset_id)
        provenance = self._provenance(dataset_id, campaign_id=request.stage_3_campaign_id)
        if provenance[2] != request.stage_3_result_id:
            raise TrainingServiceError("prerequisite_drift")
        base = self.artifacts.get(request.base_model_artifact_id)
        adapters = {
            name: self.artifacts.get(value)
            for name, value in request.lightning_adapter_artifact_ids.items()
        }
        snapshot = _source_snapshot_bytes(self.source_root)
        digest = "sha256:" + hashlib.sha256(snapshot).hexdigest()
        material = {
            "dataset_id": dataset_id,
            "request": request.model_dump(mode="json"),
            "source_snapshot_digest": digest,
            "source_snapshot_byte_size": len(snapshot),
            "base_model_digest": f"sha256:{base.sha256}",
            "lightning_adapter_digests": {
                name: f"sha256:{artifact.sha256}" for name, artifact in adapters.items()
            },
            "runtime_identity": dict(self.runtime.identity()),
        }
        return TrainingRecipePreview(
            preview_token=_json_digest(material),
            source_snapshot_digest=digest,
            source_snapshot_byte_size=len(snapshot),
            base_model_digest=f"sha256:{base.sha256}",
            lightning_adapter_digests=material["lightning_adapter_digests"],  # type: ignore[arg-type]
        )

    @_public_guard
    def create_recipe(
        self, dataset_id: str, request: RecipeInput, *, preview_token: str | None = None
    ) -> TrainingRecipeView:
        preview = self.preview_recipe(dataset_id, request)
        if preview_token is not None and preview.preview_token != preview_token:
            raise TrainingServiceError("preview_drift")
        snapshot = _source_snapshot_bytes(self.source_root)
        if "sha256:" + hashlib.sha256(snapshot).hexdigest() != preview.source_snapshot_digest:
            raise TrainingServiceError("source_drift")
        with tempfile.TemporaryDirectory(prefix="video-lab-source-") as directory:
            path = Path(directory) / "source.snapshot"
            path.write_bytes(snapshot)
            staged = self.artifacts.stage(path, "application/octet-stream")

            def persist(
                connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]
            ) -> RecipeRecord:
                source = artifacts[0]
                self.artifacts.resolve_with_connection(
                    connection,
                    source.id,
                    sha256=preview.source_snapshot_digest[7:],
                    byte_size=preview.source_snapshot_byte_size,
                    media_type="application/octet-stream",
                )
                recipe = self._resolve_recipe(dataset_id, request, source=source)
                binding = self._recipe_binding(request, recipe, source)
                return self.repository.create_recipe(recipe, binding, connection=connection)

            verified_inputs = (
                self.artifacts.get(request.base_model_artifact_id),
                *tuple(
                    self.artifacts.get(artifact_id)
                    for _, artifact_id in sorted(request.lightning_adapter_artifact_ids.items())
                ),
            )
            record = self.artifacts.commit_batch(
                (staged,), persist, verify_artifacts=verified_inputs
            )
        return self._recipe_view(record)

    @_public_guard
    def preregister_recipe(self, recipe_id: str) -> TrainingRecipeView:
        record = self.repository.get_recipe(recipe_id)
        binding = record.binding
        before_runtime = dict(self.runtime.identity())
        try:
            with self.repository.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                with ExitStack() as verified:
                    handles = self._verify_bound_artifacts(
                        connection, record.recipe, binding, verified
                    )
                    self._provenance_with_connection(connection, record.recipe, binding)
                    if before_runtime != dict(record.recipe.runtime_identity):
                        raise TrainingServiceError("runtime_drift")
                    updated = self.repository.preregister_recipe(recipe_id, connection=connection)
                    for handle in handles:
                        self.artifacts.reverify_open_handle(connection, handle)
                    if dict(self.runtime.identity()) != before_runtime:
                        raise TrainingServiceError("runtime_drift")
                    for handle in handles:
                        self.artifacts.reverify_open_handle(connection, handle)
                    connection.commit()
            return self._recipe_view(updated)
        except TrainingServiceError:
            raise
        except (ArtifactError, TrainingConflict, sqlite3.Error, OSError, ValueError):
            raise TrainingServiceError("recipe_preregistration_failed") from None

    @_public_guard
    def export_dataset_evidence(self, dataset_id: str) -> bytes:
        head = self._head(dataset_id, allow_sealed=True)
        dataset = self.repository.get_dataset(head.current_dataset_id)
        payload = {
            "schema_version": "training-dataset-evidence.v1",
            "claim_state": head.state,
            "head_version": head.head_version,
            "dataset": (dataset.frozen_dataset or dataset.draft).model_dump(mode="json"),
        }
        return canonical_json(payload).encode()

    @_public_guard
    def export_recipe_evidence(self, recipe_id: str) -> bytes:
        record = self.repository.get_recipe(recipe_id)
        payload = {
            "schema_version": "training-recipe-evidence.v1",
            "claim_state": record.state,
            "recipe_id": record.recipe_id,
            "recipe": record.recipe.model_dump(mode="json"),
        }
        return canonical_json(payload).encode()

    def _resolve_recipe(
        self, dataset_id: str, request: RecipeInput, *, source: Artifact
    ) -> TrainingRecipe:
        frozen = self._frozen(dataset_id)
        provenance = self._provenance(
            frozen.dataset_digest, campaign_id=request.stage_3_campaign_id
        )
        base = self.artifacts.get(request.base_model_artifact_id)
        adapters = {
            name: self.artifacts.get(value)
            for name, value in request.lightning_adapter_artifact_ids.items()
        }
        runtime_identity = self.runtime.identity()
        recipe = TrainingRecipe(
            schema_version="character-training-recipe.v1",
            dataset_digest=frozen.dataset_digest,
            character_pack_digest=frozen.character_pack_digest,
            stage_2_manifest_digest=provenance[0],
            stage_2_evidence_digest=provenance[1],
            stage_3_evidence_digest=provenance[2],
            base_model_digest=f"sha256:{base.sha256}",
            lightning_adapter_digests={
                key: f"sha256:{value.sha256}" for key, value in adapters.items()
            },
            source_code_digest=f"sha256:{source.sha256}",
            runtime_identity=runtime_identity,
            backend_id=request.backend_id,
            backend_config=request.backend_config,
            trainable_targets=request.trainable_targets,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            optimizer=request.optimizer,
            learning_rate=request.learning_rate,
            total_steps=request.total_steps,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            source_tier_weights=request.source_tier_weights,
            checkpoint_plan=request.checkpoint_plan,
            calibration_steps=request.calibration_steps,
            wall_clock_limit_seconds=request.wall_clock_limit_seconds,
            seed=request.seed,
        )
        validate_training_recipe_dataset(recipe, frozen)
        return recipe

    def _recipe_binding(
        self, request: RecipeInput, recipe: TrainingRecipe, source: Artifact
    ) -> RecipeMaterialBinding:
        return RecipeMaterialBinding(
            stage_3_campaign_id=request.stage_3_campaign_id,
            stage_3_result_id=request.stage_3_result_id,
            base_model_artifact_id=request.base_model_artifact_id,
            lightning_adapter_artifact_ids=dict(request.lightning_adapter_artifact_ids),
            source_snapshot_artifact_id=source.id,
            runtime_digest=_json_digest(dict(recipe.runtime_identity)),
        )

    def _verify_recipe_materials(self, recipe: TrainingRecipe) -> None:
        self._artifact_for_digest(recipe.base_model_digest)
        for digest in recipe.lightning_adapter_digests.values():
            self._artifact_for_digest(digest)
        if _source_digest(self.source_root) != recipe.source_code_digest:
            raise TrainingServiceError("source_drift")
        if dict(self.runtime.identity()) != dict(recipe.runtime_identity):
            raise TrainingServiceError("runtime_drift")
        self._provenance(
            recipe.dataset_digest,
            campaign_id="",
            expected=recipe,
        )

    def _verify_bound_artifacts(
        self,
        connection: sqlite3.Connection,
        recipe: TrainingRecipe,
        binding: RecipeMaterialBinding,
        stack: ExitStack,
    ) -> tuple[VerifiedArtifactHandle, ...]:
        bindings = (
            (
                binding.base_model_artifact_id,
                recipe.base_model_digest,
                "application/octet-stream",
            ),
            (
                binding.source_snapshot_artifact_id,
                recipe.source_code_digest,
                "application/octet-stream",
            ),
            *tuple(
                (
                    artifact_id,
                    recipe.lightning_adapter_digests[name],
                    "application/octet-stream",
                )
                for name, artifact_id in binding.lightning_adapter_artifact_ids.items()
            ),
        )
        handles: list[VerifiedArtifactHandle] = []
        for artifact_id, digest, media_type in bindings:
            row = connection.execute(
                "SELECT byte_size,media_type FROM artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
            if row is None or str(row["media_type"]) != media_type:
                raise TrainingServiceError("material_missing")
            handles.append(
                stack.enter_context(
                    self.artifacts.open_verified_with_connection(
                        connection,
                        artifact_id,
                        sha256=digest.removeprefix("sha256:"),
                        byte_size=int(row["byte_size"]),
                        media_type=media_type,
                    )
                )
            )
        return tuple(handles)

    def _provenance_with_connection(
        self,
        connection: sqlite3.Connection,
        recipe: TrainingRecipe,
        binding: RecipeMaterialBinding,
    ) -> None:
        row = connection.execute(
            "SELECT result.content_digest,"
            "json_extract(campaign.definition_json,'$.character_pack_digest'),"
            "json_extract(campaign.definition_json,'$.stage_2_manifest_digest'),"
            "json_extract(campaign.definition_json,'$.stage_2_evidence_digest'),"
            "dataset.dataset_digest,dataset.character_pack_digest,dataset.character_id "
            "FROM character_benchmark_campaigns campaign "
            "JOIN character_benchmark_results result ON result.campaign_id=campaign.id "
            "JOIN training_datasets dataset ON dataset.character_pack_storage_id="
            "campaign.character_pack_storage_id AND dataset.reviewer_id=campaign.reviewer_id "
            "WHERE campaign.id=? AND campaign.reviewer_id=? "
            "AND dataset.dataset_digest=? AND dataset.state='sealed' "
            "AND character_benchmark_provenance_is_valid(campaign.id)=1 "
            "AND json_extract(result.result_json,'$.baseline_disposition')="
            "'baseline_established' AND (SELECT new_state FROM character_benchmark_events "
            "WHERE campaign_id=campaign.id ORDER BY sequence DESC LIMIT 1)='completed'",
            (binding.stage_3_campaign_id, self.reviewer_id, recipe.dataset_digest),
        ).fetchone()
        if row is None or tuple(str(row[index]) for index in range(6)) != (
            binding.stage_3_result_id,
            recipe.character_pack_digest,
            recipe.stage_2_manifest_digest,
            recipe.stage_2_evidence_digest,
            recipe.dataset_digest,
            recipe.character_pack_digest,
        ):
            raise TrainingServiceError("prerequisite_drift")
        if binding.runtime_digest != _json_digest(dict(recipe.runtime_identity)):
            raise TrainingServiceError("runtime_drift")

    def _artifact_for_digest(self, digest: str) -> Artifact:
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT id FROM artifacts WHERE sha256=?", (digest[7:],)
            ).fetchone()
        if row is None:
            raise TrainingServiceError("material_missing")
        try:
            return self.artifacts.get(str(row[0]))
        except ArtifactError:
            raise TrainingServiceError("material_drift") from None

    def _provenance(
        self,
        dataset_digest: str,
        *,
        campaign_id: str,
        expected: TrainingRecipe | None = None,
    ) -> tuple[str, str, str]:
        dataset = self.repository.get_dataset(dataset_digest)
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT json_extract(b.definition_json,'$.stage_2_manifest_digest'),"
                "json_extract(b.definition_json,'$.stage_2_evidence_digest'),r.content_digest "
                "FROM character_benchmark_campaigns b JOIN character_benchmark_results r "
                "ON r.campaign_id=b.id WHERE b.id=? AND b.character_pack_storage_id=(SELECT "
                "character_pack_storage_id FROM training_datasets WHERE storage_id=?) "
                "AND b.reviewer_id=? AND character_benchmark_provenance_is_valid(b.id)=1 "
                "AND json_extract(r.result_json,'$.baseline_disposition')='baseline_established' "
                "AND (SELECT new_state FROM character_benchmark_events WHERE campaign_id=b.id "
                "ORDER BY sequence DESC LIMIT 1)='completed' LIMIT 1",
                (campaign_id, dataset.storage_id, self.reviewer_id),
            ).fetchone()
        if row is None:
            raise TrainingServiceError("prerequisite_missing")
        values = str(row[0]), str(row[1]), str(row[2])
        if expected is not None and values != (
            expected.stage_2_manifest_digest,
            expected.stage_2_evidence_digest,
            expected.stage_3_evidence_digest,
        ):
            raise TrainingServiceError("prerequisite_drift")
        return values

    def _head(self, handle: str, *, allow_sealed: bool = False) -> DatasetLineageRecord:
        try:
            try:
                head = self.repository.get_dataset_lineage(handle)
            except TrainingNotFound:
                record = self.repository.get_dataset(handle)
                heads = [
                    x
                    for x in self.repository.list_dataset_lineages()
                    if x.current_dataset_id == record.dataset_id
                ]
                if len(heads) != 1:
                    raise TrainingServiceError("stale_dataset") from None
                head = heads[0]
            if handle not in (head.lineage_id, head.current_dataset_id):
                raise TrainingServiceError("stale_dataset")
            if head.state == "sealed" and not allow_sealed:
                raise TrainingServiceError("sealed_dataset")
            return head
        except TrainingServiceError:
            raise
        except (TrainingNotFound, TrainingConflict):
            raise TrainingServiceError("dataset_not_found") from None

    def _frozen(self, handle: str) -> FrozenTrainingDataset:
        head = self._head(handle, allow_sealed=True)
        frozen = self.repository.get_dataset(head.current_dataset_id).frozen_dataset
        if frozen is None:
            raise TrainingServiceError("dataset_not_sealed")
        return frozen

    @staticmethod
    def _recipe_view(record: RecipeRecord) -> TrainingRecipeView:
        return TrainingRecipeView(
            recipe_id=record.recipe_id, state=record.state, recipe=record.recipe
        )

    def _discard(self, prepared: list[tuple[Path, TrainingMedia, StagedArtifact]]) -> None:
        for _, _, staged in prepared:
            try:
                self.artifacts.discard(staged)
            except ArtifactError:
                pass


def _imported(
    *,
    reviewer_id: str,
    character_id: str,
    lineage_id: str,
    artifact: Artifact,
    media: TrainingMedia,
    source_lineage: str,
    suggestions: TrainingCurationSuggestions | None,
) -> ImportedTrainingSample:
    provisional = ImportedTrainingSample.model_construct(
        import_id="pending",
        reviewer_id=reviewer_id,
        character_id=character_id,
        lineage_id=lineage_id,
        artifact_id=artifact.id,
        artifact_digest=media.sha256,
        artifact_byte_size=media.byte_size,
        media=media,
        source_lineage=source_lineage,
        suggestions=suggestions,
        imported_at=datetime.now(UTC),
    )
    return ImportedTrainingSample.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "import_id": imported_training_sample_id(provisional),
        }
    )


def _confirmation(
    reviewer_id: str, import_id: str, decision: CurationInput
) -> TrainingCurationConfirmation:
    provisional = TrainingCurationConfirmation.model_construct(
        confirmation_id="pending",
        import_id=import_id,
        reviewer_id=reviewer_id,
        decision=decision.decision,
        confirmed_at=datetime.now(UTC),
        version=1,
        sample_id=decision.sample_id,
        source_tier=decision.source_tier,
        source_lineage=decision.source_lineage,
        caption=decision.caption,
        split=decision.split,
        annotation=decision.annotation,
    )
    return TrainingCurationConfirmation.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "confirmation_id": training_confirmation_id(provisional),
        }
    )


def _sort_key(path: Path) -> tuple[str, str]:
    value = unicodedata.normalize("NFC", path.as_posix())
    return value.casefold(), value


def _json_digest(value: dict[str, object]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _calibration_replay_is_equivalent(
    operation: TrainingOrchestrationOperation,
    current_payload: Mapping[str, object],
) -> bool:
    stored = dict(operation.request_payload)
    current = dict(current_payload)
    if training_operation_payload_digest(stored) == training_operation_payload_digest(current):
        return True
    if "lora" in stored:
        return False
    legacy = dict(current)
    legacy.pop("lora", None)
    return training_operation_payload_digest(stored) == training_operation_payload_digest(legacy)


def _source_digest(root: Path) -> str:
    return "sha256:" + hashlib.sha256(_source_snapshot_bytes(root)).hexdigest()


def _source_snapshot_bytes(root: Path) -> bytes:
    """Snapshot a bounded source tree through one held root descriptor."""
    root_descriptor = -1
    try:
        root_descriptor = os.open(
            root,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        root_identity = _stat_identity(os.fstat(root_descriptor))
        output = bytearray(b"video-lab-source-snapshot.v1\0")
        _snapshot_source_directory(root_descriptor, "", output, 0)
        current = os.stat(root, follow_symlinks=False)
        if _stat_identity(current) != root_identity or not stat.S_ISDIR(current.st_mode):
            raise TrainingServiceError("source_drift")
        return bytes(output)
    except TrainingServiceError:
        raise
    except OSError:
        raise TrainingServiceError("unsafe_source_root") from None
    finally:
        if root_descriptor >= 0:
            os.close(root_descriptor)


def _snapshot_source_directory(directory: int, prefix: str, output: bytearray, total: int) -> int:
    before = _stat_identity(os.fstat(directory))
    names = sorted(os.listdir(directory), key=lambda name: _source_name_key(prefix, name))
    normalized = [_source_name_key(prefix, name)[1] for name in names]
    if len(set(normalized)) != len(normalized):
        raise TrainingServiceError("source_normalization_collision")
    for name in names:
        entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
        relative = unicodedata.normalize("NFC", f"{prefix}/{name}" if prefix else name)
        if stat.S_ISDIR(entry.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            try:
                total = _snapshot_source_directory(child, relative, output, total)
                if _stat_identity(os.fstat(child)) != _stat_identity(entry):
                    raise TrainingServiceError("source_drift")
            finally:
                os.close(child)
        elif stat.S_ISREG(entry.st_mode):
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            try:
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    total += len(chunk)
                    if total > _SOURCE_LIMIT:
                        raise TrainingServiceError("source_snapshot_too_large")
                    chunks.append(chunk)
                if _stat_identity(os.fstat(descriptor)) != _stat_identity(entry):
                    raise TrainingServiceError("source_drift")
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if _stat_identity(current) != _stat_identity(entry):
                    raise TrainingServiceError("source_drift")
                data = b"".join(chunks)
            finally:
                os.close(descriptor)
            encoded = relative.encode()
            output.extend(len(encoded).to_bytes(8, "big"))
            output.extend(encoded)
            output.extend(len(data).to_bytes(8, "big"))
            output.extend(data)
        else:
            raise TrainingServiceError("unsafe_source_root")
    if (
        _stat_identity(os.fstat(directory)) != before
        or sorted(os.listdir(directory), key=lambda name: _source_name_key(prefix, name)) != names
    ):
        raise TrainingServiceError("source_drift")
    return total


def _source_name_key(prefix: str, name: str) -> tuple[str, str]:
    value = unicodedata.normalize("NFC", f"{prefix}/{name}" if prefix else name)
    return value.casefold(), value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
