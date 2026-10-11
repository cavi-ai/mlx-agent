"""Hardened, stable CLI boundary for local dataset compilation services."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import signal
import sqlite3
import stat
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from video_lab.artifacts import (
    Artifact,
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactStore,
    VerifiedArtifactHandle,
)
from video_lab.curated_models import (
    CuratedDatasetManifestV1,
    SequenceTransformRecipeV1,
    StructuredCaptionV1,
)
from video_lab.curated_source import CuratedSourceLease
from video_lab.curated_splits import assign_identity_aware_splits
from video_lab.curated_validation import (
    CuratedValidationReport,
    FrameFinding,
    ValidationThresholdsV1,
    validate_curated_dataset,
)
from video_lab.database import Database
from video_lab.dataset_compiler import (
    DatasetCompilationError,
    DatasetCompilationService,
    DatasetCompilerBackend,
    TransformReceipt,
    temporal_sample_indices,
)
from video_lab.dataset_receipts import (
    CompilationConflict,
    CompilationOutput,
    DatasetCompilationReceipt,
    DatasetCompilationRequest,
    InputBinding,
)
from video_lab.domain import canonical_json, utc_now
from video_lab.trainers.wan_batch import WAN_BATCH_TENSORS, validated_safetensors_header_prefix


class DatasetExit(IntEnum):
    """Stable process contract (sysexits-compatible where practical)."""

    ACCEPTED = 0
    USAGE = 64
    REJECTED = 65
    NEEDS_REVIEW = 66
    CONFLICT = 67
    IN_PROGRESS = 68
    CANCELLED = 69
    INTEGRITY = 70
    INTERNAL = 71


class DatasetCliError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: DatasetExit) -> None:
        self.code = code
        self.safe_message = message
        self.exit_code = exit_code
        super().__init__(message)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class FrameSourceBinding(_FrozenModel):
    frame_id: str
    artifact_id: str
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    media_type: Literal["image/png"] = "image/png"


class DatasetValidationRequest(_FrozenModel):
    schema_version: Literal["dataset-validation-request.v1"] = "dataset-validation-request.v1"
    operation_key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    manifest: CuratedDatasetManifestV1
    frame_bindings: tuple[FrameSourceBinding, ...]
    thresholds: ValidationThresholdsV1 = ValidationThresholdsV1()
    human_dispositions: dict[str, Literal["accepted", "rejected"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bindings(self) -> DatasetValidationRequest:
        identities = tuple(binding.frame_id for binding in self.frame_bindings)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError("frame bindings must be unique and canonically ordered")
        return self


class InspectRequest(_FrozenModel):
    schema_version: Literal["dataset-inspect-request.v1"] = "dataset-inspect-request.v1"
    receipt_id: str


class ReproduceRequest(_FrozenModel):
    schema_version: Literal["dataset-reproduce-request.v1"] = "dataset-reproduce-request.v1"
    receipt_id: str
    operation_key: str


class _WanRecipeV1(_FrozenModel):
    schema_version: Literal["wan-curated-compiler-recipe.v1"]
    transform_recipe: SequenceTransformRecipeV1
    timestep: int = Field(ge=0)
    timestep_boundary: int = Field(gt=0)
    latent_channels: int = Field(gt=0)
    vae_scaling_factor: float = Field(gt=0)


class DatasetResultEnvelope(_FrozenModel):
    schema_version: Literal["dataset-cli-result.v1"] = "dataset-cli-result.v1"
    ok: bool
    data: object | None = None
    error: dict[str, str] | None = None


class InspectionArtifact(_FrozenModel):
    role: str
    artifact_id: str
    sha256: str
    byte_size: int
    media_type: str
    document: dict[str, object] | None = None


class _WanSourceV1(_FrozenModel):
    schema_version: Literal["wan-curated-source.v1"]
    shape: tuple[int, int, int, int]
    dtype: Literal["uint8"]
    frames_base64: str
    frame_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_receipt: TransformReceipt
    transform_receipt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    caption: StructuredCaptionV1
    caption_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_evidence: dict[str, str]
    selected_sequence_ids: tuple[str, ...]
    selected_frame_source_digests: tuple[tuple[str, str], ...]


class _WanTensorV1(_FrozenModel):
    name: str
    dtype: str
    rank: int = Field(gt=0)
    shape: tuple[int, ...]
    byte_offsets: tuple[int, int]
    digest: str
    finite: Literal[True]


class _WanObjectiveV1(_FrozenModel):
    target_kind: Literal["flow_noise_minus_clean"]
    formula: Literal["target=noise_latents-clean_latents"]
    clean_latents_digest: str
    noise_latents_digest: str
    target_digest: str


class _WanScheduleV1(_FrozenModel):
    timestep_min: int
    timestep_max: int
    timestep_boundary: int
    selector_rule: Literal["high_if_timestep_gte_boundary_else_low"]


class _WanManifestV1(_FrozenModel):
    schema_version: Literal["wan-precomputed-training-batch.v1"]
    sha256: str
    batch_size: int = Field(gt=0)
    tensors: tuple[_WanTensorV1, ...]
    objective: _WanObjectiveV1
    noise_schedule: _WanScheduleV1

    @model_validator(mode="after")
    def validate_inventory(self) -> _WanManifestV1:
        names = tuple(tensor.name for tensor in self.tensors)
        if names != tuple(sorted(WAN_BATCH_TENSORS)) or len(names) != len(set(names)):
            raise ValueError("tensor inventory must be exact and canonical")
        return self


class _WanProvenanceV1(_FrozenModel):
    schema_version: Literal["wan-curated-batch-provenance.v1"]
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_document_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_receipt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    caption_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    rendered_caption: str
    token_ids: tuple[tuple[int, ...], ...]
    recipe_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_digests: dict[str, dict[str, str]]
    seed: int
    batch_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetValidationReceipt(_FrozenModel):
    schema_version: Literal["dataset-validation-receipt.v1"] = "dataset-validation-receipt.v1"
    receipt_id: str
    operation_key: str
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal["accepted", "rejected", "needs_review"]
    report_artifact: InspectionArtifact
    report: CuratedValidationReport
    created_at: str


class DatasetInspectionView(_FrozenModel):
    schema_version: Literal["dataset-inspection-view.v1"] = "dataset-inspection-view.v1"
    receipt: DatasetCompilationReceipt
    source: InspectionArtifact
    recipe: InspectionArtifact
    outputs: tuple[InspectionArtifact, ...]
    transform_receipt: TransformReceipt
    tensor_inventory: tuple[_WanTensorV1, ...]
    provenance: _WanProvenanceV1
    reproduction: dict[str, str | None]


class CompilationService(Protocol):
    def validate(self, request: DatasetCompilationRequest) -> str: ...
    def compile(self, request: DatasetCompilationRequest) -> DatasetCompilationReceipt: ...
    def inspect(self, receipt_id: str) -> DatasetCompilationReceipt: ...
    def reproduce(self, receipt_id: str, *, operation_key: str) -> DatasetCompilationReceipt: ...


class DatasetApplicationService:
    def __init__(
        self,
        compilation: DatasetCompilationService | None = None,
        *,
        database: Database | None = None,
        artifacts: ArtifactStore | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        self.compilation = compilation
        self.database = database
        self.artifacts = artifacts
        self.cancelled = cancelled

    def validate(
        self, request: DatasetValidationRequest, *, source_root: Path
    ) -> DatasetValidationReceipt:
        if self.database is None or self.artifacts is None:
            raise DatasetCompilationError("validation persistence is unavailable")
        request_json = canonical_json(request.model_dump(mode="json"))
        request_digest = hashlib.sha256(request_json.encode()).hexdigest()
        replay = self._validation_replay(request.operation_key, request_digest)
        if replay is not None:
            return replay
        if self.cancelled():
            raise DatasetCompilationError("validation cancelled before source decoding")
        frames = {
            frame.frame_id: frame
            for sequence in request.manifest.sequences
            for frame in sequence.frames
        }
        bindings = {binding.frame_id: binding for binding in request.frame_bindings}
        if set(bindings) != set(frames):
            raise DatasetCompilationError("frame bindings do not match manifest")
        artifacts: list[Artifact] = []
        for frame_id in sorted(frames):
            frame, binding = frames[frame_id], bindings[frame_id]
            if binding.relative_path != frame.source_path or binding.sha256 != frame.source_digest:
                raise DatasetCompilationError("frame binding conflicts with manifest")
            artifacts.append(
                Artifact(
                    id=binding.artifact_id,
                    job_id=None,
                    sha256=binding.sha256,
                    byte_size=binding.byte_size,
                    media_type=binding.media_type,
                    relative_path=binding.relative_path,
                    created_at="1970-01-01T00:00:00+00:00",
                )
            )
        lease = CuratedSourceLease.open(root=source_root, expected=tuple(artifacts))
        try:
            decoded = {}
            for frame_id in sorted(frames):
                if self.cancelled():
                    raise DatasetCompilationError("validation cancelled at frame boundary")
                decoded[frame_id] = lease.read_frame(frames[frame_id])
            lease.reverify()
        finally:
            lease.close()
        if self.cancelled():
            raise DatasetCompilationError("validation cancelled before metric work")
        try:
            report = validate_curated_dataset(
                request.manifest,
                decoded,
                request.thresholds,
                human_dispositions=request.human_dispositions,
                cancelled=self.cancelled,
            )
        except ValueError as error:
            if "cancelled" in str(error):
                raise DatasetCompilationError("validation cancelled during metric work") from error
            raise DatasetCompilationError("validation evidence was rejected") from error
        if report.split_assignment_evidence is None:
            raise DatasetCompilationError("complete split evidence is required")
        try:
            assign_identity_aware_splits(
                request.manifest,
                request.manifest.split_policy,
                decoded_frames=decoded,
                evidence=report.split_assignment_evidence,
                enforce_declared=True,
                cancelled=self.cancelled,
            )
        except ValueError as error:
            raise DatasetCompilationError("declared dataset splits were rejected") from error
        if self.cancelled():
            raise DatasetCompilationError("validation cancelled before publication")
        return self._publish_validation(request, request_json, request_digest, report)

    def _validation_replay(
        self, operation_key: str, request_digest: str
    ) -> DatasetValidationReceipt | None:
        assert self.database is not None and self.artifacts is not None
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT operation_key,request_digest,state,report_artifact_id,receipt_id,"
                "receipt_digest,receipt_json FROM curated_validation_operations "
                "WHERE operation_key=?",
                (operation_key,),
            ).fetchone()
        if row is None:
            return None
        if str(row["request_digest"]) != request_digest:
            raise CompilationConflict("validation operation key conflict")
        receipt = DatasetValidationReceipt.model_validate_json(
            str(row["receipt_json"]), strict=True
        )
        receipt_json = canonical_json(receipt.model_dump(mode="json"))
        if (
            receipt.operation_key != operation_key
            or receipt.request_digest != request_digest
            or receipt.receipt_id != str(row["receipt_id"])
            or receipt.state != str(row["state"])
            or receipt.report_artifact.artifact_id != str(row["report_artifact_id"])
            or hashlib.sha256(receipt_json.encode()).hexdigest()
            != str(row["receipt_digest"])
            or receipt.receipt_id != "dvr_" + request_digest[:32]
        ):
            raise DatasetCompilationError("validation receipt authority mismatch")
        binding = receipt.report_artifact
        with self.database.connect() as connection:
            with self.artifacts.open_verified_with_connection(
                connection,
                binding.artifact_id,
                sha256=binding.sha256,
                byte_size=binding.byte_size,
                media_type=binding.media_type,
                cancelled=self.cancelled,
            ) as handle:
                raw = _read_exact_bounded(handle, binding.byte_size, _MAX_INSPECTION_DOCUMENT_BYTES)
                self.artifacts.reverify_open_handle(
                    connection, handle, cancelled=self.cancelled
                )
        if raw != canonical_json(receipt.report.model_dump(mode="json")).encode():
            raise DatasetCompilationError("validation report artifact mismatch")
        return receipt

    def _publish_validation(
        self,
        request: DatasetValidationRequest,
        request_json: str,
        request_digest: str,
        report: CuratedValidationReport,
    ) -> DatasetValidationReceipt:
        assert self.database is not None and self.artifacts is not None
        report_bytes = canonical_json(report.model_dump(mode="json")).encode()
        staged = self.artifacts.stage_bytes(
            report_bytes, "application/vnd.video-lab.curated-validation+json"
        )
        now = utc_now().isoformat()

        def persist(
            connection: sqlite3.Connection, committed: tuple[Artifact, ...]
        ) -> DatasetValidationReceipt:
            artifact = committed[0]
            binding = InspectionArtifact(
                role="validation_report",
                artifact_id=artifact.id,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
            )
            receipt = DatasetValidationReceipt(
                receipt_id="dvr_" + request_digest[:32],
                operation_key=request.operation_key,
                request_digest=request_digest,
                state=report.state,
                report_artifact=binding,
                report=report,
                created_at=now,
            )
            receipt_json = canonical_json(receipt.model_dump(mode="json"))
            connection.execute(
                "INSERT INTO curated_validation_operations(operation_key,request_digest,"
                "request_json,state,report_artifact_id,receipt_id,receipt_digest,receipt_json,"
                "created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    request.operation_key,
                    request_digest,
                    request_json,
                    report.state,
                    artifact.id,
                    receipt.receipt_id,
                    hashlib.sha256(receipt_json.encode()).hexdigest(),
                    receipt_json,
                    now,
                ),
            )
            return receipt

        try:
            return self.artifacts.commit_batch((staged,), persist)
        except sqlite3.IntegrityError:
            self.artifacts.discard(staged)
            replay = self._validation_replay(request.operation_key, request_digest)
            if replay is None:
                raise
            return replay

    def inspect(self, receipt_id: str) -> DatasetInspectionView:
        try:
            return self._inspect_authoritative(receipt_id)
        except ArtifactError as error:
            if self.cancelled():
                raise DatasetCompilationError(
                    "inspection cancelled during integrity verification"
                ) from error
            raise

    def compile(self, request: DatasetCompilationRequest) -> DatasetCompilationReceipt:
        if self.compilation is None or self.database is None or self.artifacts is None:
            raise DatasetCompilationError("compilation service is unavailable")
        with self.database.connect() as connection:
            with self.artifacts.open_verified_with_connection(
                connection,
                request.source.artifact_id,
                sha256=request.source.sha256,
                byte_size=request.source.byte_size,
                media_type=request.source.media_type,
                cancelled=self.cancelled,
            ) as handle:
                raw = _read_exact_bounded(
                    handle, request.source.byte_size, _MAX_INSPECTION_DOCUMENT_BYTES
                )
                self.artifacts.reverify_open_handle(
                    connection, handle, cancelled=self.cancelled
                )
        source = _WanSourceV1.model_validate_json(raw, strict=True)
        self._verify_validation_evidence(source)
        try:
            return self.compilation.compile(request)
        except ValueError as error:
            raise DatasetCompilationError("compiler authority was rejected") from error

    def reproduce(
        self, receipt_id: str, *, operation_key: str
    ) -> DatasetCompilationReceipt:
        if self.compilation is None:
            raise DatasetCompilationError("reproduction service is unavailable")
        self.inspect(receipt_id)
        return self.compilation.reproduce(receipt_id, operation_key=operation_key)

    def _verify_validation_evidence(self, source: _WanSourceV1) -> None:
        assert self.database is not None
        evidence = source.validation_evidence
        if set(evidence) != {
            "complete_manifest_digest",
            "dispositions_digest",
            "report_digest",
            "request_digest",
            "split_evidence_digest",
            "validation_receipt_id",
        }:
            raise DatasetCompilationError("validation evidence inventory was rejected")
        with self.database.connect() as connection:
            validation_row = connection.execute(
                "SELECT request_json,receipt_json FROM curated_validation_operations "
                "WHERE receipt_id=?",
                (evidence["validation_receipt_id"],),
            ).fetchone()
        if validation_row is None:
            raise DatasetCompilationError("validation receipt was unavailable")
        validation_receipt = DatasetValidationReceipt.model_validate_json(
            str(validation_row["receipt_json"]), strict=True
        )
        validation_request = DatasetValidationRequest.model_validate_json(
            str(validation_row["request_json"]), strict=True
        )
        durable_validation = self._validation_replay(
            validation_receipt.operation_key, evidence["request_digest"]
        )
        split_evidence = validation_receipt.report.split_assignment_evidence
        if (
            durable_validation != validation_receipt
            or split_evidence is None
            or evidence["complete_manifest_digest"] != source.source_digest
            or evidence["complete_manifest_digest"] != validation_receipt.report.dataset_digest
            or evidence["report_digest"] != validation_receipt.report_artifact.sha256
            or evidence["request_digest"] != validation_receipt.request_digest
            or evidence["split_evidence_digest"]
            != hashlib.sha256(canonical_json(split_evidence).encode()).hexdigest()
            or evidence["dispositions_digest"]
            != hashlib.sha256(
                canonical_json(validation_request.human_dispositions).encode()
            ).hexdigest()
        ):
            raise DatasetCompilationError("validation authority was rejected")
        sequence_ids = source.selected_sequence_ids
        if len(sequence_ids) != 1:
            raise DatasetCompilationError("validation_selection_multiple_sequences_unsupported")
        if sequence_ids != tuple(sorted(sequence_ids)):
            raise DatasetCompilationError("validation_selection_inventory_invalid")
        sequences = {
            sequence.sequence_id: sequence for sequence in validation_request.manifest.sequences
        }
        if any(sequence_id not in sequences for sequence_id in sequence_ids):
            raise DatasetCompilationError("validation_selection_sequence_unknown")
        selected = tuple(sequences[sequence_id] for sequence_id in sequence_ids)
        if any(sequence.split != "train" for sequence in selected):
            raise DatasetCompilationError("validation_selection_split_invalid")
        captions = tuple(sequence.caption for sequence in selected)
        if any(caption != source.caption for caption in captions):
            raise DatasetCompilationError("validation_selection_caption_mismatch")
        recipe = validation_request.manifest.transform_recipe
        if (
            source.transform_receipt.recipe_id != recipe.recipe_id
            or source.transform_receipt.seed != recipe.seed
        ):
            raise DatasetCompilationError("validation_selection_recipe_mismatch")
        sequence = selected[0]
        sampled = temporal_sample_indices(len(sequence.frames), recipe)
        expected_inputs = tuple(
            (sequence.frames[index].frame_id, sequence.frames[index].pixel_digest)
            for index in sampled
        )
        actual_inputs = tuple(
            (item.frame_id, item.input_pixel_digest) for item in source.transform_receipt.inputs
        )
        if actual_inputs != expected_inputs:
            raise DatasetCompilationError("validation_selection_frames_mismatch")
        expected_sources = tuple(
            (sequence.frames[index].frame_id, sequence.frames[index].source_digest)
            for index in sampled
        )
        if source.selected_frame_source_digests != expected_sources:
            raise DatasetCompilationError("validation_selection_source_digests_mismatch")
        selected_frames = {frame_id for frame_id, _ in expected_inputs}
        findings = tuple(validation_receipt.report.frame_findings) + tuple(
            validation_receipt.report.sequence_findings
        )
        for finding in findings:
            applies = (
                finding.sequence_id in sequence_ids
                or (
                    not isinstance(finding, FrameFinding)
                    and bool(set(finding.related_sequence_ids) & set(sequence_ids))
                )
                or (
                    not isinstance(finding, FrameFinding)
                    and any(
                        item.frame_id in selected_frames for item in finding.implicated_frames
                    )
                )
                or (isinstance(finding, FrameFinding) and finding.frame_id in selected_frames)
            )
            if (
                applies
                and finding.state in {"rejected", "needs_review"}
                and finding.human_disposition != "accepted"
            ):
                raise DatasetCompilationError("validation_selection_finding_unaccepted")
            if finding.human_disposition is not None and (
                validation_request.human_dispositions.get(finding.finding_id)
                != finding.human_disposition
            ):
                raise DatasetCompilationError("validation_selection_disposition_stale")

    def _inspect_authoritative(self, receipt_id: str) -> DatasetInspectionView:
        if self.compilation is None or self.database is None or self.artifacts is None:
            raise DatasetCompilationError("inspection service is unavailable")
        receipt = self.compilation.inspect(receipt_id)
        if receipt.compiler_id != "wan_curated_v1":
            raise DatasetCompilationError("unsupported inspection compiler")
        if tuple(output.role for output in receipt.outputs) != ("batch", "manifest", "provenance"):
            raise DatasetCompilationError("inspection output roles were rejected")
        bindings: tuple[tuple[str, InputBinding | CompilationOutput], ...] = (
            ("source_manifest", receipt.source),
            ("transform_recipe", receipt.recipe),
            *((output.role, output) for output in receipt.outputs),
        )
        if (
            len(bindings) > _MAX_INSPECTION_ARTIFACTS
            or sum(binding.byte_size for _, binding in bindings) > _MAX_INSPECTION_TOTAL_BYTES
        ):
            raise DatasetCompilationError("inspection aggregate budget exceeded")
        inspected: list[InspectionArtifact] = []
        document_bytes: dict[str, bytes] = {}
        with self.database.connect() as connection:
            for role, binding in bindings:
                if self.cancelled():
                    raise DatasetCompilationError("inspection cancelled at artifact boundary")
                expected_media = {
                    "source_manifest": "application/json",
                    "transform_recipe": "application/json",
                    "batch": "application/vnd.safetensors",
                    "manifest": "application/json",
                    "provenance": "application/json",
                }[role]
                if binding.media_type != expected_media:
                    raise DatasetCompilationError("inspection artifact media type was rejected")
                with self.artifacts.open_verified_with_connection(
                    connection,
                    binding.artifact_id,
                    sha256=binding.sha256,
                    byte_size=binding.byte_size,
                    media_type=binding.media_type,
                    cancelled=self.cancelled,
                ) as handle:
                    if role == "batch":
                        batch_prefix = _read_safetensors_prefix(handle, binding.byte_size)
                        raw = b""
                    else:
                        raw = _read_exact_bounded(
                            handle, binding.byte_size, _MAX_INSPECTION_DOCUMENT_BYTES
                        )
                        document_bytes[role] = raw
                    self.artifacts.reverify_open_handle(
                        connection, handle, cancelled=self.cancelled
                    )
                document = _bounded_canonical_document(raw, binding.media_type) if raw else None
                inspected.append(
                    InspectionArtifact(
                        role=role,
                        artifact_id=binding.artifact_id,
                        sha256=binding.sha256,
                        byte_size=binding.byte_size,
                        media_type=binding.media_type,
                        document=document,
                    )
                )
        if self.cancelled():
            raise DatasetCompilationError("inspection cancelled before result publication")
        try:
            source_document = cast(dict[str, Any], inspected[0].document)
            recipe_document = cast(dict[str, Any], inspected[1].document)
            manifest_document = cast(dict[str, Any], inspected[3].document)
            provenance_document = cast(dict[str, Any], inspected[4].document)
            source = _WanSourceV1.model_validate_json(
                canonical_json(source_document), strict=True
            )
            self._verify_validation_evidence(source)
            _WanRecipeV1.model_validate_json(
                canonical_json(recipe_document), strict=True
            )
            manifest = _WanManifestV1.model_validate_json(
                canonical_json(manifest_document), strict=True
            )
            provenance = _WanProvenanceV1.model_validate_json(
                canonical_json(provenance_document), strict=True
            )
            header = validated_safetensors_header_prefix(
                batch_prefix,
                total_size=inspected[2].byte_size,
                expected_names=WAN_BATCH_TENSORS,
                batch_size=manifest.batch_size,
            )
            if set(header) != {tensor.name for tensor in manifest.tensors}:
                raise ValueError
            manifest_header = {
                tensor.name: {
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "data_offsets": list(tensor.byte_offsets),
                }
                for tensor in manifest.tensors
            }
            if header != manifest_header:
                raise ValueError
            batch_binding = receipt.outputs[0]
            manifest_binding = receipt.outputs[1]
            if (
                manifest.sha256 != "sha256:" + batch_binding.sha256
                or provenance.batch_digest != batch_binding.sha256
                or provenance.manifest_digest != manifest_binding.sha256
                or provenance.source_document_digest != receipt.source.sha256
                or provenance.recipe_digest != receipt.recipe.sha256
                or provenance.source_digest != source.source_digest
                or provenance.frame_digest != source.frame_digest
                or provenance.transform_receipt_digest != source.transform_receipt_digest
                or provenance.caption_digest != source.caption_digest
                or source.transform_receipt_digest
                != hashlib.sha256(
                    canonical_json(source.transform_receipt.model_dump(mode="json")).encode()
                ).hexdigest()
                or source.caption_digest
                != hashlib.sha256(
                    canonical_json(source.caption.model_dump(mode="json")).encode()
                ).hexdigest()
                or hashlib.sha256(document_bytes["manifest"]).hexdigest()
                != manifest_binding.sha256
                or hashlib.sha256(document_bytes["provenance"]).hexdigest()
                != receipt.outputs[2].sha256
                or hashlib.sha256(document_bytes["source_manifest"]).hexdigest()
                != receipt.source.sha256
                or hashlib.sha256(document_bytes["transform_recipe"]).hexdigest()
                != receipt.recipe.sha256
                or provenance.seed != source.transform_receipt.seed
            ):
                raise ValueError
            with self.database.connect() as connection:
                with self.artifacts.open_verified_with_connection(
                    connection,
                    batch_binding.artifact_id,
                    sha256=batch_binding.sha256,
                    byte_size=batch_binding.byte_size,
                    media_type=batch_binding.media_type,
                    cancelled=self.cancelled,
                ) as handle:
                    tensor_digests = _stream_tensor_digests(
                        handle, manifest, cancelled=self.cancelled
                    )
                    if self.cancelled():
                        raise DatasetCompilationError(
                            "inspection cancelled before post-stream verification"
                        )
                    self.artifacts.reverify_open_handle(
                        connection, handle, cancelled=self.cancelled
                    )
            if tensor_digests != {tensor.name: tensor.digest for tensor in manifest.tensors}:
                raise ValueError
        except (ValidationError, ValueError, TypeError) as error:
            raise DatasetCompilationError("inspection evidence was rejected") from error
        return DatasetInspectionView(
            receipt=receipt,
            source=inspected[0],
            recipe=inspected[1],
            outputs=tuple(inspected[2:]),
            transform_receipt=source.transform_receipt,
            tensor_inventory=manifest.tensors,
            provenance=provenance,
            reproduction={
                "receipt_id": receipt.receipt_id,
                "reproduces_receipt_id": receipt.reproduces_receipt_id,
                "request_digest": receipt.request_digest,
            },
        )


_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_JSON_DEPTH = 32
_MAX_INSPECTION_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_INSPECTION_ARTIFACTS = 32
_MAX_INSPECTION_TOTAL_BYTES = 16 * 1024 * 1024 * 1024
_MAX_REQUEST_PATH_COMPONENTS = 32
_MAX_RETAINED_REQUEST_FDS = 33


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _depth(value: object, level: int = 0) -> int:
    if level > _MAX_JSON_DEPTH:
        raise ValueError("JSON nesting limit exceeded")
    if isinstance(value, dict):
        for item in value.values():
            _depth(item, level + 1)
    elif isinstance(value, list):
        for item in value:
            _depth(item, level + 1)
    return level


def _bounded_canonical_document(raw: bytes, media_type: str) -> dict[str, object] | None:
    if "json" not in media_type:
        return None
    if len(raw) > _MAX_INSPECTION_DOCUMENT_BYTES:
        raise DatasetCompilationError("inspection document budget exceeded")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
        _depth(document)
        if canonical_json(document).encode() != raw:
            raise ValueError("noncanonical inspection document")
        if not isinstance(document, dict):
            raise ValueError("inspection document must be an object")
        return cast(dict[str, object], document)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetCompilationError("inspection document was rejected") from error


def _read_exact_bounded(handle: VerifiedArtifactHandle, declared_size: int, limit: int) -> bytes:
    if declared_size < 0 or declared_size > limit:
        raise DatasetCompilationError("inspection artifact budget exceeded")
    raw = handle.read(declared_size + 1)
    if len(raw) != declared_size:
        raise DatasetCompilationError("inspection artifact was truncated or oversized")
    return raw


def _read_safetensors_prefix(handle: VerifiedArtifactHandle, declared_size: int) -> bytes:
    if declared_size < 9 or declared_size > _MAX_INSPECTION_TOTAL_BYTES:
        raise DatasetCompilationError("inspection artifact budget exceeded")
    first = handle.read(8)
    if len(first) != 8:
        raise DatasetCompilationError("inspection artifact was truncated")
    header_size = struct.unpack("<Q", first)[0]
    if header_size <= 1 or header_size > _MAX_INSPECTION_DOCUMENT_BYTES:
        raise DatasetCompilationError("inspection header budget exceeded")
    header = handle.read(header_size)
    if len(header) != header_size:
        raise DatasetCompilationError("inspection artifact was truncated")
    return first + header


def _stream_tensor_digests(
    handle: VerifiedArtifactHandle,
    manifest: _WanManifestV1,
    *,
    cancelled: Callable[[], bool],
) -> dict[str, str]:
    first = handle.read(8)
    if len(first) != 8:
        raise DatasetCompilationError("inspection artifact was truncated")
    header_size = struct.unpack("<Q", first)[0]
    header = handle.read(header_size)
    if len(header) != header_size:
        raise DatasetCompilationError("inspection artifact was truncated")
    result: dict[str, str] = {}
    cursor = 0
    for tensor in sorted(manifest.tensors, key=lambda item: item.byte_offsets):
        start, end = tensor.byte_offsets
        if start != cursor:
            raise DatasetCompilationError("inspection tensor inventory was rejected")
        digest = hashlib.sha256()
        remaining = end - start
        while remaining:
            if cancelled():
                raise DatasetCompilationError("inspection cancelled during tensor hashing")
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise DatasetCompilationError("inspection artifact was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        result[tensor.name] = "sha256:" + digest.hexdigest()
        cursor = end
    if 8 + header_size + cursor != handle.artifact.byte_size:
        raise DatasetCompilationError("inspection tensor inventory was rejected")
    return result


def _identity(fd: int) -> tuple[int, int, int, int, int, int, int]:
    info = os.fstat(fd)
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _close_descriptor(descriptor: int, failures: list[str]) -> None:
    try:
        os.close(descriptor)
    except OSError:
        failures.append("descriptor_close_failed")


def _open_request(root: Path, relative: str) -> bytes:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetCliError("usage_error", "request path is invalid", DatasetExit.USAGE)
    if (
        len(path.parts) > _MAX_REQUEST_PATH_COMPONENTS
        or len(path.parts) + 1 > _MAX_RETAINED_REQUEST_FDS
    ):
        raise DatasetCliError(
            "integrity_failure", "request traversal budget exceeded", DatasetExit.INTEGRITY
        )
    descriptors: list[int] = []
    identities: list[tuple[int, int, int, int, int, int, int]] = []
    close_failures: list[str] = []
    try:
        current = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(current)
        identities.append(_identity(current))
        for component in path.parts[:-1]:
            current = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            descriptors.append(current)
            identities.append(_identity(current))
        fd = os.open(
            path.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=current,
        )
        descriptors.append(fd)
        before = _identity(fd)
        identities.append(before)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > _MAX_REQUEST_BYTES
        ):
            raise OSError(errno.EINVAL, "unsafe request")
        chunks: list[bytes] = []
        remaining = _MAX_REQUEST_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > _MAX_REQUEST_BYTES or any(
            _identity(descriptor) != expected
            for descriptor, expected in zip(descriptors, identities, strict=True)
        ):
            raise OSError(errno.EIO, "request changed")
        verify = os.dup(descriptors[0])
        try:
            for index, component in enumerate(path.parts):
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if index < len(path.parts) - 1:
                    flags |= os.O_DIRECTORY
                next_fd = os.open(component, flags, dir_fd=verify)
                _close_descriptor(verify, close_failures)
                verify = next_fd
            if _identity(verify) != before:
                raise OSError(errno.EIO, "request replaced")
        finally:
            _close_descriptor(verify, close_failures)
        return data
    except DatasetCliError:
        raise
    except OSError as error:
        raise DatasetCliError(
            "integrity_failure", "request integrity verification failed", DatasetExit.INTEGRITY
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            _close_descriptor(descriptor, close_failures)
        if close_failures:
            raise DatasetCliError(
                "integrity_failure", "request descriptor cleanup failed", DatasetExit.INTEGRITY
            )


def load_request[T: BaseModel](root: Path, relative: str, model: type[T]) -> T:
    raw = _open_request(root, relative)
    try:
        text = raw.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
        _depth(document)
        if canonical_json(document).encode() != raw:
            raise ValueError("request is not canonical JSON")
        return model.model_validate_json(raw, strict=True)
    except (UnicodeError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise DatasetCliError(
            "rejected", "request document was rejected", DatasetExit.REJECTED
        ) from error


class _BindingBackend:
    def __init__(self, compiler_id: str) -> None:
        self.compiler_id = compiler_id

    def validate_recipe(self, recipe_bytes: bytes) -> None:
        try:
            document = json.loads(
                recipe_bytes.decode("utf-8"),
                object_pairs_hook=_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
            )
            _depth(document)
            if canonical_json(document).encode() != recipe_bytes:
                raise ValueError("recipe is not canonical")
            _WanRecipeV1.model_validate_json(recipe_bytes, strict=True)
        except (UnicodeError, ValueError, ValidationError, json.JSONDecodeError) as error:
            raise DatasetCompilationError("compiler recipe was rejected") from error

    def compile_batch(self, source_bytes: bytes, recipe_bytes: bytes) -> tuple[()]:
        raise DatasetCompilationError("validation backend cannot compile")


def build_dataset_service(
    *,
    database_path: Path,
    artifact_root: Path,
    command: str,
    request: DatasetCompilationRequest | None,
    component_roots: tuple[Path, Path, Path, Path] | None,
    cancelled: Callable[[], bool],
) -> DatasetCompilationService:
    database = Database(database_path)
    database.migrate()
    backends: dict[str, DatasetCompilerBackend] = {}
    if command == "validate" and request is not None:
        backends[request.compiler_id] = _BindingBackend(request.compiler_id)
    elif command in {"compile", "reproduce"}:
        if component_roots is None or any(not root.is_dir() for root in component_roots):
            raise DatasetCliError(
                "rejected", "required local component root is unavailable", DatasetExit.REJECTED
            )
        from video_lab.compilers.wan_curated import WanCompilerSnapshot, WanCuratedBatchCompiler
        from video_lab.trainers.wan_factory import WanPinnedLocalRoot

        roots = tuple(WanPinnedLocalRoot(path=root, revision="local") for root in component_roots)
        backend = WanCuratedBatchCompiler(snapshot=WanCompilerSnapshot(*roots))
        backends[backend.compiler_id] = backend
    return DatasetCompilationService(
        database, ArtifactStore(database, artifact_root), backends, cancelled=cancelled
    )


def build_dataset_application(
    *,
    database_path: Path,
    artifact_root: Path,
    command: str,
    request: DatasetCompilationRequest | None,
    component_roots: tuple[Path, Path, Path, Path] | None,
    cancelled: Callable[[], bool],
) -> DatasetApplicationService:
    if command == "validate":
        database = Database(database_path)
        database.migrate()
        return DatasetApplicationService(
            database=database,
            artifacts=ArtifactStore(database, artifact_root),
            cancelled=cancelled,
        )
    compilation = build_dataset_service(
        database_path=database_path,
        artifact_root=artifact_root,
        command=command,
        request=request,
        component_roots=component_roots,
        cancelled=cancelled,
    )
    return DatasetApplicationService(
        compilation,
        database=compilation.database,
        artifacts=compilation.artifacts,
        cancelled=cancelled,
    )


@contextmanager
def _cancellation() -> Iterator[Callable[[], bool]]:
    requested = False

    def handle(_signum: int, _frame: object) -> None:
        nonlocal requested
        requested = True

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, handle)
    try:
        yield lambda: requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _emit(
    envelope: DatasetResultEnvelope, *, json_output: bool, human: str, exit_code: DatasetExit
) -> None:
    if json_output:
        typer.echo(canonical_json(envelope.model_dump(mode="json")))
    elif envelope.ok:
        typer.echo(human)
    else:
        typer.echo(
            envelope.error["message"] if envelope.error else "dataset command failed", err=True
        )
    if exit_code:
        raise typer.Exit(int(exit_code))


def emit_usage_error(*, json_output: bool, message: str) -> None:
    _emit(
        DatasetResultEnvelope(ok=False, error={"code": "usage_error", "message": message}),
        json_output=json_output,
        human="",
        exit_code=DatasetExit.USAGE,
    )


def _human_result(command: str, result: object, status: object) -> str:
    if not isinstance(result, dict):
        return str(status)
    if command == "validate":
        codes = result.get("reason_codes")
        suffix = "" if not codes else " reasons=" + ",".join(str(code) for code in codes)
        return str(status) + suffix
    if command == "inspect":
        receipt = result.get("receipt")
        receipt_id = (
            receipt.get("receipt_id", "verified") if isinstance(receipt, dict) else "verified"
        )
        outputs = result.get("outputs")
        roles = (
            ",".join(str(item.get("role")) for item in outputs if isinstance(item, dict))
            if isinstance(outputs, list)
            else ""
        )
        reproduction = result.get("reproduction")
        parent = (
            reproduction.get("reproduces_receipt_id") if isinstance(reproduction, dict) else None
        )
        return f"accepted receipt={receipt_id} outputs={roles} reproduces={parent or 'none'}"
    return str(status)


def run_command(
    *,
    command: Literal["validate", "compile", "inspect", "reproduce"],
    database_path: Path,
    artifact_root: Path,
    request_root: Path,
    request_file: str,
    json_output: bool,
    source_root: Path | None = None,
    component_roots: tuple[Path | None, Path | None, Path | None, Path | None] | None = None,
) -> None:
    try:
        if command in {"validate", "inspect"} and component_roots is not None:
            raise DatasetCliError(
                "usage_error",
                "component roots are not accepted by this command",
                DatasetExit.USAGE,
            )
        if component_roots is not None and any(root is None for root in component_roots):
            raise DatasetCliError(
                "usage_error",
                "all component roots are required together",
                DatasetExit.USAGE,
            )
        exact_roots = cast(
            tuple[Path, Path, Path, Path] | None,
            component_roots,
        )
        model: type[BaseModel] = (
            DatasetValidationRequest
            if command == "validate"
            else DatasetCompilationRequest
            if command == "compile"
            else InspectRequest
            if command == "inspect"
            else ReproduceRequest
        )
        parsed = load_request(request_root, request_file, model)
        compilation_request = parsed if isinstance(parsed, DatasetCompilationRequest) else None
        with _cancellation() as cancelled:
            service = build_dataset_application(
                database_path=database_path,
                artifact_root=artifact_root,
                command=command,
                request=compilation_request,
                component_roots=exact_roots,
                cancelled=cancelled,
            )
            if command == "validate":
                assert isinstance(parsed, DatasetValidationRequest)
                if source_root is None:
                    raise DatasetCliError(
                        "usage_error", "source root is required", DatasetExit.USAGE
                    )
                validation_receipt = service.validate(parsed, source_root=source_root)
                report = validation_receipt.report
                result: object = {
                    "status": validation_receipt.state,
                    "validation_receipt": validation_receipt.model_dump(mode="json"),
                    "reason_codes": sorted(
                        {item.reason_code for item in report.frame_findings}
                        | {item.reason_code for item in report.sequence_findings}
                    ),
                }
            elif command == "compile":
                assert compilation_request is not None
                compile_method = getattr(service, "compile", None)
                if compile_method is None:
                    assert service.compilation is not None
                    compile_method = service.compilation.compile
                result = compile_method(compilation_request).model_dump(mode="json")
            elif command == "inspect":
                assert isinstance(parsed, InspectRequest)
                result = service.inspect(parsed.receipt_id).model_dump(mode="json")
            else:
                assert isinstance(parsed, ReproduceRequest)
                reproduce_method = getattr(service, "reproduce", None)
                if reproduce_method is None:
                    assert service.compilation is not None
                    reproduce_method = service.compilation.reproduce
                result = reproduce_method(
                    parsed.receipt_id, operation_key=parsed.operation_key
                ).model_dump(mode="json")
        result_status = result.get("status") if isinstance(result, dict) else "accepted"
        exit_code = (
            DatasetExit.REJECTED
            if result_status == "rejected"
            else DatasetExit.NEEDS_REVIEW
            if result_status == "needs_review"
            else DatasetExit.ACCEPTED
        )
        _emit(
            DatasetResultEnvelope(ok=True, data=result),
            json_output=json_output,
            human=_human_result(command, result, result_status),
            exit_code=exit_code,
        )
    except typer.Exit:
        raise
    except DatasetCliError as error:
        _emit(
            DatasetResultEnvelope(
                ok=False, error={"code": error.code, "message": error.safe_message}
            ),
            json_output=json_output,
            human="",
            exit_code=error.exit_code,
        )
    except CompilationConflict:
        _emit(
            DatasetResultEnvelope(
                ok=False,
                error={"code": "conflict", "message": "request conflicts with durable evidence"},
            ),
            json_output=json_output,
            human="",
            exit_code=DatasetExit.CONFLICT,
        )
    except (ArtifactIntegrityError, ArtifactError, OSError):
        _emit(
            DatasetResultEnvelope(
                ok=False,
                error={
                    "code": "integrity_failure",
                    "message": "local evidence integrity verification failed",
                },
            ),
            json_output=json_output,
            human="",
            exit_code=DatasetExit.INTEGRITY,
        )
    except DatasetCompilationError as error:
        message = str(error).lower()
        code, status = (
            ("cancelled", DatasetExit.CANCELLED)
            if "cancel" in message
            else ("in_progress", DatasetExit.IN_PROGRESS)
            if "progress" in message
            else ("rejected", DatasetExit.REJECTED)
        )
        _emit(
            DatasetResultEnvelope(
                ok=False, error={"code": code, "message": "dataset operation was not accepted"}
            ),
            json_output=json_output,
            human="",
            exit_code=status,
        )
    except Exception:
        _emit(
            DatasetResultEnvelope(
                ok=False, error={"code": "internal_failure", "message": "dataset command failed"}
            ),
            json_output=json_output,
            human="",
            exit_code=DatasetExit.INTERNAL,
        )
