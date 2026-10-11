from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from lab_tests.test_character_benchmark_repository import NOW
from lab_tests.test_training_dataset import _draft, _ready_samples, _recipe
from lab_tests.test_training_models import _confirmation, _imported
from lab_tests.test_validation_service import (
    _complete_validation_jobs,
    panel_input,
)
from lab_tests.test_validation_service import (
    service as validation_service,
)
from video_lab.artifacts import ArtifactStore
from video_lab.database import (
    _MIGRATIONS,
    Database,
    _migration_statements,
    _validation_evidence_digest,
)
from video_lab.domain import canonical_json
from video_lab.training_dataset import freeze_training_dataset
from video_lab.training_models import (
    CurationDecision,
    TrainingCalibrationEvidence,
    TrainingCancellationEvidence,
    TrainingCheckpoint,
    TrainingCheckpointArtifact,
    TrainingCheckpointEvidence,
    TrainingEvaluationEvidence,
    TrainingFailure,
    TrainingMedia,
    TrainingOperationKind,
    TrainingOperationStatus,
    TrainingOrchestrationOperation,
    TrainingProbeEvidence,
    TrainingResumeEvidence,
    TrainingRun,
    TrainingRunEvent,
    TrainingTelemetry,
    checkpoint_steps,
    training_dataset_id,
    training_recipe_id,
)
from video_lab.training_repository import (
    RecipeMaterialBinding,
    TrainingConflict,
    TrainingNotFound,
    TrainingOperationClaim,
    TrainingRepository,
    recipe_material_digest,
)

PACK_MANIFEST = {
    "pack_id": "character_pack_" + "6" * 64,
    "character_id": "character_1",
    "duplicate_hash_algorithm_version": "dhash64-exif-transpose-lanczos.v1",
    "near_duplicate_threshold_inclusive": 5,
    "references": [],
    "duplicate_warnings": [],
    "supersedes_pack_id": None,
}
PACK_DIGEST = "sha256:" + hashlib.sha256(canonical_json(PACK_MANIFEST).encode()).hexdigest()
STAGE3_RESULT = {
    "benchmark_id": "character_benchmark_training",
    "baseline_disposition": "baseline_established",
}
STAGE3_RESULT_DIGEST = (
    "sha256:" + hashlib.sha256(canonical_json(STAGE3_RESULT).encode()).hexdigest()
)
STAGE2_MANIFEST_DIGEST = "sha256:" + "2" * 64
STAGE2_EVIDENCE_DIGEST = "sha256:" + "3" * 64


def _frozen():
    draft = _draft(_ready_samples()).model_copy(
        update={"character_id": "character_1", "character_pack_digest": PACK_DIGEST}
    )
    return freeze_training_dataset(draft)


def test_orchestration_receipt_exact_replay_claim_and_terminal_immutability(
    repository: TrainingRepository,
) -> None:
    run, _, probe, calibration = _prepare_calibrated_scientific_run(repository, "receipt-run")
    repository.atomic(
        lambda connection: repository.bind_idempotency_with_connection(
            connection,
            run_id=run.run_id,
            idempotency_key=f"{run.run_id}-request",
            request_digest="sha256:" + "a" * 64,
            probe_evidence_id=probe.evidence_id,
            calibration_evidence_id=calibration.evidence_id,
        )
    )
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=3,
            previous_state="calibrated",
            new_state="preregistered",
            payload={
                "idempotency_key": f"{run.run_id}-request",
                "request_digest": "sha256:" + "a" * 64,
                "probe_evidence_id": probe.evidence_id,
                "calibration_evidence_id": calibration.evidence_id,
            },
        ),
        expected_version=2,
    )
    request = {
        "request_id": "start-request-1",
        "run_id": run.run_id,
        "recipe_digest": probe.recipe_digest,
        "dataset_digest": probe.dataset_digest,
        "backend_digest": probe.backend_digest,
        "runtime_digest": probe.runtime_digest,
        "material_digest": probe.material_digest,
        "source_digest": probe.source_digest,
        "preregistration_id": f"{run.run_id}-request",
        "preregistration_request_digest": "sha256:" + "a" * 64,
        "checkpoint_steps": list(checkpoint_steps(run.total_steps)),
        "checkpoint_fractions": [0.25, 0.5, 0.75, 1.0],
        "material_artifact_ids": {
            "base_model": "artifact-base",
            "lightning:adapter": "artifact-lightning",
            "source_snapshot": "artifact-source",
        },
    }
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        invalid_requests = (
            {**request, "checkpoint_steps": list(reversed(request["checkpoint_steps"]))},
            {**request, "checkpoint_fractions": [0.5, 0.25, 0.75, 1.0]},
            {
                **request,
                "material_artifact_ids": {
                    "base_model": "artifact-base",
                    "source_snapshot": "artifact-source",
                },
            },
            {
                **request,
                "material_artifact_ids": {
                    **request["material_artifact_ids"],
                    "extra": "artifact-base",
                },
            },
        )
        for index, invalid in enumerate(invalid_requests):
            with pytest.raises(TrainingConflict, match="identity conflict"):
                repository.begin_operation_with_connection(
                    connection,
                    operation_id=f"invalid-start-{index}",
                    run_id=run.run_id,
                    operation_kind=TrainingOperationKind.START,
                    idempotency_key=f"invalid-start-{index}",
                    request_payload=invalid,
                )
        created = repository.begin_operation_with_connection(
            connection,
            operation_id="operation-start-1",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="start-key",
            request_payload=request,
        )
        reopened = repository.begin_operation_with_connection(
            connection,
            operation_id="ignored-on-replay",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="start-key",
            request_payload=request,
        )
        assert reopened.operation_id == created.operation_id
        with pytest.raises(TrainingConflict, match="digest conflict"):
            repository.begin_operation_with_connection(
                connection,
                operation_id="conflict",
                run_id=run.run_id,
                operation_kind=TrainingOperationKind.START,
                idempotency_key="start-key",
                request_payload={**request, "request_id": "different"},
            )
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="start-key",
            claimant_id="worker-1",
            lease_seconds=60,
        )
        assert claim.claimed and claim.operation.attempt == 1
        assert not repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="start-key",
            claimant_id="worker-2",
            lease_seconds=60,
        ).claimed
        assert claim.operation.claim_token is not None
        event_payload: dict[str, object] = {}
        repository.append_event_with_connection(
            connection,
            run.run_id,
            TrainingRunEvent(
                sequence=4,
                previous_state="preregistered",
                new_state="training",
                payload=event_payload,
            ),
            expected_version=3,
        )
        event_id = "4"
        event_digest = str(
            connection.execute(
                "SELECT training_operation_evidence_digest(?,?,?)",
                (created.operation_id, "event", event_id),
            ).fetchone()[0]
        )
        repository.link_operation_evidence_with_connection(
            connection,
            operation_id=created.operation_id,
            attempt=claim.operation.attempt,
            claim_token=claim.operation.claim_token,
            evidence_kind="event",
            evidence_id=event_id,
            evidence_digest=event_digest,
        )
        receipt_payload = {
            "run_id": run.run_id,
            "lifecycle_version": 4,
            "lifecycle_state": "training",
            "event_ids": [event_id],
            "event_digests": [event_digest],
            "telemetry_ids": [],
            "telemetry_digest": "sha256:" + "3" * 64,
            "checkpoint_ids": [],
            "checkpoint_digests": [],
            "artifact_ids": [],
            "artifact_digests": [],
            "inventory_ids": [],
            "inventory_digests": [],
            "evaluation_evidence_id": None,
            "evaluation_evidence_digest": None,
        }
        with pytest.raises(sqlite3.IntegrityError, match="transition mismatch|not derived"):
            repository.complete_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.START,
                idempotency_key="start-key",
                claimant_id="worker-1",
                claim_token=claim.operation.claim_token,
                attempt=claim.operation.attempt,
                receipt_payload=receipt_payload,
            )
        terminal = repository.fail_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="start-key",
            claimant_id="worker-1",
            claim_token=claim.operation.claim_token,
            attempt=claim.operation.attempt,
            failure_digest="sha256:" + "f" * 64,
        )
        assert terminal.status is TrainingOperationStatus.FAILED
        assert repository.get_operation_with_connection(connection, "operation-start-1") == terminal
        with pytest.raises(sqlite3.IntegrityError, match="transition mismatch"):
            connection.execute(
                "UPDATE training_orchestration_operations SET updated_at='forged' "
                "WHERE id='operation-start-1'"
            )
        connection.commit()


def test_actual_probe_insert_and_completion_triggers_enforce_lifecycle(
    repository: TrainingRepository,
) -> None:
    run, probe = _prepare_draft_scientific_run(repository, "actual-operation-lifecycle-run")
    request = {
        "request_id": "actual-operation-lifecycle-request",
        "run_id": run.run_id,
        "recipe_digest": probe.recipe_digest,
        "dataset_digest": probe.dataset_digest,
        "backend_digest": probe.backend_digest,
        "runtime_digest": probe.runtime_digest,
        "material_digest": probe.material_digest,
        "source_digest": probe.source_digest,
    }
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="actual-operation-lifecycle-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="actual-operation-lifecycle-key",
            request_payload=request,
        )
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="actual-operation-lifecycle-key",
            claimant_id="actual-operation-lifecycle-worker",
            lease_seconds=60,
        )
        assert claim.operation.claim_token is not None
        repository.record_probe_with_connection(connection, probe)
        evidence_digest = str(
            connection.execute(
                "SELECT training_operation_evidence_digest(?,?,?)",
                (operation.operation_id, "probe", probe.evidence_id),
            ).fetchone()[0]
        )
        repository.link_operation_evidence_with_connection(
            connection,
            operation_id=operation.operation_id,
            attempt=claim.operation.attempt,
            claim_token=claim.operation.claim_token,
            evidence_kind="probe",
            evidence_id=probe.evidence_id,
            evidence_digest=evidence_digest,
        )
        receipt = {
            "run_id": run.run_id,
            "lifecycle_version": 0,
            "lifecycle_state": "draft",
            "evidence_id": probe.evidence_id,
            "evidence_digest": evidence_digest,
        }
        with pytest.raises(sqlite3.IntegrityError, match="not derived"):
            repository.complete_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.PROBE,
                idempotency_key="actual-operation-lifecycle-key",
                claimant_id="actual-operation-lifecycle-worker",
                claim_token=claim.operation.claim_token,
                attempt=claim.operation.attempt,
                receipt_payload=receipt,
            )
        repository.append_event_with_connection(
            connection,
            run.run_id,
            TrainingRunEvent(sequence=1, previous_state="draft", new_state="probed", payload={}),
            expected_version=0,
        )
        repository.complete_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="actual-operation-lifecycle-key",
            claimant_id="actual-operation-lifecycle-worker",
            claim_token=claim.operation.claim_token,
            attempt=claim.operation.attempt,
            receipt_payload={**receipt, "lifecycle_version": 1, "lifecycle_state": "probed"},
        )
        with pytest.raises(TrainingConflict, match="identity conflict"):
            repository.begin_operation_with_connection(
                connection,
                operation_id="actual-operation-lifecycle-invalid-state",
                run_id=run.run_id,
                operation_kind=TrainingOperationKind.PROBE,
                idempotency_key="actual-operation-lifecycle-invalid-state",
                request_payload={**request, "request_id": "invalid-after-probe"},
            )
        connection.commit()


def test_orchestration_operation_raw_sql_fails_closed_and_expired_claim_reclaims(
    repository: TrainingRepository,
) -> None:
    run, _, probe, calibration = _prepare_calibrated_scientific_run(
        repository, "receipt-recovery-run"
    )
    _start_scientific_training(repository, run, probe, calibration)
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        repository.request_cancellation_with_connection(
            connection,
            request_id="cancel-request-1",
            run_id=run.run_id,
            requested_step=0,
            request_digest="sha256:" + "6" * 64,
        )
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="operation-recovery-1",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.CANCEL,
            idempotency_key="cancel-recovery",
            request_payload={
                "request_id": "cancel-request-operation-1",
                "run_id": run.run_id,
                "recipe_digest": probe.recipe_digest,
                "dataset_digest": probe.dataset_digest,
                "backend_digest": probe.backend_digest,
                "runtime_digest": probe.runtime_digest,
                "material_digest": probe.material_digest,
                "source_digest": probe.source_digest,
                "cancellation_request_id": "cancel-request-1",
                "request_digest": "sha256:" + "6" * 64,
                "requested_step": 0,
            },
        )
        with pytest.raises(sqlite3.IntegrityError, match="mismatch"):
            connection.execute(
                "INSERT INTO training_orchestration_operations "
                "(id,reviewer_id,run_id,recipe_storage_id,operation_kind,idempotency_key,"
                "request_digest,request_payload_json,lifecycle_version,lifecycle_state,status,"
                "attempt,claimant_id,claim_token,lease_expires_at,receipt_digest,"
                "receipt_payload_json,failure_digest,created_at,updated_at) VALUES "
                "('forged','owner',?,(SELECT recipe_storage_id FROM training_runs WHERE id=?),"
                "'probe','forged','sha256:'||printf('%064d',0),'{\"run_id\": \"wrong\"}',"
                "2,'calibrated','pending',0,NULL,NULL,NULL,NULL,NULL,NULL,'now','now')",
                (run.run_id, run.run_id),
            )
        # Simulate a worker dying after its atomic claim; an already-expired lease is legal.
        connection.execute(
            "UPDATE training_orchestration_operations SET status='running',attempt=1,"
            "claimant_id='dead-worker',claim_token='dead-token',"
            "lease_expires_at='2000-01-01T00:00:00+00:00',"
            "updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (operation.operation_id,),
        )
        recovered = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.CANCEL,
            idempotency_key="cancel-recovery",
            claimant_id="dead-worker",
            lease_seconds=60,
        )
        assert recovered.claimed
        assert recovered.operation.attempt == 2
        assert recovered.operation.claimant_id == "dead-worker"
        assert recovered.operation.claim_token not in (None, "dead-token")
        with pytest.raises(TrainingConflict, match="fence conflict"):
            repository.link_operation_evidence_with_connection(
                connection,
                operation_id=operation.operation_id,
                attempt=1,
                claim_token="dead-token",
                evidence_kind="probe",
                evidence_id=probe.evidence_id,
                evidence_digest=probe.capability_digest,
            )
        with pytest.raises(TrainingConflict, match="claim mismatch"):
            repository.fail_operation_with_connection(
                connection,
                operation_kind=TrainingOperationKind.CANCEL,
                idempotency_key="cancel-recovery",
                claimant_id="dead-worker",
                claim_token="dead-token",
                attempt=1,
                failure_digest="sha256:" + "f" * 64,
            )
        connection.rollback()


def test_cancellation_operation_atomically_consumes_request_and_tracks_reclaim(
    repository: TrainingRepository,
) -> None:
    run, _, probe, calibration = _prepare_calibrated_scientific_run(
        repository, "cancel-consumption-run"
    )
    _start_scientific_training(repository, run, probe, calibration)
    request_digest = "sha256:" + "6" * 64
    payload = {
        "request_id": "cancel-operation-request",
        "run_id": run.run_id,
        "recipe_digest": probe.recipe_digest,
        "dataset_digest": probe.dataset_digest,
        "backend_digest": probe.backend_digest,
        "runtime_digest": probe.runtime_digest,
        "material_digest": probe.material_digest,
        "source_digest": probe.source_digest,
        "cancellation_request_id": "cancel-request",
        "request_digest": request_digest,
        "requested_step": 0,
    }
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        repository.request_cancellation_with_connection(
            connection,
            request_id="cancel-request",
            run_id=run.run_id,
            requested_step=0,
            request_digest=request_digest,
        )
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="cancel-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.CANCEL,
            idempotency_key="cancel-key",
            request_payload=payload,
        )
        with pytest.raises(TrainingConflict, match="identity conflict"):
            repository.begin_operation_with_connection(
                connection,
                operation_id="cancel-operation-reuse",
                run_id=run.run_id,
                operation_kind=TrainingOperationKind.CANCEL,
                idempotency_key="cancel-key-reuse",
                request_payload={**payload, "request_id": "other"},
            )
        connection.execute(
            "UPDATE training_orchestration_operations SET status='running',attempt=1,"
            "claimant_id='same-worker',claim_token='first-token',"
            "lease_expires_at='2000-01-01T00:00:00+00:00',"
            "updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (operation.operation_id,),
        )
        first = repository.get_operation_with_connection(connection, operation.operation_id)
        assert first.claim_token == "first-token"
        consumed = connection.execute(
            "SELECT operation_id,attempt,claim_token FROM training_cancellation_operation_claims "
            "WHERE cancellation_request_id='cancel-request'"
        ).fetchone()
        assert tuple(consumed) == (operation.operation_id, 1, first.claim_token)
        reclaimed = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.CANCEL,
            idempotency_key="cancel-key",
            claimant_id="same-worker",
            lease_seconds=60,
        )
        assert reclaimed.claimed and reclaimed.operation.claim_token != first.claim_token
        consumed = connection.execute(
            "SELECT attempt,claim_token FROM training_cancellation_operation_claims "
            "WHERE cancellation_request_id='cancel-request'"
        ).fetchone()
        assert tuple(consumed) == (2, reclaimed.operation.claim_token)
        connection.rollback()


def test_calibration_receipt_binds_exact_request_and_live_claim(
    repository: TrainingRepository,
) -> None:
    run, _, probe = _prepare_probed_scientific_run(repository, "calibration-binding-run")
    request = {
        "request_id": "calibration-request",
        "run_id": run.run_id,
        "recipe_digest": probe.recipe_digest,
        "dataset_digest": probe.dataset_digest,
        "backend_digest": probe.backend_digest,
        "runtime_digest": probe.runtime_digest,
        "material_digest": probe.material_digest,
        "source_digest": probe.source_digest,
        "probe_operation_id": f"{run.run_id}-probe-operation",
        "capability_digest": probe.capability_digest,
        "steps": 10,
        "trainable_targets": ["layers.1"],
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
        "seed": 7,
    }
    calibration = TrainingCalibrationEvidence(
        evidence_id="calibration-bound-evidence",
        run_id=run.run_id,
        probe_evidence_id=probe.evidence_id,
        calibration_digest="sha256:" + "9" * 64,
        recipe_digest=probe.recipe_digest,
        dataset_digest=probe.dataset_digest,
        backend_digest=probe.backend_digest,
        runtime_digest=probe.runtime_digest,
        material_digest=probe.material_digest,
        source_digest=probe.source_digest,
        succeeded=True,
    )
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(TrainingConflict, match="identity conflict"):
            repository.begin_operation_with_connection(
                connection,
                operation_id="legacy-shaped-new-calibration",
                run_id=run.run_id,
                operation_kind=TrainingOperationKind.CALIBRATION,
                idempotency_key="legacy-shaped-new-calibration",
                request_payload={key: value for key, value in request.items() if key != "lora"},
            )
        for field, replacement in (
            ("steps", 11),
            ("trainable_targets", ["layers.2"]),
            ("lora", {"rank": 32, "alpha": 64, "dropout": 0.1}),
            ("seed", 8),
        ):
            with pytest.raises(TrainingConflict, match="identity conflict"):
                repository.begin_operation_with_connection(
                    connection,
                    operation_id=f"bad-calibration-{field}",
                    run_id=run.run_id,
                    operation_kind=TrainingOperationKind.CALIBRATION,
                    idempotency_key=f"bad-calibration-{field}",
                    request_payload={**request, field: replacement},
                )
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="calibration-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.CALIBRATION,
            idempotency_key="calibration-key",
            request_payload=request,
        )
        persisted_request = json.loads(
            str(
                connection.execute(
                    "SELECT request_payload_json FROM training_orchestration_operations WHERE id=?",
                    (operation.operation_id,),
                ).fetchone()[0]
            )
        )
        assert persisted_request["lora"] == request["lora"]
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.CALIBRATION,
            idempotency_key="calibration-key",
            claimant_id="calibration-worker",
            lease_seconds=60,
        )
        assert claim.operation.claim_token is not None
        repository.record_calibration_with_connection(connection, calibration)
        evidence_digest = str(
            connection.execute(
                "SELECT training_operation_evidence_digest(?,?,?)",
                (operation.operation_id, "calibration", calibration.evidence_id),
            ).fetchone()[0]
        )
        with pytest.raises(TrainingConflict, match="fence conflict"):
            repository.link_operation_evidence_with_connection(
                connection,
                operation_id=operation.operation_id,
                attempt=claim.operation.attempt,
                claim_token="stale-token",
                evidence_kind="calibration",
                evidence_id=calibration.evidence_id,
                evidence_digest=evidence_digest,
            )
        repository.link_operation_evidence_with_connection(
            connection,
            operation_id=operation.operation_id,
            attempt=claim.operation.attempt,
            claim_token=claim.operation.claim_token,
            evidence_kind="calibration",
            evidence_id=calibration.evidence_id,
            evidence_digest=evidence_digest,
        )
        repository.append_event_with_connection(
            connection,
            run.run_id,
            TrainingRunEvent(
                sequence=2, previous_state="probed", new_state="calibrated", payload={}
            ),
            expected_version=1,
        )
        completed = repository.complete_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.CALIBRATION,
            idempotency_key="calibration-key",
            claimant_id="calibration-worker",
            claim_token=claim.operation.claim_token,
            attempt=claim.operation.attempt,
            receipt_payload={
                "run_id": run.run_id,
                "lifecycle_version": 2,
                "lifecycle_state": "calibrated",
                "evidence_id": calibration.evidence_id,
                "evidence_digest": evidence_digest,
            },
        )
        assert completed.status is TrainingOperationStatus.COMPLETED
        binding = connection.execute(
            "SELECT operation_id,request_digest,attempt,claim_token,steps,"
            "trainable_targets_json,seed FROM training_calibration_operation_bindings"
        ).fetchone()
        assert tuple(binding) == (
            operation.operation_id,
            operation.request_digest,
            1,
            claim.operation.claim_token,
            10,
            '["layers.1"]',
            7,
        )
        connection.commit()


def test_committed_evidence_survives_crash_reclaim_and_exact_receipt_reopens(
    repository: TrainingRepository,
) -> None:
    run, probe = _prepare_draft_scientific_run(repository, "receipt-crash-run")
    request = {
        "request_id": "crash-probe-request",
        "run_id": run.run_id,
        "recipe_digest": probe.recipe_digest,
        "dataset_digest": probe.dataset_digest,
        "backend_digest": probe.backend_digest,
        "runtime_digest": probe.runtime_digest,
        "material_digest": probe.material_digest,
        "source_digest": probe.source_digest,
    }
    with repository.database.connect() as first:
        first.execute("BEGIN IMMEDIATE")
        operation = repository.begin_operation_with_connection(
            first,
            operation_id="crash-probe-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="crash-probe-key",
            request_payload=request,
        )
        first.execute(
            "UPDATE training_orchestration_operations SET status='running',attempt=1,"
            "claimant_id='same-worker',claim_token='stale-token',"
            "lease_expires_at='2000-01-01T00:00:00+00:00',"
            "updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (operation.operation_id,),
        )
        repository.record_probe_with_connection(first, probe)
        repository.append_event_with_connection(
            first,
            run.run_id,
            TrainingRunEvent(sequence=1, previous_state="draft", new_state="probed", payload={}),
            expected_version=0,
        )
        digest = str(
            first.execute(
                "SELECT training_operation_evidence_digest(?,?,?)",
                (operation.operation_id, "probe", probe.evidence_id),
            ).fetchone()[0]
        )
        repository.link_operation_evidence_with_connection(
            first,
            operation_id=operation.operation_id,
            attempt=1,
            claim_token="stale-token",
            evidence_kind="probe",
            evidence_id=probe.evidence_id,
            evidence_digest=digest,
        )
        first.commit()

    with repository.database.connect() as second:
        second.execute("BEGIN IMMEDIATE")
        reclaimed = repository.claim_operation_with_connection(
            second,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="crash-probe-key",
            claimant_id="same-worker",
            lease_seconds=60,
        )
        assert reclaimed.claimed and reclaimed.operation.attempt == 2
        assert reclaimed.operation.claim_token not in (None, "stale-token")
        with pytest.raises(TrainingConflict, match="claim mismatch"):
            repository.fail_operation_with_connection(
                second,
                operation_kind=TrainingOperationKind.PROBE,
                idempotency_key="crash-probe-key",
                claimant_id="same-worker",
                claim_token="stale-token",
                attempt=1,
                failure_digest="sha256:" + "f" * 64,
            )
        with pytest.raises(TrainingConflict, match="fence conflict"):
            repository.link_operation_evidence_with_connection(
                second,
                operation_id=operation.operation_id,
                attempt=2,
                claim_token=reclaimed.operation.claim_token,
                evidence_kind="probe",
                evidence_id=probe.evidence_id,
                evidence_digest=digest,
            )
        completed = repository.complete_operation_with_connection(
            second,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="crash-probe-key",
            claimant_id="same-worker",
            claim_token=reclaimed.operation.claim_token,
            attempt=2,
            receipt_payload={
                "run_id": run.run_id,
                "lifecycle_version": 1,
                "lifecycle_state": "probed",
                "evidence_id": probe.evidence_id,
                "evidence_digest": digest,
            },
        )
        assert (
            repository.reopen_operation_with_connection(
                second, TrainingOperationKind.PROBE, "crash-probe-key"
            )
            == completed
        )
        second.commit()


def test_two_connection_claim_race_has_exactly_one_backend_claimant(
    repository: TrainingRepository,
) -> None:
    run, probe = _prepare_draft_scientific_run(repository, "receipt-claim-race-run")
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        repository.begin_operation_with_connection(
            connection,
            operation_id="claim-race-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="claim-race-key",
            request_payload={
                "request_id": "claim-race-request",
                "run_id": run.run_id,
                "recipe_digest": probe.recipe_digest,
                "dataset_digest": probe.dataset_digest,
                "backend_digest": probe.backend_digest,
                "runtime_digest": probe.runtime_digest,
                "material_digest": probe.material_digest,
                "source_digest": probe.source_digest,
            },
        )
        connection.commit()

    barrier = threading.Barrier(2)
    results: list[tuple[str, bool, int, str | None]] = []
    errors: list[BaseException] = []

    def claim(worker: str) -> None:
        try:
            barrier.wait()
            with repository.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                outcome = repository.claim_operation_with_connection(
                    connection,
                    operation_kind=TrainingOperationKind.PROBE,
                    idempotency_key="claim-race-key",
                    claimant_id=worker,
                    lease_seconds=60,
                )
                results.append(
                    (
                        worker,
                        outcome.claimed,
                        outcome.operation.attempt,
                        outcome.operation.claim_token,
                    )
                )
                connection.commit()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=claim, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not errors
    assert sum(claimed for _, claimed, _, _ in results) == 1
    assert {attempt for _, _, attempt, _ in results} == {1}
    assert len({token for _, _, _, token in results}) == 1


def test_complete_start_receipt_reopens_multi_checkpoint_execution(
    repository: TrainingRepository,
) -> None:
    prepared = _prepare_calibrated_scientific_run(repository, "complete-start-receipt-run")
    run, recipe_id, probe, calibration = prepared
    _preregister_scientific_training(repository, run, probe, calibration)
    request = {
        "request_id": "complete-start-request",
        "run_id": run.run_id,
        "recipe_digest": probe.recipe_digest,
        "dataset_digest": probe.dataset_digest,
        "backend_digest": probe.backend_digest,
        "runtime_digest": probe.runtime_digest,
        "material_digest": probe.material_digest,
        "source_digest": probe.source_digest,
        "preregistration_id": f"{run.run_id}-request",
        "preregistration_request_digest": "sha256:" + "a" * 64,
        "checkpoint_steps": list(checkpoint_steps(run.total_steps)),
        "checkpoint_fractions": [0.25, 0.5, 0.75, 1.0],
        "material_artifact_ids": {
            "base_model": "artifact-base",
            "lightning:adapter": "artifact-lightning",
            "source_snapshot": "artifact-source",
        },
    }
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="complete-start-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="complete-start-key",
            request_payload=request,
        )
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="complete-start-key",
            claimant_id="start-worker",
            lease_seconds=300,
        )
        assert claim.operation.claim_token is not None
        connection.commit()
    _finish_start_receipt(repository, run, prepared, operation, claim)


def test_complete_resume_receipt_binds_parent_inventory_and_reopens(
    repository: TrainingRepository,
) -> None:
    prepared = _prepare_calibrated_scientific_run(repository, "complete-resume-receipt-run")
    run, _, probe, calibration = prepared
    _start_scientific_training(repository, run, probe, calibration)
    _, _, initial = _publish_scientific_checkpoint_chain(
        repository, run.run_id, count=1, prepared=prepared, bind_artifacts=True
    )
    parent_checkpoint, parent_evidence = initial[0]
    with repository.database.connect() as connection:
        parent_digest = str(
            connection.execute(
                "SELECT evidence_digest FROM training_checkpoints WHERE id=?",
                (parent_checkpoint.checkpoint_id,),
            ).fetchone()[0]
        )
        rows = connection.execute(
            "SELECT role,artifact_id,artifact_digest,size_bytes "
            "FROM training_checkpoint_inventory_artifacts WHERE checkpoint_id=? ORDER BY ordinal",
            (parent_checkpoint.checkpoint_id,),
        ).fetchall()
    parent_inventory = [
        {
            "role": row["role"],
            "artifact_id": row["artifact_id"],
            "digest": row["artifact_digest"],
            "size_bytes": row["size_bytes"],
        }
        for row in rows
    ]
    request = {
        "request_id": "complete-resume-request",
        "run_id": run.run_id,
        "recipe_digest": probe.recipe_digest,
        "dataset_digest": probe.dataset_digest,
        "backend_digest": probe.backend_digest,
        "runtime_digest": probe.runtime_digest,
        "material_digest": probe.material_digest,
        "source_digest": probe.source_digest,
        "parent_checkpoint_id": parent_checkpoint.checkpoint_id,
        "parent_checkpoint_digest": parent_digest,
        "parent_inventory": parent_inventory,
    }
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO artifacts VALUES ('duplicate-byte-substitute',NULL,?,?,"
            "'application/x-substitute','duplicate-byte-substitute',?)",
            (
                parent_inventory[0]["digest"].removeprefix("sha256:"),
                parent_inventory[0]["size_bytes"],
                NOW.isoformat(),
            ),
        )
        replaced_inventory = [
            {**parent_inventory[0], "artifact_id": "duplicate-byte-substitute"},
            *parent_inventory[1:],
        ]
        for suffix, mutation in (
            ("substitute", {"parent_inventory": replaced_inventory}),
            ("stale-parent", {"parent_checkpoint_digest": "sha256:" + "0" * 64}),
        ):
            with pytest.raises(TrainingConflict, match="identity conflict"):
                repository.begin_operation_with_connection(
                    connection,
                    operation_id=f"bad-resume-{suffix}",
                    run_id=run.run_id,
                    operation_kind=TrainingOperationKind.RESUME,
                    idempotency_key=f"bad-resume-{suffix}",
                    request_payload={**request, **mutation},
                )
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="complete-resume-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.RESUME,
            idempotency_key="complete-resume-key",
            request_payload=request,
        )
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.RESUME,
            idempotency_key="complete-resume-key",
            claimant_id="resume-worker",
            lease_seconds=300,
        )
        assert claim.operation.claim_token is not None
        resume = TrainingResumeEvidence(
            resume_id="complete-resume-evidence",
            run_id=run.run_id,
            parent_checkpoint_id=parent_checkpoint.checkpoint_id,
            parent_checkpoint_digest=parent_digest,
            parent_artifact_digest=parent_evidence.safetensors_digest,
            recipe_digest=probe.recipe_digest,
            dataset_digest=probe.dataset_digest,
            backend_digest=probe.backend_digest,
            runtime_digest=probe.runtime_digest,
            material_digest=probe.material_digest,
            source_digest=probe.source_digest,
            optimizer_state_digest=parent_evidence.optimizer_state_digest,
        )
        repository.record_resume_with_connection(connection, resume)
        connection.commit()

    _, _, remaining = _publish_scientific_checkpoint_chain(
        repository,
        run.run_id,
        prepared=prepared,
        bind_artifacts=True,
        start_ordinal=1,
        parent=(parent_checkpoint.checkpoint_id, parent_digest),
    )
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=5,
            previous_state="training",
            new_state="evaluating",
            payload={"telemetry_digest": remaining[-1][1].telemetry_digest},
        ),
        expected_version=4,
    )
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        identifiers: dict[str, list[str]] = {
            "resume": [resume.resume_id],
            "event": ["5"],
            "telemetry": ["2", "3", "4"],
            "checkpoint": [item.checkpoint_id for item, _ in remaining],
        }
        bindings = connection.execute(
            "SELECT binding.checkpoint_id,binding.role,binding.artifact_id "
            "FROM training_checkpoint_inventory_artifacts binding "
            "JOIN training_checkpoints checkpoint ON checkpoint.id=binding.checkpoint_id "
            "WHERE checkpoint.run_id=? AND checkpoint.step>? "
            "ORDER BY binding.checkpoint_id,binding.ordinal",
            (run.run_id, parent_checkpoint.step),
        ).fetchall()
        identifiers["inventory"] = [f"{row['checkpoint_id']}:{row['role']}" for row in bindings]
        identifiers["artifact"] = sorted({str(row["artifact_id"]) for row in bindings})
        receipt_lists: dict[str, list[dict[str, object]]] = {
            "events": [],
            "telemetry": [],
            "checkpoints": [],
            "artifacts": [],
            "inventory": [],
        }
        resume_entry: dict[str, object] | None = None
        for kind, values in identifiers.items():
            for identifier in values:
                digest = str(
                    connection.execute(
                        "SELECT training_operation_evidence_digest(?,?,?)",
                        (operation.operation_id, kind, identifier),
                    ).fetchone()[0]
                )
                assert (
                    connection.execute(
                        "SELECT training_operation_evidence_link_is_valid(?,?,?,?,?,?)",
                        (
                            operation.operation_id,
                            claim.operation.attempt,
                            claim.operation.claim_token,
                            kind,
                            identifier,
                            digest,
                        ),
                    ).fetchone()[0]
                    == 1
                ), (kind, identifier, digest)
                repository.link_operation_evidence_with_connection(
                    connection,
                    operation_id=operation.operation_id,
                    attempt=claim.operation.attempt,
                    claim_token=claim.operation.claim_token,
                    evidence_kind=kind,
                    evidence_id=identifier,
                    evidence_digest=digest,
                )
                base = {"evidence_id": identifier, "digest": digest}
                if kind == "resume":
                    resume_entry = base
                elif kind == "event":
                    receipt_lists["events"].append({**base, "sequence": int(identifier)})
                elif kind == "telemetry":
                    row = connection.execute(
                        "SELECT step FROM training_telemetry WHERE run_id=? AND sequence=?",
                        (run.run_id, int(identifier)),
                    ).fetchone()
                    receipt_lists["telemetry"].append(
                        {**base, "sequence": int(identifier), "step": int(row[0])}
                    )
                elif kind == "checkpoint":
                    step = connection.execute(
                        "SELECT step FROM training_checkpoints WHERE id=?", (identifier,)
                    ).fetchone()[0]
                    receipt_lists["checkpoints"].append({**base, "step": int(step)})
                elif kind == "artifact":
                    size = connection.execute(
                        "SELECT byte_size FROM artifacts WHERE id=?", (identifier,)
                    ).fetchone()[0]
                    receipt_lists["artifacts"].append({**base, "size_bytes": int(size)})
                else:
                    checkpoint_id, role = identifier.rsplit(":", 1)
                    receipt_lists["inventory"].append(
                        {**base, "checkpoint_id": checkpoint_id, "role": role}
                    )
        assert resume_entry is not None
        receipt = {
            "run_id": run.run_id,
            "lifecycle_version": 5,
            "lifecycle_state": "evaluating",
            **receipt_lists,
            "evaluation": None,
            "resume": resume_entry,
        }
        completed = repository.complete_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.RESUME,
            idempotency_key="complete-resume-key",
            claimant_id="resume-worker",
            claim_token=claim.operation.claim_token,
            attempt=claim.operation.attempt,
            receipt_payload=receipt,
        )
        assert (
            repository.reopen_operation_with_connection(
                connection, TrainingOperationKind.RESUME, "complete-resume-key"
            )
            == completed
        )
        assert len(receipt_lists["telemetry"]) == 3
        assert len(receipt_lists["checkpoints"]) == 3
        assert len(receipt_lists["inventory"]) == 21
        connection.commit()


def test_resume_operation_rejects_linked_evidence_for_another_valid_parent(
    repository: TrainingRepository,
) -> None:
    prepared = _prepare_calibrated_scientific_run(repository, "resume-parent-link-run")
    run, _, probe, calibration = prepared
    _start_scientific_training(repository, run, probe, calibration)
    _, _, checkpoints = _publish_scientific_checkpoint_chain(
        repository, run.run_id, count=2, prepared=prepared, bind_artifacts=True
    )
    parent_a, evidence_a = checkpoints[0]
    parent_b, evidence_b = checkpoints[1]
    with repository.database.connect() as connection:
        parent_a_digest = str(
            connection.execute(
                "SELECT evidence_digest FROM training_checkpoints WHERE id=?",
                (parent_a.checkpoint_id,),
            ).fetchone()[0]
        )
        parent_b_digest = str(
            connection.execute(
                "SELECT evidence_digest FROM training_checkpoints WHERE id=?",
                (parent_b.checkpoint_id,),
            ).fetchone()[0]
        )
        inventory = [
            {
                "role": row["role"],
                "artifact_id": row["artifact_id"],
                "digest": row["artifact_digest"],
                "size_bytes": row["size_bytes"],
            }
            for row in connection.execute(
                "SELECT role,artifact_id,artifact_digest,size_bytes "
                "FROM training_checkpoint_inventory_artifacts WHERE checkpoint_id=? "
                "ORDER BY ordinal",
                (parent_a.checkpoint_id,),
            )
        ]
        connection.execute("BEGIN IMMEDIATE")
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="resume-parent-link-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.RESUME,
            idempotency_key="resume-parent-link-key",
            request_payload={
                "request_id": "resume-parent-link-request",
                "run_id": run.run_id,
                "recipe_digest": probe.recipe_digest,
                "dataset_digest": probe.dataset_digest,
                "backend_digest": probe.backend_digest,
                "runtime_digest": probe.runtime_digest,
                "material_digest": probe.material_digest,
                "source_digest": probe.source_digest,
                "parent_checkpoint_id": parent_a.checkpoint_id,
                "parent_checkpoint_digest": parent_a_digest,
                "parent_inventory": inventory,
            },
        )
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.RESUME,
            idempotency_key="resume-parent-link-key",
            claimant_id="resume-parent-link-worker",
            lease_seconds=300,
        )
        assert claim.operation.claim_token is not None
        wrong_parent = TrainingResumeEvidence(
            resume_id="resume-parent-link-wrong-evidence",
            run_id=run.run_id,
            parent_checkpoint_id=parent_b.checkpoint_id,
            parent_checkpoint_digest=parent_b_digest,
            parent_artifact_digest=evidence_b.safetensors_digest,
            recipe_digest=probe.recipe_digest,
            dataset_digest=probe.dataset_digest,
            backend_digest=probe.backend_digest,
            runtime_digest=probe.runtime_digest,
            material_digest=probe.material_digest,
            source_digest=probe.source_digest,
            optimizer_state_digest=evidence_b.optimizer_state_digest,
        )
        repository.record_resume_with_connection(connection, wrong_parent)
        digest = str(
            connection.execute(
                "SELECT training_operation_evidence_digest(?,?,?)",
                (operation.operation_id, "resume", wrong_parent.resume_id),
            ).fetchone()[0]
        )
        with pytest.raises(TrainingConflict, match="fence conflict"):
            repository.link_operation_evidence_with_connection(
                connection,
                operation_id=operation.operation_id,
                attempt=claim.operation.attempt,
                claim_token=claim.operation.claim_token,
                evidence_kind="resume",
                evidence_id=wrong_parent.resume_id,
                evidence_digest=digest,
            )
        assert evidence_a.safetensors_digest != evidence_b.safetensors_digest
        connection.rollback()


def test_cancellation_claim_rows_only_follow_exact_live_operation_reclaim(
    repository: TrainingRepository,
) -> None:
    run, _, probe, calibration = _prepare_calibrated_scientific_run(
        repository, "cancel-claim-update-guard-run"
    )
    _start_scientific_training(repository, run, probe, calibration)
    request_digest = "sha256:" + "6" * 64
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        repository.request_cancellation_with_connection(
            connection,
            request_id="cancel-claim-update-request",
            run_id=run.run_id,
            requested_step=0,
            request_digest=request_digest,
        )
        operation = repository.begin_operation_with_connection(
            connection,
            operation_id="cancel-claim-update-operation",
            run_id=run.run_id,
            operation_kind=TrainingOperationKind.CANCEL,
            idempotency_key="cancel-claim-update-key",
            request_payload={
                "request_id": "cancel-claim-update-operation-request",
                "run_id": run.run_id,
                "recipe_digest": probe.recipe_digest,
                "dataset_digest": probe.dataset_digest,
                "backend_digest": probe.backend_digest,
                "runtime_digest": probe.runtime_digest,
                "material_digest": probe.material_digest,
                "source_digest": probe.source_digest,
                "cancellation_request_id": "cancel-claim-update-request",
                "request_digest": request_digest,
                "requested_step": 0,
            },
        )
        for mutation in (
            "cancellation_request_id='other'",
            "operation_id='other'",
            "request_digest='sha256:'||printf('%064d',0)",
            "requested_step=1",
            "attempt=1",
            "claim_token='forged'",
            "created_at='forged'",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable|reclaim"):
                connection.execute(
                    f"UPDATE training_cancellation_operation_claims SET {mutation} "  # noqa: S608
                    "WHERE operation_id=?",
                    (operation.operation_id,),
                )
        connection.execute(
            "UPDATE training_orchestration_operations SET status='running',attempt=1,"
            "claimant_id='worker',claim_token='first-token',"
            "lease_expires_at='2000-01-01T00:00:00+00:00',"
            "updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (operation.operation_id,),
        )
        assert tuple(
            connection.execute(
                "SELECT attempt,claim_token FROM training_cancellation_operation_claims "
                "WHERE operation_id=?",
                (operation.operation_id,),
            ).fetchone()
        ) == (1, "first-token")
        with pytest.raises(sqlite3.IntegrityError, match="immutable|reclaim"):
            connection.execute(
                "UPDATE training_cancellation_operation_claims SET attempt=2,"
                "claim_token='forged-reclaim' WHERE operation_id=?",
                (operation.operation_id,),
            )
        repository.fail_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.CANCEL,
            idempotency_key="cancel-claim-update-key",
            claimant_id="worker",
            claim_token="first-token",
            attempt=1,
            failure_digest="sha256:" + "f" * 64,
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable|reclaim"):
            connection.execute(
                "UPDATE training_cancellation_operation_claims SET attempt=2,"
                "claim_token='post-terminal' WHERE operation_id=?",
                (operation.operation_id,),
            )
        connection.rollback()


def _finish_start_receipt(
    repository: TrainingRepository,
    run: TrainingRun,
    prepared: tuple[TrainingRun, str, TrainingProbeEvidence, TrainingCalibrationEvidence],
    operation: TrainingOrchestrationOperation,
    claim: TrainingOperationClaim,
) -> None:
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=4, previous_state="preregistered", new_state="training", payload={}
        ),
        expected_version=3,
    )
    _, _, published = _publish_scientific_checkpoint_chain(
        repository,
        run.run_id,
        prepared=prepared,
        bind_artifacts=True,
    )
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=5,
            previous_state="training",
            new_state="evaluating",
            payload={"telemetry_digest": published[-1][1].telemetry_digest},
        ),
        expected_version=4,
    )
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        entries: dict[str, list[dict[str, object]]] = {
            "events": [],
            "telemetry": [],
            "checkpoints": [],
            "artifacts": [],
            "inventory": [],
        }
        evidence_ids = {
            "event": ["4", "5"],
            "telemetry": [str(index) for index in range(1, 5)],
            "checkpoint": [checkpoint.checkpoint_id for checkpoint, _ in published],
        }
        inventory_rows = connection.execute(
            "SELECT binding.checkpoint_id,binding.role,binding.artifact_id,"
            "binding.artifact_digest,binding.size_bytes "
            "FROM training_checkpoint_inventory_artifacts binding "
            "JOIN training_checkpoints checkpoint ON checkpoint.id=binding.checkpoint_id "
            "WHERE checkpoint.run_id=? ORDER BY binding.checkpoint_id,binding.ordinal",
            (run.run_id,),
        ).fetchall()
        evidence_ids["inventory"] = [
            f"{row['checkpoint_id']}:{row['role']}" for row in inventory_rows
        ]
        evidence_ids["artifact"] = sorted({str(row["artifact_id"]) for row in inventory_rows})
        for kind, identifiers in evidence_ids.items():
            for identifier in identifiers:
                digest = str(
                    connection.execute(
                        "SELECT training_operation_evidence_digest(?,?,?)",
                        (operation.operation_id, kind, identifier),
                    ).fetchone()[0]
                )
                repository.link_operation_evidence_with_connection(
                    connection,
                    operation_id=operation.operation_id,
                    attempt=claim.operation.attempt,
                    claim_token=claim.operation.claim_token,
                    evidence_kind=kind,
                    evidence_id=identifier,
                    evidence_digest=digest,
                )
                if kind == "event":
                    entries["events"].append(
                        {"evidence_id": identifier, "digest": digest, "sequence": int(identifier)}
                    )
                elif kind == "telemetry":
                    step = int(
                        connection.execute(
                            "SELECT step FROM training_telemetry WHERE run_id=? AND sequence=?",
                            (run.run_id, int(identifier)),
                        ).fetchone()[0]
                    )
                    entries["telemetry"].append(
                        {
                            "evidence_id": identifier,
                            "digest": digest,
                            "sequence": int(identifier),
                            "step": step,
                        }
                    )
                elif kind == "checkpoint":
                    step = int(
                        connection.execute(
                            "SELECT step FROM training_checkpoints WHERE id=?", (identifier,)
                        ).fetchone()[0]
                    )
                    entries["checkpoints"].append(
                        {"evidence_id": identifier, "digest": digest, "step": step}
                    )
                elif kind == "artifact":
                    size = int(
                        connection.execute(
                            "SELECT byte_size FROM artifacts WHERE id=?", (identifier,)
                        ).fetchone()[0]
                    )
                    entries["artifacts"].append(
                        {"evidence_id": identifier, "digest": digest, "size_bytes": size}
                    )
                else:
                    checkpoint_id, role = identifier.rsplit(":", 1)
                    entries["inventory"].append(
                        {
                            "evidence_id": identifier,
                            "digest": digest,
                            "checkpoint_id": checkpoint_id,
                            "role": role,
                        }
                    )
        receipt = {
            "run_id": run.run_id,
            "lifecycle_version": 5,
            "lifecycle_state": "evaluating",
            **entries,
            "evaluation": None,
        }
        completed = repository.complete_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.START,
            idempotency_key="complete-start-key",
            claimant_id="start-worker",
            claim_token=claim.operation.claim_token,
            attempt=claim.operation.attempt,
            receipt_payload=receipt,
        )
        assert (
            repository.reopen_operation_with_connection(
                connection, TrainingOperationKind.START, "complete-start-key"
            )
            == completed
        )
        assert len(entries["telemetry"]) == 4
        assert len(entries["checkpoints"]) == 4
        assert len(entries["inventory"]) == 28
        assert len(entries["artifacts"]) >= 10
        connection.commit()


def test_populated_physical_v38_upgrades_and_receipt_reopens(
    repository: TrainingRepository,
) -> None:
    run, recipe_id, probe, _ = _prepare_calibrated_scientific_run(repository, "receipt-v38-run")
    with repository.database.connect() as connection:
        connection.execute("DROP TABLE training_calibration_operation_bindings")
        connection.execute("DROP TABLE training_cancellation_operation_claims")
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND "
            "(name LIKE '%v41' OR name LIKE '%v47')"
        ).fetchall():
            connection.execute(f'DROP TRIGGER "{row[0]}"')
        connection.execute(
            next(
                statement
                for statement in _migration_statements(_MIGRATIONS[37])
                if statement.lstrip().startswith(
                    "CREATE TRIGGER validate_training_event_insert_v37"
                )
            )
        )
        connection.execute("DROP TABLE training_resume_attempt_failures")
        connection.execute("DROP TABLE training_checkpoint_inventory_artifacts")
        connection.execute("DROP TABLE training_operation_evidence_links")
        connection.execute("DROP TABLE training_orchestration_operations")
        connection.execute(
            "DELETE FROM schema_migrations WHERE version BETWEEN 39 AND 47"
        )
        connection.commit()
    repository.database.migrate()
    reopened = TrainingRepository(repository.database, reviewer_id="owner")
    assert reopened.get_run(run.run_id).state == "calibrated"
    receipt_run = TrainingRun(
        run_id="post-v38-probe-run",
        recipe_id=recipe_id,
        dataset_digest=run.dataset_digest,
        total_steps=run.total_steps,
    )
    reopened.create_run(receipt_run)
    with reopened.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        operation = reopened.begin_operation_with_connection(
            connection,
            operation_id="v38-probe-operation",
            run_id=receipt_run.run_id,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="v38-probe",
            request_payload={
                "request_id": "v38-probe-request",
                "run_id": receipt_run.run_id,
                "recipe_digest": probe.recipe_digest,
                "dataset_digest": probe.dataset_digest,
                "backend_digest": probe.backend_digest,
                "runtime_digest": probe.runtime_digest,
                "material_digest": probe.material_digest,
                "source_digest": probe.source_digest,
            },
        )
        claim = reopened.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="v38-probe",
            claimant_id="worker",
            lease_seconds=60,
        )
        assert claim.operation.claim_token is not None
        new_probe = probe.model_copy(
            update={"evidence_id": "post-v38-probe", "run_id": receipt_run.run_id}
        )
        reopened.record_probe_with_connection(connection, new_probe)
        new_probe_digest = str(
            connection.execute(
                "SELECT training_operation_evidence_digest(?,?,?)",
                (operation.operation_id, "probe", new_probe.evidence_id),
            ).fetchone()[0]
        )
        reopened.append_event_with_connection(
            connection,
            receipt_run.run_id,
            TrainingRunEvent(sequence=1, previous_state="draft", new_state="probed", payload={}),
            expected_version=0,
        )
        reopened.link_operation_evidence_with_connection(
            connection,
            operation_id=operation.operation_id,
            attempt=claim.operation.attempt,
            claim_token=claim.operation.claim_token,
            evidence_kind="probe",
            evidence_id=new_probe.evidence_id,
            evidence_digest=new_probe_digest,
        )
        completed = reopened.complete_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key="v38-probe",
            claimant_id="worker",
            claim_token=claim.operation.claim_token,
            attempt=claim.operation.attempt,
            receipt_payload={
                "run_id": receipt_run.run_id,
                "lifecycle_version": 1,
                "lifecycle_state": "probed",
                "evidence_id": new_probe.evidence_id,
                "evidence_digest": new_probe_digest,
            },
        )
        assert (
            reopened.reopen_operation_with_connection(
                connection, TrainingOperationKind.PROBE, "v38-probe"
            )
            == completed
        )
        connection.commit()


def test_v40_checkpoint_without_derivable_artifact_ids_fails_v41_atomically(
    repository: TrainingRepository,
) -> None:
    _publish_scientific_checkpoint_chain(repository, "nonderivable-v40-run", count=1)
    with repository.database.connect() as connection:
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%v41'"
        ).fetchall():
            connection.execute(f'DROP TRIGGER "{row[0]}"')
        connection.execute("DROP TABLE training_checkpoint_inventory_artifacts")
        connection.execute("DROP TRIGGER validate_training_operation_evidence_link_v43")
        connection.execute("DELETE FROM schema_migrations WHERE version>=41")
        connection.commit()
    with pytest.raises(sqlite3.IntegrityError, match="not uniquely derivable"):
        repository.database.migrate()
    with repository.database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 40
        assert (
            connection.execute(
                "SELECT count(*) FROM training_checkpoints WHERE run_id='nonderivable-v40-run'"
            ).fetchone()[0]
            == 1
        )


def _valid_recipe(dataset_digest: str):
    return _recipe(dataset_digest).model_copy(
        update={
            "character_pack_digest": PACK_DIGEST,
            "stage_2_manifest_digest": STAGE2_MANIFEST_DIGEST,
            "stage_2_evidence_digest": STAGE2_EVIDENCE_DIGEST,
            "stage_3_evidence_digest": STAGE3_RESULT_DIGEST,
        }
    )


def _binding() -> RecipeMaterialBinding:
    return RecipeMaterialBinding(
        stage_3_campaign_id="character_benchmark_training",
        stage_3_result_id=STAGE3_RESULT_DIGEST,
        base_model_artifact_id="artifact-base",
        lightning_adapter_artifact_ids={"adapter": "artifact-lightning"},
        source_snapshot_artifact_id="artifact-source",
        runtime_digest="sha256:"
        + hashlib.sha256(canonical_json({"python": "3.12"}).encode()).hexdigest(),
    )


def _checkpoint(
    *,
    checkpoint_id: str,
    step: int,
    fraction: float,
    parent: str | None,
    artifact: str,
    recipe_id: str,
    dataset_digest: str,
) -> TrainingCheckpoint:
    dummy_inventory = [
        {
            "relative_path": f"{role}.bin",
            "digest": "sha256:" + "9" * 64,
            "size_bytes": 1,
            "role": role,
        }
        for role in (
            "weights",
            "optimizer",
            "trainer_state",
            "telemetry",
            "trainable_targets",
            "frozen_parameters",
            "finite_gradient_proof",
        )
    ]
    manifest = {
        "schema_version": "character-training-checkpoint.v1",
        "run_id": "run-1",
        "reviewer_id": "owner",
        "recipe_id": recipe_id,
        "dataset_digest": dataset_digest,
        "character_id": "character_1",
        "step": step,
        "fraction": fraction,
        "parent_checkpoint_id": parent,
        "artifact_digest": artifact,
        "safetensors_size_bytes": 1,
        "backend_digest": "sha256:" + "1" * 64,
        "runtime_digest": "sha256:" + "2" * 64,
        "material_digest": "sha256:" + "3" * 64,
        "source_digest": "sha256:" + "4" * 64,
        "optimizer_state_digest": "sha256:" + "5" * 64,
        "trainer_state_digest": "sha256:" + "6" * 64,
        "telemetry_digest": "sha256:" + "7" * 64,
        "telemetry_sequence_start": 1,
        "telemetry_sequence_end": 1,
        "trainable_targets_digest": "sha256:" + "8" * 64,
        "frozen_parameters_digest": "sha256:" + "9" * 64,
        "finite_gradient_proof_digest": "sha256:" + "a" * 64,
        "cancellation_boundary": False,
        "inventory": dummy_inventory,
    }
    digest = "sha256:" + hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    return TrainingCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id="run-1",
        step=step,
        fraction=fraction,
        parent_checkpoint_id=parent,
        artifact_digest=artifact,
        manifest_digest=digest,
        manifest=manifest,
    )


def _scientific_checkpoint(
    run: TrainingRun,
    recipe_id: str,
    identity: TrainingProbeEvidence,
    *,
    ordinal: int,
    parent_id: str | None,
    parent_digest: str | None,
    cancellation_boundary: bool = False,
) -> tuple[TrainingCheckpoint, TrainingCheckpointEvidence]:
    fraction = (0.25, 0.5, 0.75, 1.0)[ordinal]
    step = checkpoint_steps(run.total_steps)[ordinal]
    names = ("weights", "optimizer", "trainer", "telemetry", "targets", "frozen", "gradients")
    digests = {
        name: "sha256:" + f"{ordinal + index + 1:x}" * 64 for index, name in enumerate(names)
    }
    inventory = (
        TrainingCheckpointArtifact(
            relative_path="adapter.safetensors",
            digest=digests["weights"],
            size_bytes=10,
            role="weights",
        ),
        TrainingCheckpointArtifact(
            relative_path="optimizer.bin",
            digest=digests["optimizer"],
            size_bytes=10,
            role="optimizer",
        ),
        TrainingCheckpointArtifact(
            relative_path="trainer.json",
            digest=digests["trainer"],
            size_bytes=10,
            role="trainer_state",
        ),
        TrainingCheckpointArtifact(
            relative_path="telemetry.json",
            digest=digests["telemetry"],
            size_bytes=10,
            role="telemetry",
        ),
        TrainingCheckpointArtifact(
            relative_path="targets.json",
            digest=digests["targets"],
            size_bytes=10,
            role="trainable_targets",
        ),
        TrainingCheckpointArtifact(
            relative_path="frozen.json",
            digest=digests["frozen"],
            size_bytes=10,
            role="frozen_parameters",
        ),
        TrainingCheckpointArtifact(
            relative_path="gradients.json",
            digest=digests["gradients"],
            size_bytes=10,
            role="finite_gradient_proof",
        ),
    )
    checkpoint_id = f"{run.run_id}-cp-{ordinal}"
    manifest = {
        "schema_version": "character-training-checkpoint.v1",
        "run_id": run.run_id,
        "reviewer_id": "owner",
        "recipe_id": recipe_id,
        "dataset_digest": run.dataset_digest,
        "character_id": "character_1",
        "step": step,
        "fraction": fraction,
        "parent_checkpoint_id": parent_id,
        "artifact_digest": digests["weights"],
        "safetensors_size_bytes": 10,
        "backend_digest": identity.backend_digest,
        "runtime_digest": identity.runtime_digest,
        "material_digest": identity.material_digest,
        "source_digest": identity.source_digest,
        "optimizer_state_digest": digests["optimizer"],
        "trainer_state_digest": digests["trainer"],
        "telemetry_digest": digests["telemetry"],
        "telemetry_sequence_start": 1,
        "telemetry_sequence_end": ordinal + 1,
        "trainable_targets_digest": digests["targets"],
        "frozen_parameters_digest": digests["frozen"],
        "finite_gradient_proof_digest": digests["gradients"],
        "cancellation_boundary": cancellation_boundary,
        "inventory": [item.model_dump(mode="json") for item in inventory],
    }
    checkpoint = TrainingCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id=run.run_id,
        step=step,
        fraction=fraction,
        parent_checkpoint_id=parent_id,
        artifact_digest=digests["weights"],
        manifest_digest="sha256:" + hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        manifest=manifest,
    )
    evidence = TrainingCheckpointEvidence(
        run_id=run.run_id,
        recipe_digest=recipe_id,
        dataset_digest=run.dataset_digest,
        backend_digest=identity.backend_digest,
        runtime_digest=identity.runtime_digest,
        material_digest=identity.material_digest,
        source_digest=identity.source_digest,
        step=step,
        fraction=fraction,
        parent_checkpoint_digest=parent_digest,
        safetensors_digest=digests["weights"],
        safetensors_size_bytes=10,
        optimizer_state_digest=digests["optimizer"],
        trainer_state_digest=digests["trainer"],
        telemetry_digest=digests["telemetry"],
        telemetry_sequence_start=1,
        telemetry_sequence_end=ordinal + 1,
        trainable_targets_digest=digests["targets"],
        frozen_parameters_digest=digests["frozen"],
        finite_gradient_proof_digest=digests["gradients"],
        cancellation_boundary=cancellation_boundary,
        inventory=inventory,
    )
    return checkpoint, evidence


def _prepare_calibrated_scientific_run(
    repository: TrainingRepository, run_id: str
) -> tuple[TrainingRun, str, TrainingProbeEvidence, TrainingCalibrationEvidence]:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    recipe = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    repository.preregister_recipe(recipe.recipe_id)
    run = TrainingRun(
        run_id=run_id,
        recipe_id=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        total_steps=recipe.recipe.total_steps,
    )
    repository.create_run(run)
    with repository.database.connect() as connection:
        identity = connection.execute(
            "SELECT canonical_json_digest(json_extract(recipe.recipe_json,'$.backend_config')) "
            "AS backend_digest,materials.runtime_digest,'sha256:'||source.sha256 AS source_digest "
            "FROM training_recipes recipe JOIN training_recipe_materials materials "
            "ON materials.recipe_storage_id=recipe.storage_id JOIN artifacts source "
            "ON source.id=materials.source_snapshot_artifact_id WHERE recipe.id=?",
            (recipe.recipe_id,),
        ).fetchone()
    assert identity is not None
    probe = TrainingProbeEvidence(
        evidence_id=f"{run_id}-probe",
        run_id=run_id,
        recipe_digest=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        backend_digest=str(identity["backend_digest"]),
        runtime_digest=str(identity["runtime_digest"]),
        material_digest=recipe_material_digest(_binding()),
        source_digest=str(identity["source_digest"]),
        capability_digest="sha256:" + "8" * 64,
        succeeded=True,
    )
    repository.atomic(
        lambda connection: (
            repository.record_probe_with_connection(connection, probe),
            repository.append_event_with_connection(
                connection,
                run_id,
                TrainingRunEvent(
                    sequence=1, previous_state="draft", new_state="probed", payload={}
                ),
                expected_version=0,
            ),
        )
    )
    calibration = TrainingCalibrationEvidence(
        evidence_id=f"{run_id}-calibration",
        run_id=run_id,
        probe_evidence_id=probe.evidence_id,
        calibration_digest="sha256:" + "9" * 64,
        recipe_digest=probe.recipe_digest,
        dataset_digest=probe.dataset_digest,
        backend_digest=probe.backend_digest,
        runtime_digest=probe.runtime_digest,
        material_digest=probe.material_digest,
        source_digest=probe.source_digest,
        succeeded=True,
    )
    repository.atomic(
        lambda connection: (
            repository.record_calibration_with_connection(connection, calibration),
            repository.append_event_with_connection(
                connection,
                run_id,
                TrainingRunEvent(
                    sequence=2, previous_state="probed", new_state="calibrated", payload={}
                ),
                expected_version=1,
            ),
        )
    )
    return run, recipe.recipe_id, probe, calibration


def _prepare_probed_scientific_run(
    repository: TrainingRepository, run_id: str
) -> tuple[TrainingRun, str, TrainingProbeEvidence]:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    recipe = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    repository.preregister_recipe(recipe.recipe_id)
    run = TrainingRun(
        run_id=run_id,
        recipe_id=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        total_steps=recipe.recipe.total_steps,
    )
    repository.create_run(run)
    with repository.database.connect() as connection:
        identity = connection.execute(
            "SELECT canonical_json_digest(json_extract(recipe.recipe_json,'$.backend_config')) "
            "AS backend_digest,materials.runtime_digest,'sha256:'||source.sha256 AS source_digest "
            "FROM training_recipes recipe JOIN training_recipe_materials materials "
            "ON materials.recipe_storage_id=recipe.storage_id JOIN artifacts source "
            "ON source.id=materials.source_snapshot_artifact_id WHERE recipe.id=?",
            (recipe.recipe_id,),
        ).fetchone()
    assert identity is not None
    probe = TrainingProbeEvidence(
        evidence_id=f"{run_id}-probe",
        run_id=run_id,
        recipe_digest=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        backend_digest=str(identity["backend_digest"]),
        runtime_digest=str(identity["runtime_digest"]),
        material_digest=recipe_material_digest(_binding()),
        source_digest=str(identity["source_digest"]),
        capability_digest="sha256:" + "8" * 64,
        succeeded=True,
    )
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        probe_operation = repository.begin_operation_with_connection(
            connection,
            operation_id=f"{run_id}-probe-operation",
            run_id=run_id,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key=f"{run_id}-probe-key",
            request_payload={
                "request_id": f"{run_id}-probe-request",
                "run_id": run_id,
                "recipe_digest": probe.recipe_digest,
                "dataset_digest": probe.dataset_digest,
                "backend_digest": probe.backend_digest,
                "runtime_digest": probe.runtime_digest,
                "material_digest": probe.material_digest,
                "source_digest": probe.source_digest,
            },
        )
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key=f"{run_id}-probe-key",
            claimant_id="probe-worker",
            lease_seconds=60,
        )
        assert claim.operation.claim_token is not None
        repository.record_probe_with_connection(connection, probe)
        repository.append_event_with_connection(
            connection,
            run_id,
            TrainingRunEvent(sequence=1, previous_state="draft", new_state="probed", payload={}),
            expected_version=0,
        )
        digest = str(
            connection.execute(
                "SELECT training_operation_evidence_digest(?,?,?)",
                (probe_operation.operation_id, "probe", probe.evidence_id),
            ).fetchone()[0]
        )
        repository.link_operation_evidence_with_connection(
            connection,
            operation_id=probe_operation.operation_id,
            attempt=claim.operation.attempt,
            claim_token=claim.operation.claim_token,
            evidence_kind="probe",
            evidence_id=probe.evidence_id,
            evidence_digest=digest,
        )
        repository.complete_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.PROBE,
            idempotency_key=f"{run_id}-probe-key",
            claimant_id="probe-worker",
            claim_token=claim.operation.claim_token,
            attempt=claim.operation.attempt,
            receipt_payload={
                "run_id": run_id,
                "lifecycle_version": 1,
                "lifecycle_state": "probed",
                "evidence_id": probe.evidence_id,
                "evidence_digest": digest,
            },
        )
        connection.commit()
    return run, recipe.recipe_id, probe


def _prepare_draft_scientific_run(
    repository: TrainingRepository, run_id: str
) -> tuple[TrainingRun, TrainingProbeEvidence]:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    recipe = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    repository.preregister_recipe(recipe.recipe_id)
    run = TrainingRun(
        run_id=run_id,
        recipe_id=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        total_steps=recipe.recipe.total_steps,
    )
    repository.create_run(run)
    with repository.database.connect() as connection:
        identity = connection.execute(
            "SELECT canonical_json_digest(json_extract(recipe.recipe_json,'$.backend_config')) "
            "AS backend_digest,materials.runtime_digest,'sha256:'||source.sha256 AS source_digest "
            "FROM training_recipes recipe JOIN training_recipe_materials materials "
            "ON materials.recipe_storage_id=recipe.storage_id JOIN artifacts source "
            "ON source.id=materials.source_snapshot_artifact_id WHERE recipe.id=?",
            (recipe.recipe_id,),
        ).fetchone()
    assert identity is not None
    return run, TrainingProbeEvidence(
        evidence_id=f"{run_id}-probe",
        run_id=run_id,
        recipe_digest=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        backend_digest=str(identity["backend_digest"]),
        runtime_digest=str(identity["runtime_digest"]),
        material_digest=recipe_material_digest(_binding()),
        source_digest=str(identity["source_digest"]),
        capability_digest="sha256:" + "8" * 64,
        succeeded=True,
    )


def _preregister_scientific_training(
    repository: TrainingRepository,
    run: TrainingRun,
    probe: TrainingProbeEvidence,
    calibration: TrainingCalibrationEvidence,
) -> None:
    key = f"{run.run_id}-request"
    digest = "sha256:" + "a" * 64
    repository.atomic(
        lambda connection: repository.bind_idempotency_with_connection(
            connection,
            run_id=run.run_id,
            idempotency_key=key,
            request_digest=digest,
            probe_evidence_id=probe.evidence_id,
            calibration_evidence_id=calibration.evidence_id,
        )
    )
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=3,
            previous_state="calibrated",
            new_state="preregistered",
            payload={
                "idempotency_key": key,
                "request_digest": digest,
                "probe_evidence_id": probe.evidence_id,
                "calibration_evidence_id": calibration.evidence_id,
            },
        ),
        expected_version=2,
    )


def _start_scientific_training(
    repository: TrainingRepository,
    run: TrainingRun,
    probe: TrainingProbeEvidence,
    calibration: TrainingCalibrationEvidence,
) -> None:
    _preregister_scientific_training(repository, run, probe, calibration)
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=4, previous_state="preregistered", new_state="training", payload={}
        ),
        expected_version=3,
    )


def _publish_scientific_checkpoint_chain(
    repository: TrainingRepository,
    run_id: str,
    *,
    count: int = 4,
    prepared: tuple[TrainingRun, str, TrainingProbeEvidence, TrainingCalibrationEvidence]
    | None = None,
    bind_artifacts: bool = False,
    start_ordinal: int = 0,
    parent: tuple[str, str] | None = None,
) -> tuple[
    TrainingRun,
    TrainingProbeEvidence,
    list[tuple[TrainingCheckpoint, TrainingCheckpointEvidence]],
]:
    if prepared is None:
        run, recipe_id, probe, calibration = _prepare_calibrated_scientific_run(repository, run_id)
        _start_scientific_training(repository, run, probe, calibration)
    else:
        run, recipe_id, probe, _ = prepared
    parent_id, parent_digest = parent if parent is not None else (None, None)
    published: list[tuple[TrainingCheckpoint, TrainingCheckpointEvidence]] = []
    for ordinal in range(start_ordinal, count):
        step = checkpoint_steps(run.total_steps)[ordinal]
        repository.append_telemetry(
            run_id,
            TrainingTelemetry(sequence=ordinal + 1, step=step, payload={"loss": ordinal + 1}),
        )
        checkpoint, evidence = _scientific_checkpoint(
            run,
            recipe_id,
            probe,
            ordinal=ordinal,
            parent_id=parent_id,
            parent_digest=parent_digest,
        )
        with repository.database.connect() as connection:
            telemetry_digest = repository.telemetry_digest_with_connection(
                connection, run_id, 1, ordinal + 1
            )
        inventory = tuple(
            item.model_copy(update={"digest": telemetry_digest})
            if item.role == "telemetry"
            else item
            for item in evidence.inventory
        )
        evidence = TrainingCheckpointEvidence.model_validate(
            {
                **evidence.model_dump(mode="python"),
                "telemetry_digest": telemetry_digest,
                "inventory": inventory,
            }
        )
        manifest = dict(checkpoint.manifest)
        manifest["telemetry_digest"] = telemetry_digest
        manifest["inventory"] = [item.model_dump(mode="json") for item in inventory]
        checkpoint = TrainingCheckpoint.model_validate(
            {
                **checkpoint.model_dump(mode="python"),
                "manifest": manifest,
                "manifest_digest": "sha256:"
                + hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
            }
        )

        def publish(
            connection: sqlite3.Connection,
            checkpoint: TrainingCheckpoint = checkpoint,
            evidence: TrainingCheckpointEvidence = evidence,
        ) -> None:
            artifact_ids: dict[str, str] = {}
            if bind_artifacts:
                for item in evidence.inventory:
                    existing = connection.execute(
                        "SELECT id FROM artifacts WHERE sha256=? "
                        "AND media_type='application/octet-stream'",
                        (item.digest.removeprefix("sha256:"),),
                    ).fetchone()
                    artifact_id = (
                        str(existing[0])
                        if existing is not None
                        else f"{checkpoint.checkpoint_id}-artifact-{item.role}"
                    )
                    if existing is None:
                        connection.execute(
                            "INSERT INTO artifacts VALUES "
                            "(?,NULL,?,?, 'application/octet-stream',?,?)",
                            (
                                artifact_id,
                                item.digest.removeprefix("sha256:"),
                                item.size_bytes,
                                artifact_id,
                                NOW.isoformat(),
                            ),
                        )
                    artifact_ids[item.role] = artifact_id
            repository.add_checkpoint_with_connection(connection, checkpoint, evidence)
            if bind_artifacts:
                repository.bind_checkpoint_inventory_artifacts_with_connection(
                    connection,
                    checkpoint_id=checkpoint.checkpoint_id,
                    artifact_ids_by_role=artifact_ids,
                )

        repository.atomic(publish)
        published.append((checkpoint, evidence))
        parent_id = checkpoint.checkpoint_id
        parent_digest = (
            "sha256:"
            + hashlib.sha256(canonical_json(evidence.model_dump(mode="json")).encode()).hexdigest()
        )
    return run, probe, published


def _downgrade_checkpoint_publication_to_physical_v36(repository: TrainingRepository) -> None:
    with repository.database.connect() as connection:
        trigger_names = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND (name LIKE '%v37' OR name LIKE '%v38' OR name LIKE '%v39' "
                "OR name LIKE '%v40' OR name LIKE '%v41' OR name LIKE '%v42' "
                    "OR name LIKE '%v43' OR name LIKE '%v44' OR name LIKE '%v45' "
                    "OR name LIKE '%v46' OR name LIKE '%v47')"
            )
        )
        for trigger_name in trigger_names:
            connection.execute(f'DROP TRIGGER "{trigger_name}"')
        connection.execute("DROP TABLE training_resume_attempt_failures")
        connection.execute("DROP TABLE training_calibration_operation_bindings")
        connection.execute("DROP TABLE training_cancellation_operation_claims")
        connection.execute("DROP TABLE training_operation_evidence_links")
        connection.execute("DROP TABLE training_orchestration_operations")
        connection.execute("DROP TABLE training_checkpoint_inventory_artifacts")
        restored_v36_triggers = (
            "CREATE TRIGGER validate_training_event_insert_v36",
            "CREATE TRIGGER validate_training_checkpoint_insert_v36",
            "CREATE TRIGGER validate_training_probe_insert_v36",
            "CREATE TRIGGER validate_training_checkpoint_inventory_insert_v36",
            "CREATE TRIGGER finalize_training_checkpoint_v36",
            "CREATE TRIGGER guard_finalized_training_checkpoint_v36",
            "CREATE TRIGGER validate_training_cancellation_insert_v36",
        )
        for statement in _migration_statements(_MIGRATIONS[36]):
            if statement.startswith(restored_v36_triggers):
                connection.execute(statement)
        for statement in _migration_statements(_MIGRATIONS[35]):
            if statement.startswith("CREATE TRIGGER validate_training_calibration_insert"):
                connection.execute(statement)
        connection.execute(
            "ALTER TABLE training_cancellation_evidence DROP COLUMN cancellation_request_id"
        )
        connection.execute("ALTER TABLE training_cancellation_evidence DROP COLUMN request_digest")
        connection.execute("DELETE FROM schema_migrations WHERE version>=37")
        connection.commit()


@pytest.fixture
def repository(tmp_path: Path) -> TrainingRepository:
    global STAGE2_EVIDENCE_DIGEST, STAGE2_MANIFEST_DIGEST
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    with database.connect() as connection:
        for artifact_id, digest in (
            ("artifact-base", _valid_recipe("sha256:" + "0" * 64).base_model_digest),
            (
                "artifact-lightning",
                _valid_recipe("sha256:" + "0" * 64).lightning_adapter_digests["adapter"],
            ),
            ("artifact-source", _valid_recipe("sha256:" + "0" * 64).source_code_digest),
        ):
            connection.execute(
                "INSERT INTO artifacts VALUES (?,NULL,?,1,'application/octet-stream',?,?)",
                (artifact_id, digest.removeprefix("sha256:"), artifact_id, NOW.isoformat()),
            )
        connection.execute(
            "INSERT INTO characters VALUES ('character_1','owner','Character',?)",
            (NOW.isoformat(),),
        )
        connection.execute(
            "INSERT INTO character_packs(storage_id,id,character_id,reviewer_id,state,"
            "supersedes_storage_id,manifest_json,created_at,preregistered_at) "
            "VALUES ('pack-storage',?,'character_1','owner','preregistered',NULL,?,?,?)",
            (
                "character_pack_" + "6" * 64,
                canonical_json(PACK_MANIFEST),
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
    campaign_service = validation_service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)
    view = campaign_service.resume(created.campaign_id)
    for _ in range(40):
        assert view.review_id is not None
        with database.connect() as connection:
            trial = connection.execute(
                "SELECT winner_display_position FROM validation_trials WHERE campaign_id=? "
                "ORDER BY ordinal LIMIT 1 OFFSET (SELECT count(*) FROM validation_judgments "
                "WHERE campaign_id=?)",
                (created.campaign_id, created.campaign_id),
            ).fetchone()
        assert trial is not None
        choice = "a" if trial["winner_display_position"] == "left" else "b"
        view = campaign_service.record_choice(created.campaign_id, choice, review_id=view.review_id)
    evidence = campaign_service.export_evidence(created.campaign_id)
    stage_2_manifest = evidence["campaign"]["base_manifest"]
    STAGE2_MANIFEST_DIGEST = (
        "sha256:" + hashlib.sha256(canonical_json(stage_2_manifest).encode()).hexdigest()
    )
    STAGE2_EVIDENCE_DIGEST = (
        "sha256:" + hashlib.sha256(canonical_json(evidence).encode()).hexdigest()
    )
    with database.connect() as connection:
        rebuilt = _validation_evidence_digest(connection, created.campaign_id, "owner")
    assert rebuilt == STAGE2_EVIDENCE_DIGEST, (rebuilt, STAGE2_EVIDENCE_DIGEST)
    definition_json = {
        "id": "character_benchmark_training",
        "character_pack_digest": PACK_DIGEST,
        "stage_2_manifest_digest": _valid_recipe("sha256:" + "0" * 64).stage_2_manifest_digest,
        "stage_2_evidence_digest": STAGE2_EVIDENCE_DIGEST,
    }
    campaign_payload = {
        "definition": definition_json,
        "character_pack_id": "character_pack_" + "6" * 64,
        "character_pack_storage_id": "pack-storage",
        "stage_2_campaign_id": created.campaign_id,
        "stage_2_manifest": stage_2_manifest,
        "identity_prompt": "identity",
    }
    campaign_digest = (
        "sha256:" + hashlib.sha256(canonical_json(campaign_payload).encode()).hexdigest()
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO character_benchmark_campaigns VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "character_benchmark_training",
                "owner",
                campaign_digest,
                canonical_json(definition_json),
                "character_pack_" + "6" * 64,
                "pack-storage",
                created.campaign_id,
                canonical_json(stage_2_manifest),
                "identity",
                NOW.isoformat(),
            ),
        )
        for sequence, (previous, new) in enumerate(
            (
                ("draft", "preregistered"),
                ("preregistered", "generating"),
                ("generating", "ready"),
                ("ready", "judging"),
                ("judging", "analyzing"),
            ),
            start=1,
        ):
            connection.execute(
                "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
                (
                    "character_benchmark_training",
                    sequence,
                    previous,
                    new,
                    "sha256:" + f"{sequence}" * 64,
                    NOW.isoformat(),
                ),
            )
        connection.execute(
            "INSERT INTO character_benchmark_results VALUES (?,?,?,?)",
            (
                "character_benchmark_training",
                canonical_json(STAGE3_RESULT),
                STAGE3_RESULT_DIGEST,
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
            (
                "character_benchmark_training",
                6,
                "analyzing",
                "completed",
                "sha256:" + "6" * 64,
                NOW.isoformat(),
            ),
        )
    return TrainingRepository(database, reviewer_id="owner")


def test_dataset_recipe_round_trip_and_reopen_preserve_canonical_identity(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    created = repository.create_dataset(frozen.character_id, frozen)
    sealed = repository.seal_dataset(created.dataset_id, frozen)
    recipe = _valid_recipe(frozen.dataset_digest)
    recipe_record = repository.create_recipe(recipe, _binding())
    preregistered = repository.preregister_recipe(recipe_record.recipe_id)
    reopened = TrainingRepository(repository.database, reviewer_id="owner")
    assert sealed.dataset_digest == frozen.dataset_digest
    assert reopened.get_dataset(created.dataset_id).frozen_dataset == frozen
    assert reopened.get_recipe(recipe_record.recipe_id).recipe == recipe
    assert preregistered.state == "preregistered"
    with repository.database.connect() as connection:
        for sql in (
            "UPDATE training_recipes SET recipe_json='{}' WHERE storage_id=?",
            "DELETE FROM training_recipes WHERE storage_id=?",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(sql, (recipe_record.storage_id,))


def test_recipe_creation_can_participate_in_caller_transaction(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    created = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(created.dataset_id, frozen)
    recipe = _valid_recipe(frozen.dataset_digest)

    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        record = repository.create_recipe(recipe, _binding(), connection=connection)
        repository.preregister_recipe(record.recipe_id, connection=connection)
        connection.rollback()

    with pytest.raises(TrainingNotFound):
        repository.get_recipe(training_recipe_id(recipe))


def test_recipe_material_ledger_is_immutable_and_typed_on_load(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    created = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(created.dataset_id, frozen)
    recipe = _valid_recipe(frozen.dataset_digest)
    record = repository.create_recipe(recipe, _binding())

    assert repository.get_recipe(record.recipe_id).recipe == recipe
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="materials are immutable"):
            connection.execute(
                "UPDATE training_recipe_materials SET runtime_digest=? WHERE recipe_storage_id=?",
                ("sha256:" + "f" * 64, record.storage_id),
            )


@pytest.mark.parametrize("invalidate", (False, True))
def test_populated_v32_material_upgrade_is_exact_or_fails_atomically(
    repository: TrainingRepository, invalidate: bool
) -> None:
    frozen = _frozen()
    created = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(created.dataset_id, frozen)
    record = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    with repository.database.connect() as connection:
        for trigger in (
            "validate_training_recipe_material_insert_v34",
            "validate_training_recipe_materials_preregister_v34",
            "validate_training_recipe_material_insert_v33",
            "validate_training_recipe_lightning_insert_v33",
            "guard_training_recipe_lightning_update_v33",
            "guard_training_recipe_lightning_delete_v33",
            "validate_training_recipe_materials_preregister_v33",
        ):
            connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        connection.execute("DROP TABLE training_recipe_lightning_materials")
        connection.execute("DROP INDEX training_recipe_materials_owner_v33")
        connection.execute("DELETE FROM schema_migrations WHERE version IN (33,34)")
        if invalidate:
            connection.execute("DROP TRIGGER guard_training_recipe_material_update_v32")
            connection.execute(
                "UPDATE training_recipe_materials SET stage_3_result_id=? "
                "WHERE recipe_storage_id=?",
                ("sha256:" + "0" * 64, record.storage_id),
            )
        connection.commit()

    if invalidate:
        with pytest.raises(sqlite3.IntegrityError, match="v32 training recipe materials"):
            repository.database.migrate()
        with repository.database.connect() as connection:
            assert (
                connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
                == max(_MIGRATIONS)
            )
            assert (
                connection.execute(
                    "SELECT stage_3_result_id FROM training_recipe_materials"
                ).fetchone()[0]
                == "sha256:" + "0" * 64
            )
    else:
        repository.database.migrate()
        assert repository.get_recipe(record.recipe_id).binding == _binding()
        with repository.database.connect() as connection:
            assert (
                connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
                == max(_MIGRATIONS)
            )
            assert dict(
                connection.execute(
                    "SELECT material_name,artifact_id FROM training_recipe_lightning_materials"
                ).fetchall()
            ) == {"adapter": "artifact-lightning"}


def _downgrade_populated_recipe_to_actual_v33(repository: TrainingRepository) -> None:
    with repository.database.connect() as connection:
        connection.execute("DROP TRIGGER validate_training_recipe_material_insert_v34")
        connection.execute("DROP TRIGGER validate_training_recipe_materials_preregister_v34")
        for statement in _migration_statements(_MIGRATIONS[33]):
            if statement.startswith("CREATE TRIGGER validate_training_recipe_material_insert_v33"):
                connection.execute(statement)
            if statement.startswith(
                "CREATE TRIGGER validate_training_recipe_materials_preregister_v33"
            ):
                connection.execute(statement)
        connection.execute("DELETE FROM schema_migrations WHERE version=34")
        connection.commit()


def _downgrade_populated_run_to_physical_v34(repository: TrainingRepository) -> None:
    with repository.database.connect() as connection:
        v35_tables = {
            "training_probe_evidence",
            "training_calibration_evidence",
            "training_cancellation_evidence",
            "training_resume_evidence",
            "training_evaluation_evidence",
            "training_checkpoint_inventory",
        }
        triggers = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND "
                "(name LIKE '%v36' OR name LIKE '%v37' OR name LIKE '%v38' "
                "OR name LIKE '%v39' OR name LIKE '%v40' OR name LIKE '%v41' "
                "OR name LIKE '%v42' OR name LIKE '%v43' OR name LIKE '%v44' "
                "OR name LIKE '%v45' OR name LIKE '%v46' OR name LIKE '%v47' OR "
                "(sql LIKE '%lifecycle_state%' OR sql LIKE '%training_probe_evidence%' "
                "OR sql LIKE '%training_calibration_evidence%' "
                "OR sql LIKE '%training_cancellation_evidence%' "
                "OR sql LIKE '%training_resume_evidence%' "
                "OR sql LIKE '%training_evaluation_evidence%' "
                "OR sql LIKE '%training_checkpoint_inventory%'))"
            )
        )
        for trigger in triggers:
            connection.execute(f'DROP TRIGGER "{trigger}"')
        connection.execute("DROP TABLE training_resume_attempt_failures")
        connection.execute("DROP TABLE training_calibration_operation_bindings")
        connection.execute("DROP TABLE training_cancellation_operation_claims")
        connection.execute("DROP TABLE training_operation_evidence_links")
        connection.execute("DROP TABLE training_orchestration_operations")
        connection.execute("DROP TABLE training_checkpoint_inventory_artifacts")
        connection.execute("DROP TABLE training_run_idempotency")
        connection.execute("DROP TABLE training_cancellation_requests")
        connection.execute("ALTER TABLE training_checkpoints DROP COLUMN publication_state")
        connection.execute("DROP INDEX training_runs_idempotency")
        connection.execute("DROP INDEX training_probe_one_per_run")
        connection.execute("DROP INDEX training_calibration_one_per_run")
        for table in v35_tables:
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("ALTER TABLE training_checkpoints DROP COLUMN evidence_json")
        connection.execute("ALTER TABLE training_checkpoints DROP COLUMN evidence_digest")
        for column in ("lifecycle_state", "lifecycle_version", "idempotency_key", "request_digest"):
            connection.execute(f'ALTER TABLE training_runs DROP COLUMN "{column}"')
        for statement in _migration_statements(_MIGRATIONS[23]):
            if statement.startswith("CREATE TRIGGER validate_training_event_insert"):
                connection.execute(statement)
            elif statement.startswith("CREATE TRIGGER apply_training_event"):
                connection.execute(statement)
            elif statement.startswith("CREATE TRIGGER validate_training_telemetry_insert"):
                connection.execute(statement)
        for statement in _migration_statements(_MIGRATIONS[24]):
            if statement.startswith("CREATE TRIGGER validate_training_checkpoint_insert"):
                connection.execute(statement)
            elif statement.startswith("CREATE TRIGGER guard_training_checkpoint_update"):
                connection.execute(statement)
            elif statement.startswith("CREATE TRIGGER guard_training_checkpoint_delete"):
                connection.execute("DROP TRIGGER IF EXISTS guard_training_checkpoint_delete")
                connection.execute(statement)
        connection.execute("DELETE FROM schema_migrations WHERE version>=35")
        connection.commit()


@pytest.mark.parametrize("invalid", (False, True))
def test_real_populated_physical_v34_run_upgrade_is_exact_and_atomic(
    repository: TrainingRepository, invalid: bool
) -> None:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    recipe = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    repository.preregister_recipe(recipe.recipe_id)
    run = TrainingRun(
        run_id="legacy-v34-run",
        recipe_id=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        total_steps=recipe.recipe.total_steps,
    )
    repository.create_run(run)
    _downgrade_populated_run_to_physical_v34(repository)
    if invalid:
        with repository.database.connect() as connection:
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DROP TRIGGER validate_training_run_insert")
            connection.execute(
                "UPDATE training_runs SET recipe_storage_id='missing' WHERE id=?", (run.run_id,)
            )
            connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="v34 training runs"):
            repository.database.migrate()
        with repository.database.connect() as connection:
            assert (
                connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 34
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(training_runs)")
            }
            assert "lifecycle_state" not in columns
    else:
        repository.database.migrate()
        reopened = TrainingRepository(repository.database, reviewer_id="owner")
        assert reopened.get_run(run.run_id).state == "draft"
        with repository.database.connect() as connection:
            assert (
                connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
                == max(_MIGRATIONS)
            )
            row = connection.execute(
                "SELECT recipe_storage_id,dataset_storage_id,total_steps "
                "FROM training_runs WHERE id=?",
                (run.run_id,),
            ).fetchone()
            assert row is not None
            assert tuple(row) == (recipe.storage_id, dataset.storage_id, run.total_steps)


@pytest.mark.parametrize("legacy_state", ("running", "completed", "failed", "cancelled"))
def test_noncreated_legacy_v34_runs_are_rejected_without_invented_failure(
    repository: TrainingRepository, legacy_state: str
) -> None:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    recipe = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    repository.preregister_recipe(recipe.recipe_id)
    run = TrainingRun(
        run_id=f"legacy-{legacy_state}",
        recipe_id=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        total_steps=recipe.recipe.total_steps,
    )
    repository.create_run(run)
    _downgrade_populated_run_to_physical_v34(repository)
    terminal_at = None if legacy_state == "running" else NOW.isoformat()
    with repository.database.connect() as connection:
        connection.execute(
            "UPDATE training_runs SET state=?,terminal_at=? WHERE id=?",
            (legacy_state, terminal_at, run.run_id),
        )
        connection.commit()
    with pytest.raises(sqlite3.IntegrityError, match="non-runnable"):
        repository.database.migrate()
    with repository.database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 34
        assert (
            connection.execute(
                "SELECT count(*) FROM training_failures WHERE run_id=?", (run.run_id,)
            ).fetchone()[0]
            == 0
        )


def test_actual_populated_v33_upgrades_and_remains_loadable_and_preregisterable(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    created = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(created.dataset_id, frozen)
    record = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    _downgrade_populated_recipe_to_actual_v33(repository)

    repository.database.migrate()

    reopened = TrainingRepository(repository.database, reviewer_id="owner")
    assert reopened.get_recipe(record.recipe_id).binding == _binding()
    assert reopened.preregister_recipe(record.recipe_id).state == "preregistered"
    with repository.database.connect() as connection:
        assert (
            connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            == max(_MIGRATIONS)
        )


@pytest.mark.parametrize("malformation", ["owner", "parent_json", "child_binding"])
def test_malformed_populated_v33_fails_atomically_before_v34(
    repository: TrainingRepository, malformation: str
) -> None:
    frozen = _frozen()
    created = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(created.dataset_id, frozen)
    record = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    _downgrade_populated_recipe_to_actual_v33(repository)
    with repository.database.connect() as connection:
        if malformation == "owner":
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DROP TRIGGER guard_training_recipe_material_update_v32")
            connection.execute(
                "UPDATE training_recipe_materials SET reviewer_id='intruder' "
                "WHERE recipe_storage_id=?",
                (record.storage_id,),
            )
        elif malformation == "parent_json":
            connection.execute("DROP TRIGGER guard_training_recipe_material_update_v32")
            connection.execute(
                "UPDATE training_recipe_materials "
                'SET lightning_adapter_artifact_ids_json=\'{"adapter" : "artifact-lightning"}\' '
                "WHERE recipe_storage_id=?",
                (record.storage_id,),
            )
        else:
            connection.execute("DROP TRIGGER guard_training_recipe_lightning_update_v33")
            connection.execute(
                "UPDATE training_recipe_lightning_materials SET artifact_id='artifact-base' "
                "WHERE recipe_storage_id=?",
                (record.storage_id,),
            )
        connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="v33 training recipe materials"):
        repository.database.migrate()
    with repository.database.connect() as connection:
        assert (
            connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            == max(_MIGRATIONS)
        )
        assert connection.execute("SELECT count(*) FROM training_recipes").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE name='validate_training_recipe_material_insert_v34'"
            ).fetchone()
            is None
        )


def test_alias_collision_rolls_back_entire_dataset_write(repository: TrainingRepository) -> None:
    frozen = _frozen()
    repository.create_dataset(frozen.character_id, frozen)
    with pytest.raises(TrainingConflict, match="dataset identity conflict"):
        repository.create_dataset(frozen.character_id, frozen)
    with repository.database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM training_datasets").fetchone()[0] == 1


def test_dataset_lineage_shared_connection_cas_reopen_and_terminal_seal(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    lineage = repository.create_dataset_for_lineage(
        "lineage-1", frozen.character_id, repository._draft_from(frozen)
    )
    changed = frozen.model_copy(
        update={
            "samples": (
                frozen.samples[0].model_copy(update={"caption": "<mirza> human revised caption"}),
            )
            + frozen.samples[1:]
        }
    )
    changed = repository._draft_from(changed)
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        new_id = repository.publish_dataset_head_with_connection(
            connection,
            lineage_id=lineage.lineage_id,
            expected_head_version=1,
            draft=changed,
        )
        connection.commit()
    reopened = TrainingRepository(repository.database, reviewer_id="owner")
    assert reopened.get_dataset_lineage("lineage-1").current_dataset_id == new_id
    assert tuple(event.version for event in reopened.list_dataset_head_history("lineage-1")) == (
        1,
        2,
    )
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(TrainingConflict, match="stale"):
            repository.publish_dataset_head_with_connection(
                connection,
                lineage_id="lineage-1",
                expected_head_version=1,
                draft=changed,
            )
        connection.rollback()
    sealed_dataset = freeze_training_dataset(changed)
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="lineage-1",
            expected_head_version=2,
            draft=sealed_dataset,
            seal=True,
        )
        connection.commit()
    assert reopened.get_dataset_lineage("lineage-1").state == "sealed"
    sealed_import = _imported(lineage_id="lineage-1", character_id="character_1")
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(TrainingConflict, match="sealed"):
            repository.persist_imported_sample_with_connection(connection, sealed_import)
        with pytest.raises(TrainingConflict, match="stale or invalid"):
            repository.publish_dataset_head_with_connection(
                connection,
                lineage_id="lineage-1",
                expected_head_version=3,
                draft=changed,
            )
        connection.rollback()


def test_shared_connection_failure_rolls_back_snapshot_and_head(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-rollback", frozen.character_id, repository._draft_from(frozen)
    )
    changed = frozen.model_copy(
        update={
            "samples": (
                frozen.samples[0].model_copy(update={"caption": "<mirza> rollback caption"}),
            )
            + frozen.samples[1:]
        }
    )
    changed = repository._draft_from(changed)
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="lineage-rollback",
            expected_head_version=1,
            draft=changed,
        )
        connection.rollback()
    assert repository.get_dataset_lineage("lineage-rollback").head_version == 1
    with repository.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM training_dataset_head_events "
                "WHERE lineage_id='lineage-rollback'"
            ).fetchone()[0]
            == 1
        )


def test_import_and_confirmation_shared_transaction_reopen_and_artifact_binding(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-import", frozen.character_id, repository._draft_from(frozen)
    )
    imported = _imported(lineage_id="lineage-import", character_id="character_1")
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest.removeprefix("sha256:"),
                imported.artifact_byte_size,
                imported.media.media_type,
                "canonical/path",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        assert connection.in_transaction
        repository.persist_imported_sample_with_connection(connection, imported)
        confirmation = _confirmation(imported.import_id)
        repository.persist_confirmation_with_connection(connection, confirmation)
        assert connection.in_transaction
        connection.commit()
    reopened = TrainingRepository(repository.database, reviewer_id="owner")
    assert reopened.list_imported_samples("lineage-import") == (imported,)
    assert reopened.list_confirmations(imported.import_id) == (confirmation,)


def test_dataset_snapshot_binds_complete_import_and_confirmation_ledgers(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-ledger", frozen.character_id, repository._draft_from(frozen)
    )
    imported = _imported(lineage_id="lineage-ledger", character_id="character_1")
    confirmation = _confirmation(imported.import_id, sample_id="curated-ledger")
    accepted = confirmation.accepted_sample(imported)
    draft = repository._draft_from(frozen).model_copy(
        update={
            "samples": frozen.samples + (accepted,),
            "imported_sample_ids": (imported.import_id,),
            "curation_confirmation_ids": (confirmation.confirmation_id,),
        }
    )
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "path",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        repository.persist_imported_sample_with_connection(connection, imported)
        import_draft = repository._draft_from(frozen).model_copy(
            update={"imported_sample_ids": (imported.import_id,)}
        )
        repository.publish_dataset_head_with_connection(
            connection, lineage_id="lineage-ledger", expected_head_version=1, draft=import_draft
        )
        repository.persist_confirmation_with_connection(connection, confirmation)
        dataset_id = repository.publish_dataset_head_with_connection(
            connection, lineage_id="lineage-ledger", expected_head_version=2, draft=draft
        )
        connection.commit()
    reopened = repository.get_dataset(dataset_id)
    assert reopened.draft.imported_sample_ids == (imported.import_id,)
    assert reopened.draft.curation_confirmation_ids == (confirmation.confirmation_id,)
    with repository.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT import_id FROM training_dataset_snapshot_imports "
                "WHERE dataset_storage_id=?",
                (reopened.storage_id,),
            ).fetchone()[0]
            == imported.import_id
        )


def test_dataset_snapshot_rejects_missing_confirmation_binding(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-missing", frozen.character_id, repository._draft_from(frozen)
    )
    draft = repository._draft_from(frozen).model_copy(
        update={
            "imported_sample_ids": ("training_import_" + "a" * 64,),
        }
    )
    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(TrainingConflict):
            repository.publish_dataset_head_with_connection(
                connection, lineage_id="lineage-missing", expected_head_version=1, draft=draft
            )
        connection.rollback()


def test_rejected_confirmation_advances_identity_without_accepted_sample(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-reject", frozen.character_id, repository._draft_from(frozen)
    )
    imported = _imported(lineage_id="lineage-reject", character_id="character_1")
    rejected = _confirmation(
        imported.import_id,
        decision=CurationDecision.REJECT,
        sample_id=None,
        source_tier=None,
        source_lineage=None,
        caption=None,
        split=None,
        annotation=None,
    )
    initial = repository._draft_from(frozen)
    import_draft = initial.model_copy(update={"imported_sample_ids": (imported.import_id,)})
    rejected_draft = import_draft.model_copy(
        update={"curation_confirmation_ids": (rejected.confirmation_id,)}
    )
    assert training_dataset_id(initial) != training_dataset_id(import_draft)
    assert training_dataset_id(import_draft) != training_dataset_id(rejected_draft)
    assert rejected_draft.samples == import_draft.samples
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "reject/path",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        repository.persist_imported_sample_with_connection(connection, imported)
        repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="lineage-reject",
            expected_head_version=1,
            draft=import_draft,
        )
        repository.persist_confirmation_with_connection(connection, rejected)
        rejected_id = repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="lineage-reject",
            expected_head_version=2,
            draft=rejected_draft,
        )
        connection.commit()
    assert repository.get_dataset(rejected_id).draft.samples == initial.samples

    orphan = _imported(
        lineage_id="lineage-reject",
        character_id="character_1",
        artifact_id="orphan-artifact",
        media=TrainingMedia(
            sha256="sha256:" + "8" * 64,
            media_type="image/png",
            byte_size=14,
            width=4,
            height=3,
        ),
        artifact_digest="sha256:" + "8" * 64,
        artifact_byte_size=14,
    )
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                orphan.artifact_id,
                None,
                orphan.artifact_digest[7:],
                orphan.artifact_byte_size,
                orphan.media.media_type,
                "orphan/path",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        repository.persist_imported_sample_with_connection(connection, orphan)
        with pytest.raises(TrainingConflict):
            repository.publish_dataset_head_with_connection(
                connection,
                lineage_id="lineage-reject",
                expected_head_version=3,
                draft=freeze_training_dataset(rejected_draft),
                seal=True,
            )
        connection.rollback()


def test_snapshot_must_bind_complete_persisted_lineage_ledgers(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-complete", frozen.character_id, repository._draft_from(frozen)
    )
    first = _imported(lineage_id="lineage-complete", character_id="character_1")
    second = _imported(
        lineage_id="lineage-complete",
        character_id="character_1",
        artifact_id="artifact-2",
        media=TrainingMedia(
            sha256="sha256:" + "9" * 64, media_type="image/png", byte_size=13, width=4, height=3
        ),
        artifact_digest="sha256:" + "9" * 64,
        artifact_byte_size=13,
    )
    incomplete = repository._draft_from(frozen).model_copy(
        update={"imported_sample_ids": (first.import_id,)}
    )
    with repository.database.connect() as connection:
        for imported in (first, second):
            connection.execute(
                "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
                (
                    imported.artifact_id,
                    None,
                    imported.artifact_digest[7:],
                    imported.artifact_byte_size,
                    imported.media.media_type,
                    imported.artifact_id,
                    NOW.isoformat(),
                ),
            )
            repository.persist_imported_sample_with_connection(connection, imported)
        with pytest.raises(TrainingConflict):
            repository.publish_dataset_head_with_connection(
                connection,
                lineage_id="lineage-complete",
                expected_head_version=1,
                draft=incomplete,
            )


def test_import_action_cannot_change_curated_samples(repository: TrainingRepository) -> None:
    frozen = _frozen()
    initial = repository._draft_from(frozen)
    repository.create_dataset_for_lineage("lineage-import-delta", frozen.character_id, initial)
    imported = _imported(lineage_id="lineage-import-delta", character_id="character_1")
    changed_samples = (
        initial.samples[0].model_copy(update={"caption": "<mirza> unrelated replacement"}),
    ) + initial.samples[1:]
    invalid = initial.model_copy(
        update={
            "samples": changed_samples,
            "imported_sample_ids": (imported.import_id,),
        }
    )
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "delta/path",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        repository.persist_imported_sample_with_connection(connection, imported)
        with pytest.raises(TrainingConflict):
            repository.publish_dataset_head_with_connection(
                connection,
                lineage_id="lineage-import-delta",
                expected_head_version=1,
                draft=invalid,
            )


def test_sealed_lineage_rejects_direct_import_and_direct_or_repository_confirmation(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    initial = repository._draft_from(frozen)
    repository.create_dataset_for_lineage("lineage-sealed-ledger", frozen.character_id, initial)
    imported = _imported(lineage_id="lineage-sealed-ledger", character_id="character_1")
    import_draft = initial.model_copy(update={"imported_sample_ids": (imported.import_id,)})
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "sealed/path",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        repository.persist_imported_sample_with_connection(connection, imported)
        repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="lineage-sealed-ledger",
            expected_head_version=1,
            draft=import_draft,
        )
        repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="lineage-sealed-ledger",
            expected_head_version=2,
            draft=freeze_training_dataset(import_draft),
            seal=True,
        )
        connection.commit()

    confirmation = _confirmation(
        imported.import_id,
        decision=CurationDecision.REJECT,
        sample_id=None,
        source_tier=None,
        source_lineage=None,
        caption=None,
        split=None,
        annotation=None,
    )
    later = _imported(
        lineage_id="lineage-sealed-ledger",
        character_id="character_1",
        artifact_id="later-artifact",
        media=TrainingMedia(
            sha256="sha256:" + "7" * 64, media_type="image/png", byte_size=15, width=4, height=3
        ),
        artifact_digest="sha256:" + "7" * 64,
        artifact_byte_size=15,
    )
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                later.artifact_id,
                None,
                later.artifact_digest[7:],
                later.artifact_byte_size,
                later.media.media_type,
                "later/path",
                NOW.isoformat(),
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="sealed"):
            connection.execute(
                "INSERT INTO training_imported_samples VALUES (?,?,?,?,?,?,?,?)",
                (
                    later.import_id,
                    later.reviewer_id,
                    later.character_id,
                    later.lineage_id,
                    later.artifact_id,
                    later.artifact_digest,
                    canonical_json(later.model_dump(mode="json")),
                    later.model_dump(mode="json")["imported_at"],
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="sealed"):
            connection.execute(
                "INSERT INTO training_curation_confirmations VALUES (?,?,?,?,?,?,?)",
                (
                    confirmation.confirmation_id,
                    confirmation.import_id,
                    confirmation.reviewer_id,
                    confirmation.version,
                    confirmation.decision.value,
                    canonical_json(confirmation.model_dump(mode="json")),
                    confirmation.model_dump(mode="json")["confirmed_at"],
                ),
            )
        with pytest.raises(TrainingConflict, match="sealed"):
            repository.persist_confirmation_with_connection(connection, confirmation)


def test_raw_sqlite_without_registered_udfs_fails_closed(repository: TrainingRepository) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-raw", frozen.character_id, repository._draft_from(frozen)
    )
    before = repository.get_dataset_lineage("lineage-raw")
    connection = sqlite3.connect(repository.database.path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            connection.execute(
                "UPDATE training_dataset_lineages SET head_version=head_version+1 "
                "WHERE id='lineage-raw'"
            )
        connection.rollback()
    finally:
        connection.close()
    assert repository.get_dataset_lineage("lineage-raw") == before


def test_direct_sql_rejects_denormalized_import_and_confirmation_timestamps(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-time", frozen.character_id, repository._draft_from(frozen)
    )
    imported = _imported(lineage_id="lineage-time", character_id="character_1")
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "path",
                NOW.isoformat(),
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_imported_samples VALUES (?,?,?,?,?,?,?,?)",
                (
                    imported.import_id,
                    "owner",
                    "character_1",
                    "lineage-time",
                    imported.artifact_id,
                    imported.artifact_digest,
                    canonical_json(imported.model_dump(mode="json")),
                    "wrong-time",
                ),
            )
        connection.execute("BEGIN IMMEDIATE")
        repository.persist_imported_sample_with_connection(connection, imported)
        connection.commit()
        confirmation = _confirmation(imported.import_id)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_curation_confirmations VALUES (?,?,?,?,?,?,?)",
                (
                    confirmation.confirmation_id,
                    imported.import_id,
                    "owner",
                    1,
                    "accept",
                    canonical_json(confirmation.model_dump(mode="json")),
                    "wrong-time",
                ),
            )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("$.artifact_digest", "sha256:" + "0" * 64),
        ("$.artifact_byte_size", 999),
        ("$.media.byte_size", 999),
        ("$.media.media_type", "image/jpeg"),
    ],
)
def test_direct_import_sql_rejects_artifact_metadata_mismatch(
    repository: TrainingRepository, path: str, value: object
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-import", frozen.character_id, repository._draft_from(frozen)
    )
    imported = _imported(lineage_id="lineage-import", character_id="character_1")
    payload = imported.model_dump(mode="json")
    target = payload
    parts = path.removeprefix("$.").split(".")
    for part in parts[:-1]:
        target = target[part]  # type: ignore[assignment,index]
    target[parts[-1]] = value  # type: ignore[index]
    with repository.database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "path",
                NOW.isoformat(),
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_imported_samples VALUES (?,?,?,?,?,?,?,?)",
                (
                    imported.import_id,
                    "owner",
                    "character_1",
                    "lineage-import",
                    imported.artifact_id,
                    imported.artifact_digest,
                    canonical_json(payload),
                    NOW.isoformat(),
                ),
            )


def test_head_history_is_derived_and_snapshot_cannot_cross_lineage_or_rollback(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    derived = repository.create_dataset_for_lineage(
        "lineage-derived", frozen.character_id, repository._draft_from(frozen)
    )
    initial = repository.get_dataset(derived.current_dataset_id)
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="derived"):
            connection.execute(
                "INSERT INTO training_dataset_head_events VALUES (?,?,?,?,?,?,?,?)",
                (
                    "lineage-derived",
                    2,
                    "owner",
                    "character_1",
                    initial.storage_id,
                    initial.storage_id,
                    "draft",
                    NOW.isoformat(),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_dataset_lineages VALUES (?,?,?,?,1,'draft',?)",
                ("other-lineage", "owner", "character_1", initial.storage_id, NOW.isoformat()),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE training_dataset_lineages SET current_dataset_storage_id=?,"
                "head_version=2 WHERE id='lineage-derived'",
                (initial.storage_id,),
            )


def test_connection_aware_methods_reject_closed_and_foreign_connections(
    repository: TrainingRepository, tmp_path: Path
) -> None:
    imported = _imported()
    closed = repository.database.connect()
    closed.close()
    with pytest.raises(TrainingConflict, match="closed"):
        repository.persist_imported_sample_with_connection(closed, imported)
    foreign_database = Database(tmp_path / "foreign.sqlite3")
    foreign_database.migrate()
    with foreign_database.connect() as foreign:
        foreign.execute("BEGIN IMMEDIATE")
        with pytest.raises(TrainingConflict, match="foreign"):
            repository.persist_imported_sample_with_connection(foreign, imported)
        assert foreign.in_transaction
        foreign.rollback()


def test_dataset_declared_lineage_and_derived_registry_reject_direct_rebinding(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    lineage = repository.create_dataset_for_lineage(
        "lineage-schema", frozen.character_id, repository._draft_from(frozen)
    )
    snapshot = repository.get_dataset(lineage.current_dataset_id)
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="declared lineage is immutable"):
            connection.execute(
                "UPDATE training_datasets SET lineage_id='other-lineage' WHERE storage_id=?",
                (snapshot.storage_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="lineage is derived"):
            connection.execute(
                "INSERT INTO training_dataset_snapshot_lineages VALUES (?,?,?,?,?)",
                (snapshot.storage_id, "other-lineage", "owner", "character_1", 1),
            )
        with pytest.raises(sqlite3.IntegrityError, match="invalid training dataset head CAS"):
            connection.execute(
                "UPDATE training_dataset_lineages SET id='other-lineage',head_version=2 "
                "WHERE id='lineage-schema'"
            )
    reopened = TrainingRepository(repository.database, reviewer_id="owner")
    assert reopened.get_dataset_lineage("lineage-schema") == lineage


def test_two_open_connections_same_expected_head_only_first_wins(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-race", frozen.character_id, repository._draft_from(frozen)
    )
    changed = repository._draft_from(
        frozen.model_copy(
            update={
                "samples": (frozen.samples[0].model_copy(update={"caption": "<mirza> race"}),)
                + frozen.samples[1:]
            }
        )
    )
    first = repository.database.connect()
    second = repository.database.connect()
    try:
        first.execute("BEGIN IMMEDIATE")
        repository.publish_dataset_head_with_connection(
            first, lineage_id="lineage-race", expected_head_version=1, draft=changed
        )
        first.commit()
        second.execute("BEGIN IMMEDIATE")
        with pytest.raises(TrainingConflict, match="stale"):
            repository.publish_dataset_head_with_connection(
                second, lineage_id="lineage-race", expected_head_version=1, draft=changed
            )
        second.rollback()
    finally:
        first.close()
        second.close()


def test_overlapping_connections_race_one_cas_commit_and_one_stable_conflict(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-overlap", frozen.character_id, repository._draft_from(frozen)
    )
    barrier = threading.Barrier(2)
    results: list[str] = []
    result_lock = threading.Lock()

    def compete(label: str) -> None:
        changed = repository._draft_from(
            frozen.model_copy(
                update={
                    "samples": (
                        frozen.samples[0].model_copy(update={"caption": f"<mirza> {label}"}),
                    )
                    + frozen.samples[1:]
                }
            )
        )
        with repository.database.connect() as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            barrier.wait(timeout=5)
            try:
                connection.execute("BEGIN IMMEDIATE")
                repository.publish_dataset_head_with_connection(
                    connection,
                    lineage_id="lineage-overlap",
                    expected_head_version=1,
                    draft=changed,
                )
                connection.commit()
                outcome = "committed"
            except TrainingConflict:
                connection.rollback()
                outcome = "conflict"
            with result_lock:
                results.append(outcome)

    threads = tuple(threading.Thread(target=compete, args=(label,)) for label in ("a", "b"))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sorted(results) == ["committed", "conflict"]
    assert repository.get_dataset_lineage("lineage-overlap").head_version == 2
    assert len(repository.list_dataset_head_history("lineage-overlap")) == 2


def test_artifact_batch_import_reresolution_failure_rolls_back_everything(
    repository: TrainingRepository, tmp_path: Path
) -> None:
    frozen = _frozen()
    repository.create_dataset_for_lineage(
        "lineage-artifact", frozen.character_id, repository._draft_from(frozen)
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"canonical artifact")
    store = ArtifactStore(repository.database, tmp_path / "artifacts")
    staged = store.stage(source, "image/png")
    with repository.database.connect() as connection:
        artifact_count = connection.execute("SELECT count(*) FROM artifacts").fetchone()[0]

    def persist(connection: sqlite3.Connection, artifacts: tuple[object, ...]) -> None:
        artifact = artifacts[0]
        imported = _imported(
            lineage_id="lineage-artifact",
            character_id="character_1",
            artifact_id=artifact.id,  # type: ignore[attr-defined]
            artifact_digest="sha256:" + artifact.sha256,  # type: ignore[attr-defined]
            artifact_byte_size=artifact.byte_size + 1,  # type: ignore[attr-defined]
            media=TrainingMedia(
                sha256="sha256:" + artifact.sha256,  # type: ignore[attr-defined]
                media_type="image/png",
                byte_size=artifact.byte_size + 1,  # type: ignore[attr-defined]
                width=4,
                height=3,
            ),
        )
        repository.persist_imported_sample_with_connection(connection, imported)

    with pytest.raises(TrainingConflict):
        store.commit_batch((staged,), persist)
    with repository.database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == artifact_count
        assert (
            connection.execute("SELECT count(*) FROM training_imported_samples").fetchone()[0] == 0
        )


def test_partial_dataset_construction_cannot_commit_or_poison_reopen(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    draft = repository._draft_from(frozen)
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_datasets(storage_id,id,reviewer_id,character_id,"
                "character_pack_storage_id,character_pack_digest,state,canonical_outfit_id,"
                "reserved_token,source_tier_weights_json,manifest_json,dataset_digest,created_at,"
                "sealed_at,lineage_id) VALUES (?,?,?,?,?,?,'building',?,?,?,?,NULL,?,NULL,NULL)",
                (
                    "partial-storage",
                    frozen.dataset_digest,
                    "owner",
                    frozen.character_id,
                    "pack-storage",
                    PACK_DIGEST,
                    draft.canonical_outfit_id,
                    draft.reserved_token,
                    canonical_json(
                        {key.value: value for key, value in draft.source_tier_weights.items()}
                    ),
                    canonical_json(draft.model_dump(mode="json")),
                    NOW.isoformat(),
                ),
            )
    reopened = TrainingRepository(Database(repository.database.path), reviewer_id="owner")
    assert reopened.list_datasets() == ()
    created = reopened.create_dataset(frozen.character_id, frozen)
    assert reopened.get_dataset(created.dataset_id) == created


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("character_id", "other"),
        ("character_pack_digest", "sha256:" + "e" * 64),
        ("canonical_outfit_id", "other-outfit"),
        ("reserved_token", "<other>"),
        ("source_tier_weights_json", "{}"),
        ("id", "sha256:" + "e" * 64),
    ),
)
def test_dataset_parent_metadata_is_bound_to_manifest_by_direct_sql(
    repository: TrainingRepository, column: str, value: str
) -> None:
    frozen = _frozen()
    draft = repository._draft_from(frozen)
    values = {
        "storage_id": "forged-parent",
        "id": frozen.dataset_digest,
        "reviewer_id": "owner",
        "character_id": frozen.character_id,
        "character_pack_storage_id": "pack-storage",
        "character_pack_digest": PACK_DIGEST,
        "canonical_outfit_id": draft.canonical_outfit_id,
        "reserved_token": draft.reserved_token,
        "source_tier_weights_json": canonical_json(
            {key.value: weight for key, weight in draft.source_tier_weights.items()}
        ),
    }
    values[column] = value
    with repository.database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO training_datasets(storage_id,id,reviewer_id,character_id,"
            "character_pack_storage_id,character_pack_digest,state,canonical_outfit_id,"
            "reserved_token,source_tier_weights_json,manifest_json,dataset_digest,created_at,"
            "sealed_at) VALUES (?,?,?,?,?,?,'draft',?,?,?,?,NULL,?,NULL)",
            (
                values["storage_id"],
                values["id"],
                values["reviewer_id"],
                values["character_id"],
                values["character_pack_storage_id"],
                values["character_pack_digest"],
                values["canonical_outfit_id"],
                values["reserved_token"],
                values["source_tier_weights_json"],
                canonical_json(draft.model_dump(mode="json")),
                NOW.isoformat(),
            ),
        )


def test_replace_draft_sample_is_atomic_and_rekeys_alias(repository: TrainingRepository) -> None:
    frozen = _frozen()
    created = repository.create_dataset(frozen.character_id, frozen)
    replacement = frozen.samples[0].model_copy(
        update={"caption": "<mirza> wearing the canonical jacket, replacement"}
    )
    changed = repository.replace_draft_sample(created.dataset_id, replacement)
    assert changed.dataset_id != created.dataset_id
    assert changed.storage_id != created.storage_id
    assert repository.get_dataset(created.dataset_id).dataset_id == created.dataset_id
    assert repository.get_dataset(created.dataset_id).draft == created.draft
    replaced = next(
        item for item in changed.draft.samples if item.sample_id == replacement.sample_id
    )
    assert replaced.caption.endswith("replacement")


def test_sealed_dataset_and_children_are_immutable_by_direct_sql(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    record = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(record.dataset_id, frozen)
    with repository.database.connect() as connection:
        for sql in (
            "UPDATE training_datasets SET reserved_token='<x>' WHERE storage_id=?",
            "DELETE FROM training_datasets WHERE storage_id=?",
            "UPDATE training_samples SET caption='forged' WHERE dataset_storage_id=?",
            "DELETE FROM training_samples WHERE dataset_storage_id=?",
            "UPDATE training_sample_annotations SET annotation_json='{}' "
            "WHERE dataset_storage_id=?",
            "DELETE FROM training_sample_annotations WHERE dataset_storage_id=?",
            "UPDATE training_dataset_aliases SET alias_id='forged' WHERE dataset_storage_id=?",
            "DELETE FROM training_dataset_aliases WHERE dataset_storage_id=?",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(sql, (record.storage_id,))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_dataset_aliases VALUES ('late-alias',?)",
                (record.storage_id,),
            )


def test_dataset_requires_owned_preregistered_pack_and_draft_sql_cannot_rekey(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    wrong = frozen.model_copy(update={"character_pack_digest": "sha256:" + "f" * 64})
    with pytest.raises(TrainingConflict, match="pack provenance"):
        repository.create_dataset(wrong.character_id, wrong)
    record = repository.create_dataset(frozen.character_id, frozen)
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="snapshots"):
            connection.execute(
                "UPDATE training_datasets SET id='forged',manifest_json='{}' WHERE storage_id=?",
                (record.storage_id,),
            )
        sample = frozen.samples[0]
        mutations = (
            (
                "INSERT INTO training_samples VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record.storage_id,
                    "extra",
                    sample.source_tier.value,
                    sample.sample_digest,
                    sample.source_lineage,
                    sample.caption,
                    sample.split.value,
                    canonical_json(sample.media.model_dump(mode="json")),
                    1,
                    99,
                ),
            ),
            (
                "INSERT INTO training_sample_annotations VALUES (?,?,?)",
                (record.storage_id, sample.sample_id, "{}"),
            ),
            (
                "INSERT INTO training_dataset_aliases VALUES (?,?)",
                ("late", record.storage_id),
            ),
            (
                "UPDATE training_samples SET caption='x' WHERE dataset_storage_id=?",
                (record.storage_id,),
            ),
            (
                "DELETE FROM training_samples WHERE dataset_storage_id=?",
                (record.storage_id,),
            ),
            (
                "UPDATE training_sample_annotations SET annotation_json='{}' "
                "WHERE dataset_storage_id=?",
                (record.storage_id,),
            ),
            (
                "DELETE FROM training_sample_annotations WHERE dataset_storage_id=?",
                (record.storage_id,),
            ),
            (
                "DELETE FROM training_dataset_aliases WHERE dataset_storage_id=?",
                (record.storage_id,),
            ),
            (
                "DELETE FROM training_datasets WHERE storage_id=?",
                (record.storage_id,),
            ),
        )
        for sql, parameters in mutations:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(sql, parameters)


def test_recipe_rejects_forged_digest_and_draft_dataset(repository: TrainingRepository) -> None:
    frozen = _frozen()
    repository.create_dataset(frozen.character_id, frozen)
    with pytest.raises(TrainingConflict, match="sealed dataset"):
        repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    record = repository.get_dataset(frozen.dataset_digest)
    repository.seal_dataset(record.dataset_id, frozen)
    forged = _recipe("sha256:" + "0" * 64)
    with pytest.raises(TrainingConflict, match="dataset identity"):
        repository.create_recipe(forged, _binding())
    recipe = _valid_recipe(frozen.dataset_digest)
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_recipes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "forged-storage",
                    "sha256:" + "f" * 64,
                    "owner",
                    record.storage_id,
                    frozen.dataset_digest,
                    frozen.character_id,
                    frozen.character_pack_digest,
                    "draft",
                    canonical_json(recipe.model_dump(mode="json")),
                    "2026-01-01T00:00:00+00:00",
                    None,
                ),
            )


@pytest.mark.parametrize(
    "field",
    (
        "character_pack_digest",
        "stage_2_manifest_digest",
        "stage_2_evidence_digest",
        "stage_3_evidence_digest",
    ),
)
def test_recipe_requires_exact_terminal_provenance(
    repository: TrainingRepository, field: str
) -> None:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    recipe = _valid_recipe(frozen.dataset_digest).model_copy(update={field: "sha256:" + "e" * 64})
    with pytest.raises(TrainingConflict, match="provenance|identity"):
        repository.create_recipe(recipe, _binding())


def test_recipe_rejects_forged_canonical_stage3_definition_and_result(
    repository: TrainingRepository,
) -> None:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    forged_result = {"benchmark_id": "character_benchmark_forged"}
    forged_digest = "sha256:" + hashlib.sha256(canonical_json(forged_result).encode()).hexdigest()
    with repository.database.connect() as connection:
        stage_2 = connection.execute(
            "SELECT stage_2_campaign_id,stage_2_manifest_json FROM character_benchmark_campaigns "
            "WHERE id='character_benchmark_training'"
        ).fetchone()
        assert stage_2 is not None
        forged_definition = {
            "id": "character_benchmark_forged",
            "character_pack_digest": PACK_DIGEST,
            "stage_2_manifest_digest": _valid_recipe(frozen.dataset_digest).stage_2_manifest_digest,
            "stage_2_evidence_digest": "sha256:" + "e" * 64,
        }
        forged_campaign = {
            "definition": forged_definition,
            "character_pack_id": "character_pack_" + "6" * 64,
            "character_pack_storage_id": "pack-storage",
            "stage_2_campaign_id": stage_2["stage_2_campaign_id"],
            "stage_2_manifest": json.loads(str(stage_2["stage_2_manifest_json"])),
            "identity_prompt": "identity",
        }
        forged_campaign_digest = (
            "sha256:" + hashlib.sha256(canonical_json(forged_campaign).encode()).hexdigest()
        )
        connection.execute(
            "INSERT INTO character_benchmark_campaigns VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "character_benchmark_forged",
                "owner",
                forged_campaign_digest,
                canonical_json(forged_definition),
                "character_pack_" + "6" * 64,
                "pack-storage",
                stage_2["stage_2_campaign_id"],
                stage_2["stage_2_manifest_json"],
                "identity",
                NOW.isoformat(),
            ),
        )
        for sequence, (previous, new) in enumerate(
            (
                ("draft", "preregistered"),
                ("preregistered", "generating"),
                ("generating", "ready"),
                ("ready", "judging"),
                ("judging", "analyzing"),
            ),
            start=1,
        ):
            connection.execute(
                "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
                (
                    "character_benchmark_forged",
                    sequence,
                    previous,
                    new,
                    "sha256:" + f"{sequence}" * 64,
                    NOW.isoformat(),
                ),
            )
        connection.execute(
            "INSERT INTO character_benchmark_results VALUES (?,?,?,?)",
            (
                "character_benchmark_forged",
                canonical_json(forged_result),
                forged_digest,
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
            (
                "character_benchmark_forged",
                6,
                "analyzing",
                "completed",
                "sha256:" + "6" * 64,
                NOW.isoformat(),
            ),
        )
    recipe = _valid_recipe(frozen.dataset_digest).model_copy(
        update={"stage_3_evidence_digest": forged_digest}
    )
    with pytest.raises(TrainingConflict, match="provenance"):
        repository.create_recipe(recipe, _binding())
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO training_recipes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "forged-recipe-storage",
                    training_recipe_id(recipe),
                    "owner",
                    dataset.storage_id,
                    frozen.dataset_digest,
                    frozen.character_id,
                    PACK_DIGEST,
                    "draft",
                    canonical_json(recipe.model_dump(mode="json")),
                    NOW.isoformat(),
                    None,
                ),
            )


def test_direct_sql_cannot_forge_sealed_dataset_identity(repository: TrainingRepository) -> None:
    frozen = _frozen()
    record = repository.create_dataset(frozen.character_id, frozen)
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE training_datasets SET state='sealed',dataset_digest=?,manifest_json=?,"
                "sealed_at=? WHERE storage_id=?",
                (
                    "sha256:" + "0" * 64,
                    canonical_json(frozen.model_dump(mode="json")),
                    "2026-01-01T00:00:00+00:00",
                    record.storage_id,
                ),
            )


@pytest.mark.parametrize("terminal", ("qualified", "rejected"))
def test_scientific_run_lifecycle_probe_calibration_cas_and_atomic_rollback(
    repository: TrainingRepository, terminal: str
) -> None:
    frozen = _frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    recipe = repository.create_recipe(_valid_recipe(frozen.dataset_digest), _binding())
    repository.preregister_recipe(recipe.recipe_id)
    run = TrainingRun(
        run_id="scientific-run",
        recipe_id=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        total_steps=recipe.recipe.total_steps,
    )
    repository.create_run(run)
    with repository.database.connect() as connection:
        identity = connection.execute(
            "SELECT canonical_json_digest(json_extract(recipe.recipe_json,'$.backend_config')) "
            "AS backend_digest,materials.runtime_digest,'sha256:'||source.sha256 AS source_digest "
            "FROM training_recipes recipe JOIN training_recipe_materials materials "
            "ON materials.recipe_storage_id=recipe.storage_id JOIN artifacts source "
            "ON source.id=materials.source_snapshot_artifact_id WHERE recipe.id=?",
            (recipe.recipe_id,),
        ).fetchone()
    assert identity is not None
    material_digest = recipe_material_digest(_binding())
    probe = TrainingProbeEvidence(
        evidence_id="probe-1",
        run_id=run.run_id,
        recipe_digest=recipe.recipe_id,
        dataset_digest=frozen.dataset_digest,
        backend_digest=str(identity["backend_digest"]),
        runtime_digest=str(identity["runtime_digest"]),
        material_digest=material_digest,
        source_digest=str(identity["source_digest"]),
        capability_digest="sha256:" + "8" * 64,
        succeeded=True,
    )
    with pytest.raises(TrainingConflict, match="probe"):
        repository.record_probe(probe.model_copy(update={"material_digest": "sha256:" + "0" * 64}))
    repository.atomic(
        lambda connection: (
            repository.record_probe_with_connection(connection, probe),
            repository.append_event_with_connection(
                connection,
                run.run_id,
                TrainingRunEvent(
                    sequence=1, previous_state="draft", new_state="probed", payload={}
                ),
                expected_version=0,
            ),
        )
    )
    calibration = TrainingCalibrationEvidence(
        evidence_id="calibration-1",
        run_id=run.run_id,
        probe_evidence_id=probe.evidence_id,
        calibration_digest="sha256:" + "9" * 64,
        recipe_digest=probe.recipe_digest,
        dataset_digest=probe.dataset_digest,
        backend_digest=probe.backend_digest,
        runtime_digest=probe.runtime_digest,
        material_digest=probe.material_digest,
        source_digest=probe.source_digest,
        succeeded=True,
    )
    repository.record_calibration(calibration)
    repository.append_event(
        run.run_id,
        TrainingRunEvent(sequence=2, previous_state="probed", new_state="calibrated", payload={}),
        expected_version=1,
    )
    with pytest.raises(TrainingConflict, match="conflict"):
        repository.append_event(
            run.run_id,
            TrainingRunEvent(
                sequence=3,
                previous_state="calibrated",
                new_state="preregistered",
                payload={
                    "idempotency_key": "request",
                    "request_digest": "sha256:" + "a" * 64,
                    "probe_evidence_id": probe.evidence_id,
                    "calibration_evidence_id": calibration.evidence_id,
                },
            ),
            expected_version=1,
        )
    repository.atomic(
        lambda connection: repository.bind_idempotency_with_connection(
            connection,
            run_id=run.run_id,
            idempotency_key="request",
            request_digest="sha256:" + "a" * 64,
            probe_evidence_id=probe.evidence_id,
            calibration_evidence_id=calibration.evidence_id,
        )
    )
    before = len(repository.list_events(run.run_id))
    with pytest.raises(RuntimeError, match="rollback"):

        def fail(connection: sqlite3.Connection) -> None:
            repository.append_event_with_connection(
                connection,
                run.run_id,
                TrainingRunEvent(
                    sequence=3,
                    previous_state="calibrated",
                    new_state="preregistered",
                    payload={
                        "idempotency_key": "request",
                        "request_digest": "sha256:" + "a" * 64,
                        "probe_evidence_id": probe.evidence_id,
                        "calibration_evidence_id": calibration.evidence_id,
                    },
                ),
                expected_version=2,
            )
            raise RuntimeError("rollback")

        repository.atomic(fail)
    assert len(repository.list_events(run.run_id)) == before
    assert repository.get_run(run.run_id).state == "calibrated"
    repository.atomic(
        lambda connection: repository.bind_idempotency_with_connection(
            connection,
            run_id=run.run_id,
            idempotency_key="request",
            request_digest="sha256:" + "a" * 64,
            probe_evidence_id=probe.evidence_id,
            calibration_evidence_id=calibration.evidence_id,
        )
    )
    preregistration_payload = {
        "idempotency_key": "request",
        "request_digest": "sha256:" + "a" * 64,
        "probe_evidence_id": probe.evidence_id,
        "calibration_evidence_id": calibration.evidence_id,
    }
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=3,
            previous_state="calibrated",
            new_state="preregistered",
            payload=preregistration_payload,
        ),
        expected_version=2,
    )
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=4, previous_state="preregistered", new_state="training", payload={}
        ),
        expected_version=3,
    )
    parent_id: str | None = None
    parent_digest: str | None = None
    checkpoints: list[tuple[TrainingCheckpoint, TrainingCheckpointEvidence]] = []
    for ordinal, step in enumerate(checkpoint_steps(run.total_steps)):
        repository.append_telemetry(
            run.run_id,
            TrainingTelemetry(sequence=ordinal + 1, step=step, payload={"loss": 1 / (ordinal + 1)}),
        )
        checkpoint, evidence = _scientific_checkpoint(
            run,
            recipe.recipe_id,
            probe,
            ordinal=ordinal,
            parent_id=parent_id,
            parent_digest=parent_digest,
        )
        with repository.database.connect() as connection:
            telemetry_digest = repository.telemetry_digest_with_connection(
                connection, run.run_id, 1, ordinal + 1
            )
        updated_inventory = tuple(
            item.model_copy(update={"digest": telemetry_digest})
            if item.role == "telemetry"
            else item
            for item in evidence.inventory
        )
        evidence = TrainingCheckpointEvidence.model_validate(
            {
                **evidence.model_dump(mode="python"),
                "telemetry_digest": telemetry_digest,
                "inventory": updated_inventory,
            }
        )
        manifest = dict(checkpoint.manifest)
        manifest["telemetry_digest"] = telemetry_digest
        manifest["inventory"] = [item.model_dump(mode="json") for item in updated_inventory]
        checkpoint = TrainingCheckpoint.model_validate(
            {
                **checkpoint.model_dump(mode="python"),
                "manifest": manifest,
                "manifest_digest": "sha256:"
                + hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
            }
        )
        if ordinal > 0:
            bad_parents: list[tuple[str | None, str | None]] = [
                (None, None),
                (checkpoint.checkpoint_id, "sha256:" + "0" * 64),
                ("future-checkpoint", "sha256:" + "0" * 64),
            ]
            if ordinal > 1:
                bad_parents.append(
                    (
                        checkpoints[0][0].checkpoint_id,
                        "sha256:"
                        + hashlib.sha256(
                            canonical_json(checkpoints[0][1].model_dump(mode="json")).encode()
                        ).hexdigest(),
                    )
                )
            for bad_parent_id, bad_parent_digest in bad_parents:
                bad_evidence = evidence.model_copy(
                    update={"parent_checkpoint_digest": bad_parent_digest}
                )
                bad_manifest = dict(checkpoint.model_dump(mode="json")["manifest"])
                bad_manifest["parent_checkpoint_id"] = bad_parent_id
                bad_checkpoint = TrainingCheckpoint.model_validate(
                    {
                        **checkpoint.model_dump(mode="python"),
                        "checkpoint_id": (
                            f"{checkpoint.checkpoint_id}-bad-{len(bad_parent_id or '')}"
                        ),
                        "parent_checkpoint_id": bad_parent_id,
                        "manifest": bad_manifest,
                        "manifest_digest": "sha256:"
                        + hashlib.sha256(canonical_json(bad_manifest).encode()).hexdigest(),
                    }
                )
                with pytest.raises(sqlite3.IntegrityError, match="parent chain"):

                    def publish_bad_parent(
                        connection: sqlite3.Connection,
                        checkpoint_to_publish: TrainingCheckpoint = bad_checkpoint,
                        evidence_to_publish: TrainingCheckpointEvidence = bad_evidence,
                    ) -> None:
                        repository.add_checkpoint_with_connection(
                            connection, checkpoint_to_publish, evidence_to_publish
                        )

                    repository.atomic(publish_bad_parent)
        if ordinal == 1:
            wrong = evidence.model_copy(update={"runtime_digest": "sha256:" + "0" * 64})
            with pytest.raises(sqlite3.IntegrityError, match="canonical"):
                repository.atomic(
                    lambda connection, checkpoint=checkpoint, wrong=wrong: (
                        repository.add_checkpoint_with_connection(connection, checkpoint, wrong)
                    )
                )
            with pytest.raises(ValueError, match="manifest identity"):
                checkpoint.model_copy(update={"parent_checkpoint_id": None})
        repository.atomic(
            lambda connection, checkpoint=checkpoint, evidence=evidence: (
                repository.add_checkpoint_with_connection(connection, checkpoint, evidence)
            )
        )
        checkpoints.append((checkpoint, evidence))
        parent_id = checkpoint.checkpoint_id
        evidence_json = canonical_json(evidence.model_dump(mode="json"))
        parent_digest = "sha256:" + hashlib.sha256(evidence_json.encode()).hexdigest()
        if ordinal == 0:
            resume = TrainingResumeEvidence(
                resume_id="resume-1",
                run_id=run.run_id,
                parent_checkpoint_id=checkpoint.checkpoint_id,
                parent_checkpoint_digest=parent_digest,
                parent_artifact_digest=checkpoint.artifact_digest,
                recipe_digest=recipe.recipe_id,
                dataset_digest=run.dataset_digest,
                backend_digest=probe.backend_digest,
                runtime_digest=probe.runtime_digest,
                material_digest=probe.material_digest,
                source_digest=probe.source_digest,
                optimizer_state_digest=evidence.optimizer_state_digest,
            )
            with pytest.raises(TrainingConflict, match="resume"):
                repository.record_resume(
                    resume.model_copy(update={"source_digest": "sha256:" + "0" * 64})
                )
            repository.record_resume(resume)
    with repository.database.connect() as connection:
        finalized_id = checkpoints[0][0].checkpoint_id
        for sql in (
            "UPDATE training_checkpoint_inventory SET size_bytes=size_bytes+1 "
            "WHERE checkpoint_id=? AND ordinal=0",
            "DELETE FROM training_checkpoint_inventory WHERE checkpoint_id=? AND ordinal=0",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute(sql, (finalized_id,))
        with pytest.raises(sqlite3.IntegrityError, match="derived"):
            connection.execute(
                "INSERT INTO training_checkpoint_inventory VALUES (?,?,?,?,?,?)",
                (finalized_id, 7, "extra.bin", "sha256:" + "e" * 64, 1, "weights"),
            )
        assert (
            connection.execute(
                "SELECT count(*) FROM training_checkpoint_inventory WHERE checkpoint_id=?",
                (finalized_id,),
            ).fetchone()[0]
            == 7
        )
    with repository.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="change only through"):
            connection.execute(
                "UPDATE training_runs SET lifecycle_state='qualified' WHERE id=?", (run.run_id,)
            )
    repository.append_event(
        run.run_id,
        TrainingRunEvent(sequence=5, previous_state="training", new_state="evaluating", payload={}),
        expected_version=4,
    )
    evaluation = TrainingEvaluationEvidence(
        evidence_id="evaluation-1",
        run_id=run.run_id,
        disposition=terminal,  # type: ignore[arg-type]
        evidence_digest="sha256:" + "f" * 64,
    )
    repository.record_evaluation(evaluation)
    repository.append_event(
        run.run_id,
        TrainingRunEvent(
            sequence=6,
            previous_state="evaluating",
            new_state=terminal,  # type: ignore[arg-type]
            payload={"evaluation_evidence_digest": evaluation.evidence_digest},
        ),
        expected_version=5,
    )
    assert repository.get_run(run.run_id).state == terminal
    assert (
        repository.atomic(
            lambda connection: repository.bind_idempotency_with_connection(
                connection,
                run_id=run.run_id,
                idempotency_key="request",
                request_digest="sha256:" + "a" * 64,
                probe_evidence_id=probe.evidence_id,
                calibration_evidence_id=calibration.evidence_id,
            )
        )
        == run.run_id
    )
    with pytest.raises(TrainingConflict, match="idempotency"):
        repository.atomic(
            lambda connection: repository.bind_idempotency_with_connection(
                connection,
                run_id=run.run_id,
                idempotency_key="request",
                request_digest="sha256:" + "0" * 64,
                probe_evidence_id=probe.evidence_id,
                calibration_evidence_id=calibration.evidence_id,
            )
        )
    with pytest.raises(TrainingConflict, match="failure"):
        repository.record_failure(
            TrainingFailure(failure_id="late", run_id=run.run_id, code="late", message="safe")
        )
    cancel_run = run.model_copy(update={"run_id": "cancel-run"})
    repository.create_run(cancel_run)
    cancel_probe = probe.model_copy(
        update={"evidence_id": "probe-cancel", "run_id": cancel_run.run_id}
    )
    repository.atomic(
        lambda connection: (
            repository.record_probe_with_connection(connection, cancel_probe),
            repository.append_event_with_connection(
                connection,
                cancel_run.run_id,
                TrainingRunEvent(
                    sequence=1, previous_state="draft", new_state="probed", payload={}
                ),
                expected_version=0,
            ),
        )
    )
    cancel_calibration = calibration.model_copy(
        update={
            "evidence_id": "calibration-cancel",
            "run_id": cancel_run.run_id,
            "probe_evidence_id": cancel_probe.evidence_id,
        }
    )
    repository.record_calibration(cancel_calibration)
    repository.append_event(
        cancel_run.run_id,
        TrainingRunEvent(sequence=2, previous_state="probed", new_state="calibrated", payload={}),
        expected_version=1,
    )
    repository.atomic(
        lambda connection: repository.bind_idempotency_with_connection(
            connection,
            run_id=cancel_run.run_id,
            idempotency_key="cancel-request",
            request_digest="sha256:" + "4" * 64,
            probe_evidence_id=cancel_probe.evidence_id,
            calibration_evidence_id=cancel_calibration.evidence_id,
        )
    )
    repository.append_event(
        cancel_run.run_id,
        TrainingRunEvent(
            sequence=3,
            previous_state="calibrated",
            new_state="preregistered",
            payload={
                "idempotency_key": "cancel-request",
                "request_digest": "sha256:" + "4" * 64,
                "probe_evidence_id": cancel_probe.evidence_id,
                "calibration_evidence_id": cancel_calibration.evidence_id,
            },
        ),
        expected_version=2,
    )
    repository.append_event(
        cancel_run.run_id,
        TrainingRunEvent(
            sequence=4, previous_state="preregistered", new_state="training", payload={}
        ),
        expected_version=3,
    )
    cancel_step = checkpoint_steps(cancel_run.total_steps)[0]
    repository.append_telemetry(
        cancel_run.run_id, TrainingTelemetry(sequence=1, step=cancel_step, payload={"loss": 1.0})
    )
    cancel_checkpoint, cancel_checkpoint_evidence = _scientific_checkpoint(
        cancel_run,
        recipe.recipe_id,
        cancel_probe,
        ordinal=0,
        parent_id=None,
        parent_digest=None,
        cancellation_boundary=True,
    )
    with repository.database.connect() as connection:
        cancel_telemetry_digest = repository.telemetry_digest_with_connection(
            connection, cancel_run.run_id, 1, 1
        )
    cancel_inventory = tuple(
        item.model_copy(update={"digest": cancel_telemetry_digest})
        if item.role == "telemetry"
        else item
        for item in cancel_checkpoint_evidence.inventory
    )
    cancel_checkpoint_evidence = TrainingCheckpointEvidence.model_validate(
        {
            **cancel_checkpoint_evidence.model_dump(mode="python"),
            "telemetry_digest": cancel_telemetry_digest,
            "inventory": cancel_inventory,
        }
    )
    cancel_manifest = dict(cancel_checkpoint.manifest)
    cancel_manifest["telemetry_digest"] = cancel_telemetry_digest
    cancel_manifest["inventory"] = [item.model_dump(mode="json") for item in cancel_inventory]
    cancel_checkpoint = TrainingCheckpoint.model_validate(
        {
            **cancel_checkpoint.model_dump(mode="python"),
            "manifest": cancel_manifest,
            "manifest_digest": "sha256:"
            + hashlib.sha256(canonical_json(cancel_manifest).encode()).hexdigest(),
        }
    )
    repository.atomic(
        lambda connection: repository.add_checkpoint_with_connection(
            connection, cancel_checkpoint, cancel_checkpoint_evidence
        )
    )
    cancellation = TrainingCancellationEvidence(
        cancellation_id="cancellation-1",
        run_id=cancel_run.run_id,
        cancellation_request_id="cancel-request-1",
        request_digest="sha256:" + "6" * 64,
        requested_step=cancel_step,
        boundary_step=cancel_step,
        checkpoint_digest=cancel_checkpoint.artifact_digest,
    )
    repository.atomic(
        lambda connection: repository.request_cancellation_with_connection(
            connection,
            request_id="cancel-request-1",
            run_id=cancel_run.run_id,
            requested_step=cancel_step,
            request_digest="sha256:" + "6" * 64,
        )
    )
    with pytest.raises(TrainingConflict, match="cancellation"):
        repository.record_cancellation(
            cancellation.model_copy(update={"checkpoint_digest": "sha256:" + "0" * 64})
        )
    for update in (
        {"cancellation_request_id": "wrong-request"},
        {"request_digest": "sha256:" + "0" * 64},
    ):
        with pytest.raises(TrainingConflict, match="cancellation"):
            repository.record_cancellation(cancellation.model_copy(update=update))
    repository.record_cancellation(cancellation)
    cancellation_payload = {
        "cancellation_id": cancellation.cancellation_id,
        "cancellation_request_id": cancellation.cancellation_request_id,
        "request_digest": cancellation.request_digest,
    }
    with pytest.raises(TrainingConflict):
        repository.append_event(
            cancel_run.run_id,
            TrainingRunEvent(
                sequence=5,
                previous_state="training",
                new_state="cancelled",
                payload={**cancellation_payload, "request_digest": "sha256:" + "0" * 64},
            ),
            expected_version=4,
        )
    repository.append_event(
        cancel_run.run_id,
        TrainingRunEvent(
            sequence=5,
            previous_state="training",
            new_state="cancelled",
            payload=cancellation_payload,
        ),
        expected_version=4,
    )
    assert repository.get_run(cancel_run.run_id).state == "cancelled"
    failed_run = run.model_copy(update={"run_id": "failed-run"})
    repository.create_run(failed_run)
    failure = TrainingFailure(
        failure_id="failure-1", run_id=failed_run.run_id, code="backend", message="safe failure"
    )
    repository.atomic(
        lambda connection: (
            repository.record_failure_with_connection(connection, failure),
            repository.append_event_with_connection(
                connection,
                failed_run.run_id,
                TrainingRunEvent(
                    sequence=1, previous_state="draft", new_state="failed", payload={}
                ),
                expected_version=0,
            ),
        )
    )
    assert repository.get_run(failed_run.run_id).state == "failed"


@pytest.mark.parametrize("same_digest", (True, False), ids=("replay", "conflict"))
def test_idempotency_binding_is_atomic_under_two_connection_thread_race(
    repository: TrainingRepository, same_digest: bool
) -> None:
    run, _, probe, calibration = _prepare_calibrated_scientific_run(repository, "race-run")
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, str]] = []
    lock = threading.Lock()

    def compete(label: str) -> None:
        digest = "sha256:" + ("a" if same_digest or label == "a" else "b") * 64
        local = TrainingRepository(repository.database, reviewer_id="owner")
        barrier.wait(timeout=5)
        try:
            reopened = local.atomic(
                lambda connection: local.bind_idempotency_with_connection(
                    connection,
                    run_id=run.run_id,
                    idempotency_key="shared-request",
                    request_digest=digest,
                    probe_evidence_id=probe.evidence_id,
                    calibration_evidence_id=calibration.evidence_id,
                )
            )
            outcome = ("reopened", str(reopened))
        except TrainingConflict as error:
            outcome = ("conflict", str(error))
        with lock:
            outcomes.append(outcome)

    threads = tuple(threading.Thread(target=compete, args=(label,)) for label in ("a", "b"))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    if same_digest:
        assert sorted(outcomes) == [("reopened", run.run_id), ("reopened", run.run_id)]
    else:
        assert sorted(kind for kind, _ in outcomes) == ["conflict", "reopened"]
        assert next(value for kind, value in outcomes if kind == "reopened") == run.run_id
    with repository.database.connect() as connection:
        rows = connection.execute(
            "SELECT run_id,request_digest FROM training_run_idempotency "
            "WHERE reviewer_id='owner' AND idempotency_key='shared-request'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["run_id"] == run.run_id
    if not same_digest:
        assert rows[0]["request_digest"] in {"sha256:" + "a" * 64, "sha256:" + "b" * 64}
    assert repository.get_run(run.run_id).state == "calibrated"


@pytest.mark.parametrize(
    "corruption",
    (
        "valid",
        "missing",
        "updated",
        "extra",
        "wrong-role",
        "parent-chain",
        "older-parent",
        "skipped-schedule",
        "role-digest-swap",
        "wrong-weights-size",
    ),
)
def test_physical_v36_finalized_checkpoint_preflight_is_exact_and_atomic(
    repository: TrainingRepository, corruption: str
) -> None:
    _, _, checkpoints = _publish_scientific_checkpoint_chain(
        repository, "v36-chain-run", bind_artifacts=True
    )
    _downgrade_checkpoint_publication_to_physical_v36(repository)
    with repository.database.connect() as connection:
        first_id = checkpoints[0][0].checkpoint_id
        second_id = checkpoints[1][0].checkpoint_id
        if corruption == "missing":
            connection.execute(
                "DELETE FROM training_checkpoint_inventory WHERE checkpoint_id=? AND ordinal=6",
                (first_id,),
            )
        elif corruption == "updated":
            connection.execute(
                "UPDATE training_checkpoint_inventory SET size_bytes=size_bytes+1 "
                "WHERE checkpoint_id=? AND ordinal=0",
                (first_id,),
            )
        elif corruption == "extra":
            connection.execute("DROP TRIGGER validate_training_checkpoint_inventory_insert_v36")
            connection.execute(
                "INSERT INTO training_checkpoint_inventory VALUES (?,?,?,?,?,?)",
                (first_id, 7, "extra.bin", "sha256:" + "e" * 64, 1, "weights"),
            )
        elif corruption == "wrong-role":
            connection.execute(
                "UPDATE training_checkpoint_inventory SET role='weights' "
                "WHERE checkpoint_id=? AND ordinal=6",
                (first_id,),
            )
        elif corruption in {"role-digest-swap", "wrong-weights-size"}:
            connection.execute("DROP TRIGGER guard_finalized_training_checkpoint_v36")
            row = connection.execute(
                "SELECT manifest_json,evidence_json FROM training_checkpoints WHERE id=?",
                (first_id,),
            ).fetchone()
            assert row is not None
            manifest = json.loads(str(row["manifest_json"]))
            evidence = json.loads(str(row["evidence_json"]))
            if corruption == "role-digest-swap":
                for document in (manifest, evidence):
                    document["inventory"][0]["digest"], document["inventory"][1]["digest"] = (
                        document["inventory"][1]["digest"],
                        document["inventory"][0]["digest"],
                    )
                connection.execute(
                    "UPDATE training_checkpoint_inventory SET artifact_digest=CASE ordinal "
                    "WHEN 0 THEN ? WHEN 1 THEN ? ELSE artifact_digest END "
                    "WHERE checkpoint_id=?",
                    (
                        evidence["inventory"][0]["digest"],
                        evidence["inventory"][1]["digest"],
                        first_id,
                    ),
                )
            else:
                manifest["inventory"][0]["size_bytes"] += 1
                evidence["inventory"][0]["size_bytes"] += 1
                connection.execute(
                    "UPDATE training_checkpoint_inventory SET size_bytes=size_bytes+1 "
                    "WHERE checkpoint_id=? AND ordinal=0",
                    (first_id,),
                )
            manifest_json = canonical_json(manifest)
            evidence_json = canonical_json(evidence)
            connection.execute(
                "UPDATE training_checkpoints SET manifest_json=?,manifest_digest=?,"
                "evidence_json=?,evidence_digest=? WHERE id=?",
                (
                    manifest_json,
                    "sha256:" + hashlib.sha256(manifest_json.encode()).hexdigest(),
                    evidence_json,
                    "sha256:" + hashlib.sha256(evidence_json.encode()).hexdigest(),
                    first_id,
                ),
            )
        elif corruption in {"parent-chain", "older-parent", "skipped-schedule"}:
            connection.execute("DROP TRIGGER guard_finalized_training_checkpoint_v36")
            target_id = (
                second_id if corruption == "parent-chain" else checkpoints[2][0].checkpoint_id
            )
            row = connection.execute(
                "SELECT manifest_json,evidence_json FROM training_checkpoints WHERE id=?",
                (target_id,),
            ).fetchone()
            assert row is not None
            manifest = json.loads(str(row["manifest_json"]))
            evidence = json.loads(str(row["evidence_json"]))
            parent_id = None if corruption == "parent-chain" else first_id
            parent_digest = (
                None
                if corruption == "parent-chain"
                else "sha256:"
                + hashlib.sha256(
                    canonical_json(checkpoints[0][1].model_dump(mode="json")).encode()
                ).hexdigest()
            )
            manifest["parent_checkpoint_id"] = parent_id
            evidence["parent_checkpoint_digest"] = parent_digest
            manifest_json = canonical_json(manifest)
            evidence_json = canonical_json(evidence)
            connection.execute(
                "UPDATE training_checkpoints SET parent_checkpoint_id=?,manifest_json=?,"
                "manifest_digest=?,evidence_json=?,evidence_digest=? WHERE id=?",
                (
                    parent_id,
                    manifest_json,
                    "sha256:" + hashlib.sha256(manifest_json.encode()).hexdigest(),
                    evidence_json,
                    "sha256:" + hashlib.sha256(evidence_json.encode()).hexdigest(),
                    target_id,
                ),
            )
            if corruption == "skipped-schedule":
                third_digest = "sha256:" + hashlib.sha256(evidence_json.encode()).hexdigest()
                fourth_id = checkpoints[3][0].checkpoint_id
                fourth = connection.execute(
                    "SELECT evidence_json FROM training_checkpoints WHERE id=?", (fourth_id,)
                ).fetchone()
                assert fourth is not None
                fourth_evidence = json.loads(str(fourth["evidence_json"]))
                fourth_evidence["parent_checkpoint_digest"] = third_digest
                fourth_evidence_json = canonical_json(fourth_evidence)
                connection.execute(
                    "UPDATE training_checkpoints SET evidence_json=?,evidence_digest=? WHERE id=?",
                    (
                        fourth_evidence_json,
                        "sha256:" + hashlib.sha256(fourth_evidence_json.encode()).hexdigest(),
                        fourth_id,
                    ),
                )
                connection.execute("DROP TRIGGER guard_training_checkpoint_delete")
                connection.execute(
                    "DROP TRIGGER IF EXISTS guard_training_checkpoint_inventory_delete"
                )
                connection.execute(
                    "DROP TRIGGER IF EXISTS guard_training_checkpoint_inventory_delete_v36"
                )
                connection.execute(
                    "DELETE FROM training_checkpoint_inventory WHERE checkpoint_id=?", (second_id,)
                )
                connection.execute("DELETE FROM training_checkpoints WHERE id=?", (second_id,))
        connection.commit()
    if corruption == "valid":
        repository.database.migrate()
        with repository.database.connect() as connection:
            assert (
                connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
                == max(_MIGRATIONS)
            )
            assert (
                connection.execute(
                    "SELECT count(*) FROM training_checkpoints WHERE run_id='v36-chain-run'"
                ).fetchone()[0]
                == 4
            )
        return
    with repository.database.connect() as connection:
        before_checkpoints = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM training_checkpoints ORDER BY id"
            ).fetchall()
        )
        before_inventory = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM training_checkpoint_inventory ORDER BY checkpoint_id,ordinal"
            ).fetchall()
        )
    with pytest.raises(sqlite3.IntegrityError, match="finalized checkpoints"):
        repository.database.migrate()
    with repository.database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 36
        assert (
            tuple(
                tuple(row)
                for row in connection.execute("SELECT * FROM training_checkpoints ORDER BY id")
            )
            == before_checkpoints
        )
        assert (
            tuple(
                tuple(row)
                for row in connection.execute(
                    "SELECT * FROM training_checkpoint_inventory ORDER BY checkpoint_id,ordinal"
                )
            )
            == before_inventory
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM pragma_table_info('training_cancellation_evidence') "
                "WHERE name='cancellation_request_id'"
            ).fetchone()[0]
            == 0
        )


def test_checkpoint_parent_cannot_cross_otherwise_valid_runs(
    repository: TrainingRepository,
) -> None:
    run_a, probe_a, run_a_checkpoints = _publish_scientific_checkpoint_chain(
        repository, "parent-run-a", count=1
    )
    run_b = run_a.model_copy(update={"run_id": "parent-run-b"})
    repository.create_run(run_b)
    probe_b = probe_a.model_copy(
        update={"evidence_id": "parent-run-b-probe", "run_id": run_b.run_id}
    )
    repository.atomic(
        lambda connection: (
            repository.record_probe_with_connection(connection, probe_b),
            repository.append_event_with_connection(
                connection,
                run_b.run_id,
                TrainingRunEvent(
                    sequence=1, previous_state="draft", new_state="probed", payload={}
                ),
                expected_version=0,
            ),
        )
    )
    calibration_b = TrainingCalibrationEvidence(
        evidence_id="parent-run-b-calibration",
        run_id=run_b.run_id,
        probe_evidence_id=probe_b.evidence_id,
        calibration_digest="sha256:" + "9" * 64,
        recipe_digest=probe_b.recipe_digest,
        dataset_digest=probe_b.dataset_digest,
        backend_digest=probe_b.backend_digest,
        runtime_digest=probe_b.runtime_digest,
        material_digest=probe_b.material_digest,
        source_digest=probe_b.source_digest,
        succeeded=True,
    )
    repository.atomic(
        lambda connection: (
            repository.record_calibration_with_connection(connection, calibration_b),
            repository.append_event_with_connection(
                connection,
                run_b.run_id,
                TrainingRunEvent(
                    sequence=2, previous_state="probed", new_state="calibrated", payload={}
                ),
                expected_version=1,
            ),
        )
    )
    _start_scientific_training(repository, run_b, probe_b, calibration_b)
    first_step = checkpoint_steps(run_b.total_steps)[0]
    repository.append_telemetry(
        run_b.run_id, TrainingTelemetry(sequence=1, step=first_step, payload={"loss": 1})
    )
    first_b, first_b_evidence = _scientific_checkpoint(
        run_b,
        run_b.recipe_id,
        probe_b,
        ordinal=0,
        parent_id=None,
        parent_digest=None,
    )
    with repository.database.connect() as connection:
        first_telemetry_digest = repository.telemetry_digest_with_connection(
            connection, run_b.run_id, 1, 1
        )
    first_inventory = tuple(
        item.model_copy(update={"digest": first_telemetry_digest})
        if item.role == "telemetry"
        else item
        for item in first_b_evidence.inventory
    )
    first_b_evidence = TrainingCheckpointEvidence.model_validate(
        {
            **first_b_evidence.model_dump(mode="python"),
            "telemetry_digest": first_telemetry_digest,
            "inventory": first_inventory,
        }
    )
    first_manifest = dict(first_b.model_dump(mode="json")["manifest"])
    first_manifest["telemetry_digest"] = first_telemetry_digest
    first_manifest["inventory"] = [item.model_dump(mode="json") for item in first_inventory]
    first_b = TrainingCheckpoint.model_validate(
        {
            **first_b.model_dump(mode="python"),
            "manifest": first_manifest,
            "manifest_digest": "sha256:"
            + hashlib.sha256(canonical_json(first_manifest).encode()).hexdigest(),
        }
    )
    repository.atomic(
        lambda connection: repository.add_checkpoint_with_connection(
            connection, first_b, first_b_evidence
        )
    )
    second_step = checkpoint_steps(run_b.total_steps)[1]
    repository.append_telemetry(
        run_b.run_id, TrainingTelemetry(sequence=2, step=second_step, payload={"loss": 2})
    )
    parent_a, parent_a_evidence = run_a_checkpoints[0]
    parent_a_digest = (
        "sha256:"
        + hashlib.sha256(
            canonical_json(parent_a_evidence.model_dump(mode="json")).encode()
        ).hexdigest()
    )
    checkpoint, evidence = _scientific_checkpoint(
        run_b,
        run_b.recipe_id,
        probe_b,
        ordinal=1,
        parent_id=parent_a.checkpoint_id,
        parent_digest=parent_a_digest,
    )
    with repository.database.connect() as connection:
        telemetry_digest = repository.telemetry_digest_with_connection(
            connection, run_b.run_id, 1, 2
        )
    inventory = tuple(
        item.model_copy(update={"digest": telemetry_digest}) if item.role == "telemetry" else item
        for item in evidence.inventory
    )
    evidence = TrainingCheckpointEvidence.model_validate(
        {
            **evidence.model_dump(mode="python"),
            "telemetry_digest": telemetry_digest,
            "inventory": inventory,
        }
    )
    manifest = dict(checkpoint.model_dump(mode="json")["manifest"])
    manifest["telemetry_digest"] = telemetry_digest
    manifest["inventory"] = [item.model_dump(mode="json") for item in inventory]
    checkpoint = TrainingCheckpoint.model_validate(
        {
            **checkpoint.model_dump(mode="python"),
            "manifest": manifest,
            "manifest_digest": "sha256:"
            + hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        }
    )
    with pytest.raises(sqlite3.IntegrityError, match="parent chain"):
        repository.atomic(
            lambda connection: repository.add_checkpoint_with_connection(
                connection, checkpoint, evidence
            )
        )
    assert len(repository.list_checkpoints(run_b.run_id)) == 1
    assert repository.list_checkpoints(run_b.run_id)[0].checkpoint == first_b


_CHECKPOINT_JSON_MISMATCHES = (
    "manifest:schema_version",
    "manifest:run_id",
    "manifest:reviewer_id",
    "manifest:recipe_id",
    "manifest:dataset_digest",
    "manifest:character_id",
    "manifest:backend_digest",
    "manifest:runtime_digest",
    "manifest:source_digest",
    "manifest:material_digest",
    "manifest:step",
    "manifest:fraction",
    "manifest:parent_checkpoint_id",
    "manifest:artifact_digest",
    "manifest:safetensors_size_bytes",
    "manifest:optimizer_state_digest",
    "manifest:trainer_state_digest",
    "manifest:telemetry_digest",
    "manifest:telemetry_sequence_start",
    "manifest:telemetry_sequence_end",
    "manifest:trainable_targets_digest",
    "manifest:frozen_parameters_digest",
    "manifest:finite_gradient_proof_digest",
    "manifest:cancellation_boundary",
    "manifest:inventory",
    "evidence:schema_version",
    "evidence:run_id",
    "evidence:recipe_digest",
    "evidence:dataset_digest",
    "evidence:backend_digest",
    "evidence:runtime_digest",
    "evidence:source_digest",
    "evidence:material_digest",
    "evidence:step",
    "evidence:fraction",
    "evidence:parent_checkpoint_digest",
    "evidence:safetensors_digest",
    "evidence:safetensors_size_bytes",
    "evidence:optimizer_state_digest",
    "evidence:trainer_state_digest",
    "evidence:telemetry_digest",
    "evidence:telemetry_sequence_start",
    "evidence:telemetry_sequence_end",
    "evidence:trainable_targets_digest",
    "evidence:frozen_parameters_digest",
    "evidence:finite_gradient_proof_digest",
    "evidence:cancellation_boundary",
    "evidence:inventory",
)

_CHECKPOINT_COLUMN_MISMATCHES = (
    "run_id",
    "reviewer_id",
    "recipe_storage_id",
    "dataset_storage_id",
    "step",
    "fraction",
    "parent_checkpoint_id",
    "artifact_digest",
    "manifest_digest",
    "evidence_digest",
)


@pytest.mark.parametrize("section", ("columns", "manifest", "evidence", "extras", "role_bindings"))
def test_direct_sql_rejects_every_checkpoint_manifest_and_evidence_mismatch(
    repository: TrainingRepository, section: str
) -> None:
    run, recipe_id, probe, calibration = _prepare_calibrated_scientific_run(
        repository, "direct-checkpoint-run"
    )
    _start_scientific_training(repository, run, probe, calibration)
    step = checkpoint_steps(run.total_steps)[0]
    repository.append_telemetry(
        run.run_id, TrainingTelemetry(sequence=1, step=step, payload={"loss": 1.0})
    )
    checkpoint, evidence = _scientific_checkpoint(
        run, recipe_id, probe, ordinal=0, parent_id=None, parent_digest=None
    )
    with repository.database.connect() as connection:
        telemetry_digest = repository.telemetry_digest_with_connection(connection, run.run_id, 1, 1)
        binding = connection.execute(
            "SELECT recipe_storage_id,dataset_storage_id FROM training_runs WHERE id=?",
            (run.run_id,),
        ).fetchone()
    assert binding is not None
    inventory = tuple(
        item.model_copy(update={"digest": telemetry_digest}) if item.role == "telemetry" else item
        for item in evidence.inventory
    )
    evidence_json = evidence.model_dump(mode="json")
    evidence_json["telemetry_digest"] = telemetry_digest
    evidence_json["inventory"] = [item.model_dump(mode="json") for item in inventory]
    manifest = dict(checkpoint.model_dump(mode="json")["manifest"])
    manifest["telemetry_digest"] = telemetry_digest
    manifest["inventory"] = evidence_json["inventory"]
    fields: list[object] = (
        list(_CHECKPOINT_COLUMN_MISMATCHES)
        if section == "columns"
        else [
            target_field
            for target_section, target_field in (
                target.split(":", 1) for target in _CHECKPOINT_JSON_MISMATCHES
            )
            if target_section == section
        ]
    )
    if section == "extras":
        fields = ["manifest", "evidence", "manifest_inventory", "evidence_inventory"]
    elif section == "role_bindings":
        roles = tuple(item["role"] for item in evidence_json["inventory"])
        fields = [
            (roles[left], roles[right])
            for left in range(len(roles))
            for right in range(left + 1, len(roles))
        ] + [("weights-size", "weights-size")]
    for field in fields:
        changed_manifest = json.loads(canonical_json(manifest))
        changed_evidence = json.loads(canonical_json(evidence_json))
        if section == "role_bindings":
            assert isinstance(field, tuple)
            if field[0] == "weights-size":
                changed_manifest["inventory"][0]["size_bytes"] += 1
                changed_evidence["inventory"][0]["size_bytes"] += 1
            else:
                indexes = {
                    item["role"]: index for index, item in enumerate(changed_evidence["inventory"])
                }
                left, right = indexes[field[0]], indexes[field[1]]
                for document in (changed_manifest, changed_evidence):
                    (
                        document["inventory"][left]["digest"],
                        document["inventory"][right]["digest"],
                    ) = (
                        document["inventory"][right]["digest"],
                        document["inventory"][left]["digest"],
                    )
        elif section == "extras":
            assert isinstance(field, str)
            if field == "manifest":
                changed_manifest["unexpected"] = True
            elif field == "evidence":
                changed_evidence["unexpected"] = True
            elif field == "manifest_inventory":
                changed_manifest["inventory"][0]["unexpected"] = True
            else:
                changed_evidence["inventory"][0]["unexpected"] = True
        elif section != "columns":
            assert isinstance(field, str)
            document = changed_manifest if section == "manifest" else changed_evidence
            original = document[field]
            if isinstance(original, bool):
                document[field] = not original
            elif isinstance(original, int | float):
                document[field] = original + 1
            elif original is None:
                document[field] = "sha256:" + "0" * 64
            elif isinstance(original, list):
                document[field] = original[:-1]
            else:
                document[field] = (
                    "sha256:" + "0" * 64 if str(original).startswith("sha256:") else "wrong"
                )
        manifest_json = canonical_json(changed_manifest)
        evidence_serialized = canonical_json(changed_evidence)
        parameters = [
            f"bad-{section}-{field}",
            run.run_id,
            "owner",
            str(binding["recipe_storage_id"]),
            str(binding["dataset_storage_id"]),
            checkpoint.step,
            checkpoint.fraction,
            checkpoint.parent_checkpoint_id,
            checkpoint.artifact_digest,
            "sha256:" + hashlib.sha256(manifest_json.encode()).hexdigest(),
            manifest_json,
            NOW.isoformat(),
            evidence_serialized,
            "sha256:" + hashlib.sha256(evidence_serialized.encode()).hexdigest(),
        ]
        if section == "columns":
            assert isinstance(field, str)
            column_index = {
                "run_id": 1,
                "reviewer_id": 2,
                "recipe_storage_id": 3,
                "dataset_storage_id": 4,
                "step": 5,
                "fraction": 6,
                "parent_checkpoint_id": 7,
                "artifact_digest": 8,
                "manifest_digest": 9,
                "evidence_digest": 13,
            }[field]
            original = parameters[column_index]
            parameters[column_index] = (
                1.0
                if field == "fraction"
                else checkpoint.step + 1
                if field == "step"
                else "sha256:" + "0" * 64
                if field.endswith("digest")
                else "wrong"
            )
        with repository.database.connect() as connection:
            before = connection.execute("SELECT count(*) FROM training_checkpoints").fetchone()[0]
            try:
                connection.execute(
                    "INSERT INTO training_checkpoints(id,run_id,reviewer_id,recipe_storage_id,"
                    "dataset_storage_id,step,fraction,parent_checkpoint_id,artifact_digest,"
                    "manifest_digest,manifest_json,created_at,evidence_json,evidence_digest) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    parameters,
                )
            except sqlite3.IntegrityError as error:
                assert "canonical" in str(error) or "parent chain" in str(error)
            else:
                pytest.fail(f"direct SQL accepted checkpoint mismatch: {section}.{field}")
            connection.rollback()
        with repository.database.connect() as connection:
            assert (
                connection.execute("SELECT count(*) FROM training_checkpoints").fetchone()[0]
                == before
            ), field


@pytest.mark.parametrize(
    "malformation", ("missing", "extra", "duplicate-role", "wrong-role", "wrong-digest")
)
def test_direct_sql_rejects_incomplete_or_noncanonical_checkpoint_inventory(
    repository: TrainingRepository, malformation: str
) -> None:
    run, recipe_id, probe, calibration = _prepare_calibrated_scientific_run(
        repository, "inventory-run"
    )
    _start_scientific_training(repository, run, probe, calibration)
    step = checkpoint_steps(run.total_steps)[0]
    repository.append_telemetry(
        run.run_id, TrainingTelemetry(sequence=1, step=step, payload={"loss": 1.0})
    )
    checkpoint, evidence = _scientific_checkpoint(
        run, recipe_id, probe, ordinal=0, parent_id=None, parent_digest=None
    )
    with repository.database.connect() as connection:
        telemetry_digest = repository.telemetry_digest_with_connection(connection, run.run_id, 1, 1)
        binding = connection.execute(
            "SELECT recipe_storage_id,dataset_storage_id FROM training_runs WHERE id=?",
            (run.run_id,),
        ).fetchone()
    assert binding is not None
    inventory = [
        item.model_copy(update={"digest": telemetry_digest}) if item.role == "telemetry" else item
        for item in evidence.inventory
    ]
    evidence_json = evidence.model_dump(mode="json")
    evidence_json["telemetry_digest"] = telemetry_digest
    evidence_json["inventory"] = [item.model_dump(mode="json") for item in inventory]
    manifest = dict(checkpoint.model_dump(mode="json")["manifest"])
    manifest["telemetry_digest"] = telemetry_digest
    manifest["inventory"] = evidence_json["inventory"]
    if malformation in {"duplicate-role", "wrong-role"}:
        changed_role = "weights" if malformation == "duplicate-role" else "evidence"
        evidence_json["inventory"][-1]["role"] = changed_role
        manifest["inventory"] = evidence_json["inventory"]
    manifest_json = canonical_json(manifest)
    evidence_serialized = canonical_json(evidence_json)
    checkpoint_id = f"inventory-{malformation}"
    try:
        with repository.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO training_checkpoints(id,run_id,reviewer_id,recipe_storage_id,"
                "dataset_storage_id,step,fraction,parent_checkpoint_id,artifact_digest,"
                "manifest_digest,manifest_json,created_at,evidence_json,evidence_digest) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    checkpoint_id,
                    run.run_id,
                    "owner",
                    str(binding["recipe_storage_id"]),
                    str(binding["dataset_storage_id"]),
                    checkpoint.step,
                    checkpoint.fraction,
                    checkpoint.parent_checkpoint_id,
                    checkpoint.artifact_digest,
                    "sha256:" + hashlib.sha256(manifest_json.encode()).hexdigest(),
                    manifest_json,
                    NOW.isoformat(),
                    evidence_serialized,
                    "sha256:" + hashlib.sha256(evidence_serialized.encode()).hexdigest(),
                ),
            )
            rows = list(enumerate(evidence_json["inventory"]))
            if malformation == "missing":
                rows = rows[:-1]
            for ordinal, item in rows:
                digest = str(item["digest"])
                if malformation == "wrong-digest" and ordinal == 0:
                    digest = "sha256:" + "0" * 64
                connection.execute(
                    "INSERT INTO training_checkpoint_inventory VALUES (?,?,?,?,?,?)",
                    (
                        checkpoint_id,
                        ordinal,
                        item["relative_path"],
                        digest,
                        item["size_bytes"],
                        item["role"],
                    ),
                )
            if malformation == "extra":
                connection.execute(
                    "INSERT INTO training_checkpoint_inventory VALUES (?,?,?,?,?,?)",
                    (checkpoint_id, 7, "extra.bin", "sha256:" + "e" * 64, 1, "weights"),
                )
            connection.execute(
                "UPDATE training_checkpoints SET publication_state='finalized' WHERE id=?",
                (checkpoint_id,),
            )
    except sqlite3.IntegrityError:
        pass
    else:
        pytest.fail(f"direct SQL accepted malformed checkpoint inventory: {malformation}")
    with repository.database.connect() as connection:
        connection.rollback()
        assert (
            connection.execute(
                "SELECT count(*) FROM training_checkpoints WHERE id=?", (checkpoint_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM training_checkpoint_inventory WHERE checkpoint_id=?",
                (checkpoint_id,),
            ).fetchone()[0]
            == 0
        )


def test_failure_owner_and_run_binding_are_enforced(repository: TrainingRepository) -> None:
    with pytest.raises(TrainingConflict, match="run"):
        repository.record_failure(
            TrainingFailure(failure_id="failure-1", run_id="missing", code="boom", message="safe")
        )


@pytest.mark.parametrize(
    "message",
    (
        "/Volumes/private/model.bin",
        "failed at /tmp/model.bin",
        r"C:\Users\person\model.bin",
        r"\\server\share\model.bin",
        "file:///private/model.bin",
    ),
)
def test_failure_messages_reject_local_path_material(
    repository: TrainingRepository, message: str
) -> None:
    with pytest.raises(TrainingConflict, match="local path"):
        repository.record_failure(
            TrainingFailure(
                failure_id="failure-path", run_id="missing", code="boom", message=message
            )
        )
