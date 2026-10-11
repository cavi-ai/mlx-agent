from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lab_tests.test_capabilities import manifest
from video_lab.comparison import build_controlled_pair
from video_lab.database import _MIGRATION_1, _MIGRATION_2, _MIGRATION_3, SCHEMA_VERSION, Database
from video_lab.domain import (
    ExperimentDraft,
    JobState,
    RunAttempt,
    RunAttemptStatus,
    RunEvent,
    TelemetrySample,
)
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.states import InvalidTransition


def _claim_worker(
    path: str,
    barrier: multiprocessing.synchronize.Barrier,
    results: multiprocessing.queues.Queue[Any],
    worker_id: str,
) -> None:
    repository = JobRepository(Database(Path(path)))
    barrier.wait()
    claimed = repository.claim_next(worker_id)
    results.put(None if claimed is None else claimed.model_dump(mode="json"))


def _create_one_queued_job(database: Database) -> tuple[JobRepository, str]:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    stored = ExperimentRepository(database).create(pair)
    repository = JobRepository(database)
    now = datetime.now(UTC).isoformat()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
            "VALUES (?, ?, 'queued', ?, ?)",
            ("job_only", stored.variants[0].id, now, now),
        )
    return repository, "job_only"


def test_claim_is_atomic_and_creates_attempt(database: Database) -> None:
    repository, job_id = _create_one_queued_job(database)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_claim_worker,
            args=(str(database.path), barrier, results, worker_id),
        )
        for worker_id in ("worker-a", "worker-b")
    ]

    for process in processes:
        process.start()
    claimed_results = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    claimed = [result for result in claimed_results if result is not None]
    assert len(claimed) == 1
    assert claimed[0]["job"]["id"] == job_id
    assert claimed[0]["job"]["state"] == JobState.PREPARING.value
    assert claimed[0]["attempt"]["number"] == 1
    assert repository.claim_next("worker-c") is None
    assert [event.kind for event in repository.events_for_job(job_id)] == ["job_claimed"]


def test_events_and_telemetry_retain_insertion_order(database: Database) -> None:
    repository, _ = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None
    assert claimed.job.created_at.tzinfo is UTC
    assert claimed.job.updated_at.tzinfo is UTC
    assert claimed.attempt.started_at.tzinfo is UTC

    first = repository.append_event(
        claimed.job.id, claimed.attempt.id, "download_started", {"source": "模型", "part": 1}
    )
    second = repository.append_event(
        claimed.job.id, claimed.attempt.id, "download_finished", {"part": 2}
    )
    started = datetime.now(UTC)
    samples = (
        TelemetrySample(
            attempt_id=claimed.attempt.id,
            recorded_at=started,
            elapsed_seconds=1.25,
            process_rss_bytes=100,
            system_used_bytes=200,
            system_available_bytes=300,
            swap_used_bytes=400,
            thermal_pressure=None,
        ),
        TelemetrySample(
            attempt_id=claimed.attempt.id,
            recorded_at=started + timedelta(seconds=1),
            elapsed_seconds=2.25,
            process_rss_bytes=101,
            system_used_bytes=201,
            system_available_bytes=301,
            swap_used_bytes=401,
            thermal_pressure="nominal",
        ),
    )

    stored_samples = tuple(repository.append_telemetry(sample) for sample in samples)

    events = repository.events_for_job(claimed.job.id)
    assert [event.kind for event in events] == [
        "job_claimed",
        "download_started",
        "download_finished",
    ]
    assert all(
        previous.id < current.id for previous, current in zip(events, events[1:], strict=False)
    )
    assert [event.id for event in events[1:]] == [first.id, second.id]
    assert [event.payload for event in events][1:] == [
        {"part": 1, "source": "模型"},
        {"part": 2},
    ]
    assert repository.telemetry_for_attempt(claimed.attempt.id) == stored_samples


def test_run_models_reject_naive_timestamps_and_normalize_aware_values() -> None:
    non_utc = timezone(timedelta(hours=5, minutes=30))
    aware = datetime(2026, 7, 16, 12, 0, tzinfo=non_utc)

    attempt = RunAttempt(
        id="attempt_1",
        job_id="job_1",
        number=1,
        worker_id="worker-a",
        status=RunAttemptStatus.ACTIVE,
        started_at=aware,
        finished_at=aware,
    )
    event = RunEvent(
        id=1,
        job_id="job_1",
        attempt_id="attempt_1",
        kind="test",
        payload={},
        created_at=aware,
    )
    sample = TelemetrySample(
        attempt_id="attempt_1",
        recorded_at=aware,
        elapsed_seconds=0,
        process_rss_bytes=0,
        system_used_bytes=0,
        system_available_bytes=0,
        swap_used_bytes=0,
    )

    assert attempt.started_at.tzinfo is UTC
    assert attempt.finished_at is not None and attempt.finished_at.tzinfo is UTC
    assert event.created_at.tzinfo is UTC
    assert sample.recorded_at.tzinfo is UTC

    naive = datetime(2026, 7, 16, 12, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        attempt.model_copy(update={"started_at": naive})
    with pytest.raises(ValidationError, match="timezone-aware"):
        event.model_copy(update={"created_at": naive})
    with pytest.raises(ValidationError, match="timezone-aware"):
        sample.model_copy(update={"recorded_at": naive})


def test_run_attempt_status_is_closed() -> None:
    with pytest.raises(ValidationError):
        RunAttempt(
            id="attempt_1",
            job_id="job_1",
            number=1,
            worker_id="worker-a",
            status="unknown",
            started_at=datetime.now(UTC),
        )


def test_append_telemetry_rejects_naive_timestamp_at_repository_boundary(
    database: Database,
) -> None:
    repository, _ = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None
    invalid = TelemetrySample.model_construct(
        attempt_id=claimed.attempt.id,
        recorded_at=datetime(2026, 7, 16, 12, 0),
        elapsed_seconds=0,
        process_rss_bytes=0,
        system_used_bytes=0,
        system_available_bytes=0,
        swap_used_bytes=0,
    )

    with pytest.raises(ValidationError, match="timezone-aware"):
        repository.append_telemetry(invalid)

    assert repository.telemetry_for_attempt(claimed.attempt.id) == ()


def test_manifest_lookup_returns_immutable_variant_manifest(database: Database) -> None:
    repository, job_id = _create_one_queued_job(database)

    stored = repository.manifest_for_job(job_id)

    assert stored == manifest()
    with pytest.raises(TypeError):
        stored.runtime_versions["python"] = "changed"  # type: ignore[index]


def test_recovery_preserves_interruption_evidence(database: Database) -> None:
    repository, _ = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None

    recovered = repository.recover_interrupted("worker-a")

    assert [job.state for job in recovered] == [JobState.QUEUED]
    attempt = repository.attempts_for_job(claimed.job.id)[-1]
    assert attempt.status is RunAttemptStatus.INTERRUPTED
    assert attempt.finished_at is not None
    assert attempt.finished_at.tzinfo is UTC
    assert attempt.error_code == "worker_interrupted"
    assert attempt.error_message == "active attempt interrupted before completion"
    events = repository.events_for_job(claimed.job.id)
    assert events[-1].kind == "worker_interrupted"
    assert events[-1].payload == {"recovered_by": "worker-a"}
    with database.connect() as connection:
        lease = connection.execute(
            "SELECT claimed_by, claimed_at FROM jobs WHERE id = ?", (claimed.job.id,)
        ).fetchone()
    assert tuple(lease) == (None, None)
    with pytest.raises(InvalidTransition):
        repository.transition(claimed.job.id, JobState.QUEUED)


def test_recovery_does_not_steal_another_workers_live_lease(database: Database) -> None:
    repository, job_id = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None

    assert repository.recover_interrupted("worker-b") == ()

    assert repository.get(job_id).state is JobState.PREPARING
    assert repository.attempts_for_job(job_id)[0].status is RunAttemptStatus.ACTIVE
    with database.connect() as connection:
        lease = connection.execute(
            "SELECT claimed_by, claimed_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert lease[0] == "worker-a"
    assert lease[1] is not None


def test_cancellation_request_is_persisted_and_keeps_lease_until_finalized(
    database: Database,
) -> None:
    repository, job_id = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None

    requested = repository.request_cancellation(job_id)
    repeated = repository.request_cancellation(job_id)

    assert requested.state is JobState.CANCELLED
    assert repeated == requested
    assert [event.kind for event in repository.events_for_job(job_id)] == [
        "job_claimed",
        "cancellation_requested",
    ]
    with database.connect() as connection:
        lease = connection.execute(
            "SELECT claimed_by, claimed_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert lease[0] == "worker-a"
    assert lease[1] is not None

    job, attempt = repository.finalize_attempt(
        job_id,
        claimed.attempt.id,
        job_state=JobState.CANCELLED,
        attempt_status=RunAttemptStatus.CANCELLED,
        artifact_id=None,
        error_code="cancelled",
        error_message="generation cancelled",
        event_kind="job_cancelled",
        event_payload={"reason": "requested"},
    )

    assert job.state is JobState.CANCELLED
    assert attempt.status is RunAttemptStatus.CANCELLED
    assert attempt.finished_at is not None
    with database.connect() as connection:
        lease = connection.execute(
            "SELECT claimed_by, claimed_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert tuple(lease) == (None, None)


def test_queued_cancellation_has_no_attempt_and_clears_lease(database: Database) -> None:
    repository, job_id = _create_one_queued_job(database)

    cancelled = repository.request_cancellation(job_id)

    assert cancelled.state is JobState.CANCELLED
    assert repository.attempts_for_job(job_id) == ()
    assert repository.events_for_job(job_id) == ()


@pytest.mark.parametrize("terminal", [JobState.COMPLETED, JobState.FAILED])
def test_terminal_job_cancellation_is_invalid(database: Database, terminal: JobState) -> None:
    repository, job_id = _create_one_queued_job(database)
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = ? WHERE id = ?", (terminal.value, job_id))

    with pytest.raises(InvalidTransition):
        repository.request_cancellation(job_id)


def test_finalize_attempt_validates_pairing_without_partial_writes(database: Database) -> None:
    repository, job_id = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None

    with pytest.raises(ValueError, match="status"):
        repository.finalize_attempt(
            job_id,
            claimed.attempt.id,
            job_state=JobState.COMPLETED,
            attempt_status=RunAttemptStatus.FAILED,
            artifact_id=None,
            error_code="bad",
            error_message="bad",
            event_kind="job_completed",
            event_payload={},
        )

    assert repository.get(job_id).state is JobState.PREPARING
    assert repository.attempts_for_job(job_id)[0].status is RunAttemptStatus.ACTIVE
    assert [event.kind for event in repository.events_for_job(job_id)] == ["job_claimed"]


def test_finalize_attempt_rolls_back_when_terminal_event_cannot_serialize(
    database: Database,
) -> None:
    repository, job_id = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None

    with pytest.raises(TypeError):
        repository.finalize_attempt(
            job_id,
            claimed.attempt.id,
            job_state=JobState.FAILED,
            attempt_status=RunAttemptStatus.FAILED,
            artifact_id=None,
            error_code="adapter_error",
            error_message="failed",
            event_kind="job_failed",
            event_payload={"invalid": object()},
        )

    assert repository.get(job_id).state is JobState.PREPARING
    assert repository.attempts_for_job(job_id)[0].status is RunAttemptStatus.ACTIVE
    assert [event.kind for event in repository.events_for_job(job_id)] == ["job_claimed"]


@pytest.mark.parametrize("terminal", [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED])
def test_terminal_jobs_are_never_recovered(database: Database, terminal: JobState) -> None:
    repository, job_id = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-a")
    assert claimed is not None
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET state = ?, claimed_by = 'worker-a' WHERE id = ?",
            (terminal.value, job_id),
        )

    assert repository.recover_interrupted("startup-worker") == ()
    assert repository.get(job_id).state is terminal
    assert repository.attempts_for_job(job_id)[-1].status is RunAttemptStatus.ACTIVE


def test_migrations_upgrade_schema_version_3_database(tmp_path: Path) -> None:
    database = Database(tmp_path / "v3.sqlite3")
    with database.connect() as connection:
        for migration in (_MIGRATION_1, _MIGRATION_2, _MIGRATION_3):
            connection.executescript(migration)
        connection.executemany(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
            [(1,), (2,), (3,)],
        )

    database.migrate()
    database.migrate()

    with database.connect() as connection:
        versions = [row[0] for row in connection.execute("SELECT version FROM schema_migrations")]
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        job_columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert versions == list(range(1, SCHEMA_VERSION + 1))
    assert {"run_attempts", "run_events", "telemetry_samples"} <= tables
    assert {"claimed_by", "claimed_at"} <= job_columns
