from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from video_lab.comparison import ControlledPair
from video_lab.database import Database
from video_lab.database import validation_experiment_id as _validation_experiment_id
from video_lab.domain import (
    ClaimedJob,
    FrozenModel,
    Job,
    JobState,
    Manifest,
    RunAttempt,
    RunAttemptStatus,
    RunEvent,
    TelemetrySample,
    Variant,
    canonical_json,
    new_id,
    utc_now,
)
from video_lab.states import InvalidTransition, transition_job


class RecordNotFound(LookupError):
    pass


def validation_experiment_id(campaign_id: str, trial_id: str) -> str:
    return _validation_experiment_id(campaign_id, trial_id)


class AdaptiveExperimentConflict(RuntimeError):
    pass


class ValidationExperimentConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveWorkerLease:
    job_id: str
    attempt_id: str
    worker_id: str


class StoredExperiment(FrozenModel):
    id: str
    schema_version: str
    variable_path: str
    changed_paths: tuple[str, ...]
    created_at: datetime
    variants: tuple[Variant, Variant]


class ExperimentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, pair: ControlledPair) -> StoredExperiment:
        stored = _new_stored_experiment(pair)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _insert_stored_experiment(connection, stored)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return stored

    def create_with_jobs(self, pair: ControlledPair) -> tuple[StoredExperiment, tuple[Job, Job]]:
        stored = _new_stored_experiment(pair)
        now = utc_now()
        jobs = (
            Job(
                id=new_id("job"),
                variant_id=stored.variants[0].id,
                state=JobState.QUEUED,
                created_at=now,
                updated_at=now,
            ),
            Job(
                id=new_id("job"),
                variant_id=stored.variants[1].id,
                state=JobState.QUEUED,
                created_at=now,
                updated_at=now,
            ),
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _insert_stored_experiment(connection, stored)
                connection.executemany(
                    "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            job.id,
                            job.variant_id,
                            job.state.value,
                            job.created_at.isoformat(),
                            job.updated_at.isoformat(),
                        )
                        for job in jobs
                    ],
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return stored, jobs

    def create_adaptive_with_jobs(
        self,
        pair: ControlledPair,
        *,
        study_id: str,
        optimizer_output_digest: str,
    ) -> tuple[StoredExperiment, tuple[Job, Job]]:
        """Create or verify one deterministically identified adaptive aggregate."""
        stored, expected_jobs = _new_adaptive_experiment(
            pair,
            study_id=study_id,
            optimizer_output_digest=optimizer_output_digest,
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM experiments WHERE id = ?", (stored.id,)
                ).fetchone()
                if existing is None:
                    _insert_stored_experiment(connection, stored)
                    connection.executemany(
                        "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        [
                            (
                                job.id,
                                job.variant_id,
                                job.state.value,
                                job.created_at.isoformat(),
                                job.updated_at.isoformat(),
                            )
                            for job in expected_jobs
                        ],
                    )
                    persisted = stored
                    jobs = expected_jobs
                else:
                    persisted, jobs = _verify_adaptive_experiment(
                        connection, existing, stored, expected_jobs
                    )
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise AdaptiveExperimentConflict(
                    "adaptive experiment identity collided with persisted evidence"
                ) from error
            except Exception:
                connection.rollback()
                raise
        return persisted, jobs

    def create_validation_with_jobs(
        self,
        pair: ControlledPair,
        *,
        campaign_id: str,
        trial_id: str,
    ) -> tuple[StoredExperiment, tuple[Job, Job]]:
        """Create or verify one deterministic validation experiment and its two jobs."""
        stored, expected_jobs = _new_validation_experiment(
            pair, campaign_id=campaign_id, trial_id=trial_id
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM experiments WHERE id = ?", (stored.id,)
                ).fetchone()
                if existing is None:
                    _insert_stored_experiment(connection, stored)
                    connection.executemany(
                        "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        [
                            (
                                job.id,
                                job.variant_id,
                                job.state.value,
                                job.created_at.isoformat(),
                                job.updated_at.isoformat(),
                            )
                            for job in expected_jobs
                        ],
                    )
                    persisted, jobs = stored, expected_jobs
                else:
                    try:
                        persisted, jobs = _verify_adaptive_experiment(
                            connection, existing, stored, expected_jobs
                        )
                    except AdaptiveExperimentConflict as error:
                        raise ValidationExperimentConflict(
                            "validation experiment identity collided with persisted evidence"
                        ) from error
                _ensure_validation_experiment_binding(
                    connection,
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    experiment_id=stored.id,
                )
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValidationExperimentConflict(
                    "validation experiment identity collided with persisted evidence"
                ) from error
            except Exception:
                connection.rollback()
                raise
        return persisted, jobs

    def create_single_generation(self, manifest: Manifest) -> Job:
        """Persist one qualification experiment, variant, and queued job atomically."""
        now = utc_now()
        experiment_id = new_id("experiment")
        variant_id = new_id("variant")
        job = Job(
            id=new_id("job"),
            variant_id=variant_id,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
                    (experiment_id, "qualification.v1", "", "[]", now.isoformat()),
                )
                connection.execute(
                    "INSERT INTO variants(id, experiment_id, label, manifest_json) "
                    "VALUES (?, ?, 'A', ?)",
                    (variant_id, experiment_id, canonical_json(manifest)),
                )
                connection.execute(
                    "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (job.id, variant_id, job.state.value, now.isoformat(), now.isoformat()),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return job

    def get(self, experiment_id: str) -> StoredExperiment:
        return self._get(experiment_id, adaptive=False)

    def get_validation(self, experiment_id: str) -> StoredExperiment:
        if not experiment_id.startswith("validation_exp_"):
            raise RecordNotFound(f"validation experiment not found: {experiment_id}")
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ? AND schema_version = ?",
                (experiment_id, "validation-controlled.v1"),
            ).fetchone()
            if row is None:
                raise RecordNotFound(f"validation experiment not found: {experiment_id}")
            variants = connection.execute(
                "SELECT * FROM variants WHERE experiment_id = ? ORDER BY label", (experiment_id,)
            ).fetchall()
        return _stored_experiment(row, variants)

    def get_adaptive(self, experiment_id: str) -> StoredExperiment:
        return self._get(experiment_id, adaptive=True)

    def _get(self, experiment_id: str, *, adaptive: bool) -> StoredExperiment:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT experiments.* FROM experiments WHERE experiments.id = ? "
                + (
                    "AND EXISTS (SELECT 1 FROM adaptive_study_events AS event "
                    "WHERE json_extract(event.payload_json, '$.experiment_id') = experiments.id)"
                    if adaptive
                    else "AND experiments.schema_version <> 'validation-controlled.v1' "
                    "AND NOT EXISTS (SELECT 1 FROM adaptive_study_events AS event "
                    "WHERE json_extract(event.payload_json, '$.experiment_id') = experiments.id)"
                ),
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFound(f"experiment not found: {experiment_id}")
            variants = connection.execute(
                "SELECT * FROM variants WHERE experiment_id = ? ORDER BY label", (experiment_id,)
            ).fetchall()
        return _stored_experiment(row, variants)

    def list(self, limit: int = 50) -> tuple[StoredExperiment, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT experiments.id FROM experiments "
                "JOIN variants ON variants.experiment_id = experiments.id "
                "WHERE NOT EXISTS (SELECT 1 FROM adaptive_study_events AS event "
                "WHERE json_extract(event.payload_json, '$.experiment_id') = experiments.id) "
                "AND experiments.schema_version <> 'validation-controlled.v1' "
                "GROUP BY experiments.id HAVING COUNT(variants.id) = 2 "
                "ORDER BY experiments.created_at DESC, experiments.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(self.get(str(row["id"])) for row in rows)

    def list_adaptive_for_study(self, study_id: str) -> tuple[StoredExperiment, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT experiments.id, experiments.created_at FROM experiments "
                "JOIN adaptive_study_events AS event "
                "ON json_extract(event.payload_json, '$.experiment_id') = experiments.id "
                "WHERE event.study_id = ? ORDER BY experiments.created_at, experiments.id",
                (study_id,),
            ).fetchall()
        return tuple(self.get_adaptive(str(row["id"])) for row in rows)

    def is_generic_visible(self, experiment_id: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM experiments WHERE id = ? "
                "AND schema_version <> 'validation-controlled.v1' "
                "AND NOT EXISTS (SELECT 1 FROM adaptive_study_events AS event "
                "WHERE json_extract(event.payload_json, '$.experiment_id') = experiments.id)",
                (experiment_id,),
            ).fetchone()
        return row is not None


class JobRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create_for_experiment(self, experiment_id: str) -> tuple[Job, Job]:
        now = utc_now()
        with self.database.connect() as connection:
            variants = connection.execute(
                "SELECT id FROM variants WHERE experiment_id = ? ORDER BY label", (experiment_id,)
            ).fetchall()
            if not variants:
                raise RecordNotFound(f"experiment not found: {experiment_id}")
            if len(variants) != 2:
                raise ValueError(f"experiment must have exactly two variants: {experiment_id}")
            jobs = (
                Job(
                    id=new_id("job"),
                    variant_id=str(variants[0]["id"]),
                    state=JobState.QUEUED,
                    created_at=now,
                    updated_at=now,
                ),
                Job(
                    id=new_id("job"),
                    variant_id=str(variants[1]["id"]),
                    state=JobState.QUEUED,
                    created_at=now,
                    updated_at=now,
                ),
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (job.id, job.variant_id, job.state.value, now.isoformat(), now.isoformat())
                        for job in jobs
                    ],
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return jobs

    def get(self, job_id: str) -> Job:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise RecordNotFound(f"job not found: {job_id}")
        return _job(row)

    def list_for_experiment(self, experiment_id: str) -> tuple[Job, ...]:
        with self.database.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if exists is None:
                raise RecordNotFound(f"experiment not found: {experiment_id}")
            rows = connection.execute(
                "SELECT jobs.* FROM jobs JOIN variants ON variants.id = jobs.variant_id "
                "WHERE variants.experiment_id = ? ORDER BY variants.label",
                (experiment_id,),
            ).fetchall()
        return tuple(_job(row) for row in rows)

    def transition(self, job_id: str, target: JobState) -> Job:
        current = self.get(job_id)
        state = transition_job(current.state, target)
        updated_at = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    "UPDATE jobs SET state = ?, updated_at = ? WHERE id = ? AND state = ?",
                    (state.value, updated_at.isoformat(), job_id, current.state.value),
                )
                if cursor.rowcount != 1:
                    raise sqlite3.OperationalError("job changed during transition")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return current.model_copy(update={"state": state, "updated_at": updated_at})

    def requeue_failed_adaptive(
        self,
        study_id: str,
        experiment_id: str,
        job_ids: tuple[str, ...],
    ) -> tuple[Job, ...]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                event_row = connection.execute(
                    "SELECT new_state, payload_json FROM adaptive_study_events "
                    "WHERE study_id = ? ORDER BY sequence DESC LIMIT 1",
                    (study_id,),
                ).fetchone()
                if event_row is None or str(event_row["new_state"]) != "paused":
                    raise AdaptiveExperimentConflict(
                        "adaptive jobs can only be requeued for the active paused pair"
                    )
                try:
                    payload = json.loads(str(event_row["payload_json"]))
                    output_digest = str(payload["optimizer_output_digest"])
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise AdaptiveExperimentConflict(
                        "paused adaptive event has invalid generation evidence"
                    ) from error
                if (
                    payload.get("experiment_id") != experiment_id
                    or payload.get("resume_state") != "generating_pair"
                    or adaptive_experiment_id(study_id, output_digest) != experiment_id
                ):
                    raise AdaptiveExperimentConflict(
                        "adaptive jobs do not belong to the active paused pair"
                    )
                rows = connection.execute(
                    "SELECT jobs.*, variants.label FROM jobs "
                    "JOIN variants ON variants.id = jobs.variant_id "
                    "WHERE variants.experiment_id = ? ORDER BY variants.label",
                    (experiment_id,),
                ).fetchall()
                persisted_ids = tuple(str(row["id"]) for row in rows)
                expected_ids = tuple(
                    _adaptive_record_id("job", study_id, output_digest, role=f"job:{row['label']}")
                    for row in rows
                )
                if (
                    len(rows) != 2
                    or len(set(job_ids)) != 2
                    or set(job_ids) != set(persisted_ids)
                    or persisted_ids != expected_ids
                ):
                    raise AdaptiveExperimentConflict(
                        "adaptive retry jobs do not match the deterministic active pair jobs"
                    )
                retry_rows: list[tuple[sqlite3.Row, sqlite3.Row]] = []
                active_states = {
                    JobState.PREPARING,
                    JobState.DOWNLOADING,
                    JobState.LOADING,
                    JobState.GENERATING,
                    JobState.ENCODING,
                }
                for row in rows:
                    active_attempt = connection.execute(
                        "SELECT 1 FROM run_attempts WHERE job_id = ? AND status = ? LIMIT 1",
                        (str(row["id"]), RunAttemptStatus.ACTIVE.value),
                    ).fetchone()
                    if active_attempt is not None:
                        raise AdaptiveExperimentConflict(
                            "adaptive retry job still has an active attempt"
                        )
                    latest_attempt = connection.execute(
                        "SELECT * FROM run_attempts WHERE job_id = ? ORDER BY number DESC LIMIT 1",
                        (str(row["id"]),),
                    ).fetchone()
                    state = JobState(str(row["state"]))
                    if state is JobState.FAILED:
                        if latest_attempt is None or str(latest_attempt["status"]) not in {
                            RunAttemptStatus.FAILED.value,
                            RunAttemptStatus.INTERRUPTED.value,
                        }:
                            raise AdaptiveExperimentConflict(
                                "failed adaptive job has no finalized failure evidence"
                            )
                        retry_rows.append((row, latest_attempt))
                    elif state in active_states:
                        if (
                            latest_attempt is None
                            or str(latest_attempt["status"]) != RunAttemptStatus.INTERRUPTED.value
                        ):
                            raise AdaptiveExperimentConflict(
                                "adaptive generation job is active without interrupted evidence"
                            )
                        retry_rows.append((row, latest_attempt))
                    elif state not in {JobState.QUEUED, JobState.COMPLETED}:
                        raise AdaptiveExperimentConflict(
                            f"adaptive retry does not support job state {state.value}"
                        )
                for row, attempt_row in retry_rows:
                    cursor = connection.execute(
                        "UPDATE jobs SET state = ?, updated_at = ?, error_code = NULL, "
                        "error_message = NULL, claimed_by = NULL, claimed_at = NULL "
                        "WHERE id = ? AND state = ?",
                        (
                            JobState.QUEUED.value,
                            now.isoformat(),
                            str(row["id"]),
                            str(row["state"]),
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise sqlite3.OperationalError("adaptive retry job changed during reset")
                    _insert_event(
                        connection,
                        str(row["id"]),
                        str(attempt_row["id"]),
                        "adaptive_retry_queued",
                        {"study_id": study_id, "experiment_id": experiment_id},
                        now,
                    )
                persisted = tuple(
                    _job(
                        connection.execute(
                            "SELECT * FROM jobs WHERE id = ?", (str(row["id"]),)
                        ).fetchone()
                    )
                    for row in rows
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return persisted

    def claim_next(self, worker_id: str, job_id: str | None = None) -> ClaimedJob | None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    (
                        "SELECT * FROM jobs WHERE state = ? ORDER BY created_at, id LIMIT 1"
                        if job_id is None
                        else "SELECT * FROM jobs WHERE state = ? AND id = ? LIMIT 1"
                    ),
                    (JobState.QUEUED.value,) if job_id is None else (JobState.QUEUED.value, job_id),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                job = _job(row)
                state = transition_job(job.state, JobState.PREPARING)
                now = utc_now()
                cursor = connection.execute(
                    "UPDATE jobs SET state = ?, updated_at = ?, claimed_by = ?, claimed_at = ? "
                    "WHERE id = ? AND state = ? AND claimed_by IS NULL",
                    (
                        state.value,
                        now.isoformat(),
                        worker_id,
                        now.isoformat(),
                        job.id,
                        JobState.QUEUED.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise sqlite3.OperationalError("job changed during claim")
                next_number = int(
                    connection.execute(
                        "SELECT coalesce(max(number), 0) + 1 FROM run_attempts WHERE job_id = ?",
                        (job.id,),
                    ).fetchone()[0]
                )
                attempt = RunAttempt(
                    id=new_id("attempt"),
                    job_id=job.id,
                    number=next_number,
                    worker_id=worker_id,
                    status=RunAttemptStatus.ACTIVE,
                    started_at=now,
                )
                connection.execute(
                    "INSERT INTO run_attempts("
                    "id, job_id, number, worker_id, status, started_at"
                    ") VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        attempt.id,
                        attempt.job_id,
                        attempt.number,
                        attempt.worker_id,
                        attempt.status.value,
                        attempt.started_at.isoformat(),
                    ),
                )
                _insert_event(
                    connection,
                    job.id,
                    attempt.id,
                    "job_claimed",
                    {"worker_id": worker_id},
                    now,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        claimed_job = job.model_copy(update={"state": state, "updated_at": now})
        return ClaimedJob(job=claimed_job, attempt=attempt)

    def append_event(
        self,
        job_id: str,
        attempt_id: str,
        kind: str,
        payload: dict[str, object],
    ) -> RunEvent:
        now = utc_now()
        with self.database.connect() as connection:
            event_id = _insert_event(connection, job_id, attempt_id, kind, payload, now)
        return RunEvent(
            id=event_id,
            job_id=job_id,
            attempt_id=attempt_id,
            kind=kind,
            payload=payload,
            created_at=now,
        )

    def events_for_job(self, job_id: str) -> tuple[RunEvent, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        return tuple(_run_event(row) for row in rows)

    def append_telemetry(self, sample: TelemetrySample) -> TelemetrySample:
        sample = TelemetrySample.model_validate(sample.model_dump(round_trip=True))
        with self.database.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO telemetry_samples("
                "attempt_id, recorded_at, elapsed_seconds, process_rss_bytes, "
                "system_used_bytes, system_available_bytes, swap_used_bytes, thermal_pressure"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sample.attempt_id,
                    sample.recorded_at.isoformat(),
                    sample.elapsed_seconds,
                    sample.process_rss_bytes,
                    sample.system_used_bytes,
                    sample.system_available_bytes,
                    sample.swap_used_bytes,
                    sample.thermal_pressure,
                ),
            )
        sample_id = cursor.lastrowid
        if sample_id is None:
            raise sqlite3.OperationalError("telemetry insert did not produce an id")
        return sample.model_copy(update={"id": sample_id})

    def telemetry_for_attempt(self, attempt_id: str) -> tuple[TelemetrySample, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM telemetry_samples WHERE attempt_id = ? ORDER BY id", (attempt_id,)
            ).fetchall()
        return tuple(_telemetry_sample(row) for row in rows)

    def attempts_for_job(self, job_id: str) -> tuple[RunAttempt, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_attempts WHERE job_id = ? ORDER BY number", (job_id,)
            ).fetchall()
        return tuple(_run_attempt(row) for row in rows)

    def active_worker_leases(self) -> tuple[ActiveWorkerLease, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT jobs.id AS job_id, run_attempts.id AS attempt_id, "
                "run_attempts.worker_id FROM jobs JOIN run_attempts "
                "ON run_attempts.job_id = jobs.id WHERE run_attempts.status = ? "
                "AND jobs.claimed_by = run_attempts.worker_id "
                "ORDER BY jobs.created_at, jobs.id, run_attempts.number",
                (RunAttemptStatus.ACTIVE.value,),
            ).fetchall()
        return tuple(
            ActiveWorkerLease(
                job_id=str(row["job_id"]),
                attempt_id=str(row["attempt_id"]),
                worker_id=str(row["worker_id"]),
            )
            for row in rows
        )

    def manifest_for_job(self, job_id: str) -> Manifest:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT variants.manifest_json FROM jobs "
                "JOIN variants ON variants.id = jobs.variant_id WHERE jobs.id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFound(f"job not found: {job_id}")
        return Manifest.model_validate_json(str(row["manifest_json"]))

    def request_cancellation(self, job_id: str) -> Job:
        now = utc_now()
        active_states = {
            JobState.PREPARING,
            JobState.DOWNLOADING,
            JobState.LOADING,
            JobState.GENERATING,
            JobState.ENCODING,
        }
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise RecordNotFound(f"job not found: {job_id}")
                job = _job(row)
                if job.state is JobState.CANCELLED:
                    connection.commit()
                    return job
                if job.state in {JobState.COMPLETED, JobState.FAILED}:
                    raise InvalidTransition(f"cannot cancel terminal job in state {job.state}")
                if job.state is JobState.QUEUED:
                    connection.execute(
                        "UPDATE jobs SET state = ?, updated_at = ?, claimed_by = NULL, "
                        "claimed_at = NULL WHERE id = ?",
                        (JobState.CANCELLED.value, now.isoformat(), job_id),
                    )
                elif job.state in active_states:
                    attempt_row = connection.execute(
                        "SELECT id FROM run_attempts WHERE job_id = ? AND status = ? "
                        "ORDER BY number DESC LIMIT 1",
                        (job_id, RunAttemptStatus.ACTIVE.value),
                    ).fetchone()
                    if attempt_row is None:
                        raise sqlite3.OperationalError("active job has no active attempt")
                    connection.execute(
                        "UPDATE jobs SET state = ?, updated_at = ? WHERE id = ?",
                        (JobState.CANCELLED.value, now.isoformat(), job_id),
                    )
                    _insert_event(
                        connection,
                        job_id,
                        str(attempt_row["id"]),
                        "cancellation_requested",
                        {},
                        now,
                    )
                else:  # pragma: no cover - closed enum protects this branch
                    raise InvalidTransition(f"cannot cancel job in state {job.state}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return job.model_copy(update={"state": JobState.CANCELLED, "updated_at": now})

    def finalize_attempt(
        self,
        job_id: str,
        attempt_id: str,
        *,
        job_state: JobState,
        attempt_status: RunAttemptStatus,
        artifact_id: str | None,
        error_code: str | None,
        error_message: str | None,
        event_kind: str,
        event_payload: dict[str, object],
    ) -> tuple[Job, RunAttempt]:
        expected = {
            JobState.COMPLETED: RunAttemptStatus.COMPLETED,
            JobState.FAILED: RunAttemptStatus.FAILED,
            JobState.CANCELLED: RunAttemptStatus.CANCELLED,
        }
        if expected.get(job_state) is not attempt_status:
            raise ValueError("job state and attempt status do not match")
        if job_state is JobState.COMPLETED:
            if artifact_id is None or error_code is not None or error_message is not None:
                raise ValueError("completed attempt requires an artifact and no error")
        elif artifact_id is not None or error_code is None or error_message is None:
            raise ValueError("failed or cancelled attempt requires an error and no artifact")
        if not event_kind:
            raise ValueError("terminal event kind must not be empty")

        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                job_row = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if job_row is None:
                    raise RecordNotFound(f"job not found: {job_id}")
                job = _job(job_row)
                if job.state is JobState.CANCELLED:
                    if job_state is not JobState.CANCELLED:
                        raise InvalidTransition("cancelled job may only finalize as cancelled")
                else:
                    transition_job(job.state, job_state)
                attempt_row = connection.execute(
                    "SELECT * FROM run_attempts WHERE id = ? AND job_id = ?",
                    (attempt_id, job_id),
                ).fetchone()
                if attempt_row is None:
                    raise RecordNotFound(f"attempt not found: {attempt_id}")
                attempt = _run_attempt(attempt_row)
                if attempt.status is not RunAttemptStatus.ACTIVE:
                    raise ValueError("attempt is already finalized")
                cursor = connection.execute(
                    "UPDATE run_attempts SET status = ?, finished_at = ?, error_code = ?, "
                    "error_message = ?, artifact_id = ? WHERE id = ? AND status = ?",
                    (
                        attempt_status.value,
                        now.isoformat(),
                        error_code,
                        error_message,
                        artifact_id,
                        attempt_id,
                        RunAttemptStatus.ACTIVE.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise sqlite3.OperationalError("attempt changed during finalization")
                _insert_event(connection, job_id, attempt_id, event_kind, event_payload, now)
                if artifact_id is not None:
                    connection.execute(
                        "INSERT OR IGNORE INTO job_artifacts(job_id, artifact_id, created_at) "
                        "VALUES (?, ?, ?)",
                        (job_id, artifact_id, now.isoformat()),
                    )
                connection.execute(
                    "UPDATE jobs SET state = ?, updated_at = ?, error_code = ?, "
                    "error_message = ?, claimed_by = NULL, claimed_at = NULL WHERE id = ?",
                    (job_state.value, now.isoformat(), error_code, error_message, job_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return (
            job.model_copy(update={"state": job_state, "updated_at": now}),
            attempt.model_copy(
                update={
                    "status": attempt_status,
                    "finished_at": now,
                    "error_code": error_code,
                    "error_message": error_message,
                    "artifact_id": artifact_id,
                }
            ),
        )

    def recover_interrupted(self, worker_id: str) -> tuple[Job, ...]:
        active_states = (
            JobState.PREPARING,
            JobState.DOWNLOADING,
            JobState.LOADING,
            JobState.GENERATING,
            JobState.ENCODING,
        )
        placeholders = ",".join("?" for _ in active_states)
        now = utc_now()
        recovered: list[Job] = []
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT jobs.*, run_attempts.id AS attempt_id FROM jobs "
                    "JOIN run_attempts ON run_attempts.job_id = jobs.id "
                    f"WHERE jobs.state IN ({placeholders}) AND run_attempts.status = ? "
                    "AND jobs.claimed_by = ? AND run_attempts.worker_id = ? "
                    "ORDER BY jobs.created_at, jobs.id, run_attempts.number",
                    (
                        *tuple(state.value for state in active_states),
                        RunAttemptStatus.ACTIVE.value,
                        worker_id,
                        worker_id,
                    ),
                ).fetchall()
                for row in rows:
                    attempt_cursor = connection.execute(
                        "UPDATE run_attempts SET status = ?, finished_at = ?, "
                        "error_code = ?, error_message = ? WHERE id = ? AND status = ? "
                        "AND worker_id = ?",
                        (
                            RunAttemptStatus.INTERRUPTED.value,
                            now.isoformat(),
                            "worker_interrupted",
                            "active attempt interrupted before completion",
                            str(row["attempt_id"]),
                            RunAttemptStatus.ACTIVE.value,
                            worker_id,
                        ),
                    )
                    if attempt_cursor.rowcount != 1:
                        raise sqlite3.OperationalError("attempt lease changed during recovery")
                    _insert_event(
                        connection,
                        str(row["id"]),
                        str(row["attempt_id"]),
                        "worker_interrupted",
                        {"recovered_by": worker_id},
                        now,
                    )
                    job_cursor = connection.execute(
                        "UPDATE jobs SET state = ?, updated_at = ?, claimed_by = NULL, "
                        "claimed_at = NULL WHERE id = ? AND claimed_by = ?",
                        (
                            JobState.QUEUED.value,
                            now.isoformat(),
                            str(row["id"]),
                            worker_id,
                        ),
                    )
                    if job_cursor.rowcount != 1:
                        raise sqlite3.OperationalError("job lease changed during recovery")
                    recovered.append(
                        _job(row).model_copy(update={"state": JobState.QUEUED, "updated_at": now})
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return tuple(recovered)


def _stored_experiment(row: sqlite3.Row, variant_rows: list[sqlite3.Row]) -> StoredExperiment:
    variants = tuple(
        Variant(
            id=str(item["id"]),
            experiment_id=str(item["experiment_id"]),
            label=str(item["label"]),
            manifest=Manifest.model_validate_json(str(item["manifest_json"])),
        )
        for item in variant_rows
    )
    if len(variants) != 2:
        raise ValueError(f"experiment must have exactly two variants: {row['id']}")
    return StoredExperiment(
        id=str(row["id"]),
        schema_version=str(row["schema_version"]),
        variable_path=str(row["variable_path"]),
        changed_paths=tuple(json.loads(str(row["changed_paths_json"]))),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        variants=(variants[0], variants[1]),
    )


def _new_stored_experiment(pair: ControlledPair) -> StoredExperiment:
    experiment_id = new_id("exp")
    variants = (
        Variant(
            id=new_id("var"),
            experiment_id=experiment_id,
            label="A",
            manifest=pair.variant_a,
        ),
        Variant(
            id=new_id("var"),
            experiment_id=experiment_id,
            label="B",
            manifest=pair.variant_b,
        ),
    )
    return StoredExperiment(
        id=experiment_id,
        schema_version=pair.variant_a.schema_version,
        variable_path=pair.variable_path,
        changed_paths=pair.changed_paths,
        created_at=utc_now(),
        variants=variants,
    )


def adaptive_experiment_id(study_id: str, optimizer_output_digest: str) -> str:
    return _adaptive_record_id("exp", study_id, optimizer_output_digest, role="experiment")


def _adaptive_record_id(
    record_kind: str,
    study_id: str,
    optimizer_output_digest: str,
    *,
    role: str,
) -> str:
    if not study_id:
        raise ValueError("adaptive study id must not be empty")
    if len(optimizer_output_digest) != 64 or set(optimizer_output_digest) - set("0123456789abcdef"):
        raise ValueError("optimizer output digest must be lowercase sha256 hex")
    payload = canonical_json(
        {
            "schema_version": "adaptive-experiment-identity.v1",
            "domain": f"video-lab.adaptive.{record_kind}",
            "study_id": study_id,
            "optimizer_output_digest": optimizer_output_digest,
            "role": role,
        }
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"adaptive_{record_kind}_{digest}"


def _validation_record_id(record_kind: str, campaign_id: str, trial_id: str, *, role: str) -> str:
    if not campaign_id or not trial_id:
        raise ValueError("validation campaign and trial ids must not be empty")
    payload = canonical_json(
        {
            "schema_version": "validation-experiment-identity.v1",
            "domain": f"video-lab.validation.{record_kind}",
            "campaign_id": campaign_id,
            "trial_id": trial_id,
            "role": role,
        }
    )
    return f"validation_{record_kind}_{hashlib.sha256(payload.encode()).hexdigest()}"


def _new_validation_experiment(
    pair: ControlledPair, *, campaign_id: str, trial_id: str
) -> tuple[StoredExperiment, tuple[Job, Job]]:
    experiment_id = validation_experiment_id(campaign_id, trial_id)
    now = utc_now()
    variants = tuple(
        Variant(
            id=_validation_record_id("var", campaign_id, trial_id, role=f"variant:{label}"),
            experiment_id=experiment_id,
            label=label,
            manifest=manifest,
        )
        for label, manifest in (("A", pair.variant_a), ("B", pair.variant_b))
    )
    stored = StoredExperiment(
        id=experiment_id,
        schema_version="validation-controlled.v1",
        variable_path=pair.variable_path,
        changed_paths=pair.changed_paths,
        created_at=now,
        variants=(variants[0], variants[1]),
    )
    jobs = tuple(
        Job(
            id=_validation_record_id("job", campaign_id, trial_id, role=f"job:{variant.label}"),
            variant_id=variant.id,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
        )
        for variant in variants
    )
    return stored, (jobs[0], jobs[1])


def _ensure_validation_experiment_binding(
    connection: sqlite3.Connection,
    *,
    campaign_id: str,
    trial_id: str,
    experiment_id: str,
) -> None:
    row = connection.execute(
        "SELECT experiment_id FROM validation_experiment_bindings "
        "WHERE campaign_id=? AND trial_id=?",
        (campaign_id, trial_id),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO validation_experiment_bindings(campaign_id, trial_id, experiment_id) "
            "VALUES (?, ?, ?)",
            (campaign_id, trial_id, experiment_id),
        )
    elif str(row["experiment_id"]) != experiment_id:
        raise ValidationExperimentConflict(
            "validation trial is bound to a different controlled experiment"
        )


def _new_adaptive_experiment(
    pair: ControlledPair,
    *,
    study_id: str,
    optimizer_output_digest: str,
) -> tuple[StoredExperiment, tuple[Job, Job]]:
    experiment_id = adaptive_experiment_id(study_id, optimizer_output_digest)
    now = utc_now()
    variants = (
        Variant(
            id=_adaptive_record_id("var", study_id, optimizer_output_digest, role="variant:A"),
            experiment_id=experiment_id,
            label="A",
            manifest=pair.variant_a,
        ),
        Variant(
            id=_adaptive_record_id("var", study_id, optimizer_output_digest, role="variant:B"),
            experiment_id=experiment_id,
            label="B",
            manifest=pair.variant_b,
        ),
    )
    stored = StoredExperiment(
        id=experiment_id,
        schema_version=pair.variant_a.schema_version,
        variable_path=pair.variable_path,
        changed_paths=pair.changed_paths,
        created_at=now,
        variants=variants,
    )
    jobs = tuple(
        Job(
            id=_adaptive_record_id(
                "job", study_id, optimizer_output_digest, role=f"job:{variant.label}"
            ),
            variant_id=variant.id,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
        )
        for variant in variants
    )
    return stored, (jobs[0], jobs[1])


def _verify_adaptive_experiment(
    connection: sqlite3.Connection,
    experiment_row: sqlite3.Row,
    expected: StoredExperiment,
    expected_jobs: tuple[Job, Job],
) -> tuple[StoredExperiment, tuple[Job, Job]]:
    variant_rows = connection.execute(
        "SELECT * FROM variants WHERE experiment_id = ? ORDER BY label", (expected.id,)
    ).fetchall()
    try:
        persisted = _stored_experiment(experiment_row, variant_rows)
    except (TypeError, ValueError) as error:
        raise AdaptiveExperimentConflict(
            "adaptive experiment aggregate is partial or invalid"
        ) from error
    if (
        persisted.schema_version != expected.schema_version
        or persisted.variable_path != expected.variable_path
        or persisted.changed_paths != expected.changed_paths
        or tuple(
            (variant.id, variant.experiment_id, variant.label, variant.manifest)
            for variant in persisted.variants
        )
        != tuple(
            (variant.id, variant.experiment_id, variant.label, variant.manifest)
            for variant in expected.variants
        )
    ):
        raise AdaptiveExperimentConflict(
            "adaptive experiment identity does not match the controlled pair"
        )
    job_rows = connection.execute(
        "SELECT jobs.* FROM jobs JOIN variants ON variants.id = jobs.variant_id "
        "WHERE variants.experiment_id = ? ORDER BY variants.label, jobs.id",
        (expected.id,),
    ).fetchall()
    jobs = tuple(_job(row) for row in job_rows)
    if tuple((job.id, job.variant_id) for job in jobs) != tuple(
        (job.id, job.variant_id) for job in expected_jobs
    ):
        raise AdaptiveExperimentConflict(
            "adaptive experiment jobs are partial or do not match deterministic identities"
        )
    return persisted, (jobs[0], jobs[1])


def _insert_stored_experiment(connection: sqlite3.Connection, stored: StoredExperiment) -> None:
    connection.execute(
        "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
        (
            stored.id,
            stored.schema_version,
            stored.variable_path,
            json.dumps(list(stored.changed_paths), separators=(",", ":")),
            stored.created_at.isoformat(),
        ),
    )
    connection.executemany(
        "INSERT INTO variants(id, experiment_id, label, manifest_json) VALUES (?, ?, ?, ?)",
        [
            (item.id, item.experiment_id, item.label, canonical_json(item.manifest))
            for item in stored.variants
        ],
    )


def _job(row: sqlite3.Row) -> Job:
    return Job(
        id=str(row["id"]),
        variant_id=str(row["variant_id"]),
        state=JobState(str(row["state"])),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _run_attempt(row: sqlite3.Row) -> RunAttempt:
    finished_at = row["finished_at"]
    return RunAttempt(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        number=int(row["number"]),
        worker_id=str(row["worker_id"]),
        status=RunAttemptStatus(str(row["status"])),
        started_at=datetime.fromisoformat(str(row["started_at"])),
        finished_at=None if finished_at is None else datetime.fromisoformat(str(finished_at)),
        error_code=None if row["error_code"] is None else str(row["error_code"]),
        error_message=None if row["error_message"] is None else str(row["error_message"]),
        artifact_id=None if row["artifact_id"] is None else str(row["artifact_id"]),
    )


def _run_event(row: sqlite3.Row) -> RunEvent:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise ValueError(f"event payload must be an object: {row['id']}")
    return RunEvent(
        id=int(row["id"]),
        job_id=str(row["job_id"]),
        attempt_id=str(row["attempt_id"]),
        kind=str(row["kind"]),
        payload=payload,
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _telemetry_sample(row: sqlite3.Row) -> TelemetrySample:
    return TelemetrySample(
        id=int(row["id"]),
        attempt_id=str(row["attempt_id"]),
        recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
        elapsed_seconds=float(row["elapsed_seconds"]),
        process_rss_bytes=int(row["process_rss_bytes"]),
        system_used_bytes=int(row["system_used_bytes"]),
        system_available_bytes=int(row["system_available_bytes"]),
        swap_used_bytes=int(row["swap_used_bytes"]),
        thermal_pressure=(
            None if row["thermal_pressure"] is None else str(row["thermal_pressure"])
        ),
    )


def _insert_event(
    connection: sqlite3.Connection,
    job_id: str,
    attempt_id: str,
    kind: str,
    payload: dict[str, object],
    created_at: datetime,
) -> int:
    cursor = connection.execute(
        "INSERT INTO run_events(job_id, attempt_id, kind, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (job_id, attempt_id, kind, canonical_json(payload), created_at.isoformat()),
    )
    event_id = cursor.lastrowid
    if event_id is None:
        raise sqlite3.OperationalError("event insert did not produce an id")
    return event_id
