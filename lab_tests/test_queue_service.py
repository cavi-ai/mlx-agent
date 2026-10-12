from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from lab_tests.test_capabilities import manifest
from video_lab.artifacts import ArtifactStore
from video_lab.comparison import build_controlled_pair
from video_lab.domain import ExperimentDraft, JobState, TelemetrySample
from video_lab.queue_service import QueueService
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository


def _service(database, tmp_path, worker=None):
    experiments = ExperimentRepository(database)
    jobs = JobRepository(database)
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    reviews = ReviewService(experiments, jobs, artifacts, ReviewRepository(database))
    return QueueService(experiments, jobs, artifacts, reviews, worker_status=worker)


def _experiment(database):
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    return ExperimentRepository(database).create_with_jobs(pair)


def test_empty_snapshot_has_neutral_worker_status(database, tmp_path) -> None:
    view = _service(database, tmp_path).snapshot()
    assert view.schema_version == "queue.v1"
    assert view.experiments == ()
    assert view.worker == {"state": "unavailable"}


@pytest.mark.parametrize("state", list(JobState))
def test_projects_every_persisted_job_state_and_one_variable_label(
    database, tmp_path, state
) -> None:
    experiment, jobs = _experiment(database)
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = ? WHERE id = ?", (state.value, jobs[0].id))

    snapshot = _service(
        database, tmp_path, lambda: {"state": "idle", "stderr": "/secret"}
    ).snapshot()
    projected = snapshot.experiments[0]
    assert projected.variable_label == "inference_steps: 4 -> 8"
    assert projected.jobs[0].state == state.value
    assert snapshot.worker == {"state": "idle"}
    assert "/secret" not in json.dumps(snapshot.model_dump(mode="json"))


def test_attempts_events_and_telemetry_are_ordered_bounded_and_path_free(
    database, tmp_path
) -> None:
    _experiment_value, jobs = _experiment(database)
    repository = JobRepository(database)
    claimed = repository.claim_next("worker-one")
    assert claimed is not None
    for index in range(30):
        repository.append_event(
            claimed.job.id,
            claimed.attempt.id,
            "progress" if index < 29 else "error",
            {
                "phase": "generating",
                "step": index,
                "total": 30,
                "message": "/private/raw/model.safetensors " + "x" * 1000,
            },
        )
    repository.append_telemetry(
        TelemetrySample(
            attempt_id=claimed.attempt.id,
            recorded_at=datetime.now(UTC),
            elapsed_seconds=12.5,
            process_rss_bytes=2048,
            system_used_bytes=4096,
            system_available_bytes=8192,
            swap_used_bytes=512,
        )
    )

    job = next(
        job
        for job in _service(database, tmp_path).snapshot().experiments[0].jobs
        if job.id == claimed.job.id
    )
    assert job.attempt_count == 1
    assert len(job.evidence) <= 20
    assert [item.sequence for item in job.evidence] == sorted(
        item.sequence for item in job.evidence
    )
    assert job.progress == {"phase": "generating", "step": 28, "total": 30}
    assert job.telemetry is not None
    assert job.telemetry.elapsed_seconds == 12.5
    assert job.telemetry.process_rss_bytes == 2048
    serialized = json.dumps(job.model_dump(mode="json"))
    assert "/private/" not in serialized
    assert len(serialized) < 10_000


def test_retry_and_interruption_evidence_survive_requeue(database, tmp_path) -> None:
    _experiment_value, _jobs = _experiment(database)
    repository = JobRepository(database)
    first = repository.claim_next("worker-one")
    assert first is not None
    recovered = repository.recover_interrupted("worker-one")
    assert recovered[0].state is JobState.QUEUED
    second = repository.claim_next("worker-two")
    assert second is not None

    job = next(
        job
        for job in _service(database, tmp_path).snapshot().experiments[0].jobs
        if job.id == second.job.id
    )
    assert job.attempt_count == 2
    assert job.recovery_required is True
    assert any(item.kind == "worker_interrupted" for item in job.evidence)


def test_artifact_and_reviewability_are_metadata_only(database, tmp_path) -> None:
    experiment, jobs = _experiment(database)
    service = _service(database, tmp_path)
    initial = service.snapshot().experiments[0]
    assert initial.reviewable is False
    assert "jobs_not_completed" in initial.reviewability_reasons
    assert all(not job.artifacts for job in initial.jobs)
    assert initial.review_url is None
    assert experiment.id in initial.id


def test_reviewable_handoff_targets_blind_session_api_without_identity_in_url(
    database, tmp_path
) -> None:
    experiment, jobs = _experiment(database)
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    for job in jobs:
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state = 'completed' WHERE id = ?", (job.id,))
        source = tmp_path / f"{job.id}.mp4"
        source.write_bytes(job.id.encode())
        artifacts.commit(artifacts.stage(source, "video/mp4"), job.id)

    projected = _service(database, tmp_path).snapshot().experiments[0]

    assert projected.reviewable is True
    assert projected.review_url == "/api/reviews"
    assert experiment.id not in projected.review_url
