from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lab_tests.test_capabilities import manifest as base_manifest
from video_lab.adapter_protocol import (
    AdapterExecutionError,
    GenerationCancelled,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
)
from video_lab.adapter_registry import AdapterRegistry
from video_lab.adapters.fake import FakeVideoAdapter
from video_lab.artifacts import ArtifactStore
from video_lab.comparison import build_controlled_pair
from video_lab.database import Database
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    JobState,
    ReviewDimension,
    RubricScore,
    RunAttemptStatus,
)
from video_lab.normal_generation import RepositoryWorkerGenerationRunner
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.worker import Worker, WorkerOutcome, main


def manifest():
    return base_manifest().model_copy(
        update={
            "model_id": "video-lab/fake",
            "adapter_id": "video-lab.fake.v1",
            "width": 64,
            "height": 48,
            "scheduler": "fake",
            "dtype": "uint8",
            "backend": "cpu",
            "device": "cpu",
        }
    )


def create_job(database: Database) -> tuple[JobRepository, str]:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    stored = ExperimentRepository(database).create(pair)
    now = datetime.now(UTC).isoformat()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
            "VALUES ('job_worker', ?, 'queued', ?, ?)",
            (stored.variants[0].id, now, now),
        )
    return JobRepository(database), "job_worker"


def make_worker(
    database: Database,
    tmp_path: Path,
    registry: AdapterRegistry | None = None,
) -> tuple[Worker, JobRepository, ArtifactStore]:
    repository, _ = create_job(database)
    store = ArtifactStore(database, tmp_path / "artifacts")
    return (
        Worker(
            repository=repository,
            artifact_store=store,
            adapter_registry=registry or AdapterRegistry.with_defaults(),
            scratch_root=tmp_path / "scratch",
            worker_id="worker-test",
        ),
        repository,
        store,
    )


def test_worker_reports_no_job(database: Database, tmp_path: Path) -> None:
    worker = Worker(
        repository=JobRepository(database),
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
    )

    outcome = worker.run_once()

    assert outcome.status == "idle"
    assert outcome.job_id is None


def test_worker_completes_exactly_one_fake_job(database: Database, tmp_path: Path) -> None:
    worker, repository, artifact_store = make_worker(database, tmp_path)

    outcome = worker.run_once()

    assert outcome.status == "completed"
    assert outcome.job_id is not None
    assert repository.get(outcome.job_id).state is JobState.COMPLETED
    artifacts = artifact_store.list_for_job(outcome.job_id)
    assert len(artifacts) == 1
    assert artifacts[0].media_type == "video/mp4"
    attempt = repository.attempts_for_job(outcome.job_id)[0]
    assert attempt.status is RunAttemptStatus.COMPLETED
    assert attempt.artifact_id == artifacts[0].id
    assert len(repository.telemetry_for_attempt(attempt.id)) == 20
    events = repository.events_for_job(outcome.job_id)
    assert events[-1].kind == "job_completed"
    assert events[-1].payload["artifact_sha256"] == artifacts[0].sha256
    assert len(events[-1].payload["decoded_frame_sha256"]) == 17
    assert all(len(value) == 64 for value in events[-1].payload["decoded_frame_sha256"])
    process_event = next(event for event in events if event.kind == "worker_process_started")
    assert process_event.payload == {"worker_id": "worker-test", "process_id": os.getpid()}
    assert any(event.kind == "generation_progress" for event in events)
    assert [
        event.payload["completed"] for event in events if event.kind == "generation_progress"
    ] == list(range(1, 18))
    assert repository.manifest_for_job(outcome.job_id) == manifest()


def test_run_once_claims_only_one_of_two_queued_jobs(database: Database, tmp_path: Path) -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    _, jobs = ExperimentRepository(database).create_with_jobs(pair)
    worker = Worker(
        repository=JobRepository(database),
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
    )

    outcome = worker.run_once()

    assert outcome.status == "completed"
    assert [JobRepository(database).get(job.id).state for job in jobs].count(
        JobState.COMPLETED
    ) == 1
    assert [JobRepository(database).get(job.id).state for job in jobs].count(JobState.QUEUED) == 1


def test_worker_reuses_one_adapter_instance_across_multiple_jobs(
    database: Database, tmp_path: Path
) -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    ExperimentRepository(database).create_with_jobs(pair)
    factory_calls = 0

    def factory() -> FakeVideoAdapter:
        nonlocal factory_calls
        factory_calls += 1
        return FakeVideoAdapter()

    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", factory)
    worker = Worker(
        repository=JobRepository(database),
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=registry,
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
    )

    assert worker.run_once().status == "completed"
    assert worker.run_once().status == "completed"
    assert factory_calls == 1


def test_worker_run_until_idle_drains_queue_without_returning_idle_outcome(
    database: Database, tmp_path: Path
) -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    _, jobs = ExperimentRepository(database).create_with_jobs(pair)
    worker = Worker(
        repository=JobRepository(database),
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
    )

    outcomes = worker.run_until_idle()

    assert [outcome.status for outcome in outcomes] == ["completed", "completed"]
    assert {outcome.job_id for outcome in outcomes} == {job.id for job in jobs}
    assert worker.run_until_idle() == ()


def test_worker_run_until_idle_honors_positive_job_limit(
    database: Database, tmp_path: Path
) -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    ExperimentRepository(database).create_with_jobs(pair)
    worker = Worker(
        repository=JobRepository(database),
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
    )

    outcomes = worker.run_until_idle(max_jobs=1)

    assert len(outcomes) == 1
    assert outcomes[0].status == "completed"
    assert len(worker.run_until_idle()) == 1


def test_worker_run_until_idle_stops_after_worker_configuration_failure(
    database: Database, tmp_path: Path
) -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    _, jobs = ExperimentRepository(database).create_with_jobs(pair)
    repository = JobRepository(database)
    worker = Worker(
        repository=repository,
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
        ffprobe_resolver=lambda: None,
    )

    outcomes = worker.run_until_idle()

    assert len(outcomes) == 1
    assert outcomes[0].status == "failed"
    assert outcomes[0].error_code == "worker_configuration_error"
    assert [repository.get(job.id).state for job in sorted(jobs, key=lambda item: item.id)] == [
        JobState.FAILED,
        JobState.QUEUED,
    ]


@pytest.mark.parametrize("max_jobs", [0, -1])
def test_worker_run_until_idle_rejects_non_positive_job_limit(
    database: Database, tmp_path: Path, max_jobs: int
) -> None:
    worker = Worker(
        repository=JobRepository(database),
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
    )

    with pytest.raises(ValueError, match="max_jobs must be positive"):
        worker.run_until_idle(max_jobs=max_jobs)


class FailingAdapter(FakeVideoAdapter):
    def generate(self, request, progress, cancelled):
        raise AdapterExecutionError("adapter exploded")


class TrackingAdapter(FakeVideoAdapter):
    generated = False

    def generate(self, request, progress, cancelled):
        type(self).generated = True
        return super().generate(request, progress, cancelled)


class EvidenceFakeAdapter(FakeVideoAdapter):
    def generate(self, request, progress, cancelled):
        result = super().generate(request, progress, cancelled)
        return result.model_copy(
            update={
                "adapter_telemetry": {
                    "observed_configuration": {
                        "source_fingerprint": "a" * 64,
                        "placement_fingerprint": "b" * 64,
                    },
                    "fallback_violations": [],
                }
            }
        )


def test_repository_runner_loads_typed_committed_evidence_from_real_worker_path(
    database: Database, tmp_path: Path
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", EvidenceFakeAdapter)
    jobs = JobRepository(database)
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    runner = RepositoryWorkerGenerationRunner(
        experiments=ExperimentRepository(database),
        jobs=jobs,
        artifacts=artifacts,
        worker=Worker(
            repository=jobs,
            artifact_store=artifacts,
            adapter_registry=registry,
            scratch_root=tmp_path / "scratch",
            worker_id="qualification-worker",
        ),
    )

    evidence = runner.run_one(manifest())

    assert jobs.get(evidence.job_id).state is JobState.COMPLETED
    assert evidence.artifact["sha256"] == evidence.artifact_sha256
    assert len(evidence.artifact["decoded_frame_sha256"]) == 17
    assert evidence.source_fingerprint == "a" * 64
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1

    baseline = manifest()
    candidate = baseline.model_copy(update={"inference_steps": baseline.inference_steps + 1})
    unrelated = ExperimentRepository(database).create_single_generation(
        baseline.model_copy(update={"seed": baseline.seed + 99})
    )
    pair_evidence = runner.run_pair(baseline, candidate, variable_path="inference_steps")
    assert len(pair_evidence) == 2
    assert jobs.get(unrelated.id).state is JobState.QUEUED
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 4


def test_repository_runner_persists_guidance_pair_through_blinded_review_and_reveal(
    database: Database, tmp_path: Path
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", EvidenceFakeAdapter)
    jobs = JobRepository(database)
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    experiments = ExperimentRepository(database)
    runner = RepositoryWorkerGenerationRunner(
        experiments=experiments,
        jobs=jobs,
        artifacts=artifacts,
        worker=Worker(
            repository=jobs,
            artifact_store=artifacts,
            adapter_registry=registry,
            scratch_root=tmp_path / "scratch",
            worker_id="qualification-worker",
        ),
    )
    baseline = manifest().model_copy(update={"guidance_scale": 1.0})
    candidate = baseline.model_copy(update={"guidance_scale": 3.5})

    with pytest.raises(TypeError):
        runner.run_pair(baseline, candidate)

    first, second = runner.run_pair(baseline, candidate, variable_path="guidance_scale")

    assert first.experiment_id is not None
    assert first.experiment_id == second.experiment_id
    experiment = experiments.get(first.experiment_id)
    assert experiment.variable_path == "guidance_scale"
    assert experiment.changed_paths == ("guidance_scale",)
    assert tuple(jobs.list_for_experiment(experiment.id)) == tuple(
        jobs.get(evidence.job_id) for evidence in (first, second)
    )
    assert [item.manifest.guidance_scale for item in experiment.variants] == [1.0, 3.5]

    review = ReviewService(experiments, jobs, artifacts, ReviewRepository(database))
    view = review.start(experiment.id, "reviewer")
    assert tuple(media.display_position for media in view.media) == ("left", "right")
    presentation = json.dumps(view.model_dump(mode="json"))
    assert "guidance_scale" not in presentation
    assert "3.5" not in presentation
    submission = review.submit(
        view.session_id,
        "reviewer",
        DisplayPreference.LEFT,
        tuple(
            RubricScore(display_position=position, dimension=dimension, score=4)
            for position in ("left", "right")
            for dimension in ReviewDimension
        ),
        "guided comparison",
    )
    reveal = review.reveal(view.session_id, "reviewer")

    assert submission.status == "submitted"
    assert reveal.variable_path == "guidance_scale"
    assert (reveal.baseline_value, reveal.alternate_value) == (1.0, 3.5)
    assert reveal.structural_diff == ("guidance_scale",)
    assert set(reveal.canonical_by_display.values()) == {"A", "B"}


def test_repository_runner_preserves_worker_failure_detail(
    database: Database, tmp_path: Path
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", FailingAdapter)
    jobs = JobRepository(database)
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    runner = RepositoryWorkerGenerationRunner(
        experiments=ExperimentRepository(database),
        jobs=jobs,
        artifacts=artifacts,
        worker=Worker(
            repository=jobs,
            artifact_store=artifacts,
            adapter_registry=registry,
            scratch_root=tmp_path / "scratch",
            worker_id="qualification-worker",
        ),
    )

    with pytest.raises(RuntimeError, match="adapter exploded"):
        runner.run_one(manifest())


def test_missing_ffprobe_fails_cleanly_before_adapter_generation(
    database: Database, tmp_path: Path
) -> None:
    TrackingAdapter.generated = False
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", TrackingAdapter)
    worker, repository, store = make_worker(database, tmp_path, registry)
    worker.ffprobe_resolver = lambda: None

    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "worker_configuration_error"
    assert TrackingAdapter.generated is False
    assert repository.get("job_worker").state is JobState.FAILED
    attempt = repository.attempts_for_job("job_worker")[0]
    assert attempt.status is RunAttemptStatus.FAILED
    assert attempt.error_message == "ffprobe is required on PATH"
    assert store.list_for_job("job_worker") == ()


class EveryPhaseAdapter(FakeVideoAdapter):
    def generate(self, request, progress, cancelled):
        progress(
            GenerationProgress(
                phase="downloading", completed=1, total=1, message="ready", elapsed_seconds=0
            )
        )
        progress(
            GenerationProgress(
                phase="loading", completed=1, total=1, message="loaded", elapsed_seconds=0
            )
        )
        return super().generate(request, progress, cancelled)


def test_worker_accepts_progress_through_downloading_loading_and_generating(
    database: Database, tmp_path: Path
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", EveryPhaseAdapter)
    worker, repository, _ = make_worker(database, tmp_path, registry)

    assert worker.run_once().status == "completed"
    phases = [
        event.payload["phase"]
        for event in repository.events_for_job("job_worker")
        if event.kind == "generation_progress"
    ]
    assert phases[:3] == ["downloading", "loading", "generating"]


def test_worker_maps_adapter_failure_and_registers_no_partial(
    database: Database, tmp_path: Path
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", FailingAdapter)
    worker, repository, store = make_worker(database, tmp_path, registry)

    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "adapter_execution_failed"
    assert repository.get("job_worker").state is JobState.FAILED
    assert repository.attempts_for_job("job_worker")[0].status is RunAttemptStatus.FAILED
    assert store.list_for_job("job_worker") == ()
    assert repository.events_for_job("job_worker")[-1].kind == "job_failed"


class FailingTelemetrySampler:
    def sample(self, attempt_id, elapsed_seconds):
        raise RuntimeError("telemetry unavailable")


def test_telemetry_failure_never_prevents_success(database: Database, tmp_path: Path) -> None:
    worker, repository, _ = make_worker(database, tmp_path)
    worker.telemetry_sampler = FailingTelemetrySampler()

    outcome = worker.run_once()

    assert outcome.status == "completed"
    assert repository.get("job_worker").state is JobState.COMPLETED
    assert "telemetry_error" in [event.kind for event in repository.events_for_job("job_worker")]


def test_append_telemetry_failure_never_prevents_success(
    database: Database, tmp_path: Path
) -> None:
    worker, repository, _ = make_worker(database, tmp_path)

    def fail_append(_sample):
        raise OSError("database telemetry failure")

    repository.append_telemetry = fail_append  # type: ignore[method-assign]

    outcome = worker.run_once()

    assert outcome.status == "completed"
    assert repository.get("job_worker").state is JobState.COMPLETED
    assert "telemetry_error" in [event.kind for event in repository.events_for_job("job_worker")]


def test_telemetry_error_event_failure_never_masks_success(
    database: Database, tmp_path: Path
) -> None:
    worker, repository, _ = make_worker(database, tmp_path)
    worker.telemetry_sampler = FailingTelemetrySampler()
    append_event = repository.append_event

    def fail_telemetry_event(job_id, attempt_id, kind, payload):
        if kind == "telemetry_error":
            raise OSError("event persistence failed")
        return append_event(job_id, attempt_id, kind, payload)

    repository.append_event = fail_telemetry_event  # type: ignore[method-assign]

    assert worker.run_once().status == "completed"
    assert repository.get("job_worker").state is JobState.COMPLETED


class FailTerminalTelemetry:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self, attempt_id, elapsed_seconds):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("terminal telemetry failed")
        from video_lab.telemetry import TelemetrySampler

        return TelemetrySampler().sample(attempt_id, elapsed_seconds)


def test_terminal_telemetry_failure_preserves_adapter_error(
    database: Database, tmp_path: Path
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", FailingAdapter)
    worker, repository, _ = make_worker(database, tmp_path, registry)
    worker.telemetry_sampler = FailTerminalTelemetry()

    outcome = worker.run_once()

    assert outcome.error_code == "adapter_execution_failed"
    attempt = repository.attempts_for_job("job_worker")[0]
    assert attempt.status is RunAttemptStatus.FAILED
    assert attempt.error_message == "adapter exploded"


class InvalidOutputAdapter(FakeVideoAdapter):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def generate(self, request: GenerationRequest, progress, cancelled) -> GenerationResult:
        path = request.scratch_root / "output.mp4"
        if self.mode != "missing":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not an mp4")
        return GenerationResult(
            video_path=path,
            effective_manifest=request.manifest,
            frame_count=17,
            fps=8,
            width=64,
            height=48,
            adapter_version="1",
            runtime_versions={},
            adapter_telemetry={},
        )


class MetadataMismatchAdapter(FakeVideoAdapter):
    def generate(self, request, progress, cancelled):
        result = super().generate(request, progress, cancelled)
        return result.model_copy(update={"width": result.width + 1})


class RenamedAviAdapter(FakeVideoAdapter):
    def generate(self, request, progress, cancelled):
        request.scratch_root.mkdir(parents=True, exist_ok=True)
        path = request.scratch_root / "output.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x48:r=8",
                "-frames:v",
                "17",
                "-f",
                "avi",
                str(path),
            ],
            check=True,
        )
        return GenerationResult(
            video_path=path,
            effective_manifest=request.manifest,
            frame_count=17,
            fps=8,
            width=64,
            height=48,
            adapter_version="1",
            runtime_versions={},
            adapter_telemetry={},
        )


class EscapingOutputAdapter(FakeVideoAdapter):
    def __init__(self, *, symlink: bool) -> None:
        super().__init__()
        self.symlink = symlink

    def generate(self, request, progress, cancelled):
        result = super().generate(request, progress, cancelled)
        outside = request.scratch_root.parent.parent / "outside.mp4"
        result.video_path.replace(outside)
        if self.symlink:
            result.video_path.symlink_to(outside)
            return result
        return result.model_copy(update={"video_path": outside})


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_worker_rejects_missing_or_structurally_corrupt_output(
    database: Database, tmp_path: Path, mode: str
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", lambda: InvalidOutputAdapter(mode))
    worker, repository, store = make_worker(database, tmp_path, registry)

    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "invalid_output"
    assert repository.get("job_worker").state is JobState.FAILED
    assert store.list_for_job("job_worker") == ()


def test_worker_rejects_structural_metadata_mismatch(database: Database, tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", MetadataMismatchAdapter)
    worker, _, store = make_worker(database, tmp_path, registry)

    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "invalid_output"
    assert store.list_for_job("job_worker") == ()


def test_worker_rejects_renamed_non_mp4_container(database: Database, tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", RenamedAviAdapter)
    worker, _, store = make_worker(database, tmp_path, registry)

    outcome = worker.run_once()

    assert outcome.error_code == "invalid_output"
    assert store.list_for_job("job_worker") == ()


@pytest.mark.parametrize("major_brand", ["qt  ", None])
def test_worker_accepts_credible_h264_mp4_independent_of_major_brand(
    database: Database,
    tmp_path: Path,
    monkeypatch,
    major_brand: str | None,
) -> None:
    worker, repository, store = make_worker(database, tmp_path)
    actual_run = subprocess.run

    def quicktime_probe(command, **kwargs):
        if command[0].endswith("ffprobe"):
            tags = "" if major_brand is None else f',"tags":{{"major_brand":"{major_brand}"}}'
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    '{"format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2","duration":"2.125"'
                    + tags
                    + '},"streams":['
                    '{"codec_type":"video","codec_name":"h264","width":64,"height":48,'
                    '"avg_frame_rate":"8/1","nb_frames":"17"}]}'
                ),
                stderr="",
            )
        return actual_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", quicktime_probe)

    outcome = worker.run_once()

    assert outcome.status == "completed"
    assert repository.attempts_for_job("job_worker")[0].status is RunAttemptStatus.COMPLETED
    assert len(store.list_for_job("job_worker")) == 1


@pytest.mark.parametrize("symlink", [False, True])
def test_worker_rejects_output_outside_attempt_scratch(
    database: Database, tmp_path: Path, symlink: bool
) -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", lambda: EscapingOutputAdapter(symlink=symlink))
    worker, _, store = make_worker(database, tmp_path, registry)

    outcome = worker.run_once()

    assert outcome.error_code == "invalid_output"
    assert store.list_for_job("job_worker") == ()


class BlockingAdapter(FakeVideoAdapter):
    entered = threading.Event()

    def generate(self, request, progress, cancelled):
        progress(
            type(
                "Progress",
                (),
                {
                    "phase": "generating",
                    "completed": 1,
                    "total": 17,
                    "message": "blocked",
                    "elapsed_seconds": 0.01,
                },
            )()
        )
        self.entered.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if cancelled():
                raise GenerationCancelled("requested")
            time.sleep(0.01)
        raise AssertionError("cancellation was not observed")


def test_cancellation_from_second_database_connection_never_registers_partial(
    database: Database, tmp_path: Path
) -> None:
    BlockingAdapter.entered.clear()
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", BlockingAdapter)
    worker, repository, store = make_worker(database, tmp_path, registry)
    outcomes = []
    thread = threading.Thread(target=lambda: outcomes.append(worker.run_once()))

    thread.start()
    assert BlockingAdapter.entered.wait(timeout=5)
    second = JobRepository(Database(database.path))
    second.request_cancellation("job_worker")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcomes[0].status == "cancelled"
    assert repository.get("job_worker").state is JobState.CANCELLED
    assert repository.attempts_for_job("job_worker")[0].status is RunAttemptStatus.CANCELLED
    assert store.list_for_job("job_worker") == ()


class CancellingArtifactStore(ArtifactStore):
    def commit(self, staged, job_id):
        artifact = super().commit(staged, job_id)
        JobRepository(Database(self.database.path)).request_cancellation("job_worker")
        return artifact


def test_cancellation_after_content_commit_still_never_associates_artifact(
    database: Database, tmp_path: Path
) -> None:
    repository, _ = create_job(database)
    store = CancellingArtifactStore(database, tmp_path / "artifacts")
    worker = Worker(
        repository=repository,
        artifact_store=store,
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="worker-test",
    )

    outcome = worker.run_once()

    assert outcome.status == "cancelled"
    assert store.list_for_job("job_worker") == ()
    assert repository.attempts_for_job("job_worker")[0].artifact_id is None


def test_startup_recovery_records_interruption_then_allows_retry(
    database: Database, tmp_path: Path
) -> None:
    repository, job_id = create_job(database)
    claimed = repository.claim_next("dead-worker")
    assert claimed is not None
    repository.transition(job_id, JobState.LOADING)
    worker = Worker(
        repository=JobRepository(Database(database.path)),
        artifact_store=ArtifactStore(database, tmp_path / "artifacts"),
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=tmp_path / "scratch",
        worker_id="dead-worker",
    )

    recovered = worker.recover()
    outcome = worker.run_once()

    assert [job.id for job in recovered] == [job_id]
    assert outcome.status == "completed"
    attempts = repository.attempts_for_job(job_id)
    assert [attempt.status for attempt in attempts] == [
        RunAttemptStatus.INTERRUPTED,
        RunAttemptStatus.COMPLETED,
    ]
    assert "worker_interrupted" in [event.kind for event in repository.events_for_job(job_id)]


def worker_args(tmp_path: Path, worker_id: str | None) -> list[str]:
    args = [
        "video-lab-worker",
        "--database",
        str(tmp_path / "entry.sqlite3"),
        "--artifacts",
        str(tmp_path / "entry-artifacts"),
        "--scratch",
        str(tmp_path / "entry-scratch"),
    ]
    if worker_id is not None:
        args.extend(("--worker-id", worker_id))
    return args


@pytest.mark.parametrize("worker_id", [None, "   "])
def test_process_entry_requires_nonblank_worker_id(
    tmp_path: Path, monkeypatch, worker_id: str | None
) -> None:
    monkeypatch.setattr(sys, "argv", worker_args(tmp_path, worker_id))

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code != 0
    assert not (tmp_path / "entry.sqlite3").exists()


def test_process_entry_until_idle_processes_multiple_jobs_in_one_worker(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    database = Database(tmp_path / "entry.sqlite3")
    database.migrate()
    repository = JobRepository(database)
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    _, jobs = ExperimentRepository(database).create_with_jobs(pair)
    monkeypatch.setattr(
        sys,
        "argv",
        [*worker_args(tmp_path, "stable-worker"), "--until-idle", "--max-jobs", "2"],
    )

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "completed",
        "processed": 2,
        "completed": 2,
        "failed": 0,
        "cancelled": 0,
        "outcomes": [
            {
                "status": "completed",
                "job_id": outcome.id,
                "attempt_id": repository.attempts_for_job(outcome.id)[0].id,
                "error_code": None,
            }
            for outcome in sorted(jobs, key=lambda item: item.id)
        ],
    }


def test_process_entry_rejects_max_jobs_without_until_idle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [*worker_args(tmp_path, "stable-worker"), "--max-jobs", "2"],
    )

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code != 0
    assert not (tmp_path / "entry.sqlite3").exists()


def test_process_entry_same_stable_id_recovers_across_invocations(
    tmp_path: Path, monkeypatch
) -> None:
    database = Database(tmp_path / "entry.sqlite3")
    database.migrate()
    repository, job_id = create_job(database)
    original_run_once = Worker.run_once

    def abandon(self):
        claimed = self.repository.claim_next(self.worker_id)
        assert claimed is not None
        self.repository.transition(claimed.job.id, JobState.LOADING)
        return WorkerOutcome("idle")

    monkeypatch.setattr(Worker, "run_once", abandon)
    monkeypatch.setattr(sys, "argv", worker_args(tmp_path, "stable-worker"))
    assert main() == 0
    monkeypatch.setattr(Worker, "run_once", original_run_once)

    assert main() == 0

    assert repository.get(job_id).state is JobState.COMPLETED
    assert [attempt.status for attempt in repository.attempts_for_job(job_id)] == [
        RunAttemptStatus.INTERRUPTED,
        RunAttemptStatus.COMPLETED,
    ]


def test_process_entry_different_id_cannot_recover_invocation(tmp_path: Path, monkeypatch) -> None:
    database = Database(tmp_path / "entry.sqlite3")
    database.migrate()
    repository, job_id = create_job(database)

    def abandon(self):
        claimed = self.repository.claim_next(self.worker_id)
        assert claimed is not None
        return WorkerOutcome("idle")

    monkeypatch.setattr(Worker, "run_once", abandon)
    monkeypatch.setattr(sys, "argv", worker_args(tmp_path, "worker-a"))
    assert main() == 0
    monkeypatch.setattr(Worker, "run_once", lambda self: WorkerOutcome("idle"))
    monkeypatch.setattr(sys, "argv", worker_args(tmp_path, "worker-b"))

    assert main() == 0

    assert repository.get(job_id).state is JobState.PREPARING
    assert repository.attempts_for_job(job_id)[0].status is RunAttemptStatus.ACTIVE
