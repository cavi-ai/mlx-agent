from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from video_lab.adaptive_study_models import (
    AdaptiveObservationRequest,
    AdaptiveStudyDefinition,
    AdaptiveStudyState,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import (
    AdaptiveStudyConflict,
    AdaptiveStudyRepository,
)
from video_lab.artifacts import ArtifactStore
from video_lab.capabilities import AdapterCapabilities
from video_lab.database import Database
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    JobState,
    Manifest,
    ReviewDimension,
    RubricScore,
    RunAttemptStatus,
    utc_now,
)
from video_lab.preference_optimizer import (
    CandidatePair,
    OptimizerConfig,
    OptimizerDecision,
    StoppingDecision,
    StopReason,
    optimize_preference,
)
from video_lab.queue_service import QueueService
from video_lab.repositories import (
    AdaptiveExperimentConflict,
    ExperimentRepository,
    JobRepository,
    RecordNotFound,
)
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.services import ExperimentService, ValidationFailure

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _adaptive_evidence_snapshot(
    database: Database, tmp_path: Path, study_id: str
) -> tuple[tuple[int, ...], str]:
    repository = AdaptiveStudyRepository(database)
    events = repository.list_events(study_id)
    observations = repository.list_observations(study_id)
    steps = repository.list_optimizer_steps(study_id)
    experiments = experiment_service(database, tmp_path).list_adaptive_for_study(study_id)
    with database.connect() as connection:
        revision_count = int(
            connection.execute("SELECT COUNT(*) FROM review_revisions").fetchone()[0]
        )
    payload = json.dumps(
        {
            "events": [item.model_dump(mode="json") for item in events],
            "observations": [item.model_dump(mode="json") for item in observations],
            "optimizer_steps": [item.model_dump(mode="json") for item in steps],
            "experiments": [
                {
                    "experiment": item.experiment.model_dump(mode="json"),
                    "jobs": [job.model_dump(mode="json") for job in item.jobs],
                }
                for item in experiments
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        (len(events), len(observations), len(steps), len(experiments), revision_count),
        hashlib.sha256(payload).hexdigest(),
    )


def make_manifest() -> Manifest:
    return Manifest(
        model_id="org/model",
        model_revision="abc123",
        model_checksum="sha256:weights",
        adapter_id="diffusers-mps",
        adapter_version="0.1.0",
        runtime_versions={"python": "3.12.0"},
        prompt="a blue-haired character walking",
        negative_prompt="flicker",
        seed=42,
        width=512,
        height=288,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=1.0,
        scheduler="flow-match-euler",
        dtype="float16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="unified",
        hardware_fingerprint="apple-m5-max-128gb",
        os_fingerprint="macos-test",
    )


def capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_id="diffusers-mps",
        model_patterns=("org/*",),
        backends=("mps",),
        dtypes=("float16",),
        quantizations=("none",),
        schedulers=("flow-match-euler",),
        min_width=256,
        max_width=1024,
        width_multiple=16,
        min_height=256,
        max_height=1024,
        height_multiple=16,
        min_frames=1,
        max_frames=81,
        frame_rule="16n+1",
        supports_lora=False,
    )


def experiment_service(database: Database, tmp_path: Path) -> ExperimentService:
    return ExperimentService.from_paths(database.path, tmp_path / "artifacts")


def guidance_draft() -> ExperimentDraft:
    return ExperimentDraft(
        baseline=make_manifest().model_copy(update={"guidance_scale": 2.0}),
        variable_path="guidance_scale",
        alternate_value=4.0,
    )


def rubric_scores() -> tuple[RubricScore, ...]:
    return tuple(
        RubricScore(display_position=position, dimension=dimension, score=3)
        for position in ("left", "right")
        for dimension in ReviewDimension
    )


def make_definition(study_id: str = "study_1") -> AdaptiveStudyDefinition:
    manifest = make_manifest()
    config = OptimizerConfig(posterior_draws=256)
    return AdaptiveStudyDefinition(
        id=study_id,
        prompt=manifest.prompt,
        negative_prompt=manifest.negative_prompt,
        seed=manifest.seed,
        base_manifest=manifest,
        optimizer_config=config,
        input_fingerprint=adaptive_study_input_fingerprint(manifest, config),
        created_at=NOW,
    )


def review_service(database: Database, tmp_path: Path) -> ReviewService:
    return ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        ArtifactStore(database, tmp_path / "artifacts"),
        ReviewRepository(database),
    )


def adaptive_service(database: Database, tmp_path: Path, *, optimizer=None):
    from video_lab.adaptive_study_service import AdaptiveStudyService

    kwargs = {} if optimizer is None else {"optimizer": optimizer}
    return AdaptiveStudyService(
        AdaptiveStudyRepository(database),
        experiment_service(database, tmp_path),
        review_service(database, tmp_path),
        capabilities(),
        reviewer_id="reviewer",
        **kwargs,
    )


def complete_generation(database: Database, tmp_path: Path, study_id: str) -> None:
    experiments = experiment_service(database, tmp_path).list_adaptive_for_study(study_id)
    experiment = next(
        item
        for item in experiments
        if item.experiment.id
        == str(AdaptiveStudyRepository(database).list_events(study_id)[-1].payload["experiment_id"])
    )
    store = ArtifactStore(database, tmp_path / "artifacts")
    for index, job in enumerate(experiment.jobs):
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state = 'completed' WHERE id = ?", (job.id,))
        source = tmp_path / f"{study_id}-{index}.mp4"
        source.write_bytes(f"video:{study_id}:{index}".encode())
        store.commit(store.stage(source, "video/mp4"), job.id)


def test_adaptive_experiment_creation_is_idempotent_and_domain_separated(
    database: Database, tmp_path: Path
) -> None:
    service = experiment_service(database, tmp_path)

    first = service.create_adaptive(
        guidance_draft(),
        capabilities(),
        study_id="study_1",
        optimizer_output_digest="a" * 64,
    )
    retried = service.create_adaptive(
        guidance_draft(),
        capabilities(),
        study_id="study_1",
        optimizer_output_digest="a" * 64,
    )
    other_study = service.create_adaptive(
        guidance_draft(),
        capabilities(),
        study_id="study_2",
        optimizer_output_digest="a" * 64,
    )
    other_step = service.create_adaptive(
        guidance_draft(),
        capabilities(),
        study_id="study_1",
        optimizer_output_digest="b" * 64,
    )

    assert retried == first
    assert len({first.experiment.id, other_study.experiment.id, other_step.experiment.id}) == 3
    all_first_ids = {
        first.experiment.id,
        *(variant.id for variant in first.experiment.variants),
        *(job.id for job in first.jobs),
    }
    assert len(all_first_ids) == 5
    assert all(identifier.startswith("adaptive_") for identifier in all_first_ids)
    assert len(service.list()) == 3


def test_adaptive_experiments_are_excluded_from_every_generic_projection(
    database: Database, tmp_path: Path
) -> None:
    generic = experiment_service(database, tmp_path).create(
        ExperimentDraft(
            baseline=make_manifest(),
            variable_path="inference_steps",
            alternate_value=8,
        ),
        capabilities(),
    )
    adaptive = adaptive_service(database, tmp_path).create_study(make_definition())
    event = AdaptiveStudyRepository(database).list_events(adaptive.study_id)[-1]
    adaptive_id = str(event.payload["experiment_id"])
    experiments = experiment_service(database, tmp_path)

    assert tuple(item.experiment.id for item in experiments.list()) == (generic.experiment.id,)
    with pytest.raises(RecordNotFound):
        experiments.get(adaptive_id)
    queue = QueueService(
        ExperimentRepository(database),
        JobRepository(database),
        ArtifactStore(database, tmp_path / "artifacts"),
        review_service(database, tmp_path),
    ).snapshot()
    assert tuple(item.id for item in queue.experiments) == (generic.experiment.id,)
    assert tuple(item.experiment_id for item in review_service(database, tmp_path).queue()) == (
        generic.experiment.id,
    )
    serialized = (
        json.dumps(
            [item.experiment.model_dump(mode="json") for item in experiments.list()],
            sort_keys=True,
        )
        + queue.model_dump_json()
    )
    assert adaptive_id not in serialized
    assert '"guidance_scale":2.0' not in serialized
    assert '"guidance_scale":4.0' not in serialized


def test_create_study_persists_empty_decision_then_one_controlled_pair(
    database: Database, tmp_path: Path
) -> None:
    from video_lab.adaptive_study_service import AdaptiveStudyService

    service = AdaptiveStudyService(
        AdaptiveStudyRepository(database),
        experiment_service(database, tmp_path),
        review_service(database, tmp_path),
        capabilities(),
        reviewer_id="reviewer",
    )

    view = service.create_study(make_definition())

    assert view.state is AdaptiveStudyState.GENERATING_PAIR
    assert view.next_action == "complete_generation"
    assert view.comparison_count == 0
    assert view.review_id is None
    steps = AdaptiveStudyRepository(database).list_optimizer_steps("study_1")
    assert len(steps) == 1
    assert steps[0].selected_pair is not None
    assert (steps[0].selected_pair.value_a, steps[0].selected_pair.value_b) == (2.0, 4.0)
    experiments = experiment_service(database, tmp_path).list_adaptive_for_study("study_1")
    assert len(experiments) == 1
    manifests = tuple(variant.manifest for variant in experiments[0].experiment.variants)
    assert tuple(item.guidance_scale for item in manifests) == (2.0, 4.0)
    for manifest in manifests:
        normalized = manifest.model_copy(
            update={"guidance_scale": make_definition().base_manifest.guidance_scale}
        )
        assert normalized == make_definition().base_manifest
    assert all(job.state.value == "queued" for job in experiments[0].jobs)
    assert "selected_pair" not in view.model_dump(mode="json")


def test_resume_after_restart_creates_one_sealed_review_session(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")

    reopened_database = Database(database.path)
    reopened_database.migrate()
    reopened = adaptive_service(reopened_database, tmp_path)
    first = reopened.resume("study_1")
    retried = reopened.resume("study_1")

    assert first == retried
    assert first.state is AdaptiveStudyState.AWAITING_PREFERENCE
    assert first.next_action == "record_preference"
    assert first.review_id is not None
    assert first.recommendation is None
    assert first.concentration_interval is None
    assert "guidance" not in first.model_dump_json()
    with reopened_database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM review_sessions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_awaiting_event_persists_the_complete_versioned_sealed_review_binding(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")

    awaiting = service.resume("study_1")

    assert awaiting.review_id is not None
    event = AdaptiveStudyRepository(database).list_events("study_1")[-1]
    assert event.new_state is AdaptiveStudyState.AWAITING_PREFERENCE
    assert event.payload == {
        "binding_schema_version": "adaptive-review-binding.v1",
        "study_id": "study_1",
        "optimizer_step_ordinal": 1,
        "optimizer_output_digest": AdaptiveStudyRepository(database)
        .list_optimizer_steps("study_1")[0]
        .output_digest,
        "experiment_id": str(
            AdaptiveStudyRepository(database).list_events("study_1")[-2].payload["experiment_id"]
        ),
        "review_session_id": awaiting.review_id,
        "reviewer_id": "reviewer",
        "sealed_mapping_digest": event.payload["sealed_mapping_digest"],
    }
    assert len(str(event.payload["sealed_mapping_digest"])) == 64


def test_record_preference_is_immutable_idempotent_and_derived_from_sealed_display(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None

    advanced = service.record_preference("study_1", awaiting.review_id, DisplayPreference.LEFT)
    retried = service.record_preference("study_1", awaiting.review_id, DisplayPreference.LEFT)

    assert advanced == retried
    assert advanced.state is AdaptiveStudyState.GENERATING_PAIR
    assert advanced.comparison_count == 1
    repository = AdaptiveStudyRepository(database)
    observations = repository.list_observations("study_1")
    assert len(observations) == 1
    with database.connect() as connection:
        binding = connection.execute(
            "SELECT left_variant_id FROM review_sessions WHERE id = ?",
            (awaiting.review_id,),
        ).fetchone()
        manifest_row = connection.execute(
            "SELECT manifest_json FROM variants WHERE id = ?",
            (binding["left_variant_id"],),
        ).fetchone()
        left_guidance = Manifest.model_validate_json(
            str(manifest_row["manifest_json"])
        ).guidance_scale
        assert connection.execute("SELECT COUNT(*) FROM review_revisions").fetchone()[0] == 1
    expected_outcome = "a" if left_guidance == observations[0].pair.value_a else "b"
    assert observations[0].outcome.value == expected_outcome
    assert len(repository.list_optimizer_steps("study_1")) == 2
    assert len(experiment_service(database, tmp_path).list_adaptive_for_study("study_1")) == 2

    with pytest.raises(ValueError, match="immutable"):
        service.record_preference("study_1", awaiting.review_id, DisplayPreference.RIGHT)


def test_historical_preference_retry_is_a_pure_read_while_next_comparison_is_active(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    first = service.resume("study_1")
    assert first.review_id is not None
    service.record_preference("study_1", first.review_id, DisplayPreference.TIE)
    complete_generation(database, tmp_path, "study_1")
    current = service.resume("study_1")
    assert current.state is AdaptiveStudyState.AWAITING_PREFERENCE
    assert current.review_id is not None and current.review_id != first.review_id
    before = _adaptive_evidence_snapshot(database, tmp_path, "study_1")

    retried = service.record_preference("study_1", first.review_id, DisplayPreference.TIE)

    assert retried == current
    assert _adaptive_evidence_snapshot(database, tmp_path, "study_1") == before
    with pytest.raises(ValueError, match="immutable"):
        service.record_preference("study_1", first.review_id, DisplayPreference.LEFT)
    assert _adaptive_evidence_snapshot(database, tmp_path, "study_1") == before


def test_historical_preference_retry_is_a_pure_read_after_pause(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    first = service.resume("study_1")
    assert first.review_id is not None
    service.record_preference("study_1", first.review_id, DisplayPreference.TIE)
    event = AdaptiveStudyRepository(database).list_events("study_1")[-1]
    paused = service.pause_generation_failure(
        "study_1", str(event.payload["experiment_id"]), error_code="adapter_failed"
    )
    before = _adaptive_evidence_snapshot(database, tmp_path, "study_1")

    retried = service.record_preference("study_1", first.review_id, DisplayPreference.TIE)

    assert retried == paused
    assert _adaptive_evidence_snapshot(database, tmp_path, "study_1") == before
    with pytest.raises(ValueError, match="immutable"):
        service.record_preference("study_1", first.review_id, DisplayPreference.RIGHT)
    assert _adaptive_evidence_snapshot(database, tmp_path, "study_1") == before


def test_historical_preference_retry_is_a_pure_read_after_terminal_state(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    first_review_id: str | None = None
    terminal = None
    for _ in range(10):
        complete_generation(database, tmp_path, "study_1")
        awaiting = service.resume("study_1")
        assert awaiting.review_id is not None
        first_review_id = first_review_id or awaiting.review_id
        terminal = service.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)
    assert first_review_id is not None
    assert terminal is not None
    assert terminal.state is AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED
    before = _adaptive_evidence_snapshot(database, tmp_path, "study_1")

    retried = service.record_preference("study_1", first_review_id, DisplayPreference.TIE)

    assert retried == terminal
    assert _adaptive_evidence_snapshot(database, tmp_path, "study_1") == before
    with pytest.raises(ValueError, match="immutable"):
        service.record_preference("study_1", first_review_id, DisplayPreference.LEFT)
    assert _adaptive_evidence_snapshot(database, tmp_path, "study_1") == before


def test_unconsumed_review_must_be_the_current_sealed_comparison(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    service.create_study(make_definition("study_2"))
    complete_generation(database, tmp_path, "study_2")
    extra = service.resume("study_2")
    assert extra.review_id is not None
    before = _adaptive_evidence_snapshot(database, tmp_path, "study_1")

    with pytest.raises(ValueError, match="sealed comparison"):
        service.record_preference("study_1", extra.review_id, DisplayPreference.TIE)

    assert extra.review_id != awaiting.review_id
    assert _adaptive_evidence_snapshot(database, tmp_path, "study_1") == before


def test_generation_failure_pauses_without_consuming_budget_and_resumes_same_pair(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    generating = service.create_study(make_definition())
    event = AdaptiveStudyRepository(database).list_events("study_1")[-1]
    experiment_id = str(event.payload["experiment_id"])

    paused = service.pause_generation_failure(
        "study_1", experiment_id, error_code="adapter_setup_failed"
    )
    duplicate = service.pause_generation_failure(
        "study_1", experiment_id, error_code="adapter_setup_failed"
    )

    assert paused == duplicate
    assert paused.state is AdaptiveStudyState.PAUSED
    assert paused.next_action == "resume"
    assert paused.comparison_count == 0
    repository = AdaptiveStudyRepository(database)
    assert repository.list_observations("study_1") == ()
    assert len(repository.list_optimizer_steps("study_1")) == 1
    assert len(repository.list_events("study_1")) == 3

    resumed = service.resume("study_1")
    assert resumed == generating
    assert len(experiment_service(database, tmp_path).list_adaptive_for_study("study_1")) == 1
    assert len(AdaptiveStudyRepository(database).list_events("study_1")) == 4


def test_failed_adaptive_job_requeues_with_evidence_and_reaches_awaiting_review(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    generating = service.create_study(make_definition())
    event = AdaptiveStudyRepository(database).list_events("study_1")[-1]
    experiment_id = str(event.payload["experiment_id"])
    jobs = JobRepository(database)
    pair_jobs = experiment_service(database, tmp_path).get_adaptive(experiment_id).jobs
    claimed = jobs.claim_next("worker-1", pair_jobs[0].id)
    assert claimed is not None
    jobs.finalize_attempt(
        claimed.job.id,
        claimed.attempt.id,
        job_state=JobState.FAILED,
        attempt_status=RunAttemptStatus.FAILED,
        artifact_id=None,
        error_code="generation_failed",
        error_message="forced adaptive failure",
        event_kind="generation_failed",
        event_payload={"phase": "generating"},
    )
    attempts_before = jobs.attempts_for_job(pair_jobs[0].id)
    events_before = jobs.events_for_job(pair_jobs[0].id)
    service.pause_generation_failure("study_1", experiment_id, error_code="generation_failed")

    resumed = service.resume("study_1")

    assert resumed == generating
    assert jobs.get(pair_jobs[0].id).state is JobState.QUEUED
    assert jobs.get(pair_jobs[1].id).state is JobState.QUEUED
    assert jobs.attempts_for_job(pair_jobs[0].id) == attempts_before
    requeue_events = jobs.events_for_job(pair_jobs[0].id)
    assert requeue_events[: len(events_before)] == events_before
    assert requeue_events[-1].kind == "adaptive_retry_queued"
    with database.connect() as connection:
        row = connection.execute(
            "SELECT error_code, error_message, claimed_by, claimed_at FROM jobs WHERE id = ?",
            (pair_jobs[0].id,),
        ).fetchone()
    assert tuple(row) == (None, None, None, None)
    event_count = len(requeue_events)
    assert service.resume("study_1") == generating
    assert len(jobs.events_for_job(pair_jobs[0].id)) == event_count

    store = ArtifactStore(database, tmp_path / "artifacts")
    for index, pair_job in enumerate(pair_jobs):
        next_attempt = jobs.claim_next("worker-2", pair_job.id)
        assert next_attempt is not None
        for state in (
            JobState.DOWNLOADING,
            JobState.LOADING,
            JobState.GENERATING,
            JobState.ENCODING,
        ):
            jobs.transition(pair_job.id, state)
        source = tmp_path / f"recovered-{index}.mp4"
        source.write_bytes(f"recovered-video-{index}".encode())
        artifact = store.commit(store.stage(source, "video/mp4"), None)
        jobs.finalize_attempt(
            pair_job.id,
            next_attempt.attempt.id,
            job_state=JobState.COMPLETED,
            attempt_status=RunAttemptStatus.COMPLETED,
            artifact_id=artifact.id,
            error_code=None,
            error_message=None,
            event_kind="generation_completed",
            event_payload={},
        )

    awaiting = service.resume("study_1")
    assert awaiting.state is AdaptiveStudyState.AWAITING_PREFERENCE
    assert awaiting.review_id is not None
    assert len(jobs.attempts_for_job(pair_jobs[0].id)) == 2


def test_adaptive_requeue_rejects_foreign_jobs_and_rolls_back(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    event = AdaptiveStudyRepository(database).list_events("study_1")[-1]
    experiment_id = str(event.payload["experiment_id"])
    pair_jobs = experiment_service(database, tmp_path).get_adaptive(experiment_id).jobs
    service.pause_generation_failure("study_1", experiment_id, error_code="failed")
    generic = experiment_service(database, tmp_path).create(
        ExperimentDraft(
            baseline=make_manifest(),
            variable_path="inference_steps",
            alternate_value=8,
        ),
        capabilities(),
    )

    with pytest.raises(AdaptiveExperimentConflict, match="jobs"):
        JobRepository(database).requeue_failed_adaptive(
            "study_1",
            experiment_id,
            (pair_jobs[0].id, generic.jobs[0].id),
        )

    assert tuple(job.state for job in pair_jobs) == (JobState.QUEUED, JobState.QUEUED)


def test_adaptive_requeue_rejects_an_active_attempt_without_partial_reset(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    event = AdaptiveStudyRepository(database).list_events("study_1")[-1]
    experiment_id = str(event.payload["experiment_id"])
    jobs = JobRepository(database)
    pair_jobs = experiment_service(database, tmp_path).get_adaptive(experiment_id).jobs
    claimed = jobs.claim_next("worker-active", pair_jobs[0].id)
    assert claimed is not None
    service.pause_generation_failure("study_1", experiment_id, error_code="failed")

    with pytest.raises(AdaptiveExperimentConflict, match="active attempt"):
        jobs.requeue_failed_adaptive("study_1", experiment_id, tuple(job.id for job in pair_jobs))

    assert jobs.get(pair_jobs[0].id).state is JobState.PREPARING
    assert jobs.attempts_for_job(pair_jobs[0].id)[-1].status is RunAttemptStatus.ACTIVE


def test_resume_fails_closed_for_definition_only_legacy_without_owner_binding(
    database: Database, tmp_path: Path
) -> None:
    repository = AdaptiveStudyRepository(database)
    repository.create(make_definition())

    with pytest.raises(PermissionError, match="owner binding"):
        adaptive_service(database, tmp_path).resume("study_1")
    reopened_database = Database(database.path)
    reopened_database.migrate()
    with pytest.raises(PermissionError, match="owner binding"):
        adaptive_service(reopened_database, tmp_path).resume("study_1")

    assert repository.list_optimizer_steps("study_1") == ()
    assert repository.list_events("study_1") == ()
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM variants").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("damage", ["partial_jobs", "mismatched_manifest"])
def test_adaptive_experiment_retry_fails_closed_on_partial_or_mismatched_aggregate(
    database: Database, tmp_path: Path, damage: str
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    experiment = experiment_service(database, tmp_path).list_adaptive_for_study("study_1")[0]
    with database.connect() as connection:
        if damage == "partial_jobs":
            connection.execute("DELETE FROM jobs WHERE id = ?", (experiment.jobs[1].id,))
        else:
            changed = experiment.experiment.variants[0].manifest.model_copy(
                update={"prompt": "tampered prompt"}
            )
            connection.execute(
                "UPDATE variants SET manifest_json = ? WHERE id = ?",
                (changed.model_dump_json(), experiment.experiment.variants[0].id),
            )

    with pytest.raises(AdaptiveExperimentConflict):
        service.resume("study_1")
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1


def test_generic_experiment_creation_remains_random_and_compatible(
    database: Database, tmp_path: Path
) -> None:
    service = experiment_service(database, tmp_path)

    first = service.create(guidance_draft(), capabilities())
    second = service.create(guidance_draft(), capabilities())

    assert first.experiment.id != second.experiment.id
    assert not first.experiment.id.startswith("adaptive_")
    assert not second.experiment.id.startswith("adaptive_")
    assert len(service.list()) == 2


def test_optimizer_failure_preserves_observation_and_healthy_resume_continues_once(
    database: Database, tmp_path: Path
) -> None:
    from video_lab.preference_optimizer import optimize_preference

    def fail_after_observation(config, observations):
        if observations:
            raise RuntimeError("forced optimizer failure")
        return optimize_preference(config, observations)

    failing = adaptive_service(database, tmp_path, optimizer=fail_after_observation)
    failing.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = failing.resume("study_1")
    assert awaiting.review_id is not None

    failed = failing.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)

    assert failed.state is AdaptiveStudyState.OPTIMIZER_FAILED
    assert failed.next_action == "resume"
    repository = AdaptiveStudyRepository(database)
    assert len(repository.list_observations("study_1")) == 1
    assert len(repository.list_optimizer_steps("study_1")) == 1
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM review_revisions").fetchone()[0] == 1

    reopened_database = Database(database.path)
    reopened_database.migrate()
    resumed = adaptive_service(reopened_database, tmp_path).resume("study_1")
    retried = adaptive_service(reopened_database, tmp_path).resume("study_1")
    assert resumed == retried
    assert resumed.state is AdaptiveStudyState.GENERATING_PAIR
    assert resumed.comparison_count == 1
    assert len(repository.list_observations("study_1")) == 1
    assert len(repository.list_optimizer_steps("study_1")) == 2
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 2


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("review_session_id", "review_session_wrong"),
        ("reviewer_id", "intruder"),
        ("optimizer_output_digest", "0" * 64),
        ("sealed_mapping_digest", "f" * 64),
    ],
)
def test_observation_transaction_rejects_a_mismatched_active_review_binding_and_rolls_back(
    database: Database,
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None
    revision = review_service(database, tmp_path).submit_adaptive(
        awaiting.review_id, "reviewer", DisplayPreference.TIE
    )
    with database.connect() as connection:
        connection.execute("DROP TRIGGER guard_adaptive_study_events_update")
        row = connection.execute(
            "SELECT sequence, payload_json FROM adaptive_study_events "
            "WHERE study_id = ? ORDER BY sequence DESC LIMIT 1",
            ("study_1",),
        ).fetchone()
        payload = json.loads(str(row["payload_json"]))
        payload[field] = wrong_value
        connection.execute(
            "UPDATE adaptive_study_events SET payload_json = ? WHERE study_id = ? AND sequence = ?",
            (json.dumps(payload, sort_keys=True), "study_1", int(row["sequence"])),
        )

    repository = AdaptiveStudyRepository(database)
    with pytest.raises(AdaptiveStudyConflict, match="binding"):
        repository.append_observation(
            AdaptiveObservationRequest(
                study_id="study_1", ordinal=1, review_id=revision.id, created_at=utc_now()
            )
        )

    assert repository.list_observations("study_1") == ()
    assert repository.list_events("study_1")[-1].new_state is AdaptiveStudyState.AWAITING_PREFERENCE


def test_observation_transaction_rejects_an_intruder_session_for_the_matching_experiment(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    event = AdaptiveStudyRepository(database).list_events("study_1")[-1]
    intruder = review_service(database, tmp_path).start_adaptive(
        str(event.payload["experiment_id"]), "intruder"
    )
    revision = review_service(database, tmp_path).submit_adaptive(
        intruder.session_id, "intruder", DisplayPreference.TIE
    )

    repository = AdaptiveStudyRepository(database)
    with pytest.raises(AdaptiveStudyConflict, match="binding"):
        repository.append_observation(
            AdaptiveObservationRequest(
                study_id="study_1", ordinal=1, review_id=revision.id, created_at=utc_now()
            )
        )

    assert awaiting.review_id != intruder.session_id
    assert repository.list_observations("study_1") == ()


@pytest.mark.parametrize("contamination", ["notes", "score", "superseded"])
def test_observation_transaction_rejects_noncanonical_adaptive_revision_and_rolls_back(
    database: Database, tmp_path: Path, contamination: str
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None
    revision = review_service(database, tmp_path).submit_adaptive(
        awaiting.review_id, "reviewer", DisplayPreference.TIE
    )
    consumed_review_id = revision.id
    with database.connect() as connection:
        if contamination == "notes":
            connection.execute(
                "UPDATE review_revisions SET notes = 'contaminated' WHERE id = ?",
                (revision.id,),
            )
        elif contamination == "score":
            connection.execute(
                "INSERT INTO rubric_scores(review_id, display_position, dimension, score) "
                "VALUES (?, 'left', 'prompt_adherence', 3)",
                (revision.id,),
            )
        else:
            consumed_review_id = "review_correction"
            connection.execute(
                "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
                "display_preference, notes, created_at, revision, supersedes_review_id) "
                "VALUES (?, ?, NULL, 'tie', NULL, ?, 2, ?)",
                (
                    consumed_review_id,
                    awaiting.review_id,
                    utc_now().isoformat(),
                    revision.id,
                ),
            )

    repository = AdaptiveStudyRepository(database)
    with pytest.raises(AdaptiveStudyConflict, match="revision"):
        repository.append_observation(
            AdaptiveObservationRequest(
                study_id="study_1",
                ordinal=1,
                review_id=consumed_review_id,
                created_at=utc_now(),
            )
        )

    assert repository.list_observations("study_1") == ()


def test_resume_recomputes_and_fails_closed_on_optimizer_replay_mismatch(
    database: Database, tmp_path: Path
) -> None:
    from video_lab.preference_optimizer import optimize_preference

    adaptive_service(database, tmp_path).create_study(make_definition())

    def divergent_optimizer(config, observations):
        decision = optimize_preference(config, observations)
        return decision.model_copy(
            update={"selected_pair": CandidatePair(value_a=1.0, value_b=2.0)}
        )

    with pytest.raises(AdaptiveStudyConflict, match="replay"):
        adaptive_service(database, tmp_path, optimizer=divergent_optimizer).resume("study_1")


@pytest.mark.parametrize(
    ("comparison_count", "expected_state"),
    [
        (4, AdaptiveStudyState.GENERATING_PAIR),
        (10, AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED),
    ],
)
def test_four_to_ten_step_histories_replay_identically_after_sqlite_reopen(
    database: Database,
    tmp_path: Path,
    comparison_count: int,
    expected_state: AdaptiveStudyState,
) -> None:
    service = adaptive_service(database, tmp_path)
    view = service.create_study(make_definition())
    last_review_id = None
    last_revision_id = None
    for ordinal in range(1, comparison_count + 1):
        complete_generation(database, tmp_path, "study_1")
        awaiting = service.resume("study_1")
        assert awaiting.review_id is not None
        last_review_id = awaiting.review_id
        view = service.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)
        last_revision = review_service(database, tmp_path).latest_adaptive(
            awaiting.review_id, "reviewer"
        )
        assert last_revision is not None
        last_revision_id = last_revision.id
        assert view.comparison_count == ordinal
        reopened_database = Database(database.path)
        reopened_database.migrate()
        service = adaptive_service(reopened_database, tmp_path)

    repository = AdaptiveStudyRepository(database)
    before_steps = repository.list_optimizer_steps("study_1")
    before_events = repository.list_events("study_1")
    before_experiment_count = len(
        experiment_service(database, tmp_path).list_adaptive_for_study("study_1")
    )

    replayed = service.resume("study_1")

    assert replayed.state is expected_state
    assert replayed.comparison_count == comparison_count
    assert repository.list_optimizer_steps("study_1") == before_steps
    assert repository.list_events("study_1") == before_events
    assert len(repository.list_observations("study_1")) == comparison_count
    assert (
        len(experiment_service(database, tmp_path).list_adaptive_for_study("study_1"))
        == before_experiment_count
    )
    for experiment in experiment_service(database, tmp_path).list_adaptive_for_study("study_1"):
        for manifest in (variant.manifest for variant in experiment.experiment.variants):
            assert (
                manifest.model_copy(
                    update={"guidance_scale": make_definition().base_manifest.guidance_scale}
                )
                == make_definition().base_manifest
            )

    if expected_state is AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED:
        assert replayed.next_action == "none"
        assert replayed.recommendation is not None
        assert replayed.concentration_interval is not None
        assert last_review_id is not None
        assert last_revision_id is not None
        with pytest.raises(ValueError, match="immutable"):
            service.record_preference("study_1", last_review_id, DisplayPreference.LEFT)
        with pytest.raises(ValueError, match="immutable"):
            review_service(database, tmp_path).correct(
                last_review_id,
                "reviewer",
                last_revision_id,
                DisplayPreference.LEFT,
                (),
                None,
            )


@pytest.mark.parametrize(
    "boundary",
    [
        "planned_experiment",
        "review_session",
        "review_submission",
        "observation",
        "optimizer_step",
    ],
)
def test_resume_recovers_each_partial_transition_without_duplicate_evidence(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    service = adaptive_service(database, tmp_path)
    repository = AdaptiveStudyRepository(database)

    if boundary == "planned_experiment":
        original_create = service.experiments.create_adaptive

        def crash_before_experiment(*args, **kwargs):
            raise RuntimeError("crash after generation plan")

        monkeypatch.setattr(service.experiments, "create_adaptive", crash_before_experiment)
        with pytest.raises(RuntimeError, match="generation plan"):
            service.create_study(make_definition())
        monkeypatch.setattr(service.experiments, "create_adaptive", original_create)
        recovered = service.resume("study_1")
        assert recovered.state is AdaptiveStudyState.GENERATING_PAIR
    else:
        service.create_study(make_definition())
        complete_generation(database, tmp_path, "study_1")
        if boundary == "review_session":
            original_event = service.studies.append_awaiting_preference_or_verify

            def crash_after_session(_event):
                raise RuntimeError("crash after review session")

            monkeypatch.setattr(
                service.studies,
                "append_awaiting_preference_or_verify",
                crash_after_session,
            )
            with pytest.raises(RuntimeError, match="review session"):
                service.resume("study_1")
            monkeypatch.setattr(
                service.studies,
                "append_awaiting_preference_or_verify",
                original_event,
            )
            recovered = service.resume("study_1")
            assert recovered.state is AdaptiveStudyState.AWAITING_PREFERENCE
        else:
            awaiting = service.resume("study_1")
            assert awaiting.review_id is not None
            if boundary == "review_submission":
                original_observation = service.studies.consume_observation_or_verify

                def crash_after_submission(_request):
                    raise RuntimeError("crash after review submission")

                monkeypatch.setattr(
                    service.studies,
                    "consume_observation_or_verify",
                    crash_after_submission,
                )
                with pytest.raises(RuntimeError, match="review submission"):
                    service.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)
                monkeypatch.setattr(
                    service.studies,
                    "consume_observation_or_verify",
                    original_observation,
                )
            else:
                original_event = service._append_event

                def crash_after_evidence(study_id, *, previous, target, payload):
                    if (
                        boundary == "observation"
                        and target is AdaptiveStudyState.UPDATING_POSTERIOR
                    ) or (
                        boundary == "optimizer_step"
                        and previous is AdaptiveStudyState.UPDATING_POSTERIOR
                        and target is AdaptiveStudyState.GENERATING_PAIR
                    ):
                        raise RuntimeError(f"crash after {boundary}")
                    return original_event(
                        study_id, previous=previous, target=target, payload=payload
                    )

                monkeypatch.setattr(service, "_append_event", crash_after_evidence)
                with pytest.raises(RuntimeError, match=boundary):
                    service.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)
                monkeypatch.setattr(service, "_append_event", original_event)
            recovered = service.record_preference(
                "study_1", awaiting.review_id, DisplayPreference.TIE
            )
            assert recovered.state is AdaptiveStudyState.GENERATING_PAIR

    after_preference = boundary not in {"planned_experiment", "review_session"}
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == (
            2 if after_preference else 1
        )
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == (
            4 if after_preference else 2
        )
        assert connection.execute("SELECT COUNT(*) FROM review_sessions").fetchone()[0] == (
            0 if boundary == "planned_experiment" else 1
        )
        assert connection.execute("SELECT COUNT(*) FROM review_revisions").fetchone()[0] == (
            1 if after_preference else 0
        )
    assert len(repository.list_observations("study_1")) == (1 if after_preference else 0)
    assert len(repository.list_optimizer_steps("study_1")) == (2 if after_preference else 1)
    assert len(repository.list_events("study_1")) == (
        5 if after_preference else 2 if boundary == "planned_experiment" else 3
    )


def test_create_validates_adapter_before_persisting_and_is_idempotent(
    database: Database, tmp_path: Path
) -> None:
    from video_lab.adaptive_study_service import AdaptiveStudyService

    invalid = AdaptiveStudyService(
        AdaptiveStudyRepository(database),
        experiment_service(database, tmp_path),
        review_service(database, tmp_path),
        capabilities().model_copy(update={"adapter_id": "other"}),
        reviewer_id="reviewer",
    )
    with pytest.raises(ValidationFailure):
        invalid.create_study(make_definition())
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM adaptive_studies").fetchone()[0] == 0

    service = adaptive_service(database, tmp_path)
    first = service.create_study(make_definition())
    retried = service.create_study(make_definition())
    assert retried == first
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM adaptive_studies").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1


def test_optimizer_and_generation_setup_run_without_holding_sqlite_write_lock(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_lab.preference_optimizer import optimize_preference

    def lock_checking_optimizer(config, observations):
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        return optimize_preference(config, observations)

    service = adaptive_service(database, tmp_path, optimizer=lock_checking_optimizer)
    original_create = service.experiments.create_adaptive

    def lock_checking_create(*args, **kwargs):
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        return original_create(*args, **kwargs)

    monkeypatch.setattr(service.experiments, "create_adaptive", lock_checking_create)

    view = service.create_study(make_definition())

    assert view.state is AdaptiveStudyState.GENERATING_PAIR


def test_empty_optimizer_failure_is_persisted_and_explicit_resume_recovers(
    database: Database, tmp_path: Path
) -> None:
    def fail_optimizer(_config, _observations):
        raise RuntimeError("forced empty optimizer failure")

    failed = adaptive_service(database, tmp_path, optimizer=fail_optimizer).create_study(
        make_definition()
    )

    assert failed.state is AdaptiveStudyState.OPTIMIZER_FAILED
    assert failed.comparison_count == 0
    repository = AdaptiveStudyRepository(database)
    assert repository.list_optimizer_steps("study_1") == ()
    assert experiment_service(database, tmp_path).list_adaptive_for_study("study_1") == ()

    recovered = adaptive_service(database, tmp_path).resume("study_1")
    assert recovered.state is AdaptiveStudyState.GENERATING_PAIR
    assert len(repository.list_optimizer_steps("study_1")) == 1
    assert len(experiment_service(database, tmp_path).list_adaptive_for_study("study_1")) == 1


def test_empty_optimizer_decision_cannot_replace_the_frozen_two_vs_four_pair(
    database: Database, tmp_path: Path
) -> None:
    def wrong_initial_pair(config, observations):
        decision = optimize_preference(config, observations)
        return decision.model_copy(
            update={"selected_pair": CandidatePair(value_a=1.0, value_b=5.0)}
        )

    failed = adaptive_service(database, tmp_path, optimizer=wrong_initial_pair).create_study(
        make_definition()
    )

    assert failed.state is AdaptiveStudyState.OPTIMIZER_FAILED
    assert failed.comparison_count == 0
    assert experiment_service(database, tmp_path).list_adaptive_for_study("study_1") == ()
    events = AdaptiveStudyRepository(database).list_events("study_1")
    assert events[-1].payload["error_code"] == "initial_pair_mismatch"


def test_adaptive_correction_guard_checks_owner_before_disclosing_binding(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None

    with pytest.raises(PermissionError, match="reviewer"):
        review_service(database, tmp_path).correct(
            awaiting.review_id,
            "intruder",
            "not-visible",
            DisplayPreference.LEFT,
            (),
            None,
        )


def test_generic_review_submission_cannot_bypass_adaptive_immutability(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None

    with pytest.raises(ValueError, match="adaptive"):
        review_service(database, tmp_path).submit(
            awaiting.review_id,
            "reviewer",
            DisplayPreference.LEFT,
            rubric_scores(),
            "generic submission path",
        )


def test_adaptive_repository_rejects_a_preexisting_generic_review_revision(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None
    repository = ReviewRepository(database)
    with database.connect() as connection:
        session = connection.execute(
            "SELECT left_variant_id FROM review_sessions WHERE id = ?", (awaiting.review_id,)
        ).fetchone()
        connection.execute(
            "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
            "display_preference, notes, created_at, revision, supersedes_review_id) "
            "VALUES ('review_contaminated', ?, ?, 'left', 'generic submission path', ?, 1, NULL)",
            (awaiting.review_id, str(session["left_variant_id"]), utc_now().isoformat()),
        )

    with pytest.raises(ValueError, match="immutable"):
        repository.submit_adaptive(awaiting.review_id, DisplayPreference.LEFT)


def test_generic_review_repository_submit_rejects_an_adaptive_session_without_writing(
    database: Database, tmp_path: Path
) -> None:
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None
    repository = ReviewRepository(database)

    with pytest.raises(ValueError, match="adaptive"):
        repository.submit(
            awaiting.review_id,
            DisplayPreference.LEFT,
            rubric_scores(),
            "generic submission path",
        )

    assert repository.latest_for_session(awaiting.review_id) is None


def test_display_mapping_is_excluded_from_optimizer_replay_inputs(
    database: Database, tmp_path: Path
) -> None:
    output_digests = []
    mapping_digests = []
    for study_id in ("study_1", "study_2"):
        service = adaptive_service(database, tmp_path)
        service.create_study(make_definition(study_id))
        complete_generation(database, tmp_path, study_id)
        awaiting = service.resume(study_id)
        assert awaiting.review_id is not None
        with database.connect() as connection:
            row = connection.execute(
                "SELECT left_variant_id FROM review_sessions WHERE id = ?",
                (awaiting.review_id,),
            ).fetchone()
            manifest_row = connection.execute(
                "SELECT manifest_json FROM variants WHERE id = ?",
                (row["left_variant_id"],),
            ).fetchone()
        left_value = Manifest.model_validate_json(str(manifest_row["manifest_json"])).guidance_scale
        choice = DisplayPreference.LEFT if left_value == 2.0 else DisplayPreference.RIGHT
        service.record_preference(study_id, awaiting.review_id, choice)
        repository = AdaptiveStudyRepository(database)
        output_digests.append(repository.list_optimizer_steps(study_id)[1].output_digest)
        mapping_digests.append(repository.list_observations(study_id)[0].display_mapping_digest)

    assert output_digests[0] == output_digests[1]
    assert mapping_digests[0] != mapping_digests[1]


def test_each_new_observation_runs_one_optimizer_decision_before_persisting_next_pair(
    database: Database, tmp_path: Path
) -> None:
    from video_lab.preference_optimizer import optimize_preference

    calls = []

    def counting_optimizer(config, observations):
        calls.append(len(observations))
        return optimize_preference(config, observations)

    service = adaptive_service(database, tmp_path, optimizer=counting_optimizer)
    service.create_study(make_definition())
    assert calls == [0]
    service.resume("study_1")
    assert calls == [0]
    complete_generation(database, tmp_path, "study_1")
    reopened = adaptive_service(database, tmp_path, optimizer=counting_optimizer)
    awaiting = reopened.resume("study_1")
    assert calls == [0, 0]
    assert awaiting.review_id is not None

    reopened.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)

    assert calls == [0, 0, 1]


def test_create_rejects_stage_one_definition_without_frozen_two_vs_four_pair(
    database: Database, tmp_path: Path
) -> None:
    manifest = make_manifest()
    config = OptimizerConfig(upper_bound=4.0, posterior_draws=256)
    definition = AdaptiveStudyDefinition(
        id="study_1",
        prompt=manifest.prompt,
        negative_prompt=manifest.negative_prompt,
        seed=manifest.seed,
        base_manifest=manifest,
        optimizer_config=config,
        input_fingerprint=adaptive_study_input_fingerprint(manifest, config),
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="2.0 versus 4.0"):
        adaptive_service(database, tmp_path).create_study(definition)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM adaptive_studies").fetchone()[0] == 0


def test_unsupported_optimizer_stop_reason_fails_closed_without_claiming_budget_exhaustion(
    database: Database, tmp_path: Path
) -> None:
    def exhaust_candidates(config, observations):
        decision = optimize_preference(config, observations)
        if not observations:
            return decision
        return OptimizerDecision(
            posterior=decision.posterior,
            candidate_scores=(),
            selected_pair=None,
            stopping=StoppingDecision(
                stop=True,
                reason=StopReason.CANDIDATE_SPACE_EXHAUSTED,
                comparison_count=len(observations),
                maximum_information_bits=0.0,
            ),
        )

    service = adaptive_service(database, tmp_path, optimizer=exhaust_candidates)
    service.create_study(make_definition())
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None

    failed = service.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)

    assert failed.state is AdaptiveStudyState.OPTIMIZER_FAILED
    assert failed.comparison_count == 1
    assert failed.next_action == "none"
    events = AdaptiveStudyRepository(database).list_events("study_1")
    assert events[-1].payload["stop_reason"] == "candidate_space_exhausted"
    assert len(experiment_service(database, tmp_path).list_adaptive_for_study("study_1")) == 1
    event_count = len(events)

    assert service.resume("study_1") == failed
    assert service.resume("study_1") == failed
    assert len(AdaptiveStudyRepository(database).list_events("study_1")) == event_count


def test_malformed_optimizer_cannot_continue_past_the_independent_comparison_cap(
    database: Database, tmp_path: Path
) -> None:
    calls: list[int] = []

    def continue_at_cap(config, observations):
        calls.append(len(observations))
        decision = optimize_preference(config, observations)
        if len(observations) != config.maximum_comparisons:
            return decision
        return OptimizerDecision(
            posterior=decision.posterior,
            candidate_scores=decision.candidate_scores,
            selected_pair=CandidatePair(value_a=1.0, value_b=5.0),
            stopping=StoppingDecision(
                stop=False,
                reason=StopReason.CONTINUE,
                comparison_count=len(observations),
                maximum_information_bits=decision.stopping.maximum_information_bits,
            ),
        )

    service = adaptive_service(database, tmp_path, optimizer=continue_at_cap)
    view = service.create_study(make_definition())
    for _ in range(10):
        complete_generation(database, tmp_path, "study_1")
        awaiting = service.resume("study_1")
        assert awaiting.review_id is not None
        view = service.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)

    assert view.state is AdaptiveStudyState.OPTIMIZER_FAILED
    assert view.comparison_count == 10
    repository = AdaptiveStudyRepository(database)
    assert len(repository.list_optimizer_steps("study_1")) == 10
    assert len(experiment_service(database, tmp_path).list_adaptive_for_study("study_1")) == 10
    events = repository.list_events("study_1")
    assert events[-1].payload["error_code"] == "optimizer_continued_at_hard_cap"
    event_count = len(events)
    call_count = len(calls)

    assert service.resume("study_1") == view
    assert service.resume("study_1") == view
    assert len(repository.list_events("study_1")) == event_count
    assert len(calls) == call_count
