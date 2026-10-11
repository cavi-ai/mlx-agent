from __future__ import annotations

import json

import pytest

from lab_tests.test_adaptive_study_service import (
    adaptive_service,
    complete_generation,
    experiment_service,
    make_definition,
)
from lab_tests.test_capabilities import manifest
from lab_tests.test_review_service import _scores
from lab_tests.test_validation_repository import definition, event, panel, seed_stage_1
from video_lab.artifacts import ArtifactStore
from video_lab.comparison import build_controlled_pair
from video_lab.domain import DisplayPreference, ExperimentDraft
from video_lab.evidence_service import EvidenceNotFound, EvidenceService
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.validation_models import ValidationCampaignState
from video_lab.validation_repository import ValidationRepository


def _seed(database):
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    return ExperimentRepository(database).create_with_jobs(pair)


def _complete_and_submit(database, tmp_path, experiment, jobs, reviewer="owner"):
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    for index, job in enumerate(jobs):
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state='completed' WHERE id=?", (job.id,))
        if not store.list_for_job(job.id):
            source = tmp_path / f"reviewable-{index}.mp4"
            source.write_bytes(f"reviewable-{index}-{job.id}".encode())
            store.commit(store.stage(source, "video/mp4"), job.id)
    reviews = ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        store,
        ReviewRepository(database),
    )
    session = reviews.start(experiment.id, reviewer)
    reviews.submit(session.session_id, reviewer, DisplayPreference.TIE, _scores(), None)
    return store


def _awaiting_adaptive_experiment(database, tmp_path):
    service = adaptive_service(database, tmp_path)
    service.create_study(make_definition())
    adaptive_id = (
        experiment_service(database, tmp_path).list_adaptive_for_study("study_1")[0].experiment.id
    )
    complete_generation(database, tmp_path, "study_1")
    awaiting = service.resume("study_1")
    assert awaiting.review_id is not None
    return service, awaiting, adaptive_id


def test_history_is_stable_newest_first_and_projects_all_states(database, tmp_path) -> None:
    older, older_jobs = _seed(database)
    newer, newer_jobs = _seed(database)
    with database.connect() as connection:
        connection.execute(
            "UPDATE experiments SET created_at = '2024-01-01T00:00:00+00:00' WHERE id = ?",
            (older.id,),
        )
        connection.execute(
            "UPDATE experiments SET created_at = '2025-01-01T00:00:00+00:00' WHERE id = ?",
            (newer.id,),
        )
        connection.execute("UPDATE jobs SET state = 'failed' WHERE id = ?", (older_jobs[0].id,))
        connection.execute("UPDATE jobs SET state = 'cancelled' WHERE id = ?", (newer_jobs[0].id,))
    service = EvidenceService(database, ArtifactStore(database, tmp_path / "artifacts"))

    history = service.list()

    assert [item.id for item in history.items] == [newer.id, older.id]
    assert history.items[0].states == ("cancelled", "queued")
    assert history.items[0].sealed is True
    assert history.items[0].variable_path is None
    assert history.items[0].attempt_count == 0


def test_detail_projects_canonical_diff_retry_evidence_and_path_free_errors(
    database, tmp_path
) -> None:
    experiment, jobs = _seed(database)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO run_attempts(id, job_id, number, worker_id, status, started_at, "
            "finished_at, error_code, error_message) VALUES "
            "('attempt_1', ?, 1, 'worker', 'interrupted', '2025-01-01T00:00:00+00:00', "
            "'2025-01-01T00:01:00+00:00', 'lost', '/private/models/model.bin failed')",
            (jobs[0].id,),
        )
        connection.execute(
            "INSERT INTO run_events(job_id, attempt_id, kind, payload_json, created_at) "
            "VALUES (?, 'attempt_1', 'worker_interrupted', ?, "
            "'2025-01-01T00:01:00+00:00')",
            (jobs[0].id, json.dumps({"message": "/private/models/model.bin failed"})),
        )
        connection.execute(
            "INSERT INTO run_events(job_id, attempt_id, kind, payload_json, created_at) "
            "VALUES (?, 'attempt_1', 'diagnostic', ?, '2025-01-01T00:01:01+00:00')",
            (
                jobs[0].id,
                json.dumps(
                    {
                        "message": "failed loading /Users/franco/secret/model.safetensors",
                        "nested": [
                            {"path": "/private/weights/secret.bin", "ok": False},
                            ["prefix /Volumes/MIRZA/private/data.json suffix"],
                        ],
                    }
                ),
            ),
        )
    store = _complete_and_submit(database, tmp_path, experiment, jobs)
    detail = EvidenceService(database, store).detail(experiment.id, "owner")

    assert detail.canonical_diff.path == "inference_steps"
    assert detail.canonical_diff.baseline == 4
    assert detail.canonical_diff.alternate == 8
    assert detail.jobs[0].attempts[0].status == "interrupted"
    serialized = json.dumps(detail.model_dump(mode="json"))
    assert "/private/" not in serialized
    assert "/Users/" not in serialized
    assert "/Volumes/" not in serialized
    assert "secret.bin" not in serialized
    assert "failed loading [local-path]" in serialized


def test_artifact_integrity_and_telemetry_min_max_last_are_evidence(database, tmp_path) -> None:
    experiment, jobs = _seed(database)
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"clip")
    artifact = store.commit(store.stage(source, "video/mp4"), jobs[0].id)
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = 'completed' WHERE id = ?", (jobs[0].id,))
        connection.execute(
            "INSERT INTO run_attempts(id, job_id, number, worker_id, status, started_at, "
            "finished_at, artifact_id) VALUES "
            "('attempt_t', ?, 1, 'worker', 'completed', '2025-01-01T00:00:00+00:00', "
            "'2025-01-01T00:01:00+00:00', ?)",
            (jobs[0].id, artifact.id),
        )
        for recorded_at, elapsed, rss in (
            ("2025-01-01T00:00:03+00:00", 3.0, 200),
            ("2025-01-01T00:00:01+00:00", 1.0, 100),
            ("2025-01-01T00:00:02+00:00", 2.0, 300),
        ):
            connection.execute(
                "INSERT INTO telemetry_samples(attempt_id, recorded_at, elapsed_seconds, "
                "process_rss_bytes, system_used_bytes, system_available_bytes, "
                "swap_used_bytes, thermal_pressure) VALUES "
                "('attempt_t', ?, ?, ?, 10, 20, 0, NULL)",
                (recorded_at, elapsed, rss),
            )

    store = _complete_and_submit(database, tmp_path, experiment, jobs)
    detail = EvidenceService(database, store).detail(experiment.id, "owner")

    assert detail.jobs[0].artifacts[0].integrity == "verified"
    assert detail.jobs[0].artifacts[0].sha256 == artifact.sha256
    telemetry = detail.jobs[0].attempts[0].telemetry
    assert telemetry is not None
    assert telemetry.process_rss_bytes.min == 100
    assert telemetry.process_rss_bytes.max == 300
    assert telemetry.process_rss_bytes.last == 200
    store.resolve(artifact.id).write_bytes(b"bad")
    assert (
        EvidenceService(database, store)
        .detail(experiment.id, "owner")
        .jobs[0]
        .artifacts[0]
        .integrity
        == "corrupt"
    )


@pytest.mark.parametrize("problem", ["queued", "duplicate", "wrong_media", "corrupt"])
def test_history_completeness_requires_exact_verified_completed_mp4_pair(
    database, tmp_path, problem
) -> None:
    experiment, jobs = _seed(database)
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    registered = []
    for index, job in enumerate(jobs):
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state='completed' WHERE id=?", (job.id,))
        source = tmp_path / f"complete-{index}.mp4"
        source.write_bytes(f"complete-{index}".encode())
        registered.append(store.commit(store.stage(source, "video/mp4"), job.id))
    assert EvidenceService(database, store).list().items[0].evidence_completeness == "complete"

    if problem == "queued":
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state='queued' WHERE id=?", (jobs[0].id,))
    elif problem == "duplicate":
        source = tmp_path / "duplicate.mp4"
        source.write_bytes(b"duplicate")
        store.commit(store.stage(source, "video/mp4"), jobs[0].id)
    elif problem == "wrong_media":
        with database.connect() as connection:
            connection.execute(
                "UPDATE artifacts SET media_type='image/png' WHERE id=?", (registered[0].id,)
            )
    else:
        store.resolve(registered[0].id).write_bytes(b"checksum-mismatch")

    history = EvidenceService(database, store).list()
    detail = EvidenceService(database, store).detail(experiment.id, "owner")
    assert history.items[0].evidence_completeness == "partial"
    assert detail.evidence_completeness == "partial"


def test_typed_not_found(database, tmp_path) -> None:
    with pytest.raises(EvidenceNotFound):
        EvidenceService(database, ArtifactStore(database, tmp_path / "artifacts")).detail(
            "missing", "owner"
        )


def test_adaptive_evidence_is_absent_before_preference_submission(database, tmp_path) -> None:
    generic, jobs = _seed(database)
    store = _complete_and_submit(database, tmp_path, generic, jobs)
    _, _, adaptive_id = _awaiting_adaptive_experiment(database, tmp_path)
    evidence = EvidenceService(database, store)

    history = evidence.list(reviewer_id="owner")

    assert tuple(item.id for item in history.items) == (generic.id,)
    assert adaptive_id not in history.model_dump_json()
    assert "guidance_scale" not in history.model_dump_json()
    with pytest.raises(EvidenceNotFound):
        evidence.detail(adaptive_id, "reviewer")


def test_validation_experiment_is_absent_from_generic_history_and_detail(
    database, tmp_path
) -> None:
    seed_stage_1(database)
    campaigns = ValidationRepository(database)
    campaign = campaigns.create_or_get_bound(definition("validation_campaign"), reviewer_id="owner")
    validation_panel = panel(campaign)
    campaigns.preregister_or_verify(campaign.id, validation_panel)
    campaigns.append_event_or_verify(
        event(
            campaign.id,
            2,
            ValidationCampaignState.PREREGISTERED,
            ValidationCampaignState.GENERATING,
        )
    )
    trial_id = next(item.id for item in validation_panel.trials if not item.hidden_repeat)
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="guidance_scale", alternate_value=2.0)
    )
    validation, _ = ExperimentRepository(database).create_validation_with_jobs(
        pair, campaign_id=campaign.id, trial_id=trial_id
    )
    evidence = EvidenceService(database, ArtifactStore(database, tmp_path / "artifacts"))

    assert validation.id not in evidence.list(reviewer_id="owner").model_dump_json()
    with pytest.raises(EvidenceNotFound):
        evidence.detail(validation.id, "owner")


def test_adaptive_evidence_remains_absent_after_nonterminal_preference_submission(
    database, tmp_path
) -> None:
    generic, jobs = _seed(database)
    store = _complete_and_submit(database, tmp_path, generic, jobs)
    service, awaiting, first_adaptive_id = _awaiting_adaptive_experiment(database, tmp_path)

    nonterminal = service.record_preference("study_1", awaiting.review_id, DisplayPreference.TIE)

    assert nonterminal.next_action == "complete_generation"
    adaptive_ids = tuple(
        item.experiment.id
        for item in experiment_service(database, tmp_path).list_adaptive_for_study("study_1")
    )
    assert first_adaptive_id in adaptive_ids
    evidence = EvidenceService(database, store)
    history = evidence.list(reviewer_id="owner")
    serialized = history.model_dump_json()
    assert tuple(item.id for item in history.items) == (generic.id,)
    assert all(adaptive_id not in serialized for adaptive_id in adaptive_ids)
    assert "guidance_scale" not in serialized
    for adaptive_id in adaptive_ids:
        with pytest.raises(EvidenceNotFound):
            evidence.detail(adaptive_id, "reviewer")
    assert evidence.detail(generic.id, "owner").canonical_diff is not None


def test_review_revisions_and_canonical_reveal_are_owner_only(database, tmp_path) -> None:
    experiment, jobs = _seed(database)
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    for index, job in enumerate(jobs):
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state='completed' WHERE id=?", (job.id,))
        source = tmp_path / f"review-{index}.mp4"
        source.write_bytes(f"review-{index}".encode())
        store.commit(store.stage(source, "video/mp4"), job.id)
    reviews = ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        store,
        ReviewRepository(database),
    )
    session = reviews.start(experiment.id, "owner")
    first = reviews.submit(session.session_id, "owner", DisplayPreference.LEFT, _scores(), "first")
    reviews.correct(
        session.session_id,
        "owner",
        first.review_id,
        DisplayPreference.RIGHT,
        _scores(4),
        "corrected",
    )

    owner = EvidenceService(database, store).detail(experiment.id, "owner").review
    intruder = EvidenceService(database, store).detail(experiment.id, "intruder").review

    assert [revision.revision for revision in owner.revisions] == [1, 2]
    assert owner.revisions[1].supersedes_review_id == first.review_id
    assert owner.canonical_reveal is not None
    assert owner.entry_url == f"/reviews/{session.session_id}/receipt"
    assert intruder.session_id is None
    assert intruder.revisions == ()
    assert intruder.canonical_reveal is None
    assert experiment.variants[0].id not in (intruder.entry_url or "")


def test_reviewable_evidence_links_to_its_persisted_blind_review(database, tmp_path) -> None:
    experiment, jobs = _seed(database)
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    for index, job in enumerate(jobs):
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state='completed' WHERE id=?", (job.id,))
        source = tmp_path / f"review-entry-{index}.mp4"
        source.write_bytes(f"review-entry-{index}".encode())
        store.commit(store.stage(source, "video/mp4"), job.id)

    detail = EvidenceService(database, store).detail(experiment.id, "owner")

    assert detail.sealed is True
    assert detail.review.reviewable is True
    assert detail.review.entry_url == f"/review/{experiment.id}"
    assert detail.review.session_id is None
    assert detail.review.canonical_reveal is None

    ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        store,
        ReviewRepository(database),
    ).start(experiment.id, "owner")
    resumed = EvidenceService(database, store).detail(experiment.id, "owner")

    assert resumed.sealed is True
    assert resumed.review.entry_url == f"/review/{experiment.id}"
    assert resumed.review.session_id is None
    assert resumed.review.canonical_reveal is None
