import json
from pathlib import Path

import pytest

from lab_tests.test_capabilities import manifest
from video_lab.artifacts import ArtifactIntegrityError, ArtifactStore
from video_lab.comparison import build_controlled_pair
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    ReviewDimension,
    RubricScore,
)
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository


def _service(database, root: Path) -> ReviewService:
    return ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        ArtifactStore(database, root),
        ReviewRepository(database),
    )


def _experiment(database, root: Path):
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    experiment, jobs = ExperimentRepository(database).create_with_jobs(pair)
    store = ArtifactStore(database, root)
    artifacts = []
    for index, job in enumerate(jobs):
        with database.connect() as connection:
            connection.execute("UPDATE jobs SET state = 'completed' WHERE id = ?", (job.id,))
        source = root.parent / f"source-{index}.mp4"
        source.write_bytes(f"video-{index}".encode())
        artifacts.append(store.commit(store.stage(source, "video/mp4"), job.id))
    return experiment, jobs, tuple(artifacts)


def _scores(value: int = 3):
    return tuple(
        RubricScore(display_position=position, dimension=dimension, score=value)
        for position in ("left", "right")
        for dimension in ReviewDimension
    )


def _replace_job_artifact(database, root: Path, job_id: str, content: bytes):
    source = root.parent / f"replacement-{job_id}.mp4"
    source.write_bytes(content)
    store = ArtifactStore(database, root)
    replacement = store.commit(store.stage(source, "video/mp4"), job_id)
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM job_artifacts WHERE job_id = ? AND artifact_id <> ?",
            (job_id, replacement.id),
        )
    return replacement


def _swap_job_variant_assignments(database, jobs) -> None:
    with database.connect() as connection:
        experiment_id = "experiment_lineage_decoy"
        variant_id = "variant_lineage_decoy"
        connection.execute(
            "INSERT INTO experiments(id, schema_version, variable_path, changed_paths_json, "
            "created_at) VALUES (?, 'experiment.v1', 'inference_steps', '[]', 'now')",
            (experiment_id,),
        )
        connection.execute(
            "INSERT INTO variants(id, experiment_id, label, manifest_json) "
            "VALUES (?, ?, 'A', '{}')",
            (variant_id, experiment_id),
        )
        connection.execute("UPDATE jobs SET variant_id = ? WHERE id = ?", (variant_id, jobs[0].id))
        connection.execute(
            "UPDATE jobs SET variant_id = ? WHERE id = ?", (jobs[0].variant_id, jobs[1].id)
        )
        connection.execute(
            "UPDATE jobs SET variant_id = ? WHERE id = ?", (jobs[1].variant_id, jobs[0].id)
        )


def test_queue_requires_completed_jobs_and_one_verified_mp4_each(database, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, artifacts = _experiment(database, root)
    service = _service(database, root)

    assert service.queue()[0].reviewable is True
    assert service.queue()[0].reason_codes == ()

    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = 'queued' WHERE id = ?", (jobs[0].id,))
    item = service.queue()[0]
    assert item.reviewable is False
    assert item.reason_codes == ("jobs_not_completed",)
    with pytest.raises(ValueError, match="jobs_not_completed"):
        service.start(experiment.id, "reviewer")

    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = 'completed' WHERE id = ?", (jobs[0].id,))
        connection.execute("DELETE FROM job_artifacts WHERE job_id = ?", (jobs[0].id,))
    assert service.queue()[0].reason_codes == ("missing_artifact",)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO job_artifacts(job_id, artifact_id, created_at) VALUES (?, ?, 'now')",
            (jobs[0].id, artifacts[0].id),
        )
        connection.execute(
            "UPDATE artifacts SET media_type = 'image/png' WHERE id = ?", (artifacts[0].id,)
        )
    assert service.queue()[0].reason_codes == ("non_video_artifact",)


@pytest.mark.parametrize("state", ["queued", "preparing", "generating", "failed", "cancelled"])
def test_queue_rejects_every_non_completed_job_state(database, tmp_path: Path, state: str) -> None:
    root = tmp_path / "artifacts"
    _, jobs, _ = _experiment(database, root)
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = ? WHERE id = ?", (state, jobs[0].id))
    assert _service(database, root).queue()[0].reason_codes == ("jobs_not_completed",)


def test_queue_reports_duplicate_and_corrupt_artifacts(database, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _, jobs, artifacts = _experiment(database, root)
    service = _service(database, root)
    extra = tmp_path / "extra.mp4"
    extra.write_bytes(b"extra")
    ArtifactStore(database, root).commit(
        ArtifactStore(database, root).stage(extra, "video/mp4"), jobs[0].id
    )
    assert service.queue()[0].reason_codes == ("duplicate_artifact",)
    with database.connect() as connection:
        connection.execute("DELETE FROM job_artifacts WHERE artifact_id != ?", (artifacts[0].id,))
    ArtifactStore(database, root).resolve(artifacts[0].id).write_bytes(b"corrupt")
    assert service.queue()[0].reason_codes == ("corrupt_artifact", "missing_artifact")


def test_blind_view_hides_canonical_and_artifact_information_and_is_stable(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, _, artifacts = _experiment(database, root)
    view = _service(database, root).start(experiment.id, "reviewer")
    reloaded = _service(database, root).start(experiment.id, "reviewer")
    assert view == reloaded
    assert tuple(media.display_position for media in view.media) == ("left", "right")
    assert all(
        media.url.startswith(f"/reviews/{view.session_id}/artifacts/") for media in view.media
    )
    serialized = json.dumps(view.model_dump(mode="json"))
    forbidden = [
        *[variant.id for variant in experiment.variants],
        *[artifact.id for artifact in artifacts],
        *[artifact.relative_path for artifact in artifacts],
        *[artifact.sha256 for artifact in artifacts],
        "manifest",
        "variable_path",
        "alternate_value",
        "model_revision",
        "inference_steps",
    ]
    assert all(secret not in serialized for secret in forbidden)


def test_artifact_authorization_uses_session_position_and_owner(database, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    experiment, _, artifacts = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    assert service.artifact_for_display(view.session_id, "left", "owner") in artifacts
    with pytest.raises(PermissionError, match="reviewer"):
        service.artifact_for_display(view.session_id, "left", "intruder")
    with pytest.raises(ValueError, match="position"):
        service.artifact_for_display(view.session_id, "center", "owner")


def test_artifact_delivery_keeps_the_session_pinned_artifact_after_association_replacement(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, artifacts = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    with database.connect() as connection:
        row = connection.execute(
            "SELECT left_variant_id FROM review_sessions WHERE id = ?", (view.session_id,)
        ).fetchone()
    left_variant_id = str(row["left_variant_id"])
    left_index = next(index for index, job in enumerate(jobs) if job.variant_id == left_variant_id)
    replacement = _replace_job_artifact(database, root, jobs[left_index].id, b"replacement-video")

    delivered = service.artifact_for_display(view.session_id, "left", "owner")

    assert delivered.id == artifacts[left_index].id
    assert delivered.sha256 == artifacts[left_index].sha256
    assert delivered.id != replacement.id


def test_submit_rejects_current_artifacts_that_differ_from_the_session_pins(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, _ = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    _replace_job_artifact(database, root, jobs[0].id, b"replacement-video")

    with pytest.raises(ValueError, match="session-pinned artifacts"):
        service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)


def test_reveal_keeps_session_pinned_artifact_identity_after_submission(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, artifacts = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)
    replacements = tuple(
        _replace_job_artifact(database, root, job.id, f"replacement-{index}".encode())
        for index, job in enumerate(jobs)
    )

    reveal = service.reveal(view.session_id, "owner")

    assert set(reveal.artifacts.values()) == set(artifacts)
    assert not set(reveal.artifacts.values()).intersection(replacements)


def test_reveal_keeps_original_job_artifact_pairs_after_jobs_are_reassigned(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, artifacts = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)
    expected = {artifact.id: job.id for artifact, job in zip(artifacts, jobs, strict=True)}

    _swap_job_variant_assignments(database, jobs)

    reveal = service.reveal(view.session_id, "owner")
    assert {artifact.id: artifact.job_id for artifact in reveal.artifacts.values()} == expected
    delivered = service.artifact_for_display(view.session_id, "left", "owner")
    assert delivered.job_id == expected[delivered.id]
    with database.connect() as connection:
        persisted = connection.execute(
            "SELECT left_job_id, right_job_id FROM review_sessions WHERE id = ?",
            (view.session_id,),
        ).fetchone()
    assert {str(persisted["left_job_id"]), str(persisted["right_job_id"])} == {
        job.id for job in jobs
    }


@pytest.mark.parametrize("damage", ["corrupt", "delete"])
def test_artifact_delivery_fails_when_session_pinned_bytes_are_not_intact(
    database, tmp_path: Path, damage: str
) -> None:
    root = tmp_path / "artifacts"
    experiment, _, _ = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    artifact = service.artifact_for_display(view.session_id, "left", "owner")
    path = ArtifactStore(database, root).resolve(artifact.id)
    if damage == "corrupt":
        path.write_bytes(b"corrupt")
    else:
        path.unlink()

    with pytest.raises(ArtifactIntegrityError):
        service.artifact_for_display(view.session_id, "left", "owner")


def test_submit_validates_owner_preference_scores_and_revalidates_artifacts(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, _, artifacts = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    with pytest.raises(ValueError, match="preference"):
        service.submit(view.session_id, "owner", None, _scores(), None)
    with pytest.raises(PermissionError, match="reviewer"):
        service.submit(view.session_id, "intruder", DisplayPreference.LEFT, _scores(), None)
    with pytest.raises(ValueError, match="ten unique"):
        service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores()[:-1], None)
    with pytest.raises(ValueError, match="ten unique"):
        service.submit(
            view.session_id,
            "owner",
            DisplayPreference.LEFT,
            _scores()[:-1] + (_scores()[0],),
            None,
        )
    ArtifactStore(database, root).resolve(artifacts[0].id).write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError):
        service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)


def test_reveal_denied_before_submit_and_complete_after_submit(database, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    experiment, _, artifacts = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    with pytest.raises(PermissionError, match="submitted"):
        service.reveal(view.session_id, "owner")
    submission = service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), "note")
    assert submission.model_dump().keys() == {"review_id", "revision", "status"}
    assert submission.status == "submitted"
    with pytest.raises(PermissionError, match="reviewer"):
        service.reveal(view.session_id, "intruder")
    reveal = service.reveal(view.session_id, "owner")
    assert reveal.review_id == submission.review_id
    assert reveal.canonical_by_display.keys() == {"left", "right"}
    assert set(reveal.canonical_by_display.values()) == {"A", "B"}
    assert reveal.variable_path == "inference_steps"
    assert reveal.baseline_value == manifest().inference_steps
    assert reveal.alternate_value == 8
    assert reveal.structural_diff == ("inference_steps",)
    assert reveal.manifests["A"] == experiment.variants[0].manifest
    assert reveal.manifests["B"] == experiment.variants[1].manifest
    assert set(reveal.artifacts.values()) == set(artifacts)
    assert reveal.preferred_canonical_variant == reveal.canonical_by_display["left"]


def test_correction_returns_revision_status_and_updates_reveal(database, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    experiment, _, _ = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    first = service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)
    correction = service.correct(
        view.session_id,
        "owner",
        first.review_id,
        DisplayPreference.RIGHT,
        _scores(4),
        "corrected",
    )
    assert correction.revision == 2
    assert correction.status == "corrected"
    assert (
        service.reveal(view.session_id, "owner").preferred_canonical_variant
        == service.reveal(view.session_id, "owner").canonical_by_display["right"]
    )


@pytest.mark.parametrize("operation", ["submit", "correct"])
def test_submission_revalidates_entire_experiment_after_extra_artifact(
    database, tmp_path: Path, operation: str
) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, _ = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    first = None
    if operation == "correct":
        first = service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)
    source = tmp_path / "late-extra.mp4"
    source.write_bytes(b"late-extra")
    store = ArtifactStore(database, root)
    store.commit(store.stage(source, "video/mp4"), jobs[0].id)

    with pytest.raises(ValueError, match="duplicate_artifact"):
        if operation == "submit":
            service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)
        else:
            assert first is not None
            service.correct(
                view.session_id,
                "owner",
                first.review_id,
                DisplayPreference.RIGHT,
                _scores(),
                None,
            )


def test_submission_revalidates_entire_experiment_after_job_becomes_ineligible(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, _ = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = 'queued' WHERE id = ?", (jobs[0].id,))

    with pytest.raises(ValueError, match="jobs_not_completed"):
        service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)


def test_cross_variant_duplicate_artifact_is_ineligible_at_queue_start_and_submit(
    database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    experiment, jobs, artifacts = _experiment(database, root)
    service = _service(database, root)
    view = service.start(experiment.id, "owner")
    with database.connect() as connection:
        connection.execute("DELETE FROM job_artifacts WHERE job_id = ?", (jobs[1].id,))
    duplicate_source = tmp_path / "duplicate-content.mp4"
    duplicate_source.write_bytes(b"video-0")
    store = ArtifactStore(database, root)
    duplicate = store.commit(store.stage(duplicate_source, "video/mp4"), jobs[1].id)
    assert duplicate.id == artifacts[0].id
    assert duplicate.sha256 == artifacts[0].sha256

    item = service.queue()[0]
    assert item.reason_codes == ("duplicate_variant_artifact",)
    with pytest.raises(ValueError, match="duplicate_variant_artifact"):
        service.start(experiment.id, "another-reviewer")
    with pytest.raises(ValueError, match="duplicate_variant_artifact"):
        service.submit(view.session_id, "owner", DisplayPreference.LEFT, _scores(), None)
