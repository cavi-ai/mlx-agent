import json
import sqlite3
from datetime import UTC
from pathlib import Path

import pytest
from pydantic import ValidationError

from lab_tests.test_capabilities import manifest
from video_lab.artifacts import Artifact
from video_lab.comparison import build_controlled_pair
from video_lab.database import _MIGRATIONS, SCHEMA_VERSION, Database
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    ReviewDimension,
    RubricScore,
)
from video_lab.repositories import ExperimentRepository
from video_lab.reviews import ReviewRepository

DIMENSIONS = tuple(ReviewDimension)


def _experiment(database: Database) -> tuple[str, tuple[str, str]]:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    stored = ExperimentRepository(database).create(pair)
    return stored.id, (stored.variants[0].id, stored.variants[1].id)


def _scores(value: int = 3) -> tuple[RubricScore, ...]:
    return tuple(
        RubricScore(display_position=position, dimension=dimension, score=value)
        for position in ("left", "right")
        for dimension in DIMENSIONS
    )


def _start_session(database: Database, experiment_id: str, reviewer_id: str):
    with database.connect() as connection:
        variant_ids = [
            str(row["id"])
            for row in connection.execute(
                "SELECT id FROM variants WHERE experiment_id = ? ORDER BY label",
                (experiment_id,),
            ).fetchall()
        ]
        artifacts = {}
        for index, variant_id in enumerate(variant_ids):
            artifact = Artifact(
                id=f"art_pin_{variant_id}",
                job_id=None,
                sha256=f"{'a' if index == 0 else 'b'}" * 64,
                byte_size=1,
                media_type="video/mp4",
                relative_path=f"pins/{variant_id}.mp4",
                created_at="2026-07-16T00:00:00+00:00",
            )
            connection.execute(
                "INSERT OR IGNORE INTO artifacts(id, sha256, byte_size, media_type, "
                "relative_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    artifact.id,
                    artifact.sha256,
                    artifact.byte_size,
                    artifact.media_type,
                    artifact.relative_path,
                    artifact.created_at,
                ),
            )
            job_id = f"job_pin_{variant_id}"
            connection.execute(
                "INSERT OR IGNORE INTO jobs(id, variant_id, state, created_at, updated_at) "
                "VALUES (?, ?, 'completed', 'now', 'now')",
                (job_id, variant_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO job_artifacts(job_id, artifact_id, created_at) "
                "VALUES (?, ?, 'now')",
                (job_id, artifact.id),
            )
            artifacts[variant_id] = artifact
    return ReviewRepository(database).start_session(experiment_id, reviewer_id, artifacts)


def test_migration_6_upgrades_schema_5_idempotently(tmp_path: Path) -> None:
    database = Database(tmp_path / "schema-5.sqlite3")
    with database.connect() as connection:
        for version in range(1, 6):
            connection.executescript(_MIGRATIONS[version])
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')", (version,)
            )

    database.migrate()
    database.migrate()

    with database.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(review_sessions)").fetchall()
        }
    assert [row[0] for row in versions] == list(range(1, SCHEMA_VERSION + 1))
    assert {"review_sessions", "review_revisions", "rubric_scores"} <= tables
    assert {
        "left_job_id",
        "left_artifact_id",
        "left_artifact_sha256",
        "right_job_id",
        "right_artifact_id",
        "right_artifact_sha256",
    } <= session_columns


def test_session_blinding_is_complete_and_stable_across_repositories(database: Database) -> None:
    experiment_id, variant_ids = _experiment(database)

    first = _start_session(database, experiment_id, "reviewer-1")
    second = _start_session(database, experiment_id, "reviewer-1")

    assert first == second
    assert first.display_positions == ("left", "right")
    assert first.created_at.tzinfo is UTC
    public_text = json.dumps(first.model_dump(mode="json")) + repr(first)
    assert "reviewer-1" not in public_text
    assert all(variant_id not in public_text for variant_id in variant_ids)
    with database.connect() as connection:
        private = connection.execute(
            "SELECT left_job_id, left_artifact_id, left_artifact_sha256, right_job_id, "
            "right_artifact_id, right_artifact_sha256 FROM review_sessions WHERE id = ?",
            (first.id,),
        ).fetchone()
    assert all(str(secret) not in public_text for secret in private)
    assert not hasattr(first, "reviewer_id")
    assert not hasattr(first, "display_to_variant")


def test_submit_translates_public_display_preference_using_private_mapping(
    database: Database,
) -> None:
    experiment_id, variant_ids = _experiment(database)
    repository = ReviewRepository(database)
    session = _start_session(database, experiment_id, "reviewer-1")

    revision = repository.submit(session.id, DisplayPreference.LEFT, _scores(), None)

    with database.connect() as connection:
        left_variant_id = connection.execute(
            "SELECT left_variant_id FROM review_sessions WHERE id = ?", (session.id,)
        ).fetchone()[0]
    assert revision.preferred_variant_id == left_variant_id
    assert revision.preferred_variant_id in variant_ids


def test_submit_stores_all_scores_in_deterministic_order_and_normalizes_notes(
    database: Database,
) -> None:
    experiment_id, _ = _experiment(database)
    session = _start_session(database, experiment_id, "reviewer-1")

    revision = ReviewRepository(database).submit(
        session.id,
        DisplayPreference.LEFT,
        tuple(reversed(_scores())),
        "  useful note  ",
    )

    assert revision.revision == 1
    assert revision.display_preference is DisplayPreference.LEFT
    assert revision.preferred_variant_id is not None
    assert revision.notes == "useful note"
    assert revision.created_at.tzinfo is UTC
    assert revision.scores == _scores()
    assert ReviewRepository(database).latest_for_session(session.id) == revision


def test_tie_has_no_canonical_preference_and_blank_notes_become_none(database: Database) -> None:
    experiment_id, _ = _experiment(database)
    repository = ReviewRepository(database)
    session = _start_session(database, experiment_id, "reviewer-1")

    revision = repository.submit(session.id, DisplayPreference.TIE, _scores(), "  ")

    assert revision.preferred_variant_id is None
    assert revision.notes is None


def test_first_submission_is_unique_and_correction_is_immutable(database: Database) -> None:
    experiment_id, _ = _experiment(database)
    repository = ReviewRepository(database)
    session = _start_session(database, experiment_id, "reviewer-1")
    original = repository.submit(session.id, DisplayPreference.LEFT, _scores(2), None)

    with pytest.raises(ValueError, match="supersedes"):
        repository.submit(session.id, DisplayPreference.RIGHT, _scores(4), "duplicate")

    correction = repository.submit(
        session.id,
        DisplayPreference.RIGHT,
        _scores(4),
        "corrected",
        supersedes_id=original.id,
    )

    assert correction.revision == 2
    assert correction.supersedes_review_id == original.id
    assert correction.id != original.id
    assert repository.latest_for_session(session.id) == correction
    with database.connect() as connection:
        stored_original = connection.execute(
            "SELECT revision, display_preference FROM review_revisions WHERE id = ?",
            (original.id,),
        ).fetchone()
    assert tuple(stored_original) == (1, "left")


def test_submission_validates_exactly_ten_unique_bounded_scores(database: Database) -> None:
    experiment_id, _ = _experiment(database)
    repository = ReviewRepository(database)
    session = _start_session(database, experiment_id, "reviewer-1")

    with pytest.raises(ValueError, match="ten unique"):
        repository.submit(session.id, DisplayPreference.LEFT, _scores()[:-1], None)
    with pytest.raises(ValueError, match="ten unique"):
        repository.submit(
            session.id,
            DisplayPreference.LEFT,
            _scores()[:-1] + (_scores()[0],),
            None,
        )
    with pytest.raises(ValidationError):
        RubricScore(
            display_position="left",
            dimension=ReviewDimension.VISUAL_QUALITY,
            score=6,
        )


def test_unknown_sessions_and_cross_session_supersession_fail_closed(database: Database) -> None:
    experiment_id, _ = _experiment(database)
    repository = ReviewRepository(database)
    first_session = _start_session(database, experiment_id, "reviewer-1")
    second_session = _start_session(database, experiment_id, "reviewer-2")
    original = repository.submit(first_session.id, DisplayPreference.LEFT, _scores(), None)

    assert repository.latest_for_session("session-missing") is None
    with pytest.raises(LookupError, match="session"):
        repository.submit("session-missing", DisplayPreference.LEFT, _scores(), None)
    with pytest.raises(ValueError, match="session"):
        repository.submit(
            second_session.id,
            DisplayPreference.RIGHT,
            _scores(),
            None,
            supersedes_id=original.id,
        )


def test_schema_rejects_cross_session_supersession_via_direct_sql(database: Database) -> None:
    experiment_id, _ = _experiment(database)
    repository = ReviewRepository(database)
    first_session = _start_session(database, experiment_id, "reviewer-1")
    second_session = _start_session(database, experiment_id, "reviewer-2")
    original = repository.submit(first_session.id, DisplayPreference.LEFT, _scores(), None)

    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
            "display_preference, notes, created_at, revision, supersedes_review_id) "
            "VALUES (?, ?, NULL, 'tie', NULL, ?, 1, ?)",
            (
                "review_cross_session",
                second_session.id,
                original.created_at.isoformat(),
                original.id,
            ),
        )


def test_unknown_experiment_cannot_start_session(database: Database) -> None:
    with pytest.raises(LookupError, match="experiment"):
        ReviewRepository(database).start_session("experiment-missing", "reviewer-1", {})
