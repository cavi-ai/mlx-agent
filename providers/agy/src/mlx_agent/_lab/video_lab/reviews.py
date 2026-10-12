from __future__ import annotations

import hashlib
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from video_lab.artifacts import Artifact
from video_lab.database import Database
from video_lab.domain import (
    DisplayPreference,
    ReviewDimension,
    ReviewRevision,
    ReviewSession,
    RubricScore,
    canonical_json,
    new_id,
    utc_now,
)

_POSITIONS: tuple[Literal["left", "right"], ...] = ("left", "right")
_SCORE_KEYS = {(position, dimension) for position in _POSITIONS for dimension in ReviewDimension}
ADAPTIVE_REVIEW_BINDING_SCHEMA_VERSION = "adaptive-review-binding.v1"


@dataclass(frozen=True)
class _ReviewSessionRecord:
    id: str
    experiment_id: str
    reviewer_id: str
    left_variant_id: str
    right_variant_id: str
    left_job_id: str
    left_artifact_id: str
    left_artifact_sha256: str
    right_job_id: str
    right_artifact_id: str
    right_artifact_sha256: str
    created_at: datetime


class ReviewRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def start_session(
        self,
        experiment_id: str,
        reviewer_id: str,
        artifacts_by_variant: Mapping[str, Artifact],
    ) -> ReviewSession:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM review_sessions WHERE experiment_id = ? AND reviewer_id = ?",
                    (experiment_id, reviewer_id),
                ).fetchone()
                if existing is not None:
                    connection.commit()
                    return _public_session(existing)
                variants = connection.execute(
                    "SELECT id FROM variants WHERE experiment_id = ? ORDER BY label",
                    (experiment_id,),
                ).fetchall()
                if not variants:
                    raise LookupError(f"experiment not found: {experiment_id}")
                if len(variants) != 2:
                    raise ValueError(f"experiment must have exactly two variants: {experiment_id}")
                variant_ids = [str(row["id"]) for row in variants]
                if set(artifacts_by_variant) != set(variant_ids):
                    raise ValueError("artifact pins must cover the experiment variants")
                current = connection.execute(
                    "SELECT jobs.id AS job_id, jobs.variant_id, jobs.state, "
                    "artifacts.id AS artifact_id, "
                    "artifacts.sha256, artifacts.media_type FROM jobs "
                    "JOIN variants ON variants.id = jobs.variant_id "
                    "LEFT JOIN job_artifacts ON job_artifacts.job_id = jobs.id "
                    "LEFT JOIN artifacts ON artifacts.id = job_artifacts.artifact_id "
                    "WHERE variants.experiment_id = ?",
                    (experiment_id,),
                ).fetchall()
                current_pins = {
                    str(row["variant_id"]): (
                        str(row["job_id"]),
                        str(row["artifact_id"]),
                        str(row["sha256"]),
                    )
                    for row in current
                    if row["state"] == "completed"
                    and row["artifact_id"] is not None
                    and row["media_type"] == "video/mp4"
                }
                requested_pins = {
                    variant_id: (artifact.id, artifact.sha256)
                    for variant_id, artifact in artifacts_by_variant.items()
                }
                current_artifacts = {
                    variant_id: (pin[1], pin[2]) for variant_id, pin in current_pins.items()
                }
                if len(current) != 2 or current_artifacts != requested_pins:
                    raise ValueError("artifact pins no longer match current experiment artifacts")
                shuffled = secrets.SystemRandom().sample(variant_ids, k=2)
                left_artifact = artifacts_by_variant[shuffled[0]]
                right_artifact = artifacts_by_variant[shuffled[1]]
                record = _ReviewSessionRecord(
                    id=new_id("review_session"),
                    experiment_id=experiment_id,
                    reviewer_id=reviewer_id,
                    left_variant_id=shuffled[0],
                    right_variant_id=shuffled[1],
                    left_job_id=current_pins[shuffled[0]][0],
                    left_artifact_id=left_artifact.id,
                    left_artifact_sha256=left_artifact.sha256,
                    right_job_id=current_pins[shuffled[1]][0],
                    right_artifact_id=right_artifact.id,
                    right_artifact_sha256=right_artifact.sha256,
                    created_at=utc_now(),
                )
                connection.execute(
                    "INSERT INTO review_sessions(id, experiment_id, reviewer_id, "
                    "left_variant_id, right_variant_id, left_artifact_id, "
                    "left_job_id, left_artifact_sha256, right_job_id, right_artifact_id, "
                    "right_artifact_sha256, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.experiment_id,
                        record.reviewer_id,
                        record.left_variant_id,
                        record.right_variant_id,
                        record.left_artifact_id,
                        record.left_job_id,
                        record.left_artifact_sha256,
                        record.right_job_id,
                        record.right_artifact_id,
                        record.right_artifact_sha256,
                        record.created_at.isoformat(),
                    ),
                )
                connection.commit()
                return ReviewSession(
                    id=record.id,
                    experiment_id=record.experiment_id,
                    created_at=record.created_at,
                )
            except Exception:
                connection.rollback()
                raise

    def submit(
        self,
        session_id: str,
        preference: DisplayPreference,
        scores: Sequence[RubricScore],
        notes: str | None,
        supersedes_id: str | None = None,
    ) -> ReviewRevision:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                session_row = connection.execute(
                    "SELECT * FROM review_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if session_row is None:
                    raise LookupError(f"review session not found: {session_id}")
                if _is_adaptive_experiment(connection, str(session_row["experiment_id"])):
                    raise ValueError("adaptive review must use immutable preference submission")
                ordered_scores = _validate_and_order_scores(scores)
                normalized_notes = notes.strip() or None if notes is not None else None
                latest_row = connection.execute(
                    "SELECT * FROM review_revisions WHERE session_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                superseded_row = None
                if supersedes_id is not None:
                    superseded_row = connection.execute(
                        "SELECT session_id FROM review_revisions WHERE id = ?", (supersedes_id,)
                    ).fetchone()
                    if superseded_row is None:
                        raise ValueError("superseded review does not exist")
                    if str(superseded_row["session_id"]) != session_id:
                        raise ValueError("superseded review belongs to another session")
                if latest_row is None:
                    if supersedes_id is not None:
                        raise ValueError("first review cannot supersede another review")
                    revision_number = 1
                else:
                    if supersedes_id is None:
                        raise ValueError("correction must specify the review it supersedes")
                    if supersedes_id != str(latest_row["id"]):
                        raise ValueError(
                            "correction must supersede the latest review for this session"
                        )
                    revision_number = int(latest_row["revision"]) + 1

                preferred_variant_id = None
                if preference is DisplayPreference.LEFT:
                    preferred_variant_id = str(session_row["left_variant_id"])
                elif preference is DisplayPreference.RIGHT:
                    preferred_variant_id = str(session_row["right_variant_id"])
                revision = ReviewRevision(
                    id=new_id("review"),
                    session_id=session_id,
                    preferred_variant_id=preferred_variant_id,
                    display_preference=preference,
                    scores=ordered_scores,
                    notes=normalized_notes,
                    created_at=utc_now(),
                    revision=revision_number,
                    supersedes_review_id=supersedes_id,
                )
                connection.execute(
                    "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
                    "display_preference, notes, created_at, revision, supersedes_review_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        revision.id,
                        revision.session_id,
                        revision.preferred_variant_id,
                        revision.display_preference.value,
                        revision.notes,
                        revision.created_at.isoformat(),
                        revision.revision,
                        revision.supersedes_review_id,
                    ),
                )
                connection.executemany(
                    "INSERT INTO rubric_scores(review_id, display_position, dimension, score) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (revision.id, score.display_position, score.dimension.value, score.score)
                        for score in ordered_scores
                    ],
                )
                connection.commit()
                return revision
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValueError("review violates persistence constraints") from error
            except Exception:
                connection.rollback()
                raise

    def submit_adaptive(
        self,
        session_id: str,
        preference: DisplayPreference,
    ) -> ReviewRevision:
        """Persist one immutable adaptive preference, or return the identical retry."""
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                session_row = connection.execute(
                    "SELECT * FROM review_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if session_row is None:
                    raise LookupError(f"review session not found: {session_id}")
                latest_row = connection.execute(
                    "SELECT * FROM review_revisions WHERE session_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if latest_row is not None:
                    score_rows = connection.execute(
                        "SELECT display_position, dimension, score FROM rubric_scores "
                        "WHERE review_id = ?",
                        (latest_row["id"],),
                    ).fetchall()
                    if (
                        int(latest_row["revision"]) != 1
                        or latest_row["supersedes_review_id"] is not None
                        or latest_row["notes"] is not None
                        or str(latest_row["display_preference"]) != preference.value
                        or score_rows
                    ):
                        raise ValueError("adaptive v1 review is immutable")
                    connection.commit()
                    return _revision(latest_row, score_rows)

                preferred_variant_id = None
                if preference is DisplayPreference.LEFT:
                    preferred_variant_id = str(session_row["left_variant_id"])
                elif preference is DisplayPreference.RIGHT:
                    preferred_variant_id = str(session_row["right_variant_id"])
                revision = ReviewRevision(
                    id=new_id("review"),
                    session_id=session_id,
                    preferred_variant_id=preferred_variant_id,
                    display_preference=preference,
                    scores=(),
                    notes=None,
                    created_at=utc_now(),
                    revision=1,
                    supersedes_review_id=None,
                )
                connection.execute(
                    "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
                    "display_preference, notes, created_at, revision, supersedes_review_id) "
                    "VALUES (?, ?, ?, ?, NULL, ?, 1, NULL)",
                    (
                        revision.id,
                        revision.session_id,
                        revision.preferred_variant_id,
                        revision.display_preference.value,
                        revision.created_at.isoformat(),
                    ),
                )
                connection.commit()
                return revision
            except Exception:
                connection.rollback()
                raise

    def is_adaptive_session(self, session_id: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT experiment_id FROM review_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return row is not None and _is_adaptive_experiment(
                connection, str(row["experiment_id"])
            )

    def adaptive_binding_digest(
        self,
        session_id: str,
        reviewer_id: str,
        *,
        study_id: str,
        optimizer_step_ordinal: int,
        optimizer_output_digest: str,
    ) -> str:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"review session not found: {session_id}")
            if str(row["reviewer_id"]) != reviewer_id:
                raise PermissionError("review session belongs to another reviewer")
            if not _is_adaptive_experiment(connection, str(row["experiment_id"])):
                raise ValueError("review session is not bound to an adaptive study")
            return adaptive_review_binding_digest(
                row,
                study_id=study_id,
                optimizer_step_ordinal=optimizer_step_ordinal,
                optimizer_output_digest=optimizer_output_digest,
            )

    def latest_for_session(self, session_id: str) -> ReviewRevision | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_revisions WHERE session_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            scores = connection.execute(
                "SELECT display_position, dimension, score FROM rubric_scores "
                "WHERE review_id = ? ORDER BY "
                "CASE display_position WHEN 'left' THEN 0 ELSE 1 END, "
                "CASE dimension "
                "WHEN 'prompt_adherence' THEN 0 WHEN 'visual_quality' THEN 1 "
                "WHEN 'motion_quality' THEN 2 WHEN 'temporal_consistency' THEN 3 ELSE 4 END",
                (row["id"],),
            ).fetchall()
        return _revision(row, scores)


def adaptive_review_binding_digest(
    session_row: sqlite3.Row,
    *,
    study_id: str,
    optimizer_step_ordinal: int,
    optimizer_output_digest: str,
) -> str:
    payload = {
        "schema_version": ADAPTIVE_REVIEW_BINDING_SCHEMA_VERSION,
        "domain": "video-lab.adaptive.review-binding",
        "study_id": study_id,
        "optimizer_step_ordinal": optimizer_step_ordinal,
        "optimizer_output_digest": optimizer_output_digest,
        "experiment_id": str(session_row["experiment_id"]),
        "review_session_id": str(session_row["id"]),
        "reviewer_id": str(session_row["reviewer_id"]),
        "left_variant_id": str(session_row["left_variant_id"]),
        "right_variant_id": str(session_row["right_variant_id"]),
        "left_job_id": str(session_row["left_job_id"]),
        "right_job_id": str(session_row["right_job_id"]),
        "left_artifact_id": str(session_row["left_artifact_id"]),
        "right_artifact_id": str(session_row["right_artifact_id"]),
        "left_artifact_sha256": str(session_row["left_artifact_sha256"]),
        "right_artifact_sha256": str(session_row["right_artifact_sha256"]),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _is_adaptive_experiment(connection: sqlite3.Connection, experiment_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM adaptive_study_events "
        "WHERE json_extract(payload_json, '$.experiment_id') = ? LIMIT 1",
        (experiment_id,),
    ).fetchone()
    return row is not None


def _validate_and_order_scores(scores: Sequence[RubricScore]) -> tuple[RubricScore, ...]:
    by_key = {(score.display_position, score.dimension): score for score in scores}
    if len(scores) != 10 or set(by_key) != _SCORE_KEYS:
        raise ValueError("review requires exactly ten unique rubric scores")
    return tuple(
        by_key[(position, dimension)] for position in _POSITIONS for dimension in ReviewDimension
    )


def _public_session(row: sqlite3.Row) -> ReviewSession:
    return ReviewSession(
        id=str(row["id"]),
        experiment_id=str(row["experiment_id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _revision(row: sqlite3.Row, score_rows: Sequence[sqlite3.Row]) -> ReviewRevision:
    return ReviewRevision(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        preferred_variant_id=(
            None if row["preferred_variant_id"] is None else str(row["preferred_variant_id"])
        ),
        display_preference=DisplayPreference(str(row["display_preference"])),
        scores=tuple(
            RubricScore(
                display_position=cast(Literal["left", "right"], str(score["display_position"])),
                dimension=ReviewDimension(str(score["dimension"])),
                score=int(score["score"]),
            )
            for score in score_rows
        ),
        notes=None if row["notes"] is None else str(row["notes"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        revision=int(row["revision"]),
        supersedes_review_id=(
            None if row["supersedes_review_id"] is None else str(row["supersedes_review_id"])
        ),
    )
