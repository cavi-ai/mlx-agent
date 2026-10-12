from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from video_lab.artifacts import ArtifactError, ArtifactStore
from video_lab.database import Database
from video_lab.domain import FrozenModel
from video_lab.repositories import ExperimentRepository


class EvidenceNotFound(LookupError):
    """The requested persisted experiment does not exist."""


class ScalarRange(FrozenModel):
    min: float
    max: float
    last: float


class TelemetryEvidence(FrozenModel):
    sample_count: int
    elapsed_seconds: ScalarRange
    process_rss_bytes: ScalarRange
    system_used_bytes: ScalarRange
    system_available_bytes: ScalarRange
    swap_used_bytes: ScalarRange
    last_thermal_pressure: str | None


class EventEvidence(FrozenModel):
    sequence: int
    kind: str
    payload: dict[str, object]
    created_at: str


class AttemptEvidence(FrozenModel):
    id: str
    number: int
    status: str
    started_at: str
    finished_at: str | None
    error_code: str | None
    error_message: str | None
    artifact_id: str | None
    events: tuple[EventEvidence, ...]
    telemetry: TelemetryEvidence | None


class ArtifactEvidence(FrozenModel):
    id: str
    sha256: str
    byte_size: int
    media_type: str
    created_at: str
    integrity: str
    media_url: str | None


class JobEvidence(FrozenModel):
    id: str
    variant_label: str
    state: str
    created_at: str
    updated_at: str
    error_code: str | None
    error_message: str | None
    attempts: tuple[AttemptEvidence, ...]
    artifacts: tuple[ArtifactEvidence, ...]


class CanonicalDiff(FrozenModel):
    path: str
    changed_paths: tuple[str, ...]
    baseline: Any
    alternate: Any


class VariantEvidence(FrozenModel):
    id: str
    label: str
    manifest: dict[str, object]


class ReviewRevisionEvidence(FrozenModel):
    id: str
    revision: int
    display_preference: str
    preferred_variant_label: str | None
    notes: str | None
    supersedes_review_id: str | None
    created_at: str
    scores: tuple[dict[str, object], ...]


class ReviewEvidence(FrozenModel):
    reviewable: bool
    reason_codes: tuple[str, ...]
    entry_url: str | None
    resume_available: bool = False
    session_id: str | None
    submitted: bool
    canonical_reveal: dict[str, object] | None
    revisions: tuple[ReviewRevisionEvidence, ...]


class ExperimentHistoryItem(FrozenModel):
    id: str
    sealed: bool
    variable_path: str | None
    declared_change: str | None
    created_at: str
    states: tuple[str, ...]
    attempt_count: int
    evidence_completeness: str


class ExperimentHistory(FrozenModel):
    schema_version: str = "evidence.v1"
    filters: dict[str, str]
    items: tuple[ExperimentHistoryItem, ...]


class ExperimentEvidence(FrozenModel):
    schema_version: str = "evidence.v1"
    id: str
    sealed: bool
    experiment_schema_version: str | None
    created_at: str | None
    canonical_diff: CanonicalDiff | None
    variants: tuple[VariantEvidence, ...]
    jobs: tuple[JobEvidence, ...]
    states: tuple[str, ...]
    attempt_count: int
    review: ReviewEvidence
    evidence_completeness: str


class EvidenceService:
    """Read-only projection over immutable laboratory persistence."""

    def __init__(self, database: Database, artifacts: ArtifactStore) -> None:
        self.database = database
        self.artifacts = artifacts
        self.experiments = ExperimentRepository(database)

    def list(
        self,
        *,
        reviewer_id: str | None = None,
        state: str | None = None,
        variable: str | None = None,
    ) -> ExperimentHistory:
        filters = {key: value for key, value in (("state", state), ("variable", variable)) if value}
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT e.id, e.variable_path, e.created_at, "
                "group_concat(j.state, '|') AS states, count(DISTINCT ra.id) AS attempts "
                "FROM experiments e JOIN variants v ON v.experiment_id=e.id "
                "JOIN jobs j ON j.variant_id=v.id "
                "LEFT JOIN run_attempts ra ON ra.job_id=j.id "
                "WHERE NOT EXISTS (SELECT 1 FROM adaptive_study_events AS event "
                "WHERE json_extract(event.payload_json, '$.experiment_id') = e.id) "
                "AND e.schema_version <> 'validation-controlled.v1' "
                "GROUP BY e.id ORDER BY e.created_at DESC, e.id DESC"
            ).fetchall()
        items: list[ExperimentHistoryItem] = []
        for row in rows:
            states = tuple(sorted(set(str(row["states"]).split("|"))))
            variable_path = str(row["variable_path"])
            if state and state not in states:
                continue
            revealed = reviewer_id is not None and self.owner_submitted(str(row["id"]), reviewer_id)
            if variable and (not revealed or variable != variable_path):
                continue
            jobs = self._jobs_for_experiment(str(row["id"]))
            items.append(
                ExperimentHistoryItem(
                    id=str(row["id"]),
                    sealed=not revealed,
                    variable_path=variable_path if revealed else None,
                    declared_change=variable_path if revealed else None,
                    created_at=str(row["created_at"]),
                    states=states,
                    attempt_count=int(row["attempts"]),
                    evidence_completeness="complete" if _complete_pair(jobs) else "partial",
                )
            )
        return ExperimentHistory(filters=filters, items=tuple(items))

    def detail(self, experiment_id: str, reviewer_id: str) -> ExperimentEvidence:
        if not self.experiments.is_generic_visible(experiment_id):
            raise EvidenceNotFound(f"experiment not found: {experiment_id}")
        with self.database.connect() as connection:
            experiment = connection.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if experiment is None:
                raise EvidenceNotFound(f"experiment not found: {experiment_id}")
            variant_rows = connection.execute(
                "SELECT * FROM variants WHERE experiment_id=? ORDER BY label", (experiment_id,)
            ).fetchall()
            job_rows = connection.execute(
                "SELECT j.*, v.label FROM jobs j JOIN variants v ON v.id=j.variant_id "
                "WHERE v.experiment_id=? ORDER BY v.label, j.created_at, j.id",
                (experiment_id,),
            ).fetchall()
            session = connection.execute(
                "SELECT * FROM review_sessions WHERE experiment_id=? AND reviewer_id=?",
                (experiment_id, reviewer_id),
            ).fetchone()
        variants = tuple(self._variant(row) for row in variant_rows)
        manifests = {item.label: item.manifest for item in variants}
        variable_path = str(experiment["variable_path"])
        jobs = tuple(self._job(row) for row in job_rows)
        review = self._review(experiment_id, session, variants, jobs)
        states = tuple(sorted(job.state for job in jobs))
        attempt_count = sum(len(job.attempts) for job in jobs)
        complete = "complete" if _complete_pair(jobs) else "partial"
        if not review.submitted:
            return ExperimentEvidence(
                id=experiment_id,
                sealed=True,
                experiment_schema_version=None,
                created_at=None,
                canonical_diff=None,
                variants=(),
                jobs=(),
                states=states,
                attempt_count=attempt_count,
                review=review,
                evidence_completeness=complete,
            )
        return ExperimentEvidence(
            id=experiment_id,
            sealed=False,
            experiment_schema_version=str(experiment["schema_version"]),
            created_at=str(experiment["created_at"]),
            canonical_diff=CanonicalDiff(
                path=variable_path,
                changed_paths=tuple(json.loads(str(experiment["changed_paths_json"]))),
                baseline=_value_at(manifests["A"], variable_path),
                alternate=_value_at(manifests["B"], variable_path),
            ),
            variants=variants,
            jobs=jobs,
            states=states,
            attempt_count=attempt_count,
            review=review,
            evidence_completeness=complete,
        )

    def owner_submitted(self, experiment_id: str, reviewer_id: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM review_sessions s JOIN review_revisions r ON r.session_id=s.id "
                "WHERE s.experiment_id=? AND s.reviewer_id=? LIMIT 1",
                (experiment_id, reviewer_id),
            ).fetchone()
        return row is not None

    def _jobs_for_experiment(self, experiment_id: str) -> tuple[JobEvidence, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT j.*, v.label FROM jobs j JOIN variants v ON v.id=j.variant_id "
                "WHERE v.experiment_id=? ORDER BY v.label, j.created_at, j.id",
                (experiment_id,),
            ).fetchall()
        return tuple(self._job(row) for row in rows)

    def _variant(self, row: sqlite3.Row) -> VariantEvidence:
        payload = json.loads(str(row["manifest_json"]))
        return VariantEvidence(
            id=str(row["id"]), label=str(row["label"]), manifest=_safe_mapping(payload)
        )

    def _job(self, row: sqlite3.Row) -> JobEvidence:
        job_id = str(row["id"])
        with self.database.connect() as connection:
            attempts = connection.execute(
                "SELECT * FROM run_attempts WHERE job_id=? ORDER BY number, id", (job_id,)
            ).fetchall()
            artifact_rows = connection.execute(
                "SELECT a.* FROM artifacts a JOIN job_artifacts ja ON ja.artifact_id=a.id "
                "WHERE ja.job_id=? ORDER BY ja.created_at, a.id",
                (job_id,),
            ).fetchall()
        return JobEvidence(
            id=job_id,
            variant_label=str(row["label"]),
            state=str(row["state"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error_code=_optional(row["error_code"]),
            error_message=_safe_text(_optional(row["error_message"])),
            attempts=tuple(self._attempt(item) for item in attempts),
            artifacts=tuple(self._artifact(item) for item in artifact_rows),
        )

    def _attempt(self, row: sqlite3.Row) -> AttemptEvidence:
        attempt_id = str(row["id"])
        with self.database.connect() as connection:
            events = connection.execute(
                "SELECT * FROM run_events WHERE attempt_id=? ORDER BY id", (attempt_id,)
            ).fetchall()
            samples = connection.execute(
                "SELECT * FROM telemetry_samples WHERE attempt_id=? ORDER BY recorded_at, id",
                (attempt_id,),
            ).fetchall()
        return AttemptEvidence(
            id=attempt_id,
            number=int(row["number"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            finished_at=_optional(row["finished_at"]),
            error_code=_optional(row["error_code"]),
            error_message=_safe_text(_optional(row["error_message"])),
            artifact_id=_optional(row["artifact_id"]),
            events=tuple(
                EventEvidence(
                    sequence=int(event["id"]),
                    kind=str(event["kind"])[:80],
                    payload=_safe_mapping(json.loads(str(event["payload_json"]))),
                    created_at=str(event["created_at"]),
                )
                for event in events
            ),
            telemetry=_telemetry(samples),
        )

    def _artifact(self, row: sqlite3.Row) -> ArtifactEvidence:
        artifact_id = str(row["id"])
        integrity = "verified"
        try:
            self.artifacts.resolve(artifact_id)
        except ArtifactError as error:
            integrity = "missing" if "missing" in str(error).lower() else "corrupt"
        media_type = str(row["media_type"])
        return ArtifactEvidence(
            id=artifact_id,
            sha256=str(row["sha256"]),
            byte_size=int(row["byte_size"]),
            media_type=media_type,
            created_at=str(row["created_at"]),
            integrity=integrity,
            media_url=f"/media/artifacts/{artifact_id}"
            if media_type == "video/mp4" and integrity == "verified"
            else None,
        )

    def _review(
        self,
        experiment_id: str,
        session: sqlite3.Row | None,
        variants: tuple[VariantEvidence, ...],
        jobs: tuple[JobEvidence, ...],
    ) -> ReviewEvidence:
        reasons: list[str] = []
        if len(jobs) != 2 or any(job.state != "completed" for job in jobs):
            reasons.append("jobs_not_completed")
        if any(len(job.artifacts) != 1 for job in jobs):
            reasons.append("missing_or_duplicate_artifact")
        if any(
            job.artifacts
            and (
                job.artifacts[0].media_type != "video/mp4"
                or job.artifacts[0].integrity != "verified"
            )
            for job in jobs
        ):
            reasons.append("artifact_unavailable")
        if session is None:
            return ReviewEvidence(
                reviewable=not reasons,
                reason_codes=tuple(reasons),
                entry_url=f"/review/{experiment_id}" if not reasons else None,
                session_id=None,
                submitted=False,
                canonical_reveal=None,
                revisions=(),
            )
        session_id = str(session["id"])
        with self.database.connect() as connection:
            revision_rows = connection.execute(
                "SELECT * FROM review_revisions WHERE session_id=? ORDER BY revision, id",
                (session_id,),
            ).fetchall()
            scores = {
                str(row["id"]): connection.execute(
                    "SELECT display_position, dimension, score FROM rubric_scores "
                    "WHERE review_id=? ORDER BY display_position, dimension",
                    (str(row["id"]),),
                ).fetchall()
                for row in revision_rows
            }
        label_by_id = {variant.id: variant.label for variant in variants}
        revisions = tuple(
            ReviewRevisionEvidence(
                id=str(row["id"]),
                revision=int(row["revision"]),
                display_preference=str(row["display_preference"]),
                preferred_variant_label=None
                if row["preferred_variant_id"] is None
                else label_by_id.get(str(row["preferred_variant_id"])),
                notes=_safe_text(_optional(row["notes"])),
                supersedes_review_id=_optional(row["supersedes_review_id"]),
                created_at=str(row["created_at"]),
                scores=tuple(
                    {
                        "display_position": str(score["display_position"]),
                        "dimension": str(score["dimension"]),
                        "score": int(score["score"]),
                    }
                    for score in scores[str(row["id"])]
                ),
            )
            for row in revision_rows
        )
        if not revisions:
            return ReviewEvidence(
                reviewable=not reasons,
                reason_codes=tuple(reasons),
                entry_url=f"/review/{experiment_id}" if not reasons else None,
                resume_available=not reasons,
                session_id=None,
                submitted=False,
                canonical_reveal=None,
                revisions=(),
            )
        submitted = bool(revisions)
        reveal: dict[str, object] | None = None
        if submitted:
            reveal = {
                "left_variant_label": label_by_id[str(session["left_variant_id"])],
                "right_variant_label": label_by_id[str(session["right_variant_id"])],
                "latest_review_id": revisions[-1].id,
            }
        return ReviewEvidence(
            reviewable=not reasons,
            reason_codes=tuple(reasons),
            entry_url=f"/reviews/{session_id}/receipt" if submitted else None,
            session_id=session_id,
            submitted=submitted,
            canonical_reveal=reveal,
            revisions=revisions,
        )


def _optional(value: object) -> str | None:
    return None if value is None else str(value)


_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9:])(?:~|/)(?:[^\s/\"'<>]+/)+[^\s\"'<>]+")


def _safe_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _LOCAL_PATH.sub("[local-path]", value)[:500]


def _safe_mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, object] = {}
    for key, value in payload.items():
        safe = _safe_value(value)
        if safe is not _UNSUPPORTED:
            result[str(key)] = safe
    return result


_UNSUPPORTED = object()


def _safe_value(value: object) -> object:
    if isinstance(value, str):
        return _safe_text(value) or ""
    if isinstance(value, dict):
        return _safe_mapping(value)
    if isinstance(value, list):
        return [safe for item in value if (safe := _safe_value(item)) is not _UNSUPPORTED]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _UNSUPPORTED


def _complete_pair(jobs: tuple[JobEvidence, ...]) -> bool:
    return len(jobs) == 2 and all(
        job.state == "completed"
        and len(job.artifacts) == 1
        and job.artifacts[0].media_type == "video/mp4"
        and job.artifacts[0].integrity == "verified"
        for job in jobs
    )


def _value_at(payload: dict[str, object], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _telemetry(rows: list[sqlite3.Row]) -> TelemetryEvidence | None:
    if not rows:
        return None

    def values(name: str) -> ScalarRange:
        series = [float(row[name]) for row in rows]
        return ScalarRange(min=min(series), max=max(series), last=series[-1])

    return TelemetryEvidence(
        sample_count=len(rows),
        elapsed_seconds=values("elapsed_seconds"),
        process_rss_bytes=values("process_rss_bytes"),
        system_used_bytes=values("system_used_bytes"),
        system_available_bytes=values("system_available_bytes"),
        swap_used_bytes=values("swap_used_bytes"),
        last_thermal_pressure=_optional(rows[-1]["thermal_pressure"]),
    )
