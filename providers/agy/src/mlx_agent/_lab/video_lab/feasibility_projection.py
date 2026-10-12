from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import cast

from video_lab.artifacts import ArtifactError
from video_lab.feasibility import (
    FeasibilityProtocol,
    FeasibilityStatus,
    derive_display_verdict,
    sanitize_diagnostic,
)
from video_lab.feasibility_models import FeasibilityStage, StageAttempt, StageOutcome
from video_lab.feasibility_repository import FeasibilityRepository

_SCHEMA = "feasibility-evidence.v1"
_GENERATION_STAGES = {
    FeasibilityStage.GENERATE_MINIMUM_1,
    FeasibilityStage.GENERATE_MINIMUM_2,
    FeasibilityStage.QUALIFY_STANDARD,
    FeasibilityStage.VALIDATE_CONTROLLED_PAIR,
}


def normalize_reviewer_id(reviewer_id: str | None) -> str | None:
    if reviewer_id is None:
        return None
    normalized = reviewer_id.strip()
    return normalized or None


def next_feasibility_stage(attempts: Sequence[StageAttempt]) -> str | None:
    latest = {attempt.stage: attempt for attempt in attempts}
    for definition in FeasibilityProtocol.default().stages:
        attempt = latest.get(definition.stage)
        if attempt is None:
            return definition.stage.value
        if attempt.outcome is StageOutcome.PASSED:
            continue
        if attempt.outcome in (StageOutcome.INTERRUPTED, StageOutcome.UNEXPECTED_FAILURE):
            return definition.stage.value
        return None
    return None


class FeasibilityEvidenceProjector:
    """Builds one path-neutral, read-only phase projection for CLI, web, and export."""

    def __init__(self, repository: FeasibilityRepository) -> None:
        self.repository = repository

    def for_run(
        self,
        run_id: str,
        *,
        reviewer_id: str | None = None,
        completed_attempt: StageAttempt | None = None,
    ) -> dict[str, object]:
        run = self.repository.get_run(run_id)
        attempts = self.repository.list_attempts(run_id)
        status = FeasibilityStatus(run, attempts, derive_display_verdict(attempts))
        return self.project(
            status,
            reviewer_id=reviewer_id,
            completed_attempt=completed_attempt,
        )

    def project(
        self,
        status: FeasibilityStatus,
        *,
        reviewer_id: str | None = None,
        completed_attempt: StageAttempt | None = None,
    ) -> dict[str, object]:
        attempts = status.attempts
        reviewer = normalize_reviewer_id(reviewer_id)
        authorization = self.repository.generation_authorization_bundle(status.run.adapter_id)
        authorized = (
            authorization
            if authorization is not None and authorization.run_id == status.run.id
            else None
        )
        if not self._review_submitted(attempts, reviewer):
            return self._sealed_projection(
                status,
                authorized=authorized is not None,
                completed_attempt=completed_attempt,
            )
        resources = {
            attempt.stage.value: attempt.evidence.details["preassembly_preflight"]
            for attempt in attempts
            if attempt.evidence is not None and "preassembly_preflight" in attempt.evidence.details
        }
        projected: dict[str, object] = {
            "schema_version": _SCHEMA,
            "projection_state": "revealed",
            "run": status.run.model_dump(mode="json"),
            "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
            "completed_attempt": (
                None if completed_attempt is None else completed_attempt.model_dump(mode="json")
            ),
            "next_stage": next_feasibility_stage(attempts),
            "verdict": status.verdict.value,
            "gates": [
                self._gate(attempts, definition.stage)
                for definition in FeasibilityProtocol.default().stages
            ],
            "resources": {"preassembly_checks": resources},
            "artifacts": self._artifacts(status.run.id, attempts),
            "controlled_pair": self._controlled_pair(attempts, reviewer),
            "provenance": self._provenance(attempts),
            "conclusions": {
                "display_verdict": status.verdict.value,
                "persisted_verdict": status.run.verdict.value,
                "authorized": authorized is not None,
                "authorization": (
                    None
                    if authorized is None
                    else {
                        "run_id": authorized.run_id,
                        "verdict": authorized.verdict.value,
                        "measured_profile": authorized.measured_profile,
                        "canonical_manifest_hash": authorized.canonical_manifest_hash,
                        "source_fingerprint": authorized.source_fingerprint,
                        "runtime_fingerprint": authorized.runtime_fingerprint,
                        "placement_fingerprint": authorized.placement_fingerprint,
                        "authorized_manifest": authorized.authorized_manifest.model_dump(
                            mode="json"
                        ),
                        "resolved_sources": {
                            key: asdict(value) for key, value in authorized.resolved_sources.items()
                        },
                    }
                ),
            },
        }
        return cast(dict[str, object], sanitize_diagnostic(projected))

    def _sealed_projection(
        self,
        status: FeasibilityStatus,
        *,
        authorized: bool,
        completed_attempt: StageAttempt | None,
    ) -> dict[str, object]:
        attempts = status.attempts
        projected: dict[str, object] = {
            "schema_version": _SCHEMA,
            "projection_state": "sealed",
            "run": {
                "id": status.run.id,
                "adapter_id": status.run.adapter_id,
                "protocol_version": status.run.protocol_version,
                "verdict": status.run.verdict.value,
                "created_at": status.run.created_at.isoformat(),
                "updated_at": status.run.updated_at.isoformat(),
            },
            "attempts": [self._sealed_attempt(attempt) for attempt in attempts],
            "completed_attempt": (
                None if completed_attempt is None else self._sealed_attempt(completed_attempt)
            ),
            "next_stage": next_feasibility_stage(attempts),
            "verdict": status.verdict.value,
            "gates": [
                self._sealed_gate(attempts, definition.stage)
                for definition in FeasibilityProtocol.default().stages
            ],
            "resources": {"state": "sealed"},
            "artifacts": self._sealed_artifacts(status.run.id, attempts),
            "controlled_pair": self._sealed_controlled_pair(attempts),
            "provenance": {"state": "sealed"},
            "conclusions": {
                "display_verdict": status.verdict.value,
                "persisted_verdict": status.run.verdict.value,
                "authorized": authorized,
                "authorization": None,
                "authorization_state": "sealed",
            },
        }
        return cast(dict[str, object], sanitize_diagnostic(projected))

    @staticmethod
    def _sealed_attempt(attempt: StageAttempt) -> dict[str, object]:
        return {
            "stage": attempt.stage.value,
            "ordinal": attempt.ordinal,
            "outcome": None if attempt.outcome is None else attempt.outcome.value,
            "started_at": attempt.started_at.isoformat(),
            "finished_at": (
                None if attempt.finished_at is None else attempt.finished_at.isoformat()
            ),
        }

    @classmethod
    def _sealed_gate(
        cls, attempts: Sequence[StageAttempt], stage: FeasibilityStage
    ) -> dict[str, object]:
        attempt = next((item for item in reversed(attempts) if item.stage is stage), None)
        if attempt is None:
            return {"stage": stage.value, "outcome": "pending", "attempt": None}
        return {
            "stage": stage.value,
            "outcome": None if attempt.outcome is None else attempt.outcome.value,
            "attempt": cls._sealed_attempt(attempt),
        }

    @staticmethod
    def _gate(attempts: Sequence[StageAttempt], stage: FeasibilityStage) -> dict[str, object]:
        attempt = next((item for item in reversed(attempts) if item.stage is stage), None)
        if attempt is None:
            return {"stage": stage.value, "outcome": "pending", "attempt": None}
        return {
            "stage": stage.value,
            "outcome": None if attempt.outcome is None else attempt.outcome.value,
            "attempt": attempt.model_dump(mode="json"),
        }

    @staticmethod
    def _pair_details(attempts: Sequence[StageAttempt]) -> Mapping[str, object] | None:
        pair = next(
            (
                attempt
                for attempt in reversed(attempts)
                if attempt.stage is FeasibilityStage.VALIDATE_CONTROLLED_PAIR
                and attempt.evidence is not None
            ),
            None,
        )
        if pair is None or pair.evidence is None:
            return None
        return cast(Mapping[str, object], pair.evidence.details)

    def _review_submitted(
        self,
        attempts: Sequence[StageAttempt],
        reviewer_id: str | None,
    ) -> bool:
        if reviewer_id is None:
            return False
        details = self._pair_details(attempts)
        experiment_id = None if details is None else details.get("experiment_id")
        if not isinstance(experiment_id, str):
            return False
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT review_revisions.id FROM review_revisions "
                "JOIN review_sessions ON review_sessions.id = review_revisions.session_id "
                "JOIN rubric_scores ON rubric_scores.review_id = review_revisions.id "
                "WHERE review_sessions.experiment_id = ? AND review_sessions.reviewer_id = ? "
                "GROUP BY review_revisions.id HAVING COUNT(*) = 10 "
                "ORDER BY review_revisions.created_at, review_revisions.revision LIMIT 1",
                (experiment_id, reviewer_id),
            ).fetchone()
        return row is not None

    def _sealed_controlled_pair(self, attempts: Sequence[StageAttempt]) -> dict[str, object] | None:
        details = self._pair_details(attempts)
        if details is None:
            return None
        identity = details.get("review_identity")
        session_id = identity.get("session_id") if isinstance(identity, Mapping) else None
        if not isinstance(session_id, str) or self._phase_session(session_id) is None:
            return None
        return {
            "display_positions": ["left", "right"],
            "review": {
                "state": "blinded_randomized",
                "submitted": False,
                "revisions": [],
            },
        }

    def _sealed_artifacts(
        self, run_id: str, attempts: Sequence[StageAttempt]
    ) -> list[dict[str, str]]:
        details = self._pair_details(attempts)
        identity = None if details is None else details.get("review_identity")
        session_id = identity.get("session_id") if isinstance(identity, Mapping) else None
        if not isinstance(session_id, str) or self._phase_session(session_id) is None:
            return []
        return [
            {
                "display_position": position,
                "media_url": f"/media/feasibility/{run_id}/{position}",
            }
            for position in ("left", "right")
        ]

    def media_artifact_id(
        self,
        run_id: str,
        handle: str,
        *,
        reviewer_id: str | None = None,
    ) -> str | None:
        attempts = self.repository.list_attempts(run_id)
        reviewer = normalize_reviewer_id(reviewer_id)
        details = self._pair_details(attempts)
        identity = None if details is None else details.get("review_identity")
        session_id = identity.get("session_id") if isinstance(identity, Mapping) else None
        session = self._phase_session(session_id) if isinstance(session_id, str) else None
        if handle in ("left", "right") and session is not None:
            return str(session[f"{handle}_artifact_id"])
        if not self._review_submitted(attempts, reviewer):
            return None
        projected = self._artifacts(run_id, attempts)
        allowed = {
            str(item["artifact_id"])
            for item in projected
            if isinstance(item.get("artifact_id"), str)
        }
        return handle if handle in allowed else None

    def _phase_session(self, session_id: str) -> Mapping[str, object] | None:
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_sessions WHERE id = ? AND reviewer_id LIKE 'feasibility:%'",
                (session_id,),
            ).fetchone()
        return None if row is None else cast(Mapping[str, object], row)

    def _artifacts(self, run_id: str, attempts: Sequence[StageAttempt]) -> list[dict[str, object]]:
        jobs: list[tuple[FeasibilityStage, str, str]] = []
        for attempt in attempts:
            if (
                attempt.stage not in _GENERATION_STAGES
                or attempt.outcome is not StageOutcome.PASSED
                or attempt.evidence is None
            ):
                continue
            details = attempt.evidence.details
            job_id = details.get("job_id")
            if isinstance(job_id, str):
                jobs.append((attempt.stage, "output", job_id))
            job_ids = details.get("job_ids")
            if isinstance(job_ids, tuple):
                jobs.extend(
                    (attempt.stage, f"candidate_{index + 1}", value)
                    for index, value in enumerate(job_ids)
                    if isinstance(value, str)
                )
        projected: list[dict[str, object]] = []
        for stage, slot, job_id in jobs:
            registered = (
                ()
                if self.repository.artifacts is None
                else self.repository.artifacts.list_for_job(job_id)
            )
            if len(registered) != 1:
                continue
            artifact = registered[0]
            integrity = "unverified"
            if self.repository.artifacts is not None:
                try:
                    self.repository.artifacts.resolve(artifact.id)
                except ArtifactError:
                    integrity = "corrupt"
                else:
                    integrity = "verified"
            projected.append(
                {
                    "stage": stage.value,
                    "slot": slot,
                    "job_id": job_id,
                    "artifact_id": artifact.id,
                    "sha256": artifact.sha256,
                    "byte_size": artifact.byte_size,
                    "media_type": artifact.media_type,
                    "integrity": integrity,
                    "media_url": f"/media/feasibility/{run_id}/{artifact.id}",
                    "probe": self._completion_probe(job_id),
                }
            )
        return projected

    def _completion_probe(self, job_id: str) -> Mapping[str, object] | None:
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM run_events WHERE job_id = ? "
                "AND kind = 'job_completed' ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, Mapping):
            return None
        return cast(Mapping[str, object], payload)

    def _controlled_pair(
        self,
        attempts: Sequence[StageAttempt],
        reviewer_id: str | None,
    ) -> dict[str, object] | None:
        pair = next(
            (
                attempt
                for attempt in reversed(attempts)
                if attempt.stage is FeasibilityStage.VALIDATE_CONTROLLED_PAIR
                and attempt.evidence is not None
            ),
            None,
        )
        if pair is None or pair.evidence is None:
            return None
        details = pair.evidence.details
        experiment_id = details.get("experiment_id")
        review = self._review(
            experiment_id if isinstance(experiment_id, str) else None,
            reviewer_id,
        )
        return {
            "experiment_id": details.get("experiment_id"),
            "variable_path": details.get("variable_path"),
            "changed_paths": details.get("changed_paths"),
            "controlled_values": details.get("controlled_values"),
            "pair_seed": details.get("pair_seed"),
            "job_ids": details.get("job_ids"),
            "artifact_sha256": details.get("artifact_sha256"),
            "review": review,
        }

    def _review(
        self,
        experiment_id: str | None,
        reviewer_id: str | None,
    ) -> dict[str, object]:
        empty: dict[str, object] = {
            "session_id": None,
            "state": "missing",
            "revisions": [],
        }
        if experiment_id is None or reviewer_id is None:
            return empty
        with self.repository.database.connect() as connection:
            session = connection.execute(
                "SELECT * FROM review_sessions WHERE experiment_id = ? AND reviewer_id = ?",
                (experiment_id, reviewer_id),
            ).fetchone()
            if session is None:
                return empty
            revisions = connection.execute(
                "SELECT * FROM review_revisions WHERE session_id = ? "
                "ORDER BY review_revisions.created_at, review_revisions.revision",
                (str(session["id"]),),
            ).fetchall()
            projected_revisions: list[dict[str, object]] = []
            for revision in revisions:
                scores = connection.execute(
                    "SELECT display_position, dimension, score FROM rubric_scores "
                    "WHERE review_id = ? ORDER BY display_position, dimension",
                    (str(revision["id"]),),
                ).fetchall()
                if len(scores) != 10:
                    continue
                projected_revisions.append(
                    {
                        "id": str(revision["id"]),
                        "session_id": str(revision["session_id"]),
                        "revision": int(revision["revision"]),
                        "display_preference": str(revision["display_preference"]),
                        "preferred_variant_id": revision["preferred_variant_id"],
                        "notes": revision["notes"],
                        "created_at": str(revision["created_at"]),
                        "supersedes_review_id": revision["supersedes_review_id"],
                        "scores": [dict(score) for score in scores],
                        "display_mapping": {
                            position: {
                                "variant_id": str(session[f"{position}_variant_id"]),
                                "job_id": str(session[f"{position}_job_id"]),
                                "artifact_id": str(session[f"{position}_artifact_id"]),
                                "artifact_sha256": str(session[f"{position}_artifact_sha256"]),
                            }
                            for position in ("left", "right")
                        },
                    }
                )
        return {
            "session_id": str(session["id"]),
            "state": "submitted" if projected_revisions else "blinded_randomized",
            "experiment_id": str(session["experiment_id"]),
            "left": {
                "variant_id": str(session["left_variant_id"]),
                "job_id": str(session["left_job_id"]),
                "artifact_id": str(session["left_artifact_id"]),
                "artifact_sha256": str(session["left_artifact_sha256"]),
            },
            "right": {
                "variant_id": str(session["right_variant_id"]),
                "job_id": str(session["right_job_id"]),
                "artifact_id": str(session["right_artifact_id"]),
                "artifact_sha256": str(session["right_artifact_sha256"]),
            },
            "revisions": projected_revisions,
        }

    @staticmethod
    def _provenance(attempts: Sequence[StageAttempt]) -> dict[str, object]:
        inventory = None
        identities: dict[str, dict[str, object]] = {}
        repeatability = None
        for attempt in attempts:
            if attempt.evidence is None:
                continue
            details = attempt.evidence.details
            if attempt.stage is FeasibilityStage.DOWNLOAD_SOURCES:
                inventory = details.get("inventory")
            if attempt.stage in _GENERATION_STAGES:
                identities[attempt.stage.value] = {
                    key: details.get(key)
                    for key in (
                        "canonical_manifest_hash",
                        "source_fingerprint",
                        "runtime_fingerprint",
                        "placement_fingerprint",
                        "placement",
                        "fallback_violations",
                    )
                }
            if attempt.stage is FeasibilityStage.GENERATE_MINIMUM_2:
                repeatability = details.get("repeatability")
        return {
            "source_inventory": inventory,
            "generation_identities": identities,
            "repeatability": repeatability,
        }


__all__ = ["FeasibilityEvidenceProjector", "next_feasibility_stage"]
