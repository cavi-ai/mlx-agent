from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal, cast

from video_lab.artifacts import Artifact, ArtifactError, ArtifactStore
from video_lab.domain import (
    DisplayPreference,
    FrozenModel,
    Manifest,
    ReviewRevision,
    RubricScore,
)
from video_lab.repositories import (
    ExperimentRepository,
    JobRepository,
    StoredExperiment,
    validation_experiment_id,
)
from video_lab.reviews import ReviewRepository

DisplayPosition = Literal["left", "right"]
_POSITIONS: tuple[DisplayPosition, DisplayPosition] = ("left", "right")


class ReviewQueueItem(FrozenModel):
    experiment_id: str
    reviewable: bool
    reason_codes: tuple[str, ...]


class BlindMedia(FrozenModel):
    display_position: DisplayPosition
    url: str
    fps: float = 1.0


class BlindReviewView(FrozenModel):
    session_id: str
    media: tuple[BlindMedia, BlindMedia]
    prompt: str | None = None


class ReviewSubmission(FrozenModel):
    review_id: str
    revision: int
    status: Literal["submitted", "corrected"]


class ReviewRevealView(FrozenModel):
    review_id: str
    revision: int
    canonical_by_display: dict[DisplayPosition, Literal["A", "B"]]
    variable_path: str
    baseline_value: Any
    alternate_value: Any
    structural_diff: tuple[str, ...]
    manifests: dict[Literal["A", "B"], Manifest]
    artifacts: dict[Literal["A", "B"], Artifact]
    preferred_canonical_variant: Literal["A", "B"] | None


class ReviewService:
    def __init__(
        self,
        experiments: ExperimentRepository,
        jobs: JobRepository,
        artifacts: ArtifactStore,
        reviews: ReviewRepository,
    ) -> None:
        self.experiments = experiments
        self.jobs = jobs
        self.artifacts = artifacts
        self.reviews = reviews

    def queue(self) -> tuple[ReviewQueueItem, ...]:
        return tuple(self._queue_item(experiment) for experiment in self.experiments.list())

    def start(self, experiment_id: str, reviewer_id: str) -> BlindReviewView:
        return self._start(self.experiments.get(experiment_id), reviewer_id)

    def start_adaptive(self, experiment_id: str, reviewer_id: str) -> BlindReviewView:
        return self._start(self.experiments.get_adaptive(experiment_id), reviewer_id)

    def start_validation(
        self,
        *,
        campaign_id: str,
        trial_id: str,
        reviewer_id: str,
    ) -> BlindReviewView:
        experiment_id, winner_position, hidden_repeat = self._validation_trial_authorization(
            campaign_id, trial_id, reviewer_id
        )
        effective_reviewer = _validation_session_reviewer(
            campaign_id, trial_id, reviewer_id, hidden_repeat=hidden_repeat
        )
        view = self._start(
            self.experiments.get_validation(experiment_id),
            effective_reviewer,
            validation=True,
        )
        row = self._session_row(view.session_id)
        with self.reviews.database.connect() as connection:
            baseline = connection.execute(
                "SELECT id FROM variants WHERE experiment_id = ? AND label = 'A'",
                (experiment_id,),
            ).fetchone()
        if baseline is None:
            raise ValueError("validation experiment has no winner variant")
        actual_winner = "left" if str(row["left_variant_id"]) == str(baseline["id"]) else "right"
        swap = actual_winner != winner_position
        return BlindReviewView(
            session_id=view.session_id,
            media=(
                BlindMedia(
                    display_position="left",
                    url=f"/validation/reviews/{view.session_id}/artifacts/"
                    f"{'right' if swap else 'left'}",
                ),
                BlindMedia(
                    display_position="right",
                    url=f"/validation/reviews/{view.session_id}/artifacts/"
                    f"{'left' if swap else 'right'}",
                ),
            ),
            prompt=view.prompt,
        )

    def submit_validation(
        self,
        *,
        campaign_id: str,
        trial_id: str,
        reviewer_id: str,
        preference: DisplayPreference,
    ) -> ReviewRevision:
        experiment_id, winner_position, hidden_repeat = self._validation_trial_authorization(
            campaign_id, trial_id, reviewer_id
        )
        effective_reviewer = _validation_session_reviewer(
            campaign_id, trial_id, reviewer_id, hidden_repeat=hidden_repeat
        )
        session = self.reviews.start_session(
            experiment_id,
            effective_reviewer,
            self._current_eligibility(experiment_id, propagate_artifact_errors=True)[1],
        )
        row = self._session_row(session.id)
        with self.reviews.database.connect() as connection:
            baseline = connection.execute(
                "SELECT id FROM variants WHERE experiment_id = ? AND label = 'A'",
                (experiment_id,),
            ).fetchone()
        if baseline is None:
            raise ValueError("validation experiment has no winner variant")
        actual_winner = "left" if str(row["left_variant_id"]) == str(baseline["id"]) else "right"
        actual_preference = preference
        if preference is not DisplayPreference.TIE and actual_winner != winner_position:
            actual_preference = (
                DisplayPreference.RIGHT
                if preference is DisplayPreference.LEFT
                else DisplayPreference.LEFT
            )
        return self.reviews.submit_adaptive(session.id, actual_preference)

    def validation_trial_for_session(
        self, *, campaign_id: str, session_id: str, reviewer_id: str
    ) -> str:
        session = self._session_row(session_id)
        with self.reviews.database.connect() as connection:
            rows = connection.execute(
                "SELECT trial.trial_id, trial.reuses_trial_id FROM validation_trials AS trial "
                "JOIN validation_campaigns AS campaign ON campaign.id = trial.campaign_id "
                "WHERE trial.campaign_id = ? AND campaign.reviewer_id = ?",
                (campaign_id, reviewer_id),
            ).fetchall()
        for row in rows:
            trial_id = str(row["trial_id"])
            source_trial_id = str(row["reuses_trial_id"] or trial_id)
            if str(session["reviewer_id"]) == _validation_session_reviewer(
                campaign_id,
                trial_id,
                reviewer_id,
                hidden_repeat=row["reuses_trial_id"] is not None,
            ) and str(session["experiment_id"]) == validation_experiment_id(
                campaign_id, source_trial_id
            ):
                return trial_id
        raise LookupError(f"validation review not found: {session_id}")

    def artifact_for_validation_display(
        self,
        *,
        campaign_id: str,
        trial_id: str,
        session_id: str,
        position: str,
        reviewer_id: str,
    ) -> Artifact:
        normalized = self._position(position)
        bound_trial_id = self.validation_trial_for_session(
            campaign_id=campaign_id,
            session_id=session_id,
            reviewer_id=reviewer_id,
        )
        if bound_trial_id != trial_id:
            raise LookupError(f"validation review not found: {session_id}")
        row = self._session_row(session_id)
        _, winner_position, _ = self._validation_trial_authorization(
            campaign_id, trial_id, reviewer_id
        )
        with self.reviews.database.connect() as connection:
            baseline = connection.execute(
                "SELECT id FROM variants WHERE experiment_id=? AND label='A'",
                (row["experiment_id"],),
            ).fetchone()
        if baseline is None:
            raise ValueError("validation experiment has no winner variant")
        actual_winner = "left" if str(row["left_variant_id"]) == str(baseline["id"]) else "right"
        underlying = normalized
        if actual_winner != winner_position:
            underlying = "right" if normalized == "left" else "left"
        return self._pinned_artifact(row, self._position(underlying))

    def _start(
        self, experiment: StoredExperiment, reviewer_id: str, *, validation: bool = False
    ) -> BlindReviewView:
        reasons, artifacts_by_variant = self._current_eligibility(experiment.id)
        ordered = tuple(dict.fromkeys(reasons))
        if ordered:
            raise ValueError("experiment is not reviewable: " + ",".join(ordered))
        session = self.reviews.start_session(experiment.id, reviewer_id, artifacts_by_variant)
        # Resolve through the store at the authorization boundary, even after queue validation.
        for position in session.display_positions:
            if validation:
                self._pinned_artifact(self._session_row(session.id), position)
            else:
                self.artifact_for_display(session.id, position, reviewer_id)
        return BlindReviewView(
            session_id=session.id,
            media=(
                BlindMedia(
                    display_position="left",
                    url=f"/reviews/{session.id}/artifacts/left",
                    fps=1.0,
                ),
                BlindMedia(
                    display_position="right",
                    url=f"/reviews/{session.id}/artifacts/right",
                    fps=1.0,
                ),
            ),
            prompt=self._shared_prompt(experiment),
        )

    def prompt_for_session(self, session_id: str, reviewer_id: str) -> str | None:
        self._require_generic_session(session_id)
        row = self._session_row(session_id)
        if str(row["reviewer_id"]) != reviewer_id:
            raise PermissionError("review session belongs to another reviewer")
        return self._shared_prompt(self.experiments.get(str(row["experiment_id"])))

    def artifact_for_display(self, session_id: str, position: str, reviewer_id: str) -> Artifact:
        self._require_generic_session(session_id)
        normalized = self._position(position)
        row = self._session_row(session_id)
        if str(row["reviewer_id"]) != reviewer_id:
            raise PermissionError("review session belongs to another reviewer")
        return self._pinned_artifact(row, normalized)

    def submit(
        self,
        session_id: str,
        reviewer_id: str,
        preference: DisplayPreference | None,
        scores: Sequence[RubricScore],
        notes: str | None,
    ) -> ReviewSubmission:
        self._require_generic_session(session_id)
        if preference is None:
            raise ValueError("display preference is required")
        self._validate_owner_and_artifacts(session_id, reviewer_id)
        if self.reviews.is_adaptive_session(session_id):
            raise ValueError("adaptive review must use immutable preference submission")
        revision = self.reviews.submit(session_id, preference, scores, notes)
        return self._submission(revision, "submitted")

    def submit_adaptive(
        self,
        session_id: str,
        reviewer_id: str,
        preference: DisplayPreference,
    ) -> ReviewRevision:
        self._validate_owner_and_artifacts(session_id, reviewer_id)
        if not self.reviews.is_adaptive_session(session_id):
            raise ValueError("review session is not bound to an adaptive study")
        return self.reviews.submit_adaptive(session_id, preference)

    def latest_adaptive(self, session_id: str, reviewer_id: str) -> ReviewRevision | None:
        row = self._session_row(session_id)
        if str(row["reviewer_id"]) != reviewer_id:
            raise PermissionError("review session belongs to another reviewer")
        if not self.reviews.is_adaptive_session(session_id):
            raise ValueError("review session is not bound to an adaptive study")
        return self.reviews.latest_for_session(session_id)

    def adaptive_binding_digest(
        self,
        session_id: str,
        reviewer_id: str,
        *,
        study_id: str,
        optimizer_step_ordinal: int,
        optimizer_output_digest: str,
    ) -> str:
        return self.reviews.adaptive_binding_digest(
            session_id,
            reviewer_id,
            study_id=study_id,
            optimizer_step_ordinal=optimizer_step_ordinal,
            optimizer_output_digest=optimizer_output_digest,
        )

    def correct(
        self,
        session_id: str,
        reviewer_id: str,
        supersedes_review_id: str,
        preference: DisplayPreference | None,
        scores: Sequence[RubricScore],
        notes: str | None,
    ) -> ReviewSubmission:
        self._require_generic_session(session_id)
        if preference is None:
            raise ValueError("display preference is required")
        row = self._session_row(session_id)
        if str(row["reviewer_id"]) != reviewer_id:
            raise PermissionError("review session belongs to another reviewer")
        if self.reviews.is_adaptive_session(session_id):
            raise ValueError("adaptive v1 review is immutable")
        self._validate_owner_and_artifacts(session_id, reviewer_id)
        revision = self.reviews.submit(
            session_id, preference, scores, notes, supersedes_id=supersedes_review_id
        )
        return self._submission(revision, "corrected")

    def reveal(self, session_id: str, reviewer_id: str) -> ReviewRevealView:
        self._require_generic_session(session_id)
        row = self._session_row(session_id)
        if str(row["reviewer_id"]) != reviewer_id:
            raise PermissionError("review session belongs to another reviewer")
        revision = self.reviews.latest_for_session(session_id)
        if revision is None:
            raise PermissionError("review must be submitted before reveal")
        experiment = self.experiments.get(str(row["experiment_id"]))
        variants = {variant.id: variant for variant in experiment.variants}
        canonical_by_id = {
            variant.id: cast(Literal["A", "B"], variant.label) for variant in experiment.variants
        }
        mapping: dict[DisplayPosition, Literal["A", "B"]] = {
            "left": canonical_by_id[str(row["left_variant_id"])],
            "right": canonical_by_id[str(row["right_variant_id"])],
        }
        artifacts = {
            mapping[position]: self._pinned_artifact(row, position) for position in _POSITIONS
        }
        baseline = experiment.variants[0].manifest
        alternate = experiment.variants[1].manifest
        return ReviewRevealView(
            review_id=revision.id,
            revision=revision.revision,
            canonical_by_display=mapping,
            variable_path=experiment.variable_path,
            baseline_value=_value_at(baseline, experiment.variable_path),
            alternate_value=_value_at(alternate, experiment.variable_path),
            structural_diff=experiment.changed_paths,
            manifests={
                "A": variants[experiment.variants[0].id].manifest,
                "B": variants[experiment.variants[1].id].manifest,
            },
            artifacts=artifacts,
            preferred_canonical_variant=(
                None
                if revision.preferred_variant_id is None
                else canonical_by_id[revision.preferred_variant_id]
            ),
        )

    def _queue_item(self, experiment: StoredExperiment) -> ReviewQueueItem:
        reasons, _ = self._current_eligibility(experiment.id)
        ordered = tuple(dict.fromkeys(reasons))
        return ReviewQueueItem(
            experiment_id=experiment.id, reviewable=not ordered, reason_codes=ordered
        )

    @staticmethod
    def _shared_prompt(experiment: StoredExperiment) -> str | None:
        prompts = {variant.manifest.prompt for variant in experiment.variants}
        return prompts.pop() if len(prompts) == 1 else None

    def _current_eligibility(
        self, experiment_id: str, *, propagate_artifact_errors: bool = False
    ) -> tuple[list[str], dict[str, Artifact]]:
        jobs = self.jobs.list_for_experiment(experiment_id)
        reasons: list[str] = []
        artifacts_by_variant: dict[str, Artifact] = {}
        if len(jobs) != 2 or any(job.state.value != "completed" for job in jobs):
            reasons.append("jobs_not_completed")
        for job in jobs:
            registered = self.artifacts.list_for_job(job.id)
            videos = tuple(item for item in registered if item.media_type == "video/mp4")
            if not registered:
                reasons.append("missing_artifact")
            elif not videos:
                reasons.append("non_video_artifact")
            elif len(registered) != 1 or len(videos) != 1:
                reasons.append("duplicate_artifact")
            else:
                artifact = videos[0]
                try:
                    self.artifacts.resolve(artifact.id)
                except ArtifactError:
                    if propagate_artifact_errors:
                        raise
                    reasons.append("corrupt_artifact")
                else:
                    artifacts_by_variant[job.variant_id] = artifact
        eligible_artifacts = tuple(artifacts_by_variant.values())
        if len(eligible_artifacts) == 2 and (
            eligible_artifacts[0].id == eligible_artifacts[1].id
            or eligible_artifacts[0].sha256 == eligible_artifacts[1].sha256
        ):
            reasons.append("duplicate_variant_artifact")
        return reasons, artifacts_by_variant

    def _validate_owner_and_artifacts(self, session_id: str, reviewer_id: str) -> None:
        row = self._session_row(session_id)
        if str(row["reviewer_id"]) != reviewer_id:
            raise PermissionError("review session belongs to another reviewer")
        reasons, artifacts_by_variant = self._current_eligibility(
            str(row["experiment_id"]), propagate_artifact_errors=True
        )
        ordered = tuple(dict.fromkeys(reasons))
        if ordered:
            raise ValueError("experiment is not reviewable: " + ",".join(ordered))
        for position in _POSITIONS:
            variant_id = str(row[f"{position}_variant_id"])
            current = artifacts_by_variant.get(variant_id)
            if current is None or (
                current.id,
                current.sha256,
            ) != (
                str(row[f"{position}_artifact_id"]),
                str(row[f"{position}_artifact_sha256"]),
            ):
                raise ValueError("current artifacts do not match session-pinned artifacts")

    def _pinned_artifact(self, row: sqlite3.Row, position: DisplayPosition) -> Artifact:
        artifact = self.artifacts.get(str(row[f"{position}_artifact_id"]))
        if artifact.sha256 != str(row[f"{position}_artifact_sha256"]):
            raise ArtifactError("session-pinned artifact identity does not match registry")
        job_id = str(row[f"{position}_job_id"])
        with self.reviews.database.connect() as connection:
            job = connection.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None:
            raise ValueError("session-pinned variant job no longer exists")
        return replace(artifact, job_id=job_id)

    def _session_row(self, session_id: str) -> sqlite3.Row:
        with self.reviews.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise LookupError(f"review session not found: {session_id}")
        return cast(sqlite3.Row, row)

    def _require_generic_session(self, session_id: str) -> None:
        row = self._session_row(session_id)
        with self.reviews.database.connect() as connection:
            experiment = connection.execute(
                "SELECT schema_version FROM experiments WHERE id = ?",
                (row["experiment_id"],),
            ).fetchone()
        if (
            experiment is not None
            and str(experiment["schema_version"]) == "validation-controlled.v1"
        ):
            raise LookupError(f"review session not found: {session_id}")

    def _validation_trial_authorization(
        self, campaign_id: str, trial_id: str, reviewer_id: str
    ) -> tuple[str, str, bool]:
        with self.reviews.database.connect() as connection:
            row = connection.execute(
                "SELECT campaign.reviewer_id, trial.winner_display_position, "
                "trial.reuses_trial_id FROM validation_trials AS trial "
                "JOIN validation_campaigns AS campaign ON campaign.id = trial.campaign_id "
                "WHERE trial.campaign_id = ? AND trial.trial_id = ?",
                (campaign_id, trial_id),
            ).fetchone()
        if row is None or str(row["reviewer_id"]) != reviewer_id:
            raise LookupError(f"validation review not found: {trial_id}")
        source_trial_id = str(row["reuses_trial_id"] or trial_id)
        return (
            validation_experiment_id(campaign_id, source_trial_id),
            str(row["winner_display_position"]),
            row["reuses_trial_id"] is not None,
        )

    @staticmethod
    def _position(position: str) -> DisplayPosition:
        if position not in ("left", "right"):
            raise ValueError("unknown display position")
        return cast(DisplayPosition, position)

    @staticmethod
    def _submission(
        revision: ReviewRevision, status: Literal["submitted", "corrected"]
    ) -> ReviewSubmission:
        return ReviewSubmission(review_id=revision.id, revision=revision.revision, status=status)


def _value_at(manifest: Manifest, path: str) -> Any:
    value: Any = manifest.model_dump(mode="json")
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _validation_session_reviewer(
    campaign_id: str,
    trial_id: str,
    reviewer_id: str,
    *,
    hidden_repeat: bool,
) -> str:
    if not hidden_repeat:
        return reviewer_id
    payload = f"validation-review-session.v1\x00{campaign_id}\x00{trial_id}\x00{reviewer_id}"
    return "validation_reviewer_" + hashlib.sha256(payload.encode()).hexdigest()
