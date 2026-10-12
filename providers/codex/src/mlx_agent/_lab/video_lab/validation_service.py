from __future__ import annotations

import hashlib
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal, cast

from video_lab.adaptive_study_models import AdaptiveStudyState
from video_lab.artifacts import ArtifactError
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    FrozenModel,
    JobState,
    canonical_json,
    utc_now,
)
from video_lab.repositories import RecordNotFound, validation_experiment_id
from video_lab.validation_analysis import analyze_validation
from video_lab.validation_models import (
    ValidationCampaignDefinition,
    ValidationCampaignState,
    ValidationCandidates,
    ValidationChoice,
    ValidationDisposition,
    ValidationEvent,
    ValidationIneligible,
    ValidationJudgment,
    ValidationPanelInput,
    ValidationResult,
    ValidationTrial,
    ValidationTrialBinding,
)
from video_lab.validation_panel import build_validation_panel, derive_validation_candidates
from video_lab.validation_repository import (
    ValidationConflict,
    ValidationNotFound,
    ValidationRepository,
)


class ValidationProgress(FrozenModel):
    completed_judgments: int
    total_judgments: Literal[40] = 40
    generated_primary: int
    total_primary: Literal[36] = 36


class ValidationCampaignPreview(FrozenModel):
    primary_count: Literal[36] = 36
    repeat_count: Literal[4] = 4
    candidates: ValidationCandidates


class ValidationCampaignView(FrozenModel):
    campaign_id: str
    state: ValidationCampaignState
    next_action: str
    progress: ValidationProgress
    preview: ValidationCampaignPreview | None = None
    review_id: str | None = None
    media: tuple[ValidationMediaReference, ...] = ()
    result: ValidationResult | None = None


class ValidationMediaReference(FrozenModel):
    position: Literal["left", "right"]
    media_ref: str


class ValidationCampaignListItem(FrozenModel):
    campaign_id: str
    state: ValidationCampaignState
    next_action: str
    progress: ValidationProgress


class ValidationNextAction(StrEnum):
    PREREGISTER = "preregister"
    COMPLETE_GENERATION = "complete_generation"
    RECORD_CHOICE = "record_choice"
    ANALYZE = "analyze"
    NONE = "none"


class ValidationCampaignService:
    def __init__(
        self,
        campaigns: ValidationRepository,
        stage_1: Any,
        experiments: Any,
        reviews: Any,
        adapters: Any,
        *,
        reviewer_id: str,
        clock: Callable[[], Any] = utc_now,
    ) -> None:
        self.campaigns = campaigns
        self.stage_1 = stage_1
        self.experiments = experiments
        self.reviews = reviews
        self.adapters = adapters
        self.reviewer_id = reviewer_id
        self.clock = clock

    def create(
        self, *, stage_1_id: str, panel_input: ValidationPanelInput
    ) -> ValidationCampaignView:
        validated_input = ValidationPanelInput.model_validate(
            panel_input.model_dump(round_trip=True)
        )
        try:
            stage_view = self.stage_1.show(stage_1_id)
        except (LookupError, PermissionError, ValueError) as error:
            raise ValidationIneligible("Stage 1 evidence is not owned and terminal") from error
        if stage_view.state not in {
            AdaptiveStudyState.SEARCH_CONVERGED,
            AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED,
            AdaptiveStudyState.COMPLETED,
        }:
            raise ValidationIneligible("Stage 1 evidence is not terminal")
        evidence_digest = stage_view.evidence_digest
        if not isinstance(evidence_digest, str) or not evidence_digest.startswith("sha256:"):
            raise ValidationIneligible("Stage 1 evidence digest is unavailable")
        steps = self.stage_1.studies.list_optimizer_steps(stage_1_id)
        if not steps:
            raise ValidationIneligible("Stage 1 has no persisted terminal optimizer step")
        candidates = derive_validation_candidates(steps[-1])
        stage_definition = self.stage_1.studies.get(stage_1_id)
        campaign_id = validation_campaign_id(
            self.reviewer_id,
            evidence_digest.removeprefix("sha256:"),
            validation_panel_input_digest(validated_input),
        )
        try:
            created_at = self.campaigns.get(campaign_id).created_at
        except ValidationNotFound:
            created_at = self.clock()
        definition = ValidationCampaignDefinition(
            id=campaign_id,
            stage_1_study_id=stage_1_id,
            stage_1_evidence_digest=evidence_digest.removeprefix("sha256:"),
            base_manifest=stage_definition.base_manifest,
            candidates=candidates,
            panel_input=validated_input,
            created_at=created_at,
        )
        persisted = self.campaigns.create_or_get_bound(definition, reviewer_id=self.reviewer_id)
        return self._draft_view(persisted)

    def preview(self, campaign_id: str) -> ValidationCampaignPreview:
        definition = self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        if self.campaigns.state(campaign_id) is not ValidationCampaignState.DRAFT:
            raise PermissionError("validation candidates are sealed after preregistration")
        return ValidationCampaignPreview(candidates=definition.candidates)

    def preregister(self, campaign_id: str) -> ValidationCampaignView:
        definition = self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        state = self.campaigns.state(campaign_id)
        if state is ValidationCampaignState.DRAFT:
            self.campaigns.preregister_or_verify(campaign_id, build_validation_panel(definition))
            state = ValidationCampaignState.PREREGISTERED
        if state is ValidationCampaignState.PREREGISTERED:
            self.campaigns.append_event_or_verify(
                ValidationEvent(
                    campaign_id=campaign_id,
                    sequence=2,
                    previous_state=ValidationCampaignState.PREREGISTERED,
                    new_state=ValidationCampaignState.GENERATING,
                    created_at=self.clock(),
                )
            )
            state = ValidationCampaignState.GENERATING
        if state is not ValidationCampaignState.GENERATING:
            return self.show(campaign_id)
        if self.experiments is None or self.adapters is None:
            raise RuntimeError("validation generation dependencies are unavailable")
        capabilities = self.adapters.capabilities(definition.base_manifest.adapter_id)
        panel = self.campaigns.get_panel(campaign_id)
        for trial in sorted(
            (item for item in panel.trials if not item.hidden_repeat),
            key=lambda item: item.ordinal,
        ):
            baseline = definition.base_manifest.model_copy(
                update={
                    "prompt": trial.prompt,
                    "negative_prompt": trial.negative_prompt or "",
                    "seed": trial.seed,
                    "guidance_scale": definition.candidates.winner,
                }
            )
            self.experiments.create_validation(
                ExperimentDraft(
                    baseline=baseline,
                    variable_path="guidance_scale",
                    alternate_value=trial.challenger,
                ),
                capabilities,
                campaign_id=campaign_id,
                trial_id=trial.id,
            )
        return self.show(campaign_id)

    def show(self, campaign_id: str) -> ValidationCampaignView:
        definition = self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        state = self.campaigns.state(campaign_id)
        if state is ValidationCampaignState.DRAFT:
            return self._draft_view(definition)
        generated = self._generated_primary_count(definition)
        completed = len(self.campaigns.list_judgments(campaign_id))
        next_action = (
            ValidationNextAction.COMPLETE_GENERATION
            if state
            in {
                ValidationCampaignState.PREREGISTERED,
                ValidationCampaignState.GENERATING,
            }
            else ValidationNextAction.RECORD_CHOICE
            if state in {ValidationCampaignState.READY, ValidationCampaignState.JUDGING}
            and completed < 40
            else ValidationNextAction.ANALYZE
            if state in {ValidationCampaignState.JUDGING, ValidationCampaignState.ANALYZING}
            and completed == 40
            else ValidationNextAction.NONE
        )
        return ValidationCampaignView(
            campaign_id=campaign_id,
            state=state,
            next_action=next_action,
            progress=ValidationProgress(
                completed_judgments=completed,
                generated_primary=generated,
            ),
            preview=None,
            result=self.campaigns.get_result(campaign_id),
        )

    def list(self, *, limit: int = 50) -> tuple[ValidationCampaignListItem, ...]:
        return tuple(
            ValidationCampaignListItem(
                campaign_id=view.campaign_id,
                state=view.state,
                next_action=view.next_action,
                progress=view.progress,
            )
            for definition in self.campaigns.list_bound(reviewer_id=self.reviewer_id, limit=limit)
            for view in (self.show(definition.id),)
        )

    def resume(self, campaign_id: str) -> ValidationCampaignView:
        definition = self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        state = self.campaigns.state(campaign_id)
        if state is ValidationCampaignState.DRAFT:
            return self._draft_view(definition)
        if state in {
            ValidationCampaignState.PREREGISTERED,
            ValidationCampaignState.GENERATING,
        }:
            self.preregister(campaign_id)
            panel = self.campaigns.get_panel(campaign_id)
            all_jobs = tuple(
                job
                for trial in panel.trials
                if not trial.hidden_repeat
                for job in self.experiments.get_validation(
                    validation_experiment_id(campaign_id, trial.id)
                ).jobs
            )
            if len(all_jobs) != 72 or any(job.state is not JobState.COMPLETED for job in all_jobs):
                return self.show(campaign_id)
            self.campaigns.append_event_or_verify(
                ValidationEvent(
                    campaign_id=campaign_id,
                    sequence=3,
                    previous_state=ValidationCampaignState.GENERATING,
                    new_state=ValidationCampaignState.READY,
                    created_at=self.clock(),
                )
            )
            state = ValidationCampaignState.READY
        if state is ValidationCampaignState.READY:
            self.campaigns.append_event_or_verify(
                ValidationEvent(
                    campaign_id=campaign_id,
                    sequence=4,
                    previous_state=ValidationCampaignState.READY,
                    new_state=ValidationCampaignState.JUDGING,
                    created_at=self.clock(),
                )
            )
            state = ValidationCampaignState.JUDGING
        if state is ValidationCampaignState.JUDGING:
            judgments = self.campaigns.list_judgments(campaign_id)
            if len(judgments) == 40:
                return self.analyze(campaign_id)
            trial = self._next_trial(campaign_id, judgments)
            if self.reviews is None:
                raise RuntimeError("validation review dependency is unavailable")
            review = self.reviews.start_validation(
                campaign_id=campaign_id,
                trial_id=trial.id,
                reviewer_id=self.reviewer_id,
            )
            view = self.show(campaign_id)
            return view.model_copy(
                update={
                    "review_id": review.session_id,
                    "media": tuple(
                        ValidationMediaReference(
                            position=item.display_position,
                            media_ref=item.url,
                        )
                        for item in review.media
                    ),
                }
            )
        if state is ValidationCampaignState.ANALYZING:
            return self.analyze(campaign_id)
        return self.show(campaign_id)

    def record_choice(
        self,
        campaign_id: str,
        choice: ValidationChoice | str,
        *,
        review_id: str,
    ) -> ValidationCampaignView:
        self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        if self.reviews is None:
            raise RuntimeError("validation review dependency is unavailable")
        try:
            requested_trial_id = self.reviews.validation_trial_for_session(
                campaign_id=campaign_id,
                session_id=review_id,
                reviewer_id=self.reviewer_id,
            )
        except LookupError as error:
            raise ValidationConflict("validation review does not belong to campaign") from error
        validated_choice = ValidationChoice(choice)
        judgments = self.campaigns.list_judgments(campaign_id)
        existing = next((item for item in judgments if item.trial_id == requested_trial_id), None)
        state = self.campaigns.state(campaign_id)
        if existing is not None:
            if existing.choice is not validated_choice:
                raise ValidationConflict(
                    "validation judgment retry does not match persisted choice"
                )
            if state not in {
                ValidationCampaignState.JUDGING,
                ValidationCampaignState.ANALYZING,
                ValidationCampaignState.QUALIFIED,
                ValidationCampaignState.NOT_QUALIFIED,
                ValidationCampaignState.INCONCLUSIVE,
            }:
                raise ValidationConflict("validation campaign is not accepting judgments")
            trial = next(
                item
                for item in self.campaigns.get_panel(campaign_id).trials
                if item.id == requested_trial_id
            )
        else:
            if state is not ValidationCampaignState.JUDGING:
                raise ValidationConflict("validation campaign is not accepting judgments")
            if len(judgments) >= 40:
                raise ValidationConflict("validation review has no persisted judgment")
            trial = self._next_trial(campaign_id, judgments)
            if requested_trial_id != trial.id:
                raise ValidationConflict("validation review is not the next scheduled trial")
        preference = {
            ValidationChoice.A: DisplayPreference.LEFT,
            ValidationChoice.B: DisplayPreference.RIGHT,
            ValidationChoice.TIE: DisplayPreference.TIE,
        }[validated_choice]
        try:
            revision = self.reviews.submit_validation(
                campaign_id=campaign_id,
                trial_id=trial.id,
                reviewer_id=self.reviewer_id,
                preference=preference,
            )
        except (LookupError, ValueError) as error:
            raise ValidationConflict("validation review retry does not match evidence") from error
        if trial.hidden_repeat:
            if trial.reuses_trial_id is None:
                raise ValidationConflict("validation repeat has no source trial")
            experiment_id = validation_experiment_id(campaign_id, trial.reuses_trial_id)
            bound_review_id = revision.id
        else:
            experiment_id = validation_experiment_id(campaign_id, trial.id)
            bound_review_id = revision.id
        if experiment_id is not None and bound_review_id is not None:
            self.campaigns.bind_trial_or_verify(
                ValidationTrialBinding(
                    campaign_id=campaign_id,
                    trial_id=trial.id,
                    experiment_id=experiment_id,
                    review_id=bound_review_id,
                    reused_from_trial_id=trial.reuses_trial_id,
                )
            )
        self.campaigns.consume_judgment_or_verify(
            ValidationJudgment(
                campaign_id=campaign_id,
                trial_id=trial.id,
                ordinal=trial.ordinal,
                archetype=trial.archetype,
                challenger=trial.challenger,
                seed=trial.seed,
                hidden_repeat=trial.hidden_repeat,
                winner_display_position=trial.winner_display_position,
                choice=validated_choice,
                created_at=self.clock(),
            )
        )
        return self.resume(campaign_id)

    def analyze(self, campaign_id: str) -> ValidationCampaignView:
        self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        state = self.campaigns.state(campaign_id)
        if state in {
            ValidationCampaignState.QUALIFIED,
            ValidationCampaignState.NOT_QUALIFIED,
            ValidationCampaignState.INCONCLUSIVE,
        }:
            return self.show(campaign_id)
        judgments = self.campaigns.list_judgments(campaign_id)
        if len(judgments) != 40:
            raise ValidationIneligible("validation analysis requires exactly 40 judgments")
        if state is ValidationCampaignState.JUDGING:
            self.campaigns.append_event_or_verify(
                ValidationEvent(
                    campaign_id=campaign_id,
                    sequence=5,
                    previous_state=ValidationCampaignState.JUDGING,
                    new_state=ValidationCampaignState.ANALYZING,
                    created_at=self.clock(),
                )
            )
        elif state is not ValidationCampaignState.ANALYZING:
            raise ValidationConflict("validation campaign cannot enter analysis")
        result = self.campaigns.get_result(campaign_id)
        if result is None:
            result = self.campaigns.append_result_or_verify(
                analyze_validation(
                    judgments,
                    repeat_agreements=self._repeat_agreements(campaign_id, judgments),
                )
            )
        terminal = {
            ValidationDisposition.QUALIFIED: ValidationCampaignState.QUALIFIED,
            ValidationDisposition.NOT_QUALIFIED: ValidationCampaignState.NOT_QUALIFIED,
            ValidationDisposition.INCONCLUSIVE: ValidationCampaignState.INCONCLUSIVE,
        }[result.disposition]
        self.campaigns.append_event_or_verify(
            ValidationEvent(
                campaign_id=campaign_id,
                sequence=6,
                previous_state=ValidationCampaignState.ANALYZING,
                new_state=terminal,
                created_at=self.clock(),
            )
        )
        return self.show(campaign_id)

    def export_evidence(self, campaign_id: str) -> dict[str, object]:
        definition = self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        state = self.campaigns.state(campaign_id)
        if state not in {
            ValidationCampaignState.QUALIFIED,
            ValidationCampaignState.NOT_QUALIFIED,
            ValidationCampaignState.INCONCLUSIVE,
        }:
            raise PermissionError("validation evidence remains sealed before terminal analysis")
        panel = self.campaigns.get_panel(campaign_id)
        result = self.campaigns.get_result(campaign_id)
        if result is None:
            raise ValidationConflict("terminal validation campaign has no result")
        provenance = self._evidence_provenance(campaign_id)
        return {
            "schema_version": "balanced-validation-evidence.v2",
            "campaign": definition.model_dump(mode="json"),
            "panel": panel.model_dump(mode="json"),
            "events": [
                event.model_dump(mode="json") for event in self.campaigns.list_events(campaign_id)
            ],
            "judgments": [
                judgment.model_dump(mode="json")
                for judgment in self.campaigns.list_judgments(campaign_id)
            ],
            **provenance,
            "provenance_digest": "sha256:"
            + hashlib.sha256(canonical_json(provenance).encode()).hexdigest(),
            "result": result.model_dump(mode="json"),
        }

    def _evidence_provenance(self, campaign_id: str) -> dict[str, object]:
        self.campaigns.get_bound(campaign_id, reviewer_id=self.reviewer_id)
        panel = self.campaigns.get_panel(campaign_id)
        judgments = self.campaigns.list_judgments(campaign_id)
        with self.campaigns.database.connect() as connection:
            experiments = [
                dict(row)
                for row in connection.execute(
                    "SELECT binding.campaign_id, binding.trial_id, binding.experiment_id "
                    "FROM validation_experiment_bindings AS binding "
                    "JOIN validation_trials AS trial ON trial.campaign_id=binding.campaign_id "
                    "AND trial.trial_id=binding.trial_id WHERE binding.campaign_id=? "
                    "ORDER BY trial.ordinal",
                    (campaign_id,),
                )
            ]
            rows = connection.execute(
                "SELECT binding.campaign_id, binding.trial_id, binding.experiment_id, "
                "binding.review_id, binding.reused_from_trial_id, trial.ordinal, "
                "trial.winner_display_position, judgment.choice, review.session_id, "
                "review.revision, review.display_preference, review.preferred_variant_id, "
                "review.supersedes_review_id, session.reviewer_id, "
                "session.experiment_id AS session_experiment_id, "
                "session.left_variant_id, session.left_job_id, session.left_artifact_id, "
                "session.left_artifact_sha256, session.right_variant_id, session.right_job_id, "
                "session.right_artifact_id, session.right_artifact_sha256, "
                "baseline.id AS baseline_id, left_artifact.sha256 AS left_persisted_sha256, "
                "right_artifact.sha256 AS right_persisted_sha256, "
                "left_variant.experiment_id AS left_variant_experiment_id, "
                "right_variant.experiment_id AS right_variant_experiment_id, "
                "left_job.state AS left_job_state, right_job.state AS right_job_state "
                "FROM validation_trial_bindings AS binding "
                "JOIN validation_trials AS trial ON trial.campaign_id=binding.campaign_id "
                "AND trial.trial_id=binding.trial_id "
                "JOIN validation_judgments AS judgment ON judgment.campaign_id=binding.campaign_id "
                "AND judgment.trial_id=binding.trial_id "
                "JOIN review_revisions AS review ON review.id=binding.review_id "
                "JOIN review_sessions AS session ON session.id=review.session_id "
                "JOIN variants AS left_variant ON left_variant.id=session.left_variant_id "
                "JOIN variants AS right_variant ON right_variant.id=session.right_variant_id "
                "JOIN variants AS baseline ON baseline.experiment_id=binding.experiment_id "
                "AND baseline.label='A' "
                "JOIN jobs AS left_job ON left_job.id=session.left_job_id "
                "AND left_job.variant_id=session.left_variant_id "
                "JOIN jobs AS right_job ON right_job.id=session.right_job_id "
                "AND right_job.variant_id=session.right_variant_id "
                "JOIN artifacts AS left_artifact ON left_artifact.id=session.left_artifact_id "
                "JOIN artifacts AS right_artifact ON right_artifact.id=session.right_artifact_id "
                "JOIN job_artifacts AS left_link ON left_link.job_id=session.left_job_id "
                "AND left_link.artifact_id=session.left_artifact_id "
                "JOIN job_artifacts AS right_link ON right_link.job_id=session.right_job_id "
                "AND right_link.artifact_id=session.right_artifact_id "
                "WHERE binding.campaign_id=? ORDER BY trial.ordinal",
                (campaign_id,),
            ).fetchall()
            campaign_sessions = connection.execute(
                "SELECT DISTINCT session.id FROM review_sessions AS session "
                "JOIN validation_trial_bindings AS binding "
                "ON binding.experiment_id=session.experiment_id "
                "JOIN validation_trials AS trial ON trial.campaign_id=binding.campaign_id "
                "AND trial.trial_id=binding.trial_id "
                "WHERE binding.campaign_id=? AND session.reviewer_id="
                "validation_session_reviewer(binding.campaign_id, binding.trial_id, ?, "
                "trial.hidden_repeat) ORDER BY session.id",
                (campaign_id, self.reviewer_id),
            ).fetchall()
            campaign_revision_ids = connection.execute(
                "SELECT review.id FROM review_revisions AS review "
                "WHERE review.session_id IN ("
                "SELECT DISTINCT session.id FROM review_sessions AS session "
                "JOIN validation_trial_bindings AS binding "
                "ON binding.experiment_id=session.experiment_id "
                "JOIN validation_trials AS trial ON trial.campaign_id=binding.campaign_id "
                "AND trial.trial_id=binding.trial_id "
                "WHERE binding.campaign_id=? AND session.reviewer_id="
                "validation_session_reviewer(binding.campaign_id, binding.trial_id, ?, "
                "trial.hidden_repeat)) ORDER BY review.id",
                (campaign_id, self.reviewer_id),
            ).fetchall()
        trial_by_id = {trial.id: trial for trial in panel.trials}
        judgment_by_trial = {judgment.trial_id: judgment for judgment in judgments}
        primary_ids = {trial.id for trial in panel.trials if not trial.hidden_repeat}
        experiment_trial_ids = {str(item["trial_id"]) for item in experiments}
        row_trial_ids = {str(row["trial_id"]) for row in rows}
        bound_review_ids = {str(row["review_id"]) for row in rows}
        bound_session_ids = {str(row["session_id"]) for row in rows}
        if (
            len(panel.trials) != 40
            or len(primary_ids) != 36
            or len(experiments) != 36
            or experiment_trial_ids != primary_ids
            or any(
                str(item["experiment_id"])
                != validation_experiment_id(campaign_id, str(item["trial_id"]))
                for item in experiments
            )
            or len(judgments) != 40
            or set(judgment_by_trial) != set(trial_by_id)
            or len(rows) != 40
            or len(row_trial_ids) != 40
            or row_trial_ids != set(trial_by_id)
            or len(bound_review_ids) != 40
            or len(bound_session_ids) != 40
            or len(campaign_sessions) != 40
            or len(campaign_revision_ids) != 40
            or {str(row["id"]) for row in campaign_revision_ids} != bound_review_ids
        ):
            raise ValidationConflict("validation evidence provenance is incomplete")
        bindings: list[dict[str, object]] = []
        reviews: list[dict[str, object]] = []
        for row in rows:
            trial = trial_by_id[str(row["trial_id"])]
            source_trial_id = trial.reuses_trial_id or trial.id
            expected_experiment = validation_experiment_id(campaign_id, source_trial_id)
            expected_reviewer = _validation_session_reviewer_identity(
                campaign_id,
                trial.id,
                self.reviewer_id,
                hidden_repeat=trial.hidden_repeat,
            )
            if (
                str(row["experiment_id"]) != expected_experiment
                or str(row["session_experiment_id"]) != expected_experiment
                or str(row["left_variant_experiment_id"]) != expected_experiment
                or str(row["right_variant_experiment_id"]) != expected_experiment
                or row["reused_from_trial_id"] != trial.reuses_trial_id
                or str(row["reviewer_id"]) != expected_reviewer
                or int(row["revision"]) != 1
                or row["supersedes_review_id"] is not None
                or str(row["left_job_state"]) != "completed"
                or str(row["right_job_state"]) != "completed"
                or str(row["left_artifact_sha256"]) != str(row["left_persisted_sha256"])
                or str(row["right_artifact_sha256"]) != str(row["right_persisted_sha256"])
            ):
                raise ValidationConflict("validation evidence provenance is inconsistent")
            bindings.append(
                {
                    "campaign_id": str(row["campaign_id"]),
                    "trial_id": str(row["trial_id"]),
                    "experiment_id": str(row["experiment_id"]),
                    "review_id": str(row["review_id"]),
                    "reused_from_trial_id": row["reused_from_trial_id"],
                }
            )
            actual_winner = (
                "left" if str(row["left_variant_id"]) == str(row["baseline_id"]) else "right"
            )
            swap = actual_winner != str(row["winner_display_position"])
            display_mapping: dict[str, dict[str, str]] = {}
            for position in ("left", "right"):
                underlying = ("right" if position == "left" else "left") if swap else position
                display_mapping[position] = {
                    "variant_id": str(row[f"{underlying}_variant_id"]),
                    "job_id": str(row[f"{underlying}_job_id"]),
                    "artifact_id": str(row[f"{underlying}_artifact_id"]),
                    "artifact_sha256": str(row[f"{underlying}_artifact_sha256"]),
                }
            reviews.append(
                {
                    "trial_id": str(row["trial_id"]),
                    "review_id": str(row["review_id"]),
                    "session_id": str(row["session_id"]),
                    "revision": int(row["revision"]),
                    "choice": str(row["choice"]),
                    "persisted_display_preference": str(row["display_preference"]),
                    "preferred_variant_id": row["preferred_variant_id"],
                    "display_mapping": display_mapping,
                }
            )
            public_preference = {"a": "left", "b": "right", "tie": "tie"}[str(row["choice"])]
            preferred_variant = row["preferred_variant_id"]
            if public_preference == "tie":
                preference_valid = preferred_variant is None
            else:
                preference_valid = (
                    str(preferred_variant) == display_mapping[public_preference]["variant_id"]
                )
            if not preference_valid:
                raise ValidationConflict("validation evidence provenance choice is inconsistent")
            persisted_preference = str(row["display_preference"])
            if persisted_preference == "tie":
                persisted_preference_valid = preferred_variant is None
            elif persisted_preference in {"left", "right"}:
                persisted_preference_valid = str(preferred_variant) == str(
                    row[f"{persisted_preference}_variant_id"]
                )
            else:
                persisted_preference_valid = False
            if not persisted_preference_valid:
                raise ValidationConflict("validation review revision provenance is inconsistent")
        binding_by_trial = {str(item["trial_id"]): item for item in bindings}
        review_by_id = {str(item["review_id"]): item for item in reviews}
        for trial in panel.trials:
            if not trial.hidden_repeat or trial.reuses_trial_id is None:
                continue
            repeat_binding = binding_by_trial[trial.id]
            source_binding = binding_by_trial[trial.reuses_trial_id]
            repeat_review = review_by_id[str(repeat_binding["review_id"])]
            source_review = review_by_id[str(source_binding["review_id"])]
            repeat_mapping = cast(dict[str, dict[str, str]], repeat_review["display_mapping"])
            source_mapping = cast(dict[str, dict[str, str]], source_review["display_mapping"])
            repeat_artifacts = {
                (item["artifact_id"], item["artifact_sha256"]) for item in repeat_mapping.values()
            }
            source_artifacts = {
                (item["artifact_id"], item["artifact_sha256"]) for item in source_mapping.values()
            }
            if (
                repeat_binding["experiment_id"] != source_binding["experiment_id"]
                or repeat_binding["review_id"] == source_binding["review_id"]
                or repeat_review["session_id"] == source_review["session_id"]
                or repeat_artifacts != source_artifacts
            ):
                raise ValidationConflict("validation repeat provenance is inconsistent")
        if self.reviews is None:
            raise ValidationConflict("validation evidence provenance resolver is unavailable")
        artifact_ids = {
            item["artifact_id"]
            for review in reviews
            for item in cast(dict[str, dict[str, str]], review["display_mapping"]).values()
        }
        try:
            for artifact_id in sorted(artifact_ids):
                self.reviews.artifacts.resolve(artifact_id)
        except (ArtifactError, OSError) as error:
            raise ValidationConflict(
                "validation evidence provenance media is not canonical"
            ) from error
        return {
            "experiment_bindings": experiments,
            "trial_bindings": bindings,
            "reviews": reviews,
        }

    def _generated_primary_count(self, definition: ValidationCampaignDefinition) -> int:
        if self.experiments is None:
            return 0
        try:
            panel = self.campaigns.get_panel(definition.id)
        except Exception:
            return 0
        count = 0
        for trial in panel.trials:
            if trial.hidden_repeat:
                continue
            try:
                self.experiments.experiments.get_validation(
                    validation_experiment_id(definition.id, trial.id)
                )
            except RecordNotFound:
                continue
            count += 1
        return count

    def _next_trial(
        self, campaign_id: str, judgments: tuple[ValidationJudgment, ...]
    ) -> ValidationTrial:
        consumed = {item.trial_id for item in judgments}
        return next(
            trial
            for trial in self.campaigns.get_panel(campaign_id).trials
            if trial.id not in consumed
        )

    def _repeat_agreements(
        self, campaign_id: str, judgments: tuple[ValidationJudgment, ...]
    ) -> int:
        by_trial = {item.trial_id: item for item in judgments}
        panel = self.campaigns.get_panel(campaign_id)
        agreements = 0
        for trial in panel.trials:
            if not trial.hidden_repeat or trial.reuses_trial_id is None:
                continue
            if by_trial[trial.id].outcome is by_trial[trial.reuses_trial_id].outcome:
                agreements += 1
        return agreements

    @staticmethod
    def _draft_view(definition: ValidationCampaignDefinition) -> ValidationCampaignView:
        return ValidationCampaignView(
            campaign_id=definition.id,
            state=ValidationCampaignState.DRAFT,
            next_action=ValidationNextAction.PREREGISTER,
            progress=ValidationProgress(completed_judgments=0, generated_primary=0),
            preview=ValidationCampaignPreview(candidates=definition.candidates),
        )


def validation_panel_input_digest(panel_input: ValidationPanelInput) -> str:
    payload = {
        "schema_version": "validation-panel-input.v1",
        "panel_input": panel_input.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _validation_session_reviewer_identity(
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


def validation_campaign_id(
    reviewer_id: str, stage_1_evidence_digest: str, panel_input_digest: str
) -> str:
    normalized = reviewer_id.strip()
    if not normalized or normalized != reviewer_id:
        raise ValidationIneligible("validation reviewer id must be normalized")
    payload = {
        "schema_version": "validation-campaign-id.v1",
        "domain": "video-lab.validation.campaign",
        "reviewer_id": normalized,
        "stage_1_evidence_digest": stage_1_evidence_digest,
        "panel_input_digest": panel_input_digest,
    }
    return "validation_" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
