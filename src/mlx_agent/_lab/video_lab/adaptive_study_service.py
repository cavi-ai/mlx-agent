from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Literal

from video_lab.adaptive_study_models import (
    AdaptiveObservationRequest,
    AdaptiveOptimizerStep,
    AdaptiveStudyDefinition,
    AdaptiveStudyEvent,
    AdaptiveStudyState,
    JsonValue,
    adaptive_optimizer_input_digest,
    adaptive_optimizer_output_digest,
)
from video_lab.adaptive_study_repository import (
    AdaptiveStudyConflict,
    AdaptiveStudyRepository,
)
from video_lab.capabilities import AdapterCapabilities
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    FrozenModel,
    canonical_json,
    utc_now,
)
from video_lab.preference_optimizer import (
    OptimizerConfig,
    OptimizerDecision,
    PreferenceObservation,
    StopReason,
    optimize_preference,
)
from video_lab.repositories import adaptive_experiment_id
from video_lab.review_service import ReviewService
from video_lab.services import ExperimentService

AdaptiveNextAction = Literal["complete_generation", "record_preference", "resume", "none"]


class AdaptiveStudyProgress(FrozenModel):
    completed_comparisons: int
    current_comparison: int | None
    maximum_comparisons: int


class AdaptiveMediaReference(FrozenModel):
    choice: Literal["a", "b"]
    media_ref: str


class AdaptiveStudyView(FrozenModel):
    schema_version: Literal["adaptive-study-view.v1"] = "adaptive-study-view.v1"
    study_id: str
    prompt: str
    state: AdaptiveStudyState
    next_action: AdaptiveNextAction
    comparison_count: int
    maximum_comparisons: int
    progress: AdaptiveStudyProgress
    pair_ref: str | None = None
    review_id: str | None = None
    media: tuple[AdaptiveMediaReference, ...] = ()
    stop_reason: StopReason | None = None
    disposition: str | None = None
    recommendation: float | None = None
    concentration_interval: tuple[float, float] | None = None
    concentration_probability: float | None = None
    evidence_digest: str | None = None
    limitations: tuple[str, ...] = (
        "Stage 1 evidence is limited to one fixed prompt and seed.",
        "Only one controlled parameter is varied; this is not a complete recipe optimization.",
    )


class AdaptiveStudyListItem(FrozenModel):
    schema_version: Literal["adaptive-study-list-item.v1"] = "adaptive-study-list-item.v1"
    study_id: str
    state: AdaptiveStudyState
    next_action: AdaptiveNextAction
    progress: AdaptiveStudyProgress


class AdaptiveStudyService:
    def __init__(
        self,
        studies: AdaptiveStudyRepository,
        experiments: ExperimentService,
        reviews: ReviewService,
        capabilities: AdapterCapabilities,
        *,
        reviewer_id: str,
        optimizer: Callable[
            [OptimizerConfig, tuple[PreferenceObservation, ...]], OptimizerDecision
        ] = optimize_preference,
    ) -> None:
        if not reviewer_id:
            raise ValueError("adaptive reviewer id must not be empty")
        self.studies = studies
        self.experiments = experiments
        self.reviews = reviews
        self.capabilities = capabilities
        self.reviewer_id = reviewer_id
        self.optimizer = optimizer
        self._verified_optimizer_outputs: set[tuple[str, int, str]] = set()

    def create_study(self, definition: AdaptiveStudyDefinition) -> AdaptiveStudyView:
        validated = AdaptiveStudyDefinition.model_validate(definition.model_dump(round_trip=True))
        self._validate_adapter(validated)
        self.studies.create_or_get_bound(validated, reviewer_id=self.reviewer_id)
        return self.resume(validated.id)

    def show(self, study_id: str) -> AdaptiveStudyView:
        definition = self.studies.get(study_id)
        events = self.studies.list_events(study_id)
        self._validate_reviewer_binding(study_id, events)
        observations = self.studies.list_observations(study_id)
        steps = self.studies.list_optimizer_steps(study_id)
        state = AdaptiveStudyState.DRAFT if not events else events[-1].new_state
        terminal = state in {
            AdaptiveStudyState.SEARCH_CONVERGED,
            AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED,
            AdaptiveStudyState.COMPLETED,
        }
        nonrecoverable_failure = (
            state is AdaptiveStudyState.OPTIMIZER_FAILED
            and events[-1].payload.get("recoverable") is False
        )
        latest_step = steps[-1] if steps else None
        comparison_count = len(observations)
        maximum_comparisons = definition.optimizer_config.maximum_comparisons
        optimizer_output_digest = next(
            (
                value
                for event in reversed(events)
                if isinstance((value := event.payload.get("optimizer_output_digest")), str)
            ),
            None,
        )
        pair_ref = (
            self._pair_ref(study_id, optimizer_output_digest)
            if optimizer_output_digest is not None
            else None
        )
        review_id = (
            self._payload_string(events[-1], "review_session_id")
            if state is AdaptiveStudyState.AWAITING_PREFERENCE
            else None
        )
        return AdaptiveStudyView(
            study_id=study_id,
            prompt=definition.prompt,
            state=state,
            next_action=(
                "complete_generation"
                if state is AdaptiveStudyState.GENERATING_PAIR
                else "record_preference"
                if state is AdaptiveStudyState.AWAITING_PREFERENCE
                else "none"
                if terminal or nonrecoverable_failure
                else "resume"
            ),
            comparison_count=comparison_count,
            maximum_comparisons=maximum_comparisons,
            progress=AdaptiveStudyProgress(
                completed_comparisons=comparison_count,
                current_comparison=(
                    None if terminal else min(comparison_count + 1, maximum_comparisons)
                ),
                maximum_comparisons=maximum_comparisons,
            ),
            pair_ref=pair_ref,
            review_id=review_id,
            media=(
                (
                    AdaptiveMediaReference(choice="a", media_ref=f"adaptive-review:{review_id}:a"),
                    AdaptiveMediaReference(choice="b", media_ref=f"adaptive-review:{review_id}:b"),
                )
                if review_id is not None
                else ()
            ),
            stop_reason=(
                latest_step.stop_reason
                if latest_step and latest_step.stop_reason is not StopReason.CONTINUE
                else None
            ),
            disposition=(
                latest_step.disposition.value
                if latest_step and latest_step.disposition is not None
                else None
            ),
            recommendation=(
                latest_step.posterior.recommendation if terminal and latest_step else None
            ),
            concentration_interval=(
                latest_step.posterior.concentration_interval if terminal and latest_step else None
            ),
            concentration_probability=(
                latest_step.posterior.concentration_probability
                if terminal and latest_step
                else None
            ),
            evidence_digest=(
                "sha256:"
                + hashlib.sha256(
                    canonical_json(
                        {
                            "schema_version": "adaptive-study-evidence-chain.v1",
                            "input_fingerprint": definition.input_fingerprint,
                            "events": [item.model_dump(mode="json") for item in events],
                            "observations": [item.model_dump(mode="json") for item in observations],
                            "optimizer_steps": [item.model_dump(mode="json") for item in steps],
                        }
                    ).encode("utf-8")
                ).hexdigest()
                if terminal and latest_step
                else None
            ),
        )

    def list(self, *, limit: int = 50) -> tuple[AdaptiveStudyListItem, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("adaptive study list limit must be between 1 and 100")
        items: list[AdaptiveStudyListItem] = []
        for definition in self.studies.list():
            try:
                owner = self.studies.owner(definition.id)
            except AdaptiveStudyConflict:
                continue
            if owner != self.reviewer_id:
                continue
            view = self.show(definition.id)
            items.append(
                AdaptiveStudyListItem(
                    study_id=view.study_id,
                    state=view.state,
                    next_action=view.next_action,
                    progress=view.progress,
                )
            )
            if len(items) == limit:
                break
        return tuple(items)

    def resume(self, study_id: str) -> AdaptiveStudyView:
        definition = self.studies.get(study_id)
        events = self.studies.list_events(study_id)
        self._validate_reviewer_binding(study_id, events)
        if not events:
            raise PermissionError("adaptive study has no immutable owner binding")
        if events[-1].new_state is AdaptiveStudyState.SEARCH_READY:
            observations = self.studies.list_observations(study_id)
            steps = self.studies.list_optimizer_steps(study_id)
            if observations:
                raise RuntimeError("search-ready study unexpectedly has observations")
            if not steps:
                try:
                    decision = self.optimizer(definition.optimizer_config, ())
                except Exception as error:
                    self._append_event(
                        study_id,
                        previous=AdaptiveStudyState.SEARCH_READY,
                        target=AdaptiveStudyState.OPTIMIZER_FAILED,
                        payload={
                            "comparison_count": 0,
                            "error_type": type(error).__name__,
                        },
                    )
                    return self.show(study_id)
                step = self._optimizer_step(definition, decision, created_at=definition.created_at)
                persisted_step = False
            elif len(steps) == 1:
                step = steps[0]
                self._verify_optimizer_step(definition, step)
                persisted_step = True
            else:
                raise RuntimeError("search-ready study has duplicate optimizer steps")
            if step.stop_reason is not StopReason.CONTINUE or step.selected_pair is None:
                if not persisted_step:
                    step = self.studies.append_optimizer_step(step)
                    self._remember_optimizer_step(step)
                self._append_event(
                    study_id,
                    previous=AdaptiveStudyState.SEARCH_READY,
                    target=AdaptiveStudyState.OPTIMIZER_FAILED,
                    payload={
                        "comparison_count": 0,
                        "stop_reason": step.stop_reason.value,
                        "recoverable": False,
                    },
                )
                return self.show(study_id)
            if step.selected_pair != definition.optimizer_config.initial_pair:
                if not persisted_step:
                    step = self.studies.append_optimizer_step(step)
                    self._remember_optimizer_step(step)
                self._append_event(
                    study_id,
                    previous=AdaptiveStudyState.SEARCH_READY,
                    target=AdaptiveStudyState.OPTIMIZER_FAILED,
                    payload={
                        "comparison_count": 0,
                        "error_code": "initial_pair_mismatch",
                        "recoverable": False,
                    },
                )
                return self.show(study_id)
            experiment_id = adaptive_experiment_id(study_id, step.output_digest)
            planned = self.studies.ensure_initial_plan(
                step,
                experiment_id=experiment_id,
            )
            self._remember_optimizer_step(step)
            return self._resume_generating(definition, planned, verify_optimizer=False)
        if events[-1].new_state is AdaptiveStudyState.PAUSED:
            paused = events[-1]
            self.experiments.requeue_adaptive_failures(
                study_id, self._payload_string(paused, "experiment_id")
            )
            self._append_event(
                study_id,
                previous=AdaptiveStudyState.PAUSED,
                target=AdaptiveStudyState.GENERATING_PAIR,
                payload={
                    "optimizer_step_ordinal": self._payload_int(paused, "optimizer_step_ordinal"),
                    "optimizer_output_digest": self._payload_string(
                        paused, "optimizer_output_digest"
                    ),
                    "experiment_id": self._payload_string(paused, "experiment_id"),
                },
            )
            return self.resume(study_id)
        if events[-1].new_state is AdaptiveStudyState.OPTIMIZER_FAILED:
            if events[-1].payload.get("recoverable") is False:
                return self.show(study_id)
            observations = self.studies.list_observations(study_id)
            target = (
                AdaptiveStudyState.UPDATING_POSTERIOR
                if observations
                else AdaptiveStudyState.SEARCH_READY
            )
            self._append_event(
                study_id,
                previous=AdaptiveStudyState.OPTIMIZER_FAILED,
                target=target,
                payload={
                    "recovery": "explicit_resume",
                    "comparison_count": len(observations),
                },
            )
            return self.resume(study_id)
        if events[-1].new_state is AdaptiveStudyState.UPDATING_POSTERIOR:
            return self._resume_optimizer_update(definition)
        if events[-1].new_state is not AdaptiveStudyState.GENERATING_PAIR:
            return self.show(study_id)
        return self._resume_generating(definition, events[-1], verify_optimizer=True)

    def _resume_generating(
        self,
        definition: AdaptiveStudyDefinition,
        event: AdaptiveStudyEvent,
        *,
        verify_optimizer: bool,
    ) -> AdaptiveStudyView:
        study_id = definition.id
        step_ordinal = self._payload_int(event, "optimizer_step_ordinal")
        output_digest = self._payload_string(event, "optimizer_output_digest")
        expected_experiment_id = self._payload_string(event, "experiment_id")
        steps = self.studies.list_optimizer_steps(study_id)
        if not 1 <= step_ordinal <= len(steps):
            raise RuntimeError("generating event references a missing optimizer step")
        step = steps[step_ordinal - 1]
        if step.output_digest != output_digest or step.selected_pair is None:
            raise RuntimeError("generating event does not match its optimizer step")
        if verify_optimizer:
            self._verify_optimizer_step(definition, step)
        generated = self.experiments.create_adaptive(
            self._pair_draft(definition, step.selected_pair.value_a, step.selected_pair.value_b),
            self.capabilities,
            study_id=study_id,
            optimizer_output_digest=output_digest,
        )
        if generated.experiment.id != expected_experiment_id:
            raise RuntimeError("adaptive experiment identity is not reproducible")
        try:
            review = self.reviews.start_adaptive(expected_experiment_id, self.reviewer_id)
        except ValueError:
            return self.show(study_id)
        sealed_mapping_digest = self.reviews.adaptive_binding_digest(
            review.session_id,
            self.reviewer_id,
            study_id=study_id,
            optimizer_step_ordinal=step.ordinal,
            optimizer_output_digest=step.output_digest,
        )
        self.studies.append_awaiting_preference_or_verify(
            AdaptiveStudyEvent(
                study_id=study_id,
                sequence=event.sequence + 1,
                previous_state=AdaptiveStudyState.GENERATING_PAIR,
                new_state=AdaptiveStudyState.AWAITING_PREFERENCE,
                payload={
                    "binding_schema_version": "adaptive-review-binding.v1",
                    "study_id": study_id,
                    "optimizer_step_ordinal": step.ordinal,
                    "optimizer_output_digest": step.output_digest,
                    "experiment_id": generated.experiment.id,
                    "review_session_id": review.session_id,
                    "reviewer_id": self.reviewer_id,
                    "sealed_mapping_digest": sealed_mapping_digest,
                },
                created_at=utc_now(),
            )
        )
        return self.show(study_id)

    def _validate_adapter(self, definition: AdaptiveStudyDefinition) -> None:
        initial = definition.optimizer_config.initial_pair
        if (initial.value_a, initial.value_b) != (2.0, 4.0):
            raise ValueError("adaptive Stage 1 must start with 2.0 versus 4.0")
        self.experiments.preview(
            self._pair_draft(definition, initial.value_a, initial.value_b),
            self.capabilities,
        )

    def pause_generation_failure(
        self,
        study_id: str,
        experiment_id: str,
        *,
        error_code: str,
    ) -> AdaptiveStudyView:
        if not error_code:
            raise ValueError("generation failure error code must not be empty")
        events = self.studies.list_events(study_id)
        if not events:
            raise ValueError("adaptive study has no generation to pause")
        latest = events[-1]
        if latest.new_state is AdaptiveStudyState.PAUSED:
            if (
                self._payload_string(latest, "experiment_id") == experiment_id
                and self._payload_string(latest, "error_code") == error_code
            ):
                return self.show(study_id)
            raise ValueError("adaptive study is already paused for another failure")
        if latest.new_state is not AdaptiveStudyState.GENERATING_PAIR:
            raise ValueError("adaptive study is not generating a pair")
        if self._payload_string(latest, "experiment_id") != experiment_id:
            raise ValueError("generation failure does not match the active experiment")
        self._append_event(
            study_id,
            previous=AdaptiveStudyState.GENERATING_PAIR,
            target=AdaptiveStudyState.PAUSED,
            payload={
                "optimizer_step_ordinal": self._payload_int(latest, "optimizer_step_ordinal"),
                "optimizer_output_digest": self._payload_string(latest, "optimizer_output_digest"),
                "experiment_id": experiment_id,
                "error_code": error_code,
                "resume_state": AdaptiveStudyState.GENERATING_PAIR.value,
            },
        )
        return self.show(study_id)

    def record_preference(
        self,
        study_id: str,
        review_id: str,
        display_choice: DisplayPreference,
    ) -> AdaptiveStudyView:
        definition = self.studies.get(study_id)
        events = self.studies.list_events(study_id)
        self._validate_reviewer_binding(study_id, events)
        if not events:
            raise ValueError("adaptive study is not awaiting a preference")
        latest_review = self.reviews.latest_adaptive(review_id, self.reviewer_id)
        observations = self.studies.list_observations(study_id)
        existing = (
            None
            if latest_review is None
            else next(
                (item for item in observations if item.review_id == latest_review.id),
                None,
            )
        )
        if existing is not None:
            if latest_review is None or latest_review.display_preference is not display_choice:
                raise ValueError("adaptive v1 review is immutable")
            latest_event = events[-1]
            if (
                latest_event.new_state is AdaptiveStudyState.AWAITING_PREFERENCE
                and self._payload_string(latest_event, "review_session_id") == review_id
            ):
                self._append_event(
                    study_id,
                    previous=AdaptiveStudyState.AWAITING_PREFERENCE,
                    target=AdaptiveStudyState.UPDATING_POSTERIOR,
                    payload={
                        "observation_ordinal": existing.ordinal,
                        "review_id": existing.review_id,
                    },
                )
                return self.resume(study_id)
            if (
                latest_event.new_state is AdaptiveStudyState.UPDATING_POSTERIOR
                and self._payload_string(latest_event, "review_id") == existing.review_id
            ):
                return self.resume(study_id)
            return self.show(study_id)

        if events[-1].new_state in {
            AdaptiveStudyState.SEARCH_CONVERGED,
            AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED,
            AdaptiveStudyState.COMPLETED,
        }:
            raise ValueError("terminal adaptive study cannot record a preference")
        if events[-1].new_state is not AdaptiveStudyState.AWAITING_PREFERENCE:
            raise ValueError("adaptive study is not awaiting a preference")
        expected_review_id = self._payload_string(events[-1], "review_session_id")
        if review_id != expected_review_id:
            raise ValueError("review does not match the study's sealed comparison")
        revision = self.reviews.submit_adaptive(review_id, self.reviewer_id, display_choice)
        observation, consumed = self.studies.consume_observation_or_verify(
            AdaptiveObservationRequest(
                study_id=study_id,
                ordinal=len(observations) + 1,
                review_id=revision.id,
                created_at=utc_now(),
            )
        )
        if not consumed:
            return self._await_preference_winner(study_id, observation.review_id)
        self._append_event(
            study_id,
            previous=AdaptiveStudyState.AWAITING_PREFERENCE,
            target=AdaptiveStudyState.UPDATING_POSTERIOR,
            payload={
                "observation_ordinal": observation.ordinal,
                "review_id": observation.review_id,
            },
        )
        return self._resume_optimizer_update(definition)

    def _await_preference_winner(
        self,
        study_id: str,
        review_id: str,
        *,
        timeout_seconds: float = 5.0,
    ) -> AdaptiveStudyView:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            view = self.show(study_id)
            if view.state not in {
                AdaptiveStudyState.AWAITING_PREFERENCE,
                AdaptiveStudyState.UPDATING_POSTERIOR,
            }:
                return view
            time.sleep(0.01)
        events = self.studies.list_events(study_id)
        latest = events[-1]
        if (
            latest.new_state is AdaptiveStudyState.AWAITING_PREFERENCE
            and self._payload_string(latest, "review_session_id") == review_id
        ):
            observations = self.studies.list_observations(study_id)
            observation = next(item for item in observations if item.review_id == review_id)
            self._append_event(
                study_id,
                previous=AdaptiveStudyState.AWAITING_PREFERENCE,
                target=AdaptiveStudyState.UPDATING_POSTERIOR,
                payload={
                    "observation_ordinal": observation.ordinal,
                    "review_id": observation.review_id,
                },
            )
        return self.resume(study_id)

    def _validate_reviewer_binding(
        self, study_id: str, events: tuple[AdaptiveStudyEvent, ...]
    ) -> None:
        try:
            owner = self.studies.owner(study_id)
        except AdaptiveStudyConflict as error:
            raise PermissionError("adaptive study owner binding is unavailable") from error
        if owner != self.reviewer_id:
            raise PermissionError("adaptive study belongs to another reviewer")
        bound_reviewers = {
            reviewer
            for event in events
            if isinstance((reviewer := event.payload.get("reviewer_id")), str)
        }
        if bound_reviewers != {self.reviewer_id}:
            raise PermissionError("adaptive study reviewer evidence is inconsistent")

    @staticmethod
    def _pair_ref(study_id: str, optimizer_output_digest: str) -> str:
        digest = hashlib.sha256(
            b"adaptive-pair-ref.v1\x00"
            + study_id.encode("utf-8")
            + b"\x00"
            + optimizer_output_digest.encode("ascii")
        ).hexdigest()
        return f"adaptive-pair-ref.v1:{digest}"

    def _resume_optimizer_update(self, definition: AdaptiveStudyDefinition) -> AdaptiveStudyView:
        observations = self.studies.list_observations(definition.id)
        optimizer_observations = tuple(item.preference_observation for item in observations)
        steps = self.studies.list_optimizer_steps(definition.id)
        expected_ordinal = len(observations) + 1
        if len(steps) == expected_ordinal - 1:
            try:
                decision = self.optimizer(definition.optimizer_config, optimizer_observations)
            except Exception as error:
                self._append_event(
                    definition.id,
                    previous=AdaptiveStudyState.UPDATING_POSTERIOR,
                    target=AdaptiveStudyState.OPTIMIZER_FAILED,
                    payload={
                        "comparison_count": len(observations),
                        "error_type": type(error).__name__,
                    },
                )
                return self.show(definition.id)
            if (
                decision.stopping.reason is StopReason.CONTINUE
                and len(observations) >= definition.optimizer_config.maximum_comparisons
            ):
                self._append_event(
                    definition.id,
                    previous=AdaptiveStudyState.UPDATING_POSTERIOR,
                    target=AdaptiveStudyState.OPTIMIZER_FAILED,
                    payload={
                        "comparison_count": len(observations),
                        "error_code": "optimizer_continued_at_hard_cap",
                        "recoverable": False,
                    },
                )
                return self.show(definition.id)
            step = self.studies.append_optimizer_step(
                self._optimizer_step(
                    definition,
                    decision,
                    optimizer_observations=optimizer_observations,
                )
            )
            self._remember_optimizer_step(step)
        elif len(steps) == expected_ordinal:
            step = steps[-1]
            self._verify_optimizer_step(definition, step)
        else:
            raise RuntimeError("adaptive optimizer evidence has an invalid ordinal gap")
        if step.stop_reason is StopReason.CONTINUE:
            if len(observations) >= definition.optimizer_config.maximum_comparisons:
                self._append_event(
                    definition.id,
                    previous=AdaptiveStudyState.UPDATING_POSTERIOR,
                    target=AdaptiveStudyState.OPTIMIZER_FAILED,
                    payload={
                        "comparison_count": len(observations),
                        "error_code": "persisted_optimizer_continue_at_hard_cap",
                        "optimizer_step_ordinal": step.ordinal,
                        "optimizer_output_digest": step.output_digest,
                        "recoverable": False,
                    },
                )
                return self.show(definition.id)
            if step.selected_pair is None:
                raise RuntimeError("continuing optimizer step has no selected pair")
            experiment_id = adaptive_experiment_id(definition.id, step.output_digest)
            planned = self._append_event(
                definition.id,
                previous=AdaptiveStudyState.UPDATING_POSTERIOR,
                target=AdaptiveStudyState.GENERATING_PAIR,
                payload={
                    "optimizer_step_ordinal": step.ordinal,
                    "optimizer_output_digest": step.output_digest,
                    "experiment_id": experiment_id,
                },
            )
            return self._resume_generating(definition, planned, verify_optimizer=False)
        if step.stop_reason not in {
            StopReason.SEARCH_CONVERGED,
            StopReason.SEARCH_BUDGET_EXHAUSTED,
        }:
            self._append_event(
                definition.id,
                previous=AdaptiveStudyState.UPDATING_POSTERIOR,
                target=AdaptiveStudyState.OPTIMIZER_FAILED,
                payload={
                    "comparison_count": len(observations),
                    "optimizer_step_ordinal": step.ordinal,
                    "optimizer_output_digest": step.output_digest,
                    "stop_reason": step.stop_reason.value,
                    "recoverable": False,
                },
            )
            return self.show(definition.id)
        target = (
            AdaptiveStudyState.SEARCH_CONVERGED
            if step.stop_reason is StopReason.SEARCH_CONVERGED
            else AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED
        )
        self._append_event(
            definition.id,
            previous=AdaptiveStudyState.UPDATING_POSTERIOR,
            target=target,
            payload={
                "optimizer_step_ordinal": step.ordinal,
                "optimizer_output_digest": step.output_digest,
                "stop_reason": step.stop_reason.value,
            },
        )
        return self.show(definition.id)

    def _verify_optimizer_step(
        self,
        definition: AdaptiveStudyDefinition,
        step: AdaptiveOptimizerStep,
    ) -> None:
        key = (definition.id, step.ordinal, step.output_digest)
        if key in self._verified_optimizer_outputs:
            return
        observations = self.studies.list_observations(definition.id)
        prefix = tuple(
            item.preference_observation for item in observations[: step.comparison_count]
        )
        if len(prefix) != step.comparison_count:
            raise AdaptiveStudyConflict("optimizer replay references missing observations")
        try:
            replayed = self.optimizer(definition.optimizer_config, prefix)
        except Exception as error:
            raise AdaptiveStudyConflict("optimizer replay failed") from error
        if adaptive_optimizer_output_digest(replayed) != step.output_digest:
            raise AdaptiveStudyConflict("optimizer replay output does not match persisted evidence")
        self._verified_optimizer_outputs.add(key)

    def _remember_optimizer_step(self, step: AdaptiveOptimizerStep) -> None:
        self._verified_optimizer_outputs.add((step.study_id, step.ordinal, step.output_digest))

    def _append_event(
        self,
        study_id: str,
        *,
        previous: AdaptiveStudyState,
        target: AdaptiveStudyState,
        payload: Mapping[str, JsonValue],
    ) -> AdaptiveStudyEvent:
        return self.studies.append_event(
            AdaptiveStudyEvent(
                study_id=study_id,
                sequence=len(self.studies.list_events(study_id)) + 1,
                previous_state=previous,
                new_state=target,
                payload=payload,
                created_at=utc_now(),
            )
        )

    @staticmethod
    def _pair_draft(
        definition: AdaptiveStudyDefinition, value_a: float, value_b: float
    ) -> ExperimentDraft:
        return ExperimentDraft(
            baseline=definition.base_manifest.model_copy(update={"guidance_scale": value_a}),
            variable_path="guidance_scale",
            alternate_value=value_b,
        )

    @staticmethod
    def _optimizer_step(
        definition: AdaptiveStudyDefinition,
        decision: OptimizerDecision,
        *,
        optimizer_observations: tuple[PreferenceObservation, ...] = (),
        created_at: datetime | None = None,
    ) -> AdaptiveOptimizerStep:
        return AdaptiveOptimizerStep(
            study_id=definition.id,
            ordinal=len(optimizer_observations) + 1,
            input_digest=adaptive_optimizer_input_digest(
                definition.optimizer_config, optimizer_observations
            ),
            output_digest=adaptive_optimizer_output_digest(decision),
            posterior=decision.posterior,
            acquisition_ledger=decision.candidate_scores,
            selected_pair=decision.selected_pair,
            stop_reason=decision.stopping.reason,
            disposition=decision.stopping.disposition,
            comparison_count=decision.stopping.comparison_count,
            maximum_information_bits=decision.stopping.maximum_information_bits,
            created_at=utc_now() if created_at is None else created_at,
        )

    @staticmethod
    def _payload_string(event: AdaptiveStudyEvent, key: str) -> str:
        value = event.payload.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"adaptive event payload is missing {key}")
        return value

    @staticmethod
    def _payload_int(event: AdaptiveStudyEvent, key: str) -> int:
        value = event.payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"adaptive event payload is missing {key}")
        return value
