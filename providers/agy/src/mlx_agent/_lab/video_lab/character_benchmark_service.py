from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from typing import Any, Literal, cast

from video_lab.artifacts import ArtifactError
from video_lab.character_benchmark_analysis import analyze_character_benchmark
from video_lab.character_benchmark_models import (
    CharacterBenchmarkDefinition,
    CharacterBenchmarkEvidence,
    CharacterBenchmarkPanelInput,
    CharacterBenchmarkResult,
    CharacterRecipePair,
    CharacterTrial,
    IdentityChoice,
    IdentityObservation,
    SecondaryDimension,
    SecondaryObservation,
)
from video_lab.character_benchmark_protocol import build_character_panel, schedule_character_trials
from video_lab.character_benchmark_repository import (
    CharacterBenchmarkCampaign,
    CharacterBenchmarkConflict,
    CharacterBenchmarkEvent,
    CharacterBenchmarkNotFound,
    CharacterBenchmarkRepository,
    CharacterBenchmarkState,
    CharacterReviewBinding,
)
from video_lab.database import character_benchmark_reviewer_id
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    FrozenModel,
    JobState,
    Manifest,
    canonical_json,
    utc_now,
)


class CharacterBenchmarkProgress(FrozenModel):
    completed_identity: int
    total_identity: Literal[40] = 40
    generated_primary: int
    total_primary: Literal[36] = 36


class CharacterBenchmarkPreview(FrozenModel):
    definition: CharacterBenchmarkDefinition
    primary_count: Literal[36] = 36
    repeat_count: Literal[4] = 4


class CharacterBenchmarkMedia(FrozenModel):
    position: Literal["left", "right"]
    media_ref: str


class CharacterBenchmarkView(FrozenModel):
    campaign_id: str
    state: CharacterBenchmarkState
    next_action: str
    progress: CharacterBenchmarkProgress
    trial_id: str | None = None
    review_id: str | None = None
    media: tuple[CharacterBenchmarkMedia, ...] = ()
    result: CharacterBenchmarkResult | None = None


class CharacterBenchmarkService:
    def __init__(
        self,
        campaigns: CharacterBenchmarkRepository,
        character_packs: Any,
        stage_2: Any,
        experiments: Any,
        reviews: Any,
        adapters: Any,
        *,
        reviewer_id: str,
        clock: Callable[[], Any] = utc_now,
    ) -> None:
        self.campaigns = campaigns
        self.character_packs = character_packs
        self.stage_2 = stage_2
        self.experiments = experiments
        self.reviews = reviews
        self.adapters = adapters
        self.reviewer_id = reviewer_id
        self.clock = clock

    def create(
        self,
        *,
        character_pack_id: str,
        stage_2_campaign_id: str,
        identity_prompt: str,
        panel_input: CharacterBenchmarkPanelInput,
    ) -> CharacterBenchmarkView:
        panel = CharacterBenchmarkPanelInput.model_validate(panel_input.model_dump())
        pack_export = self._pack_export(character_pack_id)
        stage_2_evidence, base_manifest = self._stage_2_evidence(stage_2_campaign_id)
        if base_manifest.loras:
            raise CharacterBenchmarkConflict("character benchmark forbids LoRA bindings")
        pack_digest = _digest(pack_export.pack.model_dump(mode="json"))
        manifest_digest = _digest(base_manifest.model_dump(mode="json"))
        evidence_digest = _digest(stage_2_evidence)
        recipes = _recipe_pair(identity_prompt)
        definition = build_character_panel(
            character_pack_digest=pack_digest,
            stage_2_manifest_digest=manifest_digest,
            stage_2_evidence_digest=evidence_digest,
            recipes=recipes,
            panel_input=panel,
        )
        with self.campaigns.database.connect() as connection:
            pack = connection.execute(
                "SELECT storage_id, id, reviewer_id, state FROM character_packs WHERE id=?",
                (character_pack_id,),
            ).fetchone()
        if (
            pack is None
            or str(pack["reviewer_id"]) != self.reviewer_id
            or str(pack["state"]) != "preregistered"
            or str(pack["id"]) != character_pack_id
        ):
            raise CharacterBenchmarkConflict("character pack is not owned and frozen")
        try:
            created_at = self.campaigns.get(definition.id).created_at
        except CharacterBenchmarkNotFound:
            created_at = self.clock()
        content_digest = _digest(
            {
                "definition": definition.model_dump(mode="json"),
                "character_pack_id": character_pack_id,
                "character_pack_storage_id": str(pack["storage_id"]),
                "stage_2_campaign_id": stage_2_campaign_id,
                "stage_2_manifest": base_manifest.model_dump(mode="json"),
                "identity_prompt": identity_prompt,
            }
        )
        campaign = self.campaigns.create_or_get_bound(
            definition,
            character_pack_id=character_pack_id,
            character_pack_storage_id=str(pack["storage_id"]),
            stage_2_campaign_id=stage_2_campaign_id,
            stage_2_manifest=base_manifest,
            identity_prompt=identity_prompt,
            content_digest=content_digest,
            created_at=created_at,
        )
        return self._view(campaign)

    def preview(self, campaign_id: str) -> CharacterBenchmarkPreview:
        campaign = self.campaigns.get(campaign_id)
        if campaign.state is not CharacterBenchmarkState.DRAFT:
            raise PermissionError("character benchmark schedule is sealed after preregistration")
        return CharacterBenchmarkPreview(definition=campaign.definition)

    def preregister(self, campaign_id: str) -> CharacterBenchmarkView:
        campaign = self.campaigns.get(campaign_id)
        state = campaign.state
        if state is CharacterBenchmarkState.DRAFT:
            self._validate_bound_inputs(campaign)
            self.campaigns.preregister_or_verify(
                campaign_id,
                schedule_character_trials(campaign.definition),
                created_at=self.clock(),
            )
            state = CharacterBenchmarkState.PREREGISTERED
        if state is CharacterBenchmarkState.PREREGISTERED:
            self.campaigns.append_event_or_verify(
                self._event(campaign_id, 2, state, CharacterBenchmarkState.GENERATING, "generation")
            )
            state = CharacterBenchmarkState.GENERATING
        if state is not CharacterBenchmarkState.GENERATING:
            return self.show(campaign_id)
        capabilities = self.adapters.capabilities(campaign.stage_2_manifest.adapter_id)
        for trial in self.campaigns.list_trials(campaign_id):
            if trial.hidden_repeat:
                continue
            pair = self.experiments.preview(self._draft(campaign, trial), capabilities)
            if pair.changed_paths != ("prompt",) or pair.variant_a.loras or pair.variant_b.loras:
                raise CharacterBenchmarkConflict(
                    "character benchmark candidate recipe must be prompt-only without LoRAs"
                )
            self.campaigns.bind_experiment_or_verify(
                campaign_id, trial, pair, created_at=self.clock()
            )
        return self.show(campaign_id)

    def show(self, campaign_id: str) -> CharacterBenchmarkView:
        return self._view(self.campaigns.get(campaign_id))

    def resume(self, campaign_id: str) -> CharacterBenchmarkView:
        campaign = self.campaigns.get(campaign_id)
        state = campaign.state
        if state is CharacterBenchmarkState.DRAFT:
            return self._view(campaign)
        if state in {CharacterBenchmarkState.PREREGISTERED, CharacterBenchmarkState.GENERATING}:
            self.preregister(campaign_id)
            bindings = self.campaigns.list_experiment_bindings(campaign_id)
            jobs = tuple(
                job
                for binding in bindings
                for job in self.experiments.get(binding.experiment_id).jobs
            )
            if (
                len(bindings) != 36
                or len(jobs) != 72
                or any(job.state is not JobState.COMPLETED for job in jobs)
            ):
                return self.show(campaign_id)
            self.campaigns.append_event_or_verify(
                self._event(
                    campaign_id,
                    3,
                    CharacterBenchmarkState.GENERATING,
                    CharacterBenchmarkState.READY,
                    "ready",
                )
            )
            state = CharacterBenchmarkState.READY
        if state is CharacterBenchmarkState.READY:
            self.campaigns.append_event_or_verify(
                self._event(campaign_id, 4, state, CharacterBenchmarkState.JUDGING, "judging")
            )
            state = CharacterBenchmarkState.JUDGING
        if state is CharacterBenchmarkState.JUDGING:
            trial = self._next_incomplete_trial(campaign_id)
            if trial is None:
                return self.analyze(campaign_id)
            return self._judging_view(campaign_id, trial)
        if state in {CharacterBenchmarkState.ANALYZING, CharacterBenchmarkState.COMPLETED}:
            return self.analyze(campaign_id)
        return self.show(campaign_id)

    def record_identity(
        self, campaign_id: str, review_id: str, observation: IdentityObservation
    ) -> CharacterBenchmarkView:
        campaign = self.campaigns.get(campaign_id)
        validated = IdentityObservation.model_validate(observation.model_dump())
        if validated.benchmark_id != campaign_id:
            raise CharacterBenchmarkConflict("identity observation belongs to another benchmark")
        existing_binding = self._review_binding_for_session(campaign_id, review_id)
        if existing_binding is not None:
            bound_trial = self._trial(campaign_id, existing_binding.trial_id)
            if validated.trial_id != bound_trial.id:
                raise CharacterBenchmarkConflict(
                    "identity observation retry does not match bound review"
                )
            existing_identity = next(
                (
                    item
                    for item in self.campaigns.list_identity(campaign_id)
                    if item.observation.trial_id == bound_trial.id
                ),
                None,
            )
            if existing_identity is not None:
                self._validate_bound_identity_review(
                    campaign,
                    bound_trial,
                    existing_binding,
                    existing_identity.observation,
                )
                if existing_identity.observation != validated:
                    raise CharacterBenchmarkConflict(
                        "identity observation retry does not match immutable evidence"
                    )
                return self.show(campaign_id)
        trial = self._next_incomplete_trial(campaign_id)
        if trial is None or trial.id != validated.trial_id:
            existing = next(
                (
                    item
                    for item in self.campaigns.list_identity(campaign_id)
                    if item.observation.trial_id == validated.trial_id
                ),
                None,
            )
            if existing is None:
                raise CharacterBenchmarkConflict("identity observation is not next in schedule")
        session, experiment_id, reviewer_alias = self._authorized_session(
            campaign, trial or self._trial(campaign_id, validated.trial_id), review_id
        )
        bound_trial = trial or self._trial(campaign_id, validated.trial_id)
        mapping = self._display_mapping(bound_trial, session)
        display_choice = _display_choice(validated.choice, bound_trial.candidate_a_position)
        preference = _underlying_preference(display_choice, mapping)
        try:
            revision = self.reviews.reviews.submit_adaptive(review_id, preference)
        except (LookupError, ValueError) as error:
            raise CharacterBenchmarkConflict(
                "character benchmark review retry does not match evidence"
            ) from error
        self.campaigns.bind_review_or_verify(
            CharacterReviewBinding(
                campaign_id=campaign_id,
                trial_id=bound_trial.id,
                experiment_id=experiment_id,
                review_id=revision.id,
                session_id=review_id,
                reviewer_alias=reviewer_alias,
                reused_from_trial_id=bound_trial.reuses_trial_id,
                display_mapping=mapping,
            )
        )
        self.campaigns.record_identity_or_verify(
            validated,
            review_id=revision.id,
            display_choice=display_choice,
            created_at=self.clock(),
        )
        if self._requires_secondary(bound_trial, validated.choice):
            secondary = {
                item.observation.dimension
                for item in self.campaigns.list_secondary(campaign_id)
                if item.observation.trial_id == bound_trial.id
            }
            if secondary != set(SecondaryDimension):
                return self._judging_view(campaign_id, bound_trial)
        return self.resume(campaign_id)

    def record_secondary(
        self, campaign_id: str, review_id: str, observation: SecondaryObservation
    ) -> CharacterBenchmarkView:
        campaign = self.campaigns.get(campaign_id)
        validated = SecondaryObservation.model_validate(observation.model_dump())
        if validated.benchmark_id != campaign_id:
            raise CharacterBenchmarkConflict("secondary observation belongs to another benchmark")
        binding = self._review_binding_for_session(campaign_id, review_id)
        if binding is None or binding.trial_id != validated.trial_id:
            raise CharacterBenchmarkConflict("secondary observation is not bound to review")
        trial = self._trial(campaign_id, binding.trial_id)
        identity = next(
            (
                item
                for item in self.campaigns.list_identity(campaign_id)
                if item.observation.trial_id == trial.id
            ),
            None,
        )
        if identity is None or not self._requires_secondary(trial, identity.observation.choice):
            raise CharacterBenchmarkConflict("secondary observation is not eligible")
        self._validate_bound_identity_review(campaign, trial, binding, identity.observation)
        existing = next(
            (
                item.observation
                for item in self.campaigns.list_secondary(campaign_id)
                if item.observation.trial_id == trial.id
                and item.observation.dimension is validated.dimension
            ),
            None,
        )
        if existing is not None:
            if existing != validated:
                raise CharacterBenchmarkConflict(
                    "secondary observation retry does not match immutable evidence"
                )
            return self.show(campaign_id)
        if self._next_incomplete_trial(campaign_id) != trial:
            raise CharacterBenchmarkConflict("secondary observation is not next in schedule")
        self.campaigns.record_secondary_or_verify(validated, created_at=self.clock())
        supplied = {
            item.observation.dimension
            for item in self.campaigns.list_secondary(campaign_id)
            if item.observation.trial_id == trial.id
        }
        if supplied == set(SecondaryDimension):
            return self.resume(campaign_id)
        return self._judging_view(campaign_id, trial)

    def analyze(self, campaign_id: str) -> CharacterBenchmarkView:
        campaign = self.campaigns.get(campaign_id)
        evidence = self._validated_evidence(campaign, require_terminal=True)
        state = campaign.state
        if state is CharacterBenchmarkState.JUDGING:
            self.campaigns.append_event_or_verify(
                self._event(campaign_id, 5, state, CharacterBenchmarkState.ANALYZING, "analysis")
            )
            state = CharacterBenchmarkState.ANALYZING
        if state not in {CharacterBenchmarkState.ANALYZING, CharacterBenchmarkState.COMPLETED}:
            raise CharacterBenchmarkConflict("character benchmark cannot enter analysis")
        calculated = analyze_character_benchmark(evidence)
        persisted = self.campaigns.get_result(campaign_id)
        if persisted is None:
            persisted = self.campaigns.append_result_or_verify(calculated, created_at=self.clock())
        elif persisted != calculated:
            raise CharacterBenchmarkConflict("character benchmark result digest is inconsistent")
        if state is CharacterBenchmarkState.ANALYZING:
            self.campaigns.append_event_or_verify(
                self._event(campaign_id, 6, state, CharacterBenchmarkState.COMPLETED, "completed")
            )
        return self.show(campaign_id)

    def export_evidence(self, campaign_id: str) -> CharacterBenchmarkEvidence:
        campaign = self.campaigns.get(campaign_id)
        if campaign.state is not CharacterBenchmarkState.COMPLETED:
            if self.campaigns.get_result(campaign_id) is not None:
                self._validated_evidence(campaign, require_terminal=True)
                raise CharacterBenchmarkConflict(
                    "character benchmark terminal lifecycle is inconsistent"
                )
            raise PermissionError("character benchmark evidence remains sealed")
        return self._validated_evidence(campaign, require_terminal=True)

    def _pack_export(self, pack_id: str) -> Any:
        try:
            exported = self.character_packs.export_pack(pack_id)
        except (
            LookupError,
            PermissionError,
            RuntimeError,
            ValueError,
            ArtifactError,
            OSError,
        ) as error:
            raise CharacterBenchmarkConflict("character pack evidence is not canonical") from error
        if getattr(exported.pack, "pack_id", None) != pack_id:
            raise CharacterBenchmarkConflict("character pack version does not match binding")
        return exported

    def _stage_2_evidence(self, campaign_id: str) -> tuple[dict[str, object], Manifest]:
        try:
            view = self.stage_2.show(campaign_id)
            state = getattr(view.state, "value", view.state)
            evidence = self.stage_2.export_evidence(campaign_id)
        except (LookupError, PermissionError, RuntimeError, ValueError) as error:
            raise CharacterBenchmarkConflict(
                "Stage 2 evidence is not owned and terminal"
            ) from error
        if state != "qualified" or not isinstance(evidence, dict):
            raise CharacterBenchmarkConflict("Stage 2 evidence is not qualified")
        try:
            result = cast(dict[str, object], evidence["result"])
            campaign = cast(dict[str, object], evidence["campaign"])
            base = Manifest.model_validate(campaign["base_manifest"])
        except (KeyError, TypeError, ValueError) as error:
            raise CharacterBenchmarkConflict("Stage 2 evidence is incomplete") from error
        if result.get("disposition") != "qualified":
            raise CharacterBenchmarkConflict("Stage 2 evidence is not qualified")
        return evidence, base

    def _validate_bound_inputs(self, campaign: CharacterBenchmarkCampaign) -> None:
        exported = self._pack_export(campaign.character_pack_id)
        stage_evidence, manifest = self._stage_2_evidence(campaign.stage_2_campaign_id)
        expected_definition = build_character_panel(
            character_pack_digest=_digest(exported.pack.model_dump(mode="json")),
            stage_2_manifest_digest=_digest(manifest.model_dump(mode="json")),
            stage_2_evidence_digest=_digest(stage_evidence),
            recipes=_recipe_pair(campaign.identity_prompt),
            panel_input=campaign.definition.panel_input,
        )
        expected_content_digest = _digest(
            {
                "definition": campaign.definition.model_dump(mode="json"),
                "character_pack_id": campaign.character_pack_id,
                "character_pack_storage_id": campaign.character_pack_storage_id,
                "stage_2_campaign_id": campaign.stage_2_campaign_id,
                "stage_2_manifest": manifest.model_dump(mode="json"),
                "identity_prompt": campaign.identity_prompt,
            }
        )
        if (
            expected_definition != campaign.definition
            or manifest != campaign.stage_2_manifest
            or expected_content_digest != campaign.content_digest
        ):
            raise CharacterBenchmarkConflict("character benchmark bound evidence digest changed")

    def _validated_evidence(
        self, campaign: CharacterBenchmarkCampaign, *, require_terminal: bool
    ) -> CharacterBenchmarkEvidence:
        self._validate_bound_inputs(campaign)
        trials = self.campaigns.list_trials(campaign.definition.id)
        if trials != schedule_character_trials(campaign.definition):
            raise CharacterBenchmarkConflict("character benchmark trial binding is inconsistent")
        experiment_bindings = self.campaigns.list_experiment_bindings(campaign.definition.id)
        by_trial = {item.trial_id: item for item in experiment_bindings}
        primary = tuple(item for item in trials if not item.hidden_repeat)
        if len(experiment_bindings) != 36 or set(by_trial) != {item.id for item in primary}:
            raise CharacterBenchmarkConflict(
                "character benchmark experiment provenance is incomplete"
            )
        artifact_sets: dict[str, frozenset[tuple[str, str]]] = {}
        for trial in primary:
            experiment_binding = by_trial[trial.id]
            pair = self.experiments.preview(
                self._draft(campaign, trial),
                self.adapters.capabilities(campaign.stage_2_manifest.adapter_id),
            )
            view = self.experiments.get(experiment_binding.experiment_id)
            expected_jobs = {
                _record_id("job", campaign.definition.id, spec.id): spec.id
                for spec in trial.generation_specs
            }
            if (
                experiment_binding.experiment_id
                != _record_id("exp", campaign.definition.id, trial.id)
                or view.experiment.schema_version != "character-controlled.v1"
                or view.experiment.variable_path != "prompt"
                or view.experiment.changed_paths != ("prompt",)
                or tuple(item.id for item in view.experiment.variants)
                != tuple(item.id for item in trial.generation_specs)
                or tuple(item.manifest for item in view.experiment.variants)
                != (pair.variant_a, pair.variant_b)
                or len(view.jobs) != 2
                or {item.id: item.variant_id for item in view.jobs} != expected_jobs
                or any(item.state is not JobState.COMPLETED for item in view.jobs)
            ):
                raise CharacterBenchmarkConflict(
                    "character benchmark experiment recipe is inconsistent"
                )
            artifacts: set[tuple[str, str]] = set()
            for job in view.jobs:
                registered = self.experiments.artifacts.list_for_job(job.id)
                if len(registered) != 1 or registered[0].media_type != "video/mp4":
                    raise CharacterBenchmarkConflict(
                        "character benchmark artifact ownership is inconsistent"
                    )
                artifact = registered[0]
                try:
                    self.experiments.artifacts.resolve(artifact.id)
                except (ArtifactError, OSError) as error:
                    raise CharacterBenchmarkConflict(
                        "character benchmark artifact bytes are not canonical"
                    ) from error
                artifacts.add((artifact.id, artifact.sha256))
            if len(artifacts) != 2:
                raise CharacterBenchmarkConflict(
                    "character benchmark variant artifacts are not distinct"
                )
            artifact_sets[trial.id] = frozenset(artifacts)
        identities = self.campaigns.list_identity(campaign.definition.id)
        secondary = self.campaigns.list_secondary(campaign.definition.id)
        reviews = self.campaigns.list_review_bindings(campaign.definition.id)
        if len(identities) != 40 or len(reviews) != 40:
            raise CharacterBenchmarkConflict("character benchmark review evidence is incomplete")
        identity_by_trial = {item.observation.trial_id: item for item in identities}
        review_by_trial = {item.trial_id: item for item in reviews}
        if set(identity_by_trial) != {item.id for item in trials} or set(review_by_trial) != set(
            identity_by_trial
        ):
            raise CharacterBenchmarkConflict(
                "character benchmark identity judgment relation is inconsistent"
            )
        with self.campaigns.database.connect() as connection:
            revision_rows = connection.execute(
                "SELECT revision.* FROM review_revisions AS revision WHERE session_id IN "
                "(SELECT session_id FROM character_benchmark_review_bindings "
                "WHERE campaign_id=?)",
                (campaign.definition.id,),
            ).fetchall()
            result_row = connection.execute(
                "SELECT result_json, content_digest FROM character_benchmark_results "
                "WHERE campaign_id=?",
                (campaign.definition.id,),
            ).fetchone()
        events = self.campaigns.list_events(campaign.definition.id)
        expected_event_count = 6 if result_row is not None else 4
        expected_states = (
            (CharacterBenchmarkState.DRAFT, CharacterBenchmarkState.PREREGISTERED),
            (CharacterBenchmarkState.PREREGISTERED, CharacterBenchmarkState.GENERATING),
            (CharacterBenchmarkState.GENERATING, CharacterBenchmarkState.READY),
            (CharacterBenchmarkState.READY, CharacterBenchmarkState.JUDGING),
            (CharacterBenchmarkState.JUDGING, CharacterBenchmarkState.ANALYZING),
            (CharacterBenchmarkState.ANALYZING, CharacterBenchmarkState.COMPLETED),
        )
        expected_event_digests = (
            _digest([item.model_dump(mode="json") for item in trials]),
            *(
                _digest(
                    {
                        "domain": domain,
                        "campaign_id": campaign.definition.id,
                        "sequence": sequence,
                    }
                )
                for sequence, domain in enumerate(
                    ("generation", "ready", "judging", "analysis", "completed"),
                    start=2,
                )
            ),
        )
        if (
            len(events) != expected_event_count
            or tuple(item.sequence for item in events) != tuple(range(1, expected_event_count + 1))
            or tuple((item.previous_state, item.new_state) for item in events)
            != expected_states[:expected_event_count]
            or tuple(item.payload_digest for item in events)
            != expected_event_digests[:expected_event_count]
        ):
            raise CharacterBenchmarkConflict(
                "character benchmark event cardinality is inconsistent"
            )
        revisions_by_session = {str(row["session_id"]): row for row in revision_rows}
        if (
            len(revision_rows) != 40
            or len(revisions_by_session) != 40
            or set(revisions_by_session) != {item.session_id for item in reviews}
            or {str(row["id"]) for row in revision_rows} != {item.review_id for item in reviews}
        ):
            raise CharacterBenchmarkConflict(
                "character benchmark review revision cardinality is inconsistent"
            )
        expected_secondary: set[tuple[str, SecondaryDimension]] = set()
        for trial in trials:
            review_binding = review_by_trial[trial.id]
            source_id = trial.reuses_trial_id or trial.id
            if (
                review_binding.experiment_id != by_trial[source_id].experiment_id
                or review_binding.reused_from_trial_id != trial.reuses_trial_id
            ):
                raise CharacterBenchmarkConflict(
                    "character benchmark repeat source is inconsistent"
                )
            session = self._session(review_binding.session_id)
            if (
                str(session["reviewer_id"])
                != _reviewer_alias(campaign.definition.id, trial.id, self.reviewer_id)
                or str(session["experiment_id"]) != review_binding.experiment_id
                or review_binding.display_mapping != self._display_mapping(trial, session)
            ):
                raise CharacterBenchmarkConflict(
                    "character benchmark display mapping is inconsistent"
                )
            revision = revisions_by_session.get(review_binding.session_id)
            if (
                revision is None
                or str(revision["id"]) != review_binding.review_id
                or int(revision["revision"]) != 1
                or revision["supersedes_review_id"] is not None
            ):
                raise CharacterBenchmarkConflict(
                    "character benchmark review revision is inconsistent"
                )
            if trial.hidden_repeat:
                if artifact_sets[source_id] != self._session_artifacts(session):
                    raise CharacterBenchmarkConflict(
                        "character benchmark repeat media reuse is inconsistent"
                    )
            identity = identity_by_trial[trial.id]
            if identity.review_id != review_binding.review_id:
                raise CharacterBenchmarkConflict(
                    "character benchmark identity judgment review is inconsistent"
                )
            expected_display = _display_choice(
                identity.observation.choice, trial.candidate_a_position
            )
            expected_identity_digest = _digest(
                {
                    "observation": identity.observation.model_dump(mode="json"),
                    "review_id": identity.review_id,
                    "display_choice": expected_display,
                }
            )
            underlying_preference = _underlying_preference(
                expected_display, review_binding.display_mapping
            )
            expected_preferred_variant_id = (
                None
                if underlying_preference is DisplayPreference.TIE
                else str(session[f"{underlying_preference.value}_variant_id"])
            )
            if (
                identity.display_choice != expected_display
                or identity.content_digest != expected_identity_digest
                or str(revision["display_preference"]) != underlying_preference.value
                or revision["preferred_variant_id"] != expected_preferred_variant_id
            ):
                raise CharacterBenchmarkConflict(
                    "character benchmark identity judgment content is inconsistent"
                )
            if self._requires_secondary(trial, identity.observation.choice):
                expected_secondary.update((trial.id, dimension) for dimension in SecondaryDimension)
        actual_secondary = {
            (item.observation.trial_id, item.observation.dimension) for item in secondary
        }
        if actual_secondary != expected_secondary:
            raise CharacterBenchmarkConflict(
                "character benchmark secondary judgment relation is inconsistent"
            )
        if any(
            item.review_id != identity_by_trial[item.observation.trial_id].review_id
            for item in secondary
        ):
            raise CharacterBenchmarkConflict(
                "character benchmark secondary judgment review is inconsistent"
            )
        if any(
            item.content_digest != _digest(item.observation.model_dump(mode="json"))
            for item in secondary
        ):
            raise CharacterBenchmarkConflict(
                "character benchmark secondary judgment content is inconsistent"
            )
        evidence = CharacterBenchmarkEvidence(
            definition=campaign.definition,
            trials=trials,
            identity_observations=tuple(item.observation for item in identities),
            secondary_observations=tuple(item.observation for item in secondary),
        )
        if require_terminal and result_row is not None:
            calculated = analyze_character_benchmark(evidence)
            if (
                str(result_row["content_digest"]) != _digest(calculated.model_dump(mode="json"))
                or CharacterBenchmarkResult.model_validate_json(str(result_row["result_json"]))
                != calculated
            ):
                raise CharacterBenchmarkConflict(
                    "character benchmark result digest is inconsistent"
                )
        return evidence

    def _draft(
        self, campaign: CharacterBenchmarkCampaign, trial: CharacterTrial
    ) -> ExperimentDraft:
        context_prompt = trial.context.prompt
        control = campaign.stage_2_manifest.model_copy(
            update={
                "prompt": context_prompt,
                "negative_prompt": trial.context.negative_prompt
                or campaign.stage_2_manifest.negative_prompt,
                "seed": trial.seed,
                "loras": (),
            }
        )
        identity = f"{context_prompt} {campaign.identity_prompt}"
        return ExperimentDraft(baseline=control, variable_path="prompt", alternate_value=identity)

    def _next_incomplete_trial(self, campaign_id: str) -> CharacterTrial | None:
        identities = {
            item.observation.trial_id: item for item in self.campaigns.list_identity(campaign_id)
        }
        secondary = self.campaigns.list_secondary(campaign_id)
        dimensions = {
            trial_id: {
                item.observation.dimension
                for item in secondary
                if item.observation.trial_id == trial_id
            }
            for trial_id in identities
        }
        for trial in self.campaigns.list_trials(campaign_id):
            identity = identities.get(trial.id)
            if identity is None:
                return trial
            if self._requires_secondary(trial, identity.observation.choice) and dimensions.get(
                trial.id, set()
            ) != set(SecondaryDimension):
                return trial
        return None

    def _judging_view(self, campaign_id: str, trial: CharacterTrial) -> CharacterBenchmarkView:
        campaign = self.campaigns.get(campaign_id)
        existing = next(
            (
                item
                for item in self.campaigns.list_review_bindings(campaign_id)
                if item.trial_id == trial.id
            ),
            None,
        )
        source_id = trial.reuses_trial_id or trial.id
        experiment_id = next(
            item.experiment_id
            for item in self.campaigns.list_experiment_bindings(campaign_id)
            if item.trial_id == source_id
        )
        alias = _reviewer_alias(campaign_id, trial.id, self.reviewer_id)
        session_id = (
            existing.session_id
            if existing is not None
            else self.reviews.start(experiment_id, alias).session_id
        )
        session = self._session(session_id)
        mapping = self._display_mapping(trial, session)
        update = {
            "trial_id": trial.id,
            "review_id": session_id,
            "media": tuple(
                CharacterBenchmarkMedia(
                    position=cast(Literal["left", "right"], position),
                    media_ref=f"/reviews/{session_id}/artifacts/{item['underlying_position']}",
                )
                for position, item in sorted(mapping.items())
            ),
            "next_action": (
                "record_secondary"
                if trial.id
                in {item.observation.trial_id for item in self.campaigns.list_identity(campaign_id)}
                else "record_identity"
            ),
        }
        return self._view(campaign).model_copy(update=update)

    def _authorized_session(
        self, campaign: CharacterBenchmarkCampaign, trial: CharacterTrial, session_id: str
    ) -> tuple[sqlite3.Row, str, str]:
        session = self._session(session_id)
        source_id = trial.reuses_trial_id or trial.id
        experiment_id = next(
            item.experiment_id
            for item in self.campaigns.list_experiment_bindings(campaign.definition.id)
            if item.trial_id == source_id
        )
        alias = _reviewer_alias(campaign.definition.id, trial.id, self.reviewer_id)
        if str(session["experiment_id"]) != experiment_id or str(session["reviewer_id"]) != alias:
            raise CharacterBenchmarkConflict("character benchmark review does not belong to trial")
        with self.campaigns.database.connect() as connection:
            for position in ("left", "right"):
                job_id = str(session[f"{position}_job_id"])
                variant_id = str(session[f"{position}_variant_id"])
                artifact_id = str(session[f"{position}_artifact_id"])
                row = connection.execute(
                    "SELECT job.state, job.variant_id, link.artifact_id FROM jobs AS job "
                    "JOIN job_artifacts AS link ON link.job_id=job.id WHERE job.id=?",
                    (job_id,),
                ).fetchone()
                artifact = self.experiments.artifacts.get(artifact_id)
                if (
                    row is None
                    or str(row["state"]) != "completed"
                    or str(row["variant_id"]) != variant_id
                    or str(row["artifact_id"]) != artifact_id
                    or artifact.sha256 != str(session[f"{position}_artifact_sha256"])
                ):
                    raise CharacterBenchmarkConflict(
                        "character benchmark review artifact ownership is inconsistent"
                    )
                try:
                    self.experiments.artifacts.resolve(artifact_id)
                except (ArtifactError, OSError) as error:
                    raise CharacterBenchmarkConflict(
                        "character benchmark review artifact bytes are not canonical"
                    ) from error
        return session, experiment_id, alias

    def _review_binding_for_session(
        self, campaign_id: str, session_id: str
    ) -> CharacterReviewBinding | None:
        matches = tuple(
            item
            for item in self.campaigns.list_review_bindings(campaign_id)
            if item.session_id == session_id
        )
        if len(matches) > 1:
            raise CharacterBenchmarkConflict(
                "character benchmark review session has multiple bindings"
            )
        return matches[0] if matches else None

    def _validate_bound_identity_review(
        self,
        campaign: CharacterBenchmarkCampaign,
        trial: CharacterTrial,
        binding: CharacterReviewBinding,
        observation: IdentityObservation,
    ) -> None:
        session, experiment_id, reviewer_alias = self._authorized_session(
            campaign, trial, binding.session_id
        )
        mapping = self._display_mapping(trial, session)
        display_choice = _display_choice(observation.choice, trial.candidate_a_position)
        preference = _underlying_preference(display_choice, mapping)
        expected_preferred_variant_id = (
            None
            if preference is DisplayPreference.TIE
            else str(session[f"{preference.value}_variant_id"])
        )
        with self.campaigns.database.connect() as connection:
            revision = connection.execute(
                "SELECT * FROM review_revisions WHERE id=? AND session_id=?",
                (binding.review_id, binding.session_id),
            ).fetchone()
        if (
            binding.campaign_id != campaign.definition.id
            or binding.trial_id != trial.id
            or binding.experiment_id != experiment_id
            or binding.reviewer_alias != reviewer_alias
            or binding.reused_from_trial_id != trial.reuses_trial_id
            or binding.display_mapping != mapping
            or revision is None
            or int(revision["revision"]) != 1
            or revision["supersedes_review_id"] is not None
            or str(revision["display_preference"]) != preference.value
            or revision["preferred_variant_id"] != expected_preferred_variant_id
        ):
            raise CharacterBenchmarkConflict(
                "character benchmark bound review is not immutable evidence"
            )

    def _display_mapping(
        self, trial: CharacterTrial, session: sqlite3.Row
    ) -> dict[str, dict[str, str]]:
        if trial.hidden_repeat and trial.reuses_trial_id is not None:
            source = self._trial(trial.benchmark_id, trial.reuses_trial_id)
        else:
            source = trial
        candidate_a_underlying = (
            "left" if str(session["left_variant_id"]) == source.generation_specs[0].id else "right"
        )
        mapping: dict[str, dict[str, str]] = {}
        for public in ("left", "right"):
            candidate = "a" if public == trial.candidate_a_position else "b"
            underlying = (
                candidate_a_underlying if candidate == "a" else _opposite(candidate_a_underlying)
            )
            mapping[public] = {
                "candidate": candidate,
                "underlying_position": underlying,
                "variant_id": str(session[f"{underlying}_variant_id"]),
                "job_id": str(session[f"{underlying}_job_id"]),
                "artifact_id": str(session[f"{underlying}_artifact_id"]),
                "artifact_sha256": str(session[f"{underlying}_artifact_sha256"]),
            }
        return mapping

    def _session(self, session_id: str) -> sqlite3.Row:
        with self.campaigns.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise CharacterBenchmarkConflict("character benchmark review session is missing")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _session_artifacts(session: sqlite3.Row) -> frozenset[tuple[str, str]]:
        return frozenset(
            (str(session[f"{position}_artifact_id"]), str(session[f"{position}_artifact_sha256"]))
            for position in ("left", "right")
        )

    def _trial(self, campaign_id: str, trial_id: str) -> CharacterTrial:
        try:
            return next(
                item for item in self.campaigns.list_trials(campaign_id) if item.id == trial_id
            )
        except StopIteration as error:
            raise CharacterBenchmarkConflict("character benchmark trial is missing") from error

    @staticmethod
    def _requires_secondary(trial: CharacterTrial, choice: IdentityChoice) -> bool:
        return not trial.hidden_repeat and choice is not IdentityChoice.NEITHER

    def _view(self, campaign: CharacterBenchmarkCampaign) -> CharacterBenchmarkView:
        generated = (
            len(self.campaigns.list_experiment_bindings(campaign.definition.id))
            if campaign.state is not CharacterBenchmarkState.DRAFT
            else 0
        )
        completed = (
            len(self.campaigns.list_identity(campaign.definition.id))
            if campaign.state is not CharacterBenchmarkState.DRAFT
            else 0
        )
        next_action = {
            CharacterBenchmarkState.DRAFT: "preregister",
            CharacterBenchmarkState.PREREGISTERED: "complete_generation",
            CharacterBenchmarkState.GENERATING: "complete_generation",
            CharacterBenchmarkState.READY: "record_identity",
            CharacterBenchmarkState.JUDGING: "record_identity",
            CharacterBenchmarkState.ANALYZING: "analyze",
            CharacterBenchmarkState.COMPLETED: "none",
        }[campaign.state]
        return CharacterBenchmarkView(
            campaign_id=campaign.definition.id,
            state=campaign.state,
            next_action=next_action,
            progress=CharacterBenchmarkProgress(
                completed_identity=completed,
                generated_primary=generated,
            ),
            result=self.campaigns.get_result(campaign.definition.id),
        )

    def _event(
        self,
        campaign_id: str,
        sequence: int,
        previous: CharacterBenchmarkState,
        new: CharacterBenchmarkState,
        domain: str,
    ) -> CharacterBenchmarkEvent:
        return CharacterBenchmarkEvent(
            campaign_id=campaign_id,
            sequence=sequence,
            previous_state=previous,
            new_state=new,
            payload_digest=_digest(
                {"domain": domain, "campaign_id": campaign_id, "sequence": sequence}
            ),
            created_at=self.clock(),
        )


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda item: item.model_dump(mode="json"),
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _recipe_pair(identity_prompt: str) -> CharacterRecipePair:
    return CharacterRecipePair(
        control_digest=_digest(
            {"schema_version": "character-prompt-recipe.v1", "template": "{context}"}
        ),
        identity_digest=_digest(
            {
                "schema_version": "character-prompt-recipe.v1",
                "template": "{context} {identity_prompt}",
                "identity_prompt": identity_prompt,
            }
        ),
    )


def _reviewer_alias(campaign_id: str, trial_id: str, reviewer_id: str) -> str:
    return character_benchmark_reviewer_id(campaign_id, trial_id, reviewer_id)


def _record_id(kind: str, campaign_id: str, identity: str) -> str:
    payload = {
        "domain": f"video-lab.character-benchmark.{kind}",
        "campaign_id": campaign_id,
        "identity": identity,
    }
    return f"character_{kind}_{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"


def _display_choice(
    choice: IdentityChoice, candidate_a_position: Literal["left", "right"]
) -> Literal["left", "right", "both", "neither"]:
    if choice is IdentityChoice.A:
        return candidate_a_position
    if choice is IdentityChoice.B:
        return _opposite(candidate_a_position)
    return cast(Literal["both", "neither"], choice.value)


def _underlying_preference(
    display_choice: str, mapping: dict[str, dict[str, str]]
) -> DisplayPreference:
    if display_choice in {"both", "neither"}:
        return DisplayPreference.TIE
    underlying = mapping[display_choice]["underlying_position"]
    return DisplayPreference(underlying)


def _opposite(value: str) -> Literal["left", "right"]:
    return "right" if value == "left" else "left"
