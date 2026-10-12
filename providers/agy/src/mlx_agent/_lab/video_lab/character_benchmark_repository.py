from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Literal, cast

from pydantic import field_validator

from video_lab.character_benchmark_models import (
    CharacterBenchmarkDefinition,
    CharacterBenchmarkResult,
    CharacterTrial,
    IdentityObservation,
    SecondaryObservation,
)
from video_lab.comparison import ControlledPair
from video_lab.database import Database, character_benchmark_reviewer_id
from video_lab.domain import FrozenModel, JobState, Manifest, _aware_utc, canonical_json


class CharacterBenchmarkNotFound(LookupError):
    pass


class CharacterBenchmarkConflict(RuntimeError):
    pass


class CharacterBenchmarkState(StrEnum):
    DRAFT = "draft"
    PREREGISTERED = "preregistered"
    GENERATING = "generating"
    READY = "ready"
    JUDGING = "judging"
    ANALYZING = "analyzing"
    COMPLETED = "completed"


class CharacterBenchmarkEvent(FrozenModel):
    campaign_id: str
    sequence: int
    previous_state: CharacterBenchmarkState
    new_state: CharacterBenchmarkState
    payload_digest: str
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class CharacterBenchmarkCampaign(FrozenModel):
    definition: CharacterBenchmarkDefinition
    reviewer_id: str
    character_pack_id: str
    character_pack_storage_id: str
    stage_2_campaign_id: str
    stage_2_manifest: Manifest
    identity_prompt: str
    content_digest: str
    created_at: datetime
    state: CharacterBenchmarkState

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class CharacterIdentityJudgment(FrozenModel):
    observation: IdentityObservation
    review_id: str
    display_choice: Literal["left", "right", "both", "neither"]
    content_digest: str
    created_at: datetime


class CharacterSecondaryJudgment(FrozenModel):
    observation: SecondaryObservation
    review_id: str
    content_digest: str
    created_at: datetime


class CharacterExperimentBinding(FrozenModel):
    campaign_id: str
    trial_id: str
    experiment_id: str


class CharacterReviewBinding(FrozenModel):
    campaign_id: str
    trial_id: str
    experiment_id: str
    review_id: str
    session_id: str
    reviewer_alias: str
    reused_from_trial_id: str | None = None
    display_mapping: dict[str, dict[str, str]]


class CharacterBenchmarkRepository:
    """Owner-bound append-only storage for character benchmark evidence."""

    def __init__(self, database: Database, *, reviewer_id: str) -> None:
        normalized = unicodedata.normalize("NFKC", reviewer_id).strip()
        if not normalized or normalized != reviewer_id:
            raise ValueError("reviewer_id must be normalized non-empty text")
        self.database = database
        self.reviewer_id = normalized

    def create_or_get_bound(
        self,
        definition: CharacterBenchmarkDefinition,
        *,
        character_pack_id: str,
        character_pack_storage_id: str,
        stage_2_campaign_id: str,
        stage_2_manifest: Manifest,
        identity_prompt: str,
        content_digest: str,
        created_at: datetime,
    ) -> CharacterBenchmarkCampaign:
        validated = CharacterBenchmarkDefinition.model_validate(definition.model_dump())
        manifest = Manifest.model_validate(stage_2_manifest.model_dump())
        prompt = unicodedata.normalize("NFKC", identity_prompt).strip()
        if not prompt or prompt != identity_prompt:
            raise ValueError("identity prompt must be normalized non-empty text")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM character_benchmark_campaigns WHERE id=?", (validated.id,)
                ).fetchone()
                by_digest = connection.execute(
                    "SELECT id FROM character_benchmark_campaigns WHERE content_digest=?",
                    (content_digest,),
                ).fetchone()
                if row is None and by_digest is not None:
                    raise CharacterBenchmarkConflict(
                        "character benchmark content digest already belongs to another campaign"
                    )
                if row is None:
                    self._require_dependency_owners(
                        connection, character_pack_storage_id, stage_2_campaign_id
                    )
                    connection.execute(
                        "INSERT INTO character_benchmark_campaigns("
                        "id, reviewer_id, content_digest, definition_json, character_pack_id, "
                        "character_pack_storage_id, stage_2_campaign_id, stage_2_manifest_json, "
                        "identity_prompt, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            validated.id,
                            self.reviewer_id,
                            content_digest,
                            canonical_json(validated),
                            character_pack_id,
                            character_pack_storage_id,
                            stage_2_campaign_id,
                            canonical_json(manifest),
                            prompt,
                            _aware_utc(created_at).isoformat(),
                        ),
                    )
                    row = connection.execute(
                        "SELECT * FROM character_benchmark_campaigns WHERE id=?", (validated.id,)
                    ).fetchone()
                assert row is not None
                persisted = self._campaign_from_row(connection, row)
                expected = CharacterBenchmarkCampaign(
                    definition=validated,
                    reviewer_id=self.reviewer_id,
                    character_pack_id=character_pack_id,
                    character_pack_storage_id=character_pack_storage_id,
                    stage_2_campaign_id=stage_2_campaign_id,
                    stage_2_manifest=manifest,
                    identity_prompt=prompt,
                    content_digest=content_digest,
                    created_at=created_at,
                    state=persisted.state,
                )
                if persisted != expected:
                    if persisted.reviewer_id != self.reviewer_id:
                        raise CharacterBenchmarkConflict(
                            "character benchmark belongs to another reviewer"
                        )
                    raise CharacterBenchmarkConflict(
                        "character benchmark retry does not match immutable content"
                    )
                connection.commit()
                return persisted
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise

    def get(self, campaign_id: str) -> CharacterBenchmarkCampaign:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM character_benchmark_campaigns WHERE id=?", (campaign_id,)
            ).fetchone()
            if row is None:
                raise CharacterBenchmarkNotFound("character benchmark not found")
            campaign = self._campaign_from_row(connection, row)
        if campaign.reviewer_id != self.reviewer_id:
            raise CharacterBenchmarkConflict("character benchmark belongs to another reviewer")
        return campaign

    def preregister_or_verify(
        self,
        campaign_id: str,
        trials: tuple[CharacterTrial, ...],
        *,
        created_at: datetime,
    ) -> CharacterBenchmarkEvent:
        validated = tuple(CharacterTrial.model_validate(item.model_dump()) for item in trials)
        payload_digest = _digest([item.model_dump(mode="json") for item in validated])
        event = CharacterBenchmarkEvent(
            campaign_id=campaign_id,
            sequence=1,
            previous_state=CharacterBenchmarkState.DRAFT,
            new_state=CharacterBenchmarkState.PREREGISTERED,
            payload_digest=payload_digest,
            created_at=created_at,
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_owned(connection, campaign_id)
                rows = connection.execute(
                    "SELECT trial_json, expected_reviewer_id "
                    "FROM character_benchmark_trials WHERE campaign_id=? "
                    "ORDER BY ordinal",
                    (campaign_id,),
                ).fetchall()
                if rows:
                    persisted = tuple(
                        CharacterTrial.model_validate_json(str(row["trial_json"])) for row in rows
                    )
                    expected_reviewers = tuple(
                        character_benchmark_reviewer_id(campaign_id, trial.id, self.reviewer_id)
                        for trial in validated
                    )
                    existing = self._event_at(connection, campaign_id, 1)
                    if (
                        persisted != validated
                        or tuple(str(row["expected_reviewer_id"]) for row in rows)
                        != expected_reviewers
                        or existing != event
                    ):
                        raise CharacterBenchmarkConflict(
                            "character benchmark preregistration retry does not match evidence"
                        )
                    connection.commit()
                    return existing
                self._insert_event(connection, event)
                connection.executemany(
                    "INSERT INTO character_benchmark_trials("
                    "campaign_id, trial_id, ordinal, trial_json, hidden_repeat, "
                    "reuses_trial_id, candidate_a_position, expected_reviewer_id) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    [
                        (
                            campaign_id,
                            trial.id,
                            trial.ordinal,
                            canonical_json(trial),
                            int(trial.hidden_repeat),
                            trial.reuses_trial_id,
                            trial.candidate_a_position,
                            character_benchmark_reviewer_id(
                                campaign_id, trial.id, self.reviewer_id
                            ),
                        )
                        for trial in validated
                    ],
                )
                connection.commit()
                return event
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(
                    "character benchmark preregistration violates persistence"
                ) from error
            except Exception:
                connection.rollback()
                raise

    def append_event_or_verify(self, event: CharacterBenchmarkEvent) -> CharacterBenchmarkEvent:
        validated = CharacterBenchmarkEvent.model_validate(event.model_dump())
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_owned(connection, validated.campaign_id)
                existing = self._event_at(
                    connection, validated.campaign_id, validated.sequence, required=False
                )
                if existing is not None:
                    if existing != validated:
                        raise CharacterBenchmarkConflict(
                            "character benchmark event retry does not match evidence"
                        )
                    connection.commit()
                    return existing
                self._insert_event(connection, validated)
                connection.commit()
                return validated
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise

    def list_trials(self, campaign_id: str) -> tuple[CharacterTrial, ...]:
        with self.database.connect() as connection:
            self._require_owned(connection, campaign_id)
            rows = connection.execute(
                "SELECT trial_json FROM character_benchmark_trials WHERE campaign_id=? "
                "ORDER BY ordinal",
                (campaign_id,),
            ).fetchall()
        return tuple(CharacterTrial.model_validate_json(str(row["trial_json"])) for row in rows)

    def list_events(self, campaign_id: str) -> tuple[CharacterBenchmarkEvent, ...]:
        with self.database.connect() as connection:
            self._require_owned(connection, campaign_id)
            rows = connection.execute(
                "SELECT * FROM character_benchmark_events WHERE campaign_id=? ORDER BY sequence",
                (campaign_id,),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def bind_experiment_or_verify(
        self,
        campaign_id: str,
        trial: CharacterTrial,
        pair: ControlledPair,
        *,
        created_at: datetime,
    ) -> CharacterExperimentBinding:
        if trial.hidden_repeat or len(trial.generation_specs) != 2:
            raise CharacterBenchmarkConflict("only primary character trials create experiments")
        experiment_id = _record_id("exp", campaign_id, trial.id)
        variant_ids = tuple(spec.id for spec in trial.generation_specs)
        job_ids = tuple(_record_id("job", campaign_id, spec.id) for spec in trial.generation_specs)
        binding = CharacterExperimentBinding(
            campaign_id=campaign_id,
            trial_id=trial.id,
            experiment_id=experiment_id,
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_owned(connection, campaign_id)
                existing = connection.execute(
                    "SELECT * FROM character_benchmark_experiment_bindings "
                    "WHERE campaign_id=? AND trial_id=?",
                    (campaign_id, trial.id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO experiments(id, schema_version, variable_path, "
                        "changed_paths_json, created_at) VALUES (?,?,?,?,?)",
                        (
                            experiment_id,
                            "character-controlled.v1",
                            pair.variable_path,
                            json.dumps(
                                list(pair.changed_paths),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            _aware_utc(created_at).isoformat(),
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO variants(id, experiment_id, label, manifest_json) "
                        "VALUES (?,?,?,?)",
                        [
                            (variant_ids[0], experiment_id, "A", canonical_json(pair.variant_a)),
                            (variant_ids[1], experiment_id, "B", canonical_json(pair.variant_b)),
                        ],
                    )
                    connection.executemany(
                        "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
                        "VALUES (?,?,?,?,?)",
                        [
                            (
                                job_id,
                                variant_id,
                                JobState.QUEUED.value,
                                _aware_utc(created_at).isoformat(),
                                _aware_utc(created_at).isoformat(),
                            )
                            for job_id, variant_id in zip(job_ids, variant_ids, strict=True)
                        ],
                    )
                    connection.execute(
                        "INSERT INTO character_benchmark_experiment_bindings VALUES (?,?,?)",
                        (campaign_id, trial.id, experiment_id),
                    )
                else:
                    persisted = _experiment_binding_from_row(existing)
                    if persisted != binding:
                        raise CharacterBenchmarkConflict(
                            "character benchmark experiment retry does not match evidence"
                        )
                    experiment = connection.execute(
                        "SELECT * FROM experiments WHERE id=?", (experiment_id,)
                    ).fetchone()
                    variants = connection.execute(
                        "SELECT * FROM variants WHERE experiment_id=? ORDER BY label",
                        (experiment_id,),
                    ).fetchall()
                    jobs = connection.execute(
                        "SELECT job.id, job.variant_id FROM jobs AS job JOIN variants AS variant "
                        "ON variant.id=job.variant_id WHERE variant.experiment_id=? "
                        "ORDER BY variant.label",
                        (experiment_id,),
                    ).fetchall()
                    if (
                        experiment is None
                        or str(experiment["schema_version"]) != "character-controlled.v1"
                        or str(experiment["variable_path"]) != pair.variable_path
                        or tuple(json.loads(str(experiment["changed_paths_json"])))
                        != pair.changed_paths
                        or len(variants) != 2
                        or tuple(str(row["id"]) for row in variants) != variant_ids
                        or tuple(
                            Manifest.model_validate_json(str(row["manifest_json"]))
                            for row in variants
                        )
                        != (pair.variant_a, pair.variant_b)
                        or tuple(str(row["id"]) for row in jobs) != job_ids
                        or tuple(str(row["variant_id"]) for row in jobs) != variant_ids
                    ):
                        raise CharacterBenchmarkConflict(
                            "character benchmark experiment retry does not match evidence"
                        )
                connection.commit()
                return binding
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(
                    "character benchmark experiment violates persistence"
                ) from error
            except Exception:
                connection.rollback()
                raise

    def list_experiment_bindings(self, campaign_id: str) -> tuple[CharacterExperimentBinding, ...]:
        with self.database.connect() as connection:
            self._require_owned(connection, campaign_id)
            rows = connection.execute(
                "SELECT binding.* FROM character_benchmark_experiment_bindings AS binding "
                "JOIN character_benchmark_trials AS trial ON trial.campaign_id=binding.campaign_id "
                "AND trial.trial_id=binding.trial_id WHERE binding.campaign_id=? "
                "ORDER BY trial.ordinal",
                (campaign_id,),
            ).fetchall()
        return tuple(_experiment_binding_from_row(row) for row in rows)

    def bind_review_or_verify(self, binding: CharacterReviewBinding) -> CharacterReviewBinding:
        validated = CharacterReviewBinding.model_validate(binding.model_dump())
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_owned(connection, validated.campaign_id)
                row = connection.execute(
                    "SELECT * FROM character_benchmark_review_bindings "
                    "WHERE campaign_id=? AND trial_id=?",
                    (validated.campaign_id, validated.trial_id),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO character_benchmark_review_bindings VALUES (?,?,?,?,?,?,?,?)",
                        (
                            validated.campaign_id,
                            validated.trial_id,
                            validated.experiment_id,
                            validated.review_id,
                            validated.session_id,
                            validated.reviewer_alias,
                            validated.reused_from_trial_id,
                            canonical_json(validated.display_mapping),
                        ),
                    )
                    persisted = validated
                else:
                    persisted = _review_binding_from_row(row)
                    if persisted != validated:
                        raise CharacterBenchmarkConflict(
                            "character benchmark review retry does not match evidence"
                        )
                connection.commit()
                return persisted
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(
                    "character benchmark review violates persistence"
                ) from error
            except Exception:
                connection.rollback()
                raise

    def list_review_bindings(self, campaign_id: str) -> tuple[CharacterReviewBinding, ...]:
        with self.database.connect() as connection:
            self._require_owned(connection, campaign_id)
            rows = connection.execute(
                "SELECT binding.* FROM character_benchmark_review_bindings AS binding "
                "JOIN character_benchmark_trials AS trial ON trial.campaign_id=binding.campaign_id "
                "AND trial.trial_id=binding.trial_id WHERE binding.campaign_id=? "
                "ORDER BY trial.ordinal",
                (campaign_id,),
            ).fetchall()
        return tuple(_review_binding_from_row(row) for row in rows)

    def record_identity_or_verify(
        self,
        observation: IdentityObservation,
        *,
        review_id: str,
        display_choice: Literal["left", "right", "both", "neither"],
        created_at: datetime,
    ) -> CharacterIdentityJudgment:
        validated = IdentityObservation.model_validate(observation.model_dump())
        payload = {
            "observation": validated.model_dump(mode="json"),
            "review_id": review_id,
            "display_choice": display_choice,
        }
        judgment = CharacterIdentityJudgment(
            observation=validated,
            review_id=review_id,
            display_choice=display_choice,
            content_digest=_digest(payload),
            created_at=_aware_utc(created_at),
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_owned(connection, validated.benchmark_id)
                row = connection.execute(
                    "SELECT * FROM character_benchmark_identity_judgments "
                    "WHERE campaign_id=? AND trial_id=?",
                    (validated.benchmark_id, validated.trial_id),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO character_benchmark_identity_judgments VALUES "
                        "(?,?,?,?,?,?,?,?)",
                        (
                            validated.benchmark_id,
                            validated.trial_id,
                            review_id,
                            validated.choice.value,
                            display_choice,
                            canonical_json(validated),
                            judgment.content_digest,
                            judgment.created_at.isoformat(),
                        ),
                    )
                    persisted = judgment
                else:
                    persisted = _identity_from_row(row)
                    if persisted != judgment:
                        raise CharacterBenchmarkConflict(
                            "character benchmark identity retry does not match evidence"
                        )
                connection.commit()
                return persisted
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise

    def record_secondary_or_verify(
        self, observation: SecondaryObservation, *, created_at: datetime
    ) -> CharacterSecondaryJudgment:
        validated = SecondaryObservation.model_validate(observation.model_dump())
        digest = _digest(validated.model_dump(mode="json"))
        normalized_created_at = _aware_utc(created_at)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_owned(connection, validated.benchmark_id)
                identity = connection.execute(
                    "SELECT review_id FROM character_benchmark_identity_judgments "
                    "WHERE campaign_id=? AND trial_id=?",
                    (validated.benchmark_id, validated.trial_id),
                ).fetchone()
                if identity is None:
                    raise CharacterBenchmarkConflict(
                        "character benchmark secondary requires identity evidence"
                    )
                review_id = str(identity["review_id"])
                judgment = CharacterSecondaryJudgment(
                    observation=validated,
                    review_id=review_id,
                    content_digest=digest,
                    created_at=normalized_created_at,
                )
                row = connection.execute(
                    "SELECT * FROM character_benchmark_secondary_judgments WHERE "
                    "campaign_id=? AND trial_id=? AND dimension=?",
                    (validated.benchmark_id, validated.trial_id, validated.dimension.value),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO character_benchmark_secondary_judgments("
                        "campaign_id,trial_id,dimension,choice,observation_json,"
                        "content_digest,created_at,review_id) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            validated.benchmark_id,
                            validated.trial_id,
                            validated.dimension.value,
                            validated.choice.value,
                            canonical_json(validated),
                            digest,
                            judgment.created_at.isoformat(),
                            review_id,
                        ),
                    )
                    persisted = judgment
                else:
                    persisted = _secondary_from_row(row)
                    if persisted != judgment:
                        raise CharacterBenchmarkConflict(
                            "character benchmark secondary retry does not match evidence"
                        )
                connection.commit()
                return persisted
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise

    def list_identity(self, campaign_id: str) -> tuple[CharacterIdentityJudgment, ...]:
        with self.database.connect() as connection:
            self._require_owned(connection, campaign_id)
            rows = connection.execute(
                "SELECT judgment.* FROM character_benchmark_identity_judgments AS judgment "
                "JOIN character_benchmark_trials AS trial "
                "ON trial.campaign_id=judgment.campaign_id "
                "AND trial.trial_id=judgment.trial_id WHERE judgment.campaign_id=? "
                "ORDER BY trial.ordinal",
                (campaign_id,),
            ).fetchall()
        return tuple(_identity_from_row(row) for row in rows)

    def list_secondary(self, campaign_id: str) -> tuple[CharacterSecondaryJudgment, ...]:
        with self.database.connect() as connection:
            self._require_owned(connection, campaign_id)
            rows = connection.execute(
                "SELECT judgment.* FROM character_benchmark_secondary_judgments AS judgment "
                "JOIN character_benchmark_trials AS trial "
                "ON trial.campaign_id=judgment.campaign_id "
                "AND trial.trial_id=judgment.trial_id WHERE judgment.campaign_id=? "
                "ORDER BY trial.ordinal, judgment.dimension",
                (campaign_id,),
            ).fetchall()
        return tuple(_secondary_from_row(row) for row in rows)

    def append_result_or_verify(
        self, result: CharacterBenchmarkResult, *, created_at: datetime
    ) -> CharacterBenchmarkResult:
        validated = CharacterBenchmarkResult.model_validate(result.model_dump())
        digest = _digest(validated.model_dump(mode="json"))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_owned(connection, validated.benchmark_id)
                row = connection.execute(
                    "SELECT * FROM character_benchmark_results WHERE campaign_id=?",
                    (validated.benchmark_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO character_benchmark_results VALUES (?,?,?,?)",
                        (
                            validated.benchmark_id,
                            canonical_json(validated),
                            digest,
                            _aware_utc(created_at).isoformat(),
                        ),
                    )
                    persisted = validated
                else:
                    persisted = CharacterBenchmarkResult.model_validate_json(
                        str(row["result_json"])
                    )
                    if persisted != validated or str(row["content_digest"]) != digest:
                        raise CharacterBenchmarkConflict(
                            "character benchmark result retry does not match evidence"
                        )
                connection.commit()
                return persisted
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise CharacterBenchmarkConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise

    def get_result(self, campaign_id: str) -> CharacterBenchmarkResult | None:
        with self.database.connect() as connection:
            self._require_owned(connection, campaign_id)
            row = connection.execute(
                "SELECT result_json FROM character_benchmark_results WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
        return (
            None
            if row is None
            else CharacterBenchmarkResult.model_validate_json(str(row["result_json"]))
        )

    def _require_dependency_owners(
        self, connection: sqlite3.Connection, pack_storage_id: str, stage_2_campaign_id: str
    ) -> None:
        pack = connection.execute(
            "SELECT reviewer_id, state FROM character_packs WHERE storage_id=?",
            (pack_storage_id,),
        ).fetchone()
        stage_2 = connection.execute(
            "SELECT reviewer_id FROM validation_campaigns WHERE id=?", (stage_2_campaign_id,)
        ).fetchone()
        if (
            pack is None
            or str(pack["reviewer_id"]) != self.reviewer_id
            or str(pack["state"]) != "preregistered"
            or stage_2 is None
            or str(stage_2["reviewer_id"]) != self.reviewer_id
        ):
            raise CharacterBenchmarkConflict(
                "character benchmark dependencies do not belong to reviewer"
            )

    def _require_owned(self, connection: sqlite3.Connection, campaign_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM character_benchmark_campaigns WHERE id=?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise CharacterBenchmarkNotFound("character benchmark not found")
        if str(row["reviewer_id"]) != self.reviewer_id:
            raise CharacterBenchmarkConflict("character benchmark belongs to another reviewer")
        return cast(sqlite3.Row, row)

    def _campaign_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> CharacterBenchmarkCampaign:
        latest = connection.execute(
            "SELECT new_state FROM character_benchmark_events WHERE campaign_id=? "
            "ORDER BY sequence DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        return CharacterBenchmarkCampaign(
            definition=CharacterBenchmarkDefinition.model_validate_json(
                str(row["definition_json"])
            ),
            reviewer_id=str(row["reviewer_id"]),
            character_pack_id=str(row["character_pack_id"]),
            character_pack_storage_id=str(row["character_pack_storage_id"]),
            stage_2_campaign_id=str(row["stage_2_campaign_id"]),
            stage_2_manifest=Manifest.model_validate_json(str(row["stage_2_manifest_json"])),
            identity_prompt=str(row["identity_prompt"]),
            content_digest=str(row["content_digest"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            state=CharacterBenchmarkState(str(latest["new_state"]) if latest else "draft"),
        )

    @staticmethod
    def _insert_event(connection: sqlite3.Connection, event: CharacterBenchmarkEvent) -> None:
        connection.execute(
            "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
            (
                event.campaign_id,
                event.sequence,
                event.previous_state.value,
                event.new_state.value,
                event.payload_digest,
                event.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _event_at(
        connection: sqlite3.Connection,
        campaign_id: str,
        sequence: int,
        *,
        required: bool = True,
    ) -> CharacterBenchmarkEvent | None:
        row = connection.execute(
            "SELECT * FROM character_benchmark_events WHERE campaign_id=? AND sequence=?",
            (campaign_id, sequence),
        ).fetchone()
        if row is None:
            if required:
                raise CharacterBenchmarkConflict("character benchmark event is missing")
            return None
        return _event_from_row(row)


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda item: item.model_dump(mode="json"),
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _event_from_row(row: sqlite3.Row) -> CharacterBenchmarkEvent:
    return CharacterBenchmarkEvent(
        campaign_id=str(row["campaign_id"]),
        sequence=int(row["sequence"]),
        previous_state=CharacterBenchmarkState(str(row["previous_state"])),
        new_state=CharacterBenchmarkState(str(row["new_state"])),
        payload_digest=str(row["payload_digest"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _identity_from_row(row: sqlite3.Row) -> CharacterIdentityJudgment:
    display_choice = str(row["display_choice"])
    if display_choice not in {"left", "right", "both", "neither"}:
        raise CharacterBenchmarkConflict("character benchmark display choice is invalid")
    observation = IdentityObservation.model_validate_json(str(row["observation_json"]))
    expected_digest = _digest(
        {
            "observation": observation.model_dump(mode="json"),
            "review_id": str(row["review_id"]),
            "display_choice": display_choice,
        }
    )
    if (
        str(row["campaign_id"]) != observation.benchmark_id
        or str(row["trial_id"]) != observation.trial_id
        or str(row["choice"]) != observation.choice.value
        or str(row["content_digest"]) != expected_digest
    ):
        raise CharacterBenchmarkConflict(
            "character benchmark identity judgment content is inconsistent"
        )
    return CharacterIdentityJudgment(
        observation=observation,
        review_id=str(row["review_id"]),
        display_choice=cast(Literal["left", "right", "both", "neither"], display_choice),
        content_digest=str(row["content_digest"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _secondary_from_row(row: sqlite3.Row) -> CharacterSecondaryJudgment:
    observation = SecondaryObservation.model_validate_json(str(row["observation_json"]))
    expected_digest = _digest(observation.model_dump(mode="json"))
    if (
        str(row["campaign_id"]) != observation.benchmark_id
        or str(row["trial_id"]) != observation.trial_id
        or str(row["dimension"]) != observation.dimension.value
        or str(row["choice"]) != observation.choice.value
        or str(row["content_digest"]) != expected_digest
    ):
        raise CharacterBenchmarkConflict(
            "character benchmark secondary judgment content is inconsistent"
        )
    return CharacterSecondaryJudgment(
        observation=observation,
        review_id=str(row["review_id"]),
        content_digest=str(row["content_digest"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _record_id(kind: str, campaign_id: str, identity: str) -> str:
    payload = {
        "domain": f"video-lab.character-benchmark.{kind}",
        "campaign_id": campaign_id,
        "identity": identity,
    }
    return f"character_{kind}_{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"


def _experiment_binding_from_row(row: sqlite3.Row) -> CharacterExperimentBinding:
    return CharacterExperimentBinding(
        campaign_id=str(row["campaign_id"]),
        trial_id=str(row["trial_id"]),
        experiment_id=str(row["experiment_id"]),
    )


def _review_binding_from_row(row: sqlite3.Row) -> CharacterReviewBinding:
    return CharacterReviewBinding(
        campaign_id=str(row["campaign_id"]),
        trial_id=str(row["trial_id"]),
        experiment_id=str(row["experiment_id"]),
        review_id=str(row["review_id"]),
        session_id=str(row["session_id"]),
        reviewer_alias=str(row["reviewer_alias"]),
        reused_from_trial_id=(
            None if row["reused_from_trial_id"] is None else str(row["reused_from_trial_id"])
        ),
        display_mapping=json.loads(str(row["display_mapping_json"])),
    )
