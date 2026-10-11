from __future__ import annotations

# ruff: noqa: E501
import json
import sqlite3
from typing import cast

from video_lab.database import Database
from video_lab.domain import canonical_json
from video_lab.validation_models import (
    ValidationCampaignDefinition,
    ValidationCampaignState,
    ValidationEvent,
    ValidationExperimentBinding,
    ValidationJudgment,
    ValidationPanel,
    ValidationResult,
    ValidationTrial,
    ValidationTrialBinding,
    validation_panel_digest,
)


class ValidationNotFound(LookupError):
    pass


class ValidationConflict(RuntimeError):
    pass


class ValidationRepository:
    """Append-only repository for the fixed balanced-validation protocol."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_or_get_bound(
        self, definition: ValidationCampaignDefinition, *, reviewer_id: str
    ) -> ValidationCampaignDefinition:
        validated = ValidationCampaignDefinition.model_validate(
            definition.model_dump(round_trip=True)
        )
        _require_reviewer(reviewer_id)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM validation_campaigns WHERE id = ?", (validated.id,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO validation_campaigns("
                        "id, schema_version, reviewer_id, stage_1_study_id, "
                        "stage_1_evidence_digest, base_manifest_json, candidates_json, "
                        "panel_input_json, protocol_version, analysis_version, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            validated.id,
                            validated.schema_version,
                            reviewer_id,
                            validated.stage_1_study_id,
                            validated.stage_1_evidence_digest,
                            canonical_json(validated.base_manifest),
                            canonical_json(validated.candidates),
                            canonical_json(validated.panel_input),
                            validated.protocol_version,
                            validated.analysis_version,
                            validated.created_at.isoformat(),
                        ),
                    )
                    persisted = validated
                else:
                    persisted = _definition_from_row(row)
                    if not _same_campaign_definition(persisted, validated):
                        raise ValidationConflict(
                            "validation campaign id already has a different immutable definition"
                        )
                    if str(row["reviewer_id"]) != reviewer_id:
                        raise ValidationConflict("validation campaign belongs to another reviewer")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return persisted

    def bind_experiment_or_verify(
        self, binding: ValidationExperimentBinding
    ) -> ValidationExperimentBinding:
        validated = ValidationExperimentBinding.model_validate(binding.model_dump())
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_campaign(connection, validated.campaign_id)
                existing = connection.execute(
                    "SELECT * FROM validation_experiment_bindings "
                    "WHERE campaign_id = ? AND trial_id = ?",
                    (validated.campaign_id, validated.trial_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO validation_experiment_bindings("
                        "campaign_id, trial_id, experiment_id) VALUES (?, ?, ?)",
                        (
                            validated.campaign_id,
                            validated.trial_id,
                            validated.experiment_id,
                        ),
                    )
                    persisted = validated
                else:
                    persisted = _experiment_binding_from_row(existing)
                    if persisted != validated:
                        raise ValidationConflict(
                            "validation experiment binding retry does not match evidence"
                        )
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValidationConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise
        return persisted

    def list_experiment_bindings(self, campaign_id: str) -> tuple[ValidationExperimentBinding, ...]:
        with self.database.connect() as connection:
            self._require_campaign(connection, campaign_id)
            rows = connection.execute(
                "SELECT binding.* FROM validation_experiment_bindings AS binding "
                "JOIN validation_trials AS trial "
                "ON trial.campaign_id=binding.campaign_id AND trial.trial_id=binding.trial_id "
                "WHERE binding.campaign_id=? ORDER BY trial.ordinal",
                (campaign_id,),
            ).fetchall()
        return tuple(_experiment_binding_from_row(row) for row in rows)

    def preregister_or_verify(self, campaign_id: str, panel: ValidationPanel) -> ValidationEvent:
        validated = ValidationPanel.model_validate(panel.model_dump(round_trip=True))
        if validated.campaign_id != campaign_id:
            raise ValidationConflict("validation panel belongs to another campaign")
        if validation_panel_digest(campaign_id, validated.trials) != validated.content_digest:
            raise ValidationConflict("validation panel digest does not match frozen trials")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                campaign = self._require_campaign(connection, campaign_id)
                existing = connection.execute(
                    "SELECT * FROM validation_events WHERE campaign_id = ? ORDER BY sequence",
                    (campaign_id,),
                ).fetchall()
                if not existing:
                    event = ValidationEvent(
                        campaign_id=campaign_id,
                        sequence=1,
                        previous_state=ValidationCampaignState.DRAFT,
                        new_state=ValidationCampaignState.PREREGISTERED,
                        created_at=_definition_from_row(campaign).created_at,
                    )
                    self._insert_event(connection, event)
                    self._insert_panel(connection, validated)
                    persisted = event
                else:
                    persisted = _event_from_row(existing[0])
                    if (
                        persisted.sequence != 1
                        or persisted.previous_state is not ValidationCampaignState.DRAFT
                        or persisted.new_state is not ValidationCampaignState.PREREGISTERED
                        or self._panel_from_rows(connection, campaign_id) != validated
                    ):
                        raise ValidationConflict(
                            "validation preregistration does not match evidence"
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return persisted

    def append_event_or_verify(self, event: ValidationEvent) -> ValidationEvent:
        validated = ValidationEvent.model_validate(event.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_campaign(connection, validated.campaign_id)
                existing = connection.execute(
                    "SELECT * FROM validation_events WHERE campaign_id = ? AND sequence = ?",
                    (validated.campaign_id, validated.sequence),
                ).fetchone()
                if existing is not None:
                    persisted = _event_from_row(existing)
                    if not _same_event(persisted, validated):
                        raise ValidationConflict(
                            "validation event retry does not match persisted evidence"
                        )
                else:
                    self._insert_event(connection, validated)
                    persisted = validated
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValidationConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise
        return persisted

    def bind_trial_or_verify(self, trial: ValidationTrialBinding) -> ValidationTrialBinding:
        validated = ValidationTrialBinding.model_validate(trial)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_campaign(connection, validated.campaign_id)
                existing = connection.execute(
                    "SELECT * FROM validation_trial_bindings WHERE campaign_id = ? AND trial_id = ?",
                    (validated.campaign_id, validated.trial_id),
                ).fetchone()
                if existing is None:
                    trial_row = connection.execute(
                        "SELECT reuses_trial_id FROM validation_trials "
                        "WHERE campaign_id = ? AND trial_id = ?",
                        (validated.campaign_id, validated.trial_id),
                    ).fetchone()
                    if trial_row is None:
                        raise ValidationConflict("validation trial does not belong to campaign")
                    if trial_row["reuses_trial_id"] != validated.reused_from_trial_id:
                        raise ValidationConflict(
                            "validation trial binding has mismatched repeat provenance"
                        )
                    connection.execute(
                        "INSERT INTO validation_trial_bindings("
                        "campaign_id, trial_id, experiment_id, review_id, reused_from_trial_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            validated.campaign_id,
                            validated.trial_id,
                            validated.experiment_id,
                            validated.review_id,
                            validated.reused_from_trial_id,
                        ),
                    )
                    persisted = validated
                else:
                    persisted = _binding_from_row(existing)
                    if persisted != validated:
                        raise ValidationConflict(
                            "validation trial binding retry does not match evidence"
                        )
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValidationConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise
        return persisted

    def consume_judgment_or_verify(
        self, judgment: ValidationJudgment
    ) -> tuple[ValidationJudgment, bool]:
        validated = ValidationJudgment.model_validate(judgment.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_campaign(connection, validated.campaign_id)
                existing = connection.execute(
                    "SELECT * FROM validation_judgments WHERE campaign_id = ? AND trial_id = ?",
                    (validated.campaign_id, validated.trial_id),
                ).fetchone()
                if existing is not None:
                    persisted = _judgment_from_row(existing)
                    if not _same_judgment(persisted, validated):
                        raise ValidationConflict(
                            "validation judgment retry does not match persisted choice"
                        )
                    consumed = False
                else:
                    connection.execute(
                        "INSERT INTO validation_judgments("
                        "campaign_id, trial_id, ordinal, archetype, challenger, seed, hidden_repeat, "
                        "winner_display_position, choice, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            validated.campaign_id,
                            validated.trial_id,
                            validated.ordinal,
                            validated.archetype.value,
                            validated.challenger,
                            validated.seed,
                            int(validated.hidden_repeat),
                            validated.winner_display_position,
                            validated.choice.value,
                            validated.created_at.isoformat(),
                        ),
                    )
                    persisted = validated
                    consumed = True
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValidationConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise
        return persisted, consumed

    def append_result_or_verify(self, result: ValidationResult) -> ValidationResult:
        validated = ValidationResult.model_validate(result.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_campaign(connection, validated.campaign_id)
                existing = connection.execute(
                    "SELECT * FROM validation_results WHERE campaign_id = ?",
                    (validated.campaign_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO validation_results("
                        "campaign_id, disposition, reason, challengers_json, archetype_reversals_json, "
                        "repeat_agreements, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            validated.campaign_id,
                            validated.disposition.value,
                            validated.reason.value,
                            json.dumps(
                                [item.model_dump(mode="json") for item in validated.challengers],
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            json.dumps(
                                [
                                    item.model_dump(mode="json")
                                    for item in validated.archetype_reversals
                                ],
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            validated.repeat_agreements,
                            self._latest_event_created_at(connection, validated.campaign_id),
                        ),
                    )
                    persisted = validated
                else:
                    persisted = _result_from_row(existing)
                    if persisted != validated:
                        raise ValidationConflict("validation result retry does not match evidence")
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValidationConflict(str(error)) from error
            except Exception:
                connection.rollback()
                raise
        return persisted

    def get(self, campaign_id: str) -> ValidationCampaignDefinition:
        with self.database.connect() as connection:
            row = self._require_campaign(connection, campaign_id)
        return _definition_from_row(row)

    def get_bound(self, campaign_id: str, *, reviewer_id: str) -> ValidationCampaignDefinition:
        _require_reviewer(reviewer_id)
        with self.database.connect() as connection:
            row = self._require_campaign(connection, campaign_id)
            if str(row["reviewer_id"]) != reviewer_id:
                raise ValidationNotFound(f"validation campaign not found: {campaign_id}")
        return _definition_from_row(row)

    def list_bound(
        self, *, reviewer_id: str, limit: int = 50
    ) -> tuple[ValidationCampaignDefinition, ...]:
        _require_reviewer(reviewer_id)
        if not 1 <= limit <= 100:
            raise ValueError("validation campaign list limit must be between 1 and 100")
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM validation_campaigns WHERE reviewer_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (reviewer_id, limit),
            ).fetchall()
        return tuple(_definition_from_row(row) for row in rows)

    def get_panel(self, campaign_id: str) -> ValidationPanel:
        with self.database.connect() as connection:
            self._require_campaign(connection, campaign_id)
            return self._panel_from_rows(connection, campaign_id)

    def state(self, campaign_id: str) -> ValidationCampaignState:
        events = self.list_events(campaign_id)
        return ValidationCampaignState.DRAFT if not events else events[-1].new_state

    def list_events(self, campaign_id: str) -> tuple[ValidationEvent, ...]:
        with self.database.connect() as connection:
            self._require_campaign(connection, campaign_id)
            rows = connection.execute(
                "SELECT * FROM validation_events WHERE campaign_id = ? ORDER BY sequence",
                (campaign_id,),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def list_trials(self, campaign_id: str) -> tuple[ValidationTrialBinding, ...]:
        with self.database.connect() as connection:
            self._require_campaign(connection, campaign_id)
            rows = connection.execute(
                "SELECT trial.campaign_id, trial.trial_id, binding.experiment_id, binding.review_id, "
                "COALESCE(binding.reused_from_trial_id, trial.reuses_trial_id) AS reused_from_trial_id "
                "FROM validation_trials AS trial LEFT JOIN validation_trial_bindings AS binding "
                "ON binding.campaign_id = trial.campaign_id AND binding.trial_id = trial.trial_id "
                "WHERE trial.campaign_id = ? ORDER BY trial.ordinal",
                (campaign_id,),
            ).fetchall()
        return tuple(_binding_from_row(row) for row in rows)

    def list_judgments(self, campaign_id: str) -> tuple[ValidationJudgment, ...]:
        with self.database.connect() as connection:
            self._require_campaign(connection, campaign_id)
            rows = connection.execute(
                "SELECT * FROM validation_judgments WHERE campaign_id = ? ORDER BY ordinal",
                (campaign_id,),
            ).fetchall()
        return tuple(_judgment_from_row(row) for row in rows)

    def get_result(self, campaign_id: str) -> ValidationResult | None:
        with self.database.connect() as connection:
            self._require_campaign(connection, campaign_id)
            row = connection.execute(
                "SELECT * FROM validation_results WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
        return None if row is None else _result_from_row(row)

    @staticmethod
    def _require_campaign(connection: sqlite3.Connection, campaign_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM validation_campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise ValidationNotFound(f"validation campaign not found: {campaign_id}")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _latest_event_created_at(connection: sqlite3.Connection, campaign_id: str) -> str:
        row = connection.execute(
            "SELECT created_at FROM validation_events WHERE campaign_id = ? "
            "ORDER BY sequence DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        if row is None:
            raise ValidationConflict("validation result has no lifecycle evidence")
        return str(row["created_at"])

    @staticmethod
    def _insert_event(connection: sqlite3.Connection, event: ValidationEvent) -> None:
        connection.execute(
            "INSERT INTO validation_events("
            "campaign_id, sequence, previous_state, new_state, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                event.campaign_id,
                event.sequence,
                event.previous_state.value,
                event.new_state.value,
                event.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _insert_panel(connection: sqlite3.Connection, panel: ValidationPanel) -> None:
        primary = sorted(
            (trial for trial in panel.trials if not trial.hidden_repeat),
            key=lambda item: item.ordinal,
        )
        repeats = sorted(
            (trial for trial in panel.trials if trial.hidden_repeat), key=lambda item: item.ordinal
        )
        for trial in (*primary, *repeats):
            connection.execute(
                "INSERT INTO validation_trials("
                "campaign_id, trial_id, ordinal, archetype, prompt, negative_prompt, seed, challenger, "
                "hidden_repeat, reuses_trial_id, winner_display_position, panel_digest) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    panel.campaign_id,
                    trial.id,
                    trial.ordinal,
                    trial.archetype.value,
                    trial.prompt,
                    trial.negative_prompt or "",
                    trial.seed,
                    trial.challenger,
                    int(trial.hidden_repeat),
                    trial.reuses_trial_id,
                    trial.winner_display_position,
                    panel.content_digest,
                ),
            )

    @staticmethod
    def _panel_from_rows(connection: sqlite3.Connection, campaign_id: str) -> ValidationPanel:
        rows = connection.execute(
            "SELECT * FROM validation_trials WHERE campaign_id = ? ORDER BY ordinal", (campaign_id,)
        ).fetchall()
        if not rows:
            raise ValidationConflict("validation preregistration has no trial evidence")
        digests = {str(row["panel_digest"]) for row in rows}
        if len(digests) != 1:
            raise ValidationConflict("validation preregistration has inconsistent panel digests")
        return ValidationPanel(
            campaign_id=campaign_id,
            trials=tuple(_trial_from_row(row) for row in rows),
            content_digest=next(iter(digests)),
        )


def _require_reviewer(reviewer_id: str) -> None:
    if not reviewer_id or reviewer_id != reviewer_id.strip():
        raise ValidationConflict("validation campaign owner must be a normalized identity")


def _definition_from_row(row: sqlite3.Row) -> ValidationCampaignDefinition:
    return ValidationCampaignDefinition.model_validate(
        {
            "id": row["id"],
            "schema_version": row["schema_version"],
            "stage_1_study_id": row["stage_1_study_id"],
            "stage_1_evidence_digest": row["stage_1_evidence_digest"],
            "base_manifest": json.loads(str(row["base_manifest_json"])),
            "candidates": json.loads(str(row["candidates_json"])),
            "panel_input": json.loads(str(row["panel_input_json"])),
            "protocol_version": row["protocol_version"],
            "analysis_version": row["analysis_version"],
            "created_at": row["created_at"],
        }
    )


def _same_campaign_definition(
    left: ValidationCampaignDefinition, right: ValidationCampaignDefinition
) -> bool:
    return left.model_dump(mode="json", exclude={"created_at"}) == right.model_dump(
        mode="json", exclude={"created_at"}
    )


def _experiment_binding_from_row(row: sqlite3.Row) -> ValidationExperimentBinding:
    return ValidationExperimentBinding(
        campaign_id=str(row["campaign_id"]),
        trial_id=str(row["trial_id"]),
        experiment_id=str(row["experiment_id"]),
    )


def _event_from_row(row: sqlite3.Row) -> ValidationEvent:
    return ValidationEvent.model_validate(
        {
            "campaign_id": row["campaign_id"],
            "sequence": row["sequence"],
            "previous_state": row["previous_state"],
            "new_state": row["new_state"],
            "created_at": row["created_at"],
        }
    )


def _trial_from_row(row: sqlite3.Row) -> ValidationTrial:
    return ValidationTrial.model_validate(
        {
            "id": row["trial_id"],
            "ordinal": row["ordinal"],
            "archetype": row["archetype"],
            "prompt": row["prompt"],
            "negative_prompt": row["negative_prompt"] or None,
            "seed": row["seed"],
            "challenger": row["challenger"],
            "hidden_repeat": bool(row["hidden_repeat"]),
            "reuses_trial_id": row["reuses_trial_id"],
            "winner_display_position": row["winner_display_position"],
        }
    )


def _binding_from_row(row: sqlite3.Row) -> ValidationTrialBinding:
    return ValidationTrialBinding.model_validate(
        {
            "campaign_id": row["campaign_id"],
            "trial_id": row["trial_id"],
            "experiment_id": row["experiment_id"],
            "review_id": row["review_id"],
            "reused_from_trial_id": row["reused_from_trial_id"],
        }
    )


def _judgment_from_row(row: sqlite3.Row) -> ValidationJudgment:
    return ValidationJudgment.model_validate(
        {
            "campaign_id": row["campaign_id"],
            "trial_id": row["trial_id"],
            "ordinal": row["ordinal"],
            "archetype": row["archetype"],
            "challenger": row["challenger"],
            "seed": row["seed"],
            "hidden_repeat": bool(row["hidden_repeat"]),
            "winner_display_position": row["winner_display_position"],
            "choice": row["choice"],
            "created_at": row["created_at"],
        }
    )


def _result_from_row(row: sqlite3.Row) -> ValidationResult:
    return ValidationResult.model_validate(
        {
            "campaign_id": row["campaign_id"],
            "disposition": row["disposition"],
            "reason": row["reason"],
            "challengers": json.loads(str(row["challengers_json"])),
            "archetype_reversals": json.loads(str(row["archetype_reversals_json"])),
            "repeat_agreements": row["repeat_agreements"],
        }
    )


def _same_event(left: ValidationEvent, right: ValidationEvent) -> bool:
    return (
        left.campaign_id,
        left.sequence,
        left.previous_state,
        left.new_state,
    ) == (
        right.campaign_id,
        right.sequence,
        right.previous_state,
        right.new_state,
    )


def _same_judgment(left: ValidationJudgment, right: ValidationJudgment) -> bool:
    return left.model_dump(mode="json", exclude={"created_at"}) == right.model_dump(
        mode="json", exclude={"created_at"}
    )
