from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Literal, cast

from pydantic import TypeAdapter

from video_lab.adaptive_study_models import (
    ADAPTIVE_OBSERVATION_SCHEMA_VERSION,
    ADAPTIVE_STUDY_OWNER_BINDING_SCHEMA_VERSION,
    ADAPTIVE_STUDY_TRANSITIONS,
    AdaptiveObservation,
    AdaptiveObservationRequest,
    AdaptiveOptimizerStep,
    AdaptiveStudyDefinition,
    AdaptiveStudyEvent,
    AdaptiveStudyState,
    adaptive_optimizer_input_digest,
)
from video_lab.database import Database
from video_lab.domain import Manifest, canonical_json, utc_now
from video_lab.preference_optimizer import (
    AcquisitionScore,
    CandidatePair,
    OptimizerConfig,
    PosteriorSummary,
    PreferenceOutcome,
    SearchDisposition,
    StopReason,
)
from video_lab.reviews import (
    ADAPTIVE_REVIEW_BINDING_SCHEMA_VERSION,
    adaptive_review_binding_digest,
)

_ACQUISITION_LEDGER_ADAPTER = TypeAdapter(tuple[AcquisitionScore, ...])


class AdaptiveStudyNotFound(LookupError):
    pass


class AdaptiveStudyConflict(RuntimeError):
    pass


class AdaptiveStudyRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, definition: AdaptiveStudyDefinition) -> AdaptiveStudyDefinition:
        validated = AdaptiveStudyDefinition.model_validate(definition.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO adaptive_studies("
                    "id, schema_version, prompt, negative_prompt, seed, base_manifest_json, "
                    "optimizer_config_json, input_fingerprint, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        validated.id,
                        validated.schema_version,
                        validated.prompt,
                        validated.negative_prompt,
                        validated.seed,
                        canonical_json(validated.base_manifest),
                        canonical_json(validated.optimizer_config),
                        validated.input_fingerprint,
                        validated.created_at.isoformat(),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return validated

    def create_or_get_bound(
        self,
        definition: AdaptiveStudyDefinition,
        *,
        reviewer_id: str,
    ) -> AdaptiveStudyDefinition:
        """Atomically create one deterministic study and its immutable owner binding."""
        validated = AdaptiveStudyDefinition.model_validate(definition.model_dump(round_trip=True))
        if not reviewer_id or reviewer_id != reviewer_id.strip():
            raise AdaptiveStudyConflict("adaptive study owner must be a normalized identity")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM adaptive_studies WHERE id = ?", (validated.id,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO adaptive_studies("
                        "id, schema_version, prompt, negative_prompt, seed, base_manifest_json, "
                        "optimizer_config_json, input_fingerprint, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            validated.id,
                            validated.schema_version,
                            validated.prompt,
                            validated.negative_prompt,
                            validated.seed,
                            canonical_json(validated.base_manifest),
                            canonical_json(validated.optimizer_config),
                            validated.input_fingerprint,
                            validated.created_at.isoformat(),
                        ),
                    )
                    connection.execute(
                        "INSERT INTO adaptive_study_events("
                        "study_id, schema_version, sequence, previous_state, new_state, "
                        "payload_json, created_at) VALUES (?, ?, 1, ?, ?, ?, ?)",
                        (
                            validated.id,
                            "adaptive-study-event.v1",
                            AdaptiveStudyState.DRAFT.value,
                            AdaptiveStudyState.SEARCH_READY.value,
                            canonical_json(
                                {
                                    "owner_binding_schema_version": (
                                        ADAPTIVE_STUDY_OWNER_BINDING_SCHEMA_VERSION
                                    ),
                                    "reviewer_id": reviewer_id,
                                    "input_fingerprint": validated.input_fingerprint,
                                }
                            ),
                            validated.created_at.isoformat(),
                        ),
                    )
                    persisted = validated
                else:
                    persisted = _definition_from_row(row)
                    if persisted != validated:
                        raise AdaptiveStudyConflict(
                            "adaptive study id already has a different immutable definition"
                        )
                    first = connection.execute(
                        "SELECT * FROM adaptive_study_events WHERE study_id = ? "
                        "ORDER BY sequence LIMIT 1",
                        (validated.id,),
                    ).fetchone()
                    if first is None:
                        raise AdaptiveStudyConflict("adaptive study has no immutable owner binding")
                    event = _event_from_row(first)
                    if (
                        event.sequence != 1
                        or event.previous_state is not AdaptiveStudyState.DRAFT
                        or event.new_state is not AdaptiveStudyState.SEARCH_READY
                        or event.payload.get("owner_binding_schema_version")
                        != ADAPTIVE_STUDY_OWNER_BINDING_SCHEMA_VERSION
                        or event.payload.get("input_fingerprint") != validated.input_fingerprint
                    ):
                        raise AdaptiveStudyConflict("adaptive study owner binding is inconsistent")
                    if event.payload.get("reviewer_id") != reviewer_id:
                        raise PermissionError("adaptive study belongs to another reviewer")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return persisted

    def get(self, study_id: str) -> AdaptiveStudyDefinition:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM adaptive_studies WHERE id = ?", (study_id,)
            ).fetchone()
        if row is None:
            raise AdaptiveStudyNotFound(f"adaptive study not found: {study_id}")
        return _definition_from_row(row)

    def list(self) -> tuple[AdaptiveStudyDefinition, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM adaptive_studies ORDER BY created_at, id"
            ).fetchall()
        return tuple(_definition_from_row(row) for row in rows)

    def bind_owner(
        self,
        study_id: str,
        *,
        input_fingerprint: str,
        reviewer_id: str,
    ) -> AdaptiveStudyEvent:
        if not reviewer_id or reviewer_id != reviewer_id.strip():
            raise AdaptiveStudyConflict("adaptive study owner must be a normalized identity")
        definition = self.get(study_id)
        if definition.input_fingerprint != input_fingerprint:
            raise AdaptiveStudyConflict("adaptive study owner binding fingerprint mismatch")
        return self.append_event(
            AdaptiveStudyEvent(
                study_id=study_id,
                sequence=1,
                previous_state=AdaptiveStudyState.DRAFT,
                new_state=AdaptiveStudyState.SEARCH_READY,
                payload={
                    "owner_binding_schema_version": ADAPTIVE_STUDY_OWNER_BINDING_SCHEMA_VERSION,
                    "reviewer_id": reviewer_id,
                    "input_fingerprint": input_fingerprint,
                },
                created_at=utc_now(),
            )
        )

    def owner(self, study_id: str) -> str:
        definition = self.get(study_id)
        events = self.list_events(study_id)
        if not events:
            raise AdaptiveStudyConflict("adaptive study has no immutable owner binding")
        first = events[0]
        expected_keys = {
            "owner_binding_schema_version",
            "reviewer_id",
            "input_fingerprint",
        }
        if (
            first.sequence != 1
            or first.previous_state is not AdaptiveStudyState.DRAFT
            or first.new_state is not AdaptiveStudyState.SEARCH_READY
            or set(first.payload) != expected_keys
            or first.payload.get("owner_binding_schema_version")
            != ADAPTIVE_STUDY_OWNER_BINDING_SCHEMA_VERSION
            or first.payload.get("input_fingerprint") != definition.input_fingerprint
        ):
            raise AdaptiveStudyConflict("adaptive study owner binding is invalid")
        reviewer_id = first.payload.get("reviewer_id")
        if (
            not isinstance(reviewer_id, str)
            or not reviewer_id
            or reviewer_id != reviewer_id.strip()
        ):
            raise AdaptiveStudyConflict("adaptive study owner binding is invalid")
        return reviewer_id

    def append_event(self, event: AdaptiveStudyEvent) -> AdaptiveStudyEvent:
        validated = AdaptiveStudyEvent.model_validate(event.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_study(connection, validated.study_id)
                latest = connection.execute(
                    "SELECT sequence, new_state FROM adaptive_study_events "
                    "WHERE study_id = ? ORDER BY sequence DESC LIMIT 1",
                    (validated.study_id,),
                ).fetchone()
                next_sequence = 1 if latest is None else int(latest["sequence"]) + 1
                current_state = (
                    AdaptiveStudyState.DRAFT
                    if latest is None
                    else AdaptiveStudyState(str(latest["new_state"]))
                )
                if validated.sequence != next_sequence:
                    raise AdaptiveStudyConflict(
                        f"expected event sequence {next_sequence}, got {validated.sequence}"
                    )
                if validated.previous_state is not current_state:
                    raise AdaptiveStudyConflict(
                        f"expected prior state {current_state.value}, "
                        f"got {validated.previous_state.value}"
                    )
                if validated.new_state not in ADAPTIVE_STUDY_TRANSITIONS[current_state]:
                    raise AdaptiveStudyConflict(
                        "disallowed adaptive study transition "
                        f"{current_state.value} -> {validated.new_state.value}"
                    )
                connection.execute(
                    "INSERT INTO adaptive_study_events("
                    "study_id, schema_version, sequence, previous_state, new_state, "
                    "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        validated.study_id,
                        validated.schema_version,
                        validated.sequence,
                        validated.previous_state.value,
                        validated.new_state.value,
                        canonical_json(validated.model_dump(mode="json")["payload"]),
                        validated.created_at.isoformat(),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return validated

    def append_awaiting_preference_or_verify(self, event: AdaptiveStudyEvent) -> AdaptiveStudyEvent:
        """Append one sealed-review transition or return its exact concurrent winner."""
        validated = AdaptiveStudyEvent.model_validate(event.model_dump(round_trip=True))
        expected_keys = {
            "binding_schema_version",
            "study_id",
            "optimizer_step_ordinal",
            "optimizer_output_digest",
            "experiment_id",
            "review_session_id",
            "reviewer_id",
            "sealed_mapping_digest",
        }
        if (
            validated.previous_state is not AdaptiveStudyState.GENERATING_PAIR
            or validated.new_state is not AdaptiveStudyState.AWAITING_PREFERENCE
            or set(validated.payload) != expected_keys
        ):
            raise AdaptiveStudyConflict("sealed review transition is invalid")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_study(connection, validated.study_id)
                latest_row = connection.execute(
                    "SELECT * FROM adaptive_study_events WHERE study_id = ? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (validated.study_id,),
                ).fetchone()
                if latest_row is None:
                    raise AdaptiveStudyConflict("sealed review transition has no prior event")
                latest = _event_from_row(latest_row)
                if (
                    latest.sequence == validated.sequence - 1
                    and latest.new_state is AdaptiveStudyState.GENERATING_PAIR
                ):
                    connection.execute(
                        "INSERT INTO adaptive_study_events("
                        "study_id, schema_version, sequence, previous_state, new_state, "
                        "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            validated.study_id,
                            validated.schema_version,
                            validated.sequence,
                            validated.previous_state.value,
                            validated.new_state.value,
                            canonical_json(validated.model_dump(mode="json")["payload"]),
                            validated.created_at.isoformat(),
                        ),
                    )
                    persisted_row = connection.execute(
                        "SELECT * FROM adaptive_study_events WHERE study_id = ? AND sequence = ?",
                        (validated.study_id, validated.sequence),
                    ).fetchone()
                    if persisted_row is None:
                        raise RuntimeError("persisted sealed review event disappeared")
                    persisted = _event_from_row(persisted_row)
                elif (
                    latest.sequence == validated.sequence
                    and latest.schema_version == validated.schema_version
                    and latest.previous_state is validated.previous_state
                    and latest.new_state is validated.new_state
                    and latest.payload == validated.payload
                ):
                    persisted = latest
                else:
                    raise AdaptiveStudyConflict(
                        "sealed review transition lost an unrelated or mismatched race"
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return persisted

    def list_events(self, study_id: str) -> tuple[AdaptiveStudyEvent, ...]:
        with self.database.connect() as connection:
            self._require_study(connection, study_id)
            rows = connection.execute(
                "SELECT * FROM adaptive_study_events WHERE study_id = ? ORDER BY sequence",
                (study_id,),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def append_observation(self, request: AdaptiveObservationRequest) -> AdaptiveObservation:
        observation, _consumed = self._consume_observation(request, verify_retry=False)
        return observation

    def consume_observation_or_verify(
        self, request: AdaptiveObservationRequest
    ) -> tuple[AdaptiveObservation, bool]:
        """Atomically consume one optimizer step or verify its exact review retry."""
        return self._consume_observation(request, verify_retry=True)

    def _consume_observation(
        self,
        request: AdaptiveObservationRequest,
        *,
        verify_retry: bool,
    ) -> tuple[AdaptiveObservation, bool]:
        validated = AdaptiveObservationRequest.model_validate(request.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                study_row = self._require_study(connection, validated.study_id)
                definition = _definition_from_row(study_row)
                existing_row = connection.execute(
                    "SELECT * FROM adaptive_observations WHERE study_id = ? AND ordinal = ?",
                    (validated.study_id, validated.ordinal),
                ).fetchone()
                if existing_row is not None:
                    if not verify_retry:
                        next_ordinal = int(
                            connection.execute(
                                "SELECT COALESCE(MAX(ordinal), 0) + 1 "
                                "FROM adaptive_observations WHERE study_id = ?",
                                (validated.study_id,),
                            ).fetchone()[0]
                        )
                        raise AdaptiveStudyConflict(
                            f"expected observation ordinal {next_ordinal}, got {validated.ordinal}"
                        )
                    existing = _observation_from_row(existing_row)
                    if existing.review_id != validated.review_id:
                        raise AdaptiveStudyConflict(
                            "observation retry does not match persisted review evidence"
                        )
                    connection.commit()
                    return existing, False
                current_state = self._current_state(connection, validated.study_id)
                if current_state is not AdaptiveStudyState.AWAITING_PREFERENCE:
                    raise AdaptiveStudyConflict(
                        "observation cannot be appended while study is "
                        f"{current_state.value}; expected awaiting_preference"
                    )
                next_ordinal = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM adaptive_observations "
                        "WHERE study_id = ?",
                        (validated.study_id,),
                    ).fetchone()[0]
                )
                if validated.ordinal != next_ordinal:
                    raise AdaptiveStudyConflict(
                        f"expected observation ordinal {next_ordinal}, got {validated.ordinal}"
                    )
                optimizer_row = connection.execute(
                    "SELECT * FROM adaptive_optimizer_steps WHERE study_id = ? AND ordinal = ?",
                    (validated.study_id, validated.ordinal),
                ).fetchone()
                if optimizer_row is None:
                    raise AdaptiveStudyConflict(
                        f"optimizer step {validated.ordinal} does not exist"
                    )
                optimizer_step = _optimizer_step_from_row(optimizer_row)
                if (
                    optimizer_step.stop_reason is not StopReason.CONTINUE
                    or optimizer_step.selected_pair is None
                ):
                    raise AdaptiveStudyConflict(
                        f"optimizer step {validated.ordinal} did not select a comparison pair"
                    )
                if optimizer_step.comparison_count != validated.ordinal - 1:
                    raise AdaptiveStudyConflict(
                        "optimizer step comparison count does not match observation ordinal"
                    )
                already_consumed = connection.execute(
                    "SELECT 1 FROM adaptive_observations "
                    "WHERE study_id = ? AND optimizer_step_ordinal = ?",
                    (validated.study_id, validated.ordinal),
                ).fetchone()
                if already_consumed is not None:
                    raise AdaptiveStudyConflict(
                        f"optimizer step {validated.ordinal} was already consumed"
                    )
                observation = _derive_observation_from_review(
                    connection,
                    definition=definition,
                    optimizer_step=optimizer_step,
                    request=validated,
                )
                existing_rows = connection.execute(
                    "SELECT * FROM adaptive_observations WHERE study_id = ? ORDER BY ordinal",
                    (validated.study_id,),
                ).fetchall()
                optimizer_observations = tuple(
                    _observation_from_row(row).preference_observation for row in existing_rows
                ) + (observation.preference_observation,)
                definition.optimizer_config.validate_observations(optimizer_observations)
                connection.execute(
                    "INSERT INTO adaptive_observations("
                    "study_id, schema_version, ordinal, optimizer_step_ordinal, value_a, value_b, "
                    "experiment_id, review_id, display_mapping_digest, outcome, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        observation.study_id,
                        observation.schema_version,
                        observation.ordinal,
                        observation.optimizer_step_ordinal,
                        observation.pair.value_a,
                        observation.pair.value_b,
                        observation.experiment_id,
                        observation.review_id,
                        observation.display_mapping_digest,
                        observation.outcome.value,
                        observation.created_at.isoformat(),
                    ),
                )
                persisted_row = connection.execute(
                    "SELECT * FROM adaptive_observations WHERE study_id = ? AND ordinal = ?",
                    (observation.study_id, observation.ordinal),
                ).fetchone()
                if persisted_row is None:
                    raise RuntimeError("persisted observation disappeared inside transaction")
                persisted = _observation_from_row(persisted_row)
                if persisted != observation:
                    raise RuntimeError("persisted observation is not byte-stable")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return persisted, True

    def list_observations(self, study_id: str) -> tuple[AdaptiveObservation, ...]:
        with self.database.connect() as connection:
            self._require_study(connection, study_id)
            rows = connection.execute(
                "SELECT * FROM adaptive_observations WHERE study_id = ? ORDER BY ordinal",
                (study_id,),
            ).fetchall()
        return tuple(_observation_from_row(row) for row in rows)

    def ensure_initial_plan(
        self,
        step: AdaptiveOptimizerStep,
        *,
        experiment_id: str,
    ) -> AdaptiveStudyEvent:
        """Persist or verify the deterministic initial optimizer step and generation event."""
        validated = AdaptiveOptimizerStep.model_validate(step.model_dump(round_trip=True))
        if (
            validated.ordinal != 1
            or validated.comparison_count != 0
            or validated.stop_reason is not StopReason.CONTINUE
            or validated.selected_pair is None
            or not experiment_id
        ):
            raise AdaptiveStudyConflict("initial adaptive optimizer plan is invalid")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                definition = _definition_from_row(
                    self._require_study(connection, validated.study_id)
                )
                expected_input = adaptive_optimizer_input_digest(definition.optimizer_config, ())
                if validated.input_digest != expected_input:
                    raise AdaptiveStudyConflict(
                        "initial optimizer input digest does not match persisted definition"
                    )
                latest_row = connection.execute(
                    "SELECT * FROM adaptive_study_events WHERE study_id = ? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (validated.study_id,),
                ).fetchone()
                if latest_row is None:
                    raise AdaptiveStudyConflict("adaptive study has no owner binding")
                latest = _event_from_row(latest_row)
                existing_row = connection.execute(
                    "SELECT * FROM adaptive_optimizer_steps WHERE study_id = ? AND ordinal = 1",
                    (validated.study_id,),
                ).fetchone()
                if existing_row is None:
                    if latest.new_state is not AdaptiveStudyState.SEARCH_READY:
                        raise AdaptiveStudyConflict(
                            "initial optimizer plan is missing after study advanced"
                        )
                    _insert_optimizer_step_row(connection, validated)
                    persisted_step = validated
                else:
                    persisted_step = _optimizer_step_from_row(existing_row)
                    if persisted_step != validated:
                        raise AdaptiveStudyConflict(
                            "initial optimizer plan conflicts with persisted evidence"
                        )

                payload = {
                    "optimizer_step_ordinal": persisted_step.ordinal,
                    "optimizer_output_digest": persisted_step.output_digest,
                    "experiment_id": experiment_id,
                }
                if latest.new_state is AdaptiveStudyState.SEARCH_READY:
                    connection.execute(
                        "INSERT INTO adaptive_study_events("
                        "study_id, schema_version, sequence, previous_state, new_state, "
                        "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            validated.study_id,
                            "adaptive-study-event.v1",
                            latest.sequence + 1,
                            AdaptiveStudyState.SEARCH_READY.value,
                            AdaptiveStudyState.GENERATING_PAIR.value,
                            canonical_json(payload),
                            persisted_step.created_at.isoformat(),
                        ),
                    )
                    planned_row = connection.execute(
                        "SELECT * FROM adaptive_study_events WHERE study_id = ? AND sequence = ?",
                        (validated.study_id, latest.sequence + 1),
                    ).fetchone()
                    if planned_row is None:
                        raise RuntimeError("persisted initial generation plan disappeared")
                    planned = _event_from_row(planned_row)
                elif latest.new_state is AdaptiveStudyState.GENERATING_PAIR:
                    if latest.payload != payload:
                        raise AdaptiveStudyConflict(
                            "initial generation event conflicts with optimizer plan"
                        )
                    planned = latest
                else:
                    raise AdaptiveStudyConflict(
                        "initial optimizer plan cannot be ensured after study advanced"
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return planned

    def append_optimizer_step(self, step: AdaptiveOptimizerStep) -> AdaptiveOptimizerStep:
        validated = AdaptiveOptimizerStep.model_validate(step.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                study_row = self._require_study(connection, validated.study_id)
                current_state = self._current_state(connection, validated.study_id)
                expected_state = (
                    AdaptiveStudyState.SEARCH_READY
                    if validated.ordinal == 1
                    else AdaptiveStudyState.UPDATING_POSTERIOR
                )
                if current_state is not expected_state:
                    raise AdaptiveStudyConflict(
                        "optimizer step cannot be appended while study is "
                        f"{current_state.value}; expected {expected_state.value}"
                    )
                definition = _definition_from_row(study_row)
                observation_rows = connection.execute(
                    "SELECT * FROM adaptive_observations WHERE study_id = ? ORDER BY ordinal",
                    (validated.study_id,),
                ).fetchall()
                observations = tuple(
                    _observation_from_row(row).preference_observation for row in observation_rows
                )
                expected_input = adaptive_optimizer_input_digest(
                    definition.optimizer_config, observations
                )
                if validated.input_digest != expected_input:
                    raise AdaptiveStudyConflict(
                        "optimizer input digest does not match persisted canonical observations"
                    )
                next_ordinal = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(ordinal), 0) + 1 "
                        "FROM adaptive_optimizer_steps WHERE study_id = ?",
                        (validated.study_id,),
                    ).fetchone()[0]
                )
                if validated.ordinal != next_ordinal:
                    raise AdaptiveStudyConflict(
                        f"expected optimizer step ordinal {next_ordinal}, got {validated.ordinal}"
                    )
                if validated.comparison_count != len(observations):
                    raise AdaptiveStudyConflict(
                        "optimizer comparison count does not match persisted observations"
                    )
                _insert_optimizer_step_row(connection, validated)
                persisted_row = connection.execute(
                    "SELECT * FROM adaptive_optimizer_steps WHERE study_id = ? AND ordinal = ?",
                    (validated.study_id, validated.ordinal),
                ).fetchone()
                if persisted_row is None:
                    raise RuntimeError("persisted optimizer step disappeared inside transaction")
                persisted = _optimizer_step_from_row(persisted_row)
                persisted_input = adaptive_optimizer_input_digest(
                    definition.optimizer_config, observations
                )
                if persisted.input_digest != persisted_input:
                    raise RuntimeError("persisted optimizer input digest is not reproducible")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return persisted

    def list_optimizer_steps(self, study_id: str) -> tuple[AdaptiveOptimizerStep, ...]:
        with self.database.connect() as connection:
            definition = _definition_from_row(self._require_study(connection, study_id))
            rows = connection.execute(
                "SELECT * FROM adaptive_optimizer_steps WHERE study_id = ? ORDER BY ordinal",
                (study_id,),
            ).fetchall()
            observation_rows = connection.execute(
                "SELECT * FROM adaptive_observations WHERE study_id = ? ORDER BY ordinal",
                (study_id,),
            ).fetchall()
        observations = tuple(
            _observation_from_row(row).preference_observation for row in observation_rows
        )
        steps = tuple(_optimizer_step_from_row(row) for row in rows)
        for step in steps:
            prefix = observations[: step.comparison_count]
            if len(prefix) != step.comparison_count:
                raise AdaptiveStudyConflict(
                    "persisted optimizer step references missing observations"
                )
            expected_input = adaptive_optimizer_input_digest(definition.optimizer_config, prefix)
            if step.input_digest != expected_input:
                raise AdaptiveStudyConflict(
                    "persisted optimizer input digest does not match reloaded evidence"
                )
        return steps

    @staticmethod
    def _require_study(connection: sqlite3.Connection, study_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM adaptive_studies WHERE id = ?", (study_id,)
        ).fetchone()
        if row is None:
            raise AdaptiveStudyNotFound(f"adaptive study not found: {study_id}")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _current_state(connection: sqlite3.Connection, study_id: str) -> AdaptiveStudyState:
        row = connection.execute(
            "SELECT new_state FROM adaptive_study_events WHERE study_id = ? "
            "ORDER BY sequence DESC LIMIT 1",
            (study_id,),
        ).fetchone()
        return (
            AdaptiveStudyState.DRAFT if row is None else AdaptiveStudyState(str(row["new_state"]))
        )


def _insert_optimizer_step_row(connection: sqlite3.Connection, step: AdaptiveOptimizerStep) -> None:
    connection.execute(
        "INSERT INTO adaptive_optimizer_steps("
        "study_id, schema_version, ordinal, input_digest, output_digest, "
        "posterior_json, acquisition_ledger_json, selected_pair_json, "
        "stop_reason, disposition, comparison_count, maximum_information_bits, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            step.study_id,
            step.schema_version,
            step.ordinal,
            step.input_digest,
            step.output_digest,
            canonical_json(step.posterior),
            _canonical_array_json(step.acquisition_ledger),
            None if step.selected_pair is None else canonical_json(step.selected_pair),
            step.stop_reason.value,
            None if step.disposition is None else step.disposition.value,
            step.comparison_count,
            step.maximum_information_bits,
            step.created_at.isoformat(),
        ),
    )


def _definition_from_row(row: sqlite3.Row) -> AdaptiveStudyDefinition:
    return AdaptiveStudyDefinition(
        id=str(row["id"]),
        schema_version=cast(Literal["adaptive-study.v1"], str(row["schema_version"])),
        prompt=str(row["prompt"]),
        negative_prompt=(None if row["negative_prompt"] is None else str(row["negative_prompt"])),
        seed=int(row["seed"]),
        base_manifest=Manifest.model_validate_json(str(row["base_manifest_json"])),
        optimizer_config=OptimizerConfig.model_validate_json(str(row["optimizer_config_json"])),
        input_fingerprint=str(row["input_fingerprint"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _event_from_row(row: sqlite3.Row) -> AdaptiveStudyEvent:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise ValueError("adaptive study event payload must be an object")
    return AdaptiveStudyEvent(
        study_id=str(row["study_id"]),
        schema_version=cast(Literal["adaptive-study-event.v1"], str(row["schema_version"])),
        sequence=int(row["sequence"]),
        previous_state=AdaptiveStudyState(str(row["previous_state"])),
        new_state=AdaptiveStudyState(str(row["new_state"])),
        payload=payload,
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _observation_from_row(row: sqlite3.Row) -> AdaptiveObservation:
    return AdaptiveObservation(
        study_id=str(row["study_id"]),
        schema_version=cast(Literal["adaptive-observation.v1"], str(row["schema_version"])),
        ordinal=int(row["ordinal"]),
        optimizer_step_ordinal=int(row["optimizer_step_ordinal"]),
        pair=CandidatePair(value_a=float(row["value_a"]), value_b=float(row["value_b"])),
        experiment_id=str(row["experiment_id"]),
        review_id=str(row["review_id"]),
        display_mapping_digest=str(row["display_mapping_digest"]),
        outcome=PreferenceOutcome(str(row["outcome"])),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _canonical_array_json(values: tuple[AcquisitionScore, ...]) -> str:
    return json.dumps(
        [value.model_dump(mode="json") for value in values],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _optimizer_step_from_row(row: sqlite3.Row) -> AdaptiveOptimizerStep:
    selected_pair_json = row["selected_pair_json"]
    disposition = row["disposition"]
    return AdaptiveOptimizerStep(
        study_id=str(row["study_id"]),
        schema_version=cast(Literal["adaptive-optimizer-step.v1"], str(row["schema_version"])),
        ordinal=int(row["ordinal"]),
        input_digest=str(row["input_digest"]),
        output_digest=str(row["output_digest"]),
        posterior=PosteriorSummary.model_validate_json(str(row["posterior_json"])),
        acquisition_ledger=_ACQUISITION_LEDGER_ADAPTER.validate_json(
            str(row["acquisition_ledger_json"])
        ),
        selected_pair=(
            None
            if selected_pair_json is None
            else CandidatePair.model_validate_json(str(selected_pair_json))
        ),
        stop_reason=StopReason(str(row["stop_reason"])),
        disposition=(None if disposition is None else SearchDisposition(str(disposition))),
        comparison_count=int(row["comparison_count"]),
        maximum_information_bits=float(row["maximum_information_bits"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _derive_observation_from_review(
    connection: sqlite3.Connection,
    *,
    definition: AdaptiveStudyDefinition,
    optimizer_step: AdaptiveOptimizerStep,
    request: AdaptiveObservationRequest,
) -> AdaptiveObservation:
    revision = connection.execute(
        "SELECT id, session_id, revision FROM review_revisions WHERE id = ?",
        (request.review_id,),
    ).fetchone()
    if revision is None:
        raise AdaptiveStudyConflict(f"review does not exist: {request.review_id}")
    latest = connection.execute(
        "SELECT id FROM review_revisions WHERE session_id = ? ORDER BY revision DESC LIMIT 1",
        (str(revision["session_id"]),),
    ).fetchone()
    if latest is None or str(latest["id"]) != request.review_id:
        raise AdaptiveStudyConflict("observation must consume the latest review revision")
    row = connection.execute(
        "SELECT "
        "rr.id AS review_id, rr.session_id, rr.preferred_variant_id, "
        "rr.display_preference, rr.notes, rr.created_at AS review_created_at, "
        "rr.revision, rr.supersedes_review_id, "
        "rs.experiment_id, rs.reviewer_id, rs.created_at AS session_created_at, "
        "rs.left_variant_id, rs.right_variant_id, rs.left_job_id, rs.right_job_id, "
        "rs.left_artifact_id, rs.right_artifact_id, "
        "rs.left_artifact_sha256, rs.right_artifact_sha256, "
        "e.schema_version AS experiment_schema_version, e.variable_path, "
        "e.changed_paths_json, "
        "lv.experiment_id AS left_variant_experiment_id, lv.label AS left_label, "
        "lv.manifest_json AS left_manifest_json, "
        "rv.experiment_id AS right_variant_experiment_id, rv.label AS right_label, "
        "rv.manifest_json AS right_manifest_json, "
        "lj.variant_id AS left_job_variant_id, lj.state AS left_job_state, "
        "rj.variant_id AS right_job_variant_id, rj.state AS right_job_state, "
        "la.sha256 AS left_persisted_sha256, ra.sha256 AS right_persisted_sha256 "
        "FROM review_revisions rr "
        "JOIN review_sessions rs ON rs.id = rr.session_id "
        "JOIN experiments e ON e.id = rs.experiment_id "
        "JOIN variants lv ON lv.id = rs.left_variant_id "
        "JOIN variants rv ON rv.id = rs.right_variant_id "
        "JOIN jobs lj ON lj.id = rs.left_job_id "
        "JOIN jobs rj ON rj.id = rs.right_job_id "
        "JOIN artifacts la ON la.id = rs.left_artifact_id "
        "JOIN artifacts ra ON ra.id = rs.right_artifact_id "
        "JOIN job_artifacts lja ON lja.job_id = rs.left_job_id "
        "AND lja.artifact_id = rs.left_artifact_id "
        "JOIN job_artifacts rja ON rja.job_id = rs.right_job_id "
        "AND rja.artifact_id = rs.right_artifact_id "
        "WHERE rr.id = ?",
        (request.review_id,),
    ).fetchone()
    if row is None:
        raise AdaptiveStudyConflict("review provenance is incomplete")
    _require_active_review_binding(
        connection,
        definition=definition,
        optimizer_step=optimizer_step,
        request=request,
        review_row=row,
    )
    if not _review_relations_are_consistent(row):
        raise AdaptiveStudyConflict("review provenance does not form one sealed experiment")
    try:
        changed_paths = json.loads(str(row["changed_paths_json"]))
        left_manifest = Manifest.model_validate_json(str(row["left_manifest_json"]))
        right_manifest = Manifest.model_validate_json(str(row["right_manifest_json"]))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise AdaptiveStudyConflict("review provenance contains invalid manifests") from error
    if (
        str(row["experiment_schema_version"]) != "manifest.v1"
        or str(row["variable_path"]) != "guidance_scale"
        or changed_paths != ["guidance_scale"]
        or not _matches_study_identity(left_manifest, definition.base_manifest)
        or not _matches_study_identity(right_manifest, definition.base_manifest)
    ):
        raise AdaptiveStudyConflict("review provenance does not match the adaptive study")
    pair = CandidatePair(
        value_a=min(left_manifest.guidance_scale, right_manifest.guidance_scale),
        value_b=max(left_manifest.guidance_scale, right_manifest.guidance_scale),
    )
    if pair != optimizer_step.selected_pair:
        raise AdaptiveStudyConflict("review comparison pair does not match optimizer-selected pair")
    outcome = _preference_outcome(row, left_manifest, right_manifest, pair)
    digest_row = connection.execute(
        "SELECT adaptive_observation_provenance_digest(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            definition.id,
            ADAPTIVE_OBSERVATION_SCHEMA_VERSION,
            request.ordinal,
            optimizer_step.ordinal,
            pair.value_a,
            pair.value_b,
            str(row["experiment_id"]),
            request.review_id,
            outcome.value,
            request.created_at.isoformat(),
        ),
    ).fetchone()
    digest = None if digest_row is None else digest_row[0]
    if not isinstance(digest, str):
        raise AdaptiveStudyConflict("observation provenance digest could not be derived")
    return AdaptiveObservation(
        study_id=definition.id,
        ordinal=request.ordinal,
        optimizer_step_ordinal=optimizer_step.ordinal,
        pair=pair,
        experiment_id=str(row["experiment_id"]),
        review_id=request.review_id,
        display_mapping_digest=digest,
        outcome=outcome,
        created_at=request.created_at,
    )


def _require_active_review_binding(
    connection: sqlite3.Connection,
    *,
    definition: AdaptiveStudyDefinition,
    optimizer_step: AdaptiveOptimizerStep,
    request: AdaptiveObservationRequest,
    review_row: sqlite3.Row,
) -> None:
    score_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM rubric_scores WHERE review_id = ?", (request.review_id,)
        ).fetchone()[0]
    )
    if (
        int(review_row["revision"]) != 1
        or review_row["supersedes_review_id"] is not None
        or review_row["notes"] is not None
        or score_count != 0
    ):
        raise AdaptiveStudyConflict(
            "adaptive observation requires one pristine immutable review revision"
        )
    session_row = connection.execute(
        "SELECT * FROM review_sessions WHERE id = ?", (str(review_row["session_id"]),)
    ).fetchone()
    event_row = connection.execute(
        "SELECT new_state, payload_json FROM adaptive_study_events "
        "WHERE study_id = ? ORDER BY sequence DESC LIMIT 1",
        (definition.id,),
    ).fetchone()
    if (
        session_row is None
        or event_row is None
        or str(event_row["new_state"]) != "awaiting_preference"
    ):
        raise AdaptiveStudyConflict("adaptive observation has no active review binding")
    try:
        payload = json.loads(str(event_row["payload_json"]))
    except (TypeError, json.JSONDecodeError) as error:
        raise AdaptiveStudyConflict("adaptive active review binding is invalid") from error
    expected_keys = {
        "binding_schema_version",
        "study_id",
        "optimizer_step_ordinal",
        "optimizer_output_digest",
        "experiment_id",
        "review_session_id",
        "reviewer_id",
        "sealed_mapping_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise AdaptiveStudyConflict("adaptive active review binding is incomplete")
    expected_digest = adaptive_review_binding_digest(
        session_row,
        study_id=definition.id,
        optimizer_step_ordinal=optimizer_step.ordinal,
        optimizer_output_digest=optimizer_step.output_digest,
    )
    if (
        payload["binding_schema_version"] != ADAPTIVE_REVIEW_BINDING_SCHEMA_VERSION
        or payload["study_id"] != definition.id
        or payload["optimizer_step_ordinal"] != optimizer_step.ordinal
        or payload["optimizer_output_digest"] != optimizer_step.output_digest
        or payload["experiment_id"] != str(session_row["experiment_id"])
        or payload["review_session_id"] != str(session_row["id"])
        or payload["reviewer_id"] != str(session_row["reviewer_id"])
        or payload["sealed_mapping_digest"] != expected_digest
    ):
        raise AdaptiveStudyConflict("adaptive observation does not match the active review binding")


def _review_relations_are_consistent(row: sqlite3.Row) -> bool:
    experiment_id = str(row["experiment_id"])
    return all(
        (
            str(row["left_variant_experiment_id"]) == experiment_id,
            str(row["right_variant_experiment_id"]) == experiment_id,
            str(row["left_job_variant_id"]) == str(row["left_variant_id"]),
            str(row["right_job_variant_id"]) == str(row["right_variant_id"]),
            str(row["left_job_state"]) == "completed",
            str(row["right_job_state"]) == "completed",
            str(row["left_persisted_sha256"]) == str(row["left_artifact_sha256"]),
            str(row["right_persisted_sha256"]) == str(row["right_artifact_sha256"]),
        )
    )


def _matches_study_identity(candidate: Manifest, base: Manifest) -> bool:
    payload = candidate.model_dump(mode="json")
    payload["guidance_scale"] = base.guidance_scale
    return payload == base.model_dump(mode="json")


def _preference_outcome(
    row: sqlite3.Row,
    left_manifest: Manifest,
    right_manifest: Manifest,
    pair: CandidatePair,
) -> PreferenceOutcome:
    preference = str(row["display_preference"])
    preferred_variant_id = row["preferred_variant_id"]
    if preference == "tie":
        if preferred_variant_id is not None:
            raise AdaptiveStudyConflict("tie review has a preferred variant")
        return PreferenceOutcome.TIE
    position = "left" if preference == "left" else "right" if preference == "right" else None
    if position is None:
        raise AdaptiveStudyConflict("review has an unsupported display preference")
    expected_variant_id = str(row[f"{position}_variant_id"])
    if preferred_variant_id is None or str(preferred_variant_id) != expected_variant_id:
        raise AdaptiveStudyConflict("review preference does not match sealed display mapping")
    preferred_value = (
        left_manifest.guidance_scale if position == "left" else right_manifest.guidance_scale
    )
    return PreferenceOutcome.A if preferred_value == pair.value_a else PreferenceOutcome.B
