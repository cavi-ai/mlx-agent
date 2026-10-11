from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError

from video_lab.adaptive_study_models import (
    AdaptiveObservation,
    AdaptiveObservationRequest,
    AdaptiveOptimizerStep,
    AdaptiveStudyDefinition,
    AdaptiveStudyEvent,
    AdaptiveStudyState,
    adaptive_optimizer_input_digest,
    adaptive_optimizer_output_digest,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import AdaptiveStudyConflict, AdaptiveStudyRepository
from video_lab.database import Database
from video_lab.domain import Manifest, canonical_json
from video_lab.preference_optimizer import (
    AcquisitionScore,
    CandidatePair,
    OptimizerConfig,
    PreferenceOutcome,
    StopReason,
    optimize_preference,
)
from video_lab.reviews import (
    ADAPTIVE_REVIEW_BINDING_SCHEMA_VERSION,
    adaptive_review_binding_digest,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def make_manifest() -> Manifest:
    return Manifest(
        model_id="org/model",
        model_revision="abc123",
        model_checksum="sha256:weights",
        adapter_id="diffusers-mps",
        adapter_version="0.1.0",
        runtime_versions={"python": "3.12.0"},
        prompt="a blue-haired character walking — 蒼",
        negative_prompt="flicker",
        seed=42,
        width=512,
        height=288,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=1.0,
        scheduler="flow-match-euler",
        dtype="float16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="unified",
        hardware_fingerprint="apple-m5-max-128gb",
        os_fingerprint="macos-test",
    )


def make_definition(
    study_id: str = "study_1",
    *,
    seed: int = 42,
    created_at: datetime = NOW,
) -> AdaptiveStudyDefinition:
    manifest = make_manifest().model_copy(update={"seed": seed})
    config = OptimizerConfig(posterior_draws=256)
    return AdaptiveStudyDefinition(
        id=study_id,
        prompt=manifest.prompt,
        negative_prompt=manifest.negative_prompt,
        seed=manifest.seed,
        base_manifest=manifest,
        optimizer_config=config,
        input_fingerprint=adaptive_study_input_fingerprint(manifest, config),
        created_at=created_at,
    )


def persist_review_binding(
    database: Database,
    *,
    experiment_id: str = "exp_1",
    review_id: str = "review_1",
    left_guidance: float = 2.0,
    right_guidance: float = 4.0,
    display_preference: str = "left",
    include_revision: bool = True,
) -> None:
    suffix = experiment_id.removeprefix("exp_")
    variant_a = f"var_a_{suffix}"
    variant_b = f"var_b_{suffix}"
    job_a = f"job_a_{suffix}"
    job_b = f"job_b_{suffix}"
    artifact_a = f"art_a_{suffix}"
    artifact_b = f"art_b_{suffix}"
    session_id = f"session_{suffix}"
    sha_a = hashlib.sha256(artifact_a.encode()).hexdigest()
    sha_b = hashlib.sha256(artifact_b.encode()).hexdigest()
    left_manifest = make_manifest().model_copy(update={"guidance_scale": left_guidance})
    right_manifest = make_manifest().model_copy(update={"guidance_scale": right_guidance})
    preferred_variant = {
        "left": variant_a,
        "right": variant_b,
        "tie": None,
    }[display_preference]
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO experiments VALUES "
            "(?, 'manifest.v1', 'guidance_scale', '[\"guidance_scale\"]', 'now')",
            (experiment_id,),
        )
        connection.executemany(
            "INSERT INTO variants VALUES (?, ?, ?, ?)",
            (
                (variant_a, experiment_id, "A", canonical_json(left_manifest)),
                (variant_b, experiment_id, "B", canonical_json(right_manifest)),
            ),
        )
        connection.executemany(
            "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
            "VALUES (?, ?, 'completed', 'now', 'now')",
            ((job_a, variant_a), (job_b, variant_b)),
        )
        connection.executemany(
            "INSERT INTO artifacts(id, sha256, byte_size, media_type, relative_path, created_at) "
            "VALUES (?, ?, 1, 'video/mp4', ?, 'now')",
            (
                (artifact_a, sha_a, f"{artifact_a}.mp4"),
                (artifact_b, sha_b, f"{artifact_b}.mp4"),
            ),
        )
        connection.execute(
            "INSERT INTO review_sessions VALUES (?, ?, 'reviewer', ?, ?, ?, ?, ?, ?, ?, ?, 'now')",
            (
                session_id,
                experiment_id,
                variant_a,
                variant_b,
                job_a,
                artifact_a,
                sha_a,
                job_b,
                artifact_b,
                sha_b,
            ),
        )
        connection.executemany(
            "INSERT INTO job_artifacts(job_id, artifact_id, created_at) VALUES (?, ?, 'now')",
            ((job_a, artifact_a), (job_b, artifact_b)),
        )
        if include_revision:
            connection.execute(
                "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
                "display_preference, created_at, revision) "
                "VALUES (?, ?, ?, ?, 'now', 1)",
                (review_id, session_id, preferred_variant, display_preference),
            )
    repository = AdaptiveStudyRepository(database)
    events = repository.list_events("study_1")
    if events[-1].new_state is AdaptiveStudyState.GENERATING_PAIR:
        steps = repository.list_optimizer_steps("study_1")
        step_ordinal = steps[-1].ordinal if steps else 1
        output_digest = steps[-1].output_digest if steps else "0" * 64
        with database.connect() as connection:
            session_row = connection.execute(
                "SELECT * FROM review_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        repository.append_event(
            AdaptiveStudyEvent(
                study_id="study_1",
                sequence=len(events) + 1,
                previous_state=AdaptiveStudyState.GENERATING_PAIR,
                new_state=AdaptiveStudyState.AWAITING_PREFERENCE,
                payload={
                    "binding_schema_version": ADAPTIVE_REVIEW_BINDING_SCHEMA_VERSION,
                    "study_id": "study_1",
                    "optimizer_step_ordinal": step_ordinal,
                    "optimizer_output_digest": output_digest,
                    "experiment_id": experiment_id,
                    "review_session_id": session_id,
                    "reviewer_id": "reviewer",
                    "sealed_mapping_digest": adaptive_review_binding_digest(
                        session_row,
                        study_id="study_1",
                        optimizer_step_ordinal=step_ordinal,
                        optimizer_output_digest=output_digest,
                    ),
                },
                created_at=NOW,
            )
        )


def make_observation(
    *,
    ordinal: int = 1,
    experiment_id: str = "exp_1",
    review_id: str = "review_1",
    display_mapping_digest: str | None = None,
) -> AdaptiveObservation:
    return AdaptiveObservation(
        study_id="study_1",
        ordinal=ordinal,
        optimizer_step_ordinal=ordinal,
        pair=CandidatePair(value_a=2.0, value_b=4.0),
        experiment_id=experiment_id,
        review_id=review_id,
        display_mapping_digest=(
            display_mapping_digest or hashlib.sha256(f"mapping:{review_id}".encode()).hexdigest()
        ),
        outcome=PreferenceOutcome.A,
        created_at=NOW,
    )


def make_observation_request(
    *,
    ordinal: int = 1,
    review_id: str = "review_1",
) -> AdaptiveObservationRequest:
    return AdaptiveObservationRequest(
        study_id="study_1",
        ordinal=ordinal,
        review_id=review_id,
        created_at=NOW,
    )


def make_optimizer_step(
    definition: AdaptiveStudyDefinition,
    observations: tuple[AdaptiveObservation, ...] = (),
) -> AdaptiveOptimizerStep:
    optimizer_observations = tuple(item.preference_observation for item in observations)
    decision = optimize_preference(definition.optimizer_config, optimizer_observations)
    return AdaptiveOptimizerStep(
        study_id=definition.id,
        ordinal=len(observations) + 1,
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
        created_at=NOW,
    )


def transition(
    repository: AdaptiveStudyRepository,
    previous_state: AdaptiveStudyState,
    new_state: AdaptiveStudyState,
) -> AdaptiveStudyEvent:
    return repository.append_event(
        AdaptiveStudyEvent(
            study_id="study_1",
            sequence=len(repository.list_events("study_1")) + 1,
            previous_state=previous_state,
            new_state=new_state,
            created_at=NOW,
        )
    )


def prepare_awaiting_preference(
    repository: AdaptiveStudyRepository,
    definition: AdaptiveStudyDefinition,
    *,
    step: AdaptiveOptimizerStep | None = None,
) -> AdaptiveOptimizerStep:
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    persisted_step = repository.append_optimizer_step(step or make_optimizer_step(definition))
    transition(
        repository,
        AdaptiveStudyState.SEARCH_READY,
        AdaptiveStudyState.GENERATING_PAIR,
    )
    return persisted_step


def test_awaiting_preference_event_retry_verifies_exact_winner_and_rejects_other_races(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = make_definition()
    repository.create(definition)
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database, include_revision=False)
    winner = repository.list_events(definition.id)[-1]

    retried = repository.append_awaiting_preference_or_verify(winner)

    assert retried == winner
    assert len(repository.list_events(definition.id)) == winner.sequence
    mismatched_payload = dict(winner.payload)
    mismatched_payload["sealed_mapping_digest"] = "f" * 64
    mismatched = AdaptiveStudyEvent(
        study_id=winner.study_id,
        sequence=winner.sequence,
        previous_state=winner.previous_state,
        new_state=winner.new_state,
        payload=mismatched_payload,
        created_at=winner.created_at,
    )
    with pytest.raises(AdaptiveStudyConflict, match="sealed review"):
        repository.append_awaiting_preference_or_verify(mismatched)

    transition(
        repository,
        AdaptiveStudyState.AWAITING_PREFERENCE,
        AdaptiveStudyState.PAUSED,
    )
    with pytest.raises(AdaptiveStudyConflict, match="sealed review"):
        repository.append_awaiting_preference_or_verify(winner)


def test_create_and_get_definition_round_trip_canonically(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = make_definition()

    created = repository.create(definition)
    reopened = AdaptiveStudyRepository(Database(database.path)).get(definition.id)

    assert created == definition
    assert reopened == definition
    assert canonical_json(reopened) == canonical_json(definition)


@pytest.mark.parametrize(
    ("malformation", "expected_error"),
    (
        ("manifest_missing_required", "invalid adaptive study manifest"),
        ("manifest_unknown_field", "invalid adaptive study manifest"),
        ("manifest_wrong_nested_type", "invalid adaptive study manifest"),
        ("manifest_unsupported_version", "invalid adaptive study manifest"),
        ("manifest_unsupported_protocol", "invalid adaptive study manifest"),
        ("manifest_noncanonical", "invalid adaptive study manifest"),
        ("config_missing_canonical_field", "invalid adaptive optimizer config"),
        ("config_unknown_field", "invalid adaptive optimizer config"),
        ("config_wrong_type", "invalid adaptive optimizer config"),
        ("config_unsupported_version", "invalid adaptive optimizer config"),
        ("config_unsupported_optimizer", "invalid adaptive optimizer config"),
        ("config_noncanonical", "invalid adaptive optimizer config"),
    ),
)
def test_direct_sql_rejects_definition_json_that_cannot_exactly_hydrate(
    database: Database,
    malformation: str,
    expected_error: str,
) -> None:
    definition = make_definition()
    manifest = definition.base_manifest.model_dump(mode="json")
    config = definition.optimizer_config.model_dump(mode="json")
    if malformation == "manifest_missing_required":
        del manifest["model_id"]
    elif malformation == "manifest_unknown_field":
        manifest["unexpected"] = True
    elif malformation == "manifest_wrong_nested_type":
        manifest["runtime_versions"] = []
    elif malformation == "manifest_unsupported_version":
        manifest["schema_version"] = "manifest.v999"
    elif malformation == "manifest_unsupported_protocol":
        manifest["protocol_version"] = "ab.v999"
    elif malformation == "config_missing_canonical_field":
        del config["posterior_draws"]
    elif malformation == "config_unknown_field":
        config["unexpected"] = True
    elif malformation == "config_wrong_type":
        config["posterior_draws"] = "256"
    elif malformation == "config_unsupported_version":
        config["schema_version"] = "preference-optimizer-config.v999"
    elif malformation == "config_unsupported_optimizer":
        config["optimizer_version"] = "preference-gp-laplace.v999"
    manifest_json = canonical_json(manifest)
    config_json = canonical_json(config)
    if malformation == "manifest_noncanonical":
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
    elif malformation == "config_noncanonical":
        config_json = json.dumps(config, indent=2, ensure_ascii=False)

    with database.connect() as connection:
        with pytest.raises(sqlite3.DatabaseError, match=expected_error):
            connection.execute(
                "INSERT INTO adaptive_studies("
                "id, schema_version, prompt, negative_prompt, seed, base_manifest_json, "
                "optimizer_config_json, input_fingerprint, created_at) "
                "VALUES (?, 'adaptive-study.v1', ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"study_{malformation}",
                    definition.prompt,
                    definition.negative_prompt,
                    definition.seed,
                    manifest_json,
                    config_json,
                    definition.input_fingerprint,
                    definition.created_at.isoformat(),
                ),
            )


def test_direct_sql_accepts_only_canonical_typed_definition_json(
    database: Database,
) -> None:
    definition = make_definition()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO adaptive_studies("
            "id, schema_version, prompt, negative_prompt, seed, base_manifest_json, "
            "optimizer_config_json, input_fingerprint, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                definition.id,
                definition.schema_version,
                definition.prompt,
                definition.negative_prompt,
                definition.seed,
                canonical_json(definition.base_manifest),
                canonical_json(definition.optimizer_config),
                definition.input_fingerprint,
                definition.created_at.isoformat(),
            ),
        )

    assert AdaptiveStudyRepository(Database(database.path)).get(definition.id) == definition


def test_list_definitions_uses_stable_created_at_and_id_order(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    later = make_definition("study_b", seed=43, created_at=datetime(2026, 7, 20, 13, 0, tzinfo=UTC))
    earlier_b = make_definition("study_c", seed=44)
    earlier_a = make_definition("study_a", seed=45)
    for definition in (later, earlier_b, earlier_a):
        repository.create(definition)

    assert repository.list() == (earlier_a, earlier_b, later)


def test_distinct_studies_may_repeat_the_same_frozen_inputs(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    first = make_definition("study_a")
    repeated = first.model_copy(
        update={
            "id": "study_b",
            "created_at": datetime(2026, 7, 20, 13, 0, tzinfo=UTC),
        }
    )

    repository.create(first)
    repository.create(repeated)

    assert repository.list() == (first, repeated)


def test_append_events_enforces_and_returns_canonical_sequence(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    repository.create(make_definition())
    first = AdaptiveStudyEvent(
        study_id="study_1",
        sequence=1,
        previous_state=AdaptiveStudyState.DRAFT,
        new_state=AdaptiveStudyState.SEARCH_READY,
        payload={"reason": "definition_frozen"},
        created_at=NOW,
    )
    second = AdaptiveStudyEvent(
        study_id="study_1",
        sequence=2,
        previous_state=AdaptiveStudyState.SEARCH_READY,
        new_state=AdaptiveStudyState.GENERATING_PAIR,
        payload={"optimizer_step_ordinal": 1},
        created_at=NOW,
    )

    assert repository.append_event(first) == first
    assert repository.append_event(second) == second

    assert repository.list_events("study_1") == (first, second)


def test_owner_binding_is_versioned_immutable_and_fingerprint_bound(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())

    event = repository.bind_owner(
        definition.id,
        input_fingerprint=definition.input_fingerprint,
        reviewer_id="reviewer",
    )

    assert repository.owner(definition.id) == "reviewer"
    assert event.payload == {
        "owner_binding_schema_version": "adaptive-study-owner-binding.v1",
        "reviewer_id": "reviewer",
        "input_fingerprint": definition.input_fingerprint,
    }
    with pytest.raises(AdaptiveStudyConflict, match="expected event sequence 2"):
        repository.bind_owner(
            definition.id,
            input_fingerprint=definition.input_fingerprint,
            reviewer_id="intruder",
        )
    assert repository.owner(definition.id) == "reviewer"

    other = repository.create(make_definition("study_2"))
    with pytest.raises(AdaptiveStudyConflict, match="fingerprint mismatch"):
        repository.bind_owner(
            other.id,
            input_fingerprint="f" * 64,
            reviewer_id="reviewer",
        )
    assert repository.list_events(other.id) == ()


def test_append_observation_binds_review_and_canonical_pair(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)
    observation = repository.append_observation(make_observation_request())

    assert repository.list_observations("study_1") == (observation,)


def test_observation_consume_or_verify_accepts_exact_retry_and_rejects_mismatch(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)
    request = make_observation_request()

    first, consumed = repository.consume_observation_or_verify(request)
    retried, retry_consumed = repository.consume_observation_or_verify(
        request.model_copy(update={"created_at": NOW + timedelta(seconds=1)})
    )

    assert consumed is True
    assert retry_consumed is False
    assert retried == first
    with pytest.raises(AdaptiveStudyConflict, match="observation retry"):
        repository.consume_observation_or_verify(
            request.model_copy(update={"review_id": "review_other"})
        )


def test_observation_rejects_review_from_a_different_experiment(database: Database) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AdaptiveObservationRequest(
            study_id="study_1",
            ordinal=1,
            review_id="review_1",
            experiment_id="exp_2",
            created_at=NOW,
        )


def test_append_optimizer_step_round_trips_full_decision(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    step = make_optimizer_step(definition)

    assert repository.append_optimizer_step(step) == step

    assert repository.list_optimizer_steps(definition.id) == (step,)


def test_event_sequence_and_prior_state_conflicts_roll_back_completely(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    repository.create(make_definition())
    skipped = AdaptiveStudyEvent(
        study_id="study_1",
        sequence=2,
        previous_state=AdaptiveStudyState.DRAFT,
        new_state=AdaptiveStudyState.SEARCH_READY,
        created_at=NOW,
    )

    with pytest.raises(AdaptiveStudyConflict, match="expected event sequence 1"):
        repository.append_event(skipped)
    assert repository.list_events("study_1") == ()

    first = skipped.model_copy(update={"sequence": 1})
    repository.append_event(first)
    stale = AdaptiveStudyEvent(
        study_id="study_1",
        sequence=2,
        previous_state=AdaptiveStudyState.DRAFT,
        new_state=AdaptiveStudyState.GENERATING_PAIR,
        created_at=NOW,
    )
    with pytest.raises(AdaptiveStudyConflict, match="expected prior state search_ready"):
        repository.append_event(stale)
    assert repository.list_events("study_1") == (first,)


def test_observation_ordinal_conflict_and_duplicate_review_roll_back(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)
    first = repository.append_observation(make_observation_request())

    with pytest.raises(AdaptiveStudyConflict, match="expected observation ordinal 2"):
        repository.append_observation(make_observation_request())
    assert repository.list_observations("study_1") == (first,)


def test_observations_cannot_exceed_the_frozen_comparison_budget(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    observations: list[AdaptiveObservation] = []
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    step = repository.append_optimizer_step(make_optimizer_step(definition))
    for ordinal in range(1, 11):
        assert step.selected_pair is not None
        previous_state = (
            AdaptiveStudyState.SEARCH_READY
            if ordinal == 1
            else AdaptiveStudyState.UPDATING_POSTERIOR
        )
        transition(repository, previous_state, AdaptiveStudyState.GENERATING_PAIR)
        experiment_id = f"exp_{ordinal}"
        review_id = f"review_{ordinal}"
        persist_review_binding(
            database,
            experiment_id=experiment_id,
            review_id=review_id,
            left_guidance=step.selected_pair.value_a,
            right_guidance=step.selected_pair.value_b,
            display_preference="tie",
        )
        observations.append(
            repository.append_observation(
                make_observation_request(ordinal=ordinal, review_id=review_id)
            )
        )
        transition(
            repository,
            AdaptiveStudyState.AWAITING_PREFERENCE,
            AdaptiveStudyState.UPDATING_POSTERIOR,
        )
        step = repository.append_optimizer_step(
            make_optimizer_step(definition, tuple(observations))
        )

    assert step.stop_reason is StopReason.SEARCH_BUDGET_EXHAUSTED
    transition(
        repository,
        AdaptiveStudyState.UPDATING_POSTERIOR,
        AdaptiveStudyState.SEARCH_BUDGET_EXHAUSTED,
    )
    with pytest.raises(AdaptiveStudyConflict, match="search_budget_exhausted"):
        repository.append_observation(make_observation_request(ordinal=11, review_id="review_11"))
    assert repository.list_observations("study_1") == tuple(observations)


def test_optimizer_step_rejects_duplicate_and_noncanonical_input(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    first = make_optimizer_step(definition)
    repository.append_optimizer_step(first)

    with pytest.raises(AdaptiveStudyConflict, match="expected optimizer step ordinal 2"):
        repository.append_optimizer_step(first)

    transition(
        repository,
        AdaptiveStudyState.SEARCH_READY,
        AdaptiveStudyState.GENERATING_PAIR,
    )
    persist_review_binding(database)
    observation = repository.append_observation(make_observation_request())
    transition(
        repository,
        AdaptiveStudyState.AWAITING_PREFERENCE,
        AdaptiveStudyState.UPDATING_POSTERIOR,
    )
    second = make_optimizer_step(definition, (observation,))
    corrupt = second.model_copy(update={"input_digest": "f" * 64})
    with pytest.raises(AdaptiveStudyConflict, match="input digest"):
        repository.append_optimizer_step(corrupt)
    assert repository.list_optimizer_steps("study_1") == (first,)

    repository.append_optimizer_step(second)
    assert repository.list_optimizer_steps("study_1") == (first, second)


def test_optimizer_input_digest_excludes_review_and_display_identity() -> None:
    config = OptimizerConfig(posterior_draws=256)
    first = make_observation(
        review_id="review_1",
        experiment_id="exp_1",
        display_mapping_digest="a" * 64,
    )
    second = make_observation(
        review_id="review_2",
        experiment_id="exp_2",
        display_mapping_digest="b" * 64,
    )

    assert adaptive_optimizer_input_digest(
        config, (first.preference_observation,)
    ) == adaptive_optimizer_input_digest(config, (second.preference_observation,))


def test_adaptive_models_are_deeply_immutable_and_require_utc_timestamps() -> None:
    definition = make_definition()
    event = AdaptiveStudyEvent(
        study_id="study_1",
        sequence=1,
        previous_state=AdaptiveStudyState.DRAFT,
        new_state=AdaptiveStudyState.SEARCH_READY,
        payload={"nested": {"values": [1, 2]}},
        created_at=NOW,
    )

    with pytest.raises(ValidationError, match="Instance is frozen"):
        definition.prompt = "changed"
    with pytest.raises(TypeError):
        cast(Any, definition.base_manifest.runtime_versions)["python"] = "3.13"
    with pytest.raises(TypeError):
        cast(Any, event.payload)["new"] = True
    with pytest.raises(TypeError):
        cast(Any, cast(Any, event.payload["nested"])["values"])[0] = 3
    with pytest.raises(ValidationError, match="timezone-aware"):
        event.model_copy(update={"created_at": datetime(2026, 7, 20, 12, 0)})


def test_optimizer_step_rejects_noncanonical_ledger_and_nonfinite_posterior() -> None:
    definition = make_definition()
    step = make_optimizer_step(definition)
    reversed_score = AcquisitionScore(
        pair=CandidatePair(value_a=4.0, value_b=2.0),
        expected_information_bits=0.1,
    )
    reversed_decision = step.decision.model_copy(update={"candidate_scores": (reversed_score,)})
    with pytest.raises(ValidationError, match="acquisition ledger pairs must be canonical"):
        AdaptiveOptimizerStep(
            **{
                **step.model_dump(round_trip=True),
                "acquisition_ledger": reversed_decision.candidate_scores,
                "output_digest": adaptive_optimizer_output_digest(reversed_decision),
            }
        )

    nonfinite_posterior = step.posterior.model_copy(
        update={"mean": (float("inf"), *step.posterior.mean[1:])}
    )
    nonfinite_decision = step.decision.model_copy(update={"posterior": nonfinite_posterior})
    with pytest.raises(ValidationError, match="posterior values must be finite"):
        AdaptiveOptimizerStep(
            **{
                **step.model_dump(round_trip=True),
                "posterior": nonfinite_posterior,
                "output_digest": adaptive_optimizer_output_digest(nonfinite_decision),
            }
        )


@pytest.mark.parametrize(
    ("table", "update_sql", "delete_sql"),
    (
        (
            "adaptive_studies",
            "UPDATE adaptive_studies SET prompt = prompt",
            "DELETE FROM adaptive_studies",
        ),
        (
            "adaptive_study_events",
            "UPDATE adaptive_study_events SET payload_json = payload_json",
            "DELETE FROM adaptive_study_events",
        ),
        (
            "adaptive_observations",
            "UPDATE adaptive_observations SET outcome = outcome",
            "DELETE FROM adaptive_observations",
        ),
        (
            "adaptive_optimizer_steps",
            "UPDATE adaptive_optimizer_steps SET output_digest = output_digest",
            "DELETE FROM adaptive_optimizer_steps",
        ),
    ),
)
def test_evidence_tables_reject_update_and_delete(
    database: Database,
    table: str,
    update_sql: str,
    delete_sql: str,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    repository.append_event(
        AdaptiveStudyEvent(
            study_id="study_1",
            sequence=1,
            previous_state=AdaptiveStudyState.DRAFT,
            new_state=AdaptiveStudyState.SEARCH_READY,
            created_at=NOW,
        )
    )
    repository.append_optimizer_step(make_optimizer_step(definition))
    transition(
        repository,
        AdaptiveStudyState.SEARCH_READY,
        AdaptiveStudyState.GENERATING_PAIR,
    )
    persist_review_binding(database)
    repository.append_observation(make_observation_request())

    with database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(update_sql)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(delete_sql)
        count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    assert count == (3 if table == "adaptive_study_events" else 1)


def test_reopen_preserves_canonical_json_and_digest_bytes(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    event = repository.append_event(
        AdaptiveStudyEvent(
            study_id="study_1",
            sequence=1,
            previous_state=AdaptiveStudyState.DRAFT,
            new_state=AdaptiveStudyState.SEARCH_READY,
            payload={"z": 1, "a": {"β": [2, 1]}},
            created_at=NOW,
        )
    )
    step = repository.append_optimizer_step(make_optimizer_step(definition))
    with database.connect() as connection:
        before = (
            tuple(
                str(value)
                for value in connection.execute(
                    "SELECT base_manifest_json, optimizer_config_json FROM adaptive_studies"
                ).fetchone()
            )
            + tuple(
                str(value)
                for value in connection.execute(
                    "SELECT payload_json FROM adaptive_study_events"
                ).fetchone()
            )
            + tuple(
                str(value)
                for value in connection.execute(
                    "SELECT input_digest, output_digest, posterior_json, "
                    "acquisition_ledger_json, selected_pair_json "
                    "FROM adaptive_optimizer_steps"
                ).fetchone()
            )
        )

    reopened = AdaptiveStudyRepository(Database(database.path))
    assert reopened.get("study_1") == definition
    assert reopened.list_events("study_1") == (event,)
    assert reopened.list_optimizer_steps("study_1") == (step,)
    with database.connect() as connection:
        after = (
            tuple(
                str(value)
                for value in connection.execute(
                    "SELECT base_manifest_json, optimizer_config_json FROM adaptive_studies"
                ).fetchone()
            )
            + tuple(
                str(value)
                for value in connection.execute(
                    "SELECT payload_json FROM adaptive_study_events"
                ).fetchone()
            )
            + tuple(
                str(value)
                for value in connection.execute(
                    "SELECT input_digest, output_digest, posterior_json, "
                    "acquisition_ledger_json, selected_pair_json "
                    "FROM adaptive_optimizer_steps"
                ).fetchone()
            )
        )
    assert after == before
    assert before == (
        canonical_json(definition.base_manifest),
        canonical_json(definition.optimizer_config),
        canonical_json(event.model_dump(mode="json")["payload"]),
        adaptive_optimizer_input_digest(definition.optimizer_config, ()),
        adaptive_optimizer_output_digest(step.decision),
        canonical_json(step.posterior),
        json.dumps(
            [score.model_dump(mode="json") for score in step.acquisition_ledger],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        canonical_json(step.selected_pair) if step.selected_pair is not None else "None",
    )


def test_database_constraints_reject_invalid_child_evidence(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    step = make_optimizer_step(definition)
    with database.connect() as connection:
        invalid_events = (
            (
                "INSERT INTO adaptive_study_events("
                "study_id, schema_version, sequence, previous_state, new_state, "
                "payload_json, created_at) VALUES (?, 'adaptive-study-event.v1', "
                "0, 'draft', 'search_ready', '{}', ?)",
                (definition.id, NOW.isoformat()),
                "invalid adaptive study event sequence",
            ),
            (
                "INSERT INTO adaptive_study_events("
                "study_id, schema_version, sequence, previous_state, new_state, "
                "payload_json, created_at) VALUES (?, 'adaptive-study-event.v1', "
                "0.5, 'draft', 'search_ready', '{}', ?)",
                (definition.id, NOW.isoformat()),
                "invalid adaptive study event sequence",
            ),
            (
                "INSERT INTO adaptive_study_events("
                "study_id, schema_version, sequence, previous_state, new_state, "
                "payload_json, created_at) VALUES (?, 'adaptive-study-event.v1', "
                "1, 'draft', 'unknown', '{}', ?)",
                (definition.id, NOW.isoformat()),
                "invalid adaptive study transition",
            ),
            (
                "INSERT INTO adaptive_study_events("
                "study_id, schema_version, sequence, previous_state, new_state, "
                "payload_json, created_at) VALUES (?, 'adaptive-study-event.v1', "
                "1, 'draft', 'search_ready', '[]', ?)",
                (definition.id, NOW.isoformat()),
                "CHECK constraint failed",
            ),
        )
        for statement, parameters, expected_error in invalid_events:
            with pytest.raises(sqlite3.DatabaseError, match=expected_error):
                connection.execute(statement, parameters)

    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    with database.connect() as connection:
        invalid_optimizer_steps = (
            (
                "INSERT INTO adaptive_optimizer_steps("
                "study_id, schema_version, ordinal, input_digest, output_digest, "
                "posterior_json, acquisition_ledger_json, selected_pair_json, stop_reason, "
                "disposition, comparison_count, maximum_information_bits, created_at) "
                "VALUES (?, 'adaptive-optimizer-step.v1', 1, ?, ?, ?, '{}', ?, "
                "'continue', NULL, 0, 0.0, ?)",
                (
                    definition.id,
                    step.input_digest,
                    step.output_digest,
                    canonical_json(step.posterior),
                    canonical_json(step.selected_pair),
                    NOW.isoformat(),
                ),
                "CHECK constraint failed",
            ),
            (
                "INSERT INTO adaptive_optimizer_steps("
                "study_id, schema_version, ordinal, input_digest, output_digest, "
                "posterior_json, acquisition_ledger_json, selected_pair_json, stop_reason, "
                "disposition, comparison_count, maximum_information_bits, created_at) "
                "VALUES (?, 'adaptive-optimizer-step.v1', 1.5, ?, ?, ?, '[]', ?, "
                "'continue', NULL, 0.5, 0.0, ?)",
                (
                    definition.id,
                    step.input_digest,
                    step.output_digest,
                    canonical_json(step.posterior),
                    canonical_json(step.selected_pair),
                    NOW.isoformat(),
                ),
                "invalid adaptive optimizer step ordinal",
            ),
            (
                "INSERT INTO adaptive_optimizer_steps("
                "study_id, schema_version, ordinal, input_digest, output_digest, "
                "posterior_json, acquisition_ledger_json, selected_pair_json, stop_reason, "
                "disposition, comparison_count, maximum_information_bits, created_at) "
                "VALUES (?, 'adaptive-optimizer-step.v1', 1, ?, ?, '{}', '[]', ?, "
                "'continue', NULL, 0, 0.0, ?)",
                (
                    definition.id,
                    step.input_digest,
                    step.output_digest,
                    canonical_json(step.selected_pair),
                    NOW.isoformat(),
                ),
                "invalid adaptive optimizer posterior object",
            ),
        )
        for statement, parameters, expected_error in invalid_optimizer_steps:
            with pytest.raises(sqlite3.DatabaseError, match=expected_error):
                connection.execute(statement, parameters)

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                "INSERT INTO adaptive_study_events VALUES ('missing', "
                "'adaptive-study-event.v1', 1, 'draft', 'search_ready', '{}', ?)",
                (NOW.isoformat(),),
            )


def test_lifecycle_rejects_disallowed_state_jump_and_rolls_back(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    repository.create(make_definition())
    jump = AdaptiveStudyEvent(
        study_id="study_1",
        sequence=1,
        previous_state=AdaptiveStudyState.DRAFT,
        new_state=AdaptiveStudyState.GENERATING_PAIR,
        created_at=NOW,
    )

    with pytest.raises(AdaptiveStudyConflict, match="transition draft -> generating_pair"):
        repository.append_event(jump)

    assert repository.list_events("study_1") == ()


def test_optimizer_step_rejects_draft_state_and_rolls_back(database: Database) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())

    with pytest.raises(AdaptiveStudyConflict, match="optimizer step.*draft"):
        repository.append_optimizer_step(make_optimizer_step(definition))

    assert repository.list_optimizer_steps("study_1") == ()


def test_observation_request_has_no_caller_controlled_evidence_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AdaptiveObservationRequest(
            study_id="study_1",
            ordinal=1,
            review_id="review_1",
            created_at=NOW,
            pair={"value_a": 1.0, "value_b": 5.0},
            outcome="b",
            display_mapping_digest="f" * 64,
        )


def test_observation_is_derived_from_latest_persisted_blind_review(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    selected = prepare_awaiting_preference(repository, definition)
    persist_review_binding(
        database,
        left_guidance=4.0,
        right_guidance=2.0,
        display_preference="left",
    )

    observation = repository.append_observation(make_observation_request())

    assert observation.pair == selected.selected_pair == CandidatePair(value_a=2.0, value_b=4.0)
    assert observation.optimizer_step_ordinal == 1
    assert observation.experiment_id == "exp_1"
    assert observation.review_id == "review_1"
    assert observation.outcome is PreferenceOutcome.B
    assert len(observation.display_mapping_digest) == 64
    assert repository.list_observations("study_1") == (observation,)


def test_observation_rejects_missing_or_superseded_review_and_rolls_back(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database, include_revision=False)

    with pytest.raises(AdaptiveStudyConflict, match="review does not exist"):
        repository.append_observation(make_observation_request())
    assert repository.list_observations("study_1") == ()

    with database.connect() as connection:
        connection.execute(
            "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
            "display_preference, created_at, revision) "
            "VALUES ('review_1', 'session_1', 'var_a_1', 'left', 'now', 1)"
        )
        connection.execute(
            "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
            "display_preference, created_at, revision, supersedes_review_id) "
            "VALUES ('review_2', 'session_1', 'var_b_1', 'right', 'now', 2, 'review_1')"
        )

    with pytest.raises(AdaptiveStudyConflict, match="latest review revision"):
        repository.append_observation(make_observation_request())
    assert repository.list_observations("study_1") == ()


def test_observation_rejects_mismatched_experiment_session_provenance(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO experiments VALUES "
            "('exp_other', 'manifest.v1', 'guidance_scale', "
            "'[\"guidance_scale\"]', 'now')"
        )
        connection.execute(
            "UPDATE review_sessions SET experiment_id = 'exp_other' WHERE id = 'session_1'"
        )

    with pytest.raises(AdaptiveStudyConflict, match="active review binding"):
        repository.append_observation(make_observation_request())

    assert repository.list_observations("study_1") == ()


def test_observation_requires_matching_unconsumed_nonterminal_optimizer_step(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    repository.create(make_definition())
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    transition(
        repository,
        AdaptiveStudyState.SEARCH_READY,
        AdaptiveStudyState.GENERATING_PAIR,
    )
    persist_review_binding(database)

    with pytest.raises(AdaptiveStudyConflict, match="optimizer step 1 does not exist"):
        repository.append_observation(make_observation_request())
    assert repository.list_observations("study_1") == ()

    other_database = Database(database.path.parent / "stopped.sqlite3")
    other_database.migrate()
    stopped_repository = AdaptiveStudyRepository(other_database)
    stopped_definition = stopped_repository.create(make_definition())
    continuing = make_optimizer_step(stopped_definition)
    stopped_decision = continuing.decision.model_copy(
        update={
            "selected_pair": None,
            "stopping": continuing.decision.stopping.model_copy(
                update={
                    "stop": True,
                    "reason": StopReason.CANDIDATE_SPACE_EXHAUSTED,
                }
            ),
        }
    )
    stopped = continuing.model_copy(
        update={
            "selected_pair": None,
            "stop_reason": StopReason.CANDIDATE_SPACE_EXHAUSTED,
            "output_digest": adaptive_optimizer_output_digest(stopped_decision),
        }
    )
    prepare_awaiting_preference(stopped_repository, stopped_definition, step=stopped)
    persist_review_binding(other_database)

    with pytest.raises(AdaptiveStudyConflict, match="did not select a comparison pair"):
        stopped_repository.append_observation(make_observation_request())
    assert stopped_repository.list_observations("study_1") == ()


def test_observation_rejects_optimizer_pair_mismatch_and_duplicate_consumption(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    initial = make_optimizer_step(definition)
    mismatched_pair = CandidatePair(value_a=1.0, value_b=5.0)
    mismatched_decision = initial.decision.model_copy(update={"selected_pair": mismatched_pair})
    mismatched_step = initial.model_copy(
        update={
            "selected_pair": mismatched_pair,
            "output_digest": adaptive_optimizer_output_digest(mismatched_decision),
        }
    )
    prepare_awaiting_preference(repository, definition, step=mismatched_step)
    persist_review_binding(database)

    with pytest.raises(AdaptiveStudyConflict, match="does not match optimizer-selected pair"):
        repository.append_observation(make_observation_request())
    assert repository.list_observations("study_1") == ()

    other_database = Database(database.path.parent / "consumed.sqlite3")
    other_database.migrate()
    other_repository = AdaptiveStudyRepository(other_database)
    other_definition = other_repository.create(make_definition())
    prepare_awaiting_preference(other_repository, other_definition)
    persist_review_binding(other_database)
    persisted = other_repository.append_observation(make_observation_request())

    with pytest.raises(AdaptiveStudyConflict, match="expected observation ordinal 2"):
        other_repository.append_observation(make_observation_request())
    assert other_repository.list_observations("study_1") == (persisted,)


def test_paused_and_terminal_states_reject_new_evidence_and_invalid_recovery(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    repository.append_optimizer_step(make_optimizer_step(definition))
    transition(
        repository,
        AdaptiveStudyState.SEARCH_READY,
        AdaptiveStudyState.GENERATING_PAIR,
    )
    transition(repository, AdaptiveStudyState.GENERATING_PAIR, AdaptiveStudyState.PAUSED)

    with pytest.raises(AdaptiveStudyConflict, match="observation.*paused"):
        repository.append_observation(make_observation_request())
    with pytest.raises(AdaptiveStudyConflict, match="optimizer step.*paused"):
        repository.append_optimizer_step(make_optimizer_step(definition))
    with pytest.raises(AdaptiveStudyConflict, match="transition paused -> completed"):
        transition(repository, AdaptiveStudyState.PAUSED, AdaptiveStudyState.COMPLETED)

    transition(repository, AdaptiveStudyState.PAUSED, AdaptiveStudyState.GENERATING_PAIR)
    transition(
        repository,
        AdaptiveStudyState.GENERATING_PAIR,
        AdaptiveStudyState.AWAITING_PREFERENCE,
    )
    transition(
        repository,
        AdaptiveStudyState.AWAITING_PREFERENCE,
        AdaptiveStudyState.UPDATING_POSTERIOR,
    )
    transition(
        repository,
        AdaptiveStudyState.UPDATING_POSTERIOR,
        AdaptiveStudyState.SEARCH_CONVERGED,
    )
    transition(
        repository,
        AdaptiveStudyState.SEARCH_CONVERGED,
        AdaptiveStudyState.VALIDATING,
    )
    transition(repository, AdaptiveStudyState.VALIDATING, AdaptiveStudyState.COMPLETED)
    event_count = len(repository.list_events("study_1"))

    with pytest.raises(AdaptiveStudyConflict, match="optimizer step.*completed"):
        repository.append_optimizer_step(make_optimizer_step(definition))
    with pytest.raises(AdaptiveStudyConflict, match="observation.*completed"):
        repository.append_observation(make_observation_request())
    with pytest.raises(AdaptiveStudyConflict, match="transition completed -> paused"):
        transition(repository, AdaptiveStudyState.COMPLETED, AdaptiveStudyState.PAUSED)

    assert len(repository.list_events("study_1")) == event_count
    assert repository.list_observations("study_1") == ()
    assert len(repository.list_optimizer_steps("study_1")) == 1


@pytest.mark.parametrize(
    "malformation",
    (
        "posterior_scalar_type",
        "posterior_covariance_shape",
        "posterior_unknown_key",
        "concentration_interval_element_type",
        "argmax_probability_element_type",
        "ledger_pair_type",
        "ledger_unknown_key",
        "selected_pair_type",
    ),
)
def test_direct_sql_rejects_nested_optimizer_json_that_cannot_hydrate(
    database: Database,
    malformation: str,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    step = make_optimizer_step(definition)
    posterior = step.posterior.model_dump(mode="json")
    ledger = [score.model_dump(mode="json") for score in step.acquisition_ledger]
    if not ledger:
        ledger = [
            {
                "pair": {"value_a": 2.0, "value_b": 4.0},
                "expected_information_bits": 0.1,
            }
        ]
    selected_pair = step.selected_pair.model_dump(mode="json") if step.selected_pair else None
    if malformation == "posterior_scalar_type":
        posterior["grid"][0] = "not-a-number"
    elif malformation == "posterior_covariance_shape":
        posterior["covariance"][0] = []
    elif malformation == "posterior_unknown_key":
        posterior["unexpected"] = True
    elif malformation == "concentration_interval_element_type":
        posterior["concentration_interval"][0] = "not-a-number"
    elif malformation == "argmax_probability_element_type":
        posterior["argmax_probability"][0] = "not-a-number"
    elif malformation == "ledger_pair_type":
        ledger[0]["pair"]["value_a"] = "not-a-number"
    elif malformation == "ledger_unknown_key":
        ledger[0]["unexpected"] = True
    elif malformation == "selected_pair_type":
        assert selected_pair is not None
        selected_pair["value_a"] = "not-a-number"

    with database.connect() as connection:
        with pytest.raises(sqlite3.DatabaseError, match="invalid adaptive optimizer"):
            connection.execute(
                "INSERT INTO adaptive_optimizer_steps("
                "study_id, schema_version, ordinal, input_digest, output_digest, "
                "posterior_json, acquisition_ledger_json, selected_pair_json, stop_reason, "
                "disposition, comparison_count, maximum_information_bits, created_at) "
                "VALUES (?, 'adaptive-optimizer-step.v1', 1, ?, ?, ?, ?, ?, "
                "'continue', NULL, 0, 0.0, ?)",
                (
                    definition.id,
                    step.input_digest,
                    step.output_digest,
                    json.dumps(posterior, sort_keys=True, separators=(",", ":")),
                    json.dumps(ledger, sort_keys=True, separators=(",", ":")),
                    json.dumps(selected_pair, sort_keys=True, separators=(",", ":")),
                    NOW.isoformat(),
                ),
            )

    assert repository.list_optimizer_steps("study_1") == ()


@pytest.mark.parametrize(
    ("value_a", "value_b", "outcome"),
    (
        (1.0, 5.0, "a"),
        (2.0, 4.0, "b"),
    ),
)
def test_direct_sql_cannot_inject_pair_or_outcome_not_derived_from_review(
    database: Database,
    value_a: float,
    value_b: float,
    outcome: str,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)

    with database.connect() as connection:
        with pytest.raises(sqlite3.DatabaseError, match="invalid adaptive observation"):
            connection.execute(
                "INSERT INTO adaptive_observations VALUES ("
                "'study_1', 'adaptive-observation.v1', 1, 1, ?, ?, 'exp_1', "
                "'review_1', ?, ?, ?)",
                (value_a, value_b, "a" * 64, outcome, NOW.isoformat()),
            )

    assert repository.list_observations("study_1") == ()


def test_direct_sql_rejects_forged_observation_provenance_digest(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)

    with database.connect() as connection:
        with pytest.raises(
            sqlite3.DatabaseError,
            match="invalid adaptive observation provenance digest",
        ):
            connection.execute(
                "INSERT INTO adaptive_observations("
                "study_id, schema_version, ordinal, optimizer_step_ordinal, value_a, value_b, "
                "experiment_id, review_id, display_mapping_digest, outcome, created_at) "
                "VALUES ('study_1', 'adaptive-observation.v1', 1, 1, 2.0, 4.0, "
                "'exp_1', 'review_1', ?, 'a', ?)",
                ("f" * 64, NOW.isoformat()),
            )

    assert repository.list_observations("study_1") == ()


def test_valid_repository_digest_is_accepted_by_direct_sql_boundary(
    database: Database,
) -> None:
    source_database = Database(database.path.parent / "digest-source.sqlite3")
    source_database.migrate()
    source_repository = AdaptiveStudyRepository(source_database)
    source_definition = source_repository.create(make_definition())
    prepare_awaiting_preference(source_repository, source_definition)
    persist_review_binding(source_database)
    expected = source_repository.append_observation(make_observation_request())

    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO adaptive_observations("
            "study_id, schema_version, ordinal, optimizer_step_ordinal, value_a, value_b, "
            "experiment_id, review_id, display_mapping_digest, outcome, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                expected.study_id,
                expected.schema_version,
                expected.ordinal,
                expected.optimizer_step_ordinal,
                expected.pair.value_a,
                expected.pair.value_b,
                expected.experiment_id,
                expected.review_id,
                expected.display_mapping_digest,
                expected.outcome.value,
                expected.created_at.isoformat(),
            ),
        )

    assert repository.list_observations("study_1") == (expected,)


def test_observation_provenance_digest_binds_new_row_fields(database: Database) -> None:
    source_database = Database(database.path.parent / "digest-row-fields.sqlite3")
    source_database.migrate()
    source_repository = AdaptiveStudyRepository(source_database)
    source_definition = source_repository.create(make_definition())
    prepare_awaiting_preference(source_repository, source_definition)
    persist_review_binding(source_database)
    expected = source_repository.append_observation(make_observation_request())

    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    prepare_awaiting_preference(repository, definition)
    persist_review_binding(database)
    with database.connect() as connection:
        with pytest.raises(
            sqlite3.DatabaseError,
            match="invalid adaptive observation provenance digest",
        ):
            connection.execute(
                "INSERT INTO adaptive_observations("
                "study_id, schema_version, ordinal, optimizer_step_ordinal, value_a, value_b, "
                "experiment_id, review_id, display_mapping_digest, outcome, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    expected.study_id,
                    expected.schema_version,
                    expected.ordinal,
                    expected.optimizer_step_ordinal,
                    expected.pair.value_a,
                    expected.pair.value_b,
                    expected.experiment_id,
                    expected.review_id,
                    expected.display_mapping_digest,
                    expected.outcome.value,
                    NOW.replace(minute=1).isoformat(),
                ),
            )

    assert repository.list_observations("study_1") == ()


def test_reopen_recomputes_optimizer_input_digest_from_persisted_evidence(
    database: Database,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    step = make_optimizer_step(definition)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO adaptive_optimizer_steps("
            "study_id, schema_version, ordinal, input_digest, output_digest, posterior_json, "
            "acquisition_ledger_json, selected_pair_json, stop_reason, disposition, "
            "comparison_count, maximum_information_bits, created_at) "
            "VALUES (?, 'adaptive-optimizer-step.v1', 1, ?, ?, ?, ?, ?, "
            "'continue', NULL, 0, 0.0, ?)",
            (
                definition.id,
                "f" * 64,
                step.output_digest,
                canonical_json(step.posterior),
                json.dumps(
                    [score.model_dump(mode="json") for score in step.acquisition_ledger],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                canonical_json(step.selected_pair) if step.selected_pair is not None else None,
                NOW.isoformat(),
            ),
        )

    with pytest.raises(AdaptiveStudyConflict, match="persisted optimizer input digest"):
        AdaptiveStudyRepository(Database(database.path)).list_optimizer_steps("study_1")


@pytest.mark.parametrize(
    ("stop_reason", "disposition", "selected_pair_json"),
    (
        ("continue", None, None),
        ("candidate_space_exhausted", None, '{"value_a":2.0,"value_b":4.0}'),
        ("search_budget_exhausted", None, None),
        ("search_converged", "inconclusive_within_budget", None),
    ),
)
def test_direct_sql_rejects_incompatible_stop_and_selected_pair_evidence(
    database: Database,
    stop_reason: str,
    disposition: str | None,
    selected_pair_json: str | None,
) -> None:
    repository = AdaptiveStudyRepository(database)
    definition = repository.create(make_definition())
    transition(repository, AdaptiveStudyState.DRAFT, AdaptiveStudyState.SEARCH_READY)
    step = make_optimizer_step(definition)

    with database.connect() as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(
                "INSERT INTO adaptive_optimizer_steps("
                "study_id, schema_version, ordinal, input_digest, output_digest, posterior_json, "
                "acquisition_ledger_json, selected_pair_json, stop_reason, disposition, "
                "comparison_count, maximum_information_bits, created_at) "
                "VALUES (?, 'adaptive-optimizer-step.v1', 1, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, ?)",
                (
                    definition.id,
                    step.input_digest,
                    step.output_digest,
                    canonical_json(step.posterior),
                    json.dumps(
                        [score.model_dump(mode="json") for score in step.acquisition_ledger],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    selected_pair_json,
                    stop_reason,
                    disposition,
                    NOW.isoformat(),
                ),
            )
