from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from video_lab.adaptive_study_models import (
    AdaptiveStudyDefinition,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import AdaptiveStudyRepository
from video_lab.database import Database
from video_lab.domain import Manifest
from video_lab.preference_optimizer import OptimizerConfig
from video_lab.validation_models import (
    ValidationArchetype,
    ValidationCampaignDefinition,
    ValidationCampaignState,
    ValidationChoice,
    ValidationDisposition,
    ValidationEvent,
    ValidationExperimentBinding,
    ValidationJudgment,
    ValidationPanelInput,
    ValidationPrompt,
    ValidationReason,
    ValidationResult,
)
from video_lab.validation_panel import build_validation_panel
from video_lab.validation_repository import ValidationConflict, ValidationRepository

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def manifest() -> Manifest:
    return Manifest(
        model_id="org/model",
        model_revision="abc123",
        model_checksum="sha256:weights",
        adapter_id="diffusers-mps",
        adapter_version="0.1.0",
        runtime_versions={"python": "3.12"},
        prompt="a blue-haired character walking",
        negative_prompt="flicker",
        seed=42,
        width=512,
        height=288,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=3.0,
        scheduler="flow-match-euler",
        dtype="float16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="unified",
        hardware_fingerprint="test",
        os_fingerprint="test",
    )


def definition(campaign_id: str = "validation_1") -> ValidationCampaignDefinition:
    return ValidationCampaignDefinition(
        id=campaign_id,
        stage_1_study_id="study_1",
        stage_1_evidence_digest="a" * 64,
        base_manifest=manifest(),
        candidates={"winner": 3.0, "lower": 2.0, "higher": 4.0, "distant": 5.0},
        panel_input=ValidationPanelInput(
            prompts=tuple(
                ValidationPrompt(
                    archetype=archetype,
                    prompt=f"character {archetype.value}",
                    negative_prompt="flicker",
                    seeds=(101, 202, 303),
                )
                for archetype in ValidationArchetype
            ),
            schedule_seed=73021,
        ),
        created_at=NOW,
    )


def seed_stage_1(database: Database, study_id: str = "study_1") -> None:
    base_manifest = manifest()
    optimizer_config = OptimizerConfig(posterior_draws=256)
    AdaptiveStudyRepository(database).create(
        AdaptiveStudyDefinition(
            id=study_id,
            prompt=base_manifest.prompt,
            negative_prompt=base_manifest.negative_prompt,
            seed=base_manifest.seed,
            base_manifest=base_manifest,
            optimizer_config=optimizer_config,
            input_fingerprint=adaptive_study_input_fingerprint(base_manifest, optimizer_config),
            created_at=NOW,
        )
    )


@pytest.fixture
def repository(database: Database) -> ValidationRepository:
    seed_stage_1(database)
    return ValidationRepository(database)


def panel(campaign: ValidationCampaignDefinition | None = None):
    return build_validation_panel(campaign or definition())


def changed_panel(campaign: ValidationCampaignDefinition | None = None):
    original = panel(campaign)
    changed = original.trials[0].model_copy(update={"winner_display_position": "right"})
    trials = (changed, *original.trials[1:])
    return original.model_copy(update={"trials": trials, "content_digest": "sha256:" + "b" * 64})


def event(
    campaign_id: str, sequence: int, before: ValidationCampaignState, after: ValidationCampaignState
) -> ValidationEvent:
    return ValidationEvent(
        campaign_id=campaign_id,
        sequence=sequence,
        previous_state=before,
        new_state=after,
        created_at=NOW + timedelta(seconds=sequence),
    )


def preregistered(repository: ValidationRepository):
    created = repository.create_or_get_bound(definition(), reviewer_id="owner")
    registered = repository.preregister_or_verify(created.id, panel(created))
    return created, registered


def judging(repository: ValidationRepository):
    created, _ = preregistered(repository)
    repository.append_event_or_verify(
        event(
            created.id, 2, ValidationCampaignState.PREREGISTERED, ValidationCampaignState.GENERATING
        )
    )
    repository.append_event_or_verify(
        event(created.id, 3, ValidationCampaignState.GENERATING, ValidationCampaignState.READY)
    )
    repository.append_event_or_verify(
        event(created.id, 4, ValidationCampaignState.READY, ValidationCampaignState.JUDGING)
    )
    return created, panel(created).trials[0]


def judgment(
    campaign_id: str, trial, choice: ValidationChoice = ValidationChoice.A
) -> ValidationJudgment:
    return ValidationJudgment(
        campaign_id=campaign_id,
        trial_id=trial.id,
        ordinal=trial.ordinal,
        archetype=trial.archetype,
        challenger=trial.challenger,
        seed=trial.seed,
        hidden_repeat=trial.hidden_repeat,
        winner_display_position=trial.winner_display_position,
        choice=choice,
        created_at=NOW,
    )


def test_create_or_get_is_owner_bound_and_definition_exact(
    repository: ValidationRepository,
) -> None:
    created = repository.create_or_get_bound(definition(), reviewer_id="owner")

    assert repository.create_or_get_bound(definition(), reviewer_id="owner") == created
    with pytest.raises(ValidationConflict):
        repository.create_or_get_bound(definition(), reviewer_id="other")
    with pytest.raises(ValidationConflict):
        repository.create_or_get_bound(
            definition().model_copy(update={"stage_1_evidence_digest": "b" * 64}),
            reviewer_id="owner",
        )


def test_preregister_is_exactly_idempotent_and_mismatch_conflicts(
    repository: ValidationRepository,
) -> None:
    created = repository.create_or_get_bound(definition(), reviewer_id="owner")
    first = repository.preregister_or_verify(created.id, panel(created))

    assert repository.preregister_or_verify(created.id, panel(created)) == first
    with pytest.raises(ValidationConflict):
        repository.preregister_or_verify(created.id, changed_panel())
    assert repository.list_events(created.id) == (first,)
    assert len(repository.list_trials(created.id)) == 40


def test_event_retry_requires_identical_payload_and_lifecycle(
    repository: ValidationRepository,
) -> None:
    created, _ = preregistered(repository)
    generated = event(
        created.id, 2, ValidationCampaignState.PREREGISTERED, ValidationCampaignState.GENERATING
    )

    assert repository.append_event_or_verify(generated) == generated
    assert (
        repository.append_event_or_verify(generated.model_copy(update={"created_at": NOW}))
        == generated
    )
    with pytest.raises(ValidationConflict):
        repository.append_event_or_verify(
            generated.model_copy(update={"new_state": ValidationCampaignState.READY})
        )


def test_bind_trial_is_exactly_idempotent(repository: ValidationRepository) -> None:
    created, trial = judging(repository)
    binding = {
        "campaign_id": created.id,
        "trial_id": trial.id,
        "reused_from_trial_id": trial.reuses_trial_id,
    }

    first = repository.bind_trial_or_verify(binding)
    assert repository.bind_trial_or_verify(binding) == first
    with pytest.raises(ValidationConflict):
        repository.bind_trial_or_verify({**binding, "reused_from_trial_id": "wrong"})


def test_experiment_binding_is_typed_concurrent_exact_and_append_only(
    database: Database,
) -> None:
    seed_stage_1(database)
    repository = ValidationRepository(database)
    created, _ = preregistered(repository)
    repository.append_event_or_verify(
        event(
            created.id,
            2,
            ValidationCampaignState.PREREGISTERED,
            ValidationCampaignState.GENERATING,
        )
    )
    trial = next(item for item in panel(created).trials if not item.hidden_repeat)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO experiments("
            "id, schema_version, variable_path, changed_paths_json, created_at) "
            "VALUES ('validation_exp_manual', 'validation-controlled.v1', "
            "'guidance_scale', '[\"guidance_scale\"]', ?)",
            (NOW.isoformat(),),
        )
    binding = ValidationExperimentBinding(
        campaign_id=created.id,
        trial_id=trial.id,
        experiment_id="validation_exp_manual",
    )
    barrier = Barrier(2)

    def bind() -> ValidationExperimentBinding:
        barrier.wait()
        return ValidationRepository(Database(database.path)).bind_experiment_or_verify(binding)

    with ThreadPoolExecutor(max_workers=2) as executor:
        persisted = tuple(executor.map(lambda _: bind(), range(2)))
    assert persisted == (binding, binding)
    assert repository.list_experiment_bindings(created.id) == (binding,)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO experiments("
            "id, schema_version, variable_path, changed_paths_json, created_at) "
            "VALUES ('validation_exp_other', 'validation-controlled.v1', "
            "'guidance_scale', '[\"guidance_scale\"]', ?)",
            (NOW.isoformat(),),
        )
    with pytest.raises(ValidationConflict):
        repository.bind_experiment_or_verify(
            binding.model_copy(update={"experiment_id": "validation_exp_other"})
        )

    with database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE validation_experiment_bindings SET experiment_id=experiment_id"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM validation_experiment_bindings")


def test_judgment_consume_or_verify_returns_persisted_winner(
    repository: ValidationRepository,
) -> None:
    created, trial = judging(repository)
    first, consumed = repository.consume_judgment_or_verify(judgment(created.id, trial))
    retried, retry_consumed = repository.consume_judgment_or_verify(
        judgment(created.id, trial).model_copy(update={"created_at": NOW + timedelta(seconds=1)})
    )

    assert consumed is True
    assert retry_consumed is False
    assert retried == first
    with pytest.raises(ValidationConflict):
        repository.consume_judgment_or_verify(judgment(created.id, trial, ValidationChoice.B))


def test_result_is_exactly_idempotent(repository: ValidationRepository) -> None:
    created, trial = judging(repository)
    repository.consume_judgment_or_verify(judgment(created.id, trial))
    repository.append_event_or_verify(
        event(created.id, 5, ValidationCampaignState.JUDGING, ValidationCampaignState.ANALYZING)
    )
    result = ValidationResult(
        campaign_id=created.id,
        disposition=ValidationDisposition.INCONCLUSIVE,
        reason=ValidationReason.INCOMPLETE_PRIMARY_EVIDENCE,
        challengers=(),
        archetype_reversals=(),
        repeat_agreements=0,
    )

    assert repository.append_result_or_verify(result) == result
    assert repository.append_result_or_verify(result) == result
    with pytest.raises(ValidationConflict):
        repository.append_result_or_verify(result.model_copy(update={"repeat_agreements": 1}))
    assert repository.get_result(created.id) == result


@pytest.mark.parametrize(
    ("table", "column"),
    (
        ("validation_campaigns", "reviewer_id"),
        ("validation_events", "new_state"),
        ("validation_trials", "prompt"),
        ("validation_trial_bindings", "review_id"),
        ("validation_judgments", "choice"),
        ("validation_results", "reason"),
    ),
)
def test_validation_evidence_rejects_update_and_delete(
    database: Database, table: str, column: str
) -> None:
    seed_stage_1(database)
    repository = ValidationRepository(database)
    created, trial = judging(repository)
    repository.bind_trial_or_verify(
        {
            "campaign_id": created.id,
            "trial_id": trial.id,
            "reused_from_trial_id": trial.reuses_trial_id,
        }
    )
    repository.consume_judgment_or_verify(judgment(created.id, trial))
    repository.append_event_or_verify(
        event(created.id, 5, ValidationCampaignState.JUDGING, ValidationCampaignState.ANALYZING)
    )
    repository.append_result_or_verify(
        ValidationResult(
            campaign_id=created.id,
            disposition=ValidationDisposition.INCONCLUSIVE,
            reason=ValidationReason.INCOMPLETE_PRIMARY_EVIDENCE,
            challengers=(),
            archetype_reversals=(),
            repeat_agreements=0,
        )
    )

    with (
        database.connect() as connection,
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
    ):
        connection.execute(f"UPDATE {table} SET {column} = {column}")
    with (
        database.connect() as connection,
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
    ):
        connection.execute(f"DELETE FROM {table}")


def test_concurrent_identical_preregistration_and_judgment_converge(database: Database) -> None:
    seed_stage_1(database)
    owner = ValidationRepository(database)
    created = owner.create_or_get_bound(definition(), reviewer_id="owner")
    barrier = Barrier(2)

    def register() -> object:
        barrier.wait()
        return ValidationRepository(Database(database.path)).preregister_or_verify(
            created.id, panel(created)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        registered = tuple(executor.map(lambda _: register(), range(2)))
    assert registered[0] == registered[1]
    assert len(owner.list_events(created.id)) == 1

    _, trial = judging(ValidationRepository(database))
    barrier = Barrier(2)

    def judge(choice: ValidationChoice) -> tuple[bool, object]:
        barrier.wait()
        try:
            return True, ValidationRepository(Database(database.path)).consume_judgment_or_verify(
                judgment(created.id, trial, choice)
            )
        except ValidationConflict:
            return False, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(judge, (ValidationChoice.A, ValidationChoice.B)))
    assert sum(success for success, _ in outcomes) == 1
    assert len(owner.list_judgments(created.id)) == 1


def test_concurrent_mismatched_preregistration_yields_one_conflict(database: Database) -> None:
    seed_stage_1(database)
    repository = ValidationRepository(database)
    created = repository.create_or_get_bound(definition("validation_race"), reviewer_id="owner")
    barrier = Barrier(2)

    def register(candidate) -> bool:
        barrier.wait()
        try:
            ValidationRepository(Database(database.path)).preregister_or_verify(
                created.id, candidate
            )
        except ValidationConflict:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(register, (panel(created), changed_panel(created))))
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1
    assert len(repository.list_events(created.id)) == 1
