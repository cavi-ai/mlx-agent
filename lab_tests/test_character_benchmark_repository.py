from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime

import pytest

from video_lab.adaptive_study_models import (
    AdaptiveStudyDefinition,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import AdaptiveStudyRepository
from video_lab.character_benchmark_models import (
    CharacterBenchmarkPanelInput,
    CharacterRecipePair,
    IdentityChoice,
    IdentityObservation,
    SecondaryChoice,
    SecondaryDimension,
    SecondaryObservation,
)
from video_lab.character_benchmark_protocol import (
    DEFAULT_CHARACTER_CONTEXTS,
    build_character_panel,
    schedule_character_trials,
)
from video_lab.character_benchmark_repository import (
    CharacterBenchmarkConflict,
    CharacterBenchmarkEvent,
    CharacterBenchmarkRepository,
    CharacterBenchmarkState,
    CharacterReviewBinding,
)
from video_lab.comparison import build_controlled_pair
from video_lab.domain import ExperimentDraft, Manifest, canonical_json
from video_lab.preference_optimizer import OptimizerConfig

NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def manifest() -> Manifest:
    return Manifest(
        model_id="org/model",
        model_revision="a" * 40,
        model_checksum="sha256:weights",
        adapter_id="test",
        adapter_version="1",
        runtime_versions={"python": "3.12"},
        prompt="stage two prompt",
        negative_prompt="flicker",
        seed=7,
        width=512,
        height=288,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=3.0,
        scheduler="euler",
        dtype="float16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="standard",
        hardware_fingerprint="hw",
        os_fingerprint="os",
    )


def definition():
    return build_character_panel(
        character_pack_digest="sha256:" + "1" * 64,
        stage_2_manifest_digest="sha256:" + "2" * 64,
        stage_2_evidence_digest="sha256:" + "3" * 64,
        recipes=CharacterRecipePair(
            control_digest="sha256:" + "4" * 64,
            identity_digest="sha256:" + "5" * 64,
        ),
        panel_input=CharacterBenchmarkPanelInput(
            contexts=DEFAULT_CHARACTER_CONTEXTS,
            seeds=(101, 202, 303),
            schedule_seed=73021,
        ),
    )


def create(repository: CharacterBenchmarkRepository):
    return repository.create_or_get_bound(
        definition(),
        character_pack_id="character_pack_" + "6" * 64,
        character_pack_storage_id="pack-storage",
        stage_2_campaign_id="validation_stage_2",
        stage_2_manifest=manifest(),
        identity_prompt="same character, preserve identity",
        content_digest="sha256:" + "7" * 64,
        created_at=NOW,
    )


def create_alternate(repository: CharacterBenchmarkRepository):
    alternate = build_character_panel(
        character_pack_digest="sha256:" + "1" * 64,
        stage_2_manifest_digest="sha256:" + "2" * 64,
        stage_2_evidence_digest="sha256:" + "3" * 64,
        recipes=definition().recipes,
        panel_input=definition().panel_input.model_copy(update={"schedule_seed": 73022}),
    )
    return repository.create_or_get_bound(
        alternate,
        character_pack_id="character_pack_" + "6" * 64,
        character_pack_storage_id="pack-storage",
        stage_2_campaign_id="validation_stage_2",
        stage_2_manifest=manifest(),
        identity_prompt="same character, preserve identity",
        content_digest="sha256:" + "8" * 64,
        created_at=NOW,
    )


def seed_dependencies(database) -> None:
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO characters VALUES ('character_1', 'owner', 'Character', ?)",
            (NOW.isoformat(),),
        )
        connection.execute(
            "INSERT INTO character_packs(storage_id, id, character_id, reviewer_id, state, "
            "supersedes_storage_id, manifest_json, created_at, preregistered_at) "
            "VALUES ('pack-storage', ?, 'character_1', 'owner', 'preregistered', NULL, ?, ?, ?)",
            (
                "character_pack_" + "6" * 64,
                '{"pack_id":"character_pack_' + "6" * 64 + '","character_id":"character_1",'
                '"duplicate_hash_algorithm_version":"dhash64-exif-transpose-lanczos.v1",'
                '"near_duplicate_threshold_inclusive":5,"references":[],'
                '"duplicate_warnings":[],"supersedes_pack_id":null}',
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
    optimizer = OptimizerConfig(posterior_draws=256)
    AdaptiveStudyRepository(database).create(
        AdaptiveStudyDefinition(
            id="study_1",
            prompt=manifest().prompt,
            negative_prompt=manifest().negative_prompt,
            seed=manifest().seed,
            base_manifest=manifest(),
            optimizer_config=optimizer,
            input_fingerprint=adaptive_study_input_fingerprint(manifest(), optimizer),
            created_at=NOW,
        )
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO validation_campaigns(id, schema_version, reviewer_id, stage_1_study_id, "
            "stage_1_evidence_digest, base_manifest_json, candidates_json, panel_input_json, "
            "protocol_version, analysis_version, created_at) VALUES "
            "('validation_stage_2','validation-campaign.v1','owner','study_1',?,?,'{}','{}',"
            "'balanced-validation.v1','dirichlet-validation.v1',?)",
            ("a" * 64, canonical_json(manifest()), NOW.isoformat()),
        )
        for sequence, (previous, new) in enumerate(
            (
                ("draft", "preregistered"),
                ("preregistered", "generating"),
                ("generating", "ready"),
                ("ready", "judging"),
                ("judging", "analyzing"),
            ),
            start=1,
        ):
            connection.execute(
                "INSERT INTO validation_events VALUES (?,?,?,?,?)",
                ("validation_stage_2", sequence, previous, new, NOW.isoformat()),
            )
        connection.execute(
            "INSERT INTO validation_results VALUES "
            "('validation_stage_2','qualified','all_gates_passed','[]','[]',4,?)",
            (NOW.isoformat(),),
        )
        connection.execute(
            "INSERT INTO validation_events VALUES "
            "('validation_stage_2',6,'analyzing','qualified',?)",
            (NOW.isoformat(),),
        )


def append_lifecycle_through(
    repository: CharacterBenchmarkRepository, campaign_id: str, target_sequence: int
) -> None:
    transitions = (
        (CharacterBenchmarkState.DRAFT, CharacterBenchmarkState.PREREGISTERED),
        (CharacterBenchmarkState.PREREGISTERED, CharacterBenchmarkState.GENERATING),
        (CharacterBenchmarkState.GENERATING, CharacterBenchmarkState.READY),
        (CharacterBenchmarkState.READY, CharacterBenchmarkState.JUDGING),
        (CharacterBenchmarkState.JUDGING, CharacterBenchmarkState.ANALYZING),
        (CharacterBenchmarkState.ANALYZING, CharacterBenchmarkState.COMPLETED),
    )
    first_missing = len(repository.list_events(campaign_id)) + 1
    for sequence in range(first_missing, target_sequence + 1):
        previous, new = transitions[sequence - 1]
        repository.append_event_or_verify(
            CharacterBenchmarkEvent(
                campaign_id=campaign_id,
                sequence=sequence,
                previous_state=previous,
                new_state=new,
                payload_digest="sha256:" + f"{sequence:x}" * 64,
                created_at=NOW,
            )
        )


def bind_trial_experiment(repository: CharacterBenchmarkRepository, campaign_id: str, trial):
    baseline = manifest().model_copy(
        update={
            "prompt": trial.context.prompt,
            "negative_prompt": trial.context.negative_prompt or manifest().negative_prompt,
            "seed": trial.seed,
        }
    )
    pair = build_controlled_pair(
        ExperimentDraft(
            baseline=baseline,
            variable_path="prompt",
            alternate_value=baseline.prompt + " preserve identity",
        )
    )
    return repository.bind_experiment_or_verify(campaign_id, trial, pair, created_at=NOW)


def expected_reviewer_id(campaign_id: str, trial_id: str, reviewer_id: str = "owner") -> str:
    payload = f"character-benchmark-review.v1\x00{campaign_id}\x00{trial_id}\x00{reviewer_id}"
    return "character_reviewer_" + hashlib.sha256(payload.encode()).hexdigest()


def insert_review_session(
    database,
    *,
    experiment_id: str,
    reviewer_id: str,
    session_id: str,
    review_id: str,
) -> None:
    with database.connect() as connection:
        variants = connection.execute(
            "SELECT variant.id AS variant_id, job.id AS job_id "
            "FROM variants AS variant JOIN jobs AS job ON job.variant_id=variant.id "
            "WHERE variant.experiment_id=? ORDER BY variant.label",
            (experiment_id,),
        ).fetchall()
        assert len(variants) == 2
        artifacts: list[tuple[str, str]] = []
        for position, variant in zip(("left", "right"), variants, strict=True):
            digest = hashlib.sha256(f"{session_id}:{position}".encode()).hexdigest()
            artifact_id = f"artifact-{digest}"
            connection.execute(
                "INSERT INTO artifacts(id, job_id, sha256, byte_size, media_type, "
                "relative_path, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    variant["job_id"],
                    digest,
                    1,
                    "video/mp4",
                    f"{digest[:2]}/{digest}",
                    NOW.isoformat(),
                ),
            )
            artifacts.append((artifact_id, digest))
        connection.execute(
            "INSERT INTO review_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                experiment_id,
                reviewer_id,
                variants[0]["variant_id"],
                variants[1]["variant_id"],
                variants[0]["job_id"],
                artifacts[0][0],
                artifacts[0][1],
                variants[1]["job_id"],
                artifacts[1][0],
                artifacts[1][1],
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO review_revisions VALUES (?,?,NULL,'tie',NULL,?,1,NULL)",
            (review_id, session_id, NOW.isoformat()),
        )


def review_dependencies(database, repository: CharacterBenchmarkRepository):
    campaign = create(repository)
    trials = schedule_character_trials(campaign.definition)
    repository.preregister_or_verify(campaign.definition.id, trials, created_at=NOW)
    append_lifecycle_through(repository, campaign.definition.id, 2)
    first_binding = bind_trial_experiment(repository, campaign.definition.id, trials[0])
    second_binding = bind_trial_experiment(repository, campaign.definition.id, trials[1])
    append_lifecycle_through(repository, campaign.definition.id, 4)
    alias = expected_reviewer_id(campaign.definition.id, trials[0].id)
    insert_review_session(
        database,
        experiment_id=first_binding.experiment_id,
        reviewer_id=alias,
        session_id="review-session-1",
        review_id="review-revision-1",
    )
    insert_review_session(
        database,
        experiment_id=first_binding.experiment_id,
        reviewer_id="wrong-reviewer",
        session_id="review-session-2",
        review_id="review-revision-2",
    )
    return campaign, trials, first_binding, second_binding, alias


def valid_review_binding(
    campaign_id: str, trial_id: str, experiment_id: str, reviewer_alias: str
) -> CharacterReviewBinding:
    return CharacterReviewBinding(
        campaign_id=campaign_id,
        trial_id=trial_id,
        experiment_id=experiment_id,
        review_id="review-revision-1",
        session_id="review-session-1",
        reviewer_alias=reviewer_alias,
        display_mapping={},
    )


@pytest.fixture
def repository(database) -> CharacterBenchmarkRepository:
    seed_dependencies(database)
    return CharacterBenchmarkRepository(database, reviewer_id="owner")


def test_campaign_is_owner_bound_and_unique_per_content_digest(
    database, repository: CharacterBenchmarkRepository
) -> None:
    first = create(repository)
    assert create(repository) == first
    assert first.definition == definition()
    assert first.state is CharacterBenchmarkState.DRAFT

    with pytest.raises(CharacterBenchmarkConflict, match="reviewer"):
        CharacterBenchmarkRepository(database, reviewer_id="intruder").get(first.definition.id)
    changed = build_character_panel(
        character_pack_digest="sha256:" + "8" * 64,
        stage_2_manifest_digest="sha256:" + "2" * 64,
        stage_2_evidence_digest="sha256:" + "3" * 64,
        recipes=definition().recipes,
        panel_input=definition().panel_input,
    )
    with pytest.raises(CharacterBenchmarkConflict, match="content digest"):
        repository.create_or_get_bound(
            changed,
            character_pack_id="character_pack_" + "6" * 64,
            character_pack_storage_id="pack-storage",
            stage_2_campaign_id="validation_stage_2",
            stage_2_manifest=manifest(),
            identity_prompt="different",
            content_digest="sha256:" + "7" * 64,
            created_at=NOW,
        )


def test_preregister_freezes_exact_trials_events_and_repeat_sources(
    repository: CharacterBenchmarkRepository,
) -> None:
    campaign = create(repository)
    trials = schedule_character_trials(campaign.definition)

    event = repository.preregister_or_verify(campaign.definition.id, trials, created_at=NOW)

    assert event == CharacterBenchmarkEvent(
        campaign_id=campaign.definition.id,
        sequence=1,
        previous_state=CharacterBenchmarkState.DRAFT,
        new_state=CharacterBenchmarkState.PREREGISTERED,
        payload_digest="sha256:"
        + __import__("hashlib")
        .sha256(canonical_json([trial.model_dump(mode="json") for trial in trials]).encode())
        .hexdigest(),
        created_at=NOW,
    )
    assert repository.list_trials(campaign.definition.id) == trials
    assert len([trial for trial in trials if trial.hidden_repeat]) == 4
    assert repository.preregister_or_verify(campaign.definition.id, trials, created_at=NOW) == event
    with pytest.raises(CharacterBenchmarkConflict, match="preregistration"):
        repository.preregister_or_verify(
            campaign.definition.id,
            trials[:-1] + (trials[-1].model_copy(update={"ordinal": 39}),),
            created_at=NOW,
        )


def test_identity_secondary_events_and_result_are_idempotent_append_only(
    database, repository: CharacterBenchmarkRepository
) -> None:
    campaign, trials, first, _, alias = review_dependencies(database, repository)
    repository.bind_review_or_verify(
        valid_review_binding(campaign.definition.id, trials[0].id, first.experiment_id, alias)
    )
    trial = trials[0]
    identity = IdentityObservation(
        benchmark_id=campaign.definition.id,
        trial_id=trial.id,
        choice=IdentityChoice.BOTH,
    )
    persisted = repository.record_identity_or_verify(
        identity,
        review_id="review-revision-1",
        display_choice="both",
        created_at=NOW,
    )
    assert (
        repository.record_identity_or_verify(
            identity,
            review_id="review-revision-1",
            display_choice="both",
            created_at=NOW,
        )
        == persisted
    )
    with pytest.raises(CharacterBenchmarkConflict, match="identity"):
        repository.record_identity_or_verify(
            identity.model_copy(update={"choice": IdentityChoice.NEITHER}),
            review_id="review-revision-1",
            display_choice="neither",
            created_at=NOW,
        )
    secondary = SecondaryObservation(
        benchmark_id=campaign.definition.id,
        trial_id=trial.id,
        dimension=SecondaryDimension.VISUAL_QUALITY,
        choice=SecondaryChoice.TIE,
    )
    assert repository.record_secondary_or_verify(secondary, created_at=NOW).observation == secondary

    with database.connect() as connection:
        for table in (
            "character_benchmark_campaigns",
            "character_benchmark_trials",
            "character_benchmark_identity_judgments",
            "character_benchmark_secondary_judgments",
            "character_benchmark_events",
        ):
            row = connection.execute(f"SELECT rowid FROM {table} LIMIT 1").fetchone()
            assert row is not None
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"UPDATE {table} SET rowid=rowid WHERE rowid=?", (row[0],))


def test_sql_rejects_draft_pack_campaign_binding(
    database, repository: CharacterBenchmarkRepository
) -> None:
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO character_packs(storage_id,id,character_id,reviewer_id,state,"
            "supersedes_storage_id,manifest_json,created_at,preregistered_at) "
            "VALUES ('draft-pack','draft-pack','character_1','owner','draft',NULL,NULL,?,NULL)",
            (NOW.isoformat(),),
        )
    campaign = create(repository)
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO character_benchmark_campaigns SELECT ?, reviewer_id, ?, "
            "definition_json, 'draft-pack', 'draft-pack', stage_2_campaign_id, "
            "stage_2_manifest_json, identity_prompt, created_at "
            "FROM character_benchmark_campaigns WHERE id=?",
            (
                "character_benchmark_" + "d" * 64,
                "sha256:" + "d" * 64,
                campaign.definition.id,
            ),
        )


def test_sql_rejects_cross_experiment_trial_binding(
    database, repository: CharacterBenchmarkRepository
) -> None:
    campaign = create(repository)
    trial = schedule_character_trials(campaign.definition)[0]
    repository.preregister_or_verify(
        campaign.definition.id, schedule_character_trials(campaign.definition), created_at=NOW
    )
    append_lifecycle_through(repository, campaign.definition.id, 2)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO experiments VALUES ('rogue-experiment','character-controlled.v1',"
            "'prompt','[\"prompt\"]',?)",
            (NOW.isoformat(),),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO character_benchmark_experiment_bindings VALUES (?,?,?)",
                (campaign.definition.id, trial.id, "rogue-experiment"),
            )


@pytest.mark.parametrize(
    "attack",
    (
        "nonexistent_review",
        "cross_experiment",
        "cross_session_revision",
        "cross_reviewer",
        "cross_campaign",
        "cross_trial",
    ),
)
def test_sql_rejects_inconsistent_review_binding(
    database, repository: CharacterBenchmarkRepository, attack: str
) -> None:
    campaign, trials, first, second, alias = review_dependencies(database, repository)
    values = [
        campaign.definition.id,
        trials[0].id,
        first.experiment_id,
        "review-revision-1",
        "review-session-1",
        alias,
        None,
        "{}",
    ]
    if attack == "nonexistent_review":
        values[3] = "missing-review"
    elif attack == "cross_experiment":
        values[2] = second.experiment_id
    elif attack == "cross_session_revision":
        values[3] = "review-revision-2"
    elif attack == "cross_reviewer":
        values[3] = "review-revision-2"
        values[4] = "review-session-2"
        values[5] = "wrong-reviewer"
    elif attack == "cross_campaign":
        alternate = create_alternate(repository)
        alternate_trials = schedule_character_trials(alternate.definition)
        repository.preregister_or_verify(alternate.definition.id, alternate_trials, created_at=NOW)
        append_lifecycle_through(repository, alternate.definition.id, 4)
        values[0] = alternate.definition.id
        values[1] = alternate_trials[0].id
    else:
        values[1] = trials[1].id
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO character_benchmark_review_bindings VALUES (?,?,?,?,?,?,?,?)",
            values,
        )


def test_repository_rejects_revision_from_another_session(
    database, repository: CharacterBenchmarkRepository
) -> None:
    campaign, trials, first, _, alias = review_dependencies(database, repository)
    binding = valid_review_binding(
        campaign.definition.id, trials[0].id, first.experiment_id, alias
    ).model_copy(update={"review_id": "review-revision-2"})
    with pytest.raises(CharacterBenchmarkConflict, match="persistence"):
        repository.bind_review_or_verify(binding)


@pytest.mark.parametrize("attack", ("nonexistent_review", "cross_review", "cross_trial"))
def test_sql_rejects_identity_without_exact_review_binding(
    database, repository: CharacterBenchmarkRepository, attack: str
) -> None:
    campaign, trials, first, _, alias = review_dependencies(database, repository)
    repository.bind_review_or_verify(
        valid_review_binding(campaign.definition.id, trials[0].id, first.experiment_id, alias)
    )
    trial_id = trials[1].id if attack == "cross_trial" else trials[0].id
    review_id = {
        "nonexistent_review": "missing-review",
        "cross_review": "review-revision-2",
        "cross_trial": "review-revision-1",
    }[attack]
    observation = IdentityObservation(
        benchmark_id=campaign.definition.id,
        trial_id=trial_id,
        choice=IdentityChoice.BOTH,
    )
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO character_benchmark_identity_judgments VALUES (?,?,?,?,?,?,?,?)",
            (
                campaign.definition.id,
                trial_id,
                review_id,
                "both",
                "both",
                canonical_json(observation),
                "sha256:" + "e" * 64,
                NOW.isoformat(),
            ),
        )


@pytest.mark.parametrize(
    "attack", ("wrong_review", "neither_identity", "wrong_phase", "wrong_dimension")
)
def test_sql_rejects_secondary_without_exact_eligible_identity(
    database, repository: CharacterBenchmarkRepository, attack: str
) -> None:
    campaign, trials, first, _, alias = review_dependencies(database, repository)
    repository.bind_review_or_verify(
        valid_review_binding(campaign.definition.id, trials[0].id, first.experiment_id, alias)
    )
    identity = IdentityObservation(
        benchmark_id=campaign.definition.id,
        trial_id=trials[0].id,
        choice=(IdentityChoice.NEITHER if attack == "neither_identity" else IdentityChoice.BOTH),
    )
    repository.record_identity_or_verify(
        identity,
        review_id="review-revision-1",
        display_choice=("neither" if attack == "neither_identity" else "both"),
        created_at=NOW,
    )
    if attack == "wrong_phase":
        append_lifecycle_through(repository, campaign.definition.id, 5)
    observation = SecondaryObservation(
        benchmark_id=campaign.definition.id,
        trial_id=trials[0].id,
        dimension=SecondaryDimension.VISUAL_QUALITY,
        choice=SecondaryChoice.TIE,
    )
    dimension = (
        SecondaryDimension.MOTION_QUALITY.value
        if attack == "wrong_dimension"
        else observation.dimension.value
    )
    review_id = "review-revision-2" if attack == "wrong_review" else "review-revision-1"
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO character_benchmark_secondary_judgments("
            "campaign_id,trial_id,dimension,choice,observation_json,content_digest,"
            "created_at,review_id) VALUES (?,?,?,?,?,?,?,?)",
            (
                campaign.definition.id,
                trials[0].id,
                dimension,
                observation.choice.value,
                canonical_json(observation),
                "sha256:" + "f" * 64,
                NOW.isoformat(),
                review_id,
            ),
        )


def test_sql_rejects_secondary_for_hidden_repeat(
    database, repository: CharacterBenchmarkRepository
) -> None:
    campaign = create(repository)
    trials = schedule_character_trials(campaign.definition)
    hidden = next(trial for trial in trials if trial.hidden_repeat)
    source = next(trial for trial in trials if trial.id == hidden.reuses_trial_id)
    repository.preregister_or_verify(campaign.definition.id, trials, created_at=NOW)
    append_lifecycle_through(repository, campaign.definition.id, 2)
    experiment = bind_trial_experiment(repository, campaign.definition.id, source)
    append_lifecycle_through(repository, campaign.definition.id, 4)
    alias = expected_reviewer_id(campaign.definition.id, hidden.id)
    insert_review_session(
        database,
        experiment_id=experiment.experiment_id,
        reviewer_id=alias,
        session_id="review-session-1",
        review_id="review-revision-1",
    )
    repository.bind_review_or_verify(
        CharacterReviewBinding(
            campaign_id=campaign.definition.id,
            trial_id=hidden.id,
            experiment_id=experiment.experiment_id,
            review_id="review-revision-1",
            session_id="review-session-1",
            reviewer_alias=alias,
            reused_from_trial_id=hidden.reuses_trial_id,
            display_mapping={},
        )
    )
    repository.record_identity_or_verify(
        IdentityObservation(
            benchmark_id=campaign.definition.id,
            trial_id=hidden.id,
            choice=IdentityChoice.BOTH,
        ),
        review_id="review-revision-1",
        display_choice="both",
        created_at=NOW,
    )
    observation = SecondaryObservation(
        benchmark_id=campaign.definition.id,
        trial_id=hidden.id,
        dimension=SecondaryDimension.VISUAL_QUALITY,
        choice=SecondaryChoice.TIE,
    )
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO character_benchmark_secondary_judgments("
            "campaign_id,trial_id,dimension,choice,observation_json,content_digest,"
            "created_at,review_id) VALUES (?,?,?,?,?,?,?,?)",
            (
                campaign.definition.id,
                hidden.id,
                observation.dimension.value,
                observation.choice.value,
                canonical_json(observation),
                "sha256:" + "f" * 64,
                NOW.isoformat(),
                "review-revision-1",
            ),
        )


def test_sql_rejects_cross_pack_version_binding(
    database, repository: CharacterBenchmarkRepository
) -> None:
    campaign = create(repository)
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO character_benchmark_campaigns SELECT ?, reviewer_id, ?, "
            "definition_json, ?, character_pack_storage_id, stage_2_campaign_id, "
            "stage_2_manifest_json, identity_prompt, created_at "
            "FROM character_benchmark_campaigns WHERE id=?",
            (
                "character_benchmark_" + "8" * 64,
                "sha256:" + "8" * 64,
                "character_pack_" + "9" * 64,
                campaign.definition.id,
            ),
        )


@pytest.mark.parametrize("attack", ("identity_before_judging", "result_before_analysis"))
def test_sql_rejects_wrong_lifecycle_evidence(
    database, repository: CharacterBenchmarkRepository, attack: str
) -> None:
    campaign = create(repository)
    trial = schedule_character_trials(campaign.definition)[0]
    if attack == "identity_before_judging":
        repository.preregister_or_verify(
            campaign.definition.id,
            schedule_character_trials(campaign.definition),
            created_at=NOW,
        )
        statement = "INSERT INTO character_benchmark_identity_judgments VALUES (?,?,?,?,?,?,?,?)"
        values = (
            campaign.definition.id,
            trial.id,
            "missing-review",
            "neither",
            "neither",
            canonical_json(
                IdentityObservation(
                    benchmark_id=campaign.definition.id,
                    trial_id=trial.id,
                    choice=IdentityChoice.NEITHER,
                )
            ),
            "sha256:" + "8" * 64,
            NOW.isoformat(),
        )
    else:
        statement = "INSERT INTO character_benchmark_results VALUES (?,?,?,?)"
        values = (
            campaign.definition.id,
            "{}",
            "sha256:" + "8" * 64,
            NOW.isoformat(),
        )
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(statement, values)
