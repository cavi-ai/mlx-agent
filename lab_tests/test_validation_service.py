from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from lab_tests.test_capabilities import manifest
from video_lab.adaptive_study_models import (
    AdaptiveStudyDefinition,
    AdaptiveStudyState,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import AdaptiveStudyRepository
from video_lab.artifacts import ArtifactStore
from video_lab.capabilities import AdapterCapabilities
from video_lab.database import _MIGRATIONS, Database, _migration_statements
from video_lab.domain import ExperimentDraft
from video_lab.preference_optimizer import OptimizerConfig
from video_lab.repositories import (
    ExperimentRepository,
    JobRepository,
    _insert_stored_experiment,
    _new_validation_experiment,
    validation_experiment_id,
)
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.services import ExperimentService, ExperimentView
from video_lab.validation_models import (
    ValidationArchetype,
    ValidationCampaignState,
    ValidationEvent,
    ValidationIneligible,
    ValidationPanelInput,
    ValidationPrompt,
)
from video_lab.validation_repository import ValidationConflict, ValidationRepository
from video_lab.validation_service import ValidationCampaignService, ValidationCampaignView

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def panel_input() -> ValidationPanelInput:
    return ValidationPanelInput(
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
    )


class StageOneStub:
    def __init__(self, repository: AdaptiveStudyRepository, *, owner: str = "owner") -> None:
        self.studies = repository
        self.reviewer_id = owner
        posterior = SimpleNamespace(
            grid=(1.0, 2.0, 3.0, 4.0, 5.0),
            recommendation=3.0,
            mean=(0.05, 0.15, 0.6, 0.15, 0.05),
        )
        self.step = SimpleNamespace(posterior=posterior)
        self.state = AdaptiveStudyState.SEARCH_CONVERGED

    def show(self, study_id: str):
        if self.reviewer_id != "owner":
            raise PermissionError("adaptive study belongs to another reviewer")
        return SimpleNamespace(
            study_id=study_id,
            state=self.state,
            evidence_digest="sha256:" + "a" * 64,
        )


class AdapterStub:
    @staticmethod
    def capabilities(_adapter_id: str) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="test",
            model_patterns=("org/*",),
            backends=("mps",),
            dtypes=("float16",),
            quantizations=("none",),
            schedulers=("euler",),
            min_width=256,
            max_width=1024,
            width_multiple=16,
            min_height=256,
            max_height=1024,
            height_multiple=16,
            min_frames=1,
            max_frames=81,
            frame_rule="16n+1",
            supports_lora=False,
        )


def service(database, tmp_path=None, *, reviewer_id: str = "owner") -> ValidationCampaignService:
    base = manifest()
    config = OptimizerConfig(posterior_draws=256)
    studies = AdaptiveStudyRepository(database)
    studies.create(
        AdaptiveStudyDefinition(
            id="study_1",
            prompt=base.prompt,
            negative_prompt=base.negative_prompt,
            seed=base.seed,
            base_manifest=base,
            optimizer_config=config,
            input_fingerprint=adaptive_study_input_fingerprint(base, config),
            created_at=NOW,
        )
    )
    stage_one = StageOneStub(studies, owner=reviewer_id)
    studies.list_optimizer_steps = lambda _study_id: (stage_one.step,)  # type: ignore[method-assign]
    artifact_store = None if tmp_path is None else ArtifactStore(database, tmp_path / "artifacts")
    experiments = (
        None
        if artifact_store is None
        else ExperimentService(
            ExperimentRepository(database), JobRepository(database), artifact_store
        )
    )
    reviews = (
        None
        if artifact_store is None
        else ReviewService(
            ExperimentRepository(database),
            JobRepository(database),
            artifact_store,
            ReviewRepository(database),
        )
    )
    return ValidationCampaignService(
        ValidationRepository(database),
        stage_one,  # type: ignore[arg-type]
        experiments=experiments,
        reviews=reviews,
        adapters=AdapterStub(),
        reviewer_id=reviewer_id,
        clock=lambda: NOW,
    )


def _complete_validation_jobs(campaign_service: ValidationCampaignService, tmp_path: Path) -> None:
    assert campaign_service.experiments is not None
    store = campaign_service.experiments.artifacts
    with campaign_service.campaigns.database.connect() as connection:
        jobs = connection.execute("SELECT id FROM jobs ORDER BY id").fetchall()
    for index, row in enumerate(jobs):
        job_id = str(row["id"])
        source = tmp_path / f"validation-{index}.mp4"
        source.write_bytes(f"validation-{index}".encode())
        store.commit(store.stage(source, "video/mp4"), job_id)
        with campaign_service.campaigns.database.connect() as connection:
            connection.execute("UPDATE jobs SET state='completed' WHERE id=?", (job_id,))


def _persist_40th_judgment_without_resuming(
    campaign_service: ValidationCampaignService,
    campaign_id: str,
) -> str:
    view = campaign_service.resume(campaign_id)
    for _ in range(39):
        assert view.review_id is not None
        view = campaign_service.record_choice(campaign_id, "a", review_id=view.review_id)
    assert view.review_id is not None
    final_review_id = view.review_id
    resume = campaign_service.resume

    def do_not_resume(campaign_id: str) -> ValidationCampaignView:
        return campaign_service.show(campaign_id)

    campaign_service.resume = do_not_resume
    try:
        campaign_service.record_choice(campaign_id, "a", review_id=final_review_id)
    finally:
        campaign_service.resume = resume
    assert len(campaign_service.campaigns.list_judgments(campaign_id)) == 40
    assert campaign_service.campaigns.state(campaign_id) is ValidationCampaignState.JUDGING
    return final_review_id


def test_create_requires_owned_terminal_interior_stage_1(database) -> None:
    foreign = service(database, reviewer_id="other")
    with pytest.raises(ValidationIneligible):
        foreign.create(stage_1_id="study_1", panel_input=panel_input())


def test_create_derives_three_challengers_from_persisted_terminal_step(database) -> None:
    view = service(database).create(stage_1_id="study_1", panel_input=panel_input())

    assert view.state is ValidationCampaignState.DRAFT
    assert view.preview is not None
    assert view.preview.primary_count == 36
    assert view.preview.repeat_count == 4
    assert view.preview.candidates.challengers == (2.0, 4.0, 1.0)


def test_concurrent_create_with_divergent_clocks_converges_on_repository_winner(database) -> None:
    first = service(database)
    clock_barrier = Barrier(2)

    def clock(offset: int):
        def now() -> datetime:
            clock_barrier.wait()
            return NOW + timedelta(seconds=offset)

        return now

    services = (
        ValidationCampaignService(
            ValidationRepository(Database(database.path)),
            first.stage_1,
            experiments=None,
            reviews=None,
            adapters=first.adapters,
            reviewer_id="owner",
            clock=clock(1),
        ),
        ValidationCampaignService(
            ValidationRepository(Database(database.path)),
            first.stage_1,
            experiments=None,
            reviews=None,
            adapters=first.adapters,
            reviewer_id="owner",
            clock=clock(2),
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        views = tuple(
            executor.map(
                lambda item: item.create(stage_1_id="study_1", panel_input=panel_input()),
                services,
            )
        )

    assert views[0] == views[1]
    assert len(first.campaigns.list_bound(reviewer_id="owner")) == 1


def test_preregister_creates_exactly_36_experiments_and_72_jobs_once(database, tmp_path) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())

    first = campaign_service.preregister(created.campaign_id)
    second = campaign_service.preregister(created.campaign_id)

    assert first == second
    assert first.state is ValidationCampaignState.GENERATING
    assert first.preview is None
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 36
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 72
        assert connection.execute("SELECT count(*) FROM validation_trials").fetchone()[0] == 40
        assert (
            connection.execute("SELECT count(*) FROM validation_experiment_bindings").fetchone()[0]
            == 36
        )

        connection.execute("DROP TRIGGER guard_validation_experiment_bindings_delete")
        missing = connection.execute(
            "SELECT experiment_id FROM validation_experiment_bindings LIMIT 1"
        ).fetchone()["experiment_id"]
        connection.execute(
            "DELETE FROM validation_experiment_bindings WHERE experiment_id=?", (missing,)
        )

    recovered = campaign_service.preregister(created.campaign_id)
    assert recovered == first
    with database.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM validation_experiment_bindings").fetchone()[0]
            == 36
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM validation_experiment_bindings WHERE experiment_id=?",
                (missing,),
            ).fetchone()[0]
            == 1
        )


def _seed_version_11_validation_campaign(
    database: Database, tmp_path: Path
) -> tuple[ValidationCampaignService, str]:
    campaign_service = service(database, tmp_path)
    assert campaign_service.experiments is not None
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    create_validation = campaign_service.experiments.create_validation

    def create_as_version_11(
        draft: ExperimentDraft,
        capabilities: AdapterCapabilities,
        *,
        campaign_id: str,
        trial_id: str,
    ) -> ExperimentView:
        pair = campaign_service.experiments.preview(draft, capabilities)
        stored, jobs = _new_validation_experiment(pair, campaign_id=campaign_id, trial_id=trial_id)
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _insert_stored_experiment(connection, stored)
            connection.executemany(
                "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                tuple(
                    (
                        job.id,
                        job.variant_id,
                        job.state.value,
                        job.created_at.isoformat(),
                        job.updated_at.isoformat(),
                    )
                    for job in jobs
                ),
            )
            connection.commit()
        return ExperimentView(experiment=stored, jobs=jobs)

    campaign_service.experiments.create_validation = create_as_version_11
    campaign_service.preregister(created.campaign_id)
    campaign_service.experiments.create_validation = create_validation
    return campaign_service, created.campaign_id


def _advance_version_11_campaign(
    database: Database,
    campaign_id: str,
    target: ValidationCampaignState,
) -> None:
    transitions = (
        (3, ValidationCampaignState.GENERATING, ValidationCampaignState.READY),
        (4, ValidationCampaignState.READY, ValidationCampaignState.JUDGING),
        (5, ValidationCampaignState.JUDGING, ValidationCampaignState.ANALYZING),
    )
    with database.connect() as connection:
        for sequence, previous, new in transitions:
            connection.execute(
                "INSERT INTO validation_events(campaign_id, sequence, previous_state, "
                "new_state, created_at) VALUES (?, ?, ?, ?, ?)",
                (campaign_id, sequence, previous.value, new.value, NOW.isoformat()),
            )
            if target is new:
                return
        connection.execute(
            "INSERT INTO validation_results(campaign_id, disposition, reason, "
            "challengers_json, archetype_reversals_json, repeat_agreements, created_at) "
            "VALUES (?, ?, ?, '[]', '[]', 4, ?)",
            (
                campaign_id,
                target.value,
                {
                    ValidationCampaignState.QUALIFIED: "all_gates_passed",
                    ValidationCampaignState.NOT_QUALIFIED: "challenger_rejection",
                    ValidationCampaignState.INCONCLUSIVE: "incomplete_primary_evidence",
                }[target],
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO validation_events(campaign_id, sequence, previous_state, "
            "new_state, created_at) VALUES (?, 6, 'analyzing', ?, ?)",
            (campaign_id, target.value, NOW.isoformat()),
        )


@pytest.mark.parametrize(
    "target",
    [
        ValidationCampaignState.READY,
        ValidationCampaignState.JUDGING,
        ValidationCampaignState.ANALYZING,
        ValidationCampaignState.QUALIFIED,
        ValidationCampaignState.NOT_QUALIFIED,
        ValidationCampaignState.INCONCLUSIVE,
    ],
)
def test_version_11_completed_campaign_migrates_with_exact_bindings_in_every_later_state(
    tmp_path: Path,
    target: ValidationCampaignState,
) -> None:
    database = Database(tmp_path / "version-11-validation.sqlite3")
    with database.connect() as connection:
        for version in range(1, 12):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )

    campaign_service, campaign_id = _seed_version_11_validation_campaign(database, tmp_path)
    _advance_version_11_campaign(database, campaign_id, target)

    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 36
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 72
        connection.execute(
            "INSERT INTO experiments(id, schema_version, variable_path, changed_paths_json, "
            "created_at) VALUES ('extra_validation_experiment', 'validation-controlled.v1', "
            "'guidance_scale', '[\"guidance_scale\"]', ?)",
            (NOW.isoformat(),),
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_master "
                "WHERE type='table' AND name='validation_experiment_bindings'"
            ).fetchone()[0]
            == 0
        )
        prior_evidence = {
            table: tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table}"))
            for table in (
                "validation_campaigns",
                "validation_trials",
                "validation_events",
                "validation_judgments",
                "validation_results",
            )
        }

    database.migrate()
    database.migrate()

    assert campaign_service.show(campaign_id).state is target
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 37
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 72
        assert (
            connection.execute("SELECT count(*) FROM validation_experiment_bindings").fetchone()[0]
            == 36
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM validation_experiment_bindings "
                "WHERE experiment_id='extra_validation_experiment'"
            ).fetchone()[0]
            == 0
        )
        assert all(
            str(row["experiment_id"])
            == validation_experiment_id(str(row["campaign_id"]), str(row["trial_id"]))
            for row in connection.execute("SELECT * FROM validation_experiment_bindings")
        )
        assert prior_evidence == {
            table: tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table}"))
            for table in prior_evidence
        }
        if target not in {
            ValidationCampaignState.READY,
            ValidationCampaignState.JUDGING,
        }:
            repeat_id = str(
                connection.execute(
                    "SELECT trial_id FROM validation_trials "
                    "WHERE campaign_id=? AND hidden_repeat=1 LIMIT 1",
                    (campaign_id,),
                ).fetchone()["trial_id"]
            )
            with pytest.raises(sqlite3.IntegrityError, match="invalid lifecycle state"):
                connection.execute(
                    "INSERT INTO validation_experiment_bindings"
                    "(campaign_id, trial_id, experiment_id) VALUES (?, ?, ?)",
                    (campaign_id, repeat_id, "extra_validation_experiment"),
                )


def test_version_12_migration_rejects_mismatched_deterministic_experiment(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "version-11-mismatch.sqlite3")
    with database.connect() as connection:
        for version in range(1, 12):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )
    _, campaign_id = _seed_version_11_validation_campaign(database, tmp_path)
    _advance_version_11_campaign(database, campaign_id, ValidationCampaignState.READY)
    with database.connect() as connection:
        experiment_id = str(
            connection.execute("SELECT id FROM experiments ORDER BY id LIMIT 1").fetchone()["id"]
        )
        connection.execute(
            "UPDATE experiments SET schema_version='mismatched.v1' WHERE id=?",
            (experiment_id,),
        )

    with pytest.raises(sqlite3.IntegrityError, match="does not match"):
        database.migrate()

    with database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 11
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_master "
                "WHERE type='table' AND name='validation_experiment_bindings'"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 36
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 72


def test_resume_uses_40_blind_sessions_reuses_repeat_media_and_analyzes_only_at_40(
    database, tmp_path
) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)

    view = campaign_service.resume(created.campaign_id)
    first_session = view.review_id
    assert first_session is not None
    with pytest.raises(LookupError):
        campaign_service.reviews.artifact_for_display(first_session, "left", "owner")
    with database.connect() as connection:
        validation_experiment = str(
            connection.execute(
                "SELECT experiment_id FROM review_sessions WHERE id=?", (first_session,)
            ).fetchone()["experiment_id"]
        )
    with pytest.raises(LookupError):
        campaign_service.reviews.start(validation_experiment, "owner")
    with pytest.raises(TypeError):
        campaign_service.record_choice(created.campaign_id, "a")  # type: ignore[call-arg]

    choice_barrier = Barrier(2)

    def choose() -> ValidationCampaignView:
        choice_barrier.wait()
        return campaign_service.record_choice(created.campaign_id, "a", review_id=first_session)

    with ThreadPoolExecutor(max_workers=2) as executor:
        converged = tuple(executor.map(lambda _: choose(), range(2)))
    assert converged[0] == converged[1]
    view = converged[0]
    retried = campaign_service.record_choice(created.campaign_id, "a", review_id=first_session)
    assert retried.progress.completed_judgments == 1
    with pytest.raises((ValidationConflict, ValueError)):
        campaign_service.record_choice(created.campaign_id, "b", review_id=first_session)
    sessions = [first_session, view.review_id]
    for _ in range(38):
        assert view.review_id is not None
        view = campaign_service.record_choice(created.campaign_id, "a", review_id=view.review_id)
        sessions.append(view.review_id)

    assert len(set(sessions)) == 40
    with pytest.raises(ValidationIneligible, match="40 judgments"):
        campaign_service.analyze(created.campaign_id)

    assert view.review_id is not None
    terminal = campaign_service.record_choice(created.campaign_id, "a", review_id=view.review_id)
    assert terminal.state in {
        ValidationCampaignState.QUALIFIED,
        ValidationCampaignState.NOT_QUALIFIED,
        ValidationCampaignState.INCONCLUSIVE,
    }
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM review_sessions").fetchone()[0] == 40
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 36
        repeat_experiments = connection.execute(
            "SELECT count(DISTINCT session.experiment_id) FROM review_sessions AS session"
        ).fetchone()[0]
        assert (
            connection.execute("SELECT count(*) FROM validation_trial_bindings").fetchone()[0] == 40
        )
    assert repeat_experiments == 36
    evidence = campaign_service.export_evidence(created.campaign_id)
    assert campaign_service.export_evidence(created.campaign_id) == evidence
    assert evidence["schema_version"] == "balanced-validation-evidence.v2"
    assert str(evidence["provenance_digest"]).startswith("sha256:")
    assert evidence["result"]
    assert len(evidence["experiment_bindings"]) == 36
    assert len(evidence["trial_bindings"]) == 40
    assert len(evidence["reviews"]) == 40
    assert all(
        set(display) == {"left", "right"}
        for review in evidence["reviews"]
        for display in (review["display_mapping"],)
    )
    assert all(
        {"variant_id", "job_id", "artifact_id", "artifact_sha256"} <= set(item)
        for review in evidence["reviews"]
        for item in review["display_mapping"].values()
    )
    by_trial = {item["trial_id"]: item for item in evidence["trial_bindings"]}
    reviews_by_id = {item["review_id"]: item for item in evidence["reviews"]}
    repeat_trials = [item for item in evidence["panel"]["trials"] if item["hidden_repeat"]]
    assert len(repeat_trials) == 4
    for repeat in repeat_trials:
        binding = by_trial[repeat["id"]]
        source = by_trial[repeat["reuses_trial_id"]]
        assert binding["experiment_id"] == source["experiment_id"]
        assert binding["review_id"] != source["review_id"]
        assert (
            reviews_by_id[binding["review_id"]]["session_id"]
            != reviews_by_id[source["review_id"]]["session_id"]
        )
        assert {
            item["artifact_sha256"]
            for item in reviews_by_id[binding["review_id"]]["display_mapping"].values()
        } == {
            item["artifact_sha256"]
            for item in reviews_by_id[source["review_id"]]["display_mapping"].values()
        }

    assert view.review_id is not None
    assert (
        campaign_service.record_choice(created.campaign_id, "a", review_id=view.review_id)
        == terminal
    )
    with pytest.raises(ValidationConflict):
        campaign_service.record_choice(created.campaign_id, "b", review_id=view.review_id)
    with pytest.raises(ValidationConflict):
        campaign_service.record_choice(created.campaign_id, "a", review_id="arbitrary-review")


def test_nonterminal_evidence_keeps_review_and_artifact_identities_sealed(
    database, tmp_path
) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)

    with pytest.raises(PermissionError, match="sealed"):
        campaign_service.export_evidence(created.campaign_id)


def test_version_13_repairs_legacy_repeat_review_bindings(database, tmp_path) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)
    view = campaign_service.resume(created.campaign_id)
    for _ in range(40):
        assert view.review_id is not None
        view = campaign_service.record_choice(created.campaign_id, "a", review_id=view.review_id)

    with database.connect() as connection:
        correct = {
            str(row["trial_id"]): str(row["review_id"])
            for row in connection.execute(
                "SELECT trial_id, review_id FROM validation_trial_bindings WHERE campaign_id=?",
                (created.campaign_id,),
            )
        }
        connection.execute("DROP TRIGGER guard_validation_trial_bindings_update")
        connection.execute(
            "UPDATE validation_trial_bindings SET review_id=("
            "SELECT source.review_id FROM validation_trial_bindings AS source "
            "WHERE source.campaign_id=validation_trial_bindings.campaign_id "
            "AND source.trial_id=validation_trial_bindings.reused_from_trial_id) "
            "WHERE campaign_id=? AND reused_from_trial_id IS NOT NULL",
            (created.campaign_id,),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=13")

    database.migrate()

    with database.connect() as connection:
        repaired = {
            str(row["trial_id"]): str(row["review_id"])
            for row in connection.execute(
                "SELECT trial_id, review_id FROM validation_trial_bindings WHERE campaign_id=?",
                (created.campaign_id,),
            )
        }
    assert repaired == correct


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_binding",
        "wrong_repeat_experiment",
        "wrong_repeat_reviewer",
        "wrong_artifact_pin",
        "orphan_revision",
    ),
)
def test_terminal_export_fails_closed_for_incomplete_or_inconsistent_provenance(
    database, tmp_path, corruption: str
) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)
    view = campaign_service.resume(created.campaign_id)
    for _ in range(40):
        assert view.review_id is not None
        view = campaign_service.record_choice(created.campaign_id, "a", review_id=view.review_id)

    with database.connect() as connection:
        repeat = connection.execute(
            "SELECT binding.trial_id, binding.review_id FROM validation_trial_bindings AS binding "
            "WHERE binding.campaign_id=? AND binding.reused_from_trial_id IS NOT NULL LIMIT 1",
            (created.campaign_id,),
        ).fetchone()
        assert repeat is not None
        if corruption == "missing_binding":
            connection.execute("DROP TRIGGER guard_validation_trial_bindings_delete")
            connection.execute(
                "DELETE FROM validation_trial_bindings WHERE campaign_id=? AND trial_id=?",
                (created.campaign_id, repeat["trial_id"]),
            )
        elif corruption == "wrong_repeat_experiment":
            connection.execute("DROP TRIGGER guard_validation_trial_bindings_update")
            wrong_experiment = connection.execute(
                "SELECT experiment_id FROM validation_experiment_bindings "
                "WHERE campaign_id=? AND trial_id<>(SELECT reused_from_trial_id "
                "FROM validation_trial_bindings WHERE campaign_id=? AND trial_id=?) LIMIT 1",
                (created.campaign_id, created.campaign_id, repeat["trial_id"]),
            ).fetchone()
            assert wrong_experiment is not None
            connection.execute(
                "UPDATE validation_trial_bindings SET experiment_id=? "
                "WHERE campaign_id=? AND trial_id=?",
                (wrong_experiment["experiment_id"], created.campaign_id, repeat["trial_id"]),
            )
        elif corruption == "wrong_repeat_reviewer":
            connection.execute(
                "UPDATE review_sessions SET reviewer_id='intruder' WHERE id=("
                "SELECT session_id FROM review_revisions WHERE id=?)",
                (repeat["review_id"],),
            )
        elif corruption == "wrong_artifact_pin":
            connection.execute(
                "UPDATE review_sessions SET left_artifact_sha256=? WHERE id=("
                "SELECT session_id FROM review_revisions WHERE id=?)",
                ("0" * 64, repeat["review_id"]),
            )
        else:
            revision = connection.execute(
                "SELECT * FROM review_revisions WHERE id=?", (repeat["review_id"],)
            ).fetchone()
            assert revision is not None
            connection.execute(
                "INSERT INTO review_revisions(id, session_id, preferred_variant_id, "
                "display_preference, notes, created_at, revision, supersedes_review_id) "
                "VALUES ('orphan_repeat_revision', ?, ?, ?, NULL, ?, 2, ?)",
                (
                    revision["session_id"],
                    revision["preferred_variant_id"],
                    revision["display_preference"],
                    revision["created_at"],
                    revision["id"],
                ),
            )

    with pytest.raises(ValidationConflict, match="provenance"):
        campaign_service.export_evidence(created.campaign_id)


@pytest.mark.parametrize("corruption", ("deleted", "truncated", "path_redirected"))
def test_terminal_export_fails_closed_when_pinned_media_bytes_are_not_canonical(
    database, tmp_path, corruption: str
) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)
    view = campaign_service.resume(created.campaign_id)
    for _ in range(40):
        assert view.review_id is not None
        view = campaign_service.record_choice(created.campaign_id, "a", review_id=view.review_id)

    assert campaign_service.reviews is not None
    with database.connect() as connection:
        artifacts = connection.execute(
            "SELECT id, relative_path FROM artifacts ORDER BY id LIMIT 2"
        ).fetchall()
        assert len(artifacts) == 2
        artifact_id = str(artifacts[0]["id"])
        artifact_path = campaign_service.reviews.artifacts.root / str(artifacts[0]["relative_path"])
        if corruption == "path_redirected":
            connection.execute(
                "UPDATE artifacts SET relative_path=? WHERE id=?",
                (artifacts[1]["relative_path"], artifact_id),
            )
    if corruption == "deleted":
        artifact_path.unlink()
    elif corruption == "truncated":
        artifact_path.write_bytes(b"truncated")

    with pytest.raises(ValidationConflict, match="provenance"):
        campaign_service.export_evidence(created.campaign_id)


def test_terminal_export_rejects_session_coherently_swapped_to_another_experiment(
    database, tmp_path
) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)
    view = campaign_service.resume(created.campaign_id)
    for _ in range(40):
        assert view.review_id is not None
        view = campaign_service.record_choice(created.campaign_id, "a", review_id=view.review_id)

    with database.connect() as connection:
        target = connection.execute(
            "SELECT session.* FROM review_sessions AS session "
            "JOIN review_revisions AS review ON review.session_id=session.id "
            "JOIN validation_trial_bindings AS binding ON binding.review_id=review.id "
            "WHERE binding.campaign_id=? AND binding.reused_from_trial_id IS NOT NULL LIMIT 1",
            (created.campaign_id,),
        ).fetchone()
        assert target is not None
        donor = connection.execute(
            "SELECT session.* FROM review_sessions AS session "
            "JOIN review_revisions AS review ON review.session_id=session.id "
            "JOIN validation_trial_bindings AS binding ON binding.review_id=review.id "
            "WHERE binding.campaign_id=? AND binding.experiment_id<>? LIMIT 1",
            (created.campaign_id, target["experiment_id"]),
        ).fetchone()
        assert donor is not None
        assert target["experiment_id"] != donor["experiment_id"]
        connection.execute(
            "UPDATE review_sessions SET experiment_id=?, left_variant_id=?, left_job_id=?, "
            "left_artifact_id=?, left_artifact_sha256=?, right_variant_id=?, right_job_id=?, "
            "right_artifact_id=?, right_artifact_sha256=? WHERE id=?",
            (
                donor["experiment_id"],
                donor["left_variant_id"],
                donor["left_job_id"],
                donor["left_artifact_id"],
                donor["left_artifact_sha256"],
                donor["right_variant_id"],
                donor["right_job_id"],
                donor["right_artifact_id"],
                donor["right_artifact_sha256"],
                target["id"],
            ),
        )

    with pytest.raises(ValidationConflict, match="provenance"):
        campaign_service.export_evidence(created.campaign_id)


@pytest.mark.parametrize("boundary", ["judging", "analyzing"])
def test_retry_after_40_validates_exact_review_and_choice_before_resuming(
    database: Database,
    tmp_path: Path,
    boundary: str,
) -> None:
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)
    final_review_id = _persist_40th_judgment_without_resuming(campaign_service, created.campaign_id)
    if boundary == "analyzing":
        campaign_service.campaigns.append_event_or_verify(
            ValidationEvent(
                campaign_id=created.campaign_id,
                sequence=5,
                previous_state=ValidationCampaignState.JUDGING,
                new_state=ValidationCampaignState.ANALYZING,
                created_at=NOW,
            )
        )

    with pytest.raises(ValidationConflict):
        campaign_service.record_choice(created.campaign_id, "b", review_id=final_review_id)
    with pytest.raises(ValidationConflict):
        campaign_service.record_choice(created.campaign_id, "a", review_id="arbitrary-review")

    retry_barrier = Barrier(2)

    def retry() -> ValidationCampaignView:
        retry_barrier.wait()
        return campaign_service.record_choice(created.campaign_id, "a", review_id=final_review_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        converged = tuple(executor.map(lambda _: retry(), range(2)))

    assert converged[0] == converged[1]
    assert converged[0].state in {
        ValidationCampaignState.QUALIFIED,
        ValidationCampaignState.NOT_QUALIFIED,
        ValidationCampaignState.INCONCLUSIVE,
    }
