from __future__ import annotations

import gc
import json
import shutil
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_tests.test_character_benchmark_repository import NOW, manifest, seed_dependencies
from video_lab.artifacts import ArtifactStore
from video_lab.capabilities import AdapterCapabilities
from video_lab.character_benchmark_models import (
    CharacterBenchmarkPanelInput,
    IdentityChoice,
    IdentityObservation,
    SecondaryChoice,
    SecondaryDimension,
    SecondaryObservation,
)
from video_lab.character_benchmark_protocol import DEFAULT_CHARACTER_CONTEXTS
from video_lab.character_benchmark_repository import (
    CharacterBenchmarkConflict,
    CharacterBenchmarkRepository,
    CharacterBenchmarkState,
)
from video_lab.character_benchmark_service import CharacterBenchmarkService
from video_lab.character_pack import CharacterPack
from video_lab.database import Database
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.services import ExperimentService


@pytest.fixture(autouse=True)
def collect_sqlite_connections_after_case():
    """Keep the full corruption matrix below the process file-descriptor limit."""
    yield
    gc.collect()


class PackStub:
    def __init__(self) -> None:
        self.corrupt_bytes = False
        self.pack = CharacterPack(
            pack_id="character_pack_" + "6" * 64,
            character_id="character_1",
            duplicate_hash_algorithm_version="dhash64-exif-transpose-lanczos.v1",
            near_duplicate_threshold_inclusive=5,
            references=(),
            duplicate_warnings=(),
            supersedes_pack_id=None,
        )

    def export_pack(self, _pack_id: str):
        if self.corrupt_bytes:
            raise RuntimeError("pack artifact bytes changed")
        return SimpleNamespace(pack=self.pack, references=())


class StageTwoStub:
    def __init__(self) -> None:
        self.evidence = {
            "schema_version": "balanced-validation-evidence.v2",
            "campaign": {"base_manifest": manifest().model_dump(mode="json")},
            "result": {"disposition": "qualified"},
        }

    @staticmethod
    def show(_campaign_id: str):
        return SimpleNamespace(state=SimpleNamespace(value="qualified"))

    def export_evidence(self, _campaign_id: str):
        return self.evidence


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


def panel_input() -> CharacterBenchmarkPanelInput:
    return CharacterBenchmarkPanelInput(
        contexts=DEFAULT_CHARACTER_CONTEXTS,
        seeds=(101, 202, 303),
        schedule_seed=73021,
    )


def service(database, tmp_path: Path, *, seed: bool = True) -> CharacterBenchmarkService:
    if seed:
        seed_dependencies(database)
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    experiment_repository = ExperimentRepository(database)
    jobs = JobRepository(database)
    experiments = ExperimentService(experiment_repository, jobs, artifacts)
    reviews = ReviewService(
        experiment_repository,
        jobs,
        artifacts,
        ReviewRepository(database),
    )
    return CharacterBenchmarkService(
        CharacterBenchmarkRepository(database, reviewer_id="owner"),
        PackStub(),
        StageTwoStub(),
        experiments,
        reviews,
        AdapterStub(),
        reviewer_id="owner",
        clock=lambda: NOW,
    )


def complete_jobs(benchmark: CharacterBenchmarkService, tmp_path: Path) -> None:
    with benchmark.campaigns.database.connect() as connection:
        job_ids = [
            str(row["id"])
            for row in connection.execute(
                "SELECT job.id FROM jobs AS job JOIN variants AS variant ON "
                "variant.id=job.variant_id JOIN character_benchmark_experiment_bindings AS binding "
                "ON binding.experiment_id=variant.experiment_id ORDER BY job.id"
            )
        ]
    assert len(job_ids) == 72
    for index, job_id in enumerate(job_ids):
        source = tmp_path / f"generated-{index}.mp4"
        source.write_bytes(f"video-{index}".encode())
        staged = benchmark.experiments.artifacts.stage(source, "video/mp4")
        benchmark.experiments.artifacts.commit(staged, job_id)
        with benchmark.campaigns.database.connect() as connection:
            connection.execute("UPDATE jobs SET state='completed' WHERE id=?", (job_id,))


def finish(benchmark: CharacterBenchmarkService, campaign_id: str):
    view = benchmark.resume(campaign_id)
    while view.state is CharacterBenchmarkState.JUDGING:
        assert view.review_id is not None and view.trial_id is not None
        identity = IdentityObservation(
            benchmark_id=view.campaign_id,
            trial_id=view.trial_id,
            choice=IdentityChoice.B,
        )
        view = benchmark.record_identity(view.campaign_id, view.review_id, identity)
        if view.next_action == "record_secondary":
            for dimension in SecondaryDimension:
                view = benchmark.record_secondary(
                    view.campaign_id,
                    view.review_id or "",
                    SecondaryObservation(
                        benchmark_id=view.campaign_id,
                        trial_id=identity.trial_id,
                        dimension=dimension,
                        choice=SecondaryChoice.B,
                    ),
                )
    return view


def evidence_counts(database, campaign_id: str) -> tuple[int, ...]:
    with database.connect() as connection:
        return tuple(
            int(connection.execute(statement, (campaign_id,)).fetchone()[0])
            for statement in (
                "SELECT count(*) FROM character_benchmark_identity_judgments WHERE campaign_id=?",
                "SELECT count(*) FROM character_benchmark_secondary_judgments WHERE campaign_id=?",
                "SELECT count(*) FROM character_benchmark_review_bindings WHERE campaign_id=?",
                "SELECT count(*) FROM character_benchmark_events WHERE campaign_id=?",
                "SELECT count(*) FROM character_benchmark_results WHERE campaign_id=?",
            )
        )


def test_create_preregister_and_resume_create_exact_prompt_only_work(
    database, tmp_path: Path
) -> None:
    benchmark = service(database, tmp_path)
    created = benchmark.create(
        character_pack_id="character_pack_" + "6" * 64,
        stage_2_campaign_id="validation_stage_2",
        identity_prompt="same character, preserve identity",
        panel_input=panel_input(),
    )
    assert benchmark.preview(created.campaign_id).primary_count == 36
    preregistered = benchmark.preregister(created.campaign_id)
    assert preregistered.progress.total_primary == 36

    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM character_benchmark_experiment_bindings"
            ).fetchone()[0]
            == 36
        )
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 72
        manifests = [
            str(row[0])
            for row in connection.execute(
                "SELECT manifest_json FROM variants ORDER BY experiment_id, label"
            )
        ]
    assert all('"loras":[]' in value for value in manifests)
    complete_jobs(benchmark, tmp_path)
    judging = benchmark.resume(created.campaign_id)
    assert judging.state is CharacterBenchmarkState.JUDGING
    assert judging.review_id is not None


def test_restart_retry_uses_40_distinct_reviews_and_four_exact_media_reuses(
    database, tmp_path: Path
) -> None:
    benchmark = service(database, tmp_path)
    created = benchmark.create(
        character_pack_id="character_pack_" + "6" * 64,
        stage_2_campaign_id="validation_stage_2",
        identity_prompt="same character, preserve identity",
        panel_input=panel_input(),
    )
    assert benchmark.preregister(created.campaign_id) == benchmark.preregister(created.campaign_id)
    complete_jobs(benchmark, tmp_path)
    terminal = finish(benchmark, created.campaign_id)
    assert terminal.state is CharacterBenchmarkState.COMPLETED
    evidence = benchmark.export_evidence(created.campaign_id)
    assert len(evidence.identity_observations) == 40

    with database.connect() as connection:
        bindings = connection.execute(
            "SELECT * FROM character_benchmark_review_bindings ORDER BY trial_id"
        ).fetchall()
        assert len({str(row["review_id"]) for row in bindings}) == 40
        assert len({str(row["session_id"]) for row in bindings}) == 40
        repeats = [row for row in bindings if row["reused_from_trial_id"] is not None]
        assert len(repeats) == 4
        for repeat in repeats:
            source = connection.execute(
                "SELECT experiment_id FROM character_benchmark_review_bindings "
                "WHERE campaign_id=? AND trial_id=?",
                (repeat["campaign_id"], repeat["reused_from_trial_id"]),
            ).fetchone()
            assert source is not None and source["experiment_id"] == repeat["experiment_id"]
    restarted = service(database, tmp_path, seed=False)
    assert restarted.export_evidence(created.campaign_id) == evidence
    assert restarted.analyze(created.campaign_id).result == terminal.result


def test_neither_skips_secondary_and_export_replays_deterministic_analysis(
    database, tmp_path: Path
) -> None:
    benchmark = service(database, tmp_path)
    created = benchmark.create(
        character_pack_id="character_pack_" + "6" * 64,
        stage_2_campaign_id="validation_stage_2",
        identity_prompt="same character, preserve identity",
        panel_input=panel_input(),
    )
    benchmark.preregister(created.campaign_id)
    complete_jobs(benchmark, tmp_path)
    view = benchmark.resume(created.campaign_id)
    assert view.review_id and view.trial_id
    next_view = benchmark.record_identity(
        created.campaign_id,
        view.review_id,
        IdentityObservation(
            benchmark_id=created.campaign_id,
            trial_id=view.trial_id,
            choice=IdentityChoice.NEITHER,
        ),
    )
    assert next_view.trial_id != view.trial_id
    with pytest.raises(CharacterBenchmarkConflict, match="secondary"):
        benchmark.record_secondary(
            created.campaign_id,
            view.review_id,
            SecondaryObservation(
                benchmark_id=created.campaign_id,
                trial_id=view.trial_id,
                dimension=SecondaryDimension.VISUAL_QUALITY,
                choice=SecondaryChoice.TIE,
            ),
        )


def test_exact_judgment_retries_survive_advancement_restart_and_terminal_state(
    database, tmp_path: Path
) -> None:
    benchmark = service(database, tmp_path)
    created = benchmark.create(
        character_pack_id="character_pack_" + "6" * 64,
        stage_2_campaign_id="validation_stage_2",
        identity_prompt="same character, preserve identity",
        panel_input=panel_input(),
    )
    benchmark.preregister(created.campaign_id)
    complete_jobs(benchmark, tmp_path)

    first = benchmark.resume(created.campaign_id)
    assert first.review_id and first.trial_id
    neither = IdentityObservation(
        benchmark_id=created.campaign_id,
        trial_id=first.trial_id,
        choice=IdentityChoice.NEITHER,
    )
    advanced = benchmark.record_identity(created.campaign_id, first.review_id, neither)
    assert advanced.trial_id != first.trial_id

    restarted = service(database, tmp_path, seed=False)
    before_neither_retry = evidence_counts(database, created.campaign_id)
    assert restarted.record_identity(
        created.campaign_id, first.review_id, neither
    ) == restarted.show(created.campaign_id)
    assert evidence_counts(database, created.campaign_id) == before_neither_retry
    with pytest.raises(CharacterBenchmarkConflict):
        restarted.record_identity(
            created.campaign_id,
            first.review_id,
            neither.model_copy(update={"choice": IdentityChoice.A}),
        )
    assert evidence_counts(database, created.campaign_id) == before_neither_retry

    second = restarted.resume(created.campaign_id)
    assert second.review_id and second.trial_id
    while next(
        trial
        for trial in restarted.campaigns.list_trials(created.campaign_id)
        if trial.id == second.trial_id
    ).hidden_repeat:
        restarted.record_identity(
            created.campaign_id,
            second.review_id,
            IdentityObservation(
                benchmark_id=created.campaign_id,
                trial_id=second.trial_id,
                choice=IdentityChoice.B,
            ),
        )
        second = restarted.resume(created.campaign_id)
        assert second.review_id and second.trial_id
    identity = IdentityObservation(
        benchmark_id=created.campaign_id,
        trial_id=second.trial_id,
        choice=IdentityChoice.B,
    )
    secondary_observations = tuple(
        SecondaryObservation(
            benchmark_id=created.campaign_id,
            trial_id=second.trial_id,
            dimension=dimension,
            choice=SecondaryChoice.B,
        )
        for dimension in SecondaryDimension
    )
    restarted.record_identity(created.campaign_id, second.review_id, identity)
    for observation in secondary_observations:
        restarted.record_secondary(created.campaign_id, second.review_id, observation)
    after_secondary = service(database, tmp_path, seed=False)
    before_secondary_retries = evidence_counts(database, created.campaign_id)
    assert after_secondary.record_identity(
        created.campaign_id, second.review_id, identity
    ) == after_secondary.show(created.campaign_id)
    for observation in secondary_observations:
        assert after_secondary.record_secondary(
            created.campaign_id, second.review_id, observation
        ) == after_secondary.show(created.campaign_id)
    with pytest.raises(CharacterBenchmarkConflict):
        after_secondary.record_secondary(
            created.campaign_id,
            second.review_id,
            secondary_observations[0].model_copy(update={"choice": SecondaryChoice.A}),
        )
    assert evidence_counts(database, created.campaign_id) == before_secondary_retries

    terminal = finish(after_secondary, created.campaign_id)
    assert terminal.state is CharacterBenchmarkState.COMPLETED
    repeat_trial = next(
        trial
        for trial in after_secondary.campaigns.list_trials(created.campaign_id)
        if trial.hidden_repeat
    )
    repeat_binding = next(
        binding
        for binding in after_secondary.campaigns.list_review_bindings(created.campaign_id)
        if binding.trial_id == repeat_trial.id
    )
    repeat_identity = next(
        item.observation
        for item in after_secondary.campaigns.list_identity(created.campaign_id)
        if item.observation.trial_id == repeat_trial.id
    )
    terminal_restart = service(database, tmp_path, seed=False)
    before_terminal_retry = evidence_counts(database, created.campaign_id)
    assert terminal_restart.record_identity(
        created.campaign_id, second.review_id, identity
    ) == terminal_restart.show(created.campaign_id)
    assert terminal_restart.record_secondary(
        created.campaign_id, second.review_id, secondary_observations[0]
    ) == terminal_restart.show(created.campaign_id)
    assert terminal_restart.record_identity(
        created.campaign_id, repeat_binding.session_id, repeat_identity
    ) == terminal_restart.show(created.campaign_id)
    with pytest.raises(CharacterBenchmarkConflict):
        terminal_restart.record_identity(
            created.campaign_id,
            repeat_binding.session_id,
            repeat_identity.model_copy(update={"choice": IdentityChoice.A}),
        )
    assert evidence_counts(database, created.campaign_id) == before_terminal_retry


@pytest.fixture(scope="module")
def completed_benchmark_seed(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Database, Path, str]:
    """Build the completed prerequisite through the real review workflow."""
    root = tmp_path_factory.mktemp("completed-benchmark-seed")
    database = Database(root / "lab.sqlite3")
    database.migrate()
    benchmark = service(database, root)
    created = benchmark.create(
        character_pack_id="character_pack_" + "6" * 64,
        stage_2_campaign_id="validation_stage_2",
        identity_prompt="same character, preserve identity",
        panel_input=panel_input(),
    )
    benchmark.preregister(created.campaign_id)
    complete_jobs(benchmark, root)
    finish(benchmark, created.campaign_id)
    return Database(database.path, read_only=True), root, created.campaign_id


@pytest.fixture
def completed_benchmark(
    database: Database, tmp_path: Path, completed_benchmark_seed: tuple[Database, Path, str]
) -> tuple[CharacterBenchmarkService, str]:
    """Confine corrupted relations and media bytes to each test."""
    seed, root, campaign_id = completed_benchmark_seed
    with closing(seed.connect()) as source, closing(database.connect()) as destination:
        source.backup(destination)
    shutil.copytree(root / "artifacts", tmp_path / "artifacts")
    return service(database, tmp_path, seed=False), campaign_id


@pytest.mark.parametrize(
    "corruption",
    (
        "pack_artifact_bytes",
        "pack_version",
        "stage_2_evidence",
        "experiment_recipe",
        "job_variant_owner",
        "generated_artifact_bytes",
        "artifact_path",
        "artifact_sha",
        "trial_binding",
        "repeat_source",
        "cross_experiment",
        "cross_pack",
        "cross_reviewer",
        "review_revision",
        "display_mapping",
        "identity_judgment",
        "secondary_judgment",
        "secondary_review_revision",
        "event_cardinality",
        "result_digest",
    ),
)
def test_analysis_and_export_fail_closed_for_each_corrupt_relation(
    database, tmp_path: Path, corruption: str, completed_benchmark
) -> None:
    benchmark, campaign_id = completed_benchmark

    if corruption == "pack_artifact_bytes":
        benchmark.character_packs.corrupt_bytes = True
    elif corruption == "pack_version":
        benchmark.character_packs.pack = benchmark.character_packs.pack.model_copy(
            update={"pack_id": "character_pack_" + "9" * 64}
        )
    elif corruption == "stage_2_evidence":
        benchmark.stage_2.evidence["schema_version"] = "corrupt"
    elif corruption == "generated_artifact_bytes":
        with database.connect() as connection:
            artifact_id = str(
                connection.execute("SELECT id FROM artifacts ORDER BY id LIMIT 1").fetchone()[0]
            )
        benchmark.experiments.artifacts.resolve(artifact_id).write_bytes(b"corrupt")
    else:
        with database.connect() as connection:
            if corruption == "experiment_recipe":
                row = connection.execute(
                    "SELECT id, manifest_json FROM variants LIMIT 1"
                ).fetchone()
                payload = json.loads(str(row["manifest_json"]))
                payload["prompt"] = "corrupt recipe"
                connection.execute(
                    "UPDATE variants SET manifest_json=? WHERE id=?",
                    (json.dumps(payload), row["id"]),
                )
            elif corruption == "job_variant_owner":
                row = connection.execute(
                    "SELECT job.id, variant.manifest_json FROM jobs AS job "
                    "JOIN variants AS variant "
                    "ON variant.id=job.variant_id LIMIT 1"
                ).fetchone()
                connection.execute(
                    "INSERT INTO experiments(id,schema_version,variable_path,"
                    "changed_paths_json,created_at) "
                    "VALUES ('rogue_exp','manifest.v1','prompt','[\"prompt\"]',?)",
                    (NOW.isoformat(),),
                )
                connection.execute(
                    "INSERT INTO variants(id,experiment_id,label,manifest_json) "
                    "VALUES ('rogue_variant','rogue_exp','A',?)",
                    (row["manifest_json"],),
                )
                connection.execute(
                    "UPDATE jobs SET variant_id='rogue_variant' WHERE id=?", (row["id"],)
                )
            elif corruption in {"artifact_path", "artifact_sha"}:
                rows = connection.execute(
                    "SELECT id, relative_path FROM artifacts ORDER BY id LIMIT 2"
                ).fetchall()
                if corruption == "artifact_path":
                    connection.execute(
                        "UPDATE artifacts SET relative_path=? WHERE id=?",
                        (rows[1]["relative_path"], rows[0]["id"]),
                    )
                else:
                    connection.execute(
                        "UPDATE artifacts SET sha256=? WHERE id=?", ("0" * 64, rows[0]["id"])
                    )
            elif corruption == "trial_binding":
                connection.execute("DROP TRIGGER guard_character_benchmark_trial_update")
                row = connection.execute(
                    "SELECT campaign_id, trial_id, trial_json "
                    "FROM character_benchmark_trials LIMIT 1"
                ).fetchone()
                payload = json.loads(str(row["trial_json"]))
                payload["candidate_a_position"] = (
                    "right" if payload["candidate_a_position"] == "left" else "left"
                )
                connection.execute(
                    "UPDATE character_benchmark_trials SET trial_json=? "
                    "WHERE campaign_id=? AND trial_id=?",
                    (json.dumps(payload), row["campaign_id"], row["trial_id"]),
                )
            elif corruption == "repeat_source":
                connection.execute("DROP TRIGGER guard_character_benchmark_review_binding_update")
                row = connection.execute(
                    "SELECT campaign_id, trial_id FROM character_benchmark_review_bindings "
                    "WHERE reused_from_trial_id IS NOT NULL LIMIT 1"
                ).fetchone()
                connection.execute(
                    "UPDATE character_benchmark_review_bindings SET reused_from_trial_id=? "
                    "WHERE campaign_id=? AND trial_id=?",
                    (row["trial_id"], row["campaign_id"], row["trial_id"]),
                )
            elif corruption == "cross_experiment":
                connection.execute("DROP TRIGGER guard_character_benchmark_review_binding_update")
                rows = connection.execute(
                    "SELECT experiment_id FROM character_benchmark_experiment_bindings "
                    "ORDER BY trial_id LIMIT 2"
                ).fetchall()
                connection.execute(
                    "UPDATE character_benchmark_review_bindings SET experiment_id=? "
                    "WHERE experiment_id=?",
                    (rows[1]["experiment_id"], rows[0]["experiment_id"]),
                )
            elif corruption == "cross_pack":
                connection.execute("DROP TRIGGER guard_character_benchmark_campaign_update")
                connection.execute(
                    "INSERT INTO character_packs(storage_id,id,character_id,reviewer_id,state,"
                    "supersedes_storage_id,manifest_json,created_at,preregistered_at) "
                    "SELECT 'other-pack','character_pack_' || ?,character_id,reviewer_id,state,"
                    "NULL,manifest_json,created_at,preregistered_at FROM character_packs "
                    "WHERE storage_id='pack-storage'",
                    ("8" * 64,),
                )
                connection.execute(
                    "UPDATE character_benchmark_campaigns "
                    "SET character_pack_storage_id='other-pack', character_pack_id=? "
                    "WHERE id=?",
                    ("character_pack_" + "8" * 64, campaign_id),
                )
            elif corruption == "cross_reviewer":
                connection.execute(
                    "UPDATE review_sessions SET reviewer_id='wrong-reviewer' "
                    "WHERE id=(SELECT session_id FROM character_benchmark_review_bindings LIMIT 1)"
                )
            elif corruption == "review_revision":
                review_id = connection.execute(
                    "SELECT review_id FROM character_benchmark_review_bindings LIMIT 1"
                ).fetchone()[0]
                connection.execute(
                    "UPDATE review_revisions SET revision=2 WHERE id=?", (review_id,)
                )
            elif corruption == "display_mapping":
                connection.execute("DROP TRIGGER guard_character_benchmark_review_binding_update")
                connection.execute(
                    "UPDATE character_benchmark_review_bindings SET display_mapping_json='{}' "
                    "WHERE rowid=(SELECT rowid FROM character_benchmark_review_bindings LIMIT 1)"
                )
            elif corruption == "identity_judgment":
                connection.execute("DROP TRIGGER guard_character_benchmark_identity_update")
                connection.execute(
                    "UPDATE character_benchmark_identity_judgments SET content_digest=? "
                    "WHERE rowid=(SELECT rowid FROM "
                    "character_benchmark_identity_judgments LIMIT 1)",
                    ("sha256:" + "0" * 64,),
                )
            elif corruption == "secondary_judgment":
                connection.execute("DROP TRIGGER guard_character_benchmark_secondary_update")
                connection.execute(
                    "UPDATE character_benchmark_secondary_judgments SET content_digest=? "
                    "WHERE rowid=(SELECT rowid FROM "
                    "character_benchmark_secondary_judgments LIMIT 1)",
                    ("sha256:" + "0" * 64,),
                )
            elif corruption == "secondary_review_revision":
                connection.execute("DROP TRIGGER guard_character_benchmark_secondary_update")
                rows = connection.execute(
                    "SELECT DISTINCT review_id FROM character_benchmark_secondary_judgments "
                    "ORDER BY review_id LIMIT 2"
                ).fetchall()
                connection.execute(
                    "UPDATE character_benchmark_secondary_judgments SET review_id=? "
                    "WHERE review_id=?",
                    (rows[1]["review_id"], rows[0]["review_id"]),
                )
            elif corruption == "event_cardinality":
                connection.execute("DROP TRIGGER guard_character_benchmark_event_delete")
                connection.execute(
                    "DELETE FROM character_benchmark_events WHERE campaign_id=? AND sequence=6",
                    (campaign_id,),
                )
            else:
                connection.execute("DROP TRIGGER guard_character_benchmark_result_update")
                connection.execute(
                    "UPDATE character_benchmark_results SET content_digest=? WHERE campaign_id=?",
                    ("sha256:" + "0" * 64, campaign_id),
                )

    with pytest.raises(CharacterBenchmarkConflict):
        benchmark.analyze(campaign_id)
    with pytest.raises(CharacterBenchmarkConflict):
        benchmark.export_evidence(campaign_id)
