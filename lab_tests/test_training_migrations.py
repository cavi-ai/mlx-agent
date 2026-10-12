from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from lab_tests.test_training_models import _confirmation, _imported
from lab_tests.test_training_repository import NOW, PACK_DIGEST, PACK_MANIFEST, _frozen
from video_lab.database import _MIGRATIONS, SCHEMA_VERSION, Database, _migration_statements
from video_lab.domain import canonical_json
from video_lab.training_models import (
    CurationDecision,
    TrainingDatasetDraft,
    TrainingMedia,
    training_dataset_id,
)
from video_lab.training_repository import TrainingRepository


def test_scientific_lifecycle_migration_is_append_only() -> None:
    assert SCHEMA_VERSION >= 35
    assert 35 in _MIGRATIONS
    assert "training_probe_evidence" in _MIGRATIONS[35]


TABLES = {
    "training_datasets",
    "training_dataset_aliases",
    "training_samples",
    "training_sample_annotations",
    "training_recipes",
    "training_runs",
    "training_run_events",
    "training_checkpoints",
    "training_telemetry",
    "training_failures",
    "training_orchestration_operations",
    "training_calibration_operation_bindings",
    "training_cancellation_operation_claims",
    "training_checkpoint_inventory_artifacts",
    "training_dataset_lineages",
    "training_imported_samples",
    "training_curation_confirmations",
    "training_dataset_snapshots",
}


def _populated_v25_database(path: Path) -> tuple[Database, str, str]:
    database = Database(path)
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for version in range(1, 26):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if not statement.lstrip().startswith("CREATE TABLE schema_migrations"):
                    connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
        connection.execute(
            "INSERT INTO characters VALUES ('character_1','owner','Character',?)",
            (NOW.isoformat(),),
        )
        connection.execute(
            "INSERT INTO character_packs(storage_id,id,character_id,reviewer_id,state,"
            "supersedes_storage_id,manifest_json,created_at,preregistered_at) "
            "VALUES ('pack-storage',?,'character_1','owner','preregistered',NULL,?,?,?)",
            (
                "character_pack_" + "6" * 64,
                canonical_json(PACK_MANIFEST),
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        frozen = _frozen()
        draft = TrainingDatasetDraft.model_validate(
            frozen.model_dump(mode="python", exclude={"coverage", "dataset_digest"})
        )
        storage = "v25-snapshot-" + uuid.uuid4().hex
        dataset_id = training_dataset_id(draft)
        connection.execute(
            "INSERT INTO training_datasets(storage_id,id,reviewer_id,character_id,"
            "character_pack_storage_id,character_pack_digest,state,canonical_outfit_id,"
            "reserved_token,source_tier_weights_json,manifest_json,dataset_digest,"
            "created_at,sealed_at) "
            "VALUES (?,?,?,?,?,?,'draft',?,?,?,?,NULL,?,NULL)",
            (
                storage,
                dataset_id,
                "owner",
                "character_1",
                "pack-storage",
                PACK_DIGEST,
                draft.canonical_outfit_id,
                draft.reserved_token,
                canonical_json(
                    {tier.value: weight for tier, weight in draft.source_tier_weights.items()}
                ),
                canonical_json(draft.model_dump(mode="json")),
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO training_dataset_lineages VALUES (?,?,?,?,1,'draft',?)",
            ("v25-lineage", "owner", "character_1", storage, NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO training_dataset_head_events VALUES (?,?,?,?,NULL,?,'draft',?)",
            ("v25-lineage", 1, "owner", "character_1", storage, NOW.isoformat()),
        )
    return database, storage, dataset_id


def test_fresh_replay_creates_stage4_schema_at_current_version(tmp_path: Path) -> None:
    database = Database(tmp_path / "fresh.sqlite3")
    database.migrate()
    with database.connect() as connection:
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        versions = tuple(
            row[0]
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        head_history_kind = connection.execute(
            "SELECT type FROM sqlite_master WHERE name='training_dataset_head_events'"
        ).fetchone()[0]
        lineage_registry_kind = connection.execute(
            "SELECT type FROM sqlite_master WHERE name='training_dataset_snapshot_lineages'"
        ).fetchone()[0]
    assert SCHEMA_VERSION == max(_MIGRATIONS)
    assert 45 in _MIGRATIONS
    assert "CHECK(steps>=0)" in _MIGRATIONS[45]
    assert TABLES <= names
    assert {"training_dataset_snapshot_imports", "training_dataset_snapshot_confirmations"} <= names
    assert head_history_kind == "view"
    assert lineage_registry_kind == "view"
    assert versions == tuple(range(1, SCHEMA_VERSION + 1))


def _empty_physical_v44_database(path: Path) -> Database:
    database = Database(path)
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for version in range(1, 45):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if not statement.lstrip().startswith("CREATE TABLE schema_migrations"):
                    connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
        connection.commit()
    return database


def test_physical_v44_binding_rebuild_rolls_back_atomically_on_orphaned_row(
    tmp_path: Path,
) -> None:
    database = _empty_physical_v44_database(tmp_path / "malformed-v44.sqlite3")
    with database.connect() as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='validate_training_calibration_operation_binding_v42'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER validate_training_calibration_operation_binding_v42")
        connection.execute(
            "INSERT INTO training_calibration_operation_bindings "
            "(evidence_id,operation_id,request_digest,attempt,claim_token,steps,"
            "trainable_targets_json,seed,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "missing-evidence",
                "missing-operation",
                "sha256:" + "1" * 64,
                1,
                "claim-token",
                1,
                '["layers.1"]',
                7,
                NOW.isoformat(),
            ),
        )
        connection.execute(trigger_sql)
        connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        database.migrate()

    with database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 44
        assert connection.execute(
            "SELECT steps FROM training_calibration_operation_bindings"
        ).fetchone()[0] == 1
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='training_calibration_operation_bindings'"
        ).fetchone()[0]
        assert "CHECK(steps>0)" in table_sql
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master "
            "WHERE name='training_calibration_operation_bindings_v44'"
        ).fetchone()[0] == 0
        trigger_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND tbl_name='training_calibration_operation_bindings'"
            )
        }
        assert trigger_names == {
            "validate_training_calibration_operation_binding_v42",
            "guard_training_calibration_operation_binding_update_v42",
            "guard_training_calibration_operation_binding_delete_v42",
        }


def test_operation_kind_lifecycle_and_completion_target_sql_matrices(tmp_path: Path) -> None:
    database = Database(tmp_path / "operation-lifecycle-matrix.sqlite3")
    database.migrate()
    creation = {
        "probe": "draft",
        "calibration": "probed",
        "start": "preregistered",
        "resume": "training",
        "cancel": "training",
    }
    completion = {
        "probe": "probed",
        "calibration": "calibrated",
        "start": "evaluating",
        "resume": "evaluating",
        "cancel": "cancelled",
    }
    states = (
        "draft",
        "probed",
        "calibrated",
        "preregistered",
        "training",
        "evaluating",
        "qualified",
        "rejected",
        "failed",
        "cancelled",
    )
    with database.connect() as connection:
        for kind in creation:
            for state in states:
                expected_creation_states = (
                    {"training", "cancelled"} if kind == "resume" else {creation[kind]}
                )
                assert connection.execute(
                    "SELECT training_operation_lifecycle_is_valid(?,?)", (kind, state)
                ).fetchone()[0] == int(state in expected_creation_states)
                assert connection.execute(
                    "SELECT training_operation_completion_target_is_valid(?,?)", (kind, state)
                ).fetchone()[0] == int(state == completion[kind])


def test_real_v38_upgrade_appends_receipt_ledger_and_reopens(tmp_path: Path) -> None:
    database = Database(tmp_path / "physical-v38.sqlite3")
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for version in range(1, 39):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if not statement.lstrip().startswith("CREATE TABLE schema_migrations"):
                    connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
        connection.commit()
    database.migrate()
    database.migrate()
    with database.connect() as connection:
        assert (
            connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            == SCHEMA_VERSION
        )
        assert (
            connection.execute("SELECT count(*) FROM training_orchestration_operations").fetchone()[
                0
            ]
            == 0
        )
        trigger_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE '%training_operation%v39'"
            )
        }
        assert trigger_names == {"guard_training_operation_delete_v39"}


def test_v36_building_checkpoint_fails_v37_upgrade_atomically(tmp_path: Path) -> None:
    database = Database(tmp_path / "building-v36.sqlite3")
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for version in range(1, 37):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if not statement.lstrip().startswith("CREATE TABLE schema_migrations"):
                    connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
        connection.execute("DROP TRIGGER validate_training_checkpoint_insert_v36")
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "INSERT INTO training_checkpoints(id,run_id,reviewer_id,recipe_storage_id,"
            "dataset_storage_id,step,fraction,parent_checkpoint_id,artifact_digest,manifest_digest,"
            "manifest_json,created_at,evidence_json,evidence_digest,publication_state) VALUES "
            "('poison','missing','owner','recipe','dataset',1,0.25,NULL,?,?,"
            "'{}','now','{}',?,'building')",
            ("sha256:" + "1" * 64, "sha256:" + "2" * 64, "sha256:" + "3" * 64),
        )
        connection.commit()
    with pytest.raises(sqlite3.IntegrityError, match="building checkpoints"):
        database.migrate()
    with database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 36
        assert (
            connection.execute(
                "SELECT publication_state FROM training_checkpoints WHERE id='poison'"
            ).fetchone()[0]
            == "building"
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM pragma_table_info('training_cancellation_evidence') "
                "WHERE name='cancellation_request_id'"
            ).fetchone()[0]
            == 0
        )


def _database_at_v31(path: Path, *, populated_recipe: bool) -> Database:
    database = Database(path)
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for version in range(1, 32):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if not statement.lstrip().startswith("CREATE TABLE schema_migrations"):
                    connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
        connection.commit()
        if populated_recipe:
            connection.execute("DROP TRIGGER validate_training_recipe_insert")
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute(
                "INSERT INTO training_recipes VALUES "
                "('legacy-storage','legacy-id','owner','missing-dataset','sha256:'||printf('%064d',0),"
                "'character','sha256:'||printf('%064d',0),'draft','{}','now',NULL)"
            )
            connection.commit()
    return database


def test_empty_v31_migrates_but_populated_v31_recipe_fails_atomically(tmp_path: Path) -> None:
    empty = _database_at_v31(tmp_path / "empty-v31.sqlite3", populated_recipe=False)
    empty.migrate()
    with empty.connect() as connection:
        assert (
            connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            == SCHEMA_VERSION
        )

    populated = _database_at_v31(tmp_path / "populated-v31.sqlite3", populated_recipe=True)
    with pytest.raises(sqlite3.IntegrityError, match="explicit material migration"):
        populated.migrate()
    with populated.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 31
        assert connection.execute("SELECT id FROM training_recipes").fetchone()[0] == "legacy-id"
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='training_recipe_materials'"
            ).fetchone()
            is None
        )


def test_upgrade_from_21_preserves_existing_data_and_reopen_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "upgrade.sqlite3"
    database = Database(path)
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for version in range(1, 22):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if not statement.lstrip().startswith("CREATE TABLE schema_migrations"):
                    connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version,applied_at) VALUES (?,'now')", (version,)
            )
        connection.execute(
            "INSERT INTO experiments VALUES "
            "('legacy','manifest.v1','seed','[]','2026-01-01T00:00:00+00:00')"
        )
    database.migrate()
    database.migrate()
    with database.connect() as connection:
        assert (
            connection.execute("SELECT id FROM experiments WHERE id='legacy'").fetchone()[0]
            == "legacy"
        )
        assert (
            connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            == SCHEMA_VERSION
        )


def test_upgrade_from_24_preserves_training_rows(tmp_path: Path) -> None:
    path = tmp_path / "upgrade24.sqlite3"
    database = Database(path)
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for version in range(1, 25):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if not statement.lstrip().startswith("CREATE TABLE schema_migrations"):
                    connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
        connection.execute(
            "INSERT INTO experiments VALUES ('kept','manifest.v1','seed','[]','now')"
        )
    database.migrate()
    database.migrate()
    with database.connect() as connection:
        assert (
            connection.execute("SELECT id FROM experiments WHERE id='kept'").fetchone()[0] == "kept"
        )
        assert (
            connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            == SCHEMA_VERSION
        )


def test_conflicting_populated_v25_history_fails_before_partial_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conflicting-v25.sqlite3"
    database, first_storage, _ = _populated_v25_database(path)
    with database.connect() as connection:
        frozen = _frozen()
        changed = TrainingDatasetDraft.model_validate(
            frozen.model_dump(mode="python", exclude={"coverage", "dataset_digest"})
        ).model_copy(
            update={
                "samples": (
                    frozen.samples[0].model_copy(update={"caption": "<mirza> v25 successor"}),
                )
                + frozen.samples[1:]
            }
        )
        second_storage = "v25-successor-" + uuid.uuid4().hex
        connection.execute(
            "INSERT INTO training_datasets(storage_id,id,reviewer_id,character_id,"
            "character_pack_storage_id,character_pack_digest,state,canonical_outfit_id,"
            "reserved_token,source_tier_weights_json,manifest_json,dataset_digest,"
            "created_at,sealed_at) "
            "VALUES (?,?,?,?,?,?,'draft',?,?,?,?,NULL,?,NULL)",
            (
                second_storage,
                training_dataset_id(changed),
                "owner",
                "character_1",
                "pack-storage",
                PACK_DIGEST,
                changed.canonical_outfit_id,
                changed.reserved_token,
                canonical_json(
                    {tier.value: weight for tier, weight in changed.source_tier_weights.items()}
                ),
                canonical_json(changed.model_dump(mode="json")),
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO training_dataset_head_events VALUES (?,?,?,?,?,?,?,?)",
            (
                "v25-lineage",
                2,
                "owner",
                "character_1",
                first_storage,
                second_storage,
                "draft",
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "UPDATE training_dataset_lineages SET current_dataset_storage_id=?,head_version=2 "
            "WHERE id='v25-lineage'",
            (second_storage,),
        )
        connection.execute(
            "INSERT INTO training_dataset_lineages VALUES (?,?,?,?,1,'draft',?)",
            ("v25-conflict", "owner", "character_1", first_storage, NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO training_dataset_head_events VALUES (?,?,?,?,NULL,?,'draft',?)",
            ("v25-conflict", 1, "owner", "character_1", first_storage, NOW.isoformat()),
        )
    with pytest.raises(
        sqlite3.IntegrityError, match="snapshot appears in multiple dataset lineages"
    ):
        database.migrate()
    with database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 25
        assert (
            connection.execute("SELECT count(*) FROM training_dataset_head_events").fetchone()[0]
            == 3
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_master WHERE name='training_dataset_snapshots'"
            ).fetchone()[0]
            == 0
        )


def test_actual_populated_v25_upgrade_preserves_exact_lineage_head_and_history(
    tmp_path: Path,
) -> None:
    database, storage_id, dataset_id = _populated_v25_database(tmp_path / "actual-v25.sqlite3")
    database.migrate()
    database.migrate()
    with database.connect() as connection:
        head = connection.execute(
            "SELECT current_dataset_storage_id,head_version,state "
            "FROM training_dataset_lineages WHERE id='v25-lineage'"
        ).fetchone()
        history = connection.execute(
            "SELECT lineage_id,version,previous_dataset_storage_id,dataset_storage_id,state "
            "FROM training_dataset_head_events WHERE lineage_id='v25-lineage' ORDER BY version"
        ).fetchall()
        dataset = connection.execute(
            "SELECT id,lineage_id FROM training_datasets WHERE storage_id=?", (storage_id,)
        ).fetchone()
        assert tuple(head) == (storage_id, 1, "draft")
        assert tuple(history[0]) == ("v25-lineage", 1, None, storage_id, "draft")
        assert tuple(dataset) == (dataset_id, "v25-lineage")
        assert (
            connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            == SCHEMA_VERSION
        )


def test_actual_v29_incompatible_curation_ledger_fails_atomically(tmp_path: Path) -> None:
    database, _, _ = _populated_v25_database(tmp_path / "conflicting-v29.sqlite3")
    with database.connect() as connection:
        for version in range(26, 30):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
        imported = _imported(lineage_id="v25-lineage", character_id="character_1")
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "v29/path",
                NOW.isoformat(),
            ),
        )
    repository = TrainingRepository(database, reviewer_id="owner")
    with database.connect() as connection:
        repository.persist_imported_sample_with_connection(connection, imported)
        repository.persist_confirmation_with_connection(
            connection, _confirmation(imported.import_id)
        )
    with pytest.raises(sqlite3.IntegrityError, match="v29 training dataset ledger"):
        database.migrate()
    with database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 29
        assert (
            connection.execute("SELECT count(*) FROM training_curation_confirmations").fetchone()[0]
            == 1
        )


def test_actual_populated_coherent_v29_upgrades_and_remains_publishable(tmp_path: Path) -> None:
    database, _, dataset_id = _populated_v25_database(tmp_path / "coherent-v29.sqlite3")
    with database.connect() as connection:
        for version in range(26, 30):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute("INSERT INTO schema_migrations VALUES (?,'now')", (version,))
    repository = TrainingRepository(database, reviewer_id="owner")
    initial = repository.get_dataset(dataset_id).draft
    imported = _imported(lineage_id="v25-lineage", character_id="character_1")
    rejected = _confirmation(
        imported.import_id,
        decision=CurationDecision.REJECT,
        sample_id=None,
        source_tier=None,
        source_lineage=None,
        caption=None,
        split=None,
        annotation=None,
    )
    imported_draft = initial.model_copy(update={"imported_sample_ids": (imported.import_id,)})
    curated = imported_draft.model_copy(
        update={"curation_confirmation_ids": (rejected.confirmation_id,)}
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                imported.artifact_id,
                None,
                imported.artifact_digest[7:],
                imported.artifact_byte_size,
                imported.media.media_type,
                "coherent/path",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        repository.persist_imported_sample_with_connection(connection, imported)
        repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="v25-lineage",
            expected_head_version=1,
            draft=imported_draft,
        )
        repository.persist_confirmation_with_connection(connection, rejected)
        current_id = repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="v25-lineage",
            expected_head_version=2,
            draft=curated,
        )
        connection.commit()
    current_before = repository.get_dataset(current_id)
    history = repository.list_dataset_head_history("v25-lineage")
    database.migrate()
    reopened = TrainingRepository(database, reviewer_id="owner")
    lineage = reopened.get_dataset_lineage("v25-lineage")
    current_after = reopened.get_dataset(current_id)
    assert lineage.current_dataset_id == current_id
    assert lineage.head_version == 3
    assert lineage.state == "draft"
    assert current_after.storage_id == current_before.storage_id
    assert current_after.dataset_id == current_id == training_dataset_id(curated)
    assert current_after.state == "draft"
    assert current_after.dataset_digest is None
    assert current_after.draft == curated
    assert current_after.draft.samples == initial.samples
    assert current_after.draft.imported_sample_ids == (imported.import_id,)
    assert current_after.draft.curation_confirmation_ids == (rejected.confirmation_id,)
    assert reopened.list_imported_samples("v25-lineage") == (imported,)
    assert reopened.list_confirmations(imported.import_id) == (rejected,)
    assert reopened.list_dataset_head_history("v25-lineage") == history
    with database.connect() as connection:
        assert (
            tuple(
                row[0]
                for row in connection.execute(
                    "SELECT import_id FROM training_dataset_snapshot_imports "
                    "WHERE dataset_storage_id=? ORDER BY import_id",
                    (current_after.storage_id,),
                )
            )
            == current_after.draft.imported_sample_ids
        )
        assert (
            tuple(
                row[0]
                for row in connection.execute(
                    "SELECT confirmation_id FROM training_dataset_snapshot_confirmations "
                    "WHERE dataset_storage_id=? ORDER BY confirmation_id",
                    (current_after.storage_id,),
                )
            )
            == current_after.draft.curation_confirmation_ids
        )

    second = _imported(
        lineage_id="v25-lineage",
        character_id="character_1",
        artifact_id="second-artifact",
        media=TrainingMedia(
            sha256="sha256:" + "6" * 64, media_type="image/png", byte_size=16, width=4, height=3
        ),
        artifact_digest="sha256:" + "6" * 64,
        artifact_byte_size=16,
    )
    next_draft = curated.model_copy(
        update={"imported_sample_ids": (imported.import_id, second.import_id)}
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (
                second.artifact_id,
                None,
                second.artifact_digest[7:],
                second.artifact_byte_size,
                second.media.media_type,
                "coherent/second",
                NOW.isoformat(),
            ),
        )
        connection.execute("BEGIN IMMEDIATE")
        reopened.persist_imported_sample_with_connection(connection, second)
        reopened.publish_dataset_head_with_connection(
            connection,
            lineage_id="v25-lineage",
            expected_head_version=3,
            draft=next_draft,
        )
        connection.commit()
    assert len(reopened.list_dataset_head_history("v25-lineage")) == 4
