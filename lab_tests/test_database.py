import hashlib
import sqlite3
from pathlib import Path

import pytest

from lab_tests.test_character_benchmark_repository import (
    NOW,
    definition,
    manifest,
    review_dependencies,
    seed_dependencies,
    valid_review_binding,
)
from video_lab.character_benchmark_models import (
    IdentityChoice,
    IdentityObservation,
    SecondaryChoice,
    SecondaryDimension,
    SecondaryObservation,
)
from video_lab.character_benchmark_protocol import schedule_character_trials
from video_lab.character_benchmark_repository import CharacterBenchmarkRepository
from video_lab.character_models import Expression, WarningResolution
from video_lab.character_repository import CharacterRepository
from video_lab.database import (
    _MIGRATION_1,
    _MIGRATION_2,
    _MIGRATION_17,
    _MIGRATION_18,
    _MIGRATIONS,
    SCHEMA_VERSION,
    Database,
    _migration_statements,
    character_benchmark_experiment_id,
    character_benchmark_reviewer_id,
)
from video_lab.domain import canonical_json

ADAPTIVE_EVIDENCE_TABLES = {
    "adaptive_studies",
    "adaptive_study_events",
    "adaptive_observations",
    "adaptive_optimizer_steps",
}
ADAPTIVE_WEB_GRANT_TABLE = "adaptive_web_creation_grants"
CHARACTER_CURATION_GRANT_TABLE = "character_curation_grants"
VALIDATION_EVIDENCE_TABLES = {
    "validation_campaigns",
    "validation_experiment_bindings",
    "validation_events",
    "validation_trials",
    "validation_judgments",
    "validation_results",
}
VALIDATION_IMMUTABLE_TABLES = {*VALIDATION_EVIDENCE_TABLES, "validation_trial_bindings"}
CHARACTER_TABLES = {
    "characters",
    "character_packs",
    "character_references",
    "character_annotations",
    "character_pack_warnings",
    "character_pack_events",
}
CHARACTER_BENCHMARK_TABLES = {
    "character_benchmark_campaigns",
    "character_benchmark_trials",
    "character_benchmark_experiment_bindings",
    "character_benchmark_review_bindings",
    "character_benchmark_identity_judgments",
    "character_benchmark_secondary_judgments",
    "character_benchmark_events",
    "character_benchmark_results",
}


def test_migration_is_idempotent(database: Database) -> None:
    database.migrate()
    with database.connect() as connection:
        version = connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_stage4_migrations_are_append_only_after_version_21(database: Database) -> None:
    assert tuple(_MIGRATIONS) == tuple(range(1, SCHEMA_VERSION + 1))
    assert all(
        _MIGRATIONS[version] is not None for version in range(22, SCHEMA_VERSION + 1)
    )


def test_latest_migration_creates_validation_schema(database: Database) -> None:
    assert SCHEMA_VERSION == 49
    with database.connect() as connection:
        names = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert VALIDATION_EVIDENCE_TABLES <= names
    assert CHARACTER_CURATION_GRANT_TABLE in names


def test_migration_14_appends_character_pack_schema(database: Database) -> None:
    with database.connect() as connection:
        names = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        versions = tuple(
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
    assert versions == tuple(range(1, SCHEMA_VERSION + 1))
    assert CHARACTER_TABLES <= names


def test_connection_context_closes_underlying_handle(database: Database) -> None:
    with database.connect() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_migration_17_appends_immutable_character_benchmark_schema(database: Database) -> None:
    assert hashlib.sha256(_MIGRATION_17.encode()).hexdigest() == (
        "ee297f772013e67badbcdccf2042108a1a88c3d15d4e185ffb3adbadb395cf1b"
    )
    with database.connect() as connection:
        names = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        triggers = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        }

    assert CHARACTER_BENCHMARK_TABLES <= names
    assert {
        "guard_character_benchmark_campaign_update",
        "guard_character_benchmark_trial_update",
        "guard_character_benchmark_event_update",
        "guard_character_benchmark_result_update",
    } <= triggers


def test_migration_18_hardens_character_benchmark_relations(database: Database) -> None:
    assert hashlib.sha256(_MIGRATION_18.encode()).hexdigest() == (
        "19801f7953626e14854ed79e3fc75931c734ef0be0184613f9c41fa8fd99e51c"
    )
    with database.connect() as connection:
        trial_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(character_benchmark_trials)")
        }
        objects = {
            (str(row["type"]), str(row["name"]))
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master WHERE name LIKE '%character_benchmark%'"
            )
        }
    assert "expected_reviewer_id" in trial_columns
    assert {
        ("trigger", "validate_character_benchmark_campaign_insert_v18"),
        ("trigger", "validate_character_benchmark_experiment_binding_insert_v18"),
        ("trigger", "validate_character_benchmark_review_binding_insert_v18"),
        ("trigger", "validate_character_benchmark_identity_insert_v18"),
        ("trigger", "validate_character_benchmark_result_insert_v18"),
    } <= objects


def test_migration_19_restricts_secondary_to_primary_trials(database: Database) -> None:
    with database.connect() as connection:
        triggers = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        }

    assert "validate_character_benchmark_secondary_insert_v19" in triggers
    assert "validate_character_benchmark_secondary_insert_v18" not in triggers


def test_migration_18_upgrades_representative_version_17_evidence(tmp_path: Path) -> None:
    database = Database(tmp_path / "version-17.sqlite3")
    with database.connect() as connection:
        for version in range(1, 18):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )

    seed_dependencies(database)
    campaign = definition()
    trial = schedule_character_trials(campaign)[0]
    campaign_id = campaign.id
    experiment_id = character_benchmark_experiment_id(campaign_id, trial.id)
    reviewer_id = character_benchmark_reviewer_id(campaign_id, trial.id, "owner")
    variant_ids = tuple(spec.id for spec in trial.generation_specs)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO character_benchmark_campaigns VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                campaign_id,
                "owner",
                "sha256:" + "7" * 64,
                canonical_json(campaign),
                "character_pack_" + "6" * 64,
                "pack-storage",
                "validation_stage_2",
                canonical_json(manifest()),
                "same character, preserve identity",
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
            (
                campaign_id,
                1,
                "draft",
                "preregistered",
                "sha256:" + "1" * 64,
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO character_benchmark_trials VALUES (?,?,?,?,?,?,?)",
            (
                campaign_id,
                trial.id,
                trial.ordinal,
                canonical_json(trial),
                0,
                None,
                trial.candidate_a_position,
            ),
        )
        connection.execute(
            "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
            (
                campaign_id,
                2,
                "preregistered",
                "generating",
                "sha256:" + "2" * 64,
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO experiments VALUES (?,?,?,?,?)",
            (
                experiment_id,
                "character-controlled.v1",
                "prompt",
                '["prompt"]',
                NOW.isoformat(),
            ),
        )
        for label, variant_id, job_id in zip(
            ("A", "B"), variant_ids, ("job-v17-a", "job-v17-b"), strict=True
        ):
            connection.execute(
                "INSERT INTO variants VALUES (?,?,?,?)",
                (variant_id, experiment_id, label, canonical_json(manifest())),
            )
            connection.execute(
                "INSERT INTO jobs(id,variant_id,state,created_at,updated_at) "
                "VALUES (?,?,'completed',?,?)",
                (job_id, variant_id, NOW.isoformat(), NOW.isoformat()),
            )
        connection.execute(
            "INSERT INTO character_benchmark_experiment_bindings VALUES (?,?,?)",
            (campaign_id, trial.id, experiment_id),
        )
        for sequence, previous, new in (
            (3, "generating", "ready"),
            (4, "ready", "judging"),
        ):
            connection.execute(
                "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
                (
                    campaign_id,
                    sequence,
                    previous,
                    new,
                    "sha256:" + f"{sequence}" * 64,
                    NOW.isoformat(),
                ),
            )
        for artifact_id, job_id, digest in (
            ("artifact-v17-a", "job-v17-a", "a" * 64),
            ("artifact-v17-b", "job-v17-b", "b" * 64),
        ):
            connection.execute(
                "INSERT INTO artifacts(id,job_id,sha256,byte_size,media_type,relative_path,"
                "created_at) VALUES (?,?,?,1,'video/mp4',?,?)",
                (artifact_id, job_id, digest, f"{digest[:2]}/{digest}", NOW.isoformat()),
            )
        connection.execute(
            "INSERT INTO review_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "session-v17",
                experiment_id,
                reviewer_id,
                variant_ids[0],
                variant_ids[1],
                "job-v17-a",
                "artifact-v17-a",
                "a" * 64,
                "job-v17-b",
                "artifact-v17-b",
                "b" * 64,
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO review_revisions VALUES "
            "('review-v17','session-v17',NULL,'tie',NULL,?,1,NULL)",
            (NOW.isoformat(),),
        )
        connection.execute(
            "INSERT INTO character_benchmark_review_bindings VALUES (?,?,?,?,?,?,?,?)",
            (
                campaign_id,
                trial.id,
                experiment_id,
                "review-v17",
                "session-v17",
                reviewer_id,
                None,
                "{}",
            ),
        )
        identity_json = canonical_json(
            {"benchmark_id": campaign_id, "trial_id": trial.id, "choice": "both"}
        )
        connection.execute(
            "INSERT INTO character_benchmark_identity_judgments VALUES (?,?,?,?,?,?,?,?)",
            (
                campaign_id,
                trial.id,
                "review-v17",
                "both",
                "both",
                identity_json,
                "sha256:" + "c" * 64,
                NOW.isoformat(),
            ),
        )
        secondary_json = canonical_json(
            {
                "benchmark_id": campaign_id,
                "trial_id": trial.id,
                "dimension": "visual_quality",
                "choice": "tie",
            }
        )
        connection.execute(
            "INSERT INTO character_benchmark_secondary_judgments VALUES (?,?,?,?,?,?,?)",
            (
                campaign_id,
                trial.id,
                "visual_quality",
                "tie",
                secondary_json,
                "sha256:" + "d" * 64,
                NOW.isoformat(),
            ),
        )

    database.migrate()
    database.migrate()

    with database.connect() as connection:
        trial_row = connection.execute(
            "SELECT expected_reviewer_id FROM character_benchmark_trials"
        ).fetchone()
        secondary_row = connection.execute(
            "SELECT review_id FROM character_benchmark_secondary_judgments"
        ).fetchone()
        versions = tuple(
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert trial_row is not None and trial_row["expected_reviewer_id"] == reviewer_id
        assert secondary_row is not None and secondary_row["review_id"] == "review-v17"
        assert versions == tuple(range(1, SCHEMA_VERSION + 1))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE character_benchmark_trials SET expected_reviewer_id=expected_reviewer_id"
            )


def test_migration_19_upgrades_representative_version_18_evidence(tmp_path: Path) -> None:
    database = Database(tmp_path / "version-18.sqlite3")
    with database.connect() as connection:
        for version in range(1, 19):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )

    seed_dependencies(database)
    repository = CharacterBenchmarkRepository(database, reviewer_id="owner")
    campaign, trials, first, _, alias = review_dependencies(database, repository)
    repository.bind_review_or_verify(
        valid_review_binding(campaign.definition.id, trials[0].id, first.experiment_id, alias)
    )
    identity = IdentityObservation(
        benchmark_id=campaign.definition.id,
        trial_id=trials[0].id,
        choice=IdentityChoice.BOTH,
    )
    repository.record_identity_or_verify(
        identity,
        review_id="review-revision-1",
        display_choice="both",
        created_at=NOW,
    )
    observation = SecondaryObservation(
        benchmark_id=campaign.definition.id,
        trial_id=trials[0].id,
        dimension=SecondaryDimension.VISUAL_QUALITY,
        choice=SecondaryChoice.TIE,
    )
    repository.record_secondary_or_verify(observation, created_at=NOW)

    database.migrate()
    database.migrate()

    with database.connect() as connection:
        versions = tuple(
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        persisted = connection.execute(
            "SELECT observation_json, review_id FROM character_benchmark_secondary_judgments"
        ).fetchone()
        triggers = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    assert versions == tuple(range(1, SCHEMA_VERSION + 1))
    assert persisted is not None
    assert persisted["observation_json"] == canonical_json(observation)
    assert persisted["review_id"] == "review-revision-1"
    assert "validate_character_benchmark_secondary_insert_v19" in triggers


def test_migration_15_upgrades_exact_version_14_character_data(tmp_path: Path) -> None:
    database = Database(tmp_path / "version-14.sqlite3")
    with database.connect() as connection:
        for version in range(1, 15):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )

        pack_columns = tuple(
            str(row["name"]) for row in connection.execute("PRAGMA table_info(character_packs)")
        )
        assert pack_columns == (
            "id",
            "character_id",
            "reviewer_id",
            "state",
            "supersedes_pack_id",
            "manifest_json",
            "created_at",
            "preregistered_at",
        )
        timestamp = "2026-07-21T12:00:00+00:00"
        connection.execute(
            "INSERT INTO characters VALUES ('character-1', 'reviewer', 'Mara', ?)",
            (timestamp,),
        )
        for artifact_id, digest in (("artifact-a", "a" * 64), ("artifact-b", "b" * 64)):
            connection.execute(
                "INSERT INTO artifacts(id, sha256, byte_size, media_type, relative_path, "
                "created_at) VALUES (?, ?, 1, 'image/png', ?, ?)",
                (artifact_id, digest, f"aa/{digest}", timestamp),
            )
        connection.execute(
            "INSERT INTO character_packs VALUES "
            "('draft-pack', 'character-1', 'reviewer', 'draft', NULL, NULL, ?, NULL)",
            (timestamp,),
        )
        connection.execute(
            "INSERT INTO character_packs VALUES "
            "('frozen-pack', 'character-1', 'reviewer', 'draft', NULL, NULL, ?, NULL)",
            (timestamp,),
        )
        connection.execute(
            "INSERT INTO character_packs VALUES "
            "('legacy-successor', 'character-1', 'reviewer', 'draft', "
            "'draft-pack', NULL, ?, NULL)",
            (timestamp,),
        )
        connection.execute(
            "INSERT INTO character_pack_aliases VALUES ('draft-alias', 'draft-pack')"
        )
        for pack_id in ("draft-pack", "frozen-pack"):
            for ordinal, (reference_id, artifact_id, digest, dhash) in enumerate(
                (
                    ("ref-a", "artifact-a", "a" * 64, "0000000000000000"),
                    ("ref-b", "artifact-b", "b" * 64, "0000000000000001"),
                )
            ):
                connection.execute(
                    "INSERT INTO character_references VALUES (?, ?, ?, ?, ?, 'image/png', ?, "
                    "'active', ?)",
                    (
                        pack_id,
                        reference_id,
                        artifact_id,
                        f"{reference_id}.png",
                        digest,
                        dhash,
                        ordinal,
                    ),
                )
                connection.execute(
                    "INSERT INTO character_annotations VALUES (?, ?, 'portrait', 'front', "
                    "'clear_face', 'neutral', 'ordinary', 'none', 1)",
                    (pack_id, reference_id),
                )
            connection.execute(
                "INSERT INTO character_pack_warnings VALUES (?, 'ref-a', 'ref-b', 1, ?)",
                (pack_id, "unresolved" if pack_id == "draft-pack" else "accepted"),
            )
            connection.execute(
                "INSERT INTO character_pack_events VALUES "
                "(?, 1, 'reviewer', 'draft_created', '{\"pack_id\":\"original\"}', ?)",
                (pack_id, timestamp),
            )
        frozen_manifest = (
            '{"character_id":"character-1","duplicate_warnings":[],"pack_id":"frozen-pack",'
            '"references":[],"supersedes_pack_id":null}'
        )
        connection.execute(
            "UPDATE character_packs SET state='preregistered', manifest_json=?, "
            "preregistered_at=? WHERE id='frozen-pack'",
            (frozen_manifest, timestamp),
        )
        connection.execute(
            "INSERT INTO character_pack_events VALUES "
            "('frozen-pack', 2, 'reviewer', 'preregistered', '{\"digest\":\"kept\"}', ?)",
            (timestamp,),
        )
        connection.execute(
            "INSERT INTO character_pack_events VALUES "
            "('legacy-successor', 1, 'reviewer', 'draft_created', "
            '\'{"pack_id":"legacy-successor"}\', ?)',
            (timestamp,),
        )

    database.migrate()
    database.migrate()

    with database.connect() as connection:
        versions = tuple(
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        packs = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT storage_id, id, state, reviewer_id FROM character_packs ORDER BY storage_id"
            )
        )
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in CHARACTER_TABLES | {"character_pack_aliases"}
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert versions == tuple(range(1, SCHEMA_VERSION + 1))
    assert tuple((row[0], row[2], row[3]) for row in packs) == (
        ("draft-pack", "draft", "reviewer"),
        ("frozen-pack", "preregistered", "reviewer"),
        ("legacy-successor", "draft", "reviewer"),
    )
    assert packs[0][1].startswith("character_pack_") and packs[0][1] != "draft-pack"
    assert packs[1][1] == "frozen-pack"
    assert packs[2][1].startswith("character_pack_") and packs[2][1] != "legacy-successor"
    assert counts == {
        "characters": 1,
        "character_packs": 3,
        "character_pack_aliases": 3,
        "character_references": 4,
        "character_annotations": 4,
        "character_pack_warnings": 2,
        "character_pack_events": 4,
    }

    repository = CharacterRepository(database, reviewer_id="reviewer")
    migrated = repository.load_pack("draft-alias")
    assert migrated.record.pack_id == packs[0][1]
    assert [event.payload for event in migrated.events] == [{"pack_id": "original"}]
    legacy_successor = repository.load_pack("legacy-successor")
    assert legacy_successor.record.pack_id == packs[2][1]
    assert legacy_successor.record.supersedes_pack_id == migrated.record.pack_id
    replacement = migrated.references[0].model_copy(
        update={
            "reference": migrated.references[0].reference.model_copy(
                update={
                    "annotation": migrated.references[0].reference.annotation.model_copy(
                        update={"expression": Expression.ANGRY}
                    )
                }
            )
        }
    )
    repository.replace_draft_curation("draft-alias", replacement)
    rekeyed_predecessor = repository.load_pack("draft-alias")
    assert rekeyed_predecessor.record.pack_id != "draft-pack"
    reloaded_successor = repository.load_pack("legacy-successor")
    assert reloaded_successor.record.supersedes_pack_id == rekeyed_predecessor.record.pack_id
    with database.connect() as connection:
        lineage = connection.execute(
            "SELECT supersedes_storage_id FROM character_packs WHERE storage_id='legacy-successor'"
        ).fetchone()
        assert lineage is not None
        assert lineage["supersedes_storage_id"] == "draft-pack"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM character_pack_events").fetchone()[0] == 4
    repository.resolve_warning("draft-alias", "ref-a", "ref-b", WarningResolution.ACCEPTED)
    rekeyed = repository.load_pack("draft-alias")
    assert rekeyed.warnings[0].resolution is WarningResolution.ACCEPTED


def test_migration_11_rejects_campaign_for_nonexistent_stage_1_study(database: Database) -> None:
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO validation_campaigns("
            "id, schema_version, reviewer_id, stage_1_study_id, stage_1_evidence_digest, "
            "base_manifest_json, candidates_json, panel_input_json, protocol_version, "
            "analysis_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "validation_missing_stage_1",
                "validation-campaign.v1",
                "owner",
                "missing_study",
                "a" * 64,
                "{}",
                "{}",
                "{}",
                "balanced-validation.v1",
                "dirichlet-validation.v1",
                "2026-07-21T12:00:00+00:00",
            ),
        )


def test_migration_10_adds_guarded_adaptive_web_creation_grants(
    database: Database,
) -> None:
    with database.connect() as connection:
        tables = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        triggers = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        }

    assert {*ADAPTIVE_EVIDENCE_TABLES, ADAPTIVE_WEB_GRANT_TABLE} <= tables
    expected_guards = {
        f"guard_{table}_{operation}"
        for table in ADAPTIVE_EVIDENCE_TABLES
        for operation in ("update", "delete")
    }
    validation_guards = {
        f"guard_{table}_{operation}"
        for table in VALIDATION_IMMUTABLE_TABLES
        for operation in ("update", "delete")
    }
    assert {
        "guard_feasibility_stage_attempt_updates",
        "validate_adaptive_study_definition_insert",
        "validate_adaptive_study_event_insert",
        "validate_adaptive_optimizer_step_insert",
        "validate_adaptive_observation_insert",
        "validate_adaptive_web_creation_grant_update",
        *expected_guards,
        *validation_guards,
    } <= triggers


def test_migration_10_upgrades_a_version_9_database_idempotently(tmp_path: Path) -> None:
    database = Database(tmp_path / "version-9.sqlite3")
    with database.connect() as connection:
        for version in range(1, 10):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )

    database.migrate()
    database.migrate()

    with database.connect() as connection:
        versions = tuple(
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        tables = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert versions == tuple(range(1, SCHEMA_VERSION + 1))
    assert {
        *ADAPTIVE_EVIDENCE_TABLES,
        ADAPTIVE_WEB_GRANT_TABLE,
        *VALIDATION_EVIDENCE_TABLES,
    } <= tables


def test_migration_11_remains_historical_and_12_adds_experiment_bindings(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "version-11.sqlite3")
    with database.connect() as connection:
        for version in range(1, 12):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )
        tables_before = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert "validation_experiment_bindings" not in tables_before

    database.migrate()

    with database.connect() as connection:
        versions = tuple(
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        objects = {
            (str(row["type"]), str(row["name"]))
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name LIKE '%validation_experiment_binding%'"
            )
        }

    assert versions == tuple(range(1, SCHEMA_VERSION + 1))
    assert {
        ("table", "validation_experiment_bindings"),
        ("index", "validation_experiment_binding_experiment"),
        ("trigger", "validate_validation_experiment_binding_insert"),
        ("trigger", "guard_validation_experiment_bindings_update"),
        ("trigger", "guard_validation_experiment_bindings_delete"),
    } <= objects


@pytest.mark.parametrize(
    ("fingerprint", "created_at", "manifest_json", "config_json"),
    (
        ("A" * 64, "2026-07-20T12:00:00+00:00", None, None),
        ("a" * 64, "2026-07-20T12:00:00", None, None),
        ("a" * 64, "2026-07-20T12:00:00+00:00", "[]", None),
        ("a" * 64, "2026-07-20T12:00:00+00:00", None, "[]"),
    ),
)
def test_adaptive_study_schema_rejects_invalid_digest_timestamp_and_json_shapes(
    database: Database,
    fingerprint: str,
    created_at: str,
    manifest_json: str | None,
    config_json: str | None,
) -> None:
    valid_manifest = (
        '{"schema_version":"manifest.v1","prompt":"prompt","negative_prompt":"","seed":1}'
    )
    valid_config = (
        '{"schema_version":"preference-optimizer-config.v1",'
        '"optimizer_version":"preference-gp-laplace.v1"}'
    )
    with database.connect() as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(
                "INSERT INTO adaptive_studies VALUES (?, 'adaptive-study.v1', "
                "'prompt', NULL, 1, ?, ?, ?, ?)",
                (
                    f"study_{fingerprint[:4]}_{created_at}_{manifest_json}_{config_json}",
                    valid_manifest if manifest_json is None else manifest_json,
                    valid_config if config_json is None else config_json,
                    fingerprint,
                    created_at,
                ),
            )


def test_migration_3_backfills_legacy_job_artifact_relationship(tmp_path: Path) -> None:
    database = Database(tmp_path / "legacy.sqlite3")
    with database.connect() as connection:
        for migration in (_MIGRATION_1, _MIGRATION_2):
            connection.executescript(migration)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, 'now'), (2, 'now')"
        )
        connection.execute("INSERT INTO experiments VALUES ('exp', 'v1', 'seed', '[]', 'now')")
        connection.execute("INSERT INTO variants VALUES ('var', 'exp', 'A', '{}')")
        connection.execute(
            "INSERT INTO jobs(id, variant_id, state, created_at, updated_at) "
            "VALUES ('job', 'var', 'queued', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO artifacts VALUES ('art', 'job', 'hash', 1, 'video/mp4', 'path', 'now')"
        )

    database.migrate()
    database.migrate()

    with database.connect() as connection:
        links = connection.execute("SELECT job_id, artifact_id FROM job_artifacts").fetchall()
    assert [tuple(row) for row in links] == [("job", "art")]


def test_connection_enables_foreign_keys_and_wal(database: Database) -> None:
    with database.connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_connection_registers_adaptive_boundary_functions_for_every_mode(
    database: Database,
) -> None:
    for configured in (database, Database(database.path, read_only=True)):
        with configured.connect() as connection:
            row = connection.execute(
                "SELECT adaptive_manifest_json_is_valid('{}'), "
                "adaptive_optimizer_config_json_is_valid('{}'), "
                "adaptive_observation_provenance_digest("
                "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
            ).fetchone()
        assert tuple(row) == (0, 0, None)
