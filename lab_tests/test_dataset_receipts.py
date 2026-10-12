from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

import video_lab.database as database_module
from video_lab.database import SCHEMA_VERSION, Database
from video_lab.dataset_receipts import (
    CompilationConflict,
    CompilationOutput,
    DatasetCompilationReceipt,
    DatasetCompilationRepository,
    DatasetCompilationRequest,
    InputBinding,
)
from video_lab.domain import canonical_json

SHA = "a" * 64


def request(*, digest: str = SHA) -> DatasetCompilationRequest:
    return DatasetCompilationRequest(
        operation_key="compile_fixture",
        compiler_id="test_compiler_v1",
        source=InputBinding(
            role="source_manifest",
            artifact_id="art_source",
            sha256=digest,
            byte_size=12,
            media_type="application/json",
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id="art_recipe",
            sha256="b" * 64,
            byte_size=13,
            media_type="application/json",
        ),
        output_roles=("batch", "manifest"),
    )


def seed_artifacts(database: Database) -> None:
    with database.connect() as connection:
        for artifact_id, digest, size, media in (
            ("art_source", SHA, 12, "application/json"),
            ("art_recipe", "b" * 64, 13, "application/json"),
            ("art_batch", "d" * 64, 4, "application/octet-stream"),
            ("art_manifest", "e" * 64, 5, "application/json"),
        ):
            connection.execute(
                "INSERT INTO artifacts(id,job_id,sha256,byte_size,media_type,"
                "relative_path,created_at) "
                "VALUES (?,NULL,?,?,?,?,?)",
                (artifact_id, digest, size, media, digest, "2026-07-29T00:00:00+00:00"),
            )


def test_latest_dataset_migrations_are_append_only_and_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    database.migrate()
    with database.connect() as connection:
        versions = tuple(
            row[0]
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert SCHEMA_VERSION == 49
    assert versions == tuple(range(1, 50))
    assert {
        "dataset_compilation_operations",
        "dataset_compilation_inputs",
        "dataset_compilation_outputs",
        "dataset_compilation_events",
        "dataset_compilation_failures",
        "dataset_compilation_reproductions",
        "curated_validation_operations",
    } <= names


def test_physical_v47_upgrades_once_and_preserves_durable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    migrations = database_module._MIGRATIONS
    monkeypatch.setattr(
        database_module,
        "_MIGRATIONS",
        {version: migration for version, migration in migrations.items() if version <= 47},
    )
    database.migrate()
    with database.connect() as connection:
        before = {
            (row[0], row[1])
            for row in connection.execute(
                "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            )
        }
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 47
        connection.execute(
            "INSERT INTO artifacts(id,job_id,sha256,byte_size,media_type,relative_path,created_at) "
            "VALUES (?,NULL,?,?,?,?,?)",
            (
                "art_durable",
                "9" * 64,
                1,
                "application/octet-stream",
                "durable",
                "2026-07-29T00:00:00+00:00",
            ),
        )
    monkeypatch.setattr(database_module, "_MIGRATIONS", migrations)
    database.migrate()
    database.migrate()
    with database.connect() as connection:
        after = {
            (row[0], row[1])
            for row in connection.execute(
                "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            )
        }
        assert after - before == {
            ("table", "curated_validation_operations"),
            ("index", "curated_validation_request_digest"),
            ("trigger", "guard_curated_validation_update_v49"),
            ("trigger", "guard_curated_validation_delete_v49"),
            *(
                ("table", name)
                for name in (
                    "dataset_compilation_operations",
                    "dataset_compilation_inputs",
                    "dataset_compilation_outputs",
                    "dataset_compilation_events",
                    "dataset_compilation_failures",
                    "dataset_compilation_reproductions",
                )
            ),
            *(
                ("index", name)
                for name in (
                    "dataset_compilation_operation_status",
                    "dataset_compilation_input_artifact",
                    "dataset_compilation_output_artifact",
                )
            ),
            *(
                ("trigger", name)
                for name in (
                    "validate_dataset_compilation_input_insert_v48",
                    "validate_dataset_compilation_output_insert_v48",
                    "guard_dataset_compilation_operation_authority_v48",
                    "guard_dataset_compilation_input_update_v48",
                    "guard_dataset_compilation_input_delete_v48",
                    "guard_dataset_compilation_output_update_v48",
                    "guard_dataset_compilation_output_delete_v48",
                    "guard_dataset_compilation_event_update_v48",
                    "guard_dataset_compilation_event_delete_v48",
                    "guard_dataset_compilation_failure_update_v48",
                    "guard_dataset_compilation_failure_delete_v48",
                    "guard_dataset_compilation_reproduction_update_v48",
                    "guard_dataset_compilation_reproduction_delete_v48",
                )
            ),
        }
        assert (
            connection.execute("SELECT sha256 FROM artifacts WHERE id='art_durable'").fetchone()[0]
            == "9" * 64
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM schema_migrations WHERE version=48"
            ).fetchone()[0]
            == 1
        )
        assert connection.execute(
            "SELECT count(*) FROM schema_migrations WHERE version=49"
        ).fetchone()[0] == 1


def test_physical_v48_to_v49_is_exact_preserving_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "physical-v48.sqlite3")
    migrations = database_module._MIGRATIONS
    monkeypatch.setattr(
        database_module,
        "_MIGRATIONS",
        {version: migration for version, migration in migrations.items() if version <= 48},
    )
    database.migrate()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO artifacts(id,job_id,sha256,byte_size,media_type,relative_path,created_at) "
            "VALUES ('art_v48',NULL,?,1,'application/octet-stream','v48','now')",
            ("8" * 64,),
        )
        before = {
            (row[0], row[1])
            for row in connection.execute(
                "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            )
        }
    monkeypatch.setattr(database_module, "_MIGRATIONS", migrations)
    database.migrate()
    database.migrate()
    with database.connect() as connection:
        after = {
            (row[0], row[1])
            for row in connection.execute(
                "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            )
        }
        assert after - before == {
            ("table", "curated_validation_operations"),
            ("index", "curated_validation_request_digest"),
            ("trigger", "guard_curated_validation_update_v49"),
            ("trigger", "guard_curated_validation_delete_v49"),
        }
        assert tuple(
            connection.execute(
                "SELECT sha256,byte_size,media_type FROM artifacts WHERE id='art_v48'"
            ).fetchone()
        ) == ("8" * 64, 1, "application/octet-stream")
        assert connection.execute(
            "SELECT count(*) FROM schema_migrations WHERE version=49"
        ).fetchone()[0] == 1


def test_exact_request_replays_and_conflict_is_rejected(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    seed_artifacts(database)
    repository = DatasetCompilationRepository(database)
    first = repository.begin(request())
    assert repository.begin(request()).operation_id == first.operation_id
    with pytest.raises(CompilationConflict, match="digest conflict"):
        repository.begin(request(digest="c" * 64))


def test_claim_is_fenced_and_expired_attempt_is_reclaimed(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    seed_artifacts(database)
    repository = DatasetCompilationRepository(database)
    repository.begin(request())
    first = repository.claim("compile_fixture", claimant_id="worker_a", lease_seconds=60)
    assert first.claimed and first.attempt == 1 and first.claim_token
    assert not repository.claim("compile_fixture", claimant_id="worker_b", lease_seconds=60).claimed
    with database.connect() as connection:
        connection.execute(
            "UPDATE dataset_compilation_operations SET lease_expires_at=? WHERE operation_key=?",
            ("2000-01-01T00:00:00+00:00", "compile_fixture"),
        )
    second = repository.claim("compile_fixture", claimant_id="worker_b", lease_seconds=60)
    assert second.claimed and second.attempt == 2
    assert second.claim_token != first.claim_token


def test_outputs_and_events_are_immutable(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    seed_artifacts(database)
    repository = DatasetCompilationRepository(database)
    repository.begin(request())
    claim = repository.claim("compile_fixture", claimant_id="worker", lease_seconds=60)
    receipt = DatasetCompilationReceipt(
        receipt_id="dcr_fixture",
        operation_id=claim.operation_id,
        operation_key="compile_fixture",
        request_digest=claim.request_digest,
        compiler_id="test_compiler_v1",
        attempt=claim.attempt,
        source=request().source,
        recipe=request().recipe,
        outputs=(
            CompilationOutput(
                role="batch",
                artifact_id="art_batch",
                sha256="d" * 64,
                byte_size=4,
                media_type="application/octet-stream",
            ),
            CompilationOutput(
                role="manifest",
                artifact_id="art_manifest",
                sha256="e" * 64,
                byte_size=5,
                media_type="application/json",
            ),
        ),
        created_at="2026-07-29T00:00:00+00:00",
    )
    receipt_json = canonical_json(receipt.model_dump(mode="json"))
    repository.complete(
        claim,
        outputs={
            "batch": ("art_batch", "d" * 64, 4, "application/octet-stream"),
            "manifest": ("art_manifest", "e" * 64, 5, "application/json"),
        },
        receipt_digest=hashlib.sha256(receipt_json.encode()).hexdigest(),
        receipt_json=receipt_json,
    )
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE dataset_compilation_outputs SET sha256=?", (hashlib.sha256(b"x").hexdigest(),)
        )


@pytest.mark.parametrize("action", ["complete", "fail", "cancel"])
@pytest.mark.parametrize("expiry", ["2000-01-01T00:00:00+00:00", "now"])
def test_expired_or_boundary_lease_cannot_publish(tmp_path: Path, action: str, expiry: str) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    seed_artifacts(database)
    repository = DatasetCompilationRepository(database)
    repository.begin(request())
    claim = repository.claim("compile_fixture", claimant_id="worker", lease_seconds=60)
    with database.connect() as connection:
        value = (
            connection.execute("SELECT datetime('now')").fetchone()[0]
            if expiry == "now"
            else expiry
        )
        connection.execute(
            "UPDATE dataset_compilation_operations SET lease_expires_at=? WHERE id=?",
            (value, claim.operation_id),
        )
    with pytest.raises(CompilationConflict, match="claim mismatch"):
        if action == "fail":
            repository.fail(claim, code="failed", failure_digest="f" * 64)
        elif action == "cancel":
            repository.cancel(claim, evidence_digest="f" * 64)
        else:
            receipt = DatasetCompilationReceipt(
                receipt_id="dcr_expired",
                operation_id=claim.operation_id,
                operation_key="compile_fixture",
                request_digest=claim.request_digest,
                compiler_id="test_compiler_v1",
                attempt=claim.attempt,
                source=request().source,
                recipe=request().recipe,
                outputs=(
                    CompilationOutput(
                        role="batch",
                        artifact_id="art_batch",
                        sha256="d" * 64,
                        byte_size=4,
                        media_type="application/octet-stream",
                    ),
                    CompilationOutput(
                        role="manifest",
                        artifact_id="art_manifest",
                        sha256="e" * 64,
                        byte_size=5,
                        media_type="application/json",
                    ),
                ),
                created_at="2026-07-29T00:00:00+00:00",
            )
            payload = canonical_json(receipt.model_dump(mode="json"))
            repository.complete(
                claim,
                outputs={
                    "batch": ("art_batch", "d" * 64, 4, "application/octet-stream"),
                    "manifest": ("art_manifest", "e" * 64, 5, "application/json"),
                },
                receipt_digest=hashlib.sha256(payload.encode()).hexdigest(),
                receipt_json=payload,
            )
    assert repository.get("compile_fixture").status == "running"


def test_receipt_rejects_forged_bound_field_and_inspect_rechecks_rows(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    seed_artifacts(database)
    repository = DatasetCompilationRepository(database)
    repository.begin(request())
    claim = repository.claim("compile_fixture", claimant_id="worker", lease_seconds=60)
    forged = DatasetCompilationReceipt(
        receipt_id="dcr_forged",
        operation_id=claim.operation_id,
        operation_key="wrong_key",
        request_digest=claim.request_digest,
        compiler_id="test_compiler_v1",
        attempt=claim.attempt,
        source=request().source,
        recipe=request().recipe,
        outputs=(
            CompilationOutput(
                role="batch",
                artifact_id="art_batch",
                sha256="d" * 64,
                byte_size=4,
                media_type="application/octet-stream",
            ),
            CompilationOutput(
                role="manifest",
                artifact_id="art_manifest",
                sha256="e" * 64,
                byte_size=5,
                media_type="application/json",
            ),
        ),
        created_at="2026-07-29T00:00:00+00:00",
    )
    payload = canonical_json(forged.model_dump(mode="json"))
    with pytest.raises(CompilationConflict, match="receipt binding mismatch"):
        repository.complete(
            claim,
            outputs={
                "batch": ("art_batch", "d" * 64, 4, "application/octet-stream"),
                "manifest": ("art_manifest", "e" * 64, 5, "application/json"),
            },
            receipt_digest=hashlib.sha256(payload.encode()).hexdigest(),
            receipt_json=payload,
        )
