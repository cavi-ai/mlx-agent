from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest

from video_lab.database import SCHEMA_VERSION, Database
from video_lab.feasibility_models import (
    FeasibilityRun,
    FeasibilityStage,
    StageEvidence,
    StageOutcome,
    SupportVerdict,
)
from video_lab.feasibility_repository import (
    ActiveStageAttemptError,
    FeasibilityRepository,
    FeasibilityRunNotFound,
    StageAttemptFinishedError,
    StageAttemptNotFound,
)

FINGERPRINT = "b" * 64


def _run(run_id: str = "feasibility_1") -> FeasibilityRun:
    now = datetime.now(UTC)
    return FeasibilityRun(
        id=run_id,
        adapter_id="diffusers.animegen-t2v.wan2.2-a14b.mps.v1",
        protocol_version="animegen-feasibility.v1",
        input_fingerprint=FINGERPRINT,
        configuration={
            "manifest_profiles": {
                "minimum": {"width": 128, "height": 80, "num_frames": 17},
                "standard": {"width": 512, "height": 288, "num_frames": 17},
            },
            "sources": ["animegen", "wan"],
        },
        verdict=SupportVerdict.PENDING,
        created_at=now,
        updated_at=now,
    )


def test_migration_is_idempotent_and_adds_checked_feasibility_tables(database: Database) -> None:
    database.migrate()
    with database.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == list(range(1, SCHEMA_VERSION + 1))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO feasibility_runs(id, adapter_id, protocol_version, "
                "input_fingerprint, configuration_json, verdict, created_at, updated_at) "
                "VALUES ('bad', 'adapter', 'v1', ?, 'not-json', 'pending', ?, ?)",
                (FINGERPRINT, datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat()),
            )


def test_migration_upgrades_existing_version_7_database_with_update_guard(
    database: Database,
) -> None:
    with database.connect() as connection:
        connection.execute("DROP TRIGGER guard_feasibility_stage_attempt_updates")
        connection.execute("DELETE FROM schema_migrations WHERE version = 8")

    database.migrate()

    with database.connect() as connection:
        trigger = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            ("guard_feasibility_stage_attempt_updates",),
        ).fetchone()
        assert trigger is not None
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 8"
            ).fetchone()[0]
            == 1
        )


def test_create_and_get_run_preserve_typed_immutable_content(database: Database) -> None:
    repository = FeasibilityRepository(database)
    original = _run()

    assert repository.create_run(original) == original
    assert repository.get_run(original.id) == original
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_run(original)
    with pytest.raises(FeasibilityRunNotFound):
        repository.get_run("missing")


def test_authorization_reads_only_persisted_supported_verdict_and_measured_profile(
    database: Database,
) -> None:
    repository = FeasibilityRepository(database)
    run = repository.create_run(_run())
    assert repository.latest_authorized_profile(run.adapter_id) is None

    with pytest.raises(ValueError, match="not proven"):
        repository.set_verdict(run.id, SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID)
    with pytest.raises(ValueError, match="not proven"):
        repository.set_verdict(run.id, SupportVerdict.STANDARD_LAB_QUALIFIED)
    assert repository.latest_authorized_profile(run.adapter_id) is None


def test_attempt_transitions_are_atomic_monotonic_and_finished_rows_are_immutable(
    database: Database,
) -> None:
    repository = FeasibilityRepository(database)
    run = repository.create_run(_run())

    active = repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES, FINGERPRINT)
    assert active.ordinal == 1
    assert active.outcome is None
    with pytest.raises(ActiveStageAttemptError):
        repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES, FINGERPRINT)

    evidence = StageEvidence(measurements={"elapsed_seconds": 1.5}, details={"revision": "abc"})
    finished = repository.finish_attempt(active.id, StageOutcome.PASSED, evidence)
    assert finished.outcome is StageOutcome.PASSED
    assert finished.evidence == evidence
    assert finished.finished_at is not None and finished.finished_at.tzinfo is UTC
    with pytest.raises(StageAttemptFinishedError):
        repository.finish_attempt(active.id, StageOutcome.INTERRUPTED, StageEvidence())

    second = repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES, FINGERPRINT)
    assert second.ordinal == 2
    assert [attempt.id for attempt in repository.list_attempts(run.id)] == [active.id, second.id]
    with pytest.raises(FeasibilityRunNotFound):
        repository.list_attempts("missing")
    with pytest.raises(StageAttemptNotFound):
        repository.finish_attempt("missing", StageOutcome.PASSED, StageEvidence())


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("outcome", "'interrupted'"),
        ("finished_at", "'2099-01-01T00:00:00+00:00'"),
        ("evidence_json", '\'{"details":{"revised":true},"measurements":{}}\''),
        ("diagnostic_code", "'revised_code'"),
        ("diagnostic_message", "'revised message'"),
    ],
)
def test_database_rejects_direct_revision_of_finished_attempts(
    database: Database, column: str, replacement: str
) -> None:
    repository = FeasibilityRepository(database)
    run = repository.create_run(_run())
    active = repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES)
    repository.finish_attempt(
        active.id,
        StageOutcome.PASSED,
        StageEvidence(details={"original": True}),
        diagnostic_code="original_code",
        diagnostic_message="original message",
    )

    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            f"UPDATE feasibility_stage_attempts SET {column} = {replacement} WHERE id = ?",
            (active.id,),
        )


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE feasibility_stage_attempts SET finished_at = "
        "'2099-01-01T00:00:00+00:00' WHERE id = ?",
        "UPDATE feasibility_stage_attempts SET outcome = 'passed', finished_at = "
        "'2099-01-01T00:00:00+00:00' WHERE id = ?",
        "UPDATE feasibility_stage_attempts SET evidence_json = "
        '\'{"details":{},"measurements":{}}\' WHERE id = ?',
        "UPDATE feasibility_stage_attempts SET diagnostic_code = 'partial' WHERE id = ?",
        "UPDATE feasibility_stage_attempts SET ordinal = 99 WHERE id = ?",
    ],
)
def test_database_rejects_direct_partial_or_identity_updates_of_active_attempts(
    database: Database, statement: str
) -> None:
    repository = FeasibilityRepository(database)
    run = repository.create_run(_run())
    active = repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES)

    with (
        database.connect() as connection,
        pytest.raises(sqlite3.IntegrityError, match="terminal transition"),
    ):
        connection.execute(statement, (active.id,))

    assert repository.list_attempts(run.id) == (active,)


def test_attempt_fingerprint_must_match_its_run(database: Database) -> None:
    repository = FeasibilityRepository(database)
    run = repository.create_run(_run())

    with pytest.raises(ValueError, match="fingerprint"):
        repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES, "c" * 64)
    assert repository.list_attempts(run.id) == ()


def test_recovery_interrupts_active_attempts_once_and_preserves_stable_order(
    database: Database,
) -> None:
    repository = FeasibilityRepository(database)
    run = repository.create_run(_run())
    first = repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES, FINGERPRINT)
    repository.finish_attempt(first.id, StageOutcome.PASSED, StageEvidence())
    active = repository.start_attempt(run.id, FeasibilityStage.INVENTORY_SOURCES, FINGERPRINT)

    recovered = repository.recover_interrupted(run.id)

    assert [attempt.id for attempt in recovered] == [active.id]
    assert recovered[0].outcome is StageOutcome.INTERRUPTED
    assert recovered[0].diagnostic_code == "process_interrupted"
    assert repository.recover_interrupted(run.id) == ()
    assert [attempt.id for attempt in repository.list_attempts(run.id)] == [first.id, active.id]


def test_concurrent_start_allows_exactly_one_active_writer(database: Database) -> None:
    repository = FeasibilityRepository(database)
    run = repository.create_run(_run())
    barrier = Barrier(2)

    def start() -> str:
        concurrent_repository = FeasibilityRepository(Database(database.path))
        barrier.wait()
        try:
            return concurrent_repository.start_attempt(
                run.id, FeasibilityStage.RESOLVE_SOURCES, FINGERPRINT
            ).id
        except ActiveStageAttemptError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start(), range(2)))

    assert results.count("blocked") == 1
    assert len(repository.list_attempts(run.id)) == 1
