from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from video_lab.adaptive_web_creation_repository import (
    ADAPTIVE_WEB_CREATION_GRANT_TTL,
    ADAPTIVE_WEB_CREATION_PER_OWNER_CAP,
    ADAPTIVE_WEB_CREATION_TOTAL_CAP,
    AdaptiveWebCreationCapacityError,
    AdaptiveWebCreationGrantError,
    AdaptiveWebCreationGrantRepository,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
FINGERPRINT = "a" * 64


def test_grant_persists_only_token_digest_and_binds_once_idempotently(database) -> None:
    clock = [NOW]
    repository = AdaptiveWebCreationGrantRepository(database, clock=lambda: clock[0])

    token = repository.issue("owner-a")

    with database.connect() as connection:
        row = connection.execute("SELECT * FROM adaptive_web_creation_grants").fetchone()
    assert row is not None
    assert str(row["token_digest"]) == hashlib.sha256(token.encode("ascii")).hexdigest()
    assert token not in "|".join("" if value is None else str(value) for value in tuple(row))

    bound = repository.bind("owner-a", token, FINGERPRINT)
    retry = AdaptiveWebCreationGrantRepository(database, clock=lambda: clock[0]).bind(
        "owner-a", token, FINGERPRINT
    )

    assert retry == bound
    assert bound.input_fingerprint == FINGERPRINT
    assert bound.study_id is not None and bound.study_id.startswith("study_")
    assert bound.bound_at == NOW
    with pytest.raises(AdaptiveWebCreationGrantError):
        repository.bind("owner-a", token, "b" * 64)
    with database.connect() as connection:
        with pytest.raises(sqlite3.DatabaseError, match="grant transition"):
            connection.execute(
                "UPDATE adaptive_web_creation_grants SET study_id = 'tampered' "
                "WHERE token_digest = ? AND reviewer_id = 'owner-a'",
                (bound.token_digest,),
            )


def test_grant_rejects_foreign_and_expired_tokens_and_cleans_expiry(database) -> None:
    clock = [NOW]
    repository = AdaptiveWebCreationGrantRepository(database, clock=lambda: clock[0])
    token = repository.issue("owner-a")

    with pytest.raises(AdaptiveWebCreationGrantError):
        repository.bind("owner-b", token, FINGERPRINT)

    clock[0] += ADAPTIVE_WEB_CREATION_GRANT_TTL + timedelta(seconds=1)
    with pytest.raises(AdaptiveWebCreationGrantError):
        repository.bind("owner-a", token, FINGERPRINT)
    replacement = repository.issue("owner-b")
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT token_digest, reviewer_id FROM adaptive_web_creation_grants"
        ).fetchall()
    assert [(str(row["token_digest"]), str(row["reviewer_id"])) for row in rows] == [
        (hashlib.sha256(replacement.encode("ascii")).hexdigest(), "owner-b")
    ]


def test_grant_issue_enforces_per_owner_live_capacity(database) -> None:
    repository = AdaptiveWebCreationGrantRepository(database, clock=lambda: NOW)
    for _ in range(ADAPTIVE_WEB_CREATION_PER_OWNER_CAP):
        repository.issue("owner-a")

    with pytest.raises(AdaptiveWebCreationCapacityError):
        repository.issue("owner-a")

    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM adaptive_web_creation_grants WHERE reviewer_id = 'owner-a'"
            ).fetchone()[0]
            == ADAPTIVE_WEB_CREATION_PER_OWNER_CAP
        )


def test_grant_issue_enforces_total_cap_then_globally_cleans_250_expired_owners(
    database,
) -> None:
    clock = [NOW]
    repository = AdaptiveWebCreationGrantRepository(database, clock=lambda: clock[0])
    for index in range(ADAPTIVE_WEB_CREATION_TOTAL_CAP):
        repository.issue(f"owner-{index}")

    with database.connect() as connection:
        connection.execute(
            "INSERT INTO adaptive_web_creation_grants("
            "token_digest, reviewer_id, created_at, expires_at"
            ") VALUES (?, 'expired-owner', ?, ?)",
            (
                "f" * 64,
                (NOW - timedelta(hours=2)).isoformat(),
                (NOW - timedelta(hours=1)).isoformat(),
            ),
        )

    with pytest.raises(AdaptiveWebCreationCapacityError):
        repository.issue("owner-over-cap")
    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM adaptive_web_creation_grants").fetchone()[0]
            == ADAPTIVE_WEB_CREATION_TOTAL_CAP
        )

    clock[0] += ADAPTIVE_WEB_CREATION_GRANT_TTL + timedelta(seconds=1)
    replacement = repository.issue("owner-after-cleanup")
    assert replacement
    with database.connect() as connection:
        rows = connection.execute("SELECT reviewer_id FROM adaptive_web_creation_grants").fetchall()
    assert [str(row["reviewer_id"]) for row in rows] == ["owner-after-cleanup"]
