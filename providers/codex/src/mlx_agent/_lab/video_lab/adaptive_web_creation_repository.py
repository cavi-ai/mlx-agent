from __future__ import annotations

import hashlib
import secrets
import sqlite3
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Self

from pydantic import Field, field_validator, model_validator

from video_lab.database import Database
from video_lab.domain import FrozenModel, _aware_utc, utc_now

ADAPTIVE_WEB_CREATION_GRANT_TTL = timedelta(minutes=15)
ADAPTIVE_WEB_CREATION_PER_OWNER_CAP = 8
ADAPTIVE_WEB_CREATION_TOTAL_CAP = 250


class AdaptiveWebCreationGrantError(PermissionError):
    pass


class AdaptiveWebCreationCapacityError(RuntimeError):
    pass


class AdaptiveWebCreationGrant(FrozenModel):
    token_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1)
    created_at: datetime
    expires_at: datetime
    input_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    study_id: str | None = None
    bound_at: datetime | None = None

    @field_validator("reviewer_id")
    @classmethod
    def normalized_reviewer(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("creation grant reviewer must be normalized")
        return value

    @field_validator("created_at", "expires_at", "bound_at")
    @classmethod
    def normalized_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value)

    @model_validator(mode="after")
    def valid_lifecycle(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("creation grant expiry must follow creation")
        binding = (self.input_fingerprint, self.study_id, self.bound_at)
        if any(value is None for value in binding) != all(value is None for value in binding):
            raise ValueError("creation grant binding fields must transition together")
        if self.bound_at is not None and not self.created_at <= self.bound_at < self.expires_at:
            raise ValueError("creation grant binding timestamp is outside its lifetime")
        return self


class AdaptiveWebCreationGrantRepository:
    def __init__(
        self,
        database: Database,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.database = database
        self.clock = clock

    def issue(self, reviewer_id: str) -> str:
        reviewer = _normalized_reviewer(reviewer_id)
        now = self._now()
        expires_at = now + ADAPTIVE_WEB_CREATION_GRANT_TTL
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _delete_expired(connection, now)
                total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM adaptive_web_creation_grants"
                    ).fetchone()[0]
                )
                owner_total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM adaptive_web_creation_grants WHERE reviewer_id = ?",
                        (reviewer,),
                    ).fetchone()[0]
                )
                if (
                    total >= ADAPTIVE_WEB_CREATION_TOTAL_CAP
                    or owner_total >= ADAPTIVE_WEB_CREATION_PER_OWNER_CAP
                ):
                    connection.commit()
                    raise AdaptiveWebCreationCapacityError(
                        "adaptive web creation grant capacity reached"
                    )
                token = self._insert_random_token(
                    connection,
                    reviewer=reviewer,
                    created_at=now,
                    expires_at=expires_at,
                )
                connection.commit()
                return token
            except Exception:
                connection.rollback()
                raise

    def bind(
        self,
        reviewer_id: str,
        token: str,
        input_fingerprint: str,
    ) -> AdaptiveWebCreationGrant:
        reviewer = _normalized_reviewer(reviewer_id)
        if (
            len(input_fingerprint) != 64
            or input_fingerprint != input_fingerprint.lower()
            or any(character not in "0123456789abcdef" for character in input_fingerprint)
        ):
            raise ValueError("adaptive creation input fingerprint is invalid")
        token_digest = _token_digest(token)
        now = self._now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _delete_expired(connection, now)
                row = connection.execute(
                    "SELECT * FROM adaptive_web_creation_grants "
                    "WHERE token_digest = ? AND reviewer_id = ?",
                    (token_digest, reviewer),
                ).fetchone()
                if row is None:
                    raise AdaptiveWebCreationGrantError(
                        "adaptive web creation grant is unavailable"
                    )
                current = _grant_from_row(row)
                if current.input_fingerprint is not None:
                    if current.input_fingerprint != input_fingerprint:
                        raise AdaptiveWebCreationGrantError(
                            "adaptive web creation grant is already bound"
                        )
                    connection.commit()
                    return current

                study_id = _study_id(reviewer, input_fingerprint, token)
                connection.execute(
                    "UPDATE adaptive_web_creation_grants "
                    "SET input_fingerprint = ?, study_id = ?, bound_at = ? "
                    "WHERE token_digest = ? AND reviewer_id = ? "
                    "AND input_fingerprint IS NULL AND study_id IS NULL AND bound_at IS NULL",
                    (
                        input_fingerprint,
                        study_id,
                        now.isoformat(),
                        token_digest,
                        reviewer,
                    ),
                )
                persisted_row = connection.execute(
                    "SELECT * FROM adaptive_web_creation_grants "
                    "WHERE token_digest = ? AND reviewer_id = ?",
                    (token_digest, reviewer),
                ).fetchone()
                if persisted_row is None:
                    raise RuntimeError("bound adaptive web creation grant disappeared")
                persisted = _grant_from_row(persisted_row)
                connection.commit()
                return persisted
            except Exception:
                connection.rollback()
                raise

    def _insert_random_token(
        self,
        connection: sqlite3.Connection,
        *,
        reviewer: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> str:
        for _ in range(8):
            token = secrets.token_urlsafe(32)
            try:
                connection.execute(
                    "INSERT INTO adaptive_web_creation_grants("
                    "token_digest, reviewer_id, created_at, expires_at"
                    ") VALUES (?, ?, ?, ?)",
                    (
                        _token_digest(token),
                        reviewer,
                        created_at.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                continue
            return token
        raise RuntimeError("could not allocate a unique adaptive web creation token")

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("adaptive web creation clock must return an aware timestamp")
        return now.astimezone(UTC)


def _delete_expired(connection: sqlite3.Connection, now: datetime) -> None:
    connection.execute(
        "DELETE FROM adaptive_web_creation_grants WHERE julianday(expires_at) <= julianday(?)",
        (now.isoformat(),),
    )


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalized_reviewer(reviewer_id: str) -> str:
    reviewer = unicodedata.normalize("NFKC", reviewer_id).strip()
    if not reviewer or reviewer != reviewer_id:
        raise ValueError("adaptive creation reviewer must be a normalized identity")
    return reviewer


def _study_id(reviewer: str, input_fingerprint: str, token: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"video-lab.adaptive-study-id.v2\x00")
    digest.update(reviewer.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(input_fingerprint.encode("ascii"))
    digest.update(b"\x00")
    digest.update(token.encode("utf-8"))
    return f"study_{digest.hexdigest()}"


def _grant_from_row(row: sqlite3.Row) -> AdaptiveWebCreationGrant:
    return AdaptiveWebCreationGrant(
        token_digest=str(row["token_digest"]),
        reviewer_id=str(row["reviewer_id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        expires_at=datetime.fromisoformat(str(row["expires_at"])),
        input_fingerprint=(
            None if row["input_fingerprint"] is None else str(row["input_fingerprint"])
        ),
        study_id=None if row["study_id"] is None else str(row["study_id"]),
        bound_at=None if row["bound_at"] is None else datetime.fromisoformat(str(row["bound_at"])),
    )
