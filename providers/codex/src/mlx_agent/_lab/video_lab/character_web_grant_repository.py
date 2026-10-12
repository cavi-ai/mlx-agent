from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from video_lab.database import Database
from video_lab.domain import FrozenModel, utc_now

CHARACTER_CURATION_GRANT_TTL = timedelta(minutes=10)
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")


class CharacterCurationGrantError(PermissionError):
    pass


class CharacterCurationClaim(FrozenModel):
    reviewer_id: str
    character_id: str
    pack_storage_id: str


class CharacterCurationGrantRepository:
    def __init__(
        self,
        database: Database,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.database = database
        self.clock = clock

    def issue(self, reviewer_id: str, character_id: str, pack_id: str) -> str:
        reviewer = _normalized_reviewer(reviewer_id)
        character = _normalized_identity(character_id)
        pack = _normalized_identity(pack_id)
        now = self._now()
        expires_at = now + CHARACTER_CURATION_GRANT_TTL
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _delete_expired(connection, now)
                row = connection.execute(
                    "SELECT character_pack.storage_id FROM character_packs AS character_pack "
                    "LEFT JOIN character_pack_aliases AS alias "
                    "ON alias.pack_storage_id=character_pack.storage_id "
                    "WHERE character_pack.reviewer_id=? AND character_pack.character_id=? "
                    "AND (character_pack.id=? OR alias.alias_id=?) "
                    "ORDER BY character_pack.storage_id LIMIT 1",
                    (reviewer, character, pack, pack),
                ).fetchone()
                if row is None:
                    raise CharacterCurationGrantError("character curation evidence is unavailable")
                token = self._insert_random_token(
                    connection,
                    reviewer=reviewer,
                    character_id=character,
                    pack_storage_id=str(row["storage_id"]),
                    created_at=now,
                    expires_at=expires_at,
                )
                connection.commit()
                return token
            except Exception:
                connection.rollback()
                raise

    def consume(self, token: str) -> CharacterCurationClaim:
        token_digest = _token_digest(token)
        now = self._now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _delete_expired(connection, now)
                row = connection.execute(
                    "SELECT grant.reviewer_id, grant.character_id, grant.pack_storage_id "
                    "FROM character_curation_grants AS grant "
                    "JOIN characters AS character "
                    "ON character.id=grant.character_id "
                    "AND character.reviewer_id=grant.reviewer_id "
                    "JOIN character_packs AS character_pack "
                    "ON character_pack.storage_id=grant.pack_storage_id "
                    "AND character_pack.character_id=grant.character_id "
                    "AND character_pack.reviewer_id=grant.reviewer_id "
                    "WHERE grant.token_digest=?",
                    (token_digest,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    raise CharacterCurationGrantError("character curation grant is unavailable")
                deleted = connection.execute(
                    "DELETE FROM character_curation_grants WHERE token_digest=?",
                    (token_digest,),
                )
                if deleted.rowcount != 1:
                    raise CharacterCurationGrantError("character curation grant is unavailable")
                claim = CharacterCurationClaim(
                    reviewer_id=str(row["reviewer_id"]),
                    character_id=str(row["character_id"]),
                    pack_storage_id=str(row["pack_storage_id"]),
                )
                connection.commit()
                return claim
            except Exception:
                connection.rollback()
                raise

    def _insert_random_token(
        self,
        connection: sqlite3.Connection,
        *,
        reviewer: str,
        character_id: str,
        pack_storage_id: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> str:
        for _ in range(8):
            token = secrets.token_urlsafe(32)
            try:
                connection.execute(
                    "INSERT INTO character_curation_grants("
                    "token_digest, reviewer_id, character_id, pack_storage_id, "
                    "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        _token_digest(token),
                        reviewer,
                        character_id,
                        pack_storage_id,
                        created_at.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                continue
            return token
        raise RuntimeError("could not allocate a unique character curation token")

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("character curation clock must return an aware timestamp")
        return now.astimezone(UTC)


def _delete_expired(connection: sqlite3.Connection, now: datetime) -> None:
    connection.execute(
        "DELETE FROM character_curation_grants WHERE julianday(expires_at) <= julianday(?)",
        (now.isoformat(),),
    )


def _token_digest(token: str) -> str:
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise CharacterCurationGrantError("character curation grant is unavailable")
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _normalized_reviewer(reviewer_id: str) -> str:
    reviewer = unicodedata.normalize("NFKC", reviewer_id).strip()
    if not reviewer or reviewer != reviewer_id:
        raise ValueError("character curation reviewer must be a normalized identity")
    return reviewer


def _normalized_identity(value: str) -> str:
    if not value or value != value.strip():
        raise CharacterCurationGrantError("character curation evidence is unavailable")
    return value
