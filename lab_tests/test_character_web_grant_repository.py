from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from video_lab.character_models import CharacterPackDraft
from video_lab.character_repository import CharacterRepository
from video_lab.character_web_grant_repository import (
    CHARACTER_CURATION_GRANT_TTL,
    CharacterCurationGrantError,
    CharacterCurationGrantRepository,
)

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _owned_pack(database):  # type: ignore[no-untyped-def]
    characters = CharacterRepository(database, reviewer_id="owner")
    character = characters.create_character("Mara")
    pack = characters.create_pack(
        CharacterPackDraft(character_id=character.character_id, references=())
    )
    with database.connect() as connection:
        storage_id = str(
            connection.execute(
                "SELECT storage_id FROM character_packs WHERE id=?", (pack.pack_id,)
            ).fetchone()[0]
        )
    return character.character_id, pack.pack_id, storage_id


def test_curation_grant_persists_only_digest_and_consumes_atomically(database) -> None:
    character_id, pack_id, storage_id = _owned_pack(database)
    repository = CharacterCurationGrantRepository(database, clock=lambda: NOW)

    token = repository.issue("owner", character_id, pack_id)

    with database.connect() as connection:
        row = connection.execute("SELECT * FROM character_curation_grants").fetchone()
    assert row is not None
    assert str(row["token_digest"]) == hashlib.sha256(token.encode("ascii")).hexdigest()
    assert token not in "|".join(str(value) for value in tuple(row))
    assert str(row["pack_storage_id"]) == storage_id

    claim = repository.consume(token)
    assert claim.reviewer_id == "owner"
    assert claim.character_id == character_id
    assert claim.pack_storage_id == storage_id
    with database.connect() as connection:
        remaining = connection.execute("SELECT COUNT(*) FROM character_curation_grants").fetchone()[
            0
        ]
    assert remaining == 0
    with pytest.raises(CharacterCurationGrantError):
        repository.consume(token)


def test_curation_grant_rejects_expiry_wrong_owner_character_and_pack(database) -> None:
    character_id, pack_id, _storage_id = _owned_pack(database)
    other = CharacterRepository(database, reviewer_id="other")
    other_character = other.create_character("Other")
    other_pack = other.create_pack(
        CharacterPackDraft(character_id=other_character.character_id, references=())
    )
    clock = [NOW]
    repository = CharacterCurationGrantRepository(database, clock=lambda: clock[0])

    for reviewer, character, pack in (
        ("other", character_id, pack_id),
        ("owner", other_character.character_id, other_pack.pack_id),
        ("owner", character_id, other_pack.pack_id),
    ):
        with pytest.raises(CharacterCurationGrantError):
            repository.issue(reviewer, character, pack)

    token = repository.issue("owner", character_id, pack_id)
    clock[0] += CHARACTER_CURATION_GRANT_TTL + timedelta(seconds=1)
    with pytest.raises(CharacterCurationGrantError):
        repository.consume(token)
    with database.connect() as connection:
        remaining = connection.execute("SELECT COUNT(*) FROM character_curation_grants").fetchone()[
            0
        ]
    assert remaining == 0
