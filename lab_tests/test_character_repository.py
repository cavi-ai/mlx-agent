from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from video_lab.character_models import (
    BodyVisibility,
    CharacterAnnotation,
    CharacterPackDraft,
    CharacterReferenceInput,
    CuratorDisposition,
    Expression,
    Framing,
    HeadAngle,
    Lighting,
    Occlusion,
    WarningResolution,
)
from video_lab.character_pack import freeze_pack
from video_lab.character_repository import (
    CharacterConflict,
    CharacterPackEvent,
    CharacterRepository,
    CuratedReference,
)
from video_lab.database import _MIGRATIONS, Database, _migration_statements


def _annotation(*, confirmed: bool = True) -> CharacterAnnotation:
    return CharacterAnnotation(
        framing=Framing.PORTRAIT,
        head_angle=HeadAngle.FRONT,
        body_visibility=BodyVisibility.CLEAR_FACE,
        expression=Expression.NEUTRAL,
        lighting=Lighting.ORDINARY,
        occlusion=Occlusion.NONE,
        identity_confirmed=confirmed,
    )


def _reference(
    index: int,
    *,
    annotation: CharacterAnnotation | None = None,
    dhash_hex: str | None = None,
) -> CuratedReference:
    reference_id = f"ref_{index:02d}"
    return CuratedReference(
        reference=CharacterReferenceInput(
            reference_id=reference_id,
            image_sha256=f"{index:064x}",
            media_type="image/png",
            dhash_hex=dhash_hex or f"{index * 0x9E3779B97F4A7C15 & ((1 << 64) - 1):016x}",
            disposition=CuratorDisposition.ACTIVE,
            annotation=annotation or _annotation(),
        ),
        artifact_id=f"art_{index:02d}",
        source_name=f"nested/{index:02d}.png",
    )


def _complete_references() -> tuple[CuratedReference, ...]:
    annotations = []
    for index in range(12):
        annotations.append(
            CharacterAnnotation(
                framing=Framing.PORTRAIT if index < 6 else Framing.FULL_BODY,
                head_angle=(HeadAngle.FRONT, HeadAngle.THREE_QUARTER, HeadAngle.PROFILE)[index % 3],
                body_visibility=(
                    BodyVisibility.CLEAR_FACE if index < 6 else BodyVisibility.FULL_BODY
                ),
                expression=(Expression.NEUTRAL, Expression.SMILE, Expression.SERIOUS)[index % 3],
                lighting=Lighting.ORDINARY if index % 2 == 0 else Lighting.DIRECTIONAL,
                identity_confirmed=True,
            )
        )
    return tuple(
        _reference(index + 1, annotation=annotation) for index, annotation in enumerate(annotations)
    )


def _seed_artifacts(database: Database, references: tuple[CuratedReference, ...]) -> None:
    with database.connect() as connection:
        for curated in references:
            connection.execute(
                "INSERT INTO artifacts("
                "id, sha256, byte_size, media_type, relative_path, created_at"
                ") "
                "VALUES (?, ?, 1, ?, ?, 'now')",
                (
                    curated.artifact_id,
                    curated.reference.image_sha256,
                    curated.reference.media_type,
                    f"aa/{curated.reference.image_sha256}",
                ),
            )


def test_repository_round_trips_pack_and_preregistration_is_replay_safe(
    database: Database,
) -> None:
    references = _complete_references()
    _seed_artifacts(database, references)
    repository = CharacterRepository(database, reviewer_id="reviewer-1")
    character = repository.create_character("Mara")
    draft = CharacterPackDraft(
        character_id=character.character_id,
        references=tuple(curated.reference for curated in references),
    )
    record = repository.create_pack(draft, curated_references=references)

    frozen = freeze_pack(draft)
    repository.preregister(frozen)
    repository.preregister(frozen)

    persisted = repository.load_pack(record.pack_id)
    assert persisted.record.state == "preregistered"
    assert persisted.frozen_pack == frozen
    assert [event.sequence for event in persisted.events] == [1, 2]


def test_database_rejects_preregistering_manifest_without_duplicate_provenance(
    database: Database,
) -> None:
    repository = CharacterRepository(database, reviewer_id="owner")
    character = repository.create_character("Mara")
    record = repository.create_pack(
        CharacterPackDraft(character_id=character.character_id, references=())
    )
    legacy_manifest = (
        '{"character_id":"'
        + character.character_id
        + '","duplicate_warnings":[],"pack_id":"'
        + record.pack_id
        + '","references":[],"supersedes_pack_id":null}'
    )

    with (
        database.connect() as connection,
        pytest.raises(sqlite3.IntegrityError, match="duplicate provenance"),
    ):
        connection.execute(
            "UPDATE character_packs SET state='preregistered', manifest_json=?, "
            "preregistered_at='2026-07-27T12:00:00+00:00' WHERE id=?",
            (legacy_manifest, record.pack_id),
        )


def test_repository_fails_closed_for_legacy_frozen_manifest_without_duplicate_provenance(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "legacy-pack.sqlite3")
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        for version in range(1, 20):
            for statement in _migration_statements(_MIGRATIONS[version]):
                if statement.strip() and not statement.lstrip().startswith(
                    "CREATE TABLE schema_migrations"
                ):
                    connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )
    repository = CharacterRepository(database, reviewer_id="owner")
    character = repository.create_character("Mara")
    record = repository.create_pack(
        CharacterPackDraft(character_id=character.character_id, references=())
    )
    legacy_manifest = (
        '{"character_id":"'
        + character.character_id
        + '","duplicate_warnings":[],"pack_id":"'
        + record.pack_id
        + '","references":[],"supersedes_pack_id":null}'
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE character_packs SET state='preregistered', manifest_json=?, "
            "preregistered_at='2026-07-27T12:00:00+00:00' WHERE id=?",
            (legacy_manifest, record.pack_id),
        )

    database.migrate()

    with pytest.raises(ValidationError, match="duplicate_hash_algorithm_version"):
        repository.load_pack(record.pack_id)


def test_repository_enforces_reviewer_ownership(database: Database) -> None:
    owner = CharacterRepository(database, reviewer_id="owner")
    character = owner.create_character("Mara")
    other = CharacterRepository(database, reviewer_id="other")

    with pytest.raises(CharacterConflict, match="another reviewer"):
        other.get_character(character.character_id)


def test_successor_pack_requires_a_frozen_predecessor(database: Database) -> None:
    repository = CharacterRepository(database, reviewer_id="owner")
    first = repository.create_character("First")
    predecessor = repository.create_pack(
        CharacterPackDraft(character_id=first.character_id, references=())
    )

    with pytest.raises(CharacterConflict, match="preregistered"):
        repository.create_pack(
            CharacterPackDraft(
                character_id=first.character_id,
                references=(),
                supersedes_pack_id=predecessor.pack_id,
            )
        )


def test_successor_resolves_old_alias_to_frozen_predecessor_and_cannot_break_link(
    database: Database,
) -> None:
    references = _complete_references()
    _seed_artifacts(database, references)
    repository = CharacterRepository(database, reviewer_id="owner")
    character = repository.create_character("Mara")
    original = CharacterPackDraft(
        character_id=character.character_id,
        references=tuple(item.reference for item in references),
    )
    initial = repository.create_pack(original, curated_references=references)
    replacement = references[0].model_copy(
        update={
            "reference": references[0].reference.model_copy(
                update={
                    "annotation": references[0].reference.annotation.model_copy(
                        update={"expression": Expression.ANGRY}
                    )
                }
            )
        }
    )
    repository.replace_draft_curation(initial.pack_id, replacement)
    current = repository.load_pack(initial.pack_id)
    repository.preregister(freeze_pack(current.draft))

    successor = repository.create_pack(
        CharacterPackDraft(
            character_id=character.character_id,
            references=(),
            supersedes_pack_id=initial.pack_id,
        )
    )

    persisted = repository.load_pack(successor.pack_id)
    assert persisted.record.supersedes_pack_id == current.record.pack_id
    with pytest.raises(CharacterConflict, match="immutable"):
        repository.replace_draft_curation(initial.pack_id, replacement)
    assert (
        repository.load_pack(successor.pack_id).record.supersedes_pack_id == current.record.pack_id
    )


def test_preregistered_pack_evidence_is_append_only(database: Database) -> None:
    references = _complete_references()
    _seed_artifacts(database, references)
    repository = CharacterRepository(database, reviewer_id="owner")
    character = repository.create_character("Mara")
    draft = CharacterPackDraft(
        character_id=character.character_id,
        references=tuple(curated.reference for curated in references),
    )
    record = repository.create_pack(draft, curated_references=references)
    repository.preregister(freeze_pack(draft))

    statements = (
        ("UPDATE character_packs SET manifest_json='{}' WHERE id=?", (record.pack_id,)),
        (
            "UPDATE character_references SET disposition='rejected' "
            "WHERE pack_id=? AND reference_id=?",
            (record.pack_id, references[0].reference.reference_id),
        ),
        (
            "DELETE FROM character_annotations WHERE pack_id=? AND reference_id=?",
            (record.pack_id, references[0].reference.reference_id),
        ),
    )
    for statement, parameters in statements:
        with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
            connection.execute(statement, parameters)

    with database.connect() as connection:
        for statement, parameters in (
            (
                "INSERT INTO character_references VALUES (?, 'late', ?, 'late.png', ?, "
                "'image/png', 'ffffffffffffffff', 'active', 99)",
                (
                    record.pack_id,
                    references[0].artifact_id,
                    references[0].reference.image_sha256,
                ),
            ),
            (
                "INSERT INTO character_pack_events VALUES "
                "(?, 3, 'owner', 'preregistered', '{}', 'now')",
                (record.pack_id,),
            ),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement, parameters)


def test_resolved_warning_cannot_be_changed_or_deleted(database: Database) -> None:
    repository = CharacterRepository(database, reviewer_id="owner")
    character = repository.create_character("Mara")
    left = _reference(1, dhash_hex="0000000000000000")
    right = _reference(2, dhash_hex="0000000000000001")
    _seed_artifacts(database, (left, right))
    record = repository.create_pack(
        CharacterPackDraft(
            character_id=character.character_id, references=(left.reference, right.reference)
        ),
        curated_references=(left, right),
    )
    repository.resolve_warning(
        record.pack_id,
        left.reference.reference_id,
        right.reference.reference_id,
        WarningResolution.ACCEPTED,
    )
    current_pack_id = repository.load_pack(record.pack_id).record.pack_id
    with database.connect() as connection:
        current_storage_id = str(
            connection.execute(
                "SELECT storage_id FROM character_packs WHERE id=?", (current_pack_id,)
            ).fetchone()["storage_id"]
        )

    for statement in (
        "UPDATE character_pack_warnings SET resolution='dismissed' WHERE pack_id=?",
        "DELETE FROM character_pack_warnings WHERE pack_id=?",
    ):
        with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
            connection.execute(statement, (current_storage_id,))


def test_frozen_events_and_resolved_warnings_cannot_move_between_packs_or_reviewers(
    database: Database,
) -> None:
    owner = CharacterRepository(database, reviewer_id="owner")
    character = owner.create_character("Mara")
    left = _reference(1, dhash_hex="0000000000000000")
    right = _reference(2, dhash_hex="0000000000000001")
    _seed_artifacts(database, (left, right))
    warning_pack = owner.create_pack(
        CharacterPackDraft(
            character_id=character.character_id,
            references=(left.reference, right.reference),
        ),
        curated_references=(left, right),
    )
    owner.resolve_warning(
        warning_pack.pack_id,
        left.reference.reference_id,
        right.reference.reference_id,
        WarningResolution.ACCEPTED,
    )
    warning_current = owner.load_pack(warning_pack.pack_id).record.pack_id
    same_owner_target = owner.create_pack(
        CharacterPackDraft(character_id=character.character_id, references=())
    )
    other = CharacterRepository(database, reviewer_id="other")
    other_character = other.create_character("Other")
    other_target = other.create_pack(
        CharacterPackDraft(character_id=other_character.character_id, references=())
    )

    references = _complete_references()
    with database.connect() as connection:
        for curated in references[2:]:
            connection.execute(
                "INSERT INTO artifacts("
                "id, sha256, byte_size, media_type, relative_path, created_at"
                ") VALUES (?, ?, 1, ?, ?, 'now')",
                (
                    curated.artifact_id,
                    curated.reference.image_sha256,
                    curated.reference.media_type,
                    f"bb/{curated.reference.image_sha256}",
                ),
            )
    frozen_references = references
    frozen_draft = CharacterPackDraft(
        character_id=character.character_id,
        references=tuple(item.reference for item in frozen_references),
    )
    frozen_record = owner.create_pack(frozen_draft, curated_references=frozen_references)
    owner.preregister(freeze_pack(frozen_draft))

    with database.connect() as connection:
        storage = {
            str(row["id"]): str(row["storage_id"])
            for row in connection.execute(
                "SELECT storage_id, id FROM character_packs WHERE id IN (?, ?, ?, ?)",
                (
                    warning_current,
                    same_owner_target.pack_id,
                    other_target.pack_id,
                    frozen_record.pack_id,
                ),
            )
        }
        for target in (same_owner_target.pack_id, other_target.pack_id):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE character_pack_warnings SET pack_id=? WHERE pack_id=?",
                    (storage[target], storage[warning_current]),
                )
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE character_pack_events SET pack_id=? WHERE pack_id=? AND sequence=2",
                    (storage[target], storage[frozen_record.pack_id]),
                )


def test_character_pack_event_payload_is_deeply_immutable() -> None:
    event = CharacterPackEvent.model_validate(
        {
            "pack_id": "pack",
            "sequence": 1,
            "reviewer_id": "owner",
            "event_type": "draft_created",
            "payload": {"nested": {"items": [1, 2]}},
            "created_at": "2026-07-21T12:00:00+00:00",
        }
    )

    payload = cast(Any, event.payload)
    with pytest.raises(TypeError):
        payload["new"] = True
    with pytest.raises(TypeError):
        payload["nested"]["items"][0] = 9
    assert event.model_dump(mode="json")["payload"] == {"nested": {"items": [1, 2]}}
