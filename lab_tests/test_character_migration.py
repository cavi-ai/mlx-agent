from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_lab.artifacts import ArtifactStore
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
)
from video_lab.character_pack import character_pack_id, evaluate_pack
from video_lab.character_repository import CharacterConflict, CharacterRepository
from video_lab.character_service import CharacterService
from video_lab.database import _MIGRATIONS, SCHEMA_VERSION, Database, _migration_statements
from video_lab.domain import canonical_json


def _version_19_database(tmp_path: Path, name: str = "version-19.sqlite3") -> Database:
    database = Database(tmp_path / name)
    with database.connect() as connection:
        for version in range(1, 20):
            for statement in _migration_statements(_MIGRATIONS[version]):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
                (version,),
            )
    return database


def _legacy_pack_id(draft: CharacterPackDraft) -> str:
    coverage = evaluate_pack(draft)
    legacy_payload = {
        "character_id": draft.character_id,
        "references": [
            reference.model_dump(mode="json")
            for reference in sorted(draft.references, key=lambda item: item.reference_id)
        ],
        "duplicate_warnings": [
            warning.model_dump(mode="json") for warning in coverage.duplicate_warnings
        ],
        "supersedes_pack_id": draft.supersedes_pack_id,
    }
    return "character_pack_" + hashlib.sha256(canonical_json(legacy_payload).encode()).hexdigest()


def _version_19_ready_draft(
    tmp_path: Path,
) -> tuple[Database, Path, str, str]:
    database = _version_19_database(tmp_path)
    timestamp = "2026-07-27T12:00:00+00:00"

    artifact_root = tmp_path / "artifacts"
    references: list[CharacterReferenceInput] = []
    artifacts: list[tuple[str, str, bytes, str]] = []
    for index in range(12):
        content = f"version-19-character-reference-{index}".encode()
        digest = hashlib.sha256(content).hexdigest()
        relative_path = f"{digest[:2]}/{digest[2:4]}/{digest}"
        path = artifact_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        references.append(
            CharacterReferenceInput(
                reference_id=f"ref-{index:02d}",
                image_sha256=digest,
                media_type="image/png",
                dhash_hex=f"{(index + 1) * 0x9E3779B97F4A7C15 & ((1 << 64) - 1):016x}",
                disposition=CuratorDisposition.ACTIVE,
                annotation=CharacterAnnotation(
                    framing=Framing.PORTRAIT if index < 6 else Framing.FULL_BODY,
                    head_angle=(HeadAngle.FRONT, HeadAngle.THREE_QUARTER, HeadAngle.PROFILE)[
                        index % 3
                    ],
                    body_visibility=(
                        BodyVisibility.CLEAR_FACE if index < 6 else BodyVisibility.FULL_BODY
                    ),
                    expression=(Expression.NEUTRAL, Expression.SMILE, Expression.SERIOUS)[
                        index % 3
                    ],
                    lighting=(Lighting.ORDINARY if index % 2 == 0 else Lighting.DIRECTIONAL),
                    identity_confirmed=True,
                ),
            )
        )
        artifacts.append((f"artifact-{index:02d}", digest, content, relative_path))

    draft = CharacterPackDraft(character_id="character-1", references=tuple(references))
    legacy_id = _legacy_pack_id(draft)
    current_id = character_pack_id(draft)
    assert legacy_id != current_id

    with database.connect() as connection:
        connection.execute(
            "INSERT INTO characters(id, reviewer_id, name, created_at) "
            "VALUES ('character-1', 'reviewer-1', 'Mara', ?)",
            (timestamp,),
        )
        for artifact_id, digest, content, relative_path in artifacts:
            connection.execute(
                "INSERT INTO artifacts(id, sha256, byte_size, media_type, relative_path, "
                "created_at) VALUES (?, ?, ?, 'image/png', ?, ?)",
                (artifact_id, digest, len(content), relative_path, timestamp),
            )
        connection.execute(
            "INSERT INTO character_packs("
            "storage_id, id, character_id, reviewer_id, state, supersedes_storage_id, "
            "manifest_json, created_at, preregistered_at"
            ") VALUES (?, ?, 'character-1', 'reviewer-1', 'draft', NULL, NULL, ?, NULL)",
            (legacy_id, legacy_id, timestamp),
        )
        for ordinal, (reference, artifact) in enumerate(zip(references, artifacts, strict=True)):
            connection.execute(
                "INSERT INTO character_references("
                "pack_id, reference_id, artifact_id, source_name, image_sha256, media_type, "
                "dhash_hex, disposition, ordinal"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    legacy_id,
                    reference.reference_id,
                    artifact[0],
                    f"{reference.reference_id}.png",
                    reference.image_sha256,
                    reference.media_type,
                    reference.dhash_hex,
                    reference.disposition.value,
                    ordinal,
                ),
            )
            annotation = reference.annotation
            connection.execute(
                "INSERT INTO character_annotations("
                "pack_id, reference_id, framing, head_angle, body_visibility, expression, "
                "lighting, occlusion, identity_confirmed"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    legacy_id,
                    reference.reference_id,
                    annotation.framing.value,
                    annotation.head_angle.value,
                    annotation.body_visibility.value,
                    annotation.expression.value,
                    annotation.lighting.value,
                    annotation.occlusion.value,
                    int(annotation.identity_confirmed),
                ),
            )
        connection.execute(
            "INSERT INTO character_pack_events("
            "pack_id, sequence, reviewer_id, event_type, payload_json, created_at"
            ") VALUES (?, 1, 'reviewer-1', 'draft_created', ?, ?)",
            (legacy_id, canonical_json({"pack_id": legacy_id}), timestamp),
        )
    return database, artifact_root, legacy_id, current_id


def test_migration_21_rekeys_a_ready_version_19_draft_for_preregistration_and_export(
    tmp_path: Path,
) -> None:
    database, artifact_root, legacy_id, current_id = _version_19_ready_draft(tmp_path)

    database.migrate()
    Database(database.path).migrate()
    with database.connect() as connection:
        pack_row = connection.execute(
            "SELECT storage_id, id, reviewer_id, state FROM character_packs"
        ).fetchone()
        assert pack_row is not None
        assert tuple(pack_row) == (legacy_id, current_id, "reviewer-1", "draft")
        assert (
            connection.execute(
                "SELECT pack_storage_id FROM character_pack_aliases WHERE alias_id=?",
                (legacy_id,),
            ).fetchone()["pack_storage_id"]
            == legacy_id
        )
        assert connection.execute("SELECT COUNT(*) FROM character_references").fetchone()[0] == 12
        assert connection.execute("SELECT COUNT(*) FROM character_annotations").fetchone()[0] == 12
        assert connection.execute("SELECT COUNT(*) FROM character_pack_warnings").fetchone()[0] == 0
        event = connection.execute(
            "SELECT reviewer_id, event_type, payload_json FROM character_pack_events"
        ).fetchone()
        assert event is not None
        assert tuple(event) == (
            "reviewer-1",
            "draft_created",
            canonical_json({"pack_id": legacy_id}),
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert tuple(
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ) == tuple(range(1, SCHEMA_VERSION + 1))
    service = CharacterService(
        CharacterRepository(database, reviewer_id="reviewer-1"),
        ArtifactStore(database, artifact_root),
    )
    preregistered = service.preregister_pack(legacy_id)

    assert preregistered.pack_id == current_id
    assert service.pack_status(legacy_id).pack_id == current_id
    assert service.pack_status(current_id).pack_id == current_id
    assert service.export_pack(legacy_id) == service.export_pack(current_id)
    with pytest.raises(CharacterConflict, match="another reviewer"):
        CharacterRepository(database, reviewer_id="reviewer-2").load_pack(legacy_id)
    with pytest.raises(CharacterConflict, match="another reviewer"):
        CharacterRepository(database, reviewer_id="reviewer-2").load_pack(current_id)


def test_migration_21_rekeys_draft_successors_against_the_migrated_predecessor(
    tmp_path: Path,
) -> None:
    database = _version_19_database(tmp_path)
    predecessor_draft = CharacterPackDraft(character_id="character-1", references=())
    predecessor_legacy_id = _legacy_pack_id(predecessor_draft)
    successor_draft = CharacterPackDraft(
        character_id="character-1",
        references=(),
        supersedes_pack_id=predecessor_legacy_id,
    )
    successor_legacy_id = _legacy_pack_id(successor_draft)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO characters(id, reviewer_id, name, created_at) "
            "VALUES ('character-1', 'reviewer-1', 'Mara', '2026-07-27T12:00:00+00:00')"
        )
        connection.execute(
            "INSERT INTO character_packs VALUES "
            "(?, ?, 'character-1', 'reviewer-1', 'draft', NULL, NULL, "
            "'2026-07-27T12:00:00+00:00', NULL)",
            (predecessor_legacy_id, predecessor_legacy_id),
        )
        connection.execute(
            "INSERT INTO character_packs VALUES "
            "(?, ?, 'character-1', 'reviewer-1', 'draft', ?, NULL, "
            "'2026-07-27T12:00:00+00:00', NULL)",
            (successor_legacy_id, successor_legacy_id, predecessor_legacy_id),
        )
        for storage_id, pack_id in (
            (predecessor_legacy_id, predecessor_legacy_id),
            (successor_legacy_id, successor_legacy_id),
        ):
            connection.execute(
                "INSERT INTO character_pack_events VALUES "
                "(?, 1, 'reviewer-1', 'draft_created', ?, "
                "'2026-07-27T12:00:00+00:00')",
                (storage_id, canonical_json({"pack_id": pack_id})),
            )

    database.migrate()

    repository = CharacterRepository(database, reviewer_id="reviewer-1")
    predecessor = repository.load_pack(predecessor_legacy_id)
    successor = repository.load_pack(successor_legacy_id)
    assert successor.record.supersedes_pack_id == predecessor.record.pack_id
    assert successor.record.pack_id == character_pack_id(successor.draft)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT supersedes_storage_id FROM character_packs WHERE storage_id=?",
            (successor_legacy_id,),
        ).fetchone()
        assert row is not None
        assert row["supersedes_storage_id"] == predecessor_legacy_id
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("collision_kind", ["pack", "alias"])
def test_migration_21_fails_closed_on_cross_character_handle_collisions(
    tmp_path: Path,
    collision_kind: str,
) -> None:
    database, _, legacy_id, current_id = _version_19_ready_draft(tmp_path)
    other_storage_id = f"other-{collision_kind}-storage"
    other_id = current_id if collision_kind == "pack" else f"other-{collision_kind}-id"
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO characters(id, reviewer_id, name, created_at) "
            "VALUES ('character-2', 'reviewer-2', 'Iris', "
            "'2026-07-27T12:00:00+00:00')"
        )
        connection.execute(
            "INSERT INTO character_packs VALUES "
            "(?, ?, 'character-2', 'reviewer-2', 'draft', NULL, NULL, "
            "'2026-07-27T12:00:00+00:00', NULL)",
            (other_storage_id, other_id),
        )
        connection.execute(
            "INSERT INTO character_pack_events VALUES "
            "(?, 1, 'reviewer-2', 'draft_created', ?, "
            "'2026-07-27T12:00:00+00:00')",
            (other_storage_id, canonical_json({"pack_id": other_id})),
        )
        if collision_kind == "alias":
            connection.execute(
                "INSERT INTO character_pack_aliases(alias_id, pack_storage_id) VALUES (?, ?)",
                (legacy_id, other_storage_id),
            )

    expected = f"character pack identity migration {collision_kind} collision"
    with pytest.raises(RuntimeError, match=expected):
        database.migrate()
    with pytest.raises(RuntimeError, match=expected):
        Database(database.path).migrate()

    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT id FROM character_packs WHERE storage_id=?", (legacy_id,)
            ).fetchone()["id"]
            == legacy_id
        )
        assert tuple(
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ) == tuple(range(1, 20))
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
