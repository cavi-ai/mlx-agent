from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal, cast

from pydantic import Field, field_serializer, field_validator

from video_lab.character_models import (
    CharacterAnnotation,
    CharacterPackDraft,
    CharacterReferenceInput,
    CuratorDisposition,
    DuplicateWarning,
    WarningResolution,
)
from video_lab.character_pack import CharacterPack, character_pack_id, evaluate_pack, freeze_pack
from video_lab.database import Database
from video_lab.domain import FrozenModel, canonical_json, utc_now


class CharacterNotFound(LookupError):
    pass


class CharacterConflict(RuntimeError):
    pass


class CharacterRecord(FrozenModel):
    character_id: str
    name: str
    reviewer_id: str
    created_at: datetime


class CharacterPackRecord(FrozenModel):
    pack_id: str
    character_id: str
    reviewer_id: str
    state: Literal["draft", "preregistered"]
    supersedes_pack_id: str | None
    created_at: datetime
    preregistered_at: datetime | None = None


class CuratedReference(FrozenModel):
    reference: CharacterReferenceInput
    artifact_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)

    @field_validator("source_name")
    @classmethod
    def validate_source_name(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("source_name must be a normalized relative path")
        return value


class CharacterPackEvent(FrozenModel):
    pack_id: str
    sequence: int
    reviewer_id: str
    event_type: Literal["draft_created", "preregistered"]
    payload: Mapping[str, object]
    created_at: datetime

    @field_validator("payload", mode="after")
    @classmethod
    def freeze_payload(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return cast(Mapping[str, object], _deep_freeze(value))

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, object]) -> dict[str, object]:
        return cast(dict[str, object], _deep_thaw(value))


class PersistedCharacterPack(FrozenModel):
    record: CharacterPackRecord
    references: tuple[CuratedReference, ...]
    warnings: tuple[DuplicateWarning, ...]
    events: tuple[CharacterPackEvent, ...]
    frozen_pack: CharacterPack | None = None

    @property
    def draft(self) -> CharacterPackDraft:
        return CharacterPackDraft(
            character_id=self.record.character_id,
            references=tuple(curated.reference for curated in self.references),
            duplicate_warnings=self.warnings,
            supersedes_pack_id=self.record.supersedes_pack_id,
        )


class CharacterRepository:
    def __init__(self, database: Database, *, reviewer_id: str) -> None:
        reviewer = unicodedata.normalize("NFKC", reviewer_id).strip()
        if not reviewer or reviewer != reviewer_id:
            raise ValueError("reviewer_id must be normalized non-empty text")
        self.database = database
        self.reviewer_id = reviewer

    def create_character(self, name: str) -> CharacterRecord:
        normalized = unicodedata.normalize("NFKC", name).strip()
        if not normalized or normalized != name:
            raise ValueError("character name must be normalized non-empty text")
        character_id = _content_id(
            "character",
            {
                "schema_version": "character-identity.v1",
                "reviewer_id": self.reviewer_id,
                "name": name,
            },
        )
        created_at = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM characters WHERE id = ?", (character_id,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO characters(id, reviewer_id, name, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (character_id, self.reviewer_id, name, created_at.isoformat()),
                    )
                    record = CharacterRecord(
                        character_id=character_id,
                        name=name,
                        reviewer_id=self.reviewer_id,
                        created_at=created_at,
                    )
                else:
                    record = _character_from_row(row)
                    self._require_owner(record.reviewer_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return record

    def get_character(self, character_id: str) -> CharacterRecord:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM characters WHERE id = ?", (character_id,)
            ).fetchone()
        if row is None:
            raise CharacterNotFound("character not found")
        record = _character_from_row(row)
        self._require_owner(record.reviewer_id)
        return record

    def create_pack(
        self,
        pack: CharacterPackDraft,
        *,
        curated_references: tuple[CuratedReference, ...] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> CharacterPackRecord:
        if connection is not None:
            return self._create_pack_with_connection(connection, pack, curated_references)
        with self.database.connect() as owned_connection:
            owned_connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._create_pack_with_connection(
                    owned_connection, pack, curated_references
                )
                owned_connection.commit()
                return record
            except Exception:
                owned_connection.rollback()
                raise

    def _create_pack_with_connection(
        self,
        connection: sqlite3.Connection,
        pack: CharacterPackDraft,
        curated_references: tuple[CuratedReference, ...] | None,
    ) -> CharacterPackRecord:
        validated = CharacterPackDraft.model_validate(pack.model_dump(round_trip=True))
        character_row = connection.execute(
            "SELECT * FROM characters WHERE id=?", (validated.character_id,)
        ).fetchone()
        if character_row is None:
            raise CharacterNotFound("character not found")
        self._require_owner(str(character_row["reviewer_id"]))
        supersedes_storage_id: str | None = None
        if validated.supersedes_pack_id is not None:
            predecessor = self._load_pack_with_connection(connection, validated.supersedes_pack_id)
            if predecessor.record.character_id != validated.character_id:
                raise CharacterConflict("successor pack must belong to the same character")
            if predecessor.record.state != "preregistered":
                raise CharacterConflict("successor pack requires a preregistered predecessor")
            validated = validated.model_copy(
                update={"supersedes_pack_id": predecessor.record.pack_id}
            )
            supersedes_storage_id = self._pack_storage_id(connection, predecessor.record.pack_id)
        pack_id = character_pack_id(validated)
        curated = curated_references or ()
        if (
            validated.references
            and tuple(item.reference for item in curated) != validated.references
        ):
            raise CharacterConflict("curated references do not match character pack draft")
        coverage = evaluate_pack(validated)
        created_at = utc_now()
        try:
            row = self._find_pack_row(connection, pack_id)
            if row is None:
                connection.execute(
                    "INSERT INTO character_packs("
                    "storage_id, id, character_id, reviewer_id, state, "
                    "supersedes_storage_id, manifest_json, created_at, preregistered_at) "
                    "VALUES (?, ?, ?, ?, 'draft', ?, NULL, ?, NULL)",
                    (
                        pack_id,
                        pack_id,
                        validated.character_id,
                        self.reviewer_id,
                        supersedes_storage_id,
                        created_at.isoformat(),
                    ),
                )
                self._insert_references(connection, pack_id, curated)
                self._replace_warnings(connection, pack_id, coverage.duplicate_warnings)
                self._insert_event(
                    connection, pack_id, 1, "draft_created", {"pack_id": pack_id}, created_at
                )
                record = CharacterPackRecord(
                    pack_id=pack_id,
                    character_id=validated.character_id,
                    reviewer_id=self.reviewer_id,
                    state="draft",
                    supersedes_pack_id=validated.supersedes_pack_id,
                    created_at=created_at,
                )
            else:
                record = _pack_record_from_row(row)
                self._require_owner(record.reviewer_id)
                existing = self._load_pack_with_connection(connection, record.pack_id)
                if (
                    existing.draft.character_id != validated.character_id
                    or existing.draft.supersedes_pack_id != validated.supersedes_pack_id
                    or _immutable_reference_evidence(existing.references)
                    != _immutable_reference_evidence(curated)
                ):
                    raise CharacterConflict(
                        "character pack replay does not match persisted evidence"
                    )
        except sqlite3.IntegrityError as error:
            raise CharacterConflict(str(error)) from error
        return record

    def replace_draft_curation(self, pack_id: str, reference: CuratedReference) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pack = self._require_pack(connection, pack_id)
                current_pack_id = pack.pack_id
                pack_storage_id = self._pack_storage_id(connection, current_pack_id)
                if pack.state != "draft":
                    raise CharacterConflict("preregistered character packs are immutable")
                existing = connection.execute(
                    "SELECT stored.*, annotation.* FROM character_references AS stored "
                    "JOIN character_annotations AS annotation USING(pack_id, reference_id) "
                    "WHERE stored.pack_id=? AND stored.reference_id=?",
                    (pack_storage_id, reference.reference.reference_id),
                ).fetchone()
                if existing is None:
                    raise CharacterNotFound("character reference not found")
                persisted_reference = _curated_reference_from_row(existing)
                if _immutable_reference_evidence((persisted_reference,)) != (
                    _immutable_reference_evidence((reference,))
                ):
                    raise CharacterConflict("character curation cannot replace artifact evidence")
                item = reference.reference
                connection.execute(
                    "UPDATE character_references SET disposition=? "
                    "WHERE pack_id=? AND reference_id=?",
                    (item.disposition.value, pack_storage_id, item.reference_id),
                )
                annotation = item.annotation
                connection.execute(
                    "UPDATE character_annotations SET framing=?, head_angle=?, body_visibility=?, "
                    "expression=?, lighting=?, occlusion=?, identity_confirmed=? "
                    "WHERE pack_id=? AND reference_id=?",
                    (
                        annotation.framing.value,
                        annotation.head_angle.value,
                        annotation.body_visibility.value,
                        annotation.expression.value,
                        annotation.lighting.value,
                        annotation.occlusion.value,
                        int(annotation.identity_confirmed),
                        pack_storage_id,
                        item.reference_id,
                    ),
                )
                self._rekey_draft(connection, current_pack_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def resolve_warning(
        self,
        pack_id: str,
        left_reference_id: str,
        right_reference_id: str,
        resolution: WarningResolution,
    ) -> None:
        if resolution is WarningResolution.UNRESOLVED:
            raise ValueError("warning resolution must be final")
        left, right = sorted((left_reference_id, right_reference_id))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pack = self._require_pack(connection, pack_id)
                current_pack_id = pack.pack_id
                pack_storage_id = self._pack_storage_id(connection, current_pack_id)
                if pack.state != "draft":
                    raise CharacterConflict("preregistered character packs are immutable")
                row = connection.execute(
                    "SELECT resolution FROM character_pack_warnings WHERE pack_id=? "
                    "AND left_reference_id=? AND right_reference_id=?",
                    (pack_storage_id, left, right),
                ).fetchone()
                if row is None:
                    raise CharacterNotFound("character warning not found")
                current = WarningResolution(str(row["resolution"]))
                if current is resolution:
                    connection.commit()
                    return
                if current is not WarningResolution.UNRESOLVED:
                    raise CharacterConflict("resolved character warnings are immutable")
                connection.execute(
                    "UPDATE character_pack_warnings SET resolution=? WHERE pack_id=? "
                    "AND left_reference_id=? AND right_reference_id=?",
                    (resolution.value, pack_storage_id, left, right),
                )
                self._rekey_draft(connection, current_pack_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def preregister(self, frozen: CharacterPack) -> None:
        validated = CharacterPack.model_validate(frozen.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                persisted = self._load_pack_with_connection(connection, validated.pack_id)
                if persisted.record.state == "preregistered":
                    if persisted.frozen_pack != validated:
                        raise CharacterConflict("character pack preregistration replay mismatch")
                    connection.commit()
                    return
                if freeze_pack(persisted.draft) != validated:
                    raise CharacterConflict("frozen character pack does not match persisted draft")
                preregistered_at = utc_now()
                manifest_json = canonical_json(validated.model_dump(mode="json"))
                storage_id = self._pack_storage_id(connection, validated.pack_id)
                connection.execute(
                    "UPDATE character_packs SET state='preregistered', manifest_json=?, "
                    "preregistered_at=? WHERE id=?",
                    (manifest_json, preregistered_at.isoformat(), validated.pack_id),
                )
                self._insert_event(
                    connection,
                    storage_id,
                    2,
                    "preregistered",
                    {"manifest_sha256": hashlib.sha256(manifest_json.encode()).hexdigest()},
                    preregistered_at,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def load_pack(
        self, pack_id: str, *, connection: sqlite3.Connection | None = None
    ) -> PersistedCharacterPack:
        if connection is not None:
            return self._load_pack_with_connection(connection, pack_id)
        with self.database.connect() as owned_connection:
            return self._load_pack_with_connection(owned_connection, pack_id)

    def _load_pack_with_connection(
        self, connection: sqlite3.Connection, pack_id: str
    ) -> PersistedCharacterPack:
        record = self._require_pack(connection, pack_id)
        current_pack_id = record.pack_id
        pack_storage_id = self._pack_storage_id(connection, current_pack_id)
        rows = connection.execute(
            "SELECT reference.*, annotation.* FROM character_references AS reference "
            "JOIN character_annotations AS annotation USING(pack_id, reference_id) "
            "WHERE reference.pack_id=? ORDER BY reference.ordinal",
            (pack_storage_id,),
        ).fetchall()
        references = tuple(_curated_reference_from_row(row) for row in rows)
        warning_rows = connection.execute(
            "SELECT * FROM character_pack_warnings WHERE pack_id=? "
            "ORDER BY left_reference_id, right_reference_id",
            (pack_storage_id,),
        ).fetchall()
        warnings = tuple(_warning_from_row(row) for row in warning_rows)
        event_rows = connection.execute(
            "SELECT * FROM character_pack_events WHERE pack_id=? ORDER BY sequence",
            (pack_storage_id,),
        ).fetchall()
        events = tuple(_event_from_row(row, current_pack_id) for row in event_rows)
        manifest_row = connection.execute(
            "SELECT manifest_json FROM character_packs WHERE id=?", (current_pack_id,)
        ).fetchone()
        frozen = None
        if manifest_row is not None and manifest_row["manifest_json"] is not None:
            frozen = CharacterPack.model_validate_json(str(manifest_row["manifest_json"]))
        return PersistedCharacterPack(
            record=record,
            references=references,
            warnings=warnings,
            events=events,
            frozen_pack=frozen,
        )

    def _require_pack(self, connection: sqlite3.Connection, pack_id: str) -> CharacterPackRecord:
        row = self._find_pack_row(connection, pack_id)
        if row is None:
            raise CharacterNotFound("character pack not found")
        record = _pack_record_from_row(row)
        self._require_owner(record.reviewer_id)
        return record

    @staticmethod
    def _find_pack_row(connection: sqlite3.Connection, pack_id: str) -> sqlite3.Row | None:
        row = cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT pack.*, predecessor.id AS supersedes_pack_id "
                "FROM character_packs AS pack "
                "LEFT JOIN character_packs AS predecessor "
                "ON predecessor.storage_id=pack.supersedes_storage_id WHERE pack.id=?",
                (pack_id,),
            ).fetchone(),
        )
        if row is not None:
            return row
        return cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT pack.*, predecessor.id AS supersedes_pack_id "
                "FROM character_pack_aliases AS alias "
                "JOIN character_packs AS pack ON pack.storage_id=alias.pack_storage_id "
                "LEFT JOIN character_packs AS predecessor "
                "ON predecessor.storage_id=pack.supersedes_storage_id WHERE alias.alias_id=?",
                (pack_id,),
            ).fetchone(),
        )

    @staticmethod
    def _pack_storage_id(connection: sqlite3.Connection, pack_id: str) -> str:
        row = connection.execute(
            "SELECT storage_id FROM character_packs WHERE id=?", (pack_id,)
        ).fetchone()
        if row is None:
            raise CharacterNotFound("character pack not found")
        return str(row["storage_id"])

    def _require_owner(self, reviewer_id: str) -> None:
        if reviewer_id != self.reviewer_id:
            raise CharacterConflict("character evidence belongs to another reviewer")

    def _rekey_draft(self, connection: sqlite3.Connection, pack_id: str) -> str:
        persisted = self._load_pack_with_connection(connection, pack_id)
        new_pack_id = character_pack_id(persisted.draft)
        if new_pack_id != pack_id:
            storage_id = self._pack_storage_id(connection, pack_id)
            connection.execute(
                "UPDATE character_packs SET id=? WHERE storage_id=? AND state='draft'",
                (new_pack_id, storage_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO character_pack_aliases(alias_id, pack_storage_id) "
                "VALUES (?, ?)",
                (pack_id, storage_id),
            )
        return new_pack_id

    @staticmethod
    def _insert_references(
        connection: sqlite3.Connection,
        pack_id: str,
        references: tuple[CuratedReference, ...],
    ) -> None:
        for ordinal, curated in enumerate(references):
            reference = curated.reference
            connection.execute(
                "INSERT INTO character_references(pack_id, reference_id, artifact_id, source_name, "
                "image_sha256, media_type, dhash_hex, disposition, ordinal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pack_id,
                    reference.reference_id,
                    curated.artifact_id,
                    curated.source_name,
                    reference.image_sha256,
                    reference.media_type,
                    reference.dhash_hex,
                    reference.disposition.value,
                    ordinal,
                ),
            )
            annotation = reference.annotation
            connection.execute(
                "INSERT INTO character_annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pack_id,
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

    @staticmethod
    def _replace_warnings(
        connection: sqlite3.Connection,
        pack_id: str,
        warnings: tuple[DuplicateWarning, ...],
    ) -> None:
        for warning in warnings:
            connection.execute(
                "INSERT INTO character_pack_warnings VALUES (?, ?, ?, ?, ?)",
                (
                    pack_id,
                    warning.left_reference_id,
                    warning.right_reference_id,
                    warning.distance,
                    warning.resolution.value,
                ),
            )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        pack_id: str,
        sequence: int,
        event_type: Literal["draft_created", "preregistered"],
        payload: dict[str, object],
        created_at: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO character_pack_events VALUES (?, ?, ?, ?, ?, ?)",
            (
                pack_id,
                sequence,
                self.reviewer_id,
                event_type,
                canonical_json(payload),
                created_at.isoformat(),
            ),
        )


def _content_id(prefix: str, payload: dict[str, object]) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"


def _immutable_reference_evidence(
    references: tuple[CuratedReference, ...],
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    return tuple(
        (
            curated.reference.reference_id,
            curated.reference.image_sha256,
            curated.reference.media_type,
            curated.reference.dhash_hex,
            curated.artifact_id,
            curated.source_name,
        )
        for curated in references
    )


def _character_from_row(row: sqlite3.Row) -> CharacterRecord:
    return CharacterRecord(
        character_id=str(row["id"]),
        name=str(row["name"]),
        reviewer_id=str(row["reviewer_id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _pack_record_from_row(row: sqlite3.Row) -> CharacterPackRecord:
    return CharacterPackRecord(
        pack_id=str(row["id"]),
        character_id=str(row["character_id"]),
        reviewer_id=str(row["reviewer_id"]),
        state=cast(Literal["draft", "preregistered"], str(row["state"])),
        supersedes_pack_id=None
        if row["supersedes_pack_id"] is None
        else str(row["supersedes_pack_id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        preregistered_at=(
            None
            if row["preregistered_at"] is None
            else datetime.fromisoformat(str(row["preregistered_at"]))
        ),
    )


def _curated_reference_from_row(row: sqlite3.Row) -> CuratedReference:
    annotation = CharacterAnnotation.model_validate(
        {
            "framing": str(row["framing"]),
            "head_angle": str(row["head_angle"]),
            "body_visibility": str(row["body_visibility"]),
            "expression": str(row["expression"]),
            "lighting": str(row["lighting"]),
            "occlusion": str(row["occlusion"]),
            "identity_confirmed": bool(row["identity_confirmed"]),
        }
    )
    return CuratedReference(
        reference=CharacterReferenceInput.model_validate(
            {
                "reference_id": str(row["reference_id"]),
                "image_sha256": str(row["image_sha256"]),
                "media_type": str(row["media_type"]),
                "dhash_hex": str(row["dhash_hex"]),
                "disposition": CuratorDisposition(str(row["disposition"])),
                "annotation": annotation,
            }
        ),
        artifact_id=str(row["artifact_id"]),
        source_name=str(row["source_name"]),
    )


def _warning_from_row(row: sqlite3.Row) -> DuplicateWarning:
    return DuplicateWarning(
        left_reference_id=str(row["left_reference_id"]),
        right_reference_id=str(row["right_reference_id"]),
        distance=int(row["distance"]),
        resolution=WarningResolution(str(row["resolution"])),
    )


def _event_from_row(row: sqlite3.Row, public_pack_id: str) -> CharacterPackEvent:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise CharacterConflict("character event payload is invalid")
    return CharacterPackEvent(
        pack_id=public_pack_id,
        sequence=int(row["sequence"]),
        reviewer_id=str(row["reviewer_id"]),
        event_type=cast(Literal["draft_created", "preregistered"], str(row["event_type"])),
        payload=payload,
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("character event payload must contain JSON values")


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value
