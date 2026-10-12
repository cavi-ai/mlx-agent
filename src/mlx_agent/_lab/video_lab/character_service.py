from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from PIL import Image, ImageOps, UnidentifiedImageError

from video_lab.artifacts import Artifact, ArtifactError, ArtifactStore, StagedArtifact
from video_lab.character_images import difference_hash
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
from video_lab.character_pack import (
    CharacterCoverage,
    CharacterPack,
    character_pack_id,
    evaluate_pack,
    freeze_pack,
)
from video_lab.character_repository import (
    CharacterRecord,
    CharacterRepository,
    CuratedReference,
    PersistedCharacterPack,
)
from video_lab.domain import FrozenModel

_SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_READ_CHUNK_SIZE = 1024 * 1024


class CharacterImportError(ValueError):
    pass


class ImportReportEntry(FrozenModel):
    source_name: str
    reason: Literal["unsupported_extension", "exact_duplicate"]


@dataclass(frozen=True)
class CharacterImageDiscovery:
    candidates: tuple[Path, ...]
    ignored: tuple[ImportReportEntry, ...]

    def __iter__(self) -> Iterator[Path]:
        return iter(self.candidates)


class CharacterReferenceView(FrozenModel):
    reference_id: str
    artifact_id: str
    source_name: str
    image_sha256: str
    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    dhash_hex: str
    disposition: CuratorDisposition
    annotation: CharacterAnnotation


class CharacterPackView(FrozenModel):
    pack_id: str
    character_id: str
    state: Literal["draft", "preregistered"]
    supersedes_pack_id: str | None
    references: tuple[CharacterReferenceView, ...]
    coverage: CharacterCoverage
    ignored: tuple[ImportReportEntry, ...] = ()


class CharacterPackExport(FrozenModel):
    pack: CharacterPack
    references: tuple[CharacterReferenceView, ...]


@dataclass(frozen=True)
class _DecodedImport:
    source_name: str
    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    sha256: str
    byte_size: int
    dhash_hex: str
    staged: StagedArtifact


def discover_character_images(source: Path) -> CharacterImageDiscovery:
    try:
        root_stat = source.lstat()
    except OSError as error:
        raise CharacterImportError("character source root is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise CharacterImportError("character source root must not be a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise CharacterImportError("character source root must be a directory")
    candidates: list[Path] = []
    ignored: list[ImportReportEntry] = []

    def traversal_error(error: OSError) -> None:
        raise CharacterImportError("character source cannot be traversed") from error

    try:
        for directory, directory_names, file_names in os.walk(
            source, onerror=traversal_error, followlinks=False
        ):
            current = Path(directory)
            for name in tuple(directory_names):
                path = current / name
                if path.is_symlink():
                    raise CharacterImportError("character source contains a symlink")
            directory_names.sort(key=_lexical_name)
            for name in sorted(file_names, key=_lexical_name):
                path = current / name
                if path.is_symlink():
                    raise CharacterImportError("character source contains a symlink")
                relative = path.relative_to(source)
                source_name = _normalized_relative_name(relative)
                if path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    candidates.append(Path(source_name))
                else:
                    ignored.append(
                        ImportReportEntry(
                            source_name=source_name,
                            reason="unsupported_extension",
                        )
                    )
    except CharacterImportError:
        raise
    except (OSError, ValueError) as error:
        raise CharacterImportError("character source cannot be traversed") from error
    candidates.sort(key=lambda path: _lexical_name(path.as_posix()))
    ignored.sort(key=lambda entry: _lexical_name(entry.source_name))
    return CharacterImageDiscovery(candidates=tuple(candidates), ignored=tuple(ignored))


class CharacterService:
    def __init__(self, repository: CharacterRepository, artifacts: ArtifactStore) -> None:
        self.repository = repository
        self.artifacts = artifacts

    def create_character(self, name: str) -> CharacterRecord:
        return self.repository.create_character(name)

    def import_folder(self, character_id: str, source: Path) -> CharacterPackView:
        self.repository.get_character(character_id)
        root_descriptor = _open_root_descriptor(source)
        prepared: list[_DecodedImport] = []
        ignored: list[ImportReportEntry] = []
        observed_content: dict[tuple[str, str], str] = {}
        try:
            discovery = discover_character_images(source)
            ignored.extend(discovery.ignored)
            for candidate in discovery.candidates:
                decoded = self._prepare_stable(root_descriptor, candidate)
                content_key = (decoded.sha256, decoded.media_type)
                if content_key in observed_content:
                    self._discard_bounded(decoded.staged)
                    ignored.append(
                        ImportReportEntry(
                            source_name=decoded.source_name,
                            reason="exact_duplicate",
                        )
                    )
                    continue
                observed_content[content_key] = decoded.source_name
                prepared.append(decoded)
        except Exception:
            for decoded in prepared:
                self._discard_bounded(decoded.staged)
            raise
        finally:
            os.close(root_descriptor)

        def persist(
            connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]
        ) -> PersistedCharacterPack:
            curated = tuple(
                CuratedReference(
                    reference=CharacterReferenceInput(
                        reference_id=f"character_ref_{decoded.sha256}",
                        image_sha256=decoded.sha256,
                        media_type=decoded.media_type,
                        dhash_hex=decoded.dhash_hex,
                        disposition=CuratorDisposition.ACTIVE,
                        annotation=_uncurated_annotation(),
                    ),
                    artifact_id=artifact.id,
                    source_name=decoded.source_name,
                )
                for decoded, artifact in zip(prepared, artifacts, strict=True)
            )
            draft = CharacterPackDraft(
                character_id=character_id,
                references=tuple(item.reference for item in curated),
            )
            record = self.repository.create_pack(
                draft,
                curated_references=curated,
                connection=connection,
            )
            return self.repository.load_pack(record.pack_id, connection=connection)

        try:
            persisted = self.artifacts.commit_batch(
                tuple(decoded.staged for decoded in prepared), persist
            )
        except (OSError, ArtifactError) as error:
            for decoded in prepared:
                self._discard_bounded(decoded.staged)
            source_name = prepared[0].source_name if prepared else "artifact"
            raise CharacterImportError(
                f"character artifact commit failed: {source_name}"
            ) from error
        except Exception as error:
            for decoded in prepared:
                self._discard_bounded(decoded.staged)
            raise CharacterImportError("character pack persistence failed") from error

        ignored.sort(key=lambda entry: (_lexical_name(entry.source_name), entry.reason))
        return self._view(persisted, ignored=tuple(ignored))

    def curate_reference(
        self,
        pack_id: str,
        reference_id: str,
        *,
        annotation: CharacterAnnotation,
        disposition: CuratorDisposition,
    ) -> CharacterPackView:
        persisted = self.repository.load_pack(pack_id)
        replacement: CuratedReference | None = None
        references: list[CharacterReferenceInput] = []
        for curated in persisted.references:
            if curated.reference.reference_id == reference_id:
                updated = curated.reference.model_copy(
                    update={"annotation": annotation, "disposition": disposition}
                )
                replacement = curated.model_copy(update={"reference": updated})
                references.append(updated)
            else:
                references.append(curated.reference)
        if replacement is None:
            raise LookupError("character reference not found")
        updated_draft = CharacterPackDraft(
            character_id=persisted.record.character_id,
            references=tuple(references),
            duplicate_warnings=persisted.warnings,
            supersedes_pack_id=persisted.record.supersedes_pack_id,
        )
        new_pack_id = character_pack_id(updated_draft)
        self.repository.replace_draft_curation(pack_id, replacement)
        return self._view(self.repository.load_pack(new_pack_id))

    def resolve_warning(
        self,
        pack_id: str,
        left_reference_id: str,
        right_reference_id: str,
        resolution: WarningResolution,
    ) -> CharacterPackView:
        persisted = self.repository.load_pack(pack_id)
        left, right = sorted((left_reference_id, right_reference_id))
        found = False
        warnings = []
        for warning in persisted.warnings:
            if (warning.left_reference_id, warning.right_reference_id) == (left, right):
                found = True
                warnings.append(warning.model_copy(update={"resolution": resolution}))
            else:
                warnings.append(warning)
        if not found:
            raise LookupError("character warning not found")
        updated_draft = persisted.draft.model_copy(update={"duplicate_warnings": tuple(warnings)})
        new_pack_id = character_pack_id(updated_draft)
        self.repository.resolve_warning(
            pack_id,
            left_reference_id,
            right_reference_id,
            resolution,
        )
        return self._view(self.repository.load_pack(new_pack_id))

    def pack_status(self, pack_id: str) -> CharacterPackView:
        return self._view(self.repository.load_pack(pack_id))

    def preregister_pack(self, pack_id: str) -> CharacterPackView:
        persisted = self.repository.load_pack(pack_id)
        frozen = freeze_pack(persisted.draft)
        if frozen.pack_id != persisted.record.pack_id:
            raise RuntimeError("persisted draft identity does not match frozen pack identity")
        self._verify_artifacts(persisted)
        self.repository.preregister(frozen)
        return self._view(self.repository.load_pack(frozen.pack_id))

    def export_pack(self, pack_id: str) -> CharacterPackExport:
        persisted = self.repository.load_pack(pack_id)
        if persisted.frozen_pack is None:
            raise PermissionError("character pack must be preregistered before export")
        self._verify_artifacts(persisted)
        return CharacterPackExport(
            pack=persisted.frozen_pack,
            references=tuple(_reference_view(item) for item in persisted.references),
        )

    def _prepare_stable(self, root_descriptor: int, candidate: Path) -> _DecodedImport:
        source_name = _normalized_relative_name(candidate)
        directory_descriptor = os.dup(root_descriptor)
        file_descriptor: int | None = None
        staged: StagedArtifact | None = None
        prepared_ok = False
        try:
            parts = PurePosixPath(source_name).parts
            for component in parts[:-1]:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
                opened_directory = os.fstat(next_descriptor)
                if not stat.S_ISDIR(opened_directory.st_mode):
                    os.close(next_descriptor)
                    raise OSError("source component is not a directory")
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            candidate_stat = os.stat(parts[-1], dir_fd=directory_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(candidate_stat.st_mode):
                raise OSError("source candidate is not a regular file")
            file_descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_descriptor,
            )
            opened = os.fstat(file_descriptor)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(candidate_stat):
                raise OSError("source candidate changed before open")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            byte_size = 0
            while True:
                chunk = os.read(file_descriptor, _READ_CHUNK_SIZE)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
                byte_size += len(chunk)
            after_read = os.fstat(file_descriptor)
            if _identity(opened) != _identity(after_read):
                raise CharacterImportError(f"character image changed during import: {source_name}")
            data = b"".join(chunks)
            try:
                media_type, dhash_hex = _decode_image_bytes(data)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                raise CharacterImportError(
                    f"character image cannot be decoded: {source_name}"
                ) from error
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            try:
                staged = self.artifacts.stage(_descriptor_path(file_descriptor), media_type)
            except (OSError, ArtifactError) as error:
                raise CharacterImportError(
                    f"character artifact staging failed: {source_name}"
                ) from error
            after_stage = os.fstat(file_descriptor)
            if (
                _identity(opened) != _identity(after_stage)
                or staged.sha256 != digest.hexdigest()
                or staged.byte_size != byte_size
            ):
                raise CharacterImportError(f"character image changed during import: {source_name}")
            prepared_ok = True
        except CharacterImportError:
            raise
        except OSError as error:
            raise CharacterImportError(
                f"character image cannot be opened: {source_name}"
            ) from error
        finally:
            os.close(directory_descriptor)
            if file_descriptor is not None:
                os.close(file_descriptor)
            if staged is not None and not prepared_ok:
                self._discard_bounded(staged)
        assert staged is not None
        return _DecodedImport(
            source_name=source_name,
            media_type=media_type,
            sha256=digest.hexdigest(),
            byte_size=byte_size,
            dhash_hex=dhash_hex,
            staged=staged,
        )

    def _discard_bounded(self, staged: StagedArtifact) -> None:
        try:
            self.artifacts.discard(staged)
        except (OSError, ArtifactError):
            return

    def _verify_artifacts(self, persisted: PersistedCharacterPack) -> None:
        for curated in persisted.references:
            path = self.artifacts.resolve(curated.artifact_id)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != curated.reference.image_sha256:
                raise RuntimeError("character artifact digest does not match frozen reference")

    @staticmethod
    def _view(
        persisted: PersistedCharacterPack,
        *,
        ignored: tuple[ImportReportEntry, ...] = (),
    ) -> CharacterPackView:
        return CharacterPackView(
            pack_id=persisted.record.pack_id,
            character_id=persisted.record.character_id,
            state=persisted.record.state,
            supersedes_pack_id=persisted.record.supersedes_pack_id,
            references=tuple(_reference_view(item) for item in persisted.references),
            coverage=evaluate_pack(persisted.draft),
            ignored=ignored,
        )


def _uncurated_annotation() -> CharacterAnnotation:
    return CharacterAnnotation(
        framing=Framing.PORTRAIT,
        head_angle=HeadAngle.FRONT,
        body_visibility=BodyVisibility.CLEAR_FACE,
        expression=Expression.NEUTRAL,
        lighting=Lighting.ORDINARY,
        occlusion=Occlusion.NONE,
        identity_confirmed=False,
    )


def _reference_view(curated: CuratedReference) -> CharacterReferenceView:
    reference = curated.reference
    return CharacterReferenceView(
        reference_id=reference.reference_id,
        artifact_id=curated.artifact_id,
        source_name=curated.source_name,
        image_sha256=reference.image_sha256,
        media_type=reference.media_type,
        dhash_hex=reference.dhash_hex,
        disposition=reference.disposition,
        annotation=reference.annotation,
    )


def _decode_image_bytes(
    data: bytes,
) -> tuple[Literal["image/jpeg", "image/png", "image/webp"], str]:
    with Image.open(BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        media_type = _media_type(opened.format)
        return media_type, f"{difference_hash(image):016x}"


def _media_type(image_format: str | None) -> Literal["image/jpeg", "image/png", "image/webp"]:
    mapping = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    if image_format is None or image_format not in mapping:
        raise ValueError("unsupported character image format")
    return cast(Literal["image/jpeg", "image/png", "image/webp"], mapping[image_format])


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _open_root_descriptor(source: Path) -> int:
    descriptor: int | None = None
    try:
        before = source.lstat()
    except OSError as error:
        raise CharacterImportError("character source root is unavailable") from error
    if stat.S_ISLNK(before.st_mode):
        raise CharacterImportError("character source root must not be a symlink")
    if not stat.S_ISDIR(before.st_mode):
        raise CharacterImportError("character source root must be a directory")
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(before):
            raise OSError("character source root changed")
        return descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise CharacterImportError("character source root cannot be opened") from error


def _descriptor_path(descriptor: int) -> Path:
    for root in (Path("/dev/fd"), Path("/proc/self/fd")):
        if root.is_dir():
            return root / str(descriptor)
    raise CharacterImportError("character descriptor staging is unavailable")


def _normalized_relative_name(path: Path) -> str:
    value = unicodedata.normalize("NFC", PurePosixPath(path).as_posix())
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or value in {"", "."}:
        raise CharacterImportError("character source path is not a safe relative name")
    return value


def _lexical_name(value: str) -> str:
    return unicodedata.normalize("NFC", value)
