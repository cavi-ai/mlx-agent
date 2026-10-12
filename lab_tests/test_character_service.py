from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from PIL import Image

import video_lab.character_service as character_service_module
from video_lab.artifacts import ArtifactIntegrityError, ArtifactStore
from video_lab.character_models import (
    BodyVisibility,
    CharacterAnnotation,
    CuratorDisposition,
    Expression,
    Framing,
    HeadAngle,
    Lighting,
    WarningResolution,
)
from video_lab.character_repository import CharacterConflict, CharacterRepository
from video_lab.character_service import (
    CharacterImportError,
    CharacterService,
    discover_character_images,
)
from video_lab.database import Database


def _service(database: Database, tmp_path: Path) -> CharacterService:
    repository = CharacterRepository(database, reviewer_id="reviewer")
    return CharacterService(repository, ArtifactStore(database, tmp_path / "artifacts"))


def _image(path: Path, image_format: str, *, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (24, 24))
    pixels = image.load()
    for y in range(24):
        for x in range(24):
            value = (x * 31 + y * 17 + seed * 47 + (x * y * (seed + 1))) % 256
            pixels[x, y] = (value, (value * 3 + seed) % 256, (value * 7 + x) % 256)
    image.save(path, format=image_format)


def test_discovery_is_nested_normalized_lexical_and_reports_unsupported(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _image(source / "z.webp", "WEBP")
    _image(source / "nested" / "B.PNG", "PNG")
    _image(source / "a.jpeg", "JPEG")
    (source / "nested" / "ignore.txt").write_text("ignored")

    discovery = discover_character_images(source)

    assert tuple(path.as_posix() for path in discovery.candidates) == (
        "a.jpeg",
        "nested/B.PNG",
        "z.webp",
    )
    assert [entry.model_dump(mode="json") for entry in discovery.ignored] == [
        {"source_name": "nested/ignore.txt", "reason": "unsupported_extension"}
    ]


def test_import_is_replay_safe_deduplicates_bytes_warns_and_leaks_no_absolute_path(
    database: Database, tmp_path: Path
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "private" / "references"
    _image(source / "a.png", "PNG", seed=1)
    (source / "duplicate.png").write_bytes((source / "a.png").read_bytes())
    with Image.open(source / "a.png") as same_pixels:
        same_pixels.save(source / "near.png", format="PNG", compress_level=0)
    (source / "notes.md").write_text("private")

    first = service.import_folder(character.character_id, source)
    second = service.import_folder(character.character_id, source)

    assert second == first
    assert len(first.references) == 2
    assert first.references[0].source_name == "a.png"
    assert len(first.coverage.duplicate_warnings) == 1
    assert first.ignored[-1].reason == "unsupported_extension"
    serialized = first.model_dump_json()
    assert str(source) not in serialized
    assert str(tmp_path) not in serialized


def test_import_replay_preserves_existing_human_curation(
    database: Database, tmp_path: Path
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    _image(source / "a.png", "PNG", seed=3)
    imported = service.import_folder(character.character_id, source)
    annotation = CharacterAnnotation(
        framing=Framing.FULL_BODY,
        head_angle=HeadAngle.PROFILE,
        body_visibility=BodyVisibility.FULL_BODY,
        expression=Expression.SERIOUS,
        lighting=Lighting.DIRECTIONAL,
        identity_confirmed=True,
    )
    curated = service.curate_reference(
        imported.pack_id,
        imported.references[0].reference_id,
        annotation=annotation,
        disposition=CuratorDisposition.ACTIVE,
    )

    replayed = service.import_folder(character.character_id, source)

    assert replayed.pack_id == curated.pack_id
    assert replayed.references[0].annotation == annotation


def test_import_rejects_symlinks_and_symlink_roots(database: Database, tmp_path: Path) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    outside = tmp_path / "outside.png"
    _image(outside, "PNG")
    source.mkdir()
    (source / "escape.png").symlink_to(outside)

    with pytest.raises(CharacterImportError, match="symlink"):
        service.import_folder(character.character_id, source)

    root_link = tmp_path / "root-link"
    root_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(CharacterImportError, match="source root"):
        service.import_folder(character.character_id, root_link)


def test_import_rejects_ancestor_swap_to_outside_symlink_at_component_open(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    nested = source / "nested"
    _image(nested / "inside.png", "PNG", seed=1)
    outside = tmp_path / "outside"
    _image(outside / "inside.png", "PNG", seed=99)
    original_open = character_service_module.os.open
    swapped = False

    def adversarial_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "nested" and dir_fd is not None and not swapped:
            swapped = True
            nested.rename(source / "original-nested")
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(character_service_module.os, "open", adversarial_open)

    with pytest.raises(
        CharacterImportError,
        match=r"^character image cannot be opened: nested/inside\.png$",
    ) as captured:
        service.import_folder(character.character_id, source)

    assert str(outside) not in str(captured.value)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


def test_import_rejects_fifo_candidate_promptly_without_opening_it(
    database: Database, tmp_path: Path
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    source.mkdir()
    fifo = source / "blocked.png"
    os.mkfifo(fifo)
    errors: list[BaseException] = []

    def import_fifo() -> None:
        try:
            service.import_folder(character.character_id, source)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=import_fifo, daemon=True)
    thread.start()
    thread.join(timeout=0.3)

    assert not thread.is_alive(), "supported FIFO candidate blocked during open"
    assert len(errors) == 1
    assert isinstance(errors[0], CharacterImportError)
    assert str(errors[0]) == "character image cannot be opened: blocked.png"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


def test_discovery_onerror_is_bounded(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "private" / "denied"

    def failing_walk(  # type: ignore[no-untyped-def]
        top, topdown=True, onerror=None, followlinks=False
    ):
        assert onerror is not None
        onerror(PermissionError(str(secret)))
        return iter(())

    monkeypatch.setattr(character_service_module.os, "walk", failing_walk)

    with pytest.raises(
        CharacterImportError,
        match="^character source cannot be traversed$",
    ) as captured:
        service.import_folder(character.character_id, source)
    assert str(secret) not in str(captured.value)


def test_missing_source_lstat_error_is_bounded(database: Database, tmp_path: Path) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    missing = tmp_path / "private" / "missing"

    with pytest.raises(
        CharacterImportError,
        match="^character source root is unavailable$",
    ) as captured:
        service.import_folder(character.character_id, missing)
    assert str(missing) not in str(captured.value)


def test_import_rejects_unreadable_image_without_persisting_pack(
    database: Database, tmp_path: Path
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    source.mkdir()
    (source / "bad.png").write_bytes(b"not an image")

    with pytest.raises(CharacterImportError, match="bad.png"):
        service.import_folder(character.character_id, source)

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM character_packs").fetchone()[0] == 0


def test_late_unreadable_image_leaves_no_artifact_rows_files_or_staging(
    database: Database, tmp_path: Path
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    _image(source / "a.png", "PNG", seed=1)
    (source / "z.png").write_bytes(b"not an image")

    with pytest.raises(
        CharacterImportError,
        match=r"^character image cannot be decoded: z\.png$",
    ):
        service.import_folder(character.character_id, source)

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert not service.artifacts.staging_root.exists() or not tuple(
        service.artifacts.staging_root.iterdir()
    )
    assert not tuple(
        path
        for path in service.artifacts.root.rglob("*")
        if path.is_file() and ".staging" not in path.parts and ".locks" not in path.parts
    )


def test_repository_failure_cleans_only_new_unreferenced_artifacts(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    shared_source = tmp_path / "shared.png"
    _image(shared_source, "PNG", seed=5)
    shared = service.artifacts.commit(
        service.artifacts.stage(shared_source, "image/png"), job_id=None
    )
    shared_path = service.artifacts.resolve(shared.id)
    source = tmp_path / "source"
    source.mkdir()
    (source / "a-shared.png").write_bytes(shared_source.read_bytes())
    _image(source / "b-new.png", "PNG", seed=6)

    def fail_create(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise CharacterConflict(str(source / "private"))

    monkeypatch.setattr(service.repository, "create_pack", fail_create)

    with pytest.raises(
        CharacterImportError,
        match="^character pack persistence failed$",
    ) as captured:
        service.import_folder(character.character_id, source)

    assert str(source) not in str(captured.value)
    with database.connect() as connection:
        rows = connection.execute("SELECT id FROM artifacts ORDER BY id").fetchall()
    assert [str(row["id"]) for row in rows] == [shared.id]
    assert shared_path.exists()
    assert service.artifacts.resolve(shared.id) == shared_path
    assert not service.artifacts.staging_root.exists() or not tuple(
        service.artifacts.staging_root.iterdir()
    )
    content_files = tuple(
        path
        for path in service.artifacts.root.rglob("*")
        if path.is_file() and ".staging" not in path.parts and ".locks" not in path.parts
    )
    assert content_files == (shared_path,)


def test_repository_failure_never_deletes_concurrently_reused_content(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    _image(source / "same.png", "PNG", seed=8)
    concurrent_source = tmp_path / "concurrent.png"
    concurrent_source.write_bytes((source / "same.png").read_bytes())
    registration_reached = threading.Event()
    allow_failure = threading.Event()
    concurrent_finished = threading.Event()
    concurrent_artifacts = []
    concurrent_errors: list[BaseException] = []
    import_errors: list[BaseException] = []

    def fail_create(*args, **kwargs):  # type: ignore[no-untyped-def]
        registration_reached.set()
        assert allow_failure.wait(timeout=3)
        raise CharacterConflict("forced repository failure")

    def import_content() -> None:
        try:
            service.import_folder(character.character_id, source)
        except BaseException as error:
            import_errors.append(error)

    def commit_same_content() -> None:
        try:
            assert registration_reached.wait(timeout=3)
            other = ArtifactStore(Database(database.path), service.artifacts.root)
            concurrent_artifacts.append(
                other.commit(other.stage(concurrent_source, "image/png"), job_id=None)
            )
        except BaseException as error:
            concurrent_errors.append(error)
        finally:
            concurrent_finished.set()

    monkeypatch.setattr(service.repository, "create_pack", fail_create)
    import_thread = threading.Thread(target=import_content)
    import_thread.start()
    assert registration_reached.wait(timeout=3)
    concurrent_thread = threading.Thread(target=commit_same_content)
    concurrent_thread.start()

    assert not concurrent_finished.wait(timeout=0.2)
    allow_failure.set()
    import_thread.join(timeout=3)
    concurrent_thread.join(timeout=3)

    assert not import_thread.is_alive()
    assert not concurrent_thread.is_alive()
    assert len(import_errors) == 1
    assert isinstance(import_errors[0], CharacterImportError)
    assert str(import_errors[0]) == "character pack persistence failed"
    assert concurrent_errors == []
    assert len(concurrent_artifacts) == 1
    concurrent = concurrent_artifacts[0]
    assert service.artifacts.resolve(concurrent.id).read_bytes() == concurrent_source.read_bytes()
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1


@pytest.mark.parametrize("operation", ("stage", "commit_batch"))
def test_artifact_failures_are_bounded_and_do_not_leak_source_path(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "private" / "source"
    _image(source / "a.png", "PNG", seed=1)

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        if operation == "stage":
            raise OSError(str(source))
        raise ArtifactIntegrityError(str(source))

    monkeypatch.setattr(service.artifacts, operation, fail)
    expected = (
        "character artifact staging failed: a.png"
        if operation == "stage"
        else "character artifact commit failed: a.png"
    )
    with pytest.raises(CharacterImportError, match=f"^{expected}$") as captured:
        service.import_folder(character.character_id, source)
    assert str(source) not in str(captured.value)


def test_import_detects_source_mutation_and_discards_staging(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    image_path = source / "race.png"
    _image(image_path, "PNG")
    original_stage = service.artifacts.stage

    def mutating_stage(path: Path, media_type: str):  # type: ignore[no-untyped-def]
        staged = original_stage(path, media_type)
        image_path.write_bytes(image_path.read_bytes() + b"changed")
        return staged

    monkeypatch.setattr(service.artifacts, "stage", mutating_stage)
    with pytest.raises(
        CharacterImportError,
        match=r"^character image changed during import: race\.png$",
    ) as captured:
        service.import_folder(character.character_id, source)

    assert str(source) not in str(captured.value)

    staging = service.artifacts.staging_root
    assert not staging.exists() or tuple(staging.iterdir()) == ()


def test_preregister_reverifies_actual_artifact_bytes(database: Database, tmp_path: Path) -> None:
    service = _service(database, tmp_path)
    character = service.create_character("Mara")
    source = tmp_path / "source"
    for index in range(12):
        _image(source / f"{index:02d}.png", "PNG", seed=index + 10)
    view = service.import_folder(character.character_id, source)
    for index, reference in enumerate(view.references):
        annotation = CharacterAnnotation(
            framing=Framing.PORTRAIT if index < 6 else Framing.FULL_BODY,
            head_angle=(HeadAngle.FRONT, HeadAngle.THREE_QUARTER, HeadAngle.PROFILE)[index % 3],
            body_visibility=(BodyVisibility.CLEAR_FACE if index < 6 else BodyVisibility.FULL_BODY),
            expression=(Expression.NEUTRAL, Expression.SMILE, Expression.SERIOUS)[index % 3],
            lighting=Lighting.ORDINARY if index % 2 == 0 else Lighting.DIRECTIONAL,
            identity_confirmed=True,
        )
        service.curate_reference(
            view.pack_id,
            reference.reference_id,
            annotation=annotation,
            disposition=CuratorDisposition.ACTIVE,
        )
    current = service.pack_status(view.pack_id)
    for warning in current.coverage.duplicate_warnings:
        service.resolve_warning(
            view.pack_id,
            warning.left_reference_id,
            warning.right_reference_id,
            WarningResolution.ACCEPTED,
        )

    artifact_path = service.artifacts.resolve(current.references[0].artifact_id)
    os.chmod(artifact_path, 0o600)
    artifact_path.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        service.preregister_pack(view.pack_id)
