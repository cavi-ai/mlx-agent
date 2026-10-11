from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner, Result

from video_lab.artifacts import ArtifactStore
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
from video_lab.character_repository import CharacterRepository
from video_lab.character_service import CharacterService
from video_lab.cli import app
from video_lab.database import Database

runner = CliRunner()


def _common(tmp_path: Path, reviewer: str = "owner") -> list[str]:
    return [
        "--database",
        str(tmp_path / "lab.sqlite3"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--reviewer-id",
        reviewer,
    ]


def _invoke(tmp_path: Path, *arguments: str, reviewer: str = "owner") -> Result:
    return runner.invoke(app, [*_common(tmp_path, reviewer), "character", *arguments])


def _payload(result: Result) -> dict[str, object]:
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    value = json.loads(result.stdout)
    assert result.stdout == json.dumps(value, separators=(",", ":")) + "\n"
    return value


def _image(path: Path, seed: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (24, 24))
    pixels = image.load()
    for y in range(24):
        for x in range(24):
            value = (x * 31 + y * 17 + seed * 47 + x * y * (seed + 1)) % 256
            pixels[x, y] = (value, (value * 3 + seed) % 256, (value * 7 + x) % 256)
    image.save(path, format="PNG")


def _create_and_import(tmp_path: Path) -> tuple[str, dict[str, object]]:
    created = _invoke(tmp_path, "create", "Mara")
    assert created.exit_code == 0
    character = _payload(created)["data"]
    assert isinstance(character, dict)
    character_id = character["character_id"]
    assert isinstance(character_id, str)
    source = tmp_path / "private" / "source"
    _image(source / "portrait.png")
    imported = _invoke(tmp_path, "import", character_id, "--source", str(source))
    assert imported.exit_code == 0
    data = _payload(imported)["data"]
    assert isinstance(data, dict)
    return character_id, data


def _ready_pack(tmp_path: Path) -> tuple[str, str]:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    service = CharacterService(
        CharacterRepository(database, reviewer_id="owner"),
        ArtifactStore(database, tmp_path / "artifacts"),
    )
    character = service.create_character("Ready Mara")
    source = tmp_path / "ready-source"
    for index in range(12):
        _image(source / f"{index:02d}.png", seed=index + 1)
    imported = service.import_folder(character.character_id, source)
    for index, reference in enumerate(imported.references):
        service.curate_reference(
            imported.pack_id,
            reference.reference_id,
            annotation=CharacterAnnotation(
                framing=Framing.PORTRAIT if index < 6 else Framing.FULL_BODY,
                head_angle=(HeadAngle.FRONT, HeadAngle.THREE_QUARTER, HeadAngle.PROFILE)[index % 3],
                body_visibility=(
                    BodyVisibility.CLEAR_FACE if index < 6 else BodyVisibility.FULL_BODY
                ),
                expression=(Expression.NEUTRAL, Expression.SMILE, Expression.SERIOUS)[index % 3],
                lighting=Lighting.ORDINARY if index % 2 == 0 else Lighting.DIRECTIONAL,
                identity_confirmed=True,
            ),
            disposition=CuratorDisposition.ACTIVE,
        )
    current = service.pack_status(imported.pack_id)
    for warning in current.coverage.duplicate_warnings:
        if warning.resolution is WarningResolution.UNRESOLVED:
            service.resolve_warning(
                imported.pack_id,
                warning.left_reference_id,
                warning.right_reference_id,
                WarningResolution.ACCEPTED,
            )
    ready = service.pack_status(imported.pack_id)
    assert ready.coverage.ready
    return character.character_id, ready.pack_id


def test_character_create_import_status_are_json_only_and_exactly_replayable(
    tmp_path: Path,
) -> None:
    character_id, imported = _create_and_import(tmp_path)
    source = tmp_path / "private" / "source"

    replayed = _invoke(tmp_path, "import", character_id, "--source", str(source))
    status = _invoke(tmp_path, "status", str(imported["pack_id"]))

    assert replayed.exit_code == status.exit_code == 0
    assert _payload(replayed)["data"] == imported
    shown = _payload(status)["data"]
    assert isinstance(shown, dict)
    assert shown["pack_id"] == imported["pack_id"]
    assert str(source) not in json.dumps(imported, sort_keys=True)
    assert "reviewer_id" not in json.dumps(imported, sort_keys=True)


def test_character_curate_requires_complete_annotation_and_explicit_disposition(
    tmp_path: Path,
) -> None:
    _, imported = _create_and_import(tmp_path)
    references = imported["references"]
    assert isinstance(references, list)
    reference = references[0]
    assert isinstance(reference, dict)
    base = ["curate", str(imported["pack_id"]), str(reference["reference_id"])]

    incomplete = _invoke(tmp_path, *base, "--accept")
    missing_disposition = _invoke(
        tmp_path,
        *base,
        "--framing",
        "portrait",
        "--head-angle",
        "front",
        "--body-visibility",
        "clear_face",
        "--expression",
        "neutral",
        "--lighting",
        "ordinary",
        "--occlusion",
        "none",
        "--identity-confirmed",
    )

    for result in (incomplete, missing_disposition):
        assert result.exit_code == 2
        assert _payload(result)["error"] == {
            "code": "invalid_character_input",
            "message": "character input is invalid",
        }


def test_character_curate_accepts_all_explicit_human_annotations(tmp_path: Path) -> None:
    _, imported = _create_and_import(tmp_path)
    reference = imported["references"][0]  # type: ignore[index]

    result = _invoke(
        tmp_path,
        "curate",
        str(imported["pack_id"]),
        str(reference["reference_id"]),  # type: ignore[index]
        "--framing",
        "full_body",
        "--head-angle",
        "profile",
        "--body-visibility",
        "full_body",
        "--expression",
        "serious",
        "--lighting",
        "directional",
        "--occlusion",
        "partial",
        "--identity-confirmed",
        "--accept",
    )

    assert result.exit_code == 0
    data = _payload(result)["data"]
    assert isinstance(data, dict)
    curated = data["references"][0]  # type: ignore[index]
    assert curated["disposition"] == "active"
    assert curated["annotation"] == {
        "framing": "full_body",
        "head_angle": "profile",
        "body_visibility": "full_body",
        "expression": "serious",
        "lighting": "directional",
        "occlusion": "partial",
        "identity_confirmed": True,
    }


def test_character_export_is_write_once(tmp_path: Path) -> None:
    _character_id, pack_id = _ready_pack(tmp_path)
    output = tmp_path / "pack.json"

    preregistered = _invoke(tmp_path, "preregister", pack_id)
    result = _invoke(tmp_path, "export-pack", pack_id, "--output", str(output))

    assert preregistered.exit_code == result.exit_code == 0
    receipt = _payload(result)["data"]
    assert isinstance(receipt, dict)
    original = output.read_bytes()
    exported = json.loads(original)
    assert exported["pack"]["pack_id"] == pack_id
    assert exported["pack"]["duplicate_hash_algorithm_version"] == (
        "dhash64-exif-transpose-lanczos.v1"
    )
    assert exported["pack"]["near_duplicate_threshold_inclusive"] == 5
    assert len(exported["references"]) == 12
    assert receipt["sha256"] == hashlib.sha256(original).hexdigest()

    second = _invoke(tmp_path, "export-pack", pack_id, "--output", str(output))

    assert second.exit_code == 2
    assert _payload(second)["error"] == {
        "code": "invalid_character_input",
        "message": "character input is invalid",
    }
    assert output.read_bytes() == original
