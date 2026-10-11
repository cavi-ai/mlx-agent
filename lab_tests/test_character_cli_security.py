from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_tests.test_character_cli import _create_and_import, _invoke, _payload

_HELP = {
    "create": {
        "usage": "create NAME",
        "arguments": ["name"],
        "options": [],
        "choices": {},
    },
    "import": {
        "usage": "import CHARACTER_ID --source SOURCE",
        "arguments": ["character_id"],
        "options": ["--source"],
        "choices": {},
    },
    "status": {
        "usage": "status PACK_ID",
        "arguments": ["pack_id"],
        "options": [],
        "choices": {},
    },
    "curate": {
        "usage": (
            "curate PACK_ID REFERENCE_ID --framing FRAMING --head-angle HEAD_ANGLE "
            "--body-visibility BODY_VISIBILITY --expression EXPRESSION --lighting LIGHTING "
            "--occlusion OCCLUSION --identity-confirmed|--identity-not-confirmed "
            "--accept|--reject"
        ),
        "arguments": ["pack_id", "reference_id"],
        "options": [
            "--framing",
            "--head-angle",
            "--body-visibility",
            "--expression",
            "--lighting",
            "--occlusion",
            "--identity-confirmed|--identity-not-confirmed",
            "--accept|--reject",
        ],
        "choices": {
            "--framing": ["portrait", "full_body"],
            "--head-angle": ["front", "three_quarter", "profile"],
            "--body-visibility": ["clear_face", "partial_body", "full_body"],
            "--expression": ["neutral", "smile", "serious", "angry", "surprised"],
            "--lighting": ["ordinary", "directional"],
            "--occlusion": ["none", "partial", "heavy"],
        },
    },
    "resolve-warning": {
        "usage": (
            "resolve-warning PACK_ID LEFT_REFERENCE_ID RIGHT_REFERENCE_ID "
            "--resolution {accepted|dismissed}"
        ),
        "arguments": ["pack_id", "left_reference_id", "right_reference_id"],
        "options": ["--resolution"],
        "choices": {"--resolution": ["accepted", "dismissed"]},
    },
    "preregister": {
        "usage": "preregister PACK_ID",
        "arguments": ["pack_id"],
        "options": [],
        "choices": {},
    },
    "export-pack": {
        "usage": "export-pack PACK_ID --output OUTPUT_PATH",
        "arguments": ["pack_id"],
        "options": ["--output"],
        "choices": {},
    },
    "open": {
        "usage": "open CHARACTER_ID --pack PACK_ID",
        "arguments": ["character_id"],
        "options": ["--pack"],
        "choices": {},
    },
}
_GLOBAL_OPTIONS = [
    {"option": "--database", "value": "PATH", "required": True},
    {"option": "--artifacts", "value": "PATH", "required": True},
    {"option": "--reviewer-id", "value": "TEXT", "required": True},
]
_GLOBAL_USAGE = "video-lab --database PATH --artifacts PATH --reviewer-id TEXT character"


def test_character_root_help_is_machine_readable(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "--help")

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload == {
        "schema_version": "cli.v1",
        "ok": True,
        "data": {
            "schema_version": "character-cli-help.v1",
            "command": "character",
            "usage": f"{_GLOBAL_USAGE} <command>",
            "global_options": _GLOBAL_OPTIONS,
            "required_arguments": [],
            "required_options": [],
            "choices": {},
            "commands": list(_HELP),
        },
        "error": None,
    }


@pytest.mark.parametrize("command", tuple(_HELP))
def test_character_subcommand_help_is_machine_readable(tmp_path: Path, command: str) -> None:
    result = _invoke(tmp_path, command, "--help")

    assert result.exit_code == 0
    payload = _payload(result)
    expected = _HELP[command]
    assert payload == {
        "schema_version": "cli.v1",
        "ok": True,
        "data": {
            "schema_version": "character-cli-help.v1",
            "command": f"character {command}",
            "usage": f"{_GLOBAL_USAGE} {expected['usage']}",
            "global_options": _GLOBAL_OPTIONS,
            "required_arguments": expected["arguments"],
            "required_options": expected["options"],
            "choices": expected["choices"],
        },
        "error": None,
    }


def test_character_requires_reviewer_and_enforces_ownership(tmp_path: Path) -> None:
    _, imported = _create_and_import(tmp_path)

    missing = _invoke(tmp_path, "status", str(imported["pack_id"]), reviewer=" ")
    intruder = _invoke(tmp_path, "status", str(imported["pack_id"]), reviewer="intruder")

    assert missing.exit_code == intruder.exit_code == 2
    assert _payload(missing)["error"] == {
        "code": "invalid_character_input",
        "message": "character input is invalid",
    }
    assert _payload(intruder)["error"] == {
        "code": "character_not_found",
        "message": "character evidence was not found",
    }


def test_character_invalid_enum_and_unknown_command_are_bounded_json(tmp_path: Path) -> None:
    _, imported = _create_and_import(tmp_path)
    reference = imported["references"][0]  # type: ignore[index]
    invalid = _invoke(
        tmp_path,
        "curate",
        str(imported["pack_id"]),
        str(reference["reference_id"]),  # type: ignore[index]
        "--framing",
        "cinematic",
    )
    unknown = _invoke(tmp_path, "unknown")
    unknown_option = _invoke(tmp_path, "status", str(imported["pack_id"]), "--private")

    for result in (invalid, unknown, unknown_option):
        assert result.exit_code == 2
        payload = _payload(result)
        assert payload["error"] == {
            "code": "invalid_character_input",
            "message": "character input is invalid",
        }
        assert str(tmp_path) not in json.dumps(payload, sort_keys=True)


def test_character_open_rejects_wrong_pack_and_reviewer_without_identifiers(
    tmp_path: Path,
) -> None:
    character_id, imported = _create_and_import(tmp_path)
    wrong_pack = _invoke(tmp_path, "open", character_id, "--pack", "character_pack_missing")
    intruder = _invoke(
        tmp_path,
        "open",
        character_id,
        "--pack",
        str(imported["pack_id"]),
        reviewer="intruder",
    )

    for result in (wrong_pack, intruder):
        assert result.exit_code == 2
        payload = _payload(result)
        assert payload["error"] == {
            "code": "character_not_found",
            "message": "character evidence was not found",
        }
        assert character_id not in json.dumps(payload, sort_keys=True)
        assert str(imported["pack_id"]) not in json.dumps(payload, sort_keys=True)


def test_character_import_failure_never_leaks_absolute_source_path(tmp_path: Path) -> None:
    created = _invoke(tmp_path, "create", "Mara")
    character = _payload(created)["data"]
    assert isinstance(character, dict)
    source = tmp_path / "private" / "missing"

    result = _invoke(
        tmp_path,
        "import",
        str(character["character_id"]),
        "--source",
        str(source),
    )

    assert result.exit_code == 2
    payload = _payload(result)
    assert payload["error"] == {
        "code": "invalid_character_input",
        "message": "character input is invalid",
    }
    assert str(source) not in json.dumps(payload, sort_keys=True)
