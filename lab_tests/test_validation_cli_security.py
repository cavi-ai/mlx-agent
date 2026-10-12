from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_tests.test_validation_cli import _panel_path, _payload, _stage_1_id, _validation
from video_lab.database import Database

_GLOBAL_OPTIONS = [
    {"option": "--database", "value": "PATH", "required": True},
    {"option": "--artifacts", "value": "PATH", "required": True},
    {"option": "--reviewer-id", "value": "TEXT", "required": True},
]
_GLOBAL_USAGE = "video-lab --database PATH --artifacts PATH --reviewer-id TEXT validation"
_VALIDATION_HELP = {
    "create": {
        "usage": "create --study STUDY_ID --panel PANEL_PATH",
        "arguments": [],
        "options": ["--study", "--panel"],
        "choices": {},
    },
    "preview": {
        "usage": "preview CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "preregister": {
        "usage": "preregister CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "status": {
        "usage": "status CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "resume": {
        "usage": "resume CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "submit": {
        "usage": "submit CAMPAIGN_ID --review-id REVIEW_ID --choice {a|b|tie}",
        "arguments": ["campaign_id"],
        "options": ["--review-id", "--choice"],
        "choices": {"--choice": ["a", "b", "tie"]},
    },
    "analyze": {
        "usage": "analyze CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "export-evidence": {
        "usage": "export-evidence CAMPAIGN_ID --output OUTPUT_PATH",
        "arguments": ["campaign_id"],
        "options": ["--output"],
        "choices": {},
    },
}


def test_validation_root_help_is_a_single_versioned_json_projection(tmp_path: Path) -> None:
    result = _validation(tmp_path, "--help")

    assert result.exit_code == 0
    assert result.stderr == ""
    payload = _payload(result)
    assert payload["schema_version"] == "cli.v1"
    assert payload["ok"] is True
    assert payload["error"] is None
    data = payload["data"]
    assert isinstance(data, dict)
    assert data == {
        "schema_version": "validation-cli-help.v1",
        "command": "validation",
        "usage": f"{_GLOBAL_USAGE} <command>",
        "global_options": _GLOBAL_OPTIONS,
        "required_arguments": [],
        "required_options": [],
        "choices": {},
        "commands": list(_VALIDATION_HELP),
    }


@pytest.mark.parametrize("command", tuple(_VALIDATION_HELP))
def test_validation_subcommand_help_is_a_single_agent_json_projection(
    tmp_path: Path, command: str
) -> None:
    result = _validation(tmp_path, command, "--help")

    assert result.exit_code == 0
    assert result.stderr == ""
    payload = _payload(result)
    assert payload["schema_version"] == "cli.v1"
    assert payload["ok"] is True
    assert payload["error"] is None
    data = payload["data"]
    assert isinstance(data, dict)
    expected = _VALIDATION_HELP[command]
    assert data == {
        "schema_version": "validation-cli-help.v1",
        "command": f"validation {command}",
        "usage": f"{_GLOBAL_USAGE} {expected['usage']}",
        "global_options": _GLOBAL_OPTIONS,
        "required_arguments": expected["arguments"],
        "required_options": expected["options"],
        "choices": expected["choices"],
    }


def test_validation_group_without_command_is_a_single_json_error() -> None:
    result = _validation(Path("."))

    assert result.exit_code == 2
    assert result.stderr == ""
    assert _payload(result) == {
        "schema_version": "cli.v1",
        "ok": False,
        "data": None,
        "error": {
            "code": "invalid_input",
            "message": "validation command is required",
        },
    }


def test_validation_rejects_malformed_panel_and_missing_reviewer_with_json_only_stdout(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    malformed_result = _validation(
        tmp_path, "create", "--study", "study_missing", "--panel", str(malformed)
    )
    missing_reviewer = _validation(
        tmp_path,
        "create",
        "--study",
        "study_missing",
        "--panel",
        str(malformed),
        reviewer_id=" ",
    )

    for result in (malformed_result, missing_reviewer):
        assert result.exit_code == 2
        assert result.stderr == ""
        assert _payload(result)["error"] == {
            "code": "invalid_validation_input",
            "message": "validation input is invalid",
        }


def test_validation_unknown_subcommand_is_a_single_json_error(tmp_path: Path) -> None:
    result = _validation(tmp_path, "unknown")

    assert result.exit_code == 2
    assert result.stderr == ""
    assert _payload(result)["error"] == {
        "code": "invalid_validation_input",
        "message": "validation input is invalid",
    }


def test_validation_unknown_subcommand_help_is_a_single_json_error(tmp_path: Path) -> None:
    result = _validation(tmp_path, "unknown", "--help")

    assert result.exit_code == 2
    assert result.stderr == ""
    assert _payload(result) == {
        "schema_version": "cli.v1",
        "ok": False,
        "data": None,
        "error": {
            "code": "invalid_input",
            "message": "invalid validation command input",
        },
    }


def test_validation_campaigns_are_owner_scoped_across_restart_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_id = _stage_1_id(tmp_path, monkeypatch, reviewer_id="owner-a")
    created = _validation(
        tmp_path,
        "create",
        "--study",
        study_id,
        "--panel",
        str(_panel_path(tmp_path)),
        reviewer_id="owner-a",
    )
    data = _payload(created)["data"]
    assert isinstance(data, dict)
    campaign_id = data["campaign_id"]
    assert isinstance(campaign_id, str)

    replay = _validation(
        tmp_path,
        "status",
        campaign_id,
        reviewer_id="owner-a",
    )
    intruder = _validation(
        tmp_path,
        "status",
        campaign_id,
        reviewer_id="owner-b",
    )

    assert replay.exit_code == 0
    assert _payload(replay)["data"] == data
    assert intruder.exit_code == 2
    assert _payload(intruder)["error"] == {
        "code": "validation_not_found",
        "message": "validation campaign was not found",
    }


def test_validation_rejects_cross_domain_and_stale_opaque_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_id = _stage_1_id(tmp_path, monkeypatch)
    created = _validation(
        tmp_path, "create", "--study", study_id, "--panel", str(_panel_path(tmp_path))
    )
    data = _payload(created)["data"]
    assert isinstance(data, dict)
    campaign_id = data["campaign_id"]
    assert isinstance(campaign_id, str)
    assert _validation(tmp_path, "preregister", campaign_id).exit_code == 0
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    with database.connect() as connection:
        job_id = str(connection.execute("SELECT id FROM jobs LIMIT 1").fetchone()["id"])

    for foreign in (study_id, job_id, "artifact_missing"):
        result = _validation(tmp_path, "status", foreign)
        assert result.exit_code == 2
        assert _payload(result)["error"] == {
            "code": "validation_not_found",
            "message": "validation campaign was not found",
        }
    stale = _validation(tmp_path, "submit", campaign_id, "--review-id", "opaque", "--choice", "a")
    assert stale.exit_code == 2
    assert _payload(stale)["error"] == {
        "code": "validation_conflict",
        "message": "validation action conflicts with sealed evidence",
    }
    assert "guidance_scale" not in json.dumps(_payload(created), sort_keys=True)
