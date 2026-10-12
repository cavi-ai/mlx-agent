import json
from pathlib import Path

from pydantic import BaseModel
from typer.testing import CliRunner, Result

from lab_tests.test_capabilities import manifest
from lab_tests.test_services import capabilities
from video_lab.cli import app
from video_lab.domain import ExperimentDraft

runner = CliRunner()


def write_json(path: Path, value: object) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    path.write_text(json.dumps(value), encoding="utf-8")


def common(tmp_path: Path) -> list[str]:
    return [
        "--database",
        str(tmp_path / "lab.sqlite3"),
        "--artifacts",
        str(tmp_path / "data"),
    ]


def draft_files(tmp_path: Path, draft: ExperimentDraft) -> tuple[Path, Path]:
    draft_path = tmp_path / "draft.json"
    capabilities_path = tmp_path / "capabilities.json"
    write_json(draft_path, draft)
    write_json(capabilities_path, capabilities())
    return draft_path, capabilities_path


def test_create_get_and_list_emit_versioned_json(tmp_path: Path) -> None:
    draft_path, capabilities_path = draft_files(
        tmp_path,
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8),
    )

    created = runner.invoke(
        app,
        [
            *common(tmp_path),
            "experiment",
            "create",
            "--draft",
            str(draft_path),
            "--capabilities",
            str(capabilities_path),
            "--json",
        ],
    )

    assert created.exit_code == 0
    payload = json.loads(created.stdout)
    assert payload["schema_version"] == "cli.v1"
    assert payload["ok"] is True
    assert payload["data"]["experiment"]["changed_paths"] == ["inference_steps"]
    experiment_id = payload["data"]["experiment"]["id"]

    fetched = runner.invoke(app, [*common(tmp_path), "experiment", "get", experiment_id, "--json"])
    listed = runner.invoke(app, [*common(tmp_path), "experiment", "list", "--json"])

    assert fetched.exit_code == listed.exit_code == 0
    assert json.loads(fetched.stdout)["data"]["experiment"]["id"] == experiment_id
    assert json.loads(listed.stdout)["data"][0]["experiment"]["id"] == experiment_id


def test_validate_returns_changed_path_without_writing(tmp_path: Path) -> None:
    draft_path, capabilities_path = draft_files(
        tmp_path,
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8),
    )

    validated = runner.invoke(
        app,
        [
            *common(tmp_path),
            "experiment",
            "validate",
            "--draft",
            str(draft_path),
            "--capabilities",
            str(capabilities_path),
            "--json",
        ],
    )
    listed = runner.invoke(app, [*common(tmp_path), "experiment", "list", "--json"])

    assert validated.exit_code == 0
    assert json.loads(validated.stdout)["data"] == {"changed_paths": ["inference_steps"]}
    assert json.loads(listed.stdout)["data"] == []


def test_validation_error_is_json_and_exit_2(tmp_path: Path) -> None:
    draft_path, capabilities_path = draft_files(
        tmp_path,
        ExperimentDraft(
            baseline=manifest().model_copy(update={"dtype": "bfloat16"}),
            variable_path="inference_steps",
            alternate_value=8,
        ),
    )
    result = runner.invoke(
        app,
        [
            *common(tmp_path),
            "experiment",
            "validate",
            "--draft",
            str(draft_path),
            "--capabilities",
            str(capabilities_path),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "manifest_unsupported"


def test_mismatched_adapter_is_compact_json_exit_2_without_persistence(
    tmp_path: Path,
) -> None:
    draft_path, capabilities_path = draft_files(
        tmp_path,
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8),
    )
    write_json(capabilities_path, capabilities().model_copy(update={"adapter_id": "other"}))

    result = runner.invoke(
        app,
        [
            *common(tmp_path),
            "experiment",
            "create",
            "--draft",
            str(draft_path),
            "--capabilities",
            str(capabilities_path),
            "--json",
        ],
    )
    listed = runner.invoke(app, [*common(tmp_path), "experiment", "list", "--json"])

    assert result.exit_code == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert result.stdout == json.dumps(payload, separators=(",", ":")) + "\n"
    assert payload["schema_version"] == "cli.v1"
    assert payload["error"]["code"] == "manifest_unsupported"
    assert payload["error"]["details"][0]["field"] == "adapter_id"
    assert json.loads(listed.stdout)["data"] == []


def test_invalid_json_is_exit_2_and_stdout_is_one_compact_envelope(tmp_path: Path) -> None:
    draft_path = tmp_path / "draft.json"
    capabilities_path = tmp_path / "capabilities.json"
    draft_path.write_text("not json", encoding="utf-8")
    write_json(capabilities_path, capabilities())

    result = runner.invoke(
        app,
        [
            *common(tmp_path),
            "experiment",
            "create",
            "--draft",
            str(draft_path),
            "--capabilities",
            str(capabilities_path),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    assert result.stdout == json.dumps(payload, separators=(",", ":")) + "\n"
    assert payload["error"]["code"] == "invalid_input"


def assert_json_error(result: Result, code: str, exit_code: int) -> None:
    assert result.exit_code == exit_code
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert result.stdout == json.dumps(payload, separators=(",", ":")) + "\n"
    assert set(payload) == {"schema_version", "ok", "data", "error"}
    assert payload["schema_version"] == "cli.v1"
    assert payload["ok"] is False
    assert payload["data"] is None
    assert set(payload["error"]) == {"code", "message"}
    assert payload["error"]["code"] == code


def test_nonexistent_draft_is_invalid_input_exit_2(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities.json"
    write_json(capabilities_path, capabilities())

    result = runner.invoke(
        app,
        [
            *common(tmp_path),
            "experiment",
            "create",
            "--draft",
            str(tmp_path / "missing.json"),
            "--capabilities",
            str(capabilities_path),
            "--json",
        ],
    )

    assert_json_error(result, "invalid_input", 2)


def test_directory_input_is_invalid_input_exit_2(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities.json"
    write_json(capabilities_path, capabilities())

    result = runner.invoke(
        app,
        [
            *common(tmp_path),
            "experiment",
            "validate",
            "--draft",
            str(tmp_path),
            "--capabilities",
            str(capabilities_path),
            "--json",
        ],
    )

    assert_json_error(result, "invalid_input", 2)


def test_missing_required_values_are_json_exit_2(tmp_path: Path) -> None:
    missing_id = runner.invoke(app, [*common(tmp_path), "experiment", "get", "--json"])
    missing_files = runner.invoke(app, [*common(tmp_path), "experiment", "create", "--json"])

    assert_json_error(missing_id, "invalid_input", 2)
    assert_json_error(missing_files, "invalid_input", 2)
    assert "experiment_id" in json.loads(missing_id.stdout)["error"]["message"]
    message = json.loads(missing_files.stdout)["error"]["message"]
    assert "draft" in message
    assert "capabilities" in message


def test_missing_required_value_is_human_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(app, [*common(tmp_path), "experiment", "get"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == "missing required value: experiment_id\n"


def test_missing_global_paths_are_json_exit_2() -> None:
    result = runner.invoke(app, ["db", "init", "--json"])

    assert_json_error(result, "invalid_input", 2)
    message = json.loads(result.stdout)["error"]["message"]
    assert "database" in message
    assert "artifacts" in message


def test_bad_limit_is_json_parser_envelope() -> None:
    result = runner.invoke(app, ["experiment", "list", "--limit", "nope", "--json"])

    assert_json_error(result, "invalid_input", 2)


def test_unknown_option_is_json_parser_envelope() -> None:
    result = runner.invoke(app, ["experiment", "list", "--unknown", "--json"])

    assert_json_error(result, "invalid_input", 2)


def test_unknown_subcommand_is_json_parser_envelope() -> None:
    result = runner.invoke(app, ["experiment", "unknown", "--json"])

    assert_json_error(result, "invalid_input", 2)


def test_bad_limit_without_json_keeps_human_parser_diagnostic() -> None:
    result = runner.invoke(app, ["experiment", "list", "--limit", "nope"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Usage:" in result.stderr
    assert "Invalid value" in result.stderr


def test_help_remains_standard_human_output() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "--database" in result.stdout
    assert result.stderr == ""


def test_missing_experiment_is_json_and_exit_3(tmp_path: Path) -> None:
    result = runner.invoke(app, [*common(tmp_path), "experiment", "get", "exp_missing", "--json"])

    assert result.exit_code == 3
    assert json.loads(result.stdout)["error"]["code"] == "record_not_found"


def test_missing_experiment_is_human_text_and_exit_3(tmp_path: Path) -> None:
    result = runner.invoke(app, [*common(tmp_path), "experiment", "get", "exp_missing"])

    assert result.exit_code == 3
    assert result.stdout == ""
    assert result.stderr == "experiment not found: exp_missing\n"


def test_db_init_and_human_output(tmp_path: Path) -> None:
    initialized = runner.invoke(app, [*common(tmp_path), "db", "init", "--json"])
    human = runner.invoke(app, [*common(tmp_path), "experiment", "list"])

    assert initialized.exit_code == 0
    assert json.loads(initialized.stdout) == {
        "schema_version": "cli.v1",
        "ok": True,
        "data": {"initialized": True},
        "error": None,
    }
    assert human.exit_code == 0
    assert human.stdout == "No experiments.\n"
