from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_tests.test_character_benchmark_cli import _args, _payload, runner
from lab_tests.test_character_benchmark_service import complete_jobs, panel_input, service
from video_lab.character_benchmark_repository import CharacterBenchmarkConflict
from video_lab.cli import app
from video_lab.database import Database

COMMANDS = (
    "create",
    "preview",
    "preregister",
    "status",
    "resume",
    "submit-identity",
    "submit-secondary",
    "analyze",
    "export-evidence",
)


@pytest.mark.parametrize("command", COMMANDS)
def test_character_benchmark_help_is_json(tmp_path: Path, command: str) -> None:
    result = runner.invoke(app, _args(tmp_path, command, "--help"))
    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["data"]["command"] == f"character-benchmark {command}"


def test_character_benchmark_group_help_and_missing_command_are_json(tmp_path: Path) -> None:
    help_result = runner.invoke(app, _args(tmp_path, "--help"))
    missing = runner.invoke(app, _args(tmp_path))
    assert help_result.exit_code == 0
    assert _payload(help_result)["data"]["commands"] == list(COMMANDS)
    assert missing.exit_code == 2
    assert _payload(missing)["error"]["message"] == "character benchmark command is required"


def test_invalid_choice_is_sanitized_json(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        _args(
            tmp_path,
            "submit-identity",
            "opaque",
            "--review-id",
            "raw",
            "--trial-id",
            "raw",
            "--choice",
            "candidate-a",
        ),
    )
    assert result.exit_code == 2
    assert result.stderr == ""
    assert _payload(result)["error"] == {
        "code": "invalid_character_benchmark_input",
        "message": "character benchmark input is invalid",
    }


@pytest.mark.parametrize(
    ("message", "code"),
    (
        ("character pack evidence is not canonical", "character_benchmark_incomplete_curation"),
        ("stage 2 evidence is not qualified", "character_benchmark_stage_2_ineligible"),
        ("character benchmark cannot enter analysis", "character_benchmark_invalid_phase"),
        (
            "character benchmark review retry does not match evidence",
            "character_benchmark_stale_review",
        ),
        (
            "character benchmark artifact ownership is inconsistent",
            "character_benchmark_corrupt_provenance",
        ),
    ),
)
def test_service_failures_have_stable_sanitized_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    code: str,
) -> None:
    from video_lab import cli

    class Broken:
        def show(self, _campaign_id: str) -> None:
            raise CharacterBenchmarkConflict(message)

    monkeypatch.setattr(cli, "_character_benchmark_service_for_id", lambda *_: Broken())
    result = runner.invoke(app, _args(tmp_path, "status", "character_benchmark_" + "a" * 64))

    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": code,
        "message": "character benchmark action conflicts with sealed evidence",
    }


def _ready_benchmark(tmp_path: Path):
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    benchmark = service(database, tmp_path)
    created = benchmark.create(
        character_pack_id="character_pack_" + "6" * 64,
        stage_2_campaign_id="validation_stage_2",
        identity_prompt="private identity recipe",
        panel_input=panel_input(),
    )
    benchmark.preregister(created.campaign_id)
    complete_jobs(benchmark, tmp_path)
    view = benchmark.resume(created.campaign_id)
    return database, benchmark, created.campaign_id, view


def _use_service(monkeypatch: pytest.MonkeyPatch, benchmark) -> None:
    from video_lab import cli

    monkeypatch.setattr(cli, "_character_benchmark_service_for_id", lambda *_: benchmark)


def test_real_corrupt_review_artifact_is_provenance_not_stale_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, benchmark, campaign_id, view = _ready_benchmark(tmp_path)
    assert view.review_id is not None
    with database.connect() as connection:
        artifact_id = str(
            connection.execute(
                "SELECT left_artifact_id FROM review_sessions WHERE id=?", (view.review_id,)
            ).fetchone()[0]
        )
    benchmark.experiments.artifacts.resolve(artifact_id).write_bytes(b"corrupt")
    _use_service(monkeypatch, benchmark)

    result = runner.invoke(
        app,
        _args(
            tmp_path,
            "submit-identity",
            campaign_id,
            "--review-id",
            view.review_id,
            "--trial-id",
            view.trial_id,
            "--choice",
            "a",
        ),
    )

    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": "character_benchmark_corrupt_provenance",
        "message": "character benchmark action conflicts with sealed evidence",
    }


def test_real_campaign_is_owner_scoped_and_cross_owner_campaign_is_hidden(
    tmp_path: Path,
) -> None:
    _database, _benchmark, campaign_id, _view = _ready_benchmark(tmp_path)
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "lab.sqlite3"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--reviewer-id",
            "intruder",
            "character-benchmark",
            "status",
            campaign_id,
        ],
    )

    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": "character_benchmark_not_found",
        "message": "character benchmark was not found",
    }


def test_real_review_rejects_future_trial_and_cross_campaign_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _database, benchmark, campaign_id, view = _ready_benchmark(tmp_path)
    assert view.review_id is not None
    future_trial = benchmark.campaigns.list_trials(campaign_id)[1].id
    _use_service(monkeypatch, benchmark)
    future = runner.invoke(
        app,
        _args(
            tmp_path,
            "submit-identity",
            campaign_id,
            "--review-id",
            view.review_id,
            "--trial-id",
            future_trial,
            "--choice",
            "a",
        ),
    )

    other_root = tmp_path / "other"
    other_root.mkdir()
    _other_db, _other, other_campaign, other_view = _ready_benchmark(other_root)
    assert other_view.review_id is not None
    cross = runner.invoke(
        app,
        _args(
            tmp_path,
            "submit-identity",
            campaign_id,
            "--review-id",
            other_view.review_id,
            "--trial-id",
            view.trial_id,
            "--choice",
            "a",
        ),
    )

    assert future.exit_code == cross.exit_code == 2
    assert _payload(future)["error"]["code"] == "character_benchmark_conflict"
    assert _payload(cross)["error"]["code"] == "character_benchmark_stale_review"
    assert other_campaign != ""


def test_real_status_hides_schedule_repeats_recipes_and_internal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _database, benchmark, campaign_id, _view = _ready_benchmark(tmp_path)
    assert sum(trial.hidden_repeat for trial in benchmark.campaigns.list_trials(campaign_id)) == 4
    _use_service(monkeypatch, benchmark)

    result = runner.invoke(app, _args(tmp_path, "status", campaign_id))
    encoded = result.stdout

    assert result.exit_code == 0
    for secret in (
        "private identity recipe",
        "recipe_digest",
        "hidden_repeat",
        "reuses_trial_id",
        str(tmp_path),
    ):
        assert secret not in encoded


def test_real_resume_file_uris_are_authorized_and_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _database, benchmark, campaign_id, view = _ready_benchmark(tmp_path)
    artifact = benchmark.experiments.artifacts.list_for_job(
        str(benchmark._session(view.review_id)["left_job_id"])
    )[0]
    benchmark.character_packs.export_pack = lambda _pack: SimpleNamespace(
        pack=benchmark.character_packs.pack,
        references=(SimpleNamespace(artifact_id=artifact.id),),
    )
    _use_service(monkeypatch, benchmark)

    result = runner.invoke(app, _args(tmp_path, "resume", campaign_id))
    payload = _payload(result)
    data = payload["data"]

    assert result.exit_code == 0
    assert data["prompt"]
    assert len(data["references"]) == 1
    assert len(data["media"]) == 2
    for uri in [*data["references"], *(item["media_ref"] for item in data["media"])]:
        assert uri.startswith("file://")
        assert Path(uri.removeprefix("file://")).is_file()
        assert str(tmp_path) in uri
