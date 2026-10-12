import json
from pathlib import Path

from typer.testing import CliRunner, Result

from lab_tests.test_review_service import _experiment, _scores, _service
from video_lab.cli import app
from video_lab.database import Database

runner = CliRunner()


def _common(tmp_path: Path, reviewer_id: str = "local-reviewer") -> list[str]:
    return [
        "--database",
        str(tmp_path / "lab.sqlite3"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--reviewer-id",
        reviewer_id,
    ]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    return database


def _payload(result: Result, *, ok: bool = True) -> dict[str, object]:
    assert result.exit_code == (0 if ok else 2)
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    assert result.stdout == json.dumps(payload, separators=(",", ":")) + "\n"
    assert payload["schema_version"] == "cli.v1"
    assert payload["ok"] is ok
    if ok:
        assert payload["error"] is None
        assert payload["data"]["schema_version"] == "review.v1"
    return payload


def _submission(path: Path, *, preference: str = "left", score: int = 3, **extra: object) -> None:
    document = {
        "schema_version": "review.v1",
        "display_preference": preference,
        "scores": [item.model_dump(mode="json") | {"score": score} for item in _scores()],
        "notes": "careful local review",
        **extra,
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_queue_emits_review_v1_reason_codes(tmp_path: Path) -> None:
    database = _database(tmp_path)
    experiment, jobs, _ = _experiment(database, tmp_path / "artifacts")
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state = 'queued' WHERE id = ?", (jobs[0].id,))

    payload = _payload(runner.invoke(app, [*_common(tmp_path), "review", "queue", "--json"]))

    assert payload["data"] == {
        "schema_version": "review.v1",
        "items": [
            {
                "experiment_id": experiment.id,
                "reviewable": False,
                "reason_codes": ["jobs_not_completed"],
            }
        ],
    }


def test_start_and_show_are_blind_and_match_direct_service_session(tmp_path: Path) -> None:
    database = _database(tmp_path)
    experiment, _, artifacts = _experiment(database, tmp_path / "artifacts")
    direct = _service(database, tmp_path / "artifacts").start(experiment.id, "local-reviewer")

    started = _payload(
        runner.invoke(app, [*_common(tmp_path), "review", "start", experiment.id, "--json"])
    )
    shown = _payload(
        runner.invoke(app, [*_common(tmp_path), "review", "show", direct.session_id, "--json"])
    )

    assert (
        started["data"]
        == shown["data"]
        == {
            "schema_version": "review.v1",
            "session": direct.model_dump(mode="json"),
        }
    )
    serialized = json.dumps(started)
    forbidden = [
        *(variant.id for variant in experiment.variants),
        *(artifact.id for artifact in artifacts),
        *(artifact.relative_path for artifact in artifacts),
        *(artifact.sha256 for artifact in artifacts),
        "manifest",
        "variable_path",
        "alternate_value",
        "inference_steps",
    ]
    assert all(secret not in serialized for secret in forbidden)


def test_submit_correct_and_reveal_persist_through_service(tmp_path: Path) -> None:
    database = _database(tmp_path)
    experiment, _, _ = _experiment(database, tmp_path / "artifacts")
    started = _payload(
        runner.invoke(app, [*_common(tmp_path), "review", "start", experiment.id, "--json"])
    )
    session_id = started["data"]["session"]["session_id"]
    input_path = tmp_path / "submission.json"
    _submission(input_path)

    submitted = _payload(
        runner.invoke(
            app,
            [
                *_common(tmp_path),
                "review",
                "submit",
                session_id,
                "--input",
                str(input_path),
                "--json",
            ],
        )
    )
    assert submitted["data"]["submission"] == {
        "review_id": submitted["data"]["submission"]["review_id"],
        "revision": 1,
        "status": "submitted",
    }
    first_id = submitted["data"]["submission"]["review_id"]
    _submission(input_path, preference="right", score=4, supersedes_review_id=first_id)
    corrected = _payload(
        runner.invoke(
            app,
            [
                *_common(tmp_path),
                "review",
                "correct",
                session_id,
                "--input",
                str(input_path),
                "--json",
            ],
        )
    )
    assert corrected["data"]["submission"]["revision"] == 2
    assert corrected["data"]["submission"]["status"] == "corrected"

    reveal = _payload(
        runner.invoke(app, [*_common(tmp_path), "review", "reveal", session_id, "--json"])
    )
    assert reveal["data"]["reveal"]["review_id"] == corrected["data"]["submission"]["review_id"]
    assert (
        reveal["data"]["reveal"]["preferred_canonical_variant"]
        == reveal["data"]["reveal"]["canonical_by_display"]["right"]
    )
    persisted = _service(database, tmp_path / "artifacts").reveal(session_id, "local-reviewer")
    assert reveal["data"]["reveal"] == persisted.model_dump(mode="json")


def test_reveal_before_submit_is_stable_forbidden_without_secret_details(tmp_path: Path) -> None:
    database = _database(tmp_path)
    experiment, _, artifacts = _experiment(database, tmp_path / "artifacts")
    session = _service(database, tmp_path / "artifacts").start(experiment.id, "local-reviewer")

    result = runner.invoke(
        app, [*_common(tmp_path), "review", "reveal", session.session_id, "--json"]
    )
    payload = _payload(result, ok=False)

    assert payload["error"]["code"] == "review_reveal_forbidden"
    serialized = json.dumps(payload)
    secrets = [
        *(variant.id for variant in experiment.variants),
        *(artifact.id for artifact in artifacts),
        *(artifact.sha256 for artifact in artifacts),
        "manifest",
    ]
    assert all(secret not in serialized for secret in secrets)


def test_submission_file_rejects_invalid_json_schema_and_score_range(tmp_path: Path) -> None:
    database = _database(tmp_path)
    experiment, _, _ = _experiment(database, tmp_path / "artifacts")
    session = _service(database, tmp_path / "artifacts").start(experiment.id, "local-reviewer")
    input_path = tmp_path / "submission.json"

    documents = [
        "not json",
        json.dumps({"schema_version": "other.v1", "display_preference": "left", "scores": []}),
        json.dumps(
            {
                "schema_version": "review.v1",
                "display_preference": "left",
                "scores": [item.model_dump(mode="json") | {"score": 6} for item in _scores()],
            }
        ),
    ]
    for document in documents:
        input_path.write_text(document, encoding="utf-8")
        payload = _payload(
            runner.invoke(
                app,
                [
                    *_common(tmp_path),
                    "review",
                    "submit",
                    session.session_id,
                    "--input",
                    str(input_path),
                    "--json",
                ],
            ),
            ok=False,
        )
        assert payload["error"]["code"] == "invalid_review_input"


def test_reviewer_identity_is_required_and_owns_show_and_correction(tmp_path: Path) -> None:
    database = _database(tmp_path)
    experiment, _, _ = _experiment(database, tmp_path / "artifacts")
    session = _service(database, tmp_path / "artifacts").start(experiment.id, "owner")

    missing = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "lab.sqlite3"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "review",
            "show",
            session.session_id,
            "--json",
        ],
    )
    wrong = runner.invoke(
        app, [*_common(tmp_path, "intruder"), "review", "show", session.session_id, "--json"]
    )

    assert _payload(missing, ok=False)["error"]["code"] == "invalid_review_input"
    assert _payload(wrong, ok=False)["error"]["code"] == "review_not_owned"
