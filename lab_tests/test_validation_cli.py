from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from typer.testing import Result

from lab_tests.test_adaptive_study_cli import _common, _manifest, _write_manifest, runner
from lab_tests.test_adaptive_study_service import complete_generation
from lab_tests.test_validation_service import _complete_validation_jobs, panel_input, service
from video_lab.adaptive_study_repository import AdaptiveStudyRepository
from video_lab.cli import app
from video_lab.database import Database
from video_lab.preference_optimizer import optimize_preference
from video_lab.validation_models import (
    ValidationArchetype,
    ValidationPanelInput,
    ValidationPrompt,
)


def _payload(result: Result) -> dict[str, object]:
    assert result.stdout.count("\n") == 1
    value = json.loads(result.stdout)
    assert result.stdout == json.dumps(value, separators=(",", ":")) + "\n"
    return value


def _panel_path(tmp_path: Path) -> Path:
    panel = ValidationPanelInput(
        prompts=tuple(
            ValidationPrompt(
                archetype=archetype,
                prompt=f"character {archetype.value}",
                negative_prompt="flicker",
                seeds=(101, 202, 303),
            )
            for archetype in ValidationArchetype
        ),
        schedule_seed=73021,
    )
    path = tmp_path / "validation-panel.json"
    path.write_text(panel.model_dump_json(), encoding="utf-8")
    return path


def _stage_1_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reviewer_id: str = "reviewer"
) -> str:
    from video_lab import cli

    original_service_factory = cli._adaptive_study_service

    def interior_optimizer(config, observations):
        decision = optimize_preference(config, observations)
        midpoint = decision.posterior.grid[len(decision.posterior.grid) // 2]
        return decision.model_copy(
            update={"posterior": decision.posterior.model_copy(update={"recommendation": midpoint})}
        )

    def service_factory(context, capabilities):
        service = original_service_factory(context, capabilities)
        service.optimizer = interior_optimizer
        return service

    monkeypatch.setattr(cli, "_adaptive_study_service", service_factory)
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    manifest_path = _write_manifest(
        tmp_path, _manifest().model_copy(update={"guidance_scale": 3.0, "seed": 42})
    )
    created = runner.invoke(
        app,
        [*_common(tmp_path, reviewer_id), "study", "create", "--manifest", str(manifest_path)],
    )
    assert created.exit_code == 0
    data = _payload(created)["data"]
    assert isinstance(data, dict)
    study_id = data["study_id"]
    assert isinstance(study_id, str)
    for _ in range(10):
        complete_generation(database, tmp_path, study_id)
        resumed = runner.invoke(app, [*_common(tmp_path, reviewer_id), "study", "resume", study_id])
        assert resumed.exit_code == 0
        review_id = _payload(resumed)["data"]
        assert isinstance(review_id, dict)
        value = review_id["review_id"]
        assert isinstance(value, str)
        with database.connect() as connection:
            row = connection.execute(
                "SELECT variant.manifest_json FROM review_sessions AS session "
                "JOIN variants AS variant ON variant.id=session.left_variant_id "
                "WHERE session.id=?",
                (value,),
            ).fetchone()
        assert row is not None
        left_manifest = json.loads(str(row["manifest_json"]))
        choice = "a" if left_manifest["guidance_scale"] == 3.0 else "b"
        submitted = runner.invoke(
            app,
            [
                *_common(tmp_path, reviewer_id),
                "study",
                "submit",
                study_id,
                "--review-id",
                value,
                "--choice",
                choice,
            ],
        )
        assert submitted.exit_code == 0
    step = AdaptiveStudyRepository(database).list_optimizer_steps(study_id)[-1]
    assert step.posterior.recommendation in step.posterior.grid[1:-1]
    return study_id


def _validation(tmp_path: Path, *arguments: str, reviewer_id: str = "reviewer") -> Result:
    return runner.invoke(app, [*_common(tmp_path, reviewer_id), "validation", *arguments])


def test_validation_create_preview_preregister_status_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_id = _stage_1_id(tmp_path, monkeypatch)
    created = _validation(
        tmp_path, "create", "--study", study_id, "--panel", str(_panel_path(tmp_path))
    )

    assert created.exit_code == 0
    campaign_data = _payload(created)["data"]
    assert isinstance(campaign_data, dict)
    campaign_id = campaign_data["campaign_id"]
    assert isinstance(campaign_id, str)
    previewed = _validation(tmp_path, "preview", campaign_id)
    preregistered = _validation(tmp_path, "preregister", campaign_id)
    status = _validation(tmp_path, "status", campaign_id)

    assert previewed.exit_code == preregistered.exit_code == status.exit_code == 0
    preview = _payload(previewed)["data"]
    assert isinstance(preview, dict)
    assert preview["primary_count"] == 36
    preregistration = _payload(preregistered)["data"]
    shown = _payload(status)["data"]
    assert isinstance(preregistration, dict) and isinstance(shown, dict)
    assert preregistration["state"] == shown["state"] == "generating"
    assert shown["preview"] is None


def test_validation_create_is_idempotent_and_draft_preview_is_sealed_after_preregistration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_id = _stage_1_id(tmp_path, monkeypatch)
    panel = _panel_path(tmp_path)
    first = _validation(tmp_path, "create", "--study", study_id, "--panel", str(panel))
    retry = _validation(tmp_path, "create", "--study", study_id, "--panel", str(panel))
    first_data = _payload(first)["data"]
    retry_data = _payload(retry)["data"]
    assert first.exit_code == retry.exit_code == 0
    assert first_data == retry_data
    assert isinstance(first_data, dict)
    campaign_id = first_data["campaign_id"]
    assert isinstance(campaign_id, str)

    assert _validation(tmp_path, "preregister", campaign_id).exit_code == 0
    sealed = _validation(tmp_path, "preview", campaign_id)

    assert sealed.exit_code == 2
    assert _payload(sealed)["error"] == {
        "code": "validation_preview_sealed",
        "message": "validation preview is sealed after preregistration",
    }


def test_validation_export_uses_exclusive_output_creation_and_sha256_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    result = _validation(tmp_path, "export-evidence", "validation_missing", "--output", str(output))

    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": "validation_not_found",
        "message": "validation campaign was not found",
    }
    assert not output.exists()

    payload = b'{"evidence":"sealed"}\n'
    output.write_bytes(payload)
    existing = _validation(
        tmp_path, "export-evidence", "validation_missing", "--output", str(output)
    )
    assert existing.exit_code == 2
    assert _payload(existing)["error"] == {
        "code": "invalid_validation_input",
        "message": "validation input is invalid",
    }
    assert output.read_bytes() == payload
    assert hashlib.sha256(payload).hexdigest()


def test_validation_submit_accepts_only_a_b_or_tie(tmp_path: Path) -> None:
    result = _validation(
        tmp_path,
        "submit",
        "validation_missing",
        "--review-id",
        "review_missing",
        "--choice",
        "invalid",
    )

    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": "invalid_validation_input",
        "message": "validation input is invalid",
    }


def test_validation_resume_returns_dereferenceable_local_media_uris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    campaign_service = service(database, tmp_path)
    created = campaign_service.create(stage_1_id="study_1", panel_input=panel_input())
    campaign_service.preregister(created.campaign_id)
    _complete_validation_jobs(campaign_service, tmp_path)
    from video_lab import cli

    monkeypatch.setattr(cli, "_validation_campaign_service_for_id", lambda *_: campaign_service)
    resumed = _validation(tmp_path, "resume", created.campaign_id, reviewer_id="owner")

    assert resumed.exit_code == 0
    data = _payload(resumed)["data"]
    assert isinstance(data, dict)
    media = data["media"]
    assert isinstance(media, list) and len(media) == 2
    for item in media:
        parsed = urlparse(item["media_ref"])
        assert parsed.scheme == "file"
        path = Path(unquote(parsed.path))
        assert path.is_file()
        assert path.read_bytes().startswith(b"validation-")
