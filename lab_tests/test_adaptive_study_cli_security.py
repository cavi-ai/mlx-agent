from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lab_tests.test_adaptive_study_cli import (
    _common,
    _manifest,
    _payload,
    _service,
    _write_manifest,
    runner,
)
from lab_tests.test_adaptive_study_service import complete_generation
from video_lab.adaptive_study_models import (
    AdaptiveStudyDefinition,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import AdaptiveStudyRepository
from video_lab.cli import app
from video_lab.database import Database
from video_lab.domain import DisplayPreference
from video_lab.evidence_service import EvidenceNotFound, EvidenceService
from video_lab.preference_optimizer import OptimizerConfig
from video_lab.queue_service import QueueService
from video_lab.repositories import ExperimentRepository, JobRepository

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _definition(study_id: str, *, seed: int, created_at: datetime) -> AdaptiveStudyDefinition:
    manifest = _manifest().model_copy(update={"seed": seed})
    config = OptimizerConfig(posterior_draws=256)
    return AdaptiveStudyDefinition(
        id=study_id,
        prompt=manifest.prompt,
        negative_prompt=manifest.negative_prompt,
        seed=manifest.seed,
        base_manifest=manifest,
        optimizer_config=config,
        input_fingerprint=adaptive_study_input_fingerprint(manifest, config),
        created_at=created_at,
    )




def test_creator_binding_is_first_versioned_event_and_enforced_through_terminal(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    definition = _definition("study_owned", seed=41, created_at=NOW)
    creator = _service(tmp_path, reviewer_id="creator")
    intruder = _service(tmp_path, reviewer_id="intruder")

    generating = creator.create_study(definition)
    first_event = AdaptiveStudyRepository(database).list_events(definition.id)[0]

    assert first_event.previous_state.value == "draft"
    assert first_event.new_state.value == "search_ready"
    assert first_event.payload == {
        "owner_binding_schema_version": "adaptive-study-owner-binding.v1",
        "reviewer_id": "creator",
        "input_fingerprint": definition.input_fingerprint,
    }
    assert generating.state.value == "generating_pair"
    with pytest.raises(PermissionError):
        intruder.create_study(definition)
    with pytest.raises(PermissionError):
        intruder.show(definition.id)
    with pytest.raises(PermissionError):
        intruder.resume(definition.id)
    assert intruder.list(limit=50) == ()

    complete_generation(database, tmp_path, definition.id)
    awaiting = creator.resume(definition.id)
    assert awaiting.review_id is not None
    with pytest.raises(PermissionError):
        intruder.show(definition.id)
    with pytest.raises(PermissionError):
        intruder.resume(definition.id)
    with pytest.raises(PermissionError):
        intruder.record_preference(definition.id, awaiting.review_id, DisplayPreference.TIE)

    terminal = creator.record_preference(definition.id, awaiting.review_id, DisplayPreference.TIE)
    for _ in range(9):
        complete_generation(database, tmp_path, definition.id)
        awaiting = creator.resume(definition.id)
        assert awaiting.review_id is not None
        terminal = creator.record_preference(
            definition.id, awaiting.review_id, DisplayPreference.TIE
        )
    assert terminal.state.value == "search_budget_exhausted"
    with pytest.raises(PermissionError):
        intruder.show(definition.id)
    with pytest.raises(PermissionError):
        intruder.resume(definition.id)
    assert intruder.list(limit=50) == ()


def test_authorized_list_filters_owner_before_limit_and_skips_unbound_legacy(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    owner_a = _service(tmp_path, reviewer_id="owner-a")
    owner_b = _service(tmp_path, reviewer_id="owner-b")
    owner_a.create_study(_definition("study_a1", seed=11, created_at=NOW))
    owner_b.create_study(_definition("study_b1", seed=12, created_at=NOW + timedelta(seconds=1)))
    owner_a.create_study(_definition("study_a2", seed=13, created_at=NOW + timedelta(seconds=2)))
    AdaptiveStudyRepository(database).create(
        _definition("study_unbound", seed=14, created_at=NOW + timedelta(seconds=3))
    )

    assert tuple(item.study_id for item in owner_a.list(limit=1)) == ("study_a1",)
    assert tuple(item.study_id for item in owner_a.list(limit=2)) == (
        "study_a1",
        "study_a2",
    )
    assert tuple(item.study_id for item in owner_b.list(limit=2)) == ("study_b1",)
    assert _service(tmp_path, reviewer_id="nobody").list(limit=2) == ()
    with pytest.raises(PermissionError):
        owner_a.show("study_unbound")
    with pytest.raises(PermissionError):
        owner_a.resume("study_unbound")


def test_study_without_subcommand_is_one_typed_json_error() -> None:
    result = runner.invoke(app, ["study"])

    assert result.exit_code == 2
    assert result.stderr == ""
    assert _payload(result) == {
        "schema_version": "cli.v1",
        "ok": False,
        "data": None,
        "error": {
            "code": "invalid_input",
            "message": "adaptive study command is required",
        },
    }


def test_study_error_handling_does_not_capture_an_unrelated_positional_value(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, [*_common(tmp_path), "experiment", "get", "study"])

    assert result.exit_code == 3
    assert result.stdout == ""
    assert result.stderr == "experiment not found: study\n"


def test_study_create_requires_reviewer_identity(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "lab.sqlite3"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "study",
            "create",
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": "invalid_study_input",
        "message": "adaptive study input is invalid",
    }
