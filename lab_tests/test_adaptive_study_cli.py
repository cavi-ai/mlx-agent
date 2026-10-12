from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from lab_tests.test_adaptive_study_service import complete_generation
from lab_tests.test_fake_adapter import manifest as fake_manifest_fixture
from video_lab.adapter_registry import AdapterRegistry
from video_lab.adaptive_study_models import (
    AdaptiveStudyDefinition,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import AdaptiveStudyRepository
from video_lab.adaptive_study_service import AdaptiveStudyService
from video_lab.artifacts import ArtifactStore
from video_lab.cli import app
from video_lab.database import Database
from video_lab.domain import DisplayPreference, Manifest, utc_now
from video_lab.preference_optimizer import OptimizerConfig
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.services import ExperimentService

runner = CliRunner()


def _manifest() -> Manifest:
    return fake_manifest_fixture.__wrapped__()  # type: ignore[attr-defined, no-any-return]


def _common(tmp_path: Path, reviewer_id: str = "reviewer") -> list[str]:
    return [
        "--database",
        str(tmp_path / "lab.sqlite3"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--reviewer-id",
        reviewer_id,
    ]


def _write_manifest(tmp_path: Path, value: Manifest | None = None) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text((value or _manifest()).model_dump_json(), encoding="utf-8")
    return path


def _payload(result: Result) -> dict[str, object]:
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    assert result.stdout == json.dumps(payload, separators=(",", ":")) + "\n"
    return payload


def _service(tmp_path: Path, reviewer_id: str = "reviewer") -> AdaptiveStudyService:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    return AdaptiveStudyService(
        AdaptiveStudyRepository(database),
        ExperimentService(ExperimentRepository(database), JobRepository(database), artifacts),
        ReviewService(
            ExperimentRepository(database),
            JobRepository(database),
            artifacts,
            ReviewRepository(database),
        ),
        AdapterRegistry.with_defaults().capabilities("video-lab.fake.v1"),
        reviewer_id=reviewer_id,
    )


def test_study_create_show_and_list_are_stable_sealed_service_parity(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    created = runner.invoke(
        app, [*_common(tmp_path), "study", "create", "--manifest", str(manifest_path)]
    )

    assert created.exit_code == 0
    created_payload = _payload(created)
    assert created_payload["schema_version"] == "cli.v1"
    assert created_payload["ok"] is True
    created_data = created_payload["data"]
    assert isinstance(created_data, dict)
    study_id = created_data["study_id"]
    assert isinstance(study_id, str)
    assert created_data == _service(tmp_path).show(study_id).model_dump(mode="json")
    assert created_data["prompt"] == _manifest().prompt
    assert created_data["state"] == "generating_pair"
    assert created_data["next_action"] == "complete_generation"
    assert created_data["progress"] == {
        "completed_comparisons": 0,
        "current_comparison": 1,
        "maximum_comparisons": 10,
    }

    shown = runner.invoke(app, [*_common(tmp_path), "study", "show", study_id])
    listed = runner.invoke(app, [*_common(tmp_path), "study", "list"])

    assert shown.exit_code == listed.exit_code == 0
    assert _payload(shown)["data"] == created_data
    listed_data = _payload(listed)["data"]
    assert isinstance(listed_data, list) and len(listed_data) == 1
    assert listed_data == [
        item.model_dump(mode="json") for item in _service(tmp_path).list(limit=50)
    ]
    encoded_list = json.dumps(listed_data, sort_keys=True)
    for forbidden in (
        "prompt",
        "manifest",
        "guidance",
        "selected_pair",
        "display_mapping",
        "model_revision",
        "negative_prompt",
        _manifest().model_revision,
    ):
        assert forbidden not in encoded_list


def test_study_create_is_idempotent_per_reviewer_and_independent_across_reviewers(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)

    owner_first = runner.invoke(
        app,
        [*_common(tmp_path, "owner-a"), "study", "create", "--manifest", str(manifest_path)],
    )
    owner_retry = runner.invoke(
        app,
        [*_common(tmp_path, " owner-a "), "study", "create", "--manifest", str(manifest_path)],
    )
    independent = runner.invoke(
        app,
        [*_common(tmp_path, "owner-b"), "study", "create", "--manifest", str(manifest_path)],
    )

    assert owner_first.exit_code == owner_retry.exit_code == independent.exit_code == 0
    first_data = _payload(owner_first)["data"]
    retry_data = _payload(owner_retry)["data"]
    independent_data = _payload(independent)["data"]
    assert isinstance(first_data, dict)
    assert isinstance(retry_data, dict)
    assert isinstance(independent_data, dict)
    assert retry_data == first_data
    assert independent_data["study_id"] != first_data["study_id"]
    repository = AdaptiveStudyRepository(Database(tmp_path / "lab.sqlite3"))
    assert len(repository.list()) == 2
    assert {repository.owner(item.id) for item in repository.list()} == {"owner-a", "owner-b"}


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("a", DisplayPreference.LEFT),
        ("b", DisplayPreference.RIGHT),
        ("tie", DisplayPreference.TIE),
    ],
)
def test_study_resume_submit_and_reopen_use_a_b_display_choices(
    tmp_path: Path,
    choice: str,
    expected: DisplayPreference,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    created = _payload(
        runner.invoke(
            app, [*_common(tmp_path), "study", "create", "--manifest", str(manifest_path)]
        )
    )
    study_id = created["data"]["study_id"]  # type: ignore[index]
    assert isinstance(study_id, str)
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    complete_generation(database, tmp_path, study_id)

    resumed = runner.invoke(app, [*_common(tmp_path), "study", "resume", study_id])
    duplicate_resume = runner.invoke(app, [*_common(tmp_path), "study", "resume", study_id])

    assert resumed.exit_code == duplicate_resume.exit_code == 0
    resumed_data = _payload(resumed)["data"]
    assert _payload(duplicate_resume)["data"] == resumed_data
    assert isinstance(resumed_data, dict)
    review_id = resumed_data["review_id"]
    assert isinstance(review_id, str)
    assert resumed_data["prompt"] == _manifest().prompt
    assert resumed_data["state"] == "awaiting_preference"
    assert resumed_data["next_action"] == "record_preference"
    assert resumed_data["media"] == [
        {"choice": "a", "media_ref": f"adaptive-review:{review_id}:a"},
        {"choice": "b", "media_ref": f"adaptive-review:{review_id}:b"},
    ]
    sealed = json.dumps(resumed_data, sort_keys=True)
    for forbidden in (
        "guidance_scale",
        "selected_pair",
        "manifest",
        "acquisition",
        "optimizer_output_digest",
        "display_mapping",
        "model_revision",
        "negative_prompt",
        _manifest().model_revision,
    ):
        assert forbidden not in sealed

    submitted = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "submit",
            study_id,
            "--review-id",
            review_id,
            "--choice",
            choice,
        ],
    )
    retried = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "submit",
            study_id,
            "--review-id",
            review_id,
            "--choice",
            choice,
        ],
    )
    reopened = runner.invoke(app, [*_common(tmp_path), "study", "show", study_id])

    assert submitted.exit_code == retried.exit_code == reopened.exit_code == 0
    assert _payload(submitted)["data"] == _payload(retried)["data"]
    assert _payload(reopened)["data"] == _service(tmp_path).show(study_id).model_dump(mode="json")
    latest = _service(tmp_path).reviews.latest_adaptive(review_id, "reviewer")
    assert latest is not None and latest.display_preference is expected

    conflicting = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "submit",
            study_id,
            "--review-id",
            review_id,
            "--choice",
            "b" if choice != "b" else "a",
        ],
    )
    assert conflicting.exit_code == 2
    assert _payload(conflicting)["error"] == {
        "code": "invalid_study_state",
        "message": "adaptive study action is invalid in its current state",
    }


def test_study_commands_return_typed_sanitized_errors(tmp_path: Path) -> None:
    missing = runner.invoke(app, [*_common(tmp_path), "study", "show", "study_missing"])
    bad_choice = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "submit",
            "study_missing",
            "--review-id",
            "review_missing",
            "--choice",
            "left",
        ],
    )

    assert missing.exit_code == 3
    assert _payload(missing)["error"] == {
        "code": "study_not_found",
        "message": "adaptive study was not found",
    }
    assert bad_choice.exit_code == 2
    error = _payload(bad_choice)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "invalid_input"
    assert "left" not in error["message"]


def test_study_create_rejects_conflicting_manifest_overrides_and_adapter(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    prompt = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "create",
            "--manifest",
            str(manifest_path),
            "--prompt",
            "different prompt",
        ],
    )
    seed = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "create",
            "--manifest",
            str(manifest_path),
            "--seed",
            "99",
        ],
    )
    adapter = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "create",
            "--manifest",
            str(manifest_path),
            "--adapter",
            "diffusers.wan2.1-t2v-1.3b.mps.v1",
        ],
    )

    for result in (prompt, seed, adapter):
        assert result.exit_code == 2
        error = _payload(result)["error"]
        assert isinstance(error, dict)
        assert error["code"] == "invalid_study_input"


def test_study_create_validates_registered_adapter_capabilities_before_persistence(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path, _manifest().model_copy(update={"dtype": "float16"}))

    result = runner.invoke(
        app, [*_common(tmp_path), "study", "create", "--manifest", str(manifest_path)]
    )
    listed = runner.invoke(app, [*_common(tmp_path), "study", "list"])

    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": "manifest_unsupported",
        "message": "manifest is unsupported by the selected adapter",
    }
    assert listed.exit_code == 0
    assert _payload(listed)["data"] == []


def test_study_submit_rejects_wrong_owner_without_identity_leak(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    created = _payload(
        runner.invoke(
            app, [*_common(tmp_path), "study", "create", "--manifest", str(manifest_path)]
        )
    )
    study_id = created["data"]["study_id"]  # type: ignore[index]
    assert isinstance(study_id, str)
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    complete_generation(database, tmp_path, study_id)
    resumed = _payload(runner.invoke(app, [*_common(tmp_path), "study", "resume", study_id]))
    review_id = resumed["data"]["review_id"]  # type: ignore[index]
    assert isinstance(review_id, str)

    missing_review = runner.invoke(
        app,
        [
            *_common(tmp_path),
            "study",
            "submit",
            study_id,
            "--review-id",
            "review_missing",
            "--choice",
            "tie",
        ],
    )
    intruder_show = runner.invoke(
        app,
        [*_common(tmp_path, reviewer_id="intruder"), "study", "show", study_id],
    )
    intruder_resume = runner.invoke(
        app,
        [*_common(tmp_path, reviewer_id="intruder"), "study", "resume", study_id],
    )
    result = runner.invoke(
        app,
        [
            *_common(tmp_path, reviewer_id="intruder"),
            "study",
            "submit",
            study_id,
            "--review-id",
            review_id,
            "--choice",
            "tie",
        ],
    )

    assert missing_review.exit_code == 3
    assert _payload(missing_review)["error"] == {
        "code": "study_review_not_found",
        "message": "adaptive review was not found",
    }
    assert result.exit_code == 2
    assert _payload(result)["error"] == {
        "code": "study_review_not_owned",
        "message": "adaptive review operation is not permitted",
    }
    for denied in (intruder_show, intruder_resume):
        assert denied.exit_code == 2
        assert _payload(denied)["error"] == {
            "code": "study_review_not_owned",
            "message": "adaptive review operation is not permitted",
        }


def test_study_terminal_show_exposes_only_safe_posterior_summary(tmp_path: Path) -> None:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    service = _service(tmp_path)
    manifest = _manifest()
    config = OptimizerConfig(posterior_draws=256)
    definition = AdaptiveStudyDefinition(
        id="study_terminal",
        prompt=manifest.prompt,
        negative_prompt=manifest.negative_prompt,
        seed=manifest.seed,
        base_manifest=manifest,
        optimizer_config=config,
        input_fingerprint=adaptive_study_input_fingerprint(manifest, config),
        created_at=utc_now(),
    )
    terminal = service.create_study(definition)
    for _ in range(10):
        complete_generation(database, tmp_path, definition.id)
        awaiting = service.resume(definition.id)
        assert awaiting.review_id is not None
        terminal = service.record_preference(
            definition.id, awaiting.review_id, DisplayPreference.TIE
        )

    shown = runner.invoke(app, [*_common(tmp_path), "study", "show", definition.id])

    assert terminal.state.value == "search_budget_exhausted"
    assert shown.exit_code == 0
    data = _payload(shown)["data"]
    assert data == service.show(definition.id).model_dump(mode="json")
    assert isinstance(data, dict)
    assert data["stop_reason"] == "search_budget_exhausted"
    assert data["disposition"] == "inconclusive_within_budget"
    assert isinstance(data["recommendation"], float)
    assert isinstance(data["concentration_interval"], list)
    assert isinstance(data["concentration_probability"], float)
    serialized = json.dumps(data, sort_keys=True)
    for forbidden in (
        '"grid"',
        '"mean"',
        '"covariance"',
        "acquisition_ledger",
        "optimizer_output_digest",
        "content_digest",
        "display_mapping",
        "manifest",
    ):
        assert forbidden not in serialized
