import sqlite3
from pathlib import Path

import pytest

from lab_tests.test_capabilities import manifest
from video_lab.capabilities import AdapterCapabilities
from video_lab.database import Database
from video_lab.domain import ExperimentDraft, JobState
from video_lab.services import ExperimentService, ValidationFailure


def capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_id="test",
        model_patterns=("org/*",),
        backends=("mps",),
        dtypes=("float16",),
        quantizations=("none",),
        schedulers=("euler",),
        min_width=256,
        max_width=1024,
        width_multiple=16,
        min_height=256,
        max_height=1024,
        height_multiple=16,
        min_frames=1,
        max_frames=81,
        frame_rule="16n+1",
        supports_lora=False,
    )


def test_creates_validated_experiment_with_two_jobs(database: Database, tmp_path: Path) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")
    view = service.create(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8),
        capabilities(),
    )

    assert view.experiment.changed_paths == ("inference_steps",)
    assert len(view.jobs) == 2
    assert service.get(view.experiment.id) == view
    assert service.list() == (view,)

    transitioned = service.transition_job(view.jobs[0].id, JobState.PREPARING)
    assert transitioned.state is JobState.PREPARING
    assert service.get(view.experiment.id).jobs[0] == transitioned


def test_rejects_unsupported_baseline_before_persistence(
    database: Database, tmp_path: Path
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")

    with pytest.raises(ValidationFailure) as error:
        service.create(
            ExperimentDraft(
                baseline=manifest().model_copy(update={"dtype": "bfloat16"}),
                variable_path="inference_steps",
                alternate_value=8,
            ),
            capabilities(),
        )

    assert error.value.violations[0].field == "dtype"
    assert str(error.value) == "manifest is not supported by the selected adapter"
    assert service.list() == ()


def test_rejects_mismatched_adapter_before_persistence(database: Database, tmp_path: Path) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")
    selected = capabilities().model_copy(update={"adapter_id": "other"})

    with pytest.raises(ValidationFailure) as error:
        service.create(
            ExperimentDraft(
                baseline=manifest(), variable_path="inference_steps", alternate_value=8
            ),
            selected,
        )

    assert error.value.violations[0].field == "adapter_id"
    assert service.list() == ()


def test_rejects_unsupported_alternate_before_persistence(
    database: Database, tmp_path: Path
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")

    with pytest.raises(ValidationFailure) as error:
        service.create(
            ExperimentDraft(
                baseline=manifest(),
                variable_path="scheduler",
                alternate_value="unsupported",
            ),
            capabilities(),
        )

    assert error.value.violations[0].field == "scheduler"
    assert service.list() == ()


def test_rolls_back_entire_aggregate_when_second_job_insert_fails(
    database: Database, tmp_path: Path
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")
    with database.connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_second_job BEFORE INSERT ON jobs "
            "WHEN (SELECT label FROM variants WHERE id = NEW.variant_id) = 'B' "
            "BEGIN SELECT RAISE(ABORT, 'forced second job failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced second job failure"):
        service.create(
            ExperimentDraft(
                baseline=manifest(), variable_path="inference_steps", alternate_value=8
            ),
            capabilities(),
        )

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM variants").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_from_paths_migrates_database_and_configures_artifact_root(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "lab.sqlite3"
    artifact_root = tmp_path / "artifacts"

    service = ExperimentService.from_paths(database_path, artifact_root)

    assert service.list() == ()
    assert service.artifacts.root == artifact_root.resolve()
