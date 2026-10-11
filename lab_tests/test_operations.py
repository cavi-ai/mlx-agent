import subprocess
import sys

import pytest
from pydantic import ValidationError

from lab_tests.test_capabilities import manifest
from video_lab.adapter_registry import AdapterRegistry, UnknownAdapterError
from video_lab.comparison import ComparisonError
from video_lab.database import Database
from video_lab.operations import LaboratoryOperations
from video_lab.services import ExperimentService, ValidationFailure


def operations(database: Database, tmp_path: object) -> LaboratoryOperations:
    return LaboratoryOperations(
        AdapterRegistry.with_defaults(),
        ExperimentService.from_paths(database.path, tmp_path / "artifacts"),  # type: ignore[operator]
    )


def draft(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "baseline": manifest()
        .model_copy(
            update={
                "adapter_id": "video-lab.fake.v1",
                "backend": "cpu",
                "dtype": "uint8",
                "scheduler": "fake",
                "num_frames": 17,
            }
        )
        .model_dump(mode="json"),
        "variable_path": "inference_steps",
        "alternate_value": "8",
    }
    payload.update(updates)
    return payload


def test_catalog_is_lazy_json_safe_and_guides_manifest_fields() -> None:
    code = """
import json, sys
from video_lab.adapter_registry import AdapterRegistry
from video_lab.operations import LaboratoryOperations
catalog = LaboratoryOperations.catalog_for(AdapterRegistry.with_defaults())
assert [entry.adapter_id for entry in catalog] == [
    'video-lab.fake.v1',
    'diffusers.wan2.1-t2v-1.3b.mps.v1',
    'diffusers.animegen-t2v.wan2.2-a14b.mps.v1',
]
json.dumps([entry.model_dump(mode='json') for entry in catalog])
candidate = catalog[-1]
assert candidate.generation_enabled is False
assert candidate.source_model == 'aidealab/AnimeGen-T2V'
assert candidate.frame_rule == '16n+1'
assert candidate.declared_profiles == ('minimum', 'standard-512x288-17f-8fps')
assert any(
    field.path == 'inference_steps' and field.value_type == 'integer'
    for field in catalog[0].fields
)
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
assert not any(name == 'diffusers' or name.startswith('diffusers.') for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_catalog_expresses_exact_frame_count_rule() -> None:
    catalog = LaboratoryOperations.catalog_for(AdapterRegistry.with_defaults())
    entries = {entry.adapter_id: entry for entry in catalog}
    for adapter_id in (
        "video-lab.fake.v1",
        "diffusers.wan2.1-t2v-1.3b.mps.v1",
    ):
        frames = next(field for field in entries[adapter_id].fields if field.path == "num_frames")
        assert frames.rule == "16n+1"
        assert (17 - 1) % 16 == 0
        assert (33 - 1) % 16 == 0
        assert (18 - 1) % 16 != 0


def test_preview_canonicalizes_values_and_never_writes(
    database: Database, tmp_path: object
) -> None:
    subject = operations(database, tmp_path)
    preview = subject.preview(draft(), "video-lab.fake.v1")

    assert preview.variant_a.inference_steps == manifest().inference_steps
    assert preview.variant_b.inference_steps == 8
    assert preview.changed_paths == ("inference_steps",)
    assert subject.service.list() == ()


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"variable_path": "missing"}, ComparisonError),
        ({"variable_path": "backend", "alternate_value": "mps"}, ValidationFailure),
        ({"variable_path": "inference_steps", "alternate_value": "not-an-int"}, ComparisonError),
    ],
)
def test_preview_rejects_invalid_pairs_without_writes(
    database: Database, tmp_path: object, payload: dict[str, object], error: type[Exception]
) -> None:
    subject = operations(database, tmp_path)
    with pytest.raises(error):
        subject.preview(draft(**payload), "video-lab.fake.v1")
    assert subject.service.list() == ()


def test_preview_rejects_unknown_adapter_and_invalid_baseline(
    database: Database, tmp_path: object
) -> None:
    subject = operations(database, tmp_path)
    with pytest.raises(UnknownAdapterError):
        subject.preview(draft(), "missing")
    with pytest.raises(ValidationError):
        subject.preview(draft(baseline={}), "video-lab.fake.v1")
    assert subject.service.list() == ()


def test_create_persists_one_experiment_two_variants_and_two_queued_jobs(
    database: Database, tmp_path: object
) -> None:
    subject = operations(database, tmp_path)
    view = subject.create(draft(), "video-lab.fake.v1")

    assert len(subject.service.list()) == 1
    assert len(view.experiment.variants) == 2
    assert [job.state.value for job in view.jobs] == ["queued", "queued"]


def test_failed_create_persists_zero_rows(database: Database, tmp_path: object) -> None:
    subject = operations(database, tmp_path)
    with pytest.raises(ValidationFailure):
        subject.create(draft(variable_path="backend", alternate_value="mps"), "video-lab.fake.v1")

    with database.connect() as connection:
        for table in ("experiments", "variants", "jobs"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
