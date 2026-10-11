from pathlib import Path

import pytest
from pydantic import ValidationError

import video_lab.trainer_protocol as trainer_protocol
from video_lab.trainer_protocol import (
    CalibrationRequest,
    CancellationToken,
    ProbeTrainingRequest,
    TrainerBackend,
    TrainerCapabilityReport,
    TrainingMaterialHandles,
    TrainingProgress,
    TrainingRequest,
)
from video_lab.trainers.fake import FakeTrainerBackend

DIGEST = "sha256:" + "1" * 64


def test_protocol_models_are_frozen_and_reject_extra_fields(tmp_path: Path) -> None:
    materials = TrainingMaterialHandles(
        recipe_digest=DIGEST,
        dataset_digest=DIGEST,
        base_model_digest=DIGEST,
        backend_digest=DIGEST,
        source_digest=DIGEST,
        runtime_digest=DIGEST,
        material_digest=DIGEST,
    )
    with pytest.raises(ValidationError):
        materials.recipe_digest = "sha256:" + "2" * 64
    with pytest.raises(ValidationError):
        TrainingProgress(step=1, total_steps=4, fraction=0.25, loss=1.0, unexpected=True)
    request = ProbeTrainingRequest(
        request_id="probe-1",
        backend_id="fake",
        expected_backend_digest=DIGEST,
        materials=materials,
        scratch_root=tmp_path,
    )
    assert request.scratch_root == tmp_path


def test_progress_rejects_nonfinite_and_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        TrainingProgress(step=5, total_steps=4, fraction=1.0, loss=1.0)
    with pytest.raises(ValidationError):
        TrainingProgress(step=1, total_steps=4, fraction=0.25, loss=float("nan"))


def test_backend_protocol_is_runtime_checkable() -> None:
    assert isinstance(FakeTrainerBackend(), TrainerBackend)
    assert TrainerCapabilityReport(
        backend_id="fake",
        backend_digest=DIGEST,
        runtime_digest=DIGEST,
        supported=True,
        max_steps=100,
        safe_boundaries=(25, 50, 75, 100),
    ).supported
    token = CancellationToken()
    assert not token.cancelled()
    token.request()
    assert token.cancelled()


def test_lora_configuration_is_required_exact_and_strictly_validated(tmp_path: Path) -> None:
    configuration_type = getattr(trainer_protocol, "LoRAConfiguration", None)
    assert configuration_type is not None, "LoRAConfiguration is absent"
    materials = TrainingMaterialHandles(
        recipe_digest=DIGEST,
        dataset_digest=DIGEST,
        base_model_digest=DIGEST,
        backend_digest=DIGEST,
        source_digest=DIGEST,
        runtime_digest=DIGEST,
        material_digest=DIGEST,
    )
    lora = configuration_type(rank=16, alpha=32, dropout=0.05)
    shared = {
        "request_id": "request-1",
        "run_id": "run-1",
        "backend_id": "fake",
        "materials": materials,
        "scratch_root": tmp_path,
        "trainable_targets": ("transformer.attn.to_q",),
        "seed": 7,
        "deadline_seconds": 60,
    }

    calibration = CalibrationRequest(**shared, steps=1, lora=lora)
    training = TrainingRequest(
        **shared,
        total_steps=4,
        checkpoint_steps=(1, 2, 3, 4),
        lora=lora,
    )

    assert calibration.lora == training.lora == lora
    assert calibration.model_dump(mode="json")["lora"] == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    }
    for request_type, specific in (
        (CalibrationRequest, {"steps": 1}),
        (
            TrainingRequest,
            {"total_steps": 4, "checkpoint_steps": (1, 2, 3, 4)},
        ),
    ):
        with pytest.raises(ValidationError, match="lora"):
            request_type(**shared, **specific)
    for values in (
        {"rank": True, "alpha": 32, "dropout": 0.05},
        {"rank": 16, "alpha": 0, "dropout": 0.05},
        {"rank": 16, "alpha": 32, "dropout": float("nan")},
        {"rank": 16, "alpha": 32, "dropout": 1.0},
    ):
        with pytest.raises(ValidationError):
            configuration_type(**values)


def test_calibration_request_accepts_exact_zero_steps_and_rejects_negative(
    tmp_path: Path,
) -> None:
    values = {
        "request_id": "request-1",
        "run_id": "run-1",
        "backend_id": "fake",
        "materials": TrainingMaterialHandles(
            recipe_digest=DIGEST,
            dataset_digest=DIGEST,
            base_model_digest=DIGEST,
            backend_digest=DIGEST,
            source_digest=DIGEST,
            runtime_digest=DIGEST,
            material_digest=DIGEST,
        ),
        "scratch_root": tmp_path,
        "steps": 0,
        "trainable_targets": ("transformer.attn.to_q",),
        "lora": trainer_protocol.LoRAConfiguration(rank=16, alpha=32, dropout=0.05),
        "seed": 7,
        "deadline_seconds": 60,
    }

    assert CalibrationRequest.model_validate(values).steps == 0
    with pytest.raises(ValidationError, match="steps"):
        CalibrationRequest.model_validate({**values, "steps": -1})
