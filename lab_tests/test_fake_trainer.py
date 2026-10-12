import hashlib
import json
from pathlib import Path

import pytest

from video_lab.trainer_protocol import (
    CalibrationRequest,
    CancellationToken,
    LoRAConfiguration,
    TrainerResultInvalid,
    TrainingCancelled,
    TrainingMaterialHandles,
    TrainingRequest,
    validate_cancelled_training_result,
    validate_training_result,
)
from video_lab.trainers.fake import FakeTrainerBackend, FakeTrainerFault


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _request(root: Path, **updates: object) -> TrainingRequest:
    values: dict[str, object] = {
        "request_id": "request-1",
        "run_id": "run-1",
        "backend_id": "fake",
        "materials": TrainingMaterialHandles(
            recipe_digest=_digest("1"),
            dataset_digest=_digest("2"),
            base_model_digest=_digest("3"),
            backend_digest=_digest("7"),
            source_digest=_digest("4"),
            runtime_digest=_digest("5"),
            material_digest=_digest("6"),
        ),
        "scratch_root": root,
        "total_steps": 8,
        "checkpoint_steps": (2, 4, 6, 8),
        "trainable_targets": ("transformer.attn.to_q",),
        "lora": LoRAConfiguration(rank=16, alpha=32, dropout=0.05),
        "seed": 7,
        "deadline_seconds": 60,
    }
    values.update(updates)
    return TrainingRequest.model_validate(values)


def test_fake_zero_step_calibration_emits_no_progress_and_returns_result(
    tmp_path: Path,
) -> None:
    training = _request(tmp_path)
    request = CalibrationRequest(
        request_id=training.request_id,
        run_id=training.run_id,
        backend_id=training.backend_id,
        materials=training.materials,
        scratch_root=tmp_path,
        steps=0,
        trainable_targets=training.trainable_targets,
        lora=training.lora,
        seed=training.seed,
        deadline_seconds=training.deadline_seconds,
    )
    progress = []

    result = FakeTrainerBackend().calibrate(request, progress.append, lambda: False)

    assert progress == []
    assert result.request_id == request.request_id
    assert result.finite_gradients is True
    assert result.frozen_parameters_unchanged is True


def _replace_evidence_payload(result, checkpoint_index: int, role: str, payload: object):
    checkpoint = result.checkpoints[checkpoint_index]
    target = next(item for item in checkpoint.files if item.role == role)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    (result.output_root / target.relative_path).write_bytes(data)
    changed = target.model_copy(
        update={
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    )
    files = tuple(changed if item.role == role else item for item in checkpoint.files)
    checkpoints = list(result.checkpoints)
    checkpoints[checkpoint_index] = checkpoint.model_copy(update={"files": files})
    return result.model_copy(update={"checkpoints": tuple(checkpoints)})


def test_fake_training_is_byte_deterministic(tmp_path: Path) -> None:
    progress: list[int] = []
    left = FakeTrainerBackend().train(
        _request(tmp_path / "left"), lambda item: progress.append(item.step), lambda: False
    )
    right = FakeTrainerBackend().train(
        _request(tmp_path / "right"), lambda item: None, lambda: False
    )
    assert left.model_dump(mode="json", exclude={"output_root"}) == right.model_dump(
        mode="json", exclude={"output_root"}
    )
    assert progress == list(range(1, 9))
    validate_training_result(_request(tmp_path / "left"), left)
    for checkpoint in left.checkpoints:
        weights = checkpoint.files[0]
        data = (left.output_root / weights.relative_path).read_bytes()
        assert data[:1] != b"{" and hashlib.sha256(data).hexdigest() in weights.digest

    telemetry = next(item for item in left.checkpoints[0].files if item.role == "telemetry")
    records = json.loads((left.output_root / telemetry.relative_path).read_bytes())
    assert records[0] == {
        "payload": {
            "checkpoint_boundary": False,
            "fraction": 0.125,
            "gradient_norm": 1.0,
            "loss": 1 / 9,
            "step": 1,
            "total_steps": 8,
        },
        "sequence": 1,
        "step": 1,
    }


def test_fake_checkpoint_weights_are_deterministic_and_distinct_per_step(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    digests = tuple(
        next(item.digest for item in checkpoint.files if item.role == "weights")
        for checkpoint in result.checkpoints
    )
    assert len(set(digests)) == len(result.checkpoints)


def test_fake_observes_cancellation_only_at_checkpoint_boundary(tmp_path: Path) -> None:
    token = CancellationToken()
    seen: list[int] = []

    def progress(item: object) -> None:
        step = int(item.step)
        seen.append(step)
        if step == 3:
            token.request()

    with pytest.raises(TrainingCancelled) as error:
        FakeTrainerBackend().train(_request(tmp_path), progress, token.cancelled)
    assert error.value.boundary_step == 4
    assert seen == [1, 2, 3, 4]
    assert error.value.result is not None
    assert [item.step for item in error.value.result.checkpoints] == [2, 4]
    validate_cancelled_training_result(_request(tmp_path), error.value.result, 4)


def test_cancelled_result_gets_full_digest_and_identity_validation(tmp_path: Path) -> None:
    token = CancellationToken()

    def progress(item: object) -> None:
        if int(item.step) == 3:
            token.request()

    with pytest.raises(TrainingCancelled) as caught:
        FakeTrainerBackend().train(_request(tmp_path), progress, token.cancelled)
    result = caught.value.result
    assert result is not None
    bad = result.model_copy(update={"changed_targets": ("undeclared",)})
    with pytest.raises(TrainerResultInvalid):
        validate_cancelled_training_result(_request(tmp_path), bad, 4)


def test_result_rejects_noncanonical_and_aliased_output_paths(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    first = result.checkpoints[0]
    files = list(first.files)
    files[0] = files[0].model_copy(update={"relative_path": "./" + files[0].relative_path})
    bad = result.model_copy(
        update={
            "checkpoints": (first.model_copy(update={"files": tuple(files)}),)
            + result.checkpoints[1:]
        }
    )
    with pytest.raises(TrainerResultInvalid, match="trainer_output_unsafe"):
        validate_training_result(request, bad)


def test_result_rejects_noncanonical_json_evidence(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    checkpoint = result.checkpoints[0]
    target = next(item for item in checkpoint.files if item.role == "optimizer")
    path = result.output_root / target.relative_path
    path.write_bytes(path.read_bytes() + b" ")
    changed = target.model_copy(
        update={
            "size_bytes": path.stat().st_size,
            "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    files = tuple(changed if item.role == "optimizer" else item for item in checkpoint.files)
    bad = result.model_copy(
        update={
            "checkpoints": (checkpoint.model_copy(update={"files": files}),)
            + result.checkpoints[1:]
        }
    )
    with pytest.raises(TrainerResultInvalid, match="trainer_evidence_payload_invalid"):
        validate_training_result(request, bad)


@pytest.mark.parametrize(
    "role",
    [
        "weights",
        "optimizer",
        "trainer_state",
        "trainable_targets",
        "frozen_parameters",
        "finite_gradient_proof",
    ],
)
def test_result_rejects_scientifically_impossible_role_evidence(
    tmp_path: Path, role: str
) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    checkpoint = result.checkpoints[0]
    target = next(item for item in checkpoint.files if item.role == role)
    path = result.output_root / target.relative_path
    if role == "weights":
        header = json.dumps(
            {"weight": {"dtype": "F32", "shape": [999], "data_offsets": [0, 1]}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        header += b" " * ((-len(header)) % 8)
        payload = len(header).to_bytes(8, "little") + header + b"x"
    else:
        payload = json.dumps(
            {
                "request": request.model_dump(mode="json", exclude={"scratch_root"}),
                "role": role,
                "step": checkpoint.step,
                "telemetry": None,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    path.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    changed = target.model_copy(update={"size_bytes": len(payload), "digest": digest})
    files = tuple(changed if item.role == role else item for item in checkpoint.files)
    updates: dict[str, object] = {"files": files}
    summary_fields = {
        "trainable_targets": "trainable_targets_digest",
        "frozen_parameters": "frozen_parameters_digest",
        "finite_gradient_proof": "finite_gradient_proof_digest",
    }
    if role in summary_fields:
        updates[summary_fields[role]] = digest
    bad_checkpoint = checkpoint.model_copy(update=updates)
    bad = result.model_copy(
        update={"checkpoints": (bad_checkpoint,) + result.checkpoints[1:]}
    )
    with pytest.raises(TrainerResultInvalid):
        validate_training_result(request, bad)


def test_fake_resume_binds_parent_and_global_telemetry_sequence(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        checkpoint_steps=(6, 8),
        start_step=4,
        telemetry_sequence_start=5,
        parent_checkpoint_id="run-1-cp-parent",
        parent_checkpoint_digest=_digest("8"),
        optimizer_state_digest=_digest("9"),
    )
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    validate_training_result(request, result)
    assert result.checkpoints[0].parent_checkpoint_id == "run-1-cp-parent"
    telemetry = next(item for item in result.checkpoints[0].files if item.role == "telemetry")
    records = json.loads((result.output_root / telemetry.relative_path).read_bytes())
    assert [item["sequence"] for item in records] == [5, 6]
    assert [item["step"] for item in records] == [5, 6]


def test_non_adam_optimizer_configuration_is_bound_and_validated(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        optimizer={
            "algorithm": "sgd",
            "parameters": [
                {"name": "learning_rate", "value": 0.1},
                {"name": "momentum", "value": 0.9},
            ],
        },
    )
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    validate_training_result(request, result)
    optimizer = next(item for item in result.checkpoints[0].files if item.role == "optimizer")
    payload = json.loads((result.output_root / optimizer.relative_path).read_bytes())
    assert payload["algorithm"] == "sgd"
    assert payload["config"] == {"learning_rate": 0.1, "momentum": 0.9}


def test_metadata_only_optimizer_state_is_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    checkpoint = result.checkpoints[0]
    target = next(item for item in checkpoint.files if item.role == "optimizer")
    path = result.output_root / target.relative_path
    data = json.dumps(
        {
            "schema_version": "trainer-optimizer-state.v1",
            "request_id": request.request_id,
            "run_id": request.run_id,
            "step": checkpoint.step,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(data)
    changed = target.model_copy(
        update={
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    )
    files = tuple(changed if item.role == "optimizer" else item for item in checkpoint.files)
    changed_checkpoint = checkpoint.model_copy(update={"files": files})
    changed_result = result.model_copy(
        update={"checkpoints": (changed_checkpoint,) + result.checkpoints[1:]}
    )
    with pytest.raises(TrainerResultInvalid, match="trainer_evidence_payload_invalid"):
        validate_training_result(request, changed_result)


def test_legitimate_namespaced_backend_optimizer_state_is_accepted(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    checkpoint = result.checkpoints[0]
    target = next(item for item in checkpoint.files if item.role == "optimizer")
    path = result.output_root / target.relative_path
    payload = {
        "schema_version": "trainer-optimizer-state.v2",
        "request_id": request.request_id,
        "run_id": request.run_id,
        "step": checkpoint.step,
        "algorithm": "adamw",
        "config": {"learning_rate": 0.001},
        "global_step": checkpoint.step,
        "parameter_groups": [
            {
                "group_id": "default",
                "parameters": list(request.trainable_targets),
                "hyperparameters": {"learning_rate": 0.001},
            }
        ],
        "parameter_state": {
            request.trainable_targets[0]: {
                "step": checkpoint.step,
                "slots": {"first_moment": 0.25, "second_moment": 0.125},
            }
        },
        "backend_state": {"fake": {"restored_updates": checkpoint.step}},
        "parent_optimizer_state_digest": request.optimizer_state_digest,
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(data)
    changed = target.model_copy(
        update={
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    )
    files = tuple(changed if item.role == "optimizer" else item for item in checkpoint.files)
    changed_checkpoint = checkpoint.model_copy(update={"files": files})
    changed_result = result.model_copy(
        update={"checkpoints": (changed_checkpoint,) + result.checkpoints[1:]}
    )
    validate_training_result(request, changed_result)


@pytest.mark.parametrize(
    ("role", "mutate"),
    [
        (
            "optimizer",
            lambda payload: (
                payload["parameter_state"].__setitem__(
                    "transformer.attn.to_q", {"step": 2, "slots": {"present": True}}
                ),
                payload.__setitem__("backend_state", {"fake": {"present": True}}),
            ),
        ),
        (
            "trainer_state",
            lambda payload: payload.__setitem__(
                "backend_state", {"fake": {"present": True}}
            ),
        ),
    ],
)
def test_boolean_only_checkpoint_state_is_rejected(
    tmp_path: Path, role: str, mutate
) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    checkpoint = result.checkpoints[0]
    target = next(item for item in checkpoint.files if item.role == role)
    payload = json.loads((result.output_root / target.relative_path).read_bytes())
    mutate(payload)
    changed = _replace_evidence_payload(result, 0, role, payload)
    with pytest.raises(TrainerResultInvalid, match="trainer_evidence_payload_invalid"):
        validate_training_result(request, changed)


@pytest.mark.parametrize("role", ["optimizer", "trainer_state"])
@pytest.mark.parametrize(
    "backend_state",
    [
        {"foreign": {"restored_updates": 2}},
        {"fake": {"restored_updates": 2}, "foreign": {"restored_updates": 2}},
        {},
    ],
    ids=["foreign", "multiple", "missing"],
)
def test_checkpoint_state_rejects_invalid_backend_namespace(
    tmp_path: Path, role: str, backend_state: dict[str, object]
) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    checkpoint = result.checkpoints[0]
    target = next(item for item in checkpoint.files if item.role == role)
    payload = json.loads((result.output_root / target.relative_path).read_bytes())
    payload["backend_state"] = backend_state
    changed = _replace_evidence_payload(result, 0, role, payload)
    with pytest.raises(TrainerResultInvalid, match="trainer_evidence_payload_invalid"):
        validate_training_result(request, changed)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_optimizer_slots_reject_nonfinite_numeric_state(
    tmp_path: Path, nonfinite: float
) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    checkpoint = result.checkpoints[0]
    target = next(item for item in checkpoint.files if item.role == "optimizer")
    payload = json.loads((result.output_root / target.relative_path).read_bytes())
    payload["parameter_state"][request.trainable_targets[0]]["slots"] = {"moment": nonfinite}
    changed = _replace_evidence_payload(result, 0, "optimizer", payload)
    with pytest.raises(TrainerResultInvalid, match="trainer_evidence_payload_invalid"):
        validate_training_result(request, changed)


def test_numeric_zero_is_substantive_checkpoint_state(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    for role in ("optimizer", "trainer_state"):
        checkpoint = result.checkpoints[0]
        target = next(item for item in checkpoint.files if item.role == role)
        payload = json.loads((result.output_root / target.relative_path).read_bytes())
        payload["backend_state"] = {request.backend_id: {"restored_updates": 0}}
        if role == "optimizer":
            payload["parameter_state"][request.trainable_targets[0]]["slots"] = {"moment": 0}
        result = _replace_evidence_payload(result, 0, role, payload)
    validate_training_result(request, result)


@pytest.mark.parametrize(
    "fault",
    [
        FakeTrainerFault.NONFINITE_GRADIENT,
        FakeTrainerFault.FROZEN_MUTATION,
        FakeTrainerFault.CORRUPT_CHECKPOINT,
        FakeTrainerFault.MISSING_CHECKPOINT,
        FakeTrainerFault.EXTRA_CHECKPOINT,
        FakeTrainerFault.WRONG_STEP,
        FakeTrainerFault.WRONG_FRACTION,
        FakeTrainerFault.WRONG_PARENT,
        FakeTrainerFault.WRONG_TARGET,
        FakeTrainerFault.DEADLINE,
        FakeTrainerFault.CRASH,
    ],
)
def test_fake_supports_adversarial_faults(tmp_path: Path, fault: FakeTrainerFault) -> None:
    backend = FakeTrainerBackend(fault=fault)
    if fault == FakeTrainerFault.CRASH:
        with pytest.raises(RuntimeError, match="fake_trainer_crash"):
            backend.train(_request(tmp_path), lambda item: None, lambda: False)
    else:
        result = backend.train(_request(tmp_path), lambda item: None, lambda: False)
        assert result.fault == fault
        with pytest.raises(TrainerResultInvalid):
            validate_training_result(_request(tmp_path), result)


def test_fake_rejects_symlink_scratch_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe_scratch_root"):
        FakeTrainerBackend().train(_request(link), lambda item: None, lambda: False)


def test_result_validator_rejects_backend_digest_identity_mutation(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeTrainerBackend().train(request, lambda item: None, lambda: False)
    with pytest.raises(TrainerResultInvalid, match="trainer_result_identity_invalid"):
        validate_training_result(
            request, result.model_copy(update={"backend_digest": _digest("0")})
        )
