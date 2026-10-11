from __future__ import annotations

import errno
import hashlib
import json
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import video_lab.trainers.wan_checkpoint as wan_checkpoint
from video_lab.domain import canonical_json
from video_lab.trainer_protocol import (
    CheckpointResult,
    LoRAConfiguration,
    OptimizerConfiguration,
    OptimizerParameter,
    TrainingMaterialHandles,
    TrainingRequest,
    TrainingResult,
    validate_training_result,
)
from video_lab.trainers.wan_checkpoint import (
    WanCheckpointCleanupError,
    WanCheckpointCodec,
    WanCheckpointState,
    WanCheckpointValidationContext,
    WanParentCheckpointDescriptor,
    WanParentCheckpointFileDescriptor,
)
from video_lab.wan_pending import TrainableParameterSpec, trainable_parameter_spec_records

BACKEND_ID = "diffusers.wan2.2-animegen-lora.mps.v1"


def _physical_specs() -> tuple[TrainableParameterSpec, ...]:
    return (
        TrainableParameterSpec(
            "transformer.attn.to_q.lora_A.default.weight",
            "transformer.attn.to_q", "A", (4, 2), "bfloat16", "mps",
        ),
        TrainableParameterSpec(
            "transformer.attn.to_q.lora_B.default.weight",
            "transformer.attn.to_q", "B", (2, 4), "bfloat16", "mps",
        ),
    )


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _rng_states(seed: int, step: int) -> dict[str, object]:
    states: dict[str, object] = {}
    for device in ("cpu", "mps"):
        state = hashlib.sha256(f"{device}:{seed}:{step}".encode()).digest()
        states[device] = {
            "algorithm": "torch.Generator",
            "device": device,
            "state_hex": state.hex(),
            "digest": _sha(state),
        }
    return states


def _request(root: Path) -> TrainingRequest:
    digest = _sha(b"identity")
    return TrainingRequest(
        request_id="request-1",
        run_id="run-1",
        backend_id=BACKEND_ID,
        materials=TrainingMaterialHandles(
            recipe_digest=digest,
            dataset_digest=digest,
            base_model_digest=digest,
            backend_digest=digest,
            source_digest=digest,
            runtime_digest=digest,
            material_digest=digest,
        ),
        scratch_root=root,
        total_steps=4,
        checkpoint_steps=(4,),
        trainable_targets=("transformer.attn.to_q",),
        lora=LoRAConfiguration(rank=4, alpha=8, dropout=0.0),
        seed=11,
        deadline_seconds=60,
        optimizer=OptimizerConfiguration(
            algorithm="adamw",
            parameters=(OptimizerParameter(name="learning_rate", value=0.001),),
        ),
    )


def _safetensors(
    tensors: tuple[tuple[str, str, tuple[int, ...]], ...] | None = None,
) -> bytes:
    specs = tensors or (
        ("transformer.attn.to_q.lora_A.weight", "F32", (4, 2)),
        ("transformer.attn.to_q.lora_B.weight", "F32", (2, 4)),
    )
    widths = {"U8": 1, "F16": 2, "BF16": 2, "F32": 4}
    header_value: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape in sorted(specs):
        start = len(payload)
        elements = 1
        for dimension in shape:
            elements *= dimension
        if dtype == "F32":
            payload.extend(
                struct.pack(f"<{elements}f", *(float(index + 1) for index in range(elements)))
            )
        else:
            payload.extend(bytes((index % 251) + 1 for index in range(elements * widths[dtype])))
        header_value[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    header = json.dumps(
        header_value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    header += b" " * ((-len(header)) % 8)
    return len(header).to_bytes(8, "little") + header + bytes(payload)


def _state(request: TrainingRequest, *, weights: bytes | None = None) -> WanCheckpointState:
    config = {"learning_rate": 0.001}
    progress = {
        "step": 4,
        "total_steps": 4,
        "fraction": 1.0,
        "loss": 0.25,
        "gradient_norm": 1.5,
        "checkpoint_boundary": True,
    }
    return WanCheckpointState(
        checkpoint_id="checkpoint-1",
        step=4,
        total_steps=4,
        parent_checkpoint_id=None,
        weights=weights if weights is not None else _safetensors(),
        optimizer={
            "schema_version": "trainer-optimizer-state.v2",
            "request_id": "request-1",
            "run_id": "run-1",
            "step": 4,
            "algorithm": "adamw",
            "config": config,
            "global_step": 4,
            "parameter_groups": [
                {
                    "group_id": "default",
                    "parameters": ["transformer.attn.to_q"],
                    "hyperparameters": config,
                }
            ],
            "parameter_state": {
                "transformer.attn.to_q": {
                    "step": 4,
                    "slots": {"exp_avg_norm": 0.2, "exp_avg_sq_norm": 0.04},
                }
            },
            "backend_state": {BACKEND_ID: {"optimizer_updates": 4}},
            "parent_optimizer_state_digest": None,
        },
        trainer_state={
            "schema_version": "trainer-state.v3",
            "request_id": "request-1",
            "run_id": "run-1",
            "step": 4,
            "backend_id": BACKEND_ID,
            "seed": 11,
            "total_steps": 4,
            "global_step": 4,
            "rng_states": _rng_states(11, 4),
            "scheduler_state": {"step": 4, "learning_rate_scale": 1.0},
            "scaler_state": {"enabled": False, "scale": 1.0},
            "backend_state": {BACKEND_ID: {"data_cursor": 4}},
        },
        telemetry=(
            {
                "sequence": 1,
                "step": 1,
                "payload": {**progress, "step": 1, "fraction": 0.25, "checkpoint_boundary": False},
            },
            {
                "sequence": 2,
                "step": 2,
                "payload": {**progress, "step": 2, "fraction": 0.5, "checkpoint_boundary": False},
            },
            {
                "sequence": 3,
                "step": 3,
                "payload": {**progress, "step": 3, "fraction": 0.75, "checkpoint_boundary": False},
            },
            {"sequence": 4, "step": 4, "payload": progress},
        ),
        trainable_targets={
            "schema_version": "trainer-trainable-targets.v2",
            "request_id": "request-1",
            "run_id": "run-1",
            "step": 4,
            "targets": ["transformer.attn.to_q"],
            "physical_parameters": trainable_parameter_spec_records(_physical_specs()),
        },
        frozen_parameters={
            "schema_version": "trainer-frozen-parameters.v1",
            "request_id": "request-1",
            "run_id": "run-1",
            "step": 4,
            "trainable_targets": ["transformer.attn.to_q"],
            "unchanged": True,
        },
        finite_gradient_proof={
            "schema_version": "trainer-finite-gradient-proof.v1",
            "request_id": "request-1",
            "run_id": "run-1",
            "step": 4,
            "finite": True,
            "loss": 0.25,
            "gradient_norm": 1.5,
        },
    )


def _context(request: TrainingRequest) -> WanCheckpointValidationContext:
    return WanCheckpointValidationContext(
        checkpoint_id="checkpoint-1",
        request_id=request.request_id,
        run_id=request.run_id,
        backend_id=request.backend_id,
        seed=request.seed,
        step=request.total_steps,
        total_steps=request.total_steps,
        parent_checkpoint_id=request.parent_checkpoint_id,
        parent_optimizer_state_digest=request.optimizer_state_digest,
        trainable_targets=request.trainable_targets,
        trainable_parameter_specs=_physical_specs(),
        lora_rank=request.lora.rank,
        lora_alpha=request.lora.alpha,
        lora_dropout=request.lora.dropout,
        optimizer_algorithm=request.optimizer.algorithm,
        optimizer_config=tuple(
            (item.name, item.value) for item in request.optimizer.parameters
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "dtype", "device", "span"),
)
def test_pending_gradient_validator_rejects_non_exact_parameter_inventory_and_payload(
    mutation: str,
) -> None:
    payload = b"\x00" * 8
    record: dict[str, object] = {
        "accumulation_position": 2,
        "parameters": {
            "transformer.attn.to_q": {
                "digest": _sha(payload),
                "serialized_b64": __import__("base64").b64encode(payload).decode(),
                "shape": [2, 2],
                "dtype": "bfloat16",
                "device": "mps",
            }
        },
    }
    parameters = record["parameters"]
    assert isinstance(parameters, dict)
    if mutation == "missing":
        parameters.clear()
    elif mutation == "extra":
        parameters["rogue"] = dict(parameters["transformer.attn.to_q"])
    elif mutation == "dtype":
        parameters["transformer.attn.to_q"]["dtype"] = "float16"  # type: ignore[index]
    elif mutation == "device":
        parameters["transformer.attn.to_q"]["device"] = "cpu"  # type: ignore[index]
    else:
        parameters["transformer.attn.to_q"]["shape"] = [1, 4]  # type: ignore[index]
    with pytest.raises(ValueError):
        wan_checkpoint._validated_pending_gradients(  # type: ignore[attr-defined]
            record,
            2,
            expected_parameters={"transformer.attn.to_q"},
            expected_specs={
                "transformer.attn.to_q": ((2, 2), "bfloat16", "mps")
            },
        )


def _parent_descriptor(
    root: Path, result: CheckpointResult, *, evidence_digest: str | None = None
) -> WanParentCheckpointDescriptor:
    files = result.files
    descriptors = []
    for item in files:
        path = root / item.relative_path
        info = path.stat()
        descriptors.append(
            WanParentCheckpointFileDescriptor(
                role=item.role,
                path=path,
                digest=item.digest,
                size_bytes=item.size_bytes,
                device=info.st_dev,
                inode=info.st_ino,
            )
        )
    return WanParentCheckpointDescriptor(
        checkpoint_id=result.checkpoint_id,
        checkpoint_digest=evidence_digest or _sha(b"checkpoint-evidence"),
        files=tuple(descriptors),
    )


def test_resume_restore_rehashes_and_schema_validates_all_seven_parent_files(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)

    restored = codec.restore(descriptor, _context(request))

    assert restored == _state(request)
    codec.reverify_restored()
    codec.close()


def test_resume_restore_rejects_replaced_parent_before_decoding(tmp_path: Path) -> None:
    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)
    weights = next(item for item in descriptor.files if item.role == "weights")
    path = weights.path
    path.unlink()
    path.write_bytes(_safetensors())

    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        codec.restore(descriptor, _context(request))
    codec.close()


def test_resume_restored_parent_handle_detects_path_replacement_before_use(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)
    codec.restore(descriptor, _context(request))
    optimizer = next(item for item in descriptor.files if item.role == "optimizer")
    path = optimizer.path
    original = path.read_bytes()
    path.unlink()
    path.write_bytes(original)

    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        codec.reverify_restored()
    codec.close()


def test_resume_rejects_directory_rename_with_symlink_alias_to_same_leaf(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)
    codec.restore(descriptor, _context(request))
    checkpoint = tmp_path / result.checkpoint_id
    moved = tmp_path / "moved-parent"
    checkpoint.rename(moved)
    checkpoint.symlink_to(moved, target_is_directory=True)

    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        codec.reverify_restored()
    codec.close()


def test_resume_rejects_duplicate_parent_path_and_inode(tmp_path: Path) -> None:
    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)
    duplicated = replace(
        descriptor,
        files=descriptor.files[:-1]
        + (
            replace(
                descriptor.files[-1],
                path=descriptor.files[0].path,
                device=descriptor.files[0].device,
                inode=descriptor.files[0].inode,
            ),
        ),
    )

    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        codec.restore(duplicated, _context(request))
    codec.close()


def test_resume_rejects_replaced_checkpoint_root_alias(tmp_path: Path) -> None:
    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)
    codec.restore(descriptor, _context(request))
    moved = tmp_path.with_name(f"{tmp_path.name}-moved")
    tmp_path.rename(moved)
    tmp_path.symlink_to(moved, target_is_directory=True)

    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        codec.reverify_restored()
    codec.close()


def test_resume_missing_parent_and_fifo_fail_without_blocking(tmp_path: Path) -> None:
    for fault in ("missing", "fifo"):
        root = tmp_path / fault
        root.mkdir()
        request = _request(root)
        codec = WanCheckpointCodec(root)
        result = codec.write(_state(request), _context(request))
        descriptor = _parent_descriptor(root, result)
        victim = next(item for item in descriptor.files if item.role == "optimizer")
        path = victim.path
        path.unlink()
        if fault == "fifo":
            os.mkfifo(path)
        with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
            codec.restore(descriptor, _context(request))
        codec.close()


def test_resume_rejects_descriptor_subclasses(tmp_path: Path) -> None:
    @dataclass(frozen=True, slots=True)
    class DescriptorSubclass(WanParentCheckpointDescriptor):
        pass

    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)
    subclass = DescriptorSubclass(
        checkpoint_id=descriptor.checkpoint_id,
        checkpoint_digest=descriptor.checkpoint_digest,
        files=descriptor.files,
        step=descriptor.step,
        total_steps=descriptor.total_steps,
        parent_checkpoint_id=descriptor.parent_checkpoint_id,
        parent_optimizer_state_digest=descriptor.parent_optimizer_state_digest,
    )

    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        codec.restore(subclass, _context(request))
    codec.close()


def test_resume_rejects_mutation_between_seven_file_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    codec = WanCheckpointCodec(tmp_path)
    result = codec.write(_state(request), _context(request))
    descriptor = _parent_descriptor(tmp_path, result)
    optimizer = next(item for item in descriptor.files if item.role == "optimizer")
    original_read = wan_checkpoint._read_open_regular
    calls = 0

    def read_then_replace(descriptor_fd: int, expected_size: int) -> bytes:
        nonlocal calls
        payload = original_read(descriptor_fd, expected_size)
        calls += 1
        if calls == 1:
            original = optimizer.path.read_bytes()
            optimizer.path.unlink()
            optimizer.path.write_bytes(original)
        return payload

    monkeypatch.setattr(wan_checkpoint, "_read_open_regular", read_then_replace)

    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        codec.restore(descriptor, _context(request))
    codec.close()


class _HostileMapping(dict[str, object]):
    def __init__(self, value: dict[str, object], behavior: str) -> None:
        super().__init__(value)
        self.behavior = behavior

    def get(self, key: str, default: object = None) -> object:
        if self.behavior == "get" and key == "request_id":
            return "rogue-request"
        return super().get(key, default)

    def items(self):  # type: ignore[no-untyped-def]
        if self.behavior == "items":
            return (("rogue", True),)
        return super().items()

    def __iter__(self) -> Iterator[str]:
        if self.behavior == "iteration":
            return iter(("rogue",))
        return super().__iter__()


class _ValidationContextSubclass(WanCheckpointValidationContext):
    pass


def test_codec_durably_publishes_exactly_seven_validator_accepted_roles(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "scratch")

    with WanCheckpointCodec(request.scratch_root) as codec:
        checkpoint = codec.write(_state(request), _context(request))

    result = TrainingResult(
        request_id=request.request_id,
        run_id=request.run_id,
        backend_id=request.backend_id,
        backend_digest=request.materials.backend_digest,
        materials=request.materials,
        output_root=request.scratch_root,
        checkpoints=(checkpoint,),
        telemetry_digest=checkpoint.telemetry_digest,
        changed_targets=request.trainable_targets,
        frozen_parameters_unchanged=True,
        finite_gradients=True,
        elapsed_seconds=1.0,
    )
    validate_training_result(request, result)

    assert tuple(item.role for item in checkpoint.files) == (
        "weights",
        "optimizer",
        "trainer_state",
        "telemetry",
        "trainable_targets",
        "frozen_parameters",
        "finite_gradient_proof",
    )
    assert all(item.relative_path.startswith("checkpoint-1/") for item in checkpoint.files)
    optimizer = json.loads((request.scratch_root / "checkpoint-1" / "optimizer.json").read_bytes())
    assert optimizer["parameter_state"]["transformer.attn.to_q"]["slots"] == {
        "exp_avg_norm": 0.2,
        "exp_avg_sq_norm": 0.04,
    }
    assert (
        request.scratch_root / "checkpoint-1" / "telemetry.json"
    ).read_bytes() == canonical_json(list(_state(request).telemetry)).encode()


def test_codec_rejects_malformed_safetensors_without_partial_publication(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "scratch")

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(_state(request, weights=b"not-safetensors"), _context(request))

    assert list(request.scratch_root.iterdir()) == []


def test_codec_rejects_replaced_root_before_publication(tmp_path: Path) -> None:
    request = _request(tmp_path / "scratch")
    codec = WanCheckpointCodec(request.scratch_root)
    original = tmp_path / "original-scratch"
    request.scratch_root.rename(original)
    request.scratch_root.mkdir()
    try:
        with pytest.raises(ValueError, match="unsafe_scratch_root"):
            codec.write(_state(request), _context(request))
    finally:
        codec.close()

    assert list(original.iterdir()) == []
    assert list(request.scratch_root.iterdir()) == []


def test_codec_rejects_symlinked_scratch_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe_scratch_root"):
        WanCheckpointCodec(linked / "scratch")

    assert list(real.iterdir()) == []


def test_codec_cleans_staging_when_a_later_exclusive_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    real_write = wan_checkpoint._write_exclusive
    writes = 0

    def fail_second(directory_fd: int, name: str, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected full disk")
        real_write(directory_fd, name, payload)

    monkeypatch.setattr(wan_checkpoint, "_write_exclusive", fail_second)

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_write_failed"):
            codec.write(_state(request), _context(request))

    assert list(request.scratch_root.iterdir()) == []


def test_codec_preserves_primary_and_unlink_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    real_write = wan_checkpoint._write_exclusive
    writes = 0

    def fail_second(directory_fd: int, name: str, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected primary write failure")
        real_write(directory_fd, name, payload)

    def fail_unlink(directory_fd: int, name: str) -> None:
        raise OSError(f"injected unlink failure: {name}")

    monkeypatch.setattr(wan_checkpoint, "_write_exclusive", fail_second)
    monkeypatch.setattr(
        wan_checkpoint, "_unlink_checkpoint_entry", fail_unlink, raising=False
    )

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(RuntimeError, match="wan_checkpoint_cleanup_failed") as raised:
            codec.write(_state(request), _context(request))

    assert isinstance(raised.value, WanCheckpointCleanupError)
    assert str(raised.value.primary_error) == "injected primary write failure"
    cleanup_messages = [str(error) for error in raised.value.cleanup_errors]
    assert cleanup_messages[0] == "injected unlink failure: weights.safetensors"
    assert len(cleanup_messages) == 2
    assert "Directory not empty" in cleanup_messages[1]
    assert (request.scratch_root / ".checkpoint-1.staging").is_dir()


def test_codec_post_publish_directory_close_failure_discards_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")

    def fail_after_close(descriptor: int) -> None:
        os.close(descriptor)
        raise OSError("injected staging close failure")

    monkeypatch.setattr(
        wan_checkpoint, "_close_directory", fail_after_close, raising=False
    )
    codec = WanCheckpointCodec(request.scratch_root)
    try:
        with pytest.raises(RuntimeError, match="wan_checkpoint_cleanup_failed") as raised:
            codec.write(_state(request), _context(request))
    finally:
        codec.close()

    assert isinstance(raised.value, WanCheckpointCleanupError)
    assert raised.value.primary_error is None
    assert [str(error) for error in raised.value.cleanup_errors] == [
        "injected staging close failure"
    ]
    assert list(request.scratch_root.iterdir()) == []


def test_codec_post_publish_close_and_discard_failures_are_aggregated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    real_unlink = os.unlink

    def fail_after_close(descriptor: int) -> None:
        os.close(descriptor)
        raise OSError("injected staging close failure")

    def fail_weights_unlink(directory_fd: int, name: str) -> None:
        if name == "weights.safetensors":
            raise OSError("injected published unlink failure")
        real_unlink(name, dir_fd=directory_fd)

    monkeypatch.setattr(
        wan_checkpoint, "_close_directory", fail_after_close, raising=False
    )
    monkeypatch.setattr(
        wan_checkpoint, "_unlink_checkpoint_entry", fail_weights_unlink
    )
    codec = WanCheckpointCodec(request.scratch_root)
    try:
        with pytest.raises(RuntimeError, match="wan_checkpoint_cleanup_failed") as raised:
            codec.write(_state(request), _context(request))
    finally:
        codec.close()

    assert isinstance(raised.value, WanCheckpointCleanupError)
    cleanup_messages = [str(error) for error in raised.value.cleanup_errors]
    assert cleanup_messages[0:2] == [
        "injected staging close failure",
        "injected published unlink failure",
    ]
    assert "Directory not empty" in cleanup_messages[2]
    retained = request.scratch_root / "checkpoint-1"
    assert [item.name for item in retained.iterdir()] == ["weights.safetensors"]


def test_codec_discard_surfaces_directory_removal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")

    with WanCheckpointCodec(request.scratch_root) as codec:
        checkpoint = codec.write(_state(request), _context(request))

        def fail_rmdir(root_fd: int, directory_name: str) -> None:
            raise OSError(f"injected rmdir failure: {directory_name}")

        monkeypatch.setattr(
            wan_checkpoint, "_remove_checkpoint_directory", fail_rmdir, raising=False
        )
        with pytest.raises(RuntimeError, match="wan_checkpoint_cleanup_failed") as raised:
            codec.discard(checkpoint.checkpoint_id)

    assert isinstance(raised.value, WanCheckpointCleanupError)
    assert raised.value.primary_error is None
    assert "injected rmdir failure" in str(raised.value.cleanup_errors[0])
    retained = request.scratch_root / "checkpoint-1"
    assert retained.is_dir()
    assert list(retained.iterdir()) == []


def test_codec_discard_surfaces_root_fsync_failure_after_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")

    with WanCheckpointCodec(request.scratch_root) as codec:
        checkpoint = codec.write(_state(request), _context(request))
        root_fd = codec._root_fd
        real_fsync = os.fsync

        def fail_root_fsync(descriptor: int) -> None:
            if descriptor == root_fd:
                raise OSError("injected root fsync failure")
            real_fsync(descriptor)

        monkeypatch.setattr(
            wan_checkpoint, "_fsync_directory", fail_root_fsync, raising=False
        )
        with pytest.raises(RuntimeError, match="wan_checkpoint_cleanup_failed") as raised:
            codec.discard(checkpoint.checkpoint_id)

    assert isinstance(raised.value, WanCheckpointCleanupError)
    assert "injected root fsync failure" in str(raised.value.cleanup_errors[0])
    assert not (request.scratch_root / "checkpoint-1").exists()


def test_codec_discard_root_fsyncs_successful_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")

    with WanCheckpointCodec(request.scratch_root) as codec:
        checkpoint = codec.write(_state(request), _context(request))
        root_fd = codec._root_fd
        synced: list[int] = []

        def record_fsync(descriptor: int) -> None:
            synced.append(descriptor)
            os.fsync(descriptor)

        monkeypatch.setattr(
            wan_checkpoint, "_fsync_directory", record_fsync, raising=False
        )
        codec.discard(checkpoint.checkpoint_id)

    assert synced == [root_fd]
    assert not (request.scratch_root / "checkpoint-1").exists()


def test_codec_rejects_nonfinite_optimizer_state_before_writing(tmp_path: Path) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    state = replace(
        state,
        optimizer={
            **state.optimizer,
            "backend_state": {BACKEND_ID: {"loss": float("nan")}},
        },
    )

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(state, _context(request))

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize("behavior", ("get", "items", "iteration"))
def test_codec_rejects_hostile_mapping_without_publishing(
    tmp_path: Path, behavior: str
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    state = replace(
        state,
        trainable_targets=_HostileMapping(dict(state.trainable_targets), behavior),
    )

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(state, _context(request))

    assert list(request.scratch_root.iterdir()) == []


def test_codec_validates_and_writes_the_same_owned_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    original_targets = dict(state.trainable_targets)
    real_validate = wan_checkpoint._validate_role_schemas
    validated_targets: dict[str, object] | None = None

    def validate_snapshot(
        snapshot: WanCheckpointState, context: WanCheckpointValidationContext
    ) -> None:
        nonlocal validated_targets
        validated_targets = dict(snapshot.trainable_targets)
        original_targets["request_id"] = "mutated-after-snapshot"
        real_validate(snapshot, context)

    state = replace(state, trainable_targets=original_targets)
    monkeypatch.setattr(wan_checkpoint, "_validate_role_schemas", validate_snapshot)

    with WanCheckpointCodec(request.scratch_root) as codec:
        codec.write(state, _context(request))

    written = (request.scratch_root / "checkpoint-1" / "trainable_targets.json").read_bytes()
    assert validated_targets is not None
    assert written == canonical_json(validated_targets).encode()
    assert json.loads(written)["request_id"] == "request-1"
    assert original_targets["request_id"] == "mutated-after-snapshot"


def test_codec_validates_and_writes_the_same_captured_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    expected = state.weights
    real_validate = wan_checkpoint._validate_role_schemas

    def mutate_frozen_state(
        snapshot: WanCheckpointState, context: WanCheckpointValidationContext
    ) -> None:
        object.__setattr__(
            state,
            "weights",
            _safetensors((("rogue.lora_A.weight", "F32", (4, 2)),)),
        )
        real_validate(snapshot, context)

    monkeypatch.setattr(wan_checkpoint, "_validate_role_schemas", mutate_frozen_state)

    with WanCheckpointCodec(request.scratch_root) as codec:
        codec.write(state, _context(request))

    assert (request.scratch_root / "checkpoint-1" / "weights.safetensors").read_bytes() == expected
    assert state.weights != expected


def test_codec_uses_one_owned_scalar_identity_snapshot_after_validation_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    expected_weights = state.weights
    expected_targets = canonical_json(dict(state.trainable_targets)).encode()
    real_validate = wan_checkpoint._validate_role_schemas

    def mutate_frozen_state(
        snapshot: WanCheckpointState, context: WanCheckpointValidationContext
    ) -> None:
        object.__setattr__(state, "checkpoint_id", "rogue-checkpoint")
        object.__setattr__(state, "step", 1)
        object.__setattr__(state, "total_steps", 99)
        object.__setattr__(state, "parent_checkpoint_id", "rogue-parent")
        real_validate(snapshot, context)

    monkeypatch.setattr(wan_checkpoint, "_validate_role_schemas", mutate_frozen_state)

    with WanCheckpointCodec(request.scratch_root) as codec:
        result = codec.write(state, _context(request))

    checkpoint = request.scratch_root / "checkpoint-1"
    assert result.checkpoint_id == "checkpoint-1"
    assert result.step == 4
    assert result.fraction == 1.0
    assert result.parent_checkpoint_id is None
    assert (checkpoint / "weights.safetensors").read_bytes() == expected_weights
    assert (checkpoint / "trainable_targets.json").read_bytes() == expected_targets
    assert not (request.scratch_root / "rogue-checkpoint").exists()


def test_codec_uses_one_owned_validation_context_after_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    context = _context(request)
    real_validate = wan_checkpoint._validate_role_schemas

    def mutate_original_context(
        snapshot: WanCheckpointState, owned: WanCheckpointValidationContext
    ) -> None:
        object.__setattr__(context, "request_id", "rogue-request")
        object.__setattr__(context, "lora_rank", 99)
        object.__setattr__(context, "trainable_targets", ("rogue.target",))
        object.__setattr__(context, "optimizer_config", (("learning_rate", 9.0),))
        real_validate(snapshot, owned)

    monkeypatch.setattr(wan_checkpoint, "_validate_role_schemas", mutate_original_context)

    with WanCheckpointCodec(request.scratch_root) as codec:
        result = codec.write(state, context)

    assert result.checkpoint_id == "checkpoint-1"
    assert (request.scratch_root / "checkpoint-1").is_dir()
    assert not (request.scratch_root / "rogue-checkpoint").exists()


def test_codec_rejects_owned_context_mismatch_despite_late_mutation_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    context = replace(_context(request), lora_rank=2)
    real_validate = wan_checkpoint._validate_role_schemas

    def attempt_repair(
        snapshot: WanCheckpointState, owned: WanCheckpointValidationContext
    ) -> None:
        object.__setattr__(context, "lora_rank", 4)
        object.__setattr__(context, "request_id", "request-1")
        real_validate(snapshot, owned)

    monkeypatch.setattr(wan_checkpoint, "_validate_role_schemas", attempt_repair)

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(_state(request), context)

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize(
    "case",
    ("context_subclass", "targets_list", "config_list", "config_entry_list"),
)
def test_codec_rejects_context_subclasses_and_mutable_nested_values(
    tmp_path: Path, case: str
) -> None:
    request = _request(tmp_path / "scratch")
    context = _context(request)
    if case == "context_subclass":
        context = _ValidationContextSubclass(
            context.checkpoint_id,
            context.request_id,
            context.run_id,
            context.backend_id,
            context.seed,
            context.step,
            context.total_steps,
            context.parent_checkpoint_id,
                context.parent_optimizer_state_digest,
                context.trainable_targets,
                context.trainable_parameter_specs,
                context.lora_rank,
            context.lora_alpha,
            context.lora_dropout,
            context.optimizer_algorithm,
            context.optimizer_config,
        )
    elif case == "targets_list":
        object.__setattr__(context, "trainable_targets", ["transformer.attn.to_q"])
    elif case == "config_list":
        object.__setattr__(context, "optimizer_config", [("learning_rate", 0.001)])
    else:
        object.__setattr__(context, "optimizer_config", (["learning_rate", 0.001],))

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(_state(request), context)

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize(
    "case",
    (
        "empty_optimizer",
        "empty_trainer_state",
        "malformed_parameter_groups",
        "malformed_telemetry",
    ),
)
def test_codec_rejects_empty_or_malformed_role_schemas_before_writing(
    tmp_path: Path, case: str
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    if case == "empty_optimizer":
        state = replace(state, optimizer={})
    elif case == "empty_trainer_state":
        state = replace(state, trainer_state={})
    elif case == "malformed_parameter_groups":
        state = replace(state, optimizer={**state.optimizer, "parameter_groups": []})
    else:
        state = replace(state, telemetry=({"sequence": 1, "step": 1},))

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(state, _context(request))

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize(
    "role",
    (
        "optimizer",
        "trainer_state",
        "telemetry",
        "trainable_targets",
        "frozen_parameters",
        "finite_gradient_proof",
    ),
)
def test_codec_rejects_rogue_keys_in_every_json_role_before_writing(
    tmp_path: Path, role: str
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    if role == "optimizer":
        state = replace(state, optimizer={**state.optimizer, "rogue": True})
    elif role == "trainer_state":
        state = replace(state, trainer_state={**state.trainer_state, "rogue": True})
    elif role == "telemetry":
        first, *rest = state.telemetry
        state = replace(state, telemetry=({**first, "rogue": True}, *rest))
    elif role == "trainable_targets":
        state = replace(
            state, trainable_targets={**state.trainable_targets, "rogue": True}
        )
    elif role == "frozen_parameters":
        state = replace(
            state, frozen_parameters={**state.frozen_parameters, "rogue": True}
        )
    else:
        state = replace(
            state,
            finite_gradient_proof={**state.finite_gradient_proof, "rogue": True},
        )

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(state, _context(request))

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize(
    "case",
    (
        "optimizer_target_mismatch",
        "trainer_backend_mismatch",
        "telemetry_step_mismatch",
        "frozen_target_mismatch",
        "finite_proof_false",
        "coerced_scalar_type",
    ),
)
def test_codec_rejects_semantically_inconsistent_roles_before_writing(
    tmp_path: Path, case: str
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    if case == "optimizer_target_mismatch":
        state = replace(
            state,
            optimizer={
                **state.optimizer,
                "parameter_state": {
                    "transformer.attn.rogue": {
                        "step": 4,
                        "slots": {"exp_avg_norm": 0.2},
                    }
                },
            },
        )
    elif case == "trainer_backend_mismatch":
        state = replace(
            state, trainer_state={**state.trainer_state, "backend_id": "rogue.backend"}
        )
    elif case == "telemetry_step_mismatch":
        first, *rest = state.telemetry
        state = replace(state, telemetry=({**first, "step": 2}, *rest))
    elif case == "frozen_target_mismatch":
        state = replace(
            state,
            frozen_parameters={
                **state.frozen_parameters,
                "trainable_targets": ["transformer.attn.rogue"],
            },
        )
    elif case == "finite_proof_false":
        state = replace(
            state,
            finite_gradient_proof={**state.finite_gradient_proof, "finite": False},
        )
    else:
        state = replace(state, optimizer={**state.optimizer, "global_step": 4.0})

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(state, _context(request))

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize(
    "case",
    (
        "backend",
        "seed",
        "algorithm",
        "config",
        "parent_optimizer_digest",
        "parent_checkpoint_id",
        "request_binding",
        "targets",
        "rank",
    ),
)
def test_codec_rejects_internally_valid_state_that_violates_authoritative_request(
    tmp_path: Path, case: str
) -> None:
    request = _request(tmp_path / "scratch")
    state = _state(request)
    if case == "backend":
        backend_id = "rogue.backend"
        state = replace(
            state,
            optimizer={
                **state.optimizer,
                "backend_state": {backend_id: {"optimizer_updates": 4}},
            },
            trainer_state={
                **state.trainer_state,
                "backend_id": backend_id,
                "backend_state": {backend_id: {"data_cursor": 4}},
            },
        )
    elif case == "seed":
        state = replace(
            state,
            trainer_state={
                **state.trainer_state,
                "seed": 12,
                "rng_states": _rng_states(12, 4),
            },
        )
    elif case == "algorithm":
        state = replace(state, optimizer={**state.optimizer, "algorithm": "sgd"})
    elif case == "config":
        config = {"learning_rate": 0.002}
        group = dict(state.optimizer["parameter_groups"][0])  # type: ignore[index]
        group["hyperparameters"] = config
        state = replace(
            state,
            optimizer={
                **state.optimizer,
                "config": config,
                "parameter_groups": [group],
            },
        )
    elif case == "parent_optimizer_digest":
        state = replace(
            state,
            optimizer={
                **state.optimizer,
                "parent_optimizer_state_digest": _sha(b"rogue-parent"),
            },
        )
    elif case == "parent_checkpoint_id":
        state = replace(state, parent_checkpoint_id="rogue-parent")
    elif case == "request_binding":
        request_id = "rogue-request"
        state = replace(
            state,
            optimizer={**state.optimizer, "request_id": request_id},
            trainer_state={**state.trainer_state, "request_id": request_id},
            trainable_targets={**state.trainable_targets, "request_id": request_id},
            frozen_parameters={**state.frozen_parameters, "request_id": request_id},
            finite_gradient_proof={
                **state.finite_gradient_proof,
                "request_id": request_id,
            },
        )
    elif case == "targets":
        target = "transformer.attn.to_k"
        group = dict(state.optimizer["parameter_groups"][0])  # type: ignore[index]
        group["parameters"] = [target]
        state = replace(
            state,
            weights=_safetensors(
                (
                    (f"{target}.lora_A.weight", "F32", (4, 2)),
                    (f"{target}.lora_B.weight", "F32", (2, 4)),
                )
            ),
            optimizer={
                **state.optimizer,
                "parameter_groups": [group],
                "parameter_state": {target: state.optimizer["parameter_state"][  # type: ignore[index]
                    "transformer.attn.to_q"
                ]},
            },
            trainable_targets={**state.trainable_targets, "targets": [target]},
            frozen_parameters={
                **state.frozen_parameters,
                "trainable_targets": [target],
            },
        )
    else:
        state = replace(
            state,
            weights=_safetensors(
                (
                    ("transformer.attn.to_q.lora_A.weight", "F32", (2, 2)),
                    ("transformer.attn.to_q.lora_B.weight", "F32", (2, 2)),
                )
            ),
        )

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(state, _context(request))

    assert list(request.scratch_root.iterdir()) == []


def test_codec_rejects_structurally_valid_weights_for_undeclared_target(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "scratch")

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(
                _state(
                    request,
                    weights=_safetensors(
                        (
                            ("text_encoder.rogue.lora_A.weight", "F32", (2, 2)),
                            ("text_encoder.rogue.lora_B.weight", "F32", (2, 2)),
                        )
                    ),
                ),
                _context(request),
            )

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize(
    "weights",
    (
        _safetensors(
            (
                ("transformer.attn.to_q.lora_A.weight", "U8", (2, 2)),
                ("transformer.attn.to_q.lora_B.weight", "U8", (2, 2)),
            )
        ),
        _safetensors(
            (
                ("transformer.attn.to_q.lora_A.weight", "F32", (0, 2)),
                ("transformer.attn.to_q.lora_B.weight", "F32", (2, 0)),
            )
        ),
        _safetensors(
            (("transformer.attn.to_q.lora_A.weight", "F32", (2, 2)),)
        ),
        _safetensors(
            (
                ("transformer.attn.to_q.lora_projection.weight", "F32", (2, 2)),
                ("transformer.attn.to_q.lora_projection_2.weight", "F32", (2, 2)),
            )
        ),
    ),
    ids=("integer", "empty", "unpaired", "non-lora"),
)
def test_codec_rejects_non_substantive_lora_tensor_payloads_before_writing(
    tmp_path: Path, weights: bytes
) -> None:
    request = _request(tmp_path / "scratch")

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_state_invalid"):
            codec.write(_state(request, weights=weights), _context(request))

    assert list(request.scratch_root.iterdir()) == []


def test_codec_reopen_rehash_rejects_bytes_changed_after_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")
    real_write = wan_checkpoint._write_exclusive
    corrupted = False

    def write_then_corrupt(directory_fd: int, name: str, payload: bytes) -> None:
        nonlocal corrupted
        real_write(directory_fd, name, payload)
        if not corrupted:
            corrupted = True
            descriptor = os.open(name, os.O_WRONLY, dir_fd=directory_fd)
            try:
                os.lseek(descriptor, -1, os.SEEK_END)
                os.write(descriptor, bytes((payload[-1] ^ 0xFF,)))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    monkeypatch.setattr(wan_checkpoint, "_write_exclusive", write_then_corrupt)

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_write_failed"):
            codec.write(_state(request), _context(request))

    assert list(request.scratch_root.iterdir()) == []


@pytest.mark.parametrize("nonempty", (False, True))
def test_codec_preserves_existing_destination_directory(tmp_path: Path, nonempty: bool) -> None:
    request = _request(tmp_path / "scratch")
    request.scratch_root.mkdir()
    destination = request.scratch_root / "checkpoint-1"
    destination.mkdir()
    if nonempty:
        (destination / "sentinel").write_bytes(b"adversary")
    identity = destination.stat().st_ino

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_write_failed"):
            codec.write(_state(request), _context(request))

    assert destination.stat().st_ino == identity
    assert (
        (destination / "sentinel").read_bytes() if nonempty else list(destination.iterdir())
    ) == (b"adversary" if nonempty else [])
    assert not (request.scratch_root / ".checkpoint-1.staging").exists()


@pytest.mark.parametrize("nonempty", (False, True))
def test_codec_atomic_publish_loses_destination_creation_race_without_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonempty: bool
) -> None:
    request = _request(tmp_path / "scratch")
    codec = WanCheckpointCodec(request.scratch_root)
    raced_identity: int | None = None

    def race_after_absence_check(root_fd: int, destination: str) -> None:
        nonlocal raced_identity
        os.mkdir(destination, dir_fd=root_fd)
        if nonempty:
            sentinel = os.open(
                f"{destination}/sentinel",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            )
            try:
                os.write(sentinel, b"adversary")
                os.fsync(sentinel)
            finally:
                os.close(sentinel)
        raced_identity = os.stat(destination, dir_fd=root_fd, follow_symlinks=False).st_ino

    monkeypatch.setattr(
        wan_checkpoint,
        "_require_destination_absent",
        race_after_absence_check,
        raising=False,
    )
    try:
        with pytest.raises(ValueError, match="wan_checkpoint_write_failed"):
            codec.write(_state(request), _context(request))
    finally:
        codec.close()

    destination = request.scratch_root / "checkpoint-1"
    assert raced_identity is not None
    assert destination.stat().st_ino == raced_identity
    assert (
        (destination / "sentinel").read_bytes() if nonempty else list(destination.iterdir())
    ) == (b"adversary" if nonempty else [])
    assert not (request.scratch_root / ".checkpoint-1.staging").exists()


def test_codec_fails_closed_when_atomic_no_replace_primitive_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path / "scratch")

    def unsupported(root_fd: int, source: str, destination: str) -> None:
        raise OSError(errno.ENOTSUP, "atomic no-replace unsupported")

    monkeypatch.setattr(
        wan_checkpoint,
        "_publish_directory_no_replace",
        unsupported,
        raising=False,
    )

    with WanCheckpointCodec(request.scratch_root) as codec:
        with pytest.raises(ValueError, match="wan_checkpoint_write_failed"):
            codec.write(_state(request), _context(request))

    assert list(request.scratch_root.iterdir()) == []
