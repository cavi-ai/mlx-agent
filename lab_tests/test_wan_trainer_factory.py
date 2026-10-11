from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

import video_lab.trainers.wan_diffusers_runtime as wan_runtime
import video_lab.trainers.wan_factory as wan_factory
import video_lab.trainers.wan_mps as wan_mps
import video_lab.wan_pending as wan_pending
from video_lab.trainer_protocol import TrainerBackend
from video_lab.trainers.wan_diffusers_runtime import (
    WanDiffusersAdapterPaths,
    validate_wan_batch_artifact,
)
from video_lab.trainers.wan_factory import (
    WAN_ANIMEGEN_ADAPTER_ID,
    WanPinnedLocalRoot,
    WanTrainingAdapterUnsupported,
    create_wan_lora_mps_trainer,
    local_wan_training_adapter_capability,
    require_component_on_mps,
    root_manifest_digest,
)
from video_lab.trainers.wan_mps import WanSourceDescriptor, WanTrainingSourceResolver


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class _Runtime:
    closed = False

    def observe_host(self) -> object:
        raise AssertionError("factory construction must be lazy")

    def prepare(self, sources: object, request: object) -> object:
        raise AssertionError("factory construction must be lazy")

    def release(self, prepared: object | None) -> None:
        raise AssertionError("factory construction must be lazy")

    def close(self) -> None:
        self.closed = True


def _descriptor(path: Path, name: str = "base_model") -> WanSourceDescriptor:
    info = path.stat()
    payload = path.read_bytes()
    return WanSourceDescriptor(
        name=name,
        path=lambda: path,
        expected_digest=_sha(payload),
        expected_size=len(payload),
        expected_device=info.st_dev,
        expected_inode=info.st_ino,
    )


def _root(path: Path, revision: str = "a" * 40) -> WanPinnedLocalRoot:
    return WanPinnedLocalRoot(path, revision, root_manifest_digest(path))


def _descriptor_digest(*descriptors: WanSourceDescriptor) -> str:
    with WanTrainingSourceResolver(tuple(descriptors)).acquire() as lease:
        return lease.sources.source_digest


def _roots(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source, cache, artifacts = (tmp_path / name for name in ("source", "cache", "artifacts"))
    source.mkdir()
    cache.mkdir()
    artifacts.mkdir()
    model = source / "model.safetensors"
    model.write_bytes(b"model")
    return source, cache, artifacts, model


def test_factory_import_and_construction_are_dependency_free_and_lazy(tmp_path: Path) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    descriptor = _descriptor(model)
    backend = create_wan_lora_mps_trainer(
        backend_digest=_sha(b"backend"),
        material_digest=_sha(b"materials"),
        orchestration_source_digest=_sha(b"orchestration-source"),
        descriptor_source_digest=_descriptor_digest(descriptor),
        source=_root(source),
        cache=_root(cache, "b" * 40),
        artifact_root=artifacts,
        source_descriptors=(descriptor,),
        runtime=_Runtime(),
    )
    assert isinstance(backend, TrainerBackend)
    code = """
import sys
import video_lab.trainers.wan_factory
for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'peft', 'safetensors'):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_factory_backend_close_is_idempotent_and_rejects_further_operations(
    tmp_path: Path,
) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    descriptor = _descriptor(model)
    runtime = _Runtime()
    backend = create_wan_lora_mps_trainer(
        backend_digest=_sha(b"backend"),
        material_digest=_sha(b"materials"),
        orchestration_source_digest=_sha(b"orchestration"),
        descriptor_source_digest=_descriptor_digest(descriptor),
        source=_root(source),
        cache=_root(cache, "b" * 40),
        artifact_root=artifacts,
        source_descriptors=(descriptor,),
        runtime=runtime,
    )
    concrete = backend  # protocol object is the owned factory backend
    concrete.close()  # type: ignore[attr-defined]
    concrete.close()  # type: ignore[attr-defined]
    assert runtime.closed
    with pytest.raises(ValueError, match="wan_trainer_closed"):
        concrete._require_open()  # type: ignore[attr-defined]


def test_factory_backend_close_fails_busy_without_closing_active_operation(
    tmp_path: Path,
) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    descriptor = _descriptor(model)
    runtime = _Runtime()
    backend = create_wan_lora_mps_trainer(
        backend_digest=_sha(b"backend"),
        material_digest=_sha(b"materials"),
        orchestration_source_digest=_sha(b"orchestration"),
        descriptor_source_digest=_descriptor_digest(descriptor),
        source=_root(source),
        cache=_root(cache, "b" * 40),
        artifact_root=artifacts,
        source_descriptors=(descriptor,),
        runtime=runtime,
    )
    entered, release = threading.Event(), threading.Event()

    def operate() -> None:
        with backend._operation():  # type: ignore[attr-defined]
            entered.set()
            assert release.wait(timeout=10)

    thread = threading.Thread(target=operate)
    thread.start()
    assert entered.wait(timeout=10)
    with pytest.raises(ValueError, match="wan_trainer_busy"):
        backend.close()  # type: ignore[attr-defined]
    assert not runtime.closed
    release.set()
    thread.join(timeout=10)
    backend.close()  # type: ignore[attr-defined]
    assert runtime.closed


def test_factory_backend_close_failure_is_recoverable(tmp_path: Path) -> None:
    class _FailOnceRuntime(_Runtime):
        attempts = 0

        def close(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("injected cleanup failure")
            self.closed = True

    source, cache, artifacts, model = _roots(tmp_path)
    descriptor = _descriptor(model)
    runtime = _FailOnceRuntime()
    backend = create_wan_lora_mps_trainer(
        backend_digest=_sha(b"backend"),
        material_digest=_sha(b"materials"),
        orchestration_source_digest=_sha(b"orchestration"),
        descriptor_source_digest=_descriptor_digest(descriptor),
        source=_root(source),
        cache=_root(cache, "b" * 40),
        artifact_root=artifacts,
        source_descriptors=(descriptor,),
        runtime=runtime,
    )
    with pytest.raises(OSError, match="injected cleanup failure"):
        backend.close()  # type: ignore[attr-defined]
    backend._require_open()  # type: ignore[attr-defined]
    backend.close()  # type: ignore[attr-defined]
    assert runtime.closed


def test_factory_requires_exact_production_adapter_inputs(tmp_path: Path) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    descriptor = _descriptor(model)
    with pytest.raises(
        WanTrainingAdapterUnsupported,
        match="adapter_paths_and_runtime_digest_required",
    ):
        create_wan_lora_mps_trainer(
            backend_digest=_sha(b"backend"),
            material_digest=_sha(b"materials"),
            orchestration_source_digest=_sha(b"orchestration-source"),
            descriptor_source_digest=_descriptor_digest(descriptor),
            source=_root(source),
            cache=_root(cache),
            artifact_root=artifacts,
            source_descriptors=(descriptor,),
        )


def test_factory_requires_distinct_orchestration_and_descriptor_source_digests(
    tmp_path: Path,
) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    with pytest.raises(ValueError, match="wan_factory_source_invalid"):
        create_wan_lora_mps_trainer(
            backend_digest=_sha(b"backend"),
            material_digest=_sha(b"materials"),
            source=_root(source),
            cache=_root(cache),
            artifact_root=artifacts,
            source_descriptors=(_descriptor(model),),
            runtime=_Runtime(),
        )


def test_capability_manifest_is_explicit_and_dependency_free() -> None:
    capability = local_wan_training_adapter_capability()
    assert capability.adapter_id == WAN_ANIMEGEN_ADAPTER_ID
    assert capability.supported is True
    assert capability.required_batch_format == "wan-precomputed-training-batch.v1"


@pytest.mark.parametrize("revision", ["", "main", "latest", "a" * 39, "g" * 40])
def test_factory_rejects_unpinned_revision(tmp_path: Path, revision: str) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    descriptor = _descriptor(model)
    with pytest.raises(ValueError, match="wan_factory_root_invalid"):
        create_wan_lora_mps_trainer(
            backend_digest=_sha(b"backend"),
            material_digest=_sha(b"materials"),
            orchestration_source_digest=_sha(b"orchestration-source"),
            descriptor_source_digest=_descriptor_digest(descriptor),
            source=WanPinnedLocalRoot(source, revision, root_manifest_digest(source)),
            cache=_root(cache),
            artifact_root=artifacts,
            source_descriptors=(descriptor,),
            runtime=_Runtime(),
        )


def test_factory_rejects_root_content_drift_and_symlink(tmp_path: Path) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    pinned = _root(source)
    model.write_bytes(b"replacement")
    descriptor = _descriptor(model)
    with pytest.raises(ValueError, match="wan_factory_root_invalid"):
        create_wan_lora_mps_trainer(
            backend_digest=_sha(b"backend"),
            material_digest=_sha(b"materials"),
            orchestration_source_digest=_sha(b"orchestration-source"),
            descriptor_source_digest=_descriptor_digest(descriptor),
            source=pinned,
            cache=_root(cache),
            artifact_root=artifacts,
            source_descriptors=(descriptor,),
            runtime=_Runtime(),
        )
    link = tmp_path / "linked"
    link.symlink_to(cache, target_is_directory=True)
    with pytest.raises(ValueError, match="wan_factory_root_invalid"):
        root_manifest_digest(link)


class _Tensor:
    def __init__(self, device: object, dtype: str = "torch.bfloat16") -> None:
        self.device = device
        self.dtype = dtype


def _component(
    *,
    device: object = "mps",
    parameters: tuple[object, ...] = ("mps",),
    buffers: tuple[object, ...] = ("mps",),
    device_map: object = None,
) -> object:
    mapping = {"": "mps"} if device_map is None else device_map
    return type(
        "Component",
        (),
        {
            "device": device,
            "hf_device_map": mapping,
            "parameters": lambda self: tuple(_Tensor(item) for item in parameters),
            "buffers": lambda self: tuple(_Tensor(item) for item in buffers),
        },
    )()


@pytest.mark.parametrize(
    "component",
    [
        _component(parameters=("cpu",)),
        _component(buffers=("meta",)),
        _component(device_map={"block": "disk"}),
        _component(device_map={}),
        _component(device="cpu"),
    ],
)
def test_component_placement_rejects_every_incomplete_or_non_mps_inventory(
    component: object,
) -> None:
    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        require_component_on_mps(component)


def test_component_placement_accepts_complete_mps_inventory() -> None:
    require_component_on_mps(_component(device_map={"": "mps", "block": "mps"}))


def test_component_placement_accepts_actual_indexed_mps_device() -> None:
    device = torch.device("mps:0")
    require_component_on_mps(
        _component(
            device=device,
            parameters=(device,),
            buffers=(device,),
            device_map={"": device, "block": device},
        )
    )


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_component_placement_rejects_non_bfloat16_tensor(kind: str) -> None:
    component = _component(device_map={"": "mps"})
    tensors = (_Tensor("mps", "torch.float32"),)
    setattr(component, "parameters" if kind == "parameter" else "buffers", lambda: iter(tensors))
    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        require_component_on_mps(component)


def test_precomputed_batch_manifest_rejects_digest_only_legacy_envelope(tmp_path: Path) -> None:
    tensors = tmp_path / "batch.safetensors"
    manifest = tmp_path / "batch.json"
    tensors.write_bytes(b"exact-safe-tensor-payload")
    manifest.write_text(
        '{"schema_version":"wan-precomputed-training-batch.v1",'
        f'"sha256":"{_sha(tensors.read_bytes())}"}}'
    )

    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        validate_wan_batch_artifact(manifest, tensors)


def test_precomputed_batch_manifest_rejects_unbound_tensor_metadata(tmp_path: Path) -> None:
    tensors = tmp_path / "batch.safetensors"
    manifest = tmp_path / "batch.json"
    tensors.write_bytes(b"not-even-safetensors")
    manifest.write_text(
        __import__("json").dumps(
            {
                "schema_version": "wan-precomputed-training-batch.v1",
                "sha256": _sha(tensors.read_bytes()),
                "batch_size": 1,
                "tensors": [],
                "objective": {
                    "target_kind": "flow_noise_minus_clean",
                    "formula": "target=noise_latents-clean_latents",
                },
                "noise_schedule": {
                    "timestep_boundary": 500,
                    "selector_rule": "high_if_timestep_gte_boundary_else_low",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        validate_wan_batch_artifact(manifest, tensors)


def _raw_safetensors(header: dict[str, object], data: bytes) -> bytes:
    encoded = __import__("json").dumps(header, separators=(",", ":")).encode()
    return len(encoded).to_bytes(8, "little") + encoded + data


@pytest.mark.parametrize(
    ("metadata", "data"),
    [
        ({"dtype": "BOGUS", "shape": [1], "data_offsets": [0, 1]}, b"x"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [0, 3]}, b"xxx"),
        ({"dtype": "F32", "shape": [1] * 65, "data_offsets": [0, 4]}, b"xxxx"),
        ({"dtype": "F32", "shape": [2**62, 2**62], "data_offsets": [0, 4]}, b"xxxx"),
    ],
)
def test_safetensors_envelope_rejects_unknown_or_unbounded_metadata(
    metadata: dict[str, object], data: bytes
) -> None:
    payload = _raw_safetensors({"only": metadata}, data)
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        wan_runtime._validated_safetensors_header(  # type: ignore[attr-defined]
            payload, expected_names={"only"}, batch_size=1
        )


def _loaded_batch() -> dict[str, torch.Tensor]:
    clean = torch.tensor([[[0.25]], [[0.5]]], dtype=torch.bfloat16)
    noise = torch.tensor([[[1.0]], [[1.5]]], dtype=torch.bfloat16)
    timestep = torch.tensor([750.0, 250.0], dtype=torch.float32)
    sigma = (timestep / 1000.0).to(dtype=torch.bfloat16).reshape(2, 1, 1)
    return {
        "hidden_states": sigma * noise + (1 - sigma) * clean,
        "encoder_hidden_states": torch.ones((2, 1, 2), dtype=torch.bfloat16),
        "clean_latents": clean,
        "noise_latents": noise,
        "target": noise - clean,
        "timestep": timestep,
        "transformer_selector": torch.tensor([0, 1], dtype=torch.int64),
    }


def test_loaded_batch_rejects_hidden_states_outside_declared_flow_interpolation() -> None:
    batch = _loaded_batch()
    batch["hidden_states"] = batch["hidden_states"] + 1
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        wan_runtime._validate_loaded_batch(  # type: ignore[attr-defined]
            batch,
            batch_size=2,
            timestep_min=0,
            timestep_max=1000,
            boundary=500,
            in_channels=1,
        )


def test_loaded_batch_rejects_target_that_is_not_declared_flow_objective() -> None:
    batch = _loaded_batch()
    batch["target"] = batch["target"] + 1
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        wan_runtime._validate_loaded_batch(  # type: ignore[attr-defined]
            batch,
            batch_size=2,
            timestep_min=0,
            timestep_max=1000,
            boundary=500,
            in_channels=1,
        )


def test_runtime_rng_receipt_round_trips_observed_torch_state_not_echoed_bytes() -> None:
    torch.manual_seed(91)
    state = torch.get_rng_state()
    receipt = wan_runtime._torch_rng_state(state, "cpu")  # type: ignore[attr-defined]
    restored = wan_runtime._decode_torch_rng(  # type: ignore[attr-defined]
        torch, receipt, "cpu"
    )
    assert torch.equal(restored, state)
    with pytest.raises(ValueError, match="wan_runtime_restore_invalid"):
        wan_runtime._decode_torch_rng(  # type: ignore[attr-defined]
            torch, {**receipt, "state_hex": "00"}, "cpu"
        )


def test_mps_rng_receipt_restores_the_global_stream_continuation() -> None:
    class _MpsRng:
        def __init__(self) -> None:
            self.state = torch.tensor([4], dtype=torch.uint8)

        def get_rng_state(self) -> torch.Tensor:
            return self.state.clone()

        def set_rng_state(self, state: torch.Tensor) -> None:
            self.state = state.clone()

        def consume(self) -> int:
            value = int(self.state.item())
            self.state += 1
            return value

    mps = _MpsRng()
    fake_torch = type(
        "Torch", (), {"mps": mps, "tensor": staticmethod(torch.tensor), "uint8": torch.uint8}
    )()
    receipt = wan_runtime._capture_global_rng(fake_torch, "mps")  # type: ignore[attr-defined]
    expected = mps.consume()
    mps.consume()
    wan_runtime._restore_global_rng(fake_torch, receipt, "mps")  # type: ignore[attr-defined]
    assert mps.consume() == expected


def test_adamw_recipe_normalizes_every_executed_option() -> None:
    assert wan_runtime._normalized_adamw_config(  # type: ignore[attr-defined]
        {
            "learning_rate": 0.0002,
            "gradient_accumulation_steps": 4,
            "beta1": 0.8,
            "beta2": 0.95,
            "eps": 1e-7,
            "weight_decay": 0.02,
            "amsgrad": True,
            "maximize": False,
            "foreach": False,
            "fused": False,
            "capturable": False,
            "differentiable": False,
        }
    ) == {
        "lr": 0.0002,
        "gradient_accumulation_steps": 4,
        "betas": (0.8, 0.95),
        "eps": 1e-7,
        "weight_decay": 0.02,
        "amsgrad": True,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "differentiable": False,
    }


def test_adamw_recipe_rejects_unknown_options_before_prepare() -> None:
    with pytest.raises(ValueError, match="wan_runtime_optimizer_config_invalid"):
        wan_runtime._normalized_adamw_config(  # type: ignore[attr-defined]
            {"learning_rate": 0.001, "mystery": 1}
        )


def test_gradient_accumulation_executes_only_complete_or_terminal_updates() -> None:
    assert [
        step
        for step in range(1, 11)
        if wan_runtime._optimizer_update_due(  # type: ignore[attr-defined]
            step=step, total_steps=10, accumulation_steps=4
        )
    ] == [4, 8, 10]


def test_terminal_partial_accumulation_normalizes_by_actual_residual_count() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([6.0])
    wan_runtime._normalize_accumulated_gradients(  # type: ignore[attr-defined]
        (parameter,), accumulation_position=2
    )
    assert torch.equal(parameter.grad, torch.tensor([3.0]))


def test_pending_gradient_record_round_trips_before_first_optimizer_update() -> None:
    parameter = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.bfloat16))
    parameter.grad = torch.full_like(parameter, 2)
    record = wan_runtime._pending_gradient_record(  # type: ignore[attr-defined]
        {"transformer.block.lora_A.weight": parameter}, accumulation_position=2
    )
    parameter.grad = None
    wan_runtime._restore_pending_gradients(  # type: ignore[attr-defined]
        torch, {"transformer.block.lora_A.weight": parameter}, record
    )
    assert record["accumulation_position"] == 2
    assert torch.equal(parameter.grad, torch.full_like(parameter, 2))
    tampered = __import__("copy").deepcopy(record)
    tampered["parameters"]["transformer.block.lora_A.weight"]["digest"] = _sha(b"bad")
    with pytest.raises(ValueError, match="wan_runtime_optimizer_invalid"):
        wan_runtime._restore_pending_gradients(  # type: ignore[attr-defined]
            torch, {"transformer.block.lora_A.weight": parameter}, tampered
        )


def test_physical_trainable_specs_require_exact_declared_ab_pairs() -> None:
    observed = tuple(
        wan_mps.WanParameterObservation(
            name=f"transformer.attn.to_q.lora_{role}.default.weight",
            module="transformer.attn.to_q",
            requires_grad=True,
            digest=_sha(role.encode()),
            device="mps",
            dtype="bfloat16",
            shape=(4, 2) if role == "A" else (2, 4),
        )
        for role in ("A", "B")
    )
    validated = wan_mps._trainable_specs_from_inventory(  # type: ignore[attr-defined]
        observed, ("transformer.attn.to_q",)
    )
    assert tuple(item.physical_name for item in validated) == (
        "transformer.attn.to_q.lora_A.default.weight",
        "transformer.attn.to_q.lora_B.default.weight",
    )
    with pytest.raises(ValueError):
        wan_pending.validate_trainable_parameter_specs(
            validated[:1], declared_targets=("transformer.attn.to_q",)
        )
    with pytest.raises(ValueError):
        wan_pending.validate_trainable_parameter_specs(
            (
                validated[0],
                wan_pending.TrainableParameterSpec(
                    physical_name="transformer.rogue.lora_B.default.weight",
                    logical_target="transformer.rogue",
                    role="B",
                    shape=(2, 4),
                    dtype="bfloat16",
                    device="cpu",
                ),
            ),
            declared_targets=("transformer.attn.to_q",),
        )


def test_physical_specs_are_derived_from_the_owned_observed_inventory() -> None:
    observed = (
        wan_mps.WanParameterObservation(
            name="transformer.attn.to_q.lora_A.weight",
            module="transformer.attn.to_q",
            requires_grad=True,
            digest=_sha(b"a"),
            device="mps",
            dtype="bfloat16",
            shape=(4, 2),
        ),
        wan_mps.WanParameterObservation(
            name="transformer.attn.to_q.lora_B.weight",
            module="transformer.rogue",
            requires_grad=True,
            digest=_sha(b"b"),
            device="mps",
            dtype="bfloat16",
            shape=(2, 4),
        ),
    )
    with pytest.raises(ValueError):
        wan_mps._trainable_specs_from_inventory(  # type: ignore[attr-defined]
            observed, ("transformer.attn.to_q",)
        )


def test_accumulation_four_cancel_step_two_resume_matches_uninterrupted() -> None:
    def run(parameter: torch.nn.Parameter, start: int, stop: int) -> None:
        for step in range(start, stop + 1):
            gradient = torch.tensor([float(step)])
            parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient
            if step % 4 == 0:
                wan_runtime._normalize_accumulated_gradients(  # type: ignore[attr-defined]
                    (parameter,), accumulation_position=4
                )
                parameter.data.add_(parameter.grad, alpha=-0.1)
                parameter.grad = None

    uninterrupted = torch.nn.Parameter(torch.tensor([1.0]))
    run(uninterrupted, 1, 4)

    resumed = torch.nn.Parameter(torch.tensor([1.0]))
    run(resumed, 1, 2)
    pending = wan_runtime._pending_gradient_record(  # type: ignore[attr-defined]
        {"parameter": resumed}, accumulation_position=2
    )
    resumed.grad = None
    wan_runtime._restore_pending_gradients(  # type: ignore[attr-defined]
        torch, {"parameter": resumed}, pending
    )
    run(resumed, 3, 4)
    assert torch.equal(resumed, uninterrupted)


def test_adamw_slots_require_exact_schema_shape_dtype_and_step_device() -> None:
    parameter = torch.zeros((2, 3), dtype=torch.bfloat16)
    valid = {
        "step": torch.zeros((), dtype=torch.float32),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    wan_runtime._validate_adamw_slots(  # type: ignore[attr-defined]
        valid, parameter, amsgrad=False, capturable=False, fused=False
    )
    for invalid in (
        {**valid, "rogue": torch.zeros(())},
        {key: value for key, value in valid.items() if key != "exp_avg_sq"},
        {**valid, "exp_avg": torch.zeros((1,), dtype=torch.bfloat16)},
        {**valid, "exp_avg_sq": torch.zeros_like(parameter, dtype=torch.float32)},
    ):
        with pytest.raises(ValueError, match="wan_runtime_optimizer_invalid"):
            wan_runtime._validate_adamw_slots(  # type: ignore[attr-defined]
                invalid, parameter, amsgrad=False, capturable=False, fused=False
            )


def test_adapter_snapshot_is_private_immutable_and_detached_from_source_mutation(
    tmp_path: Path,
) -> None:
    source, artifacts = tmp_path / "source", tmp_path / "artifacts"
    source.mkdir()
    artifacts.mkdir()
    wan = source / "wan"
    wan.mkdir()
    (wan / "config.json").write_bytes(b"config")
    for name in ("high.safetensors", "low.safetensors", "batch.json", "batch.safetensors"):
        (source / name).write_bytes(name.encode())
    paths = WanDiffusersAdapterPaths(
        wan_snapshot=Path("wan"),
        high_transformer=Path("high.safetensors"),
        low_transformer=Path("low.safetensors"),
        batch_manifest=Path("batch.json"),
        batch_tensors=Path("batch.safetensors"),
    )
    source_files = tuple(path for path in sorted(source.rglob("*")) if path.is_file())
    descriptors = tuple(
        _descriptor(
            path,
            "base_model" if index == 0 else path.relative_to(source).as_posix(),
        )
        for index, path in enumerate(source_files)
    )
    with WanTrainingSourceResolver(descriptors).acquire() as lease:
        snapshot = wan_factory._snapshot_adapter_inputs(  # type: ignore[attr-defined]
            source, paths, artifacts, lease=lease
        )
    (source / "high.safetensors").write_bytes(b"mutated")
    assert (snapshot / "high.safetensors").read_bytes() == b"high.safetensors"
    assert snapshot.stat().st_mode & 0o777 == 0o500
    assert (snapshot / "high.safetensors").stat().st_mode & 0o777 == 0o400


def test_secure_snapshot_rejects_same_content_inode_swap_after_retention_before_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "weights.safetensors"
    target.write_bytes(b"approved")
    pinned = WanPinnedLocalRoot(source, "a" * 40, root_manifest_digest(source))

    def replace_after_retention() -> None:
        target.rename(source / "detached.safetensors")
        target.write_bytes(b"approved")

    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        wan_factory.secure_snapshot_pinned_root(
            pinned,
            tmp_path / "snapshot",
            after_retain=replace_after_retention,
        )
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    "primary",
    [KeyboardInterrupt(), SystemExit(23), GeneratorExit()],
    ids=["keyboard_interrupt", "system_exit", "generator_exit"],
)
def test_secure_snapshot_cleans_partial_destination_for_base_exceptions_during_write(
    primary: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"approved")
    destination = tmp_path / "snapshot"
    pinned = WanPinnedLocalRoot(source, "a" * 40, root_manifest_digest(source))

    def interrupt_write(*_args: object, **_kwargs: object) -> int:
        raise primary

    monkeypatch.setattr(wan_factory.os, "write", interrupt_write)
    with pytest.raises(type(primary)) as raised:
        wan_factory.secure_snapshot_pinned_root(pinned, destination)

    assert raised.value is primary
    assert not destination.exists()


def test_retained_root_close_attempts_every_descriptor_and_preserves_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "first").write_bytes(b"first")
    (source / "second").write_bytes(b"second")
    pinned = WanPinnedLocalRoot(source, "a" * 40, root_manifest_digest(source))
    original_close = wan_factory.os.close
    retained_descriptors: tuple[int, ...] = ()
    attempted: list[int] = []

    def fail_retained_close(descriptor: int) -> None:
        if descriptor in retained_descriptors:
            attempted.append(descriptor)
            raise OSError("private descriptor")
        original_close(descriptor)

    with monkeypatch.context() as patch, pytest.raises(BaseExceptionGroup) as raised:
        with wan_factory.retain_pinned_root(pinned) as lease:
            retained_descriptors = lease.descriptors
            patch.setattr(wan_factory.os, "close", fail_retained_close)
            raise KeyboardInterrupt()

    primary, cleanup = raised.value.exceptions
    assert isinstance(primary, KeyboardInterrupt)
    assert isinstance(cleanup, ExceptionGroup)
    assert attempted == list(reversed(retained_descriptors))
    assert len(cleanup.exceptions) == len(retained_descriptors)
    assert {str(failure) for failure in cleanup.exceptions} == {
        "wan_retained_root_close_failed"
    }
    assert str(tmp_path) not in str(raised.value)
    for descriptor in reversed(retained_descriptors):
        original_close(descriptor)


def test_snapshot_construction_preserves_primary_and_exhaustive_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_bytes(b"approved")
    destination = tmp_path / "snapshot"
    target = destination / "weights"
    pinned = WanPinnedLocalRoot(source, "a" * 40, root_manifest_digest(source))
    original_unlink = Path.unlink
    original_rmdir = Path.rmdir
    attempts: list[str] = []

    def interrupt_write(*_args: object, **_kwargs: object) -> int:
        raise KeyboardInterrupt()

    def fail_mode(*_args: object, **_kwargs: object) -> None:
        attempts.append("chmod")
        raise OSError("private mode path")

    def fail_unlink(path: Path, *_args: object, **_kwargs: object) -> None:
        if path == target:
            attempts.append("unlink")
            raise OSError("private unlink path")
        original_unlink(path)

    def fail_rmdir(path: Path, *_args: object, **_kwargs: object) -> None:
        if path == destination:
            attempts.append("rmdir")
            raise OSError("private root path")
        original_rmdir(path)

    def fail_fsync(_path: Path) -> None:
        attempts.append("fsync")
        raise OSError("private fsync path")

    with monkeypatch.context() as patch, pytest.raises(BaseExceptionGroup) as raised:
        patch.setattr(wan_factory.os, "write", interrupt_write)
        patch.setattr(wan_factory.os, "chmod", fail_mode)
        patch.setattr(Path, "unlink", fail_unlink)
        patch.setattr(Path, "rmdir", fail_rmdir)
        patch.setattr(wan_factory, "_fsync_directory", fail_fsync)
        wan_factory.secure_snapshot_pinned_root(pinned, destination)

    primary, cleanup = raised.value.exceptions
    assert isinstance(primary, KeyboardInterrupt)
    assert isinstance(cleanup, ExceptionGroup)
    assert attempts == ["chmod", "chmod", "unlink", "rmdir", "fsync"]
    assert str(tmp_path) not in str(raised.value)
    wan_factory._remove_snapshot(destination)  # type: ignore[attr-defined]


def test_retained_source_copy_streams_large_file_in_fixed_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large.safetensors"
    source.write_bytes(b"x" * (3 * 1024 * 1024 + 17))
    descriptor = _descriptor(source)
    destination = tmp_path / "copy.safetensors"
    requests: list[int] = []
    original_read = wan_factory.os.read

    def observed_read(file_descriptor: int, size: int) -> bytes:
        requests.append(size)
        return original_read(file_descriptor, size)

    monkeypatch.setattr("video_lab.trainers.wan_mps.os.read", observed_read)
    with WanTrainingSourceResolver((descriptor,)).acquire() as lease:
        opened = wan_factory.os.open(
            destination, wan_factory.os.O_RDWR | wan_factory.os.O_CREAT | wan_factory.os.O_EXCL
        )
        try:
            digest, size = lease.copy_file_to(lease.sources.files[0], opened)
        finally:
            wan_factory.os.close(opened)
    assert digest == descriptor.expected_digest
    assert size == descriptor.expected_size
    assert max(requests) <= 1024 * 1024
    assert sum(1 for size in requests if size == 1024 * 1024) >= 3


def test_open_file_digest_streams_without_aggregate_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sparse.safetensors"
    size = 5 * 1024 * 1024 + 19
    with source.open("wb") as stream:
        stream.seek(size - 1)
        stream.write(b"z")
    requests: list[int] = []
    original_read = os.read

    def observed_read(descriptor: int, count: int) -> bytes:
        requests.append(count)
        return original_read(descriptor, count)

    class AggregateBufferForbidden:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("digest must not aggregate file bytes")

    monkeypatch.setattr("video_lab.trainers.wan_mps.os.read", observed_read)
    monkeypatch.setattr(
        "video_lab.trainers.wan_mps.bytearray", AggregateBufferForbidden, raising=False
    )
    descriptor = os.open(source, os.O_RDONLY)
    try:
        digest = __import__("video_lab.trainers.wan_mps", fromlist=["x"])._digest_open_file(
            descriptor, size
        )
    finally:
        os.close(descriptor)
    assert digest == _sha(source.read_bytes())
    assert requests and max(requests) <= 1024 * 1024


def test_root_manifest_digest_streams_large_files_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    payload = b"x" * (4 * 1024 * 1024 + 31)
    (nested / "large.bin").write_bytes(payload)
    expected_record = b"nested/large.bin\0" + str(len(payload)).encode() + b"\0"
    expected = (
        "sha256:" + hashlib.sha256(expected_record + hashlib.sha256(payload).digest()).hexdigest()
    )
    requests: list[int] = []
    original_read = os.read

    def observed_read(descriptor: int, count: int) -> bytes:
        requests.append(count)
        return original_read(descriptor, count)

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError("manifest hashing must not materialize a whole file")

    monkeypatch.setattr(wan_factory.os, "read", observed_read)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    assert root_manifest_digest(root) == expected
    assert requests and max(requests) <= 1024 * 1024


def test_root_manifest_rejects_same_size_rewrite_with_restored_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "weights.bin"
    target.write_bytes(b"approved")
    initial = target.stat()
    original_read = os.read
    mutated = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            target.write_bytes(b"replaced")
            os.utime(target, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        return chunk

    monkeypatch.setattr(wan_factory.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="wan_factory_root_invalid"):
        root_manifest_digest(root)
    assert mutated


def test_snapshot_directory_fsync_failure_is_surfaced_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, artifacts = tmp_path / "source", tmp_path / "artifacts"
    source.mkdir()
    artifacts.mkdir()
    (source / "wan").mkdir()
    (source / "wan" / "config.json").write_bytes(b"config")
    for name in ("high", "low", "batch.json", "batch.safe"):
        (source / name).write_bytes(name.encode())
    paths = WanDiffusersAdapterPaths(
        Path("wan"), Path("high"), Path("low"), Path("batch.json"), Path("batch.safe")
    )
    files = tuple(path for path in sorted(source.rglob("*")) if path.is_file())
    descriptors = tuple(
        _descriptor(path, "base_model" if index == 0 else path.relative_to(source).as_posix())
        for index, path in enumerate(files)
    )
    real_fsync = os.fsync
    failed = False

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal failed
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not failed:
            failed = True
            raise OSError("injected snapshot directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(wan_factory.os, "fsync", fail_first_directory_fsync)
    with WanTrainingSourceResolver(descriptors).acquire() as lease:
        with pytest.raises(ValueError, match="wan_factory_source_invalid"):
            wan_factory._snapshot_adapter_inputs(source, paths, artifacts, lease)  # type: ignore[attr-defined]
    assert failed
    assert list(artifacts.iterdir()) == []


def test_factory_constructor_failure_cleans_owned_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, cache, artifacts, model = _roots(tmp_path)
    model.unlink()
    (source / "wan").mkdir()
    (source / "wan" / "config.json").write_bytes(b"config")
    for name in ("high", "low", "batch.json", "batch.safe"):
        (source / name).write_bytes(name.encode())
    paths = WanDiffusersAdapterPaths(
        Path("wan"), Path("high"), Path("low"), Path("batch.json"), Path("batch.safe")
    )
    files = tuple(path for path in sorted(source.rglob("*")) if path.is_file())
    descriptors = tuple(
        _descriptor(path, "base_model" if index == 0 else path.relative_to(source).as_posix())
        for index, path in enumerate(files)
    )

    def fail_capture(_root: Path) -> object:
        raise ValueError("injected runtime identity failure")

    monkeypatch.setattr(wan_runtime, "_capture_root_identity", fail_capture)
    with pytest.raises(ValueError, match="injected runtime identity failure"):
        create_wan_lora_mps_trainer(
            backend_digest=_sha(b"backend"),
            material_digest=_sha(b"material"),
            orchestration_source_digest=_sha(b"orchestration"),
            descriptor_source_digest=_descriptor_digest(*descriptors),
            source=_root(source),
            cache=_root(cache),
            artifact_root=artifacts,
            source_descriptors=descriptors,
            adapter_paths=paths,
            runtime_digest=_sha(b"runtime"),
        )
    assert list(artifacts.iterdir()) == []


def test_retained_source_copy_rejects_mutation_after_approval(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"approved")
    descriptor = _descriptor(source)
    destination = tmp_path / "copy.safetensors"
    with WanTrainingSourceResolver((descriptor,)).acquire() as lease:
        source.write_bytes(b"mutated!")
        opened = wan_factory.os.open(
            destination, wan_factory.os.O_RDWR | wan_factory.os.O_CREAT | wan_factory.os.O_EXCL
        )
        try:
            with pytest.raises(ValueError, match="wan_source_invalid"):
                lease.copy_file_to(lease.sources.files[0], opened)
        finally:
            wan_factory.os.close(opened)


def test_owned_runtime_close_recursively_removes_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights").write_bytes(b"large-private-model")
    torch = type(
        "Torch",
        (),
        {
            "mps": type(
                "Mps",
                (),
                {
                    "synchronize": staticmethod(lambda: None),
                    "empty_cache": staticmethod(lambda: None),
                },
            )()
        },
    )()
    runtime = wan_runtime.WanDiffusersTrainingRuntime(
        root=snapshot,
        paths=WanDiffusersAdapterPaths(
            Path("wan"), Path("high"), Path("low"), Path("batch.json"), Path("batch.safe")
        ),
        runtime_digest=_sha(b"runtime"),
        importer=lambda _name: torch,
        cleanup=lambda: wan_factory._remove_snapshot(snapshot),  # type: ignore[attr-defined]
    )
    runtime.close()
    assert not snapshot.exists()


def test_snapshot_cleanup_reports_every_mode_failure_without_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "private-snapshot"
    nested = snapshot / "nested"
    nested.mkdir(parents=True)
    (nested / "weights").write_bytes(b"private")

    def fail_mode(*_args: object, **_kwargs: object) -> None:
        raise OSError("private path must not escape")

    monkeypatch.setattr(wan_factory.os, "chmod", fail_mode)
    with pytest.raises(ExceptionGroup) as raised:
        wan_factory._remove_snapshot(snapshot)  # type: ignore[attr-defined]

    failures = raised.value.exceptions
    assert len(failures) == 3
    assert {str(failure) for failure in failures} == {"wan_snapshot_cleanup_mode_failed"}
    assert str(tmp_path) not in str(raised.value)
    assert not snapshot.exists()


def test_snapshot_cleanup_scope_preserves_primary_and_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "private-snapshot"
    snapshot.mkdir()
    (snapshot / "weights").write_bytes(b"private")

    def fail_mode(*_args: object, **_kwargs: object) -> None:
        raise OSError("private path must not escape")

    monkeypatch.setattr(wan_factory.os, "chmod", fail_mode)
    with pytest.raises(ExceptionGroup) as raised:
        with wan_factory.wan_snapshot_cleanup_scope(snapshot):
            raise RuntimeError("primary_training_failure")

    primary, cleanup = raised.value.exceptions
    assert str(primary) == "primary_training_failure"
    assert isinstance(cleanup, ExceptionGroup)
    assert {str(failure) for failure in cleanup.exceptions} == {
        "wan_snapshot_cleanup_mode_failed"
    }
    assert str(tmp_path) not in str(raised.value)
    assert not snapshot.exists()


@pytest.mark.parametrize(
    "primary",
    [KeyboardInterrupt(), SystemExit(19), GeneratorExit()],
    ids=["keyboard_interrupt", "system_exit", "generator_exit"],
)
def test_snapshot_cleanup_scope_preserves_base_exceptions(
    primary: BaseException, tmp_path: Path
) -> None:
    snapshot = tmp_path / "private-snapshot"
    snapshot.mkdir()
    (snapshot / "weights").write_bytes(b"private")

    with pytest.raises(type(primary)) as raised:
        with wan_factory.wan_snapshot_cleanup_scope(snapshot):
            raise primary

    assert raised.value is primary
    assert not snapshot.exists()


def test_snapshot_cleanup_attempts_all_operations_and_aggregates_heterogeneous_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "private-snapshot"
    nested = snapshot / "nested"
    weights = nested / "weights"
    nested.mkdir(parents=True)
    weights.write_bytes(b"private")
    attempts: list[str] = []
    original_unlink = Path.unlink
    original_rmdir = Path.rmdir

    def fail_mode(*_args: object, **_kwargs: object) -> None:
        attempts.append("chmod")
        raise OSError("secret chmod path")

    def fail_unlink(path: Path, *_args: object, **_kwargs: object) -> None:
        if path == weights:
            attempts.append("unlink")
            raise OSError("secret unlink path")
        original_unlink(path)

    def fail_rmdir(path: Path, *_args: object, **_kwargs: object) -> None:
        if path in {nested, snapshot}:
            attempts.append("rmdir")
            raise OSError("secret rmdir path")
        original_rmdir(path)

    def fail_fsync(_path: Path) -> None:
        attempts.append("fsync")
        raise OSError("secret fsync path")

    with monkeypatch.context() as patch:
        patch.setattr(wan_factory.os, "chmod", fail_mode)
        patch.setattr(Path, "unlink", fail_unlink)
        patch.setattr(Path, "rmdir", fail_rmdir)
        patch.setattr(wan_factory, "_fsync_directory", fail_fsync)
        with pytest.raises(BaseExceptionGroup) as raised:
            with wan_factory.wan_snapshot_cleanup_scope(snapshot):
                raise KeyboardInterrupt()

    primary, cleanup = raised.value.exceptions
    assert isinstance(primary, KeyboardInterrupt)
    assert isinstance(cleanup, ExceptionGroup)
    messages = [str(failure) for failure in cleanup.exceptions]
    assert messages.count("wan_snapshot_cleanup_mode_failed") == 3
    assert messages.count("wan_snapshot_cleanup_remove_failed") == 2
    assert messages.count("wan_snapshot_cleanup_root_failed") == 1
    assert messages.count("wan_snapshot_cleanup_fsync_failed") == 1
    assert attempts.count("chmod") == 3
    assert attempts.count("unlink") == 1
    assert attempts.count("rmdir") == 2
    assert attempts.count("fsync") == 1
    assert str(tmp_path) not in str(raised.value)
    wan_factory._remove_snapshot(snapshot)  # type: ignore[attr-defined]


def test_production_runtime_global_rng_lease_rejects_concurrent_runtime(
    tmp_path: Path,
) -> None:
    paths = WanDiffusersAdapterPaths(
        Path("wan"), Path("high"), Path("low"), Path("batch.json"), Path("batch.safe")
    )
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = wan_runtime.WanDiffusersTrainingRuntime(
        root=first_root, paths=paths, runtime_digest=_sha(b"first")
    )
    second = wan_runtime.WanDiffusersTrainingRuntime(
        root=second_root, paths=paths, runtime_digest=_sha(b"second")
    )
    first._acquire_production_operation()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="wan_runtime_concurrent_operation"):
        second._acquire_production_operation()  # type: ignore[attr-defined]
    first._release_production_operation()  # type: ignore[attr-defined]
    second._acquire_production_operation()  # type: ignore[attr-defined]
    second._release_production_operation()  # type: ignore[attr-defined]
