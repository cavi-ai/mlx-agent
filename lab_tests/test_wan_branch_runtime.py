from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from video_lab.trainer_protocol import LoRAConfiguration
from video_lab.trainers.wan_batch import build_wan_batch_manifest
from video_lab.trainers.wan_branch_loader import WanMpsPreparedFactory
from video_lab.trainers.wan_branch_runtime import (
    WanBranchStepEvidence,
    WanMpsBranchOperations,
    WanMpsOutOfMemoryError,
    is_recognized_mps_oom,
)
from video_lab.trainers.wan_continuity import compare_wan_branch_continuity
from video_lab.trainers.wan_mps import (
    WanPreparedTraining,
    WanPreparedTrainingOperations,
    WanSourceDescriptor,
    WanTrainingSourceResolver,
    WanTrainingSources,
)
from video_lab.trainers.wan_production_models import (
    WanBranchRecipeV1,
    WanComponentInventoryV1,
    WanNumericalToleranceV1,
)


class _Mps:
    def __init__(self) -> None:
        self.state = torch.tensor([3], dtype=torch.uint8)

    def get_rng_state(self) -> torch.Tensor:
        return self.state.clone()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self.state = state.clone()

    def manual_seed(self, seed: int) -> None:
        self.state = torch.tensor([seed % 255], dtype=torch.uint8)

    def current_allocated_memory(self) -> int:
        return 10

    def driver_allocated_memory(self) -> int:
        return 20


class _Torch:
    def __init__(self) -> None:
        self.mps = _Mps()
        self.random = torch.random
        self.optim = torch.optim
        self.nn = torch.nn
        self.uint8 = torch.uint8
        self.float16 = torch.float16
        self.bfloat16 = torch.bfloat16
        self.float32 = torch.float32
        self.tensor = torch.tensor
        self.frombuffer = torch.frombuffer

    def manual_seed(self, seed: int) -> torch.Generator:
        return torch.manual_seed(seed)

    @staticmethod
    def isfinite(value: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(value)


class _MpsTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, value: torch.Tensor) -> _MpsTensor:
        return torch.Tensor._make_subclass(cls, value.detach(), value.requires_grad)

    @property
    def device(self) -> torch.device:
        return torch.device("mps:0")


def _mark_module_mps(module: torch.nn.Module) -> None:
    for child in module.modules():
        for name, parameter in tuple(child.named_parameters(recurse=False)):
            value = _MpsTensor(parameter.detach().to(dtype=torch.bfloat16))
            setattr(child, name, torch.nn.Parameter(value, requires_grad=parameter.requires_grad))


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(2, 2, bias=False)
        self.to_k = torch.nn.Linear(2, 2, bias=False)
        self.to_v = torch.nn.Linear(2, 2, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])


class _Transformer(torch.nn.Module):
    def __init__(self, *, fake_mps: bool = True) -> None:
        super().__init__()
        self.attn = _Attention()
        self.fake_mps = fake_mps
        if fake_mps:
            _mark_module_mps(self)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def add_adapter(self, config: object, *, adapter_name: str) -> None:
        assert adapter_name == "character"
        for module in (self.attn.to_q, self.attn.to_k, self.attn.to_v, self.attn.to_out[0]):
            module.add_module(
                "lora_A", torch.nn.ModuleDict({"character": torch.nn.Linear(2, 2, bias=False)})
            )
            module.add_module(
                "lora_B", torch.nn.ModuleDict({"character": torch.nn.Linear(2, 2, bias=False)})
            )
            if self.fake_mps:
                _mark_module_mps(module.lora_A)
                _mark_module_mps(module.lora_B)
            module.forward = lambda value, base=module: (
                torch.nn.functional.linear(value, base.weight)
                + base.lora_B["character"](base.lora_A["character"](value))
            )  # type: ignore[method-assign]
        self.config_seen = config

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool,
    ) -> object:
        assert return_dict
        self.calls.append((hidden_states.clone(), timestep.clone(), encoder_hidden_states.clone()))
        sample = (
            self.attn.to_q(hidden_states)
            + self.attn.to_k(hidden_states)
            + self.attn.to_v(hidden_states)
            + self.attn.to_out[0](hidden_states)
        )
        return SimpleNamespace(sample=sample)


def _recipe(branch: str = "high_noise") -> WanBranchRecipeV1:
    return WanBranchRecipeV1(
        schema_version="wan-mps-branch-recipe.v1",
        branch=branch,
        rank=2,
        alpha=2,
        dropout=0.0,
        target_suffixes=("to_k", "to_out.0", "to_q", "to_v"),
        optimizer_algorithm="adamw",
        optimizer_parameters=(("learning_rate", 0.01),),
        gradient_clip_norm=1.0,
        seed=7,
        tolerance=WanNumericalToleranceV1(atol=1e-5, rtol=1e-4),
    )


def _operations(
    *, branch: str = "high_noise", cancelled=lambda: False, real_mps: bool = False
) -> tuple[WanMpsBranchOperations, _Transformer, _Torch]:
    transformer, fake_torch = _Transformer(fake_mps=not real_mps), _Torch()
    if real_mps:
        transformer.to(device="mps", dtype=torch.bfloat16)
    batch = {
        "hidden_states": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16
        ),
        "timestep": torch.tensor([900.0, 100.0]),
        "encoder_hidden_states": torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16
        ),
        "clean_latents": torch.zeros(2, 2, dtype=torch.bfloat16),
        "noise_latents": torch.ones(2, 2, dtype=torch.bfloat16),
        "target": torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16
        ),
        "transformer_selector": torch.tensor([0, 1]),
    }
    if real_mps:
        batch = {name: value.to("mps") for name, value in batch.items()}
    inventory = WanComponentInventoryV1(
        schema_version="wan-mps-component-inventory.v1",
        component_manifests=tuple(
            (name, "sha256:" + character * 64)
            for name, character in zip(
                ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"),
                "12345",
                strict=True,
            )
        ),
    )
    loaded = SimpleNamespace(branch=branch, transformer=transformer, inventory=inventory)
    loaded.close = lambda: setattr(loaded, "transformer", None)

    def importer(name: str) -> object:
        if name == "torch":
            return fake_torch
        if name == "peft":
            return SimpleNamespace(LoraConfig=lambda **kw: SimpleNamespace(**kw))
        import importlib

        return importlib.import_module(name)

    return (
        WanMpsBranchOperations(
            loaded_branch=loaded,
            recipe=_recipe(branch),
            batch=batch,
            batch_digest="sha256:" + "6" * 64,
            importer=importer,
            cancelled=cancelled,
        ),
        transformer,
        fake_torch,
    )  # type: ignore[arg-type]


def _prepared_operations() -> WanMpsBranchOperations:
    operations, _transformer, _torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    operations.train_step(step=1, total_steps=1)
    return operations


def _rng_states(
    fake_torch: _Torch,
) -> tuple[object, tuple[object, ...], torch.Tensor, torch.Tensor]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
        fake_torch.mps.get_rng_state(),
    )


def _assert_rng_states_restored(
    before: tuple[object, tuple[object, ...], torch.Tensor, torch.Tensor], fake_torch: _Torch
) -> None:
    python_state, numpy_state, cpu_state, mps_state = before
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    assert torch.equal(fake_torch.mps.get_rng_state(), mps_state)


def _mutate_all_rng(fake_torch: _Torch) -> None:
    random.random()
    np.random.random()
    torch.rand(1)
    fake_torch.mps.manual_seed(99)


def test_branch_runtime_public_types_exist() -> None:
    assert WanMpsBranchOperations.__name__ == "WanMpsBranchOperations"
    assert WanBranchStepEvidence.__name__ == "WanBranchStepEvidence"


@pytest.mark.parametrize(("branch", "selector"), [("high_noise", 0), ("low_noise", 1)])
def test_branch_selector_authority(branch: str, selector: int) -> None:
    assert WanMpsBranchOperations.selector_for_branch(branch) == selector


def test_branch_selector_rejects_unknown_branch() -> None:
    with pytest.raises(ValueError, match="wan_branch_recipe_invalid"):
        WanMpsBranchOperations.selector_for_branch("other")


def test_branch_operations_declares_complete_training_protocol() -> None:
    required = {
        "eligible_lora_modules",
        "inject_lora",
        "parameter_inventory",
        "train_step",
        "checkpoint_state",
        "restore_checkpoint",
        "observe_restored_state",
    }
    assert required <= set(dir(WanMpsBranchOperations))


def test_prepared_training_keeps_measurement_boundary_and_explicit_evidence() -> None:
    class Operations:
        def eligible_lora_modules(self) -> tuple[str, ...]:
            return ("transformer.block.to_q",)

        def inject_lora(self, **_kwargs: object) -> None:
            pass

        def parameter_inventory(self) -> tuple[object, ...]:
            return ()

        def train_step(self, *, step: int, total_steps: int) -> object:
            del step, total_steps
            return {"loss": 1.0, "gradient_norm": 2.0}

        def checkpoint_state(self, *, step: int) -> object:
            raise AssertionError(step)

        def restore_checkpoint(self, state: object) -> None:
            raise AssertionError(state)

        def observe_restored_state(self) -> object:
            raise AssertionError

    operations = Operations()
    assert isinstance(operations, WanPreparedTrainingOperations)
    measurement = WanPreparedTraining(handle=operations).train_step(step=1, total_steps=1)
    assert measurement.model_dump() == {"loss": 1.0, "gradient_norm": 2.0}


def test_realistic_double_runs_exact_masked_one_step_and_preserves_rng() -> None:
    operations, transformer, fake_torch = _operations()
    targets = operations.eligible_lora_modules()
    assert tuple(name.rsplit(".", 1)[-1] for name in targets if not name.endswith("out.0")) == (
        "to_k",
        "to_q",
        "to_v",
    )
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    assert transformer.config_seen.r == 2
    assert transformer.config_seen.lora_alpha == 2
    assert transformer.config_seen.lora_dropout == 0.0
    assert set(transformer.config_seen.target_modules) == {
        target.removeprefix("transformer.") for target in targets
    }
    trainable = operations._named_trainable()  # noqa: SLF001
    optimized = {
        id(parameter)
        for group in operations.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in trainable.values()}
    rng_before = _rng_states(fake_torch)
    injected_before = {
        item.name: item.digest for item in operations.parameter_inventory() if item.requires_grad
    }
    base_before = {
        item.name: item.digest
        for item in operations.parameter_inventory()
        if not item.requires_grad
    }
    measurement = WanPreparedTraining(handle=operations).train_step(step=1, total_steps=1)
    evidence = operations.branch_step_evidence()
    assert measurement.loss == evidence.loss
    assert evidence.selector_count == 1 and evidence.optimizer_update_count == 1
    assert len(transformer.calls) == 1
    assert torch.equal(transformer.calls[0][0], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(transformer.calls[0][1], torch.tensor([900.0]))
    assert torch.equal(transformer.calls[0][2], torch.tensor([[5.0, 6.0]]))
    _assert_rng_states_restored(rng_before, fake_torch)
    assert evidence.changed_adapter_inventory
    after = {item.name: item for item in operations.parameter_inventory()}
    assert {name: after[name].digest for name in base_before} == base_before
    assert all(
        injected_before[item.name] != item.digest
        and any(item.module.endswith(suffix) for suffix in operations.recipe.target_suffixes)
        for item in evidence.changed_adapter_inventory
    )
    assert all(item.requires_grad for item in evidence.changed_adapter_inventory)
    assert all(not item.requires_grad for item in evidence.unchanged_base_inventory)


def test_real_peft_wrapping_preserves_base_inventory_identity() -> None:
    import peft

    operations, transformer, _fake_torch = _operations(real_mps=True)
    original_importer = operations.importer
    operations.importer = lambda name: peft if name == "peft" else original_importer(name)

    def add_adapter(config: object, *, adapter_name: str) -> None:
        peft.inject_adapter_in_model(config, transformer, adapter_name=adapter_name)

    transformer.add_adapter = add_adapter  # type: ignore[method-assign]
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    measurement = operations.train_step(step=1, total_steps=1)
    assert measurement["loss"] > 0.0
    inventory = WanMpsBranchOperations.parameter_inventory(operations)
    assert operations.branch_step_evidence().unchanged_base_inventory
    assert all(".base_layer." not in item.name for item in inventory)


def test_zero_selected_rows_rejected_without_update() -> None:
    operations, _transformer, _torch = _operations(branch="high_noise")
    operations.batch["transformer_selector"] = torch.ones(2, dtype=torch.int64)
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="wan_branch_selector_empty"):
        operations.train_step(step=1, total_steps=1)
    assert operations.optimizer_updates == 0


def test_cancellation_race_immediately_before_step_prevents_update() -> None:
    checks = iter((False, False, True))
    operations, _transformer, _torch = _operations(cancelled=lambda: next(checks))
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="wan_branch_cancelled"):
        operations.train_step(step=1, total_steps=1)
    assert operations.optimizer_updates == 0


def test_python_numpy_cpu_and_mps_rng_restore_on_injection_and_cancellation() -> None:
    checks = iter((False, True))
    operations, _transformer, fake_torch = _operations(cancelled=lambda: next(checks))
    targets = operations.eligible_lora_modules()
    before = _rng_states(fake_torch)
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    _assert_rng_states_restored(before, fake_torch)
    with pytest.raises(ValueError, match="wan_branch_cancelled"):
        operations.train_step(step=1, total_steps=1)
    _assert_rng_states_restored(before, fake_torch)


@pytest.mark.parametrize("missing", ("to_q", "to_k", "to_v", "to_out.0"))
def test_each_missing_attention_suffix_is_rejected(missing: str) -> None:
    operations, transformer, _torch = _operations()
    if missing == "to_out.0":
        transformer.attn.to_out[0] = torch.nn.Identity()
    else:
        setattr(transformer.attn, missing, torch.nn.Identity())
    with pytest.raises(ValueError, match="wan_branch_targets_invalid"):
        operations.eligible_lora_modules()


@pytest.mark.parametrize(
    ("field", "code"),
    [("hidden_states", "wan_branch_input_nonfinite"), ("target", "wan_branch_input_nonfinite")],
)
def test_nonfinite_inputs_fail_path_free(field: str, code: str) -> None:
    operations, _transformer, _torch = _operations()
    operations.batch[field][0, 0] = float("nan")
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    with pytest.raises(ValueError, match=f"^{code}$"):
        operations.train_step(step=1, total_steps=1)
    assert operations.optimizer_updates == 0


def test_extra_or_ambiguous_declared_target_is_rejected() -> None:
    operations, _transformer, _torch = _operations()
    targets = operations.eligible_lora_modules() + ("transformer.undeclared",)
    with pytest.raises(ValueError, match="wan_branch_recipe_invalid"):
        operations.inject_lora(
            targets=targets,
            configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
            seed=7,
            device="mps",
            dtype="bfloat16",
        )


def test_factory_prepared_training_reaches_and_completes_branch_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, _transformer, _torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )

    @dataclass(frozen=True)
    class Request:
        branch: str = "high_noise"
        _authority_signature: tuple[object, ...] = ()
        _snapshot_parent_identity: tuple[int, int] = (1, 1)

    monkeypatch.setattr(
        "video_lab.trainers.wan_branch_loader.load_wan_branch",
        lambda *_args, **_kwargs: operations.loaded_branch,
    )
    monkeypatch.setattr(
        WanMpsBranchOperations,
        "from_sources",
        classmethod(lambda cls, **_kwargs: operations),
    )
    factory = WanMpsPreparedFactory(request=Request())  # type: ignore[arg-type]
    sources = WanTrainingSources(
        files=(), base_model_digest="sha256:" + "0" * 64, source_digest="sha256:" + "1" * 64
    )
    prepared = factory.prepare_branch(sources, object(), _recipe())
    measurement = prepared.train_step(step=1, total_steps=1)
    assert measurement.loss == operations.branch_step_evidence().loss


@pytest.mark.parametrize(
    ("stage", "code"),
    [
        ("output", "wan_branch_output_nonfinite"),
        ("loss", "wan_branch_loss_nonfinite"),
        ("gradient", "wan_branch_gradient_nonfinite"),
        ("parameter", "wan_branch_parameter_nonfinite"),
        ("optimizer", "wan_branch_optimizer_nonfinite"),
    ],
)
def test_each_computed_nonfinite_boundary_has_stable_code(
    monkeypatch: pytest.MonkeyPatch, stage: str, code: str
) -> None:
    operations, transformer, _torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    if stage == "output":
        monkeypatch.setattr(
            transformer,
            "forward",
            lambda **_kwargs: SimpleNamespace(sample=torch.full((1, 2), float("nan"))),
        )
    elif stage == "loss":
        monkeypatch.setattr(
            torch.nn.functional,
            "mse_loss",
            lambda *_args, **_kwargs: torch.tensor(float("nan"), requires_grad=True),
        )
    elif stage == "gradient":
        next(iter(operations._named_trainable().values())).register_hook(  # noqa: SLF001
            lambda value: torch.full_like(value, float("nan"))
        )
    else:
        original_step = operations.optimizer.step

        def corrupt_step() -> None:
            original_step()
            parameter = next(iter(operations._named_trainable().values()))  # noqa: SLF001
            if stage == "parameter":
                parameter.data.fill_(float("nan"))
            else:
                operations.optimizer.state[parameter]["bad"] = torch.tensor(float("nan"))

        monkeypatch.setattr(operations.optimizer, "step", corrupt_step)
    with pytest.raises(ValueError, match=f"^{code}$"):
        operations.train_step(step=1, total_steps=1)


def test_low_branch_exact_mask_and_expected_mse() -> None:
    operations, transformer, _torch = _operations(branch="low_noise")
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    with torch.no_grad():
        expected_prediction = (
            transformer.attn.to_q(torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16))
            + transformer.attn.to_k(torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16))
            + transformer.attn.to_v(torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16))
            + transformer.attn.to_out[0](
                torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
            )
        )
        expected = torch.nn.functional.mse_loss(
            expected_prediction.float(), torch.tensor([[1.0, 0.0]])
        ).item()
    measurement = operations.train_step(step=1, total_steps=1)
    assert measurement["loss"] == pytest.approx(expected)
    assert torch.equal(transformer.calls[0][0], torch.tensor([[3.0, 4.0]]))
    with pytest.raises(ValueError, match="wan_branch_optimizer_update_invalid"):
        operations.train_step(step=1, total_steps=1)
    assert operations.optimizer_updates == 1


@pytest.mark.parametrize("failure", ("validation", "forward", "oom", "cleanup"))
def test_rng_restored_for_each_failure_surface(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    operations, transformer, fake_torch = _operations()
    before = _rng_states(fake_torch)
    if failure == "validation":
        original = operations.eligible_lora_modules

        def fail_validation() -> tuple[str, ...]:
            _mutate_all_rng(fake_torch)
            return original() + ("transformer.undeclared",)

        monkeypatch.setattr(operations, "eligible_lora_modules", fail_validation)
        with pytest.raises(ValueError, match="wan_branch_recipe_invalid"):
            operations.inject_lora(
                targets=original(),
                configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
                seed=7,
                device="mps",
                dtype="bfloat16",
            )
    elif failure == "cleanup":

        def fail_close() -> None:
            _mutate_all_rng(fake_torch)
            raise RuntimeError("cleanup failed")

        operations.loaded_branch.close = fail_close
        with pytest.raises(RuntimeError, match="cleanup failed"):
            operations.close()
    else:
        targets = operations.eligible_lora_modules()
        operations.inject_lora(
            targets=targets,
            configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
            seed=7,
            device="mps",
            dtype="bfloat16",
        )

        def fail_forward(**_kwargs: object) -> object:
            _mutate_all_rng(fake_torch)
            message = "MPS backend out of memory" if failure == "oom" else "forward failed"
            raise RuntimeError(message)

        monkeypatch.setattr(transformer, "forward", fail_forward)
        with pytest.raises(RuntimeError):
            operations.train_step(step=1, total_steps=1)
    _assert_rng_states_restored(before, fake_torch)


@pytest.mark.parametrize("cancel_at", range(4))
def test_complete_cancellation_matrix_restores_rng_and_cannot_checkpoint(
    cancel_at: int,
) -> None:
    events: list[str] = []
    operations, _transformer, fake_torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        _mutate_all_rng(fake_torch)
        result = checks == cancel_at
        events.append(f"cancel:{str(result).lower()}")
        checks += 1
        return result

    operations.cancelled = cancelled
    before = _rng_states(fake_torch)
    with pytest.raises(ValueError, match="wan_branch_cancelled"):
        WanPreparedTraining(handle=operations).train_step(step=1, total_steps=1)
    assert operations.optimizer_updates == 0
    with pytest.raises(ValueError, match="wan_branch_checkpoint_state_invalid"):
        WanPreparedTraining(handle=operations).checkpoint_state(step=1)
    assert events[-1] == "cancel:true"
    _assert_rng_states_restored(before, fake_torch)


def test_ambiguous_physical_trainable_target_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, transformer, _torch = _operations()
    targets = operations.eligible_lora_modules()
    original_named = operations._named_trainable  # noqa: SLF001

    def ambiguous_named() -> dict[str, object]:
        named = original_named()
        first_name, first_parameter = next(iter(named.items()))
        named[first_name.replace(".lora_A.", ".lora_A.character.lora_B.")] = first_parameter
        return named

    monkeypatch.setattr(operations, "_named_trainable", ambiguous_named)
    with pytest.raises(ValueError, match="wan_branch_trainable_inventory_invalid"):
        operations.inject_lora(
            targets=targets,
            configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
            seed=7,
            device="mps",
            dtype="bfloat16",
        )


def test_undeclared_trainable_parameter_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    operations, transformer, _torch = _operations()
    transformer.register_parameter(
        "rogue",
        torch.nn.Parameter(_MpsTensor(torch.ones(2, 2, dtype=torch.bfloat16))),
    )
    original_add_adapter = transformer.add_adapter

    def add_adapter(config: object, *, adapter_name: str) -> None:
        original_add_adapter(config, adapter_name=adapter_name)
        transformer.rogue.requires_grad_(True)

    monkeypatch.setattr(transformer, "add_adapter", add_adapter)
    targets = operations.eligible_lora_modules()
    with pytest.raises(ValueError, match="wan_branch_trainable_inventory_invalid"):
        operations.inject_lora(
            targets=targets,
            configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
            seed=7,
            device="mps",
            dtype="bfloat16",
        )


@pytest.mark.parametrize("cancel_final", (False, True))
def test_backward_clip_cancel_step_order_is_exactly_once(
    monkeypatch: pytest.MonkeyPatch, cancel_final: bool
) -> None:
    operations, _transformer, _torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    events: list[str] = []
    original_loss = operations._loss  # noqa: SLF001
    original_clip = operations._clip_gradient_once  # noqa: SLF001
    original_step = operations.optimizer.step
    checks = iter((False, False, False, cancel_final))

    def loss() -> tuple[object, int]:
        value, count = original_loss()
        value.register_hook(lambda gradient: events.append("backward") or gradient)
        return value, count

    def clip() -> float:
        events.append("clip")
        return original_clip()

    def cancelled() -> bool:
        result = next(checks)
        events.append(f"cancel:{str(result).lower()}")
        return result

    def step() -> None:
        events.append("step")
        original_step()

    monkeypatch.setattr(operations, "_loss", loss)
    monkeypatch.setattr(operations, "_clip_gradient_once", clip)
    monkeypatch.setattr(operations.optimizer, "step", step)
    operations.cancelled = cancelled
    if cancel_final:
        with pytest.raises(ValueError, match="wan_branch_cancelled"):
            operations.train_step(step=1, total_steps=1)
        assert events[-4:] == ["backward", "cancel:false", "clip", "cancel:true"]
        assert "step" not in events
    else:
        operations.train_step(step=1, total_steps=1)
        assert events[-5:] == ["backward", "cancel:false", "clip", "cancel:false", "step"]
    assert events.count("backward") == 1
    assert events.count("clip") == 1
    assert events.count("step") == (0 if cancel_final else 1)


def test_v2_checkpoint_restores_fresh_runtime_and_matches_uninterrupted_probe() -> None:
    reference = _prepared_operations()
    checkpoint = reference.checkpoint_state_v2(parent_checkpoint_id=None)
    reference_probe = reference.continuity_probe()

    restored, restored_transformer, _torch = _operations()
    reference_base = dict(reference.transformer.named_parameters())
    for name, target in restored_transformer.named_parameters():
        target.data.copy_(reference_base[name].data)
    targets = restored.eligible_lora_modules()
    restored.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    receipt = restored.restore_checkpoint_v2(checkpoint, checkpoint.manifest)
    comparison = compare_wan_branch_continuity(
        reference_probe, restored.continuity_probe(), restored.recipe
    )
    assert receipt.schema_version == "wan-branch-restore-receipt.v2"
    assert comparison.accepted


@pytest.mark.parametrize(
    "mutation",
    [
        "adapter_name",
        "adapter_shape",
        "optimizer_mapping",
        "optimizer_slot",
        "cpu_rng",
        "mps_rng",
        "branch",
        "batch",
        "recipe",
        "component",
        "topology",
    ],
)
def test_v2_restore_rejects_every_mutated_identity_before_probe(mutation: str) -> None:
    from dataclasses import replace

    source = _prepared_operations()
    checkpoint = source.checkpoint_state_v2(parent_checkpoint_id=None)
    fresh, _transformer, _torch = _operations()
    targets = fresh.eligible_lora_modules()
    fresh.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    before = {item.name: item.digest for item in fresh.parameter_inventory()}
    state, manifest = checkpoint, checkpoint.manifest
    if mutation in {"branch", "batch", "recipe", "component"}:
        field = {
            "branch": "branch",
            "batch": "batch_digest",
            "recipe": "recipe_digest",
            "component": "component_inventory_digest",
        }[mutation]
        value = "low_noise" if mutation == "branch" else "sha256:" + "9" * 64
        manifest = replace(manifest, **{field: value})
    elif mutation == "topology":
        manifest = replace(manifest, adapter_tensors=manifest.adapter_tensors[:-1])
    elif mutation in {"adapter_name", "adapter_shape"}:
        specs = list(manifest.adapter_tensors)
        first = specs[0]
        specs[0] = replace(
            first,
            name=first.name + ".renamed" if mutation == "adapter_name" else first.name,
            shape=(99, 99) if mutation == "adapter_shape" else first.shape,
        )
        manifest = replace(manifest, adapter_tensors=tuple(specs))
    elif mutation in {"optimizer_mapping", "optimizer_slot"}:
        optimizer = {key: value for key, value in checkpoint.optimizer.items()}
        states = {key: dict(value) for key, value in optimizer["parameter_state"].items()}
        if mutation == "optimizer_mapping":
            states["rogue"] = states.pop(next(iter(states)))
        else:
            states[next(iter(states))]["rogue"] = states[next(iter(states))].pop("step")
        state = replace(checkpoint, optimizer={**optimizer, "parameter_state": states})
    elif mutation == "cpu_rng":
        state = replace(checkpoint, cpu_rng_state=b"mutated")
    else:
        state = replace(checkpoint, mps_rng_state=b"mutated")
    if manifest != checkpoint.manifest:
        state = replace(state, manifest=manifest)
    probe_count = len(fresh.transformer.calls)
    with pytest.raises(ValueError, match="wan_branch_checkpoint_v2_invalid"):
        fresh.restore_checkpoint_v2(state, manifest)
    after = {item.name: item.digest for item in fresh.parameter_inventory()}
    assert after == before
    assert len(fresh.transformer.calls) == probe_count


@pytest.mark.parametrize(
    "mutation",
    [
        "renamed_slot",
        "additional_slot",
        "missing_slot",
        "wrong_dtype",
        "wrong_shape",
        "wrong_scalar",
        "wrong_parameter_mapping",
    ],
)
def test_v2_restore_derives_adamw_slots_from_runtime_before_mutation(mutation: str) -> None:
    from dataclasses import replace

    source = _prepared_operations()
    checkpoint = source.checkpoint_state_v2(parent_checkpoint_id=None)
    fresh, _transformer, _torch = _operations()
    fresh.inject_lora(
        targets=fresh.eligible_lora_modules(),
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    before = {item.name: item.digest for item in fresh.parameter_inventory()}
    probe_count = len(fresh.transformer.calls)
    manifest = checkpoint.manifest
    optimizer = dict(checkpoint.optimizer)
    states = {
        name: {slot: dict(payload) for slot, payload in value.items()}
        for name, value in optimizer["parameter_state"].items()
    }
    first_name = next(iter(states))
    if mutation in {"renamed_slot", "additional_slot", "missing_slot"}:
        for slots in states.values():
            if mutation == "renamed_slot":
                slots["momentum"] = slots.pop("exp_avg")
            elif mutation == "additional_slot":
                slots["rogue"] = dict(slots["step"])
            else:
                slots.pop("exp_avg_sq")
        slot_names = tuple(sorted(next(iter(states.values()))))
        manifest = replace(manifest, optimizer_slot_names=slot_names)
    elif mutation == "wrong_dtype":
        states[first_name]["exp_avg"]["dtype"] = "torch.float16"
    elif mutation == "wrong_shape":
        states[first_name]["exp_avg"]["shape"] = [4]
    elif mutation == "wrong_scalar":
        states[first_name]["step"]["shape"] = [1]
    else:
        states["rogue"] = states.pop(first_name)
        manifest = replace(
            manifest,
            optimizer_parameter_names=tuple(sorted(states)),
        )
    state = replace(
        checkpoint,
        manifest=manifest,
        optimizer={**optimizer, "parameter_state": states},
    )
    with pytest.raises(ValueError, match="wan_branch_checkpoint_v2_invalid"):
        fresh.restore_checkpoint_v2(state, manifest)
    assert {item.name: item.digest for item in fresh.parameter_inventory()} == before
    assert len(fresh.transformer.calls) == probe_count


def test_illicit_base_parameter_mutation_is_rejected_with_stable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, transformer, _torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    original_step = operations.optimizer.step

    def mutate_base() -> None:
        original_step()
        transformer.attn.to_q.weight.data.add_(1)

    monkeypatch.setattr(operations.optimizer, "step", mutate_base)
    with pytest.raises(ValueError, match="^wan_branch_base_parameter_changed$"):
        operations.train_step(step=1, total_steps=1)


@pytest.mark.parametrize(
    "exit_path",
    (
        "zero_selector",
        "output",
        "loss",
        "gradient",
        "parameter",
        "optimizer",
        "repeated_step",
    ),
)
def test_complete_rng_state_restored_for_every_step_exit(
    monkeypatch: pytest.MonkeyPatch, exit_path: str
) -> None:
    operations, transformer, fake_torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )
    if exit_path == "zero_selector":
        operations.batch["transformer_selector"] = torch.ones(2, dtype=torch.int64)
    elif exit_path == "output":
        monkeypatch.setattr(
            transformer,
            "forward",
            lambda **_kwargs: SimpleNamespace(sample=torch.full((1, 2), float("nan"))),
        )
    elif exit_path == "loss":
        monkeypatch.setattr(
            torch.nn.functional,
            "mse_loss",
            lambda *_args, **_kwargs: torch.tensor(float("nan"), requires_grad=True),
        )
    elif exit_path == "gradient":
        next(iter(operations._named_trainable().values())).register_hook(  # noqa: SLF001
            lambda value: torch.full_like(value, float("nan"))
        )
    elif exit_path in {"parameter", "optimizer"}:
        original_step = operations.optimizer.step

        def corrupt_step() -> None:
            original_step()
            parameter = next(iter(operations._named_trainable().values()))  # noqa: SLF001
            if exit_path == "parameter":
                parameter.data.fill_(float("nan"))
            else:
                operations.optimizer.state[parameter]["bad"] = torch.tensor(float("nan"))

        monkeypatch.setattr(operations.optimizer, "step", corrupt_step)
    elif exit_path == "repeated_step":
        operations.train_step(step=1, total_steps=1)
    before = _rng_states(fake_torch)
    with pytest.raises(ValueError):
        operations.train_step(step=1, total_steps=1)
    _assert_rng_states_restored(before, fake_torch)


def test_mps_oom_classification_is_explicit_and_restores_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, transformer, fake_torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )

    def oom(**_kwargs: object) -> object:
        _mutate_all_rng(fake_torch)
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(transformer, "forward", oom)
    before = _rng_states(fake_torch)
    with pytest.raises(WanMpsOutOfMemoryError, match="^wan_branch_mps_out_of_memory$") as raised:
        operations.train_step(step=1, total_steps=1)
    assert is_recognized_mps_oom(raised.value)
    _assert_rng_states_restored(before, fake_torch)


def test_unrelated_runtime_error_is_not_classified_as_mps_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, transformer, _torch = _operations()
    targets = operations.eligible_lora_modules()
    operations.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )

    def unrelated(**_kwargs: object) -> object:
        raise RuntimeError("forward failed")

    monkeypatch.setattr(transformer, "forward", unrelated)
    with pytest.raises(RuntimeError, match="^forward failed$") as raised:
        operations.train_step(step=1, total_steps=1)
    assert not isinstance(raised.value, WanMpsOutOfMemoryError)
    assert not is_recognized_mps_oom(raised.value)


def _source_descriptor(name: str, path: Path) -> WanSourceDescriptor:
    payload = path.read_bytes()
    info = path.stat(follow_symlinks=False)
    return WanSourceDescriptor(
        name=name,
        path=lambda: path,
        expected_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        expected_device=info.st_dev,
        expected_inode=info.st_ino,
    )


def test_compiled_batch_decode_remains_bound_to_retained_source_descriptors(
    tmp_path: Path,
) -> None:
    import safetensors.torch

    tensors = {
        "hidden_states": torch.ones((1, 2), dtype=torch.bfloat16),
        "timestep": torch.tensor((900.0,), dtype=torch.float32),
        "encoder_hidden_states": torch.ones((1, 1, 2), dtype=torch.bfloat16),
        "clean_latents": torch.zeros((1, 2), dtype=torch.bfloat16),
        "noise_latents": torch.ones((1, 2), dtype=torch.bfloat16),
        "target": torch.ones((1, 2), dtype=torch.bfloat16),
        "transformer_selector": torch.zeros((1,), dtype=torch.int64),
    }
    payload = bytes(safetensors.torch.save(tensors))
    manifest_payload = build_wan_batch_manifest(
        payload,
        batch_size=1,
        timestep_min=0,
        timestep_max=1000,
        timestep_boundary=500,
    )
    base = tmp_path / "base-model.json"
    manifest = tmp_path / "batch.json"
    batch = tmp_path / "batch.safetensors"
    base.write_bytes(b"base")
    manifest.write_bytes(manifest_payload)
    batch.write_bytes(payload)
    descriptors = (
        _source_descriptor("base_model", base),
        _source_descriptor("batch_manifest", manifest),
        _source_descriptor("batch_tensors", batch),
    )
    decoded_payloads: list[bytes] = []

    class MovableTensor:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value

        def to(self, *, device: str, dtype: torch.dtype) -> torch.Tensor:
            assert device == "mps"
            return self.value.to(dtype=dtype)

    def decode(retained_payload: bytes) -> dict[str, MovableTensor]:
        decoded_payloads.append(retained_payload)
        replacement = batch.with_suffix(".replacement")
        replacement.write_bytes(b"attacker-batch")
        os.replace(replacement, batch)
        return {name: MovableTensor(value) for name, value in tensors.items()}

    torch_double = SimpleNamespace(
        int64=torch.int64,
        float32=torch.float32,
        bfloat16=torch.bfloat16,
    )

    def importer(name: str) -> object:
        if name == "torch":
            return torch_double
        if name == "safetensors.torch":
            return SimpleNamespace(load=decode)
        raise AssertionError(name)

    loaded = SimpleNamespace(
        branch="high_noise",
        transformer=object(),
        inventory=_operations()[0].loaded_branch.inventory,
    )
    with WanTrainingSourceResolver(descriptors).acquire() as lease:
        with pytest.raises(ValueError, match="wan_source_invalid"):
            WanMpsBranchOperations.from_source_lease(
                loaded_branch=loaded,
                recipe=_recipe(),
                source_lease=lease,
                importer=importer,
            )

    assert decoded_payloads == [payload]


def test_close_releases_tensor_owners_while_handle_remains_referenced() -> None:
    operations, _transformer, _torch = _operations()
    batch_tensors = tuple(operations.batch.values())
    operations.inject_lora(
        targets=operations.eligible_lora_modules(),
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )

    operations.close()

    assert operations.batch == {}
    assert operations.optimizer is None
    assert operations.transformer is None
    assert operations.loaded_branch.transformer is None
    assert batch_tensors


@pytest.mark.parametrize("fault", ("device", "dtype"))
def test_adapter_placement_is_observed_after_peft_injection(fault: str) -> None:
    operations, transformer, _torch = _operations()
    original_add_adapter = transformer.add_adapter

    def add_adapter(config: object, *, adapter_name: str) -> None:
        original_add_adapter(config, adapter_name=adapter_name)
        module = transformer.attn.to_q.lora_A["character"]
        replacement = torch.tensor(
            module.weight.detach().to("cpu").tolist(), dtype=torch.bfloat16
        )
        if fault == "dtype":
            replacement = replacement.to(dtype=torch.float64)
        module.weight = torch.nn.Parameter(replacement)

    transformer.add_adapter = add_adapter  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        operations.inject_lora(
            targets=operations.eligible_lora_modules(),
            configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
            seed=7,
            device="mps",
            dtype="bfloat16",
        )
