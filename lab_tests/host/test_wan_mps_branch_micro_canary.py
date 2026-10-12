from __future__ import annotations

import gc
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_HEAVY_MODULES = ("torch", "peft", "diffusers", "transformers")
_TARGETS = (
    "transformer.attn.to_k",
    "transformer.attn.to_out.0",
    "transformer.attn.to_q",
    "transformer.attn.to_v",
)
_MAX_MPS_RESIDUAL_BYTES = 8 * 1024 * 1024
pytestmark = pytest.mark.wan_mps_micro_host


def _require_enabled() -> dict[str, Any]:
    if os.environ.get("VIDEO_LAB_RUN_WAN_MPS_MICRO_CANARY") != "1":
        pytest.skip("set VIDEO_LAB_RUN_WAN_MPS_MICRO_CANARY=1 to enable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        pytest.fail("wan_mps_fallback_forbidden")
    if platform.system() != "Darwin":
        pytest.skip("Wan MPS branch micro-canary requires Darwin")
    modules: dict[str, Any] = {}
    for name in _HEAVY_MODULES:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            pytest.fail(f"enabled Wan MPS branch micro-canary requires {name}")
    torch = modules["torch"]
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.fail("enabled Wan MPS branch micro-canary requires available MPS")
    return modules


def test_disabled_micro_canary_skips_before_heavy_imports() -> None:
    test_path = Path(__file__).resolve()
    probe = """
import json
import sys

import pytest

status = pytest.main([
    sys.argv[1] + "::test_wan_mps_branch_micro_canary",
    "-q",
    "-p",
    "no:cacheprovider",
])
print("WAN_MICRO_IMPORT_PROBE=" + json.dumps({
    name: name in sys.modules
    for name in ("torch", "peft", "diffusers", "transformers")
}, sort_keys=True))
raise SystemExit(status)
"""
    environment = os.environ.copy()
    environment.pop("VIDEO_LAB_RUN_WAN_MPS_MICRO_CANARY", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe, str(test_path)],
        cwd=test_path.parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(
        item for item in result.stdout.splitlines() if item.startswith("WAN_MICRO_IMPORT_PROBE=")
    )
    imported = json.loads(line.removeprefix("WAN_MICRO_IMPORT_PROBE="))
    assert imported == {name: False for name in _HEAVY_MODULES}


@pytest.mark.parametrize("fallback", ("", "1", "false", "true"))
def test_enabled_canary_rejects_mps_fallback_before_heavy_imports(fallback: str) -> None:
    test_path = Path(__file__).resolve()
    probe = """
import json
import sys

import pytest

status = pytest.main([
    sys.argv[1] + "::test_wan_mps_branch_micro_canary",
    "-q",
    "-p",
    "no:cacheprovider",
])
print("WAN_MICRO_FALLBACK_PROBE=" + json.dumps({
    name: name in sys.modules
    for name in ("torch", "peft", "diffusers", "transformers")
}, sort_keys=True))
raise SystemExit(status)
"""
    environment = os.environ.copy()
    environment["VIDEO_LAB_RUN_WAN_MPS_MICRO_CANARY"] = "1"
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = fallback
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe, str(test_path)],
        cwd=test_path.parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "wan_mps_fallback_forbidden" in result.stdout
    line = next(
        item for item in result.stdout.splitlines() if item.startswith("WAN_MICRO_FALLBACK_PROBE=")
    )
    imported = json.loads(line.removeprefix("WAN_MICRO_FALLBACK_PROBE="))
    assert imported == {name: False for name in _HEAVY_MODULES}


def _inventory() -> Any:
    from video_lab.trainers.wan_production_models import WanComponentInventoryV1

    return WanComponentInventoryV1(
        schema_version="wan-mps-component-inventory.v1",
        component_manifests=tuple(
            (name, "sha256:" + digit * 64)
            for name, digit in zip(
                ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"),
                "12345",
                strict=True,
            )
        ),
    )


def _recipe(branch: str) -> Any:
    from video_lab.trainers.wan_production_models import (
        WanBranchRecipeV1,
        WanNumericalToleranceV1,
    )

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


def _tiny_transformer(torch: Any, peft: Any) -> Any:
    class TinyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = torch.nn.Linear(2, 2, bias=False)
            self.to_k = torch.nn.Linear(2, 2, bias=False)
            self.to_v = torch.nn.Linear(2, 2, bias=False)
            self.to_out = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

    class TinyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = TinyAttention()
            self.force_nonfinite = False
            self.forward_devices: list[tuple[str, str]] = []

        def add_adapter(self, config: object, *, adapter_name: str) -> None:
            peft.inject_adapter_in_model(config, self, adapter_name=adapter_name)

        def forward(
            self,
            *,
            hidden_states: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            return_dict: bool,
        ) -> object:
            del timestep, encoder_hidden_states
            assert return_dict
            self.forward_devices.append((str(hidden_states.device), str(hidden_states.dtype)))
            sample = (
                self.attn.to_q(hidden_states)
                + self.attn.to_k(hidden_states)
                + self.attn.to_v(hidden_states)
                + self.attn.to_out[0](hidden_states)
            )
            if self.force_nonfinite:
                sample = sample * torch.tensor(float("nan"), device="mps")
            return SimpleNamespace(sample=sample)

    transformer = TinyTransformer()
    base_weights = (
        ((0.25, -0.5), (0.75, 0.125)),
        ((-0.125, 0.5), (0.25, -0.75)),
        ((0.625, 0.25), (-0.375, 0.5)),
        ((0.5, -0.25), (0.125, 0.875)),
    )
    with torch.no_grad():
        for module, values in zip(
            (
                transformer.attn.to_q,
                transformer.attn.to_k,
                transformer.attn.to_v,
                transformer.attn.to_out[0],
            ),
            base_weights,
            strict=True,
        ):
            module.weight.copy_(torch.tensor(values, dtype=torch.float32))
    return transformer.to(device="mps", dtype=torch.bfloat16)


def _tensor_bytes(torch: Any, transformer: Any, *, trainable: bool) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for name, parameter in transformer.named_parameters():
        if bool(parameter.requires_grad) is not trainable:
            continue
        logical_name = name if trainable else name.replace(".base_layer.", ".")
        result[f"transformer.{logical_name}"] = (
            parameter.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
        )
    return result


def _batch(torch: Any) -> dict[str, Any]:
    def bf16(values: object) -> Any:
        return torch.tensor(values, device="mps", dtype=torch.bfloat16)

    return {
        "hidden_states": bf16(((1.0, 2.0), (3.0, 4.0))),
        "timestep": torch.tensor((900.0, 100.0), device="mps", dtype=torch.float32),
        "encoder_hidden_states": bf16(((0.5, 0.25), (0.75, 0.125))),
        "clean_latents": bf16(((0.0, 0.0), (0.0, 0.0))),
        "noise_latents": bf16(((1.0, 1.0), (1.0, 1.0))),
        "target": bf16(((0.0, 1.0), (1.0, 0.0))),
        "transformer_selector": torch.tensor((0, 1), device="mps", dtype=torch.int64),
    }


def _new_runtime(
    modules: dict[str, Any], branch: str, *, cancelled: Any = lambda: False
) -> tuple[Any, Any]:
    from video_lab.trainers.wan_branch_runtime import WanMpsBranchOperations

    torch, peft = modules["torch"], modules["peft"]
    transformer = _tiny_transformer(torch, peft)

    class TinyLoadedBranch:
        def __init__(self) -> None:
            self.branch = branch
            self.inventory = _inventory()
            self.transformer: Any | None = transformer
            self.closed = False

        def close(self) -> None:
            self.transformer = None
            self.closed = True
            torch.mps.synchronize()
            torch.mps.empty_cache()

    loaded = TinyLoadedBranch()
    runtime = WanMpsBranchOperations(
        loaded_branch=loaded,
        recipe=_recipe(branch),
        batch=_batch(torch),
        batch_digest="sha256:" + ("6" if branch == "high_noise" else "7") * 64,
        importer=importlib.import_module,
        cancelled=cancelled,
    )
    return runtime, transformer


def _inject(runtime: Any) -> None:
    from video_lab.trainer_protocol import LoRAConfiguration

    targets = runtime.eligible_lora_modules()
    assert targets == _TARGETS
    runtime.inject_lora(
        targets=targets,
        configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
        seed=7,
        device="mps",
        dtype="bfloat16",
    )


def _memory_baseline(torch: Any) -> int:
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    return int(torch.mps.current_allocated_memory())


def _close_runtime(torch: Any, runtime: Any) -> int:
    allocated_before_close = int(torch.mps.current_allocated_memory())
    runtime.close()
    loaded = runtime.loaded_branch
    if (
        runtime.transformer is not None
        or runtime.optimizer is not None
        or not loaded.closed
        or loaded.transformer is not None
    ):
        raise AssertionError("wan_micro_runtime_ownership_not_released")
    return allocated_before_close


def _observe_release(
    torch: Any, *, allocated_before_load: int, allocated_before_close: int
) -> dict[str, int]:
    allocated_after_cleanup = _memory_baseline(torch)
    residual = max(0, allocated_after_cleanup - allocated_before_load)
    if (
        allocated_before_close <= allocated_before_load
        or allocated_after_cleanup >= allocated_before_close
    ):
        raise AssertionError("wan_micro_runtime_memory_not_released")
    if residual > _MAX_MPS_RESIDUAL_BYTES:
        raise AssertionError("wan_micro_runtime_residual_exceeded")
    return {
        "allocated_before_load": allocated_before_load,
        "allocated_before_close": allocated_before_close,
        "allocated_after_cleanup": allocated_after_cleanup,
        "bounded_residual_bytes": residual,
    }


def test_noop_runtime_close_is_rejected_by_release_evidence() -> None:
    modules = _require_enabled()
    torch = modules["torch"]
    allocated_before_load = _memory_baseline(torch)
    runtime, _transformer = _new_runtime(modules, "high_noise")
    _inject(runtime)
    runtime.close = lambda: None
    try:
        with pytest.raises(AssertionError, match="wan_micro_runtime_ownership_not_released"):
            _close_runtime(torch, runtime)
    finally:
        del runtime.close
        allocated_before_close = _close_runtime(torch, runtime)
        runtime = None
        _transformer = None
        _observe_release(
            torch,
            allocated_before_load=allocated_before_load,
            allocated_before_close=allocated_before_close,
        )


def _prove_branch(modules: dict[str, Any], branch: str) -> dict[str, object]:
    from video_lab.trainers.wan_continuity import compare_wan_branch_continuity

    torch = modules["torch"]
    trained_before_load = _memory_baseline(torch)
    reference, reference_transformer = _new_runtime(modules, branch)
    base_before = _tensor_bytes(torch, reference_transformer, trainable=True)
    _inject(reference)
    adapter_before = _tensor_bytes(torch, reference_transformer, trainable=True)
    measurement = reference.train_step(step=1, total_steps=1)
    evidence = reference.branch_step_evidence()
    checkpoint = reference.checkpoint_state_v2(parent_checkpoint_id=None)
    reference_probe = reference.continuity_probe()
    base_after = _tensor_bytes(torch, reference_transformer, trainable=False)
    adapter_after = _tensor_bytes(torch, reference_transformer, trainable=True)
    assert base_after == base_before
    changed = tuple(
        sorted(name for name in adapter_before if adapter_before[name] != adapter_after[name])
    )
    assert changed
    assert all(".lora_" in name for name in changed)
    assert changed == tuple(item.name for item in evidence.changed_adapter_inventory)
    assert evidence.optimizer_update_count == 1
    assert all(
        str(parameter.device) == "mps:0" and parameter.dtype == torch.bfloat16
        for parameter in reference_transformer.parameters()
    )
    assert reference_transformer.forward_devices == [("mps:0", "torch.bfloat16")] * 2
    trained_before_close = _close_runtime(torch, reference)
    reference = None
    reference_transformer = None
    trained_cleanup = _observe_release(
        torch,
        allocated_before_load=trained_before_load,
        allocated_before_close=trained_before_close,
    )

    restored_before_load = _memory_baseline(torch)
    restored, restored_transformer = _new_runtime(modules, branch)
    assert _tensor_bytes(torch, restored_transformer, trainable=True) == base_before
    _inject(restored)
    receipt = restored.restore_checkpoint_v2(checkpoint, checkpoint.manifest)
    restored_state = restored.checkpoint_state_v2(parent_checkpoint_id=None)
    comparison = compare_wan_branch_continuity(
        reference_probe, restored.continuity_probe(), restored.recipe
    )
    assert receipt.schema_version == "wan-branch-restore-receipt.v2"
    assert restored_state.weights == checkpoint.weights
    assert restored_state.optimizer == checkpoint.optimizer
    assert restored_state.cpu_rng_state == checkpoint.cpu_rng_state
    assert restored_state.mps_rng_state == checkpoint.mps_rng_state
    assert comparison.accepted
    assert restored_transformer.forward_devices == [("mps:0", "torch.bfloat16")]
    restored_before_close = _close_runtime(torch, restored)
    restored = None
    restored_transformer = None
    restored_cleanup = _observe_release(
        torch,
        allocated_before_load=restored_before_load,
        allocated_before_close=restored_before_close,
    )
    return {
        "branch": branch,
        "loss": measurement["loss"],
        "changed_adapter_tensors": changed,
        "continuity_max_absolute_error": comparison.max_absolute_error,
        "trained_cleanup": trained_cleanup,
        "restored_cleanup": restored_cleanup,
    }


def _prove_failures(modules: dict[str, Any]) -> dict[str, object]:
    from video_lab.trainer_protocol import LoRAConfiguration

    torch = modules["torch"]
    codes: list[str] = []
    cleanup: list[dict[str, int]] = []

    cancelled_before_load = _memory_baseline(torch)
    cancelled, cancelled_transformer = _new_runtime(modules, "high_noise", cancelled=lambda: True)
    _inject(cancelled)
    cancelled_before = _tensor_bytes(torch, cancelled_transformer, trainable=True)
    with pytest.raises(ValueError, match="^wan_branch_cancelled$"):
        cancelled.train_step(step=1, total_steps=1)
    assert _tensor_bytes(torch, cancelled_transformer, trainable=True) == cancelled_before
    codes.append("wan_branch_cancelled")
    cancelled_before_close = _close_runtime(torch, cancelled)
    cancelled = None
    cancelled_transformer = None
    cleanup.append(
        _observe_release(
            torch,
            allocated_before_load=cancelled_before_load,
            allocated_before_close=cancelled_before_close,
        )
    )

    nonfinite_before_load = _memory_baseline(torch)
    nonfinite, nonfinite_transformer = _new_runtime(modules, "high_noise")
    _inject(nonfinite)
    nonfinite_transformer.force_nonfinite = True
    with pytest.raises(ValueError, match="^wan_branch_output_nonfinite$"):
        nonfinite.train_step(step=1, total_steps=1)
    codes.append("wan_branch_output_nonfinite")
    nonfinite_before_close = _close_runtime(torch, nonfinite)
    nonfinite = None
    nonfinite_transformer = None
    cleanup.append(
        _observe_release(
            torch,
            allocated_before_load=nonfinite_before_load,
            allocated_before_close=nonfinite_before_close,
        )
    )

    mismatched_before_load = _memory_baseline(torch)
    mismatched, _mismatched_transformer = _new_runtime(modules, "high_noise")
    with pytest.raises(ValueError, match="^wan_branch_recipe_invalid$"):
        mismatched.inject_lora(
            targets=_TARGETS[:-1],
            configuration=LoRAConfiguration(rank=2, alpha=2, dropout=0.0),
            seed=7,
            device="mps",
            dtype="bfloat16",
        )
    codes.append("wan_branch_recipe_invalid")
    mismatched_before_close = _close_runtime(torch, mismatched)
    mismatched = None
    _mismatched_transformer = None
    cleanup.append(
        _observe_release(
            torch,
            allocated_before_load=mismatched_before_load,
            allocated_before_close=mismatched_before_close,
        )
    )

    checkpoint_before_load = _memory_baseline(torch)
    checkpoint_failure, _checkpoint_transformer = _new_runtime(modules, "high_noise")
    _inject(checkpoint_failure)
    checkpoint_failure.train_step(step=1, total_steps=1)
    real_importer = checkpoint_failure.importer

    class FailingSafetensors:
        @staticmethod
        def save(_state: object) -> bytes:
            raise RuntimeError("wan_micro_checkpoint_injected")

    checkpoint_failure.importer = lambda name: (
        FailingSafetensors if name == "safetensors.torch" else real_importer(name)
    )
    with pytest.raises(RuntimeError, match="^wan_micro_checkpoint_injected$"):
        checkpoint_failure.checkpoint_state_v2(parent_checkpoint_id=None)
    codes.append("wan_micro_checkpoint_injected")
    checkpoint_before_close = _close_runtime(torch, checkpoint_failure)
    checkpoint_failure = None
    _checkpoint_transformer = None
    cleanup.append(
        _observe_release(
            torch,
            allocated_before_load=checkpoint_before_load,
            allocated_before_close=checkpoint_before_close,
        )
    )
    return {"codes": tuple(codes), "cleanup": tuple(cleanup)}


def _run_branch_matrix(modules: dict[str, Any], scratch_root: Path) -> dict[str, object]:
    torch = modules["torch"]
    before_load = _memory_baseline(torch)
    branches = tuple(_prove_branch(modules, branch) for branch in ("high_noise", "low_noise"))
    failure_evidence = _prove_failures(modules)
    after_cleanup = _memory_baseline(torch)
    residual = max(0, after_cleanup - before_load)
    assert residual <= _MAX_MPS_RESIDUAL_BYTES
    assert not tuple(scratch_root.iterdir())
    return {
        "branches": tuple(item["branch"] for item in branches),
        "branch_evidence": branches,
        "failure_codes": failure_evidence["codes"],
        "failure_cleanup": failure_evidence["cleanup"],
        "allocated_before_load": before_load,
        "allocated_after_cleanup": after_cleanup,
        "bounded_residual_bytes": residual,
    }


def test_wan_mps_branch_micro_canary(tmp_path: Path) -> None:
    modules = _require_enabled()
    evidence = _run_branch_matrix(modules, tmp_path)
    assert evidence["branches"] == ("high_noise", "low_noise")
    assert evidence["failure_codes"] == (
        "wan_branch_cancelled",
        "wan_branch_output_nonfinite",
        "wan_branch_recipe_invalid",
        "wan_micro_checkpoint_injected",
    )
