from __future__ import annotations

import math

import pytest
import torch
from safetensors.torch import save

from video_lab.trainers.wan_checkpoint import (
    WanBranchCheckpointManifestV2,
    WanBranchCheckpointStateV2,
    WanBranchTensorSpecV2,
    validate_wan_branch_checkpoint_v2,
)
from video_lab.trainers.wan_continuity import compare_wan_branch_continuity
from video_lab.trainers.wan_production_models import WanBranchRecipeV1, WanNumericalToleranceV1


def _recipe() -> WanBranchRecipeV1:
    return WanBranchRecipeV1(
        schema_version="wan-mps-branch-recipe.v1",
        branch="high_noise",
        rank=2,
        alpha=2,
        dropout=0.0,
        target_suffixes=("to_k", "to_out.0", "to_q", "to_v"),
        optimizer_algorithm="adamw",
        optimizer_parameters=(("learning_rate", 0.01),),
        gradient_clip_norm=1.0,
        seed=7,
        tolerance=WanNumericalToleranceV1(atol=1e-6, rtol=1e-4),
    )


@pytest.mark.parametrize(
    ("reference", "candidate", "atol", "rtol", "accepted"),
    [
        (1.0, 1.0, 1e-6, 1e-6, True),
        (1.0, 1.00001, 1e-6, 1e-4, True),
        (1.0, 1.01, 1e-6, 1e-4, False),
    ],
)
def test_continuity_uses_preregistered_tolerance(
    reference: float, candidate: float, atol: float, rtol: float, accepted: bool
) -> None:
    recipe = _recipe().model_copy(
        update={"tolerance": WanNumericalToleranceV1(atol=atol, rtol=rtol)}
    )
    result = compare_wan_branch_continuity(
        {"transformer.attn.to_q": (reference,)},
        {"transformer.attn.to_q": (candidate,)},
        recipe,
    )
    assert result.accepted is accepted
    assert result.failing_tensor_names == (() if accepted else ("transformer.attn.to_q",))


def test_continuity_reports_canonical_maxima_without_raw_values() -> None:
    result = compare_wan_branch_continuity(
        {"a": (2.0, 0.0), "b": (4.0,)},
        {"a": (2.5, 0.25), "b": (4.0,)},
        _recipe(),
    )
    assert result.max_absolute_error == 0.5
    assert result.max_relative_error == 250000.0
    assert result.failing_tensor_names == ("a",)
    assert "2.5" not in result.model_dump_json()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_continuity_rejects_nonfinite_before_comparison(value: float) -> None:
    with pytest.raises(ValueError, match="wan_continuity_nonfinite"):
        compare_wan_branch_continuity({"a": (1.0,)}, {"a": (value,)}, _recipe())


def test_continuity_rejects_renamed_or_reshaped_state() -> None:
    with pytest.raises(ValueError, match="wan_continuity_structure_invalid"):
        compare_wan_branch_continuity({"a": (1.0,)}, {"b": (1.0,)}, _recipe())
    with pytest.raises(ValueError, match="wan_continuity_structure_invalid"):
        compare_wan_branch_continuity({"a": (1.0,)}, {"a": (1.0, 2.0)}, _recipe())
    with pytest.raises(ValueError, match="wan_continuity_structure_invalid"):
        compare_wan_branch_continuity(
            {"a": ((1.0, 2.0),)}, {"a": ((1.0,), (2.0,))}, _recipe()
        )


def test_continuity_rejects_equal_shape_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="wan_continuity_structure_invalid"):
        compare_wan_branch_continuity(
            {"a": torch.ones(2, dtype=torch.float32)},
            {"a": torch.ones(2, dtype=torch.float16)},
            _recipe(),
        )


def test_continuity_accepts_matching_bfloat16_next_probe() -> None:
    reference = torch.tensor(((1.0, -2.0),), dtype=torch.bfloat16)
    candidate = reference.clone()
    result = compare_wan_branch_continuity(
        {"probe.output": reference}, {"probe.output": candidate}, _recipe()
    )
    assert result.accepted
    assert result.max_absolute_error == 0.0
    assert result.max_relative_error == 0.0


def test_tolerance_digest_is_bound_to_complete_recipe() -> None:
    first = compare_wan_branch_continuity({"a": (1.0,)}, {"a": (1.0,)}, _recipe())
    changed = _recipe().model_copy(update={"seed": 8})
    second = compare_wan_branch_continuity({"a": (1.0,)}, {"a": (1.0,)}, changed)
    assert first.recipe_digest != second.recipe_digest
    assert first.tolerance_digest != second.tolerance_digest


def _digest(value: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _checkpoint() -> WanBranchCheckpointStateV2:
    cpu_rng, mps_rng = b"cpu-rng", b"mps-rng"
    names = ("transformer.attn.to_q.lora_A.weight",)
    manifest = WanBranchCheckpointManifestV2(
        schema_version="wan-branch-checkpoint.v2",
        branch="high_noise",
        selector=0,
        step=1,
        total_steps=1,
        parent_checkpoint_id=None,
        batch_digest="sha256:" + "1" * 64,
        recipe_digest="sha256:" + "2" * 64,
        component_inventory_digest="sha256:" + "3" * 64,
        adapter_tensors=(WanBranchTensorSpecV2(names[0], (2, 2), "F32"),),
        optimizer_parameter_names=names,
        optimizer_slot_names=("exp_avg", "exp_avg_sq", "step"),
        cpu_rng_digest=_digest(cpu_rng),
        mps_rng_digest=_digest(mps_rng),
        finite=True,
    )
    return WanBranchCheckpointStateV2(
        manifest=manifest,
        weights=save({names[0]: torch.ones(2, 2)}),
        optimizer={
            "schema_version": "wan-branch-optimizer-state.v2",
            "parameter_state": {
                names[0]: {"exp_avg": [0.0], "exp_avg_sq": [0.0], "step": 1}
            }
        },
        cpu_rng_state=cpu_rng,
        mps_rng_state=mps_rng,
    )


def test_v2_checkpoint_binds_adapter_optimizer_rng_and_branch() -> None:
    checkpoint = _checkpoint()
    assert validate_wan_branch_checkpoint_v2(checkpoint, checkpoint.manifest) is checkpoint


@pytest.mark.parametrize("field", ["branch", "batch_digest", "recipe_digest"])
def test_v2_checkpoint_rejects_cross_identity_before_restore(field: str) -> None:
    from dataclasses import replace

    checkpoint = _checkpoint()
    replacement = "low_noise" if field == "branch" else "sha256:" + "9" * 64
    expected = replace(checkpoint.manifest, **{field: replacement})
    with pytest.raises(ValueError, match="wan_branch_checkpoint_v2_invalid"):
        validate_wan_branch_checkpoint_v2(checkpoint, expected)


@pytest.mark.parametrize("case", ["nonfinite", "extra_optimizer", "missing_rng", "wrong_dtype"])
def test_v2_checkpoint_rejects_mutated_continuation_state(case: str) -> None:
    from dataclasses import replace

    checkpoint = _checkpoint()
    if case == "nonfinite":
        checkpoint = replace(
            checkpoint,
            weights=save(
                {"transformer.attn.to_q.lora_A.weight": torch.full((2, 2), math.nan)}
            ),
        )
    elif case == "extra_optimizer":
        checkpoint = replace(checkpoint, optimizer={**checkpoint.optimizer, "extra": 1})
    elif case == "missing_rng":
        checkpoint = replace(checkpoint, cpu_rng_state=b"")
    else:
        checkpoint = replace(
            checkpoint,
            weights=save(
                {"transformer.attn.to_q.lora_A.weight": torch.ones(2, 2).half()}
            ),
        )
    with pytest.raises(ValueError, match="wan_branch_checkpoint_v2_invalid"):
        validate_wan_branch_checkpoint_v2(checkpoint, checkpoint.manifest)


def test_v2_checkpoint_rejects_safetensors_metadata_even_when_tensor_state_is_valid() -> None:
    checkpoint = _checkpoint()
    checkpoint = checkpoint.__class__(
        manifest=checkpoint.manifest,
        weights=save(
            {"transformer.attn.to_q.lora_A.weight": torch.ones(2, 2)},
            metadata={"private_path": "/Users/example/model"},
        ),
        optimizer=checkpoint.optimizer,
        cpu_rng_state=checkpoint.cpu_rng_state,
        mps_rng_state=checkpoint.mps_rng_state,
    )
    with pytest.raises(ValueError, match="wan_branch_checkpoint_v2_invalid"):
        validate_wan_branch_checkpoint_v2(checkpoint, checkpoint.manifest)
