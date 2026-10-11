"""Contract tests for local Wan MPS production evidence models."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from video_lab.trainers.wan_production_models import (
    WanBranchCanaryResultV1,
    WanBranchInventoryResultV1,
    WanBranchRecipeV1,
    WanCapacityResultV1,
    WanComponentInventoryV1,
    WanNumericalToleranceV1,
    WanProductionCanaryReceiptV1,
    WanProductionCanaryRequestV1,
    wan_branch_recipe_digest,
    wan_inventory_digest,
    wan_production_receipt_id,
)


def _digest(value: str) -> str:
    return f"sha256:{value * 64}"


def _recipe(branch: str = "high_noise") -> WanBranchRecipeV1:
    return WanBranchRecipeV1(
        schema_version="wan-mps-branch-recipe.v1",
        branch=branch,
        rank=2,
        alpha=2,
        dropout=0.0,
        target_suffixes=("to_k", "to_out.0", "to_q", "to_v"),
        optimizer_algorithm="adamw",
        optimizer_parameters=(("learning_rate", 1e-4), ("weight_decay", 0.01)),
        gradient_clip_norm=1.0,
        seed=7,
        tolerance=WanNumericalToleranceV1(atol=1e-5, rtol=1e-4),
    )


def _inventory() -> WanComponentInventoryV1:
    return WanComponentInventoryV1(
        schema_version="wan-mps-component-inventory.v1",
        component_manifests=(
            ("scheduler", _digest("a")),
            ("text_encoder", _digest("b")),
            ("tokenizer", _digest("c")),
            ("transformer", _digest("d")),
            ("vae", _digest("e")),
        ),
    )


def test_available_branch_inventory_result_requires_one_complete_inventory() -> None:
    result = WanBranchInventoryResultV1(
        available=True,
        inventory=_inventory(),
        missing_roles=(),
    )

    assert result.inventory == _inventory()
    assert result.missing_roles == ()
    for invalid in (
        {"available": True, "inventory": None, "missing_roles": ()},
        {"available": True, "inventory": _inventory(), "missing_roles": ("tokenizer",)},
        {"available": 1, "inventory": _inventory(), "missing_roles": ()},
    ):
        with pytest.raises(ValidationError):
            WanBranchInventoryResultV1(**invalid)  # type: ignore[arg-type]


def test_unavailable_branch_inventory_result_requires_canonical_closed_roles() -> None:
    result = WanBranchInventoryResultV1(
        available=False,
        inventory=None,
        missing_roles=("high_transformer", "tokenizer", "transformer_config"),
    )

    assert result.inventory is None
    assert result.missing_roles == (
        "high_transformer",
        "tokenizer",
        "transformer_config",
    )
    for missing_roles in (
        (),
        ("tokenizer", "high_transformer"),
        ("tokenizer", "tokenizer"),
        ("/private/model",),
        ["tokenizer"],
    ):
        with pytest.raises(ValidationError) as raised:
            WanBranchInventoryResultV1(
                available=False,
                inventory=None,
                missing_roles=missing_roles,  # type: ignore[arg-type]
            )
        assert "/private/model" not in str(raised.value)


def test_branch_inventory_result_is_frozen_forbids_locations_and_hides_inputs() -> None:
    result = WanBranchInventoryResultV1(
        available=False,
        inventory=None,
        missing_roles=("scheduler",),
    )
    with pytest.raises(ValidationError):
        result.model_copy(update={"missing_roles": ()})
    with pytest.raises(ValidationError) as raised:
        result.model_copy(update={"location": "/private/model"})
    assert "/private/model" not in str(raised.value)
    assert "location" not in result.model_dump(mode="json")


def _request(branch: str = "high_noise") -> WanProductionCanaryRequestV1:
    return WanProductionCanaryRequestV1(
        schema_version="wan-mps-production-canary-request.v1",
        recipe=_recipe(branch),
        inventory=_inventory(),
        device="mps",
        dtype="bfloat16",
        batch_digest=_digest("f"),
        relative_work_path="canary/high-noise",
    )


def _result(branch: str) -> WanBranchCanaryResultV1:
    return WanBranchCanaryResultV1(
        schema_version="wan-mps-branch-canary-result.v1",
        branch=branch,
        recipe_digest=wan_branch_recipe_digest(_recipe(branch)),
        output_digest=_digest("1" if branch == "high_noise" else "2"),
        max_absolute_error=1e-6,
        max_relative_error=1e-5,
    )


def _receipt() -> WanProductionCanaryReceiptV1:
    return WanProductionCanaryReceiptV1(
        schema_version="wan-mps-production-canary-receipt.v1",
        branch_results=(_result("high_noise"), _result("low_noise")),
        compilation_receipt_digest=_digest("3"),
        batch_digest=_digest("f"),
        recipe_family_digest=_digest("4"),
        dependency_versions=(("diffusers", "0.32.2"), ("torch", "2.7.0")),
        hardware_identity=(("chip", "Apple M4 Max"), ("device", "mps")),
        cleanup_state="cleaned",
    )


def test_branch_recipe_is_exact_rank_two_attention_policy() -> None:
    recipe = _recipe()

    assert recipe.branch == "high_noise"
    with pytest.raises(ValidationError):
        recipe.model_copy(update={"rank": 4})


@pytest.mark.parametrize(
    ("field", "invalid_values"),
    [
        ("rank", (2.0, True, np.int64(2), np.float64(2.0), "2")),
        ("dropout", (0, False, np.int64(0), np.float64(0.0), "0.0")),
        ("alpha", (2.0, True, np.int64(2), np.float64(2.0), "2")),
        ("gradient_clip_norm", (1, True, np.int64(1), np.float64(1.0), "1.0")),
        ("seed", (7.0, True, np.int64(7), np.float64(7.0), "7")),
    ],
)
def test_recipe_requires_exact_builtin_numeric_types_before_canonicalization(
    field: str, invalid_values: tuple[object, ...]
) -> None:
    recipe = _recipe()
    for value in invalid_values:
        with pytest.raises(ValidationError):
            recipe.model_copy(update={field: value})


def test_recipe_preserves_authentic_builtin_rank_and_dropout_identities() -> None:
    recipe = _recipe()
    assert type(recipe.rank) is int
    assert recipe.rank == 2
    assert type(recipe.dropout) is float
    assert recipe.dropout == 0.0


def test_contracts_are_frozen_and_forbid_unknown_keys() -> None:
    recipe = _recipe()
    with pytest.raises(ValidationError):
        WanNumericalToleranceV1(atol=1e-5, rtol=1e-4, unknown=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        recipe.model_copy(update={"unknown": True})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", True),
        ("alpha", True),
        ("seed", True),
        ("gradient_clip_norm", True),
    ],
)
def test_recipe_rejects_bool_values_for_numeric_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _recipe().model_copy(update={field: value})


@pytest.mark.parametrize("field", ["atol", "rtol"])
@pytest.mark.parametrize("value", [1, True, np.int64(1), np.float64(1e-5), "0.00001"])
def test_tolerance_requires_exact_builtin_float_values(field: str, value: object) -> None:
    values: dict[str, object] = {"atol": 1e-5, "rtol": 1e-4}
    values[field] = value
    with pytest.raises(ValidationError):
        WanNumericalToleranceV1(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_tolerance_rejects_non_positive_or_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        WanNumericalToleranceV1(atol=value, rtol=1e-4)
    with pytest.raises(ValidationError):
        WanNumericalToleranceV1(atol=1e-5, rtol=value)


def test_recipe_rejects_noncanonical_target_and_optimizer_tuples() -> None:
    with pytest.raises(ValidationError):
        _recipe().model_copy(update={"target_suffixes": ("to_q", "to_k", "to_v", "to_out.0")})
    with pytest.raises(ValidationError):
        _recipe().model_copy(update={"target_suffixes": ("to_k", "to_k")})
    with pytest.raises(ValidationError):
        _recipe().model_copy(
            update={"optimizer_parameters": (("weight_decay", 0.01), ("learning_rate", 1e-4))}
        )
    with pytest.raises(ValidationError):
        _recipe().model_copy(
            update={"optimizer_parameters": (("learning_rate", 1e-4), ("learning_rate", 2e-4))}
        )


@pytest.mark.parametrize("value", [np.int64(1), np.float64(1e-4), "0.0001"])
def test_optimizer_numeric_evidence_requires_exact_builtin_types(value: object) -> None:
    with pytest.raises(ValidationError):
        _recipe().model_copy(
            update={"optimizer_parameters": (("learning_rate", value), ("weight_decay", 0.01))}
        )


@pytest.mark.parametrize(
    "parameters",
    [
        [("learning_rate", 1e-4), ("weight_decay", 0.01)],
        (["learning_rate", 1e-4], ("weight_decay", 0.01)),
        ({"learning_rate": 1e-4}, ("weight_decay", 0.01)),
        (("learning_rate", 1e-4, "extra"), ("weight_decay", 0.01)),
        (("learning_rate",), ("weight_decay", 0.01)),
        ((np.str_("learning_rate"), 1e-4), ("weight_decay", 0.01)),
        (("learning_rate", np.float64(1e-4)), ("weight_decay", 0.01)),
        (("learning_rate", 1e-4), ("weight_decay", np.int64(1))),
        (("learning_rate", "0.0001"), ("weight_decay", 0.01)),
    ],
)
def test_optimizer_parameters_require_exact_tuple_entries_before_pydantic_conversion(
    parameters: object,
) -> None:
    recipe = _recipe()
    with pytest.raises(ValidationError):
        recipe.model_copy(update={"optimizer_parameters": parameters})


def test_optimizer_parameters_retain_exact_tuple_and_scalar_identities() -> None:
    recipe = _recipe()
    assert type(recipe.optimizer_parameters) is tuple
    for name, value in recipe.optimizer_parameters:
        assert type(name) is str
        assert type(value) is float


def test_inventory_requires_each_canonical_component_manifest() -> None:
    inventory = _inventory()
    with pytest.raises(ValidationError):
        inventory.model_copy(update={"component_manifests": inventory.component_manifests[:-1]})


def test_request_requires_bfloat16_mps_and_relative_work_path() -> None:
    request = _request()
    invalid_values = (
        ("device", "cpu"),
        ("dtype", "float16"),
        ("relative_work_path", "/private/tmp"),
        ("relative_work_path", "\\private\\tmp"),
        ("relative_work_path", "C:\\private\\tmp"),
        ("relative_work_path", "C:private\\tmp"),
        ("relative_work_path", "\\\\server\\share\\tmp"),
        ("relative_work_path", "\\\\?\\C:\\private\\tmp"),
        ("relative_work_path", "\\\\.\\PhysicalDrive0"),
        ("relative_work_path", "..\\private\\tmp"),
        ("relative_work_path", "../private/tmp"),
        ("relative_work_path", ""),
        ("relative_work_path", "."),
        ("relative_work_path", "canary/./high-noise"),
        ("relative_work_path", "canary\\high-noise"),
    )
    for field, value in invalid_values:
        with pytest.raises(ValidationError):
            request.model_copy(update={field: value})


@pytest.mark.parametrize(
    "component",
    (
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    ),
)
@pytest.mark.parametrize("suffix", ("", ".weights", ".", " ", " .manifest"))
def test_request_rejects_reserved_windows_dos_device_components(
    component: str, suffix: str
) -> None:
    request = _request()
    value = f"canary/{component.lower()}{suffix}/artifact"
    with pytest.raises(ValidationError):
        request.model_copy(update={"relative_work_path": value})


def test_request_allows_ordinary_portable_relative_components() -> None:
    request = _request()
    assert request.model_copy(
        update={"relative_work_path": "control/printer/com10/lpt10/model.v1"}
    ).relative_work_path == "control/printer/com10/lpt10/model.v1"


@pytest.mark.parametrize(
    "relative_work_path",
    (
        "canary/CON:stream/artifact",
        "canary/conin$/artifact",
        "canary/ConOut$/artifact",
        "canary/COM¹/artifact",
        "canary/com².txt/artifact",
        "canary/LPT³ /artifact",
        "canary/nested/COM¹/artifact",
        "canary/nested/CON:stream/artifact",
    ),
)
def test_request_rejects_every_component_reserved_by_pure_windows_path(
    relative_work_path: str,
) -> None:
    with pytest.raises(ValidationError):
        _request().model_copy(update={"relative_work_path": relative_work_path})


def test_request_keeps_nonreserved_component_names_portable() -> None:
    relative_work_path = "context/com10/lpt10/console/artifact"
    updated = _request().model_copy(update={"relative_work_path": relative_work_path})
    assert updated.relative_work_path == relative_work_path


def test_recipe_digest_is_domain_separated_and_sensitive_to_scientific_fields() -> None:
    recipe = _recipe()
    digest = wan_branch_recipe_digest(recipe)
    assert digest.startswith("sha256:")

    def changed_optimizer(name: str, value: float) -> WanBranchRecipeV1:
        parameters = tuple(
            (parameter_name, value if parameter_name == name else parameter_value)
            for parameter_name, parameter_value in recipe.optimizer_parameters
        )
        return recipe.model_copy(update={"optimizer_parameters": parameters})

    changes = (
        recipe.model_copy(update={"branch": "low_noise"}),
        recipe.model_copy(update={"alpha": 4}),
        recipe.model_copy(update={"gradient_clip_norm": 2.0}),
        recipe.model_copy(update={"seed": 8}),
        recipe.model_copy(update={"tolerance": WanNumericalToleranceV1(atol=2e-5, rtol=1e-4)}),
        recipe.model_copy(update={"tolerance": WanNumericalToleranceV1(atol=1e-5, rtol=2e-4)}),
        changed_optimizer("learning_rate", 2e-4),
        changed_optimizer("weight_decay", 0.02),
    )
    for changed in changes:
        assert digest != wan_branch_recipe_digest(changed)


@pytest.mark.parametrize(
    ("field", "alternative"),
    [
        ("schema_version", "wan-mps-branch-recipe.v2"),
        ("rank", 4),
        ("dropout", 0.1),
        ("target_suffixes", ("to_q", "to_k", "to_v", "to_out.0")),
        ("optimizer_algorithm", "sgd"),
    ],
)
def test_recipe_rejects_fixed_identity_alternatives_before_digest_creation(
    field: str, alternative: object
) -> None:
    with pytest.raises(ValidationError):
        _recipe().model_copy(update={field: alternative})


def test_recipe_digest_matches_the_canonical_full_payload_golden_value() -> None:
    assert wan_branch_recipe_digest(_recipe()) == (
        "sha256:8cd047a1d8360ee09c7922c9489d7a2b09da8d9d829ed7d5867981c28ad20f5b"
    )


def test_inventory_digest_changes_with_every_component_manifest() -> None:
    inventory = _inventory()
    assert wan_inventory_digest(inventory).startswith("sha256:")
    for index, (role, _) in enumerate(inventory.component_manifests):
        manifests = list(inventory.component_manifests)
        manifests[index] = (role, _digest(str(index)))
        changed = inventory.model_copy(update={"component_manifests": tuple(manifests)})
        assert wan_inventory_digest(inventory) != wan_inventory_digest(changed)


@pytest.mark.parametrize("field", ["max_absolute_error", "max_relative_error"])
@pytest.mark.parametrize("value", [1, True, np.int64(1), np.float64(1e-6), "0.000001"])
def test_branch_result_numeric_evidence_requires_exact_builtin_float_types(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        _result("high_noise").model_copy(update={field: value})


def test_receipt_binds_one_result_per_branch_in_canonical_order() -> None:
    receipt = _receipt()
    assert wan_production_receipt_id(receipt).startswith("sha256:")
    with pytest.raises(ValidationError):
        receipt.model_copy(update={"branch_results": tuple(reversed(receipt.branch_results))})
    with pytest.raises(ValidationError):
        receipt.model_copy(
            update={"branch_results": (_result("high_noise"), _result("high_noise"))}
        )


def test_receipt_requires_canonical_dependency_hardware_and_cleanup_evidence() -> None:
    receipt = _receipt()
    with pytest.raises(ValidationError):
        receipt.model_copy(
            update={"dependency_versions": tuple(reversed(receipt.dependency_versions))}
        )
    with pytest.raises(ValidationError):
        receipt.model_copy(update={"hardware_identity": ()})
    with pytest.raises(ValidationError):
        receipt.model_copy(update={"cleanup_state": "pending"})


def test_capacity_requires_nonnegative_memory_and_mps_bfloat16_placement() -> None:
    capacity = WanCapacityResultV1(
        schema_version="wan-mps-capacity-result.v1",
        device="mps",
        dtype="bfloat16",
        total_memory_bytes=64,
        available_memory_bytes=32,
        required_memory_bytes=16,
        fits=True,
    )
    assert capacity.fits is True
    with pytest.raises(ValidationError):
        capacity.model_copy(update={"available_memory_bytes": -1})
    with pytest.raises(ValidationError):
        capacity.model_copy(update={"device": "cuda"})

    for field in ("total_memory_bytes", "available_memory_bytes", "required_memory_bytes"):
        for value in (16.0, True, np.int64(16), np.float64(16.0), "16"):
            with pytest.raises(ValidationError):
                capacity.model_copy(update={field: value})


def test_public_contract_symbols_are_exported_from_trainers() -> None:
    from video_lab import trainers

    for symbol in (
        "WanBranchCanaryResultV1",
        "WanBranchInventoryResultV1",
        "WanBranchRecipeV1",
        "WanCapacityResultV1",
        "WanComponentInventoryV1",
        "WanNumericalToleranceV1",
        "WanProductionCanaryReceiptV1",
        "WanProductionCanaryRequestV1",
        "WanTransformerBranch",
        "wan_branch_recipe_digest",
        "wan_inventory_digest",
        "wan_production_receipt_id",
    ):
        assert hasattr(trainers, symbol)
