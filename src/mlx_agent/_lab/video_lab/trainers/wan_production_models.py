"""Immutable local-only evidence contracts for Wan MPS production canaries."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import PurePath, PureWindowsPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from video_lab.domain import FrozenModel, canonical_json
from video_lab.training_models import Sha256Digest

WanTransformerBranch = Literal["high_noise", "low_noise"]
WanCanaryPhase = Literal[
    "inventory",
    "load_trained",
    "step",
    "checkpoint",
    "close_trained",
    "memory_release_trained",
    "load_fresh_restore",
    "restore",
    "continuity",
    "close_restored",
    "memory_release_final",
]
WanComponentName = Literal["scheduler", "text_encoder", "tokenizer", "transformer", "vae"]
WanComponentManifest = tuple[WanComponentName, Sha256Digest]
WanBranchInventoryRole = Literal[
    "high_transformer",
    "low_transformer",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer_config",
    "vae",
]
WanTensorRole = Literal["activation", "optimizer", "parameter"]
WanTensorDtype = Literal["bfloat16", "float32", "int64"]
_TARGET_SUFFIXES: tuple[Literal["to_k", "to_out.0", "to_q", "to_v"], ...] = (
    "to_k",
    "to_out.0",
    "to_q",
    "to_v",
)
_COMPONENT_NAMES: tuple[WanComponentName, ...] = (
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)


class _StrictWanV2Model(FrozenModel):
    """V2 evidence base that validates construction and every model copy."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        values = {name: getattr(self, name) for name in type(self).model_fields}
        if deep:
            values = deepcopy(values)
        if update is not None:
            values.update(update)
        return type(self).model_validate(values, strict=True)


def _exact_tuple(value: object, *, label: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be an exact tuple")
    return value


def _revalidate_exact_model(value: object, expected: type[FrozenModel]) -> FrozenModel:
    if type(value) is expected:
        payload = value.model_dump(mode="python")
    elif type(value) is dict:
        payload = value
    else:
        raise ValueError("nested model must have the exact declared type")
    return expected.model_validate(payload, strict=True)


def _domain_digest(domain: str, payload: dict[str, object]) -> str:
    serialized = canonical_json({"domain": domain, "payload": payload})
    return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"


def _canonical_pairs(
    value: tuple[tuple[str, object], ...], *, label: str
) -> tuple[tuple[str, object], ...]:
    names = tuple(item[0] for item in value)
    if not value or names != tuple(sorted(set(names))):
        raise ValueError(f"{label} must be sorted and unique")
    return value


def _builtin_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("value must be an exact built-in int")
    return value


def _builtin_float(value: object) -> float:
    if type(value) is not float:
        raise ValueError("value must be an exact built-in float")
    return value


def _relative_path(value: str) -> str:
    if type(value) is not str:
        raise ValueError("path must be a string")
    path = PurePath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or windows_path.anchor != ""
        or windows_path.drive != ""
        or ".." in path.parts
        or ".." in windows_path.parts
        or value == "."
        or value.startswith("./")
        or value.endswith("/.")
        or "/./" in value
        or any(PureWindowsPath(component).is_reserved() for component in path.parts)
    ):
        raise ValueError("path must be relative and contained")
    return value


class WanNumericalToleranceV1(FrozenModel):
    schema_version: Literal["wan-numerical-tolerance.v1"] = "wan-numerical-tolerance.v1"
    atol: Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]
    rtol: Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]

    @field_validator("atol", "rtol", mode="before")
    @classmethod
    def validate_builtin_float(cls, value: object) -> float:
        return _builtin_float(value)


class WanBranchRecipeV1(FrozenModel):
    schema_version: Literal["wan-mps-branch-recipe.v1"]
    branch: WanTransformerBranch
    rank: Literal[2]
    alpha: Annotated[StrictInt, Field(gt=0)]
    dropout: Literal[0.0]  # type: ignore[valid-type]
    target_suffixes: tuple[Literal["to_k", "to_out.0", "to_q", "to_v"], ...]
    optimizer_algorithm: Literal["adamw"]
    optimizer_parameters: tuple[tuple[str, StrictInt | StrictFloat | StrictBool], ...]
    gradient_clip_norm: Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]
    seed: StrictInt
    tolerance: WanNumericalToleranceV1

    @field_validator("rank", "alpha", "seed", mode="before")
    @classmethod
    def validate_builtin_int(cls, value: object) -> int:
        return _builtin_int(value)

    @field_validator("dropout", "gradient_clip_norm", mode="before")
    @classmethod
    def validate_builtin_float(cls, value: object) -> float:
        return _builtin_float(value)

    @field_validator("target_suffixes")
    @classmethod
    def validate_target_suffixes(
        cls, value: tuple[Literal["to_k", "to_out.0", "to_q", "to_v"], ...]
    ) -> tuple[Literal["to_k", "to_out.0", "to_q", "to_v"], ...]:
        if value != _TARGET_SUFFIXES:
            raise ValueError("target suffixes must be the exact canonical attention policy")
        return value

    @field_validator("optimizer_parameters")
    @classmethod
    def validate_optimizer_parameters(
        cls, value: tuple[tuple[str, StrictInt | StrictFloat | StrictBool], ...]
    ) -> tuple[tuple[str, StrictInt | StrictFloat | StrictBool], ...]:
        _canonical_pairs(value, label="optimizer parameters")
        if any(not name for name, _ in value):
            raise ValueError("optimizer parameter names must be non-empty")
        return value

    @field_validator("optimizer_parameters", mode="before")
    @classmethod
    def validate_optimizer_builtin_values(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("optimizer parameters must be an exact tuple")
        for entry in value:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("optimizer parameter entries must be exact pairs")
            parameter_name, parameter_value = entry
            if type(parameter_name) is not str:
                raise ValueError("optimizer parameter names must use exact built-in strings")
            if type(parameter_value) not in (int, float, bool):
                raise ValueError("optimizer values must use exact built-in scalar types")
        return value


class WanComponentInventoryV1(FrozenModel):
    schema_version: Literal["wan-mps-component-inventory.v1"]
    component_manifests: tuple[WanComponentManifest, ...]

    @field_validator("component_manifests")
    @classmethod
    def validate_component_manifests(
        cls,
        value: tuple[WanComponentManifest, ...],
    ) -> tuple[WanComponentManifest, ...]:
        names = tuple(item[0] for item in value)
        if names != _COMPONENT_NAMES:
            raise ValueError("component manifests must contain every canonical component once")
        return value


class WanBranchInventoryResultV1(FrozenModel):
    """Complete inventory or path-free roles required for local acquisition."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )

    available: StrictBool
    inventory: WanComponentInventoryV1 | None
    missing_roles: tuple[WanBranchInventoryRole, ...]

    @field_validator("missing_roles", mode="before")
    @classmethod
    def validate_exact_role_tuple(cls, value: object) -> object:
        if type(value) is not tuple or any(type(role) is not str for role in value):
            raise ValueError("missing roles must be an exact tuple of logical roles")
        return value

    @field_validator("missing_roles")
    @classmethod
    def validate_canonical_roles(
        cls, value: tuple[WanBranchInventoryRole, ...]
    ) -> tuple[WanBranchInventoryRole, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("missing roles must be sorted, unique, and nonempty when unavailable")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> WanBranchInventoryResultV1:
        if self.available:
            if self.inventory is None or self.missing_roles:
                raise ValueError("available inventory result must contain one complete inventory")
        elif self.inventory is not None or not self.missing_roles:
            raise ValueError("unavailable inventory result must contain missing logical roles")
        return self


class WanProductionCanaryRequestV1(FrozenModel):
    schema_version: Literal["wan-mps-production-canary-request.v1"]
    recipe: WanBranchRecipeV1
    inventory: WanComponentInventoryV1
    device: Literal["mps"]
    dtype: Literal["bfloat16"]
    batch_digest: Sha256Digest
    relative_work_path: str = Field(min_length=1)

    @field_validator("relative_work_path")
    @classmethod
    def validate_relative_work_path(cls, value: str) -> str:
        return _relative_path(value)


class WanBranchCanaryResultV1(FrozenModel):
    schema_version: Literal["wan-mps-branch-canary-result.v1"]
    branch: WanTransformerBranch
    recipe_digest: Sha256Digest
    output_digest: Sha256Digest
    max_absolute_error: Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]
    max_relative_error: Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]

    @field_validator("max_absolute_error", "max_relative_error", mode="before")
    @classmethod
    def validate_builtin_float(cls, value: object) -> float:
        return _builtin_float(value)


class WanProductionCanaryReceiptV1(FrozenModel):
    schema_version: Literal["wan-mps-production-canary-receipt.v1"]
    branch_results: tuple[WanBranchCanaryResultV1, WanBranchCanaryResultV1]
    compilation_receipt_digest: Sha256Digest
    batch_digest: Sha256Digest
    recipe_family_digest: Sha256Digest
    dependency_versions: tuple[tuple[str, str], ...]
    hardware_identity: tuple[tuple[str, str], ...]
    cleanup_state: Literal["cleaned"]

    @model_validator(mode="after")
    def validate_receipt_bindings(self) -> WanProductionCanaryReceiptV1:
        if tuple(result.branch for result in self.branch_results) != ("high_noise", "low_noise"):
            raise ValueError("branch results must bind high_noise then low_noise exactly once")
        _canonical_pairs(self.dependency_versions, label="dependency versions")
        _canonical_pairs(self.hardware_identity, label="hardware identity")
        if any(not value for _, value in self.dependency_versions + self.hardware_identity):
            raise ValueError("receipt identity values must be non-empty")
        return self


class WanCapacityResultV1(FrozenModel):
    schema_version: Literal["wan-mps-capacity-result.v1"]
    device: Literal["mps"]
    dtype: Literal["bfloat16"]
    total_memory_bytes: Annotated[StrictInt, Field(ge=0)]
    available_memory_bytes: Annotated[StrictInt, Field(ge=0)]
    required_memory_bytes: Annotated[StrictInt, Field(ge=0)]
    fits: StrictBool

    @field_validator(
        "total_memory_bytes", "available_memory_bytes", "required_memory_bytes", mode="before"
    )
    @classmethod
    def validate_builtin_int(cls, value: object) -> int:
        return _builtin_int(value)

    @field_validator("fits", mode="before")
    @classmethod
    def validate_builtin_bool(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("fits must be an exact built-in bool")
        return value

    @model_validator(mode="after")
    def validate_capacity(self) -> WanCapacityResultV1:
        if self.available_memory_bytes > self.total_memory_bytes:
            raise ValueError("available memory cannot exceed total memory")
        if self.fits != (self.required_memory_bytes <= self.available_memory_bytes):
            raise ValueError("capacity result must match required and available memory")
        return self


class WanProductionCanaryRequestV2(_StrictWanV2Model):
    """Complete immutable authority for the two sequential branch runs."""

    schema_version: Literal["wan-mps-production-canary-request.v2"]
    recipes: tuple[WanBranchRecipeV1, WanBranchRecipeV1]
    inventories: tuple[WanComponentInventoryV1, WanComponentInventoryV1]
    compilation_receipt_digest: Sha256Digest
    batch_digest: Sha256Digest
    primary_run_id: str = Field(min_length=8)
    independent_run_id: str = Field(min_length=8)
    requested_tensor_envelope: tuple[WanTensorEnvelopeV2, ...]
    memory_release_bound_bytes: Annotated[StrictInt, Field(ge=0)]
    required_artifact_bytes: Annotated[StrictInt, Field(gt=0)]
    dependency_versions: tuple[tuple[str, str], ...]
    hardware_identity: tuple[tuple[str, str], ...]

    @field_validator(
        "recipes",
        "inventories",
        "requested_tensor_envelope",
        "dependency_versions",
        "hardware_identity",
        mode="before",
    )
    @classmethod
    def validate_exact_containers(cls, value: object) -> object:
        return _exact_tuple(value, label="v2 request container")

    @field_validator("recipes", mode="before")
    @classmethod
    def revalidate_recipes(cls, value: object) -> object:
        items = _exact_tuple(value, label="recipes")
        return tuple(_revalidate_exact_model(item, WanBranchRecipeV1) for item in items)

    @field_validator("inventories", mode="before")
    @classmethod
    def revalidate_inventories(cls, value: object) -> object:
        items = _exact_tuple(value, label="inventories")
        return tuple(_revalidate_exact_model(item, WanComponentInventoryV1) for item in items)

    @field_validator("requested_tensor_envelope", mode="before")
    @classmethod
    def revalidate_envelope(cls, value: object) -> object:
        items = _exact_tuple(value, label="requested tensor envelope")
        return tuple(_revalidate_exact_model(item, WanTensorEnvelopeV2) for item in items)

    @field_validator(
        "memory_release_bound_bytes", "required_artifact_bytes", mode="before"
    )
    @classmethod
    def validate_exact_ints(cls, value: object) -> int:
        return _builtin_int(value)

    @model_validator(mode="after")
    def validate_authority(self) -> WanProductionCanaryRequestV2:
        if tuple(recipe.branch for recipe in self.recipes) != ("high_noise", "low_noise"):
            raise ValueError("wan_canary_branch_order_invalid")
        if self.primary_run_id == self.independent_run_id:
            raise ValueError("wan_canary_independent_run_invalid")
        names = tuple(item.name for item in self.requested_tensor_envelope)
        roles = {item.role for item in self.requested_tensor_envelope}
        if (
            not names
            or names != tuple(sorted(set(names)))
            or roles != {"activation", "optimizer", "parameter"}
        ):
            raise ValueError("wan_canary_tensor_envelope_invalid")
        _canonical_pairs(self.dependency_versions, label="dependency versions")
        _canonical_pairs(self.hardware_identity, label="hardware identity")
        if any(not value for _, value in self.dependency_versions + self.hardware_identity):
            raise ValueError("wan_canary_identity_invalid")
        hardware = dict(self.hardware_identity)
        if (
            set(hardware) != {"device", "machine", "system"}
            or hardware["device"] != "mps"
            or hardware["system"] != "Darwin"
        ):
            raise ValueError("wan_canary_hardware_identity_invalid")
        return self


class WanCapacityResultV2(_StrictWanV2Model):
    schema_version: Literal["wan-mps-capacity-result.v2"]
    branch: WanTransformerBranch
    phase: WanCanaryPhase
    device: Literal["mps"]
    dtype: Literal["bfloat16"]
    requested_tensor_envelope: tuple[WanTensorEnvelopeV2, ...]
    driver_allocated_before_bytes: Annotated[StrictInt, Field(ge=0)]
    driver_allocated_after_failure_bytes: Annotated[StrictInt, Field(ge=0)]
    driver_allocated_after_cleanup_bytes: Annotated[StrictInt, Field(ge=0)]
    recommended_maximum_bytes: Annotated[StrictInt, Field(ge=0)]
    required_memory_bytes: Annotated[StrictInt, Field(gt=0)]
    recipe_digest: Sha256Digest
    inventory_digest: Sha256Digest
    fits: Literal[False]
    classification: Literal["recognized_mps_allocation_failure"]

    @field_validator("requested_tensor_envelope", mode="before")
    @classmethod
    def revalidate_envelope(cls, value: object) -> object:
        items = _exact_tuple(value, label="requested tensor envelope")
        return tuple(_revalidate_exact_model(item, WanTensorEnvelopeV2) for item in items)

    @field_validator(
        "driver_allocated_before_bytes",
        "driver_allocated_after_failure_bytes",
        "driver_allocated_after_cleanup_bytes",
        "recommended_maximum_bytes",
        "required_memory_bytes",
        mode="before",
    )
    @classmethod
    def validate_exact_ints(cls, value: object) -> int:
        return _builtin_int(value)

    @field_validator("fits", mode="before")
    @classmethod
    def validate_exact_false(cls, value: object) -> bool:
        if type(value) is not bool or value:
            raise ValueError("fits must be exact false")
        return value

    @model_validator(mode="after")
    def validate_capacity(self) -> WanCapacityResultV2:
        names = tuple(item.name for item in self.requested_tensor_envelope)
        roles = {item.role for item in self.requested_tensor_envelope}
        if (
            not names
            or names != tuple(sorted(set(names)))
            or roles != {"activation", "optimizer", "parameter"}
        ):
            raise ValueError("wan_capacity_envelope_invalid")
        if self.required_memory_bytes != wan_required_tensor_bytes(
            self.requested_tensor_envelope
        ):
            raise ValueError("wan_capacity_required_bytes_invalid")
        return self


class WanTensorEnvelopeV2(_StrictWanV2Model):
    schema_version: Literal["wan-mps-tensor-envelope.v2"] = "wan-mps-tensor-envelope.v2"
    name: str = Field(min_length=1)
    role: WanTensorRole
    shape: tuple[StrictInt, ...]
    dtype: WanTensorDtype
    copies: Annotated[StrictInt, Field(gt=0)]

    @field_validator("copies", mode="before")
    @classmethod
    def validate_exact_copies(cls, value: object) -> int:
        return _builtin_int(value)

    @field_validator("shape", mode="before")
    @classmethod
    def validate_shape(cls, value: object) -> object:
        if (
            type(value) is not tuple
            or not value
            or any(type(item) is not int or item <= 0 for item in value)
        ):
            raise ValueError("wan_tensor_envelope_shape_invalid")
        return value


class WanProductionCanaryUnavailableV2(_StrictWanV2Model):
    schema_version: Literal["wan-mps-production-canary-unavailable.v2"]
    branch: WanTransformerBranch
    missing_roles: tuple[WanBranchInventoryRole, ...]

    @field_validator("missing_roles", mode="before")
    @classmethod
    def validate_exact_roles(cls, value: object) -> object:
        roles = _exact_tuple(value, label="missing roles")
        if any(type(role) is not str for role in roles):
            raise ValueError("wan_canary_missing_roles_invalid")
        return roles

    @field_validator("missing_roles")
    @classmethod
    def validate_roles(
        cls, value: tuple[WanBranchInventoryRole, ...]
    ) -> tuple[WanBranchInventoryRole, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("wan_canary_missing_roles_invalid")
        return value


class WanBranchNumericalProbeV2(_StrictWanV2Model):
    schema_version: Literal["wan-mps-branch-numerical-probe.v2"] = (
        "wan-mps-branch-numerical-probe.v2"
    )
    branch: WanTransformerBranch
    recipe_digest: Sha256Digest
    tensor_name: Literal["probe.output"]
    shape: tuple[StrictInt, ...]
    dtype: Literal["bfloat16", "float32"]
    values: tuple[StrictFloat, ...]

    @field_validator("shape", mode="before")
    @classmethod
    def validate_shape(cls, value: object) -> object:
        items = _exact_tuple(value, label="probe shape")
        if not items or any(type(item) is not int or item <= 0 for item in items):
            raise ValueError("wan_canary_probe_invalid")
        return items

    @field_validator("values", mode="before")
    @classmethod
    def validate_values(cls, value: object) -> object:
        items = _exact_tuple(value, label="probe values")
        if not items or any(type(item) is not float for item in items):
            raise ValueError("wan_canary_probe_invalid")
        return items

    @model_validator(mode="after")
    def validate_probe(self) -> WanBranchNumericalProbeV2:
        if math.prod(self.shape) != len(self.values) or any(
            not math.isfinite(value) for value in self.values
        ):
            raise ValueError("wan_canary_probe_invalid")
        return self


class WanProductionCanaryReceiptV2(_StrictWanV2Model):
    schema_version: Literal["wan-mps-production-canary-receipt.v2"]
    branch_results: tuple[WanBranchCanaryResultV1, WanBranchCanaryResultV1]
    branch_probes: tuple[WanBranchNumericalProbeV2, WanBranchNumericalProbeV2]
    compilation_receipt_digest: Sha256Digest
    batch_digest: Sha256Digest
    inventory_family_digest: Sha256Digest
    recipe_family_digest: Sha256Digest
    primary_run_id: str = Field(min_length=8)
    independent_run_id: str = Field(min_length=8)
    dependency_versions: tuple[tuple[str, str], ...]
    hardware_identity: tuple[tuple[str, str], ...]
    memory_release_bytes: tuple[tuple[str, StrictInt], ...]
    cleanup_state: Literal["cleaned"]

    @field_validator(
        "branch_results",
        "branch_probes",
        "dependency_versions",
        "hardware_identity",
        "memory_release_bytes",
        mode="before",
    )
    @classmethod
    def validate_exact_containers(cls, value: object) -> object:
        return _exact_tuple(value, label="v2 receipt container")

    @field_validator("branch_results", mode="before")
    @classmethod
    def revalidate_results(cls, value: object) -> object:
        items = _exact_tuple(value, label="branch results")
        return tuple(_revalidate_exact_model(item, WanBranchCanaryResultV1) for item in items)

    @field_validator("branch_probes", mode="before")
    @classmethod
    def revalidate_probes(cls, value: object) -> object:
        items = _exact_tuple(value, label="branch probes")
        return tuple(_revalidate_exact_model(item, WanBranchNumericalProbeV2) for item in items)

    @model_validator(mode="after")
    def validate_receipt(self) -> WanProductionCanaryReceiptV2:
        if tuple(result.branch for result in self.branch_results) != ("high_noise", "low_noise"):
            raise ValueError("wan_canary_branch_order_invalid")
        if tuple(probe.branch for probe in self.branch_probes) != ("high_noise", "low_noise"):
            raise ValueError("wan_canary_branch_order_invalid")
        if any(
            probe.recipe_digest != result.recipe_digest
            for probe, result in zip(self.branch_probes, self.branch_results, strict=True)
        ):
            raise ValueError("wan_canary_probe_invalid")
        if self.primary_run_id == self.independent_run_id:
            raise ValueError("wan_canary_independent_run_invalid")
        if tuple(name for name, _ in self.memory_release_bytes) != (
            "high_noise.trained",
            "high_noise.restored",
            "low_noise.trained",
            "low_noise.restored",
        ):
            raise ValueError("wan_canary_memory_release_invalid")
        if any(value < 0 for _, value in self.memory_release_bytes):
            raise ValueError("wan_canary_memory_release_invalid")
        _canonical_pairs(self.dependency_versions, label="dependency versions")
        _canonical_pairs(self.hardware_identity, label="hardware identity")
        hardware = dict(self.hardware_identity)
        if (
            set(hardware) != {"device", "machine", "system"}
            or hardware["device"] != "mps"
            or hardware["system"] != "Darwin"
        ):
            raise ValueError("wan_canary_hardware_identity_invalid")
        return self


def wan_branch_recipe_digest(recipe: WanBranchRecipeV1) -> Sha256Digest:
    return _domain_digest("wan-mps-branch-recipe-digest.v1", recipe.model_dump(mode="json"))


def wan_inventory_digest(inventory: WanComponentInventoryV1) -> Sha256Digest:
    return _domain_digest(
        "wan-mps-component-inventory-digest.v1", inventory.model_dump(mode="json")
    )


def wan_inventory_family_digest(
    inventories: tuple[WanComponentInventoryV1, WanComponentInventoryV1],
) -> Sha256Digest:
    return _domain_digest(
        "wan-mps-inventory-family-digest.v2",
        {"inventories": [item.model_dump(mode="json") for item in inventories]},
    )


def wan_recipe_family_digest(
    recipes: tuple[WanBranchRecipeV1, WanBranchRecipeV1],
) -> Sha256Digest:
    return _domain_digest(
        "wan-mps-recipe-family-digest.v2",
        {"recipes": [item.model_dump(mode="json") for item in recipes]},
    )


def wan_required_tensor_bytes(envelope: tuple[WanTensorEnvelopeV2, ...]) -> int:
    widths = {"bfloat16": 2, "float32": 4, "int64": 8}
    total = 0
    for item in envelope:
        elements = 1
        for extent in item.shape:
            elements *= extent
        total += elements * widths[item.dtype] * item.copies
    return total


def wan_production_receipt_id(receipt: WanProductionCanaryReceiptV1) -> Sha256Digest:
    return _domain_digest(
        "wan-mps-production-canary-receipt-id.v1", receipt.model_dump(mode="json")
    )


def wan_production_receipt_v2_id(receipt: WanProductionCanaryReceiptV2) -> Sha256Digest:
    return _domain_digest(
        "wan-mps-production-canary-receipt-id.v2", receipt.model_dump(mode="json")
    )


__all__ = (
    "WanBranchCanaryResultV1",
    "WanBranchNumericalProbeV2",
    "WanBranchInventoryResultV1",
    "WanBranchInventoryRole",
    "WanBranchRecipeV1",
    "WanCanaryPhase",
    "WanCapacityResultV1",
    "WanCapacityResultV2",
    "WanComponentInventoryV1",
    "WanNumericalToleranceV1",
    "WanProductionCanaryReceiptV1",
    "WanProductionCanaryReceiptV2",
    "WanProductionCanaryUnavailableV2",
    "WanProductionCanaryRequestV1",
    "WanProductionCanaryRequestV2",
    "WanTransformerBranch",
    "WanTensorEnvelopeV2",
    "wan_branch_recipe_digest",
    "wan_inventory_digest",
    "wan_inventory_family_digest",
    "wan_production_receipt_id",
    "wan_production_receipt_v2_id",
    "wan_recipe_family_digest",
    "wan_required_tensor_bytes",
)
