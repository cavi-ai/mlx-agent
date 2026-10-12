"""One-branch, one-update Wan attention-LoRA production runtime."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, StrictFloat, StrictInt, field_validator

from video_lab.domain import FrozenModel, canonical_json
from video_lab.trainer_protocol import LoRAConfiguration
from video_lab.trainers.wan_branch_loader import WanLoadedBranch
from video_lab.trainers.wan_checkpoint import (
    WanBranchCheckpointManifestV2,
    WanBranchCheckpointStateV2,
    WanBranchRestoreReceiptV2,
    WanBranchTensorSpecV2,
    WanCheckpointState,
    validate_wan_branch_checkpoint_v2,
)
from video_lab.trainers.wan_mps import (
    WanParameterObservation,
    WanRuntimeCheckpointState,
    WanRuntimeRestoreReceipt,
    WanTrainingSourceLease,
    WanTrainingSources,
)
from video_lab.trainers.wan_production_models import (
    WanBranchRecipeV1,
    WanTransformerBranch,
    wan_branch_recipe_digest,
    wan_inventory_digest,
)

_ADAPTER_NAME = "character"
_LORA_MARKERS = (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.")
_ADAMW_SLOT_NAMES = ("exp_avg", "exp_avg_sq", "step")


def _is_mps_device(value: object) -> bool:
    device_type = getattr(value, "type", None)
    if device_type is not None:
        return type(device_type) is str and device_type == "mps"
    if type(value) is not str:
        return False
    if value == "mps":
        return True
    prefix, separator, index = value.partition(":")
    return prefix == "mps" and separator == ":" and index.isdigit()


def _is_bfloat16_dtype(value: object) -> bool:
    return str(value).removeprefix("torch.") == "bfloat16"


class WanMpsOutOfMemoryError(RuntimeError):
    """Normalized boundary for a recognized local MPS allocation failure."""


def is_recognized_mps_oom(error: BaseException) -> bool:
    """Return whether an exception is the normalized or native MPS OOM condition."""

    if isinstance(error, WanMpsOutOfMemoryError):
        return True
    message = str(error).lower()
    return isinstance(error, RuntimeError) and "mps" in message and "out of memory" in message


class WanBranchStepEvidence(FrozenModel):
    """Strict evidence from exactly one complete branch optimizer update."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    branch: WanTransformerBranch
    selector_count: StrictInt = Field(gt=0)
    loss: StrictFloat
    gradient_norm: StrictFloat = Field(ge=0)
    changed_adapter_inventory: tuple[WanParameterObservation, ...]
    unchanged_base_inventory: tuple[WanParameterObservation, ...]
    optimizer_update_count: Literal[1]
    memory_observations: tuple[tuple[str, StrictInt], ...]

    @field_validator("loss", "gradient_norm")
    @classmethod
    def finite_measurements(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("wan_branch_numeric_nonfinite")
        return value

    @field_validator("memory_observations")
    @classmethod
    def canonical_memory(
        cls, value: tuple[tuple[str, StrictInt], ...]
    ) -> tuple[tuple[str, StrictInt], ...]:
        names = tuple(name for name, _ in value)
        if not value or names != tuple(sorted(set(names))):
            raise ValueError("wan_branch_memory_invalid")
        return value


class WanMpsBranchOperations:
    """Train exactly one already-loaded transformer with one attention LoRA update."""

    def __init__(
        self,
        *,
        loaded_branch: WanLoadedBranch,
        recipe: WanBranchRecipeV1,
        batch: Mapping[str, Any],
        batch_digest: str | None = None,
        importer: Callable[[str], object],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        if loaded_branch.branch != recipe.branch or loaded_branch.transformer is None:
            raise ValueError("wan_branch_recipe_invalid")
        self.loaded_branch = loaded_branch
        self.recipe = recipe
        self.transformer = cast(Any, loaded_branch.transformer)
        self.batch = dict(batch)
        self.batch_digest = batch_digest
        self.importer = importer
        self.torch = cast(Any, importer("torch"))
        self.cancelled = cancelled
        self.optimizer: Any = None
        self.optimizer_updates = 0
        self._baseline: tuple[WanParameterObservation, ...] | None = None
        self._injected: tuple[WanParameterObservation, ...] | None = None
        self._last_evidence: WanBranchStepEvidence | None = None
        self._restored: WanRuntimeRestoreReceipt | None = None
        self._validate_batch()

    @classmethod
    def from_sources(
        cls,
        *,
        loaded_branch: WanLoadedBranch,
        recipe: WanBranchRecipeV1,
        sources: WanTrainingSources,
        importer: Callable[[str], object],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> WanMpsBranchOperations:
        """Decode the already-authorized precomputed batch without model-side components."""

        from video_lab.trainers.wan_diffusers_runtime import validate_wan_batch_artifact

        by_name = {item.name: item.path for item in sources.files}
        try:
            manifest = by_name["batch_manifest"]
            tensors = by_name["batch_tensors"]
        except KeyError:
            raise ValueError("wan_training_batch_invalid") from None
        payload = validate_wan_batch_artifact(manifest, tensors)
        torch = cast(Any, importer("torch"))
        decoded = cast(Any, importer("safetensors.torch")).load(payload)
        batch = {
            name: tensor.to(
                device="mps",
                dtype=(
                    torch.int64
                    if name == "transformer_selector"
                    else torch.float32
                    if name == "timestep"
                    else torch.bfloat16
                ),
            )
            for name, tensor in decoded.items()
        }
        return cls(
            loaded_branch=loaded_branch,
            recipe=recipe,
            batch=batch,
            batch_digest=sources.source_digest,
            importer=importer,
            cancelled=cancelled,
        )

    @classmethod
    def from_source_lease(
        cls,
        *,
        loaded_branch: WanLoadedBranch,
        recipe: WanBranchRecipeV1,
        source_lease: WanTrainingSourceLease,
        importer: Callable[[str], object],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> WanMpsBranchOperations:
        """Decode only bytes read from the retained source descriptors."""

        from video_lab.trainers.wan_batch import validate_wan_batch_envelope

        if type(source_lease) is not WanTrainingSourceLease:
            raise ValueError("wan_training_batch_invalid")
        by_name = {item.name: item for item in source_lease.sources.files}
        try:
            manifest_source = by_name["batch_manifest"]
            tensor_source = by_name["batch_tensors"]
        except KeyError:
            raise ValueError("wan_training_batch_invalid") from None
        try:
            manifest_payload = source_lease.read_file_bytes(manifest_source)
            tensor_payload = source_lease.read_file_bytes(tensor_source)
            validate_wan_batch_envelope(manifest_payload, tensor_payload)
            decoded = cast(Any, importer("safetensors.torch")).load(tensor_payload)
            source_lease.reverify()
        except ValueError as error:
            if str(error) == "wan_source_invalid":
                raise
            raise ValueError("wan_training_batch_invalid") from None
        torch = cast(Any, importer("torch"))
        batch = {
            name: tensor.to(
                device="mps",
                dtype=(
                    torch.int64
                    if name == "transformer_selector"
                    else torch.float32
                    if name == "timestep"
                    else torch.bfloat16
                ),
            )
            for name, tensor in decoded.items()
        }
        source_lease.reverify()
        return cls(
            loaded_branch=loaded_branch,
            recipe=recipe,
            batch=batch,
            batch_digest=tensor_source.digest,
            importer=importer,
            cancelled=cancelled,
        )

    @staticmethod
    def selector_for_branch(branch: object) -> int:
        if branch == "high_noise":
            return 0
        if branch == "low_noise":
            return 1
        raise ValueError("wan_branch_recipe_invalid")

    def eligible_lora_modules(self) -> tuple[str, ...]:
        found: list[str] = []
        suffix_counts = {suffix: 0 for suffix in self.recipe.target_suffixes}
        for name, module in self.transformer.named_modules():
            matches = tuple(
                suffix for suffix in self.recipe.target_suffixes if _matches_suffix(name, suffix)
            )
            if len(matches) > 1:
                raise ValueError("wan_branch_targets_invalid")
            if matches and hasattr(module, "weight"):
                suffix_counts[matches[0]] += 1
                found.append(f"transformer.{name}")
        if not found or any(count == 0 for count in suffix_counts.values()):
            raise ValueError("wan_branch_targets_invalid")
        result = tuple(sorted(found))
        if len(result) != len(set(result)):
            raise ValueError("wan_branch_targets_invalid")
        return result

    def inject_lora(
        self,
        *,
        targets: tuple[str, ...],
        configuration: LoRAConfiguration,
        seed: int,
        device: str,
        dtype: str,
    ) -> None:
        with self._rng_fence(seed):
            self._inject_lora(
                targets=targets,
                configuration=configuration,
                seed=seed,
                device=device,
                dtype=dtype,
            )

    def _inject_lora(
        self,
        *,
        targets: tuple[str, ...],
        configuration: LoRAConfiguration,
        seed: int,
        device: str,
        dtype: str,
    ) -> None:
        if self.optimizer is not None:
            raise ValueError("wan_branch_adapter_already_injected")
        if (
            configuration.rank != 2
            or configuration.alpha != self.recipe.alpha
            or configuration.dropout != 0.0
            or device != "mps"
            or dtype != "bfloat16"
            or targets != self.eligible_lora_modules()
        ):
            raise ValueError("wan_branch_recipe_invalid")
        for parameter in self.transformer.parameters():
            parameter.requires_grad_(False)
        self._baseline = self.parameter_inventory()
        local_targets = [target.removeprefix("transformer.") for target in targets]
        peft = cast(Any, self.importer("peft"))
        with self._rng_fence(seed):
            config = peft.LoraConfig(
                r=2,
                lora_alpha=self.recipe.alpha,
                lora_dropout=0.0,
                target_modules=local_targets,
            )
            self.transformer.add_adapter(config, adapter_name=_ADAPTER_NAME)
        trainable = self._named_trainable()
        if not trainable or any(_logical_target(name) not in targets for name in trainable):
            raise ValueError("wan_branch_trainable_inventory_invalid")
        if set(_logical_target(name) for name in trainable) != set(targets):
            raise ValueError("wan_branch_trainable_inventory_invalid")
        declared = {id(parameter) for parameter in trainable.values()}
        all_trainable = {
            id(parameter) for parameter in self.transformer.parameters() if parameter.requires_grad
        }
        if declared != all_trainable:
            raise ValueError("wan_branch_trainable_inventory_invalid")
        optimizer_config = _adamw_config(self.recipe.optimizer_parameters)
        self.optimizer = self.torch.optim.AdamW(tuple(trainable.values()), **optimizer_config)
        owned = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        if owned != declared:
            raise ValueError("wan_branch_optimizer_invalid")
        self._injected = self.parameter_inventory()

    def parameter_inventory(self) -> tuple[WanParameterObservation, ...]:
        named = tuple(sorted(self.transformer.named_parameters()))
        parameter_names = frozenset(name for name, _parameter in named)
        values = []
        for name, parameter in named:
            device = getattr(parameter, "device", None)
            dtype = getattr(parameter, "dtype", None)
            if not _is_mps_device(device) or not _is_bfloat16_dtype(dtype):
                raise ValueError("wan_runtime_placement_invalid")
            device_type = getattr(cast(Any, device), "type", None)
            device_name = cast(
                Literal["mps"],
                device_type
                if type(device_type) is str
                else str(device).partition(":")[0],
            )
            dtype_name = cast(Literal["bfloat16"], str(dtype).removeprefix("torch."))
            logical_name = f"transformer.{_inventory_parameter_name(name, parameter_names)}"
            values.append(
                WanParameterObservation(
                    name=logical_name,
                    module=_logical_target(logical_name),
                    requires_grad=bool(parameter.requires_grad),
                    digest=self._parameter_digest(parameter),
                    device=device_name,
                    dtype=dtype_name,
                    shape=tuple(int(value) for value in parameter.shape),
                )
            )
        if not values:
            raise ValueError("wan_branch_parameter_inventory_invalid")
        return tuple(values)

    def measure_forward_backward(self) -> object:
        raise ValueError("wan_branch_canary_requires_one_optimizer_step")

    def train_step(self, *, step: int, total_steps: int) -> dict[str, float]:
        seed = _domain_seed(self.recipe.seed, self.recipe.branch)
        with self._rng_fence(seed):
            try:
                return self._train_step(step=step, total_steps=total_steps)
            except RuntimeError as error:
                if is_recognized_mps_oom(error):
                    raise WanMpsOutOfMemoryError("wan_branch_mps_out_of_memory") from error
                raise

    def _train_step(self, *, step: int, total_steps: int) -> dict[str, float]:
        if (step, total_steps) != (1, 1) or self.optimizer is None:
            raise ValueError("wan_branch_canary_requires_one_optimizer_step")
        if self.optimizer_updates:
            raise ValueError("wan_branch_optimizer_update_invalid")
        if self.cancelled():
            raise ValueError("wan_branch_cancelled")
        self.optimizer.zero_grad(set_to_none=True)
        loss, selector_count = self._loss()
        if self.cancelled():
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("wan_branch_cancelled")
        loss.backward()
        self._require_finite_gradients()
        if self.cancelled():
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("wan_branch_cancelled")
        gradient = self._clip_gradient_once()
        if self.cancelled():
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("wan_branch_cancelled")
        self.optimizer.step()
        self.optimizer_updates = 1
        self._require_finite_parameters_and_optimizer()
        after = self.parameter_inventory()
        assert self._baseline is not None and self._injected is not None
        baseline = {item.name: item.digest for item in self._baseline}
        unchanged = tuple(
            item
            for item in after
            if not item.requires_grad and baseline.get(item.name) == item.digest
        )
        base_after = tuple(item for item in after if not item.requires_grad)
        if len(unchanged) != len(base_after) or len(base_after) != len(self._baseline):
            raise ValueError("wan_branch_base_parameter_changed")
        injected = {item.name: item.digest for item in self._injected if item.requires_grad}
        changed = tuple(
            item for item in after if item.requires_grad and injected.get(item.name) != item.digest
        )
        if not changed or any(item.name not in injected for item in changed):
            raise ValueError("wan_branch_adapter_parameter_unchanged")
        evidence = WanBranchStepEvidence(
            branch=self.recipe.branch,
            selector_count=selector_count,
            loss=float(loss.detach().item()),
            gradient_norm=gradient,
            changed_adapter_inventory=changed,
            unchanged_base_inventory=unchanged,
            optimizer_update_count=1,
            memory_observations=self._memory_observations(),
        )
        self._last_evidence = evidence
        return {"loss": evidence.loss, "gradient_norm": evidence.gradient_norm}

    def branch_step_evidence(self) -> WanBranchStepEvidence:
        if self._last_evidence is None:
            raise ValueError("wan_branch_step_evidence_missing")
        return self._last_evidence

    def checkpoint_state(self, *, step: int) -> WanRuntimeCheckpointState:
        if step != 1 or self.optimizer_updates != 1:
            raise ValueError("wan_branch_checkpoint_state_invalid")
        safetensors = cast(Any, self.importer("safetensors.torch"))
        weights = {
            name: parameter.detach().to("cpu").contiguous()
            for name, parameter in self._named_trainable().items()
        }
        return WanRuntimeCheckpointState(
            weights=bytes(safetensors.save(weights)),
            optimizer={"schema_version": "wan-branch-optimizer-state.v1", "step": 1},
            trainer_state={
                "schema_version": "wan-branch-trainer-state.v1",
                "step": 1,
                "branch": self.recipe.branch,
            },
        )

    def checkpoint_state_v2(
        self, *, parent_checkpoint_id: str | None
    ) -> WanBranchCheckpointStateV2:
        """Capture adapter, optimizer, and RNG continuation state at step one."""

        if self.optimizer_updates != 1 or self.batch_digest is None:
            raise ValueError("wan_branch_checkpoint_state_invalid")
        named = dict(sorted(self._named_trainable().items()))
        safetensors = cast(Any, self.importer("safetensors.torch"))
        weights = bytes(
            safetensors.save(
                {
                    name: parameter.detach().to("cpu").contiguous()
                    for name, parameter in named.items()
                }
            )
        )
        specs = self._adapter_specs(named)
        optimizer = self._optimizer_state_v2(named)
        self._validate_optimizer_contract_v2(optimizer, named)
        cpu_rng = self.torch.random.get_rng_state().detach().to("cpu").numpy().tobytes()
        mps_rng = self.torch.mps.get_rng_state().detach().to("cpu").numpy().tobytes()
        manifest = WanBranchCheckpointManifestV2(
            schema_version="wan-branch-checkpoint.v2",
            branch=self.recipe.branch,
            selector=self.selector_for_branch(self.recipe.branch),
            step=1,
            total_steps=1,
            parent_checkpoint_id=parent_checkpoint_id,
            batch_digest=self.batch_digest,
            recipe_digest=wan_branch_recipe_digest(self.recipe),
            component_inventory_digest=wan_inventory_digest(self.loaded_branch.inventory),
            adapter_tensors=specs,
            optimizer_parameter_names=tuple(named),
            optimizer_slot_names=_ADAMW_SLOT_NAMES,
            cpu_rng_digest="sha256:" + hashlib.sha256(cpu_rng).hexdigest(),
            mps_rng_digest="sha256:" + hashlib.sha256(mps_rng).hexdigest(),
            finite=True,
        )
        return WanBranchCheckpointStateV2(
            manifest=manifest,
            weights=weights,
            optimizer=optimizer,
            cpu_rng_state=cpu_rng,
            mps_rng_state=mps_rng,
        )

    def restore_checkpoint_v2(
        self,
        state: WanBranchCheckpointStateV2,
        expected: WanBranchCheckpointManifestV2,
    ) -> WanBranchRestoreReceiptV2:
        """Restore a validated checkpoint into this freshly injected branch runtime."""

        validate_wan_branch_checkpoint_v2(state, expected)
        named = dict(sorted(self._named_trainable().items()))
        local_specs = self._adapter_specs(named)
        local_expected = self._expected_manifest_v2(expected, local_specs)
        if expected != local_expected or self.optimizer_updates != 0:
            raise ValueError("wan_branch_checkpoint_v2_invalid")
        tensors = cast(Any, self.importer("safetensors.torch")).load(state.weights)
        self._validate_optimizer_contract_v2(state.optimizer, named)
        decoded_optimizer = self._decode_optimizer_state_v2(state.optimizer, named)
        cpu_rng = self._rng_tensor(state.cpu_rng_state)
        mps_rng = self._rng_tensor(state.mps_rng_state)
        for name, tensor in tensors.items():
            named[name].data.copy_(tensor.to(named[name].device))
        self.optimizer.state.clear()
        self.optimizer.state.update(decoded_optimizer)
        self.torch.random.set_rng_state(cpu_rng)
        self.torch.mps.set_rng_state(mps_rng)
        self.optimizer_updates = 1
        observed = self.checkpoint_state_v2(parent_checkpoint_id=expected.parent_checkpoint_id)
        expected_optimizer = canonical_json(dict(state.optimizer)).encode()
        if (
            observed.weights != state.weights
            or canonical_json(dict(observed.optimizer)).encode() != expected_optimizer
            or observed.cpu_rng_state != state.cpu_rng_state
            or observed.mps_rng_state != state.mps_rng_state
        ):
            raise ValueError("wan_branch_checkpoint_v2_observation_invalid")
        manifest_payload = canonical_json(asdict(expected)).encode()
        return WanBranchRestoreReceiptV2(
            schema_version="wan-branch-restore-receipt.v2",
            manifest_digest="sha256:" + hashlib.sha256(manifest_payload).hexdigest(),
            weights_digest="sha256:" + hashlib.sha256(state.weights).hexdigest(),
            optimizer_digest="sha256:" + hashlib.sha256(expected_optimizer).hexdigest(),
            cpu_rng_digest=expected.cpu_rng_digest,
            mps_rng_digest=expected.mps_rng_digest,
        )

    def continuity_probe(self) -> dict[str, Any]:
        """Run the preregistered branch forward probe without optimizer mutation."""

        selector = self.batch["transformer_selector"]
        mask = selector == self.selector_for_branch(self.recipe.branch)
        result = self.transformer(
            hidden_states=self.batch["hidden_states"][mask],
            timestep=self.batch["timestep"][mask],
            encoder_hidden_states=self.batch["encoder_hidden_states"][mask],
            return_dict=True,
        )
        sample = result.sample.detach().to("cpu").clone()
        if not bool(self.torch.isfinite(sample).all().item()):
            raise ValueError("wan_branch_output_nonfinite")
        return {"probe.output": sample}

    def restore_checkpoint(self, state: WanCheckpointState) -> None:
        if state.step != 1 or state.total_steps != 1:
            raise ValueError("wan_branch_checkpoint_restore_invalid")
        tensors = cast(Any, self.importer("safetensors.torch")).load(state.weights)
        named = self._named_trainable()
        if set(tensors) != set(named):
            raise ValueError("wan_branch_checkpoint_restore_invalid")
        for name, tensor in tensors.items():
            parameter = named[name]
            if tuple(tensor.shape) != tuple(parameter.shape) or tensor.dtype != parameter.dtype:
                raise ValueError("wan_branch_checkpoint_restore_invalid")
            parameter.data.copy_(tensor.to(parameter.device))
        self.optimizer_updates = 1
        self._restored = WanRuntimeRestoreReceipt(
            weights_digest="sha256:" + hashlib.sha256(state.weights).hexdigest(),
            optimizer_payload=canonical_json(dict(state.optimizer)).encode(),
            trainer_state_payload=canonical_json(dict(state.trainer_state)).encode(),
        )

    def observe_restored_state(self) -> WanRuntimeRestoreReceipt:
        if self._restored is None:
            raise ValueError("wan_branch_checkpoint_restore_missing")
        return self._restored

    @property
    def last_evidence(self) -> WanBranchStepEvidence | None:
        return self._last_evidence

    def close(self) -> None:
        seed = _domain_seed(self.recipe.seed, self.recipe.branch)
        with self._rng_fence(seed):
            self.optimizer = None
            self.transformer = None
            self.batch.clear()
            self._baseline = None
            self._injected = None
            self.loaded_branch.close()
    def _adapter_specs(self, named: Mapping[str, Any]) -> tuple[WanBranchTensorSpecV2, ...]:
        dtype_names = {
            str(self.torch.float16): "F16",
            str(self.torch.bfloat16): "BF16",
            str(self.torch.float32): "F32",
        }
        try:
            return tuple(
                WanBranchTensorSpecV2(
                    name=name,
                    shape=tuple(int(size) for size in parameter.shape),
                    dtype=dtype_names[str(parameter.dtype)],
                )
                for name, parameter in named.items()
            )
        except KeyError:
            raise ValueError("wan_branch_checkpoint_v2_invalid") from None

    def _expected_manifest_v2(
        self,
        claimed: WanBranchCheckpointManifestV2,
        specs: tuple[WanBranchTensorSpecV2, ...],
    ) -> WanBranchCheckpointManifestV2:
        if self.batch_digest is None:
            raise ValueError("wan_branch_checkpoint_v2_invalid")
        return WanBranchCheckpointManifestV2(
            schema_version="wan-branch-checkpoint.v2",
            branch=self.recipe.branch,
            selector=self.selector_for_branch(self.recipe.branch),
            step=1,
            total_steps=1,
            parent_checkpoint_id=claimed.parent_checkpoint_id,
            batch_digest=self.batch_digest,
            recipe_digest=wan_branch_recipe_digest(self.recipe),
            component_inventory_digest=wan_inventory_digest(self.loaded_branch.inventory),
            adapter_tensors=specs,
            optimizer_parameter_names=tuple(item.name for item in specs),
            optimizer_slot_names=_ADAMW_SLOT_NAMES,
            cpu_rng_digest=claimed.cpu_rng_digest,
            mps_rng_digest=claimed.mps_rng_digest,
            finite=True,
        )

    def _optimizer_state_v2(self, named: Mapping[str, Any]) -> dict[str, object]:
        parameter_state: dict[str, object] = {}
        for name, parameter in named.items():
            slots = self.optimizer.state.get(parameter)
            if not isinstance(slots, Mapping) or not slots:
                raise ValueError("wan_branch_checkpoint_state_invalid")
            encoded: dict[str, object] = {}
            for slot_name, value in sorted(slots.items()):
                if type(slot_name) is not str or not hasattr(value, "detach"):
                    raise ValueError("wan_branch_checkpoint_state_invalid")
                tensor = value.detach().to("cpu")
                encoded[slot_name] = {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "values": tensor.reshape(-1).tolist(),
                }
            parameter_state[name] = encoded
        return {
            "schema_version": "wan-branch-optimizer-state.v2",
            "parameter_state": parameter_state,
        }

    def _decode_optimizer_state_v2(
        self, payload: Mapping[str, object], named: Mapping[str, Any]
    ) -> dict[Any, dict[str, Any]]:
        raw_states = cast(Mapping[str, Mapping[str, object]], payload["parameter_state"])
        result: dict[Any, dict[str, Any]] = {}
        for name, parameter in named.items():
            decoded: dict[str, Any] = {}
            for slot_name, raw in raw_states[name].items():
                slot = cast(Mapping[str, object], raw)
                dtype_name = cast(str, slot["dtype"]).removeprefix("torch.")
                dtype = getattr(self.torch, dtype_name, None)
                shape = tuple(cast(list[int], slot["shape"]))
                values = cast(list[float], slot["values"])
                if dtype is None or math.prod(shape) != len(values):
                    raise ValueError("wan_branch_checkpoint_v2_invalid")
                decoded[slot_name] = (
                    self.torch.tensor(values, dtype=dtype).reshape(shape).to(parameter.device)
                )
            result[parameter] = decoded
        return result

    def _validate_optimizer_contract_v2(
        self, payload: Mapping[str, object], named: Mapping[str, Any]
    ) -> None:
        """Derive exact AdamW continuation slots from trusted runtime parameters."""

        try:
            if self.recipe.optimizer_algorithm != "adamw":
                raise ValueError
            raw_states = cast(Mapping[str, Mapping[str, object]], payload["parameter_state"])
            if set(raw_states) != set(named):
                raise ValueError
            for name, parameter in named.items():
                slots = raw_states[name]
                if tuple(sorted(slots)) != _ADAMW_SLOT_NAMES:
                    raise ValueError
                for slot_name in _ADAMW_SLOT_NAMES:
                    raw = slots[slot_name]
                    if not isinstance(raw, Mapping) or set(raw) != {"dtype", "shape", "values"}:
                        raise ValueError
                    dtype = raw["dtype"]
                    shape = raw["shape"]
                    values = raw["values"]
                    expected_dtype = (
                        "torch.float32" if slot_name == "step" else str(parameter.dtype)
                    )
                    expected_shape = [] if slot_name == "step" else list(parameter.shape)
                    if (
                        dtype != expected_dtype
                        or shape != expected_shape
                        or not isinstance(values, list)
                        or len(values) != max(1, math.prod(expected_shape))
                        or any(
                            type(value) not in (int, float) or not math.isfinite(float(value))
                            for value in values
                        )
                    ):
                        raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("wan_branch_checkpoint_v2_invalid") from None

    def _rng_tensor(self, payload: bytes) -> Any:
        return self.torch.frombuffer(bytearray(payload), dtype=self.torch.uint8).clone()

    def _loss(self) -> tuple[Any, int]:
        self._require_finite_batch()
        selector = self.batch["transformer_selector"]
        mask = selector == self.selector_for_branch(self.recipe.branch)
        count = int(mask.sum().item())
        if count <= 0:
            raise ValueError("wan_branch_selector_empty")
        result = self.transformer(
            hidden_states=self.batch["hidden_states"][mask],
            timestep=self.batch["timestep"][mask],
            encoder_hidden_states=self.batch["encoder_hidden_states"][mask],
            return_dict=True,
        )
        prediction = result.sample
        if not bool(self.torch.isfinite(prediction).all().item()):
            raise ValueError("wan_branch_output_nonfinite")
        target = self.batch["target"][mask]
        loss = self.torch.nn.functional.mse_loss(prediction.float(), target.float())
        if not bool(self.torch.isfinite(loss).all().item()):
            raise ValueError("wan_branch_loss_nonfinite")
        return loss, count

    def _validate_batch(self) -> None:
        required = {
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "clean_latents",
            "noise_latents",
            "target",
            "transformer_selector",
        }
        if set(self.batch) != required:
            raise ValueError("wan_training_batch_invalid")

    def _require_finite_batch(self) -> None:
        for name, tensor in self.batch.items():
            if name != "transformer_selector" and not bool(
                self.torch.isfinite(tensor).all().item()
            ):
                raise ValueError("wan_branch_input_nonfinite")

    def _named_trainable(self) -> dict[str, Any]:
        return {
            f"transformer.{name}": parameter
            for name, parameter in self.transformer.named_parameters()
            if parameter.requires_grad
        }

    def _parameter_digest(self, parameter: Any) -> str:
        payload = parameter.detach().to("cpu").contiguous().view(self.torch.uint8).numpy().tobytes()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def _require_finite_gradients(self) -> None:
        named = self._named_trainable()
        if not named or any(
            parameter.grad is None or not bool(self.torch.isfinite(parameter.grad).all().item())
            for parameter in named.values()
        ):
            raise ValueError("wan_branch_gradient_nonfinite")

    def _clip_gradient_once(self) -> float:
        value = self.torch.nn.utils.clip_grad_norm_(
            tuple(self._named_trainable().values()), self.recipe.gradient_clip_norm
        )
        result = float(value.detach().item() if hasattr(value, "detach") else value)
        if not math.isfinite(result):
            raise ValueError("wan_branch_gradient_nonfinite")
        return result

    def _require_finite_parameters_and_optimizer(self) -> None:
        if any(
            not bool(self.torch.isfinite(parameter).all().item())
            for parameter in self._named_trainable().values()
        ):
            raise ValueError("wan_branch_parameter_nonfinite")
        for state in self.optimizer.state.values():
            if not isinstance(state, Mapping):
                raise ValueError("wan_branch_optimizer_invalid")
            for value in state.values():
                if hasattr(value, "dtype") and not bool(self.torch.isfinite(value).all().item()):
                    raise ValueError("wan_branch_optimizer_nonfinite")

    def _memory_observations(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(
                (
                    ("allocated_bytes", int(self.torch.mps.current_allocated_memory())),
                    ("driver_bytes", int(self.torch.mps.driver_allocated_memory())),
                )
            )
        )

    @contextmanager
    def _rng_fence(self, seed: int) -> Iterator[None]:
        python_state = random.getstate()
        numpy: Any = None
        numpy_state: object | None = None
        try:
            numpy = self.importer("numpy")
            numpy_state = numpy.random.get_state()
        except (ImportError, ModuleNotFoundError):
            pass
        cpu_state = self.torch.random.get_rng_state()
        mps_state = self.torch.mps.get_rng_state()
        try:
            random.seed(seed)
            if numpy is not None:
                numpy.random.seed(seed % (2**32))
            self.torch.manual_seed(seed)
            self.torch.mps.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            if numpy is not None and numpy_state is not None:
                numpy.random.set_state(numpy_state)
            self.torch.random.set_rng_state(cpu_state)
            self.torch.mps.set_rng_state(mps_state)


def _matches_suffix(name: str, suffix: str) -> bool:
    return name == suffix or name.endswith("." + suffix)


def _logical_target(parameter_name: str) -> str:
    matches = tuple(marker for marker in _LORA_MARKERS if marker in parameter_name)
    if len(matches) > 1:
        raise ValueError("wan_branch_trainable_inventory_invalid")
    if matches:
        return parameter_name.split(matches[0], 1)[0]
    return parameter_name.rpartition(".")[0]


def _inventory_parameter_name(parameter_name: str, parameter_names: frozenset[str]) -> str:
    if any(marker in parameter_name for marker in _LORA_MARKERS):
        return parameter_name
    prefix, marker, suffix = parameter_name.partition(".base_layer.")
    if marker and any(
        candidate.startswith(prefix + lora_marker)
        for candidate in parameter_names
        for lora_marker in _LORA_MARKERS
    ):
        return f"{prefix}.{suffix}"
    return parameter_name


def _domain_seed(seed: int, branch: WanTransformerBranch) -> int:
    payload = f"wan-mps-branch-seed.v1\0{branch}\0{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _adamw_config(parameters: tuple[tuple[str, int | float | bool], ...]) -> dict[str, object]:
    aliases = {"learning_rate": "lr", "beta1": "betas", "beta2": "betas"}
    raw = dict(parameters)
    allowed = {
        "learning_rate",
        "beta1",
        "beta2",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }
    if set(raw) - allowed or "learning_rate" not in raw:
        raise ValueError("wan_branch_optimizer_invalid")
    result: dict[str, object] = {}
    for name, value in raw.items():
        if name not in {"beta1", "beta2"}:
            result[aliases.get(name, name)] = value
    result["betas"] = (float(raw.get("beta1", 0.9)), float(raw.get("beta2", 0.999)))
    return result


__all__ = [
    "WanBranchStepEvidence",
    "WanMpsBranchOperations",
    "WanMpsOutOfMemoryError",
    "is_recognized_mps_oom",
]
