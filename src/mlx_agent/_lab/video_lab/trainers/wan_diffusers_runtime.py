"""Concrete local-only Diffusers/PEFT runtime for precomputed Wan training batches."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import shutil
import struct
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast

from video_lab.domain import canonical_json
from video_lab.trainer_protocol import CalibrationRequest, LoRAConfiguration, TrainingRequest
from video_lab.trainers.wan_batch import (
    validate_loaded_wan_batch,
    validate_wan_batch_envelope,
)
from video_lab.trainers.wan_checkpoint import WanCheckpointState
from video_lab.trainers.wan_mps import (
    WAN_LORA_MPS_BACKEND_ID,
    WanMpsObservation,
    WanPreparedTraining,
    WanRuntimeCheckpointState,
    WanRuntimeRestoreReceipt,
    WanTrainingSources,
)
from video_lab.trainers.wan_production_models import WanBranchRecipeV1
from video_lab.training_models import Sha256Digest
from video_lab.wan_pending import validate_pending_gradients

WAN_BATCH_SCHEMA = "wan-precomputed-training-batch.v1"
_WAN_BATCH_TENSORS = {
    "hidden_states",
    "timestep",
    "encoder_hidden_states",
    "clean_latents",
    "noise_latents",
    "target",
    "transformer_selector",
}
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_MAX_SAFETENSORS_RANK = 32
_ADAMW_KEYS = {
    "learning_rate",
    "gradient_accumulation_steps",
    "beta1",
    "beta2",
    "eps",
    "weight_decay",
    "amsgrad",
    "maximize",
    "foreach",
    "fused",
    "capturable",
    "differentiable",
}
_PRODUCTION_RNG_LOCK = threading.Lock()


class _WanBranchPreparedFactory(Protocol):
    def prepare_branch(
        self, sources: object, request: object, branch_recipe: WanBranchRecipeV1
    ) -> WanPreparedTraining: ...


@dataclass(frozen=True, slots=True)
class WanDiffusersAdapterPaths:
    wan_snapshot: Path
    high_transformer: Path
    low_transformer: Path
    batch_manifest: Path
    batch_tensors: Path


class WanDiffusersTrainingRuntime:
    """Repository-owned forward/loss/optimizer implementation; imports libraries lazily."""

    def __init__(
        self,
        *,
        root: Path,
        paths: WanDiffusersAdapterPaths,
        runtime_digest: Sha256Digest,
        importer: Callable[[str], object] = importlib.import_module,
        cleanup: Callable[[], None] | None = None,
        prepared_factory: _WanBranchPreparedFactory | None = None,
        branch_recipe: WanBranchRecipeV1 | None = None,
    ) -> None:
        self._root, self._paths = root, paths
        self._runtime_digest, self._importer = runtime_digest, importer
        self._cleanup = cleanup
        if (prepared_factory is None) != (branch_recipe is None):
            raise ValueError("wan_runtime_branch_factory_invalid")
        self._prepared_factory = prepared_factory
        self._branch_recipe = branch_recipe
        self._branch_prepared: set[int] = set()
        self._prepared: set[int] = set()
        self._state_lock = threading.Lock()
        self._operation_held = False
        self._root_identity = _capture_root_identity(root)

    def observe_host(self) -> WanMpsObservation:
        torch = cast(Any, self._importer("torch"))
        built = bool(torch.backends.mps.is_built())
        available = bool(torch.backends.mps.is_available())
        bf16 = False
        if built and available:
            bf16 = _is_mps_device(
                torch.empty(1, device="mps", dtype=torch.bfloat16).device
            )
        maximum = int(torch.mps.recommended_max_memory()) if available else 0
        allocated = int(torch.mps.driver_allocated_memory()) if available else 0
        return WanMpsObservation(
            runtime_digest=self._runtime_digest,
            built=built,
            available=available,
            bf16_supported=bf16,
            component_placements=tuple(
                (name, "mps") for name in ("text_encoder", "transformer", "transformer_2", "vae")
            ),
            disk_free_bytes=shutil.disk_usage(self._root).free,
            disk_required_bytes=1,
            memory_headroom_bytes=max(0, maximum - allocated),
            memory_required_bytes=1,
        )

    def prepare(
        self, sources: WanTrainingSources, request: TrainingRequest | CalibrationRequest
    ) -> WanPreparedTraining:
        self._acquire_production_operation()
        try:
            if self._prepared_factory is not None:
                assert self._branch_recipe is not None
                prepared = self._prepared_factory.prepare_branch(
                    sources, request, self._branch_recipe
                )
                with self._state_lock:
                    self._prepared.add(id(prepared.handle))
                    self._branch_prepared.add(id(prepared.handle))
                return prepared
            del sources
            self._reverify_root()
            operations = _WanDiffusersOperations(
                root=self._root,
                paths=self._paths,
                request=request,
                importer=self._importer,
                reverify=self._reverify_root,
            )
            operations.load()
            self._reverify_root()
            prepared = WanPreparedTraining(handle=operations)
            with self._state_lock:
                self._prepared.add(id(operations))
            return prepared
        except BaseException:
            self._release_production_operation()
            raise

    def _reverify_root(self) -> None:
        if _capture_root_identity(self._root) != self._root_identity:
            raise ValueError("wan_runtime_source_invalid")

    def release(self, prepared: WanPreparedTraining | None) -> None:
        release_operation = False
        branch_prepared = False
        if prepared is not None:
            with self._state_lock:
                identifier = id(prepared.handle)
                release_operation = identifier in self._prepared
                branch_prepared = identifier in self._branch_prepared
                self._prepared.discard(identifier)
                self._branch_prepared.discard(identifier)
        if branch_prepared:
            assert prepared is not None
            try:
                close = getattr(prepared.handle, "close", None)
                if not callable(close):
                    raise ValueError("wan_runtime_branch_handle_invalid")
                close()
            finally:
                if release_operation:
                    self._release_production_operation()
            return
        try:
            torch = cast(Any, self._importer("torch"))
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            pass
        if release_operation:
            self._release_production_operation()

    def close(self) -> None:
        with self._state_lock:
            if self._prepared or self._operation_held:
                raise ValueError("wan_runtime_busy")
            cleanup = self._cleanup
        if cleanup is not None:
            cleanup()
            self._cleanup = None

    def _acquire_production_operation(self) -> None:
        with self._state_lock:
            if self._operation_held or not _PRODUCTION_RNG_LOCK.acquire(blocking=False):
                raise ValueError("wan_runtime_concurrent_operation")
            self._operation_held = True

    def _release_production_operation(self) -> None:
        with self._state_lock:
            if not self._operation_held:
                raise ValueError("wan_runtime_operation_missing")
            self._operation_held = False
            _PRODUCTION_RNG_LOCK.release()


class _WanDiffusersOperations:
    def __init__(
        self,
        *,
        root: Path,
        paths: WanDiffusersAdapterPaths,
        request: TrainingRequest | CalibrationRequest,
        importer: Callable[[str], object],
        reverify: Callable[[], None],
    ) -> None:
        self.root, self.paths, self.request, self.importer = root, paths, request, importer
        self.torch: Any = None
        self.reverify = reverify
        self.components: dict[str, Any] = {}
        self.batch: dict[str, Any] = {}
        self.optimizer: Any = None
        self.step = 0
        self.batch_cursor = 0
        self.accumulation_position = 0
        self.optimizer_updates = 0
        self.scheduler_state: dict[str, object] = {"step": 0, "rate_scale": 1.0}
        self.scaler_state: dict[str, object] = {
            "schema_version": "torch-amp-scaler-state.v1",
            "enabled": False,
            "scale": 1.0,
        }
        self._restored: WanRuntimeRestoreReceipt | None = None

    def load(self) -> None:
        self.reverify()
        self.torch = cast(Any, self.importer("torch"))
        self.torch.manual_seed(self.request.seed)
        self.torch.mps.manual_seed(self.request.seed)
        self.torch.use_deterministic_algorithms(True)
        if self.request.lora.dropout != 0:
            raise ValueError("wan_runtime_stochastic_recipe_unsupported")
        diffusers = cast(Any, self.importer("diffusers"))
        transformers = cast(Any, self.importer("transformers"))
        local = {"local_files_only": True, "torch_dtype": self.torch.bfloat16}
        wan = self._path(self.paths.wan_snapshot)
        self.components = {
            "text_encoder": transformers.UMT5EncoderModel.from_pretrained(
                str(wan), subfolder="text_encoder", **local
            ),
            "vae": diffusers.AutoencoderKLWan.from_pretrained(str(wan), subfolder="vae", **local),
            "transformer": diffusers.WanTransformer3DModel.from_single_file(
                self._path(self.paths.high_transformer),
                config=str(wan),
                subfolder="transformer",
                **local,
            ),
            "transformer_2": diffusers.WanTransformer3DModel.from_single_file(
                self._path(self.paths.low_transformer),
                config=str(wan),
                subfolder="transformer",
                **local,
            ),
        }
        for component in self.components.values():
            component.to(device="mps", dtype=self.torch.bfloat16)
            _require_full_mps(component)
        self.batch = self._load_batch()
        self.reverify()

    def eligible_lora_modules(self) -> tuple[str, ...]:
        self.reverify()
        names: list[str] = []
        for prefix in ("transformer", "transformer_2"):
            for name, module in self.components[prefix].named_modules():
                if name.endswith(("to_q", "to_k", "to_v", "to_out.0")) and hasattr(
                    module, "weight"
                ):
                    names.append(f"{prefix}.{name}")
        return tuple(sorted(names))

    def inject_lora(
        self,
        *,
        targets: tuple[str, ...],
        configuration: LoRAConfiguration,
        seed: int,
        device: str,
        dtype: str,
    ) -> None:
        self.reverify()
        if device != "mps" or dtype != "bfloat16":
            raise ValueError("wan_runtime_placement_invalid")
        self.torch.manual_seed(seed)
        peft = cast(Any, self.importer("peft"))
        for component in self.components.values():
            for parameter in component.parameters():
                parameter.requires_grad_(False)
        for prefix in ("transformer", "transformer_2"):
            selected = tuple(
                name.removeprefix(prefix + ".") for name in targets if name.startswith(prefix + ".")
            )
            if selected:
                config = peft.LoraConfig(
                    r=configuration.rank,
                    lora_alpha=configuration.alpha,
                    lora_dropout=configuration.dropout,
                    target_modules=list(selected),
                )
                self.components[prefix].add_adapter(config, adapter_name="character")
        trainable = [
            parameter
            for component in self.components.values()
            for parameter in component.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("wan_runtime_targets_invalid")
        raw_config = (
            {item.name: item.value for item in self.request.optimizer.parameters}
            if isinstance(self.request, TrainingRequest)
            else {"learning_rate": 0.001}
        )
        config = _normalized_adamw_config(raw_config)
        self.gradient_accumulation_steps = cast(int, config.pop("gradient_accumulation_steps"))
        self.optimizer = self.torch.optim.AdamW(trainable, **config)
        self._verify()
        self.reverify()

    def parameter_inventory(self) -> tuple[object, ...]:
        self.reverify()
        values: list[dict[str, object]] = []
        for prefix, component in self.components.items():
            for name, parameter in component.named_parameters():
                payload = (
                    parameter.detach()
                    .to("cpu")
                    .contiguous()
                    .view(self.torch.uint8)
                    .numpy()
                    .tobytes()
                )
                values.append(
                    {
                        "name": f"{prefix}.{name}",
                        "module": _lora_target_module(prefix, name),
                        "requires_grad": bool(parameter.requires_grad),
                        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                        "device": str(parameter.device),
                        "dtype": str(parameter.dtype).removeprefix("torch."),
                        "shape": tuple(parameter.shape),
                    }
                )
        return tuple(values)

    def measure_forward_backward(self) -> object:
        self.reverify()
        started = monotonic()
        loss = self._loss()
        loss.backward()
        gradient = self._gradient_norm()
        self.optimizer.zero_grad(set_to_none=True)
        result = self._measurement(loss, gradient, monotonic() - started)
        self.reverify()
        return result

    def train_step(self, *, step: int, total_steps: int) -> object:
        self.reverify()
        accumulation = getattr(self, "gradient_accumulation_steps", 1)
        if self.accumulation_position == 0:
            self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss()
        loss.backward()
        self.accumulation_position += 1
        if _optimizer_update_due(
            step=step, total_steps=total_steps, accumulation_steps=accumulation
        ):
            _normalize_accumulated_gradients(
                tuple(self._named_trainable().values()),
                accumulation_position=self.accumulation_position,
            )
            gradient = self._gradient_norm()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_updates += 1
            self.accumulation_position = 0
        else:
            gradient = self._gradient_norm()
        self.step = step
        self.batch_cursor = step
        self.scheduler_state = {"step": step, "rate_scale": 1.0}
        self._verify()
        self.reverify()
        return {"loss": float(loss.detach().item()), "gradient_norm": gradient}

    def checkpoint_state(self, *, step: int) -> WanRuntimeCheckpointState:
        self.reverify()
        safetensors = cast(Any, self.importer("safetensors.torch"))
        weights = {
            name: parameter.detach().to("cpu").contiguous()
            for prefix, component in self.components.items()
            for name, parameter in component.named_parameters()
            if parameter.requires_grad
            for name in (f"{prefix}.{name}",)
        }
        weight_bytes = bytes(safetensors.save(weights))
        targets = self.request.trainable_targets
        config = (
            {item.name: item.value for item in self.request.optimizer.parameters}
            if isinstance(self.request, TrainingRequest)
            else {"learning_rate": 0.001}
        )
        accumulation = int(config.get("gradient_accumulation_steps", 1))
        backend_optimizer: dict[str, object] = {"optimizer_updates": self.optimizer_updates}
        if accumulation > 1:
            backend_optimizer["pending_gradients"] = _pending_gradient_record(
                self._named_trainable(),
                accumulation_position=self.accumulation_position,
            )
        optimizer = {
            "schema_version": "trainer-optimizer-state.v2",
            "request_id": self.request.request_id,
            "run_id": self.request.run_id,
            "step": step,
            "algorithm": "adamw",
            "config": config,
            "global_step": step,
            "parameter_groups": [
                {"group_id": "default", "parameters": list(targets), "hyperparameters": config}
            ],
            "parameter_state": self._optimizer_parameter_state(step, targets),
            "backend_state": {WAN_LORA_MPS_BACKEND_ID: backend_optimizer},
            "parent_optimizer_state_digest": getattr(self.request, "optimizer_state_digest", None),
        }
        trainer = self._observed_trainer_state(step)
        return WanRuntimeCheckpointState(
            weights=weight_bytes, optimizer=optimizer, trainer_state=trainer
        )

    def restore_checkpoint(self, state: WanCheckpointState) -> None:
        self.reverify()
        safetensors = cast(Any, self.importer("safetensors.torch"))
        self.optimizer_updates = cast(
            int,
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], state.optimizer["backend_state"])[
                    WAN_LORA_MPS_BACKEND_ID
                ],
            )["optimizer_updates"],
        )
        weights = safetensors.load(state.weights)
        for prefix, component in self.components.items():
            by_name = dict(component.named_parameters())
            for qualified, tensor in weights.items():
                if qualified.startswith(prefix + "."):
                    by_name[qualified.removeprefix(prefix + ".")].data.copy_(tensor.to("mps"))
        self._restore_optimizer_parameter_state(state.optimizer)
        self._restore_trainer_state(state.trainer_state, state.optimizer)
        self.step = state.step
        self._verify()
        self.reverify()
        self._restored = WanRuntimeRestoreReceipt(
            weights_digest="sha256:" + hashlib.sha256(state.weights).hexdigest(),
            optimizer_payload=canonical_json(
                self._observed_optimizer_payload(state.optimizer)
            ).encode(),
            trainer_state_payload=canonical_json(self._observed_trainer_state(state.step)).encode(),
        )

    def observe_restored_state(self) -> object:
        if self._restored is None:
            raise ValueError("wan_runtime_restore_missing")
        self._verify()
        self.reverify()
        return self._restored

    def _loss(self) -> Any:
        predicted = self.torch.empty_like(self.batch["target"])
        selector = self.batch["transformer_selector"]
        for value, name in ((0, "transformer"), (1, "transformer_2")):
            mask = selector == value
            if bool(mask.any().item()):
                result = self.components[name](
                    hidden_states=self.batch["hidden_states"][mask],
                    timestep=self.batch["timestep"][mask],
                    encoder_hidden_states=self.batch["encoder_hidden_states"][mask],
                    return_dict=True,
                )
                predicted[mask] = result.sample
        loss = self.torch.nn.functional.mse_loss(predicted, self.batch["target"])
        if not bool(self.torch.isfinite(loss).item()):
            raise ValueError("wan_training_step_invalid")
        return loss

    def _named_trainable(self) -> dict[str, Any]:
        return {
            f"{prefix}.{name}": parameter
            for prefix, component in self.components.items()
            for name, parameter in component.named_parameters()
            if parameter.requires_grad
        }

    def _optimizer_parameter_state(self, step: int, targets: tuple[str, ...]) -> dict[str, object]:
        safetensors = cast(Any, self.importer("safetensors.torch"))
        named = self._named_trainable()
        result: dict[str, object] = {}
        covered: set[str] = set()
        for target in targets:
            parameters: dict[str, object] = {}
            for name, parameter in sorted(named.items()):
                if _lora_target_module(*name.split(".", 1)) != target:
                    continue
                state = self.optimizer.state.get(parameter)
                if not isinstance(state, Mapping):
                    raise ValueError("wan_runtime_optimizer_invalid")
                if not state and self.optimizer_updates == 0:
                    parameters[name] = {"initialized": False}
                    covered.add(name)
                    continue
                if not state:
                    raise ValueError("wan_runtime_optimizer_invalid")
                group = self.optimizer.param_groups[0]
                _validate_adamw_slots(
                    state,
                    parameter,
                    amsgrad=bool(group["amsgrad"]),
                    capturable=bool(group["capturable"]),
                    fused=bool(group["fused"]),
                )
                slot_tensors = {
                    str(slot): value.detach().to("cpu").contiguous()
                    for slot, value in state.items()
                    if hasattr(value, "detach")
                }
                if set(slot_tensors) != set(state):
                    raise ValueError("wan_runtime_optimizer_invalid")
                payload = bytes(safetensors.save(slot_tensors))
                parameters[name] = {
                    "serialized_b64": base64.b64encode(payload).decode(),
                    "serialized_size": len(payload),
                    "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                }
                covered.add(name)
            if not parameters:
                raise ValueError("wan_runtime_optimizer_invalid")
            result[target] = {
                "step": step,
                "slots": {"parameter_count": len(parameters), "parameters": parameters},
            }
        if covered != set(named):
            raise ValueError("wan_runtime_optimizer_invalid")
        return result

    def _restore_optimizer_parameter_state(self, optimizer: Mapping[str, object]) -> None:
        safetensors = cast(Any, self.importer("safetensors.torch"))
        named = self._named_trainable()
        parameter_state = cast(Mapping[str, object], optimizer["parameter_state"])
        if tuple(parameter_state) != self.request.trainable_targets:
            raise ValueError("wan_runtime_optimizer_invalid")
        restored: set[str] = set()
        self.optimizer.state.clear()
        for target in self.request.trainable_targets:
            record = cast(Mapping[str, object], parameter_state[target])
            slots = cast(Mapping[str, object], record["slots"])
            parameters = cast(Mapping[str, object], slots["parameters"])
            if slots.get("parameter_count") != len(parameters):
                raise ValueError("wan_runtime_optimizer_invalid")
            for name, raw_record in parameters.items():
                if (
                    name in restored
                    or name not in named
                    or _lora_target_module(*name.split(".", 1)) != target
                ):
                    raise ValueError("wan_runtime_optimizer_invalid")
                encoded = cast(Mapping[str, object], raw_record)
                if encoded == {"initialized": False}:
                    if self.optimizer_updates != 0:
                        raise ValueError("wan_runtime_optimizer_invalid")
                    restored.add(name)
                    continue
                payload = base64.b64decode(cast(str, encoded["serialized_b64"]), validate=True)
                if (
                    encoded.get("serialized_size") != len(payload)
                    or encoded.get("digest") != "sha256:" + hashlib.sha256(payload).hexdigest()
                ):
                    raise ValueError("wan_runtime_optimizer_invalid")
                values = safetensors.load(payload)
                group = self.optimizer.param_groups[0]
                parameter = named[name]
                restored_state = {
                    slot: tensor.to(
                        device=(
                            parameter.device
                            if slot != "step" or group["capturable"] or group["fused"]
                            else "cpu"
                        )
                    )
                    for slot, tensor in values.items()
                }
                _validate_adamw_slots(
                    restored_state,
                    parameter,
                    amsgrad=bool(group["amsgrad"]),
                    capturable=bool(group["capturable"]),
                    fused=bool(group["fused"]),
                )
                self.optimizer.state[parameter] = restored_state
                restored.add(name)
        if restored != set(named):
            raise ValueError("wan_runtime_optimizer_invalid")

    def _observed_optimizer_payload(self, original: Mapping[str, object]) -> dict[str, object]:
        observed = dict(original)
        observed["parameter_state"] = self._optimizer_parameter_state(
            self.step, self.request.trainable_targets
        )
        backend: dict[str, object] = {"optimizer_updates": self.optimizer_updates}
        if getattr(self, "gradient_accumulation_steps", 1) > 1:
            backend["pending_gradients"] = _pending_gradient_record(
                self._named_trainable(),
                accumulation_position=self.accumulation_position,
            )
        observed["backend_state"] = {WAN_LORA_MPS_BACKEND_ID: backend}
        return observed

    def _observed_trainer_state(self, step: int) -> dict[str, object]:
        return _trainer_state(
            self.request,
            step,
            cpu_rng=_capture_global_rng(self.torch, "cpu"),
            mps_rng=_capture_global_rng(self.torch, "mps"),
            scheduler_state=self.scheduler_state,
            scaler_state=self.scaler_state,
            batch_cursor=self.batch_cursor,
        )

    def _restore_trainer_state(
        self, trainer: Mapping[str, object], optimizer: Mapping[str, object]
    ) -> None:
        rng_states = cast(Mapping[str, object], trainer["rng_states"])
        _restore_global_rng(self.torch, rng_states["cpu"], "cpu")
        _restore_global_rng(self.torch, rng_states["mps"], "mps")
        self.scheduler_state = dict(cast(Mapping[str, object], trainer["scheduler_state"]))
        self.scaler_state = dict(cast(Mapping[str, object], trainer["scaler_state"]))
        backend = cast(Mapping[str, object], trainer["backend_state"])
        owned = cast(Mapping[str, object], backend[WAN_LORA_MPS_BACKEND_ID])
        self.batch_cursor = cast(int, owned["data_cursor"])
        optimizer_backend = cast(
            Mapping[str, object],
            cast(Mapping[str, object], optimizer["backend_state"])[WAN_LORA_MPS_BACKEND_ID],
        )
        self.optimizer_updates = cast(int, optimizer_backend["optimizer_updates"])
        pending = optimizer_backend.get("pending_gradients")
        self.accumulation_position = (
            _restore_pending_gradients(
                self.torch,
                self._named_trainable(),
                cast(Mapping[str, object], pending),
            )
            if pending is not None
            else 0
        )

    def _gradient_norm(self) -> float:
        squares = [
            parameter.grad.detach().float().pow(2).sum()
            for component in self.components.values()
            for parameter in component.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not squares:
            raise ValueError("wan_training_step_invalid")
        value = float(self.torch.stack(squares).sum().sqrt().item())
        if not math.isfinite(value):
            raise ValueError("wan_training_step_invalid")
        return value

    def _measurement(self, loss: Any, gradient: float, elapsed: float) -> dict[str, float]:
        allocated = float(self.torch.mps.current_allocated_memory())
        driver = float(self.torch.mps.driver_allocated_memory())
        return {
            "loss": float(loss.detach().item()),
            "gradient_norm": gradient,
            "wall_seconds": elapsed,
            "rss_start_bytes": 0.0,
            "rss_end_bytes": 0.0,
            "mps_allocated_start_bytes": allocated,
            "mps_allocated_end_bytes": allocated,
            "mps_driver_start_bytes": driver,
            "mps_driver_end_bytes": driver,
        }

    def _load_batch(self) -> dict[str, Any]:
        manifest_path, tensors_path = (
            self._path(self.paths.batch_manifest),
            self._path(self.paths.batch_tensors),
        )
        payload = validate_wan_batch_artifact(manifest_path, tensors_path)
        manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
        tensors = cast(Any, self.importer("safetensors.torch")).load(payload)
        schedule = cast(dict[str, object], manifest["noise_schedule"])
        transformer_config = self.components["transformer"].config
        validate_loaded_wan_batch(
            tensors,
            batch_size=cast(int, manifest["batch_size"]),
            timestep_min=cast(int, schedule["timestep_min"]),
            timestep_max=cast(int, schedule["timestep_max"]),
            boundary=cast(int, schedule["timestep_boundary"]),
            in_channels=int(transformer_config.in_channels),
        )
        return {
            name: tensor.to(
                device="mps",
                dtype=(
                    self.torch.int64
                    if name == "transformer_selector"
                    else self.torch.float32
                    if name == "timestep"
                    else self.torch.bfloat16
                ),
            )
            for name, tensor in tensors.items()
        }

    def _path(self, relative: Path) -> Path:
        path = self.root / relative
        if (
            relative.is_absolute()
            or not path.is_relative_to(self.root)
            or not path.exists()
            or path.is_symlink()
        ):
            raise ValueError("wan_runtime_source_invalid")
        return path

    def _verify(self) -> None:
        for component in self.components.values():
            _require_full_mps(component)


def _require_full_mps(component: object) -> None:
    observed = cast(Any, component)
    device_map = getattr(observed, "hf_device_map", {"": "mps"})
    tensors = tuple(observed.parameters()) + tuple(observed.buffers())
    if (
        not isinstance(device_map, Mapping)
        or not device_map
        or not tensors
        or any(not _is_mps_device(value) for value in device_map.values())
        or any(not _is_mps_device(getattr(tensor, "device", None)) for tensor in tensors)
        or any(
            str(getattr(tensor, "dtype", "")) not in {"bfloat16", "torch.bfloat16"}
            for tensor in tensors
        )
    ):
        raise ValueError("wan_runtime_placement_invalid")


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


def _lora_target_module(prefix: str, parameter_name: str) -> str:
    for marker in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."):
        if marker in parameter_name:
            return f"{prefix}.{parameter_name.split(marker, 1)[0]}"
    return f"{prefix}.{parameter_name.rpartition('.')[0]}"


def validate_wan_batch_artifact(manifest_path: Path, tensors_path: Path) -> bytes:
    """Validate the exact dependency-free envelope before safetensors decoding."""

    try:
        if manifest_path.is_symlink() or tensors_path.is_symlink():
            raise ValueError("wan_training_batch_invalid")
        payload = tensors_path.read_bytes()
        validate_wan_batch_envelope(manifest_path.read_bytes(), payload)
        return payload
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("wan_training_batch_invalid") from None


def _batch_manifest_matches_payload(manifest: Mapping[str, object], payload: bytes) -> bool:
    try:
        header_size = struct.unpack("<Q", payload[:8])[0]
        batch_size = manifest.get("batch_size")
        records = manifest.get("tensors")
        if type(batch_size) is not int or batch_size <= 0 or not isinstance(records, list):
            return False
        header = _validated_safetensors_header(
            payload, expected_names=_WAN_BATCH_TENSORS, batch_size=batch_size
        )
        expected: list[dict[str, object]] = []
        for name in sorted(_WAN_BATCH_TENSORS):
            metadata = header[name]
            if not isinstance(metadata, dict) or set(metadata) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                return False
            dtype, shape, offsets = (metadata["dtype"], metadata["shape"], metadata["data_offsets"])
            if (
                type(dtype) is not str
                or not isinstance(shape, list)
                or not shape
                or any(type(value) is not int or value <= 0 for value in shape)
                or shape[0] != batch_size
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(value) is not int or value < 0 for value in offsets)
                or offsets[1] <= offsets[0]
            ):
                return False
            start, end = 8 + header_size + offsets[0], 8 + header_size + offsets[1]
            if end > len(payload):
                return False
            expected.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "rank": len(shape),
                    "shape": shape,
                    "byte_offsets": offsets,
                    "digest": "sha256:" + hashlib.sha256(payload[start:end]).hexdigest(),
                    "finite": True,
                }
            )
        objective = manifest.get("objective")
        schedule = manifest.get("noise_schedule")
        return (
            records == expected
            and objective
            == {
                "target_kind": "flow_noise_minus_clean",
                "formula": "target=noise_latents-clean_latents",
                "clean_latents_digest": next(
                    item["digest"] for item in expected if item["name"] == "clean_latents"
                ),
                "noise_latents_digest": next(
                    item["digest"] for item in expected if item["name"] == "noise_latents"
                ),
                "target_digest": next(
                    item["digest"] for item in expected if item["name"] == "target"
                ),
            }
            and isinstance(schedule, dict)
            and set(schedule)
            == {
                "timestep_min",
                "timestep_max",
                "timestep_boundary",
                "selector_rule",
            }
            and schedule.get("selector_rule") == "high_if_timestep_gte_boundary_else_low"
        )
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError, struct.error):
        return False


def _validated_safetensors_header(
    payload: bytes, *, expected_names: set[str], batch_size: int
) -> dict[str, dict[str, object]]:
    """Parse a bounded, canonical safetensors envelope before allocating tensors."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("wan_training_batch_invalid")
            result[key] = value
        return result

    try:
        if len(payload) < 9 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError
        header_size = struct.unpack("<Q", payload[:8])[0]
        if (
            header_size <= 1
            or header_size > _MAX_SAFETENSORS_HEADER_BYTES
            or 8 + header_size > len(payload)
        ):
            raise ValueError
        parsed = json.loads(
            payload[8 : 8 + header_size].decode().rstrip(" "),
            object_pairs_hook=unique_object,
        )
        if not isinstance(parsed, dict):
            raise ValueError
        metadata = parsed.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(
                type(key) is not str or type(value) is not str for key, value in metadata.items()
            )
        ):
            raise ValueError
        if set(parsed) != expected_names or len(parsed) > 100_000:
            raise ValueError
        data_size = len(payload) - 8 - header_size
        spans: list[tuple[int, int]] = []
        for item in parsed.values():
            if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
                raise ValueError
            dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
            if (
                type(dtype) is not str
                or dtype not in _SAFETENSORS_DTYPE_BYTES
                or not isinstance(shape, list)
                or not shape
                or len(shape) > _MAX_SAFETENSORS_RANK
                or any(type(dim) is not int or dim <= 0 or dim > 2**31 for dim in shape)
                or shape[0] != batch_size
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)
            ):
                raise ValueError
            count = math.prod(shape)
            byte_count = count * _SAFETENSORS_DTYPE_BYTES[dtype]
            start, end = offsets
            if count > data_size or end - start != byte_count or end > data_size:
                raise ValueError
            spans.append((start, end))
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError
            cursor = end
        if cursor != data_size:
            raise ValueError
        return cast(dict[str, dict[str, object]], parsed)
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, struct.error):
        raise ValueError("wan_training_batch_invalid") from None


def _validate_loaded_batch(
    tensors: Mapping[str, Any],
    *,
    batch_size: int,
    timestep_min: int,
    timestep_max: int,
    boundary: int,
    in_channels: int,
) -> None:
    """Validate exact Wan flow-training semantics after safetensors decoding."""
    try:
        if set(tensors) != _WAN_BATCH_TENSORS or not 0 <= timestep_min < boundary <= timestep_max:
            raise ValueError
        if any(
            getattr(tensor, "ndim", 0) < 1 or int(tensor.shape[0]) != batch_size
            for tensor in tensors.values()
        ):
            raise ValueError
        hidden = tensors["hidden_states"]
        clean = tensors["clean_latents"]
        noise = tensors["noise_latents"]
        target = tensors["target"]
        timestep = tensors["timestep"]
        selector = tensors["transformer_selector"]
        encoder = tensors["encoder_hidden_states"]
        sigma = (timestep / float(timestep_max)).to(dtype=hidden.dtype)
        while sigma.ndim < hidden.ndim:
            sigma = sigma.unsqueeze(-1)
        expected_hidden = sigma * noise + (1.0 - sigma) * clean
        if (
            tuple(hidden.shape) != tuple(clean.shape)
            or tuple(hidden.shape) != tuple(noise.shape)
            or tuple(hidden.shape) != tuple(target.shape)
            or hidden.ndim < 3
            or int(hidden.shape[1]) != in_channels
            or encoder.ndim != 3
            or tuple(timestep.shape) != (batch_size,)
            or str(timestep.dtype) != "torch.float32"
            or tuple(selector.shape) != (batch_size,)
            or str(selector.dtype) != "torch.int64"
            or not bool((timestep >= timestep_min).all().item())
            or not bool((timestep <= timestep_max).all().item())
            or not bool(((selector == 0) | (selector == 1)).all().item())
            or not bool((selector == (timestep < boundary).to(dtype=selector.dtype)).all().item())
            # FlowMatchEulerDiscreteScheduler.add_noise casts sigma to the
            # sample dtype, then computes sigma*noise + (1-sigma)*sample.
            or not bool((hidden == expected_hidden).all().item())
            or not bool((target == noise - clean).all().item())
            or any(
                not bool(cast(Any, tensor).isfinite().all().item())
                for name, tensor in tensors.items()
                if name != "transformer_selector"
            )
        ):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("wan_training_batch_invalid") from None


def _capture_root_identity(root: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    try:
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError("wan_runtime_source_invalid")
        values: list[tuple[str, int, int, int, int]] = []
        root_info = root.stat(follow_symlinks=False)
        values.append(
            (".", root_info.st_dev, root_info.st_ino, root_info.st_size, root_info.st_mtime_ns)
        )
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("wan_runtime_source_invalid")
            info = path.stat(follow_symlinks=False)
            values.append(
                (
                    path.relative_to(root).as_posix(),
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                )
            )
        return tuple(values)
    except (OSError, ValueError):
        raise ValueError("wan_runtime_source_invalid") from None


def _trainer_state(
    request: TrainingRequest | CalibrationRequest,
    step: int,
    *,
    cpu_rng: Mapping[str, object],
    mps_rng: Mapping[str, object],
    scheduler_state: Mapping[str, object],
    scaler_state: Mapping[str, object],
    batch_cursor: int,
) -> dict[str, object]:
    total = request.total_steps if isinstance(request, TrainingRequest) else max(4, request.steps)
    return {
        "schema_version": "trainer-state.v3",
        "request_id": request.request_id,
        "run_id": request.run_id,
        "step": step,
        "backend_id": request.backend_id,
        "seed": request.seed,
        "total_steps": total,
        "global_step": step,
        "rng_states": {"cpu": dict(cpu_rng), "mps": dict(mps_rng)},
        "scheduler_state": dict(scheduler_state),
        "scaler_state": dict(scaler_state),
        "backend_state": {WAN_LORA_MPS_BACKEND_ID: {"data_cursor": batch_cursor}},
    }


def _torch_rng_state(state: Any, device: str) -> dict[str, object]:
    raw = bytes(state.detach().to("cpu").contiguous().tolist())
    return {
        "algorithm": "torch.global_rng",
        "device": device,
        "state_hex": raw.hex(),
        "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _decode_torch_rng(torch: Any, record: object, device: str) -> Any:
    try:
        payload = cast(Mapping[str, object], record)
        raw = bytes.fromhex(cast(str, payload["state_hex"]))
        if (
            payload.get("algorithm") != "torch.global_rng"
            or payload.get("device") != device
            or payload.get("digest") != "sha256:" + hashlib.sha256(raw).hexdigest()
            or not raw
        ):
            raise ValueError
        return torch.tensor(tuple(raw), dtype=torch.uint8, device="cpu")
    except (KeyError, TypeError, ValueError):
        raise ValueError("wan_runtime_restore_invalid") from None


def _capture_global_rng(torch: Any, device: str) -> dict[str, object]:
    state = torch.get_rng_state() if device == "cpu" else torch.mps.get_rng_state()
    return _torch_rng_state(state, device)


def _restore_global_rng(torch: Any, record: object, device: str) -> None:
    state = _decode_torch_rng(torch, record, device)
    if device == "cpu":
        torch.set_rng_state(state)
    else:
        torch.mps.set_rng_state(state)


def _normalized_adamw_config(raw: Mapping[str, object]) -> dict[str, object]:
    try:
        if set(raw) - _ADAMW_KEYS:
            raise ValueError
        values: dict[str, Any] = {
            "lr": raw.get("learning_rate", 0.001),
            "gradient_accumulation_steps": raw.get("gradient_accumulation_steps", 1),
            "betas": (raw.get("beta1", 0.9), raw.get("beta2", 0.999)),
            "eps": raw.get("eps", 1e-8),
            "weight_decay": raw.get("weight_decay", 0.01),
            "amsgrad": raw.get("amsgrad", False),
            "maximize": raw.get("maximize", False),
            "foreach": raw.get("foreach", False),
            "fused": raw.get("fused", False),
            "capturable": raw.get("capturable", False),
            "differentiable": raw.get("differentiable", False),
        }
        lr = values["lr"]
        accumulation = values["gradient_accumulation_steps"]
        betas = values["betas"]
        if (
            type(lr) not in (int, float)
            or not math.isfinite(float(lr))
            or float(lr) <= 0
            or type(accumulation) is not int
            or accumulation <= 0
            or any(type(beta) not in (int, float) or not 0 <= float(beta) < 1 for beta in betas)
            or any(
                type(values[key]) is not bool
                for key in (
                    "amsgrad",
                    "maximize",
                    "foreach",
                    "fused",
                    "capturable",
                    "differentiable",
                )
            )
        ):
            raise ValueError
        values["lr"] = float(lr)
        values["betas"] = tuple(float(beta) for beta in betas)
        values["eps"] = float(values["eps"])
        values["weight_decay"] = float(values["weight_decay"])
        return values
    except (TypeError, ValueError, OverflowError):
        raise ValueError("wan_runtime_optimizer_config_invalid") from None


def _validate_adamw_slots(
    state: Mapping[str, Any],
    parameter: Any,
    *,
    amsgrad: bool,
    capturable: bool,
    fused: bool,
) -> None:
    expected = {"step", "exp_avg", "exp_avg_sq"}
    if amsgrad:
        expected.add("max_exp_avg_sq")
    try:
        step = state["step"]
        expected_step_device = str(parameter.device) if capturable or fused else "cpu"
        if (
            set(state) != expected
            or tuple(step.shape) != ()
            or str(step.dtype) != "torch.float32"
            or str(step.device) != expected_step_device
            or bool(step.requires_grad)
        ):
            raise ValueError
        for name in expected - {"step"}:
            slot = state[name]
            if (
                tuple(slot.shape) != tuple(parameter.shape)
                or slot.dtype != parameter.dtype
                or slot.device != parameter.device
                or bool(slot.requires_grad)
            ):
                raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("wan_runtime_optimizer_invalid") from None


def _optimizer_update_due(*, step: int, total_steps: int, accumulation_steps: int) -> bool:
    if not 1 <= step <= total_steps or accumulation_steps <= 0:
        raise ValueError("wan_runtime_optimizer_config_invalid")
    return step % accumulation_steps == 0 or step == total_steps


def _normalize_accumulated_gradients(
    parameters: tuple[Any, ...], *, accumulation_position: int
) -> None:
    if accumulation_position <= 0:
        raise ValueError("wan_runtime_optimizer_invalid")
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            gradient.div_(accumulation_position)


def _pending_gradient_record(
    parameters: Mapping[str, Any], *, accumulation_position: int
) -> dict[str, object]:
    if accumulation_position < 0:
        raise ValueError("wan_runtime_optimizer_invalid")
    records: dict[str, object] = {}
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad
        if gradient is None:
            if accumulation_position:
                raise ValueError("wan_runtime_optimizer_invalid")
            continue
        # Untyped storage avoids NumPy dtype support gaps such as bfloat16.
        payload = bytes(gradient.detach().to("cpu").contiguous().untyped_storage())
        records[name] = {
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "serialized_b64": base64.b64encode(payload).decode(),
            "shape": list(gradient.shape),
            "dtype": str(gradient.dtype).removeprefix("torch."),
            "device": str(gradient.device),
        }
    if bool(records) != bool(accumulation_position):
        raise ValueError("wan_runtime_optimizer_invalid")
    return {"accumulation_position": accumulation_position, "parameters": records}


def _restore_pending_gradients(
    torch: Any, parameters: Mapping[str, Any], record: Mapping[str, object]
) -> int:
    try:
        position = record["accumulation_position"]
        if type(position) is not int or position < 0:
            raise ValueError
        specs = {
            name: (
                tuple(parameter.shape),
                str(parameter.dtype).removeprefix("torch."),
                str(parameter.device),
            )
            for name, parameter in parameters.items()
        }
        validated = validate_pending_gradients(
            record,
            position,
            expected_parameters=parameters,
            expected_specs=specs,
        )
        raw_parameters = cast(Mapping[str, object], validated["parameters"])
        for name, parameter in parameters.items():
            if not position:
                parameter.grad = None
                continue
            item = cast(Mapping[str, object], raw_parameters[name])
            payload = base64.b64decode(cast(str, item["serialized_b64"]), validate=True)
            gradient = torch.frombuffer(bytearray(payload), dtype=parameter.dtype).clone()
            parameter.grad = gradient.reshape(parameter.shape).to(parameter.device)
        return position
    except (KeyError, TypeError, ValueError):
        raise ValueError("wan_runtime_optimizer_invalid") from None
