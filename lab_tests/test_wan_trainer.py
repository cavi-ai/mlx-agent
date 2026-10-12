from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import video_lab.trainers.wan_mps as wan_mps
from video_lab.domain import canonical_json
from video_lab.trainer_protocol import (
    CalibrationRequest,
    CheckpointResult,
    LoRAConfiguration,
    OptimizerConfiguration,
    OptimizerParameter,
    ProbeTrainingRequest,
    TrainerBackend,
    TrainingCancelled,
    TrainingMaterialHandles,
    TrainingRequest,
    validate_cancelled_training_result,
    validate_training_result,
)
from video_lab.trainers.wan_checkpoint import (
    WanCheckpointCleanupError,
    WanCheckpointCodec,
    WanCheckpointState,
    WanParentCheckpointDescriptor,
    WanParentCheckpointFileDescriptor,
)
from video_lab.trainers.wan_mps import (
    WAN_LORA_MPS_BACKEND_ID,
    WanLoRAMpsTrainer,
    WanMpsObservation,
    WanRuntimeCheckpointState,
    WanRuntimeRestoreReceipt,
    WanSourceDescriptor,
    WanTrainingRuntime,
    WanTrainingSourceResolver,
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


def _source_digest(descriptors: tuple[WanSourceDescriptor, ...]) -> str:
    payload = {
        "files": [
            {
                "name": item.name,
                "sha256": item.expected_digest,
                "size_bytes": item.expected_size,
            }
            for item in sorted(descriptors, key=lambda item: item.name)
        ]
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha(encoded)


def _descriptor(name: str, path: Path) -> WanSourceDescriptor:
    info = path.stat()
    data = path.read_bytes()
    return WanSourceDescriptor(
        name=name,
        path=lambda: path,
        expected_digest=_sha(data),
        expected_size=len(data),
        expected_device=info.st_dev,
        expected_inode=info.st_ino,
    )


def _materials(
    descriptors: tuple[WanSourceDescriptor, ...],
    *,
    backend_digest: str,
    runtime_digest: str,
    material_digest: str,
) -> TrainingMaterialHandles:
    base = next(item for item in descriptors if item.name == "base_model")
    return TrainingMaterialHandles(
        recipe_digest=_sha(b"recipe"),
        dataset_digest=_sha(b"dataset"),
        base_model_digest=base.expected_digest,
        backend_digest=backend_digest,
        source_digest=_source_digest(descriptors),
        runtime_digest=runtime_digest,
        material_digest=material_digest,
    )


class ObservedRuntime:
    def __init__(self, observation: WanMpsObservation) -> None:
        self.observation = observation
        self.observe_calls = 0
        self.prepare_calls = 0

    def observe_host(self) -> WanMpsObservation:
        self.observe_calls += 1
        return self.observation

    def prepare(self, sources: object, request: object) -> object:
        self.prepare_calls += 1
        raise AssertionError("probe must not load training components")

    def release(self, prepared: object | None) -> None:
        raise AssertionError("probe must not release components it did not prepare")


@pytest.fixture
def local_sources(tmp_path: Path) -> tuple[WanSourceDescriptor, ...]:
    base = tmp_path / "base.safetensors"
    high = tmp_path / "high_noise.safetensors"
    low = tmp_path / "low_noise.safetensors"
    base.write_bytes(b"base-model")
    high.write_bytes(b"animegen-high")
    low.write_bytes(b"animegen-low")
    return (
        _descriptor("base_model", base),
        _descriptor("high_noise", high),
        _descriptor("low_noise", low),
    )


def _observation(runtime_digest: str, **updates: object) -> WanMpsObservation:
    values: dict[str, object] = {
        "runtime_digest": runtime_digest,
        "built": True,
        "available": True,
        "bf16_supported": True,
        "component_placements": (
            ("text_encoder", "mps"),
            ("transformer", "mps"),
            ("transformer_2", "mps"),
            ("vae", "mps"),
        ),
        "disk_free_bytes": 20_000,
        "disk_required_bytes": 10_000,
        "memory_headroom_bytes": 12_000,
        "memory_required_bytes": 8_000,
    }
    values.update(updates)
    return WanMpsObservation(**values)


def _probe(
    local_sources: tuple[WanSourceDescriptor, ...],
    observation: WanMpsObservation,
    *,
    material_digest: str | None = None,
) -> tuple[object, ObservedRuntime]:
    backend_digest = _sha(b"backend")
    expected_material_digest = material_digest or _sha(b"materials")
    runtime = ObservedRuntime(observation)
    trainer = WanLoRAMpsTrainer(
        backend_digest=backend_digest,
        material_digest=expected_material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,
    )
    materials = _materials(
        local_sources,
        backend_digest=backend_digest,
        runtime_digest=observation.runtime_digest,
        material_digest=expected_material_digest,
    )
    report = trainer.probe(
        ProbeTrainingRequest(
            request_id="probe-1",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=backend_digest,
            materials=materials,
            scratch_root=Path("/strictly-local/scratch"),
        )
    )
    return report, runtime


def test_lazy_trainer_package_import_is_ml_dependency_free() -> None:
    code = """
import sys
import video_lab.trainers
for name in ('torch', 'diffusers', 'transformers', 'accelerate'):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_lazy_trainer_package_keeps_existing_wildcard_exports() -> None:
    namespace: dict[str, object] = {}

    exec("from video_lab.trainers import *", namespace)

    assert "FakeTrainerBackend" in namespace
    assert "FakeTrainerFault" in namespace
    assert "TrainingCancelled" in namespace
    assert "WanLoRAMpsTrainer" in namespace
    assert "WanTrainingSourceLease" in namespace
    assert "WanBackendStageUnavailable" in namespace
    assert "LoRATargetPolicy" in namespace
    assert "WanCalibrationMeasurement" in namespace
    assert "WanParameterObservation" in namespace


def test_source_resolver_rehashes_exact_local_files(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    with WanTrainingSourceResolver(local_sources).acquire() as lease:
        resolved = lease.sources

    assert tuple(item.name for item in resolved.files) == (
        "base_model",
        "high_noise",
        "low_noise",
    )
    assert resolved.base_model_digest == _sha(b"base-model")
    assert resolved.source_digest == _source_digest(local_sources)
    assert all(item.path.is_absolute() for item in resolved.files)


def test_source_lease_closes_retained_descriptors(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    lease = WanTrainingSourceResolver(local_sources).acquire()
    lease.reverify()

    lease.close()
    lease.close()

    with pytest.raises(ValueError, match="wan_source_lease_closed"):
        lease.reverify()


def test_source_lease_exposes_read_only_metadata(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    with WanTrainingSourceResolver(local_sources).acquire() as lease:
        with pytest.raises(AttributeError):
            lease.sources = lease.sources  # type: ignore[misc]


@pytest.mark.parametrize("failure", ("missing", "symlink", "replacement"))
def test_source_resolver_rejects_missing_linked_or_replaced_material(
    tmp_path: Path, failure: str
) -> None:
    original = tmp_path / "base.safetensors"
    original.write_bytes(b"exact-bytes")
    descriptor = _descriptor("base_model", original)

    if failure == "missing":
        original.unlink()
    elif failure == "symlink":
        target = tmp_path / "target.safetensors"
        target.write_bytes(b"exact-bytes")
        original.unlink()
        original.symlink_to(target)
    else:
        replacement_path = tmp_path / "replacement.safetensors"
        replacement_path.write_bytes(b"exact-bytes")
        os.replace(replacement_path, original)

    with pytest.raises(ValueError, match="wan_source_invalid"):
        WanTrainingSourceResolver((descriptor,)).acquire()


def test_probe_accepts_only_exact_local_mps_bf16_runtime(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    runtime_digest = _sha(b"runtime")
    report, runtime = _probe(local_sources, _observation(runtime_digest))

    assert report.model_dump(mode="json") == {
        "backend_id": WAN_LORA_MPS_BACKEND_ID,
        "backend_digest": _sha(b"backend"),
        "runtime_digest": runtime_digest,
        "supported": True,
        "max_steps": 1_000_000,
        "safe_boundaries": [1, 2, 3, 4],
    }
    assert runtime.prepare_calls == 0


@pytest.mark.parametrize(
    "updates",
    (
        {"built": False},
        {"available": False},
        {"bf16_supported": False},
        {"component_placements": (("transformer_high", "cpu"),)},
        {"component_placements": (("transformer_high", "offload"),)},
        {"component_placements": ()},
        {"disk_free_bytes": 9_999},
        {"memory_headroom_bytes": 7_999},
    ),
)
def test_probe_fails_closed_without_loading_for_unsafe_host_observations(
    local_sources: tuple[WanSourceDescriptor, ...], updates: dict[str, object]
) -> None:
    runtime_digest = _sha(b"runtime")
    report, runtime = _probe(local_sources, _observation(runtime_digest, **updates))

    assert report.supported is False
    assert report.safe_boundaries == (1, 2, 3, 4)
    assert runtime.prepare_calls == 0


def test_probe_fails_closed_on_backend_runtime_or_material_identity_drift(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    backend_digest = _sha(b"backend")
    material_digest = _sha(b"materials")
    runtime_digest = _sha(b"runtime")
    runtime = ObservedRuntime(_observation(runtime_digest))
    trainer = WanLoRAMpsTrainer(
        backend_digest=backend_digest,
        material_digest=material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,
    )
    materials = _materials(
        local_sources,
        backend_digest=backend_digest,
        runtime_digest=runtime_digest,
        material_digest=material_digest,
    )

    requests = (
        ProbeTrainingRequest(
            request_id="backend-drift",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=_sha(b"other-backend"),
            materials=materials,
            scratch_root=Path("/scratch"),
        ),
        ProbeTrainingRequest(
            request_id="runtime-drift",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=backend_digest,
            materials=materials.model_copy(update={"runtime_digest": _sha(b"other-runtime")}),
            scratch_root=Path("/scratch"),
        ),
        ProbeTrainingRequest(
            request_id="material-drift",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=backend_digest,
            materials=materials.model_copy(update={"material_digest": _sha(b"other-material")}),
            scratch_root=Path("/scratch"),
        ),
    )

    assert tuple(trainer.probe(request).supported for request in requests) == (
        False,
        False,
        False,
    )
    assert runtime.prepare_calls == 0


def test_probe_fails_closed_when_local_material_changes_after_authorization(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    changed_path = local_sources[1].path()
    changed_path.write_bytes(b"replaced-animegen-high")

    report, runtime = _probe(local_sources, _observation(_sha(b"runtime")))

    assert report.supported is False
    assert runtime.prepare_calls == 0


def test_probe_fails_closed_when_injected_runtime_observation_crashes(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    class CrashingRuntime:
        def observe_host(self) -> WanMpsObservation:
            raise TypeError("malformed injected runtime")

        def prepare(self, sources: object, request: object) -> object:
            raise AssertionError("probe must not load training components")

        def release(self, prepared: object | None) -> None:
            raise AssertionError("probe must not release components")

    backend_digest = _sha(b"backend")
    material_digest = _sha(b"materials")
    materials = _materials(
        local_sources,
        backend_digest=backend_digest,
        runtime_digest=_sha(b"runtime"),
        material_digest=material_digest,
    )
    trainer = WanLoRAMpsTrainer(
        backend_digest=backend_digest,
        material_digest=material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=CrashingRuntime(),
    )

    report = trainer.probe(
        ProbeTrainingRequest(
            request_id="runtime-crash",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=backend_digest,
            materials=materials,
            scratch_root=Path("/scratch"),
        )
    )

    assert report.supported is False


def _probe_observation_value(
    local_sources: tuple[WanSourceDescriptor, ...], observation: object
) -> object:
    class RawRuntime:
        def observe_host(self) -> object:
            return observation

        def prepare(self, sources: object, request: object) -> object:
            raise AssertionError("probe must not load training components")

        def release(self, prepared: object | None) -> None:
            raise AssertionError("probe must not release components")

    backend_digest = _sha(b"backend")
    runtime_digest = _sha(b"runtime")
    material_digest = _sha(b"materials")
    materials = _materials(
        local_sources,
        backend_digest=backend_digest,
        runtime_digest=runtime_digest,
        material_digest=material_digest,
    )
    trainer = WanLoRAMpsTrainer(
        backend_digest=backend_digest,
        material_digest=material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=RawRuntime(),  # type: ignore[arg-type]
    )
    return trainer.probe(
        ProbeTrainingRequest(
            request_id="raw-observation",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=backend_digest,
            materials=materials,
            scratch_root=Path("/scratch"),
        )
    )


def _valid_observation_payload() -> dict[str, object]:
    return {
        "runtime_digest": _sha(b"runtime"),
        "built": True,
        "available": True,
        "bf16_supported": True,
        "component_placements": (
            ("text_encoder", "mps"),
            ("transformer", "mps"),
            ("transformer_2", "mps"),
            ("vae", "mps"),
        ),
        "disk_free_bytes": 20_000,
        "disk_required_bytes": 10_000,
        "memory_headroom_bytes": 12_000,
        "memory_required_bytes": 8_000,
    }


def test_mps_observation_model_rejects_lenient_digest_type() -> None:
    payload = _valid_observation_payload()
    payload["runtime_digest"] = _sha(b"runtime").encode()

    with pytest.raises(ValueError):
        WanMpsObservation(**payload)


@pytest.mark.parametrize("field", ("disk_required_bytes", "memory_required_bytes"))
def test_probe_revalidates_invalid_same_instance_observation(
    local_sources: tuple[WanSourceDescriptor, ...], field: str
) -> None:
    payload = _valid_observation_payload()
    payload[field] = True
    bypassed = WanMpsObservation.model_construct(**payload)

    report = _probe_observation_value(local_sources, bypassed)

    assert report.supported is False
    assert report.runtime_digest == _sha(b"runtime")


def _special_descriptor(path: Path) -> WanSourceDescriptor:
    info = path.lstat()
    return WanSourceDescriptor(
        name="base_model",
        path=lambda: path,
        expected_digest=_sha(b""),
        expected_size=0,
        expected_device=info.st_dev,
        expected_inode=info.st_ino,
    )


def test_source_acquire_rejects_fifo_without_blocking_or_leaking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)
    script = f"""
import hashlib
import os
from pathlib import Path
from video_lab.trainers.wan_mps import WanSourceDescriptor, WanTrainingSourceResolver

path = Path({str(fifo)!r})
info = path.lstat()
descriptor = WanSourceDescriptor(
    name="base_model",
    path=lambda: path,
    expected_digest="sha256:" + hashlib.sha256(b"").hexdigest(),
    expected_size=0,
    expected_device=info.st_dev,
    expected_inode=info.st_ino,
)
before = len(os.listdir("/dev/fd"))
try:
    WanTrainingSourceResolver((descriptor,)).acquire()
except ValueError as error:
    assert str(error) == "wan_source_invalid"
else:
    raise AssertionError("FIFO source was accepted")
assert len(os.listdir("/dev/fd")) == before
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=1,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_acquire_rejects_special_leaf_matrix_without_fd_leaks(
    tmp_path: Path,
) -> None:
    unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    socket_path = Path("/private/tmp") / f"wan-source-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    unix_socket.bind(str(socket_path))
    directory = tmp_path / "source-directory"
    directory.mkdir()
    paths = (socket_path, directory, Path("/dev/null"))
    try:
        for path in paths:
            before = len(os.listdir("/dev/fd"))
            with pytest.raises(ValueError, match="wan_source_invalid"):
                WanTrainingSourceResolver((_special_descriptor(path),)).acquire()
            assert len(os.listdir("/dev/fd")) == before
    finally:
        unix_socket.close()
        socket_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime_digest", "not-a-digest"),
        ("built", "true"),
        ("available", 1),
        ("bf16_supported", None),
        ("disk_free_bytes", True),
        ("disk_free_bytes", -1),
        ("disk_free_bytes", 1.5),
        ("disk_free_bytes", float("nan")),
        ("disk_required_bytes", False),
        ("memory_headroom_bytes", -1),
        ("memory_required_bytes", 2.0),
        (
            "component_placements",
            [
                ("text_encoder", "mps"),
                ("transformer", "mps"),
                ("transformer_2", "mps"),
                ("vae", "mps"),
            ],
        ),
        (
            "component_placements",
            (("text_encoder", 1), ("transformer", "mps")),
        ),
        (
            "component_placements",
            (("vae", "mps"), ("text_encoder", "mps")),
        ),
    ),
)
def test_probe_fails_closed_for_malformed_observation_fields(
    local_sources: tuple[WanSourceDescriptor, ...], field: str, value: object
) -> None:
    payload = _valid_observation_payload()
    payload[field] = value

    report = _probe_observation_value(local_sources, payload)

    assert report.supported is False
    assert report.runtime_digest == _sha(b"runtime")


@pytest.mark.parametrize("observation", (object(), None, "mps"))
def test_probe_fails_closed_for_wrong_observation_object(
    local_sources: tuple[WanSourceDescriptor, ...], observation: object
) -> None:
    report = _probe_observation_value(local_sources, observation)

    assert report.supported is False
    assert report.runtime_digest == _sha(b"runtime")


def test_probe_fails_closed_when_observation_property_access_raises(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    class ExplodingObservation:
        @property
        def runtime_digest(self) -> str:
            raise LookupError("broken observation property")

    report = _probe_observation_value(local_sources, ExplodingObservation())

    assert report.supported is False
    assert report.runtime_digest == _sha(b"runtime")


@pytest.mark.parametrize(
    "placements",
    tuple(
        tuple(
            item
            for item in _valid_observation_payload()["component_placements"]
            if item[0] != omitted
        )
        for omitted in ("text_encoder", "transformer", "transformer_2", "vae")
    )
    + (
        (
            ("scheduler", "mps"),
            ("text_encoder", "mps"),
            ("transformer", "mps"),
            ("transformer_2", "mps"),
            ("vae", "mps"),
        ),
        (
            ("text_encoder", "mps"),
            ("transformer", "mps"),
            ("transformer", "mps"),
            ("transformer_2", "mps"),
            ("vae", "mps"),
        ),
        (("text_encoder", "mps"), ("transformer", "mps"), ("transformer_2", "mps")),
    ),
)
def test_probe_requires_exact_executable_component_inventory(
    local_sources: tuple[WanSourceDescriptor, ...],
    placements: tuple[tuple[str, str], ...],
) -> None:
    payload = _valid_observation_payload()
    payload["component_placements"] = placements

    report = _probe_observation_value(local_sources, payload)

    assert report.supported is False


@pytest.mark.parametrize("device", ("cpu", "offload", "mps:0", "MPS", ""))
def test_probe_rejects_noncanonical_executable_component_device(
    local_sources: tuple[WanSourceDescriptor, ...], device: str
) -> None:
    payload = _valid_observation_payload()
    payload["component_placements"] = (
        ("text_encoder", "mps"),
        ("transformer", device),
        ("transformer_2", "mps"),
        ("vae", "mps"),
    )

    report = _probe_observation_value(local_sources, payload)

    assert report.supported is False


def test_wan_trainer_and_runtime_conform_to_runtime_checkable_protocols(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    runtime = ObservedRuntime(_observation(_sha(b"runtime")))
    trainer = WanLoRAMpsTrainer(
        backend_digest=_sha(b"backend"),
        material_digest=_sha(b"materials"),
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,
    )

    assert isinstance(runtime, WanTrainingRuntime)
    assert isinstance(trainer, TrainerBackend)


def _lora_weights(step: int) -> bytes:
    payload = struct.pack(
        "<64f", *(float(step + index) for index in range(1, 65))
    )
    header = json.dumps(
        {
            "transformer.attn.to_q.lora_A.weight": {
                "dtype": "F32",
                "shape": [16, 2],
                "data_offsets": [0, 128],
            },
            "transformer.attn.to_q.lora_B.weight": {
                "dtype": "F32",
                "shape": [2, 16],
                "data_offsets": [128, 256],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    header += b" " * ((-len(header)) % 8)
    return len(header).to_bytes(8, "little") + header + payload


class _TrainingPreparedHandle:
    def __init__(self, request: TrainingRequest) -> None:
        self.request = request
        self.injected = False
        self.steps: list[int] = []
        self.mutate_frozen_at: int | None = None
        self.rogue_trainable_at: int | None = None
        self.nonfinite_at: int | None = None
        self.error_at: int | None = None
        self.restored: WanCheckpointState | None = None
        self._applied_weights = _lora_weights(0)
        self._applied_optimizer: dict[str, object] = {}
        self._applied_trainer_state: dict[str, object] = {}
        self._applied_step = 0
        self.omit_restore_component: str | None = None

    def eligible_lora_modules(self) -> tuple[str, ...]:
        return self.request.trainable_targets

    def inject_lora(
        self,
        *,
        targets: tuple[str, ...],
        configuration: LoRAConfiguration,
        seed: int,
        device: str,
        dtype: str,
    ) -> None:
        assert targets == self.request.trainable_targets
        assert configuration == self.request.lora
        assert seed == self.request.seed
        assert (device, dtype) == ("mps", "bfloat16")
        self.injected = True

    def train_step(self, *, step: int, total_steps: int) -> dict[str, float]:
        assert total_steps == self.request.total_steps
        self.steps.append(step)
        if self.error_at == step:
            raise RuntimeError("injected optimizer failure")
        next_state = self._state_for_step(step)
        self._applied_weights = next_state.weights
        self._applied_optimizer = dict(next_state.optimizer)
        self._applied_trainer_state = dict(next_state.trainer_state)
        self._applied_step = step
        self.measured = True
        return {
            "loss": float("nan") if self.nonfinite_at == step else 1.0 / (step + 1),
            "gradient_norm": float(step),
        }

    def parameter_inventory(self) -> tuple[dict[str, object], ...]:
        inventory: list[dict[str, object]] = [
            {
                "name": "transformer.norm.weight",
                "module": "transformer.norm",
                "requires_grad": False,
                "digest": _sha(b"frozen"),
                "device": "mps",
                "dtype": "bfloat16",
                "shape": (1,),
            }
        ]
        if self.injected:
            inventory.extend(
                {
                    "name": f"{self.request.trainable_targets[0]}.lora_{suffix}.weight",
                    "module": self.request.trainable_targets[0],
                    "requires_grad": True,
                    "digest": _sha(f"lora-{suffix}-{len(self.steps)}".encode()),
                    "device": "mps",
                    "dtype": "bfloat16",
                    "shape": (1,),
                }
                for suffix in ("A", "B")
            )
        if self.mutate_frozen_at is not None and self.steps[-1:] == [self.mutate_frozen_at]:
            inventory[0] = {**inventory[0], "digest": _sha(b"mutated-frozen")}
        if self.rogue_trainable_at is not None and self.steps[-1:] == [self.rogue_trainable_at]:
            inventory.append(
                {
                    "name": "text_encoder.rogue.weight",
                    "module": "text_encoder.rogue",
                    "requires_grad": True,
                    "digest": _sha(b"rogue"),
                    "device": "mps",
                    "dtype": "bfloat16",
                    "shape": (1,),
                }
            )
        return tuple(inventory)

    def checkpoint_state(self, *, step: int) -> WanRuntimeCheckpointState:
        if step != self._applied_step:
            raise AssertionError("checkpoint must serialize current runtime state")
        return WanRuntimeCheckpointState(
            weights=self._applied_weights,
            optimizer=self._applied_optimizer,
            trainer_state=self._applied_trainer_state,
        )

    def _state_for_step(self, step: int) -> WanRuntimeCheckpointState:
        config = {item.name: item.value for item in self.request.optimizer.parameters}
        accumulation = int(config.get("gradient_accumulation_steps", 1))
        residual = step % accumulation
        terminal_partial = step == self.request.total_steps and residual > 0
        position = 0 if residual == 0 or terminal_partial else residual
        optimizer_backend: dict[str, object] = {
            "optimizer_updates": step // accumulation + int(terminal_partial)
        }
        if accumulation > 1:
            pending_parameters: dict[str, object] = {}
            if position:
                specs = wan_mps._trainable_specs_from_inventory(  # type: ignore[attr-defined]
                    tuple(
                        wan_mps.WanParameterObservation.model_validate(item, strict=True)
                        for item in self.parameter_inventory()
                    ),
                    self.request.trainable_targets,
                )
                for spec in specs:
                    payload = b"\x00" * (math.prod(spec.shape) * 2)
                    pending_parameters[spec.physical_name] = {
                        "digest": _sha(payload),
                        "serialized_b64": __import__("base64").b64encode(payload).decode(),
                        "shape": list(spec.shape),
                        "dtype": spec.dtype,
                        "device": spec.device,
                    }
            optimizer_backend["pending_gradients"] = {
                "accumulation_position": position,
                "parameters": pending_parameters,
            }
        backend_state = {WAN_LORA_MPS_BACKEND_ID: {"data_cursor": step}}
        return WanRuntimeCheckpointState(
            weights=_lora_weights(step),
            optimizer={
                "schema_version": "trainer-optimizer-state.v2",
                "request_id": self.request.request_id,
                "run_id": self.request.run_id,
                "step": step,
                "algorithm": self.request.optimizer.algorithm,
                "config": config,
                "global_step": step,
                "parameter_groups": [
                    {
                        "group_id": "default",
                        "parameters": list(self.request.trainable_targets),
                        "hyperparameters": config,
                    }
                ],
                "parameter_state": {
                    target: {
                        "step": step,
                        "slots": {"exp_avg_norm": step / 10, "exp_avg_sq_norm": step / 100},
                    }
                    for target in self.request.trainable_targets
                },
                "backend_state": {WAN_LORA_MPS_BACKEND_ID: optimizer_backend},
                "parent_optimizer_state_digest": self.request.optimizer_state_digest,
            },
            trainer_state={
                "schema_version": "trainer-state.v3",
                "request_id": self.request.request_id,
                "run_id": self.request.run_id,
                "step": step,
                "backend_id": self.request.backend_id,
                "seed": self.request.seed,
                "total_steps": self.request.total_steps,
                "global_step": step,
                "rng_states": _rng_states(self.request.seed, step),
                "scheduler_state": {"step": step, "rate_scale": 1.0},
                "scaler_state": {"enabled": False, "scale": 1.0},
                "backend_state": backend_state,
            },
        )

    def restore_checkpoint(self, state: WanCheckpointState) -> None:
        self.restored = state
        self.steps = list(range(1, state.step + 1))
        self._applied_weights = state.weights
        self._applied_optimizer = dict(state.optimizer)
        self._applied_trainer_state = dict(state.trainer_state)
        self._applied_step = state.step
        component = self.omit_restore_component
        if component == "weights":
            self._applied_weights = _lora_weights(0)
        elif component == "optimizer_groups":
            self._applied_optimizer["parameter_groups"] = []
        elif component == "optimizer_slots":
            self._applied_optimizer["parameter_state"] = {}
        elif component == "global_step":
            self._applied_trainer_state["global_step"] = 0
        elif component == "data_cursor":
            self._applied_trainer_state["backend_state"] = {
                WAN_LORA_MPS_BACKEND_ID: {"data_cursor": 0}
            }
        elif component in {"cpu_rng", "mps_rng"}:
            rng_states = dict(self._applied_trainer_state["rng_states"])  # type: ignore[arg-type]
            device = component.removesuffix("_rng")
            rng_states[device] = _rng_states(self.request.seed, 0)[device]
            self._applied_trainer_state["rng_states"] = rng_states
        elif component == "scheduler":
            self._applied_trainer_state["scheduler_state"] = {"step": 0}
        elif component == "scaler":
            self._applied_trainer_state["scaler_state"] = {"scale": 2.0}
        elif component == "backend":
            self._applied_trainer_state["backend_state"] = {
                WAN_LORA_MPS_BACKEND_ID: {"data_cursor": state.step, "wrong": 1}
            }

    def observe_restored_state(self) -> WanRuntimeRestoreReceipt:
        return WanRuntimeRestoreReceipt(
            weights_digest=_sha(self._applied_weights),
            optimizer_payload=canonical_json(self._applied_optimizer).encode(),
            trainer_state_payload=canonical_json(self._applied_trainer_state).encode(),
        )


class _TrainingRuntime:
    def __init__(self, observation: WanMpsObservation) -> None:
        self.observation = observation
        self.observations: list[WanMpsObservation] = []
        self.handle: _TrainingPreparedHandle | None = None
        self.release_calls = 0
        self.invalid_placement_at_call: int | None = None

    def observe_host(self) -> WanMpsObservation:
        self.observations.append(self.observation)
        if self.invalid_placement_at_call == len(self.observations):
            return self.observation.model_copy(
                update={
                    "component_placements": (
                        ("text_encoder", "mps"),
                        ("transformer", "cpu"),
                        ("transformer_2", "mps"),
                        ("vae", "mps"),
                    )
                }
            )
        return self.observation

    def prepare(self, sources: object, request: object) -> object:
        assert isinstance(request, TrainingRequest)
        self.handle = _TrainingPreparedHandle(request)
        return wan_mps.WanPreparedTraining(self.handle)

    def release(self, prepared: object | None) -> None:
        self.release_calls += 1


def _training(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    *,
    checkpoint_steps: tuple[int, ...] = (2, 4),
    clock: Callable[[], float] | None = None,
) -> tuple[WanLoRAMpsTrainer, TrainingRequest, _TrainingRuntime]:
    runtime_digest = _sha(b"runtime")
    backend_digest = _sha(b"backend")
    material_digest = _sha(b"materials")
    materials = _materials(
        local_sources,
        backend_digest=backend_digest,
        runtime_digest=runtime_digest,
        material_digest=material_digest,
    )
    runtime = _TrainingRuntime(_observation(runtime_digest))
    trainer = WanLoRAMpsTrainer(
        backend_digest=backend_digest,
        material_digest=material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,
        **({"clock": clock} if clock is not None else {}),
    )
    training = TrainingRequest(
        request_id="train",
        run_id="run",
        backend_id=WAN_LORA_MPS_BACKEND_ID,
        materials=materials,
        scratch_root=tmp_path / "training",
        total_steps=4,
        checkpoint_steps=checkpoint_steps,
        trainable_targets=("transformer.attn.to_q",),
        lora=LoRAConfiguration(rank=16, alpha=32, dropout=0.05),
        seed=7,
        deadline_seconds=60,
        optimizer=OptimizerConfiguration(
            algorithm="adamw",
            parameters=(OptimizerParameter(name="learning_rate", value=0.001),),
        ),
    )

    return trainer, training, runtime


def test_training_publishes_only_scheduled_validated_boundaries(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path)
    progress = []

    result = trainer.train(training, progress.append, lambda: False)

    validate_training_result(training, result)
    assert [item.step for item in progress] == [1, 2, 3, 4]
    assert [item.checkpoint_boundary for item in progress] == [False, True, False, True]
    assert [item.step for item in result.checkpoints] == [2, 4]
    assert result.checkpoints[0].parent_checkpoint_id is None
    assert result.checkpoints[1].parent_checkpoint_id == result.checkpoints[0].checkpoint_id
    assert runtime.handle is not None
    assert runtime.handle.steps == [1, 2, 3, 4]
    assert len(runtime.observations) == 5
    assert runtime.release_calls == 1
    trainer_state = json.loads(
        (
            training.scratch_root / result.checkpoints[-1].checkpoint_id / "trainer_state.json"
        ).read_bytes()
    )
    assert trainer_state["rng_states"] == _rng_states(7, 4)
    assert trainer_state["scheduler_state"] == {"step": 4, "rate_scale": 1.0}
    assert trainer_state["scaler_state"] == {"enabled": False, "scale": 1.0}
    assert trainer_state["backend_state"] == {WAN_LORA_MPS_BACKEND_ID: {"data_cursor": 4}}


def test_training_observes_cancellation_only_after_complete_validated_boundary(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path)
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return True

    with pytest.raises(TrainingCancelled) as raised:
        trainer.train(training, lambda _item: None, cancelled)

    assert raised.value.boundary_step == 2
    assert raised.value.result is not None
    validate_cancelled_training_result(training, raised.value.result, 2)
    assert runtime.handle is not None
    assert runtime.handle.steps == [1, 2]
    assert checks == 1
    assert runtime.release_calls == 1


def test_cancellation_release_failure_discards_checkpoint_and_returns_cleanup_error(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path)

    def fail_release(prepared: object | None) -> None:
        runtime.release_calls += 1
        raise OSError("injected release failure")

    runtime.release = fail_release  # type: ignore[method-assign]

    with pytest.raises(WanCheckpointCleanupError) as raised:
        trainer.train(training, lambda _item: None, lambda: True)

    assert isinstance(raised.value.primary_error, TrainingCancelled)
    assert raised.value.primary_error.result is not None
    assert [str(error) for error in raised.value.cleanup_errors] == [
        "injected release failure"
    ]
    assert runtime.release_calls == 1
    assert list(training.scratch_root.iterdir()) == []


def test_cancellation_release_and_discard_failures_are_aggregated_without_result(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path)
    original_discard = WanCheckpointCodec.discard

    def fail_release(prepared: object | None) -> None:
        runtime.release_calls += 1
        raise OSError("injected release failure")

    def fail_discard(codec: WanCheckpointCodec, checkpoint_id: str) -> None:
        original_discard(codec, checkpoint_id)
        raise OSError("injected discard failure")

    runtime.release = fail_release  # type: ignore[method-assign]
    monkeypatch.setattr(WanCheckpointCodec, "discard", fail_discard)

    with pytest.raises(WanCheckpointCleanupError) as raised:
        trainer.train(training, lambda _item: None, lambda: True)

    assert isinstance(raised.value.primary_error, TrainingCancelled)
    assert [str(error) for error in raised.value.cleanup_errors] == [
        "injected release failure",
        "injected discard failure",
    ]
    assert runtime.release_calls == 1
    assert list(training.scratch_root.iterdir()) == []


def test_cancellation_codec_close_failure_discards_checkpoint(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, training, _runtime = _training(local_sources, tmp_path)
    original_close = WanCheckpointCodec.close
    close_calls = 0

    def fail_first_close(codec: WanCheckpointCodec) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(codec)
        if close_calls == 1:
            raise OSError("injected codec close failure")

    monkeypatch.setattr(WanCheckpointCodec, "close", fail_first_close)

    with pytest.raises(WanCheckpointCleanupError) as raised:
        trainer.train(training, lambda _item: None, lambda: True)

    assert isinstance(raised.value.primary_error, TrainingCancelled)
    assert [str(error) for error in raised.value.cleanup_errors] == [
        "injected codec close failure"
    ]
    assert list(training.scratch_root.iterdir()) == []


def test_cancellation_lease_close_failure_discards_checkpoint(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, training, _runtime = _training(local_sources, tmp_path)
    original_close = wan_mps.WanTrainingSourceLease.close

    def fail_close(lease: wan_mps.WanTrainingSourceLease) -> None:
        original_close(lease)
        raise OSError("injected lease close failure")

    monkeypatch.setattr(wan_mps.WanTrainingSourceLease, "close", fail_close)

    with pytest.raises(WanCheckpointCleanupError) as raised:
        trainer.train(training, lambda _item: None, lambda: True)

    assert isinstance(raised.value.primary_error, TrainingCancelled)
    assert [str(error) for error in raised.value.cleanup_errors] == [
        "injected lease close failure"
    ]
    assert list(training.scratch_root.iterdir()) == []


def test_cancellation_late_close_and_discard_failures_are_aggregated(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, training, _runtime = _training(local_sources, tmp_path)
    original_close = WanCheckpointCodec.close
    original_discard = WanCheckpointCodec.discard
    close_calls = 0

    def fail_first_close(codec: WanCheckpointCodec) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(codec)
        if close_calls == 1:
            raise OSError("injected codec close failure")

    def fail_discard(codec: WanCheckpointCodec, checkpoint_id: str) -> None:
        original_discard(codec, checkpoint_id)
        raise OSError("injected late discard failure")

    monkeypatch.setattr(WanCheckpointCodec, "close", fail_first_close)
    monkeypatch.setattr(WanCheckpointCodec, "discard", fail_discard)

    with pytest.raises(WanCheckpointCleanupError) as raised:
        trainer.train(training, lambda _item: None, lambda: True)

    assert isinstance(raised.value.primary_error, TrainingCancelled)
    assert [str(error) for error in raised.value.cleanup_errors] == [
        "injected codec close failure",
        "injected late discard failure",
    ]
    assert list(training.scratch_root.iterdir()) == []


def test_cancellation_release_source_replacement_discards_every_checkpoint(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path)
    victim = local_sources[1].path()
    original_release = runtime.release

    def release(prepared: object | None) -> None:
        original_release(prepared)
        replacement = tmp_path / "release-race-replacement"
        replacement.write_bytes(victim.read_bytes())
        os.replace(replacement, victim)

    runtime.release = release  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="wan_source_invalid") as raised:
        trainer.train(training, lambda _item: None, lambda: True)

    assert isinstance(raised.value.__cause__, TrainingCancelled)
    assert raised.value.__cause__.boundary_step == 2
    assert raised.value.__cause__.result is not None
    assert runtime.release_calls == 1
    assert list(training.scratch_root.iterdir()) == []


def test_training_checkpoint_ids_and_parents_are_deterministic(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    left_trainer, left_request, _left_runtime = _training(local_sources, tmp_path / "left")
    right_trainer, right_request, _right_runtime = _training(local_sources, tmp_path / "right")

    left = left_trainer.train(left_request, lambda _item: None, lambda: False)
    right = right_trainer.train(right_request, lambda _item: None, lambda: False)

    assert [item.checkpoint_id for item in left.checkpoints] == [
        item.checkpoint_id for item in right.checkpoints
    ]
    assert [item.parent_checkpoint_id for item in left.checkpoints] == [
        item.parent_checkpoint_id for item in right.checkpoints
    ]


def test_training_rejects_frozen_mutation_and_removes_unpublished_outputs(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path)
    original_prepare = runtime.prepare

    def prepare(sources: object, request: object) -> object:
        prepared = original_prepare(sources, request)
        assert runtime.handle is not None
        runtime.handle.mutate_frozen_at = 3
        return prepared

    runtime.prepare = prepare  # type: ignore[method-assign]

    with pytest.raises(
        ValueError, match="wan_(parameter_inventory_invalid|frozen_parameters_changed)"
    ):
        trainer.train(training, lambda _item: None, lambda: False)

    assert list(training.scratch_root.iterdir()) == []
    assert runtime.release_calls == 1


def _checkpoint_descriptor(
    root: Path, checkpoint: CheckpointResult, *, checkpoint_digest: str
) -> WanParentCheckpointDescriptor:
    files = []
    for item in checkpoint.files:
        path = root / item.relative_path
        info = path.stat()
        files.append(
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
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_digest=checkpoint_digest,
        files=tuple(files),
        step=checkpoint.step,
        total_steps=4,
    )


def test_resume_resolver_exception_fails_before_runtime_preparation(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, request, runtime = _training(local_sources, tmp_path)
    resumed = request.model_copy(
        update={
            "start_step": 2,
            "checkpoint_steps": (4,),
            "telemetry_sequence_start": 3,
            "parent_checkpoint_id": "parent",
            "parent_checkpoint_digest": _sha(b"evidence"),
            "optimizer_state_digest": _sha(b"optimizer"),
        }
    )

    def fail_resolver(_request: TrainingRequest) -> WanParentCheckpointDescriptor:
        raise ValueError("resolver rejected parent")

    failing = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,
        checkpoint_resolver=fail_resolver,
    )
    with pytest.raises(ValueError, match="resolver rejected parent"):
        failing.train(resumed, lambda _item: None, lambda: False)
    assert runtime.handle is None


@pytest.mark.parametrize(
    "component",
    (
        "weights",
        "optimizer_groups",
        "optimizer_slots",
        "global_step",
        "data_cursor",
        "cpu_rng",
        "mps_rng",
        "scheduler",
        "scaler",
        "backend",
    ),
)
def test_resume_rejects_each_partially_applied_runtime_component(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    component: str,
) -> None:
    first_trainer, request, _first_runtime = _training(local_sources, tmp_path)
    with pytest.raises(TrainingCancelled) as interrupted:
        first_trainer.train(request, lambda _item: None, lambda: True)
    assert interrupted.value.result is not None
    parent = interrupted.value.result.checkpoints[-1]
    evidence_digest = _sha(b"partial-apply-evidence")
    descriptor = _checkpoint_descriptor(
        request.scratch_root, parent, checkpoint_digest=evidence_digest
    )
    optimizer_digest = next(item.digest for item in parent.files if item.role == "optimizer")
    resumed = request.model_copy(
        update={
            "start_step": 2,
            "checkpoint_steps": (4,),
            "telemetry_sequence_start": 3,
            "parent_checkpoint_id": parent.checkpoint_id,
            "parent_checkpoint_digest": evidence_digest,
            "optimizer_state_digest": optimizer_digest,
        }
    )
    runtime = _TrainingRuntime(_observation(resumed.materials.runtime_digest))
    original_prepare = runtime.prepare

    def prepare_partial(sources: object, requested: object) -> object:
        prepared = original_prepare(sources, requested)
        assert runtime.handle is not None
        runtime.handle.omit_restore_component = component
        return prepared

    runtime.prepare = prepare_partial  # type: ignore[method-assign]
    trainer = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,
        checkpoint_resolver=lambda _request: descriptor,
    )

    with pytest.raises(ValueError, match="wan_training_restore_invalid"):
        trainer.train(resumed, lambda _item: None, lambda: False)
    assert runtime.handle is not None
    assert runtime.handle.steps == [1, 2]


def test_resume_training_restores_exact_parent_and_matches_uninterrupted_final_state(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted_root.mkdir()
    resumed_root.mkdir()
    full_trainer, full_request, full_runtime = _training(local_sources, uninterrupted_root)
    full = full_trainer.train(full_request, lambda _item: None, lambda: False)

    first_trainer, first_request, _first_runtime = _training(local_sources, resumed_root)
    with pytest.raises(TrainingCancelled) as interrupted:
        first_trainer.train(first_request, lambda _item: None, lambda: True)
    assert interrupted.value.result is not None
    parent = interrupted.value.result.checkpoints[-1]
    evidence_digest = _sha(b"checkpoint-evidence")
    descriptor = _checkpoint_descriptor(
        first_request.scratch_root, parent, checkpoint_digest=evidence_digest
    )
    optimizer_digest = next(item.digest for item in parent.files if item.role == "optimizer")
    resumed = first_request.model_copy(
        update={
            "start_step": 2,
            "checkpoint_steps": (4,),
            "telemetry_sequence_start": 3,
            "parent_checkpoint_id": parent.checkpoint_id,
            "parent_checkpoint_digest": evidence_digest,
            "optimizer_state_digest": optimizer_digest,
        }
    )
    cancelled_root = tmp_path / "resume-cancelled"
    cancelled_root.mkdir()
    cancelled_request = resumed.model_copy(update={"scratch_root": cancelled_root})
    cancelled_runtime = _TrainingRuntime(_observation(resumed.materials.runtime_digest))
    cancelled_trainer = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=cancelled_runtime,
        checkpoint_resolver=lambda _request: descriptor,
    )
    with pytest.raises(TrainingCancelled) as cancelled:
        cancelled_trainer.train(cancelled_request, lambda _item: None, lambda: True)
    assert cancelled.value.boundary_step == 4
    assert cancelled_runtime.handle is not None
    assert cancelled_runtime.handle.steps == [1, 2, 3, 4]

    deadline_root = tmp_path / "resume-deadline"
    deadline_root.mkdir()
    deadline_request = resumed.model_copy(update={"scratch_root": deadline_root})
    readings = iter((0.0, 61.0))
    deadline_runtime = _TrainingRuntime(_observation(resumed.materials.runtime_digest))
    deadline_trainer = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=deadline_runtime,
        checkpoint_resolver=lambda _request: descriptor,
        clock=lambda: next(readings),
    )
    with pytest.raises(RuntimeError, match="wan_training_deadline_exceeded"):
        deadline_trainer.train(deadline_request, lambda _item: None, lambda: False)
    assert deadline_runtime.handle is not None
    assert deadline_runtime.handle.steps == [1, 2]

    noop_runtime = _TrainingRuntime(_observation(resumed.materials.runtime_digest))
    noop_prepare = noop_runtime.prepare

    def prepare_without_restore(sources: object, request: object) -> object:
        prepared = noop_prepare(sources, request)
        assert noop_runtime.handle is not None
        noop_runtime.handle.restore_checkpoint = lambda _state: None  # type: ignore[method-assign,assignment]
        return prepared

    noop_runtime.prepare = prepare_without_restore  # type: ignore[method-assign]
    noop_trainer = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=noop_runtime,
        checkpoint_resolver=lambda _request: descriptor,
    )
    with pytest.raises(ValueError, match="wan_training_restore_invalid"):
        noop_trainer.train(resumed, lambda _item: None, lambda: False)
    assert noop_runtime.handle is not None
    assert noop_runtime.handle.steps == []

    wrong_runtime = _TrainingRuntime(_observation(resumed.materials.runtime_digest))
    wrong_prepare = wrong_runtime.prepare

    def prepare_with_wrong_receipt(sources: object, request: object) -> object:
        prepared = wrong_prepare(sources, request)
        assert wrong_runtime.handle is not None

        def wrong_receipt(_state: WanCheckpointState) -> object:
            return object()

        wrong_runtime.handle.restore_checkpoint = wrong_receipt  # type: ignore[method-assign]
        return prepared

    wrong_runtime.prepare = prepare_with_wrong_receipt  # type: ignore[method-assign]
    wrong_trainer = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=wrong_runtime,
        checkpoint_resolver=lambda _request: descriptor,
    )
    with pytest.raises(ValueError, match="wan_training_restore_invalid"):
        wrong_trainer.train(resumed, lambda _item: None, lambda: False)
    assert wrong_runtime.handle is not None
    assert wrong_runtime.handle.steps == []

    racing_runtime = _TrainingRuntime(_observation(resumed.materials.runtime_digest))
    racing_prepare = racing_runtime.prepare
    checkpoint_directory = first_request.scratch_root / parent.checkpoint_id
    moved_directory = first_request.scratch_root / "parent-moved-before-first-step"

    def prepare_with_parent_replacement(sources: object, request: object) -> object:
        prepared = racing_prepare(sources, request)
        assert racing_runtime.handle is not None
        original_restore = racing_runtime.handle.restore_checkpoint

        def restore_then_replace(state: WanCheckpointState) -> None:
            original_restore(state)
            checkpoint_directory.rename(moved_directory)
            checkpoint_directory.symlink_to(moved_directory, target_is_directory=True)

        racing_runtime.handle.restore_checkpoint = restore_then_replace  # type: ignore[method-assign]
        return prepared

    racing_runtime.prepare = prepare_with_parent_replacement  # type: ignore[method-assign]
    racing_trainer = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=racing_runtime,
        checkpoint_resolver=lambda _request: descriptor,
    )
    with pytest.raises(ValueError, match="wan_checkpoint_restore_invalid"):
        racing_trainer.train(resumed, lambda _item: None, lambda: False)
    assert racing_runtime.handle is not None
    assert racing_runtime.handle.steps == [1, 2]
    checkpoint_directory.unlink()
    moved_directory.rename(checkpoint_directory)

    resumed_runtime = _TrainingRuntime(_observation(resumed.materials.runtime_digest))
    resumed_trainer = WanLoRAMpsTrainer(
        backend_digest=resumed.materials.backend_digest,
        material_digest=resumed.materials.material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=resumed_runtime,
        checkpoint_resolver=lambda _request: descriptor,
    )

    result = resumed_trainer.train(resumed, lambda _item: None, lambda: False)

    assert resumed_runtime.handle is not None
    assert resumed_runtime.handle.restored is not None
    assert resumed_runtime.handle.restored.weights == (first_request.scratch_root / next(
        item.relative_path for item in parent.files if item.role == "weights"
    )).read_bytes()
    resumed_files = {item.role: item for item in result.checkpoints[-1].files}
    full_files = {item.role: item for item in full.checkpoints[-1].files}
    assert resumed_files["weights"].digest == full_files["weights"].digest
    resumed_state = resumed_runtime.handle.checkpoint_state(step=4)
    full_state = full_runtime.handle.checkpoint_state(step=4)
    assert resumed_state.weights == full_state.weights
    assert resumed_state.trainer_state == full_state.trainer_state
    assert {
        key: value
        for key, value in resumed_state.optimizer.items()
        if key != "parent_optimizer_state_digest"
    } == {
        key: value
        for key, value in full_state.optimizer.items()
        if key != "parent_optimizer_state_digest"
    }
    assert resumed_state.optimizer["parent_optimizer_state_digest"] == optimizer_digest


@pytest.mark.parametrize("fault", ("nonfinite", "rogue"))
def test_training_independently_rejects_nonfinite_or_rogue_trainable_state(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path, fault: str
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path, checkpoint_steps=(1, 4))
    original_prepare = runtime.prepare

    def prepare(sources: object, request: object) -> object:
        prepared = original_prepare(sources, request)
        assert runtime.handle is not None
        if fault == "nonfinite":
            runtime.handle.nonfinite_at = 3
        else:
            runtime.handle.rogue_trainable_at = 3
        return prepared

    runtime.prepare = prepare  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        trainer.train(training, lambda _item: None, lambda: False)

    assert list(training.scratch_root.iterdir()) == []
    assert runtime.release_calls == 1


def test_training_rechecks_placement_each_step_and_discards_prior_boundary(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path, checkpoint_steps=(1, 4))
    runtime.invalid_placement_at_call = 3

    with pytest.raises(ValueError, match="wan_training_placement_invalid"):
        trainer.train(training, lambda _item: None, lambda: False)

    assert len(runtime.observations) == 3
    assert list(training.scratch_root.iterdir()) == []
    assert runtime.release_calls == 1


def test_training_deadline_discards_prior_boundary_and_releases(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    readings = iter((0.0, 0.1, 0.2, 0.3, 0.4, 61.0))
    trainer, training, runtime = _training(
        local_sources,
        tmp_path,
        checkpoint_steps=(1, 4),
        clock=lambda: next(readings),
    )

    with pytest.raises(RuntimeError, match="wan_training_deadline_exceeded"):
        trainer.train(training, lambda _item: None, lambda: False)

    assert list(training.scratch_root.iterdir()) == []
    assert runtime.release_calls == 1


def test_training_runtime_exception_is_bounded_and_discards_prior_boundary(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path)
    original_prepare = runtime.prepare

    def prepare(sources: object, request: object) -> object:
        prepared = original_prepare(sources, request)
        assert runtime.handle is not None
        runtime.handle.error_at = 3
        return prepared

    runtime.prepare = prepare  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="wan_training_runtime_failed"):
        trainer.train(training, lambda _item: None, lambda: False)

    assert list(training.scratch_root.iterdir()) == []
    assert runtime.release_calls == 1


def test_training_keeps_source_lease_live_and_reverifies_each_step(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path, checkpoint_steps=(1, 4))
    real_observe = runtime.observe_host
    victim = local_sources[1].path()

    def observe_host() -> WanMpsObservation:
        observation = real_observe()
        if len(runtime.observations) == 3:
            replacement = tmp_path / "replacement-source"
            replacement.write_bytes(victim.read_bytes())
            os.replace(replacement, victim)
        return observation

    runtime.observe_host = observe_host  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="wan_source_invalid"):
        trainer.train(training, lambda _item: None, lambda: False)

    assert list(training.scratch_root.iterdir()) == []
    assert runtime.release_calls == 1


def test_training_rejects_replaced_scratch_root_and_cleans_retained_directory(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path
) -> None:
    trainer, training, runtime = _training(local_sources, tmp_path, checkpoint_steps=(1, 2, 4))
    real_observe = runtime.observe_host
    retained = tmp_path / "retained-scratch"

    def observe_host() -> WanMpsObservation:
        observation = real_observe()
        if len(runtime.observations) == 3:
            training.scratch_root.rename(retained)
            training.scratch_root.mkdir()
        return observation

    runtime.observe_host = observe_host  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="unsafe_scratch_root"):
        trainer.train(training, lambda _item: None, lambda: False)

    assert list(retained.iterdir()) == []
    assert list(training.scratch_root.iterdir()) == []
    assert runtime.release_calls == 1


@pytest.mark.parametrize(
    "stage",
    (
        "train_step",
        "checkpoint_state",
        "publication",
        "prefix_validation",
        "final_validation",
        "release",
    ),
)
def test_training_final_boundary_reverifies_source_after_every_untrusted_call(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    trainer, training, runtime = _training(
        local_sources, tmp_path, checkpoint_steps=(4,)
    )
    victim = local_sources[1].path()
    replaced = False

    def replace_source() -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        replacement = tmp_path / f"replacement-{stage}"
        replacement.write_bytes(victim.read_bytes())
        os.replace(replacement, victim)

    original_prepare = runtime.prepare

    def prepare(sources: object, request: object) -> object:
        prepared = original_prepare(sources, request)
        assert runtime.handle is not None
        if stage == "train_step":
            original = runtime.handle.train_step

            def train_step(*, step: int, total_steps: int) -> dict[str, float]:
                measurement = original(step=step, total_steps=total_steps)
                if step == total_steps:
                    replace_source()
                return measurement

            runtime.handle.train_step = train_step  # type: ignore[method-assign]
        elif stage == "checkpoint_state":
            original_checkpoint = runtime.handle.checkpoint_state

            def checkpoint_state(*, step: int) -> WanRuntimeCheckpointState:
                state = original_checkpoint(step=step)
                replace_source()
                return state

            runtime.handle.checkpoint_state = checkpoint_state  # type: ignore[method-assign]
        return prepared

    runtime.prepare = prepare  # type: ignore[method-assign]

    if stage == "publication":
        original_write = WanCheckpointCodec.write

        def write(codec: object, state: object, context: object) -> object:
            checkpoint = original_write(codec, state, context)  # type: ignore[arg-type]
            replace_source()
            return checkpoint

        monkeypatch.setattr(WanCheckpointCodec, "write", write)
    elif stage == "prefix_validation":
        original_prefix_validation = validate_cancelled_training_result

        def validate_prefix(request: object, result: object, boundary: int) -> None:
            original_prefix_validation(request, result, boundary)  # type: ignore[arg-type]
            replace_source()

        monkeypatch.setattr(
            "video_lab.trainers.wan_mps.validate_cancelled_training_result", validate_prefix
        )
    elif stage == "final_validation":
        original_final_validation = validate_training_result

        def validate_final(request: object, result: object) -> None:
            original_final_validation(request, result)  # type: ignore[arg-type]
            replace_source()

        monkeypatch.setattr(
            "video_lab.trainers.wan_mps.validate_training_result", validate_final
        )
    elif stage == "release":
        original_release = runtime.release

        def release(prepared: object | None) -> None:
            original_release(prepared)
            replace_source()

        runtime.release = release  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="wan_source_invalid"):
        trainer.train(training, lambda _item: None, lambda: False)

    assert replaced is True
    assert list(training.scratch_root.iterdir()) == []


def test_training_preserves_primary_and_checkpoint_cleanup_failures(
    local_sources: tuple[WanSourceDescriptor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, training, _runtime = _training(
        local_sources, tmp_path, checkpoint_steps=(4,)
    )

    def fail_validation(request: object, result: object) -> None:
        raise ValueError("injected result validation failure")

    original_discard = WanCheckpointCodec.discard

    def fail_cleanup(codec: WanCheckpointCodec, checkpoint_id: str) -> None:
        original_discard(codec, checkpoint_id)
        raise OSError("injected checkpoint cleanup failure")

    monkeypatch.setattr(
        "video_lab.trainers.wan_mps.validate_training_result", fail_validation
    )
    monkeypatch.setattr(WanCheckpointCodec, "discard", fail_cleanup)

    with pytest.raises(RuntimeError, match="wan_checkpoint_cleanup_failed") as raised:
        trainer.train(training, lambda _item: None, lambda: False)

    assert isinstance(raised.value, WanCheckpointCleanupError)
    assert str(raised.value.primary_error) == "injected result validation failure"
    assert [str(error) for error in raised.value.cleanup_errors] == [
        "injected checkpoint cleanup failure"
    ]
    assert list(training.scratch_root.iterdir()) == []


@dataclass
class _CalibrationPreparedHandle:
    eligible: tuple[str, ...]
    rogue_trainable: bool = False
    rogue_after_measurement: bool = False
    mutate_frozen: bool = False
    measurement_updates: dict[str, float] | None = None
    measurement_error: BaseException | None = None

    def __post_init__(self) -> None:
        self.injected = False
        self.measured = False
        self.inject_arguments: dict[str, object] | None = None

    def eligible_lora_modules(self) -> tuple[str, ...]:
        return self.eligible

    def inject_lora(
        self,
        *,
        targets: tuple[str, ...],
        configuration: LoRAConfiguration,
        seed: int,
        device: str,
        dtype: str,
    ) -> None:
        self.injected = True
        self.inject_arguments = {
            "targets": targets,
            "rank": configuration.rank,
            "alpha": configuration.alpha,
            "dropout": configuration.dropout,
            "seed": seed,
            "device": device,
            "dtype": dtype,
        }

    def parameter_inventory(self) -> tuple[dict[str, object], ...]:
        frozen_bytes = b"frozen-mutated" if self.measured and self.mutate_frozen else b"frozen"
        frozen_digest = _sha(frozen_bytes)
        inventory: list[dict[str, object]] = [
            {
                "name": "transformer.blocks.0.norm.weight",
                "module": "transformer.blocks.0.norm",
                "requires_grad": False,
                "digest": frozen_digest,
                "device": "mps",
                "dtype": "bfloat16",
                "shape": (1,),
            }
        ]
        if self.injected:
            for target in self.eligible:
                inventory.extend(
                    (
                        {
                            "name": f"{target}.lora_A.weight",
                            "module": target,
                            "requires_grad": True,
                            "digest": _sha(f"{target}:A".encode()),
                            "device": "mps",
                            "dtype": "bfloat16",
                            "shape": (1,),
                        },
                        {
                            "name": f"{target}.lora_B.weight",
                            "module": target,
                            "requires_grad": True,
                            "digest": _sha(f"{target}:B".encode()),
                            "device": "mps",
                            "dtype": "bfloat16",
                            "shape": (1,),
                        },
                    )
                )
        if self.rogue_trainable or (self.measured and self.rogue_after_measurement):
            inventory.append(
                {
                    "name": "text_encoder.rogue.weight",
                    "module": "text_encoder.rogue",
                    "requires_grad": True,
                    "digest": _sha(b"rogue"),
                    "device": "mps",
                    "dtype": "bfloat16",
                    "shape": (1,),
                }
            )
        return tuple(inventory)

    def measure_forward_backward(self) -> dict[str, float]:
        self.measured = True
        if self.measurement_error is not None:
            raise self.measurement_error
        measured = {
            "loss": 0.25,
            "gradient_norm": 1.5,
            "wall_seconds": 0.125,
            "rss_start_bytes": 1_000.0,
            "rss_end_bytes": 1_100.0,
            "mps_allocated_start_bytes": 200.0,
            "mps_allocated_end_bytes": 240.0,
            "mps_driver_start_bytes": 300.0,
            "mps_driver_end_bytes": 350.0,
        }
        measured.update(self.measurement_updates or {})
        return measured


class _CalibrationRuntime:
    def __init__(self, observation: WanMpsObservation, handle: _CalibrationPreparedHandle) -> None:
        self.observation = observation
        self.handle = handle
        self.prepared: object | None = None
        self.prepare_error: BaseException | None = None
        self.partial_allocation = False
        self.release_calls = 0
        self.release_error: BaseException | None = None

    def observe_host(self) -> WanMpsObservation:
        return self.observation

    def prepare(self, sources: object, request: object) -> object:
        if self.prepare_error is not None:
            self.partial_allocation = True
            raise self.prepare_error
        self.prepared = wan_mps.WanPreparedTraining(self.handle)
        return self.prepared

    def release(self, prepared: object | None) -> None:
        assert prepared is self.prepared
        self.release_calls += 1
        self.partial_allocation = False
        if self.release_error is not None:
            raise self.release_error


def _calibration(
    local_sources: tuple[WanSourceDescriptor, ...],
    handle: _CalibrationPreparedHandle,
    *,
    targets: tuple[str, ...] | None = None,
    steps: int = 1,
) -> tuple[WanLoRAMpsTrainer, CalibrationRequest, _CalibrationRuntime]:
    runtime_digest = _sha(b"runtime")
    backend_digest = _sha(b"backend")
    material_digest = _sha(b"materials")
    runtime = _CalibrationRuntime(_observation(runtime_digest), handle)
    trainer = WanLoRAMpsTrainer(
        backend_digest=backend_digest,
        material_digest=material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,  # type: ignore[arg-type]
    )
    request = CalibrationRequest(
        request_id="calibrate-1",
        run_id="run-1",
        backend_id=WAN_LORA_MPS_BACKEND_ID,
        materials=_materials(
            local_sources,
            backend_digest=backend_digest,
            runtime_digest=runtime_digest,
            material_digest=material_digest,
        ),
        scratch_root=Path("/strictly-local/calibration"),
        steps=steps,
        trainable_targets=targets or handle.eligible,
        lora=LoRAConfiguration(rank=16, alpha=32, dropout=0.05),
        seed=7,
        deadline_seconds=60,
    )
    return trainer, request, runtime


def test_calibration_exactly_maps_lora_policy_and_emits_finite_proofs(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    assert getattr(wan_mps, "LoRATargetPolicy", None) is not None
    targets = (
        "transformer.blocks.0.attn.to_k",
        "transformer.blocks.0.attn.to_q",
    )
    handle = _CalibrationPreparedHandle(targets)
    trainer, request, runtime = _calibration(local_sources, handle)
    progress = []

    result = trainer.calibrate(request, progress.append, lambda: False)

    assert handle.inject_arguments == {
        "targets": targets,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "seed": 7,
        "device": "mps",
        "dtype": "bfloat16",
    }
    assert result.changed_targets == targets
    assert result.finite_gradients is True
    assert result.frozen_parameters_unchanged is True
    assert result.measurements == {
        "loss": 0.25,
        "gradient_norm": 1.5,
        "wall_seconds": 0.125,
        "rss_start_bytes": 1_000.0,
        "rss_end_bytes": 1_100.0,
        "mps_allocated_start_bytes": 200.0,
        "mps_allocated_end_bytes": 240.0,
        "mps_driver_start_bytes": 300.0,
        "mps_driver_end_bytes": 350.0,
    }
    assert [item.step for item in progress] == [1]
    assert runtime.release_calls == 1


def test_zero_step_calibration_still_measures_canary_without_progress(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    handle = _CalibrationPreparedHandle(("transformer.blocks.0.attn.to_q",))
    trainer, request, runtime = _calibration(local_sources, handle, steps=0)
    progress = []

    result = trainer.calibrate(request, progress.append, lambda: False)

    assert handle.measured is True
    assert progress == []
    assert result.measurements["loss"] == 0.25
    assert runtime.release_calls == 1


@pytest.mark.parametrize(
    "targets",
    (
        ("transformer.blocks.0.attn.to_q",),
        (
            "transformer.blocks.0.attn.to_q",
            "transformer.blocks.0.attn.to_q",
        ),
        ("to_q", "transformer.blocks.0.attn.to_k"),
    ),
)
def test_calibration_rejects_partial_duplicate_or_missing_exact_targets_and_releases(
    local_sources: tuple[WanSourceDescriptor, ...], targets: tuple[str, ...]
) -> None:
    eligible = (
        "transformer.blocks.0.attn.to_k",
        "transformer.blocks.0.attn.to_q",
    )
    handle = _CalibrationPreparedHandle(eligible)
    trainer, request, runtime = _calibration(local_sources, handle, targets=targets)

    with pytest.raises(ValueError, match="wan_lora_targets_invalid"):
        trainer.calibrate(request, lambda _item: None, lambda: False)

    assert handle.injected is False
    assert runtime.release_calls == 1


@pytest.mark.parametrize("fault", ("rogue", "rogue_after", "frozen"))
def test_calibration_independently_rejects_trainable_or_frozen_inventory_drift(
    local_sources: tuple[WanSourceDescriptor, ...], fault: str
) -> None:
    handle = _CalibrationPreparedHandle(
        ("transformer.blocks.0.attn.to_q",),
        rogue_trainable=fault == "rogue",
        rogue_after_measurement=fault == "rogue_after",
        mutate_frozen=fault == "frozen",
    )
    trainer, request, runtime = _calibration(local_sources, handle)

    expected = "wan_(parameter_inventory_invalid|frozen_parameters_changed)"
    with pytest.raises(ValueError, match=expected):
        trainer.calibrate(request, lambda _item: None, lambda: False)

    assert runtime.release_calls == 1


@pytest.mark.parametrize(
    "updates",
    (
        {"loss": float("nan")},
        {"gradient_norm": float("inf")},
        {"rss_end_bytes": -1.0},
        {"mps_driver_end_bytes": float("nan")},
    ),
)
def test_calibration_rejects_nonfinite_measurement_matrix_and_releases(
    local_sources: tuple[WanSourceDescriptor, ...], updates: dict[str, float]
) -> None:
    handle = _CalibrationPreparedHandle(
        ("transformer.blocks.0.attn.to_q",), measurement_updates=updates
    )
    trainer, request, runtime = _calibration(local_sources, handle)

    with pytest.raises(ValueError, match="wan_calibration_measurement_invalid"):
        trainer.calibrate(request, lambda _item: None, lambda: False)

    assert runtime.release_calls == 1


@pytest.mark.parametrize("error", (MemoryError("oom details"), RuntimeError("operator details")))
def test_calibration_bounds_runtime_failures_and_releases(
    local_sources: tuple[WanSourceDescriptor, ...], error: BaseException
) -> None:
    handle = _CalibrationPreparedHandle(
        ("transformer.blocks.0.attn.to_q",), measurement_error=error
    )
    trainer, request, runtime = _calibration(local_sources, handle)

    with pytest.raises(RuntimeError) as raised:
        trainer.calibrate(request, lambda _item: None, lambda: False)

    assert str(raised.value) == "wan_calibration_runtime_failed"
    assert runtime.release_calls == 1


@pytest.mark.parametrize(
    "prepare_error",
    (RuntimeError("prepare"), MemoryError("oom"), KeyboardInterrupt("interrupt")),
)
def test_calibration_prepare_failure_releases_none_and_closes_source_lease(
    local_sources: tuple[WanSourceDescriptor, ...], prepare_error: BaseException
) -> None:
    handle = _CalibrationPreparedHandle(("transformer.blocks.0.attn.to_q",))
    trainer, request, runtime = _calibration(local_sources, handle)
    runtime.prepare_error = prepare_error
    before = len(os.listdir("/dev/fd"))

    with pytest.raises(RuntimeError) as raised:
        trainer.calibrate(request, lambda _item: None, lambda: False)

    assert str(raised.value) == "wan_calibration_runtime_failed"
    assert runtime.release_calls == 1
    assert runtime.partial_allocation is False
    assert len(os.listdir("/dev/fd")) == before


@pytest.mark.parametrize("release_error", (RuntimeError("release"), KeyboardInterrupt("interrupt")))
def test_prepare_primary_failure_wins_when_release_none_also_fails(
    local_sources: tuple[WanSourceDescriptor, ...], release_error: BaseException
) -> None:
    handle = _CalibrationPreparedHandle(("transformer.blocks.0.attn.to_q",))
    trainer, request, runtime = _calibration(local_sources, handle)
    runtime.prepare_error = MemoryError("primary details")
    runtime.release_error = release_error
    before = len(os.listdir("/dev/fd"))

    with pytest.raises(RuntimeError) as raised:
        trainer.calibrate(request, lambda _item: None, lambda: False)

    assert str(raised.value) == "wan_calibration_runtime_failed"
    assert runtime.release_calls == 1
    assert len(os.listdir("/dev/fd")) == before


def test_calibration_cancellation_after_prepare_releases_without_measuring(
    local_sources: tuple[WanSourceDescriptor, ...],
) -> None:
    handle = _CalibrationPreparedHandle(("transformer.blocks.0.attn.to_q",))
    trainer, request, runtime = _calibration(local_sources, handle)

    with pytest.raises(TrainingCancelled):
        trainer.calibrate(request, lambda _item: None, lambda: True)

    assert handle.measured is False
    assert runtime.release_calls == 1


@pytest.mark.parametrize(
    "release_error",
    (RuntimeError("details"), MemoryError("oom"), KeyboardInterrupt("interrupt")),
)
def test_calibration_release_failure_is_bounded_and_still_closes_source_lease(
    local_sources: tuple[WanSourceDescriptor, ...], release_error: BaseException
) -> None:
    handle = _CalibrationPreparedHandle(("transformer.blocks.0.attn.to_q",))
    trainer, request, runtime = _calibration(local_sources, handle)
    runtime.release_error = release_error
    before = len(os.listdir("/dev/fd"))

    with pytest.raises(RuntimeError) as raised:
        trainer.calibrate(request, lambda _item: None, lambda: False)

    assert str(raised.value) == "wan_calibration_runtime_failed"
    assert runtime.release_calls == 1
    assert len(os.listdir("/dev/fd")) == before


@pytest.mark.parametrize("race", ("replacement", "symlink", "delete"))
def test_probe_reverifies_and_closes_source_lease_after_observation_race(
    local_sources: tuple[WanSourceDescriptor, ...], tmp_path: Path, race: str
) -> None:
    victim = local_sources[1].path()

    class RacingRuntime(ObservedRuntime):
        def observe_host(self) -> WanMpsObservation:
            observation = super().observe_host()
            if race == "replacement":
                replacement = tmp_path / "replacement.safetensors"
                replacement.write_bytes(b"animegen-high")
                os.replace(replacement, victim)
            elif race == "symlink":
                target = tmp_path / "target.safetensors"
                target.write_bytes(b"animegen-high")
                victim.unlink()
                victim.symlink_to(target)
            else:
                victim.unlink()
            return observation

    before = len(os.listdir("/dev/fd"))
    runtime = RacingRuntime(_observation(_sha(b"runtime")))
    backend_digest = _sha(b"backend")
    material_digest = _sha(b"materials")
    trainer = WanLoRAMpsTrainer(
        backend_digest=backend_digest,
        material_digest=material_digest,
        source_resolver=WanTrainingSourceResolver(local_sources),
        runtime=runtime,
    )
    report = trainer.probe(
        ProbeTrainingRequest(
            request_id=f"race-{race}",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=backend_digest,
            materials=_materials(
                local_sources,
                backend_digest=backend_digest,
                runtime_digest=_sha(b"runtime"),
                material_digest=material_digest,
            ),
            scratch_root=Path("/scratch"),
        )
    )

    assert report.supported is False
    assert runtime.prepare_calls == 0
    assert len(os.listdir("/dev/fd")) == before


def test_source_resolver_rejects_symlinked_ancestor_directory(tmp_path: Path) -> None:
    authorized = tmp_path / "authorized"
    authorized.mkdir()
    source = authorized / "base.safetensors"
    source.write_bytes(b"exact-bytes")
    descriptor = _descriptor("base_model", source)
    moved = tmp_path / "moved"
    authorized.rename(moved)
    authorized.symlink_to(moved, target_is_directory=True)

    with pytest.raises(ValueError, match="wan_source_invalid"):
        WanTrainingSourceResolver((descriptor,)).acquire()
