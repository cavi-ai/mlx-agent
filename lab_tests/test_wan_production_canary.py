from __future__ import annotations

import gc
import hashlib
import multiprocessing
import subprocess
import sys
import threading
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as torch_library
from pydantic import ValidationError

import video_lab.artifacts as artifacts_module
import video_lab.host_canaries.wan_curated as canary_module
from video_lab.artifacts import ArtifactIntegrityError, ArtifactStore
from video_lab.database import Database
from video_lab.domain import canonical_json
from video_lab.host_canaries.wan_curated import (
    WanCanaryCancelled,
    WanCapacityExceeded,
    WanProductionCanaryExecutionV2,
    WanProductionCanaryUnavailableV2,
    _publish_checkpoint,
    _WanCanaryHooks,
    run_wan_curated_host_canary,
)
from video_lab.local_authority import RetainedDirectoryAuthority
from video_lab.trainers.wan_branch_loader import WanBranchLoadRequest
from video_lab.trainers.wan_factory import WanPinnedLocalRoot, root_manifest_digest
from video_lab.trainers.wan_mps import WanSourceDescriptor, WanTrainingSourceResolver
from video_lab.trainers.wan_production_models import (
    WanBranchCanaryResultV1,
    WanBranchRecipeV1,
    WanCapacityResultV2,
    WanComponentInventoryV1,
    WanNumericalToleranceV1,
    WanProductionCanaryReceiptV2,
    WanProductionCanaryRequestV2,
    WanTensorEnvelopeV2,
    wan_branch_recipe_digest,
    wan_inventory_digest,
    wan_inventory_family_digest,
    wan_recipe_family_digest,
)

_D = "sha256:" + "1" * 64


@dataclass
class _CheckpointManifest:
    branch: str


def _checkpoint(branch: str = "high_noise") -> SimpleNamespace:
    return SimpleNamespace(
        manifest=_CheckpointManifest(branch),
        weights=b"weights",
        optimizer={},
        cpu_rng_state=b"cpu",
        mps_rng_state=b"mps",
    )


def _assert_no_checkpoint_evidence(root: Path) -> None:
    assert tuple(root.glob("*.checkpoint.json")) == ()
    assert tuple(root.glob(".*.tmp")) == ()


def _checkpoint_creator_with_failed_final_reverify(
    root: str,
    installed: object,
    publisher_started: object,
    publisher_finished: object,
    results: object,
) -> None:
    authority = RetainedDirectoryAuthority.capture(
        Path(root), error_code="wan_canary_checkpoint_root_changed"
    )
    real_reverify = RetainedDirectoryAuthority.reverify

    def fail_after_install(self: RetainedDirectoryAuthority) -> None:
        real_reverify(self)
        if self is authority and tuple(Path(root).glob("*.checkpoint.json")):
            installed.set()  # type: ignore[attr-defined]
            if not publisher_started.wait(timeout=10):  # type: ignore[attr-defined]
                raise TimeoutError("deduplicating checkpoint publisher did not start")
            publisher_finished.wait(timeout=2)  # type: ignore[attr-defined]
            raise ValueError("wan_canary_checkpoint_root_changed")

    RetainedDirectoryAuthority.reverify = fail_after_install
    try:
        try:
            _publish_checkpoint(
                Path(root), "high_noise", _checkpoint(), root_authority=authority
            )
        except BaseException as error:
            results.put(("creator", "error", str(error)))  # type: ignore[attr-defined]
        else:
            results.put(("creator", "success", ""))  # type: ignore[attr-defined]
    finally:
        RetainedDirectoryAuthority.reverify = real_reverify
        authority.close()


def _checkpoint_deduplicator(
    root: str,
    publisher_started: object,
    publisher_finished: object,
    results: object,
) -> None:
    authority = RetainedDirectoryAuthority.capture(
        Path(root), error_code="wan_canary_checkpoint_root_changed"
    )
    publisher_started.set()  # type: ignore[attr-defined]
    try:
        try:
            _publish_checkpoint(
                Path(root), "high_noise", _checkpoint(), root_authority=authority
            )
        except BaseException as error:
            results.put(("deduplicator", "error", str(error)))  # type: ignore[attr-defined]
        else:
            results.put(("deduplicator", "success", ""))  # type: ignore[attr-defined]
    finally:
        publisher_finished.set()  # type: ignore[attr-defined]
        authority.close()


def test_documented_production_marker_collects_only_real_enabled_node() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "wan_mps_production_host",
            "lab_tests/host/test_wan_mps_production_adapter_canary.py",
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = tuple(line for line in completed.stdout.splitlines() if "::" in line)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert collected == (
        "lab_tests/host/test_wan_mps_production_adapter_canary.py"
        "::test_wan_mps_production_adapter_canary",
    )


def _recipe(branch: str) -> WanBranchRecipeV1:
    return WanBranchRecipeV1(
        schema_version="wan-mps-branch-recipe.v1", branch=branch, rank=2, alpha=2,
        dropout=0.0, target_suffixes=("to_k", "to_out.0", "to_q", "to_v"),
        optimizer_algorithm="adamw", optimizer_parameters=(("learning_rate", 0.0001),),
        gradient_clip_norm=1.0, seed=7,
        tolerance=WanNumericalToleranceV1(atol=1e-5, rtol=1e-4),
    )


def _inventory(suffix: str) -> WanComponentInventoryV1:
    return WanComponentInventoryV1(
        schema_version="wan-mps-component-inventory.v1",
        component_manifests=tuple(
            (name, "sha256:" + suffix * 64)
            for name in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
        ),
    )


def _request() -> WanProductionCanaryRequestV2:
    return WanProductionCanaryRequestV2(
        schema_version="wan-mps-production-canary-request.v2",
        recipes=(_recipe("high_noise"), _recipe("low_noise")),
        inventories=(_inventory("3"), _inventory("4")),
        compilation_receipt_digest=_D, batch_digest="sha256:" + "2" * 64,
        primary_run_id="run-primary-0001", independent_run_id="run-independent-0002",
        requested_tensor_envelope=(
            WanTensorEnvelopeV2(
                name="activation", role="activation", shape=(1, 3, 4),
                dtype="bfloat16", copies=1,
            ),
            WanTensorEnvelopeV2(
                name="adamw", role="optimizer", shape=(2, 4),
                dtype="float32", copies=2,
            ),
            WanTensorEnvelopeV2(
                name="adapter", role="parameter", shape=(2, 4),
                dtype="bfloat16", copies=1,
            ),
        ),
        memory_release_bound_bytes=16, required_artifact_bytes=1024,
        dependency_versions=(("torch", "2.8.0"),),
        hardware_identity=(("device", "mps"), ("machine", "arm64"), ("system", "Darwin")),
    )


def _concrete_execution(tmp_path, request=None) -> WanProductionCanaryExecutionV2:
    authority_root = tmp_path / f"authority-{uuid.uuid4().hex}"
    checkpoint_root = authority_root / "checkpoints"
    output_root = authority_root / "artifacts"
    model_root = authority_root / "model-authority"
    source_root = authority_root / "compiled-authority"
    snapshot_parent = authority_root / "snapshots"
    authority_root.mkdir()
    checkpoint_root.mkdir()
    output_root.mkdir()
    model_root.mkdir()
    source_root.mkdir()
    snapshot_parent.mkdir()
    base_model = source_root / "model_index.json"
    batch = source_root / "batch.safetensors"
    base_model.write_bytes(b"model")
    batch.write_bytes(b"batch")
    descriptors = tuple(
        WanSourceDescriptor(
            name=name,
            path=lambda path=path: path,
            expected_digest="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_size=path.stat().st_size,
            expected_device=path.stat().st_dev,
            expected_inode=path.stat().st_ino,
        )
        for name, path in (("base_model", base_model), ("batch_tensors", batch))
    )
    source_lease = WanTrainingSourceResolver(descriptors).acquire()
    base_model.chmod(0o400)
    batch.chmod(0o400)
    source_root.chmod(0o500)
    model_root.chmod(0o500)
    return WanProductionCanaryExecutionV2(
        request=request or _request(),
        branch_load_request=WanBranchLoadRequest(
            model_root=WanPinnedLocalRoot(model_root, "a" * 40, root_manifest_digest(model_root)),
            snapshot_parent=snapshot_parent,
            branch="high_noise",
            scheduler=Path("scheduler"),
            text_encoder=Path("text_encoder"),
            tokenizer=Path("tokenizer"),
            transformer_config=Path("transformer/config.json"),
            high_transformer=Path("weights/high.safetensors"),
            low_transformer=Path("weights/low.safetensors"),
            vae=Path("vae"),
        ),
        training_sources=source_lease.sources,
        training_source_lease=source_lease,
        compilation_receipt_id="receipt-1",
        checkpoint_root=checkpoint_root,
        output_root=output_root,
    )


class _FakeMps:
    def __init__(self, *, built=True, available=True) -> None:
        self._built, self._available = built, available

    def is_built(self):
        return self._built

    def is_available(self):
        return self._available

    def synchronize(self):
        return None

    def empty_cache(self):
        return None

    def driver_allocated_memory(self):
        return 10

    def recommended_max_memory(self):
        return 1_000

    def current_allocated_memory(self):
        return 0


class _FakeTorch:
    bfloat16 = "bfloat16"

    def __init__(self, *, built=True, available=True, probe_error=False) -> None:
        self.backends = SimpleNamespace(mps=_FakeMps(built=built, available=available))
        self.mps = self.backends.mps
        self.probe_error = probe_error

    def empty(self, size, *, device, dtype):
        if self.probe_error:
            raise RuntimeError("no bf16")
        return SimpleNamespace(device=device, dtype=dtype)


class _IndexedMpsDevice:
    type = "mps"

    def __str__(self) -> str:
        return "mps:0"


def _install_preflight_doubles(monkeypatch, execution, torch) -> object:
    receipt = SimpleNamespace(model_dump=lambda mode: {"receipt": "accepted"})
    payload = canonical_json(receipt.model_dump("json")).encode()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    object.__setattr__(
        execution, "request",
        execution.request.model_copy(
            update={
                "compilation_receipt_digest": digest,
                "batch_digest": next(
                    item.digest
                    for item in execution.training_sources.files
                    if item.name == "batch_tensors"
                ),
            }
        ),
    )
    monkeypatch.setattr(
        canary_module, "DatasetCompilationRepository",
        lambda database: SimpleNamespace(receipt=lambda receipt_id: receipt),
    )
    monkeypatch.setattr(canary_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(canary_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(canary_module.importlib.metadata, "version", lambda name: "2.8.0")
    monkeypatch.setattr(
        canary_module.shutil, "disk_usage", lambda root: SimpleNamespace(free=10_000)
    )
    return torch


@dataclass
class _Handle:
    branch: str
    role: str


def _hooks(events: list[str], *, fail: str | None = None) -> _WanCanaryHooks:
    def event(name: str, value: object = None) -> object:
        events.append(name)
        if name == fail:
            raise RuntimeError("injected")
        return value

    def restore(handle: object, recipe: WanBranchRecipeV1) -> None:
        typed = handle
        assert isinstance(typed, _Handle)
        event(f"restore:{typed.branch}")

    def continuity(handle: object, recipe: WanBranchRecipeV1) -> WanBranchCanaryResultV1:
        typed = handle
        assert isinstance(typed, _Handle)
        event(f"continuity:{typed.branch}")
        return WanBranchCanaryResultV1(
            schema_version="wan-mps-branch-canary-result.v1", branch=recipe.branch,
            recipe_digest=wan_branch_recipe_digest(recipe),
            output_digest="sha256:" + ("7" if recipe.branch == "high_noise" else "8") * 64,
            max_absolute_error=0.0, max_relative_error=0.0,
        )

    return _WanCanaryHooks(
        preflight=lambda: None,
        inventory=lambda branch, expected: event(f"inventory:{branch}", expected),
        device_preflight=lambda: event("device_preflight"),
        load=lambda branch, role: event(f"load_{role}:{branch}", _Handle(branch, role)),
        step=lambda handle, recipe: event(f"step:{recipe.branch}"),
        checkpoint=lambda handle: event(f"checkpoint:{handle.branch}"),
        restore=restore,
        continuity=continuity,
        close=lambda handle: event(f"close_{handle.role}:{handle.branch}"),
        memory_released=lambda branch, role, bound: event(
            f"memory_release_{role}:{branch}", 0
        ),
        publish=lambda receipt: event("receipt", receipt),
        capacity_observation=lambda: (128, 96),
        capacity_cleanup=lambda: event("capacity_cleanup"),
    )


def _run(tmp_path, hooks, cancelled=lambda: False):
    return run_wan_curated_host_canary(
        execution=_concrete_execution(tmp_path), database=object(), artifacts=object(),
        cancelled=cancelled, _hooks=hooks,
    )


def test_runs_truthful_fresh_restore_lifecycle_without_overlap(tmp_path) -> None:
    events: list[str] = []
    receipt = _run(tmp_path, _hooks(events))
    assert receipt.independent_run_id == "run-independent-0002"
    def per_branch(branch: str) -> list[str]:
        return [
            f"load_trained:{branch}", f"step:{branch}",
            f"checkpoint:{branch}", f"close_trained:{branch}",
            f"memory_release_trained:{branch}", f"load_restored:{branch}",
            f"restore:{branch}", f"continuity:{branch}", f"close_restored:{branch}",
            f"memory_release_restored:{branch}",
        ]
    assert events == [
        "inventory:high_noise", "inventory:low_noise", "device_preflight",
        *per_branch("high_noise"), *per_branch("low_noise"), "receipt",
    ]


@pytest.mark.parametrize("failure", [
    "inventory:high_noise", "load_trained:high_noise", "step:high_noise",
    "checkpoint:high_noise", "close_trained:high_noise",
    "memory_release_trained:high_noise", "load_restored:high_noise",
    "restore:high_noise", "continuity:high_noise", "close_restored:high_noise",
    "memory_release_restored:high_noise", "inventory:low_noise",
    "load_trained:low_noise", "step:low_noise", "checkpoint:low_noise",
    "close_trained:low_noise", "memory_release_trained:low_noise",
    "load_restored:low_noise", "restore:low_noise", "continuity:low_noise",
    "close_restored:low_noise", "memory_release_restored:low_noise",
])
def test_every_failure_prevents_later_work_and_publication(tmp_path, failure: str) -> None:
    events: list[str] = []
    with pytest.raises((RuntimeError, BaseExceptionGroup)):
        _run(tmp_path, _hooks(events, fail=failure))
    assert "receipt" not in events
    if failure.endswith("high_noise"):
        assert not any(
            event.endswith("low_noise") and event != "inventory:low_noise"
            for event in events
        )


def test_recognized_oom_is_bound_capacity_and_unrelated_error_is_not(tmp_path) -> None:
    events: list[str] = []
    hooks = _hooks(events)
    hooks = _WanCanaryHooks(
        preflight=hooks.preflight,
        inventory=hooks.inventory,
        device_preflight=hooks.device_preflight,
        load=hooks.load,
        step=lambda handle, recipe: (_ for _ in ()).throw(
            RuntimeError("MPS backend out of memory")
        ),
        checkpoint=hooks.checkpoint,
        restore=hooks.restore,
        continuity=hooks.continuity,
        close=hooks.close,
        memory_released=hooks.memory_released,
        publish=hooks.publish,
        capacity_observation=hooks.capacity_observation,
        capacity_cleanup=hooks.capacity_cleanup,
    )
    with pytest.raises(WanCapacityExceeded) as caught:
        _run(tmp_path, hooks)
    assert (caught.value.result.branch, caught.value.result.phase) == ("high_noise", "step")
    assert caught.value.result.required_memory_bytes == 104
    assert "receipt" not in events

    hooks = _hooks(events)
    hooks = _WanCanaryHooks(
        preflight=hooks.preflight,
        inventory=hooks.inventory,
        device_preflight=hooks.device_preflight,
        load=hooks.load,
        step=lambda handle, recipe: (_ for _ in ()).throw(RuntimeError("kernel bug")),
        checkpoint=hooks.checkpoint,
        restore=hooks.restore,
        continuity=hooks.continuity,
        close=hooks.close,
        memory_released=hooks.memory_released,
        publish=hooks.publish,
        capacity_observation=hooks.capacity_observation,
        capacity_cleanup=hooks.capacity_cleanup,
    )
    with pytest.raises(RuntimeError, match="kernel bug"):
        _run(tmp_path, hooks)


def test_stale_inventory_stops_before_any_load(tmp_path) -> None:
    events: list[str] = []
    hooks = _hooks(events)
    hooks = _WanCanaryHooks(
        preflight=hooks.preflight,
        inventory=lambda branch, expected: _inventory("9"),
        device_preflight=hooks.device_preflight,
        load=hooks.load,
        step=hooks.step,
        checkpoint=hooks.checkpoint,
        restore=hooks.restore,
        continuity=hooks.continuity,
        close=hooks.close,
        memory_released=hooks.memory_released,
        publish=hooks.publish,
        capacity_observation=hooks.capacity_observation,
        capacity_cleanup=hooks.capacity_cleanup,
    )
    with pytest.raises(ValueError, match="wan_canary_inventory_stale"):
        _run(tmp_path, hooks)
    assert not events


def test_cancellation_at_all_twenty_four_boundaries_never_publishes(tmp_path) -> None:
    for cancel_at in range(24):
        events: list[str] = []
        calls = 0

        def cancelled(target: int = cancel_at) -> bool:
            nonlocal calls
            calls += 1
            return calls == target + 1

        with pytest.raises(WanCanaryCancelled):
            _run(tmp_path, _hooks(events), cancelled)
        assert "receipt" not in events


def test_cancellation_after_second_inventory_precedes_heavy_device_work(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    inventory_calls: list[str] = []
    imports: list[str] = []
    factories: list[object] = []
    publications: list[object] = []
    cancellation_checks = 0

    monkeypatch.setattr(canary_module, "_selected", lambda request, branch: branch)

    def inventory(branch):
        inventory_calls.append(branch)
        return SimpleNamespace(
            available=True,
            inventory=execution.request.inventories[
                0 if branch == "high_noise" else 1
            ],
            missing_roles=(),
        )

    def cancelled() -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks == 3

    def import_sentinel(name: str) -> object:
        imports.append(name)
        raise AssertionError(f"heavy import after cancellation: {name}")

    class FactorySentinel:
        def __init__(self, request, importer) -> None:
            factories.append((request, importer))

    class Artifacts:
        root = execution.output_root.resolve()

        def stage_bytes(self, payload, media_type):
            publications.append((payload, media_type))
            raise AssertionError("publication after cancellation")

    monkeypatch.setattr(canary_module, "inventory_wan_branch", inventory)
    monkeypatch.setattr(canary_module, "WanMpsPreparedFactory", FactorySentinel)

    with pytest.raises(WanCanaryCancelled):
        run_wan_curated_host_canary(
            execution=execution,
            database=database,
            artifacts=Artifacts(),
            cancelled=cancelled,
            _importer=import_sentinel,
        )

    assert cancellation_checks == 3
    assert inventory_calls == ["high_noise", "low_noise"]
    assert imports == []
    assert factories == []
    assert publications == []


def test_request_rejects_reused_independent_run_identifier() -> None:
    request = _request()
    with pytest.raises(ValueError):
        request.model_copy(update={"independent_run_id": request.primary_run_id})


def test_family_digests_are_derived_and_forgery_has_no_request_surface() -> None:
    request = _request()
    assert "inventory_family_digest" not in type(request).model_fields
    assert "recipe_family_digest" not in type(request).model_fields
    assert wan_inventory_family_digest(request.inventories) != wan_inventory_family_digest(
        tuple(reversed(request.inventories))
    )
    assert wan_recipe_family_digest(request.recipes) != wan_recipe_family_digest(
        tuple(reversed(request.recipes))
    )


def test_missing_inventory_returns_typed_path_free_acquisition_outcome(tmp_path) -> None:
    events: list[str] = []
    hooks = _hooks(events)
    unavailable = WanProductionCanaryUnavailableV2(
        schema_version="wan-mps-production-canary-unavailable.v2",
        branch="high_noise",
        missing_roles=("high_transformer",),
    )
    hooks = _WanCanaryHooks(
        preflight=hooks.preflight,
        inventory=lambda branch, expected: unavailable,
        device_preflight=hooks.device_preflight,
        load=hooks.load,
        step=hooks.step,
        checkpoint=hooks.checkpoint,
        restore=hooks.restore,
        continuity=hooks.continuity,
        close=hooks.close,
        memory_released=hooks.memory_released,
        publish=hooks.publish,
        capacity_observation=hooks.capacity_observation,
        capacity_cleanup=hooks.capacity_cleanup,
    )
    assert _run(tmp_path, hooks) == unavailable
    assert not events


def test_capacity_records_distinct_before_failure_and_after_cleanup(tmp_path) -> None:
    events: list[str] = []
    hooks = _hooks(events)
    failed = False
    post_failure = iter(((150, 1_000), (90, 1_000)))

    def oom(handle, recipe):
        nonlocal failed
        failed = True
        raise RuntimeError("MPS backend out of memory")

    def observe():
        events.append("observe")
        return next(post_failure) if failed else (110, 1_000)

    hooks = _WanCanaryHooks(
        preflight=hooks.preflight,
        inventory=hooks.inventory,
        device_preflight=hooks.device_preflight,
        load=hooks.load,
        step=oom,
        checkpoint=hooks.checkpoint,
        restore=hooks.restore,
        continuity=hooks.continuity,
        close=hooks.close,
        memory_released=hooks.memory_released,
        publish=hooks.publish,
        capacity_observation=observe,
        capacity_cleanup=lambda: events.append("capacity_cleanup"),
    )
    with pytest.raises(WanCapacityExceeded) as caught:
        _run(tmp_path, hooks)
    assert caught.value.result.driver_allocated_before_bytes == 110
    assert caught.value.result.driver_allocated_after_failure_bytes == 150
    assert caught.value.result.driver_allocated_after_cleanup_bytes == 90
    assert events[-2:] == ["capacity_cleanup", "observe"]


@pytest.mark.parametrize(
    ("built", "available", "probe_error", "code"),
    [
        (False, True, False, "wan_canary_mps_unavailable"),
        (True, False, False, "wan_canary_mps_unavailable"),
        (True, True, True, "wan_canary_bf16_unsupported"),
    ],
)
def test_concrete_device_preflight_rejects_mps_or_bf16_after_inventory(
    tmp_path, database, monkeypatch, built, available, probe_error, code
) -> None:
    execution = _concrete_execution(tmp_path)
    torch = _install_preflight_doubles(
        monkeypatch, execution,
        _FakeTorch(built=built, available=available, probe_error=probe_error),
    )
    inventory_calls: list[object] = []
    monkeypatch.setattr(
        canary_module, "inventory_wan_branch",
        lambda branch: (
            inventory_calls.append(branch)
            or SimpleNamespace(
                available=True,
                inventory=execution.request.inventories[
                    0 if branch == "high_noise" else 1
                ],
                missing_roles=(),
            )
        ),
    )
    monkeypatch.setattr(canary_module, "_selected", lambda request, branch: branch)
    artifacts = SimpleNamespace(root=execution.output_root.resolve())
    with pytest.raises(ValueError, match=code):
        run_wan_curated_host_canary(
            execution=execution, database=database, artifacts=artifacts,
            _importer=lambda name: torch,
        )
    assert inventory_calls == ["high_noise", "low_noise"]


def test_concrete_default_heavy_importer_uses_importlib_semantics(
    tmp_path, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    torch = _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    imports: list[str] = []

    monkeypatch.setattr(
        canary_module.importlib,
        "import_module",
        lambda name: imports.append(name) or torch,
    )
    hooks = canary_module._production_hooks(
        execution,
        object(),
        SimpleNamespace(root=execution.output_root.resolve()),
    )
    hooks.device_preflight()

    assert imports == ["torch"]


def test_concrete_device_preflight_accepts_actual_indexed_mps_device(
    tmp_path, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    torch = _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    torch.empty = lambda *_args, **_kwargs: SimpleNamespace(
        device=_IndexedMpsDevice(), dtype="torch.bfloat16"
    )
    hooks = canary_module._production_hooks(
        execution,
        object(),
        SimpleNamespace(root=execution.output_root.resolve()),
        importer=lambda _name: torch,
    )

    hooks.device_preflight()


def test_public_preflight_accepts_actual_indexed_mps_device(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    torch = _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    torch.empty = lambda *_args, **_kwargs: SimpleNamespace(
        device=_IndexedMpsDevice(), dtype="torch.bfloat16"
    )
    monkeypatch.setattr(canary_module, "_selected", lambda request, branch: branch)
    monkeypatch.setattr(
        canary_module,
        "inventory_wan_branch",
        lambda branch: (
            SimpleNamespace(
                available=True,
                inventory=execution.request.inventories[0],
                missing_roles=(),
            )
            if branch == "high_noise"
            else SimpleNamespace(
                available=False, inventory=None, missing_roles=("low_transformer",)
            )
        ),
    )

    result = run_wan_curated_host_canary(
        execution=execution,
        database=database,
        artifacts=SimpleNamespace(root=execution.output_root.resolve()),
        _importer=lambda _name: torch,
    )

    assert isinstance(result, WanProductionCanaryUnavailableV2)
    assert result.branch == "low_noise"


def test_public_preflight_rejects_mps_fallback_before_heavy_imports(
    tmp_path, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    imports: list[str] = []

    def importer(name: str) -> object:
        imports.append(name)
        raise AssertionError(name)

    with pytest.raises(ValueError, match="^wan_mps_fallback_forbidden$"):
        run_wan_curated_host_canary(
            execution=execution,
            database=object(),
            artifacts=SimpleNamespace(root=execution.output_root.resolve()),
            _importer=importer,
        )

    assert imports == []


def test_public_execution_rejects_path_only_training_sources(tmp_path) -> None:
    execution = _concrete_execution(tmp_path)
    object.__setattr__(execution, "training_source_lease", None)

    with pytest.raises(ValueError, match="^wan_canary_execution_authority_invalid$"):
        run_wan_curated_host_canary(
            execution=execution,
            database=object(),
            artifacts=object(),
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda request: object.__setattr__(request, "memory_release_bound_bytes", False),
        lambda request: object.__setattr__(request, "recipes", list(request.recipes)),
    ),
)
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_public_entry_revalidates_mutated_v2_request(mutate, tmp_path) -> None:
    execution = _concrete_execution(tmp_path)
    events: list[str] = []
    mutate(execution.request)

    with pytest.raises(ValueError, match="^wan_canary_execution_authority_invalid$"):
        run_wan_curated_host_canary(
            execution=execution,
            database=object(),
            artifacts=object(),
            _hooks=_hooks(events),
        )

    assert events == []


def test_post_injection_placement_failure_closes_loaded_branch(monkeypatch, tmp_path) -> None:
    execution = _concrete_execution(tmp_path)
    lease_available = True
    loaded: list[object] = []

    class Operations:
        def __init__(self, *, fail_injection: bool) -> None:
            self.fail_injection = fail_injection
            self.close_calls = 0

        def eligible_lora_modules(self) -> tuple[str, ...]:
            return ("transformer.to_q",)

        def inject_lora(self, **_kwargs: object) -> None:
            if self.fail_injection:
                raise ValueError("wan_runtime_placement_invalid")

        def close(self) -> None:
            nonlocal lease_available
            self.close_calls += 1
            lease_available = True

    class Factory:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def prepare_branch(self, *_args: object, **_kwargs: object) -> object:
            nonlocal lease_available
            assert lease_available
            lease_available = False
            operation = Operations(fail_injection=not loaded)
            loaded.append(operation)
            return SimpleNamespace(handle=operation)

    monkeypatch.setattr(canary_module, "_selected", lambda *_args: object())
    monkeypatch.setattr(canary_module, "WanMpsPreparedFactory", Factory)
    hooks = canary_module._production_hooks(
        execution,
        object(),
        SimpleNamespace(root=execution.output_root.resolve()),
    )

    with pytest.raises(ValueError, match="^wan_runtime_placement_invalid$"):
        hooks.load("high_noise", "trained")

    first = loaded[0]
    assert isinstance(first, Operations)
    assert first.close_calls == 1
    second = hooks.load("high_noise", "trained")
    assert isinstance(second, Operations)
    second.close()


def test_public_cancel_callback_reaches_real_step_boundary(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    torch = _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(canary_module, "_selected", lambda request, branch: branch)
    monkeypatch.setattr(
        canary_module,
        "inventory_wan_branch",
        lambda branch: SimpleNamespace(
            available=True,
            inventory=execution.request.inventories[
                0 if branch == "high_noise" else 1
            ],
            missing_roles=(),
        ),
    )
    in_step = False
    callbacks: list[object] = []

    def cancelled() -> bool:
        return in_step

    class Operations:
        def __init__(self, branch: str, callback: object) -> None:
            self.recipe = _recipe(branch)
            self.callback = callback

        def eligible_lora_modules(self) -> tuple[str, ...]:
            return ("transformer.attn.to_q",)

        def inject_lora(self, **_kwargs: object) -> None:
            return None

        def train_step(self, *, step: int, total_steps: int) -> None:
            nonlocal in_step
            assert (step, total_steps) == (1, 1)
            in_step = True
            assert callable(self.callback) and self.callback()
            raise ValueError("wan_branch_cancelled")

        def close(self) -> None:
            return None

    class Factory:
        def __init__(self, request: str, importer: object) -> None:
            self.branch = request
            assert importer("torch") is torch  # type: ignore[operator]

        def prepare_branch(
            self,
            sources: object,
            request: object,
            recipe: WanBranchRecipeV1,
            *,
            cancelled: object,
        ) -> object:
            del sources, request
            callbacks.append(cancelled)
            return SimpleNamespace(handle=Operations(recipe.branch, cancelled))

    monkeypatch.setattr(canary_module, "WanMpsPreparedFactory", Factory)
    with pytest.raises(ValueError, match="^wan_branch_cancelled$"):
        run_wan_curated_host_canary(
            execution=execution,
            database=database,
            artifacts=SimpleNamespace(root=execution.output_root.resolve()),
            cancelled=cancelled,
            _importer=lambda _name: torch,
        )

    assert callbacks == [cancelled]


def test_public_default_executes_concrete_authorities_and_atomic_publication(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    torch = _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    events: list[str] = []
    monkeypatch.setattr(canary_module, "_selected", lambda request, branch: branch)
    monkeypatch.setattr(
        canary_module, "inventory_wan_branch",
        lambda branch: SimpleNamespace(
            available=True,
            inventory=execution.request.inventories[0 if branch == "high_noise" else 1],
            missing_roles=(),
        ),
    )

    @dataclass
    class Manifest:
        branch: str

    class Operations:
        def __init__(self, branch):
            self.recipe = _recipe(branch)
            self.branch = branch
            self.closed = False

        def eligible_lora_modules(self):
            return ("transformer.attn.to_q",)

        def inject_lora(self, **kwargs):
            events.append(f"inject:{self.branch}")

        def train_step(self, *, step, total_steps):
            events.append(f"step:{self.branch}")

        def continuity_probe(self):
            return {
                "probe.output": torch_library.tensor(
                    [1.0], dtype=torch_library.float32
                )
            }

        def checkpoint_state_v2(self, *, parent_checkpoint_id):
            events.append(f"checkpoint:{self.branch}")
            return SimpleNamespace(
                manifest=Manifest(self.branch), weights=self.branch.encode(), optimizer={},
                cpu_rng_state=b"cpu", mps_rng_state=b"mps",
            )

        def restore_checkpoint_v2(self, checkpoint, manifest):
            events.append(f"restore:{self.branch}")

        def close(self):
            assert not self.closed
            self.closed = True
            events.append(f"close:{self.branch}")

    class Factory:
        def __init__(self, request, importer):
            self.branch = request
            assert importer("torch") is torch

        def prepare_branch(self, sources, request, recipe, *, cancelled):
            assert callable(cancelled)
            events.append(f"load:{recipe.branch}")
            return SimpleNamespace(handle=Operations(recipe.branch))

    monkeypatch.setattr(canary_module, "WanMpsPreparedFactory", Factory)

    class Artifacts:
        root = execution.output_root.resolve()

        def stage_bytes(self, payload, media_type):
            events.append("stage_receipt")
            return SimpleNamespace(payload=payload)

        def commit(self, staged, job_id):
            events.append("commit_receipt")

        def discard(self, staged):
            events.append("discard_receipt")

    result = run_wan_curated_host_canary(
        execution=execution, database=database, artifacts=Artifacts(),
        _importer=lambda name: torch,
    )
    assert result.inventory_family_digest == wan_inventory_family_digest(
        execution.request.inventories
    )
    assert result.recipe_family_digest == wan_recipe_family_digest(execution.request.recipes)
    assert events[-2:] == ["stage_receipt", "commit_receipt"]
    assert len(tuple(execution.checkpoint_root.glob("*.checkpoint.json"))) == 2
    assert events.count("load:high_noise") == 2
    assert events.count("load:low_noise") == 2


def test_production_hooks_do_not_retain_closed_tensor_owners(
    tmp_path, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    torch = _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(canary_module, "_selected", lambda request, branch: branch)
    monkeypatch.setattr(
        canary_module,
        "inventory_wan_branch",
        lambda branch: SimpleNamespace(
            available=True,
            inventory=execution.request.inventories[
                0 if branch == "high_noise" else 1
            ],
            missing_roles=(),
        ),
    )
    owner_references: list[weakref.ReferenceType[object]] = []

    class TensorOwner:
        pass

    @dataclass
    class Manifest:
        branch: str

    class Operations:
        def __init__(self, branch: str) -> None:
            self.recipe = _recipe(branch)
            owner = TensorOwner()
            self.tensor_owner = owner
            owner_references.append(weakref.ref(owner))

        def eligible_lora_modules(self) -> tuple[str, ...]:
            return ("transformer.attn.to_q",)

        def inject_lora(self, **_kwargs: object) -> None:
            return None

        def train_step(self, *, step: int, total_steps: int) -> None:
            assert (step, total_steps) == (1, 1)

        def continuity_probe(self) -> dict[str, object]:
            return {
                "probe.output": torch_library.tensor(
                    [1.0], dtype=torch_library.float32
                )
            }

        def checkpoint_state_v2(self, *, parent_checkpoint_id: object) -> object:
            assert parent_checkpoint_id is None
            return SimpleNamespace(
                manifest=Manifest(self.recipe.branch),
                weights=b"weights",
                optimizer={},
                cpu_rng_state=b"cpu",
                mps_rng_state=b"mps",
            )

        def restore_checkpoint_v2(self, checkpoint: object, manifest: object) -> None:
            del checkpoint, manifest

        def close(self) -> None:
            return None

    class Factory:
        def __init__(self, request: str, importer: object) -> None:
            self.branch = request
            assert importer("torch") is torch  # type: ignore[operator]

        def prepare_branch(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return SimpleNamespace(handle=Operations(self.branch))

    monkeypatch.setattr(canary_module, "WanMpsPreparedFactory", Factory)

    class Artifacts:
        root = execution.output_root.resolve()

        def stage_bytes(self, payload: bytes, media_type: str) -> object:
            del payload, media_type
            return object()

        def commit(self, staged: object, job_id: object) -> None:
            del staged, job_id

        def discard(self, staged: object) -> None:
            del staged

    artifacts = Artifacts()
    hooks = canary_module._production_hooks(
        execution, object(), artifacts, importer=lambda _name: torch
    )
    run_wan_curated_host_canary(
        execution=execution,
        database=object(),
        artifacts=artifacts,
        _hooks=hooks,
    )
    gc.collect()

    assert owner_references and all(reference() is None for reference in owner_references)


def test_public_default_preserves_concrete_missing_inventory_without_load(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    _install_preflight_doubles(monkeypatch, execution, _FakeTorch())
    monkeypatch.setattr(canary_module, "_selected", lambda request, branch: branch)
    monkeypatch.setattr(
        canary_module, "inventory_wan_branch",
        lambda branch: (
            SimpleNamespace(
                available=True,
                inventory=execution.request.inventories[0],
                missing_roles=(),
            )
            if branch == "high_noise"
            else SimpleNamespace(
                available=False, inventory=None, missing_roles=("low_transformer",)
            )
        ),
    )
    loads: list[object] = []
    monkeypatch.setattr(
        canary_module, "WanMpsPreparedFactory",
        lambda request: loads.append(request),
    )
    artifacts = SimpleNamespace(root=execution.output_root.resolve())
    imports: list[str] = []

    def import_sentinel(name: str) -> object:
        imports.append(name)
        raise AssertionError(f"heavy import forbidden: {name}")

    result = run_wan_curated_host_canary(
        execution=execution, database=database, artifacts=artifacts,
        _importer=import_sentinel,
    )
    assert isinstance(result, WanProductionCanaryUnavailableV2)
    assert result.branch == "low_noise"
    assert result.missing_roles == ("low_transformer",)
    assert result.model_dump(mode="json") == {
        "schema_version": "wan-mps-production-canary-unavailable.v2",
        "branch": "low_noise",
        "missing_roles": ["low_transformer"],
    }
    assert imports == []
    assert loads == []


@pytest.mark.parametrize(
    ("failure_event", "phase"),
    [
        ("load_trained:high_noise", "load_trained"),
        ("step:high_noise", "step"),
        ("checkpoint:high_noise", "checkpoint"),
        ("load_restored:high_noise", "load_fresh_restore"),
        ("restore:high_noise", "restore"),
        ("continuity:high_noise", "continuity"),
    ],
)
def test_every_allocation_phase_oom_is_truthful_capacity(
    tmp_path, failure_event, phase
) -> None:
    events: list[str] = []
    hooks = _hooks(events)
    failed = False
    post_failure = iter(((30, 1_000), (5, 1_000)))

    def operation(name, value=None):
        nonlocal failed
        events.append(name)
        if name == failure_event:
            failed = True
            raise RuntimeError("MPS backend out of memory")
        return value

    def load(branch, role):
        return operation(f"load_{role}:{branch}", _Handle(branch, role))

    def restore(handle, recipe):
        operation(f"restore:{handle.branch}")

    def continuity(handle, recipe):
        operation(f"continuity:{handle.branch}")
        return WanBranchCanaryResultV1(
            schema_version="wan-mps-branch-canary-result.v1",
            branch=recipe.branch,
            recipe_digest=wan_branch_recipe_digest(recipe),
            output_digest="sha256:" + "7" * 64,
            max_absolute_error=0.0,
            max_relative_error=0.0,
        )

    def observe():
        return next(post_failure) if failed else (20, 1_000)

    hooks = _WanCanaryHooks(
        preflight=hooks.preflight,
        inventory=hooks.inventory,
        device_preflight=hooks.device_preflight,
        load=load,
        step=lambda handle, recipe: operation(f"step:{recipe.branch}"),
        checkpoint=lambda handle: operation(f"checkpoint:{handle.branch}"),
        restore=restore,
        continuity=continuity,
        close=hooks.close,
        memory_released=hooks.memory_released,
        publish=hooks.publish,
        capacity_observation=observe,
        capacity_cleanup=lambda: events.append("capacity_cleanup"),
    )
    with pytest.raises(WanCapacityExceeded) as caught:
        _run(tmp_path, hooks)
    assert caught.value.result.branch == "high_noise"
    assert caught.value.result.phase == phase
    assert caught.value.result.driver_allocated_before_bytes == 20
    assert caught.value.result.driver_allocated_after_failure_bytes == 30
    assert caught.value.result.driver_allocated_after_cleanup_bytes == 5
    assert caught.value.result.requested_tensor_envelope == _request().requested_tensor_envelope
    assert caught.value.result.recipe_digest == wan_branch_recipe_digest(
        _request().recipes[0]
    )
    assert caught.value.result.inventory_digest == wan_inventory_digest(
        _request().inventories[0]
    )
    assert "capacity_cleanup" in events
    assert "receipt" not in events


@pytest.mark.parametrize(
    "failure_event",
    [
        "load_trained:high_noise",
        "step:high_noise",
        "checkpoint:high_noise",
        "load_restored:high_noise",
        "restore:high_noise",
        "continuity:high_noise",
    ],
)
def test_unrelated_errors_in_allocation_phases_remain_errors(
    tmp_path, failure_event
) -> None:
    events: list[str] = []
    hooks = _hooks(events, fail=failure_event)
    with pytest.raises(RuntimeError, match="injected") as caught:
        _run(tmp_path, hooks)
    assert not isinstance(caught.value, WanCapacityExceeded)
    assert "capacity_cleanup" not in events
    assert "receipt" not in events


@pytest.mark.parametrize(
    ("failure", "blocked_event"),
    [
        ("step:low_noise", "checkpoint:low_noise"),
        ("close_trained:low_noise", "memory_release_trained:low_noise"),
        ("memory_release_trained:low_noise", "load_restored:low_noise"),
        ("load_restored:low_noise", "restore:low_noise"),
        ("restore:low_noise", "continuity:low_noise"),
        ("continuity:low_noise", "memory_release_restored:low_noise"),
        ("close_restored:low_noise", "memory_release_restored:low_noise"),
        ("memory_release_restored:low_noise", "receipt"),
    ],
)
def test_every_low_branch_failure_blocks_later_work_and_receipt(
    tmp_path, failure, blocked_event
) -> None:
    events: list[str] = []
    with pytest.raises(RuntimeError, match="injected"):
        _run(tmp_path, _hooks(events, fail=failure))
    assert blocked_event not in events
    assert "receipt" not in events


def _set_fake_source_authority(execution: WanProductionCanaryExecutionV2, path: Path) -> None:
    base_model = next(
        item for item in execution.training_sources.files if item.name == "base_model"
    ).path
    descriptors = tuple(
        WanSourceDescriptor(
            name=name,
            path=lambda source=source: source,
            expected_digest="sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_size=source.stat().st_size,
            expected_device=source.stat().st_dev,
            expected_inode=source.stat().st_ino,
        )
        for name, source in (("base_model", base_model), ("batch_tensors", path))
    )
    lease = WanTrainingSourceResolver(descriptors).acquire()
    execution.training_source_lease.close()
    object.__setattr__(execution, "training_sources", lease.sources)
    object.__setattr__(execution, "training_source_lease", lease)


def test_public_execution_rejects_source_output_overlap_before_hooks(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    source = execution.output_root / "compiled-batch.safetensors"
    source.write_bytes(b"batch")
    source.chmod(0o400)
    _set_fake_source_authority(execution, source)
    monkeypatch.setattr(
        canary_module,
        "_production_hooks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hooks reached after overlapping authority")
        ),
    )

    with pytest.raises(ValueError, match="^wan_canary_root_overlap$"):
        run_wan_curated_host_canary(
            execution=execution,
            database=database,
            artifacts=SimpleNamespace(root=execution.output_root.resolve()),
        )


def test_public_execution_rejects_writable_source_authority_before_hooks(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    source_root = tmp_path / "compiled"
    source_root.mkdir()
    source = source_root / "batch.safetensors"
    source.write_bytes(b"batch")
    _set_fake_source_authority(execution, source)
    monkeypatch.setattr(
        canary_module,
        "_production_hooks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hooks reached with writable source authority")
        ),
    )

    with pytest.raises(ValueError, match="^wan_canary_source_writable$"):
        run_wan_curated_host_canary(
            execution=execution,
            database=database,
            artifacts=SimpleNamespace(root=execution.output_root.resolve()),
        )


def test_public_execution_detects_output_root_replacement_after_retention(
    tmp_path, database, monkeypatch
) -> None:
    execution = _concrete_execution(tmp_path)
    source_root = tmp_path / "compiled"
    source_root.mkdir()
    source = source_root / "batch.safetensors"
    source.write_bytes(b"batch")
    source.chmod(0o400)
    source_root.chmod(0o500)
    _set_fake_source_authority(execution, source)
    detached = tmp_path / "detached-output"

    def hooks(*_args: object, **_kwargs: object) -> _WanCanaryHooks:
        result = _hooks([])

        def replace_root() -> None:
            execution.output_root.rename(detached)
            execution.output_root.mkdir()

        return _WanCanaryHooks(
            preflight=replace_root,
            inventory=result.inventory,
            device_preflight=result.device_preflight,
            load=result.load,
            step=result.step,
            checkpoint=result.checkpoint,
            restore=result.restore,
            continuity=result.continuity,
            close=result.close,
            memory_released=result.memory_released,
            publish=result.publish,
            capacity_observation=result.capacity_observation,
            capacity_cleanup=result.capacity_cleanup,
        )

    monkeypatch.setattr(canary_module, "_production_hooks", hooks)
    try:
        with pytest.raises(ValueError, match="^wan_canary_output_root_changed$"):
            run_wan_curated_host_canary(
                execution=execution,
                    database=database,
                artifacts=SimpleNamespace(root=execution.output_root.resolve()),
            )
    finally:
        if detached.exists():
            for child in execution.output_root.iterdir():
                child.unlink()
            execution.output_root.rmdir()
            detached.rename(execution.output_root)


def test_checkpoint_root_swap_during_publication_removes_detached_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "retained-checkpoints"
    root.mkdir()
    detached = tmp_path / "detached-checkpoints"
    authority = RetainedDirectoryAuthority.capture(
        root, error_code="wan_canary_checkpoint_root_changed"
    )
    checkpoint = _checkpoint()
    real_reverify = RetainedDirectoryAuthority.reverify
    reverify_calls = 0

    def swap_during_final_reverify(self: RetainedDirectoryAuthority) -> None:
        nonlocal reverify_calls
        if self is authority:
            reverify_calls += 1
            if reverify_calls == 2:
                root.rename(detached)
                root.mkdir()
        real_reverify(self)

    monkeypatch.setattr(
        RetainedDirectoryAuthority, "reverify", swap_during_final_reverify
    )
    try:
        with pytest.raises(ValueError, match="^wan_canary_checkpoint_root_changed$"):
            _publish_checkpoint(
                authority.path,
                "high_noise",
                checkpoint,
                root_authority=authority,
            )
        _assert_no_checkpoint_evidence(detached)
        _assert_no_checkpoint_evidence(root)
    finally:
        if root.exists():
            for path in sorted(root.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            root.rmdir()
        if detached.exists():
            for path in sorted(detached.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            detached.rename(root)
        authority.close()


def test_checkpoint_publication_deduplicates_without_replacing_verified_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    authority = RetainedDirectoryAuthority.capture(root)
    try:
        _publish_checkpoint(
            root, "high_noise", _checkpoint(), root_authority=authority
        )
        published = tuple(root.glob("*.checkpoint.json"))
        assert len(published) == 1
        identity = (published[0].stat().st_dev, published[0].stat().st_ino)
        _publish_checkpoint(
            root, "high_noise", _checkpoint(), root_authority=authority
        )
        repeated = tuple(root.glob("*.checkpoint.json"))
        assert len(repeated) == 1
        assert (repeated[0].stat().st_dev, repeated[0].stat().st_ino) == identity
    finally:
        authority.close()


def test_checkpoint_same_process_publishers_do_not_delete_each_others_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    authority = RetainedDirectoryAuthority.capture(root)
    first_staged = threading.Event()
    release_first = threading.Event()
    real_fsync = canary_module.os.fsync
    results: list[str] = []
    errors: list[BaseException] = []

    def coordinate_first_publisher(descriptor: int) -> None:
        real_fsync(descriptor)
        if (
            threading.current_thread().name == "checkpoint-publisher-one"
            and not first_staged.is_set()
        ):
            first_staged.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("first checkpoint publisher was not released")

    def publish() -> None:
        try:
            _publish_checkpoint(
                root, "high_noise", _checkpoint(), root_authority=authority
            )
            results.append("published")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(canary_module.os, "fsync", coordinate_first_publisher)
    first = threading.Thread(target=publish, name="checkpoint-publisher-one")
    second = threading.Thread(target=publish, name="checkpoint-publisher-two")
    try:
        first.start()
        assert first_staged.wait(timeout=5)
        second.start()
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert results == ["published", "published"]
        assert len(tuple(root.glob("*.checkpoint.json"))) == 1
        assert tuple(root.glob(".*.tmp")) == ()
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
        authority.close()


def test_checkpoint_creator_failure_cannot_remove_concurrent_publisher_success(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    context = multiprocessing.get_context("spawn")
    installed = context.Event()
    publisher_started = context.Event()
    publisher_finished = context.Event()
    results = context.Queue()
    creator = context.Process(
        target=_checkpoint_creator_with_failed_final_reverify,
        args=(
            str(root),
            installed,
            publisher_started,
            publisher_finished,
            results,
        ),
    )
    deduplicator = context.Process(
        target=_checkpoint_deduplicator,
        args=(str(root), publisher_started, publisher_finished, results),
    )
    try:
        creator.start()
        assert installed.wait(timeout=15)
        deduplicator.start()
        creator.join(timeout=20)
        deduplicator.join(timeout=20)
        assert not creator.is_alive()
        assert not deduplicator.is_alive()
        outcomes = {
            label: (status, message)
            for label, status, message in (results.get(timeout=5), results.get(timeout=5))
        }
        assert outcomes == {
            "creator": ("error", "wan_canary_checkpoint_root_changed"),
            "deduplicator": ("success", ""),
        }
        published = tuple(root.glob("*.checkpoint.json"))
        assert len(published) == 1
        assert tuple(root.glob(".*.tmp")) == ()
    finally:
        if creator.is_alive():
            creator.terminate()
        if deduplicator.is_alive():
            deduplicator.terminate()
        creator.join(timeout=5)
        deduplicator.join(timeout=5)
        results.close()
        results.join_thread()


def test_checkpoint_prewrite_root_swap_writes_neither_retained_nor_replacement_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    detached = tmp_path / "detached-checkpoints"
    authority = RetainedDirectoryAuthority.capture(
        root, error_code="wan_canary_checkpoint_root_changed"
    )
    real_reverify = RetainedDirectoryAuthority.reverify

    def swap_before_write(self: RetainedDirectoryAuthority) -> None:
        if self is authority:
            root.rename(detached)
            root.mkdir()
        real_reverify(self)

    monkeypatch.setattr(RetainedDirectoryAuthority, "reverify", swap_before_write)
    try:
        with pytest.raises(ValueError, match="^wan_canary_checkpoint_root_changed$"):
            _publish_checkpoint(
                root, "high_noise", _checkpoint(), root_authority=authority
            )
        _assert_no_checkpoint_evidence(detached)
        _assert_no_checkpoint_evidence(root)
    finally:
        if root.exists():
            root.rmdir()
        if detached.exists():
            detached.rename(root)
        authority.close()


def test_checkpoint_write_failure_removes_staging_and_final_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    authority = RetainedDirectoryAuthority.capture(root)
    real_fsync = canary_module.os.fsync
    fsync_calls = 0

    def fail_staging_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("forced checkpoint stage failure")
        real_fsync(descriptor)

    monkeypatch.setattr(canary_module.os, "fsync", fail_staging_fsync)
    try:
        with pytest.raises(OSError, match="^forced checkpoint stage failure$"):
            _publish_checkpoint(
                root, "high_noise", _checkpoint(), root_authority=authority
            )
        _assert_no_checkpoint_evidence(root)
    finally:
        authority.close()


def test_checkpoint_postverify_failure_preserves_preexisting_verified_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    detached = tmp_path / "detached-checkpoints"
    authority = RetainedDirectoryAuthority.capture(
        root, error_code="wan_canary_checkpoint_root_changed"
    )
    _publish_checkpoint(root, "high_noise", _checkpoint(), root_authority=authority)
    published = next(root.glob("*.checkpoint.json"))
    expected = published.read_bytes()
    identity = (published.stat().st_dev, published.stat().st_ino)
    real_reverify = RetainedDirectoryAuthority.reverify
    reverify_calls = 0

    def swap_during_final_reverify(self: RetainedDirectoryAuthority) -> None:
        nonlocal reverify_calls
        if self is authority:
            reverify_calls += 1
            if reverify_calls == 2:
                root.rename(detached)
                root.mkdir()
        real_reverify(self)

    monkeypatch.setattr(
        RetainedDirectoryAuthority, "reverify", swap_during_final_reverify
    )
    try:
        with pytest.raises(ValueError, match="^wan_canary_checkpoint_root_changed$"):
            _publish_checkpoint(
                root, "high_noise", _checkpoint(), root_authority=authority
            )
        retained = next(detached.glob("*.checkpoint.json"))
        assert retained.read_bytes() == expected
        assert (retained.stat().st_dev, retained.stat().st_ino) == identity
        _assert_no_checkpoint_evidence(root)
    finally:
        if root.exists():
            root.rmdir()
        if detached.exists():
            detached.rename(root)
        authority.close()


def test_checkpoint_rollback_failure_is_path_safe_after_created_file_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    detached = tmp_path / "detached-checkpoints"
    authority = RetainedDirectoryAuthority.capture(
        root, error_code="wan_canary_checkpoint_root_changed"
    )
    real_reverify = RetainedDirectoryAuthority.reverify
    real_fsync = canary_module.os.fsync
    reverify_calls = 0
    fsync_calls = 0

    def swap_during_final_reverify(self: RetainedDirectoryAuthority) -> None:
        nonlocal reverify_calls
        if self is authority:
            reverify_calls += 1
            if reverify_calls == 2:
                root.rename(detached)
                root.mkdir()
        real_reverify(self)

    def fail_cleanup_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 3:
            raise OSError("forced checkpoint cleanup fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        RetainedDirectoryAuthority, "reverify", swap_during_final_reverify
    )
    monkeypatch.setattr(canary_module.os, "fsync", fail_cleanup_fsync)
    try:
        with pytest.raises(
            ValueError, match="^wan_canary_checkpoint_rollback_failed$"
        ) as raised:
            _publish_checkpoint(
                root, "high_noise", _checkpoint(), root_authority=authority
            )
        assert str(root) not in str(raised.value)
        assert str(detached) not in str(raised.value)
        assert isinstance(raised.value.__cause__, BaseExceptionGroup)
        primary, *_rollback = raised.value.__cause__.exceptions
        assert str(primary) == "wan_canary_checkpoint_root_changed"
        assert str(root) not in str(primary)
        assert str(detached) not in str(primary)
        _assert_no_checkpoint_evidence(detached)
        _assert_no_checkpoint_evidence(root)
    finally:
        if root.exists():
            root.rmdir()
        if detached.exists():
            detached.rename(root)
        authority.close()


def test_checkpoint_authority_close_during_final_reverify_removes_created_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    authority = RetainedDirectoryAuthority.capture(
        root, error_code="wan_canary_checkpoint_root_changed"
    )
    real_reverify = RetainedDirectoryAuthority.reverify
    reverify_calls = 0

    def close_during_final_reverify(self: RetainedDirectoryAuthority) -> None:
        nonlocal reverify_calls
        if self is authority:
            reverify_calls += 1
            if reverify_calls == 2:
                self.close()
        real_reverify(self)

    monkeypatch.setattr(
        RetainedDirectoryAuthority, "reverify", close_during_final_reverify
    )
    with pytest.raises(ValueError, match="^wan_canary_checkpoint_root_changed$"):
        _publish_checkpoint(root, "high_noise", _checkpoint(), root_authority=authority)
    _assert_no_checkpoint_evidence(root)


def test_receipt_publication_root_swap_returns_no_receipt_and_compensates(
    tmp_path: Path, database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _run(tmp_path, _hooks([]))
    base = tmp_path / "receipt-root-swap"
    base.mkdir()
    execution = _concrete_execution(base)
    store = ArtifactStore(database, execution.output_root)
    detached = base / "detached-output"
    authority = RetainedDirectoryAuthority.capture(
        execution.output_root, error_code="wan_canary_output_root_changed"
    )

    def replace_root_path() -> None:
        execution.output_root.rename(detached)
        execution.output_root.mkdir()

    monkeypatch.setattr(store, "_before_database_commit", replace_root_path)
    hooks = canary_module._production_hooks(
        execution,
        database,
        store,
        importer=lambda _name: _FakeTorch(),
        output_authority=authority,
    )
    try:
        with pytest.raises(ArtifactIntegrityError):
            hooks.publish(receipt)
        assert _artifact_row_count(database) == 0
        assert not any(
            path.is_file()
            for path in detached.rglob("*")
            if ".locks" not in path.parts
        )
        assert not any(path.is_file() for path in execution.output_root.rglob("*"))
    finally:
        if execution.output_root.exists():
            execution.output_root.rmdir()
        if detached.exists():
            detached.rename(execution.output_root)
        authority.close()


def _artifact_row_count(database: Database) -> int:
    with database.connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0])


def _published_content_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in root.rglob("*")
        if path.is_file() and ".staging" not in path.parts and ".locks" not in path.parts
    )


def test_concrete_artifact_stage_failure_leaves_no_partial_publication(
    tmp_path, database, monkeypatch
) -> None:
    receipt = _run(tmp_path, _hooks([]))
    base = tmp_path / "stage-failure"
    base.mkdir()
    execution = _concrete_execution(base)
    store = ArtifactStore(database, execution.output_root)

    monkeypatch.setattr(
        artifacts_module.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(RuntimeError("stage failed")),
    )
    hooks = canary_module._production_hooks(
        execution, database, store, importer=lambda name: _FakeTorch()
    )
    with pytest.raises(RuntimeError, match="stage failed"):
        hooks.publish(receipt)

    assert _artifact_row_count(database) == 0
    assert _published_content_files(store.root) == ()
    assert not store.staging_root.exists() or tuple(store.staging_root.iterdir()) == ()


def test_concrete_artifact_commit_failure_discards_and_leaves_no_publication(
    tmp_path, database, monkeypatch
) -> None:
    receipt = _run(tmp_path, _hooks([]))
    base = tmp_path / "commit-failure"
    base.mkdir()
    execution = _concrete_execution(base)
    store = ArtifactStore(database, execution.output_root)
    discarded: list[object] = []
    real_discard = store.discard

    def discard(staged):
        discarded.append(staged)
        real_discard(staged)

    def fail_registration(connection, artifact_id, job_id):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(store, "discard", discard)
    monkeypatch.setattr(
        ArtifactStore,
        "_associate_with_connection",
        staticmethod(fail_registration),
    )
    hooks = canary_module._production_hooks(
        execution, database, store, importer=lambda name: _FakeTorch()
    )
    with pytest.raises(RuntimeError, match="commit failed"):
        hooks.publish(receipt)

    assert len(discarded) == 1
    assert _artifact_row_count(database) == 0
    assert _published_content_files(store.root) == ()
    assert tuple(store.staging_root.iterdir()) == ()


def test_v2_request_rejects_list_bool_coercion_and_malformed_nested_model_copy() -> None:
    request = _request()
    malformed_recipe = _recipe("high_noise")
    object.__setattr__(malformed_recipe, "alpha", True)

    for update in (
        {"recipes": list(request.recipes)},
        {"inventories": list(request.inventories)},
        {"requested_tensor_envelope": list(request.requested_tensor_envelope)},
        {"dependency_versions": [["torch", "2.8.0"]]},
        {"memory_release_bound_bytes": False},
        {"required_artifact_bytes": 1024.0},
        {"recipes": (malformed_recipe, request.recipes[1])},
    ):
        with pytest.raises(ValidationError):
            request.model_copy(update=update)


def test_v2_capacity_unavailable_and_receipt_reject_container_coercion(tmp_path) -> None:
    request = _request()
    capacity = WanCapacityResultV2(
        schema_version="wan-mps-capacity-result.v2",
        branch="high_noise",
        phase="step",
        device="mps",
        dtype="bfloat16",
        requested_tensor_envelope=request.requested_tensor_envelope,
        driver_allocated_before_bytes=1,
        driver_allocated_after_failure_bytes=2,
        driver_allocated_after_cleanup_bytes=0,
        recommended_maximum_bytes=1000,
        required_memory_bytes=104,
        recipe_digest=wan_branch_recipe_digest(request.recipes[0]),
        inventory_digest=wan_inventory_digest(request.inventories[0]),
        fits=False,
        classification="recognized_mps_allocation_failure",
    )
    unavailable = WanProductionCanaryUnavailableV2(
        schema_version="wan-mps-production-canary-unavailable.v2",
        branch="high_noise",
        missing_roles=("high_transformer",),
    )
    receipt = _run(tmp_path, _hooks([]))
    assert isinstance(receipt, WanProductionCanaryReceiptV2)

    invalid_copies = (
        (capacity, {"requested_tensor_envelope": list(request.requested_tensor_envelope)}),
        (capacity, {"driver_allocated_before_bytes": False}),
        (capacity, {"fits": 0}),
        (unavailable, {"missing_roles": ["high_transformer"]}),
        (receipt, {"branch_results": list(receipt.branch_results)}),
        (receipt, {"memory_release_bytes": list(receipt.memory_release_bytes)}),
    )
    for model, update in invalid_copies:
        with pytest.raises(ValidationError):
            model.model_copy(update=update)


def test_v2_validation_errors_hide_private_input_values() -> None:
    secret = "/private/models/customer-secret"
    with pytest.raises(ValidationError) as raised:
        _request().model_copy(update={"unexpected_private_path": secret})

    assert secret not in str(raised.value)
