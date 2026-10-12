"""Inventory-first tests for the isolated Wan production branch loader."""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import video_lab.trainers.wan_branch_loader as loader
import video_lab.trainers.wan_factory as wan_factory
from video_lab.trainers.wan_branch_loader import (
    WanBranchLoadRequest,
    WanLoadedBranch,
    WanMpsPreparedFactory,
    inventory_wan_branch,
    load_wan_branch,
)
from video_lab.trainers.wan_diffusers_runtime import WanDiffusersTrainingRuntime
from video_lab.trainers.wan_factory import WanPinnedLocalRoot, root_manifest_digest
from video_lab.trainers.wan_mps import WanPreparedTraining
from video_lab.trainers.wan_production_models import (
    WanBranchInventoryResultV1,
    WanBranchRecipeV1,
    WanNumericalToleranceV1,
)


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    files = {
        "scheduler/scheduler_config.json": b"scheduler",
        "text_encoder/config.json": b"encoder",
        "tokenizer/tokenizer.json": b"tokenizer",
        "transformer/config.json": b"transformer-config",
        "vae/config.json": b"vae",
        "weights/high.safetensors": b"high-weights",
        "weights/low.safetensors": b"low-weights",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def _request(tmp_path: Path, *, branch: str = "high_noise") -> WanBranchLoadRequest:
    root = _model_root(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    return WanBranchLoadRequest(
        model_root=WanPinnedLocalRoot(root, "a" * 40, root_manifest_digest(root)),
        snapshot_parent=snapshots,
        branch=branch,  # type: ignore[arg-type]
        scheduler=Path("scheduler"),
        text_encoder=Path("text_encoder"),
        tokenizer=Path("tokenizer"),
        transformer_config=Path("transformer/config.json"),
        high_transformer=Path("weights/high.safetensors"),
        low_transformer=Path("weights/low.safetensors"),
        vae=Path("vae"),
    )


class _Tensor:
    def __init__(self, device: str = "mps", dtype: str = "torch.bfloat16") -> None:
        self.device = device
        self.dtype = dtype


class _Transformer:
    def __init__(
        self,
        *,
        parameters: tuple[_Tensor, ...] = (_Tensor(),),
        buffers: tuple[_Tensor, ...] = (_Tensor(),),
        declared_device: str = "mps",
        device_map: object = None,
    ) -> None:
        self._parameters = parameters
        self._buffers = buffers
        self.device = declared_device
        self.hf_device_map = {"": "mps"} if device_map is None else device_map
        self.to_calls: list[dict[str, object]] = []

    def to(self, **kwargs: object) -> _Transformer:
        self.to_calls.append(kwargs)
        return self

    def parameters(self) -> tuple[_Tensor, ...]:
        return self._parameters

    def buffers(self) -> tuple[_Tensor, ...]:
        return self._buffers


class _Libraries:
    def __init__(self, transformer: _Transformer | None = None) -> None:
        self.transformer = transformer or _Transformer()
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.imports: list[str] = []
        self.synchronized = 0
        self.emptied = 0
        owner = self

        class TransformerModel:
            @classmethod
            def from_single_file(cls, path: object, **kwargs: object) -> _Transformer:
                owner.calls.append((path, kwargs))
                return owner.transformer

        self.torch = SimpleNamespace(
            bfloat16=SimpleNamespace(__str__=lambda _self: "torch.bfloat16"),
            mps=SimpleNamespace(
                synchronize=self._synchronize,
                empty_cache=self._empty_cache,
            ),
        )
        self.diffusers = SimpleNamespace(WanTransformer3DModel=TransformerModel)
        self.safetensors = SimpleNamespace(load_file=self._load_checkpoint)

    def _synchronize(self) -> None:
        self.synchronized += 1

    def _empty_cache(self) -> None:
        self.emptied += 1

    def _load_checkpoint(self, path: object) -> dict[str, bytes]:
        return {"retained_payload": Path(path).read_bytes()}  # type: ignore[arg-type]

    def importer(self, name: str) -> object:
        self.imports.append(name)
        if name == "torch":
            return self.torch
        if name == "diffusers":
            return self.diffusers
        if name == "safetensors.torch":
            return self.safetensors
        raise AssertionError(f"unauthorized import: {name}")


def _recipe(branch: str = "high_noise") -> WanBranchRecipeV1:
    return WanBranchRecipeV1(
        schema_version="wan-mps-branch-recipe.v1",
        branch=branch,
        rank=2,
        alpha=2,
        dropout=0.0,
        target_suffixes=("to_k", "to_out.0", "to_q", "to_v"),
        optimizer_algorithm="adamw",
        optimizer_parameters=(("learning_rate", 1e-4),),
        gradient_clip_norm=1.0,
        seed=7,
        tolerance=WanNumericalToleranceV1(atol=1e-5, rtol=1e-4),
    )


def test_inventory_import_is_ml_dependency_free_and_does_not_mutate_rng(tmp_path: Path) -> None:
    request = _request(tmp_path)
    before = __import__("random").getstate()

    inventory = inventory_wan_branch(request)

    assert type(inventory) is WanBranchInventoryResultV1
    assert inventory.available is True
    assert inventory.inventory is not None
    assert tuple(name for name, _digest in inventory.inventory.component_manifests) == (
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    )
    assert __import__("random").getstate() == before
    model_path = str(request.model_root.path)
    manifest_digest = request.model_root.manifest_digest
    snapshot_parent = str(request.snapshot_parent)
    code = f"""
import pathlib, sys
from video_lab.trainers.wan_branch_loader import WanBranchLoadRequest, inventory_wan_branch
from video_lab.trainers.wan_factory import WanPinnedLocalRoot
r = WanBranchLoadRequest(
    model_root=WanPinnedLocalRoot(
        pathlib.Path({model_path!r}), {'a' * 40!r}, {manifest_digest!r}),
    snapshot_parent=pathlib.Path({snapshot_parent!r}), branch='high_noise',
    scheduler=pathlib.Path('scheduler'), text_encoder=pathlib.Path('text_encoder'),
    tokenizer=pathlib.Path('tokenizer'), transformer_config=pathlib.Path('transformer/config.json'),
    high_transformer=pathlib.Path('weights/high.safetensors'),
    low_transformer=pathlib.Path('weights/low.safetensors'), vae=pathlib.Path('vae'))
assert inventory_wan_branch(r).available
for name in ('torch', 'diffusers', 'transformers', 'peft'):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_components_return_redacted_acquisition_inventory_without_imports(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    (request.model_root.path / "weights/high.safetensors").unlink()
    (request.model_root.path / "tokenizer/tokenizer.json").unlink()
    request = replace(
        request,
        model_root=replace(
            request.model_root,
            manifest_digest=root_manifest_digest(request.model_root.path),
        ),
    )

    inventory = inventory_wan_branch(request)

    assert inventory.available is False
    assert inventory.inventory is None
    assert inventory.missing_roles == ("high_transformer", "tokenizer")
    assert set(inventory.model_dump(mode="json")) == {
        "available",
        "inventory",
        "missing_roles",
    }
    assert str(tmp_path) not in repr(inventory)


@pytest.mark.parametrize(
    ("branch", "expected_payload"),
    [("high_noise", b"high-weights"), ("low_noise", b"low-weights")],
)
def test_loads_only_selected_transformer_from_private_snapshot(
    tmp_path: Path, branch: str, expected_payload: bytes
) -> None:
    request = _request(tmp_path, branch=branch)
    libraries = _Libraries()

    loaded = load_wan_branch(request, importer=libraries.importer)

    assert isinstance(loaded, WanLoadedBranch)
    assert libraries.imports == ["torch", "diffusers", "safetensors.torch"]
    assert len(libraries.calls) == 1
    checkpoint, kwargs = libraries.calls[0]
    assert checkpoint == {"retained_payload": expected_payload}
    config_path = Path(str(kwargs["config"]))
    assert config_path.parent == Path("/dev/fd")
    assert config_path.read_bytes() == b"transformer-config"
    assert kwargs == {
        "config": str(config_path),
        "local_files_only": True,
        "torch_dtype": libraries.torch.bfloat16,
    }
    assert libraries.transformer.to_calls == [
        {"device": "mps", "dtype": libraries.torch.bfloat16}
    ]
    assert "UMT5EncoderModel" not in repr(libraries.diffusers)
    loaded.close()
    assert libraries.synchronized == 1
    assert libraries.emptied == 1
    assert not loaded.snapshot.path.exists()
    assert loaded.transformer is None
    assert loaded._torch is None


@pytest.mark.parametrize(
    "transformer",
    [
        _Transformer(parameters=(_Tensor("cpu"),)),
        _Transformer(buffers=(_Tensor("meta"),)),
        _Transformer(device_map={"block": "disk"}),
        _Transformer(parameters=(_Tensor("mps", "torch.float32"),)),
        _Transformer(parameters=(_Tensor("mps"), _Tensor("cpu"))),
        _Transformer(parameters=(), buffers=()),
        _Transformer(declared_device="cpu"),
    ],
)
def test_invalid_or_incomplete_mps_bfloat16_placement_fails_before_return(
    tmp_path: Path, transformer: _Transformer
) -> None:
    request = _request(tmp_path)
    libraries = _Libraries(transformer)

    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        load_wan_branch(request, importer=libraries.importer)

    assert list(request.snapshot_parent.iterdir()) == []
    assert libraries.synchronized == 1
    assert libraries.emptied == 1


@pytest.mark.parametrize("special_kind", ["symlink", "fifo", "socket"])
def test_inventory_rejects_non_regular_source_entries(
    tmp_path: Path, special_kind: str
) -> None:
    request = _request(tmp_path)
    special = request.model_root.path / "unsafe"
    opened_socket: socket.socket | None = None
    socket_alias: Path | None = None
    if special_kind == "symlink":
        special.symlink_to(request.model_root.path / "vae")
    elif special_kind == "fifo":
        os.mkfifo(special)
    else:
        socket_alias = Path("/private/tmp") / f"wan-socket-{uuid.uuid4().hex}"
        socket_alias.symlink_to(request.model_root.path, target_is_directory=True)
        opened_socket = socket.socket(socket.AF_UNIX)
        opened_socket.bind(str(socket_alias / special.name))
    try:
        with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
            inventory_wan_branch(request)
    finally:
        if opened_socket is not None:
            opened_socket.close()
        if socket_alias is not None:
            socket_alias.unlink()


def test_incomplete_manifest_and_same_content_replacement_are_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        inventory_wan_branch(
            replace(request, model_root=replace(request.model_root, manifest_digest=_sha(b"bad")))
        )

    selected = request.model_root.path / request.high_transformer
    replacement = selected.with_suffix(".replacement")
    replacement.write_bytes(selected.read_bytes())
    os.replace(replacement, selected)
    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        inventory_wan_branch(request)


def test_replacement_between_inventory_and_copy_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    original = loader._inventory_from_retained_files

    def replacing_inventory(*args: object, **kwargs: object) -> object:
        inventory = original(*args, **kwargs)
        selected = request.model_root.path / request.high_transformer
        replacement = selected.with_suffix(".replacement")
        replacement.write_bytes(selected.read_bytes())
        os.replace(replacement, selected)
        return inventory

    monkeypatch.setattr(loader, "_inventory_from_retained_files", replacing_inventory)
    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        load_wan_branch(request, importer=lambda _name: pytest.fail("must fail before import"))
    assert list(request.snapshot_parent.iterdir()) == []


def test_constructor_failure_cleans_snapshot_and_releases_global_load_lease(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    class FailingModel:
        @classmethod
        def from_single_file(cls, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("private constructor detail")

    libraries = _Libraries()
    libraries.diffusers.WanTransformer3DModel = FailingModel
    with pytest.raises(RuntimeError, match="wan_branch_load_failed"):
        load_wan_branch(request, importer=libraries.importer)
    assert list(request.snapshot_parent.iterdir()) == []

    valid = _Libraries()
    loaded = load_wan_branch(request, importer=valid.importer)
    loaded.close()


def test_snapshot_parent_replacement_is_rejected_before_destination_creation(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    retained_parent = tmp_path / "retained-snapshot-parent"
    request.snapshot_parent.rename(retained_parent)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    request.snapshot_parent.symlink_to(attacker, target_is_directory=True)

    with pytest.raises(ValueError, match="wan_branch_snapshot_parent_invalid") as raised:
        load_wan_branch(
            request,
            importer=lambda _name: pytest.fail("must fail before ML import"),
        )

    assert str(tmp_path) not in str(raised.value)
    assert list(attacker.iterdir()) == []
    assert list(retained_parent.iterdir()) == []


def test_snapshot_root_open_failure_removes_created_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    real_open = wan_factory.os.open
    rejected = False

    def fail_first_snapshot_root_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal rejected
        if (
            not rejected
            and isinstance(path, str)
            and path.startswith("wan-branch-")
            and flags & os.O_DIRECTORY
            and dir_fd is not None
        ):
            rejected = True
            raise OSError("private open failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(wan_factory.os, "open", fail_first_snapshot_root_open)

    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        load_wan_branch(
            request,
            importer=lambda _name: pytest.fail("must fail before ML import"),
        )

    assert rejected is True
    assert list(request.snapshot_parent.iterdir()) == []


def test_constructor_reads_retained_files_after_snapshot_path_replacement(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    libraries = _Libraries()
    consumed: list[tuple[bytes, bytes]] = []
    replacement_root: Path | None = None

    class ReplacingModel:
        @classmethod
        def from_single_file(cls, checkpoint: object, **kwargs: object) -> _Transformer:
            nonlocal replacement_root
            snapshot_root = next(
                path
                for path in request.snapshot_parent.iterdir()
                if path.name.startswith("wan-branch-")
            )
            retained_root = request.snapshot_parent / "retained-during-constructor"
            snapshot_root.rename(retained_root)
            replacement_root = snapshot_root
            (replacement_root / "weights").mkdir(parents=True)
            (replacement_root / "transformer").mkdir()
            (replacement_root / request.high_transformer).write_bytes(b"attacker-weights")
            (replacement_root / request.transformer_config).write_bytes(b"attacker-config")
            weight_payload = (
                checkpoint["retained_payload"]
                if isinstance(checkpoint, dict)
                else Path(checkpoint).read_bytes()  # type: ignore[arg-type]
            )
            config_payload = Path(str(kwargs["config"])).read_bytes()
            consumed.append((weight_payload, config_payload))
            return libraries.transformer

    libraries.diffusers.WanTransformer3DModel = ReplacingModel
    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        load_wan_branch(request, importer=libraries.importer)

    assert consumed == [(b"high-weights", b"transformer-config")]
    assert replacement_root is not None
    assert (replacement_root / request.high_transformer).read_bytes() == b"attacker-weights"
    assert (replacement_root / request.transformer_config).read_bytes() == b"attacker-config"


def test_close_cleans_retained_snapshot_inode_without_deleting_replacement_tree(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    loaded = load_wan_branch(request, importer=_Libraries().importer)
    retained_snapshot = request.snapshot_parent / "retained-original-snapshot"
    loaded.snapshot.path.rename(retained_snapshot)
    loaded.snapshot.path.mkdir()
    marker = loaded.snapshot.path / "attacker-marker"
    marker.write_bytes(b"must-survive")

    loaded.close()

    assert marker.read_bytes() == b"must-survive"
    assert not retained_snapshot.exists()


def test_only_one_branch_load_can_be_live_at_once(tmp_path: Path) -> None:
    first_request = _request(tmp_path / "first")
    second_request = _request(tmp_path / "second")
    first = load_wan_branch(first_request, importer=_Libraries().importer)
    with pytest.raises(ValueError, match="wan_branch_concurrent_load"):
        load_wan_branch(second_request, importer=_Libraries().importer)
    first.close()
    second = load_wan_branch(second_request, importer=_Libraries().importer)
    second.close()


def test_close_is_exhaustive_idempotent_and_normalizes_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    libraries = _Libraries()
    loaded = load_wan_branch(request, importer=libraries.importer)
    attempts: list[str] = []

    def fail_synchronize() -> None:
        attempts.append("synchronize")
        raise KeyboardInterrupt("private")

    def fail_empty_cache() -> None:
        attempts.append("empty_cache")
        raise RuntimeError("private")

    original_remove = loader._remove_snapshot_at

    def fail_remove(**kwargs: object) -> None:
        attempts.append("remove_snapshot")
        original_remove(**kwargs)  # type: ignore[arg-type]
        raise OSError("private")

    libraries.torch.mps.synchronize = fail_synchronize
    libraries.torch.mps.empty_cache = fail_empty_cache
    monkeypatch.setattr(loader, "_remove_snapshot_at", fail_remove)
    with pytest.raises(BaseExceptionGroup) as raised:
        loaded.close()

    assert attempts == ["synchronize", "empty_cache", "remove_snapshot"]
    assert loaded.transformer is None
    assert str(tmp_path) not in str(raised.value)
    assert {str(item) for item in raised.value.exceptions} == {
        "wan_branch_mps_synchronize_failed",
        "wan_branch_mps_cache_cleanup_failed",
        "wan_branch_snapshot_cleanup_failed",
    }
    loaded.close()


def test_prepared_factory_and_runtime_delegate_to_single_branch_loader(tmp_path: Path) -> None:
    request = _request(tmp_path)
    libraries = _Libraries()
    factory = WanMpsPreparedFactory(request=request, importer=libraries.importer)
    recipe = _recipe("low_noise")

    prepared = factory.prepare_branch(object(), object(), recipe)

    assert isinstance(prepared, WanPreparedTraining)
    assert isinstance(prepared.handle, WanLoadedBranch)
    assert libraries.calls[0][0] == {"retained_payload": b"low-weights"}
    prepared.handle.close()

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime = WanDiffusersTrainingRuntime(
        root=runtime_root,
        paths=SimpleNamespace(),  # type: ignore[arg-type]
        runtime_digest=_sha(b"runtime"),
        prepared_factory=factory,
        branch_recipe=_recipe("high_noise"),
    )
    delegated = runtime.prepare(Any, Any)  # type: ignore[arg-type]
    assert isinstance(delegated.handle, WanLoadedBranch)
    assert libraries.calls[-1][0] == {"retained_payload": b"high-weights"}
    runtime.release(delegated)
    runtime.close()
