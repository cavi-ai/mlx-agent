from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from video_lab.artifacts import ArtifactStore
from video_lab.database import Database
from video_lab.dataset_compiler import BackendOutput, DatasetCompilationService
from video_lab.dataset_receipts import (
    DatasetCompilationReceipt,
    DatasetCompilationRequest,
    InputBinding,
)
from video_lab.trainer_protocol import (
    CalibrationRequest,
    LoRAConfiguration,
    ProbeTrainingRequest,
    TrainingCancelled,
    TrainingMaterialHandles,
    TrainingRequest,
)
from video_lab.trainers.wan_checkpoint import (
    WanParentCheckpointDescriptor,
    WanParentCheckpointFileDescriptor,
)
from video_lab.trainers.wan_diffusers_runtime import WanDiffusersAdapterPaths
from video_lab.trainers.wan_factory import (
    WanPinnedLocalRoot,
    create_wan_lora_mps_trainer,
    root_manifest_digest,
)
from video_lab.trainers.wan_mps import WAN_LORA_MPS_BACKEND_ID, WanSourceDescriptor

pytestmark = pytest.mark.wan_mps_host


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_host_evidence(path: Path, value: object) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    directory_descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("wan_host_canary_evidence_write_failed")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        )
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _VerifiedCompilation:
    receipt: DatasetCompilationReceipt
    runtime_root: Path
    batch_manifest: Path
    batch_tensors: Path
    reverify: Callable[[], None]


class _ReceiptBackend:
    compiler_id = "wan_curated_v1"

    def validate_recipe(self, recipe_bytes: bytes) -> None:
        if not recipe_bytes:
            raise ValueError

    def compile_batch(self, source_bytes: bytes, recipe_bytes: bytes) -> tuple[BackendOutput, ...]:
        if not source_bytes or not recipe_bytes:
            raise ValueError
        return (
            BackendOutput("batch", b"compiled-batch", "application/vnd.safetensors"),
            BackendOutput("manifest", b"{}", "application/json"),
            BackendOutput("provenance", b'{"bound":true}', "application/json"),
        )


def test_host_evidence_publication_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence.json"
    replace = os.replace
    observed: list[tuple[bool, bool]] = []

    def observe(source: Path, target: Path) -> None:
        observed.append((source.is_file(), target.exists()))
        replace(source, target)

    monkeypatch.setattr(os, "replace", observe)
    _write_host_evidence(destination, {"status": "retained"})
    assert observed == [(True, False)]
    assert destination.read_text(encoding="utf-8") == '{"status":"retained"}'
    assert tuple(tmp_path.iterdir()) == (destination,)


def _receipt_fixture(
    tmp_path: Path,
) -> tuple[DatasetCompilationReceipt, WanDiffusersAdapterPaths, str]:
    database = Database(tmp_path / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, tmp_path / "artifacts")
    source = store.commit(store.stage_bytes(b'{"source":1}', "application/json"), None)
    recipe = store.commit(store.stage_bytes(b'{"recipe":1}', "application/json"), None)
    request = DatasetCompilationRequest(
        operation_key="host_receipt",
        compiler_id="wan_curated_v1",
        source=InputBinding(
            role="source_manifest",
            artifact_id=source.id,
            sha256=source.sha256,
            byte_size=source.byte_size,
            media_type=source.media_type,
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id=recipe.id,
            sha256=recipe.sha256,
            byte_size=recipe.byte_size,
            media_type=recipe.media_type,
        ),
        output_roles=("batch", "manifest", "provenance"),
    )
    backend = _ReceiptBackend()
    receipt = DatasetCompilationService(database, store, {backend.compiler_id: backend}).compile(
        request
    )
    outputs = {output.role: output for output in receipt.outputs}
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "model_index.json").write_text("{}", encoding="utf-8")
    revision = hashlib.sha256(root_manifest_digest(model_root).encode()).hexdigest()[:40]
    return receipt, WanDiffusersAdapterPaths(
        wan_snapshot=Path("unused"),
        high_transformer=Path("unused-high"),
        low_transformer=Path("unused-low"),
        batch_manifest=store.resolve(outputs["manifest"].artifact_id).relative_to(tmp_path),
        batch_tensors=store.resolve(outputs["batch"].artifact_id).relative_to(tmp_path),
    ), revision


def test_compiled_receipt_authority_comes_from_durable_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, paths, revision = _receipt_fixture(tmp_path)
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID", receipt.receipt_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_OPERATION_ID", receipt.operation_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_DATABASE", str(tmp_path / "lab.sqlite3"))
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    with ExitStack() as stack:
        verified = _require_compiled_receipt(
            tmp_path / "model", revision, paths, stack, tmp_path / "runtime"
        )
        assert verified.receipt == receipt
        assert (verified.runtime_root / verified.batch_manifest).is_file()
        assert (verified.runtime_root / verified.batch_tensors).is_file()
        assert not (tmp_path / "model" / "compiled-receipt").exists()
        for item in (verified.runtime_root, *verified.runtime_root.rglob("*")):
            assert item.stat(follow_symlinks=False).st_mode & 0o777 == (
                0o700 if item.is_dir() else 0o600
            )
        verified.reverify()

    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_OPERATION_ID", "dco_swapped")
    with ExitStack() as stack, pytest.raises(ValueError, match="wan_canary_receipt_invalid"):
        _require_compiled_receipt(
            tmp_path / "model", revision, paths, stack, tmp_path / "runtime-invalid"
        )


@pytest.mark.parametrize(
    ("attack", "runtime_role"),
    [
        ("same_bytes_new_inode", "batch"),
        ("same_bytes_new_inode", "manifest"),
        ("symlink", "batch"),
        ("symlink", "manifest"),
        ("declared_path_mismatch", "batch"),
    ],
)
def test_compiled_receipt_runtime_inputs_reject_path_attacks(
    attack: str, runtime_role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, paths, revision = _receipt_fixture(tmp_path)
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID", receipt.receipt_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_OPERATION_ID", receipt.operation_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_DATABASE", str(tmp_path / "lab.sqlite3"))
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    if attack == "declared_path_mismatch":
        paths = replace(paths, batch_tensors=Path("attacker-selected.safetensors"))
        with ExitStack() as stack:
            verified = _require_compiled_receipt(
                tmp_path / "model",
                revision,
                paths,
                stack,
                tmp_path / "runtime-mismatch",
            )
            assert verified.batch_tensors != paths.batch_tensors
        return
    with ExitStack() as stack:
        verified = _require_compiled_receipt(
            tmp_path / "model",
            revision,
            paths,
            stack,
            tmp_path / f"runtime-{runtime_role}-{attack}",
        )
        target = verified.runtime_root / (
            verified.batch_tensors if runtime_role == "batch" else verified.batch_manifest
        )
        payload = target.read_bytes()
        detached = target.with_name(target.name + ".detached")
        target.rename(detached)
        if attack == "symlink":
            target.symlink_to(detached)
        else:
            target.write_bytes(payload)
        with pytest.raises(ValueError, match="wan_canary_runtime_input_changed"):
            verified.reverify()


@pytest.mark.parametrize("attack", ["same_content", "corrupt"])
def test_compiled_receipt_rejects_database_inode_replacement(
    attack: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, paths, revision = _receipt_fixture(tmp_path)
    database_path = tmp_path / "lab.sqlite3"
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID", receipt.receipt_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_OPERATION_ID", receipt.operation_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_DATABASE", str(database_path))
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    with ExitStack() as stack:
        verified = _require_compiled_receipt(
            tmp_path / "model",
            revision,
            paths,
            stack,
            tmp_path / f"runtime-db-{attack}",
        )
        payload = database_path.read_bytes()
        database_path.rename(tmp_path / "detached.sqlite3")
        database_path.write_bytes(payload if attack == "same_content" else b"corrupt")
        with pytest.raises(ValueError, match="wan_canary_database_changed"):
            verified.reverify()


@pytest.mark.parametrize("attack", ["same_content", "different_content", "ancestor_swap"])
def test_compiled_receipt_rejects_model_source_replacement_after_snapshot(
    attack: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, paths, revision = _receipt_fixture(tmp_path)
    model = tmp_path / "model"
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID", receipt.receipt_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_COMPILATION_OPERATION_ID", receipt.operation_id)
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_DATABASE", str(tmp_path / "lab.sqlite3"))
    monkeypatch.setenv("VIDEO_LAB_WAN_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    with ExitStack() as stack:
        verified = _require_compiled_receipt(
            model, revision, paths, stack, tmp_path / f"runtime-model-{attack}"
        )
        target = model / "model_index.json"
        payload = target.read_bytes()
        if attack == "ancestor_swap":
            model.rename(tmp_path / "detached-model")
            model.mkdir()
            (model / "model_index.json").write_bytes(payload)
        else:
            target.rename(model / "detached.json")
            target.write_bytes(payload if attack == "same_content" else b'{"changed":true}')
        with pytest.raises(ValueError, match="wan_canary_model_source_changed"):
            verified.reverify()


def _require_compiled_receipt(
    source: Path,
    source_revision: str,
    paths: WanDiffusersAdapterPaths,
    stack: ExitStack,
    runtime_root: Path,
) -> _VerifiedCompilation:
    receipt_value = os.environ.get("VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID")
    operation_value = os.environ.get("VIDEO_LAB_WAN_COMPILATION_OPERATION_ID")
    database_value = os.environ.get("VIDEO_LAB_WAN_ARTIFACT_DATABASE")
    artifact_value = os.environ.get("VIDEO_LAB_WAN_ARTIFACT_ROOT")
    if not all((receipt_value, operation_value, database_value, artifact_value)):
        raise ValueError("wan_canary_receipt_inputs_missing")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("wan_canary_receipt_inputs_invalid")
    database_path, artifact_root = (
        Path(database_value),  # type: ignore[arg-type]
        Path(artifact_value),  # type: ignore[arg-type]
    )
    if any(not path.is_absolute() or path.is_symlink() for path in (database_path, artifact_root)):
        raise ValueError("wan_canary_receipt_inputs_invalid")
    if artifact_root.is_relative_to(source) or source.is_relative_to(artifact_root):
        raise ValueError("wan_canary_receipt_topology_invalid")
    try:
        from video_lab.trainers.wan_factory import (
            WanPinnedLocalRoot,
            retain_pinned_root,
            root_manifest_digest,
            secure_snapshot_pinned_root,
            wan_snapshot_cleanup_scope,
        )

        database_descriptor = os.open(
            database_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        stack.callback(os.close, database_descriptor)
        opened_database = os.fstat(database_descriptor)
        database_identity = (opened_database.st_dev, opened_database.st_ino)
        database = Database(database_path)
        store = ArtifactStore(database, artifact_root)
        service = DatasetCompilationService(database, store, {})
        receipt = service.inspect(receipt_value)  # type: ignore[arg-type]
        if receipt.compiler_id != "wan_curated_v1" or tuple(
            output.role for output in receipt.outputs
        ) != ("batch", "manifest", "provenance"):
            raise ValueError
        if receipt.operation_id != operation_value:
            raise ValueError
        outputs = {output.role: output for output in receipt.outputs}
        handles = []
        with database.connect() as connection:
            for binding in (receipt.source, receipt.recipe, *receipt.outputs):
                handles.append(
                    stack.enter_context(
                        store.open_verified_with_connection(
                            connection,
                            binding.artifact_id,
                            sha256=binding.sha256,
                            byte_size=binding.byte_size,
                            media_type=binding.media_type,
                        )
                    )
                )
        if runtime_root.exists() or runtime_root.is_symlink():
            raise ValueError
        source_manifest = root_manifest_digest(source)
        source_authority = WanPinnedLocalRoot(
            source, source_revision, source_manifest
        )
        retained_source = stack.enter_context(
            retain_pinned_root(source_authority)
        )
        snapshot = secure_snapshot_pinned_root(
            source_authority,
            runtime_root,
            lease=retained_source,
        )
        if snapshot.path != runtime_root:
            raise ValueError
        stack.enter_context(wan_snapshot_cleanup_scope(runtime_root))
        runtime_root.chmod(0o700)
        for item in runtime_root.rglob("*"):
            item.chmod(0o700 if item.is_dir() else 0o600, follow_symlinks=False)
        derived: dict[str, Path] = {}
        runtime_identities: dict[str, tuple[int, int, int, int]] = {}
        for role in ("batch", "manifest"):
            output = outputs[role]
            relative = Path("compiled-receipt") / output.artifact_id
            runtime_path = runtime_root / relative
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.parent.chmod(0o700)
            handle = next(item for item in handles if item.artifact.id == output.artifact_id)
            runtime_path.write_bytes(os.pread(handle.descriptor, output.byte_size, 0))
            runtime_path.chmod(0o600)
            derived[role] = relative
            info = runtime_path.stat(follow_symlinks=False)
            runtime_identities[role] = (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            )

        def reverify() -> None:
            current_database = os.stat(database_path, follow_symlinks=False)
            if (
                (os.fstat(database_descriptor).st_dev, os.fstat(database_descriptor).st_ino)
                != database_identity
                or (current_database.st_dev, current_database.st_ino) != database_identity
            ):
                raise ValueError("wan_canary_database_changed")
            with database.connect() as connection:
                durable = service.repository.receipt(receipt.receipt_id)
                if durable != receipt:
                    raise ValueError("wan_canary_receipt_changed")
                for handle in handles:
                    store.reverify_open_handle(connection, handle)
                if root_manifest_digest(source) != source_manifest:
                    raise ValueError("wan_canary_model_source_changed")
                try:
                    retained_source.reverify()
                except ValueError:
                    raise ValueError("wan_canary_model_source_changed") from None
                for role in ("batch", "manifest"):
                    output = outputs[role]
                    current = runtime_root / derived[role]
                    if (
                        current.is_symlink()
                        or not current.is_file()
                        or (
                            lambda info: (
                                info.st_dev,
                                info.st_ino,
                                info.st_size,
                                info.st_mtime_ns,
                            )
                        )(current.stat(follow_symlinks=False))
                        != runtime_identities[role]
                        or hashlib.sha256(current.read_bytes()).hexdigest() != output.sha256
                    ):
                        raise ValueError("wan_canary_runtime_input_changed")

        reverify()
    except Exception:
        raise ValueError("wan_canary_receipt_invalid") from None
    return _VerifiedCompilation(
        receipt, runtime_root, derived["manifest"], derived["batch"], reverify
    )


def _require_host_canary() -> tuple[
    Path, Path, WanDiffusersAdapterPaths, tuple[str, ...], str, str
]:
    if os.environ.get("VIDEO_LAB_RUN_WAN_MPS_TRAINING_CANARY") != "1":
        pytest.skip("set VIDEO_LAB_RUN_WAN_MPS_TRAINING_CANARY=1 to enable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        pytest.fail("wan_mps_fallback_forbidden")
    if platform.system() != "Darwin":
        pytest.skip("Wan MPS training canary requires Darwin")
    if not all(
        os.environ.get(name)
        for name in (
            "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
            "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
            "VIDEO_LAB_WAN_ARTIFACT_DATABASE",
            "VIDEO_LAB_WAN_ARTIFACT_ROOT",
            "VIDEO_LAB_WAN_MODEL_REVISION",
            "VIDEO_LAB_WAN_CACHE_REVISION",
        )
    ):
        pytest.fail("enabled Wan MPS canary requires compiled receipt authority")
    try:
        import torch
    except ImportError:
        pytest.fail("enabled Wan MPS canary requires torch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.fail("enabled Wan MPS canary requires available MPS")
    source_value = os.environ.get("VIDEO_LAB_WAN_SOURCE_ROOT")
    cache_value = os.environ.get("VIDEO_LAB_WAN_CACHE_ROOT")
    model_revision = os.environ.get("VIDEO_LAB_WAN_MODEL_REVISION")
    cache_revision = os.environ.get("VIDEO_LAB_WAN_CACHE_REVISION")
    paths_value = os.environ.get("VIDEO_LAB_WAN_ADAPTER_PATHS_JSON")
    targets_value = os.environ.get("VIDEO_LAB_WAN_TARGETS_JSON")
    if not all(
        (
            source_value,
            cache_value,
            model_revision,
            cache_revision,
            paths_value,
            targets_value,
        )
    ):
        pytest.fail("enabled Wan MPS canary requires exact source/cache/adapter/target inputs")
    if any(
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in (model_revision, cache_revision)  # type: ignore[arg-type]
    ):
        pytest.fail("enabled Wan MPS canary requires exact lowercase revisions")
    source, cache = Path(source_value), Path(cache_value)  # type: ignore[arg-type]
    try:
        raw_paths = json.loads(paths_value)  # type: ignore[arg-type]
        targets = tuple(json.loads(targets_value))  # type: ignore[arg-type]
        paths = WanDiffusersAdapterPaths(**{key: Path(value) for key, value in raw_paths.items()})
    except Exception:
        pytest.fail("enabled Wan MPS canary adapter inputs are invalid")
    if any(not root.is_absolute() or not root.is_dir() for root in (source, cache)):
        pytest.fail("enabled Wan MPS canary local roots are unavailable")
    return (
        source,
        cache,
        paths,
        targets,
        model_revision,  # type: ignore[return-value]
        cache_revision,  # type: ignore[return-value]
    )


def _descriptors(source: Path, base_path: Path) -> tuple[WanSourceDescriptor, ...]:
    files = sorted(path for path in source.rglob("*") if path.is_file() and not path.is_symlink())
    if base_path not in files:
        pytest.fail("Wan adapter wan_snapshot/model_index.json must be in source inventory")
    ordered = (base_path,) + tuple(path for path in files if path != base_path)
    values = []
    for index, path in enumerate(ordered):
        payload, info = path.read_bytes(), path.stat()
        values.append(
            WanSourceDescriptor(
                name="base_model" if index == 0 else path.relative_to(source).as_posix(),
                path=lambda path=path: path,
                expected_digest=_sha(payload),
                expected_size=len(payload),
                expected_device=info.st_dev,
                expected_inode=info.st_ino,
            )
        )
    return tuple(values)


def _source_digest(descriptors: tuple[WanSourceDescriptor, ...]) -> str:
    payload = {
        "files": [
            {"name": item.name, "sha256": item.expected_digest, "size_bytes": item.expected_size}
            for item in sorted(descriptors, key=lambda item: item.name)
        ]
    }
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _parent(result: object, checkpoint: object) -> WanParentCheckpointDescriptor:
    output_root = result.output_root  # type: ignore[attr-defined]
    files = []
    for item in checkpoint.files:  # type: ignore[attr-defined]
        path = output_root / item.relative_path
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
        checkpoint_digest=_sha(
            json.dumps({item.role: item.digest for item in files}, sort_keys=True).encode()
        ),
        files=tuple(files),
        step=checkpoint.step,
        total_steps=4,
        parent_checkpoint_id=checkpoint.parent_checkpoint_id,
        parent_optimizer_state_digest=None,
    )


def test_wan_mps_training_canary(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    trainers = ExitStack()
    request.addfinalizer(trainers.close)
    source, cache, paths, targets, model_revision, cache_revision = _require_host_canary()
    model_source = source
    verified_compilation = _require_compiled_receipt(
        model_source,
        model_revision,
        paths,
        trainers,
        tmp_path / "compiled-runtime",
    )
    source = verified_compilation.runtime_root
    compiled_receipt = verified_compilation.receipt
    paths = replace(
        paths,
        batch_manifest=verified_compilation.batch_manifest,
        batch_tensors=verified_compilation.batch_tensors,
    )
    base = source / paths.wan_snapshot / "model_index.json"
    descriptors = _descriptors(source, base)
    backend_digest, runtime_digest, material_digest = (
        _sha(b"wan-host-backend"),
        _sha(b"wan-host-runtime"),
        _sha(b"wan-host-material"),
    )
    factory = dict(
        backend_digest=backend_digest,
        material_digest=material_digest,
        orchestration_source_digest=_source_digest(descriptors),
        descriptor_source_digest=_source_digest(descriptors),
        source=WanPinnedLocalRoot(source, model_revision, root_manifest_digest(source)),
        cache=WanPinnedLocalRoot(cache, cache_revision, root_manifest_digest(cache)),
        artifact_root=tmp_path,
        source_descriptors=descriptors,
        adapter_paths=paths,
        runtime_digest=runtime_digest,
    )
    trainer = create_wan_lora_mps_trainer(**factory)
    trainers.callback(trainer.close)  # type: ignore[attr-defined]
    materials = TrainingMaterialHandles(
        recipe_digest="sha256:" + compiled_receipt.recipe.sha256,
        dataset_digest="sha256:" + compiled_receipt.source.sha256,
        base_model_digest=descriptors[0].expected_digest,
        backend_digest=backend_digest,
        source_digest=_source_digest(descriptors),
        runtime_digest=runtime_digest,
        material_digest=material_digest,
    )
    probe = trainer.probe(
        ProbeTrainingRequest(
            request_id="host-probe",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            expected_backend_digest=backend_digest,
            materials=materials,
            scratch_root=tmp_path,
        )
    )
    assert probe.supported
    lora = LoRAConfiguration(rank=2, alpha=2, dropout=0.0)
    calibration = trainer.calibrate(
        CalibrationRequest(
            request_id="host-calibration",
            run_id="host-run",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            materials=materials,
            scratch_root=tmp_path / "calibration",
            steps=0,
            trainable_targets=targets,
            lora=lora,
            seed=7,
            deadline_seconds=3600,
        ),
        lambda _progress: None,
        lambda: False,
    )
    assert calibration.finite_gradients and calibration.frozen_parameters_unchanged

    def request(root: Path, *, start: int = 0, parent: object | None = None) -> TrainingRequest:
        root.mkdir(parents=True, exist_ok=False)
        values = dict(
            request_id="host-training",
            run_id="host-run",
            backend_id=WAN_LORA_MPS_BACKEND_ID,
            materials=materials,
            scratch_root=root,
            total_steps=4,
            checkpoint_steps=(1, 2, 3, 4) if start == 0 else (3, 4),
            trainable_targets=targets,
            lora=lora,
            seed=7,
            deadline_seconds=3600,
            start_step=start,
            telemetry_sequence_start=start + 1,
        )
        if parent is not None:
            values.update(
                parent_checkpoint_id=parent.checkpoint_id,
                parent_checkpoint_digest=parent.checkpoint_digest,
                optimizer_state_digest=next(
                    item.digest for item in parent.files if item.role == "optimizer"
                ),
            )
        return TrainingRequest(**values)

    full = trainer.train(request(tmp_path / "full"), lambda _progress: None, lambda: False)
    boundary = {"step": 0}
    with pytest.raises(TrainingCancelled) as cancelled:
        trainer.train(
            request(tmp_path / "cancelled"),
            lambda progress: boundary.update(step=progress.step),
            lambda: boundary["step"] >= 2,
        )
    assert cancelled.value.result is not None
    parent = _parent(cancelled.value.result, cancelled.value.result.checkpoints[-1])
    resumed_trainer = create_wan_lora_mps_trainer(
        **factory, checkpoint_resolver=lambda _request: parent
    )
    trainers.callback(resumed_trainer.close)  # type: ignore[attr-defined]
    resumed = resumed_trainer.train(
        request(tmp_path / "resumed", start=2, parent=parent),
        lambda _progress: None,
        lambda: False,
    )
    full_files = {item.role: item.digest for item in full.checkpoints[-1].files}
    resumed_files = {item.role: item.digest for item in resumed.checkpoints[-1].files}
    assert resumed_files == full_files
    verified_compilation.reverify()
    _write_host_evidence(
        tmp_path / "wan-host-canary-evidence.json",
        {
            "probe": probe.model_dump(mode="json"),
            "calibration": calibration.model_dump(mode="json"),
            "final_files": full_files,
        },
    )
    trainers.close()
    assert not tuple(tmp_path.glob("wan-adapter-*"))
    assert (tmp_path / "wan-host-canary-evidence.json").is_file()
