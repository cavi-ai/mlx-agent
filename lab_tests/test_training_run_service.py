import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from lab_tests import test_training_repository as fixtures
from lab_tests import test_wan_trainer as wan_fixtures
from video_lab.artifacts import ArtifactStore
from video_lab.database import _MIGRATIONS, Database, _migration_statements
from video_lab.domain import canonical_json
from video_lab.trainer_protocol import (
    CalibrationRequest,
    LoRAConfiguration,
    TrainerRegistry,
    TrainingRequest,
)
from video_lab.trainers.fake import FakeTrainerBackend
from video_lab.trainers.wan_checkpoint import (
    WanParentCheckpointDescriptor,
    WanParentCheckpointFileDescriptor,
)
from video_lab.trainers.wan_factory import (
    WanPinnedLocalRoot,
    create_wan_lora_mps_trainer,
    root_manifest_digest,
)
from video_lab.trainers.wan_mps import WAN_LORA_MPS_BACKEND_ID, WanPreparedTraining
from video_lab.training_models import (
    TrainingOperationKind,
    TrainingOperationStatus,
    TrainingRunEvent,
    checkpoint_steps,
    training_operation_payload_digest,
)
from video_lab.training_repository import RecipeMaterialBinding, TrainingRepository
from video_lab.training_service import TrainingService, TrainingServiceError


def test_backend_registry_is_owner_scoped_and_rejects_mismatch() -> None:
    registry = TrainerRegistry(reviewer_id="reviewer-a")
    registry.register(FakeTrainerBackend())
    assert registry.resolve("reviewer-a", "fake").backend_id == "fake"
    with pytest.raises(LookupError, match="backend_not_registered"):
        registry.resolve("reviewer-b", "fake")
    with pytest.raises(ValueError, match="backend_id_mismatch"):
        registry.register(FakeTrainerBackend(backend_id="wrong"), expected_id="fake")


def test_training_service_exposes_complete_scientific_run_lifecycle() -> None:
    assert callable(TrainingService.start_training)
    assert callable(TrainingService.resume_training)
    assert callable(TrainingService.request_cancellation)
    assert callable(TrainingService.begin_evaluation)
    assert callable(TrainingService.complete_evaluation)
    assert callable(TrainingService.export_terminal_evidence)


class _FactoryWanRuntime:
    def __init__(self, runtime_digest: str) -> None:
        self.observation = wan_fixtures._observation(runtime_digest)
        self.fail_resume = False
        self.block_resume = False
        self.resume_started = threading.Event()
        self.resume_release = threading.Event()

    def observe_host(self):
        return self.observation

    def prepare(self, _sources, request):
        if isinstance(request, CalibrationRequest):
            handle = wan_fixtures._CalibrationPreparedHandle(request.trainable_targets)
        else:
            handle = wan_fixtures._TrainingPreparedHandle(request)
            if self.fail_resume and request.start_step > 0:
                handle.error_at = request.start_step + 1
            if self.block_resume and request.start_step > 0:
                self.resume_started.set()
                if not self.resume_release.wait(timeout=10):
                    raise RuntimeError("resume race timeout")
        return WanPreparedTraining(handle)

    def release(self, _prepared) -> None:
        return None


class _FailingCloseArtifactStore(ArtifactStore):
    fail_close_ids: set[str]

    def __init__(self, database: Database, root: Path) -> None:
        super().__init__(database, root)
        self.fail_close_ids = set()

    @contextmanager
    def open_verified_with_connection(self, connection, artifact_id, **kwargs):
        with super().open_verified_with_connection(
            connection, artifact_id, **kwargs
        ) as handle:
            yield handle
        if artifact_id in self.fail_close_ids:
            self.fail_close_ids.remove(artifact_id)
            raise OSError("injected parent handle close failure")


def test_public_factory_service_start_cancel_resume_persists_exact_chain(tmp_path: Path) -> None:
    repository: TrainingRepository = fixtures.repository.__wrapped__(tmp_path)
    store = _FailingCloseArtifactStore(
        repository.database, tmp_path / "factory-artifacts"
    )
    base_path, adapter_path, source_path = (
        tmp_path / "factory-base", tmp_path / "factory-adapter", tmp_path / "factory-source"
    )
    base = _artifact(store, base_path, b"factory-base")
    adapter = _artifact(store, adapter_path, b"factory-adapter")
    source = _artifact(store, source_path, b"factory-source")
    frozen = fixtures._frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    runtime_identity = {"python": "3.12", "adapter": "factory-wan-double"}
    runtime_digest = "sha256:" + hashlib.sha256(
        canonical_json(runtime_identity).encode()
    ).hexdigest()
    recipe = fixtures._valid_recipe(frozen.dataset_digest).model_copy(update={
        "backend_id": WAN_LORA_MPS_BACKEND_ID, "backend_config": {},
        "base_model_digest": f"sha256:{base.sha256}",
        "lightning_adapter_digests": {"adapter": f"sha256:{adapter.sha256}"},
        "source_code_digest": f"sha256:{source.sha256}", "runtime_identity": runtime_identity,
        "total_steps": 8, "calibration_steps": 0,
        "trainable_targets": ("transformer.attn.to_q",),
    })
    binding = RecipeMaterialBinding(
        stage_3_campaign_id="character_benchmark_training",
        stage_3_result_id=fixtures.STAGE3_RESULT_DIGEST,
        base_model_artifact_id=base.id,
        lightning_adapter_artifact_ids={"adapter": adapter.id},
        source_snapshot_artifact_id=source.id,
        runtime_digest=runtime_digest,
    )
    created = repository.create_recipe(recipe, binding)
    repository.preregister_recipe(created.recipe_id)
    descriptors = (
        wan_fixtures._descriptor("base_model", base_path),
        wan_fixtures._descriptor("high_noise", adapter_path),
        wan_fixtures._descriptor("low_noise", source_path),
    )
    service_holder: dict[str, TrainingService] = {}

    def resolve_parent(request: TrainingRequest) -> WanParentCheckpointDescriptor:
        service = service_holder["service"]
        checkpoint = next(
            item.checkpoint for item in repository.list_checkpoints(request.run_id)
            if item.checkpoint.checkpoint_id == request.parent_checkpoint_id
        )
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT binding.role,binding.artifact_digest,binding.size_bytes,"
                "artifact.relative_path FROM training_checkpoint_inventory_artifacts binding "
                "JOIN artifacts artifact ON artifact.id=binding.artifact_id "
                "WHERE binding.checkpoint_id=? ORDER BY binding.ordinal",
                (checkpoint.checkpoint_id,),
            ).fetchall()
        files = []
        for row in rows:
            path = store.root / str(row["relative_path"])
            info = path.stat()
            files.append(WanParentCheckpointFileDescriptor(
                role=str(row["role"]), path=path, digest=str(row["artifact_digest"]),
                size_bytes=int(row["size_bytes"]), device=info.st_dev, inode=info.st_ino,
            ))
        return WanParentCheckpointDescriptor(
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_digest=service._checkpoint_evidence_digest(checkpoint.checkpoint_id),
            files=tuple(files), step=checkpoint.step, total_steps=8,
            parent_checkpoint_id=checkpoint.parent_checkpoint_id,
            parent_optimizer_state_digest=None,
        )

    backend_digest = "sha256:" + hashlib.sha256(canonical_json({}).encode()).hexdigest()
    material_digest = "sha256:" + hashlib.sha256(
        canonical_json(binding.model_dump(mode="json")).encode()
    ).hexdigest()
    cache = tmp_path / "factory-cache"
    cache.mkdir()
    factory_runtime = _FactoryWanRuntime(runtime_digest)
    backend = create_wan_lora_mps_trainer(
        backend_digest=backend_digest, material_digest=material_digest,
        orchestration_source_digest=f"sha256:{source.sha256}",
        descriptor_source_digest=wan_fixtures._source_digest(descriptors),
        source=WanPinnedLocalRoot(tmp_path, "a" * 40, root_manifest_digest(tmp_path)),
        cache=WanPinnedLocalRoot(cache, "b" * 40, root_manifest_digest(cache)),
        artifact_root=tmp_path, source_descriptors=descriptors,
        runtime=factory_runtime, checkpoint_resolver=resolve_parent,
    )
    registry = TrainerRegistry(reviewer_id="owner")
    registry.register(backend)
    service = TrainingService(repository, store, object(), reviewer_id="owner",
                              source_root=tmp_path, trainer_registry=registry)
    service_holder["service"] = service
    run_id = "factory-wan-service-run"
    service.probe_backend(created.recipe_id, run_id=run_id, request_id="factory-probe",
                          scratch_root=tmp_path / "probe")
    calibration = service.start_calibration(run_id, request_id="factory-calibration",
                                            scratch_root=tmp_path / "calibration")
    service.complete_calibration(run_id, calibration)
    service.preregister_run(run_id, idempotency_key="factory-preregister")
    requested = False
    def cancel(progress) -> None:
        nonlocal requested
        if progress.step >= 2 and not requested:
            requested = True
            service.request_cancellation(run_id, request_id="factory-cancel", requested_step=0)

    (tmp_path / "start").mkdir()
    with pytest.raises(TrainingServiceError, match="training_cancelled"):
        service.start_training(run_id, request_id="factory-start",
                               scratch_root=tmp_path / "start", progress=cancel)
    cancelled_checkpoints = repository.list_checkpoints(run_id)
    assert cancelled_checkpoints and cancelled_checkpoints[-1].checkpoint.step == 2
    (tmp_path / "resume").mkdir()
    with repository.database.connect() as connection:
        parent_artifact = connection.execute(
            "SELECT artifact_id FROM training_checkpoint_inventory_artifacts "
            "WHERE checkpoint_id=? ORDER BY ordinal LIMIT 1",
            (cancelled_checkpoints[-1].checkpoint.checkpoint_id,),
        ).fetchone()
    store.fail_close_ids.add(str(parent_artifact["artifact_id"]))
    with pytest.raises(TrainingServiceError, match="training_result_invalid"):
        service.resume_training(run_id, request_id="factory-resume",
                                scratch_root=tmp_path / "resume")
    assert repository.get_run(run_id).state == "cancelled"
    factory_runtime.block_resume = True
    (tmp_path / "resume-retry").mkdir()
    (tmp_path / "resume-loser").mkdir()
    with ThreadPoolExecutor(max_workers=1) as executor:
        winner = executor.submit(
            service.resume_training,
            run_id,
            request_id="factory-resume",
            scratch_root=tmp_path / "resume-retry",
        )
        assert factory_runtime.resume_started.wait(timeout=10)
        with pytest.raises(TrainingServiceError, match="operation_in_progress"):
            service.resume_training(
                run_id,
                request_id="factory-resume",
                scratch_root=tmp_path / "resume-loser",
            )
        factory_runtime.resume_release.set()
        receipt = winner.result(timeout=10)
    assert receipt.lifecycle_state == "evaluating"
    assert repository.list_checkpoints(run_id)[-1].checkpoint.step == 8
    with repository.database.connect() as connection:
        transitions = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT previous_state,new_state FROM training_run_events "
                "WHERE run_id=? ORDER BY sequence",
                (run_id,),
            )
        )
        failures = connection.execute(
            "SELECT operation_id,attempt FROM training_resume_attempt_failures"
        ).fetchall()
        operation = connection.execute(
            "SELECT status,attempt FROM training_orchestration_operations "
            "WHERE operation_kind='resume' AND idempotency_key='factory-resume'"
        ).fetchone()
    assert transitions.count(("cancelled", "training")) == 2
    assert transitions.count(("training", "cancelled")) == 2
    assert len(failures) == 1
    assert tuple(operation) == ("completed", 2)


def _artifact(store: ArtifactStore, path: Path, content: bytes):
    path.write_bytes(content)
    return store.commit(store.stage(path, "application/octet-stream"), None)


def _downgrade_calibration_bindings_to_physical_v44(
    repository: TrainingRepository,
) -> None:
    statements = tuple(_migration_statements(_MIGRATIONS[42]))
    lifecycle_statements = tuple(_migration_statements(_MIGRATIONS[37]))

    def statement(prefix: str) -> str:
        return next(item for item in statements if item.lstrip().startswith(prefix))

    with repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER validate_training_event_insert_v47")
        connection.execute("DROP TRIGGER validate_training_operation_update_v47")
        connection.execute("DROP TRIGGER validate_training_resume_attempt_failure_insert_v47")
        connection.execute("DROP TRIGGER guard_training_resume_attempt_failure_update_v47")
        connection.execute("DROP TRIGGER guard_training_resume_attempt_failure_delete_v47")
        connection.execute("DROP TABLE training_resume_attempt_failures")
        connection.execute(
            next(
                item
                for item in lifecycle_statements
                if item.lstrip().startswith("CREATE TRIGGER validate_training_event_insert_v37")
            )
        )
        connection.execute(
            next(
                item
                for item in _migration_statements(_MIGRATIONS[42])
                if item.lstrip().startswith(
                    "CREATE TRIGGER validate_training_operation_update_v42"
                )
            )
        )
        connection.execute("DROP TRIGGER validate_training_calibration_operation_binding_v42")
        connection.execute(
            "DROP TRIGGER guard_training_calibration_operation_binding_update_v42"
        )
        connection.execute(
            "DROP TRIGGER guard_training_calibration_operation_binding_delete_v42"
        )
        connection.execute(
            "ALTER TABLE training_calibration_operation_bindings "
            "RENAME TO training_calibration_operation_bindings_v45"
        )
        connection.execute(statement("CREATE TABLE training_calibration_operation_bindings"))
        connection.execute(
            "INSERT INTO training_calibration_operation_bindings "
            "SELECT * FROM training_calibration_operation_bindings_v45"
        )
        connection.execute("DROP TABLE training_calibration_operation_bindings_v45")
        for prefix in (
            "CREATE TRIGGER validate_training_calibration_operation_binding_v42",
            "CREATE TRIGGER guard_training_calibration_operation_binding_update_v42",
            "CREATE TRIGGER guard_training_calibration_operation_binding_delete_v42",
        ):
            connection.execute(statement(prefix))
        connection.execute("DELETE FROM schema_migrations WHERE version > 44")
        connection.commit()


class _CountingFakeTrainer(FakeTrainerBackend):
    def __init__(self) -> None:
        super().__init__()
        self.train_calls = 0
        self.calibration_calls = 0
        self.last_calibration_request: CalibrationRequest | None = None
        self.last_training_request: TrainingRequest | None = None

    def calibrate(self, *args, **kwargs):
        self.calibration_calls += 1
        self.last_calibration_request = args[0]
        return super().calibrate(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.train_calls += 1
        self.last_training_request = args[0]
        return super().train(*args, **kwargs)


class _ReplacingFakeTrainer(_CountingFakeTrainer):
    replacement_path: Path | None = None

    def train(self, *args, **kwargs):
        if self.replacement_path is not None:
            replacement = self.replacement_path.with_name("replacement")
            replacement.write_bytes(b"replaced-parent-artifact")
            os.replace(replacement, self.replacement_path)
        return super().train(*args, **kwargs)


def _prepared_service(
    tmp_path: Path,
    backend: FakeTrainerBackend,
    *,
    backend_config: dict[str, object] | None = None,
    calibration_steps: int = 1,
):
    repository: TrainingRepository = fixtures.repository.__wrapped__(tmp_path)
    store = ArtifactStore(repository.database, tmp_path / "run-artifacts")
    base = _artifact(store, tmp_path / "base-real", b"base-real")
    adapter = _artifact(store, tmp_path / "adapter-real", b"adapter-real")
    source = _artifact(store, tmp_path / "source-real", b"source-real")
    frozen = fixtures._frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    runtime = {"python": "3.12"}
    recipe = fixtures._valid_recipe(frozen.dataset_digest).model_copy(
        update={
            "backend_id": "fake",
            "backend_config": backend_config or {},
            "base_model_digest": f"sha256:{base.sha256}",
            "lightning_adapter_digests": {"adapter": f"sha256:{adapter.sha256}"},
            "source_code_digest": f"sha256:{source.sha256}",
            "runtime_identity": runtime,
            "total_steps": 8,
            "calibration_steps": calibration_steps,
        }
    )
    binding = RecipeMaterialBinding(
        stage_3_campaign_id="character_benchmark_training",
        stage_3_result_id=fixtures.STAGE3_RESULT_DIGEST,
        base_model_artifact_id=base.id,
        lightning_adapter_artifact_ids={"adapter": adapter.id},
        source_snapshot_artifact_id=source.id,
        runtime_digest="sha256:" + hashlib.sha256(canonical_json(runtime).encode()).hexdigest(),
    )
    created = repository.create_recipe(recipe, binding)
    repository.preregister_recipe(created.recipe_id)
    registry = TrainerRegistry(reviewer_id="owner")
    registry.register(backend)
    service = TrainingService(
        repository,
        store,
        object(),
        reviewer_id="owner",
        source_root=tmp_path,
        trainer_registry=registry,
    )
    run_id = "run-service-replay"
    service.probe_backend(
        created.recipe_id,
        run_id=run_id,
        request_id="probe-service-replay",
        scratch_root=tmp_path / "probe",
    )
    calibration = service.start_calibration(
        run_id,
        request_id="calibration-service-replay",
        scratch_root=tmp_path / "calibration",
    )
    service.complete_calibration(run_id, calibration)
    service.preregister_run(run_id, idempotency_key="service-replay")
    return repository, store, service, run_id


def test_zero_step_recipe_is_bound_exactly_and_replays_without_backend_rerun(
    tmp_path: Path,
) -> None:
    backend = _CountingFakeTrainer()
    repository, _, service, run_id = _prepared_service(
        tmp_path, backend, calibration_steps=0
    )

    assert backend.last_calibration_request is not None
    assert backend.last_calibration_request.steps == 0
    assert backend.calibration_calls == 1
    with repository.database.connect() as connection:
        row = connection.execute(
            "SELECT request_digest,request_payload_json FROM "
            "training_orchestration_operations WHERE run_id=? "
            "AND operation_kind='calibration'",
            (run_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(str(row["request_payload_json"]))
        assert payload["steps"] == 0
        assert str(row["request_digest"]) == training_operation_payload_digest(payload)
        assert str(row["request_digest"]) != training_operation_payload_digest(
            {**payload, "steps": 1}
        )
        binding = connection.execute(
            "SELECT steps FROM training_calibration_operation_bindings WHERE operation_id="
            "(SELECT id FROM training_orchestration_operations WHERE run_id=? "
            "AND operation_kind='calibration')",
            (run_id,),
        ).fetchone()
        assert binding is not None and int(binding["steps"]) == 0

    replay = service.start_calibration(
        run_id,
        request_id="calibration-service-replay",
        scratch_root=tmp_path / "zero-step-replay",
    )

    assert replay.request_id == "calibration-service-replay"
    assert backend.calibration_calls == 1


@pytest.mark.parametrize(
    ("reserved", "identical_recipe_value"),
    [("learning_rate", 0.0001), ("gradient_accumulation_steps", 4)],
)
def test_start_training_rejects_reserved_optimizer_parameter_collision(
    tmp_path: Path, reserved: str, identical_recipe_value: float | int
) -> None:
    backend = _CountingFakeTrainer()
    _, _, service, run_id = _prepared_service(
        tmp_path,
        backend,
        backend_config={"optimizer_parameters": {reserved: identical_recipe_value}},
    )
    with pytest.raises(TrainingServiceError, match="optimizer_parameters_reserved"):
        service.start_training(
            run_id,
            request_id=f"reserved-{reserved}",
            scratch_root=tmp_path / f"reserved-{reserved}",
        )
    assert backend.train_calls == 0


def test_reserved_optimizer_preflight_does_not_mutate_or_strand_start(
    tmp_path: Path,
) -> None:
    backend = _CountingFakeTrainer()
    repository, _, service, run_id = _prepared_service(
        tmp_path,
        backend,
        backend_config={"optimizer_parameters": {"learning_rate": 0.0001}},
    )

    def durable_snapshot() -> tuple[object, ...]:
        with repository.database.connect() as connection:
            return (
                tuple(
                    connection.execute(
                        "SELECT * FROM training_runs WHERE id=?", (run_id,)
                    ).fetchone()
                ),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM training_run_events WHERE run_id=? ORDER BY sequence",
                        (run_id,),
                    )
                ),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM training_orchestration_operations "
                        "WHERE run_id=? ORDER BY created_at,id",
                        (run_id,),
                    )
                ),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT link.* FROM training_operation_evidence_links link "
                        "JOIN training_orchestration_operations operation "
                        "ON operation.id=link.operation_id WHERE operation.run_id=? "
                        "ORDER BY link.operation_id,link.evidence_kind,link.evidence_id",
                        (run_id,),
                    )
                ),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM training_checkpoints WHERE run_id=? ORDER BY step", (run_id,)
                    )
                ),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM training_resume_evidence WHERE run_id=? ORDER BY created_at",
                        (run_id,),
                    )
                ),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM training_telemetry WHERE run_id=? ORDER BY sequence",
                        (run_id,),
                    )
                ),
                tuple(connection.execute("SELECT * FROM artifacts ORDER BY id")),
            )

    before = durable_snapshot()
    for _ in range(2):
        with pytest.raises(TrainingServiceError, match="optimizer_parameters_reserved"):
            service.start_training(
                run_id,
                request_id="reserved-preflight-retry",
                scratch_root=tmp_path / "reserved-preflight-retry",
            )
        assert repository.get_run(run_id).state == "preregistered"
        assert durable_snapshot() == before
    assert backend.train_calls == 0

    corrected_root = tmp_path / "corrected"
    corrected_root.mkdir()
    corrected_backend = _CountingFakeTrainer()
    _, _, corrected_service, corrected_run_id = _prepared_service(
        corrected_root,
        corrected_backend,
        backend_config={"optimizer_parameters": {"weight_decay": 0.01}},
    )
    receipt = corrected_service.start_training(
        corrected_run_id,
        request_id="corrected-preflight-start",
        scratch_root=corrected_root / "scratch",
    )
    assert receipt.lifecycle_state == "evaluating"
    assert corrected_backend.train_calls == 1


def test_calibration_exact_replay_returns_durable_result_without_backend_rerun(
    tmp_path: Path,
) -> None:
    backend = _CountingFakeTrainer()
    repository: TrainingRepository = fixtures.repository.__wrapped__(tmp_path)
    store = ArtifactStore(repository.database, tmp_path / "calibration-replay-artifacts")
    base = _artifact(store, tmp_path / "calibration-base", b"calibration-replay-base")
    adapter = _artifact(
        store, tmp_path / "calibration-adapter", b"calibration-replay-adapter"
    )
    source = _artifact(
        store, tmp_path / "calibration-source", b"calibration-replay-source"
    )
    frozen = fixtures._frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    runtime = {"python": "3.12"}
    recipe = fixtures._valid_recipe(frozen.dataset_digest).model_copy(
        update={
            "backend_id": "fake",
            "backend_config": {},
            "base_model_digest": f"sha256:{base.sha256}",
            "lightning_adapter_digests": {"adapter": f"sha256:{adapter.sha256}"},
            "source_code_digest": f"sha256:{source.sha256}",
            "runtime_identity": runtime,
            "calibration_steps": 1,
        }
    )
    created = repository.create_recipe(
        recipe,
        RecipeMaterialBinding(
            stage_3_campaign_id="character_benchmark_training",
            stage_3_result_id=fixtures.STAGE3_RESULT_DIGEST,
            base_model_artifact_id=base.id,
            lightning_adapter_artifact_ids={"adapter": adapter.id},
            source_snapshot_artifact_id=source.id,
            runtime_digest="sha256:"
            + hashlib.sha256(canonical_json(runtime).encode()).hexdigest(),
        ),
    )
    repository.preregister_recipe(created.recipe_id)
    registry = TrainerRegistry(reviewer_id="owner")
    registry.register(backend)
    service = TrainingService(
        repository,
        store,
        object(),
        reviewer_id="owner",
        source_root=tmp_path,
        trainer_registry=registry,
    )
    service.probe_backend(
        created.recipe_id,
        run_id="calibration-replay-run",
        request_id="calibration-replay-probe",
        scratch_root=tmp_path / "probe",
    )
    first = service.start_calibration(
        "calibration-replay-run",
        request_id="calibration-replay",
        scratch_root=tmp_path / "calibration-first",
    )
    service.complete_calibration("calibration-replay-run", first)
    second = service.start_calibration(
        "calibration-replay-run",
        request_id="calibration-replay",
        scratch_root=tmp_path / "calibration-second",
    )
    assert second == first
    assert backend.calibration_calls == 1


def test_physical_v44_legacy_calibration_replay_is_byte_preserving_and_lora_exact(
    tmp_path: Path,
) -> None:
    backend = _CountingFakeTrainer()
    repository: TrainingRepository = fixtures.repository.__wrapped__(tmp_path)
    _downgrade_calibration_bindings_to_physical_v44(repository)
    store = ArtifactStore(repository.database, tmp_path / "legacy-v44-artifacts")
    base = _artifact(store, tmp_path / "legacy-v44-base", b"legacy-v44-base")
    adapter = _artifact(store, tmp_path / "legacy-v44-adapter", b"legacy-v44-adapter")
    source = _artifact(store, tmp_path / "legacy-v44-source", b"legacy-v44-source")
    frozen = fixtures._frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    runtime = {"python": "3.12"}
    recipe = fixtures._valid_recipe(frozen.dataset_digest).model_copy(
        update={
            "backend_id": "fake",
            "backend_config": {},
            "base_model_digest": f"sha256:{base.sha256}",
            "lightning_adapter_digests": {"adapter": f"sha256:{adapter.sha256}"},
            "source_code_digest": f"sha256:{source.sha256}",
            "runtime_identity": runtime,
            "calibration_steps": 1,
        }
    )
    material_binding = RecipeMaterialBinding(
        stage_3_campaign_id="character_benchmark_training",
        stage_3_result_id=fixtures.STAGE3_RESULT_DIGEST,
        base_model_artifact_id=base.id,
        lightning_adapter_artifact_ids={"adapter": adapter.id},
        source_snapshot_artifact_id=source.id,
        runtime_digest="sha256:"
        + hashlib.sha256(canonical_json(runtime).encode()).hexdigest(),
    )
    created = repository.create_recipe(recipe, material_binding)
    repository.preregister_recipe(created.recipe_id)
    registry = TrainerRegistry(reviewer_id="owner")
    registry.register(backend)
    service = TrainingService(
        repository,
        store,
        object(),
        reviewer_id="owner",
        source_root=tmp_path,
        trainer_registry=registry,
    )
    run_id = "legacy-v44-calibration-run"
    request_id = "legacy-v44-calibration"
    service.probe_backend(
        created.recipe_id,
        run_id=run_id,
        request_id="legacy-v44-probe",
        scratch_root=tmp_path / "probe",
    )
    record = repository.get_recipe(created.recipe_id)
    materials = service._training_materials(record)
    with repository.database.connect() as connection:
        probe = connection.execute(
            "SELECT evidence.capability_digest,operation.id FROM training_probe_evidence evidence "
            "JOIN training_orchestration_operations operation ON operation.run_id=evidence.run_id "
            "AND operation.operation_kind='probe' WHERE evidence.run_id=?",
            (run_id,),
        ).fetchone()
        assert probe is not None
        legacy_payload = {
            "request_id": request_id,
            "run_id": run_id,
            "recipe_digest": materials.recipe_digest,
            "dataset_digest": materials.dataset_digest,
            "backend_digest": materials.backend_digest,
            "runtime_digest": materials.runtime_digest,
            "material_digest": materials.material_digest,
            "source_digest": materials.source_digest,
            "probe_operation_id": str(probe[1]),
            "capability_digest": str(probe[0]),
            "steps": 1,
            "trainable_targets": list(recipe.trainable_targets),
            "seed": recipe.seed,
        }
        connection.create_function(
            "training_operation_json_is_exact", 3, lambda _kind, _phase, _value: 1
        )
        connection.execute("BEGIN IMMEDIATE")
        for suffix, mutation in (
            ("wrong-seed", {"seed": recipe.seed + 1}),
            ("extra-field", {"historical_writer_note": "not-part-of-v44"}),
        ):
            with pytest.raises(fixtures.TrainingConflict, match="identity conflict"):
                repository.begin_operation_with_connection(
                    connection,
                    operation_id=f"legacy-v44-{suffix}",
                    run_id=run_id,
                    operation_kind=TrainingOperationKind.CALIBRATION,
                    idempotency_key=f"legacy-v44-{suffix}",
                    request_payload={**legacy_payload, **mutation},
                )
        repository.begin_operation_with_connection(
            connection,
            operation_id="legacy-v44-calibration-operation",
            run_id=run_id,
            operation_kind=TrainingOperationKind.CALIBRATION,
            idempotency_key=request_id,
            request_payload=legacy_payload,
        )
        claim = repository.claim_operation_with_connection(
            connection,
            operation_kind=TrainingOperationKind.CALIBRATION,
            idempotency_key=request_id,
            claimant_id="legacy-v44-worker",
            lease_seconds=60,
        )
        assert claim.claimed
        connection.commit()
    with pytest.raises(TrainingServiceError, match="operation_in_progress"):
        service.start_calibration(
            run_id, request_id=request_id, scratch_root=tmp_path / "legacy-in-progress"
        )
    assert backend.calibration_calls == 0
    result = backend.calibrate(
        CalibrationRequest(
            request_id=request_id,
            run_id=run_id,
            backend_id="fake",
            materials=materials,
            scratch_root=tmp_path / "legacy-calibration",
            steps=1,
            trainable_targets=recipe.trainable_targets,
            lora=LoRAConfiguration(
                rank=recipe.lora_rank,
                alpha=recipe.lora_alpha,
                dropout=recipe.lora_dropout,
            ),
            seed=recipe.seed,
            deadline_seconds=recipe.wall_clock_limit_seconds,
        ),
        lambda _item: None,
        lambda: False,
    )
    service.complete_calibration(run_id, result)

    def scientific_bytes() -> tuple[tuple[object, ...], ...]:
        with repository.database.connect() as connection:
            rows = []
            for table in (
                "training_orchestration_operations",
                "training_calibration_evidence",
                "training_operation_evidence_links",
                "training_calibration_operation_bindings",
            ):
                rows.extend(tuple(row) for row in connection.execute(f"SELECT * FROM {table}"))
            return tuple(rows)

    def binding_schema() -> tuple[tuple[tuple[object, ...], ...], ...]:
        with repository.database.connect() as connection:
            return (
                tuple(tuple(row) for row in connection.execute(
                    "PRAGMA table_info(training_calibration_operation_bindings)"
                )),
                tuple(tuple(row) for row in connection.execute(
                    "PRAGMA foreign_key_list(training_calibration_operation_bindings)"
                )),
                tuple(tuple(row) for row in connection.execute(
                    "PRAGMA index_list(training_calibration_operation_bindings)"
                )),
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
                        "AND tbl_name='training_calibration_operation_bindings' ORDER BY name"
                    )
                ),
            )

    with repository.database.connect() as connection:
        assert connection.execute(
            "SELECT max(version) FROM schema_migrations"
        ).fetchone()[0] == 44
        table_sql = str(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='training_calibration_operation_bindings'"
            ).fetchone()[0]
        )
        assert "CHECK(steps>0)" in table_sql
    before = scientific_bytes()
    schema_before = binding_schema()
    repository.database.migrate()
    after_first_migration = scientific_bytes()
    schema_after_first_migration = binding_schema()
    Database(repository.database.path).migrate()
    assert scientific_bytes() == after_first_migration
    assert binding_schema() == schema_after_first_migration
    replay_backend = _CountingFakeTrainer()
    replay_registry = TrainerRegistry(reviewer_id="owner")
    replay_registry.register(replay_backend)
    replay_service = TrainingService(
        repository,
        store,
        object(),
        reviewer_id="owner",
        source_root=tmp_path,
        trainer_registry=replay_registry,
    )

    replay = replay_service.start_calibration(
        run_id, request_id=request_id, scratch_root=tmp_path / "replay"
    )

    assert replay == result
    assert replay_backend.calibration_calls == 0
    assert scientific_bytes() == before
    assert binding_schema() == schema_before
    with repository.database.connect() as connection:
        assert connection.execute(
            "SELECT max(version) FROM schema_migrations"
        ).fetchone()[0] == max(_MIGRATIONS)
        migrated_table_sql = str(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='training_calibration_operation_bindings'"
            ).fetchone()[0]
        )
        assert "CHECK(steps>=0)" in migrated_table_sql
        with pytest.raises(fixtures.TrainingConflict, match="digest conflict"):
            repository.begin_operation_with_connection(
                connection,
                operation_id="legacy-v44-calibration-operation",
                run_id=run_id,
                operation_kind=TrainingOperationKind.CALIBRATION,
                idempotency_key=request_id,
                request_payload={
                    **legacy_payload,
                    "lora": {"rank": 32, "alpha": 64, "dropout": 0.1},
                },
            )

    zero_recipe = recipe.model_copy(
        update={"backend_config": {"upgrade_zero": True}, "calibration_steps": 0}
    )
    zero_created = repository.create_recipe(zero_recipe, material_binding)
    repository.preregister_recipe(zero_created.recipe_id)
    zero_run_id = "upgraded-v44-zero-calibration-run"
    replay_service.probe_backend(
        zero_created.recipe_id,
        run_id=zero_run_id,
        request_id="upgraded-v44-zero-probe",
        scratch_root=tmp_path / "zero-probe",
    )
    zero = replay_service.start_calibration(
        zero_run_id,
        request_id="upgraded-v44-zero-calibration",
        scratch_root=tmp_path / "zero-calibration",
    )
    assert replay_backend.last_calibration_request is not None
    assert replay_backend.last_calibration_request.steps == 0
    replay_service.complete_calibration(zero_run_id, zero)
    zero_replay = replay_service.start_calibration(
        zero_run_id,
        request_id="upgraded-v44-zero-calibration",
        scratch_root=tmp_path / "zero-replay",
    )
    assert zero_replay == zero
    assert replay_backend.calibration_calls == 1
    with repository.database.connect() as connection:
        zero_rows = connection.execute(
            "SELECT json_extract(operation.request_payload_json,'$.steps'),binding.steps "
            "FROM training_orchestration_operations operation "
            "JOIN training_calibration_operation_bindings binding "
            "ON binding.operation_id=operation.id WHERE operation.run_id=?",
            (zero_run_id,),
        ).fetchone()
        assert tuple(zero_rows) == (0, 0)


def test_start_exact_replay_does_not_rerun_backend(tmp_path: Path) -> None:
    backend = _CountingFakeTrainer()
    _, _, service, run_id = _prepared_service(tmp_path, backend)
    first = service.start_training(
        run_id, request_id="start-replay", scratch_root=tmp_path / "scratch"
    )
    second = service.start_training(
        run_id, request_id="start-replay", scratch_root=tmp_path / "different-scratch"
    )
    assert second == first
    assert backend.train_calls == 1


def test_service_maps_frozen_recipe_lora_configuration_to_backend_requests(
    tmp_path: Path,
) -> None:
    backend = _CountingFakeTrainer()
    _, _, service, run_id = _prepared_service(tmp_path, backend)

    service.start_training(
        run_id, request_id="lora-mapping", scratch_root=tmp_path / "training"
    )

    assert backend.last_calibration_request is not None
    assert backend.last_training_request is not None
    assert backend.last_calibration_request.lora.model_dump() == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    }
    assert backend.last_training_request.lora == backend.last_calibration_request.lora


def _downgrade_operations_to_physical_v43(repository: TrainingRepository) -> None:
    with repository.database.connect() as connection:
        connection.execute("DROP TRIGGER validate_training_event_insert_v47")
        connection.execute("DROP TRIGGER validate_training_operation_update_v47")
        connection.execute("DROP TRIGGER validate_training_resume_attempt_failure_insert_v47")
        connection.execute("DROP TRIGGER guard_training_resume_attempt_failure_update_v47")
        connection.execute("DROP TRIGGER guard_training_resume_attempt_failure_delete_v47")
        connection.execute("DROP TABLE training_resume_attempt_failures")
        connection.execute(
            next(
                item
                for item in _migration_statements(_MIGRATIONS[37])
                if item.lstrip().startswith("CREATE TRIGGER validate_training_event_insert_v37")
            )
        )
        connection.execute(
            next(
                item
                for item in _migration_statements(_MIGRATIONS[42])
                if item.lstrip().startswith(
                    "CREATE TRIGGER validate_training_operation_update_v42"
                )
            )
        )
        connection.execute("DROP TRIGGER validate_training_operation_durable_result_v44")
        connection.execute("DROP TRIGGER guard_training_operation_durable_result_v44")
        connection.execute(
            "ALTER TABLE training_orchestration_operations DROP COLUMN result_payload_json"
        )
        connection.execute(
            "ALTER TABLE training_orchestration_operations DROP COLUMN result_digest"
        )
        connection.execute("DELETE FROM schema_migrations WHERE version > 43")
        connection.commit()


def test_physical_v43_completed_run_fails_backfill_atomically_when_results_not_derivable(
    tmp_path: Path,
) -> None:
    backend = _CountingFakeTrainer()
    repository, _, service, run_id = _prepared_service(tmp_path, backend)
    service.start_training(
        run_id, request_id="v43-start", scratch_root=tmp_path / "v43-start"
    )
    _downgrade_operations_to_physical_v43(repository)
    with repository.database.connect() as connection:
        before = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT id,operation_kind,status,receipt_digest,receipt_payload_json "
                "FROM training_orchestration_operations ORDER BY id"
            )
        )
    with pytest.raises(sqlite3.IntegrityError, match="typed result is not derivable"):
        repository.database.migrate()
    with repository.database.connect() as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 43
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(training_orchestration_operations)"
            )
        }
        after = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT id,operation_kind,status,receipt_digest,receipt_payload_json "
                "FROM training_orchestration_operations ORDER BY id"
            )
        )
    assert "result_payload_json" not in columns
    assert after == before


def test_cancel_operation_completes_exactly_and_replays_without_backend_rerun(
    tmp_path: Path,
) -> None:
    backend = _CountingFakeTrainer()
    repository, _, service, run_id = _prepared_service(tmp_path, backend)
    requested = False

    def cancel_on_first_step(progress) -> None:
        nonlocal requested
        if not requested:
            requested = True
            service.request_cancellation(
                run_id, request_id="cancel-replay", requested_step=0
            )

    with pytest.raises(TrainingServiceError, match="training_cancelled"):
        service.start_training(
            run_id,
            request_id="start-cancelled",
            scratch_root=tmp_path / "scratch",
            progress=cancel_on_first_step,
        )
    replay = service.request_cancellation(
        run_id, request_id="cancel-replay", requested_step=0
    )
    assert replay.status is TrainingOperationStatus.COMPLETED
    assert replay.operation_kind is TrainingOperationKind.CANCEL
    assert backend.train_calls == 1


def _publish_one_checkpoint_for_resume(
    repository: TrainingRepository,
    store: ArtifactStore,
    service: TrainingService,
    backend: FakeTrainerBackend,
    run_id: str,
    tmp_path: Path,
) -> None:
    run = repository.get_run(run_id)
    record = repository.get_recipe(run.run.recipe_id)
    materials = service._training_materials(record)
    observed = []
    request = TrainingRequest(
        request_id="partial-run",
        run_id=run_id,
        backend_id="fake",
        materials=materials,
        scratch_root=tmp_path / "partial",
        total_steps=record.recipe.total_steps,
        checkpoint_steps=checkpoint_steps(record.recipe.total_steps),
        trainable_targets=record.recipe.trainable_targets,
        lora=LoRAConfiguration(
            rank=record.recipe.lora_rank,
            alpha=record.recipe.lora_alpha,
            dropout=record.recipe.lora_dropout,
        ),
        seed=record.recipe.seed,
        deadline_seconds=record.recipe.wall_clock_limit_seconds,
    )
    generated = backend.train(request, observed.append, lambda: False)
    partial = generated.model_copy(
        update={
            "checkpoints": (generated.checkpoints[0],),
            "telemetry_digest": generated.checkpoints[0].telemetry_digest,
        }
    )
    repository.append_event(
        run_id,
        TrainingRunEvent(
            sequence=4,
            previous_state="preregistered",
            new_state="training",
            payload={"request_digest": "sha256:" + "1" * 64},
        ),
        expected_version=3,
    )
    files = partial.checkpoints[0].files
    staged = tuple(
        store.stage(partial.output_root / item.relative_path, "application/octet-stream")
        for item in files
    )

    def publish(connection, artifacts) -> None:
        for sequence, item in enumerate(observed[: partial.checkpoints[0].step], start=1):
            repository.append_telemetry_with_connection(
                connection,
                run_id,
                fixtures.TrainingTelemetry(
                    sequence=sequence, step=item.step, payload=item.model_dump(mode="json")
                ),
            )
        service._publish_checkpoints(connection, record, partial)
        repository.bind_checkpoint_inventory_artifacts_with_connection(
            connection,
            checkpoint_id=partial.checkpoints[0].checkpoint_id,
            artifact_ids_by_role={
                item.role: artifact.id for item, artifact in zip(files, artifacts, strict=True)
            },
        )

    store.commit_batch(staged, publish)


def test_resume_exact_replay_does_not_rerun_backend(tmp_path: Path) -> None:
    backend = _CountingFakeTrainer()
    repository, store, service, run_id = _prepared_service(tmp_path, backend)
    _publish_one_checkpoint_for_resume(repository, store, service, backend, run_id, tmp_path)
    first = service.resume_training(
        run_id, request_id="resume-replay", scratch_root=tmp_path / "resume"
    )
    second = service.resume_training(
        run_id, request_id="resume-replay", scratch_root=tmp_path / "other-resume"
    )
    assert second == first
    assert backend.train_calls == 2


def test_resume_rejects_parent_artifact_replacement_while_handles_are_open(
    tmp_path: Path,
) -> None:
    backend = _ReplacingFakeTrainer()
    repository, store, service, run_id = _prepared_service(tmp_path, backend)
    _publish_one_checkpoint_for_resume(repository, store, service, backend, run_id, tmp_path)
    with repository.database.connect() as connection:
        row = connection.execute(
            "SELECT artifact.relative_path FROM training_checkpoint_inventory_artifacts binding "
            "JOIN artifacts artifact ON artifact.id=binding.artifact_id "
            "WHERE binding.checkpoint_id='checkpoint-1' ORDER BY binding.ordinal LIMIT 1"
        ).fetchone()
    backend.replacement_path = store.root / str(row[0])
    with pytest.raises(TrainingServiceError, match="training_result_invalid"):
        service.resume_training(
            run_id, request_id="resume-replaced", scratch_root=tmp_path / "resume"
        )


def test_resume_cancellation_reverifies_all_parent_handles_before_publication(
    tmp_path: Path,
) -> None:
    backend = _ReplacingFakeTrainer()
    repository, store, service, run_id = _prepared_service(tmp_path, backend)
    _publish_one_checkpoint_for_resume(repository, store, service, backend, run_id, tmp_path)
    service.request_cancellation(run_id, request_id="resume-cancel", requested_step=2)
    with repository.database.connect() as connection:
        row = connection.execute(
            "SELECT artifact.relative_path FROM training_checkpoint_inventory_artifacts binding "
            "JOIN artifacts artifact ON artifact.id=binding.artifact_id "
            "WHERE binding.checkpoint_id='checkpoint-1' ORDER BY binding.ordinal LIMIT 1"
        ).fetchone()
    backend.replacement_path = store.root / str(row[0])
    with pytest.raises(TrainingServiceError, match="training_result_invalid"):
        service.resume_training(
            run_id, request_id="resume-cancelled-replaced", scratch_root=tmp_path / "resume"
        )
    assert repository.get_run(run_id).state == "failed"
    with repository.database.connect() as connection:
        assert connection.execute(
            "SELECT count(*) FROM training_cancellation_evidence WHERE run_id=?", (run_id,)
        ).fetchone()[0] == 0


def test_training_service_publishes_atomic_checkpoint_chain_and_terminal_export(
    tmp_path: Path,
) -> None:
    repository: TrainingRepository = fixtures.repository.__wrapped__(tmp_path)
    store = ArtifactStore(repository.database, tmp_path / "run-artifacts")
    base = _artifact(store, tmp_path / "base-real", b"base-real")
    adapter = _artifact(store, tmp_path / "adapter-real", b"adapter-real")
    source = _artifact(store, tmp_path / "source-real", b"source-real")
    frozen = fixtures._frozen()
    dataset = repository.create_dataset(frozen.character_id, frozen)
    repository.seal_dataset(dataset.dataset_id, frozen)
    runtime = {"python": "3.12"}
    recipe = fixtures._valid_recipe(frozen.dataset_digest).model_copy(
        update={
            "backend_id": "fake",
            "backend_config": {},
            "base_model_digest": f"sha256:{base.sha256}",
            "lightning_adapter_digests": {"adapter": f"sha256:{adapter.sha256}"},
            "source_code_digest": f"sha256:{source.sha256}",
            "runtime_identity": runtime,
            "total_steps": 8,
            "calibration_steps": 1,
        }
    )
    binding = RecipeMaterialBinding(
        stage_3_campaign_id="character_benchmark_training",
        stage_3_result_id=fixtures.STAGE3_RESULT_DIGEST,
        base_model_artifact_id=base.id,
        lightning_adapter_artifact_ids={"adapter": adapter.id},
        source_snapshot_artifact_id=source.id,
        runtime_digest="sha256:" + hashlib.sha256(canonical_json(runtime).encode()).hexdigest(),
    )
    created = repository.create_recipe(recipe, binding)
    repository.preregister_recipe(created.recipe_id)
    registry = TrainerRegistry(reviewer_id="owner")
    registry.register(FakeTrainerBackend())
    service = TrainingService(
        repository,
        store,
        object(),
        reviewer_id="owner",
        source_root=tmp_path,
        trainer_registry=registry,
    )
    service.probe_backend(
        created.recipe_id,
        run_id="run-service-e2e",
        request_id="probe-service-e2e",
        scratch_root=tmp_path / "probe",
    )
    calibration = service.start_calibration(
        "run-service-e2e",
        request_id="calibration-service-e2e",
        scratch_root=tmp_path / "calibration",
    )
    service.complete_calibration("run-service-e2e", calibration)
    service.preregister_run("run-service-e2e", idempotency_key="service-e2e")
    result = service.start_training(
        "run-service-e2e",
        request_id="training-service-e2e",
        scratch_root=tmp_path / "scratch",
    )
    assert len(result.checkpoint_ids) == 4
    assert repository.get_run("run-service-e2e").state == "evaluating"
    assert len(repository.list_checkpoints("run-service-e2e")) == 4
    with repository.database.connect() as connection:
        assert connection.execute(
            "SELECT count(*) FROM training_checkpoint_inventory_artifacts binding "
            "JOIN training_checkpoints checkpoint ON checkpoint.id=binding.checkpoint_id "
            "WHERE checkpoint.run_id=?",
            ("run-service-e2e",),
        ).fetchone()[0] == 28
    with pytest.raises(TrainingServiceError, match="terminal_evidence_incomplete"):
        service.export_terminal_evidence("run-service-e2e")
    service.complete_evaluation("run-service-e2e", qualified=True, evidence={"posterior": 0.95})
    first = service.export_terminal_evidence("run-service-e2e")
    assert first == service.export_terminal_evidence("run-service-e2e")
    assert str(tmp_path).encode() not in first


def test_terminal_export_includes_every_request_referenced_material(tmp_path: Path) -> None:
    backend = _CountingFakeTrainer()
    repository, _, service, run_id = _prepared_service(tmp_path, backend)
    service.start_training(
        run_id, request_id="material-export", scratch_root=tmp_path / "training"
    )
    service.complete_evaluation(run_id, qualified=True, evidence={"posterior": 0.9})
    with repository.database.connect() as connection:
        row = connection.execute(
            "SELECT base_model_artifact_id,source_snapshot_artifact_id,"
            "lightning_adapter_artifact_ids_json FROM training_recipe_materials material "
            "JOIN training_runs run ON run.recipe_storage_id=material.recipe_storage_id "
            "WHERE run.id=?",
            (run_id,),
        ).fetchone()
    expected = {
        str(row["base_model_artifact_id"]),
        str(row["source_snapshot_artifact_id"]),
        *json.loads(str(row["lightning_adapter_artifact_ids_json"])).values(),
    }
    payload = json.loads(service.export_terminal_evidence(run_id))
    exported = {item["artifact_id"] for item in payload["artifacts"]}
    assert expected <= exported


def test_terminal_export_rejects_replaced_request_material(tmp_path: Path) -> None:
    backend = _CountingFakeTrainer()
    repository, store, service, run_id = _prepared_service(tmp_path, backend)
    service.start_training(
        run_id, request_id="material-replacement", scratch_root=tmp_path / "training"
    )
    service.complete_evaluation(run_id, qualified=True, evidence={"posterior": 0.9})
    with repository.database.connect() as connection:
        row = connection.execute(
            "SELECT artifact.relative_path FROM training_recipe_materials material "
            "JOIN training_runs run ON run.recipe_storage_id=material.recipe_storage_id "
            "JOIN artifacts artifact ON artifact.id=material.base_model_artifact_id "
            "WHERE run.id=?",
            (run_id,),
        ).fetchone()
    path = store.root / str(row["relative_path"])
    replacement = path.with_name("replacement-material")
    replacement.write_bytes(b"tampered-material")
    os.replace(replacement, path)
    with pytest.raises(TrainingServiceError):
        service.export_terminal_evidence(run_id)
