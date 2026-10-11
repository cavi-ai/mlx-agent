from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from video_lab.domain import canonical_json
from video_lab.training_models import (
    CheckpointPlan,
    CurationDecision,
    DatasetSplit,
    ImportedTrainingSample,
    OperationTelemetryEntry,
    StartOperationReceipt,
    TrainingAnnotation,
    TrainingCheckpoint,
    TrainingCheckpointArtifact,
    TrainingCheckpointEvidence,
    TrainingCurationConfirmation,
    TrainingCurationSuggestions,
    TrainingDatasetDraft,
    TrainingFailure,
    TrainingMedia,
    TrainingRecipe,
    TrainingRun,
    TrainingRunEvent,
    TrainingSample,
    TrainingSourceTier,
    TrainingTelemetry,
    checkpoint_steps,
    imported_training_sample_id,
    training_confirmation_id,
    training_dataset_id,
    training_recipe_id,
)


def test_start_receipt_preserves_distinct_per_row_telemetry_digests() -> None:
    receipt = StartOperationReceipt(
        run_id="run-1",
        lifecycle_version=4,
        lifecycle_state="evaluating",
        events=(),
        telemetry=(
            OperationTelemetryEntry(evidence_id="1", digest=_digest("t1"), sequence=1, step=1),
            OperationTelemetryEntry(evidence_id="2", digest=_digest("t2"), sequence=2, step=2),
        ),
        checkpoints=(),
        artifacts=(),
        inventory=(),
    )
    assert tuple(entry.digest for entry in receipt.telemetry) == (_digest("t1"), _digest("t2"))


def test_dataset_ledger_identity_is_order_independent_and_preserved() -> None:
    imports = ("training_import_" + "a" * 64, "training_import_" + "b" * 64)
    confirmations = ("training_confirmation_" + "c" * 64,)
    first = TrainingDatasetDraft(
        character_id="character_1",
        character_pack_digest=_digest("pack"),
        canonical_outfit_id="jacket-v1",
        reserved_token="<mirza>",
        samples=(),
        imported_sample_ids=imports,
        curation_confirmation_ids=confirmations,
    )
    reordered = first.model_copy(update={"imported_sample_ids": tuple(reversed(imports))})
    assert training_dataset_id(first) == training_dataset_id(reordered)
    frozen = first.model_dump(mode="python")
    assert tuple(frozen["imported_sample_ids"]) == imports
    assert tuple(frozen["curation_confirmation_ids"]) == confirmations


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("imported_sample_ids", ("bad",)),
        ("curation_confirmation_ids", ("training_confirmation_" + "A" * 64,)),
        ("imported_sample_ids", ("training_import_" + "a" * 64,) * 2),
    ),
)
def test_dataset_ledgers_reject_malformed_or_duplicate_ids(
    field: str, value: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError):
        TrainingDatasetDraft(
            character_id="character_1",
            character_pack_digest=_digest("pack"),
            canonical_outfit_id="jacket-v1",
            reserved_token="<mirza>",
            samples=(),
            **{field: value},
        )


def _imported(**updates: object) -> ImportedTrainingSample:
    media = TrainingMedia(
        sha256=_digest("media"), media_type="image/png", byte_size=12, width=4, height=3
    )
    values = dict(
        import_id="pending",
        reviewer_id="owner",
        character_id="character_1",
        lineage_id="lineage_1",
        artifact_id="artifact_1",
        artifact_digest=media.sha256,
        artifact_byte_size=media.byte_size,
        media=media,
        source_lineage="camera",
        suggestions=TrainingCurationSuggestions(caption="machine text"),
        imported_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    values.update(updates)
    provisional = ImportedTrainingSample.model_construct(**values)
    values["import_id"] = imported_training_sample_id(provisional)
    return ImportedTrainingSample(**values)


def _confirmation(import_id: str, **updates: object) -> TrainingCurationConfirmation:
    values = dict(
        confirmation_id="pending",
        import_id=import_id,
        reviewer_id="owner",
        decision=CurationDecision.ACCEPT,
        confirmed_at=datetime(2026, 1, 2, tzinfo=UTC),
        version=1,
        sample_id="sample-1",
        source_tier=TrainingSourceTier.GOLD_STILL,
        source_lineage="camera",
        caption="human text",
        split=DatasetSplit.TRAIN,
        annotation=_sample("annotation").annotation,
    )
    values.update(updates)
    provisional = TrainingCurationConfirmation.model_construct(**values)
    values["confirmation_id"] = training_confirmation_id(provisional)
    return TrainingCurationConfirmation(**values)


def test_machine_suggestions_never_become_human_confirmation() -> None:
    imported = _imported()
    assert imported.suggestions is not None and imported.suggestions.caption == "machine text"
    with pytest.raises(ValidationError, match="requires every"):
        TrainingCurationConfirmation(
            confirmation_id="bad",
            import_id=imported.import_id,
            reviewer_id="owner",
            decision=CurationDecision.ACCEPT,
            confirmed_at=datetime.now(UTC),
            version=1,
        )
    confirmed = _confirmation(imported.import_id)
    assert confirmed.caption == "human text"
    assert confirmed.accepted_sample(imported).sample_digest == imported.artifact_digest
    foreign_reviewer = ImportedTrainingSample.model_construct(
        **{**imported.__dict__, "reviewer_id": "another-reviewer"}
    )
    with pytest.raises(ValueError, match="does not accept"):
        confirmed.accepted_sample(foreign_reviewer)


def test_rejection_has_no_curated_fields_and_content_ids_bind_mutation() -> None:
    imported = _imported()
    changed = ImportedTrainingSample.model_construct(
        **{**imported.__dict__, "source_lineage": "other"}
    )
    assert imported_training_sample_id(changed) != imported.import_id
    values = dict(
        confirmation_id="pending",
        import_id=imported.import_id,
        reviewer_id="owner",
        decision=CurationDecision.REJECT,
        confirmed_at=datetime.now(UTC),
        version=1,
    )
    provisional = TrainingCurationConfirmation.model_construct(**values)
    values["confirmation_id"] = training_confirmation_id(provisional)
    rejected = TrainingCurationConfirmation(**values)
    assert rejected.decision is CurationDecision.REJECT
    with pytest.raises(ValidationError, match="must not contain"):
        TrainingCurationConfirmation(**{**values, "caption": "not allowed"})


def test_run_evidence_contracts_are_frozen_and_trainer_neutral() -> None:
    run = TrainingRun(
        run_id="run-1", recipe_id="recipe-1", dataset_digest=_digest("dataset"), total_steps=100
    )
    event = TrainingRunEvent(
        sequence=1, previous_state="draft", new_state="probed", payload={"worker": "local"}
    )
    telemetry = TrainingTelemetry(sequence=1, step=1, payload={"loss": 1.0})
    checkpoint_manifest = {
        "schema_version": "character-training-checkpoint.v1",
        "run_id": run.run_id,
        "reviewer_id": "reviewer",
        "recipe_id": run.recipe_id,
        "dataset_digest": run.dataset_digest,
        "character_id": "character",
        "step": 25,
        "fraction": 0.25,
        "parent_checkpoint_id": None,
        "artifact_digest": _digest("artifact"),
        "safetensors_size_bytes": 1,
        "backend_digest": _digest("backend"),
        "runtime_digest": _digest("runtime"),
        "material_digest": _digest("material"),
        "source_digest": _digest("source"),
        "optimizer_state_digest": _digest("optimizer"),
        "trainer_state_digest": _digest("state"),
        "telemetry_digest": _digest("telemetry"),
        "telemetry_sequence_start": 1,
        "telemetry_sequence_end": 1,
        "trainable_targets_digest": _digest("targets"),
        "frozen_parameters_digest": _digest("frozen"),
        "finite_gradient_proof_digest": _digest("gradients"),
        "cancellation_boundary": False,
        "inventory": [],
    }
    checkpoint = TrainingCheckpoint(
        checkpoint_id="cp-25",
        run_id=run.run_id,
        step=25,
        fraction=0.25,
        parent_checkpoint_id=None,
        artifact_digest=_digest("artifact"),
        manifest_digest="sha256:"
        + hashlib.sha256(canonical_json(checkpoint_manifest).encode()).hexdigest(),
        manifest=checkpoint_manifest,
    )
    failure = TrainingFailure(
        failure_id="failure-1", run_id=run.run_id, code="boom", message="safe"
    )
    for value in (run, event, telemetry, checkpoint, failure):
        with pytest.raises(ValidationError, match="frozen"):
            value.__setattr__(next(iter(type(value).model_fields)), "changed")


def test_scientific_run_lifecycle_is_explicit() -> None:
    event = TrainingRunEvent(
        sequence=1, previous_state="draft", new_state="probed", payload={"evidence_id": "p1"}
    )
    assert event.new_state == "probed"
    with pytest.raises(ValidationError):
        TrainingRunEvent(sequence=1, previous_state="draft", new_state="training", payload={})


def test_orchestration_operation_contract_is_frozen_typed_and_digest_bound() -> None:
    from video_lab.training_models import (
        TrainingOperationKind,
        TrainingOperationStatus,
        TrainingOrchestrationOperation,
        training_operation_payload_digest,
    )

    payload = {"run_id": "run-1", "request_id": "request-1"}
    operation = TrainingOrchestrationOperation(
        operation_id="operation-1",
        run_id="run-1",
        recipe_id="recipe-1",
        operation_kind=TrainingOperationKind.START,
        idempotency_key="key-1",
        request_digest=training_operation_payload_digest(payload),
        request_payload=payload,
        lifecycle_version=3,
        lifecycle_state="preregistered",
        status=TrainingOperationStatus.PENDING,
        attempt=0,
    )
    assert operation.request_payload == payload
    with pytest.raises(ValidationError, match="digest"):
        TrainingOrchestrationOperation.model_validate(
            {**operation.model_dump(mode="python"), "request_digest": "sha256:" + "0" * 64}
        )
    with pytest.raises(ValidationError, match="frozen"):
        operation.status = TrainingOperationStatus.RUNNING


@pytest.mark.parametrize(
    ("kind", "legal_states"),
    [
        ("probe", {"draft"}),
        ("calibration", {"probed"}),
        ("start", {"preregistered"}),
        ("resume", {"training", "cancelled"}),
        ("cancel", {"training"}),
    ],
)
def test_orchestration_operation_model_rejects_every_illegal_creation_state(
    kind: str, legal_states: set[str]
) -> None:
    from video_lab.training_models import (
        TrainingOperationKind,
        TrainingOperationStatus,
        TrainingOrchestrationOperation,
        training_operation_payload_digest,
    )

    payload = {"run_id": "run-1", "request_id": "request-1"}
    states = (
        "draft",
        "probed",
        "calibrated",
        "preregistered",
        "training",
        "evaluating",
        "qualified",
        "rejected",
        "failed",
        "cancelled",
    )
    for state in states:
        values = {
            "operation_id": "operation-1",
            "run_id": "run-1",
            "recipe_id": "recipe-1",
            "operation_kind": TrainingOperationKind(kind),
            "idempotency_key": "key-1",
            "request_digest": training_operation_payload_digest(payload),
            "request_payload": payload,
            "lifecycle_version": 0,
            "lifecycle_state": state,
            "status": TrainingOperationStatus.PENDING,
            "attempt": 0,
        }
        if state in legal_states:
            assert TrainingOrchestrationOperation.model_validate(values).lifecycle_state == state
        else:
            with pytest.raises(ValidationError, match="creation lifecycle"):
                TrainingOrchestrationOperation.model_validate(values)


def test_operation_specific_requests_reject_extra_fields_and_resume_inventory_replacement() -> None:
    from video_lab.training_models import (
        ProbeOperationRequest,
        ResumeInventoryArtifact,
        ResumeOperationRequest,
        StartOperationRequest,
        training_operation_payload_digest,
    )

    identity = {
        "request_id": "request-1",
        "run_id": "run-1",
        "recipe_digest": _digest("recipe"),
        "dataset_digest": _digest("dataset"),
        "backend_digest": _digest("backend"),
        "runtime_digest": _digest("runtime"),
        "material_digest": _digest("material"),
        "source_digest": _digest("source"),
    }
    assert ProbeOperationRequest(**identity).run_id == "run-1"
    with pytest.raises(ValidationError, match="extra"):
        ProbeOperationRequest(**identity, unbound=True)
    start = StartOperationRequest(
        **identity,
        preregistration_id="preregistered-1",
        preregistration_request_digest=_digest("preregistered"),
        checkpoint_steps=(25, 50, 75, 100),
        checkpoint_fractions=(0.25, 0.5, 0.75, 1.0),
        material_artifact_ids={
            "base_model": "base",
            "lightning:adapter": "lightning",
            "source_snapshot": "source",
        },
    )
    assert start.checkpoint_fractions == (0.25, 0.5, 0.75, 1.0)
    with pytest.raises(ValidationError, match="checkpoint_fractions"):
        StartOperationRequest.model_validate(
            start.model_dump(mode="python", exclude={"checkpoint_fractions"})
        )
    roles = (
        "weights",
        "optimizer",
        "trainer_state",
        "telemetry",
        "trainable_targets",
        "frozen_parameters",
        "finite_gradient_proof",
    )
    inventory = tuple(
        ResumeInventoryArtifact(
            role=role, artifact_id=f"artifact-{role}", digest=_digest(role), size_bytes=10
        )
        for role in roles
    )
    resume = ResumeOperationRequest(
        **identity,
        parent_checkpoint_id="checkpoint-1",
        parent_checkpoint_digest=_digest("checkpoint"),
        parent_inventory=inventory,
    )
    replacement = resume.model_copy(
        update={
            "parent_inventory": (
                inventory[0].model_copy(update={"artifact_id": "replacement"}),
                *inventory[1:],
            )
        }
    )
    assert training_operation_payload_digest(
        resume.model_dump(mode="json")
    ) != training_operation_payload_digest(replacement.model_dump(mode="json"))
    with pytest.raises(ValidationError, match="every exact role"):
        ResumeOperationRequest(
            **identity,
            parent_checkpoint_id="checkpoint-1",
            parent_checkpoint_digest=_digest("checkpoint"),
            parent_inventory=inventory[:-1] + (inventory[0],),
        )


def test_calibration_operation_digest_binds_exact_lora_configuration() -> None:
    from video_lab.trainer_protocol import LoRAConfiguration
    from video_lab.training_models import (
        CalibrationOperationRequest,
        training_operation_payload_digest,
    )

    request = CalibrationOperationRequest(
        request_id="request-1",
        run_id="run-1",
        recipe_digest=_digest("recipe"),
        dataset_digest=_digest("dataset"),
        backend_digest=_digest("backend"),
        runtime_digest=_digest("runtime"),
        material_digest=_digest("material"),
        source_digest=_digest("source"),
        probe_operation_id="probe-1",
        capability_digest=_digest("capability"),
        steps=0,
        trainable_targets=("transformer.attn.to_q",),
        lora=LoRAConfiguration(rank=16, alpha=32, dropout=0.05),
        seed=7,
    )
    changed = request.model_copy(
        update={"lora": LoRAConfiguration(rank=32, alpha=64, dropout=0.1)}
    )
    changed_steps = request.model_copy(update={"steps": 1})

    assert request.model_dump(mode="json")["lora"] == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    }
    assert training_operation_payload_digest(
        request.model_dump(mode="json")
    ) != training_operation_payload_digest(changed.model_dump(mode="json"))
    assert training_operation_payload_digest(
        request.model_dump(mode="json")
    ) != training_operation_payload_digest(changed_steps.model_dump(mode="json"))
    with pytest.raises(ValidationError, match="steps"):
        CalibrationOperationRequest.model_validate(
            {**request.model_dump(mode="json"), "steps": -1}
        )


def test_cancel_receipt_requires_exact_boundary_checkpoint() -> None:
    from video_lab.training_models import CancelOperationReceipt

    payload = {
        "run_id": "run-1",
        "lifecycle_version": 5,
        "lifecycle_state": "cancelled",
        "cancellation": {"evidence_id": "cancel-1", "digest": _digest("cancel")},
        "terminal_event": {"evidence_id": "5", "digest": _digest("event"), "sequence": 5},
        "boundary_checkpoint": {
            "evidence_id": "checkpoint-1",
            "digest": _digest("checkpoint"),
            "step": 100,
        },
        "cancellation_request_id": "request-1",
        "request_digest": _digest("request"),
        "requested_step": 75,
        "boundary_step": 100,
    }
    assert CancelOperationReceipt.model_validate(payload).boundary_checkpoint.step == 100
    with pytest.raises(ValidationError, match="boundary checkpoint"):
        CancelOperationReceipt.model_validate(
            {**payload, "boundary_checkpoint": {**payload["boundary_checkpoint"], "step": 99}}
        )
    with pytest.raises(ValidationError, match="boundary_checkpoint"):
        CancelOperationReceipt.model_validate(
            {key: value for key, value in payload.items() if key != "boundary_checkpoint"}
        )


@pytest.mark.parametrize(
    ("previous_state", "new_state"),
    [
        (previous_state, new_state)
        for previous_state in (
            "draft",
            "probed",
            "calibrated",
            "preregistered",
            "training",
            "evaluating",
            "qualified",
            "rejected",
            "failed",
            "cancelled",
        )
        for new_state in (
            "probed",
            "calibrated",
            "preregistered",
            "training",
            "evaluating",
            "qualified",
            "rejected",
            "failed",
            "cancelled",
        )
    ],
)
def test_training_run_event_exhaustively_accepts_only_scientific_lifecycle_edges(
    previous_state: str, new_state: str
) -> None:
    legal = {
        ("draft", "probed"),
        ("probed", "calibrated"),
        ("calibrated", "preregistered"),
        ("preregistered", "training"),
        ("cancelled", "training"),
        ("training", "evaluating"),
        ("evaluating", "qualified"),
        ("evaluating", "rejected"),
        *(
            (source, "failed")
            for source in (
                "draft",
                "probed",
                "calibrated",
                "preregistered",
                "training",
                "evaluating",
            )
        ),
        ("training", "cancelled"),
    }
    payload = {
        "sequence": 1,
        "previous_state": previous_state,
        "new_state": new_state,
        "payload": {},
    }
    if (previous_state, new_state) in legal:
        event = TrainingRunEvent.model_validate(payload)
        assert (event.previous_state, event.new_state) == (previous_state, new_state)
    else:
        with pytest.raises(ValidationError, match="lifecycle"):
            TrainingRunEvent.model_validate(payload)


def test_checkpoint_evidence_requires_complete_unique_inventory_and_telemetry_range() -> None:
    digests = {
        name: _digest(name)
        for name in ("weights", "optimizer", "state", "telemetry", "targets", "frozen", "gradients")
    }
    inventory = (
        TrainingCheckpointArtifact(
            relative_path="weights.bin", digest=digests["weights"], size_bytes=1, role="weights"
        ),
        TrainingCheckpointArtifact(
            relative_path="optimizer.bin",
            digest=digests["optimizer"],
            size_bytes=1,
            role="optimizer",
        ),
        TrainingCheckpointArtifact(
            relative_path="state.bin", digest=digests["state"], size_bytes=1, role="trainer_state"
        ),
        TrainingCheckpointArtifact(
            relative_path="telemetry.bin",
            digest=digests["telemetry"],
            size_bytes=1,
            role="telemetry",
        ),
        TrainingCheckpointArtifact(
            relative_path="targets.bin",
            digest=digests["targets"],
            size_bytes=1,
            role="trainable_targets",
        ),
        TrainingCheckpointArtifact(
            relative_path="frozen.bin",
            digest=digests["frozen"],
            size_bytes=1,
            role="frozen_parameters",
        ),
        TrainingCheckpointArtifact(
            relative_path="gradients.bin",
            digest=digests["gradients"],
            size_bytes=1,
            role="finite_gradient_proof",
        ),
    )
    evidence = TrainingCheckpointEvidence(
        run_id="run",
        recipe_digest=_digest("recipe"),
        dataset_digest=_digest("dataset"),
        backend_digest=_digest("backend"),
        runtime_digest=_digest("runtime"),
        material_digest=_digest("material"),
        source_digest=_digest("source"),
        step=25,
        fraction=0.25,
        parent_checkpoint_digest=None,
        safetensors_digest=digests["weights"],
        safetensors_size_bytes=1,
        optimizer_state_digest=digests["optimizer"],
        trainer_state_digest=digests["state"],
        telemetry_digest=digests["telemetry"],
        telemetry_sequence_start=1,
        telemetry_sequence_end=2,
        trainable_targets_digest=digests["targets"],
        frozen_parameters_digest=digests["frozen"],
        finite_gradient_proof_digest=digests["gradients"],
        cancellation_boundary=False,
        inventory=inventory,
    )
    assert evidence.inventory == inventory
    with pytest.raises(ValidationError, match="incomplete"):
        TrainingCheckpointEvidence.model_validate(
            {**evidence.model_dump(mode="python"), "inventory": inventory[:-1]}
        )

    role_fields = {
        "weights": "safetensors_digest",
        "optimizer": "optimizer_state_digest",
        "trainer_state": "trainer_state_digest",
        "telemetry": "telemetry_digest",
        "trainable_targets": "trainable_targets_digest",
        "frozen_parameters": "frozen_parameters_digest",
        "finite_gradient_proof": "finite_gradient_proof_digest",
    }
    payload = evidence.model_dump(mode="python")
    for left_index, _left in enumerate(inventory):
        for right in inventory[left_index + 1 :]:
            swapped = [dict(item) for item in payload["inventory"]]
            swapped[left_index]["digest"], swapped[inventory.index(right)]["digest"] = (
                swapped[inventory.index(right)]["digest"],
                swapped[left_index]["digest"],
            )
            with pytest.raises(ValidationError, match="role binding"):
                TrainingCheckpointEvidence.model_validate({**payload, "inventory": swapped})

    wrong_weight_size = [dict(item) for item in payload["inventory"]]
    wrong_weight_size[0]["size_bytes"] = evidence.safetensors_size_bytes + 1
    with pytest.raises(ValidationError, match="role binding"):
        TrainingCheckpointEvidence.model_validate({**payload, "inventory": wrong_weight_size})

    assert set(role_fields) == {item.role for item in evidence.inventory}


@pytest.mark.parametrize(
    ("total_steps", "expected"),
    ((4, (1, 2, 3, 4)), (8, (2, 4, 6, 8)), (101, (26, 51, 76, 101))),
)
def test_checkpoint_steps_are_deterministic_and_include_final_step(
    total_steps: int, expected: tuple[int, int, int, int]
) -> None:
    assert checkpoint_steps(total_steps) == expected


@pytest.mark.parametrize("total_steps", (1, 2, 3))
def test_checkpoint_steps_reject_unrepresentable_small_runs(total_steps: int) -> None:
    with pytest.raises(ValueError, match="four checkpoint"):
        checkpoint_steps(total_steps)


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _sample(sample_id: str, *, split: DatasetSplit = DatasetSplit.TRAIN) -> TrainingSample:
    digest = _digest(sample_id)
    return TrainingSample(
        sample_id=sample_id,
        source_tier=TrainingSourceTier.GOLD_STILL,
        sample_digest=digest,
        source_lineage=f"lineage-{sample_id}",
        caption="<mirza> in the canonical jacket",
        split=split,
        media=TrainingMedia(
            sha256=digest,
            media_type="image/png",
            byte_size=12,
            width=4,
            height=3,
        ),
        annotation=TrainingAnnotation(
            canonical_outfit_id="jacket-v1",
            head_angle="front",
            framing="portrait",
            motion_speed="slow",
            camera_motion="static",
            expression_change=False,
            partial_occlusion=False,
            lighting="ordinary",
            visible_outfit_details=True,
        ),
    )


def _recipe(**updates: object) -> TrainingRecipe:
    values: dict[str, object] = {
        "schema_version": "character-training-recipe.v1",
        "dataset_digest": _digest("dataset"),
        "character_pack_digest": _digest("pack"),
        "stage_2_manifest_digest": _digest("manifest"),
        "stage_2_evidence_digest": _digest("stage2"),
        "stage_3_evidence_digest": _digest("stage3"),
        "base_model_digest": _digest("base"),
        "lightning_adapter_digests": {
            "adapter-b": _digest("adapter-b"),
            "adapter-a": _digest("adapter-a"),
        },
        "source_code_digest": _digest("source"),
        "runtime_identity": {"python": "3.12", "os": "test"},
        "backend_id": "test-backend",
        "backend_config": {"precision": "bf16", "batch_size": 2},
        "trainable_targets": ("layers.2", "layers.1"),
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "total_steps": 200,
        "gradient_accumulation_steps": 4,
        "source_tier_weights": {
            TrainingSourceTier.GOLD_STILL: 1.0,
            TrainingSourceTier.GOLD_CLIP: 1.0,
            TrainingSourceTier.SYNTHETIC_CLIP: 0.5,
        },
        "checkpoint_plan": CheckpointPlan(),
        "calibration_steps": 10,
        "wall_clock_limit_seconds": 28800,
        "seed": 7,
    }
    values.update(updates)
    return TrainingRecipe(**values)


def test_frozen_sample_rejects_mutation_and_unknown_fields() -> None:
    """Changing an accepted sample or silently accepting an unknown field is a bug."""
    sample = _sample("a")

    with pytest.raises(ValidationError, match="Instance is frozen"):
        sample.caption = "changed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingMedia.model_validate({**sample.media.model_dump(), "path": "/secret/source.png"})


def test_recipe_id_binds_each_scientific_and_resource_field() -> None:
    """Omitting any scientific or resource field from recipe identity is a bug."""
    recipe = _recipe()
    changed = recipe.model_copy(update={"lora_rank": 32})

    assert training_recipe_id(recipe) != training_recipe_id(changed)
    assert recipe.checkpoint_plan.fractions == (0.25, 0.5, 0.75, 1.0)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("schema_version", "changed-schema"),
        ("dataset_digest", _digest("changed-dataset")),
        ("character_pack_digest", _digest("changed-pack")),
        ("stage_2_manifest_digest", _digest("changed-manifest")),
        ("stage_2_evidence_digest", _digest("changed-stage2")),
        ("stage_3_evidence_digest", _digest("changed-stage3")),
        ("base_model_digest", _digest("changed-base")),
        ("lightning_adapter_digests", {"adapter-a": _digest("changed-adapter")}),
        ("source_code_digest", _digest("changed-source")),
        ("runtime_identity", {"python": "3.13", "os": "test"}),
        ("backend_id", "changed-backend"),
        ("backend_config", {"precision": "fp32", "batch_size": 2}),
        ("trainable_targets", ("layers.1", "layers.3")),
        ("lora_rank", 32),
        ("lora_alpha", 64),
        ("lora_dropout", 0.1),
        ("optimizer", "sgd"),
        ("learning_rate", 0.0002),
        ("total_steps", 201),
        ("gradient_accumulation_steps", 8),
        (
            "source_tier_weights",
            {
                TrainingSourceTier.GOLD_STILL: 2.0,
                TrainingSourceTier.GOLD_CLIP: 1.0,
                TrainingSourceTier.SYNTHETIC_CLIP: 0.5,
            },
        ),
        ("checkpoint_plan", CheckpointPlan.model_construct(fractions=(0.25,))),
        ("calibration_steps", 11),
        ("wall_clock_limit_seconds", 57600),
        ("seed", 8),
    ],
)
def test_recipe_id_changes_when_any_identity_field_changes(
    field: str, changed_value: object
) -> None:
    """Omitting one recipe field from canonical identity would hide a changed experiment."""
    recipe = _recipe()

    changed = TrainingRecipe.model_construct(**{**recipe.__dict__, field: changed_value})

    assert training_recipe_id(changed) != training_recipe_id(recipe)


def test_recipe_id_normalizes_mapping_and_set_like_target_order() -> None:
    """Equivalent mappings and target sets receiving distinct IDs is a reproducibility bug."""
    left = _recipe()
    right = _recipe(
        lightning_adapter_digests={
            "adapter-a": _digest("adapter-a"),
            "adapter-b": _digest("adapter-b"),
        },
        runtime_identity={"os": "test", "python": "3.12"},
        trainable_targets=("layers.1", "layers.2"),
    )

    assert training_recipe_id(left) == training_recipe_id(right)


@pytest.mark.parametrize(
    "fractions",
    [
        (0.25, 0.5, 0.75),
        (0.25, 0.5, 0.75, 0.75, 1.0),
        (0.5, 0.25, 0.75, 1.0),
        (0.25, 0.5, 0.75, 0.9),
    ],
)
def test_checkpoint_plan_rejects_missing_duplicate_reordered_or_invalid_fractions(
    fractions: tuple[float, ...],
) -> None:
    """An incomplete or ambiguous checkpoint schedule would break frozen comparisons."""
    with pytest.raises(ValidationError):
        CheckpointPlan(fractions=fractions)


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (TrainingMedia, "sha256", "sha256:not-hex"),
        (TrainingMedia, "sha256", "sha256:" + "A" * 64),
        (TrainingSample, "sample_digest", "sha256:" + "0" * 63),
        (TrainingDatasetDraft, "character_pack_digest", "0" * 64),
        (TrainingRecipe, "dataset_digest", "sha256:dataset"),
        (TrainingRecipe, "character_pack_digest", "sha256:pack"),
        (TrainingRecipe, "stage_2_manifest_digest", "sha256:manifest"),
        (TrainingRecipe, "stage_2_evidence_digest", "sha256:stage2"),
        (TrainingRecipe, "stage_3_evidence_digest", "sha256:stage3"),
        (TrainingRecipe, "base_model_digest", "sha256:base"),
        (TrainingRecipe, "source_code_digest", "sha256:source"),
        (TrainingRecipe, "lightning_adapter_digests", {"adapter": "sha256:short"}),
    ],
)
def test_content_digest_fields_require_exact_lowercase_sha256_format(
    model: type[object], field: str, value: object
) -> None:
    """Accepting malformed content IDs would make identity evidence non-interoperable."""
    if model is TrainingMedia:
        payload = _sample("digest-media").media.model_dump()
    elif model is TrainingSample:
        payload = _sample("digest-sample").model_dump()
        payload["media"]["sha256"] = value
    elif model is TrainingDatasetDraft:
        payload = {
            "character_id": "mirza",
            "character_pack_digest": _digest("pack"),
            "canonical_outfit_id": "jacket-v1",
            "reserved_token": "<mirza>",
            "samples": (),
        }
    else:
        payload = _recipe().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError):
        model.model_validate(payload)  # type: ignore[attr-defined]
