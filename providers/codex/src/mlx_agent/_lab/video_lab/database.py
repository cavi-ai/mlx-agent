from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

from video_lab.adaptive_study_models import adaptive_observation_provenance_digest
from video_lab.character_models import CharacterPackDraft
from video_lab.character_pack import character_pack_id
from video_lab.domain import Manifest, canonical_json, utc_now
from video_lab.preference_optimizer import CandidatePair, OptimizerConfig, PreferenceOutcome
from video_lab.training_dataset import validate_training_recipe_dataset
from video_lab.training_models import (
    CalibrationOperationRequest,
    CancelOperationReceipt,
    CancelOperationRequest,
    EvidenceOperationReceipt,
    FrozenTrainingDataset,
    ImportedTrainingSample,
    ProbeOperationRequest,
    ResumeOperationReceipt,
    ResumeOperationRequest,
    StartOperationReceipt,
    StartOperationRequest,
    TrainingCheckpointEvidence,
    TrainingCurationConfirmation,
    TrainingDatasetDraft,
    TrainingRecipe,
    TrainingSample,
    checkpoint_steps,
    imported_training_sample_id,
    training_confirmation_id,
    training_dataset_id,
    training_recipe_id,
)

SCHEMA_VERSION = 49

_CHECKPOINT_MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "reviewer_id",
    "recipe_id",
    "dataset_digest",
    "character_id",
    "step",
    "fraction",
    "parent_checkpoint_id",
    "artifact_digest",
    "safetensors_size_bytes",
    "backend_digest",
    "runtime_digest",
    "material_digest",
    "source_digest",
    "optimizer_state_digest",
    "trainer_state_digest",
    "telemetry_digest",
    "telemetry_sequence_start",
    "telemetry_sequence_end",
    "trainable_targets_digest",
    "frozen_parameters_digest",
    "finite_gradient_proof_digest",
    "cancellation_boundary",
    "inventory",
}
_CHECKPOINT_EVIDENCE_KEYS = {
    "schema_version",
    "run_id",
    "recipe_digest",
    "dataset_digest",
    "backend_digest",
    "runtime_digest",
    "material_digest",
    "source_digest",
    "step",
    "fraction",
    "parent_checkpoint_digest",
    "safetensors_digest",
    "safetensors_size_bytes",
    "optimizer_state_digest",
    "trainer_state_digest",
    "telemetry_digest",
    "telemetry_sequence_start",
    "telemetry_sequence_end",
    "trainable_targets_digest",
    "frozen_parameters_digest",
    "finite_gradient_proof_digest",
    "cancellation_boundary",
    "inventory",
}
_CHECKPOINT_INVENTORY_KEYS = {"relative_path", "digest", "size_bytes", "role"}


def _sha256_digest_is_valid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _training_checkpoint_json_is_exact(manifest_json: object, evidence_json: object) -> int:
    try:
        manifest = json.loads(str(manifest_json))
        evidence = json.loads(str(evidence_json))
        if not isinstance(manifest, dict) or not isinstance(evidence, dict):
            return 0
        if set(manifest) != _CHECKPOINT_MANIFEST_KEYS or set(evidence) != _CHECKPOINT_EVIDENCE_KEYS:
            return 0
        if manifest["schema_version"] != "character-training-checkpoint.v1":
            return 0
        manifest_inventory = manifest["inventory"]
        evidence_inventory = evidence["inventory"]
        if manifest_inventory != evidence_inventory or not isinstance(evidence_inventory, list):
            return 0
        if len(evidence_inventory) != 7:
            return 0
        if any(
            not isinstance(item, dict) or set(item) != _CHECKPOINT_INVENTORY_KEYS
            for item in evidence_inventory
        ):
            return 0
        TrainingCheckpointEvidence.model_validate(evidence)
        digest_values = [
            value
            for document in (manifest, evidence)
            for key, value in document.items()
            if key.endswith("digest") and value is not None
        ] + [item["digest"] for item in evidence_inventory]
        return int(all(_sha256_digest_is_valid(value) for value in digest_values))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0


def validation_experiment_id(campaign_id: str, trial_id: str) -> str:
    if not campaign_id or not trial_id:
        raise ValueError("validation campaign and trial ids must not be empty")
    payload = canonical_json(
        {
            "schema_version": "validation-experiment-identity.v1",
            "domain": "video-lab.validation.exp",
            "campaign_id": campaign_id,
            "trial_id": trial_id,
            "role": "experiment",
        }
    )
    return f"validation_exp_{hashlib.sha256(payload.encode()).hexdigest()}"


def character_benchmark_reviewer_id(campaign_id: str, trial_id: str, reviewer_id: str) -> str:
    payload = f"character-benchmark-review.v1\x00{campaign_id}\x00{trial_id}\x00{reviewer_id}"
    return "character_reviewer_" + hashlib.sha256(payload.encode()).hexdigest()


def character_benchmark_experiment_id(campaign_id: str, trial_id: str) -> str:
    payload = canonical_json(
        {
            "domain": "video-lab.character-benchmark.exp",
            "campaign_id": campaign_id,
            "identity": trial_id,
        }
    )
    return "character_exp_" + hashlib.sha256(payload.encode()).hexdigest()


def _training_dataset_identity_is_valid(
    connection: sqlite3.Connection, storage_id: object, digest: object, manifest_json: object
) -> int:
    try:
        frozen = FrozenTrainingDataset.model_validate_json(str(manifest_json))
        row = connection.execute(
            "SELECT character_id,character_pack_digest,canonical_outfit_id,reserved_token,"
            "source_tier_weights_json FROM training_datasets WHERE storage_id=?",
            (storage_id,),
        ).fetchone()
        sample_rows = connection.execute(
            "SELECT sample.*,annotation.annotation_json FROM training_samples sample "
            "JOIN training_sample_annotations annotation USING(dataset_storage_id,sample_id) "
            "WHERE sample.dataset_storage_id=? ORDER BY sample.ordinal",
            (storage_id,),
        ).fetchall()
        samples = tuple(
            TrainingSample.model_validate(
                {
                    "sample_id": item["sample_id"],
                    "source_tier": item["source_tier"],
                    "sample_digest": item["sample_digest"],
                    "source_lineage": item["source_lineage"],
                    "caption": item["caption"],
                    "split": item["split"],
                    "media": json.loads(str(item["media_json"])),
                    "annotation": json.loads(str(item["annotation_json"])),
                    "accepted": bool(item["accepted"]),
                }
            )
            for item in sample_rows
        )
        if row is None:
            return 0
        draft = TrainingDatasetDraft(
            character_id=str(row["character_id"]),
            character_pack_digest=str(row["character_pack_digest"]),
            canonical_outfit_id=str(row["canonical_outfit_id"]),
            reserved_token=str(row["reserved_token"]),
            source_tier_weights=json.loads(str(row["source_tier_weights_json"])),
            samples=samples,
            imported_sample_ids=frozen.imported_sample_ids,
            curation_confirmation_ids=frozen.curation_confirmation_ids,
        )
        expected = training_dataset_id(draft)
        return int(
            str(digest) == expected
            and frozen.dataset_digest == expected
            and TrainingDatasetDraft.model_validate(
                frozen.model_dump(mode="python", exclude={"coverage", "dataset_digest"})
            )
            == draft
        )
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return 0


def _training_recipe_identity_is_valid(
    connection: sqlite3.Connection,
    dataset_storage_id: object,
    recipe_id: object,
    recipe_json: object,
) -> int:
    try:
        recipe = TrainingRecipe.model_validate_json(str(recipe_json))
        row = connection.execute(
            "SELECT state,dataset_digest,manifest_json FROM training_datasets WHERE storage_id=?",
            (dataset_storage_id,),
        ).fetchone()
        if row is None or str(row["state"]) != "sealed":
            return 0
        dataset = FrozenTrainingDataset.model_validate_json(str(row["manifest_json"]))
        validate_training_recipe_dataset(recipe, dataset)
        return int(
            str(recipe_id) == training_recipe_id(recipe)
            and recipe.dataset_digest == str(row["dataset_digest"])
        )
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return 0


def _canonical_json_digest(value: object) -> str | None:
    try:
        parsed = json.loads(str(value))
        return "sha256:" + hashlib.sha256(canonical_json(parsed).encode()).hexdigest()
    except (TypeError, ValueError):
        return None


def _canonical_json_is_exact(value: object) -> int:
    if not isinstance(value, str):
        return 0
    try:
        return int(canonical_json(json.loads(value)) == value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0


def _training_operation_json_is_exact(kind: object, phase: object, value: object) -> int:
    request_models = {
        "probe": ProbeOperationRequest,
        "calibration": CalibrationOperationRequest,
        "start": StartOperationRequest,
        "resume": ResumeOperationRequest,
        "cancel": CancelOperationRequest,
    }
    receipt_models = {
        "probe": EvidenceOperationReceipt,
        "calibration": EvidenceOperationReceipt,
        "start": StartOperationReceipt,
        "resume": ResumeOperationReceipt,
        "cancel": CancelOperationReceipt,
    }
    try:
        models = (
            request_models if phase == "request" else receipt_models if phase == "receipt" else {}
        )
        model = cast(Any, models[str(kind)])
        parsed = model.model_validate_json(str(value))
        return int(canonical_json(parsed.model_dump(mode="json")) == value)
    except (KeyError, TypeError, ValueError):
        return 0


def _training_operation_lifecycle_is_valid(kind: object, state: object) -> int:
    if str(kind) == "resume":
        return int(str(state) in {"training", "cancelled"})
    return int(
        str(state)
        == {
            "probe": "draft",
            "calibration": "probed",
            "start": "preregistered",
            "cancel": "training",
        }.get(str(kind))
    )


def _training_operation_completion_target_is_valid(kind: object, state: object) -> int:
    return int(
        str(state)
        == {
            "probe": "probed",
            "calibration": "calibrated",
            "start": "evaluating",
            "resume": "evaluating",
            "cancel": "cancelled",
        }.get(str(kind))
    )


def _training_operation_receipt_links_are_exact(
    connection: sqlite3.Connection, operation_id: object, receipt_json: object
) -> int:
    try:
        operation = connection.execute(
            "SELECT operation_kind FROM training_orchestration_operations WHERE id=?",
            (str(operation_id),),
        ).fetchone()
        if operation is None:
            return 0
        kind = str(operation[0])
        payload = json.loads(str(receipt_json))
        if kind in {"start", "resume"}:
            expected: set[tuple[str, str, str]] = set()
            for field, link_kind in (
                ("events", "event"),
                ("telemetry", "telemetry"),
                ("checkpoints", "checkpoint"),
                ("artifacts", "artifact"),
                ("inventory", "inventory"),
            ):
                expected.update(
                    (link_kind, entry["evidence_id"], entry["digest"]) for entry in payload[field]
                )
            if kind == "resume":
                expected.add(
                    ("resume", payload["resume"]["evidence_id"], payload["resume"]["digest"])
                )
            if payload["evaluation"] is not None:
                expected.add(
                    (
                        "evaluation",
                        payload["evaluation"]["evidence_id"],
                        payload["evaluation"]["digest"],
                    )
                )
        elif kind == "cancel":
            expected = {
                (
                    "cancel",
                    payload["cancellation"]["evidence_id"],
                    payload["cancellation"]["digest"],
                ),
                (
                    "event",
                    payload["terminal_event"]["evidence_id"],
                    payload["terminal_event"]["digest"],
                ),
                (
                    "checkpoint",
                    payload["boundary_checkpoint"]["evidence_id"],
                    payload["boundary_checkpoint"]["digest"],
                ),
            }
        else:
            expected = {(kind, payload["evidence_id"], payload["evidence_digest"])}
        actual = {
            (str(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT evidence_kind,evidence_id,evidence_digest "
                "FROM training_operation_evidence_links WHERE operation_id=?",
                (str(operation_id),),
            )
        }
        return int(actual == expected)
    except (KeyError, TypeError, ValueError, sqlite3.DatabaseError):
        return 0


def _training_operation_request_identity_is_valid(
    connection: sqlite3.Connection, kind: object, run_id: object, payload_json: object
) -> int:
    try:
        payload = json.loads(str(payload_json))
        row = connection.execute(
            "SELECT recipe.id recipe_digest,recipe.dataset_digest,recipe.recipe_json,"
            "material.runtime_digest,material.stage_3_campaign_id,material.stage_3_result_id,"
            "material.base_model_artifact_id,material.lightning_adapter_artifact_ids_json,"
            "material.source_snapshot_artifact_id,source.sha256 source_sha256 "
            "FROM training_runs run JOIN training_recipes recipe "
            "ON recipe.storage_id=run.recipe_storage_id JOIN training_recipe_materials material "
            "ON material.recipe_storage_id=recipe.storage_id JOIN artifacts source "
            "ON source.id=material.source_snapshot_artifact_id WHERE run.id=?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            return 0
        material_payload = {
            "stage_3_campaign_id": row["stage_3_campaign_id"],
            "stage_3_result_id": row["stage_3_result_id"],
            "base_model_artifact_id": row["base_model_artifact_id"],
            "lightning_adapter_artifact_ids": json.loads(
                row["lightning_adapter_artifact_ids_json"]
            ),
            "source_snapshot_artifact_id": row["source_snapshot_artifact_id"],
            "runtime_digest": row["runtime_digest"],
        }
        common = (
            payload["recipe_digest"] == row["recipe_digest"]
            and payload["dataset_digest"] == row["dataset_digest"]
            and payload["backend_digest"]
            == _canonical_json_digest(json.dumps(json.loads(row["recipe_json"])["backend_config"]))
            and payload["runtime_digest"] == row["runtime_digest"]
            and payload["source_digest"] == "sha256:" + row["source_sha256"]
            and payload["material_digest"] == _canonical_json_digest(json.dumps(material_payload))
        )
        if not common:
            return 0
        if kind == "calibration":
            recipe_payload = json.loads(row["recipe_json"])
            legacy_keys = {
                "request_id",
                "run_id",
                "recipe_digest",
                "dataset_digest",
                "backend_digest",
                "runtime_digest",
                "material_digest",
                "source_digest",
                "probe_operation_id",
                "capability_digest",
                "steps",
                "trainable_targets",
                "seed",
            }
            current_keys = legacy_keys | {"lora"}
            if set(payload) not in (legacy_keys, current_keys):
                return 0
            expected_lora = {
                "rank": recipe_payload["lora_rank"],
                "alpha": recipe_payload["lora_alpha"],
                "dropout": recipe_payload["lora_dropout"],
            }
            lora_is_valid = "lora" not in payload or payload["lora"] == expected_lora
            return int(
                payload["steps"] == recipe_payload["calibration_steps"]
                and payload["trainable_targets"] == recipe_payload["trainable_targets"]
                and lora_is_valid
                and payload["seed"] == recipe_payload["seed"]
                and connection.execute(
                    "SELECT 1 FROM training_orchestration_operations operation "
                    "JOIN training_probe_evidence evidence "
                    "ON evidence.id=json_extract(operation.receipt_payload_json,'$.evidence_id') "
                    "WHERE operation.id=? AND operation.run_id=? "
                    "AND operation.operation_kind='probe' AND operation.status='completed' "
                    "AND evidence.capability_digest=? AND evidence.succeeded=1",
                    (payload["probe_operation_id"], str(run_id), payload["capability_digest"]),
                ).fetchone()
                is not None
            )
        if kind == "start":
            preregistered = connection.execute(
                "SELECT 1 FROM training_run_idempotency WHERE run_id=? "
                "AND idempotency_key=? AND request_digest=?",
                (
                    str(run_id),
                    payload["preregistration_id"],
                    payload["preregistration_request_digest"],
                ),
            ).fetchone()
            recipe_payload = json.loads(row["recipe_json"])
            fractions = recipe_payload["checkpoint_plan"]["fractions"]
            steps = [
                _training_checkpoint_step(recipe_payload["total_steps"], fraction)
                for fraction in fractions
            ]
            exact_materials = {
                "base_model": row["base_model_artifact_id"],
                "source_snapshot": row["source_snapshot_artifact_id"],
                **{
                    f"lightning:{name}": artifact_id
                    for name, artifact_id in sorted(
                        json.loads(row["lightning_adapter_artifact_ids_json"]).items()
                    )
                },
            }
            return int(
                preregistered is not None
                and payload["checkpoint_fractions"] == fractions
                and payload["checkpoint_steps"] == steps
                and payload["material_artifact_ids"] == exact_materials
            )
        if kind == "resume":
            checkpoint = connection.execute(
                "SELECT evidence_digest FROM training_checkpoints WHERE id=? AND run_id=? "
                "AND publication_state='finalized'",
                (payload["parent_checkpoint_id"], str(run_id)),
            ).fetchone()
            if checkpoint is None or checkpoint[0] != payload["parent_checkpoint_digest"]:
                return 0
            persisted = {
                (
                    str(item["role"]),
                    str(item["artifact_id"]),
                    str(item["artifact_digest"]),
                    int(item["size_bytes"]),
                )
                for item in connection.execute(
                    "SELECT role,artifact_id,artifact_digest,size_bytes "
                    "FROM training_checkpoint_inventory_artifacts WHERE checkpoint_id=?",
                    (payload["parent_checkpoint_id"],),
                )
            }
            requested = {
                (
                    item["role"],
                    item["artifact_id"],
                    item["digest"],
                    int(item["size_bytes"]),
                )
                for item in payload["parent_inventory"]
            }
            return int(persisted == requested and len(persisted) == 7)
        if kind == "cancel":
            request = connection.execute(
                "SELECT requested_step,request_digest FROM training_cancellation_requests "
                "WHERE id=? AND run_id=? AND reviewer_id=(SELECT reviewer_id FROM training_runs WHERE id=?)",
                (payload["cancellation_request_id"], str(run_id), str(run_id)),
            ).fetchone()
            return int(
                request is not None
                and int(request[0]) == payload["requested_step"]
                and str(request[1]) == payload["request_digest"]
            )
        return 1
    except (KeyError, TypeError, ValueError, sqlite3.DatabaseError):
        return 0


def _training_operation_completion_is_derived(
    connection: sqlite3.Connection, operation_id: object
) -> int:
    try:
        operation = connection.execute(
            "SELECT operation_kind,run_id,updated_at FROM training_orchestration_operations "
            "WHERE id=? AND status='running'",
            (str(operation_id),),
        ).fetchone()
        if operation is None:
            return 0
        kind, run_id, started_at = map(str, operation)
        state = connection.execute(
            "SELECT lifecycle_state FROM training_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
        if _training_operation_completion_target_is_valid(kind, state) != 1:
            return 0
        links: dict[str, set[str]] = {}
        for row in connection.execute(
            "SELECT evidence_kind,evidence_id FROM training_operation_evidence_links "
            "WHERE operation_id=?",
            (str(operation_id),),
        ):
            links.setdefault(str(row[0]), set()).add(str(row[1]))
        if kind in {"probe", "calibration"}:
            return int(len(links.get(kind, set())) == 1 and sum(map(len, links.values())) == 1)
        events = {
            str(row[0])
            for row in connection.execute(
                "SELECT sequence FROM training_run_events WHERE run_id=? "
                "AND julianday(created_at)>=julianday(?)",
                (run_id, started_at),
            )
        }
        if links.get("event", set()) != events or not events:
            return 0
        if kind == "cancel":
            if set(links) != {"cancel", "event", "checkpoint"} or any(
                len(links[item]) != 1 for item in ("cancel", "event", "checkpoint")
            ):
                return 0
            cancellation_id = next(iter(links["cancel"]))
            event_id = next(iter(links["event"]))
            checkpoint_id = next(iter(links["checkpoint"]))
            exact = connection.execute(
                "SELECT 1 FROM training_cancellation_evidence cancellation "
                "JOIN training_checkpoints checkpoint ON checkpoint.id=? "
                "JOIN training_run_events event ON event.run_id=cancellation.run_id "
                "AND CAST(event.sequence AS TEXT)=? "
                "WHERE cancellation.id=? AND cancellation.run_id=? "
                "AND checkpoint.run_id=cancellation.run_id "
                "AND checkpoint.step=cancellation.boundary_step "
                "AND checkpoint.artifact_digest=cancellation.checkpoint_digest "
                "AND event.previous_state='training' AND event.new_state='cancelled' "
                "AND json_extract(event.payload_json,'$.cancellation_id')=cancellation.id "
                "AND json_extract(event.payload_json,'$.cancellation_request_id')="
                "cancellation.cancellation_request_id "
                "AND json_extract(event.payload_json,'$.request_digest')=cancellation.request_digest "
                "AND json_extract(event.payload_json,'$.checkpoint_digest')=checkpoint.artifact_digest "
                "AND json_extract(event.payload_json,'$.boundary_step')=checkpoint.step",
                (checkpoint_id, event_id, cancellation_id, run_id),
            ).fetchone()
            return int(exact is not None)
        telemetry = {
            str(row[0])
            for row in connection.execute(
                "SELECT sequence FROM training_telemetry WHERE run_id=? "
                "AND julianday(created_at)>=julianday(?)",
                (run_id, started_at),
            )
        }
        checkpoints = {
            str(row[0])
            for row in connection.execute(
                "SELECT id FROM training_checkpoints WHERE run_id=? "
                "AND julianday(created_at)>=julianday(?) AND publication_state='finalized'",
                (run_id, started_at),
            )
        }
        inventory = {
            f"{row[0]}:{row[1]}"
            for row in connection.execute(
                "SELECT binding.checkpoint_id,binding.role "
                "FROM training_checkpoint_inventory_artifacts binding "
                "JOIN training_checkpoints checkpoint ON checkpoint.id=binding.checkpoint_id "
                "WHERE checkpoint.run_id=? AND julianday(checkpoint.created_at)>=julianday(?)",
                (run_id, started_at),
            )
        }
        artifacts = {
            str(row[0])
            for row in connection.execute(
                "SELECT binding.artifact_id FROM training_checkpoint_inventory_artifacts binding "
                "JOIN training_checkpoints checkpoint ON checkpoint.id=binding.checkpoint_id "
                "WHERE checkpoint.run_id=? AND julianday(checkpoint.created_at)>=julianday(?)",
                (run_id, started_at),
            )
        }
        required_kinds = {"event", "telemetry", "checkpoint", "inventory", "artifact"}
        if kind == "resume":
            required_kinds.add("resume")
        if "evaluation" in links:
            required_kinds.add("evaluation")
        return int(
            bool(telemetry and checkpoints and inventory and artifacts)
            and set(links) == required_kinds
            and (kind != "resume" or len(links["resume"]) == 1)
            and links.get("telemetry", set()) == telemetry
            and links.get("checkpoint", set()) == checkpoints
            and links.get("inventory", set()) == inventory
            and links.get("artifact", set()) == artifacts
        )
    except (KeyError, TypeError, sqlite3.DatabaseError):
        return 0


def _training_operation_evidence_digest(
    connection: sqlite3.Connection, operation_id: object, kind: object, evidence_id: object
) -> str | None:
    try:
        operation = connection.execute(
            "SELECT run_id FROM training_orchestration_operations WHERE id=?",
            (str(operation_id),),
        ).fetchone()
        if operation is None:
            return None
        run_id = str(operation[0])
        table_specs = {
            "probe": ("training_probe_evidence", "id"),
            "calibration": ("training_calibration_evidence", "id"),
            "resume": ("training_resume_evidence", "id"),
            "cancel": ("training_cancellation_evidence", "id"),
            "evaluation": ("training_evaluation_evidence", "id"),
        }
        if str(kind) in table_specs:
            table, key = table_specs[str(kind)]
            row = connection.execute(
                f"SELECT * FROM {table} WHERE {key}=? AND run_id=?",  # noqa: S608
                (str(evidence_id), run_id),
            ).fetchone()
            if row is None:
                return None
            payload = {name: row[name] for name in row.keys() if name != "created_at"}
            return "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        if str(kind) in {"event", "telemetry"}:
            table = "training_run_events" if str(kind) == "event" else "training_telemetry"
            row = connection.execute(
                f"SELECT * FROM {table} WHERE run_id=? AND sequence=?",  # noqa: S608
                (run_id, int(str(evidence_id))),
            ).fetchone()
            if row is None:
                return None
            payload = {
                name: (json.loads(str(row[name])) if name == "payload_json" else row[name])
                for name in row.keys()
                if name != "created_at"
            }
            return "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        if str(kind) == "checkpoint":
            row = connection.execute(
                "SELECT evidence_digest FROM training_checkpoints WHERE id=? AND run_id=?",
                (str(evidence_id), run_id),
            ).fetchone()
            return None if row is None else str(row[0])
        if str(kind) == "artifact":
            row = connection.execute(
                "SELECT sha256 FROM artifacts WHERE id=?", (str(evidence_id),)
            ).fetchone()
            return None if row is None else "sha256:" + str(row[0])
        if str(kind) == "inventory":
            checkpoint_id, role = str(evidence_id).rsplit(":", 1)
            row = connection.execute(
                "SELECT * FROM training_checkpoint_inventory_artifacts "
                "WHERE checkpoint_id=? AND role=?",
                (checkpoint_id, role),
            ).fetchone()
            if row is None:
                return None
            payload = {name: row[name] for name in row.keys()}
            return "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        return None
    except (TypeError, ValueError, sqlite3.DatabaseError):
        return None


def _v43_training_operation_results(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, object]]:
    """Derive public results or reject before any v44 schema mutation."""
    derived: dict[str, dict[str, object]] = {}
    for row in connection.execute(
        "SELECT id,operation_kind,run_id,request_payload_json,receipt_payload_json "
        "FROM training_orchestration_operations WHERE status='completed'"
    ):
        kind = str(row["operation_kind"])
        if kind in {"probe", "calibration"}:
            raise sqlite3.IntegrityError(
                "training operation typed result is not derivable from v43 receipt"
            )
        receipt = json.loads(str(row["receipt_payload_json"]))
        if kind == "cancel":
            derived[str(row["id"])] = receipt
            continue
        if kind not in {"start", "resume"}:
            raise sqlite3.IntegrityError(
                "training operation typed result is not derivable from v43 receipt"
            )
        request = json.loads(str(row["request_payload_json"]))
        checkpoints = receipt.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise sqlite3.IntegrityError(
                "training operation typed result is not derivable from v43 receipt"
            )
        checkpoint_ids = [item.get("evidence_id") for item in checkpoints]
        if any(not isinstance(item, str) for item in checkpoint_ids):
            raise sqlite3.IntegrityError(
                "training operation typed result is not derivable from v43 receipt"
            )
        final = connection.execute(
            "SELECT json_extract(evidence_json,'$.telemetry_digest') "
            "FROM training_checkpoints WHERE id=? AND run_id=? AND publication_state='finalized'",
            (checkpoint_ids[-1], str(row["run_id"])),
        ).fetchone()
        if final is None or not isinstance(final[0], str):
            raise sqlite3.IntegrityError(
                "training operation typed result is not derivable from v43 receipt"
            )
        derived[str(row["id"])] = {
            "schema_version": "character-training-publication-receipt.v1",
            "request_id": request["request_id"],
            "run_id": str(row["run_id"]),
            "lifecycle_state": receipt["lifecycle_state"],
            "checkpoint_ids": checkpoint_ids,
            "telemetry_digest": str(final[0]),
        }
    return derived


def _apply_migration_44(connection: sqlite3.Connection) -> None:
    results = _v43_training_operation_results(connection)
    statements = _migration_statements(_MIGRATION_44)
    connection.execute(statements[0])
    connection.execute(statements[1])
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='validate_training_operation_update_v42'"
    ).fetchone()
    if trigger is None or not isinstance(trigger[0], str):
        raise sqlite3.IntegrityError("v43 training operation update trigger is missing")
    connection.execute("DROP TRIGGER validate_training_operation_update_v42")
    try:
        for operation_id, payload in results.items():
            payload_json = canonical_json(payload)
            connection.execute(
                "UPDATE training_orchestration_operations SET result_digest=?,"
                "result_payload_json=? WHERE id=? AND status='completed'",
                (
                    "sha256:" + hashlib.sha256(payload_json.encode()).hexdigest(),
                    payload_json,
                    operation_id,
                ),
            )
    finally:
        connection.execute(str(trigger[0]))
    for statement in statements[3:]:
        connection.execute(statement)


def _training_resume_evidence_matches_operation_request(
    connection: sqlite3.Connection, operation_id: object, evidence_id: object
) -> int:
    """Bind resume evidence to the exact requested parent and seven artifact handles."""
    try:
        operation = connection.execute(
            "SELECT run_id,request_payload_json FROM training_orchestration_operations "
            "WHERE id=? AND operation_kind='resume'",
            (str(operation_id),),
        ).fetchone()
        evidence = connection.execute(
            "SELECT * FROM training_resume_evidence WHERE id=?",
            (str(evidence_id),),
        ).fetchone()
        if operation is None or evidence is None:
            return 0
        request = json.loads(str(operation["request_payload_json"]))
        if not (
            evidence["run_id"] == operation["run_id"] == request["run_id"]
            and evidence["parent_checkpoint_id"] == request["parent_checkpoint_id"]
            and evidence["parent_checkpoint_digest"] == request["parent_checkpoint_digest"]
            and all(
                evidence[field] == request[field]
                for field in (
                    "recipe_digest",
                    "dataset_digest",
                    "backend_digest",
                    "runtime_digest",
                    "material_digest",
                    "source_digest",
                )
            )
        ):
            return 0
        requested = {
            (
                item["role"],
                item["artifact_id"],
                item["digest"],
                int(item["size_bytes"]),
            )
            for item in request["parent_inventory"]
        }
        persisted = {
            (
                str(row["role"]),
                str(row["artifact_id"]),
                str(row["artifact_digest"]),
                int(row["size_bytes"]),
            )
            for row in connection.execute(
                "SELECT role,artifact_id,artifact_digest,size_bytes "
                "FROM training_checkpoint_inventory_artifacts WHERE checkpoint_id=?",
                (evidence["parent_checkpoint_id"],),
            )
        }
        if len(requested) != 7 or persisted != requested:
            return 0
        by_role = {role: digest for role, _, digest, _ in persisted}
        checkpoint = connection.execute(
            "SELECT artifact_digest,evidence_json,evidence_digest FROM training_checkpoints "
            "WHERE id=? AND run_id=? AND publication_state='finalized'",
            (evidence["parent_checkpoint_id"], operation["run_id"]),
        ).fetchone()
        if checkpoint is None:
            return 0
        checkpoint_evidence = json.loads(str(checkpoint["evidence_json"]))
        role_fields = {
            "weights": "safetensors_digest",
            "optimizer": "optimizer_state_digest",
            "trainer_state": "trainer_state_digest",
            "telemetry": "telemetry_digest",
            "trainable_targets": "trainable_targets_digest",
            "frozen_parameters": "frozen_parameters_digest",
            "finite_gradient_proof": "finite_gradient_proof_digest",
        }
        return int(
            checkpoint["evidence_digest"] == request["parent_checkpoint_digest"]
            and checkpoint["artifact_digest"] == by_role["weights"]
            and evidence["parent_artifact_digest"] == by_role["weights"]
            and evidence["optimizer_state_digest"] == by_role["optimizer"]
            and all(
                checkpoint_evidence[field] == by_role[role] for role, field in role_fields.items()
            )
            and all(
                checkpoint_evidence[field] == request[field]
                for field in (
                    "recipe_digest",
                    "dataset_digest",
                    "backend_digest",
                    "runtime_digest",
                    "material_digest",
                    "source_digest",
                )
            )
        )
    except (KeyError, TypeError, ValueError, sqlite3.DatabaseError):
        return 0


def _training_operation_evidence_link_is_valid(
    connection: sqlite3.Connection,
    operation_id: object,
    attempt: object,
    claim_token: object,
    kind: object,
    evidence_id: object,
    digest: object,
) -> int:
    """Validate one link against the exact live attempt and operation-specific vocabulary."""
    try:
        operation = connection.execute(
            "SELECT operation_kind,run_id,updated_at FROM training_orchestration_operations "
            "WHERE id=? AND status='running' AND attempt=? AND claim_token=?",
            (str(operation_id), int(str(attempt)), str(claim_token)),
        ).fetchone()
        if operation is None:
            return 0
        permitted = {
            "probe": {"probe"},
            "calibration": {"calibration"},
            "start": {"event", "telemetry", "checkpoint", "inventory", "artifact", "evaluation"},
            "resume": {
                "resume",
                "event",
                "telemetry",
                "checkpoint",
                "inventory",
                "artifact",
                "evaluation",
            },
            "cancel": {"cancel", "event", "checkpoint", "inventory", "artifact"},
        }
        evidence_kind = str(kind)
        if evidence_kind not in permitted[str(operation[0])]:
            return 0
        canonical_digest = _training_operation_evidence_digest(
            connection, operation_id, evidence_kind, evidence_id
        )
        if canonical_digest != str(digest):
            return 0
        if (
            evidence_kind == "resume"
            and _training_resume_evidence_matches_operation_request(
                connection, operation_id, evidence_id
            )
            != 1
        ):
            return 0
        if (
            evidence_kind == "calibration"
            and connection.execute(
                "SELECT 1 FROM training_calibration_operation_bindings WHERE evidence_id=? "
                "AND operation_id=? AND attempt=? AND claim_token=?",
                (str(evidence_id), str(operation_id), int(str(attempt)), str(claim_token)),
            ).fetchone()
            is None
        ):
            return 0
        if (
            evidence_kind == "cancel"
            and connection.execute(
                "SELECT 1 FROM training_cancellation_operation_claims claim "
                "JOIN training_cancellation_evidence evidence "
                "ON evidence.cancellation_request_id=claim.cancellation_request_id "
                "WHERE claim.operation_id=? AND claim.attempt=? AND claim.claim_token=? "
                "AND evidence.id=? AND evidence.request_digest=claim.request_digest "
                "AND evidence.requested_step=claim.requested_step",
                (str(operation_id), int(str(attempt)), str(claim_token), str(evidence_id)),
            ).fetchone()
            is None
        ):
            return 0
        run_id, started_at = str(operation[1]), str(operation[2])
        if evidence_kind in {"probe", "calibration", "resume", "cancel", "evaluation"}:
            table = {
                "probe": "training_probe_evidence",
                "calibration": "training_calibration_evidence",
                "resume": "training_resume_evidence",
                "cancel": "training_cancellation_evidence",
                "evaluation": "training_evaluation_evidence",
            }[evidence_kind]
            row = connection.execute(
                f"SELECT created_at FROM {table} WHERE id=? AND run_id=?",  # noqa: S608
                (str(evidence_id), run_id),
            ).fetchone()
        elif evidence_kind in {"event", "telemetry"}:
            table = "training_run_events" if evidence_kind == "event" else "training_telemetry"
            row = connection.execute(
                f"SELECT created_at FROM {table} WHERE run_id=? AND sequence=?",  # noqa: S608
                (run_id, int(str(evidence_id))),
            ).fetchone()
        else:
            checkpoint_id = (
                str(evidence_id).rsplit(":", 1)[0]
                if evidence_kind == "inventory"
                else str(evidence_id)
            )
            if evidence_kind == "artifact":
                row = connection.execute(
                    "SELECT checkpoint.created_at FROM training_checkpoint_inventory_artifacts binding "
                    "JOIN training_checkpoints checkpoint ON checkpoint.id=binding.checkpoint_id "
                    "WHERE binding.artifact_id=? AND checkpoint.run_id=? "
                    "ORDER BY julianday(checkpoint.created_at) DESC LIMIT 1",
                    (checkpoint_id, run_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT created_at FROM training_checkpoints WHERE id=? AND run_id=?",
                    (checkpoint_id, run_id),
                ).fetchone()
        return int(
            row is not None
            and connection.execute(
                "SELECT julianday(?)>=julianday(?)", (row[0], started_at)
            ).fetchone()[0]
            == 1
        )
    except (KeyError, TypeError, ValueError, sqlite3.DatabaseError):
        return 0


def _training_import_identity_is_valid(identity: object, payload: object) -> int:
    try:
        model = ImportedTrainingSample.model_validate_json(str(payload))
        return int(str(identity) == model.import_id == imported_training_sample_id(model))
    except (TypeError, ValueError):
        return 0


def _training_confirmation_identity_is_valid(identity: object, payload: object) -> int:
    try:
        model = TrainingCurationConfirmation.model_validate_json(str(payload))
        return int(str(identity) == model.confirmation_id == training_confirmation_id(model))
    except (TypeError, ValueError):
        return 0


def _training_ledger_snapshot_is_complete(
    connection: sqlite3.Connection,
    manifest_json: object,
    lineage_id: object,
    reviewer_id: object,
    character_id: object,
) -> int:
    try:
        payload = json.loads(str(manifest_json))
        draft = TrainingDatasetDraft.model_validate(
            {
                key: value
                for key, value in payload.items()
                if key not in {"coverage", "dataset_digest"}
            }
        )
        imports = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT id FROM training_imported_samples WHERE lineage_id=? "
                "AND reviewer_id=? AND character_id=? ORDER BY id",
                (lineage_id, reviewer_id, character_id),
            )
        )
        confirmations = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT confirmation.id FROM training_curation_confirmations confirmation "
                "JOIN training_imported_samples imported ON imported.id=confirmation.import_id "
                "WHERE imported.lineage_id=? AND confirmation.reviewer_id=? "
                "AND imported.character_id=? ORDER BY confirmation.id",
                (lineage_id, reviewer_id, character_id),
            )
        )
        if draft.imported_sample_ids != imports or draft.curation_confirmation_ids != confirmations:
            return 0
        samples = [canonical_json(sample.model_dump(mode="json")) for sample in draft.samples]
        for confirmation_id in confirmations:
            row = connection.execute(
                "SELECT confirmation.payload_json confirmation_json,imported.payload_json import_json "
                "FROM training_curation_confirmations confirmation "
                "JOIN training_imported_samples imported ON imported.id=confirmation.import_id "
                "WHERE confirmation.id=?",
                (confirmation_id,),
            ).fetchone()
            if row is None:
                return 0
            confirmation = TrainingCurationConfirmation.model_validate_json(
                str(row["confirmation_json"])
            )
            imported = ImportedTrainingSample.model_validate_json(str(row["import_json"]))
            if confirmation.decision.value == "reject":
                if any(
                    sample.sample_digest == imported.artifact_digest for sample in draft.samples
                ):
                    return 0
            elif (
                canonical_json(confirmation.accepted_sample(imported).model_dump(mode="json"))
                not in samples
            ):
                return 0
        return 1
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return 0


def _training_ledger_transition_is_valid(
    connection: sqlite3.Connection,
    old_storage_id: object,
    new_storage_id: object,
    new_state: object,
) -> int:
    try:
        rows = connection.execute(
            "SELECT storage_id,manifest_json FROM training_datasets WHERE storage_id IN (?,?)",
            (old_storage_id, new_storage_id),
        ).fetchall()
        manifests = {str(row["storage_id"]): str(row["manifest_json"]) for row in rows}
        old_payload = json.loads(manifests[str(old_storage_id)])
        old = TrainingDatasetDraft.model_validate(
            {
                key: value
                for key, value in old_payload.items()
                if key not in {"coverage", "dataset_digest"}
            }
        )
        new_payload = json.loads(manifests[str(new_storage_id)])
        new = TrainingDatasetDraft.model_validate(
            {
                key: value
                for key, value in new_payload.items()
                if key not in {"coverage", "dataset_digest"}
            }
        )
        old_samples = sorted(canonical_json(item.model_dump(mode="json")) for item in old.samples)
        new_samples = sorted(canonical_json(item.model_dump(mode="json")) for item in new.samples)
        old_imports, new_imports = set(old.imported_sample_ids), set(new.imported_sample_ids)
        old_confirmations = set(old.curation_confirmation_ids)
        new_confirmations = set(new.curation_confirmation_ids)
        if not old_imports and not new_imports and not old_confirmations and not new_confirmations:
            return 1
        if str(new_state) == "sealed":
            return int(
                str(old_storage_id) == str(new_storage_id)
                and old_imports == new_imports
                and old_confirmations == new_confirmations
                and old_samples == new_samples
            )
        added_imports = new_imports - old_imports
        added_confirmations = new_confirmations - old_confirmations
        if added_imports and not added_confirmations:
            return int(
                old_imports < new_imports
                and old_confirmations == new_confirmations
                and old_samples == new_samples
            )
        if len(added_confirmations) != 1 or old_imports != new_imports:
            return 0
        confirmation_row = connection.execute(
            "SELECT payload_json FROM training_curation_confirmations WHERE id=?",
            (next(iter(added_confirmations)),),
        ).fetchone()
        if confirmation_row is None:
            return 0
        confirmation = TrainingCurationConfirmation.model_validate_json(
            str(confirmation_row["payload_json"])
        )
        imported_row = connection.execute(
            "SELECT payload_json FROM training_imported_samples WHERE id=?",
            (confirmation.import_id,),
        ).fetchone()
        if imported_row is None:
            return 0
        if confirmation.decision.value == "reject":
            return int(old_samples == new_samples)
        imported = ImportedTrainingSample.model_validate_json(str(imported_row["payload_json"]))
        expected = sorted(
            old_samples
            + [canonical_json(confirmation.accepted_sample(imported).model_dump(mode="json"))]
        )
        return int(new_samples == expected)
    except (KeyError, sqlite3.DatabaseError, TypeError, ValueError):
        return 0


def _training_draft_identity_is_valid(
    connection: sqlite3.Connection, storage_id: object, identity: object, manifest_json: object
) -> int:
    try:
        draft = TrainingDatasetDraft.model_validate_json(str(manifest_json))
        rows = connection.execute(
            "SELECT sample.*,annotation.annotation_json FROM training_samples sample "
            "JOIN training_sample_annotations annotation USING(dataset_storage_id,sample_id) "
            "WHERE sample.dataset_storage_id=? ORDER BY sample.ordinal",
            (storage_id,),
        ).fetchall()
        persisted = tuple(
            TrainingSample.model_validate(
                {
                    "sample_id": row["sample_id"],
                    "source_tier": row["source_tier"],
                    "sample_digest": row["sample_digest"],
                    "source_lineage": row["source_lineage"],
                    "caption": row["caption"],
                    "split": row["split"],
                    "media": json.loads(str(row["media_json"])),
                    "annotation": json.loads(str(row["annotation_json"])),
                    "accepted": bool(row["accepted"]),
                }
            )
            for row in rows
        )
        return int(draft.samples == persisted and training_dataset_id(draft) == str(identity))
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return 0


def _training_draft_manifest_is_valid(
    identity: object,
    character_id: object,
    character_pack_digest: object,
    canonical_outfit_id: object,
    reserved_token: object,
    source_tier_weights_json: object,
    manifest_json: object,
) -> int:
    """Validate every duplicated dataset column against one canonical draft manifest."""
    try:
        draft = TrainingDatasetDraft.model_validate_json(str(manifest_json))
        weights = json.loads(str(source_tier_weights_json))
        expected_weights = {
            tier.value: weight for tier, weight in draft.source_tier_weights.items()
        }
        return int(
            str(identity) == training_dataset_id(draft)
            and str(character_id) == draft.character_id
            and str(character_pack_digest) == draft.character_pack_digest
            and str(canonical_outfit_id) == draft.canonical_outfit_id
            and str(reserved_token) == draft.reserved_token
            and weights == expected_weights
        )
    except (TypeError, ValueError):
        return 0


def _training_checkpoint_step(total_steps: object, fraction: object) -> int | None:
    try:
        fractions = (0.25, 0.5, 0.75, 1.0)
        return checkpoint_steps(int(str(total_steps)))[fractions.index(float(str(fraction)))]
    except (TypeError, ValueError):
        return None


def _character_benchmark_provenance_is_valid(
    connection: sqlite3.Connection, campaign_id: object
) -> int:
    try:
        row = connection.execute(
            "SELECT campaign.*,result.result_json,result.content_digest AS result_digest "
            "FROM character_benchmark_campaigns campaign "
            "JOIN character_benchmark_results result ON result.campaign_id=campaign.id "
            "WHERE campaign.id=?",
            (campaign_id,),
        ).fetchone()
        if row is None:
            return 0
        campaign_payload = {
            "definition": json.loads(str(row["definition_json"])),
            "character_pack_id": str(row["character_pack_id"]),
            "character_pack_storage_id": str(row["character_pack_storage_id"]),
            "stage_2_campaign_id": str(row["stage_2_campaign_id"]),
            "stage_2_manifest": json.loads(str(row["stage_2_manifest_json"])),
            "identity_prompt": str(row["identity_prompt"]),
        }
        campaign_digest = (
            "sha256:" + hashlib.sha256(canonical_json(campaign_payload).encode()).hexdigest()
        )
        result_digest = (
            "sha256:"
            + hashlib.sha256(
                canonical_json(json.loads(str(row["result_json"]))).encode()
            ).hexdigest()
        )
        stage_2_manifest_digest = _canonical_json_digest(row["stage_2_manifest_json"])
        stage_2_evidence_digest = _validation_evidence_digest(
            connection, str(row["stage_2_campaign_id"]), str(row["reviewer_id"])
        )
        definition = json.loads(str(row["definition_json"]))
        return int(
            campaign_digest == str(row["content_digest"])
            and result_digest == str(row["result_digest"])
            and definition.get("stage_2_manifest_digest") == stage_2_manifest_digest
            and definition.get("stage_2_evidence_digest") == stage_2_evidence_digest
        )
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return 0


def _validation_evidence_digest(
    connection: sqlite3.Connection, campaign_id: str, reviewer_id: str
) -> str | None:
    """Rebuild the canonical Stage-2 export from its owned persisted graph."""
    try:
        campaign = connection.execute(
            "SELECT * FROM validation_campaigns WHERE id=? AND reviewer_id=?",
            (campaign_id, reviewer_id),
        ).fetchone()
        result = connection.execute(
            "SELECT * FROM validation_results WHERE campaign_id=?", (campaign_id,)
        ).fetchone()
        state = connection.execute(
            "SELECT new_state FROM validation_events WHERE campaign_id=? "
            "ORDER BY sequence DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        if campaign is None or result is None or state is None or state[0] != "qualified":
            return None
        trials = connection.execute(
            "SELECT * FROM validation_trials WHERE campaign_id=? ORDER BY ordinal", (campaign_id,)
        ).fetchall()
        events = connection.execute(
            "SELECT * FROM validation_events WHERE campaign_id=? ORDER BY sequence", (campaign_id,)
        ).fetchall()
        judgments = connection.execute(
            "SELECT * FROM validation_judgments WHERE campaign_id=? ORDER BY ordinal",
            (campaign_id,),
        ).fetchall()
        experiment_bindings = [
            dict(item)
            for item in connection.execute(
                "SELECT binding.campaign_id,binding.trial_id,binding.experiment_id "
                "FROM validation_experiment_bindings binding JOIN validation_trials trial "
                "ON trial.campaign_id=binding.campaign_id AND trial.trial_id=binding.trial_id "
                "WHERE binding.campaign_id=? ORDER BY trial.ordinal",
                (campaign_id,),
            )
        ]
        review_rows = connection.execute(
            "SELECT binding.campaign_id,binding.trial_id,binding.experiment_id,binding.review_id,"
            "binding.reused_from_trial_id,trial.ordinal,trial.winner_display_position,"
            "judgment.choice,review.session_id,review.revision,review.display_preference,"
            "review.preferred_variant_id,session.left_variant_id,session.left_job_id,"
            "session.left_artifact_id,session.left_artifact_sha256,session.right_variant_id,"
            "session.right_job_id,session.right_artifact_id,session.right_artifact_sha256,"
            "baseline.id AS baseline_id FROM validation_trial_bindings binding "
            "JOIN validation_trials trial ON trial.campaign_id=binding.campaign_id "
            "AND trial.trial_id=binding.trial_id JOIN validation_judgments judgment "
            "ON judgment.campaign_id=binding.campaign_id AND judgment.trial_id=binding.trial_id "
            "JOIN review_revisions review ON review.id=binding.review_id "
            "JOIN review_sessions session ON session.id=review.session_id "
            "JOIN variants baseline ON baseline.experiment_id=binding.experiment_id "
            "AND baseline.label='A' WHERE binding.campaign_id=? ORDER BY trial.ordinal",
            (campaign_id,),
        ).fetchall()
        if (
            len(trials) != 40
            or sum(not bool(item["hidden_repeat"]) for item in trials) != 36
            or len(events) != 6
            or len(judgments) != 40
            or len(experiment_bindings) != 36
            or len(review_rows) != 40
        ):
            return None
        trial_payloads = [
            {
                "id": item["trial_id"],
                "ordinal": item["ordinal"],
                "archetype": item["archetype"],
                "prompt": item["prompt"],
                "negative_prompt": item["negative_prompt"] or None,
                "seed": item["seed"],
                "challenger": item["challenger"],
                "hidden_repeat": bool(item["hidden_repeat"]),
                "reuses_trial_id": item["reuses_trial_id"],
                "winner_display_position": item["winner_display_position"],
            }
            for item in trials
        ]
        binding_payloads: list[dict[str, object]] = []
        review_payloads: list[dict[str, object]] = []
        for item in review_rows:
            binding_payloads.append(
                {
                    "campaign_id": item["campaign_id"],
                    "trial_id": item["trial_id"],
                    "experiment_id": item["experiment_id"],
                    "review_id": item["review_id"],
                    "reused_from_trial_id": item["reused_from_trial_id"],
                }
            )
            winner = "left" if item["left_variant_id"] == item["baseline_id"] else "right"
            swap = winner != item["winner_display_position"]
            mapping: dict[str, dict[str, str]] = {}
            for position in ("left", "right"):
                underlying = ("right" if position == "left" else "left") if swap else position
                mapping[position] = {
                    "variant_id": str(item[f"{underlying}_variant_id"]),
                    "job_id": str(item[f"{underlying}_job_id"]),
                    "artifact_id": str(item[f"{underlying}_artifact_id"]),
                    "artifact_sha256": str(item[f"{underlying}_artifact_sha256"]),
                }
            review_payloads.append(
                {
                    "trial_id": item["trial_id"],
                    "review_id": item["review_id"],
                    "session_id": item["session_id"],
                    "revision": item["revision"],
                    "choice": item["choice"],
                    "persisted_display_preference": item["display_preference"],
                    "preferred_variant_id": item["preferred_variant_id"],
                    "display_mapping": mapping,
                }
            )
        provenance = {
            "experiment_bindings": experiment_bindings,
            "trial_bindings": binding_payloads,
            "reviews": review_payloads,
        }
        campaign_payload = {
            "id": campaign["id"],
            "schema_version": campaign["schema_version"],
            "stage_1_study_id": campaign["stage_1_study_id"],
            "stage_1_evidence_digest": campaign["stage_1_evidence_digest"],
            "base_manifest": json.loads(str(campaign["base_manifest_json"])),
            "candidates": json.loads(str(campaign["candidates_json"])),
            "panel_input": json.loads(str(campaign["panel_input_json"])),
            "protocol_version": campaign["protocol_version"],
            "analysis_version": campaign["analysis_version"],
            "created_at": _pydantic_json_datetime(campaign["created_at"]),
        }
        evidence = {
            "schema_version": "balanced-validation-evidence.v2",
            "campaign": campaign_payload,
            "panel": {
                "campaign_id": campaign_id,
                "schema_version": "validation-panel.v1",
                "trials": trial_payloads,
                "content_digest": trials[0]["panel_digest"],
            },
            "events": [
                {
                    "campaign_id": item["campaign_id"],
                    "sequence": item["sequence"],
                    "previous_state": item["previous_state"],
                    "new_state": item["new_state"],
                    "created_at": _pydantic_json_datetime(item["created_at"]),
                }
                for item in events
            ],
            "judgments": [
                {
                    "campaign_id": item["campaign_id"],
                    "trial_id": item["trial_id"],
                    "ordinal": item["ordinal"],
                    "archetype": item["archetype"],
                    "challenger": item["challenger"],
                    "seed": item["seed"],
                    "hidden_repeat": bool(item["hidden_repeat"]),
                    "winner_display_position": item["winner_display_position"],
                    "choice": item["choice"],
                    "created_at": _pydantic_json_datetime(item["created_at"]),
                }
                for item in judgments
            ],
            **provenance,
            "provenance_digest": "sha256:"
            + hashlib.sha256(canonical_json(provenance).encode()).hexdigest(),
            "result": {
                "campaign_id": result["campaign_id"],
                "disposition": result["disposition"],
                "reason": result["reason"],
                "challengers": json.loads(str(result["challengers_json"])),
                "archetype_reversals": json.loads(str(result["archetype_reversals_json"])),
                "repeat_agreements": result["repeat_agreements"],
            },
        }
        return "sha256:" + hashlib.sha256(canonical_json(evidence).encode()).hexdigest()
    except (sqlite3.DatabaseError, TypeError, ValueError, KeyError):
        return None


def _pydantic_json_datetime(value: object) -> str:
    text = str(value)
    return text[:-6] + "Z" if text.endswith("+00:00") else text


_MIGRATION_1 = """
CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE experiments(
  id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL,
  variable_path TEXT NOT NULL,
  changed_paths_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE variants(
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  label TEXT NOT NULL CHECK(label IN ('A','B')),
  manifest_json TEXT NOT NULL,
  UNIQUE(experiment_id, label)
);
CREATE TABLE jobs(
  id TEXT PRIMARY KEY,
  variant_id TEXT NOT NULL UNIQUE REFERENCES variants(id),
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  error_code TEXT,
  error_message TEXT
);
"""

_MIGRATION_2 = """
CREATE TABLE artifacts(
  id TEXT PRIMARY KEY,
  job_id TEXT REFERENCES jobs(id),
  sha256 TEXT NOT NULL,
  byte_size INTEGER NOT NULL,
  media_type TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(sha256, media_type)
);
"""

_MIGRATION_3 = """
CREATE TABLE job_artifacts(
  job_id TEXT NOT NULL REFERENCES jobs(id),
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  created_at TEXT NOT NULL,
  PRIMARY KEY(job_id, artifact_id)
);
INSERT INTO job_artifacts(job_id, artifact_id, created_at)
SELECT job_id, id, created_at FROM artifacts WHERE job_id IS NOT NULL;
"""

_MIGRATION_4 = """
CREATE TABLE run_attempts(
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL REFERENCES jobs(id),
  number INTEGER NOT NULL CHECK(number > 0),
  worker_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('active','completed','failed','cancelled','interrupted')),
  started_at TEXT NOT NULL,
  finished_at TEXT,
  error_code TEXT,
  error_message TEXT,
  artifact_id TEXT REFERENCES artifacts(id),
  UNIQUE(job_id, number)
);
CREATE TABLE run_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL REFERENCES jobs(id),
  attempt_id TEXT NOT NULL REFERENCES run_attempts(id),
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE telemetry_samples(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  attempt_id TEXT NOT NULL REFERENCES run_attempts(id),
  recorded_at TEXT NOT NULL,
  elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds >= 0),
  process_rss_bytes INTEGER NOT NULL CHECK(process_rss_bytes >= 0),
  system_used_bytes INTEGER NOT NULL CHECK(system_used_bytes >= 0),
  system_available_bytes INTEGER NOT NULL CHECK(system_available_bytes >= 0),
  swap_used_bytes INTEGER NOT NULL CHECK(swap_used_bytes >= 0),
  thermal_pressure TEXT
);
"""

_MIGRATION_5 = """
ALTER TABLE jobs ADD COLUMN claimed_by TEXT;
ALTER TABLE jobs ADD COLUMN claimed_at TEXT;
"""

_MIGRATION_6 = """
CREATE TABLE review_sessions(
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  reviewer_id TEXT NOT NULL,
  left_variant_id TEXT NOT NULL REFERENCES variants(id),
  right_variant_id TEXT NOT NULL REFERENCES variants(id),
  left_job_id TEXT NOT NULL REFERENCES jobs(id),
  left_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  left_artifact_sha256 TEXT NOT NULL,
  right_job_id TEXT NOT NULL REFERENCES jobs(id),
  right_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  right_artifact_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  CHECK(left_variant_id <> right_variant_id),
  CHECK(left_job_id <> right_job_id),
  CHECK(left_artifact_id <> right_artifact_id),
  CHECK(left_artifact_sha256 <> right_artifact_sha256),
  UNIQUE(experiment_id, reviewer_id)
);
CREATE TABLE review_revisions(
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES review_sessions(id),
  preferred_variant_id TEXT REFERENCES variants(id),
  display_preference TEXT NOT NULL CHECK(display_preference IN ('left','right','tie')),
  notes TEXT,
  created_at TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK(revision > 0),
  supersedes_review_id TEXT,
  UNIQUE(session_id, revision),
  UNIQUE(id, session_id),
  FOREIGN KEY(supersedes_review_id, session_id)
    REFERENCES review_revisions(id, session_id)
);
CREATE UNIQUE INDEX one_initial_review_per_session
  ON review_revisions(session_id) WHERE supersedes_review_id IS NULL;
CREATE UNIQUE INDEX one_successor_per_review
  ON review_revisions(session_id, supersedes_review_id)
  WHERE supersedes_review_id IS NOT NULL;
CREATE TABLE rubric_scores(
  review_id TEXT NOT NULL REFERENCES review_revisions(id),
  display_position TEXT NOT NULL CHECK(display_position IN ('left','right')),
  dimension TEXT NOT NULL CHECK(dimension IN (
    'prompt_adherence','visual_quality','motion_quality',
    'temporal_consistency','character_consistency'
  )),
  score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
  UNIQUE(review_id, display_position, dimension)
);
"""

_MIGRATION_7 = """
CREATE TABLE feasibility_runs(
  id TEXT PRIMARY KEY,
  adapter_id TEXT NOT NULL,
  protocol_version TEXT NOT NULL,
  input_fingerprint TEXT NOT NULL CHECK(
    length(input_fingerprint) = 64 AND input_fingerprint = lower(input_fingerprint)
    AND input_fingerprint NOT GLOB '*[^0-9a-f]*'
  ),
  configuration_json TEXT NOT NULL CHECK(
    json_valid(configuration_json) AND json_type(configuration_json) = 'object'
  ),
  verdict TEXT NOT NULL CHECK(verdict IN (
    'pending','incompatible','resource_limited',
    'experimental_local_hybrid','standard_lab_qualified'
  )),
  created_at TEXT NOT NULL CHECK(substr(created_at, -6) = '+00:00'),
  updated_at TEXT NOT NULL CHECK(substr(updated_at, -6) = '+00:00')
);
CREATE TABLE feasibility_stage_attempts(
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES feasibility_runs(id),
  stage TEXT NOT NULL CHECK(stage IN (
    'resolve_sources','inventory_sources','storage_preflight','download_sources',
    'load_text_stack','load_vae','load_high_transformer','load_low_transformer',
    'assemble_pipeline','generate_minimum_1','generate_minimum_2','qualify_standard',
    'validate_controlled_pair','publish_verdict'
  )),
  ordinal INTEGER NOT NULL CHECK(ordinal > 0),
  input_fingerprint TEXT NOT NULL CHECK(
    length(input_fingerprint) = 64 AND input_fingerprint = lower(input_fingerprint)
    AND input_fingerprint NOT GLOB '*[^0-9a-f]*'
  ),
  outcome TEXT CHECK(outcome IN (
    'passed','incompatible','resource_exhausted','interrupted',
    'invalid_evidence','unexpected_failure'
  )),
  started_at TEXT NOT NULL CHECK(substr(started_at, -6) = '+00:00'),
  finished_at TEXT CHECK(finished_at IS NULL OR substr(finished_at, -6) = '+00:00'),
  evidence_json TEXT CHECK(
    evidence_json IS NULL OR (json_valid(evidence_json) AND json_type(evidence_json) = 'object')
  ),
  diagnostic_code TEXT CHECK(
    diagnostic_code IS NULL OR length(diagnostic_code) BETWEEN 1 AND 64
  ),
  diagnostic_message TEXT CHECK(
    diagnostic_message IS NULL OR length(diagnostic_message) BETWEEN 1 AND 1000
  ),
  UNIQUE(run_id, stage, ordinal),
  CHECK(
    (outcome IS NULL AND finished_at IS NULL AND evidence_json IS NULL
      AND diagnostic_code IS NULL AND diagnostic_message IS NULL)
    OR (outcome IS NOT NULL AND finished_at IS NOT NULL)
  )
);
CREATE UNIQUE INDEX one_active_feasibility_stage_attempt
  ON feasibility_stage_attempts(run_id, stage) WHERE outcome IS NULL;
CREATE INDEX feasibility_stage_attempt_order
  ON feasibility_stage_attempts(run_id, started_at, stage, ordinal, id);
"""

_MIGRATION_8 = """
CREATE TRIGGER guard_feasibility_stage_attempt_updates
BEFORE UPDATE ON feasibility_stage_attempts
BEGIN
  SELECT CASE
    WHEN OLD.outcome IS NOT NULL THEN
      RAISE(ABORT, 'finished feasibility stage attempts are immutable')
    WHEN NOT (
      NEW.id IS OLD.id
      AND NEW.run_id IS OLD.run_id
      AND NEW.stage IS OLD.stage
      AND NEW.ordinal IS OLD.ordinal
      AND NEW.input_fingerprint IS OLD.input_fingerprint
      AND NEW.started_at IS OLD.started_at
      AND NEW.outcome IS NOT NULL
      AND NEW.finished_at IS NOT NULL
      AND NEW.evidence_json IS NOT NULL
    ) THEN
      RAISE(ABORT, 'invalid feasibility stage terminal transition')
  END;
END;
"""

_MIGRATION_9 = """
CREATE TABLE adaptive_studies(
  id TEXT PRIMARY KEY CHECK(length(id) > 0),
  schema_version TEXT NOT NULL CHECK(schema_version = 'adaptive-study.v1'),
  prompt TEXT NOT NULL CHECK(length(prompt) > 0),
  negative_prompt TEXT,
  seed INTEGER NOT NULL CHECK(typeof(seed) = 'integer'),
  base_manifest_json TEXT NOT NULL CHECK(
    COALESCE(
      json_valid(base_manifest_json)
      AND json_type(base_manifest_json) = 'object'
      AND json_extract(base_manifest_json, '$.schema_version') = 'manifest.v1'
      AND json_extract(base_manifest_json, '$.prompt') = prompt
      AND json_extract(base_manifest_json, '$.negative_prompt') = COALESCE(negative_prompt, '')
      AND json_extract(base_manifest_json, '$.seed') = seed,
      0
    )
  ),
  optimizer_config_json TEXT NOT NULL CHECK(
    COALESCE(
      json_valid(optimizer_config_json)
      AND json_type(optimizer_config_json) = 'object'
      AND json_extract(optimizer_config_json, '$.schema_version')
        = 'preference-optimizer-config.v1'
      AND json_extract(optimizer_config_json, '$.optimizer_version')
        = 'preference-gp-laplace.v1',
      0
    )
  ),
  input_fingerprint TEXT NOT NULL CHECK(
    length(input_fingerprint) = 64
    AND input_fingerprint = lower(input_fingerprint)
    AND input_fingerprint NOT GLOB '*[^0-9a-f]*'
  ),
  created_at TEXT NOT NULL CHECK(
    julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'
  )
);
CREATE TRIGGER validate_adaptive_study_definition_insert
BEFORE INSERT ON adaptive_studies BEGIN
  SELECT CASE WHEN adaptive_manifest_json_is_valid(NEW.base_manifest_json) <> 1
    THEN RAISE(ABORT, 'invalid adaptive study manifest') END;
  SELECT CASE WHEN adaptive_optimizer_config_json_is_valid(NEW.optimizer_config_json) <> 1
    THEN RAISE(ABORT, 'invalid adaptive optimizer config') END;
END;
CREATE TABLE adaptive_study_events(
  study_id TEXT NOT NULL REFERENCES adaptive_studies(id),
  schema_version TEXT NOT NULL CHECK(schema_version = 'adaptive-study-event.v1'),
  sequence INTEGER NOT NULL CHECK(typeof(sequence) = 'integer' AND sequence > 0),
  previous_state TEXT NOT NULL CHECK(previous_state IN (
    'draft','search_ready','generating_pair','awaiting_preference','updating_posterior',
    'search_converged','search_budget_exhausted','validating','completed','paused',
    'optimizer_failed'
  )),
  new_state TEXT NOT NULL CHECK(new_state IN (
    'draft','search_ready','generating_pair','awaiting_preference','updating_posterior',
    'search_converged','search_budget_exhausted','validating','completed','paused',
    'optimizer_failed'
  )),
  payload_json TEXT NOT NULL CHECK(
    json_valid(payload_json) AND json_type(payload_json) = 'object'
  ),
  created_at TEXT NOT NULL CHECK(
    julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'
  ),
  PRIMARY KEY(study_id, sequence),
  CHECK(previous_state <> new_state)
);
CREATE TABLE adaptive_optimizer_steps(
  study_id TEXT NOT NULL REFERENCES adaptive_studies(id),
  schema_version TEXT NOT NULL CHECK(schema_version = 'adaptive-optimizer-step.v1'),
  ordinal INTEGER NOT NULL CHECK(typeof(ordinal) = 'integer' AND ordinal > 0),
  input_digest TEXT NOT NULL CHECK(
    length(input_digest) = 64
    AND input_digest = lower(input_digest)
    AND input_digest NOT GLOB '*[^0-9a-f]*'
  ),
  output_digest TEXT NOT NULL CHECK(
    length(output_digest) = 64
    AND output_digest = lower(output_digest)
    AND output_digest NOT GLOB '*[^0-9a-f]*'
  ),
  posterior_json TEXT NOT NULL CHECK(
    COALESCE(
      json_valid(posterior_json)
      AND json_type(posterior_json) = 'object'
      AND json_type(posterior_json, '$.grid') = 'array'
      AND json_array_length(posterior_json, '$.grid') > 0
      AND json_type(posterior_json, '$.mean') = 'array'
      AND json_array_length(posterior_json, '$.mean')
        = json_array_length(posterior_json, '$.grid')
      AND json_type(posterior_json, '$.covariance') = 'array'
      AND json_array_length(posterior_json, '$.covariance')
        = json_array_length(posterior_json, '$.grid')
      AND json_type(posterior_json, '$.argmax_probability') = 'array'
      AND json_array_length(posterior_json, '$.argmax_probability')
        = json_array_length(posterior_json, '$.grid')
      AND json_type(posterior_json, '$.recommendation') IN ('integer', 'real')
      AND json_type(posterior_json, '$.concentration_interval') = 'array'
      AND json_array_length(posterior_json, '$.concentration_interval') = 2
      AND json_type(posterior_json, '$.concentration_probability') IN ('integer', 'real')
      AND json_type(posterior_json, '$.content_digest') = 'text'
      AND length(json_extract(posterior_json, '$.content_digest')) = 71
      AND substr(json_extract(posterior_json, '$.content_digest'), 1, 7) = 'sha256:'
      AND substr(json_extract(posterior_json, '$.content_digest'), 8)
        = lower(substr(json_extract(posterior_json, '$.content_digest'), 8))
      AND substr(json_extract(posterior_json, '$.content_digest'), 8)
        NOT GLOB '*[^0-9a-f]*',
      0
    )
  ),
  acquisition_ledger_json TEXT NOT NULL CHECK(
    json_valid(acquisition_ledger_json) AND json_type(acquisition_ledger_json) = 'array'
  ),
  selected_pair_json TEXT CHECK(
    selected_pair_json IS NULL
    OR (json_valid(selected_pair_json) AND json_type(selected_pair_json) = 'object')
  ),
  stop_reason TEXT NOT NULL CHECK(stop_reason IN (
    'continue','search_converged','search_budget_exhausted','candidate_space_exhausted'
  )),
  disposition TEXT CHECK(disposition IS NULL OR disposition = 'inconclusive_within_budget'),
  comparison_count INTEGER NOT NULL CHECK(
    typeof(comparison_count) = 'integer' AND comparison_count >= 0
  ),
  maximum_information_bits REAL NOT NULL CHECK(
    typeof(maximum_information_bits) = 'real'
    AND maximum_information_bits >= 0
    AND maximum_information_bits <= 1.0e308
  ),
  created_at TEXT NOT NULL CHECK(
    julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'
  ),
  PRIMARY KEY(study_id, ordinal),
  CHECK(ordinal = comparison_count + 1),
  CHECK(COALESCE(
    (stop_reason = 'search_budget_exhausted' AND disposition = 'inconclusive_within_budget')
    OR (stop_reason <> 'search_budget_exhausted' AND disposition IS NULL),
    0
  )),
  CHECK(
    (stop_reason = 'continue' AND selected_pair_json IS NOT NULL)
    OR (stop_reason <> 'continue' AND selected_pair_json IS NULL)
  )
);
CREATE TABLE adaptive_observations(
  study_id TEXT NOT NULL REFERENCES adaptive_studies(id),
  schema_version TEXT NOT NULL CHECK(schema_version = 'adaptive-observation.v1'),
  ordinal INTEGER NOT NULL CHECK(typeof(ordinal) = 'integer' AND ordinal > 0),
  optimizer_step_ordinal INTEGER NOT NULL CHECK(
    typeof(optimizer_step_ordinal) = 'integer'
    AND optimizer_step_ordinal = ordinal
  ),
  value_a REAL NOT NULL CHECK(typeof(value_a) = 'real' AND abs(value_a) <= 1.0e308),
  value_b REAL NOT NULL CHECK(typeof(value_b) = 'real' AND abs(value_b) <= 1.0e308),
  experiment_id TEXT NOT NULL UNIQUE REFERENCES experiments(id),
  review_id TEXT NOT NULL UNIQUE REFERENCES review_revisions(id),
  display_mapping_digest TEXT NOT NULL CHECK(
    length(display_mapping_digest) = 64
    AND display_mapping_digest = lower(display_mapping_digest)
    AND display_mapping_digest NOT GLOB '*[^0-9a-f]*'
  ),
  outcome TEXT NOT NULL CHECK(outcome IN ('a','b','no_perceptible_difference')),
  created_at TEXT NOT NULL CHECK(
    julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'
  ),
  PRIMARY KEY(study_id, ordinal),
  UNIQUE(study_id, optimizer_step_ordinal),
  FOREIGN KEY(study_id, optimizer_step_ordinal)
    REFERENCES adaptive_optimizer_steps(study_id, ordinal),
  CHECK(value_a < value_b)
);
CREATE TRIGGER validate_adaptive_study_event_insert
BEFORE INSERT ON adaptive_study_events BEGIN
  SELECT CASE WHEN NOT (
    (NEW.previous_state = 'draft' AND NEW.new_state = 'search_ready')
    OR (NEW.previous_state = 'search_ready' AND NEW.new_state IN (
      'generating_pair','paused','optimizer_failed'
    ))
    OR (NEW.previous_state = 'generating_pair' AND NEW.new_state IN (
      'awaiting_preference','paused'
    ))
    OR (NEW.previous_state = 'awaiting_preference' AND NEW.new_state IN (
      'updating_posterior','paused'
    ))
    OR (NEW.previous_state = 'updating_posterior' AND NEW.new_state IN (
      'generating_pair','search_converged','search_budget_exhausted','paused',
      'optimizer_failed'
    ))
    OR (NEW.previous_state IN ('search_converged','search_budget_exhausted')
      AND NEW.new_state = 'validating')
    OR (NEW.previous_state = 'validating' AND NEW.new_state IN ('completed','paused'))
    OR (NEW.previous_state = 'paused' AND NEW.new_state IN (
      'search_ready','generating_pair','awaiting_preference','updating_posterior','validating'
    ))
    OR (NEW.previous_state = 'optimizer_failed' AND NEW.new_state IN (
      'search_ready','updating_posterior','paused'
    ))
  ) THEN RAISE(ABORT, 'invalid adaptive study transition') END;
  SELECT CASE WHEN NEW.sequence <> COALESCE((
    SELECT MAX(sequence) + 1 FROM adaptive_study_events WHERE study_id = NEW.study_id
  ), 1) THEN RAISE(ABORT, 'invalid adaptive study event sequence') END;
  SELECT CASE WHEN NEW.previous_state <> COALESCE((
    SELECT new_state FROM adaptive_study_events WHERE study_id = NEW.study_id
    ORDER BY sequence DESC LIMIT 1
  ), 'draft') THEN RAISE(ABORT, 'invalid adaptive study prior state') END;
END;
CREATE TRIGGER validate_adaptive_optimizer_step_insert
BEFORE INSERT ON adaptive_optimizer_steps BEGIN
  SELECT CASE WHEN NEW.ordinal <> COALESCE((
    SELECT MAX(ordinal) + 1 FROM adaptive_optimizer_steps WHERE study_id = NEW.study_id
  ), 1) THEN RAISE(ABORT, 'invalid adaptive optimizer step ordinal') END;
  SELECT CASE WHEN NEW.comparison_count <> (
    SELECT COUNT(*) FROM adaptive_observations WHERE study_id = NEW.study_id
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer comparison count') END;
  SELECT CASE WHEN NOT (
    (NEW.ordinal = 1 AND COALESCE((
      SELECT new_state FROM adaptive_study_events WHERE study_id = NEW.study_id
      ORDER BY sequence DESC LIMIT 1
    ), 'draft') = 'search_ready')
    OR (NEW.ordinal > 1 AND COALESCE((
      SELECT new_state FROM adaptive_study_events WHERE study_id = NEW.study_id
      ORDER BY sequence DESC LIMIT 1
    ), 'draft') = 'updating_posterior')
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer state') END;
  SELECT CASE WHEN (SELECT COUNT(*) FROM json_each(NEW.posterior_json)) <> 8
    THEN RAISE(ABORT, 'invalid adaptive optimizer posterior object') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM json_each(NEW.posterior_json, '$.grid')
    WHERE type NOT IN ('integer','real') OR abs(CAST(value AS REAL)) > 1.0e308
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer posterior grid') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM json_each(NEW.posterior_json, '$.mean')
    WHERE type NOT IN ('integer','real') OR abs(CAST(value AS REAL)) > 1.0e308
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer posterior mean') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM json_each(NEW.posterior_json, '$.argmax_probability')
    WHERE type NOT IN ('integer','real') OR abs(CAST(value AS REAL)) > 1.0e308
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer argmax probability') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM json_each(NEW.posterior_json, '$.concentration_interval')
    WHERE type NOT IN ('integer','real') OR abs(CAST(value AS REAL)) > 1.0e308
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer concentration interval') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM json_each(NEW.posterior_json, '$.covariance') AS covariance_row
    WHERE covariance_row.type <> 'array'
      OR json_array_length(covariance_row.value)
        <> json_array_length(NEW.posterior_json, '$.grid')
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer covariance shape') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1
    FROM json_each(NEW.posterior_json, '$.covariance') AS covariance_row,
      json_each(covariance_row.value) AS covariance_value
    WHERE covariance_value.type NOT IN ('integer','real')
      OR abs(CAST(covariance_value.value AS REAL)) > 1.0e308
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer covariance value') END;
  SELECT CASE WHEN NOT (
    json_type(NEW.posterior_json, '$.recommendation') IN ('integer','real')
    AND abs(CAST(json_extract(NEW.posterior_json, '$.recommendation') AS REAL)) <= 1.0e308
    AND json_type(NEW.posterior_json, '$.concentration_probability') IN ('integer','real')
    AND abs(CAST(
      json_extract(NEW.posterior_json, '$.concentration_probability') AS REAL
    )) <= 1.0e308
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer posterior scalar') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM json_each(NEW.acquisition_ledger_json) AS score
    WHERE score.type <> 'object'
      OR (SELECT COUNT(*) FROM json_each(score.value)) <> 2
      OR json_type(score.value, '$.pair') <> 'object'
      OR (SELECT COUNT(*) FROM json_each(json_extract(score.value, '$.pair'))) <> 2
      OR json_type(score.value, '$.pair.value_a') NOT IN ('integer','real')
      OR json_type(score.value, '$.pair.value_b') NOT IN ('integer','real')
      OR abs(CAST(json_extract(score.value, '$.pair.value_a') AS REAL)) > 1.0e308
      OR abs(CAST(json_extract(score.value, '$.pair.value_b') AS REAL)) > 1.0e308
      OR CAST(json_extract(score.value, '$.pair.value_a') AS REAL)
        >= CAST(json_extract(score.value, '$.pair.value_b') AS REAL)
      OR json_type(score.value, '$.expected_information_bits') NOT IN ('integer','real')
      OR CAST(json_extract(score.value, '$.expected_information_bits') AS REAL) < 0
      OR abs(CAST(
        json_extract(score.value, '$.expected_information_bits') AS REAL
      )) > 1.0e308
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer acquisition ledger') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1
    FROM json_each(NEW.acquisition_ledger_json) AS score
    GROUP BY
      json_extract(score.value, '$.pair.value_a'),
      json_extract(score.value, '$.pair.value_b')
    HAVING COUNT(*) > 1
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer duplicate acquisition pair') END;
  SELECT CASE WHEN NEW.selected_pair_json IS NOT NULL AND NOT (
    (SELECT COUNT(*) FROM json_each(NEW.selected_pair_json)) = 2
    AND json_type(NEW.selected_pair_json, '$.value_a') IN ('integer','real')
    AND json_type(NEW.selected_pair_json, '$.value_b') IN ('integer','real')
    AND abs(CAST(json_extract(NEW.selected_pair_json, '$.value_a') AS REAL)) <= 1.0e308
    AND abs(CAST(json_extract(NEW.selected_pair_json, '$.value_b') AS REAL)) <= 1.0e308
    AND CAST(json_extract(NEW.selected_pair_json, '$.value_a') AS REAL)
      < CAST(json_extract(NEW.selected_pair_json, '$.value_b') AS REAL)
  ) THEN RAISE(ABORT, 'invalid adaptive optimizer selected pair') END;
END;
CREATE TRIGGER validate_adaptive_observation_insert
BEFORE INSERT ON adaptive_observations BEGIN
  SELECT CASE WHEN COALESCE((
    SELECT new_state FROM adaptive_study_events WHERE study_id = NEW.study_id
    ORDER BY sequence DESC LIMIT 1
  ), 'draft') <> 'awaiting_preference'
    THEN RAISE(ABORT, 'invalid adaptive observation state') END;
  SELECT CASE WHEN NEW.ordinal <> COALESCE((
    SELECT MAX(ordinal) + 1 FROM adaptive_observations WHERE study_id = NEW.study_id
  ), 1) THEN RAISE(ABORT, 'invalid adaptive observation ordinal') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM adaptive_optimizer_steps AS optimizer
    WHERE optimizer.study_id = NEW.study_id
      AND optimizer.ordinal = NEW.optimizer_step_ordinal
      AND optimizer.comparison_count = NEW.ordinal - 1
      AND optimizer.stop_reason = 'continue'
      AND CAST(json_extract(optimizer.selected_pair_json, '$.value_a') AS REAL) = NEW.value_a
      AND CAST(json_extract(optimizer.selected_pair_json, '$.value_b') AS REAL) = NEW.value_b
  ) THEN RAISE(ABORT, 'invalid adaptive observation optimizer binding') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1
    FROM review_revisions AS review
    JOIN review_sessions AS session ON session.id = review.session_id
    JOIN experiments AS experiment ON experiment.id = session.experiment_id
    JOIN variants AS left_variant ON left_variant.id = session.left_variant_id
    JOIN variants AS right_variant ON right_variant.id = session.right_variant_id
    JOIN jobs AS left_job ON left_job.id = session.left_job_id
    JOIN jobs AS right_job ON right_job.id = session.right_job_id
    JOIN artifacts AS left_artifact ON left_artifact.id = session.left_artifact_id
    JOIN artifacts AS right_artifact ON right_artifact.id = session.right_artifact_id
    JOIN job_artifacts AS left_link ON left_link.job_id = session.left_job_id
      AND left_link.artifact_id = session.left_artifact_id
    JOIN job_artifacts AS right_link ON right_link.job_id = session.right_job_id
      AND right_link.artifact_id = session.right_artifact_id
    JOIN adaptive_studies AS study ON study.id = NEW.study_id
    WHERE review.id = NEW.review_id
      AND NEW.experiment_id = session.experiment_id
      AND review.id = (
        SELECT latest.id FROM review_revisions AS latest
        WHERE latest.session_id = session.id ORDER BY latest.revision DESC LIMIT 1
      )
      AND left_variant.experiment_id = session.experiment_id
      AND right_variant.experiment_id = session.experiment_id
      AND left_job.variant_id = session.left_variant_id
      AND right_job.variant_id = session.right_variant_id
      AND left_job.state = 'completed'
      AND right_job.state = 'completed'
      AND left_artifact.sha256 = session.left_artifact_sha256
      AND right_artifact.sha256 = session.right_artifact_sha256
      AND experiment.schema_version = 'manifest.v1'
      AND experiment.variable_path = 'guidance_scale'
      AND experiment.changed_paths_json = '["guidance_scale"]'
      AND json_remove(left_variant.manifest_json, '$.guidance_scale')
        = json_remove(study.base_manifest_json, '$.guidance_scale')
      AND json_remove(right_variant.manifest_json, '$.guidance_scale')
        = json_remove(study.base_manifest_json, '$.guidance_scale')
      AND min(
        CAST(json_extract(left_variant.manifest_json, '$.guidance_scale') AS REAL),
        CAST(json_extract(right_variant.manifest_json, '$.guidance_scale') AS REAL)
      ) = NEW.value_a
      AND max(
        CAST(json_extract(left_variant.manifest_json, '$.guidance_scale') AS REAL),
        CAST(json_extract(right_variant.manifest_json, '$.guidance_scale') AS REAL)
      ) = NEW.value_b
      AND (
        (review.display_preference = 'tie'
          AND review.preferred_variant_id IS NULL
          AND NEW.outcome = 'no_perceptible_difference')
        OR (review.display_preference = 'left'
          AND review.preferred_variant_id = session.left_variant_id
          AND NEW.outcome = CASE
            WHEN CAST(json_extract(
              left_variant.manifest_json, '$.guidance_scale'
            ) AS REAL) = NEW.value_a THEN 'a' ELSE 'b' END)
        OR (review.display_preference = 'right'
          AND review.preferred_variant_id = session.right_variant_id
          AND NEW.outcome = CASE
            WHEN CAST(json_extract(
              right_variant.manifest_json, '$.guidance_scale'
            ) AS REAL) = NEW.value_a THEN 'a' ELSE 'b' END)
      )
  ) THEN RAISE(ABORT, 'invalid adaptive observation review binding') END;
  SELECT CASE WHEN NEW.display_mapping_digest IS NOT adaptive_observation_provenance_digest(
    NEW.study_id,
    NEW.schema_version,
    NEW.ordinal,
    NEW.optimizer_step_ordinal,
    NEW.value_a,
    NEW.value_b,
    NEW.experiment_id,
    NEW.review_id,
    NEW.outcome,
    NEW.created_at
  ) THEN RAISE(ABORT, 'invalid adaptive observation provenance digest') END;
END;
CREATE TRIGGER guard_adaptive_studies_update
BEFORE UPDATE ON adaptive_studies BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
CREATE TRIGGER guard_adaptive_studies_delete
BEFORE DELETE ON adaptive_studies BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
CREATE TRIGGER guard_adaptive_study_events_update
BEFORE UPDATE ON adaptive_study_events BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
CREATE TRIGGER guard_adaptive_study_events_delete
BEFORE DELETE ON adaptive_study_events BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
CREATE TRIGGER guard_adaptive_observations_update
BEFORE UPDATE ON adaptive_observations BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
CREATE TRIGGER guard_adaptive_observations_delete
BEFORE DELETE ON adaptive_observations BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
CREATE TRIGGER guard_adaptive_optimizer_steps_update
BEFORE UPDATE ON adaptive_optimizer_steps BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
CREATE TRIGGER guard_adaptive_optimizer_steps_delete
BEFORE DELETE ON adaptive_optimizer_steps BEGIN
  SELECT RAISE(ABORT, 'adaptive study evidence is append-only');
END;
"""

_MIGRATION_10 = """
CREATE TABLE adaptive_web_creation_grants(
  token_digest TEXT NOT NULL CHECK(
    length(token_digest) = 64
    AND token_digest = lower(token_digest)
    AND token_digest NOT GLOB '*[^0-9a-f]*'
  ),
  reviewer_id TEXT NOT NULL CHECK(length(reviewer_id) > 0 AND reviewer_id = trim(reviewer_id)),
  created_at TEXT NOT NULL CHECK(
    julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'
  ),
  expires_at TEXT NOT NULL CHECK(
    julianday(expires_at) IS NOT NULL
    AND substr(expires_at, -6) = '+00:00'
    AND julianday(expires_at) > julianday(created_at)
  ),
  input_fingerprint TEXT CHECK(
    input_fingerprint IS NULL OR (
      length(input_fingerprint) = 64
      AND input_fingerprint = lower(input_fingerprint)
      AND input_fingerprint NOT GLOB '*[^0-9a-f]*'
    )
  ),
  study_id TEXT CHECK(study_id IS NULL OR length(study_id) > 0),
  bound_at TEXT CHECK(
    bound_at IS NULL OR (
      julianday(bound_at) IS NOT NULL
      AND substr(bound_at, -6) = '+00:00'
      AND julianday(bound_at) >= julianday(created_at)
      AND julianday(bound_at) < julianday(expires_at)
    )
  ),
  PRIMARY KEY(token_digest, reviewer_id),
  CHECK(
    (input_fingerprint IS NULL AND study_id IS NULL AND bound_at IS NULL)
    OR (input_fingerprint IS NOT NULL AND study_id IS NOT NULL AND bound_at IS NOT NULL)
  )
);
CREATE INDEX adaptive_web_creation_grants_owner_expiry
ON adaptive_web_creation_grants(reviewer_id, expires_at);
CREATE TRIGGER validate_adaptive_web_creation_grant_update
BEFORE UPDATE ON adaptive_web_creation_grants BEGIN
  SELECT CASE WHEN NOT (
    OLD.input_fingerprint IS NULL
    AND OLD.study_id IS NULL
    AND OLD.bound_at IS NULL
    AND NEW.token_digest IS OLD.token_digest
    AND NEW.reviewer_id IS OLD.reviewer_id
    AND NEW.created_at IS OLD.created_at
    AND NEW.expires_at IS OLD.expires_at
    AND NEW.input_fingerprint IS NOT NULL
    AND NEW.study_id IS NOT NULL
    AND NEW.bound_at IS NOT NULL
  ) THEN RAISE(ABORT, 'invalid adaptive web creation grant transition') END;
END;
"""

_MIGRATION_11 = """
CREATE TABLE validation_campaigns(
  id TEXT PRIMARY KEY CHECK(length(id) > 0),
  schema_version TEXT NOT NULL CHECK(schema_version = 'validation-campaign.v1'),
  reviewer_id TEXT NOT NULL CHECK(length(reviewer_id) > 0 AND reviewer_id = trim(reviewer_id)),
  stage_1_study_id TEXT NOT NULL REFERENCES adaptive_studies(id) CHECK(length(stage_1_study_id) > 0),
  stage_1_evidence_digest TEXT NOT NULL CHECK(
    length(stage_1_evidence_digest) = 64
    AND stage_1_evidence_digest = lower(stage_1_evidence_digest)
    AND stage_1_evidence_digest NOT GLOB '*[^0-9a-f]*'
  ),
  base_manifest_json TEXT NOT NULL CHECK(json_valid(base_manifest_json) AND json_type(base_manifest_json) = 'object'),
  candidates_json TEXT NOT NULL CHECK(json_valid(candidates_json) AND json_type(candidates_json) = 'object'),
  panel_input_json TEXT NOT NULL CHECK(json_valid(panel_input_json) AND json_type(panel_input_json) = 'object'),
  protocol_version TEXT NOT NULL CHECK(protocol_version = 'balanced-validation.v1'),
  analysis_version TEXT NOT NULL CHECK(analysis_version = 'dirichlet-validation.v1'),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00')
);
CREATE TABLE validation_events(
  campaign_id TEXT NOT NULL REFERENCES validation_campaigns(id),
  sequence INTEGER NOT NULL CHECK(typeof(sequence) = 'integer' AND sequence > 0),
  previous_state TEXT NOT NULL CHECK(previous_state IN ('draft','preregistered','generating','ready','judging','analyzing','qualified','not_qualified','inconclusive')),
  new_state TEXT NOT NULL CHECK(new_state IN ('draft','preregistered','generating','ready','judging','analyzing','qualified','not_qualified','inconclusive')),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'),
  PRIMARY KEY(campaign_id, sequence),
  CHECK(previous_state <> new_state)
);
CREATE TABLE validation_trials(
  campaign_id TEXT NOT NULL REFERENCES validation_campaigns(id),
  trial_id TEXT NOT NULL CHECK(length(trial_id) > 0),
  ordinal INTEGER NOT NULL CHECK(typeof(ordinal) = 'integer' AND ordinal > 0),
  archetype TEXT NOT NULL CHECK(archetype IN ('portrait_motion','full_body_action','camera_motion','environment_interaction')),
  prompt TEXT NOT NULL CHECK(length(prompt) > 0),
  negative_prompt TEXT NOT NULL,
  seed INTEGER NOT NULL CHECK(typeof(seed) = 'integer'),
  challenger REAL NOT NULL CHECK(typeof(challenger) = 'real' AND abs(challenger) <= 1.0e308),
  hidden_repeat INTEGER NOT NULL CHECK(hidden_repeat IN (0, 1)),
  reuses_trial_id TEXT,
  winner_display_position TEXT NOT NULL CHECK(winner_display_position IN ('left', 'right')),
  panel_digest TEXT NOT NULL CHECK(
    length(panel_digest) = 71 AND substr(panel_digest, 1, 7) = 'sha256:'
    AND substr(panel_digest, 8) = lower(substr(panel_digest, 8))
    AND substr(panel_digest, 8) NOT GLOB '*[^0-9a-f]*'
  ),
  PRIMARY KEY(campaign_id, trial_id),
  UNIQUE(campaign_id, ordinal),
  FOREIGN KEY(campaign_id, reuses_trial_id) REFERENCES validation_trials(campaign_id, trial_id),
  CHECK((hidden_repeat = 1) = (reuses_trial_id IS NOT NULL))
);
CREATE UNIQUE INDEX validation_primary_context_challenger
ON validation_trials(campaign_id, archetype, prompt, negative_prompt, seed, challenger)
WHERE hidden_repeat = 0;
CREATE TABLE validation_trial_bindings(
  campaign_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  experiment_id TEXT,
  review_id TEXT,
  reused_from_trial_id TEXT,
  PRIMARY KEY(campaign_id, trial_id),
  FOREIGN KEY(campaign_id, trial_id) REFERENCES validation_trials(campaign_id, trial_id),
  FOREIGN KEY(campaign_id, reused_from_trial_id) REFERENCES validation_trials(campaign_id, trial_id),
  FOREIGN KEY(experiment_id) REFERENCES experiments(id),
  FOREIGN KEY(review_id) REFERENCES review_revisions(id),
  CHECK((experiment_id IS NULL) = (review_id IS NULL))
);
CREATE TABLE validation_judgments(
  campaign_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK(typeof(ordinal) = 'integer' AND ordinal > 0),
  archetype TEXT NOT NULL CHECK(archetype IN ('portrait_motion','full_body_action','camera_motion','environment_interaction')),
  challenger REAL NOT NULL CHECK(typeof(challenger) = 'real' AND abs(challenger) <= 1.0e308),
  seed INTEGER NOT NULL CHECK(typeof(seed) = 'integer'),
  hidden_repeat INTEGER NOT NULL CHECK(hidden_repeat IN (0, 1)),
  winner_display_position TEXT NOT NULL CHECK(winner_display_position IN ('left', 'right')),
  choice TEXT NOT NULL CHECK(choice IN ('a', 'b', 'tie')),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'),
  PRIMARY KEY(campaign_id, trial_id),
  UNIQUE(campaign_id, ordinal),
  FOREIGN KEY(campaign_id, trial_id) REFERENCES validation_trials(campaign_id, trial_id)
);
CREATE TABLE validation_results(
  campaign_id TEXT PRIMARY KEY REFERENCES validation_campaigns(id),
  disposition TEXT NOT NULL CHECK(disposition IN ('qualified','not_qualified','inconclusive')),
  reason TEXT NOT NULL CHECK(reason IN ('all_gates_passed','challenger_rejection','archetype_reversal','repeat_reliability_failed','incomplete_primary_evidence','insufficient_primary_evidence')),
  challengers_json TEXT NOT NULL CHECK(json_valid(challengers_json) AND json_type(challengers_json) = 'array'),
  archetype_reversals_json TEXT NOT NULL CHECK(json_valid(archetype_reversals_json) AND json_type(archetype_reversals_json) = 'array'),
  repeat_agreements INTEGER NOT NULL CHECK(typeof(repeat_agreements) = 'integer' AND repeat_agreements BETWEEN 0 AND 4),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00')
);
CREATE TRIGGER validate_validation_event_insert
BEFORE INSERT ON validation_events BEGIN
  SELECT CASE WHEN NEW.sequence <> COALESCE((SELECT MAX(sequence) + 1 FROM validation_events WHERE campaign_id = NEW.campaign_id), 1)
    THEN RAISE(ABORT, 'invalid validation event sequence') END;
  SELECT CASE WHEN NEW.previous_state <> COALESCE((SELECT new_state FROM validation_events WHERE campaign_id = NEW.campaign_id ORDER BY sequence DESC LIMIT 1), 'draft')
    THEN RAISE(ABORT, 'invalid validation event prior state') END;
  SELECT CASE WHEN NOT (
    (NEW.previous_state = 'draft' AND NEW.new_state = 'preregistered')
    OR (NEW.previous_state = 'preregistered' AND NEW.new_state = 'generating')
    OR (NEW.previous_state = 'generating' AND NEW.new_state = 'ready')
    OR (NEW.previous_state = 'ready' AND NEW.new_state = 'judging')
    OR (NEW.previous_state = 'judging' AND NEW.new_state = 'analyzing')
    OR (NEW.previous_state = 'analyzing' AND NEW.new_state IN ('qualified','not_qualified','inconclusive'))
  ) THEN RAISE(ABORT, 'invalid validation lifecycle transition') END;
END;
CREATE TRIGGER validate_validation_trial_insert
BEFORE INSERT ON validation_trials BEGIN
  SELECT CASE WHEN COALESCE((SELECT new_state FROM validation_events WHERE campaign_id = NEW.campaign_id ORDER BY sequence DESC LIMIT 1), 'draft') <> 'preregistered'
    THEN RAISE(ABORT, 'validation trials require preregistered campaign') END;
  SELECT CASE WHEN NEW.hidden_repeat = 1 AND NOT EXISTS(SELECT 1 FROM validation_trials WHERE campaign_id = NEW.campaign_id AND trial_id = NEW.reuses_trial_id)
    THEN RAISE(ABORT, 'validation repeat source is missing') END;
END;
CREATE TRIGGER validate_validation_trial_binding_insert
BEFORE INSERT ON validation_trial_bindings BEGIN
  SELECT CASE WHEN COALESCE((SELECT new_state FROM validation_events WHERE campaign_id = NEW.campaign_id ORDER BY sequence DESC LIMIT 1), 'draft') NOT IN ('generating','ready','judging')
    THEN RAISE(ABORT, 'validation trial binding has invalid lifecycle state') END;
  SELECT CASE WHEN NEW.review_id IS NOT NULL AND NOT EXISTS(
    SELECT 1 FROM review_revisions AS review JOIN review_sessions AS session ON session.id = review.session_id
    JOIN validation_campaigns AS campaign ON campaign.id = NEW.campaign_id
    WHERE review.id = NEW.review_id AND session.experiment_id = NEW.experiment_id AND session.reviewer_id = campaign.reviewer_id
  ) THEN RAISE(ABORT, 'validation trial review owner mismatch') END;
END;
CREATE TRIGGER validate_validation_judgment_insert
BEFORE INSERT ON validation_judgments BEGIN
  SELECT CASE WHEN COALESCE((SELECT new_state FROM validation_events WHERE campaign_id = NEW.campaign_id ORDER BY sequence DESC LIMIT 1), 'draft') <> 'judging'
    THEN RAISE(ABORT, 'validation judgment has invalid lifecycle state') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM validation_trials AS trial
    WHERE trial.campaign_id = NEW.campaign_id AND trial.trial_id = NEW.trial_id
      AND trial.ordinal = NEW.ordinal AND trial.archetype = NEW.archetype
      AND trial.challenger = NEW.challenger AND trial.seed = NEW.seed
      AND trial.hidden_repeat = NEW.hidden_repeat AND trial.winner_display_position = NEW.winner_display_position
  ) THEN RAISE(ABORT, 'validation judgment does not match trial') END;
END;
CREATE TRIGGER validate_validation_result_insert
BEFORE INSERT ON validation_results BEGIN
  SELECT CASE WHEN COALESCE((SELECT new_state FROM validation_events WHERE campaign_id = NEW.campaign_id ORDER BY sequence DESC LIMIT 1), 'draft') <> 'analyzing'
    THEN RAISE(ABORT, 'validation result has invalid lifecycle state') END;
END;
CREATE TRIGGER guard_validation_campaigns_update BEFORE UPDATE ON validation_campaigns BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_campaigns_delete BEFORE DELETE ON validation_campaigns BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_events_update BEFORE UPDATE ON validation_events BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_events_delete BEFORE DELETE ON validation_events BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_trials_update BEFORE UPDATE ON validation_trials BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_trials_delete BEFORE DELETE ON validation_trials BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_trial_bindings_update BEFORE UPDATE ON validation_trial_bindings BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_trial_bindings_delete BEFORE DELETE ON validation_trial_bindings BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_judgments_update BEFORE UPDATE ON validation_judgments BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_judgments_delete BEFORE DELETE ON validation_judgments BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_results_update BEFORE UPDATE ON validation_results BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_results_delete BEFORE DELETE ON validation_results BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
"""

_MIGRATION_12 = """
CREATE TABLE validation_experiment_bindings(
  campaign_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  PRIMARY KEY(campaign_id, trial_id),
  FOREIGN KEY(campaign_id, trial_id) REFERENCES validation_trials(campaign_id, trial_id)
);
CREATE UNIQUE INDEX validation_experiment_binding_experiment
ON validation_experiment_bindings(experiment_id);
CREATE TRIGGER validate_validation_experiment_binding_insert
BEFORE INSERT ON validation_experiment_bindings BEGIN
  SELECT CASE WHEN COALESCE((SELECT new_state FROM validation_events WHERE campaign_id = NEW.campaign_id ORDER BY sequence DESC LIMIT 1), 'draft') NOT IN ('generating','ready','judging')
    AND EXISTS(SELECT 1 FROM schema_migrations WHERE version = 12)
    THEN RAISE(ABORT, 'validation experiment binding has invalid lifecycle state') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM validation_trials AS trial
    WHERE trial.campaign_id = NEW.campaign_id AND trial.trial_id = NEW.trial_id
      AND trial.hidden_repeat = 0
  ) THEN RAISE(ABORT, 'validation experiment binding requires a primary trial') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM experiments AS experiment
    WHERE experiment.id = NEW.experiment_id
      AND experiment.schema_version = 'validation-controlled.v1'
  ) THEN RAISE(ABORT, 'validation experiment binding requires a controlled experiment') END;
END;
CREATE TRIGGER guard_validation_experiment_bindings_update BEFORE UPDATE ON validation_experiment_bindings BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
CREATE TRIGGER guard_validation_experiment_bindings_delete BEFORE DELETE ON validation_experiment_bindings BEGIN SELECT RAISE(ABORT, 'validation evidence is append-only'); END;
"""

_MIGRATION_13 = """
DROP TRIGGER validate_validation_trial_binding_insert;
CREATE TRIGGER validate_validation_trial_binding_insert
BEFORE INSERT ON validation_trial_bindings BEGIN
  SELECT CASE WHEN COALESCE((SELECT new_state FROM validation_events WHERE campaign_id = NEW.campaign_id ORDER BY sequence DESC LIMIT 1), 'draft') NOT IN ('generating','ready','judging')
    THEN RAISE(ABORT, 'validation trial binding has invalid lifecycle state') END;
  SELECT CASE WHEN NEW.review_id IS NOT NULL AND NOT EXISTS(
    SELECT 1 FROM review_revisions AS review JOIN review_sessions AS session ON session.id = review.session_id
    JOIN validation_campaigns AS campaign ON campaign.id = NEW.campaign_id
    JOIN validation_trials AS trial ON trial.campaign_id = NEW.campaign_id AND trial.trial_id = NEW.trial_id
    WHERE review.id = NEW.review_id AND session.experiment_id = NEW.experiment_id
      AND session.reviewer_id = validation_session_reviewer(
        NEW.campaign_id, NEW.trial_id, campaign.reviewer_id, trial.hidden_repeat
      )
  ) THEN RAISE(ABORT, 'validation trial review owner mismatch') END;
END;
"""

_MIGRATION_14 = """
CREATE TABLE characters(
  id TEXT PRIMARY KEY,
  reviewer_id TEXT NOT NULL,
  name TEXT NOT NULL CHECK(length(name) > 0 AND name = trim(name)),
  created_at TEXT NOT NULL,
  UNIQUE(id, reviewer_id),
  UNIQUE(reviewer_id, name)
);
CREATE TABLE character_packs(
  id TEXT PRIMARY KEY,
  character_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('draft','preregistered')),
  supersedes_pack_id TEXT,
  manifest_json TEXT,
  created_at TEXT NOT NULL,
  preregistered_at TEXT,
  UNIQUE(character_id, id),
  FOREIGN KEY(character_id, reviewer_id) REFERENCES characters(id, reviewer_id),
  FOREIGN KEY(character_id, supersedes_pack_id) REFERENCES character_packs(character_id, id),
  CHECK((state = 'draft' AND manifest_json IS NULL AND preregistered_at IS NULL)
     OR (state = 'preregistered' AND manifest_json IS NOT NULL AND preregistered_at IS NOT NULL))
);
CREATE TABLE character_pack_aliases(
  alias_id TEXT PRIMARY KEY,
  pack_id TEXT NOT NULL REFERENCES character_packs(id) ON UPDATE CASCADE
);
CREATE TABLE character_references(
  pack_id TEXT NOT NULL REFERENCES character_packs(id) ON UPDATE CASCADE,
  reference_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  source_name TEXT NOT NULL CHECK(length(source_name) > 0 AND substr(source_name, 1, 1) != '/'),
  image_sha256 TEXT NOT NULL CHECK(length(image_sha256) = 64 AND image_sha256 = lower(image_sha256)),
  media_type TEXT NOT NULL CHECK(media_type IN ('image/jpeg','image/png','image/webp')),
  dhash_hex TEXT NOT NULL CHECK(length(dhash_hex) = 16 AND dhash_hex = lower(dhash_hex)),
  disposition TEXT NOT NULL CHECK(disposition IN ('active','rejected')),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(pack_id, reference_id),
  UNIQUE(pack_id, ordinal),
  UNIQUE(pack_id, artifact_id)
);
CREATE TABLE character_annotations(
  pack_id TEXT NOT NULL,
  reference_id TEXT NOT NULL,
  framing TEXT NOT NULL CHECK(framing IN ('portrait','full_body')),
  head_angle TEXT NOT NULL CHECK(head_angle IN ('front','three_quarter','profile')),
  body_visibility TEXT NOT NULL CHECK(body_visibility IN ('clear_face','partial_body','full_body')),
  expression TEXT NOT NULL CHECK(expression IN ('neutral','smile','serious','angry','surprised')),
  lighting TEXT NOT NULL CHECK(lighting IN ('ordinary','directional')),
  occlusion TEXT NOT NULL CHECK(occlusion IN ('none','partial','heavy')),
  identity_confirmed INTEGER NOT NULL CHECK(identity_confirmed IN (0,1)),
  PRIMARY KEY(pack_id, reference_id),
  FOREIGN KEY(pack_id, reference_id) REFERENCES character_references(pack_id, reference_id) ON UPDATE CASCADE
);
CREATE TABLE character_pack_warnings(
  pack_id TEXT NOT NULL REFERENCES character_packs(id) ON UPDATE CASCADE,
  left_reference_id TEXT NOT NULL,
  right_reference_id TEXT NOT NULL,
  distance INTEGER NOT NULL CHECK(distance BETWEEN 0 AND 64),
  resolution TEXT NOT NULL CHECK(resolution IN ('unresolved','accepted','dismissed')),
  PRIMARY KEY(pack_id, left_reference_id, right_reference_id),
  FOREIGN KEY(pack_id, left_reference_id) REFERENCES character_references(pack_id, reference_id) ON UPDATE CASCADE,
  FOREIGN KEY(pack_id, right_reference_id) REFERENCES character_references(pack_id, reference_id) ON UPDATE CASCADE,
  CHECK(left_reference_id < right_reference_id)
);
CREATE TABLE character_pack_events(
  pack_id TEXT NOT NULL REFERENCES character_packs(id) ON UPDATE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  reviewer_id TEXT NOT NULL,
  event_type TEXT NOT NULL CHECK(event_type IN ('draft_created','preregistered')),
  payload_json TEXT NOT NULL CHECK(json_valid(payload_json) AND json_type(payload_json) = 'object'),
  created_at TEXT NOT NULL,
  PRIMARY KEY(pack_id, sequence)
);
CREATE TRIGGER validate_character_pack_event_insert BEFORE INSERT ON character_pack_events BEGIN
  SELECT CASE WHEN NEW.sequence != COALESCE((SELECT MAX(sequence) + 1 FROM character_pack_events WHERE pack_id = NEW.pack_id), 1)
    THEN RAISE(ABORT, 'character pack event sequence is not append-only') END;
  SELECT CASE WHEN NEW.reviewer_id != (SELECT reviewer_id FROM character_packs WHERE id = NEW.pack_id)
    THEN RAISE(ABORT, 'character pack event reviewer mismatch') END;
  SELECT CASE WHEN NOT (
    (NEW.sequence = 1 AND NEW.event_type = 'draft_created' AND (SELECT state FROM character_packs WHERE id = NEW.pack_id) = 'draft')
    OR (NEW.sequence = 2 AND NEW.event_type = 'preregistered' AND (SELECT state FROM character_packs WHERE id = NEW.pack_id) = 'preregistered')
  ) THEN RAISE(ABORT, 'character pack event does not match lifecycle') END;
END;
CREATE TRIGGER guard_character_frozen_reference_insert BEFORE INSERT ON character_references WHEN (SELECT state FROM character_packs WHERE id = NEW.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_frozen_annotation_insert BEFORE INSERT ON character_annotations WHEN (SELECT state FROM character_packs WHERE id = NEW.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_frozen_warning_insert BEFORE INSERT ON character_pack_warnings WHEN (SELECT state FROM character_packs WHERE id = NEW.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_characters_update BEFORE UPDATE ON characters BEGIN SELECT RAISE(ABORT, 'character identity is append-only'); END;
CREATE TRIGGER guard_characters_delete BEFORE DELETE ON characters BEGIN SELECT RAISE(ABORT, 'character identity is append-only'); END;
CREATE TRIGGER guard_character_pack_alias_update BEFORE UPDATE ON character_pack_aliases WHEN EXISTS(SELECT 1 FROM character_packs WHERE id = OLD.pack_id) BEGIN SELECT RAISE(ABORT, 'character pack alias is append-only'); END;
CREATE TRIGGER guard_character_pack_alias_delete BEFORE DELETE ON character_pack_aliases BEGIN SELECT RAISE(ABORT, 'character pack alias is append-only'); END;
CREATE TRIGGER guard_character_pack_frozen_update BEFORE UPDATE ON character_packs WHEN OLD.state = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER validate_character_pack_preregister_update BEFORE UPDATE ON character_packs WHEN OLD.state = 'draft' BEGIN
  SELECT CASE WHEN NEW.character_id != OLD.character_id OR NEW.reviewer_id != OLD.reviewer_id OR NEW.supersedes_pack_id IS NOT OLD.supersedes_pack_id OR NEW.created_at != OLD.created_at
    OR NOT ((NEW.state = 'preregistered' AND NEW.id = OLD.id) OR (NEW.state = 'draft' AND NEW.manifest_json IS NULL AND NEW.preregistered_at IS NULL))
    THEN RAISE(ABORT, 'invalid character pack transition') END;
END;
CREATE TRIGGER guard_character_pack_frozen_delete BEFORE DELETE ON character_packs WHEN OLD.state = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_reference_update BEFORE UPDATE ON character_references WHEN (SELECT state FROM character_packs WHERE id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_reference_delete BEFORE DELETE ON character_references WHEN (SELECT state FROM character_packs WHERE id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_annotation_update BEFORE UPDATE ON character_annotations WHEN (SELECT state FROM character_packs WHERE id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_annotation_delete BEFORE DELETE ON character_annotations WHEN (SELECT state FROM character_packs WHERE id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_warning_update BEFORE UPDATE ON character_pack_warnings WHEN
  (OLD.resolution != 'unresolved' AND (NEW.pack_id = OLD.pack_id OR NEW.left_reference_id != OLD.left_reference_id OR NEW.right_reference_id != OLD.right_reference_id OR NEW.distance != OLD.distance OR NEW.resolution != OLD.resolution))
  OR (SELECT state FROM character_packs WHERE id = OLD.pack_id) = 'preregistered'
  BEGIN SELECT RAISE(ABORT, 'character warning is append-only'); END;
CREATE TRIGGER guard_character_warning_delete BEFORE DELETE ON character_pack_warnings WHEN OLD.resolution != 'unresolved' OR (SELECT state FROM character_packs WHERE id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character warning is append-only'); END;
CREATE TRIGGER guard_character_pack_events_update BEFORE UPDATE ON character_pack_events WHEN
  NEW.pack_id = OLD.pack_id OR NEW.sequence != OLD.sequence OR NEW.reviewer_id != OLD.reviewer_id OR NEW.event_type != OLD.event_type OR NEW.payload_json != OLD.payload_json OR NEW.created_at != OLD.created_at
  BEGIN SELECT RAISE(ABORT, 'character pack events are append-only'); END;
CREATE TRIGGER guard_character_pack_events_delete BEFORE DELETE ON character_pack_events BEGIN SELECT RAISE(ABORT, 'character pack events are append-only'); END;
"""

_MIGRATION_15 = """
PRAGMA defer_foreign_keys=ON;
DROP TRIGGER validate_character_pack_event_insert;
DROP TRIGGER guard_character_frozen_reference_insert;
DROP TRIGGER guard_character_frozen_annotation_insert;
DROP TRIGGER guard_character_frozen_warning_insert;
DROP TRIGGER guard_characters_update;
DROP TRIGGER guard_characters_delete;
DROP TRIGGER guard_character_pack_alias_update;
DROP TRIGGER guard_character_pack_alias_delete;
DROP TRIGGER guard_character_pack_frozen_update;
DROP TRIGGER validate_character_pack_preregister_update;
DROP TRIGGER guard_character_pack_frozen_delete;
DROP TRIGGER guard_character_reference_update;
DROP TRIGGER guard_character_reference_delete;
DROP TRIGGER guard_character_annotation_update;
DROP TRIGGER guard_character_annotation_delete;
DROP TRIGGER guard_character_warning_update;
DROP TRIGGER guard_character_warning_delete;
DROP TRIGGER guard_character_pack_events_update;
DROP TRIGGER guard_character_pack_events_delete;
ALTER TABLE character_pack_events RENAME TO character_pack_events_v14;
ALTER TABLE character_pack_warnings RENAME TO character_pack_warnings_v14;
ALTER TABLE character_annotations RENAME TO character_annotations_v14;
ALTER TABLE character_references RENAME TO character_references_v14;
ALTER TABLE character_pack_aliases RENAME TO character_pack_aliases_v14;
ALTER TABLE character_packs RENAME TO character_packs_v14;
CREATE TABLE character_packs(
  storage_id TEXT PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  character_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('draft','preregistered')),
  supersedes_storage_id TEXT,
  manifest_json TEXT,
  created_at TEXT NOT NULL,
  preregistered_at TEXT,
  UNIQUE(character_id, id),
  UNIQUE(character_id, storage_id),
  FOREIGN KEY(character_id, reviewer_id) REFERENCES characters(id, reviewer_id),
  FOREIGN KEY(character_id, supersedes_storage_id) REFERENCES character_packs(character_id, storage_id),
  CHECK((state = 'draft' AND manifest_json IS NULL AND preregistered_at IS NULL)
     OR (state = 'preregistered' AND manifest_json IS NOT NULL AND preregistered_at IS NOT NULL))
);
CREATE TABLE character_pack_aliases(
  alias_id TEXT PRIMARY KEY,
  pack_storage_id TEXT NOT NULL REFERENCES character_packs(storage_id)
);
CREATE TABLE character_references(
  pack_id TEXT NOT NULL REFERENCES character_packs(storage_id),
  reference_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  source_name TEXT NOT NULL CHECK(length(source_name) > 0 AND substr(source_name, 1, 1) != '/'),
  image_sha256 TEXT NOT NULL CHECK(length(image_sha256) = 64 AND image_sha256 = lower(image_sha256)),
  media_type TEXT NOT NULL CHECK(media_type IN ('image/jpeg','image/png','image/webp')),
  dhash_hex TEXT NOT NULL CHECK(length(dhash_hex) = 16 AND dhash_hex = lower(dhash_hex)),
  disposition TEXT NOT NULL CHECK(disposition IN ('active','rejected')),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(pack_id, reference_id),
  UNIQUE(pack_id, ordinal),
  UNIQUE(pack_id, artifact_id)
);
CREATE TABLE character_annotations(
  pack_id TEXT NOT NULL,
  reference_id TEXT NOT NULL,
  framing TEXT NOT NULL CHECK(framing IN ('portrait','full_body')),
  head_angle TEXT NOT NULL CHECK(head_angle IN ('front','three_quarter','profile')),
  body_visibility TEXT NOT NULL CHECK(body_visibility IN ('clear_face','partial_body','full_body')),
  expression TEXT NOT NULL CHECK(expression IN ('neutral','smile','serious','angry','surprised')),
  lighting TEXT NOT NULL CHECK(lighting IN ('ordinary','directional')),
  occlusion TEXT NOT NULL CHECK(occlusion IN ('none','partial','heavy')),
  identity_confirmed INTEGER NOT NULL CHECK(identity_confirmed IN (0,1)),
  PRIMARY KEY(pack_id, reference_id),
  FOREIGN KEY(pack_id, reference_id) REFERENCES character_references(pack_id, reference_id)
);
CREATE TABLE character_pack_warnings(
  pack_id TEXT NOT NULL REFERENCES character_packs(storage_id),
  left_reference_id TEXT NOT NULL,
  right_reference_id TEXT NOT NULL,
  distance INTEGER NOT NULL CHECK(distance BETWEEN 0 AND 64),
  resolution TEXT NOT NULL CHECK(resolution IN ('unresolved','accepted','dismissed')),
  PRIMARY KEY(pack_id, left_reference_id, right_reference_id),
  FOREIGN KEY(pack_id, left_reference_id) REFERENCES character_references(pack_id, reference_id),
  FOREIGN KEY(pack_id, right_reference_id) REFERENCES character_references(pack_id, reference_id),
  CHECK(left_reference_id < right_reference_id)
);
CREATE TABLE character_pack_events(
  pack_id TEXT NOT NULL REFERENCES character_packs(storage_id),
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  reviewer_id TEXT NOT NULL,
  event_type TEXT NOT NULL CHECK(event_type IN ('draft_created','preregistered')),
  payload_json TEXT NOT NULL CHECK(json_valid(payload_json) AND json_type(payload_json) = 'object'),
  created_at TEXT NOT NULL,
  PRIMARY KEY(pack_id, sequence)
);
INSERT INTO character_packs(
  storage_id, id, character_id, reviewer_id, state, supersedes_storage_id,
  manifest_json, created_at, preregistered_at
) SELECT id, id, character_id, reviewer_id, state, supersedes_pack_id,
  manifest_json, created_at, preregistered_at FROM character_packs_v14;
INSERT INTO character_pack_aliases(alias_id, pack_storage_id)
  SELECT alias_id, pack_id FROM character_pack_aliases_v14;
INSERT INTO character_references SELECT * FROM character_references_v14;
INSERT INTO character_annotations SELECT * FROM character_annotations_v14;
INSERT INTO character_pack_warnings SELECT * FROM character_pack_warnings_v14;
INSERT INTO character_pack_events SELECT * FROM character_pack_events_v14;
DROP TABLE character_pack_events_v14;
DROP TABLE character_pack_warnings_v14;
DROP TABLE character_annotations_v14;
DROP TABLE character_references_v14;
DROP TABLE character_pack_aliases_v14;
DROP TABLE character_packs_v14;
CREATE TRIGGER validate_character_pack_event_insert BEFORE INSERT ON character_pack_events BEGIN
  SELECT CASE WHEN NEW.sequence != COALESCE((SELECT MAX(sequence) + 1 FROM character_pack_events WHERE pack_id = NEW.pack_id), 1)
    THEN RAISE(ABORT, 'character pack event sequence is not append-only') END;
  SELECT CASE WHEN NEW.reviewer_id != (SELECT reviewer_id FROM character_packs WHERE storage_id = NEW.pack_id)
    THEN RAISE(ABORT, 'character pack event reviewer mismatch') END;
  SELECT CASE WHEN NOT (
    (NEW.sequence = 1 AND NEW.event_type = 'draft_created' AND (SELECT state FROM character_packs WHERE storage_id = NEW.pack_id) = 'draft')
    OR (NEW.sequence = 2 AND NEW.event_type = 'preregistered' AND (SELECT state FROM character_packs WHERE storage_id = NEW.pack_id) = 'preregistered')
  ) THEN RAISE(ABORT, 'character pack event does not match lifecycle') END;
END;
CREATE TRIGGER guard_character_frozen_reference_insert BEFORE INSERT ON character_references WHEN (SELECT state FROM character_packs WHERE storage_id = NEW.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_frozen_annotation_insert BEFORE INSERT ON character_annotations WHEN (SELECT state FROM character_packs WHERE storage_id = NEW.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_frozen_warning_insert BEFORE INSERT ON character_pack_warnings WHEN (SELECT state FROM character_packs WHERE storage_id = NEW.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_characters_update BEFORE UPDATE ON characters BEGIN SELECT RAISE(ABORT, 'character identity is append-only'); END;
CREATE TRIGGER guard_characters_delete BEFORE DELETE ON characters BEGIN SELECT RAISE(ABORT, 'character identity is append-only'); END;
CREATE TRIGGER guard_character_pack_alias_update BEFORE UPDATE ON character_pack_aliases BEGIN SELECT RAISE(ABORT, 'character pack alias is append-only'); END;
CREATE TRIGGER guard_character_pack_alias_delete BEFORE DELETE ON character_pack_aliases BEGIN SELECT RAISE(ABORT, 'character pack alias is append-only'); END;
CREATE TRIGGER guard_character_pack_frozen_update BEFORE UPDATE ON character_packs WHEN OLD.state = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER validate_character_pack_preregister_update BEFORE UPDATE ON character_packs WHEN OLD.state = 'draft' BEGIN
  SELECT CASE WHEN NEW.storage_id != OLD.storage_id OR NEW.character_id != OLD.character_id OR NEW.reviewer_id != OLD.reviewer_id OR NEW.supersedes_storage_id IS NOT OLD.supersedes_storage_id OR NEW.created_at != OLD.created_at
    OR NOT ((NEW.state = 'preregistered' AND NEW.id = OLD.id) OR (NEW.state = 'draft' AND NEW.manifest_json IS NULL AND NEW.preregistered_at IS NULL))
    THEN RAISE(ABORT, 'invalid character pack transition') END;
END;
CREATE TRIGGER guard_character_pack_frozen_delete BEFORE DELETE ON character_packs WHEN OLD.state = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_reference_update BEFORE UPDATE ON character_references WHEN (SELECT state FROM character_packs WHERE storage_id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_reference_delete BEFORE DELETE ON character_references WHEN (SELECT state FROM character_packs WHERE storage_id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_annotation_update BEFORE UPDATE ON character_annotations WHEN (SELECT state FROM character_packs WHERE storage_id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_annotation_delete BEFORE DELETE ON character_annotations WHEN (SELECT state FROM character_packs WHERE storage_id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character pack is append-only'); END;
CREATE TRIGGER guard_character_warning_update BEFORE UPDATE ON character_pack_warnings WHEN
  OLD.resolution != 'unresolved' OR (SELECT state FROM character_packs WHERE storage_id = OLD.pack_id) = 'preregistered'
  BEGIN SELECT RAISE(ABORT, 'character warning is append-only'); END;
CREATE TRIGGER guard_character_warning_delete BEFORE DELETE ON character_pack_warnings WHEN OLD.resolution != 'unresolved' OR (SELECT state FROM character_packs WHERE storage_id = OLD.pack_id) = 'preregistered' BEGIN SELECT RAISE(ABORT, 'character warning is append-only'); END;
CREATE TRIGGER guard_character_pack_events_update BEFORE UPDATE ON character_pack_events BEGIN SELECT RAISE(ABORT, 'character pack events are append-only'); END;
CREATE TRIGGER guard_character_pack_events_delete BEFORE DELETE ON character_pack_events BEGIN SELECT RAISE(ABORT, 'character pack events are append-only'); END;
"""

_MIGRATION_16 = """
CREATE UNIQUE INDEX character_pack_storage_reviewer
ON character_packs(storage_id, reviewer_id);
CREATE TABLE character_curation_grants(
  token_digest TEXT PRIMARY KEY
    CHECK(length(token_digest) = 64 AND token_digest = lower(token_digest)),
  reviewer_id TEXT NOT NULL CHECK(length(reviewer_id) > 0 AND reviewer_id = trim(reviewer_id)),
  character_id TEXT NOT NULL,
  pack_storage_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  FOREIGN KEY(character_id, reviewer_id) REFERENCES characters(id, reviewer_id),
  FOREIGN KEY(character_id, pack_storage_id)
    REFERENCES character_packs(character_id, storage_id),
  FOREIGN KEY(pack_storage_id, reviewer_id)
    REFERENCES character_packs(storage_id, reviewer_id),
  CHECK(julianday(expires_at) > julianday(created_at))
);
CREATE INDEX character_curation_grants_expiry
ON character_curation_grants(expires_at);
CREATE INDEX character_curation_grants_owner
ON character_curation_grants(reviewer_id);
CREATE TRIGGER guard_character_curation_grants_update
BEFORE UPDATE ON character_curation_grants BEGIN
  SELECT RAISE(ABORT, 'character curation grants cannot be changed');
END;
"""

_MIGRATION_17 = """
CREATE TABLE character_benchmark_campaigns(
  id TEXT PRIMARY KEY CHECK(substr(id, 1, 20) = 'character_benchmark_'),
  reviewer_id TEXT NOT NULL CHECK(length(reviewer_id) > 0 AND reviewer_id = trim(reviewer_id)),
  content_digest TEXT NOT NULL UNIQUE CHECK(length(content_digest) = 71 AND substr(content_digest, 1, 7) = 'sha256:'),
  definition_json TEXT NOT NULL CHECK(json_valid(definition_json) AND json_type(definition_json) = 'object'),
  character_pack_id TEXT NOT NULL,
  character_pack_storage_id TEXT NOT NULL REFERENCES character_packs(storage_id),
  stage_2_campaign_id TEXT NOT NULL REFERENCES validation_campaigns(id),
  stage_2_manifest_json TEXT NOT NULL CHECK(json_valid(stage_2_manifest_json) AND json_type(stage_2_manifest_json) = 'object'),
  identity_prompt TEXT NOT NULL CHECK(length(identity_prompt) > 0 AND identity_prompt = trim(identity_prompt)),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'),
  UNIQUE(id, reviewer_id),
  FOREIGN KEY(character_pack_storage_id, reviewer_id) REFERENCES character_packs(storage_id, reviewer_id)
);
CREATE TABLE character_benchmark_events(
  campaign_id TEXT NOT NULL REFERENCES character_benchmark_campaigns(id),
  sequence INTEGER NOT NULL CHECK(sequence BETWEEN 1 AND 6),
  previous_state TEXT NOT NULL CHECK(previous_state IN ('draft','preregistered','generating','ready','judging','analyzing','completed')),
  new_state TEXT NOT NULL CHECK(new_state IN ('draft','preregistered','generating','ready','judging','analyzing','completed')),
  payload_digest TEXT NOT NULL CHECK(length(payload_digest) = 71 AND substr(payload_digest, 1, 7) = 'sha256:'),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'),
  PRIMARY KEY(campaign_id, sequence),
  CHECK(previous_state <> new_state)
);
CREATE TABLE character_benchmark_trials(
  campaign_id TEXT NOT NULL REFERENCES character_benchmark_campaigns(id),
  trial_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK(ordinal BETWEEN 1 AND 40),
  trial_json TEXT NOT NULL CHECK(json_valid(trial_json) AND json_type(trial_json) = 'object'),
  hidden_repeat INTEGER NOT NULL CHECK(hidden_repeat IN (0,1)),
  reuses_trial_id TEXT,
  candidate_a_position TEXT NOT NULL CHECK(candidate_a_position IN ('left','right')),
  PRIMARY KEY(campaign_id, trial_id),
  UNIQUE(campaign_id, ordinal),
  FOREIGN KEY(campaign_id, reuses_trial_id) REFERENCES character_benchmark_trials(campaign_id, trial_id) DEFERRABLE INITIALLY DEFERRED,
  CHECK((hidden_repeat = 1) = (reuses_trial_id IS NOT NULL))
);
CREATE TABLE character_benchmark_experiment_bindings(
  campaign_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  experiment_id TEXT NOT NULL UNIQUE REFERENCES experiments(id),
  PRIMARY KEY(campaign_id, trial_id),
  FOREIGN KEY(campaign_id, trial_id) REFERENCES character_benchmark_trials(campaign_id, trial_id)
);
CREATE TABLE character_benchmark_review_bindings(
  campaign_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  review_id TEXT NOT NULL UNIQUE REFERENCES review_revisions(id),
  session_id TEXT NOT NULL UNIQUE REFERENCES review_sessions(id),
  reviewer_alias TEXT NOT NULL,
  reused_from_trial_id TEXT,
  display_mapping_json TEXT NOT NULL CHECK(json_valid(display_mapping_json) AND json_type(display_mapping_json) = 'object'),
  PRIMARY KEY(campaign_id, trial_id),
  FOREIGN KEY(campaign_id, trial_id) REFERENCES character_benchmark_trials(campaign_id, trial_id),
  FOREIGN KEY(campaign_id, reused_from_trial_id) REFERENCES character_benchmark_trials(campaign_id, trial_id)
);
CREATE TABLE character_benchmark_identity_judgments(
  campaign_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  review_id TEXT NOT NULL,
  choice TEXT NOT NULL CHECK(choice IN ('a','b','both','neither')),
  display_choice TEXT NOT NULL CHECK(display_choice IN ('left','right','both','neither')),
  observation_json TEXT NOT NULL CHECK(json_valid(observation_json) AND json_type(observation_json) = 'object'),
  content_digest TEXT NOT NULL CHECK(length(content_digest) = 71 AND substr(content_digest, 1, 7) = 'sha256:'),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'),
  PRIMARY KEY(campaign_id, trial_id),
  UNIQUE(review_id),
  FOREIGN KEY(campaign_id, trial_id) REFERENCES character_benchmark_trials(campaign_id, trial_id)
);
CREATE TABLE character_benchmark_secondary_judgments(
  campaign_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  dimension TEXT NOT NULL CHECK(dimension IN ('prompt_adherence','motion_quality','temporal_consistency','visual_quality')),
  choice TEXT NOT NULL CHECK(choice IN ('a','b','tie')),
  observation_json TEXT NOT NULL CHECK(json_valid(observation_json) AND json_type(observation_json) = 'object'),
  content_digest TEXT NOT NULL CHECK(length(content_digest) = 71 AND substr(content_digest, 1, 7) = 'sha256:'),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00'),
  PRIMARY KEY(campaign_id, trial_id, dimension),
  FOREIGN KEY(campaign_id, trial_id) REFERENCES character_benchmark_identity_judgments(campaign_id, trial_id)
);
CREATE TABLE character_benchmark_results(
  campaign_id TEXT PRIMARY KEY REFERENCES character_benchmark_campaigns(id),
  result_json TEXT NOT NULL CHECK(json_valid(result_json) AND json_type(result_json) = 'object'),
  content_digest TEXT NOT NULL CHECK(length(content_digest) = 71 AND substr(content_digest, 1, 7) = 'sha256:'),
  created_at TEXT NOT NULL CHECK(julianday(created_at) IS NOT NULL AND substr(created_at, -6) = '+00:00')
);
CREATE TRIGGER validate_character_benchmark_event_insert BEFORE INSERT ON character_benchmark_events BEGIN
  SELECT CASE WHEN NEW.sequence <> COALESCE((SELECT MAX(sequence)+1 FROM character_benchmark_events WHERE campaign_id=NEW.campaign_id),1) THEN RAISE(ABORT,'invalid character benchmark event sequence') END;
  SELECT CASE WHEN NEW.previous_state <> COALESCE((SELECT new_state FROM character_benchmark_events WHERE campaign_id=NEW.campaign_id ORDER BY sequence DESC LIMIT 1),'draft') THEN RAISE(ABORT,'invalid character benchmark prior state') END;
  SELECT CASE WHEN NOT ((NEW.previous_state='draft' AND NEW.new_state='preregistered') OR (NEW.previous_state='preregistered' AND NEW.new_state='generating') OR (NEW.previous_state='generating' AND NEW.new_state='ready') OR (NEW.previous_state='ready' AND NEW.new_state='judging') OR (NEW.previous_state='judging' AND NEW.new_state='analyzing') OR (NEW.previous_state='analyzing' AND NEW.new_state='completed')) THEN RAISE(ABORT,'invalid character benchmark lifecycle transition') END;
END;
CREATE TRIGGER validate_character_benchmark_trial_insert BEFORE INSERT ON character_benchmark_trials BEGIN
  SELECT CASE WHEN COALESCE((SELECT new_state FROM character_benchmark_events WHERE campaign_id=NEW.campaign_id ORDER BY sequence DESC LIMIT 1),'draft') <> 'preregistered' THEN RAISE(ABORT,'character benchmark trials require preregistration') END;
END;
CREATE TRIGGER guard_character_benchmark_campaign_update BEFORE UPDATE ON character_benchmark_campaigns BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_campaign_delete BEFORE DELETE ON character_benchmark_campaigns BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_trial_update BEFORE UPDATE ON character_benchmark_trials BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_trial_delete BEFORE DELETE ON character_benchmark_trials BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_experiment_binding_update BEFORE UPDATE ON character_benchmark_experiment_bindings BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_experiment_binding_delete BEFORE DELETE ON character_benchmark_experiment_bindings BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_review_binding_update BEFORE UPDATE ON character_benchmark_review_bindings BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_review_binding_delete BEFORE DELETE ON character_benchmark_review_bindings BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_identity_update BEFORE UPDATE ON character_benchmark_identity_judgments BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_identity_delete BEFORE DELETE ON character_benchmark_identity_judgments BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_secondary_update BEFORE UPDATE ON character_benchmark_secondary_judgments BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_secondary_delete BEFORE DELETE ON character_benchmark_secondary_judgments BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_event_update BEFORE UPDATE ON character_benchmark_events BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_event_delete BEFORE DELETE ON character_benchmark_events BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_result_update BEFORE UPDATE ON character_benchmark_results BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_result_delete BEFORE DELETE ON character_benchmark_results BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
"""

_MIGRATION_18 = """
DROP TRIGGER guard_character_benchmark_trial_update;
DROP TRIGGER guard_character_benchmark_secondary_update;
ALTER TABLE character_benchmark_trials ADD COLUMN expected_reviewer_id TEXT;
UPDATE character_benchmark_trials
SET expected_reviewer_id = character_benchmark_reviewer_id(
  campaign_id,
  trial_id,
  (SELECT campaign.reviewer_id FROM character_benchmark_campaigns AS campaign
   WHERE campaign.id = character_benchmark_trials.campaign_id)
);
ALTER TABLE character_benchmark_secondary_judgments ADD COLUMN review_id TEXT;
UPDATE character_benchmark_secondary_judgments
SET review_id = (
  SELECT identity.review_id FROM character_benchmark_identity_judgments AS identity
  WHERE identity.campaign_id = character_benchmark_secondary_judgments.campaign_id
    AND identity.trial_id = character_benchmark_secondary_judgments.trial_id
);
CREATE TRIGGER guard_character_benchmark_trial_update BEFORE UPDATE ON character_benchmark_trials BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER guard_character_benchmark_secondary_update BEFORE UPDATE ON character_benchmark_secondary_judgments BEGIN SELECT RAISE(ABORT,'character benchmark evidence is append-only'); END;
CREATE TRIGGER validate_character_benchmark_campaign_insert_v18
BEFORE INSERT ON character_benchmark_campaigns BEGIN
  SELECT CASE WHEN json_extract(NEW.definition_json, '$.id') IS NOT NEW.id
    THEN RAISE(ABORT, 'character benchmark definition campaign mismatch') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM character_packs AS pack
    WHERE pack.storage_id = NEW.character_pack_storage_id
      AND pack.id = NEW.character_pack_id
      AND pack.reviewer_id = NEW.reviewer_id
      AND pack.state = 'preregistered'
  ) THEN RAISE(ABORT, 'character benchmark pack is not owned and preregistered') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM validation_campaigns AS campaign
    JOIN validation_results AS result ON result.campaign_id = campaign.id
    WHERE campaign.id = NEW.stage_2_campaign_id
      AND campaign.reviewer_id = NEW.reviewer_id
      AND result.disposition = 'qualified'
      AND (SELECT event.new_state FROM validation_events AS event
           WHERE event.campaign_id = campaign.id
           ORDER BY event.sequence DESC LIMIT 1) = 'qualified'
  ) THEN RAISE(ABORT, 'character benchmark Stage 2 campaign is not owned and qualified') END;
END;
CREATE TRIGGER validate_character_benchmark_trial_insert_v18
BEFORE INSERT ON character_benchmark_trials BEGIN
  SELECT CASE WHEN NEW.expected_reviewer_id IS NULL OR NEW.expected_reviewer_id !=
    character_benchmark_reviewer_id(
      NEW.campaign_id,
      NEW.trial_id,
      (SELECT campaign.reviewer_id FROM character_benchmark_campaigns AS campaign
       WHERE campaign.id = NEW.campaign_id)
    ) THEN RAISE(ABORT, 'character benchmark trial reviewer mismatch') END;
  SELECT CASE WHEN json_extract(NEW.trial_json, '$.benchmark_id') IS NOT NEW.campaign_id
    OR json_extract(NEW.trial_json, '$.id') IS NOT NEW.trial_id
    OR json_extract(NEW.trial_json, '$.ordinal') IS NOT NEW.ordinal
    OR json_extract(NEW.trial_json, '$.hidden_repeat') IS NOT NEW.hidden_repeat
    OR json_extract(NEW.trial_json, '$.reuses_trial_id') IS NOT NEW.reuses_trial_id
    OR json_extract(NEW.trial_json, '$.candidate_a_position') IS NOT NEW.candidate_a_position
    THEN RAISE(ABORT, 'character benchmark trial columns do not match trial evidence') END;
END;
CREATE TRIGGER validate_character_benchmark_experiment_binding_insert_v18
BEFORE INSERT ON character_benchmark_experiment_bindings BEGIN
  SELECT CASE WHEN COALESCE(
    (SELECT event.new_state FROM character_benchmark_events AS event
     WHERE event.campaign_id = NEW.campaign_id ORDER BY event.sequence DESC LIMIT 1),
    'draft'
  ) != 'generating' THEN RAISE(ABORT, 'character benchmark experiment has invalid lifecycle') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM character_benchmark_trials AS trial
    WHERE trial.campaign_id = NEW.campaign_id
      AND trial.trial_id = NEW.trial_id
      AND trial.hidden_repeat = 0
      AND NEW.experiment_id = character_benchmark_experiment_id(
        NEW.campaign_id, NEW.trial_id
      )
  ) THEN RAISE(ABORT, 'character benchmark experiment does not match trial') END;
END;
CREATE TRIGGER validate_character_benchmark_review_binding_insert_v18
BEFORE INSERT ON character_benchmark_review_bindings BEGIN
  SELECT CASE WHEN COALESCE(
    (SELECT event.new_state FROM character_benchmark_events AS event
     WHERE event.campaign_id = NEW.campaign_id ORDER BY event.sequence DESC LIMIT 1),
    'draft'
  ) != 'judging' THEN RAISE(ABORT, 'character benchmark review has invalid lifecycle') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM character_benchmark_trials AS trial
    JOIN character_benchmark_experiment_bindings AS source
      ON source.campaign_id = trial.campaign_id
     AND source.trial_id = COALESCE(trial.reuses_trial_id, trial.trial_id)
     AND source.experiment_id = NEW.experiment_id
    JOIN review_sessions AS session
      ON session.id = NEW.session_id
     AND session.experiment_id = NEW.experiment_id
     AND session.reviewer_id = trial.expected_reviewer_id
    JOIN review_revisions AS revision
      ON revision.id = NEW.review_id
     AND revision.session_id = NEW.session_id
     AND revision.revision = 1
     AND revision.supersedes_review_id IS NULL
    WHERE trial.campaign_id = NEW.campaign_id
      AND trial.trial_id = NEW.trial_id
      AND NEW.reviewer_alias = trial.expected_reviewer_id
      AND NEW.reused_from_trial_id IS trial.reuses_trial_id
  ) THEN RAISE(ABORT, 'character benchmark review binding is inconsistent') END;
END;
CREATE TRIGGER validate_character_benchmark_identity_insert_v18
BEFORE INSERT ON character_benchmark_identity_judgments BEGIN
  SELECT CASE WHEN COALESCE(
    (SELECT event.new_state FROM character_benchmark_events AS event
     WHERE event.campaign_id = NEW.campaign_id ORDER BY event.sequence DESC LIMIT 1),
    'draft'
  ) != 'judging' THEN RAISE(ABORT, 'character benchmark identity has invalid lifecycle') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM character_benchmark_review_bindings AS binding
    WHERE binding.campaign_id = NEW.campaign_id
      AND binding.trial_id = NEW.trial_id
      AND binding.review_id = NEW.review_id
  ) THEN RAISE(ABORT, 'character benchmark identity review binding is inconsistent') END;
  SELECT CASE WHEN json_extract(NEW.observation_json, '$.benchmark_id') IS NOT NEW.campaign_id
    OR json_extract(NEW.observation_json, '$.trial_id') IS NOT NEW.trial_id
    OR json_extract(NEW.observation_json, '$.choice') IS NOT NEW.choice
    THEN RAISE(ABORT, 'character benchmark identity columns do not match observation') END;
END;
CREATE TRIGGER validate_character_benchmark_secondary_insert_v18
BEFORE INSERT ON character_benchmark_secondary_judgments BEGIN
  SELECT CASE WHEN COALESCE(
    (SELECT event.new_state FROM character_benchmark_events AS event
     WHERE event.campaign_id = NEW.campaign_id ORDER BY event.sequence DESC LIMIT 1),
    'draft'
  ) != 'judging' THEN RAISE(ABORT, 'character benchmark secondary has invalid lifecycle') END;
  SELECT CASE WHEN NEW.review_id IS NULL OR NOT EXISTS(
    SELECT 1 FROM character_benchmark_identity_judgments AS identity
    WHERE identity.campaign_id = NEW.campaign_id
      AND identity.trial_id = NEW.trial_id
      AND identity.review_id = NEW.review_id
      AND identity.choice != 'neither'
  ) THEN RAISE(ABORT, 'character benchmark secondary identity binding is inconsistent') END;
  SELECT CASE WHEN json_extract(NEW.observation_json, '$.benchmark_id') IS NOT NEW.campaign_id
    OR json_extract(NEW.observation_json, '$.trial_id') IS NOT NEW.trial_id
    OR json_extract(NEW.observation_json, '$.dimension') IS NOT NEW.dimension
    OR json_extract(NEW.observation_json, '$.choice') IS NOT NEW.choice
    THEN RAISE(ABORT, 'character benchmark secondary columns do not match observation') END;
END;
CREATE TRIGGER validate_character_benchmark_result_insert_v18
BEFORE INSERT ON character_benchmark_results BEGIN
  SELECT CASE WHEN COALESCE(
    (SELECT event.new_state FROM character_benchmark_events AS event
     WHERE event.campaign_id = NEW.campaign_id ORDER BY event.sequence DESC LIMIT 1),
    'draft'
  ) NOT IN ('analyzing','completed')
    THEN RAISE(ABORT, 'character benchmark result has invalid lifecycle') END;
  SELECT CASE WHEN json_extract(NEW.result_json, '$.benchmark_id') IS NOT NEW.campaign_id
    THEN RAISE(ABORT, 'character benchmark result campaign mismatch') END;
END;
"""

_MIGRATION_19 = """
DROP TRIGGER validate_character_benchmark_secondary_insert_v18;
CREATE TRIGGER validate_character_benchmark_secondary_insert_v19
BEFORE INSERT ON character_benchmark_secondary_judgments BEGIN
  SELECT CASE WHEN COALESCE(
    (SELECT event.new_state FROM character_benchmark_events AS event
     WHERE event.campaign_id = NEW.campaign_id ORDER BY event.sequence DESC LIMIT 1),
    'draft'
  ) != 'judging' THEN RAISE(ABORT, 'character benchmark secondary has invalid lifecycle') END;
  SELECT CASE WHEN NOT EXISTS(
    SELECT 1 FROM character_benchmark_trials AS trial
    WHERE trial.campaign_id = NEW.campaign_id
      AND trial.trial_id = NEW.trial_id
      AND trial.hidden_repeat = 0
  ) THEN RAISE(ABORT, 'character benchmark secondary requires a primary trial') END;
  SELECT CASE WHEN NEW.review_id IS NULL OR NOT EXISTS(
    SELECT 1 FROM character_benchmark_identity_judgments AS identity
    WHERE identity.campaign_id = NEW.campaign_id
      AND identity.trial_id = NEW.trial_id
      AND identity.review_id = NEW.review_id
      AND identity.choice != 'neither'
  ) THEN RAISE(ABORT, 'character benchmark secondary identity binding is inconsistent') END;
  SELECT CASE WHEN json_extract(NEW.observation_json, '$.benchmark_id') IS NOT NEW.campaign_id
    OR json_extract(NEW.observation_json, '$.trial_id') IS NOT NEW.trial_id
    OR json_extract(NEW.observation_json, '$.dimension') IS NOT NEW.dimension
    OR json_extract(NEW.observation_json, '$.choice') IS NOT NEW.choice
    THEN RAISE(ABORT, 'character benchmark secondary columns do not match observation') END;
END;
"""

_MIGRATION_20 = """
CREATE TRIGGER validate_character_pack_duplicate_provenance_insert
BEFORE INSERT ON character_packs WHEN NEW.state = 'preregistered' BEGIN
  SELECT CASE WHEN
    json_extract(NEW.manifest_json, '$.duplicate_hash_algorithm_version')
      IS NOT 'dhash64-exif-transpose-lanczos.v1'
    OR json_extract(NEW.manifest_json, '$.near_duplicate_threshold_inclusive') IS NOT 5
    THEN RAISE(ABORT, 'character pack duplicate provenance is incomplete') END;
END;
CREATE TRIGGER validate_character_pack_duplicate_provenance_update
BEFORE UPDATE ON character_packs
WHEN OLD.state = 'draft' AND NEW.state = 'preregistered' BEGIN
  SELECT CASE WHEN
    json_extract(NEW.manifest_json, '$.duplicate_hash_algorithm_version')
      IS NOT 'dhash64-exif-transpose-lanczos.v1'
    OR json_extract(NEW.manifest_json, '$.near_duplicate_threshold_inclusive') IS NOT 5
    THEN RAISE(ABORT, 'character pack duplicate provenance is incomplete') END;
END;
"""

_MIGRATION_21 = ""

_MIGRATION_22 = """
CREATE TABLE training_datasets(
  storage_id TEXT PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  reviewer_id TEXT NOT NULL,
  character_id TEXT NOT NULL,
  character_pack_storage_id TEXT NOT NULL,
  character_pack_digest TEXT NOT NULL CHECK(length(character_pack_digest)=71 AND substr(character_pack_digest,1,7)='sha256:'),
  state TEXT NOT NULL CHECK(state IN ('draft','sealed')),
  canonical_outfit_id TEXT NOT NULL,
  reserved_token TEXT NOT NULL,
  source_tier_weights_json TEXT NOT NULL CHECK(json_valid(source_tier_weights_json)),
  manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json) AND json_type(manifest_json)='object'),
  dataset_digest TEXT UNIQUE,
  created_at TEXT NOT NULL,
  sealed_at TEXT,
  UNIQUE(storage_id, reviewer_id),
  UNIQUE(storage_id, character_id),
  FOREIGN KEY(character_id, reviewer_id) REFERENCES characters(id, reviewer_id),
  FOREIGN KEY(character_id,character_pack_storage_id) REFERENCES character_packs(character_id,storage_id),
  FOREIGN KEY(character_pack_storage_id,reviewer_id) REFERENCES character_packs(storage_id,reviewer_id),
  CHECK((state='draft' AND sealed_at IS NULL AND dataset_digest IS NULL) OR (state='sealed' AND sealed_at IS NOT NULL AND dataset_digest IS NOT NULL))
);
CREATE TABLE training_dataset_aliases(
  alias_id TEXT PRIMARY KEY,
  dataset_storage_id TEXT NOT NULL REFERENCES training_datasets(storage_id)
);
CREATE TABLE training_samples(
  dataset_storage_id TEXT NOT NULL REFERENCES training_datasets(storage_id),
  sample_id TEXT NOT NULL,
  source_tier TEXT NOT NULL CHECK(source_tier IN ('gold_still','gold_clip','synthetic_clip')),
  sample_digest TEXT NOT NULL,
  source_lineage TEXT NOT NULL,
  caption TEXT NOT NULL,
  split TEXT NOT NULL CHECK(split IN ('train','validation','sealed_holdout')),
  media_json TEXT NOT NULL CHECK(json_valid(media_json)),
  accepted INTEGER NOT NULL CHECK(accepted IN (0,1)),
  ordinal INTEGER NOT NULL CHECK(ordinal>=0),
  PRIMARY KEY(dataset_storage_id,sample_id),
  UNIQUE(dataset_storage_id,ordinal),
  UNIQUE(dataset_storage_id,sample_digest)
);
CREATE TABLE training_sample_annotations(
  dataset_storage_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  annotation_json TEXT NOT NULL CHECK(json_valid(annotation_json) AND json_type(annotation_json)='object'),
  PRIMARY KEY(dataset_storage_id,sample_id),
  FOREIGN KEY(dataset_storage_id,sample_id) REFERENCES training_samples(dataset_storage_id,sample_id)
);
CREATE INDEX training_datasets_owner ON training_datasets(reviewer_id,character_id,state);
CREATE INDEX training_samples_split ON training_samples(dataset_storage_id,split,source_tier);
CREATE TRIGGER validate_training_dataset_insert BEFORE INSERT ON training_datasets BEGIN SELECT CASE WHEN NEW.state!='draft' OR training_draft_manifest_is_valid(NEW.id,NEW.character_id,NEW.character_pack_digest,NEW.canonical_outfit_id,NEW.reserved_token,NEW.source_tier_weights_json,NEW.manifest_json)!=1 OR NOT EXISTS(SELECT 1 FROM character_packs pack WHERE pack.storage_id=NEW.character_pack_storage_id AND pack.character_id=NEW.character_id AND pack.reviewer_id=NEW.reviewer_id AND pack.state='preregistered' AND canonical_json_digest(pack.manifest_json)=NEW.character_pack_digest) THEN RAISE(ABORT,'training dataset identity or character pack provenance mismatch') END; END;
CREATE TRIGGER materialize_training_dataset AFTER INSERT ON training_datasets BEGIN
 INSERT INTO training_dataset_aliases(alias_id,dataset_storage_id) VALUES(NEW.id,NEW.storage_id);
 INSERT INTO training_samples(dataset_storage_id,sample_id,source_tier,sample_digest,source_lineage,caption,split,media_json,accepted,ordinal) SELECT NEW.storage_id,json_extract(item.value,'$.sample_id'),json_extract(item.value,'$.source_tier'),json_extract(item.value,'$.sample_digest'),json_extract(item.value,'$.source_lineage'),json_extract(item.value,'$.caption'),json_extract(item.value,'$.split'),json_extract(item.value,'$.media'),json_extract(item.value,'$.accepted'),CAST(item.key AS INTEGER) FROM json_each(NEW.manifest_json,'$.samples') item;
 INSERT INTO training_sample_annotations(dataset_storage_id,sample_id,annotation_json) SELECT NEW.storage_id,json_extract(item.value,'$.sample_id'),json_extract(item.value,'$.annotation') FROM json_each(NEW.manifest_json,'$.samples') item;
END;
CREATE TRIGGER validate_training_dataset_seal BEFORE UPDATE ON training_datasets WHEN OLD.state='draft' AND NEW.state='sealed' BEGIN SELECT CASE WHEN NEW.storage_id!=OLD.storage_id OR NEW.reviewer_id!=OLD.reviewer_id OR NEW.character_id!=OLD.character_id OR NEW.character_pack_storage_id!=OLD.character_pack_storage_id OR NEW.character_pack_digest!=OLD.character_pack_digest OR NEW.canonical_outfit_id!=OLD.canonical_outfit_id OR NEW.reserved_token!=OLD.reserved_token OR NEW.source_tier_weights_json!=OLD.source_tier_weights_json OR NEW.created_at!=OLD.created_at OR NEW.dataset_digest IS NULL OR NEW.sealed_at IS NULL OR training_dataset_identity_is_valid(OLD.storage_id,NEW.dataset_digest,NEW.manifest_json)!=1 THEN RAISE(ABORT,'invalid training dataset seal') END; END;
CREATE TRIGGER guard_training_dataset_other_update BEFORE UPDATE ON training_datasets WHEN OLD.state IN ('draft','sealed') AND NOT (OLD.state='draft' AND NEW.state='sealed') BEGIN SELECT RAISE(ABORT,'training dataset snapshots are immutable'); END;
CREATE TRIGGER guard_training_dataset_delete BEFORE DELETE ON training_datasets BEGIN SELECT RAISE(ABORT,'training dataset snapshots are immutable'); END;
CREATE TRIGGER validate_training_alias_insert BEFORE INSERT ON training_dataset_aliases BEGIN SELECT CASE WHEN NEW.alias_id!=(SELECT id FROM training_datasets WHERE storage_id=NEW.dataset_storage_id) OR EXISTS(SELECT 1 FROM training_dataset_aliases WHERE dataset_storage_id=NEW.dataset_storage_id) THEN RAISE(ABORT,'training dataset alias is not canonical') END; END;
CREATE TRIGGER guard_training_alias_update BEFORE UPDATE ON training_dataset_aliases BEGIN SELECT RAISE(ABORT,'training dataset aliases are immutable'); END;
CREATE TRIGGER guard_training_alias_delete BEFORE DELETE ON training_dataset_aliases BEGIN SELECT RAISE(ABORT,'training dataset aliases are immutable'); END;
CREATE TRIGGER validate_training_sample_insert BEFORE INSERT ON training_samples BEGIN SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM training_datasets parent,json_each(parent.manifest_json,'$.samples') item WHERE parent.storage_id=NEW.dataset_storage_id AND CAST(item.key AS INTEGER)=NEW.ordinal AND json_extract(item.value,'$.sample_id')=NEW.sample_id AND json_extract(item.value,'$.source_tier')=NEW.source_tier AND json_extract(item.value,'$.sample_digest')=NEW.sample_digest AND json_extract(item.value,'$.source_lineage')=NEW.source_lineage AND json_extract(item.value,'$.caption')=NEW.caption AND json_extract(item.value,'$.split')=NEW.split AND canonical_json_digest(json_extract(item.value,'$.media'))=canonical_json_digest(NEW.media_json) AND json_extract(item.value,'$.accepted')=NEW.accepted) THEN RAISE(ABORT,'training sample does not match canonical manifest') END; END;
CREATE TRIGGER guard_training_sample_update BEFORE UPDATE ON training_samples BEGIN SELECT RAISE(ABORT,'training samples are immutable'); END;
CREATE TRIGGER guard_training_sample_delete BEFORE DELETE ON training_samples BEGIN SELECT RAISE(ABORT,'training samples are immutable'); END;
CREATE TRIGGER validate_training_annotation_insert BEFORE INSERT ON training_sample_annotations BEGIN SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM training_datasets parent,json_each(parent.manifest_json,'$.samples') item WHERE parent.storage_id=NEW.dataset_storage_id AND json_extract(item.value,'$.sample_id')=NEW.sample_id AND canonical_json_digest(json_extract(item.value,'$.annotation'))=canonical_json_digest(NEW.annotation_json)) THEN RAISE(ABORT,'training annotation does not match canonical manifest') END; END;
CREATE TRIGGER guard_training_annotation_update BEFORE UPDATE ON training_sample_annotations BEGIN SELECT RAISE(ABORT,'training annotations are immutable'); END;
CREATE TRIGGER guard_training_annotation_delete BEFORE DELETE ON training_sample_annotations BEGIN SELECT RAISE(ABORT,'training annotations are immutable'); END;
"""

_MIGRATION_23 = """
CREATE TABLE training_recipes(
  storage_id TEXT PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  reviewer_id TEXT NOT NULL,
  dataset_storage_id TEXT NOT NULL,
  dataset_digest TEXT NOT NULL,
  character_id TEXT NOT NULL,
  character_pack_digest TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('draft','preregistered')),
  recipe_json TEXT NOT NULL CHECK(json_valid(recipe_json) AND json_type(recipe_json)='object'),
  created_at TEXT NOT NULL,
  preregistered_at TEXT,
  UNIQUE(storage_id,reviewer_id),
  UNIQUE(storage_id,dataset_storage_id),
  FOREIGN KEY(dataset_storage_id,reviewer_id) REFERENCES training_datasets(storage_id,reviewer_id),
  FOREIGN KEY(dataset_storage_id,character_id) REFERENCES training_datasets(storage_id,character_id),
  CHECK((state='draft' AND preregistered_at IS NULL) OR (state='preregistered' AND preregistered_at IS NOT NULL))
);
CREATE TABLE training_runs(
  id TEXT PRIMARY KEY,
  reviewer_id TEXT NOT NULL,
  recipe_storage_id TEXT NOT NULL,
  dataset_storage_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('created','running','completed','failed','cancelled')),
  total_steps INTEGER NOT NULL CHECK(total_steps>0),
  created_at TEXT NOT NULL,
  terminal_at TEXT,
  UNIQUE(id,reviewer_id), UNIQUE(id,recipe_storage_id),
  FOREIGN KEY(recipe_storage_id,reviewer_id) REFERENCES training_recipes(storage_id,reviewer_id),
  FOREIGN KEY(recipe_storage_id,dataset_storage_id) REFERENCES training_recipes(storage_id,dataset_storage_id),
  CHECK((state IN ('created','running') AND terminal_at IS NULL) OR (state IN ('completed','failed','cancelled') AND terminal_at IS NOT NULL))
);
CREATE TABLE training_run_events(
  run_id TEXT NOT NULL, reviewer_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence>0),
  previous_state TEXT NOT NULL, new_state TEXT NOT NULL, payload_json TEXT NOT NULL CHECK(json_valid(payload_json)), created_at TEXT NOT NULL,
  PRIMARY KEY(run_id,sequence), FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id)
);
CREATE TABLE training_telemetry(
  run_id TEXT NOT NULL, reviewer_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence>0), step INTEGER NOT NULL CHECK(step>=0), payload_json TEXT NOT NULL CHECK(json_valid(payload_json)), created_at TEXT NOT NULL,
  PRIMARY KEY(run_id,sequence), UNIQUE(run_id,step), FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id)
);
CREATE TABLE training_failures(
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, reviewer_id TEXT NOT NULL, code TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id)
);
CREATE INDEX training_recipes_dataset ON training_recipes(dataset_storage_id,state);
CREATE INDEX training_runs_recipe ON training_runs(recipe_storage_id,state);
CREATE TRIGGER validate_training_recipe_insert BEFORE INSERT ON training_recipes BEGIN SELECT CASE WHEN (SELECT state FROM training_datasets WHERE storage_id=NEW.dataset_storage_id)!='sealed' OR (SELECT dataset_digest FROM training_datasets WHERE storage_id=NEW.dataset_storage_id)!=NEW.dataset_digest OR json_extract(NEW.recipe_json,'$.dataset_digest')!=NEW.dataset_digest OR json_extract(NEW.recipe_json,'$.character_pack_digest')!=NEW.character_pack_digest OR (SELECT character_pack_digest FROM training_datasets WHERE storage_id=NEW.dataset_storage_id)!=NEW.character_pack_digest OR training_recipe_identity_is_valid(NEW.dataset_storage_id,NEW.id,NEW.recipe_json)!=1 OR NOT EXISTS(SELECT 1 FROM training_datasets dataset JOIN character_benchmark_campaigns benchmark ON benchmark.character_pack_storage_id=dataset.character_pack_storage_id AND benchmark.reviewer_id=NEW.reviewer_id JOIN character_benchmark_results result ON result.campaign_id=benchmark.id WHERE dataset.storage_id=NEW.dataset_storage_id AND character_benchmark_provenance_is_valid(benchmark.id)=1 AND result.content_digest=json_extract(NEW.recipe_json,'$.stage_3_evidence_digest') AND json_extract(benchmark.definition_json,'$.character_pack_digest')=NEW.character_pack_digest AND json_extract(benchmark.definition_json,'$.stage_2_manifest_digest')=json_extract(NEW.recipe_json,'$.stage_2_manifest_digest') AND json_extract(benchmark.definition_json,'$.stage_2_evidence_digest')=json_extract(NEW.recipe_json,'$.stage_2_evidence_digest') AND (SELECT new_state FROM character_benchmark_events WHERE campaign_id=benchmark.id ORDER BY sequence DESC LIMIT 1)='completed') THEN RAISE(ABORT,'training recipe dataset identity mismatch') END; END;
CREATE TRIGGER guard_training_recipe_update BEFORE UPDATE ON training_recipes WHEN OLD.state='preregistered' BEGIN SELECT RAISE(ABORT,'preregistered training recipe is immutable'); END;
CREATE TRIGGER validate_training_recipe_preregister BEFORE UPDATE ON training_recipes WHEN OLD.state='draft' BEGIN SELECT CASE WHEN NEW.storage_id!=OLD.storage_id OR NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id OR NEW.dataset_storage_id!=OLD.dataset_storage_id OR NEW.dataset_digest!=OLD.dataset_digest OR NEW.character_id!=OLD.character_id OR NEW.character_pack_digest!=OLD.character_pack_digest OR NEW.recipe_json!=OLD.recipe_json OR NEW.created_at!=OLD.created_at OR NEW.state!='preregistered' OR NEW.preregistered_at IS NULL OR NOT EXISTS(SELECT 1 FROM training_datasets dataset JOIN character_benchmark_campaigns benchmark ON benchmark.character_pack_storage_id=dataset.character_pack_storage_id AND benchmark.reviewer_id=NEW.reviewer_id JOIN character_benchmark_results result ON result.campaign_id=benchmark.id WHERE dataset.storage_id=NEW.dataset_storage_id AND character_benchmark_provenance_is_valid(benchmark.id)=1 AND result.content_digest=json_extract(NEW.recipe_json,'$.stage_3_evidence_digest') AND json_extract(benchmark.definition_json,'$.character_pack_digest')=NEW.character_pack_digest AND json_extract(benchmark.definition_json,'$.stage_2_manifest_digest')=json_extract(NEW.recipe_json,'$.stage_2_manifest_digest') AND json_extract(benchmark.definition_json,'$.stage_2_evidence_digest')=json_extract(NEW.recipe_json,'$.stage_2_evidence_digest') AND (SELECT new_state FROM character_benchmark_events WHERE campaign_id=benchmark.id ORDER BY sequence DESC LIMIT 1)='completed') THEN RAISE(ABORT,'invalid training recipe preregistration') END; END;
CREATE TRIGGER guard_training_recipe_delete BEFORE DELETE ON training_recipes WHEN OLD.state='preregistered' BEGIN SELECT RAISE(ABORT,'preregistered training recipe is immutable'); END;
CREATE TRIGGER validate_training_run_insert BEFORE INSERT ON training_runs BEGIN SELECT CASE WHEN (SELECT state FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)!='preregistered' OR (SELECT dataset_storage_id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)!=NEW.dataset_storage_id OR NEW.total_steps!=(SELECT json_extract(recipe_json,'$.total_steps') FROM training_recipes WHERE storage_id=NEW.recipe_storage_id) THEN RAISE(ABORT,'training run identity mismatch') END; END;
CREATE TRIGGER validate_training_event_insert BEFORE INSERT ON training_run_events BEGIN
 SELECT CASE WHEN NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_run_events WHERE run_id=NEW.run_id),1) THEN RAISE(ABORT,'training event sequence is not append-only') END;
 SELECT CASE WHEN (SELECT state FROM training_runs WHERE id=NEW.run_id)!=NEW.previous_state OR NOT ((NEW.previous_state='created' AND NEW.new_state='running') OR (NEW.previous_state='running' AND NEW.new_state IN ('completed','failed','cancelled'))) THEN RAISE(ABORT,'invalid training lifecycle transition') END;
 SELECT CASE WHEN NEW.new_state='completed' AND (COALESCE((SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id),-1)!=(SELECT total_steps FROM training_runs WHERE id=NEW.run_id) OR NOT EXISTS(SELECT 1 FROM training_checkpoints WHERE run_id=NEW.run_id AND fraction=1.0)) THEN RAISE(ABORT,'training completion evidence is incomplete') END;
END;
CREATE TRIGGER apply_training_event AFTER INSERT ON training_run_events BEGIN UPDATE training_runs SET state=NEW.new_state,terminal_at=CASE WHEN NEW.new_state IN ('completed','failed','cancelled') THEN NEW.created_at ELSE NULL END WHERE id=NEW.run_id; END;
CREATE TRIGGER guard_training_event_update BEFORE UPDATE ON training_run_events BEGIN SELECT RAISE(ABORT,'training events are immutable'); END;
CREATE TRIGGER guard_training_event_delete BEFORE DELETE ON training_run_events BEGIN SELECT RAISE(ABORT,'training events are immutable'); END;
CREATE TRIGGER validate_training_telemetry_insert BEFORE INSERT ON training_telemetry BEGIN SELECT CASE WHEN (SELECT state FROM training_runs WHERE id=NEW.run_id)!='running' OR NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_telemetry WHERE run_id=NEW.run_id),1) OR NEW.step>(SELECT total_steps FROM training_runs WHERE id=NEW.run_id) OR NEW.step<=COALESCE((SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id),-1) THEN RAISE(ABORT,'invalid training telemetry sequence') END; END;
CREATE TRIGGER guard_training_telemetry_update BEFORE UPDATE ON training_telemetry BEGIN SELECT RAISE(ABORT,'training telemetry is immutable'); END;
CREATE TRIGGER guard_training_telemetry_delete BEFORE DELETE ON training_telemetry BEGIN SELECT RAISE(ABORT,'training telemetry is immutable'); END;
CREATE TRIGGER guard_training_failure_update BEFORE UPDATE ON training_failures BEGIN SELECT RAISE(ABORT,'training failures are immutable'); END;
CREATE TRIGGER guard_training_failure_delete BEFORE DELETE ON training_failures BEGIN SELECT RAISE(ABORT,'training failures are immutable'); END;
"""

_MIGRATION_24 = """
CREATE TABLE training_checkpoints(
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL, reviewer_id TEXT NOT NULL, recipe_storage_id TEXT NOT NULL, dataset_storage_id TEXT NOT NULL,
 step INTEGER NOT NULL CHECK(step>0), fraction REAL NOT NULL CHECK(fraction IN (0.25,0.5,0.75,1.0)), parent_checkpoint_id TEXT,
  artifact_digest TEXT NOT NULL CHECK(length(artifact_digest)=71 AND substr(artifact_digest,1,7)='sha256:'),
 manifest_digest TEXT NOT NULL CHECK(length(manifest_digest)=71 AND substr(manifest_digest,1,7)='sha256:'), manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json) AND json_type(manifest_json)='object'), created_at TEXT NOT NULL,
 UNIQUE(run_id,step), UNIQUE(run_id,fraction), UNIQUE(run_id,artifact_digest), UNIQUE(run_id,manifest_digest),
 FOREIGN KEY(run_id,recipe_storage_id) REFERENCES training_runs(id,recipe_storage_id),
 FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id),
 FOREIGN KEY(recipe_storage_id,dataset_storage_id) REFERENCES training_recipes(storage_id,dataset_storage_id),
 FOREIGN KEY(parent_checkpoint_id) REFERENCES training_checkpoints(id)
);
CREATE INDEX training_checkpoints_run ON training_checkpoints(run_id,step);
CREATE TRIGGER validate_training_checkpoint_insert BEFORE INSERT ON training_checkpoints BEGIN
 SELECT CASE WHEN NEW.dataset_storage_id!=(SELECT dataset_storage_id FROM training_runs WHERE id=NEW.run_id) OR NEW.step!=training_checkpoint_step((SELECT total_steps FROM training_runs WHERE id=NEW.run_id),NEW.fraction) OR NEW.step>(SELECT total_steps FROM training_runs WHERE id=NEW.run_id) OR (SELECT state FROM training_runs WHERE id=NEW.run_id)!='running' OR NEW.step>COALESCE((SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id),-1) OR NOT EXISTS(SELECT 1 FROM training_recipes recipe,json_each(recipe.recipe_json,'$.checkpoint_plan.fractions') declared WHERE recipe.storage_id=NEW.recipe_storage_id AND declared.value=NEW.fraction) OR json_extract(NEW.manifest_json,'$.schema_version')!='character-training-checkpoint.v1' OR json_extract(NEW.manifest_json,'$.run_id')!=NEW.run_id OR json_extract(NEW.manifest_json,'$.reviewer_id')!=NEW.reviewer_id OR json_extract(NEW.manifest_json,'$.recipe_id')!=(SELECT id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id) OR json_extract(NEW.manifest_json,'$.dataset_digest')!=(SELECT dataset_digest FROM training_recipes WHERE storage_id=NEW.recipe_storage_id) OR json_extract(NEW.manifest_json,'$.character_id')!=(SELECT character_id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id) OR json_extract(NEW.manifest_json,'$.step')!=NEW.step OR json_extract(NEW.manifest_json,'$.fraction')!=NEW.fraction OR json_extract(NEW.manifest_json,'$.parent_checkpoint_id') IS NOT NEW.parent_checkpoint_id OR json_extract(NEW.manifest_json,'$.artifact_digest')!=NEW.artifact_digest OR canonical_json_digest(NEW.manifest_json)!=NEW.manifest_digest THEN RAISE(ABORT,'training checkpoint identity mismatch') END;
 SELECT CASE WHEN (NEW.fraction=0.25 AND NEW.parent_checkpoint_id IS NOT NULL) OR (NEW.fraction>0.25 AND (SELECT run_id FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id) IS NOT NEW.run_id) OR (NEW.fraction>0.25 AND (SELECT step FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id)!=(SELECT max(step) FROM training_checkpoints WHERE run_id=NEW.run_id)) OR (NEW.fraction>0.25 AND (SELECT fraction FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id)!=NEW.fraction-0.25) THEN RAISE(ABORT,'training checkpoint parent mismatch') END;
END;
CREATE TRIGGER guard_training_checkpoint_update BEFORE UPDATE ON training_checkpoints BEGIN SELECT RAISE(ABORT,'training checkpoints are immutable'); END;
CREATE TRIGGER guard_training_checkpoint_delete BEFORE DELETE ON training_checkpoints BEGIN SELECT RAISE(ABORT,'training checkpoints are immutable'); END;
"""

_MIGRATION_25 = """
CREATE TABLE training_dataset_lineages(
  id TEXT PRIMARY KEY,
  reviewer_id TEXT NOT NULL,
  character_id TEXT NOT NULL,
  current_dataset_storage_id TEXT NOT NULL UNIQUE,
  head_version INTEGER NOT NULL CHECK(head_version>0),
  state TEXT NOT NULL CHECK(state IN ('draft','sealed')),
  created_at TEXT NOT NULL,
  UNIQUE(id,reviewer_id), UNIQUE(id,character_id),
  FOREIGN KEY(character_id,reviewer_id) REFERENCES characters(id,reviewer_id),
  FOREIGN KEY(current_dataset_storage_id,reviewer_id) REFERENCES training_datasets(storage_id,reviewer_id),
  FOREIGN KEY(current_dataset_storage_id,character_id) REFERENCES training_datasets(storage_id,character_id)
);
CREATE TABLE training_dataset_head_events(
  lineage_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  reviewer_id TEXT NOT NULL,
  character_id TEXT NOT NULL,
  previous_dataset_storage_id TEXT,
  dataset_storage_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('draft','sealed')),
  created_at TEXT NOT NULL,
  PRIMARY KEY(lineage_id,version),
  FOREIGN KEY(lineage_id,reviewer_id) REFERENCES training_dataset_lineages(id,reviewer_id),
  FOREIGN KEY(lineage_id,character_id) REFERENCES training_dataset_lineages(id,character_id),
  FOREIGN KEY(dataset_storage_id,reviewer_id) REFERENCES training_datasets(storage_id,reviewer_id),
  FOREIGN KEY(dataset_storage_id,character_id) REFERENCES training_datasets(storage_id,character_id),
  FOREIGN KEY(previous_dataset_storage_id) REFERENCES training_datasets(storage_id)
);
CREATE TABLE training_imported_samples(
  id TEXT PRIMARY KEY,
  reviewer_id TEXT NOT NULL,
  character_id TEXT NOT NULL,
  lineage_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL,
  artifact_digest TEXT NOT NULL CHECK(length(artifact_digest)=71 AND substr(artifact_digest,1,7)='sha256:'),
  payload_json TEXT NOT NULL CHECK(json_valid(payload_json) AND json_type(payload_json)='object'),
  created_at TEXT NOT NULL,
  UNIQUE(lineage_id,artifact_digest),
  FOREIGN KEY(lineage_id,reviewer_id) REFERENCES training_dataset_lineages(id,reviewer_id),
  FOREIGN KEY(lineage_id,character_id) REFERENCES training_dataset_lineages(id,character_id),
  FOREIGN KEY(artifact_id) REFERENCES artifacts(id)
);
CREATE TABLE training_curation_confirmations(
  id TEXT PRIMARY KEY,
  import_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  decision TEXT NOT NULL CHECK(decision IN ('accept','reject')),
  payload_json TEXT NOT NULL CHECK(json_valid(payload_json) AND json_type(payload_json)='object'),
  created_at TEXT NOT NULL,
  UNIQUE(import_id,reviewer_id,version),
  FOREIGN KEY(import_id) REFERENCES training_imported_samples(id)
);
CREATE INDEX training_imports_lineage ON training_imported_samples(lineage_id,created_at,id);
CREATE INDEX training_confirmations_import ON training_curation_confirmations(import_id,version);
CREATE TRIGGER validate_training_lineage_insert BEFORE INSERT ON training_dataset_lineages BEGIN
 SELECT CASE WHEN NEW.head_version!=1 OR NEW.state!='draft' OR NOT EXISTS(
  SELECT 1 FROM training_datasets d WHERE d.storage_id=NEW.current_dataset_storage_id
  AND d.reviewer_id=NEW.reviewer_id AND d.character_id=NEW.character_id AND d.state='draft'
 ) THEN RAISE(ABORT,'invalid initial training dataset head') END;
END;
CREATE TRIGGER validate_training_head_event_insert BEFORE INSERT ON training_dataset_head_events WHEN NEW.version>1 BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.character_id!=(SELECT character_id FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.version!=(SELECT head_version+1 FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.previous_dataset_storage_id!=(SELECT current_dataset_storage_id FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR (SELECT state FROM training_dataset_lineages WHERE id=NEW.lineage_id)!='draft'
 OR NOT EXISTS(SELECT 1 FROM training_datasets d WHERE d.storage_id=NEW.dataset_storage_id
   AND d.reviewer_id=NEW.reviewer_id AND d.character_id=NEW.character_id AND d.state=NEW.state)
 OR (NEW.state='draft' AND EXISTS(SELECT 1 FROM training_dataset_head_events old
   WHERE old.lineage_id=NEW.lineage_id AND old.dataset_storage_id=NEW.dataset_storage_id))
 OR (NEW.state='sealed' AND NEW.dataset_storage_id!=NEW.previous_dataset_storage_id)
 THEN RAISE(ABORT,'invalid training dataset head event') END;
END;
CREATE TRIGGER validate_initial_training_head_event BEFORE INSERT ON training_dataset_head_events WHEN NEW.version=1 BEGIN
 SELECT CASE WHEN NEW.previous_dataset_storage_id IS NOT NULL
 OR NEW.dataset_storage_id!=(SELECT current_dataset_storage_id FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.character_id!=(SELECT character_id FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.state!='draft' THEN RAISE(ABORT,'invalid initial training dataset head event') END;
END;
CREATE TRIGGER validate_training_lineage_update BEFORE UPDATE ON training_dataset_lineages BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id OR NEW.character_id!=OLD.character_id
 OR NEW.created_at!=OLD.created_at OR OLD.state!='draft' OR NEW.head_version!=OLD.head_version+1
 OR NOT EXISTS(SELECT 1 FROM training_dataset_head_events e WHERE e.lineage_id=OLD.id
   AND e.version=NEW.head_version AND e.previous_dataset_storage_id=OLD.current_dataset_storage_id
   AND e.dataset_storage_id=NEW.current_dataset_storage_id AND e.state=NEW.state)
 THEN RAISE(ABORT,'invalid training dataset head CAS') END;
END;
CREATE TRIGGER guard_training_lineage_delete BEFORE DELETE ON training_dataset_lineages BEGIN SELECT RAISE(ABORT,'training dataset lineages are immutable'); END;
CREATE TRIGGER guard_training_head_event_update BEFORE UPDATE ON training_dataset_head_events BEGIN SELECT RAISE(ABORT,'training dataset head history is immutable'); END;
CREATE TRIGGER guard_training_head_event_delete BEFORE DELETE ON training_dataset_head_events BEGIN SELECT RAISE(ABORT,'training dataset head history is immutable'); END;
CREATE TRIGGER validate_training_import_insert BEFORE INSERT ON training_imported_samples BEGIN
 SELECT CASE WHEN json_extract(NEW.payload_json,'$.import_id')!=NEW.id
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.character_id')!=NEW.character_id
 OR json_extract(NEW.payload_json,'$.lineage_id')!=NEW.lineage_id
 OR json_extract(NEW.payload_json,'$.artifact_id')!=NEW.artifact_id
 OR json_extract(NEW.payload_json,'$.artifact_digest')!=NEW.artifact_digest
 OR NOT EXISTS(SELECT 1 FROM artifacts a WHERE a.id=NEW.artifact_id AND 'sha256:'||a.sha256=NEW.artifact_digest)
 THEN RAISE(ABORT,'invalid imported training sample') END;
END;
CREATE TRIGGER guard_training_import_update BEFORE UPDATE ON training_imported_samples BEGIN SELECT RAISE(ABORT,'imported training samples are immutable'); END;
CREATE TRIGGER guard_training_import_delete BEFORE DELETE ON training_imported_samples BEGIN SELECT RAISE(ABORT,'imported training samples are immutable'); END;
CREATE TRIGGER validate_training_confirmation_insert BEFORE INSERT ON training_curation_confirmations BEGIN
 SELECT CASE WHEN json_extract(NEW.payload_json,'$.confirmation_id')!=NEW.id
 OR json_extract(NEW.payload_json,'$.import_id')!=NEW.import_id
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.version')!=NEW.version
 OR json_extract(NEW.payload_json,'$.decision')!=NEW.decision
 OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_imported_samples WHERE id=NEW.import_id)
 OR NEW.version!=COALESCE((SELECT max(version)+1 FROM training_curation_confirmations WHERE import_id=NEW.import_id AND reviewer_id=NEW.reviewer_id),1)
 THEN RAISE(ABORT,'invalid training curation confirmation') END;
END;
CREATE TRIGGER guard_training_confirmation_update BEFORE UPDATE ON training_curation_confirmations BEGIN SELECT RAISE(ABORT,'training confirmations are immutable'); END;
CREATE TRIGGER guard_training_confirmation_delete BEFORE DELETE ON training_curation_confirmations BEGIN SELECT RAISE(ABORT,'training confirmations are immutable'); END;
"""

_MIGRATION_26 = """
DROP TRIGGER validate_training_lineage_insert;
DROP TRIGGER validate_training_head_event_insert;
DROP TRIGGER validate_initial_training_head_event;
DROP TRIGGER validate_training_lineage_update;
DROP TRIGGER guard_training_head_event_update;
DROP TRIGGER guard_training_head_event_delete;
DROP TRIGGER validate_training_import_insert;
ALTER TABLE training_dataset_head_events RENAME TO training_dataset_head_events_v25;
CREATE TABLE training_dataset_snapshot_lineages(
  dataset_storage_id TEXT PRIMARY KEY,
  lineage_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  character_id TEXT NOT NULL,
  first_version INTEGER NOT NULL CHECK(first_version>0),
  FOREIGN KEY(lineage_id,reviewer_id) REFERENCES training_dataset_lineages(id,reviewer_id),
  FOREIGN KEY(lineage_id,character_id) REFERENCES training_dataset_lineages(id,character_id),
  FOREIGN KEY(dataset_storage_id,reviewer_id) REFERENCES training_datasets(storage_id,reviewer_id),
  FOREIGN KEY(dataset_storage_id,character_id) REFERENCES training_datasets(storage_id,character_id)
);
CREATE TABLE training_dataset_snapshots(
  lineage_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  reviewer_id TEXT NOT NULL,
  character_id TEXT NOT NULL,
  previous_dataset_storage_id TEXT,
  dataset_storage_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('draft','sealed')),
  created_at TEXT NOT NULL,
  PRIMARY KEY(lineage_id,version),
  FOREIGN KEY(lineage_id,reviewer_id) REFERENCES training_dataset_lineages(id,reviewer_id),
  FOREIGN KEY(lineage_id,character_id) REFERENCES training_dataset_lineages(id,character_id),
  FOREIGN KEY(dataset_storage_id,reviewer_id) REFERENCES training_datasets(storage_id,reviewer_id),
  FOREIGN KEY(dataset_storage_id,character_id) REFERENCES training_datasets(storage_id,character_id),
  FOREIGN KEY(previous_dataset_storage_id) REFERENCES training_datasets(storage_id)
);
INSERT INTO training_dataset_snapshot_lineages
 SELECT dataset_storage_id,lineage_id,reviewer_id,character_id,min(version)
 FROM training_dataset_head_events_v25 GROUP BY dataset_storage_id;
INSERT INTO training_dataset_snapshots
 SELECT lineage_id,version,reviewer_id,character_id,previous_dataset_storage_id,
 dataset_storage_id,state,created_at FROM training_dataset_head_events_v25;
DROP TABLE training_dataset_head_events_v25;
CREATE VIEW training_dataset_head_events AS SELECT * FROM training_dataset_snapshots;
CREATE TRIGGER reject_training_head_event_insert INSTEAD OF INSERT ON training_dataset_head_events
 BEGIN SELECT RAISE(ABORT,'training dataset head history is derived'); END;
CREATE TRIGGER reject_training_head_event_update INSTEAD OF UPDATE ON training_dataset_head_events
 BEGIN SELECT RAISE(ABORT,'training dataset head history is derived'); END;
CREATE TRIGGER reject_training_head_event_delete INSTEAD OF DELETE ON training_dataset_head_events
 BEGIN SELECT RAISE(ABORT,'training dataset head history is derived'); END;
CREATE TRIGGER validate_training_lineage_insert_v26 BEFORE INSERT ON training_dataset_lineages BEGIN
 SELECT CASE WHEN NEW.head_version!=1 OR NEW.state!='draft' OR EXISTS(
  SELECT 1 FROM training_dataset_snapshot_lineages
  WHERE dataset_storage_id=NEW.current_dataset_storage_id
 ) OR NOT EXISTS(SELECT 1 FROM training_datasets d WHERE d.storage_id=NEW.current_dataset_storage_id
  AND d.reviewer_id=NEW.reviewer_id AND d.character_id=NEW.character_id AND d.state='draft')
 THEN RAISE(ABORT,'invalid initial training dataset head') END;
END;
CREATE TRIGGER derive_initial_training_head_v26 AFTER INSERT ON training_dataset_lineages BEGIN
 INSERT INTO training_dataset_snapshot_lineages VALUES(
  NEW.current_dataset_storage_id,NEW.id,NEW.reviewer_id,NEW.character_id,1
 );
 INSERT INTO training_dataset_snapshots VALUES(
  NEW.id,1,NEW.reviewer_id,NEW.character_id,NULL,NEW.current_dataset_storage_id,'draft',NEW.created_at
 );
END;
CREATE TRIGGER validate_training_lineage_update_v26 BEFORE UPDATE ON training_dataset_lineages BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id
 OR NEW.character_id!=OLD.character_id OR NEW.created_at!=OLD.created_at OR OLD.state!='draft'
 OR NEW.head_version!=OLD.head_version+1
 OR NOT EXISTS(SELECT 1 FROM training_dataset_snapshots old_head
   WHERE old_head.lineage_id=OLD.id AND old_head.version=OLD.head_version
   AND old_head.dataset_storage_id=OLD.current_dataset_storage_id)
 OR NOT EXISTS(SELECT 1 FROM training_datasets d WHERE d.storage_id=NEW.current_dataset_storage_id
   AND d.reviewer_id=NEW.reviewer_id AND d.character_id=NEW.character_id AND d.state=NEW.state)
 OR (NEW.state='draft' AND NEW.current_dataset_storage_id=OLD.current_dataset_storage_id)
 OR (NEW.state='sealed' AND NEW.current_dataset_storage_id!=OLD.current_dataset_storage_id)
 OR (NEW.state='draft' AND EXISTS(SELECT 1 FROM training_dataset_snapshot_lineages used
   WHERE used.dataset_storage_id=NEW.current_dataset_storage_id))
 THEN RAISE(ABORT,'invalid training dataset head CAS') END;
END;
CREATE TRIGGER derive_training_head_v26 AFTER UPDATE ON training_dataset_lineages BEGIN
 INSERT INTO training_dataset_snapshot_lineages
  SELECT NEW.current_dataset_storage_id,NEW.id,NEW.reviewer_id,NEW.character_id,NEW.head_version
  WHERE NEW.state='draft';
 INSERT INTO training_dataset_snapshots VALUES(
  NEW.id,NEW.head_version,NEW.reviewer_id,NEW.character_id,OLD.current_dataset_storage_id,
  NEW.current_dataset_storage_id,NEW.state,strftime('%Y-%m-%dT%H:%M:%fZ','now')
 );
END;
CREATE TRIGGER validate_training_snapshot_insert_v26 BEFORE INSERT ON training_dataset_snapshots BEGIN
 SELECT CASE WHEN NEW.lineage_id!=(SELECT id FROM training_dataset_lineages
   WHERE id=NEW.lineage_id AND reviewer_id=NEW.reviewer_id AND character_id=NEW.character_id)
 OR NEW.version!=(SELECT head_version FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.dataset_storage_id!=(SELECT current_dataset_storage_id FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR NEW.state!=(SELECT state FROM training_dataset_lineages WHERE id=NEW.lineage_id)
 OR (NEW.version=1 AND NEW.previous_dataset_storage_id IS NOT NULL)
 OR (NEW.version>1 AND NEW.previous_dataset_storage_id!=(SELECT dataset_storage_id
   FROM training_dataset_snapshots WHERE lineage_id=NEW.lineage_id AND version=NEW.version-1))
 THEN RAISE(ABORT,'invalid derived training dataset history') END;
END;
CREATE TRIGGER guard_training_snapshot_update_v26 BEFORE UPDATE ON training_dataset_snapshots
 BEGIN SELECT RAISE(ABORT,'training dataset history is immutable'); END;
CREATE TRIGGER guard_training_snapshot_delete_v26 BEFORE DELETE ON training_dataset_snapshots
 BEGIN SELECT RAISE(ABORT,'training dataset history is immutable'); END;
CREATE TRIGGER guard_training_snapshot_lineage_update_v26
 BEFORE UPDATE ON training_dataset_snapshot_lineages
 BEGIN SELECT RAISE(ABORT,'training dataset lineage binding is immutable'); END;
CREATE TRIGGER guard_training_snapshot_lineage_delete_v26
 BEFORE DELETE ON training_dataset_snapshot_lineages
 BEGIN SELECT RAISE(ABORT,'training dataset lineage binding is immutable'); END;
CREATE TRIGGER validate_training_import_insert_v26 BEFORE INSERT ON training_imported_samples BEGIN
 SELECT CASE WHEN json_extract(NEW.payload_json,'$.import_id')!=NEW.id
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.character_id')!=NEW.character_id
 OR json_extract(NEW.payload_json,'$.lineage_id')!=NEW.lineage_id
 OR json_extract(NEW.payload_json,'$.artifact_id')!=NEW.artifact_id
 OR json_extract(NEW.payload_json,'$.artifact_digest')!=NEW.artifact_digest
 OR json_extract(NEW.payload_json,'$.media.sha256')!=NEW.artifact_digest
 OR json_extract(NEW.payload_json,'$.artifact_byte_size')!=json_extract(NEW.payload_json,'$.media.byte_size')
 OR NOT EXISTS(SELECT 1 FROM artifacts a WHERE a.id=NEW.artifact_id
   AND 'sha256:'||a.sha256=NEW.artifact_digest
   AND a.byte_size=json_extract(NEW.payload_json,'$.artifact_byte_size')
   AND a.media_type=json_extract(NEW.payload_json,'$.media.media_type'))
 THEN RAISE(ABORT,'invalid imported training sample') END;
END;
"""

_MIGRATION_27 = """
CREATE TRIGGER authorize_training_snapshot_lineage_insert_v27
 BEFORE INSERT ON training_dataset_snapshot_lineages BEGIN
 SELECT CASE WHEN training_transition_authorized(
   NEW.lineage_id,NEW.dataset_storage_id,NEW.reviewer_id,NEW.character_id,NEW.first_version
 )!=1 THEN RAISE(ABORT,'training snapshot lineage transition is not authorized') END;
END;
DROP TRIGGER validate_training_import_insert_v26;
CREATE TRIGGER validate_training_import_insert_v27 BEFORE INSERT ON training_imported_samples BEGIN
 SELECT CASE WHEN json_extract(NEW.payload_json,'$.import_id')!=NEW.id
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.character_id')!=NEW.character_id
 OR json_extract(NEW.payload_json,'$.lineage_id')!=NEW.lineage_id
 OR json_extract(NEW.payload_json,'$.artifact_id')!=NEW.artifact_id
 OR json_extract(NEW.payload_json,'$.artifact_digest')!=NEW.artifact_digest
 OR json_extract(NEW.payload_json,'$.media.sha256')!=NEW.artifact_digest
 OR json_extract(NEW.payload_json,'$.imported_at')!=NEW.created_at
 OR json_extract(NEW.payload_json,'$.artifact_byte_size')!=json_extract(NEW.payload_json,'$.media.byte_size')
 OR NOT EXISTS(SELECT 1 FROM artifacts a WHERE a.id=NEW.artifact_id
   AND 'sha256:'||a.sha256=NEW.artifact_digest
   AND a.byte_size=json_extract(NEW.payload_json,'$.artifact_byte_size')
   AND a.media_type=json_extract(NEW.payload_json,'$.media.media_type'))
 THEN RAISE(ABORT,'invalid imported training sample') END;
END;
DROP TRIGGER validate_training_confirmation_insert;
CREATE TRIGGER validate_training_confirmation_insert_v27
 BEFORE INSERT ON training_curation_confirmations BEGIN
 SELECT CASE WHEN json_extract(NEW.payload_json,'$.confirmation_id')!=NEW.id
 OR json_extract(NEW.payload_json,'$.import_id')!=NEW.import_id
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.version')!=NEW.version
 OR json_extract(NEW.payload_json,'$.decision')!=NEW.decision
 OR json_extract(NEW.payload_json,'$.confirmed_at')!=NEW.created_at
 OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_imported_samples WHERE id=NEW.import_id)
 OR NEW.version!=COALESCE((SELECT max(version)+1 FROM training_curation_confirmations
   WHERE import_id=NEW.import_id AND reviewer_id=NEW.reviewer_id),1)
 THEN RAISE(ABORT,'invalid training curation confirmation') END;
END;
"""

_MIGRATION_28 = """
ALTER TABLE training_datasets ADD COLUMN lineage_id TEXT;
DROP TRIGGER guard_training_dataset_other_update;
UPDATE training_datasets SET lineage_id=(
 SELECT binding.lineage_id FROM training_dataset_snapshot_lineages binding
 WHERE binding.dataset_storage_id=training_datasets.storage_id
) WHERE EXISTS(SELECT 1 FROM training_dataset_snapshot_lineages binding
 WHERE binding.dataset_storage_id=training_datasets.storage_id);
CREATE TRIGGER guard_training_dataset_other_update BEFORE UPDATE ON training_datasets
 WHEN OLD.state IN ('draft','sealed') AND NOT (OLD.state='draft' AND NEW.state='sealed')
 BEGIN SELECT RAISE(ABORT,'training dataset snapshots are immutable'); END;
DROP TRIGGER authorize_training_snapshot_lineage_insert_v27;
DROP TRIGGER guard_training_snapshot_lineage_update_v26;
DROP TRIGGER guard_training_snapshot_lineage_delete_v26;
ALTER TABLE training_dataset_snapshot_lineages RENAME TO training_dataset_snapshot_lineages_v27;
CREATE VIEW training_dataset_snapshot_lineages AS
 SELECT dataset.storage_id AS dataset_storage_id,dataset.lineage_id,
 dataset.reviewer_id,dataset.character_id,min(history.version) AS first_version
 FROM training_datasets dataset JOIN training_dataset_snapshots history
 ON history.dataset_storage_id=dataset.storage_id AND history.lineage_id=dataset.lineage_id
 WHERE dataset.lineage_id IS NOT NULL GROUP BY dataset.storage_id;
DROP TABLE training_dataset_snapshot_lineages_v27;
CREATE TRIGGER reject_training_snapshot_lineage_insert_v28
 INSTEAD OF INSERT ON training_dataset_snapshot_lineages
 BEGIN SELECT RAISE(ABORT,'training snapshot lineage is derived'); END;
CREATE TRIGGER reject_training_snapshot_lineage_update_v28
 INSTEAD OF UPDATE ON training_dataset_snapshot_lineages
 BEGIN SELECT RAISE(ABORT,'training snapshot lineage is derived'); END;
CREATE TRIGGER reject_training_snapshot_lineage_delete_v28
 INSTEAD OF DELETE ON training_dataset_snapshot_lineages
 BEGIN SELECT RAISE(ABORT,'training snapshot lineage is derived'); END;
DROP TRIGGER validate_training_lineage_insert_v26;
DROP TRIGGER derive_initial_training_head_v26;
DROP TRIGGER validate_training_lineage_update_v26;
DROP TRIGGER derive_training_head_v26;
CREATE TRIGGER guard_training_dataset_lineage_update_v28
 BEFORE UPDATE OF lineage_id ON training_datasets
 WHEN NEW.lineage_id IS NOT OLD.lineage_id
 BEGIN SELECT RAISE(ABORT,'training dataset declared lineage is immutable'); END;
CREATE TRIGGER validate_training_lineage_insert_v28 BEFORE INSERT ON training_dataset_lineages BEGIN
 SELECT CASE WHEN NEW.head_version!=1 OR NEW.state!='draft' OR NOT EXISTS(
  SELECT 1 FROM training_datasets d WHERE d.storage_id=NEW.current_dataset_storage_id
  AND d.reviewer_id=NEW.reviewer_id AND d.character_id=NEW.character_id
  AND d.state='draft' AND d.lineage_id=NEW.id
 ) THEN RAISE(ABORT,'invalid initial training dataset head') END;
END;
CREATE TRIGGER derive_initial_training_head_v28 AFTER INSERT ON training_dataset_lineages BEGIN
 INSERT INTO training_dataset_snapshots VALUES(
  NEW.id,1,NEW.reviewer_id,NEW.character_id,NULL,NEW.current_dataset_storage_id,'draft',NEW.created_at
 );
END;
CREATE TRIGGER validate_training_lineage_update_v28 BEFORE UPDATE ON training_dataset_lineages BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id
 OR NEW.character_id!=OLD.character_id OR NEW.created_at!=OLD.created_at OR OLD.state!='draft'
 OR NEW.head_version!=OLD.head_version+1
 OR NOT EXISTS(SELECT 1 FROM training_dataset_snapshots old_head
   WHERE old_head.lineage_id=OLD.id AND old_head.version=OLD.head_version
   AND old_head.dataset_storage_id=OLD.current_dataset_storage_id)
 OR NOT EXISTS(SELECT 1 FROM training_datasets d WHERE d.storage_id=NEW.current_dataset_storage_id
   AND d.reviewer_id=NEW.reviewer_id AND d.character_id=NEW.character_id
   AND d.state=NEW.state AND d.lineage_id=NEW.id)
 OR (NEW.state='draft' AND NEW.current_dataset_storage_id=OLD.current_dataset_storage_id)
 OR (NEW.state='sealed' AND NEW.current_dataset_storage_id!=OLD.current_dataset_storage_id)
 OR (NEW.state='draft' AND EXISTS(SELECT 1 FROM training_dataset_snapshots used
   WHERE used.dataset_storage_id=NEW.current_dataset_storage_id))
 THEN RAISE(ABORT,'invalid training dataset head CAS') END;
END;
CREATE TRIGGER derive_training_head_v28 AFTER UPDATE ON training_dataset_lineages BEGIN
 INSERT INTO training_dataset_snapshots VALUES(
  NEW.id,NEW.head_version,NEW.reviewer_id,NEW.character_id,OLD.current_dataset_storage_id,
  NEW.current_dataset_storage_id,NEW.state,strftime('%Y-%m-%dT%H:%M:%fZ','now')
 );
END;
"""

_MIGRATION_29 = """
CREATE TABLE training_dataset_snapshot_imports(
 dataset_storage_id TEXT NOT NULL,
 import_id TEXT NOT NULL,
 PRIMARY KEY(dataset_storage_id,import_id),
 FOREIGN KEY(dataset_storage_id) REFERENCES training_datasets(storage_id),
 FOREIGN KEY(import_id) REFERENCES training_imported_samples(id)
);
CREATE TABLE training_dataset_snapshot_confirmations(
 dataset_storage_id TEXT NOT NULL,
 confirmation_id TEXT NOT NULL,
 PRIMARY KEY(dataset_storage_id,confirmation_id),
 FOREIGN KEY(dataset_storage_id) REFERENCES training_datasets(storage_id),
 FOREIGN KEY(confirmation_id) REFERENCES training_curation_confirmations(id)
);
CREATE TRIGGER validate_training_ledger_snapshot_v29 BEFORE INSERT ON training_datasets
WHEN NEW.lineage_id IS NOT NULL BEGIN
 SELECT CASE WHEN EXISTS(
  SELECT 1 FROM json_each(NEW.manifest_json,'$.imported_sample_ids') ledger
  LEFT JOIN training_imported_samples imported ON imported.id=ledger.value
  WHERE imported.id IS NULL OR imported.reviewer_id!=NEW.reviewer_id
   OR imported.character_id!=NEW.character_id OR imported.lineage_id!=NEW.lineage_id
 ) OR EXISTS(
  SELECT 1 FROM json_each(NEW.manifest_json,'$.curation_confirmation_ids') ledger
  LEFT JOIN training_curation_confirmations confirmation ON confirmation.id=ledger.value
  LEFT JOIN training_imported_samples imported ON imported.id=confirmation.import_id
  WHERE confirmation.id IS NULL OR confirmation.reviewer_id!=NEW.reviewer_id
   OR imported.character_id!=NEW.character_id OR imported.lineage_id!=NEW.lineage_id
   OR NOT EXISTS(SELECT 1 FROM json_each(NEW.manifest_json,'$.imported_sample_ids') ids
                 WHERE ids.value=confirmation.import_id)
 ) OR EXISTS(
  SELECT confirmation.import_id
  FROM json_each(NEW.manifest_json,'$.curation_confirmation_ids') ledger
  JOIN training_curation_confirmations confirmation ON confirmation.id=ledger.value
  GROUP BY confirmation.import_id HAVING count(*)>1
 ) THEN RAISE(ABORT,'training dataset ledger record mismatch') END;
 SELECT CASE WHEN EXISTS(
  SELECT 1 FROM json_each(NEW.manifest_json,'$.curation_confirmation_ids') ledger
  JOIN training_curation_confirmations confirmation ON confirmation.id=ledger.value
  JOIN training_imported_samples imported ON imported.id=confirmation.import_id
  WHERE (confirmation.decision='reject' AND EXISTS(
    SELECT 1 FROM json_each(NEW.manifest_json,'$.samples') sample
    WHERE json_extract(sample.value,'$.sample_digest')=imported.artifact_digest))
  OR (confirmation.decision='accept' AND NOT EXISTS(
    SELECT 1 FROM json_each(NEW.manifest_json,'$.samples') sample
    WHERE json_extract(sample.value,'$.sample_id')=json_extract(confirmation.payload_json,'$.sample_id')
     AND json_extract(sample.value,'$.sample_digest')=imported.artifact_digest
     AND json_extract(sample.value,'$.source_tier')=json_extract(confirmation.payload_json,'$.source_tier')
     AND json_extract(sample.value,'$.source_lineage')=json_extract(confirmation.payload_json,'$.source_lineage')
     AND json_extract(sample.value,'$.caption')=json_extract(confirmation.payload_json,'$.caption')
     AND json_extract(sample.value,'$.split')=json_extract(confirmation.payload_json,'$.split')
     AND canonical_json_digest(json_extract(sample.value,'$.media'))=canonical_json_digest(json_extract(imported.payload_json,'$.media'))
     AND canonical_json_digest(json_extract(sample.value,'$.annotation'))=canonical_json_digest(json_extract(confirmation.payload_json,'$.annotation'))
     AND json_extract(sample.value,'$.accepted')=1))
 ) THEN RAISE(ABORT,'training dataset confirmation/sample mismatch') END;
END;
CREATE TRIGGER materialize_training_snapshot_ledgers_v29 AFTER INSERT ON training_datasets
WHEN NEW.lineage_id IS NOT NULL BEGIN
 INSERT INTO training_dataset_snapshot_imports SELECT NEW.storage_id,value
  FROM json_each(NEW.manifest_json,'$.imported_sample_ids');
 INSERT INTO training_dataset_snapshot_confirmations SELECT NEW.storage_id,value
  FROM json_each(NEW.manifest_json,'$.curation_confirmation_ids');
END;
CREATE TRIGGER validate_training_snapshot_import_insert_v29 BEFORE INSERT ON training_dataset_snapshot_imports BEGIN
 SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM training_datasets dataset,
  json_each(dataset.manifest_json,'$.imported_sample_ids') ledger
  WHERE dataset.storage_id=NEW.dataset_storage_id AND ledger.value=NEW.import_id)
 THEN RAISE(ABORT,'training snapshot import binding is not canonical') END;
END;
CREATE TRIGGER validate_training_snapshot_confirmation_insert_v29 BEFORE INSERT ON training_dataset_snapshot_confirmations BEGIN
 SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM training_datasets dataset,
  json_each(dataset.manifest_json,'$.curation_confirmation_ids') ledger
  WHERE dataset.storage_id=NEW.dataset_storage_id AND ledger.value=NEW.confirmation_id)
 THEN RAISE(ABORT,'training snapshot confirmation binding is not canonical') END;
END;
CREATE TRIGGER guard_training_snapshot_import_update_v29 BEFORE UPDATE ON training_dataset_snapshot_imports BEGIN SELECT RAISE(ABORT,'training snapshot import bindings are immutable'); END;
CREATE TRIGGER guard_training_snapshot_import_delete_v29 BEFORE DELETE ON training_dataset_snapshot_imports BEGIN SELECT RAISE(ABORT,'training snapshot import bindings are immutable'); END;
CREATE TRIGGER guard_training_snapshot_confirmation_update_v29 BEFORE UPDATE ON training_dataset_snapshot_confirmations BEGIN SELECT RAISE(ABORT,'training snapshot confirmation bindings are immutable'); END;
CREATE TRIGGER guard_training_snapshot_confirmation_delete_v29 BEFORE DELETE ON training_dataset_snapshot_confirmations BEGIN SELECT RAISE(ABORT,'training snapshot confirmation bindings are immutable'); END;
CREATE TRIGGER validate_training_ledger_head_advance_v29 BEFORE UPDATE ON training_dataset_lineages
WHEN NEW.state='draft' AND (
 EXISTS(SELECT 1 FROM training_dataset_snapshot_imports WHERE dataset_storage_id=OLD.current_dataset_storage_id)
 OR EXISTS(SELECT 1 FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=OLD.current_dataset_storage_id)
 OR EXISTS(SELECT 1 FROM training_dataset_snapshot_imports WHERE dataset_storage_id=NEW.current_dataset_storage_id)
 OR EXISTS(SELECT 1 FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=NEW.current_dataset_storage_id)
) BEGIN
 SELECT CASE WHEN EXISTS(
  SELECT import_id FROM training_dataset_snapshot_imports WHERE dataset_storage_id=OLD.current_dataset_storage_id
  EXCEPT SELECT import_id FROM training_dataset_snapshot_imports WHERE dataset_storage_id=NEW.current_dataset_storage_id
 ) OR EXISTS(
  SELECT confirmation_id FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=OLD.current_dataset_storage_id
  EXCEPT SELECT confirmation_id FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=NEW.current_dataset_storage_id
 ) THEN RAISE(ABORT,'training dataset ledger rollback') END;
 SELECT CASE WHEN
  (SELECT count(*) FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=NEW.current_dataset_storage_id)>
  (SELECT count(*) FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=OLD.current_dataset_storage_id)
  AND ((SELECT count(*) FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=NEW.current_dataset_storage_id)!=
       (SELECT count(*)+1 FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=OLD.current_dataset_storage_id)
   OR (SELECT count(*) FROM training_dataset_snapshot_imports WHERE dataset_storage_id=NEW.current_dataset_storage_id)!=
      (SELECT count(*) FROM training_dataset_snapshot_imports WHERE dataset_storage_id=OLD.current_dataset_storage_id))
 THEN RAISE(ABORT,'invalid training confirmation action') END;
 SELECT CASE WHEN
  (SELECT count(*) FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=NEW.current_dataset_storage_id)=
  (SELECT count(*) FROM training_dataset_snapshot_confirmations WHERE dataset_storage_id=OLD.current_dataset_storage_id)
  AND (SELECT count(*) FROM training_dataset_snapshot_imports WHERE dataset_storage_id=NEW.current_dataset_storage_id)<=
      (SELECT count(*) FROM training_dataset_snapshot_imports WHERE dataset_storage_id=OLD.current_dataset_storage_id)
 THEN RAISE(ABORT,'training dataset ledger action is empty') END;
END;
DROP TRIGGER validate_training_import_insert_v27;
CREATE TRIGGER validate_training_import_insert_v29 BEFORE INSERT ON training_imported_samples BEGIN
 SELECT CASE WHEN training_import_identity_is_valid(NEW.id,NEW.payload_json)!=1
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.character_id')!=NEW.character_id
 OR json_extract(NEW.payload_json,'$.lineage_id')!=NEW.lineage_id
 OR json_extract(NEW.payload_json,'$.artifact_id')!=NEW.artifact_id
 OR json_extract(NEW.payload_json,'$.artifact_digest')!=NEW.artifact_digest
 OR json_extract(NEW.payload_json,'$.imported_at')!=NEW.created_at
 OR NOT EXISTS(SELECT 1 FROM artifacts a WHERE a.id=NEW.artifact_id
   AND 'sha256:'||a.sha256=NEW.artifact_digest
   AND a.byte_size=json_extract(NEW.payload_json,'$.artifact_byte_size')
   AND a.media_type=json_extract(NEW.payload_json,'$.media.media_type'))
 THEN RAISE(ABORT,'invalid imported training sample') END;
END;
DROP TRIGGER validate_training_confirmation_insert_v27;
CREATE TRIGGER validate_training_confirmation_insert_v29 BEFORE INSERT ON training_curation_confirmations BEGIN
 SELECT CASE WHEN training_confirmation_identity_is_valid(NEW.id,NEW.payload_json)!=1
 OR json_extract(NEW.payload_json,'$.import_id')!=NEW.import_id
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.version')!=NEW.version
 OR json_extract(NEW.payload_json,'$.decision')!=NEW.decision
 OR json_extract(NEW.payload_json,'$.confirmed_at')!=NEW.created_at
 OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_imported_samples WHERE id=NEW.import_id)
 OR NEW.version!=COALESCE((SELECT max(version)+1 FROM training_curation_confirmations WHERE import_id=NEW.import_id AND reviewer_id=NEW.reviewer_id),1)
 THEN RAISE(ABORT,'invalid training curation confirmation') END;
END;
"""

_MIGRATION_30 = """
CREATE TRIGGER validate_complete_training_ledger_snapshot_v30
 BEFORE INSERT ON training_datasets WHEN NEW.lineage_id IS NOT NULL BEGIN
 SELECT CASE WHEN training_ledger_snapshot_is_complete(
  NEW.manifest_json,NEW.lineage_id,NEW.reviewer_id,NEW.character_id
 )!=1 THEN RAISE(ABORT,'training dataset ledger is incomplete') END;
END;
DROP TRIGGER validate_training_ledger_head_advance_v29;
CREATE TRIGGER validate_training_ledger_head_advance_v30
 BEFORE UPDATE ON training_dataset_lineages BEGIN
 SELECT CASE WHEN training_ledger_transition_is_valid(
  OLD.current_dataset_storage_id,NEW.current_dataset_storage_id,NEW.state
 )!=1 THEN RAISE(ABORT,'invalid training dataset ledger transition') END;
END;
DROP TRIGGER validate_training_confirmation_insert_v29;
CREATE TRIGGER validate_training_confirmation_insert_v30
 BEFORE INSERT ON training_curation_confirmations BEGIN
 SELECT CASE WHEN training_confirmation_identity_is_valid(NEW.id,NEW.payload_json)!=1
 OR json_extract(NEW.payload_json,'$.import_id')!=NEW.import_id
 OR json_extract(NEW.payload_json,'$.reviewer_id')!=NEW.reviewer_id
 OR json_extract(NEW.payload_json,'$.version')!=NEW.version
 OR json_extract(NEW.payload_json,'$.decision')!=NEW.decision
 OR json_extract(NEW.payload_json,'$.confirmed_at')!=NEW.created_at
 OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_imported_samples WHERE id=NEW.import_id)
 OR NEW.version!=1
 OR EXISTS(SELECT 1 FROM training_curation_confirmations WHERE import_id=NEW.import_id)
 THEN RAISE(ABORT,'invalid training curation confirmation') END;
END;
"""

_MIGRATION_31 = """
CREATE TRIGGER validate_training_ledger_seal_v31
 BEFORE UPDATE ON training_dataset_lineages WHEN NEW.state='sealed' BEGIN
 SELECT CASE WHEN training_ledger_snapshot_is_complete(
  (SELECT manifest_json FROM training_datasets WHERE storage_id=NEW.current_dataset_storage_id),
  NEW.id,NEW.reviewer_id,NEW.character_id
 )!=1 THEN RAISE(ABORT,'sealed training dataset ledger is incomplete') END;
END;
CREATE TRIGGER reject_training_import_for_sealed_lineage_v31
 BEFORE INSERT ON training_imported_samples
 WHEN (SELECT state FROM training_dataset_lineages WHERE id=NEW.lineage_id)='sealed'
 BEGIN SELECT RAISE(ABORT,'sealed training lineage rejects imports'); END;
CREATE TRIGGER reject_training_confirmation_for_sealed_lineage_v31
 BEFORE INSERT ON training_curation_confirmations
 WHEN (SELECT lineage.state FROM training_imported_samples imported
       JOIN training_dataset_lineages lineage ON lineage.id=imported.lineage_id
       WHERE imported.id=NEW.import_id)='sealed'
 BEGIN SELECT RAISE(ABORT,'sealed training lineage rejects confirmations'); END;
"""

_MIGRATION_32 = """
CREATE TABLE training_recipe_materials(
 recipe_storage_id TEXT PRIMARY KEY,
 reviewer_id TEXT NOT NULL,
 base_model_artifact_id TEXT NOT NULL,
 lightning_adapter_artifact_ids_json TEXT NOT NULL CHECK(json_valid(lightning_adapter_artifact_ids_json) AND json_type(lightning_adapter_artifact_ids_json)='object'),
 source_snapshot_artifact_id TEXT NOT NULL,
 stage_3_campaign_id TEXT NOT NULL,
 stage_3_result_id TEXT NOT NULL,
 runtime_digest TEXT NOT NULL CHECK(length(runtime_digest)=71 AND substr(runtime_digest,1,7)='sha256:'),
 FOREIGN KEY(recipe_storage_id,reviewer_id) REFERENCES training_recipes(storage_id,reviewer_id),
 FOREIGN KEY(base_model_artifact_id) REFERENCES artifacts(id),
 FOREIGN KEY(source_snapshot_artifact_id) REFERENCES artifacts(id),
 FOREIGN KEY(stage_3_campaign_id) REFERENCES character_benchmark_campaigns(id)
);
CREATE TRIGGER guard_training_recipe_material_update_v32 BEFORE UPDATE ON training_recipe_materials BEGIN SELECT RAISE(ABORT,'training recipe materials are immutable'); END;
CREATE TRIGGER guard_training_recipe_material_delete_v32 BEFORE DELETE ON training_recipe_materials BEGIN SELECT RAISE(ABORT,'training recipe materials are immutable'); END;
"""

_MIGRATION_33 = """
CREATE UNIQUE INDEX training_recipe_materials_owner_v33
ON training_recipe_materials(recipe_storage_id,reviewer_id);
CREATE TABLE training_recipe_lightning_materials(
 recipe_storage_id TEXT NOT NULL,
 reviewer_id TEXT NOT NULL,
 material_name TEXT NOT NULL CHECK(length(material_name)>0),
 artifact_id TEXT NOT NULL,
 PRIMARY KEY(recipe_storage_id,material_name),
 FOREIGN KEY(recipe_storage_id,reviewer_id)
  REFERENCES training_recipe_materials(recipe_storage_id,reviewer_id),
 FOREIGN KEY(artifact_id) REFERENCES artifacts(id)
);
INSERT INTO training_recipe_lightning_materials
SELECT material.recipe_storage_id,material.reviewer_id,item.key,item.value
FROM training_recipe_materials material,json_each(material.lightning_adapter_artifact_ids_json) item;
CREATE TRIGGER validate_training_recipe_material_insert_v33
BEFORE INSERT ON training_recipe_materials BEGIN
 SELECT CASE WHEN NOT EXISTS(
  SELECT 1 FROM training_recipes recipe
  JOIN artifacts base ON base.id=NEW.base_model_artifact_id
  JOIN artifacts source ON source.id=NEW.source_snapshot_artifact_id
  JOIN character_benchmark_campaigns campaign ON campaign.id=NEW.stage_3_campaign_id
   AND campaign.reviewer_id=NEW.reviewer_id
  JOIN character_benchmark_results result ON result.campaign_id=campaign.id
  WHERE recipe.storage_id=NEW.recipe_storage_id AND recipe.reviewer_id=NEW.reviewer_id
   AND 'sha256:'||base.sha256=json_extract(recipe.recipe_json,'$.base_model_digest')
   AND 'sha256:'||source.sha256=json_extract(recipe.recipe_json,'$.source_code_digest')
   AND NEW.runtime_digest=canonical_json_digest(json_extract(recipe.recipe_json,'$.runtime_identity'))
   AND result.content_digest=NEW.stage_3_result_id
   AND result.content_digest=json_extract(recipe.recipe_json,'$.stage_3_evidence_digest')
 ) THEN RAISE(ABORT,'training recipe material provenance mismatch') END;
END;
CREATE TRIGGER validate_training_recipe_lightning_insert_v33
BEFORE INSERT ON training_recipe_lightning_materials BEGIN
 SELECT CASE WHEN NOT EXISTS(
  SELECT 1 FROM training_recipes recipe JOIN artifacts artifact ON artifact.id=NEW.artifact_id
  WHERE recipe.storage_id=NEW.recipe_storage_id AND recipe.reviewer_id=NEW.reviewer_id
   AND 'sha256:'||artifact.sha256=json_extract(
    recipe.recipe_json,'$.lightning_adapter_digests.'||NEW.material_name)
 ) THEN RAISE(ABORT,'training recipe lightning provenance mismatch') END;
END;
CREATE TRIGGER guard_training_recipe_lightning_update_v33 BEFORE UPDATE ON training_recipe_lightning_materials BEGIN SELECT RAISE(ABORT,'training recipe lightning materials are immutable'); END;
CREATE TRIGGER guard_training_recipe_lightning_delete_v33 BEFORE DELETE ON training_recipe_lightning_materials BEGIN SELECT RAISE(ABORT,'training recipe lightning materials are immutable'); END;
CREATE TRIGGER validate_training_recipe_materials_preregister_v33
BEFORE UPDATE OF state ON training_recipes WHEN NEW.state='preregistered' BEGIN
 SELECT CASE WHEN
  (SELECT count(*) FROM training_recipe_lightning_materials WHERE recipe_storage_id=NEW.storage_id)
  !=(SELECT count(*) FROM json_each(json_extract(NEW.recipe_json,'$.lightning_adapter_digests')))
 THEN RAISE(ABORT,'training recipe material set mismatch') END;
END;
"""

_MIGRATION_34 = """
DROP TRIGGER validate_training_recipe_material_insert_v33;
DROP TRIGGER validate_training_recipe_materials_preregister_v33;
CREATE TRIGGER validate_training_recipe_material_insert_v34
BEFORE INSERT ON training_recipe_materials BEGIN
 SELECT CASE WHEN NOT EXISTS(
  SELECT 1 FROM training_recipes recipe
  JOIN training_datasets dataset ON dataset.storage_id=recipe.dataset_storage_id
   AND dataset.reviewer_id=recipe.reviewer_id AND dataset.state='sealed'
  JOIN artifacts base ON base.id=NEW.base_model_artifact_id
  JOIN artifacts source ON source.id=NEW.source_snapshot_artifact_id
  JOIN character_benchmark_campaigns campaign ON campaign.id=NEW.stage_3_campaign_id
   AND campaign.reviewer_id=NEW.reviewer_id
   AND campaign.character_pack_storage_id=dataset.character_pack_storage_id
  JOIN character_benchmark_results result ON result.campaign_id=campaign.id
  WHERE recipe.storage_id=NEW.recipe_storage_id AND recipe.reviewer_id=NEW.reviewer_id
   AND recipe.dataset_digest=dataset.dataset_digest
   AND recipe.character_pack_digest=dataset.character_pack_digest
   AND 'sha256:'||base.sha256=json_extract(recipe.recipe_json,'$.base_model_digest')
   AND 'sha256:'||source.sha256=json_extract(recipe.recipe_json,'$.source_code_digest')
   AND NEW.runtime_digest=canonical_json_digest(json_extract(recipe.recipe_json,'$.runtime_identity'))
   AND result.content_digest=NEW.stage_3_result_id
   AND result.content_digest=json_extract(recipe.recipe_json,'$.stage_3_evidence_digest')
   AND json_extract(result.result_json,'$.baseline_disposition')='baseline_established'
   AND character_benchmark_provenance_is_valid(campaign.id)=1
   AND json_extract(campaign.definition_json,'$.character_pack_digest')=recipe.character_pack_digest
   AND json_extract(campaign.definition_json,'$.stage_2_manifest_digest')=json_extract(recipe.recipe_json,'$.stage_2_manifest_digest')
   AND json_extract(campaign.definition_json,'$.stage_2_evidence_digest')=json_extract(recipe.recipe_json,'$.stage_2_evidence_digest')
   AND (SELECT new_state FROM character_benchmark_events WHERE campaign_id=campaign.id ORDER BY sequence DESC LIMIT 1)='completed'
 ) THEN RAISE(ABORT,'training recipe material provenance mismatch') END;
END;
CREATE TRIGGER validate_training_recipe_materials_preregister_v34
BEFORE UPDATE OF state ON training_recipes WHEN NEW.state='preregistered' BEGIN
 SELECT CASE WHEN EXISTS(
  SELECT 1 FROM json_each(json_extract(NEW.recipe_json,'$.lightning_adapter_digests')) expected
  LEFT JOIN training_recipe_lightning_materials child
   ON child.recipe_storage_id=NEW.storage_id AND child.material_name=expected.key
  LEFT JOIN training_recipe_materials parent ON parent.recipe_storage_id=NEW.storage_id
  LEFT JOIN json_each(parent.lightning_adapter_artifact_ids_json) binding ON binding.key=expected.key
  LEFT JOIN artifacts artifact ON artifact.id=child.artifact_id
  WHERE child.artifact_id IS NULL OR child.artifact_id IS NOT binding.value
   OR 'sha256:'||artifact.sha256 IS NOT expected.value
 ) OR EXISTS(
  SELECT 1 FROM training_recipe_lightning_materials child
  WHERE child.recipe_storage_id=NEW.storage_id AND NOT EXISTS(
   SELECT 1 FROM json_each(json_extract(NEW.recipe_json,'$.lightning_adapter_digests')) expected
   WHERE expected.key=child.material_name
  )
 ) THEN RAISE(ABORT,'training recipe material set mismatch') END;
END;
"""

_MIGRATION_35 = """
ALTER TABLE training_runs ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'draft'
 CHECK(lifecycle_state IN ('draft','probed','calibrated','preregistered','training','evaluating','qualified','rejected','failed','cancelled'));
ALTER TABLE training_runs ADD COLUMN lifecycle_version INTEGER NOT NULL DEFAULT 0 CHECK(lifecycle_version>=0);
ALTER TABLE training_runs ADD COLUMN idempotency_key TEXT;
ALTER TABLE training_runs ADD COLUMN request_digest TEXT CHECK(request_digest IS NULL OR (length(request_digest)=71 AND substr(request_digest,1,7)='sha256:'));
UPDATE training_runs SET lifecycle_state=CASE state WHEN 'created' THEN 'draft' ELSE 'failed' END,
 lifecycle_version=0;
CREATE UNIQUE INDEX training_runs_idempotency ON training_runs(reviewer_id,idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE TABLE training_probe_evidence(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL,reviewer_id TEXT NOT NULL,recipe_digest TEXT NOT NULL,dataset_digest TEXT NOT NULL,
 backend_digest TEXT NOT NULL,runtime_digest TEXT NOT NULL,material_digest TEXT NOT NULL,source_digest TEXT NOT NULL,
 capability_digest TEXT NOT NULL,succeeded INTEGER NOT NULL CHECK(succeeded IN (0,1)),created_at TEXT NOT NULL,
 UNIQUE(run_id,id),FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id)
);
CREATE TABLE training_calibration_evidence(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL,reviewer_id TEXT NOT NULL,probe_evidence_id TEXT NOT NULL,calibration_digest TEXT NOT NULL,
 recipe_digest TEXT NOT NULL,dataset_digest TEXT NOT NULL,backend_digest TEXT NOT NULL,runtime_digest TEXT NOT NULL,
 material_digest TEXT NOT NULL,source_digest TEXT NOT NULL,
 succeeded INTEGER NOT NULL CHECK(succeeded IN (0,1)),created_at TEXT NOT NULL,
 UNIQUE(run_id,id),FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id),
 FOREIGN KEY(probe_evidence_id) REFERENCES training_probe_evidence(id)
);
CREATE TABLE training_cancellation_evidence(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL,reviewer_id TEXT NOT NULL,requested_step INTEGER NOT NULL CHECK(requested_step>=0),
 boundary_step INTEGER NOT NULL CHECK(boundary_step>=requested_step),checkpoint_digest TEXT NOT NULL,created_at TEXT NOT NULL,
 UNIQUE(run_id),FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id)
);
CREATE TABLE training_resume_evidence(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL,reviewer_id TEXT NOT NULL,parent_checkpoint_id TEXT NOT NULL,parent_checkpoint_digest TEXT NOT NULL,parent_artifact_digest TEXT NOT NULL,
 recipe_digest TEXT NOT NULL,dataset_digest TEXT NOT NULL,backend_digest TEXT NOT NULL,runtime_digest TEXT NOT NULL,
 material_digest TEXT NOT NULL,source_digest TEXT NOT NULL,optimizer_state_digest TEXT NOT NULL,created_at TEXT NOT NULL,
 UNIQUE(run_id,id),FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id),
 FOREIGN KEY(parent_checkpoint_id) REFERENCES training_checkpoints(id)
);
CREATE TABLE training_evaluation_evidence(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL,reviewer_id TEXT NOT NULL,disposition TEXT NOT NULL CHECK(disposition IN ('qualified','rejected')),
 evidence_digest TEXT NOT NULL CHECK(length(evidence_digest)=71 AND substr(evidence_digest,1,7)='sha256:'),created_at TEXT NOT NULL,
 UNIQUE(run_id),FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id)
);
CREATE UNIQUE INDEX training_probe_one_per_run ON training_probe_evidence(run_id);
CREATE UNIQUE INDEX training_calibration_one_per_run ON training_calibration_evidence(run_id);
ALTER TABLE training_checkpoints ADD COLUMN evidence_json TEXT CHECK(evidence_json IS NULL OR (json_valid(evidence_json) AND json_type(evidence_json)='object'));
ALTER TABLE training_checkpoints ADD COLUMN evidence_digest TEXT CHECK(evidence_digest IS NULL OR (length(evidence_digest)=71 AND substr(evidence_digest,1,7)='sha256:'));
CREATE TABLE training_checkpoint_inventory(
 checkpoint_id TEXT NOT NULL,ordinal INTEGER NOT NULL CHECK(ordinal>=0),relative_path TEXT NOT NULL,
 artifact_digest TEXT NOT NULL CHECK(length(artifact_digest)=71 AND substr(artifact_digest,1,7)='sha256:'),
 size_bytes INTEGER NOT NULL CHECK(size_bytes>0),role TEXT NOT NULL CHECK(role IN ('weights','optimizer','trainer_state','telemetry','evidence')),
 PRIMARY KEY(checkpoint_id,ordinal),UNIQUE(checkpoint_id,relative_path),
 FOREIGN KEY(checkpoint_id) REFERENCES training_checkpoints(id)
);
DROP TRIGGER validate_training_event_insert;
DROP TRIGGER apply_training_event;
DROP TRIGGER validate_training_telemetry_insert;
DROP TRIGGER validate_training_checkpoint_insert;
CREATE TRIGGER validate_training_event_insert BEFORE INSERT ON training_run_events BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_run_events WHERE run_id=NEW.run_id),1)
  OR NEW.previous_state!=(SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
  OR NOT ((NEW.previous_state='draft' AND NEW.new_state='probed' AND EXISTS(SELECT 1 FROM training_probe_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='probed' AND NEW.new_state='calibrated' AND EXISTS(SELECT 1 FROM training_calibration_evidence c JOIN training_probe_evidence p ON p.id=c.probe_evidence_id WHERE c.run_id=NEW.run_id AND c.reviewer_id=NEW.reviewer_id AND c.succeeded=1 AND p.succeeded=1 AND c.recipe_digest=p.recipe_digest AND c.dataset_digest=p.dataset_digest AND c.backend_digest=p.backend_digest AND c.runtime_digest=p.runtime_digest AND c.material_digest=p.material_digest AND c.source_digest=p.source_digest))
   OR (NEW.previous_state='calibrated' AND NEW.new_state='preregistered' AND length(json_extract(NEW.payload_json,'$.idempotency_key'))>0 AND length(json_extract(NEW.payload_json,'$.request_digest'))=71 AND substr(json_extract(NEW.payload_json,'$.request_digest'),1,7)='sha256:' AND json_extract(NEW.payload_json,'$.probe_evidence_id')=(SELECT id FROM training_probe_evidence WHERE run_id=NEW.run_id AND succeeded=1 ORDER BY created_at DESC LIMIT 1) AND json_extract(NEW.payload_json,'$.calibration_evidence_id')=(SELECT id FROM training_calibration_evidence WHERE run_id=NEW.run_id AND succeeded=1 ORDER BY created_at DESC LIMIT 1))
   OR (NEW.previous_state='preregistered' AND NEW.new_state='training')
   OR (NEW.previous_state='training' AND NEW.new_state='evaluating' AND EXISTS(SELECT 1 FROM training_checkpoints c WHERE c.run_id=NEW.run_id AND c.fraction=1.0 AND (SELECT count(*) FROM training_checkpoint_inventory i WHERE i.checkpoint_id=c.id)=json_array_length(json_extract(c.evidence_json,'$.inventory')) AND NOT EXISTS(SELECT 1 FROM json_each(c.evidence_json,'$.inventory') item WHERE NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=c.id AND i.ordinal=CAST(item.key AS INTEGER) AND i.relative_path=json_extract(item.value,'$.relative_path') AND i.artifact_digest=json_extract(item.value,'$.digest') AND i.size_bytes=json_extract(item.value,'$.size_bytes') AND i.role=json_extract(item.value,'$.role'))) AND (SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id)=(SELECT total_steps FROM training_runs WHERE id=NEW.run_id)))
   OR (NEW.previous_state='evaluating' AND NEW.new_state IN ('qualified','rejected') AND EXISTS(SELECT 1 FROM training_evaluation_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND disposition=NEW.new_state AND evidence_digest=json_extract(NEW.payload_json,'$.evaluation_evidence_digest')))
   OR (NEW.previous_state IN ('draft','probed','calibrated','preregistered','training','evaluating') AND NEW.new_state='failed' AND EXISTS(SELECT 1 FROM training_failures WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id))
   OR (NEW.previous_state IN ('preregistered','training') AND NEW.new_state='cancelled' AND EXISTS(SELECT 1 FROM training_cancellation_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id)))
 THEN RAISE(ABORT,'invalid scientific training lifecycle transition') END;
END;
CREATE TRIGGER apply_training_event AFTER INSERT ON training_run_events BEGIN
 UPDATE training_runs SET lifecycle_state=NEW.new_state,lifecycle_version=lifecycle_version+1,
  state=CASE WHEN NEW.new_state IN ('qualified','rejected') THEN 'completed'
    WHEN NEW.new_state IN ('failed','cancelled') THEN NEW.new_state
    WHEN NEW.new_state IN ('training','evaluating') THEN 'running' ELSE 'created' END,
  terminal_at=CASE WHEN NEW.new_state IN ('qualified','rejected','failed','cancelled') THEN NEW.created_at ELSE NULL END,
  idempotency_key=CASE WHEN NEW.new_state='preregistered' THEN json_extract(NEW.payload_json,'$.idempotency_key') ELSE idempotency_key END,
  request_digest=CASE WHEN NEW.new_state='preregistered' THEN json_extract(NEW.payload_json,'$.request_digest') ELSE request_digest END
 WHERE id=NEW.run_id;
END;
CREATE TRIGGER guard_scientific_training_run_update BEFORE UPDATE ON training_runs BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id
  OR NEW.recipe_storage_id!=OLD.recipe_storage_id OR NEW.dataset_storage_id!=OLD.dataset_storage_id
  OR NEW.total_steps!=OLD.total_steps OR NEW.created_at!=OLD.created_at
  OR NEW.state!=CASE WHEN NEW.lifecycle_state IN ('qualified','rejected') THEN 'completed'
    WHEN NEW.lifecycle_state IN ('failed','cancelled') THEN NEW.lifecycle_state
    WHEN NEW.lifecycle_state IN ('training','evaluating') THEN 'running' ELSE 'created' END
  OR NEW.lifecycle_version!=OLD.lifecycle_version+1
  OR NOT EXISTS(SELECT 1 FROM training_run_events e WHERE e.run_id=OLD.id
    AND e.sequence=(SELECT max(sequence) FROM training_run_events WHERE run_id=OLD.id)
    AND e.previous_state=OLD.lifecycle_state AND e.new_state=NEW.lifecycle_state)
 THEN RAISE(ABORT,'scientific training runs change only through evidence events') END;
END;
CREATE TRIGGER validate_training_telemetry_insert BEFORE INSERT ON training_telemetry BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_telemetry WHERE run_id=NEW.run_id),1)
  OR NEW.step>(SELECT total_steps FROM training_runs WHERE id=NEW.run_id)
  OR NEW.step<=COALESCE((SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id),-1)
 THEN RAISE(ABORT,'invalid training telemetry sequence') END;
END;
CREATE TRIGGER validate_training_checkpoint_insert BEFORE INSERT ON training_checkpoints BEGIN
 SELECT CASE WHEN NEW.dataset_storage_id!=(SELECT dataset_storage_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.step!=training_checkpoint_step((SELECT total_steps FROM training_runs WHERE id=NEW.run_id),NEW.fraction)
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.step>COALESCE((SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id),-1)
  OR NOT EXISTS(SELECT 1 FROM training_recipes recipe,json_each(recipe.recipe_json,'$.checkpoint_plan.fractions') declared WHERE recipe.storage_id=NEW.recipe_storage_id AND declared.value=NEW.fraction)
  OR NEW.evidence_json IS NULL OR canonical_json_digest(NEW.evidence_json)!=NEW.evidence_digest
  OR json_extract(NEW.evidence_json,'$.schema_version')!='character-training-checkpoint-evidence.v1'
  OR json_extract(NEW.evidence_json,'$.run_id')!=NEW.run_id
  OR json_extract(NEW.evidence_json,'$.recipe_digest')!=(SELECT id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.evidence_json,'$.dataset_digest')!=(SELECT dataset_digest FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR NOT EXISTS(SELECT 1 FROM training_calibration_evidence c WHERE c.run_id=NEW.run_id AND c.succeeded=1 AND c.backend_digest=json_extract(NEW.evidence_json,'$.backend_digest') AND c.runtime_digest=json_extract(NEW.evidence_json,'$.runtime_digest') AND c.material_digest=json_extract(NEW.evidence_json,'$.material_digest') AND c.source_digest=json_extract(NEW.evidence_json,'$.source_digest'))
  OR json_extract(NEW.evidence_json,'$.step')!=NEW.step OR json_extract(NEW.evidence_json,'$.fraction')!=NEW.fraction
  OR json_extract(NEW.evidence_json,'$.parent_checkpoint_digest') IS NOT (SELECT evidence_digest FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id)
  OR json_extract(NEW.evidence_json,'$.safetensors_digest')!=NEW.artifact_digest
  OR json_array_length(json_extract(NEW.evidence_json,'$.inventory'))<4
  OR json_extract(NEW.evidence_json,'$.telemetry_sequence_start')<1
  OR json_extract(NEW.evidence_json,'$.telemetry_sequence_end')<(SELECT min(sequence) FROM training_telemetry WHERE run_id=NEW.run_id)
  OR json_extract(NEW.evidence_json,'$.telemetry_sequence_end')>(SELECT max(sequence) FROM training_telemetry WHERE run_id=NEW.run_id)
 THEN RAISE(ABORT,'training checkpoint evidence mismatch') END;
 SELECT CASE WHEN (NEW.fraction=0.25 AND NEW.parent_checkpoint_id IS NOT NULL)
  OR (NEW.fraction>0.25 AND (SELECT run_id FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id) IS NOT NEW.run_id)
  OR (NEW.fraction>0.25 AND (SELECT step FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id)!=(SELECT max(step) FROM training_checkpoints WHERE run_id=NEW.run_id))
  OR (NEW.fraction>0.25 AND (SELECT fraction FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id)!=NEW.fraction-0.25)
 THEN RAISE(ABORT,'training checkpoint parent mismatch') END;
END;
CREATE TRIGGER validate_training_probe_insert BEFORE INSERT ON training_probe_evidence BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='draft'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.recipe_digest!=(SELECT recipe.id FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.dataset_digest!=(SELECT recipe.dataset_digest FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.backend_digest!=(SELECT canonical_json_digest(json_extract(recipe.recipe_json,'$.backend_config')) FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.runtime_digest!=(SELECT material.runtime_digest FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.source_digest!=(SELECT 'sha256:'||artifact.sha256 FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id JOIN artifacts artifact ON artifact.id=material.source_snapshot_artifact_id WHERE run.id=NEW.run_id)
 THEN RAISE(ABORT,'training probe identity mismatch') END;
END;
CREATE TRIGGER validate_training_calibration_insert BEFORE INSERT ON training_calibration_evidence BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='probed'
  OR NOT EXISTS(SELECT 1 FROM training_probe_evidence p WHERE p.id=NEW.probe_evidence_id AND p.run_id=NEW.run_id AND p.reviewer_id=NEW.reviewer_id AND p.succeeded=1 AND p.recipe_digest=NEW.recipe_digest AND p.dataset_digest=NEW.dataset_digest AND p.backend_digest=NEW.backend_digest AND p.runtime_digest=NEW.runtime_digest AND p.material_digest=NEW.material_digest AND p.source_digest=NEW.source_digest)
 THEN RAISE(ABORT,'training calibration identity mismatch') END;
END;
CREATE TRIGGER validate_training_cancellation_insert BEFORE INSERT ON training_cancellation_evidence BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id) NOT IN ('preregistered','training')
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NOT EXISTS(SELECT 1 FROM training_checkpoints c WHERE c.run_id=NEW.run_id AND c.step=NEW.boundary_step AND c.artifact_digest=NEW.checkpoint_digest AND json_extract(c.evidence_json,'$.cancellation_boundary')=1)
 THEN RAISE(ABORT,'training cancellation boundary mismatch') END;
END;
CREATE TRIGGER validate_training_resume_insert BEFORE INSERT ON training_resume_evidence BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NOT EXISTS(SELECT 1 FROM training_checkpoints c JOIN training_runs parent_run ON parent_run.id=c.run_id JOIN training_recipes recipe ON recipe.storage_id=parent_run.recipe_storage_id WHERE c.id=NEW.parent_checkpoint_id AND c.run_id=NEW.run_id AND c.evidence_digest=NEW.parent_checkpoint_digest AND c.artifact_digest=NEW.parent_artifact_digest AND recipe.id=NEW.recipe_digest AND recipe.dataset_digest=NEW.dataset_digest AND json_extract(c.evidence_json,'$.backend_digest')=NEW.backend_digest AND json_extract(c.evidence_json,'$.runtime_digest')=NEW.runtime_digest AND json_extract(c.evidence_json,'$.material_digest')=NEW.material_digest AND json_extract(c.evidence_json,'$.source_digest')=NEW.source_digest AND json_extract(c.evidence_json,'$.optimizer_state_digest')=NEW.optimizer_state_digest)
 THEN RAISE(ABORT,'training resume identity mismatch') END;
END;
CREATE TRIGGER validate_training_checkpoint_inventory_insert BEFORE INSERT ON training_checkpoint_inventory BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=(SELECT run_id FROM training_checkpoints WHERE id=NEW.checkpoint_id))!='training'
  OR NOT EXISTS(SELECT 1 FROM training_checkpoints c,json_each(c.evidence_json,'$.inventory') item WHERE c.id=NEW.checkpoint_id AND CAST(item.key AS INTEGER)=NEW.ordinal AND json_extract(item.value,'$.relative_path')=NEW.relative_path AND json_extract(item.value,'$.digest')=NEW.artifact_digest AND json_extract(item.value,'$.size_bytes')=NEW.size_bytes AND json_extract(item.value,'$.role')=NEW.role)
 THEN RAISE(ABORT,'training checkpoint inventory mismatch') END;
END;
CREATE TRIGGER validate_training_failure_insert_v35 BEFORE INSERT ON training_failures BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id) NOT IN ('draft','probed','calibrated','preregistered','training','evaluating')
 THEN RAISE(ABORT,'training failure identity or lifecycle mismatch') END;
END;
CREATE TRIGGER validate_training_evaluation_insert BEFORE INSERT ON training_evaluation_evidence BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='evaluating'
 THEN RAISE(ABORT,'training evaluation identity or lifecycle mismatch') END;
END;
CREATE TRIGGER guard_training_probe_update BEFORE UPDATE ON training_probe_evidence BEGIN SELECT RAISE(ABORT,'training probe evidence is immutable'); END;
CREATE TRIGGER guard_training_probe_delete BEFORE DELETE ON training_probe_evidence BEGIN SELECT RAISE(ABORT,'training probe evidence is immutable'); END;
CREATE TRIGGER guard_training_calibration_update BEFORE UPDATE ON training_calibration_evidence BEGIN SELECT RAISE(ABORT,'training calibration evidence is immutable'); END;
CREATE TRIGGER guard_training_calibration_delete BEFORE DELETE ON training_calibration_evidence BEGIN SELECT RAISE(ABORT,'training calibration evidence is immutable'); END;
CREATE TRIGGER guard_training_cancellation_update BEFORE UPDATE ON training_cancellation_evidence BEGIN SELECT RAISE(ABORT,'training cancellation evidence is immutable'); END;
CREATE TRIGGER guard_training_cancellation_delete BEFORE DELETE ON training_cancellation_evidence BEGIN SELECT RAISE(ABORT,'training cancellation evidence is immutable'); END;
CREATE TRIGGER guard_training_resume_update BEFORE UPDATE ON training_resume_evidence BEGIN SELECT RAISE(ABORT,'training resume evidence is immutable'); END;
CREATE TRIGGER guard_training_resume_delete BEFORE DELETE ON training_resume_evidence BEGIN SELECT RAISE(ABORT,'training resume evidence is immutable'); END;
CREATE TRIGGER guard_training_evaluation_update BEFORE UPDATE ON training_evaluation_evidence BEGIN SELECT RAISE(ABORT,'training evaluation evidence is immutable'); END;
CREATE TRIGGER guard_training_evaluation_delete BEFORE DELETE ON training_evaluation_evidence BEGIN SELECT RAISE(ABORT,'training evaluation evidence is immutable'); END;
CREATE TRIGGER guard_training_checkpoint_inventory_update BEFORE UPDATE ON training_checkpoint_inventory BEGIN SELECT RAISE(ABORT,'training checkpoint inventory is immutable'); END;
CREATE TRIGGER guard_training_checkpoint_inventory_delete BEFORE DELETE ON training_checkpoint_inventory BEGIN SELECT RAISE(ABORT,'training checkpoint inventory is immutable'); END;
"""

_MIGRATION_36 = """
ALTER TABLE training_checkpoints ADD COLUMN publication_state TEXT NOT NULL DEFAULT 'building'
 CHECK(publication_state IN ('building','finalized'));
DROP TRIGGER validate_training_event_insert;
DROP TRIGGER guard_training_checkpoint_inventory_update;
DROP TRIGGER guard_training_checkpoint_inventory_delete;
ALTER TABLE training_checkpoint_inventory RENAME TO training_checkpoint_inventory_v35;
CREATE TABLE training_checkpoint_inventory(
 checkpoint_id TEXT NOT NULL,ordinal INTEGER NOT NULL CHECK(ordinal>=0),relative_path TEXT NOT NULL,
 artifact_digest TEXT NOT NULL CHECK(length(artifact_digest)=71 AND substr(artifact_digest,1,7)='sha256:'),
 size_bytes INTEGER NOT NULL CHECK(size_bytes>0),role TEXT NOT NULL CHECK(role IN
 ('weights','optimizer','trainer_state','telemetry','trainable_targets','frozen_parameters','finite_gradient_proof')),
 PRIMARY KEY(checkpoint_id,ordinal),UNIQUE(checkpoint_id,relative_path),
 FOREIGN KEY(checkpoint_id) REFERENCES training_checkpoints(id));
INSERT INTO training_checkpoint_inventory SELECT * FROM training_checkpoint_inventory_v35;
DROP TABLE training_checkpoint_inventory_v35;
CREATE TABLE training_run_idempotency(
 reviewer_id TEXT NOT NULL,recipe_storage_id TEXT NOT NULL,idempotency_key TEXT NOT NULL,
 request_digest TEXT NOT NULL CHECK(length(request_digest)=71 AND substr(request_digest,1,7)='sha256:'),
 run_id TEXT NOT NULL,probe_evidence_id TEXT NOT NULL,calibration_evidence_id TEXT NOT NULL,
 created_at TEXT NOT NULL,PRIMARY KEY(reviewer_id,recipe_storage_id,idempotency_key),
 FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id),
 FOREIGN KEY(probe_evidence_id) REFERENCES training_probe_evidence(id),
 FOREIGN KEY(calibration_evidence_id) REFERENCES training_calibration_evidence(id)
);
CREATE TABLE training_cancellation_requests(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL UNIQUE,reviewer_id TEXT NOT NULL,
 requested_step INTEGER NOT NULL CHECK(requested_step>=0),request_digest TEXT NOT NULL
 CHECK(length(request_digest)=71 AND substr(request_digest,1,7)='sha256:'),created_at TEXT NOT NULL,
 FOREIGN KEY(run_id,reviewer_id) REFERENCES training_runs(id,reviewer_id)
);
DROP TRIGGER validate_training_checkpoint_insert;
DROP TRIGGER guard_training_checkpoint_update;
DROP TRIGGER validate_training_cancellation_insert;
DROP TRIGGER validate_training_probe_insert;
CREATE TRIGGER validate_training_event_insert_v36 BEFORE INSERT ON training_run_events BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_run_events WHERE run_id=NEW.run_id),1)
  OR NEW.previous_state!=(SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
  OR NOT ((NEW.previous_state='draft' AND NEW.new_state='probed' AND EXISTS(SELECT 1 FROM training_probe_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='probed' AND NEW.new_state='calibrated' AND EXISTS(SELECT 1 FROM training_calibration_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='calibrated' AND NEW.new_state='preregistered' AND EXISTS(SELECT 1 FROM training_run_idempotency b WHERE b.run_id=NEW.run_id AND b.reviewer_id=NEW.reviewer_id AND b.idempotency_key=json_extract(NEW.payload_json,'$.idempotency_key') AND b.request_digest=json_extract(NEW.payload_json,'$.request_digest')))
   OR (NEW.previous_state='preregistered' AND NEW.new_state='training')
   OR (NEW.previous_state='training' AND NEW.new_state='evaluating' AND EXISTS(SELECT 1 FROM training_checkpoints c WHERE c.run_id=NEW.run_id AND c.fraction=1.0 AND c.publication_state='finalized') AND (SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id)=(SELECT total_steps FROM training_runs WHERE id=NEW.run_id))
   OR (NEW.previous_state='evaluating' AND NEW.new_state IN ('qualified','rejected') AND EXISTS(SELECT 1 FROM training_evaluation_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND disposition=NEW.new_state AND evidence_digest=json_extract(NEW.payload_json,'$.evaluation_evidence_digest')))
   OR (NEW.previous_state IN ('draft','probed','calibrated','preregistered','training','evaluating') AND NEW.new_state='failed' AND EXISTS(SELECT 1 FROM training_failures WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id))
   OR (NEW.previous_state='training' AND NEW.new_state='cancelled' AND EXISTS(SELECT 1 FROM training_cancellation_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id)))
 THEN RAISE(ABORT,'invalid scientific training lifecycle transition') END;
END;
CREATE TRIGGER validate_training_checkpoint_insert_v36 BEFORE INSERT ON training_checkpoints BEGIN
 SELECT CASE WHEN NEW.publication_state!='building' OR NEW.evidence_json IS NULL
  OR canonical_json_digest(NEW.manifest_json)!=NEW.manifest_digest
  OR canonical_json_digest(NEW.evidence_json)!=NEW.evidence_digest
  OR json_extract(NEW.manifest_json,'$.schema_version')!='character-training-checkpoint.v1'
  OR json_extract(NEW.evidence_json,'$.schema_version')!='character-training-checkpoint-evidence.v1'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.recipe_storage_id!=(SELECT recipe_storage_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.dataset_storage_id!=(SELECT dataset_storage_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.step!=training_checkpoint_step((SELECT total_steps FROM training_runs WHERE id=NEW.run_id),NEW.fraction)
  OR json_extract(NEW.manifest_json,'$.run_id')!=NEW.run_id
  OR json_extract(NEW.manifest_json,'$.reviewer_id')!=NEW.reviewer_id
  OR json_extract(NEW.manifest_json,'$.recipe_id')!=(SELECT id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.manifest_json,'$.dataset_digest')!=(SELECT dataset_digest FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.manifest_json,'$.character_id')!=(SELECT character_id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.manifest_json,'$.step')!=NEW.step OR json_extract(NEW.manifest_json,'$.fraction')!=NEW.fraction
  OR json_extract(NEW.manifest_json,'$.parent_checkpoint_id') IS NOT NEW.parent_checkpoint_id
  OR json_extract(NEW.manifest_json,'$.artifact_digest')!=NEW.artifact_digest
  OR json_extract(NEW.manifest_json,'$.safetensors_size_bytes')!=json_extract(NEW.evidence_json,'$.safetensors_size_bytes')
  OR json_extract(NEW.manifest_json,'$.backend_digest')!=json_extract(NEW.evidence_json,'$.backend_digest')
  OR json_extract(NEW.manifest_json,'$.runtime_digest')!=json_extract(NEW.evidence_json,'$.runtime_digest')
  OR json_extract(NEW.manifest_json,'$.material_digest')!=json_extract(NEW.evidence_json,'$.material_digest')
  OR json_extract(NEW.manifest_json,'$.source_digest')!=json_extract(NEW.evidence_json,'$.source_digest')
  OR json_extract(NEW.manifest_json,'$.optimizer_state_digest')!=json_extract(NEW.evidence_json,'$.optimizer_state_digest')
  OR json_extract(NEW.manifest_json,'$.trainer_state_digest')!=json_extract(NEW.evidence_json,'$.trainer_state_digest')
  OR json_extract(NEW.manifest_json,'$.telemetry_digest')!=json_extract(NEW.evidence_json,'$.telemetry_digest')
  OR json_extract(NEW.manifest_json,'$.telemetry_sequence_start')!=json_extract(NEW.evidence_json,'$.telemetry_sequence_start')
  OR json_extract(NEW.manifest_json,'$.telemetry_sequence_end')!=json_extract(NEW.evidence_json,'$.telemetry_sequence_end')
  OR json_extract(NEW.manifest_json,'$.trainable_targets_digest')!=json_extract(NEW.evidence_json,'$.trainable_targets_digest')
  OR json_extract(NEW.manifest_json,'$.frozen_parameters_digest')!=json_extract(NEW.evidence_json,'$.frozen_parameters_digest')
  OR json_extract(NEW.manifest_json,'$.finite_gradient_proof_digest')!=json_extract(NEW.evidence_json,'$.finite_gradient_proof_digest')
  OR json_extract(NEW.manifest_json,'$.cancellation_boundary')!=json_extract(NEW.evidence_json,'$.cancellation_boundary')
  OR json_extract(NEW.manifest_json,'$.inventory')!=json_extract(NEW.evidence_json,'$.inventory')
  OR json_extract(NEW.evidence_json,'$.run_id')!=NEW.run_id
  OR json_extract(NEW.evidence_json,'$.recipe_digest')!=(SELECT id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.evidence_json,'$.dataset_digest')!=(SELECT dataset_digest FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR NOT EXISTS(SELECT 1 FROM training_calibration_evidence c WHERE c.run_id=NEW.run_id AND c.succeeded=1
   AND c.backend_digest=json_extract(NEW.evidence_json,'$.backend_digest')
   AND c.runtime_digest=json_extract(NEW.evidence_json,'$.runtime_digest')
   AND c.material_digest=json_extract(NEW.evidence_json,'$.material_digest')
   AND c.source_digest=json_extract(NEW.evidence_json,'$.source_digest'))
  OR json_extract(NEW.evidence_json,'$.step')!=NEW.step OR json_extract(NEW.evidence_json,'$.fraction')!=NEW.fraction
  OR json_extract(NEW.evidence_json,'$.parent_checkpoint_digest') IS NOT
    (SELECT evidence_digest FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id AND publication_state='finalized')
  OR json_extract(NEW.evidence_json,'$.safetensors_digest')!=NEW.artifact_digest
  OR json_extract(NEW.evidence_json,'$.safetensors_size_bytes')<=0
  OR json_extract(NEW.evidence_json,'$.telemetry_sequence_start')<1
  OR json_extract(NEW.evidence_json,'$.telemetry_sequence_end')<json_extract(NEW.evidence_json,'$.telemetry_sequence_start')
  OR (SELECT count(*) FROM training_telemetry t WHERE t.run_id=NEW.run_id AND t.sequence BETWEEN json_extract(NEW.evidence_json,'$.telemetry_sequence_start') AND json_extract(NEW.evidence_json,'$.telemetry_sequence_end'))!=json_extract(NEW.evidence_json,'$.telemetry_sequence_end')-json_extract(NEW.evidence_json,'$.telemetry_sequence_start')+1
  OR json_extract(NEW.evidence_json,'$.telemetry_digest')!=(SELECT canonical_json_digest(json_group_array(json_object('sequence',sequence,'step',step,'payload',json(payload_json)))) FROM (SELECT sequence,step,payload_json FROM training_telemetry WHERE run_id=NEW.run_id AND sequence BETWEEN json_extract(NEW.evidence_json,'$.telemetry_sequence_start') AND json_extract(NEW.evidence_json,'$.telemetry_sequence_end') ORDER BY sequence))
 THEN RAISE(ABORT,'training checkpoint canonical identity mismatch') END;
END;
CREATE TRIGGER validate_training_probe_insert_v36 BEFORE INSERT ON training_probe_evidence BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='draft'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.recipe_digest!=(SELECT recipe.id FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.dataset_digest!=(SELECT recipe.dataset_digest FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.backend_digest!=(SELECT canonical_json_digest(json_extract(recipe.recipe_json,'$.backend_config')) FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.runtime_digest!=(SELECT material.runtime_digest FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.source_digest!=(SELECT 'sha256:'||artifact.sha256 FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id JOIN artifacts artifact ON artifact.id=material.source_snapshot_artifact_id WHERE run.id=NEW.run_id)
  OR NEW.material_digest!=(SELECT canonical_json_digest(json_object(
    'stage_3_campaign_id',material.stage_3_campaign_id,'stage_3_result_id',material.stage_3_result_id,
    'base_model_artifact_id',material.base_model_artifact_id,
    'lightning_adapter_artifact_ids',json(material.lightning_adapter_artifact_ids_json),
    'source_snapshot_artifact_id',material.source_snapshot_artifact_id,'runtime_digest',material.runtime_digest))
   FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
 THEN RAISE(ABORT,'training probe persisted identity mismatch') END;
END;
CREATE TRIGGER validate_training_checkpoint_inventory_insert_v36 BEFORE INSERT ON training_checkpoint_inventory BEGIN
 SELECT CASE WHEN (SELECT publication_state FROM training_checkpoints WHERE id=NEW.checkpoint_id)!='building'
  OR NOT EXISTS(SELECT 1 FROM training_checkpoints c,json_each(c.evidence_json,'$.inventory') item
   WHERE c.id=NEW.checkpoint_id AND CAST(item.key AS INTEGER)=NEW.ordinal
   AND json_extract(item.value,'$.relative_path')=NEW.relative_path
   AND json_extract(item.value,'$.digest')=NEW.artifact_digest
   AND json_extract(item.value,'$.size_bytes')=NEW.size_bytes
   AND json_extract(item.value,'$.role')=NEW.role)
 THEN RAISE(ABORT,'training checkpoint inventory mismatch') END;
END;
CREATE TRIGGER finalize_training_checkpoint_v36 BEFORE UPDATE OF publication_state ON training_checkpoints BEGIN
 SELECT CASE WHEN OLD.publication_state!='building' OR NEW.publication_state!='finalized'
  OR NEW.id!=OLD.id OR NEW.run_id!=OLD.run_id OR NEW.reviewer_id!=OLD.reviewer_id
  OR NEW.recipe_storage_id!=OLD.recipe_storage_id OR NEW.dataset_storage_id!=OLD.dataset_storage_id
  OR NEW.step!=OLD.step OR NEW.fraction!=OLD.fraction OR NEW.parent_checkpoint_id IS NOT OLD.parent_checkpoint_id
  OR NEW.artifact_digest!=OLD.artifact_digest OR NEW.manifest_digest!=OLD.manifest_digest
  OR NEW.manifest_json!=OLD.manifest_json OR NEW.evidence_digest!=OLD.evidence_digest
  OR NEW.evidence_json!=OLD.evidence_json OR NEW.created_at!=OLD.created_at
  OR (SELECT count(*) FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id)!=7
  OR (SELECT count(DISTINCT role) FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id)!=7
  OR NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id AND i.role='weights' AND i.artifact_digest=json_extract(OLD.evidence_json,'$.safetensors_digest') AND i.size_bytes=json_extract(OLD.evidence_json,'$.safetensors_size_bytes'))
  OR NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id AND i.role='optimizer' AND i.artifact_digest=json_extract(OLD.evidence_json,'$.optimizer_state_digest'))
  OR NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id AND i.role='trainer_state' AND i.artifact_digest=json_extract(OLD.evidence_json,'$.trainer_state_digest'))
  OR NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id AND i.role='telemetry' AND i.artifact_digest=json_extract(OLD.evidence_json,'$.telemetry_digest'))
  OR NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id AND i.role='trainable_targets' AND i.artifact_digest=json_extract(OLD.evidence_json,'$.trainable_targets_digest'))
  OR NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id AND i.role='frozen_parameters' AND i.artifact_digest=json_extract(OLD.evidence_json,'$.frozen_parameters_digest'))
  OR NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id AND i.role='finite_gradient_proof' AND i.artifact_digest=json_extract(OLD.evidence_json,'$.finite_gradient_proof_digest'))
  OR EXISTS(SELECT 1 FROM json_each(OLD.evidence_json,'$.inventory') item WHERE NOT EXISTS(
    SELECT 1 FROM training_checkpoint_inventory i WHERE i.checkpoint_id=OLD.id
    AND i.ordinal=CAST(item.key AS INTEGER) AND i.relative_path=json_extract(item.value,'$.relative_path')
    AND i.artifact_digest=json_extract(item.value,'$.digest') AND i.size_bytes=json_extract(item.value,'$.size_bytes')
    AND i.role=json_extract(item.value,'$.role')))
 THEN RAISE(ABORT,'training checkpoint finalization is incomplete') END;
END;
CREATE TRIGGER guard_finalized_training_checkpoint_v36 BEFORE UPDATE ON training_checkpoints
 WHEN OLD.publication_state='finalized' BEGIN SELECT RAISE(ABORT,'finalized checkpoint is immutable'); END;
CREATE TRIGGER validate_training_cancellation_insert_v36 BEFORE INSERT ON training_cancellation_evidence BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NOT EXISTS(SELECT 1 FROM training_cancellation_requests r WHERE r.run_id=NEW.run_id
    AND r.reviewer_id=NEW.reviewer_id AND r.requested_step=NEW.requested_step)
  OR NOT EXISTS(SELECT 1 FROM training_checkpoints c WHERE c.run_id=NEW.run_id
    AND c.publication_state='finalized' AND c.step=NEW.boundary_step
    AND c.artifact_digest=NEW.checkpoint_digest
    AND json_extract(c.evidence_json,'$.cancellation_boundary')=1)
 THEN RAISE(ABORT,'training cancellation request or boundary mismatch') END;
END;
CREATE TRIGGER guard_training_idempotency_update_v36 BEFORE UPDATE ON training_run_idempotency BEGIN SELECT RAISE(ABORT,'training idempotency is immutable'); END;
CREATE TRIGGER guard_training_idempotency_delete_v36 BEFORE DELETE ON training_run_idempotency BEGIN SELECT RAISE(ABORT,'training idempotency is immutable'); END;
CREATE TRIGGER validate_training_idempotency_insert_v36 BEFORE INSERT ON training_run_idempotency BEGIN
 SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM training_runs run
  JOIN training_probe_evidence probe ON probe.id=NEW.probe_evidence_id AND probe.run_id=run.id
  JOIN training_calibration_evidence calibration ON calibration.id=NEW.calibration_evidence_id
   AND calibration.run_id=run.id AND calibration.probe_evidence_id=probe.id
  WHERE run.id=NEW.run_id AND run.reviewer_id=NEW.reviewer_id
   AND run.recipe_storage_id=NEW.recipe_storage_id AND run.lifecycle_state='calibrated'
   AND probe.succeeded=1 AND calibration.succeeded=1)
 THEN RAISE(ABORT,'training idempotency evidence mismatch') END;
END;
CREATE TRIGGER guard_training_cancel_request_update_v36 BEFORE UPDATE ON training_cancellation_requests BEGIN SELECT RAISE(ABORT,'training cancellation request is immutable'); END;
CREATE TRIGGER guard_training_cancel_request_delete_v36 BEFORE DELETE ON training_cancellation_requests BEGIN SELECT RAISE(ABORT,'training cancellation request is immutable'); END;
CREATE TRIGGER validate_training_cancel_request_insert_v36 BEFORE INSERT ON training_cancellation_requests BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.requested_step>COALESCE((SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id),0)
 THEN RAISE(ABORT,'training cancellation request lifecycle mismatch') END;
END;
"""

_MIGRATION_37 = """
DROP TRIGGER validate_training_event_insert_v36;
DROP TRIGGER validate_training_checkpoint_insert_v36;
DROP TRIGGER finalize_training_checkpoint_v36;
DROP TRIGGER guard_finalized_training_checkpoint_v36;
DROP TRIGGER validate_training_checkpoint_inventory_insert_v36;
DROP TRIGGER validate_training_cancellation_insert_v36;
ALTER TABLE training_cancellation_evidence ADD COLUMN cancellation_request_id TEXT;
ALTER TABLE training_cancellation_evidence ADD COLUMN request_digest TEXT;
UPDATE training_cancellation_evidence SET
 cancellation_request_id=(SELECT id FROM training_cancellation_requests request
  WHERE request.run_id=training_cancellation_evidence.run_id
   AND request.reviewer_id=training_cancellation_evidence.reviewer_id
   AND request.requested_step=training_cancellation_evidence.requested_step),
 request_digest=(SELECT request_digest FROM training_cancellation_requests request
  WHERE request.run_id=training_cancellation_evidence.run_id
   AND request.reviewer_id=training_cancellation_evidence.reviewer_id
   AND request.requested_step=training_cancellation_evidence.requested_step);
CREATE TRIGGER validate_training_checkpoint_insert_v37 BEFORE INSERT ON training_checkpoints BEGIN
 SELECT CASE WHEN NEW.publication_state!='finalized'
  OR training_checkpoint_json_is_exact(NEW.manifest_json,NEW.evidence_json)!=1
  OR canonical_json_digest(NEW.manifest_json)!=NEW.manifest_digest
  OR canonical_json_digest(NEW.evidence_json)!=NEW.evidence_digest
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.recipe_storage_id!=(SELECT recipe_storage_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.dataset_storage_id!=(SELECT dataset_storage_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.step!=training_checkpoint_step((SELECT total_steps FROM training_runs WHERE id=NEW.run_id),NEW.fraction)
  OR json_extract(NEW.manifest_json,'$.run_id')!=NEW.run_id
  OR json_extract(NEW.manifest_json,'$.reviewer_id')!=NEW.reviewer_id
  OR json_extract(NEW.manifest_json,'$.recipe_id')!=(SELECT id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.manifest_json,'$.dataset_digest')!=(SELECT dataset_digest FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.manifest_json,'$.character_id')!=(SELECT character_id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.manifest_json,'$.step')!=NEW.step OR json_extract(NEW.manifest_json,'$.fraction')!=NEW.fraction
  OR json_extract(NEW.manifest_json,'$.parent_checkpoint_id') IS NOT NEW.parent_checkpoint_id
  OR json_extract(NEW.manifest_json,'$.artifact_digest')!=NEW.artifact_digest
  OR json_extract(NEW.manifest_json,'$.safetensors_size_bytes')!=json_extract(NEW.evidence_json,'$.safetensors_size_bytes')
  OR json_extract(NEW.manifest_json,'$.backend_digest')!=json_extract(NEW.evidence_json,'$.backend_digest')
  OR json_extract(NEW.manifest_json,'$.runtime_digest')!=json_extract(NEW.evidence_json,'$.runtime_digest')
  OR json_extract(NEW.manifest_json,'$.material_digest')!=json_extract(NEW.evidence_json,'$.material_digest')
  OR json_extract(NEW.manifest_json,'$.source_digest')!=json_extract(NEW.evidence_json,'$.source_digest')
  OR json_extract(NEW.manifest_json,'$.optimizer_state_digest')!=json_extract(NEW.evidence_json,'$.optimizer_state_digest')
  OR json_extract(NEW.manifest_json,'$.trainer_state_digest')!=json_extract(NEW.evidence_json,'$.trainer_state_digest')
  OR json_extract(NEW.manifest_json,'$.telemetry_digest')!=json_extract(NEW.evidence_json,'$.telemetry_digest')
  OR json_extract(NEW.manifest_json,'$.telemetry_sequence_start')!=json_extract(NEW.evidence_json,'$.telemetry_sequence_start')
  OR json_extract(NEW.manifest_json,'$.telemetry_sequence_end')!=json_extract(NEW.evidence_json,'$.telemetry_sequence_end')
  OR json_extract(NEW.manifest_json,'$.trainable_targets_digest')!=json_extract(NEW.evidence_json,'$.trainable_targets_digest')
  OR json_extract(NEW.manifest_json,'$.frozen_parameters_digest')!=json_extract(NEW.evidence_json,'$.frozen_parameters_digest')
  OR json_extract(NEW.manifest_json,'$.finite_gradient_proof_digest')!=json_extract(NEW.evidence_json,'$.finite_gradient_proof_digest')
  OR json_extract(NEW.manifest_json,'$.cancellation_boundary')!=json_extract(NEW.evidence_json,'$.cancellation_boundary')
  OR json_extract(NEW.evidence_json,'$.run_id')!=NEW.run_id
  OR json_extract(NEW.evidence_json,'$.recipe_digest')!=(SELECT id FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR json_extract(NEW.evidence_json,'$.dataset_digest')!=(SELECT dataset_digest FROM training_recipes WHERE storage_id=NEW.recipe_storage_id)
  OR NOT EXISTS(SELECT 1 FROM training_calibration_evidence c WHERE c.run_id=NEW.run_id AND c.succeeded=1
   AND c.backend_digest=json_extract(NEW.evidence_json,'$.backend_digest')
   AND c.runtime_digest=json_extract(NEW.evidence_json,'$.runtime_digest')
   AND c.material_digest=json_extract(NEW.evidence_json,'$.material_digest')
   AND c.source_digest=json_extract(NEW.evidence_json,'$.source_digest'))
  OR json_extract(NEW.evidence_json,'$.step')!=NEW.step OR json_extract(NEW.evidence_json,'$.fraction')!=NEW.fraction
  OR json_extract(NEW.evidence_json,'$.parent_checkpoint_digest') IS NOT
    (SELECT evidence_digest FROM training_checkpoints WHERE id=NEW.parent_checkpoint_id AND publication_state='finalized')
  OR json_extract(NEW.evidence_json,'$.safetensors_digest')!=NEW.artifact_digest
  OR (SELECT count(*) FROM training_telemetry t WHERE t.run_id=NEW.run_id AND t.sequence BETWEEN json_extract(NEW.evidence_json,'$.telemetry_sequence_start') AND json_extract(NEW.evidence_json,'$.telemetry_sequence_end'))!=json_extract(NEW.evidence_json,'$.telemetry_sequence_end')-json_extract(NEW.evidence_json,'$.telemetry_sequence_start')+1
  OR json_extract(NEW.evidence_json,'$.telemetry_digest')!=(SELECT canonical_json_digest(json_group_array(json_object('sequence',sequence,'step',step,'payload',json(payload_json)))) FROM (SELECT sequence,step,payload_json FROM training_telemetry WHERE run_id=NEW.run_id AND sequence BETWEEN json_extract(NEW.evidence_json,'$.telemetry_sequence_start') AND json_extract(NEW.evidence_json,'$.telemetry_sequence_end') ORDER BY sequence))
 THEN RAISE(ABORT,'training checkpoint canonical publication mismatch') END;
END;
CREATE TRIGGER materialize_training_checkpoint_inventory_v37 AFTER INSERT ON training_checkpoints BEGIN
 INSERT INTO training_checkpoint_inventory(checkpoint_id,ordinal,relative_path,artifact_digest,size_bytes,role)
 SELECT NEW.id,CAST(item.key AS INTEGER),json_extract(item.value,'$.relative_path'),
  json_extract(item.value,'$.digest'),json_extract(item.value,'$.size_bytes'),json_extract(item.value,'$.role')
 FROM json_each(NEW.evidence_json,'$.inventory') item ORDER BY CAST(item.key AS INTEGER);
END;
CREATE TRIGGER validate_training_checkpoint_inventory_insert_v37 BEFORE INSERT ON training_checkpoint_inventory BEGIN
 SELECT CASE WHEN (SELECT publication_state FROM training_checkpoints WHERE id=NEW.checkpoint_id)!='finalized'
  OR NEW.ordinal!=(SELECT count(*) FROM training_checkpoint_inventory WHERE checkpoint_id=NEW.checkpoint_id)
  OR NEW.ordinal>=7
  OR NOT EXISTS(SELECT 1 FROM training_checkpoints c,json_each(c.evidence_json,'$.inventory') item
   WHERE c.id=NEW.checkpoint_id AND CAST(item.key AS INTEGER)=NEW.ordinal
   AND json_extract(item.value,'$.relative_path')=NEW.relative_path
   AND json_extract(item.value,'$.digest')=NEW.artifact_digest
   AND json_extract(item.value,'$.size_bytes')=NEW.size_bytes
   AND json_extract(item.value,'$.role')=NEW.role)
 THEN RAISE(ABORT,'training checkpoint inventory is derived') END;
END;
CREATE TRIGGER guard_training_checkpoint_inventory_update_v37 BEFORE UPDATE ON training_checkpoint_inventory BEGIN SELECT RAISE(ABORT,'training checkpoint inventory is immutable'); END;
CREATE TRIGGER guard_training_checkpoint_inventory_delete_v37 BEFORE DELETE ON training_checkpoint_inventory BEGIN SELECT RAISE(ABORT,'training checkpoint inventory is immutable'); END;
CREATE TRIGGER guard_training_checkpoint_update_v37 BEFORE UPDATE ON training_checkpoints BEGIN SELECT RAISE(ABORT,'training checkpoints are immutable'); END;
CREATE TRIGGER validate_training_cancellation_insert_v37 BEFORE INSERT ON training_cancellation_evidence BEGIN
 SELECT CASE WHEN (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='training'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.cancellation_request_id IS NULL OR NEW.request_digest IS NULL
  OR NOT EXISTS(SELECT 1 FROM training_cancellation_requests request
    WHERE request.id=NEW.cancellation_request_id AND request.run_id=NEW.run_id
     AND request.reviewer_id=NEW.reviewer_id AND request.requested_step=NEW.requested_step
     AND request.request_digest=NEW.request_digest)
  OR NOT EXISTS(SELECT 1 FROM training_checkpoints checkpoint WHERE checkpoint.run_id=NEW.run_id
    AND checkpoint.publication_state='finalized' AND checkpoint.step=NEW.boundary_step
    AND checkpoint.artifact_digest=NEW.checkpoint_digest
    AND json_extract(checkpoint.evidence_json,'$.cancellation_boundary')=1)
 THEN RAISE(ABORT,'training cancellation request or boundary mismatch') END;
END;
CREATE TRIGGER validate_training_event_insert_v37 BEFORE INSERT ON training_run_events BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_run_events WHERE run_id=NEW.run_id),1)
  OR NEW.previous_state!=(SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
  OR NOT ((NEW.previous_state='draft' AND NEW.new_state='probed' AND EXISTS(SELECT 1 FROM training_probe_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='probed' AND NEW.new_state='calibrated' AND EXISTS(SELECT 1 FROM training_calibration_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='calibrated' AND NEW.new_state='preregistered' AND EXISTS(SELECT 1 FROM training_run_idempotency binding WHERE binding.run_id=NEW.run_id AND binding.reviewer_id=NEW.reviewer_id AND binding.idempotency_key=json_extract(NEW.payload_json,'$.idempotency_key') AND binding.request_digest=json_extract(NEW.payload_json,'$.request_digest')))
   OR (NEW.previous_state='preregistered' AND NEW.new_state='training')
   OR (NEW.previous_state='training' AND NEW.new_state='evaluating' AND EXISTS(SELECT 1 FROM training_checkpoints checkpoint WHERE checkpoint.run_id=NEW.run_id AND checkpoint.fraction=1.0 AND checkpoint.publication_state='finalized') AND (SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id)=(SELECT total_steps FROM training_runs WHERE id=NEW.run_id))
   OR (NEW.previous_state='evaluating' AND NEW.new_state IN ('qualified','rejected') AND EXISTS(SELECT 1 FROM training_evaluation_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND disposition=NEW.new_state AND evidence_digest=json_extract(NEW.payload_json,'$.evaluation_evidence_digest')))
   OR (NEW.previous_state IN ('draft','probed','calibrated','preregistered','training','evaluating') AND NEW.new_state='failed' AND EXISTS(SELECT 1 FROM training_failures WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id))
   OR (NEW.previous_state='training' AND NEW.new_state='cancelled' AND EXISTS(
    SELECT 1 FROM training_cancellation_evidence evidence WHERE evidence.run_id=NEW.run_id
     AND evidence.reviewer_id=NEW.reviewer_id
     AND evidence.id=json_extract(NEW.payload_json,'$.cancellation_id')
     AND evidence.cancellation_request_id=json_extract(NEW.payload_json,'$.cancellation_request_id')
     AND evidence.request_digest=json_extract(NEW.payload_json,'$.request_digest'))))
 THEN RAISE(ABORT,'invalid scientific training lifecycle transition') END;
END;
DROP TRIGGER validate_training_probe_insert_v36;
CREATE TRIGGER validate_training_probe_insert_v37 BEFORE INSERT ON training_probe_evidence BEGIN
 SELECT CASE WHEN length(NEW.capability_digest)!=71 OR substr(NEW.capability_digest,1,7)!='sha256:'
  OR substr(NEW.capability_digest,8) GLOB '*[^0-9a-f]*'
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='draft'
  OR NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.recipe_digest!=(SELECT recipe.id FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.dataset_digest!=(SELECT recipe.dataset_digest FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.backend_digest!=(SELECT canonical_json_digest(json_extract(recipe.recipe_json,'$.backend_config')) FROM training_runs run JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.runtime_digest!=(SELECT material.runtime_digest FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
  OR NEW.source_digest!=(SELECT 'sha256:'||artifact.sha256 FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id JOIN artifacts artifact ON artifact.id=material.source_snapshot_artifact_id WHERE run.id=NEW.run_id)
  OR NEW.material_digest!=(SELECT canonical_json_digest(json_object('stage_3_campaign_id',material.stage_3_campaign_id,'stage_3_result_id',material.stage_3_result_id,'base_model_artifact_id',material.base_model_artifact_id,'lightning_adapter_artifact_ids',json(material.lightning_adapter_artifact_ids_json),'source_snapshot_artifact_id',material.source_snapshot_artifact_id,'runtime_digest',material.runtime_digest)) FROM training_runs run JOIN training_recipe_materials material ON material.recipe_storage_id=run.recipe_storage_id WHERE run.id=NEW.run_id)
 THEN RAISE(ABORT,'training probe persisted identity mismatch') END;
END;
DROP TRIGGER validate_training_calibration_insert;
CREATE TRIGGER validate_training_calibration_insert_v37 BEFORE INSERT ON training_calibration_evidence BEGIN
 SELECT CASE WHEN length(NEW.calibration_digest)!=71 OR substr(NEW.calibration_digest,1,7)!='sha256:'
  OR substr(NEW.calibration_digest,8) GLOB '*[^0-9a-f]*'
  OR (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)!='probed'
  OR NOT EXISTS(SELECT 1 FROM training_probe_evidence probe WHERE probe.id=NEW.probe_evidence_id
   AND probe.run_id=NEW.run_id AND probe.reviewer_id=NEW.reviewer_id AND probe.succeeded=1
   AND probe.recipe_digest=NEW.recipe_digest AND probe.dataset_digest=NEW.dataset_digest
   AND probe.backend_digest=NEW.backend_digest AND probe.runtime_digest=NEW.runtime_digest
   AND probe.material_digest=NEW.material_digest AND probe.source_digest=NEW.source_digest)
 THEN RAISE(ABORT,'training calibration identity mismatch') END;
END;
CREATE TRIGGER validate_training_resume_digest_shape_v37 BEFORE INSERT ON training_resume_evidence BEGIN
 SELECT CASE WHEN sha256_digest_is_valid(NEW.parent_checkpoint_digest)!=1
  OR sha256_digest_is_valid(NEW.parent_artifact_digest)!=1
  OR sha256_digest_is_valid(NEW.recipe_digest)!=1 OR sha256_digest_is_valid(NEW.dataset_digest)!=1
  OR sha256_digest_is_valid(NEW.backend_digest)!=1 OR sha256_digest_is_valid(NEW.runtime_digest)!=1
  OR sha256_digest_is_valid(NEW.material_digest)!=1 OR sha256_digest_is_valid(NEW.source_digest)!=1
  OR sha256_digest_is_valid(NEW.optimizer_state_digest)!=1
 THEN RAISE(ABORT,'training resume digest shape mismatch') END;
END;
CREATE TRIGGER validate_training_evaluation_digest_shape_v37 BEFORE INSERT ON training_evaluation_evidence BEGIN
 SELECT CASE WHEN sha256_digest_is_valid(NEW.evidence_digest)!=1
 THEN RAISE(ABORT,'training evaluation digest shape mismatch') END;
END;
CREATE TRIGGER validate_training_idempotency_digest_shape_v37 BEFORE INSERT ON training_run_idempotency BEGIN
 SELECT CASE WHEN sha256_digest_is_valid(NEW.request_digest)!=1
 THEN RAISE(ABORT,'training idempotency digest shape mismatch') END;
END;
CREATE TRIGGER validate_training_cancel_request_digest_shape_v37 BEFORE INSERT ON training_cancellation_requests BEGIN
 SELECT CASE WHEN sha256_digest_is_valid(NEW.request_digest)!=1
 THEN RAISE(ABORT,'training cancellation request digest shape mismatch') END;
END;
"""

_MIGRATION_38 = """
CREATE TRIGGER validate_training_checkpoint_parent_chain_v38
BEFORE INSERT ON training_checkpoints BEGIN
 SELECT CASE WHEN NOT EXISTS(
  SELECT 1 FROM training_recipes recipe,
   json_each(recipe.recipe_json,'$.checkpoint_plan.fractions') declared
  WHERE recipe.storage_id=NEW.recipe_storage_id AND declared.value=NEW.fraction
   AND ((CAST(declared.key AS INTEGER)=0
     AND NEW.parent_checkpoint_id IS NULL
     AND NOT EXISTS(SELECT 1 FROM training_checkpoints prior
      WHERE prior.run_id=NEW.run_id AND prior.publication_state='finalized'))
    OR (CAST(declared.key AS INTEGER)>0
     AND (SELECT count(*) FROM training_checkpoints prior
      WHERE prior.run_id=NEW.run_id AND prior.publication_state='finalized')=CAST(declared.key AS INTEGER)
     AND EXISTS(
      SELECT 1 FROM training_checkpoints parent
      WHERE parent.id=NEW.parent_checkpoint_id AND parent.run_id=NEW.run_id
       AND parent.reviewer_id=NEW.reviewer_id
       AND parent.recipe_storage_id=NEW.recipe_storage_id
       AND parent.dataset_storage_id=NEW.dataset_storage_id
       AND parent.publication_state='finalized'
       AND parent.fraction=json_extract(
        recipe.recipe_json,'$.checkpoint_plan.fractions['||(CAST(declared.key AS INTEGER)-1)||']')
       AND parent.step=training_checkpoint_step(
        (SELECT total_steps FROM training_runs WHERE id=NEW.run_id),parent.fraction)
       AND parent.step=(SELECT max(latest.step) FROM training_checkpoints latest
        WHERE latest.run_id=NEW.run_id AND latest.publication_state='finalized')
       AND parent.evidence_digest=json_extract(NEW.evidence_json,'$.parent_checkpoint_digest'))))
 ) THEN RAISE(ABORT,'training checkpoint parent chain mismatch') END;
END;
"""

_MIGRATION_39 = """
CREATE TABLE training_orchestration_operations(
 id TEXT PRIMARY KEY,
 reviewer_id TEXT NOT NULL,
 run_id TEXT NOT NULL,
 recipe_storage_id TEXT NOT NULL,
 operation_kind TEXT NOT NULL CHECK(operation_kind IN ('probe','calibration','start','resume','cancel')),
 idempotency_key TEXT NOT NULL,
 request_digest TEXT NOT NULL,
 request_payload_json TEXT NOT NULL,
 lifecycle_version INTEGER NOT NULL CHECK(lifecycle_version>=0),
 lifecycle_state TEXT NOT NULL CHECK(lifecycle_state IN ('draft','probed','calibrated','preregistered','training','evaluating','qualified','rejected','failed','cancelled')),
 status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
 attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt>=0),
 claimant_id TEXT,
 lease_expires_at TEXT,
 receipt_digest TEXT,
 receipt_payload_json TEXT,
 failure_digest TEXT,
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,
 UNIQUE(reviewer_id,operation_kind,idempotency_key),
 FOREIGN KEY(run_id) REFERENCES training_runs(id),
 FOREIGN KEY(recipe_storage_id) REFERENCES training_recipes(storage_id)
);
CREATE TRIGGER validate_training_operation_insert_v39 BEFORE INSERT ON training_orchestration_operations BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.recipe_storage_id!=(SELECT recipe_storage_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.lifecycle_version!=(SELECT lifecycle_version FROM training_runs WHERE id=NEW.run_id)
  OR NEW.lifecycle_state!=(SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
  OR canonical_json_is_exact(NEW.request_payload_json)!=1
  OR canonical_json_digest(NEW.request_payload_json)!=NEW.request_digest
  OR json_extract(NEW.request_payload_json,'$.run_id') IS NOT NEW.run_id
  OR sha256_digest_is_valid(NEW.request_digest)!=1
  OR NEW.status!='pending' OR NEW.attempt!=0 OR NEW.claimant_id IS NOT NULL
  OR NEW.lease_expires_at IS NOT NULL OR NEW.receipt_digest IS NOT NULL
  OR NEW.receipt_payload_json IS NOT NULL OR NEW.failure_digest IS NOT NULL
 THEN RAISE(ABORT,'training operation request mismatch') END;
END;
CREATE TRIGGER validate_training_operation_update_v39 BEFORE UPDATE ON training_orchestration_operations BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id OR NEW.run_id!=OLD.run_id
  OR NEW.recipe_storage_id!=OLD.recipe_storage_id OR NEW.operation_kind!=OLD.operation_kind
  OR NEW.idempotency_key!=OLD.idempotency_key OR NEW.request_digest!=OLD.request_digest
  OR NEW.request_payload_json!=OLD.request_payload_json OR NEW.lifecycle_version!=OLD.lifecycle_version
  OR NEW.lifecycle_state!=OLD.lifecycle_state OR NEW.created_at!=OLD.created_at
  OR OLD.status IN ('completed','failed')
  OR (NEW.status='completed' AND (
      json_extract(NEW.receipt_payload_json,'$.run_id') IS NOT NEW.run_id
      OR json_extract(NEW.receipt_payload_json,'$.lifecycle_version') IS NOT (SELECT lifecycle_version FROM training_runs WHERE id=NEW.run_id)
      OR (NEW.operation_kind='probe' AND NOT EXISTS(SELECT 1 FROM training_probe_evidence e
          WHERE e.id=json_extract(NEW.receipt_payload_json,'$.evidence_id') AND e.run_id=NEW.run_id
           AND e.reviewer_id=NEW.reviewer_id AND e.succeeded=1))
      OR (NEW.operation_kind='calibration' AND NOT EXISTS(SELECT 1 FROM training_calibration_evidence e
          WHERE e.id=json_extract(NEW.receipt_payload_json,'$.evidence_id') AND e.run_id=NEW.run_id
           AND e.reviewer_id=NEW.reviewer_id AND e.succeeded=1))
      OR (NEW.operation_kind='resume' AND NOT EXISTS(SELECT 1 FROM training_resume_evidence e
          WHERE e.id=json_extract(NEW.receipt_payload_json,'$.resume_id') AND e.run_id=NEW.run_id
           AND e.reviewer_id=NEW.reviewer_id))
      OR (NEW.operation_kind='cancel' AND NOT EXISTS(SELECT 1 FROM training_cancellation_evidence e
          WHERE e.id=json_extract(NEW.receipt_payload_json,'$.cancellation_id') AND e.run_id=NEW.run_id
           AND e.reviewer_id=NEW.reviewer_id))))
  OR NOT ((OLD.status='pending' AND NEW.status='running' AND NEW.attempt=OLD.attempt+1
      AND NEW.claimant_id IS NOT NULL AND NEW.lease_expires_at IS NOT NULL
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='running' AND julianday(OLD.lease_expires_at)<=julianday('now')
      AND NEW.attempt=OLD.attempt+1 AND NEW.claimant_id IS NOT NULL AND NEW.lease_expires_at IS NOT NULL
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='completed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.lease_expires_at IS NULL AND NEW.failure_digest IS NULL
      AND NEW.receipt_payload_json IS NOT NULL AND canonical_json_is_exact(NEW.receipt_payload_json)=1
      AND canonical_json_digest(NEW.receipt_payload_json)=NEW.receipt_digest
      AND sha256_digest_is_valid(NEW.receipt_digest)=1)
   OR (OLD.status='running' AND NEW.status='failed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.lease_expires_at IS NULL
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL
      AND sha256_digest_is_valid(NEW.failure_digest)=1))
 THEN RAISE(ABORT,'training operation status transition mismatch') END;
END;
CREATE TRIGGER guard_training_operation_delete_v39 BEFORE DELETE ON training_orchestration_operations BEGIN
 SELECT RAISE(ABORT,'training operation is immutable');
END;
"""

_MIGRATION_40 = """
DROP TRIGGER validate_training_operation_insert_v39;
DROP TRIGGER validate_training_operation_update_v39;
ALTER TABLE training_orchestration_operations ADD COLUMN claim_token TEXT;
CREATE TABLE training_operation_evidence_links(
 operation_id TEXT NOT NULL,
 attempt INTEGER NOT NULL CHECK(attempt>0),
 claim_token TEXT NOT NULL,
 evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('probe','calibration','resume','cancel','event','telemetry','checkpoint','artifact','inventory','evaluation')),
 evidence_id TEXT NOT NULL,
 evidence_digest TEXT NOT NULL,
 created_at TEXT NOT NULL,
 PRIMARY KEY(operation_id,evidence_kind,evidence_id),
 UNIQUE(evidence_kind,evidence_id),
 FOREIGN KEY(operation_id) REFERENCES training_orchestration_operations(id)
);
CREATE TRIGGER validate_training_operation_insert_v40 BEFORE INSERT ON training_orchestration_operations BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.recipe_storage_id!=(SELECT recipe_storage_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.lifecycle_version!=(SELECT lifecycle_version FROM training_runs WHERE id=NEW.run_id)
  OR NEW.lifecycle_state!=(SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
  OR training_operation_json_is_exact(NEW.operation_kind,'request',NEW.request_payload_json)!=1
  OR training_operation_request_identity_is_valid(NEW.operation_kind,NEW.run_id,NEW.request_payload_json)!=1
  OR canonical_json_digest(NEW.request_payload_json)!=NEW.request_digest
  OR json_extract(NEW.request_payload_json,'$.run_id') IS NOT NEW.run_id
  OR sha256_digest_is_valid(NEW.request_digest)!=1
  OR NEW.status!='pending' OR NEW.attempt!=0 OR NEW.claimant_id IS NOT NULL
  OR NEW.claim_token IS NOT NULL OR NEW.lease_expires_at IS NOT NULL
  OR NEW.receipt_digest IS NOT NULL OR NEW.receipt_payload_json IS NOT NULL
  OR NEW.failure_digest IS NOT NULL
 THEN RAISE(ABORT,'training operation request mismatch') END;
END;
CREATE TRIGGER validate_training_operation_update_v40 BEFORE UPDATE ON training_orchestration_operations BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id OR NEW.run_id!=OLD.run_id
  OR NEW.recipe_storage_id!=OLD.recipe_storage_id OR NEW.operation_kind!=OLD.operation_kind
  OR NEW.idempotency_key!=OLD.idempotency_key OR NEW.request_digest!=OLD.request_digest
  OR NEW.request_payload_json!=OLD.request_payload_json OR NEW.lifecycle_version!=OLD.lifecycle_version
  OR NEW.lifecycle_state!=OLD.lifecycle_state OR NEW.created_at!=OLD.created_at
  OR OLD.status IN ('completed','failed')
  OR (NEW.status='completed' AND (
      training_operation_json_is_exact(NEW.operation_kind,'receipt',NEW.receipt_payload_json)!=1
      OR canonical_json_digest(NEW.receipt_payload_json)!=NEW.receipt_digest
      OR json_extract(NEW.receipt_payload_json,'$.run_id') IS NOT NEW.run_id
      OR json_extract(NEW.receipt_payload_json,'$.lifecycle_version') IS NOT (SELECT lifecycle_version FROM training_runs WHERE id=NEW.run_id)
      OR json_extract(NEW.receipt_payload_json,'$.lifecycle_state') IS NOT (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
      OR training_operation_receipt_links_are_exact(NEW.id,NEW.receipt_payload_json)!=1
      OR NOT EXISTS(SELECT 1 FROM training_operation_evidence_links link
          WHERE link.operation_id=NEW.id AND link.attempt=OLD.attempt AND link.claim_token=OLD.claim_token)))
  OR NOT ((OLD.status='pending' AND NEW.status='running' AND NEW.attempt=OLD.attempt+1
      AND NEW.claimant_id IS NOT NULL AND NEW.claim_token IS NOT NULL
      AND NEW.lease_expires_at IS NOT NULL AND NEW.receipt_digest IS NULL
      AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='running' AND julianday(OLD.lease_expires_at)<=julianday('now')
      AND NEW.attempt=OLD.attempt+1 AND NEW.claimant_id IS NOT NULL AND NEW.claim_token IS NOT NULL
      AND NEW.claim_token!=OLD.claim_token AND NEW.lease_expires_at IS NOT NULL
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='completed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.claim_token IS NULL AND NEW.lease_expires_at IS NULL
      AND NEW.failure_digest IS NULL AND NEW.receipt_payload_json IS NOT NULL
      AND sha256_digest_is_valid(NEW.receipt_digest)=1)
   OR (OLD.status='running' AND NEW.status='failed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.claim_token IS NULL AND NEW.lease_expires_at IS NULL
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL
      AND sha256_digest_is_valid(NEW.failure_digest)=1))
 THEN RAISE(ABORT,'training operation status transition mismatch') END;
END;
CREATE TRIGGER validate_training_operation_evidence_link_v40 BEFORE INSERT ON training_operation_evidence_links BEGIN
 SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM training_orchestration_operations operation
   WHERE operation.id=NEW.operation_id AND operation.status='running'
    AND operation.attempt=NEW.attempt AND operation.claim_token=NEW.claim_token)
  OR sha256_digest_is_valid(NEW.evidence_digest)!=1
  OR training_operation_evidence_digest(NEW.operation_id,NEW.evidence_kind,NEW.evidence_id)
     IS NOT NEW.evidence_digest
  OR ((SELECT operation_kind FROM training_orchestration_operations WHERE id=NEW.operation_id)
      IN ('probe','calibration','resume','cancel')
      AND NEW.evidence_kind!=(SELECT operation_kind FROM training_orchestration_operations WHERE id=NEW.operation_id))
  OR NOT (
   (NEW.evidence_kind='probe' AND EXISTS(SELECT 1 FROM training_probe_evidence e
      WHERE e.id=NEW.evidence_id AND e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='calibration' AND EXISTS(SELECT 1 FROM training_calibration_evidence e
      WHERE e.id=NEW.evidence_id AND e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='resume' AND EXISTS(SELECT 1 FROM training_resume_evidence e
      WHERE e.id=NEW.evidence_id AND e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='cancel' AND EXISTS(SELECT 1 FROM training_cancellation_evidence e
      WHERE e.id=NEW.evidence_id AND e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='event' AND EXISTS(SELECT 1 FROM training_run_events e
      WHERE e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND CAST(e.sequence AS TEXT)=NEW.evidence_id
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='telemetry' AND EXISTS(SELECT 1 FROM training_telemetry e
      WHERE e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND CAST(e.sequence AS TEXT)=NEW.evidence_id
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='checkpoint' AND EXISTS(SELECT 1 FROM training_checkpoints e
      WHERE e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND e.id=NEW.evidence_id AND e.evidence_digest=NEW.evidence_digest
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='artifact' AND EXISTS(SELECT 1 FROM artifacts e
      WHERE e.id=NEW.evidence_id AND 'sha256:'||e.sha256=NEW.evidence_digest
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='inventory' AND EXISTS(SELECT 1 FROM training_checkpoint_inventory e
      JOIN training_checkpoints checkpoint ON checkpoint.id=e.checkpoint_id
       WHERE checkpoint.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND e.checkpoint_id||':'||e.role=NEW.evidence_id
       AND julianday(checkpoint.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id))))
   OR (NEW.evidence_kind='evaluation' AND EXISTS(SELECT 1 FROM training_evaluation_evidence e
      WHERE e.run_id=(SELECT run_id FROM training_orchestration_operations WHERE id=NEW.operation_id)
       AND e.id=NEW.evidence_id
       AND julianday(e.created_at)>=julianday((SELECT updated_at FROM training_orchestration_operations WHERE id=NEW.operation_id)))))
 THEN RAISE(ABORT,'training operation evidence fence mismatch') END;
END;
CREATE TRIGGER guard_training_operation_evidence_link_update_v40 BEFORE UPDATE ON training_operation_evidence_links BEGIN SELECT RAISE(ABORT,'training operation evidence link is immutable'); END;
CREATE TRIGGER guard_training_operation_evidence_link_delete_v40 BEFORE DELETE ON training_operation_evidence_links BEGIN SELECT RAISE(ABORT,'training operation evidence link is immutable'); END;
"""

_MIGRATION_41 = """
CREATE TABLE training_checkpoint_inventory_artifacts(
 checkpoint_id TEXT NOT NULL,
 ordinal INTEGER NOT NULL,
 artifact_id TEXT NOT NULL,
 role TEXT NOT NULL,
 artifact_digest TEXT NOT NULL,
 size_bytes INTEGER NOT NULL,
 PRIMARY KEY(checkpoint_id,ordinal),
 UNIQUE(checkpoint_id,artifact_id),
 FOREIGN KEY(checkpoint_id,ordinal) REFERENCES training_checkpoint_inventory(checkpoint_id,ordinal),
 FOREIGN KEY(artifact_id) REFERENCES artifacts(id)
);
INSERT INTO training_checkpoint_inventory_artifacts
SELECT inventory.checkpoint_id,inventory.ordinal,artifact.id,inventory.role,
 inventory.artifact_digest,inventory.size_bytes
FROM training_checkpoint_inventory inventory JOIN artifacts artifact
 ON 'sha256:'||artifact.sha256=inventory.artifact_digest
 AND artifact.byte_size=inventory.size_bytes;
CREATE TRIGGER validate_training_checkpoint_inventory_artifact_insert_v41
BEFORE INSERT ON training_checkpoint_inventory_artifacts BEGIN
 SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM training_checkpoint_inventory inventory
   JOIN training_checkpoints checkpoint ON checkpoint.id=inventory.checkpoint_id
   JOIN training_runs run ON run.id=checkpoint.run_id
   JOIN artifacts artifact ON artifact.id=NEW.artifact_id
   WHERE inventory.checkpoint_id=NEW.checkpoint_id AND inventory.ordinal=NEW.ordinal
    AND inventory.role=NEW.role AND inventory.artifact_digest=NEW.artifact_digest
    AND inventory.size_bytes=NEW.size_bytes AND 'sha256:'||artifact.sha256=NEW.artifact_digest
    AND artifact.byte_size=NEW.size_bytes)
 THEN RAISE(ABORT,'training checkpoint artifact binding mismatch') END;
END;
CREATE TRIGGER guard_training_checkpoint_inventory_artifact_update_v41
BEFORE UPDATE ON training_checkpoint_inventory_artifacts BEGIN SELECT RAISE(ABORT,'training checkpoint artifact binding is immutable'); END;
CREATE TRIGGER guard_training_checkpoint_inventory_artifact_delete_v41
BEFORE DELETE ON training_checkpoint_inventory_artifacts BEGIN SELECT RAISE(ABORT,'training checkpoint artifact binding is immutable'); END;
CREATE TRIGGER validate_training_operation_lifecycle_insert_v41 BEFORE INSERT ON training_orchestration_operations BEGIN
 SELECT CASE WHEN training_operation_lifecycle_is_valid(
   NEW.operation_kind,NEW.lifecycle_state)!=1
   OR (NEW.operation_kind='resume' AND NOT EXISTS(SELECT 1 FROM training_checkpoints checkpoint
       WHERE checkpoint.id=json_extract(NEW.request_payload_json,'$.parent_checkpoint_id')
        AND checkpoint.run_id=NEW.run_id AND checkpoint.publication_state='finalized'))
 THEN RAISE(ABORT,'training operation lifecycle mismatch') END;
END;
CREATE TRIGGER validate_training_operation_completion_v41
BEFORE UPDATE OF status ON training_orchestration_operations WHEN NEW.status='completed' BEGIN
 SELECT CASE WHEN training_operation_completion_is_derived(NEW.id)!=1
 THEN RAISE(ABORT,'training operation completion is not derived from complete attempt evidence') END;
END;
"""

_MIGRATION_42 = """
DROP TRIGGER validate_training_operation_evidence_link_v40;
DROP TRIGGER validate_training_operation_update_v40;
CREATE TRIGGER validate_training_operation_update_v42
BEFORE UPDATE ON training_orchestration_operations BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id OR NEW.run_id!=OLD.run_id
  OR NEW.recipe_storage_id!=OLD.recipe_storage_id OR NEW.operation_kind!=OLD.operation_kind
  OR NEW.idempotency_key!=OLD.idempotency_key OR NEW.request_digest!=OLD.request_digest
  OR NEW.request_payload_json!=OLD.request_payload_json OR NEW.lifecycle_version!=OLD.lifecycle_version
  OR NEW.lifecycle_state!=OLD.lifecycle_state OR NEW.created_at!=OLD.created_at
  OR OLD.status IN ('completed','failed')
  OR (NEW.status='completed' AND (
      training_operation_json_is_exact(NEW.operation_kind,'receipt',NEW.receipt_payload_json)!=1
      OR canonical_json_digest(NEW.receipt_payload_json)!=NEW.receipt_digest
      OR json_extract(NEW.receipt_payload_json,'$.run_id') IS NOT NEW.run_id
      OR json_extract(NEW.receipt_payload_json,'$.lifecycle_version') IS NOT
         (SELECT lifecycle_version FROM training_runs WHERE id=NEW.run_id)
      OR json_extract(NEW.receipt_payload_json,'$.lifecycle_state') IS NOT
         (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
      OR training_operation_receipt_links_are_exact(NEW.id,NEW.receipt_payload_json)!=1
      OR NOT EXISTS(SELECT 1 FROM training_operation_evidence_links link
          WHERE link.operation_id=NEW.id)))
  OR NOT ((OLD.status='pending' AND NEW.status='running' AND NEW.attempt=OLD.attempt+1
      AND NEW.claimant_id IS NOT NULL AND NEW.claim_token IS NOT NULL
      AND NEW.lease_expires_at IS NOT NULL AND NEW.receipt_digest IS NULL
      AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='running'
      AND julianday(OLD.lease_expires_at)<=julianday('now')
      AND NEW.attempt=OLD.attempt+1 AND NEW.claimant_id IS NOT NULL
      AND NEW.claim_token IS NOT NULL AND NEW.claim_token!=OLD.claim_token
      AND NEW.lease_expires_at IS NOT NULL AND NEW.receipt_digest IS NULL
      AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='completed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.claim_token IS NULL AND NEW.lease_expires_at IS NULL
      AND NEW.failure_digest IS NULL AND NEW.receipt_payload_json IS NOT NULL
      AND sha256_digest_is_valid(NEW.receipt_digest)=1)
   OR (OLD.status='running' AND NEW.status='failed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.claim_token IS NULL AND NEW.lease_expires_at IS NULL
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL
      AND sha256_digest_is_valid(NEW.failure_digest)=1))
 THEN RAISE(ABORT,'training operation status transition mismatch') END;
END;
CREATE TABLE training_calibration_operation_bindings(
 evidence_id TEXT PRIMARY KEY,
 operation_id TEXT NOT NULL UNIQUE,
 request_digest TEXT NOT NULL,
 attempt INTEGER NOT NULL CHECK(attempt>0),
 claim_token TEXT NOT NULL,
 steps INTEGER NOT NULL CHECK(steps>0),
 trainable_targets_json TEXT NOT NULL,
 seed INTEGER NOT NULL,
 created_at TEXT NOT NULL,
 FOREIGN KEY(evidence_id) REFERENCES training_calibration_evidence(id),
 FOREIGN KEY(operation_id) REFERENCES training_orchestration_operations(id)
);
CREATE TRIGGER validate_training_calibration_operation_binding_v42
BEFORE INSERT ON training_calibration_operation_bindings BEGIN
 SELECT CASE WHEN NOT EXISTS(
  SELECT 1 FROM training_orchestration_operations operation
  JOIN training_calibration_evidence evidence ON evidence.id=NEW.evidence_id
  WHERE operation.id=NEW.operation_id AND operation.operation_kind='calibration'
   AND operation.status='running' AND operation.attempt=NEW.attempt
   AND operation.claim_token=NEW.claim_token AND operation.request_digest=NEW.request_digest
   AND evidence.run_id=operation.run_id AND evidence.reviewer_id=operation.reviewer_id
   AND evidence.recipe_digest=json_extract(operation.request_payload_json,'$.recipe_digest')
   AND evidence.dataset_digest=json_extract(operation.request_payload_json,'$.dataset_digest')
   AND evidence.backend_digest=json_extract(operation.request_payload_json,'$.backend_digest')
   AND evidence.runtime_digest=json_extract(operation.request_payload_json,'$.runtime_digest')
   AND evidence.material_digest=json_extract(operation.request_payload_json,'$.material_digest')
   AND evidence.source_digest=json_extract(operation.request_payload_json,'$.source_digest')
   AND evidence.probe_evidence_id=(SELECT json_extract(probe.receipt_payload_json,'$.evidence_id')
     FROM training_orchestration_operations probe
     WHERE probe.id=json_extract(operation.request_payload_json,'$.probe_operation_id'))
   AND NEW.steps=json_extract(operation.request_payload_json,'$.steps')
   AND NEW.trainable_targets_json=json_extract(operation.request_payload_json,'$.trainable_targets')
   AND NEW.seed=json_extract(operation.request_payload_json,'$.seed')
 ) THEN RAISE(ABORT,'training calibration operation binding mismatch') END;
END;
CREATE TRIGGER guard_training_calibration_operation_binding_update_v42
BEFORE UPDATE ON training_calibration_operation_bindings BEGIN
 SELECT RAISE(ABORT,'training calibration operation binding is immutable');
END;
CREATE TRIGGER guard_training_calibration_operation_binding_delete_v42
BEFORE DELETE ON training_calibration_operation_bindings BEGIN
 SELECT RAISE(ABORT,'training calibration operation binding is immutable');
END;
CREATE TABLE training_cancellation_operation_claims(
 cancellation_request_id TEXT PRIMARY KEY,
 operation_id TEXT NOT NULL UNIQUE,
 request_digest TEXT NOT NULL,
 requested_step INTEGER NOT NULL CHECK(requested_step>=0),
 attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt>=0),
 claim_token TEXT,
 created_at TEXT NOT NULL,
 FOREIGN KEY(cancellation_request_id) REFERENCES training_cancellation_requests(id),
 FOREIGN KEY(operation_id) REFERENCES training_orchestration_operations(id)
);
CREATE TRIGGER consume_training_cancellation_request_v42
AFTER INSERT ON training_orchestration_operations WHEN NEW.operation_kind='cancel' BEGIN
 INSERT INTO training_cancellation_operation_claims(
  cancellation_request_id,operation_id,request_digest,requested_step,attempt,claim_token,created_at)
 SELECT request.id,NEW.id,request.request_digest,request.requested_step,0,NULL,NEW.created_at
 FROM training_cancellation_requests request
 WHERE request.id=json_extract(NEW.request_payload_json,'$.cancellation_request_id')
  AND request.run_id=NEW.run_id AND request.reviewer_id=NEW.reviewer_id
  AND request.request_digest=json_extract(NEW.request_payload_json,'$.request_digest')
  AND request.requested_step=json_extract(NEW.request_payload_json,'$.requested_step');
 SELECT CASE WHEN changes()!=1
  THEN RAISE(ABORT,'training cancellation request consumption mismatch') END;
END;
CREATE TRIGGER claim_training_cancellation_request_v42
AFTER UPDATE OF status,attempt,claim_token ON training_orchestration_operations
WHEN NEW.operation_kind='cancel' AND NEW.status='running'
 AND (OLD.status!='running' OR NEW.attempt!=OLD.attempt) BEGIN
 UPDATE training_cancellation_operation_claims SET attempt=NEW.attempt,claim_token=NEW.claim_token
 WHERE operation_id=NEW.id AND attempt=OLD.attempt;
 SELECT CASE WHEN changes()!=1
  THEN RAISE(ABORT,'training cancellation request claim mismatch') END;
END;
CREATE TRIGGER guard_training_cancellation_operation_claim_delete_v42
BEFORE DELETE ON training_cancellation_operation_claims BEGIN
 SELECT RAISE(ABORT,'training cancellation request consumption is immutable');
END;
CREATE TRIGGER validate_training_operation_evidence_link_v42
BEFORE INSERT ON training_operation_evidence_links BEGIN
 SELECT CASE WHEN training_operation_evidence_link_is_valid(
   NEW.operation_id,NEW.attempt,NEW.claim_token,NEW.evidence_kind,NEW.evidence_id,NEW.evidence_digest)!=1
 THEN RAISE(ABORT,'training operation evidence fence mismatch') END;
END;
"""

_MIGRATION_43 = """
DROP TRIGGER validate_training_operation_evidence_link_v42;
CREATE TRIGGER validate_training_operation_evidence_link_v43
BEFORE INSERT ON training_operation_evidence_links BEGIN
 SELECT CASE WHEN training_operation_evidence_link_is_valid(
   NEW.operation_id,NEW.attempt,NEW.claim_token,NEW.evidence_kind,NEW.evidence_id,NEW.evidence_digest)!=1
 THEN RAISE(ABORT,'training operation evidence fence mismatch') END;
END;
CREATE TRIGGER guard_training_cancellation_operation_claim_update_v43
BEFORE UPDATE ON training_cancellation_operation_claims BEGIN
 SELECT CASE WHEN NEW.cancellation_request_id!=OLD.cancellation_request_id
  OR NEW.operation_id!=OLD.operation_id OR NEW.request_digest!=OLD.request_digest
  OR NEW.requested_step!=OLD.requested_step OR NEW.created_at!=OLD.created_at
  OR NEW.attempt!=OLD.attempt+1 OR NEW.claim_token IS NULL
  OR NEW.claim_token IS OLD.claim_token
  OR NOT EXISTS(
   SELECT 1 FROM training_orchestration_operations operation
   WHERE operation.id=OLD.operation_id AND operation.operation_kind='cancel'
    AND operation.status='running' AND operation.attempt=NEW.attempt
    AND operation.claim_token=NEW.claim_token
    AND operation.run_id=(SELECT run_id FROM training_cancellation_requests
                          WHERE id=OLD.cancellation_request_id)
    AND json_extract(operation.request_payload_json,'$.cancellation_request_id')=
        OLD.cancellation_request_id
    AND json_extract(operation.request_payload_json,'$.request_digest')=OLD.request_digest
    AND json_extract(operation.request_payload_json,'$.requested_step')=OLD.requested_step)
 THEN RAISE(ABORT,'training cancellation claim changes only through exact live reclaim') END;
END;
"""

_MIGRATION_44 = """
ALTER TABLE training_orchestration_operations ADD COLUMN result_digest TEXT
 CHECK(result_digest IS NULL OR sha256_digest_is_valid(result_digest)=1);
ALTER TABLE training_orchestration_operations ADD COLUMN result_payload_json TEXT
 CHECK(result_payload_json IS NULL OR (json_valid(result_payload_json)
  AND json_type(result_payload_json)='object'
  AND canonical_json_is_exact(result_payload_json)=1));
SELECT 1;
CREATE TRIGGER validate_training_operation_durable_result_v44
BEFORE UPDATE OF status,result_digest,result_payload_json ON training_orchestration_operations BEGIN
 SELECT CASE WHEN
  (NEW.status='completed')!=(NEW.result_digest IS NOT NULL AND NEW.result_payload_json IS NOT NULL)
  OR (NEW.result_payload_json IS NOT NULL
   AND canonical_json_digest(NEW.result_payload_json)!=NEW.result_digest)
 THEN RAISE(ABORT,'training operation durable result mismatch') END;
END;
CREATE TRIGGER guard_training_operation_durable_result_v44
BEFORE UPDATE OF result_digest,result_payload_json ON training_orchestration_operations
WHEN OLD.status='completed' BEGIN
 SELECT RAISE(ABORT,'training operation durable result is immutable');
END;
"""

_MIGRATION_45 = """
DROP TRIGGER validate_training_calibration_operation_binding_v42;
DROP TRIGGER guard_training_calibration_operation_binding_update_v42;
DROP TRIGGER guard_training_calibration_operation_binding_delete_v42;
ALTER TABLE training_calibration_operation_bindings
 RENAME TO training_calibration_operation_bindings_v44;
CREATE TABLE training_calibration_operation_bindings(
 evidence_id TEXT PRIMARY KEY,
 operation_id TEXT NOT NULL UNIQUE,
 request_digest TEXT NOT NULL,
 attempt INTEGER NOT NULL CHECK(attempt>0),
 claim_token TEXT NOT NULL,
 steps INTEGER NOT NULL CHECK(steps>=0),
 trainable_targets_json TEXT NOT NULL,
 seed INTEGER NOT NULL,
 created_at TEXT NOT NULL,
 FOREIGN KEY(evidence_id) REFERENCES training_calibration_evidence(id),
 FOREIGN KEY(operation_id) REFERENCES training_orchestration_operations(id)
);
INSERT INTO training_calibration_operation_bindings(
 evidence_id,operation_id,request_digest,attempt,claim_token,steps,
 trainable_targets_json,seed,created_at
) SELECT evidence_id,operation_id,request_digest,attempt,claim_token,steps,
 trainable_targets_json,seed,created_at
 FROM training_calibration_operation_bindings_v44;
DROP TABLE training_calibration_operation_bindings_v44;
CREATE TRIGGER validate_training_calibration_operation_binding_v42
BEFORE INSERT ON training_calibration_operation_bindings BEGIN
 SELECT CASE WHEN NOT EXISTS(
  SELECT 1 FROM training_orchestration_operations operation
  JOIN training_calibration_evidence evidence ON evidence.id=NEW.evidence_id
  WHERE operation.id=NEW.operation_id AND operation.operation_kind='calibration'
   AND operation.status='running' AND operation.attempt=NEW.attempt
   AND operation.claim_token=NEW.claim_token AND operation.request_digest=NEW.request_digest
   AND evidence.run_id=operation.run_id AND evidence.reviewer_id=operation.reviewer_id
   AND evidence.recipe_digest=json_extract(operation.request_payload_json,'$.recipe_digest')
   AND evidence.dataset_digest=json_extract(operation.request_payload_json,'$.dataset_digest')
   AND evidence.backend_digest=json_extract(operation.request_payload_json,'$.backend_digest')
   AND evidence.runtime_digest=json_extract(operation.request_payload_json,'$.runtime_digest')
   AND evidence.material_digest=json_extract(operation.request_payload_json,'$.material_digest')
   AND evidence.source_digest=json_extract(operation.request_payload_json,'$.source_digest')
   AND evidence.probe_evidence_id=(SELECT json_extract(probe.receipt_payload_json,'$.evidence_id')
     FROM training_orchestration_operations probe
     WHERE probe.id=json_extract(operation.request_payload_json,'$.probe_operation_id'))
   AND NEW.steps=json_extract(operation.request_payload_json,'$.steps')
   AND NEW.trainable_targets_json=json_extract(operation.request_payload_json,'$.trainable_targets')
   AND NEW.seed=json_extract(operation.request_payload_json,'$.seed')
 ) THEN RAISE(ABORT,'training calibration operation binding mismatch') END;
END;
CREATE TRIGGER guard_training_calibration_operation_binding_update_v42
BEFORE UPDATE ON training_calibration_operation_bindings BEGIN
 SELECT RAISE(ABORT,'training calibration operation binding is immutable');
END;
CREATE TRIGGER guard_training_calibration_operation_binding_delete_v42
BEFORE DELETE ON training_calibration_operation_bindings BEGIN
 SELECT RAISE(ABORT,'training calibration operation binding is immutable');
END;
"""


_MIGRATION_46 = """
DROP TRIGGER validate_training_event_insert_v37;
CREATE TRIGGER validate_training_event_insert_v46 BEFORE INSERT ON training_run_events BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_run_events WHERE run_id=NEW.run_id),1)
  OR NEW.previous_state!=(SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
  OR NOT ((NEW.previous_state='draft' AND NEW.new_state='probed' AND EXISTS(SELECT 1 FROM training_probe_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='probed' AND NEW.new_state='calibrated' AND EXISTS(SELECT 1 FROM training_calibration_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='calibrated' AND NEW.new_state='preregistered' AND EXISTS(SELECT 1 FROM training_run_idempotency binding WHERE binding.run_id=NEW.run_id AND binding.reviewer_id=NEW.reviewer_id AND binding.idempotency_key=json_extract(NEW.payload_json,'$.idempotency_key') AND binding.request_digest=json_extract(NEW.payload_json,'$.request_digest')))
   OR (NEW.previous_state='preregistered' AND NEW.new_state='training')
   OR (NEW.previous_state='cancelled' AND NEW.new_state='training' AND EXISTS(SELECT 1 FROM training_checkpoints checkpoint WHERE checkpoint.run_id=NEW.run_id AND checkpoint.id=json_extract(NEW.payload_json,'$.parent_checkpoint_id') AND checkpoint.evidence_digest=json_extract(NEW.payload_json,'$.parent_checkpoint_digest') AND checkpoint.publication_state='finalized'))
   OR (NEW.previous_state='training' AND NEW.new_state='evaluating' AND EXISTS(SELECT 1 FROM training_checkpoints checkpoint WHERE checkpoint.run_id=NEW.run_id AND checkpoint.fraction=1.0 AND checkpoint.publication_state='finalized') AND (SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id)=(SELECT total_steps FROM training_runs WHERE id=NEW.run_id))
   OR (NEW.previous_state='evaluating' AND NEW.new_state IN ('qualified','rejected') AND EXISTS(SELECT 1 FROM training_evaluation_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND disposition=NEW.new_state AND evidence_digest=json_extract(NEW.payload_json,'$.evaluation_evidence_digest')))
   OR (NEW.previous_state IN ('draft','probed','calibrated','preregistered','training','evaluating') AND NEW.new_state='failed' AND EXISTS(SELECT 1 FROM training_failures WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id))
   OR (NEW.previous_state='training' AND NEW.new_state='cancelled' AND EXISTS(
    SELECT 1 FROM training_cancellation_evidence evidence WHERE evidence.run_id=NEW.run_id
     AND evidence.reviewer_id=NEW.reviewer_id
     AND evidence.id=json_extract(NEW.payload_json,'$.cancellation_id')
     AND evidence.cancellation_request_id=json_extract(NEW.payload_json,'$.cancellation_request_id')
     AND evidence.request_digest=json_extract(NEW.payload_json,'$.request_digest'))))
 THEN RAISE(ABORT,'invalid scientific training lifecycle transition') END;
END;
"""


_MIGRATION_47 = """
CREATE TABLE training_resume_attempt_failures(
 operation_id TEXT NOT NULL,attempt INTEGER NOT NULL CHECK(attempt>0),
 claim_token TEXT NOT NULL,failure_digest TEXT NOT NULL,
 created_at TEXT NOT NULL,PRIMARY KEY(operation_id,attempt),
 UNIQUE(operation_id,claim_token),
 FOREIGN KEY(operation_id) REFERENCES training_orchestration_operations(id)
);
CREATE TRIGGER validate_training_resume_attempt_failure_insert_v47
BEFORE INSERT ON training_resume_attempt_failures BEGIN
 SELECT CASE WHEN sha256_digest_is_valid(NEW.failure_digest)!=1 OR NOT EXISTS(
  SELECT 1 FROM training_orchestration_operations operation
  WHERE operation.id=NEW.operation_id AND operation.operation_kind='resume'
   AND operation.status='running' AND operation.attempt=NEW.attempt
   AND operation.claim_token=NEW.claim_token)
 THEN RAISE(ABORT,'training resume attempt failure mismatch') END;
END;
CREATE TRIGGER guard_training_resume_attempt_failure_update_v47
BEFORE UPDATE ON training_resume_attempt_failures BEGIN
 SELECT RAISE(ABORT,'training resume attempt failure is immutable');
END;
CREATE TRIGGER guard_training_resume_attempt_failure_delete_v47
BEFORE DELETE ON training_resume_attempt_failures BEGIN
 SELECT RAISE(ABORT,'training resume attempt failure is immutable');
END;
DROP TRIGGER validate_training_operation_update_v42;
CREATE TRIGGER validate_training_operation_update_v47
BEFORE UPDATE ON training_orchestration_operations BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.reviewer_id!=OLD.reviewer_id OR NEW.run_id!=OLD.run_id
  OR NEW.recipe_storage_id!=OLD.recipe_storage_id OR NEW.operation_kind!=OLD.operation_kind
  OR NEW.idempotency_key!=OLD.idempotency_key OR NEW.request_digest!=OLD.request_digest
  OR NEW.request_payload_json!=OLD.request_payload_json OR NEW.lifecycle_version!=OLD.lifecycle_version
  OR NEW.lifecycle_state!=OLD.lifecycle_state OR NEW.created_at!=OLD.created_at
  OR OLD.status IN ('completed','failed')
  OR (NEW.status='completed' AND (
      training_operation_json_is_exact(NEW.operation_kind,'receipt',NEW.receipt_payload_json)!=1
      OR canonical_json_digest(NEW.receipt_payload_json)!=NEW.receipt_digest
      OR json_extract(NEW.receipt_payload_json,'$.run_id') IS NOT NEW.run_id
      OR json_extract(NEW.receipt_payload_json,'$.lifecycle_version') IS NOT
         (SELECT lifecycle_version FROM training_runs WHERE id=NEW.run_id)
      OR json_extract(NEW.receipt_payload_json,'$.lifecycle_state') IS NOT
         (SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
      OR training_operation_receipt_links_are_exact(NEW.id,NEW.receipt_payload_json)!=1
      OR NOT EXISTS(SELECT 1 FROM training_operation_evidence_links link
          WHERE link.operation_id=NEW.id)))
  OR NOT ((OLD.status='pending' AND NEW.status='running' AND NEW.attempt=OLD.attempt+1
      AND NEW.claimant_id IS NOT NULL AND NEW.claim_token IS NOT NULL
      AND NEW.lease_expires_at IS NOT NULL AND NEW.receipt_digest IS NULL
      AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='running'
      AND julianday(OLD.lease_expires_at)<=julianday('now')
      AND NEW.attempt=OLD.attempt+1 AND NEW.claimant_id IS NOT NULL
      AND NEW.claim_token IS NOT NULL AND NEW.claim_token!=OLD.claim_token
      AND NEW.lease_expires_at IS NOT NULL AND NEW.receipt_digest IS NULL
      AND NEW.receipt_payload_json IS NULL AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='running' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id=OLD.claimant_id AND NEW.claim_token=OLD.claim_token
      AND julianday(NEW.lease_expires_at)<=julianday('now')
      AND EXISTS(SELECT 1 FROM training_resume_attempt_failures failure
       WHERE failure.operation_id=OLD.id AND failure.attempt=OLD.attempt
        AND failure.claim_token=OLD.claim_token)
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL
      AND NEW.failure_digest IS NULL)
   OR (OLD.status='running' AND NEW.status='completed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.claim_token IS NULL AND NEW.lease_expires_at IS NULL
      AND NEW.failure_digest IS NULL AND NEW.receipt_payload_json IS NOT NULL
      AND sha256_digest_is_valid(NEW.receipt_digest)=1)
   OR (OLD.status='running' AND NEW.status='failed' AND NEW.attempt=OLD.attempt
      AND NEW.claimant_id IS NULL AND NEW.claim_token IS NULL AND NEW.lease_expires_at IS NULL
      AND NEW.receipt_digest IS NULL AND NEW.receipt_payload_json IS NULL
      AND sha256_digest_is_valid(NEW.failure_digest)=1))
 THEN RAISE(ABORT,'training operation status transition mismatch') END;
END;
DROP TRIGGER validate_training_event_insert_v46;
CREATE TRIGGER validate_training_event_insert_v47 BEFORE INSERT ON training_run_events BEGIN
 SELECT CASE WHEN NEW.reviewer_id!=(SELECT reviewer_id FROM training_runs WHERE id=NEW.run_id)
  OR NEW.sequence!=COALESCE((SELECT max(sequence)+1 FROM training_run_events WHERE run_id=NEW.run_id),1)
  OR NEW.previous_state!=(SELECT lifecycle_state FROM training_runs WHERE id=NEW.run_id)
  OR NOT ((NEW.previous_state='draft' AND NEW.new_state='probed' AND EXISTS(SELECT 1 FROM training_probe_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='probed' AND NEW.new_state='calibrated' AND EXISTS(SELECT 1 FROM training_calibration_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND succeeded=1))
   OR (NEW.previous_state='calibrated' AND NEW.new_state='preregistered' AND EXISTS(SELECT 1 FROM training_run_idempotency binding WHERE binding.run_id=NEW.run_id AND binding.reviewer_id=NEW.reviewer_id AND binding.idempotency_key=json_extract(NEW.payload_json,'$.idempotency_key') AND binding.request_digest=json_extract(NEW.payload_json,'$.request_digest')))
   OR (NEW.previous_state='preregistered' AND NEW.new_state='training')
   OR (NEW.previous_state='cancelled' AND NEW.new_state='training' AND EXISTS(SELECT 1 FROM training_checkpoints checkpoint WHERE checkpoint.run_id=NEW.run_id AND checkpoint.id=json_extract(NEW.payload_json,'$.parent_checkpoint_id') AND checkpoint.evidence_digest=json_extract(NEW.payload_json,'$.parent_checkpoint_digest') AND checkpoint.publication_state='finalized'))
   OR (NEW.previous_state='training' AND NEW.new_state='evaluating' AND EXISTS(SELECT 1 FROM training_checkpoints checkpoint WHERE checkpoint.run_id=NEW.run_id AND checkpoint.fraction=1.0 AND checkpoint.publication_state='finalized') AND (SELECT max(step) FROM training_telemetry WHERE run_id=NEW.run_id)=(SELECT total_steps FROM training_runs WHERE id=NEW.run_id))
   OR (NEW.previous_state='evaluating' AND NEW.new_state IN ('qualified','rejected') AND EXISTS(SELECT 1 FROM training_evaluation_evidence WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id AND disposition=NEW.new_state AND evidence_digest=json_extract(NEW.payload_json,'$.evaluation_evidence_digest')))
   OR (NEW.previous_state IN ('draft','probed','calibrated','preregistered','training','evaluating') AND NEW.new_state='failed' AND EXISTS(SELECT 1 FROM training_failures WHERE run_id=NEW.run_id AND reviewer_id=NEW.reviewer_id))
   OR (NEW.previous_state='training' AND NEW.new_state='cancelled' AND EXISTS(
    SELECT 1 FROM training_cancellation_evidence evidence WHERE evidence.run_id=NEW.run_id
     AND evidence.reviewer_id=NEW.reviewer_id
     AND evidence.id=json_extract(NEW.payload_json,'$.cancellation_id')
     AND evidence.cancellation_request_id=json_extract(NEW.payload_json,'$.cancellation_request_id')
     AND evidence.request_digest=json_extract(NEW.payload_json,'$.request_digest')))
   OR (NEW.previous_state='training' AND NEW.new_state='cancelled' AND EXISTS(
    SELECT 1 FROM training_resume_attempt_failures failure
    JOIN training_orchestration_operations operation ON operation.id=failure.operation_id
    WHERE operation.run_id=NEW.run_id AND operation.reviewer_id=NEW.reviewer_id
     AND failure.operation_id=json_extract(NEW.payload_json,'$.resume_operation_id')
     AND failure.attempt=json_extract(NEW.payload_json,'$.resume_attempt')
     AND failure.failure_digest=json_extract(NEW.payload_json,'$.resume_failure_digest'))))
 THEN RAISE(ABORT,'invalid scientific training lifecycle transition') END;
END;
"""


_MIGRATION_48 = """
CREATE TABLE IF NOT EXISTS dataset_compilation_operations(
 id TEXT PRIMARY KEY, operation_key TEXT NOT NULL UNIQUE, compiler_id TEXT NOT NULL,
 request_digest TEXT NOT NULL CHECK(length(request_digest)=64 AND request_digest NOT GLOB '*[^0-9a-f]*'),
 request_json TEXT NOT NULL CHECK(json_valid(request_json) AND canonical_json_is_exact(request_json)=1),
 status TEXT NOT NULL CHECK(status IN ('pending','running','retryable','completed','failed','cancelled')),
 attempt INTEGER NOT NULL CHECK(attempt>=0), claimant_id TEXT, claim_token TEXT,
 lease_expires_at TEXT, receipt_id TEXT UNIQUE,
 receipt_digest TEXT CHECK(receipt_digest IS NULL OR (length(receipt_digest)=64 AND receipt_digest NOT GLOB '*[^0-9a-f]*')),
 receipt_json TEXT CHECK(receipt_json IS NULL OR (json_valid(receipt_json) AND canonical_json_is_exact(receipt_json)=1)),
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 CHECK((status='running')=(claimant_id IS NOT NULL AND claim_token IS NOT NULL AND lease_expires_at IS NOT NULL)),
 CHECK((status='completed')=(receipt_id IS NOT NULL AND receipt_digest IS NOT NULL AND receipt_json IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS dataset_compilation_inputs(
 operation_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('source_manifest','transform_recipe')),
 artifact_id TEXT NOT NULL, sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
 byte_size INTEGER NOT NULL CHECK(byte_size>=0), media_type TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(operation_id,role), FOREIGN KEY(operation_id) REFERENCES dataset_compilation_operations(id),
 FOREIGN KEY(artifact_id) REFERENCES artifacts(id)
);
CREATE TABLE IF NOT EXISTS dataset_compilation_outputs(
 receipt_id TEXT NOT NULL, operation_id TEXT NOT NULL, role TEXT NOT NULL,
 artifact_id TEXT NOT NULL, sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
 byte_size INTEGER NOT NULL CHECK(byte_size>=0), media_type TEXT NOT NULL,
 attempt INTEGER NOT NULL CHECK(attempt>0), created_at TEXT NOT NULL,
 PRIMARY KEY(receipt_id,role), UNIQUE(operation_id,role),
 FOREIGN KEY(operation_id) REFERENCES dataset_compilation_operations(id),
 FOREIGN KEY(artifact_id) REFERENCES artifacts(id)
);
CREATE TABLE IF NOT EXISTS dataset_compilation_events(
 operation_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence>0),
 attempt INTEGER NOT NULL CHECK(attempt>=0), event_kind TEXT NOT NULL,
 evidence_digest TEXT NOT NULL CHECK(length(evidence_digest)=64 AND evidence_digest NOT GLOB '*[^0-9a-f]*'), created_at TEXT NOT NULL,
 PRIMARY KEY(operation_id,sequence), FOREIGN KEY(operation_id) REFERENCES dataset_compilation_operations(id)
);
CREATE TABLE IF NOT EXISTS dataset_compilation_failures(
 operation_id TEXT NOT NULL, attempt INTEGER NOT NULL CHECK(attempt>0), claim_token TEXT NOT NULL,
 code TEXT NOT NULL, failure_digest TEXT NOT NULL CHECK(length(failure_digest)=64 AND failure_digest NOT GLOB '*[^0-9a-f]*'),
 created_at TEXT NOT NULL, PRIMARY KEY(operation_id,attempt),
 FOREIGN KEY(operation_id) REFERENCES dataset_compilation_operations(id)
);
CREATE TABLE IF NOT EXISTS dataset_compilation_reproductions(
 receipt_id TEXT PRIMARY KEY, original_receipt_id TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
 FOREIGN KEY(operation_id) REFERENCES dataset_compilation_operations(id)
);
CREATE INDEX IF NOT EXISTS dataset_compilation_operation_status ON dataset_compilation_operations(status,lease_expires_at);
CREATE INDEX IF NOT EXISTS dataset_compilation_input_artifact ON dataset_compilation_inputs(artifact_id);
CREATE INDEX IF NOT EXISTS dataset_compilation_output_artifact ON dataset_compilation_outputs(artifact_id);
CREATE TRIGGER IF NOT EXISTS validate_dataset_compilation_input_insert_v48
 BEFORE INSERT ON dataset_compilation_inputs BEGIN
 SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM artifacts artifact WHERE artifact.id=NEW.artifact_id
  AND artifact.sha256=NEW.sha256 AND artifact.byte_size=NEW.byte_size
  AND artifact.media_type=NEW.media_type)
 THEN RAISE(ABORT,'dataset compilation input artifact mismatch') END;
END;
CREATE TRIGGER IF NOT EXISTS validate_dataset_compilation_output_insert_v48
 BEFORE INSERT ON dataset_compilation_outputs BEGIN
 SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM dataset_compilation_operations operation
  WHERE operation.id=NEW.operation_id AND operation.status='running'
   AND operation.attempt=NEW.attempt)
  OR NOT EXISTS(SELECT 1 FROM artifacts artifact WHERE artifact.id=NEW.artifact_id
   AND artifact.sha256=NEW.sha256 AND artifact.byte_size=NEW.byte_size
   AND artifact.media_type=NEW.media_type)
 THEN RAISE(ABORT,'dataset compilation output claim or artifact mismatch') END;
END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_operation_authority_v48
 BEFORE UPDATE ON dataset_compilation_operations BEGIN
 SELECT CASE WHEN NEW.id!=OLD.id OR NEW.operation_key!=OLD.operation_key
  OR NEW.compiler_id!=OLD.compiler_id OR NEW.request_digest!=OLD.request_digest
  OR NEW.request_json!=OLD.request_json OR NEW.created_at!=OLD.created_at
  OR OLD.status IN ('completed','failed','cancelled')
  OR (NEW.status='completed' AND (
   NEW.receipt_digest!=substr(canonical_json_digest(NEW.receipt_json),8)
   OR json_extract(NEW.receipt_json,'$.operation_id')!=NEW.id
   OR json_extract(NEW.receipt_json,'$.operation_key')!=NEW.operation_key
   OR json_extract(NEW.receipt_json,'$.compiler_id')!=NEW.compiler_id
   OR json_extract(NEW.receipt_json,'$.request_digest')!=NEW.request_digest
   OR json_extract(NEW.receipt_json,'$.attempt')!=NEW.attempt))
 THEN RAISE(ABORT,'dataset compilation operation authority mismatch') END;
END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_input_update_v48 BEFORE UPDATE ON dataset_compilation_inputs
 BEGIN SELECT RAISE(ABORT,'dataset compilation inputs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_input_delete_v48 BEFORE DELETE ON dataset_compilation_inputs
 BEGIN SELECT RAISE(ABORT,'dataset compilation inputs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_output_update_v48 BEFORE UPDATE ON dataset_compilation_outputs
 BEGIN SELECT RAISE(ABORT,'dataset compilation outputs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_output_delete_v48 BEFORE DELETE ON dataset_compilation_outputs
 BEGIN SELECT RAISE(ABORT,'dataset compilation outputs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_event_update_v48 BEFORE UPDATE ON dataset_compilation_events
 BEGIN SELECT RAISE(ABORT,'dataset compilation events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_event_delete_v48 BEFORE DELETE ON dataset_compilation_events
 BEGIN SELECT RAISE(ABORT,'dataset compilation events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_failure_update_v48 BEFORE UPDATE ON dataset_compilation_failures
 BEGIN SELECT RAISE(ABORT,'dataset compilation failures are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_failure_delete_v48 BEFORE DELETE ON dataset_compilation_failures
 BEGIN SELECT RAISE(ABORT,'dataset compilation failures are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_reproduction_update_v48 BEFORE UPDATE ON dataset_compilation_reproductions
 BEGIN SELECT RAISE(ABORT,'dataset compilation reproductions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_dataset_compilation_reproduction_delete_v48 BEFORE DELETE ON dataset_compilation_reproductions
 BEGIN SELECT RAISE(ABORT,'dataset compilation reproductions are immutable'); END;
"""


_MIGRATION_49 = """
CREATE TABLE IF NOT EXISTS curated_validation_operations(
 operation_key TEXT PRIMARY KEY,
 request_digest TEXT NOT NULL CHECK(length(request_digest)=64 AND request_digest NOT GLOB '*[^0-9a-f]*'),
 request_json TEXT NOT NULL CHECK(json_valid(request_json) AND canonical_json_is_exact(request_json)=1),
 state TEXT NOT NULL CHECK(state IN ('accepted','rejected','needs_review')),
 report_artifact_id TEXT NOT NULL,
 receipt_id TEXT NOT NULL UNIQUE,
 receipt_digest TEXT NOT NULL CHECK(length(receipt_digest)=64 AND receipt_digest NOT GLOB '*[^0-9a-f]*'),
 receipt_json TEXT NOT NULL CHECK(json_valid(receipt_json) AND canonical_json_is_exact(receipt_json)=1),
 created_at TEXT NOT NULL,
 FOREIGN KEY(report_artifact_id) REFERENCES artifacts(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS curated_validation_request_digest
 ON curated_validation_operations(request_digest);
CREATE TRIGGER IF NOT EXISTS guard_curated_validation_update_v49
 BEFORE UPDATE ON curated_validation_operations
 BEGIN SELECT RAISE(ABORT,'curated validation operations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS guard_curated_validation_delete_v49
 BEFORE DELETE ON curated_validation_operations
 BEGIN SELECT RAISE(ABORT,'curated validation operations are immutable'); END;
"""


_MIGRATIONS = {
    1: _MIGRATION_1,
    2: _MIGRATION_2,
    3: _MIGRATION_3,
    4: _MIGRATION_4,
    5: _MIGRATION_5,
    6: _MIGRATION_6,
    7: _MIGRATION_7,
    8: _MIGRATION_8,
    9: _MIGRATION_9,
    10: _MIGRATION_10,
    11: _MIGRATION_11,
    12: _MIGRATION_12,
    13: _MIGRATION_13,
    14: _MIGRATION_14,
    15: _MIGRATION_15,
    16: _MIGRATION_16,
    17: _MIGRATION_17,
    18: _MIGRATION_18,
    19: _MIGRATION_19,
    20: _MIGRATION_20,
    21: _MIGRATION_21,
    22: _MIGRATION_22,
    23: _MIGRATION_23,
    24: _MIGRATION_24,
    25: _MIGRATION_25,
    26: _MIGRATION_26,
    27: _MIGRATION_27,
    28: _MIGRATION_28,
    29: _MIGRATION_29,
    30: _MIGRATION_30,
    31: _MIGRATION_31,
    32: _MIGRATION_32,
    33: _MIGRATION_33,
    34: _MIGRATION_34,
    35: _MIGRATION_35,
    36: _MIGRATION_36,
    37: _MIGRATION_37,
    38: _MIGRATION_38,
    39: _MIGRATION_39,
    40: _MIGRATION_40,
    41: _MIGRATION_41,
    42: _MIGRATION_42,
    43: _MIGRATION_43,
    44: _MIGRATION_44,
    45: _MIGRATION_45,
    46: _MIGRATION_46,
    47: _MIGRATION_47,
    48: _MIGRATION_48,
    49: _MIGRATION_49,
}


def _validate_v31_training_recipe_materials(connection: sqlite3.Connection) -> None:
    """Version 31 has no canonical artifact handles from which exact bindings can be derived."""
    if connection.execute("SELECT 1 FROM training_recipes LIMIT 1").fetchone() is not None:
        raise sqlite3.IntegrityError("v31 training recipes require explicit material migration")


def _validate_v32_training_recipe_materials(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT recipe.recipe_json,recipe.reviewer_id AS recipe_reviewer,"
        "recipe.dataset_storage_id,"
        "material.* FROM training_recipes recipe JOIN training_recipe_materials material "
        "ON material.recipe_storage_id=recipe.storage_id"
    ).fetchall()
    recipe_count = connection.execute("SELECT count(*) FROM training_recipes").fetchone()[0]
    material_count = connection.execute(
        "SELECT count(*) FROM training_recipe_materials"
    ).fetchone()[0]
    if len(rows) != recipe_count or material_count != recipe_count:
        raise sqlite3.IntegrityError("v32 training recipe materials are incomplete")
    for row in rows:
        try:
            recipe = TrainingRecipe.model_validate_json(str(row["recipe_json"]))
            adapters = json.loads(str(row["lightning_adapter_artifact_ids_json"]))
            if not isinstance(adapters, dict) or set(adapters) != set(
                recipe.lightning_adapter_digests
            ):
                raise ValueError
            dataset = connection.execute(
                "SELECT * FROM training_datasets WHERE storage_id=? AND reviewer_id=? "
                "AND state='sealed'",
                (row["dataset_storage_id"], row["reviewer_id"]),
            ).fetchone()
            campaign = connection.execute(
                "SELECT campaign.*,result.content_digest AS result_digest,result.result_json "
                "FROM character_benchmark_campaigns campaign JOIN character_benchmark_results result "
                "ON result.campaign_id=campaign.id WHERE campaign.id=? AND campaign.reviewer_id=?",
                (row["stage_3_campaign_id"], row["reviewer_id"]),
            ).fetchone()
            base = connection.execute(
                "SELECT sha256 FROM artifacts WHERE id=?", (row["base_model_artifact_id"],)
            ).fetchone()
            source = connection.execute(
                "SELECT sha256 FROM artifacts WHERE id=?", (row["source_snapshot_artifact_id"],)
            ).fetchone()
            if dataset is None or campaign is None or base is None or source is None:
                raise ValueError
            definition = json.loads(str(campaign["definition_json"]))
            valid = (
                str(row["recipe_reviewer"]) == str(row["reviewer_id"])
                and str(dataset["dataset_digest"]) == recipe.dataset_digest
                and str(dataset["character_pack_digest"]) == recipe.character_pack_digest
                and str(campaign["character_pack_storage_id"])
                == str(dataset["character_pack_storage_id"])
                and str(campaign["result_digest"])
                == str(row["stage_3_result_id"])
                == recipe.stage_3_evidence_digest
                and json.loads(str(campaign["result_json"])).get("baseline_disposition")
                == "baseline_established"
                and definition.get("character_pack_digest") == recipe.character_pack_digest
                and definition.get("stage_2_manifest_digest") == recipe.stage_2_manifest_digest
                and definition.get("stage_2_evidence_digest") == recipe.stage_2_evidence_digest
                and str(base["sha256"]) == recipe.base_model_digest[7:]
                and str(source["sha256"]) == recipe.source_code_digest[7:]
                and str(row["runtime_digest"])
                == _canonical_json_digest(canonical_json(dict(recipe.runtime_identity)))
                and _character_benchmark_provenance_is_valid(connection, row["stage_3_campaign_id"])
                == 1
                and connection.execute(
                    "SELECT new_state FROM character_benchmark_events WHERE campaign_id=? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (row["stage_3_campaign_id"],),
                ).fetchone()[0]
                == "completed"
            )
            for name, artifact_id in adapters.items():
                artifact = connection.execute(
                    "SELECT sha256 FROM artifacts WHERE id=?", (artifact_id,)
                ).fetchone()
                valid = (
                    valid
                    and artifact is not None
                    and (
                        "sha256:" + str(artifact["sha256"])
                        == recipe.lightning_adapter_digests[name]
                    )
                )
            if not valid:
                raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError):
            raise sqlite3.IntegrityError("v32 training recipe materials are invalid") from None


def _validate_v33_training_recipe_materials(connection: sqlite3.Connection) -> None:
    """Audit every populated v33 material against the complete v34 contract."""
    try:
        _validate_v32_training_recipe_materials(connection)
    except sqlite3.IntegrityError:
        raise sqlite3.IntegrityError("v33 training recipe materials are invalid") from None
    rows = connection.execute(
        "SELECT recipe.storage_id,recipe.reviewer_id AS recipe_reviewer,recipe.recipe_json,"
        "material.reviewer_id AS material_reviewer,"
        "material.lightning_adapter_artifact_ids_json "
        "FROM training_recipes recipe JOIN training_recipe_materials material "
        "ON material.recipe_storage_id=recipe.storage_id"
    ).fetchall()
    try:
        expected_child_count = 0
        for row in rows:
            recipe = TrainingRecipe.model_validate_json(str(row["recipe_json"]))
            bindings = json.loads(str(row["lightning_adapter_artifact_ids_json"]))
            if (
                row["recipe_reviewer"] != row["material_reviewer"]
                or not isinstance(bindings, dict)
                or canonical_json(bindings) != str(row["lightning_adapter_artifact_ids_json"])
                or set(bindings) != set(recipe.lightning_adapter_digests)
            ):
                raise ValueError
            expected_child_count += len(bindings)
            children = connection.execute(
                "SELECT material_name,artifact_id,reviewer_id "
                "FROM training_recipe_lightning_materials WHERE recipe_storage_id=?",
                (row["storage_id"],),
            ).fetchall()
            child_bindings = {
                str(child["material_name"]): str(child["artifact_id"]) for child in children
            }
            if (
                len(child_bindings) != len(children)
                or child_bindings != bindings
                or any(child["reviewer_id"] != row["recipe_reviewer"] for child in children)
            ):
                raise ValueError
            for name, artifact_id in child_bindings.items():
                artifact = connection.execute(
                    "SELECT sha256 FROM artifacts WHERE id=?", (artifact_id,)
                ).fetchone()
                if (
                    artifact is None
                    or "sha256:" + str(artifact["sha256"]) != recipe.lightning_adapter_digests[name]
                ):
                    raise ValueError
        actual_child_count = connection.execute(
            "SELECT count(*) FROM training_recipe_lightning_materials"
        ).fetchone()[0]
        if actual_child_count != expected_child_count:
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise sqlite3.IntegrityError("v33 training recipe materials are invalid") from None


def _validate_v34_training_runs(connection: sqlite3.Connection) -> None:
    """Fail a v34 upgrade before DDL when persisted run identities are not exact."""
    invalid = connection.execute(
        "SELECT 1 FROM training_runs run LEFT JOIN training_recipes recipe "
        "ON recipe.storage_id=run.recipe_storage_id LEFT JOIN training_datasets dataset "
        "ON dataset.storage_id=run.dataset_storage_id WHERE recipe.storage_id IS NULL "
        "OR dataset.storage_id IS NULL OR recipe.reviewer_id!=run.reviewer_id "
        "OR recipe.dataset_storage_id!=run.dataset_storage_id "
        "OR run.total_steps!=json_extract(recipe.recipe_json,'$.total_steps') LIMIT 1"
    ).fetchone()
    if invalid is not None:
        raise sqlite3.IntegrityError("v34 training runs require exact scientific migration")


def _validate_v35_legacy_run_mapping(connection: sqlite3.Connection) -> None:
    invented_terminal = connection.execute(
        "SELECT 1 FROM training_runs WHERE lifecycle_version=0 AND state!='created' LIMIT 1"
    ).fetchone()
    if invented_terminal is not None:
        raise sqlite3.IntegrityError("legacy non-created training runs are non-runnable")
    if connection.execute("SELECT 1 FROM training_checkpoints LIMIT 1").fetchone() is not None:
        raise sqlite3.IntegrityError("v35 checkpoints require explicit scientific re-publication")


def _validate_v36_checkpoint_publication(connection: sqlite3.Connection) -> None:
    building = connection.execute(
        "SELECT count(*) FROM training_checkpoints WHERE publication_state!='finalized'"
    ).fetchone()[0]
    unmatched_cancellations = connection.execute(
        "SELECT count(*) FROM training_cancellation_evidence evidence WHERE NOT EXISTS("
        "SELECT 1 FROM training_cancellation_requests request WHERE request.run_id=evidence.run_id "
        "AND request.reviewer_id=evidence.reviewer_id "
        "AND request.requested_step=evidence.requested_step)"
    ).fetchone()[0]
    if building:
        raise sqlite3.IntegrityError("v36 building checkpoints cannot migrate")
    if unmatched_cancellations:
        raise sqlite3.IntegrityError("v36 cancellation evidence lacks an exact request")
    rows = connection.execute(
        "SELECT checkpoint.*,run.total_steps,run.reviewer_id AS run_reviewer,"
        "run.recipe_storage_id AS run_recipe_storage,"
        "run.dataset_storage_id AS run_dataset_storage,recipe.id AS recipe_id,"
        "recipe.dataset_digest,recipe.character_id,recipe.recipe_json "
        "FROM training_checkpoints checkpoint JOIN training_runs run ON run.id=checkpoint.run_id "
        "JOIN training_recipes recipe ON recipe.storage_id=run.recipe_storage_id "
        "WHERE checkpoint.publication_state='finalized' ORDER BY checkpoint.run_id,checkpoint.step"
    ).fetchall()
    previous_by_run: dict[str, sqlite3.Row] = {}
    finalized_count_by_run: dict[str, int] = {}
    try:
        for row in rows:
            manifest_json = str(row["manifest_json"])
            evidence_json = str(row["evidence_json"])
            manifest = json.loads(manifest_json)
            evidence = json.loads(evidence_json)
            if _training_checkpoint_json_is_exact(manifest_json, evidence_json) != 1:
                raise ValueError
            if _canonical_json_digest(manifest_json) != row["manifest_digest"]:
                raise ValueError
            if _canonical_json_digest(evidence_json) != row["evidence_digest"]:
                raise ValueError
            fractions = json.loads(str(row["recipe_json"]))["checkpoint_plan"]["fractions"]
            ordinal = fractions.index(row["fraction"])
            expected_step = checkpoint_steps(int(row["total_steps"]))[ordinal]
            run_id = str(row["run_id"])
            finalized_count = finalized_count_by_run.get(run_id, 0)
            previous = previous_by_run.get(run_id)
            expected_parent = None if ordinal == 0 else str(previous["id"]) if previous else None
            expected_parent_digest = None if previous is None else str(previous["evidence_digest"])
            identity = (
                ordinal == finalized_count
                and (previous is None) == (ordinal == 0)
                and row["reviewer_id"] == row["run_reviewer"]
                and row["recipe_storage_id"] == row["run_recipe_storage"]
                and row["dataset_storage_id"] == row["run_dataset_storage"]
                and row["step"] == expected_step
                and row["parent_checkpoint_id"] == expected_parent
                and manifest["run_id"] == row["run_id"]
                and manifest["reviewer_id"] == row["reviewer_id"]
                and manifest["recipe_id"] == row["recipe_id"]
                and manifest["dataset_digest"] == row["dataset_digest"]
                and manifest["character_id"] == row["character_id"]
                and manifest["step"] == row["step"]
                and manifest["fraction"] == row["fraction"]
                and manifest["parent_checkpoint_id"] == expected_parent
                and manifest["artifact_digest"] == row["artifact_digest"]
                and evidence["run_id"] == row["run_id"]
                and evidence["recipe_digest"] == row["recipe_id"]
                and evidence["dataset_digest"] == row["dataset_digest"]
                and evidence["step"] == row["step"]
                and evidence["fraction"] == row["fraction"]
                and evidence["parent_checkpoint_digest"] == expected_parent_digest
                and evidence["safetensors_digest"] == row["artifact_digest"]
            )
            shared_fields = (
                "backend_digest",
                "runtime_digest",
                "material_digest",
                "source_digest",
                "optimizer_state_digest",
                "trainer_state_digest",
                "telemetry_digest",
                "telemetry_sequence_start",
                "telemetry_sequence_end",
                "trainable_targets_digest",
                "frozen_parameters_digest",
                "finite_gradient_proof_digest",
                "cancellation_boundary",
                "inventory",
                "safetensors_size_bytes",
            )
            identity = identity and all(
                manifest[field] == evidence[field] for field in shared_fields
            )
            calibration = connection.execute(
                "SELECT 1 FROM training_calibration_evidence WHERE run_id=? AND succeeded=1 "
                "AND backend_digest=? AND runtime_digest=? AND material_digest=? AND source_digest=?",
                (
                    row["run_id"],
                    evidence["backend_digest"],
                    evidence["runtime_digest"],
                    evidence["material_digest"],
                    evidence["source_digest"],
                ),
            ).fetchone()
            telemetry_rows = connection.execute(
                "SELECT sequence,step,payload_json FROM training_telemetry WHERE run_id=? "
                "AND sequence BETWEEN ? AND ? ORDER BY sequence",
                (
                    row["run_id"],
                    evidence["telemetry_sequence_start"],
                    evidence["telemetry_sequence_end"],
                ),
            ).fetchall()
            telemetry_payload = [
                {
                    "sequence": item["sequence"],
                    "step": item["step"],
                    "payload": json.loads(str(item["payload_json"])),
                }
                for item in telemetry_rows
            ]
            expected_telemetry_digest = (
                "sha256:"
                + hashlib.sha256(canonical_json(cast(Any, telemetry_payload)).encode()).hexdigest()
            )
            identity = (
                identity
                and calibration is not None
                and len(telemetry_rows)
                == evidence["telemetry_sequence_end"] - evidence["telemetry_sequence_start"] + 1
                and evidence["telemetry_digest"] == expected_telemetry_digest
            )
            if not identity or (ordinal > 0 and previous is None):
                raise ValueError
            inventory = connection.execute(
                "SELECT relative_path,artifact_digest,size_bytes,role "
                "FROM training_checkpoint_inventory WHERE checkpoint_id=? ORDER BY ordinal",
                (row["id"],),
            ).fetchall()
            actual_inventory = [
                {
                    "relative_path": item["relative_path"],
                    "digest": item["artifact_digest"],
                    "size_bytes": item["size_bytes"],
                    "role": item["role"],
                }
                for item in inventory
            ]
            if actual_inventory != evidence["inventory"] or len(actual_inventory) != 7:
                raise ValueError
            previous_by_run[run_id] = row
            finalized_count_by_run[run_id] = finalized_count + 1
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise sqlite3.IntegrityError("v36 finalized checkpoints are not canonical") from None


def _validate_v40_checkpoint_artifact_bindings(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT inventory.checkpoint_id,inventory.ordinal,count(artifact.id) candidates "
        "FROM training_checkpoint_inventory inventory LEFT JOIN artifacts artifact "
        "ON 'sha256:'||artifact.sha256=inventory.artifact_digest "
        "AND artifact.byte_size=inventory.size_bytes "
        "GROUP BY inventory.checkpoint_id,inventory.ordinal HAVING candidates!=1"
    ).fetchall()
    if rows:
        raise sqlite3.IntegrityError(
            "v40 checkpoint inventory artifact identities are not uniquely derivable"
        )


def _migration_statements(migration: str) -> tuple[str, ...]:
    statements: list[str] = []
    pending = ""
    for line in migration.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending.strip())
            pending = ""
    if pending.strip():
        raise ValueError("incomplete migration statement")
    return tuple(statements)


def _validate_training_snapshot_lineage_history(
    connection: sqlite3.Connection,
    table: Literal["training_dataset_head_events", "training_dataset_snapshots"],
) -> None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? AND type IN ('table','view')", (table,)
    ).fetchone()
    if exists is None:
        return
    conflict = connection.execute(
        f"SELECT dataset_storage_id FROM {table} GROUP BY dataset_storage_id "
        "HAVING count(DISTINCT lineage_id)>1 ORDER BY dataset_storage_id LIMIT 1"
    ).fetchone()
    if conflict is not None:
        raise sqlite3.IntegrityError("training snapshot appears in multiple dataset lineages")


def _validate_v27_training_snapshot_bindings(connection: sqlite3.Connection) -> None:
    contradiction = connection.execute(
        "SELECT history.dataset_storage_id FROM training_dataset_snapshots history "
        "LEFT JOIN training_dataset_snapshot_lineages binding "
        "ON binding.dataset_storage_id=history.dataset_storage_id "
        "GROUP BY history.dataset_storage_id HAVING binding.dataset_storage_id IS NULL "
        "OR count(DISTINCT history.lineage_id)!=1 "
        "OR min(history.lineage_id)!=min(binding.lineage_id) LIMIT 1"
    ).fetchone()
    orphan = connection.execute(
        "SELECT binding.dataset_storage_id FROM training_dataset_snapshot_lineages binding "
        "LEFT JOIN training_dataset_snapshots history "
        "ON history.dataset_storage_id=binding.dataset_storage_id "
        "WHERE history.dataset_storage_id IS NULL LIMIT 1"
    ).fetchone()
    if contradiction is not None or orphan is not None:
        raise sqlite3.IntegrityError("training snapshot lineage history is inconsistent")


def _validate_v29_training_curation_ledgers(connection: sqlite3.Connection) -> None:
    conflict = connection.execute(
        "SELECT import_id FROM training_curation_confirmations GROUP BY import_id "
        "HAVING count(*)>1 OR min(version)!=1 OR max(version)!=1 LIMIT 1"
    ).fetchone()
    if conflict is not None:
        raise sqlite3.IntegrityError("v29 training confirmation ledger is incompatible")
    heads = connection.execute(
        "SELECT lineage.id,lineage.reviewer_id,lineage.character_id,dataset.manifest_json "
        "FROM training_dataset_lineages lineage JOIN training_datasets dataset "
        "ON dataset.storage_id=lineage.current_dataset_storage_id"
    ).fetchall()
    for head in heads:
        if (
            _training_ledger_snapshot_is_complete(
                connection,
                head["manifest_json"],
                head["id"],
                head["reviewer_id"],
                head["character_id"],
            )
            != 1
        ):
            raise sqlite3.IntegrityError("v29 training dataset ledger is incompatible")


def _rekey_character_pack_drafts_for_duplicate_provenance(
    connection: sqlite3.Connection,
) -> None:
    rows = connection.execute(
        "SELECT pack.storage_id, pack.id, pack.character_id, "
        "pack.supersedes_storage_id, predecessor.id AS supersedes_pack_id "
        "FROM character_packs AS pack "
        "LEFT JOIN character_packs AS predecessor "
        "ON predecessor.storage_id=pack.supersedes_storage_id "
        "WHERE pack.state='draft' ORDER BY pack.storage_id"
    ).fetchall()
    drafts: dict[str, tuple[str, CharacterPackDraft, str | None]] = {}
    for row in rows:
        storage_id = str(row["storage_id"])
        old_id = str(row["id"])
        reference_rows = connection.execute(
            "SELECT reference.*, annotation.* FROM character_references AS reference "
            "JOIN character_annotations AS annotation USING(pack_id, reference_id) "
            "WHERE reference.pack_id=? ORDER BY reference.ordinal",
            (storage_id,),
        ).fetchall()
        warning_rows = connection.execute(
            "SELECT * FROM character_pack_warnings WHERE pack_id=? "
            "ORDER BY left_reference_id, right_reference_id",
            (storage_id,),
        ).fetchall()
        draft = CharacterPackDraft.model_validate(
            {
                "character_id": str(row["character_id"]),
                "references": [
                    {
                        "reference_id": str(reference["reference_id"]),
                        "image_sha256": str(reference["image_sha256"]),
                        "media_type": str(reference["media_type"]),
                        "dhash_hex": str(reference["dhash_hex"]),
                        "disposition": str(reference["disposition"]),
                        "annotation": {
                            "framing": str(reference["framing"]),
                            "head_angle": str(reference["head_angle"]),
                            "body_visibility": str(reference["body_visibility"]),
                            "expression": str(reference["expression"]),
                            "lighting": str(reference["lighting"]),
                            "occlusion": str(reference["occlusion"]),
                            "identity_confirmed": bool(reference["identity_confirmed"]),
                        },
                    }
                    for reference in reference_rows
                ],
                "duplicate_warnings": [
                    {
                        "left_reference_id": str(warning["left_reference_id"]),
                        "right_reference_id": str(warning["right_reference_id"]),
                        "distance": int(warning["distance"]),
                        "resolution": str(warning["resolution"]),
                    }
                    for warning in warning_rows
                ],
                "supersedes_pack_id": (
                    None if row["supersedes_pack_id"] is None else str(row["supersedes_pack_id"])
                ),
            }
        )
        predecessor_storage_id = (
            None if row["supersedes_storage_id"] is None else str(row["supersedes_storage_id"])
        )
        drafts[storage_id] = (old_id, draft, predecessor_storage_id)

    resolved: dict[str, str] = {}
    visiting: set[str] = set()

    def resolve_identity(storage_id: str) -> str:
        if storage_id in resolved:
            return resolved[storage_id]
        if storage_id in visiting:
            raise RuntimeError("character pack identity migration lineage cycle")
        visiting.add(storage_id)
        _, draft, predecessor_storage_id = drafts[storage_id]
        if predecessor_storage_id in drafts:
            draft = draft.model_copy(
                update={"supersedes_pack_id": resolve_identity(predecessor_storage_id)}
            )
        new_id = character_pack_id(draft)
        visiting.remove(storage_id)
        resolved[storage_id] = new_id
        return new_id

    proposals = [
        (storage_id, old_id, resolve_identity(storage_id))
        for storage_id, (old_id, _, _) in drafts.items()
    ]
    destinations: dict[str, str] = {}
    for storage_id, _, new_id in proposals:
        destination = destinations.setdefault(new_id, storage_id)
        if destination != storage_id:
            raise RuntimeError("character pack identity migration destination collision")

    for storage_id, old_id, new_id in proposals:
        pack_collision = connection.execute(
            "SELECT storage_id FROM character_packs WHERE id=? AND storage_id!=?",
            (new_id, storage_id),
        ).fetchone()
        if pack_collision is not None:
            raise RuntimeError("character pack identity migration pack collision")
        for handle in {old_id, new_id}:
            alias_collision = connection.execute(
                "SELECT pack_storage_id FROM character_pack_aliases "
                "WHERE alias_id=? AND pack_storage_id!=?",
                (handle, storage_id),
            ).fetchone()
            if alias_collision is not None:
                raise RuntimeError("character pack identity migration alias collision")

    for storage_id, old_id, new_id in proposals:
        if new_id == old_id:
            continue
        updated = connection.execute(
            "UPDATE character_packs SET id=? WHERE storage_id=? AND id=? AND state='draft'",
            (new_id, storage_id, old_id),
        )
        if updated.rowcount != 1:
            raise RuntimeError("character pack identity migration lost draft")
        alias = connection.execute(
            "SELECT pack_storage_id FROM character_pack_aliases WHERE alias_id=?",
            (old_id,),
        ).fetchone()
        if alias is None:
            connection.execute(
                "INSERT INTO character_pack_aliases(alias_id, pack_storage_id) VALUES (?, ?)",
                (old_id, storage_id),
            )


def _backfill_validation_experiment_bindings(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT trial.campaign_id, trial.trial_id, "
        "COALESCE((SELECT event.new_state FROM validation_events AS event "
        "WHERE event.campaign_id = trial.campaign_id "
        "ORDER BY event.sequence DESC LIMIT 1), 'draft') AS campaign_state "
        "FROM validation_trials AS trial WHERE trial.hidden_repeat = 0 "
        "ORDER BY trial.campaign_id, trial.ordinal"
    ).fetchall()
    completed_states = {
        "ready",
        "judging",
        "analyzing",
        "qualified",
        "not_qualified",
        "inconclusive",
    }
    for row in rows:
        campaign_id = str(row["campaign_id"])
        trial_id = str(row["trial_id"])
        experiment_id = validation_experiment_id(campaign_id, trial_id)
        experiment = connection.execute(
            "SELECT schema_version FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if experiment is None:
            if str(row["campaign_state"]) in completed_states:
                raise sqlite3.IntegrityError(
                    "completed validation trial does not match a deterministic experiment"
                )
            continue
        if str(experiment["schema_version"]) != "validation-controlled.v1":
            raise sqlite3.IntegrityError(
                "validation experiment identity does not match controlled evidence"
            )
        connection.execute(
            "INSERT INTO validation_experiment_bindings(campaign_id, trial_id, experiment_id) "
            "VALUES (?, ?, ?)",
            (campaign_id, trial_id, experiment_id),
        )


def _repair_validation_repeat_review_bindings(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TRIGGER IF EXISTS guard_validation_trial_bindings_update")
    connection.execute(
        "UPDATE validation_trial_bindings AS binding SET review_id=("
        "SELECT review.id FROM validation_trials AS trial "
        "JOIN validation_campaigns AS campaign ON campaign.id=trial.campaign_id "
        "JOIN review_sessions AS session ON session.experiment_id=binding.experiment_id "
        "AND session.reviewer_id=validation_session_reviewer("
        "binding.campaign_id, binding.trial_id, campaign.reviewer_id, 1) "
        "JOIN review_revisions AS review ON review.session_id=session.id "
        "WHERE trial.campaign_id=binding.campaign_id AND trial.trial_id=binding.trial_id "
        "AND trial.hidden_repeat=1 ORDER BY review.revision DESC LIMIT 1) "
        "WHERE binding.reused_from_trial_id IS NOT NULL"
    )
    missing = connection.execute(
        "SELECT 1 FROM validation_trial_bindings WHERE reused_from_trial_id IS NOT NULL "
        "AND review_id IS NULL LIMIT 1"
    ).fetchone()
    if missing is not None:
        raise sqlite3.IntegrityError("validation repeat review binding cannot be repaired")
    connection.execute(
        "CREATE TRIGGER guard_validation_trial_bindings_update "
        "BEFORE UPDATE ON validation_trial_bindings BEGIN "
        "SELECT RAISE(ABORT, 'validation evidence is append-only'); END"
    )


def _validation_session_reviewer_sql(
    campaign_id: object, trial_id: object, reviewer_id: object, hidden_repeat: object
) -> str:
    if not all(isinstance(value, str) for value in (campaign_id, trial_id, reviewer_id)):
        raise ValueError("validation session identity requires text inputs")
    if not bool(hidden_repeat):
        return str(reviewer_id)
    payload = f"validation-review-session.v1\x00{campaign_id}\x00{trial_id}\x00{reviewer_id}"
    return "validation_reviewer_" + hashlib.sha256(payload.encode()).hexdigest()


def _adaptive_manifest_json_is_valid(value: object) -> int:
    if not isinstance(value, str):
        return 0
    try:
        manifest = Manifest.model_validate_json(value)
    except (TypeError, ValueError):
        return 0
    return int(
        manifest.schema_version == "manifest.v1"
        and manifest.protocol_version == "ab.v1"
        and value == canonical_json(manifest)
    )


def _adaptive_optimizer_config_json_is_valid(value: object) -> int:
    if not isinstance(value, str):
        return 0
    try:
        config = OptimizerConfig.model_validate_json(value)
    except (TypeError, ValueError):
        return 0
    return int(value == canonical_json(config))


def _adaptive_observation_digest(
    connection: sqlite3.Connection,
    study_id: object,
    schema_version: object,
    ordinal: object,
    optimizer_step_ordinal: object,
    value_a: object,
    value_b: object,
    experiment_id: object,
    review_id: object,
    outcome: object,
    created_at: object,
) -> str | None:
    try:
        if (
            not isinstance(study_id, str)
            or schema_version != "adaptive-observation.v1"
            or type(ordinal) is not int
            or type(optimizer_step_ordinal) is not int
            or not isinstance(value_a, (int, float))
            or not isinstance(value_b, (int, float))
            or not isinstance(experiment_id, str)
            or not isinstance(review_id, str)
            or not isinstance(outcome, str)
            or not isinstance(created_at, str)
            or ordinal <= 0
            or optimizer_step_ordinal != ordinal
        ):
            return None
        observation_created_at = datetime.fromisoformat(created_at)
        if observation_created_at.utcoffset() is None or not created_at.endswith("+00:00"):
            return None
        pair = CandidatePair(value_a=float(value_a), value_b=float(value_b))
        if pair.value_a >= pair.value_b:
            return None
        preference_outcome = PreferenceOutcome(outcome)
        row = connection.execute(
            "SELECT "
            "study.base_manifest_json, "
            "optimizer.output_digest AS optimizer_output_digest, "
            "optimizer.selected_pair_json AS optimizer_selected_pair_json, "
            "optimizer.stop_reason AS optimizer_stop_reason, "
            "optimizer.comparison_count AS optimizer_comparison_count, "
            "review.id AS review_id, review.session_id, review.preferred_variant_id, "
            "review.display_preference, review.notes, "
            "review.created_at AS review_created_at, review.revision, "
            "review.supersedes_review_id, "
            "session.experiment_id, session.reviewer_id, "
            "session.created_at AS session_created_at, "
            "session.left_variant_id, session.right_variant_id, "
            "session.left_job_id, session.right_job_id, "
            "session.left_artifact_id, session.right_artifact_id, "
            "session.left_artifact_sha256, session.right_artifact_sha256, "
            "experiment.schema_version AS experiment_schema_version, "
            "experiment.variable_path, experiment.changed_paths_json, "
            "left_variant.experiment_id AS left_variant_experiment_id, "
            "left_variant.label AS left_label, "
            "left_variant.manifest_json AS left_manifest_json, "
            "right_variant.experiment_id AS right_variant_experiment_id, "
            "right_variant.label AS right_label, "
            "right_variant.manifest_json AS right_manifest_json, "
            "left_job.variant_id AS left_job_variant_id, "
            "left_job.state AS left_job_state, "
            "right_job.variant_id AS right_job_variant_id, "
            "right_job.state AS right_job_state, "
            "left_artifact.sha256 AS left_persisted_sha256, "
            "right_artifact.sha256 AS right_persisted_sha256 "
            "FROM adaptive_studies AS study "
            "JOIN adaptive_optimizer_steps AS optimizer "
            "ON optimizer.study_id = study.id AND optimizer.ordinal = ? "
            "JOIN review_revisions AS review ON review.id = ? "
            "JOIN review_sessions AS session ON session.id = review.session_id "
            "JOIN experiments AS experiment ON experiment.id = session.experiment_id "
            "JOIN variants AS left_variant ON left_variant.id = session.left_variant_id "
            "JOIN variants AS right_variant ON right_variant.id = session.right_variant_id "
            "JOIN jobs AS left_job ON left_job.id = session.left_job_id "
            "JOIN jobs AS right_job ON right_job.id = session.right_job_id "
            "JOIN artifacts AS left_artifact ON left_artifact.id = session.left_artifact_id "
            "JOIN artifacts AS right_artifact ON right_artifact.id = session.right_artifact_id "
            "JOIN job_artifacts AS left_link ON left_link.job_id = session.left_job_id "
            "AND left_link.artifact_id = session.left_artifact_id "
            "JOIN job_artifacts AS right_link ON right_link.job_id = session.right_job_id "
            "AND right_link.artifact_id = session.right_artifact_id "
            "WHERE study.id = ? AND session.experiment_id = ? "
            "AND review.id = ("
            "SELECT latest.id FROM review_revisions AS latest "
            "WHERE latest.session_id = session.id ORDER BY latest.revision DESC LIMIT 1"
            ")",
            (optimizer_step_ordinal, review_id, study_id, experiment_id),
        ).fetchone()
        if row is None:
            return None
        base_manifest = Manifest.model_validate_json(str(row["base_manifest_json"]))
        left_manifest = Manifest.model_validate_json(str(row["left_manifest_json"]))
        right_manifest = Manifest.model_validate_json(str(row["right_manifest_json"]))
        selected_pair = CandidatePair.model_validate_json(str(row["optimizer_selected_pair_json"]))
        if (
            str(row["optimizer_stop_reason"]) != "continue"
            or int(row["optimizer_comparison_count"]) != ordinal - 1
            or selected_pair != pair
            or str(row["experiment_schema_version"]) != "manifest.v1"
            or str(row["variable_path"]) != "guidance_scale"
            or str(row["changed_paths_json"]) != '["guidance_scale"]'
            or str(row["left_variant_experiment_id"]) != experiment_id
            or str(row["right_variant_experiment_id"]) != experiment_id
            or str(row["left_job_variant_id"]) != str(row["left_variant_id"])
            or str(row["right_job_variant_id"]) != str(row["right_variant_id"])
            or str(row["left_job_state"]) != "completed"
            or str(row["right_job_state"]) != "completed"
            or str(row["left_persisted_sha256"]) != str(row["left_artifact_sha256"])
            or str(row["right_persisted_sha256"]) != str(row["right_artifact_sha256"])
            or not _manifest_matches_study(left_manifest, base_manifest)
            or not _manifest_matches_study(right_manifest, base_manifest)
            or min(left_manifest.guidance_scale, right_manifest.guidance_scale) != pair.value_a
            or max(left_manifest.guidance_scale, right_manifest.guidance_scale) != pair.value_b
            or _review_outcome(row, left_manifest, right_manifest, pair) is not preference_outcome
        ):
            return None
        return adaptive_observation_provenance_digest(
            observation={
                "study_id": study_id,
                "schema_version": schema_version,
                "ordinal": ordinal,
                "optimizer_step_ordinal": optimizer_step_ordinal,
                "pair": pair.model_dump(mode="json"),
                "experiment_id": experiment_id,
                "review_id": review_id,
                "outcome": outcome,
                "created_at": created_at,
            },
            optimizer_output_digest=str(row["optimizer_output_digest"]),
            experiment={
                "id": str(row["experiment_id"]),
                "schema_version": str(row["experiment_schema_version"]),
                "variable_path": str(row["variable_path"]),
                "changed_paths": ["guidance_scale"],
            },
            review={
                "id": str(row["review_id"]),
                "session_id": str(row["session_id"]),
                "reviewer_id": str(row["reviewer_id"]),
                "revision": int(row["revision"]),
                "supersedes_review_id": row["supersedes_review_id"],
                "display_preference": str(row["display_preference"]),
                "preferred_variant_id": row["preferred_variant_id"],
                "notes": row["notes"],
                "session_created_at": str(row["session_created_at"]),
                "review_created_at": str(row["review_created_at"]),
            },
            display_mapping={
                "left": _display_provenance(row, left_manifest, "left"),
                "right": _display_provenance(row, right_manifest, "right"),
            },
        )
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return None


def _manifest_matches_study(candidate: Manifest, base: Manifest) -> bool:
    payload = candidate.model_dump(mode="json")
    payload["guidance_scale"] = base.guidance_scale
    return payload == base.model_dump(mode="json")


def _review_outcome(
    row: sqlite3.Row,
    left_manifest: Manifest,
    right_manifest: Manifest,
    pair: CandidatePair,
) -> PreferenceOutcome | None:
    preference = str(row["display_preference"])
    preferred_variant_id = row["preferred_variant_id"]
    if preference == "tie":
        return PreferenceOutcome.TIE if preferred_variant_id is None else None
    position = "left" if preference == "left" else "right" if preference == "right" else None
    if position is None or str(preferred_variant_id) != str(row[f"{position}_variant_id"]):
        return None
    preferred_value = (
        left_manifest.guidance_scale if position == "left" else right_manifest.guidance_scale
    )
    return PreferenceOutcome.A if preferred_value == pair.value_a else PreferenceOutcome.B


def _display_provenance(row: sqlite3.Row, manifest: Manifest, position: str) -> dict[str, object]:
    return {
        "variant_id": str(row[f"{position}_variant_id"]),
        "variant_label": str(row[f"{position}_label"]),
        "manifest": manifest.model_dump(mode="json"),
        "job_id": str(row[f"{position}_job_id"]),
        "artifact_id": str(row[f"{position}_artifact_id"]),
        "artifact_sha256": str(row[f"{position}_artifact_sha256"]),
    }


class _ClosingConnection(sqlite3.Connection):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()
        return False


class Database:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only

    def connect(self) -> sqlite3.Connection:
        target = f"file:{self.path}?mode=ro" if self.read_only else self.path
        connection = sqlite3.connect(
            target,
            isolation_level=None,
            uri=self.read_only,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "adaptive_manifest_json_is_valid",
            1,
            _adaptive_manifest_json_is_valid,
            deterministic=True,
        )
        connection.create_function(
            "training_operation_request_identity_is_valid",
            3,
            lambda kind, run_id, payload: _training_operation_request_identity_is_valid(
                connection, kind, run_id, payload
            ),
        )
        connection.create_function(
            "training_operation_completion_is_derived",
            1,
            lambda operation_id: _training_operation_completion_is_derived(
                connection, operation_id
            ),
        )
        connection.create_function(
            "training_operation_evidence_digest",
            3,
            lambda operation_id, kind, evidence_id: _training_operation_evidence_digest(
                connection, operation_id, kind, evidence_id
            ),
        )
        connection.create_function(
            "training_operation_evidence_link_is_valid",
            6,
            lambda operation_id, attempt, claim_token, kind, evidence_id, digest: (
                _training_operation_evidence_link_is_valid(
                    connection, operation_id, attempt, claim_token, kind, evidence_id, digest
                )
            ),
        )
        connection.create_function(
            "training_operation_receipt_links_are_exact",
            2,
            lambda operation_id, receipt: _training_operation_receipt_links_are_exact(
                connection, operation_id, receipt
            ),
        )
        connection.create_function(
            "adaptive_optimizer_config_json_is_valid",
            1,
            _adaptive_optimizer_config_json_is_valid,
            deterministic=True,
        )
        connection.create_function(
            "adaptive_observation_provenance_digest",
            10,
            lambda *arguments: _adaptive_observation_digest(connection, *arguments),
        )
        connection.create_function(
            "validation_session_reviewer",
            4,
            _validation_session_reviewer_sql,
            deterministic=True,
        )
        connection.create_function(
            "character_benchmark_reviewer_id",
            3,
            character_benchmark_reviewer_id,
            deterministic=True,
        )
        connection.create_function(
            "character_benchmark_experiment_id",
            2,
            character_benchmark_experiment_id,
            deterministic=True,
        )
        connection.create_function(
            "training_dataset_identity_is_valid",
            3,
            lambda storage_id, digest, manifest: _training_dataset_identity_is_valid(
                connection, storage_id, digest, manifest
            ),
        )
        connection.create_function(
            "training_recipe_identity_is_valid",
            3,
            lambda dataset_storage_id, recipe_id, recipe_json: _training_recipe_identity_is_valid(
                connection, dataset_storage_id, recipe_id, recipe_json
            ),
        )
        connection.create_function(
            "training_draft_identity_is_valid",
            3,
            lambda storage_id, identity, manifest: _training_draft_identity_is_valid(
                connection, storage_id, identity, manifest
            ),
        )
        connection.create_function(
            "training_draft_manifest_is_valid",
            7,
            _training_draft_manifest_is_valid,
            deterministic=True,
        )
        connection.create_function(
            "canonical_json_digest", 1, _canonical_json_digest, deterministic=True
        )
        connection.create_function(
            "canonical_json_is_exact", 1, _canonical_json_is_exact, deterministic=True
        )
        connection.create_function(
            "training_operation_json_is_exact",
            3,
            _training_operation_json_is_exact,
            deterministic=True,
        )
        connection.create_function(
            "training_operation_lifecycle_is_valid",
            2,
            _training_operation_lifecycle_is_valid,
            deterministic=True,
        )
        connection.create_function(
            "training_operation_completion_target_is_valid",
            2,
            _training_operation_completion_target_is_valid,
            deterministic=True,
        )
        connection.create_function(
            "training_import_identity_is_valid",
            2,
            _training_import_identity_is_valid,
            deterministic=True,
        )
        connection.create_function(
            "training_confirmation_identity_is_valid",
            2,
            _training_confirmation_identity_is_valid,
            deterministic=True,
        )
        connection.create_function(
            "training_ledger_snapshot_is_complete",
            4,
            lambda manifest, lineage, reviewer, character: _training_ledger_snapshot_is_complete(
                connection, manifest, lineage, reviewer, character
            ),
        )
        connection.create_function(
            "training_ledger_transition_is_valid",
            3,
            lambda old_storage, new_storage, state: _training_ledger_transition_is_valid(
                connection, old_storage, new_storage, state
            ),
        )
        connection.create_function(
            "training_checkpoint_step", 2, _training_checkpoint_step, deterministic=True
        )
        connection.create_function(
            "training_checkpoint_json_is_exact",
            2,
            _training_checkpoint_json_is_exact,
            deterministic=True,
        )
        connection.create_function(
            "sha256_digest_is_valid",
            1,
            lambda value: int(_sha256_digest_is_valid(value)),
            deterministic=True,
        )
        connection.create_function(
            "character_benchmark_provenance_is_valid",
            1,
            lambda campaign_id: _character_benchmark_provenance_is_valid(connection, campaign_id),
        )
        connection.execute("PRAGMA foreign_keys=ON")
        if not self.read_only:
            connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def migrate(self) -> None:
        if self.read_only:
            raise RuntimeError("read-only database cannot be migrated")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations("
                    "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
                )
                applied = {
                    int(row["version"])
                    for row in connection.execute("SELECT version FROM schema_migrations")
                }
                for version, migration in _MIGRATIONS.items():
                    if version in applied:
                        continue
                    if version == 13:
                        _repair_validation_repeat_review_bindings(connection)
                    if version == 26:
                        _validate_training_snapshot_lineage_history(
                            connection, "training_dataset_head_events"
                        )
                    if version == 27:
                        _validate_training_snapshot_lineage_history(
                            connection, "training_dataset_snapshots"
                        )
                    if version == 28:
                        _validate_v27_training_snapshot_bindings(connection)
                    if version == 30:
                        _validate_v29_training_curation_ledgers(connection)
                    if version == 32:
                        _validate_v31_training_recipe_materials(connection)
                    if version == 33:
                        _validate_v32_training_recipe_materials(connection)
                    if version == 34:
                        _validate_v33_training_recipe_materials(connection)
                    if version == 35:
                        _validate_v34_training_runs(connection)
                    if version == 36:
                        _validate_v35_legacy_run_mapping(connection)
                    if version == 37:
                        _validate_v36_checkpoint_publication(connection)
                    if version == 41:
                        _validate_v40_checkpoint_artifact_bindings(connection)
                    if version == 44:
                        _apply_migration_44(connection)
                    else:
                        for statement in _migration_statements(migration):
                            if statement.strip() and not statement.lstrip().startswith(
                                "CREATE TABLE schema_migrations"
                            ):
                                connection.execute(statement)
                    if version == 12:
                        _backfill_validation_experiment_bindings(connection)
                    if version == 21:
                        _rekey_character_pack_drafts_for_duplicate_provenance(connection)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (version, utc_now().isoformat()),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
