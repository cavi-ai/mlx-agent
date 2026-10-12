"""Transactional persistence for trainer-neutral character-training evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from video_lab.database import Database
from video_lab.domain import FrozenModel, canonical_json, utc_now
from video_lab.training_dataset import validate_training_recipe_dataset
from video_lab.training_models import (
    FrozenTrainingDataset,
    ImportedTrainingSample,
    TrainingCalibrationEvidence,
    TrainingCancellationEvidence,
    TrainingCheckpoint,
    TrainingCheckpointEvidence,
    TrainingCurationConfirmation,
    TrainingDatasetDraft,
    TrainingEvaluationEvidence,
    TrainingFailure,
    TrainingOperationKind,
    TrainingOperationStatus,
    TrainingOrchestrationOperation,
    TrainingProbeEvidence,
    TrainingRecipe,
    TrainingResumeEvidence,
    TrainingRun,
    TrainingRunEvent,
    TrainingSample,
    TrainingTelemetry,
    training_dataset_id,
    training_operation_payload_digest,
    training_recipe_id,
)


class TrainingNotFound(LookupError):
    pass


class TrainingConflict(RuntimeError):
    pass


class DatasetRecord(FrozenModel):
    storage_id: str
    dataset_id: str
    reviewer_id: str
    character_id: str
    state: Literal["draft", "sealed"]
    dataset_digest: str | None
    draft: TrainingDatasetDraft
    frozen_dataset: FrozenTrainingDataset | None = None


class RecipeMaterialBinding(FrozenModel):
    """Persistence-only handles for independently resolving recipe evidence."""

    stage_3_campaign_id: str
    stage_3_result_id: str
    base_model_artifact_id: str
    lightning_adapter_artifact_ids: Mapping[str, str]
    source_snapshot_artifact_id: str
    runtime_digest: str


def recipe_material_digest(binding: RecipeMaterialBinding) -> str:
    payload = canonical_json(binding.model_dump(mode="json"))
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


class RecipeRecord(FrozenModel):
    storage_id: str
    recipe_id: str
    reviewer_id: str
    state: Literal["draft", "preregistered"]
    recipe: TrainingRecipe
    binding: RecipeMaterialBinding


class RunRecord(FrozenModel):
    run: TrainingRun
    reviewer_id: str
    state: Literal[
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
    ]
    created_at: str
    terminal_at: str | None


class EventRecord(FrozenModel):
    run_id: str
    reviewer_id: str
    event: TrainingRunEvent
    created_at: str


class TelemetryRecord(FrozenModel):
    run_id: str
    reviewer_id: str
    sample: TrainingTelemetry
    created_at: str


class CheckpointRecord(FrozenModel):
    reviewer_id: str
    checkpoint: TrainingCheckpoint
    created_at: str


class FailureRecord(FrozenModel):
    reviewer_id: str
    failure: TrainingFailure
    created_at: str


class TrainingOperationClaim(FrozenModel):
    operation: TrainingOrchestrationOperation
    claimed: bool


class DatasetLineageRecord(FrozenModel):
    lineage_id: str
    reviewer_id: str
    character_id: str
    current_dataset_id: str
    head_version: int
    state: Literal["draft", "sealed"]
    created_at: str


class DatasetHeadEventRecord(FrozenModel):
    lineage_id: str
    version: int
    previous_dataset_id: str | None
    dataset_id: str
    state: Literal["draft", "sealed"]
    created_at: str


class TrainingRepository:
    def __init__(self, database: Database, *, reviewer_id: str) -> None:
        reviewer = unicodedata.normalize("NFKC", reviewer_id).strip()
        if not reviewer or reviewer != reviewer_id:
            raise ValueError("reviewer_id must be normalized non-empty text")
        self.database = database
        self.reviewer_id = reviewer

    def create_dataset_lineage(self, lineage_id: str, dataset_id: str) -> DatasetLineageRecord:
        current = self.get_dataset(dataset_id)
        with self.database.connect() as connection:
            declared = connection.execute(
                "SELECT lineage_id FROM training_datasets WHERE storage_id=?",
                (current.storage_id,),
            ).fetchone()
        if declared is None or str(declared["lineage_id"] or "") != lineage_id:
            raise TrainingConflict("dataset snapshot did not declare this lineage at creation")
        return self.get_dataset_lineage(lineage_id)

    def create_dataset_for_lineage(
        self, lineage_id: str, character_id: str, draft: TrainingDatasetDraft
    ) -> DatasetLineageRecord:
        self.create_dataset(character_id, draft, lineage_id=lineage_id)
        return self.get_dataset_lineage(lineage_id)

    def get_dataset_lineage(self, lineage_id: str) -> DatasetLineageRecord:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT l.*,d.id dataset_id FROM training_dataset_lineages l "
                "JOIN training_datasets d "
                "ON d.storage_id=l.current_dataset_storage_id WHERE l.id=? AND l.reviewer_id=?",
                (lineage_id, self.reviewer_id),
            ).fetchone()
        if row is None:
            raise TrainingNotFound("training dataset lineage not found")
        return DatasetLineageRecord(
            lineage_id=str(row["id"]),
            reviewer_id=str(row["reviewer_id"]),
            character_id=str(row["character_id"]),
            current_dataset_id=str(row["dataset_id"]),
            head_version=int(row["head_version"]),
            state=cast(Literal["draft", "sealed"], str(row["state"])),
            created_at=str(row["created_at"]),
        )

    def list_dataset_lineages(self) -> tuple[DatasetLineageRecord, ...]:
        with self.database.connect() as connection:
            ids = tuple(
                str(r[0])
                for r in connection.execute(
                    "SELECT id FROM training_dataset_lineages WHERE reviewer_id=? "
                    "ORDER BY created_at,id",
                    (self.reviewer_id,),
                )
            )
        return tuple(self.get_dataset_lineage(value) for value in ids)

    def list_dataset_head_history(self, lineage_id: str) -> tuple[DatasetHeadEventRecord, ...]:
        self.get_dataset_lineage(lineage_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT e.*,p.id previous_id,d.id dataset_id FROM training_dataset_head_events e "
                "LEFT JOIN training_datasets p ON p.storage_id=e.previous_dataset_storage_id "
                "JOIN training_datasets d ON d.storage_id=e.dataset_storage_id "
                "WHERE e.lineage_id=? AND e.reviewer_id=? ORDER BY e.version",
                (lineage_id, self.reviewer_id),
            ).fetchall()
        return tuple(
            DatasetHeadEventRecord(
                lineage_id=lineage_id,
                version=int(r["version"]),
                previous_dataset_id=cast(str | None, r["previous_id"]),
                dataset_id=str(r["dataset_id"]),
                state=cast(Literal["draft", "sealed"], str(r["state"])),
                created_at=str(r["created_at"]),
            )
            for r in rows
        )

    def persist_imported_sample_with_connection(
        self, connection: sqlite3.Connection, imported: ImportedTrainingSample
    ) -> None:
        self._require_owned_connection(connection)
        if imported.reviewer_id != self.reviewer_id:
            raise TrainingConflict("import reviewer mismatch")
        state = connection.execute(
            "SELECT state FROM training_dataset_lineages WHERE id=? AND reviewer_id=?",
            (imported.lineage_id, self.reviewer_id),
        ).fetchone()
        if state is not None and str(state["state"]) == "sealed":
            raise TrainingConflict("sealed training lineage rejects imports")
        try:
            connection.execute(
                "INSERT INTO training_imported_samples VALUES (?,?,?,?,?,?,?,?)",
                (
                    imported.import_id,
                    imported.reviewer_id,
                    imported.character_id,
                    imported.lineage_id,
                    imported.artifact_id,
                    imported.artifact_digest,
                    canonical_json(imported.model_dump(mode="json")),
                    imported.model_dump(mode="json")["imported_at"],
                ),
            )
        except sqlite3.DatabaseError:
            raise TrainingConflict("imported sample identity conflict") from None

    def persist_confirmation_with_connection(
        self, connection: sqlite3.Connection, confirmation: TrainingCurationConfirmation
    ) -> None:
        self._require_owned_connection(connection)
        if confirmation.reviewer_id != self.reviewer_id:
            raise TrainingConflict("confirmation reviewer mismatch")
        state = connection.execute(
            "SELECT lineage.state FROM training_imported_samples imported "
            "JOIN training_dataset_lineages lineage ON lineage.id=imported.lineage_id "
            "WHERE imported.id=? AND lineage.reviewer_id=?",
            (confirmation.import_id, self.reviewer_id),
        ).fetchone()
        if state is not None and str(state["state"]) == "sealed":
            raise TrainingConflict("sealed training lineage rejects confirmations")
        try:
            connection.execute(
                "INSERT INTO training_curation_confirmations VALUES (?,?,?,?,?,?,?)",
                (
                    confirmation.confirmation_id,
                    confirmation.import_id,
                    confirmation.reviewer_id,
                    confirmation.version,
                    confirmation.decision.value,
                    canonical_json(confirmation.model_dump(mode="json")),
                    confirmation.model_dump(mode="json")["confirmed_at"],
                ),
            )
        except sqlite3.DatabaseError:
            raise TrainingConflict("training confirmation conflict") from None

    def list_imported_samples(self, lineage_id: str) -> tuple[ImportedTrainingSample, ...]:
        self.get_dataset_lineage(lineage_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM training_imported_samples "
                "WHERE lineage_id=? AND reviewer_id=? ORDER BY created_at,id",
                (lineage_id, self.reviewer_id),
            ).fetchall()
        return tuple(ImportedTrainingSample.model_validate_json(str(r[0])) for r in rows)

    def list_confirmations(self, import_id: str) -> tuple[TrainingCurationConfirmation, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT c.payload_json FROM training_curation_confirmations c "
                "JOIN training_imported_samples i ON i.id=c.import_id "
                "WHERE c.import_id=? AND c.reviewer_id=? AND i.reviewer_id=? "
                "ORDER BY c.version",
                (import_id, self.reviewer_id, self.reviewer_id),
            ).fetchall()
        return tuple(TrainingCurationConfirmation.model_validate_json(str(r[0])) for r in rows)

    def publish_dataset_head_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        lineage_id: str,
        expected_head_version: int,
        draft: TrainingDatasetDraft | FrozenTrainingDataset,
        seal: bool = False,
    ) -> str:
        """Publish one immutable snapshot and CAS its lineage head without committing."""
        self._require_owned_connection(connection)
        base = self._draft_from(draft)
        state = "sealed" if seal else "draft"
        if seal and not isinstance(draft, FrozenTrainingDataset):
            raise TrainingConflict("sealing requires a frozen dataset")
        head = connection.execute(
            "SELECT * FROM training_dataset_lineages WHERE id=? AND reviewer_id=?",
            (lineage_id, self.reviewer_id),
        ).fetchone()
        if (
            head is None
            or int(head["head_version"]) != expected_head_version
            or str(head["state"]) != "draft"
            or str(head["character_id"]) != base.character_id
        ):
            raise TrainingConflict("stale or invalid dataset head")
        previous_storage_id = str(head["current_dataset_storage_id"])
        storage_id = previous_storage_id if seal else f"training_dataset_storage_{uuid.uuid4().hex}"
        dataset_id = training_dataset_id(base)
        pack = connection.execute(
            "SELECT storage_id FROM character_packs WHERE character_id=? "
            "AND reviewer_id=? AND state='preregistered' "
            "AND canonical_json_digest(manifest_json)=?",
            (base.character_id, self.reviewer_id, base.character_pack_digest),
        ).fetchone()
        if pack is None:
            raise TrainingConflict("dataset character pack provenance mismatch")
        manifest = draft.model_dump(mode="json") if seal else base.model_dump(mode="json")
        digest = draft.dataset_digest if isinstance(draft, FrozenTrainingDataset) and seal else None
        now = utc_now().isoformat()
        try:
            if seal:
                current = connection.execute(
                    "SELECT id,manifest_json FROM training_datasets WHERE storage_id=?",
                    (previous_storage_id,),
                ).fetchone()
                if current is None or str(current["id"]) != dataset_id:
                    raise TrainingConflict("frozen dataset is not the current head")
                connection.execute(
                    "UPDATE training_datasets SET state='sealed',dataset_digest=?,"
                    "manifest_json=?,sealed_at=? WHERE storage_id=?",
                    (digest, canonical_json(manifest), now, storage_id),
                )
            else:
                connection.execute(
                    "INSERT INTO training_datasets(storage_id,id,reviewer_id,character_id,"
                    "character_pack_storage_id,character_pack_digest,state,canonical_outfit_id,"
                    "reserved_token,source_tier_weights_json,manifest_json,dataset_digest,"
                    "created_at,sealed_at,lineage_id) "
                    "VALUES (?,?,?,?,?,?,'draft',?,?,?,?,NULL,?,NULL,?)",
                    (
                        storage_id,
                        dataset_id,
                        self.reviewer_id,
                        base.character_id,
                        str(pack["storage_id"]),
                        base.character_pack_digest,
                        base.canonical_outfit_id,
                        base.reserved_token,
                        canonical_json(
                            {
                                tier.value: weight
                                for tier, weight in base.source_tier_weights.items()
                            }
                        ),
                        canonical_json(base.model_dump(mode="json")),
                        now,
                        lineage_id,
                    ),
                )
            version = expected_head_version + 1
            cursor = connection.execute(
                "UPDATE training_dataset_lineages SET current_dataset_storage_id=?,"
                "head_version=?,state=? WHERE id=? AND reviewer_id=? "
                "AND head_version=? AND state='draft'",
                (
                    storage_id,
                    version,
                    state,
                    lineage_id,
                    self.reviewer_id,
                    expected_head_version,
                ),
            )
            if cursor.rowcount != 1:
                raise TrainingConflict("stale dataset head")
        except TrainingConflict:
            raise
        except sqlite3.DatabaseError:
            raise TrainingConflict("dataset head publication conflict") from None
        return str(digest or dataset_id)

    def _require_owned_connection(self, connection: sqlite3.Connection) -> None:
        try:
            rows = connection.execute("PRAGMA database_list").fetchall()
        except sqlite3.Error:
            raise TrainingConflict("connection is closed") from None
        main = next((str(row[2]) for row in rows if str(row[1]) == "main"), "")
        if not main or Path(main).resolve() != self.database.path.resolve():
            raise TrainingConflict("connection belongs to a foreign database")

    def create_dataset(
        self,
        character_id: str,
        draft: TrainingDatasetDraft | FrozenTrainingDataset,
        *,
        lineage_id: str | None = None,
    ) -> DatasetRecord:
        base = self._draft_from(draft)
        if base.character_id != character_id:
            raise TrainingConflict("dataset character identity mismatch")
        dataset_id = training_dataset_id(base)
        storage_id = f"training_dataset_storage_{uuid.uuid4().hex}"
        created_at = utc_now().isoformat()
        tier_weights = {tier.value: weight for tier, weight in base.source_tier_weights.items()}
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                owner = connection.execute(
                    "SELECT reviewer_id FROM characters WHERE id=?", (character_id,)
                ).fetchone()
                if owner is None or str(owner["reviewer_id"]) != self.reviewer_id:
                    raise TrainingConflict("dataset owner or character mismatch")
                pack = connection.execute(
                    "SELECT storage_id FROM character_packs WHERE character_id=? "
                    "AND reviewer_id=? AND state='preregistered' "
                    "AND canonical_json_digest(manifest_json)=?",
                    (character_id, self.reviewer_id, base.character_pack_digest),
                ).fetchone()
                if pack is None:
                    raise TrainingConflict("dataset character pack provenance mismatch")
                connection.execute(
                    "INSERT INTO training_datasets(storage_id,id,reviewer_id,character_id,"
                    "character_pack_storage_id,character_pack_digest,state,canonical_outfit_id,"
                    "reserved_token,source_tier_weights_json,manifest_json,dataset_digest,"
                    "created_at,sealed_at,lineage_id) "
                    "VALUES (?,?,?,?,?,?,'draft',?,?,?,?,NULL,?,NULL,?)",
                    (
                        storage_id,
                        dataset_id,
                        self.reviewer_id,
                        character_id,
                        str(pack["storage_id"]),
                        base.character_pack_digest,
                        base.canonical_outfit_id,
                        base.reserved_token,
                        canonical_json(tier_weights),
                        canonical_json(base.model_dump(mode="json")),
                        created_at,
                        lineage_id,
                    ),
                )
                if lineage_id is not None:
                    connection.execute(
                        "INSERT INTO training_dataset_lineages VALUES (?,?,?,?,1,'draft',?)",
                        (lineage_id, self.reviewer_id, character_id, storage_id, created_at),
                    )
                connection.commit()
        except TrainingConflict:
            raise
        except TrainingNotFound:
            raise TrainingConflict("recipe dataset identity mismatch") from None
        except sqlite3.DatabaseError:
            raise TrainingConflict("dataset identity conflict") from None
        return self.get_dataset(dataset_id)

    def replace_draft_sample(self, dataset_id: str, sample: TrainingSample) -> DatasetRecord:
        current = self.get_dataset(dataset_id)
        if current.state != "draft":
            raise TrainingConflict("sealed dataset is immutable")
        samples = tuple(
            item for item in current.draft.samples if item.sample_id != sample.sample_id
        )
        changed = current.draft.model_copy(update={"samples": samples + (sample,)})
        return self.create_dataset(changed.character_id, changed)

    def seal_dataset(self, dataset_id: str, frozen: FrozenTrainingDataset) -> DatasetRecord:
        current = self.get_dataset(dataset_id)
        expected = self._draft_from(frozen)
        if current.state == "sealed":
            if current.frozen_dataset != frozen:
                raise TrainingConflict("sealed dataset replay mismatch")
            return current
        if current.draft != expected or frozen.dataset_digest != training_dataset_id(expected):
            raise TrainingConflict("frozen dataset identity mismatch")
        now = utc_now().isoformat()
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE training_datasets SET state='sealed',dataset_digest=?,"
                    "manifest_json=?,sealed_at=? WHERE storage_id=?",
                    (
                        frozen.dataset_digest,
                        canonical_json(frozen.model_dump(mode="json")),
                        now,
                        current.storage_id,
                    ),
                )
                connection.commit()
        except sqlite3.DatabaseError:
            raise TrainingConflict("dataset sealing conflict") from None
        return self.get_dataset(frozen.dataset_digest)

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT dataset.* FROM training_datasets dataset "
                "LEFT JOIN training_dataset_aliases alias "
                "ON alias.dataset_storage_id=dataset.storage_id "
                "WHERE (dataset.id=? OR dataset.dataset_digest=? OR alias.alias_id=?) "
                "AND dataset.reviewer_id=? AND dataset.state IN ('draft','sealed') "
                "AND alias.alias_id=dataset.id "
                "AND (SELECT count(*) FROM training_dataset_aliases exact_alias "
                "WHERE exact_alias.dataset_storage_id=dataset.storage_id)=1 "
                "AND ((dataset.state='draft' AND training_draft_identity_is_valid("
                "dataset.storage_id,dataset.id,dataset.manifest_json)=1) OR "
                "(dataset.state='sealed' AND training_dataset_identity_is_valid("
                "dataset.storage_id,dataset.dataset_digest,dataset.manifest_json)=1)) LIMIT 1",
                (dataset_id, dataset_id, dataset_id, self.reviewer_id),
            ).fetchone()
        if row is None:
            raise TrainingNotFound("training dataset not found")
        manifest = json.loads(str(row["manifest_json"]))
        if str(row["state"]) == "sealed":
            frozen = FrozenTrainingDataset.model_validate(manifest)
            draft = self._draft_from(frozen)
        else:
            frozen = None
            draft = TrainingDatasetDraft.model_validate(manifest)
        with self.database.connect() as connection:
            imported_ids = tuple(
                str(item[0])
                for item in connection.execute(
                    "SELECT import_id FROM training_dataset_snapshot_imports "
                    "WHERE dataset_storage_id=? ORDER BY import_id",
                    (row["storage_id"],),
                )
            )
            confirmation_ids = tuple(
                str(item[0])
                for item in connection.execute(
                    "SELECT confirmation_id FROM training_dataset_snapshot_confirmations "
                    "WHERE dataset_storage_id=? ORDER BY confirmation_id",
                    (row["storage_id"],),
                )
            )
        if (
            imported_ids != draft.imported_sample_ids
            or confirmation_ids != draft.curation_confirmation_ids
        ):
            raise TrainingNotFound("training dataset ledger bindings do not match manifest")
        return DatasetRecord(
            storage_id=str(row["storage_id"]),
            dataset_id=str(row["id"]),
            reviewer_id=str(row["reviewer_id"]),
            character_id=str(row["character_id"]),
            state=cast(Literal["draft", "sealed"], str(row["state"])),
            dataset_digest=cast(str | None, row["dataset_digest"]),
            draft=draft,
            frozen_dataset=frozen,
        )

    def create_recipe(
        self,
        recipe: TrainingRecipe,
        binding: RecipeMaterialBinding,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> RecipeRecord:
        if connection is not None:
            self._require_owned_connection(connection)
            return self._create_recipe_with_connection(connection, recipe, binding)
        with self.database.connect() as owned:
            owned.execute("BEGIN IMMEDIATE")
            try:
                result = self._create_recipe_with_connection(owned, recipe, binding)
                owned.commit()
                return result
            except Exception:
                owned.rollback()
                raise

    def _create_recipe_with_connection(
        self,
        connection: sqlite3.Connection,
        recipe: TrainingRecipe,
        binding: RecipeMaterialBinding,
    ) -> RecipeRecord:
        recipe_id = training_recipe_id(recipe)
        try:
            dataset = self.get_dataset(recipe.dataset_digest)
            if dataset.frozen_dataset is None:
                raise TrainingConflict("recipe requires an existing sealed dataset")
            validate_training_recipe_dataset(recipe, dataset.frozen_dataset)
            storage_id = f"training_recipe_storage_{uuid.uuid4().hex}"
            provenance = connection.execute(
                "SELECT benchmark.id FROM training_datasets dataset "
                "JOIN character_benchmark_campaigns benchmark "
                "ON benchmark.character_pack_storage_id=dataset.character_pack_storage_id "
                "AND benchmark.reviewer_id=? JOIN character_benchmark_results result "
                "ON result.campaign_id=benchmark.id WHERE dataset.storage_id=? "
                "AND character_benchmark_provenance_is_valid(benchmark.id)=1 "
                "AND result.content_digest=? "
                "AND json_extract(benchmark.definition_json,'$.character_pack_digest')=? "
                "AND json_extract(benchmark.definition_json,'$.stage_2_manifest_digest')=? "
                "AND json_extract(benchmark.definition_json,'$.stage_2_evidence_digest')=? "
                "AND (SELECT new_state FROM character_benchmark_events "
                "WHERE campaign_id=benchmark.id ORDER BY sequence DESC LIMIT 1)='completed'",
                (
                    self.reviewer_id,
                    dataset.storage_id,
                    recipe.stage_3_evidence_digest,
                    recipe.character_pack_digest,
                    recipe.stage_2_manifest_digest,
                    recipe.stage_2_evidence_digest,
                ),
            ).fetchone()
            if provenance is None or str(provenance["id"]) != binding.stage_3_campaign_id:
                raise TrainingConflict("recipe prerequisite provenance mismatch")
            result = connection.execute(
                "SELECT content_digest FROM character_benchmark_results WHERE campaign_id=?",
                (binding.stage_3_campaign_id,),
            ).fetchone()
            if result is None or str(result["content_digest"]) != binding.stage_3_result_id:
                raise TrainingConflict("recipe prerequisite provenance mismatch")
            self._require_recipe_artifact(
                connection, binding.base_model_artifact_id, recipe.base_model_digest
            )
            self._require_recipe_artifact(
                connection, binding.source_snapshot_artifact_id, recipe.source_code_digest
            )
            if set(binding.lightning_adapter_artifact_ids) != set(recipe.lightning_adapter_digests):
                raise TrainingConflict("recipe material provenance mismatch")
            for name, artifact_id in binding.lightning_adapter_artifact_ids.items():
                self._require_recipe_artifact(
                    connection, artifact_id, recipe.lightning_adapter_digests[name]
                )
            connection.execute(
                "INSERT INTO training_recipes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    storage_id,
                    recipe_id,
                    self.reviewer_id,
                    dataset.storage_id,
                    recipe.dataset_digest,
                    dataset.character_id,
                    recipe.character_pack_digest,
                    "draft",
                    canonical_json(recipe.model_dump(mode="json")),
                    utc_now().isoformat(),
                    None,
                ),
            )
            connection.execute(
                "INSERT INTO training_recipe_materials VALUES (?,?,?,?,?,?,?,?)",
                (
                    storage_id,
                    self.reviewer_id,
                    binding.base_model_artifact_id,
                    canonical_json(dict(binding.lightning_adapter_artifact_ids)),
                    binding.source_snapshot_artifact_id,
                    binding.stage_3_campaign_id,
                    binding.stage_3_result_id,
                    binding.runtime_digest,
                ),
            )
            connection.executemany(
                "INSERT INTO training_recipe_lightning_materials VALUES (?,?,?,?)",
                (
                    (storage_id, self.reviewer_id, name, artifact_id)
                    for name, artifact_id in sorted(binding.lightning_adapter_artifact_ids.items())
                ),
            )
        except TrainingConflict:
            raise
        except TrainingNotFound:
            raise TrainingConflict("recipe dataset identity mismatch") from None
        except sqlite3.DatabaseError:
            raise TrainingConflict("recipe dataset identity conflict") from None
        except (TypeError, ValueError):
            raise TrainingConflict("recipe dataset identity mismatch") from None
        return self._get_recipe_with_connection(connection, recipe_id)

    @staticmethod
    def _require_recipe_artifact(
        connection: sqlite3.Connection, artifact_id: str, digest: str
    ) -> None:
        artifact = connection.execute(
            "SELECT sha256 FROM artifacts WHERE id=?", (artifact_id,)
        ).fetchone()
        if artifact is None or "sha256:" + str(artifact["sha256"]) != digest:
            raise TrainingConflict("recipe material provenance mismatch")

    def preregister_recipe(
        self, recipe_id: str, *, connection: sqlite3.Connection | None = None
    ) -> RecipeRecord:
        if connection is not None:
            self._require_owned_connection(connection)
            return self._preregister_recipe_with_connection(connection, recipe_id)
        with self.database.connect() as owned:
            owned.execute("BEGIN IMMEDIATE")
            try:
                result = self._preregister_recipe_with_connection(owned, recipe_id)
                owned.commit()
                return result
            except Exception:
                owned.rollback()
                raise

    def _preregister_recipe_with_connection(
        self, connection: sqlite3.Connection, recipe_id: str
    ) -> RecipeRecord:
        current = self._get_recipe_with_connection(connection, recipe_id)
        if current.state == "preregistered":
            return current
        try:
            connection.execute(
                "UPDATE training_recipes SET state='preregistered',preregistered_at=? "
                "WHERE storage_id=?",
                (utc_now().isoformat(), current.storage_id),
            )
        except sqlite3.DatabaseError:
            raise TrainingConflict("recipe preregistration conflict") from None
        return self._get_recipe_with_connection(connection, recipe_id)

    def get_recipe(self, recipe_id: str) -> RecipeRecord:
        with self.database.connect() as connection:
            return self._get_recipe_with_connection(connection, recipe_id)

    def _get_recipe_with_connection(
        self, connection: sqlite3.Connection, recipe_id: str
    ) -> RecipeRecord:
        row = connection.execute(
            "SELECT recipe.*,materials.base_model_artifact_id,"
            "materials.lightning_adapter_artifact_ids_json,"
            "materials.source_snapshot_artifact_id,materials.stage_3_campaign_id,"
            "materials.stage_3_result_id,materials.runtime_digest "
            "FROM training_recipes recipe JOIN training_recipe_materials materials "
            "ON materials.recipe_storage_id=recipe.storage_id "
            "WHERE (recipe.id=? OR recipe.storage_id=?) AND recipe.reviewer_id=?",
            (recipe_id, recipe_id, self.reviewer_id),
        ).fetchone()
        if row is None:
            raise TrainingNotFound("training recipe not found")
        recipe = TrainingRecipe.model_validate_json(str(row["recipe_json"]))
        binding = RecipeMaterialBinding(
            base_model_artifact_id=str(row["base_model_artifact_id"]),
            lightning_adapter_artifact_ids=json.loads(
                str(row["lightning_adapter_artifact_ids_json"])
            ),
            source_snapshot_artifact_id=str(row["source_snapshot_artifact_id"]),
            stage_3_campaign_id=str(row["stage_3_campaign_id"]),
            stage_3_result_id=str(row["stage_3_result_id"]),
            runtime_digest=str(row["runtime_digest"]),
        )
        child_bindings = {
            str(item["material_name"]): str(item["artifact_id"])
            for item in connection.execute(
                "SELECT material_name,artifact_id FROM training_recipe_lightning_materials "
                "WHERE recipe_storage_id=? ORDER BY material_name",
                (row["storage_id"],),
            )
        }
        if child_bindings != dict(binding.lightning_adapter_artifact_ids):
            raise TrainingConflict("recipe material binding mismatch")
        return RecipeRecord(
            storage_id=str(row["storage_id"]),
            recipe_id=str(row["id"]),
            reviewer_id=str(row["reviewer_id"]),
            state=cast(Literal["draft", "preregistered"], str(row["state"])),
            recipe=recipe,
            binding=binding,
        )

    def list_recipes(self) -> tuple[RecipeRecord, ...]:
        with self.database.connect() as connection:
            identifiers = tuple(
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM training_recipes WHERE reviewer_id=? ORDER BY created_at,id",
                    (self.reviewer_id,),
                )
            )
        return tuple(self.get_recipe(identifier) for identifier in identifiers)

    def list_datasets(self) -> tuple[DatasetRecord, ...]:
        with self.database.connect() as connection:
            identifiers = tuple(
                str(row["id"])
                for row in connection.execute(
                    "SELECT dataset.id FROM training_datasets dataset "
                    "WHERE dataset.reviewer_id=? AND dataset.state IN ('draft','sealed') "
                    "AND EXISTS(SELECT 1 FROM training_dataset_aliases alias "
                    "WHERE alias.dataset_storage_id=dataset.storage_id "
                    "AND alias.alias_id=dataset.id) "
                    "AND (SELECT count(*) FROM training_dataset_aliases alias "
                    "WHERE alias.dataset_storage_id=dataset.storage_id)=1 "
                    "ORDER BY dataset.created_at,dataset.id",
                    (self.reviewer_id,),
                )
            )
        return tuple(self.get_dataset(identifier) for identifier in identifiers)

    def get_run(self, run_id: str) -> RunRecord:
        with self.database.connect() as connection:
            return self.get_run_with_connection(connection, run_id)

    def get_run_with_connection(self, connection: sqlite3.Connection, run_id: str) -> RunRecord:
        row = connection.execute(
            "SELECT run.*,recipe.id AS recipe_id,recipe.dataset_digest "
            "FROM training_runs run JOIN training_recipes recipe "
            "ON recipe.storage_id=run.recipe_storage_id "
            "WHERE run.id=? AND run.reviewer_id=?",
            (run_id, self.reviewer_id),
        ).fetchone()
        if row is None:
            raise TrainingNotFound("training run not found")
        return RunRecord(
            run=TrainingRun(
                run_id=str(row["id"]),
                recipe_id=str(row["recipe_id"]),
                dataset_digest=str(row["dataset_digest"]),
                total_steps=int(row["total_steps"]),
            ),
            reviewer_id=str(row["reviewer_id"]),
            state=cast(
                Literal[
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
                ],
                str(row["lifecycle_state"]),
            ),
            created_at=str(row["created_at"]),
            terminal_at=cast(str | None, row["terminal_at"]),
        )

    def list_runs(self) -> tuple[RunRecord, ...]:
        with self.database.connect() as connection:
            ids = tuple(
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM training_runs WHERE reviewer_id=? ORDER BY created_at,id",
                    (self.reviewer_id,),
                )
            )
        return tuple(self.get_run(run_id) for run_id in ids)

    def reopen_operation_with_connection(
        self,
        connection: sqlite3.Connection,
        operation_kind: TrainingOperationKind,
        idempotency_key: str,
    ) -> TrainingOrchestrationOperation:
        self._require_owned_connection(connection)
        row = connection.execute(
            "SELECT operation.*,recipe.id recipe_id "
            "FROM training_orchestration_operations operation "
            "JOIN training_recipes recipe ON recipe.storage_id=operation.recipe_storage_id "
            "WHERE operation.reviewer_id=? AND operation.operation_kind=? "
            "AND operation.idempotency_key=?",
            (self.reviewer_id, operation_kind.value, idempotency_key),
        ).fetchone()
        if row is None:
            raise TrainingNotFound("training orchestration operation not found")
        return self._operation_from_row(row)

    def get_operation_with_connection(
        self, connection: sqlite3.Connection, operation_id: str
    ) -> TrainingOrchestrationOperation:
        self._require_owned_connection(connection)
        row = connection.execute(
            "SELECT operation.*,recipe.id recipe_id "
            "FROM training_orchestration_operations operation "
            "JOIN training_recipes recipe ON recipe.storage_id=operation.recipe_storage_id "
            "WHERE operation.id=? AND operation.reviewer_id=?",
            (operation_id, self.reviewer_id),
        ).fetchone()
        if row is None:
            raise TrainingNotFound("training orchestration operation not found")
        return self._operation_from_row(row)

    def begin_operation_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        operation_id: str,
        run_id: str,
        operation_kind: TrainingOperationKind,
        idempotency_key: str,
        request_payload: Mapping[str, object],
    ) -> TrainingOrchestrationOperation:
        """Create or exactly reopen an immutable operation request."""
        self._require_owned_connection(connection)
        if request_payload.get("run_id") != run_id:
            raise TrainingConflict("training operation request run mismatch")
        request_json = canonical_json(dict(request_payload))
        request_digest = training_operation_payload_digest(request_payload)
        existing = connection.execute(
            "SELECT request_digest FROM training_orchestration_operations WHERE reviewer_id=? "
            "AND operation_kind=? AND idempotency_key=?",
            (self.reviewer_id, operation_kind.value, idempotency_key),
        ).fetchone()
        if existing is not None:
            if str(existing["request_digest"]) != request_digest:
                raise TrainingConflict("training operation request digest conflict")
            return self.reopen_operation_with_connection(
                connection, operation_kind, idempotency_key
            )
        run = connection.execute(
            "SELECT recipe_storage_id,lifecycle_version,lifecycle_state FROM training_runs "
            "WHERE id=? AND reviewer_id=?",
            (run_id, self.reviewer_id),
        ).fetchone()
        if run is None:
            raise TrainingConflict("training operation run mismatch")
        now = utc_now().isoformat()
        try:
            connection.execute(
                "INSERT INTO training_orchestration_operations "
                "(id,reviewer_id,run_id,recipe_storage_id,operation_kind,idempotency_key,"
                "request_digest,request_payload_json,lifecycle_version,lifecycle_state,status,"
                "attempt,claimant_id,lease_expires_at,receipt_digest,receipt_payload_json,"
                "failure_digest,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,'pending',0,"
                "NULL,NULL,NULL,NULL,NULL,?,?)",
                (
                    operation_id,
                    self.reviewer_id,
                    run_id,
                    run["recipe_storage_id"],
                    operation_kind.value,
                    idempotency_key,
                    request_digest,
                    request_json,
                    run["lifecycle_version"],
                    run["lifecycle_state"],
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            # A concurrent exact creator is reopened; an ID/digest collision fails closed.
            existing = connection.execute(
                "SELECT request_digest FROM training_orchestration_operations WHERE reviewer_id=? "
                "AND operation_kind=? AND idempotency_key=?",
                (self.reviewer_id, operation_kind.value, idempotency_key),
            ).fetchone()
            if existing is None or str(existing["request_digest"]) != request_digest:
                raise TrainingConflict("training operation identity conflict") from None
        return self.reopen_operation_with_connection(connection, operation_kind, idempotency_key)

    def claim_operation_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        operation_kind: TrainingOperationKind,
        idempotency_key: str,
        claimant_id: str,
        lease_seconds: int,
    ) -> TrainingOperationClaim:
        self._require_owned_connection(connection)
        if not claimant_id or lease_seconds <= 0:
            raise ValueError("claimant and positive lease are required")
        lease = (utc_now() + timedelta(seconds=lease_seconds)).isoformat()
        claim_token = uuid.uuid4().hex
        cursor = connection.execute(
            "UPDATE training_orchestration_operations SET status='running',attempt=attempt+1,"
            "claimant_id=?,claim_token=?,lease_expires_at=?,updated_at=? WHERE reviewer_id=? "
            "AND operation_kind=? AND idempotency_key=? AND (status='pending' OR "
            "(status='running' AND julianday(lease_expires_at)<=julianday('now')))",
            (
                claimant_id,
                claim_token,
                lease,
                utc_now().isoformat(),
                self.reviewer_id,
                operation_kind.value,
                idempotency_key,
            ),
        )
        operation = self.reopen_operation_with_connection(
            connection, operation_kind, idempotency_key
        )
        return TrainingOperationClaim(operation=operation, claimed=cursor.rowcount == 1)

    def complete_operation_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        operation_kind: TrainingOperationKind,
        idempotency_key: str,
        claimant_id: str,
        claim_token: str,
        attempt: int,
        receipt_payload: Mapping[str, object],
        result_payload: Mapping[str, object] | None = None,
    ) -> TrainingOrchestrationOperation:
        self._require_owned_connection(connection)
        self._validate_operation_receipt_evidence(connection, operation_kind, receipt_payload)
        receipt_json = canonical_json(dict(receipt_payload))
        receipt_digest = training_operation_payload_digest(receipt_payload)
        durable_result = receipt_payload if result_payload is None else result_payload
        result_json = canonical_json(dict(durable_result))
        result_digest = training_operation_payload_digest(durable_result)
        cursor = connection.execute(
            "UPDATE training_orchestration_operations SET status='completed',claimant_id=NULL,"
            "claim_token=NULL,lease_expires_at=NULL,receipt_digest=?,receipt_payload_json=?,"
            "result_digest=?,result_payload_json=?,updated_at=? "
            "WHERE reviewer_id=? AND operation_kind=? AND idempotency_key=? "
            "AND status='running' AND claimant_id=? AND claim_token=? AND attempt=?",
            (
                receipt_digest,
                receipt_json,
                result_digest,
                result_json,
                utc_now().isoformat(),
                self.reviewer_id,
                operation_kind.value,
                idempotency_key,
                claimant_id,
                claim_token,
                attempt,
            ),
        )
        if cursor.rowcount != 1:
            raise TrainingConflict("training operation completion claim mismatch")
        return self.reopen_operation_with_connection(connection, operation_kind, idempotency_key)

    def link_operation_evidence_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        operation_id: str,
        attempt: int,
        claim_token: str,
        evidence_kind: str,
        evidence_id: str,
        evidence_digest: str,
    ) -> None:
        """Fence already-published evidence to the exact live operation attempt."""
        self._require_owned_connection(connection)
        try:
            if evidence_kind == "calibration":
                operation = connection.execute(
                    "SELECT operation_kind,request_digest,request_payload_json "
                    "FROM training_orchestration_operations WHERE id=?",
                    (operation_id,),
                ).fetchone()
                if operation is None or str(operation["operation_kind"]) != "calibration":
                    raise sqlite3.IntegrityError("calibration operation mismatch")
                request = json.loads(str(operation["request_payload_json"]))
                connection.execute(
                    "INSERT INTO training_calibration_operation_bindings "
                    "(evidence_id,operation_id,request_digest,attempt,claim_token,steps,"
                    "trainable_targets_json,seed,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        evidence_id,
                        operation_id,
                        operation["request_digest"],
                        attempt,
                        claim_token,
                        request["steps"],
                        canonical_json(request["trainable_targets"]),
                        request["seed"],
                        utc_now().isoformat(),
                    ),
                )
            connection.execute(
                "INSERT INTO training_operation_evidence_links VALUES (?,?,?,?,?,?,?)",
                (
                    operation_id,
                    attempt,
                    claim_token,
                    evidence_kind,
                    evidence_id,
                    evidence_digest,
                    utc_now().isoformat(),
                ),
            )
        except sqlite3.DatabaseError:
            raise TrainingConflict("training operation evidence fence conflict") from None

    def fail_operation_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        operation_kind: TrainingOperationKind,
        idempotency_key: str,
        claimant_id: str,
        claim_token: str,
        attempt: int,
        failure_digest: str,
    ) -> TrainingOrchestrationOperation:
        self._require_owned_connection(connection)
        cursor = connection.execute(
            "UPDATE training_orchestration_operations SET status='failed',claimant_id=NULL,"
            "claim_token=NULL,lease_expires_at=NULL,failure_digest=?,updated_at=? "
            "WHERE reviewer_id=? "
            "AND operation_kind=? AND idempotency_key=? AND status='running' AND claimant_id=? "
            "AND claim_token=? AND attempt=?",
            (
                failure_digest,
                utc_now().isoformat(),
                self.reviewer_id,
                operation_kind.value,
                idempotency_key,
                claimant_id,
                claim_token,
                attempt,
            ),
        )
        if cursor.rowcount != 1:
            raise TrainingConflict("training operation failure claim mismatch")
        return self.reopen_operation_with_connection(connection, operation_kind, idempotency_key)

    def _operation_from_row(self, row: sqlite3.Row) -> TrainingOrchestrationOperation:
        return TrainingOrchestrationOperation(
            operation_id=str(row["id"]),
            run_id=str(row["run_id"]),
            recipe_id=str(row["recipe_id"]),
            operation_kind=TrainingOperationKind(str(row["operation_kind"])),
            idempotency_key=str(row["idempotency_key"]),
            request_digest=str(row["request_digest"]),
            request_payload=json.loads(str(row["request_payload_json"])),
            lifecycle_version=int(row["lifecycle_version"]),
            lifecycle_state=cast(Any, str(row["lifecycle_state"])),
            status=TrainingOperationStatus(str(row["status"])),
            attempt=int(row["attempt"]),
            claimant_id=cast(str | None, row["claimant_id"]),
            claim_token=cast(str | None, row["claim_token"]),
            lease_expires_at=(
                datetime.fromisoformat(str(row["lease_expires_at"]))
                if row["lease_expires_at"] is not None
                else None
            ),
            receipt_digest=cast(str | None, row["receipt_digest"]),
            receipt_payload=(
                json.loads(str(row["receipt_payload_json"]))
                if row["receipt_payload_json"] is not None
                else None
            ),
            result_digest=cast(str | None, row["result_digest"]),
            result_payload=(
                json.loads(str(row["result_payload_json"]))
                if row["result_payload_json"] is not None
                else None
            ),
            failure_digest=cast(str | None, row["failure_digest"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    def _validate_operation_receipt_evidence(
        self,
        connection: sqlite3.Connection,
        kind: TrainingOperationKind,
        payload: Mapping[str, object],
    ) -> None:
        run_id = payload.get("run_id")
        version = payload.get("lifecycle_version")
        if not isinstance(run_id, str) or not isinstance(version, int):
            raise TrainingConflict("training operation receipt is incomplete")
        run = connection.execute(
            "SELECT lifecycle_version FROM training_runs WHERE id=? AND reviewer_id=?",
            (run_id, self.reviewer_id),
        ).fetchone()
        if run is None or int(run[0]) != version:
            raise TrainingConflict("training operation receipt lifecycle mismatch")
        resume = payload.get("resume")
        cancellation = payload.get("cancellation")
        table_and_identity = {
            TrainingOperationKind.PROBE: ("training_probe_evidence", payload.get("evidence_id")),
            TrainingOperationKind.CALIBRATION: (
                "training_calibration_evidence",
                payload.get("evidence_id"),
            ),
            TrainingOperationKind.RESUME: (
                "training_resume_evidence",
                resume.get("evidence_id") if isinstance(resume, Mapping) else None,
            ),
            TrainingOperationKind.CANCEL: (
                "training_cancellation_evidence",
                cancellation.get("evidence_id") if isinstance(cancellation, Mapping) else None,
            ),
        }
        if kind in table_and_identity:
            table, identity = table_and_identity[kind]
            if (
                not isinstance(identity, str)
                or connection.execute(
                    f"SELECT 1 FROM {table} "  # noqa: S608
                    "WHERE id=? AND reviewer_id=? AND run_id=?",
                    (identity, self.reviewer_id, run_id),
                ).fetchone()
                is None
            ):
                raise TrainingConflict("training operation receipt evidence is incomplete")

    def list_events(self, run_id: str) -> tuple[EventRecord, ...]:
        self.get_run(run_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_run_events WHERE run_id=? AND reviewer_id=? "
                "ORDER BY sequence",
                (run_id, self.reviewer_id),
            ).fetchall()
        return tuple(
            EventRecord(
                run_id=run_id,
                reviewer_id=str(row["reviewer_id"]),
                event=TrainingRunEvent(
                    sequence=int(row["sequence"]),
                    previous_state=cast(Any, str(row["previous_state"])),
                    new_state=cast(Any, str(row["new_state"])),
                    payload=json.loads(str(row["payload_json"])),
                ),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    def list_telemetry(self, run_id: str) -> tuple[TelemetryRecord, ...]:
        self.get_run(run_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_telemetry WHERE run_id=? AND reviewer_id=? "
                "ORDER BY sequence",
                (run_id, self.reviewer_id),
            ).fetchall()
        return tuple(
            TelemetryRecord(
                run_id=run_id,
                reviewer_id=str(row["reviewer_id"]),
                sample=TrainingTelemetry(
                    sequence=int(row["sequence"]),
                    step=int(row["step"]),
                    payload=json.loads(str(row["payload_json"])),
                ),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    def list_checkpoints(self, run_id: str) -> tuple[CheckpointRecord, ...]:
        self.get_run(run_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_checkpoints WHERE run_id=? AND reviewer_id=? "
                "AND publication_state='finalized' ORDER BY step",
                (run_id, self.reviewer_id),
            ).fetchall()
        return tuple(
            CheckpointRecord(
                reviewer_id=str(row["reviewer_id"]),
                checkpoint=TrainingCheckpoint(
                    checkpoint_id=str(row["id"]),
                    run_id=run_id,
                    step=int(row["step"]),
                    fraction=float(row["fraction"]),
                    parent_checkpoint_id=cast(str | None, row["parent_checkpoint_id"]),
                    artifact_digest=str(row["artifact_digest"]),
                    manifest_digest=str(row["manifest_digest"]),
                    manifest=json.loads(str(row["manifest_json"])),
                ),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    def list_failures(self, run_id: str) -> tuple[FailureRecord, ...]:
        self.get_run(run_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_failures WHERE run_id=? AND reviewer_id=? "
                "ORDER BY created_at,id",
                (run_id, self.reviewer_id),
            ).fetchall()
        return tuple(
            FailureRecord(
                reviewer_id=str(row["reviewer_id"]),
                failure=TrainingFailure(
                    failure_id=str(row["id"]),
                    run_id=run_id,
                    code=str(row["code"]),
                    message=str(row["message"]),
                ),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    def create_run(self, run: TrainingRun) -> RunRecord:
        recipe = self.get_recipe(run.recipe_id)
        if recipe.state != "preregistered" or recipe.recipe.dataset_digest != run.dataset_digest:
            raise TrainingConflict("training run recipe or dataset identity mismatch")
        dataset = self.get_dataset(run.dataset_digest)
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self.create_run_with_connection(
                    connection, run, recipe.storage_id, dataset.storage_id
                )
                connection.commit()
        except sqlite3.DatabaseError:
            raise TrainingConflict("training run identity conflict") from None
        return self.get_run(run.run_id)

    def create_run_with_connection(
        self,
        connection: sqlite3.Connection,
        run: TrainingRun,
        recipe_storage_id: str,
        dataset_storage_id: str,
    ) -> None:
        """Insert a draft run without owning the caller's transaction."""
        connection.execute(
            "INSERT INTO training_runs(id,reviewer_id,recipe_storage_id,dataset_storage_id,"
            "state,total_steps,created_at,terminal_at) VALUES (?,?,?,?,?,?,?,NULL)",
            (
                run.run_id,
                self.reviewer_id,
                recipe_storage_id,
                dataset_storage_id,
                "created",
                run.total_steps,
                utc_now().isoformat(),
            ),
        )

    def append_event(
        self, run_id: str, event: TrainingRunEvent, *, expected_version: int | None = None
    ) -> None:
        self._atomic_evidence(
            lambda connection: self.append_event_with_connection(
                connection, run_id, event, expected_version=expected_version
            ),
            "training event conflict",
        )

    def append_event_with_connection(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        event: TrainingRunEvent,
        *,
        expected_version: int | None = None,
    ) -> None:
        if expected_version is not None:
            current = connection.execute(
                "SELECT lifecycle_version FROM training_runs WHERE id=? AND reviewer_id=?",
                (run_id, self.reviewer_id),
            ).fetchone()
            if current is None or int(current[0]) != expected_version:
                raise TrainingConflict("training run lifecycle compare-and-swap conflict")
        connection.execute(
            "INSERT INTO training_run_events VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                self.reviewer_id,
                event.sequence,
                event.previous_state,
                event.new_state,
                canonical_json(dict(event.payload)),
                utc_now().isoformat(),
            ),
        )

    def record_probe_with_connection(
        self, connection: sqlite3.Connection, evidence: TrainingProbeEvidence
    ) -> None:
        connection.execute(
            "INSERT INTO training_probe_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                evidence.evidence_id,
                evidence.run_id,
                self.reviewer_id,
                evidence.recipe_digest,
                evidence.dataset_digest,
                evidence.backend_digest,
                evidence.runtime_digest,
                evidence.material_digest,
                evidence.source_digest,
                evidence.capability_digest,
                int(evidence.succeeded),
                utc_now().isoformat(),
            ),
        )

    def record_probe(self, evidence: TrainingProbeEvidence) -> None:
        self._atomic_evidence(
            lambda connection: self.record_probe_with_connection(connection, evidence),
            "training probe evidence conflict",
        )

    def record_calibration_with_connection(
        self, connection: sqlite3.Connection, evidence: TrainingCalibrationEvidence
    ) -> None:
        connection.execute(
            "INSERT INTO training_calibration_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                evidence.evidence_id,
                evidence.run_id,
                self.reviewer_id,
                evidence.probe_evidence_id,
                evidence.calibration_digest,
                evidence.recipe_digest,
                evidence.dataset_digest,
                evidence.backend_digest,
                evidence.runtime_digest,
                evidence.material_digest,
                evidence.source_digest,
                int(evidence.succeeded),
                utc_now().isoformat(),
            ),
        )

    def record_calibration(self, evidence: TrainingCalibrationEvidence) -> None:
        self._atomic_evidence(
            lambda connection: self.record_calibration_with_connection(connection, evidence),
            "training calibration evidence conflict",
        )

    def record_cancellation_with_connection(
        self, connection: sqlite3.Connection, evidence: TrainingCancellationEvidence
    ) -> None:
        connection.execute(
            "INSERT INTO training_cancellation_evidence "
            "(id,run_id,reviewer_id,requested_step,boundary_step,checkpoint_digest,created_at,"
            "cancellation_request_id,request_digest) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                evidence.cancellation_id,
                evidence.run_id,
                self.reviewer_id,
                evidence.requested_step,
                evidence.boundary_step,
                evidence.checkpoint_digest,
                utc_now().isoformat(),
                evidence.cancellation_request_id,
                evidence.request_digest,
            ),
        )

    def record_cancellation(self, evidence: TrainingCancellationEvidence) -> None:
        self._atomic_evidence(
            lambda connection: self.record_cancellation_with_connection(connection, evidence),
            "training cancellation evidence conflict",
        )

    def atomic(self, callback: Callable[[sqlite3.Connection], object]) -> object:
        """Run a complete evidence publication under one caller-visible transaction."""
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = callback(connection)
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def _atomic_evidence(
        self, callback: Callable[[sqlite3.Connection], object], message: str
    ) -> None:
        try:
            self.atomic(callback)
        except sqlite3.DatabaseError:
            raise TrainingConflict(message) from None

    def append_telemetry(self, run_id: str, sample: TrainingTelemetry) -> None:
        self._atomic_evidence(
            lambda connection: self.append_telemetry_with_connection(connection, run_id, sample),
            "training telemetry conflict",
        )

    def append_telemetry_with_connection(
        self, connection: sqlite3.Connection, run_id: str, sample: TrainingTelemetry
    ) -> None:
        connection.execute(
            "INSERT INTO training_telemetry VALUES (?,?,?,?,?,?)",
            (
                run_id,
                self.reviewer_id,
                sample.sequence,
                sample.step,
                canonical_json(dict(sample.payload)),
                utc_now().isoformat(),
            ),
        )

    def telemetry_digest_with_connection(
        self, connection: sqlite3.Connection, run_id: str, start: int, end: int
    ) -> str:
        rows = connection.execute(
            "SELECT sequence,step,payload_json FROM training_telemetry "
            "WHERE run_id=? AND reviewer_id=? AND sequence BETWEEN ? AND ? ORDER BY sequence",
            (run_id, self.reviewer_id, start, end),
        ).fetchall()
        if len(rows) != end - start + 1 or start < 1 or end < start:
            raise TrainingConflict("training telemetry range is incomplete")
        payload = [
            {
                "sequence": int(row["sequence"]),
                "step": int(row["step"]),
                "payload": json.loads(str(row["payload_json"])),
            }
            for row in rows
        ]
        serialized = canonical_json(cast(Any, payload))
        return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"

    def add_checkpoint(self, checkpoint: TrainingCheckpoint) -> None:
        raise TrainingConflict("scientific checkpoint evidence is required")

    def add_checkpoint_with_connection(
        self,
        connection: sqlite3.Connection,
        checkpoint: TrainingCheckpoint,
        evidence: TrainingCheckpointEvidence,
    ) -> None:
        row = connection.execute(
            "SELECT recipe_storage_id,dataset_storage_id FROM training_runs "
            "WHERE id=? AND reviewer_id=?",
            (checkpoint.run_id, self.reviewer_id),
        ).fetchone()
        if row is None or evidence.run_id != checkpoint.run_id or evidence.step != checkpoint.step:
            raise TrainingConflict("training checkpoint run mismatch")
        evidence_json = canonical_json(evidence.model_dump(mode="json"))
        evidence_digest = f"sha256:{hashlib.sha256(evidence_json.encode()).hexdigest()}"
        connection.execute(
            "INSERT INTO training_checkpoints(id,run_id,reviewer_id,recipe_storage_id,"
            "dataset_storage_id,step,fraction,parent_checkpoint_id,artifact_digest,"
            "manifest_digest,manifest_json,created_at,"
            "evidence_json,evidence_digest,publication_state) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'finalized')",
            (
                checkpoint.checkpoint_id,
                checkpoint.run_id,
                self.reviewer_id,
                row["recipe_storage_id"],
                row["dataset_storage_id"],
                checkpoint.step,
                checkpoint.fraction,
                checkpoint.parent_checkpoint_id,
                checkpoint.artifact_digest,
                checkpoint.manifest_digest,
                canonical_json(checkpoint.model_dump(mode="json")["manifest"]),
                utc_now().isoformat(),
                evidence_json,
                evidence_digest,
            ),
        )

    def request_cancellation_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        request_id: str,
        run_id: str,
        requested_step: int,
        request_digest: str,
    ) -> None:
        connection.execute(
            "INSERT INTO training_cancellation_requests VALUES (?,?,?,?,?,?)",
            (
                request_id,
                run_id,
                self.reviewer_id,
                requested_step,
                request_digest,
                utc_now().isoformat(),
            ),
        )

    def bind_checkpoint_inventory_artifacts_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        checkpoint_id: str,
        artifact_ids_by_role: Mapping[str, str],
    ) -> None:
        """Bind each checkpoint role to one exact committed artifact identity."""
        self._require_owned_connection(connection)
        rows = connection.execute(
            "SELECT inventory.ordinal,inventory.role,inventory.artifact_digest,"
            "inventory.size_bytes FROM training_checkpoint_inventory inventory "
            "JOIN training_checkpoints checkpoint ON checkpoint.id=inventory.checkpoint_id "
            "WHERE inventory.checkpoint_id=? AND checkpoint.reviewer_id=? "
            "ORDER BY inventory.ordinal",
            (checkpoint_id, self.reviewer_id),
        ).fetchall()
        roles = {str(row["role"]) for row in rows}
        if len(rows) != 7 or set(artifact_ids_by_role) != roles:
            raise TrainingConflict("training checkpoint artifact role inventory mismatch")
        try:
            for row in rows:
                role = str(row["role"])
                connection.execute(
                    "INSERT INTO training_checkpoint_inventory_artifacts VALUES (?,?,?,?,?,?)",
                    (
                        checkpoint_id,
                        row["ordinal"],
                        artifact_ids_by_role[role],
                        role,
                        row["artifact_digest"],
                        row["size_bytes"],
                    ),
                )
        except sqlite3.DatabaseError:
            raise TrainingConflict("training checkpoint artifact binding conflict") from None

    def bind_idempotency_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        idempotency_key: str,
        request_digest: str,
        probe_evidence_id: str,
        calibration_evidence_id: str,
    ) -> str:
        row = connection.execute(
            "SELECT binding.run_id,binding.request_digest FROM training_run_idempotency binding "
            "JOIN training_runs run ON run.id=binding.run_id "
            "WHERE binding.reviewer_id=? AND binding.recipe_storage_id=run.recipe_storage_id "
            "AND binding.idempotency_key=? AND run.recipe_storage_id="
            "(SELECT recipe_storage_id FROM training_runs WHERE id=? AND reviewer_id=?)",
            (self.reviewer_id, idempotency_key, run_id, self.reviewer_id),
        ).fetchone()
        if row is not None:
            if str(row["request_digest"]) != request_digest:
                raise TrainingConflict("training idempotency request digest conflict")
            return str(row["run_id"])
        recipe = connection.execute(
            "SELECT recipe_storage_id FROM training_runs WHERE id=? AND reviewer_id=?",
            (run_id, self.reviewer_id),
        ).fetchone()
        if recipe is None:
            raise TrainingConflict("training idempotency run mismatch")
        connection.execute(
            "INSERT INTO training_run_idempotency VALUES (?,?,?,?,?,?,?,?)",
            (
                self.reviewer_id,
                recipe["recipe_storage_id"],
                idempotency_key,
                request_digest,
                run_id,
                probe_evidence_id,
                calibration_evidence_id,
                utc_now().isoformat(),
            ),
        )
        return run_id

    def record_resume_with_connection(
        self, connection: sqlite3.Connection, evidence: TrainingResumeEvidence
    ) -> None:
        connection.execute(
            "INSERT INTO training_resume_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                evidence.resume_id,
                evidence.run_id,
                self.reviewer_id,
                evidence.parent_checkpoint_id,
                evidence.parent_checkpoint_digest,
                evidence.parent_artifact_digest,
                evidence.recipe_digest,
                evidence.dataset_digest,
                evidence.backend_digest,
                evidence.runtime_digest,
                evidence.material_digest,
                evidence.source_digest,
                evidence.optimizer_state_digest,
                utc_now().isoformat(),
            ),
        )

    def record_resume(self, evidence: TrainingResumeEvidence) -> None:
        self._atomic_evidence(
            lambda connection: self.record_resume_with_connection(connection, evidence),
            "training resume evidence conflict",
        )

    def record_evaluation_with_connection(
        self, connection: sqlite3.Connection, evidence: TrainingEvaluationEvidence
    ) -> None:
        connection.execute(
            "INSERT INTO training_evaluation_evidence VALUES (?,?,?,?,?,?)",
            (
                evidence.evidence_id,
                evidence.run_id,
                self.reviewer_id,
                evidence.disposition,
                evidence.evidence_digest,
                utc_now().isoformat(),
            ),
        )

    def record_evaluation(self, evidence: TrainingEvaluationEvidence) -> None:
        self._atomic_evidence(
            lambda connection: self.record_evaluation_with_connection(connection, evidence),
            "training evaluation evidence conflict",
        )

    def record_failure_with_connection(
        self, connection: sqlite3.Connection, failure: TrainingFailure
    ) -> None:
        if _contains_local_path(failure.message):
            raise TrainingConflict("training failure message contains local path material")
        connection.execute(
            "INSERT INTO training_failures VALUES (?,?,?,?,?,?)",
            (
                failure.failure_id,
                failure.run_id,
                self.reviewer_id,
                failure.code,
                failure.message,
                utc_now().isoformat(),
            ),
        )

    def record_failure(self, failure: TrainingFailure) -> None:
        self._atomic_evidence(
            lambda connection: self.record_failure_with_connection(connection, failure),
            "training failure run or owner conflict",
        )

    def _append(self, sql: str, parameters: tuple[object, ...], message: str) -> None:
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(sql, parameters)
                connection.commit()
        except sqlite3.DatabaseError:
            raise TrainingConflict(message) from None

    @staticmethod
    def _draft_from(value: FrozenTrainingDataset | TrainingDatasetDraft) -> TrainingDatasetDraft:
        return TrainingDatasetDraft.model_validate(
            value.model_dump(mode="python", exclude={"coverage", "dataset_digest"})
        )

    @staticmethod
    def _insert_samples(
        connection: sqlite3.Connection,
        storage_id: str,
        samples: tuple[TrainingSample, ...],
        *,
        ordinal_start: int = 0,
    ) -> None:
        for ordinal, sample in enumerate(samples, ordinal_start):
            connection.execute(
                "INSERT INTO training_samples VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    storage_id,
                    sample.sample_id,
                    sample.source_tier.value,
                    sample.sample_digest,
                    sample.source_lineage,
                    sample.caption,
                    sample.split.value,
                    canonical_json(sample.media.model_dump(mode="json")),
                    int(sample.accepted),
                    ordinal,
                ),
            )
            connection.execute(
                "INSERT INTO training_sample_annotations VALUES (?,?,?)",
                (
                    storage_id,
                    sample.sample_id,
                    canonical_json(sample.annotation.model_dump(mode="json")),
                ),
            )


_WINDOWS_DRIVE_PATH = re.compile(r"(?i)(?:^|\s)[a-z]:[\\/]")


def _contains_local_path(message: str) -> bool:
    lowered = message.lower()
    return (
        "file://" in lowered
        or message.startswith("/")
        or " /" in message
        or "\\\\" in message
        or bool(_WINDOWS_DRIVE_PATH.search(message))
    )
