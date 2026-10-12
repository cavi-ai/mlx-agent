import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from lab_tests import test_training_repository as training_repository_fixtures
from lab_tests.test_training_dataset import _recipe
from lab_tests.test_training_repository import NOW, PACK_MANIFEST, _frozen
from video_lab.artifacts import ArtifactError, ArtifactStore
from video_lab.character_pack import CharacterPack
from video_lab.domain import canonical_json
from video_lab.training_media import TrainingMediaInspectionError
from video_lab.training_models import (
    CurationDecision,
    DatasetSplit,
    TrainingAnnotation,
    TrainingCurationSuggestions,
    TrainingSourceTier,
)
from video_lab.training_repository import TrainingConflict, TrainingNotFound, TrainingRepository
from video_lab.training_service import (
    CurationInput,
    DatasetCreateInput,
    RecipeInput,
    TrainingService,
    TrainingServiceError,
    _source_digest,
    discover_training_media,
)


def test_discovery_is_deterministic_and_source_digest_changes_with_bytes(tmp_path: Path) -> None:
    (tmp_path / "z.png").write_bytes(b"z")
    (tmp_path / "e\u0301.jpg").write_bytes(b"e")
    assert tuple(path.name for path in discover_training_media(tmp_path)) == (
        "z.png",
        "e\u0301.jpg",
    )
    before = _source_digest(tmp_path)
    (tmp_path / "z.png").write_bytes(b"changed")
    assert _source_digest(tmp_path) != before


def test_discovery_rejects_empty_and_unsupported(tmp_path: Path) -> None:
    with pytest.raises(TrainingServiceError, match="empty_media_folder"):
        discover_training_media(tmp_path)
    (tmp_path / "note.txt").write_text("x")
    with pytest.raises(TrainingServiceError, match="unsupported_entry"):
        discover_training_media(tmp_path)


class _PackService:
    @staticmethod
    def export_pack(pack_id: str) -> SimpleNamespace:
        pack = CharacterPack.model_validate(PACK_MANIFEST)
        assert pack.pack_id == pack_id
        return SimpleNamespace(pack=pack)


@pytest.fixture
def repository(tmp_path: Path) -> TrainingRepository:
    return training_repository_fixtures.repository.__wrapped__(tmp_path)


@pytest.fixture
def training_service(repository: TrainingRepository, tmp_path: Path) -> TrainingService:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VERSION = 1")
    return TrainingService(
        repository,
        ArtifactStore(repository.database, tmp_path / "artifacts"),
        _PackService(),
        reviewer_id="owner",
        source_root=source,
    )


def _new_dataset(service: TrainingService):
    return service.create_dataset(
        DatasetCreateInput(
            character_pack_id=PACK_MANIFEST["pack_id"],
            canonical_outfit_id="outfit",
            reserved_token="<hero>",
        )
    )


def _clone_completed_campaign(repository: TrainingRepository, campaign_id: str) -> str:
    with repository.database.connect() as connection:
        original = connection.execute(
            "SELECT * FROM character_benchmark_campaigns WHERE id='character_benchmark_training'"
        ).fetchone()
        definition = json.loads(str(original["definition_json"]))
        definition["id"] = campaign_id
        payload = {
            "definition": definition,
            "character_pack_id": str(original["character_pack_id"]),
            "character_pack_storage_id": str(original["character_pack_storage_id"]),
            "stage_2_campaign_id": str(original["stage_2_campaign_id"]),
            "stage_2_manifest": json.loads(str(original["stage_2_manifest_json"])),
            "identity_prompt": str(original["identity_prompt"]),
        }
        digest = "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        connection.execute(
            "INSERT INTO character_benchmark_campaigns VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                campaign_id,
                "owner",
                digest,
                canonical_json(definition),
                original["character_pack_id"],
                original["character_pack_storage_id"],
                original["stage_2_campaign_id"],
                original["stage_2_manifest_json"],
                original["identity_prompt"],
                NOW.isoformat(),
            ),
        )
        for sequence, (previous, new) in enumerate(
            (
                ("draft", "preregistered"),
                ("preregistered", "generating"),
                ("generating", "ready"),
                ("ready", "judging"),
                ("judging", "analyzing"),
            ),
            start=1,
        ):
            connection.execute(
                "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
                (
                    campaign_id,
                    sequence,
                    previous,
                    new,
                    "sha256:" + str(sequence) * 64,
                    NOW.isoformat(),
                ),
            )
        result = connection.execute(
            "SELECT * FROM character_benchmark_results "
            "WHERE campaign_id='character_benchmark_training'"
        ).fetchone()
        result_json = json.loads(str(result["result_json"]))
        result_json["benchmark_id"] = campaign_id
        result_json["baseline_disposition"] = "baseline_established"
        result_digest = "sha256:" + hashlib.sha256(canonical_json(result_json).encode()).hexdigest()
        connection.execute(
            "INSERT INTO character_benchmark_results VALUES (?,?,?,?)",
            (campaign_id, canonical_json(result_json), result_digest, NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO character_benchmark_events VALUES (?,?,?,?,?,?)",
            (campaign_id, 6, "analyzing", "completed", "sha256:" + "6" * 64, NOW.isoformat()),
        )
        return result_digest


def test_recipe_flow_explicitly_binds_one_of_two_eligible_campaigns(
    training_service: TrainingService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_campaign = "character_benchmark_selected"
    selected_result = _clone_completed_campaign(training_service.repository, selected_campaign)
    frozen = _frozen()
    training_service.repository.create_dataset_for_lineage(
        "training_lineage_recipe_e2e", frozen.character_id, frozen
    )
    with training_service.repository.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        training_service.repository.publish_dataset_head_with_connection(
            connection,
            lineage_id="training_lineage_recipe_e2e",
            expected_head_version=1,
            draft=frozen,
            seal=True,
        )
        connection.commit()
    base_path, adapter_path = tmp_path / "base.bin", tmp_path / "adapter.bin"
    base_path.write_bytes(b"real-base-material")
    adapter_path.write_bytes(b"real-adapter-material")
    base = training_service.artifacts.commit(
        training_service.artifacts.stage(base_path, "application/octet-stream"), None
    )
    adapter = training_service.artifacts.commit(
        training_service.artifacts.stage(adapter_path, "application/octet-stream"), None
    )
    request = RecipeInput(
        stage_3_campaign_id=selected_campaign,
        stage_3_result_id=selected_result,
        base_model_artifact_id=base.id,
        lightning_adapter_artifact_ids={"adapter": adapter.id},
        backend_id="animegen-wan",
        backend_config={},
        trainable_targets=("transformer.blocks.0.attn",),
        lora_rank=16,
        lora_alpha=16,
        lora_dropout=0,
        optimizer="adamw",
        learning_rate=1e-4,
        total_steps=100,
        gradient_accumulation_steps=1,
        source_tier_weights={tier: 1.0 for tier in TrainingSourceTier},
        calibration_steps=4,
        wall_clock_limit_seconds=28800,
        seed=7,
    )
    with training_service.repository.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT character_benchmark_provenance_is_valid(?)", (selected_campaign,)
            ).fetchone()[0]
            == 1
        )
    preview = training_service.preview_recipe(frozen.dataset_digest, request)
    created_recipe = training_service.create_recipe(
        frozen.dataset_digest, request, preview_token=preview.preview_token
    )
    record = training_service.repository.get_recipe(created_recipe.recipe_id)
    assert record.binding.stage_3_campaign_id == selected_campaign
    assert record.binding.stage_3_result_id == selected_result
    assert training_service.preregister_recipe(created_recipe.recipe_id).state == "preregistered"

    with pytest.raises(TrainingServiceError, match="prerequisite_drift"):
        training_service.preview_recipe(
            frozen.dataset_digest,
            request.model_copy(update={"stage_3_result_id": "sha256:" + "0" * 64}),
        )
    with pytest.raises(TrainingServiceError, match="prerequisite_missing"):
        training_service.preview_recipe(
            frozen.dataset_digest,
            request.model_copy(update={"stage_3_campaign_id": "character_benchmark_forged"}),
        )
    other_owner = TrainingService(
        TrainingRepository(training_service.repository.database, reviewer_id="other-owner"),
        training_service.artifacts,
        _PackService(),
        reviewer_id="other-owner",
        source_root=training_service.source_root,
    )
    with pytest.raises(TrainingServiceError, match="dataset_not_found"):
        other_owner.preview_recipe(frozen.dataset_digest, request)

    for seed, target in ((8, base), (9, adapter)):
        original_reverify = training_service.artifacts.reverify_open_handle
        observations = 0
        destination = training_service.artifacts.root / target.relative_path
        expected_bytes = destination.read_bytes()

        def replace_after_second_target_hash(
            connection: object,
            handle: object,
            original: object = original_reverify,
            target_id: str = target.id,
            target_path: Path = destination,
        ) -> None:
            nonlocal observations
            original(connection, handle)  # type: ignore[operator]
            if handle.artifact.id == target_id:  # type: ignore[attr-defined]
                observations += 1
                if observations == 2:
                    replacement = target_path.with_name(target_path.name + ".race")
                    replacement.write_bytes(b"replacement-race")
                    os.replace(replacement, target_path)

        monkeypatch.setattr(
            training_service.artifacts,
            "reverify_open_handle",
            replace_after_second_target_hash,
        )
        with training_service.repository.database.connect() as connection:
            before = tuple(
                connection.execute(
                    "SELECT (SELECT count(*) FROM training_recipes),"
                    "(SELECT count(*) FROM training_recipe_materials),"
                    "(SELECT count(*) FROM artifacts)"
                ).fetchone()
            )
        with pytest.raises(TrainingServiceError, match="training_service_failed"):
            training_service.create_recipe(
                frozen.dataset_digest, request.model_copy(update={"seed": seed})
            )
        with training_service.repository.database.connect() as connection:
            after = tuple(
                connection.execute(
                    "SELECT (SELECT count(*) FROM training_recipes),"
                    "(SELECT count(*) FROM training_recipe_materials),"
                    "(SELECT count(*) FROM artifacts)"
                ).fetchone()
            )
        assert after == before
        restoration = destination.with_name(destination.name + ".restore")
        restoration.write_bytes(expected_bytes)
        os.replace(restoration, destination)
        monkeypatch.setattr(training_service.artifacts, "reverify_open_handle", original_reverify)


def _validation_error() -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        RecipeInput.model_validate({})
    return caught.value


@pytest.mark.parametrize(
    ("operation", "failure", "expected_code"),
    (
        ("create_dataset", _validation_error(), "dataset_creation_failed"),
        (
            "import_media",
            ArtifactError("decoder stderr /private/secret"),
            "training_service_failed",
        ),
        ("curate_sample", TrainingNotFound("SELECT * FROM secret"), "training_service_failed"),
        (
            "dataset_status",
            TrainingConflict("sqlite conflict /private/db"),
            "training_service_failed",
        ),
        ("seal_dataset", ValueError("dataset not ready /private/data"), "training_service_failed"),
        (
            "preview_recipe",
            sqlite3.OperationalError("SELECT password FROM users"),
            "training_service_failed",
        ),
        ("create_recipe", OSError("/private/source.py"), "training_service_failed"),
        (
            "preregister_recipe",
            TrainingMediaInspectionError("corrupt_media"),
            "training_service_failed",
        ),
        (
            "export_dataset_evidence",
            RuntimeError("ffmpeg decoder stderr"),
            "training_service_failed",
        ),
        ("export_recipe_evidence", LookupError("/private/recipe"), "training_service_failed"),
    ),
)
def test_every_public_training_service_method_has_uniform_error_boundary(
    training_service: TrainingService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: Exception,
    expected_code: str,
) -> None:
    if operation == "create_dataset":
        monkeypatch.setattr(
            training_service.character_packs,
            "export_pack",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    elif operation in {
        "import_media",
        "curate_sample",
        "dataset_status",
        "seal_dataset",
        "export_dataset_evidence",
    }:
        monkeypatch.setattr(
            training_service, "_head", lambda *_args, **_kwargs: (_ for _ in ()).throw(failure)
        )
    elif operation in {"preview_recipe", "create_recipe"}:
        monkeypatch.setattr(
            training_service,
            "_frozen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            training_service.repository,
            "get_recipe",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )

    with training_service.repository.database.connect() as connection:
        before = tuple(
            connection.execute(
                "SELECT (SELECT count(*) FROM training_datasets),"
                "(SELECT count(*) FROM training_recipes),(SELECT count(*) FROM artifacts)"
            ).fetchone()
        )
    before_files = tuple(
        sorted(path for path in training_service.artifacts.root.rglob("*") if path.is_file())
    )
    recipe_request = RecipeInput.model_construct()
    calls = {
        "create_dataset": lambda: training_service.create_dataset(
            DatasetCreateInput.model_construct()
        ),
        "import_media": lambda: training_service.import_media("dataset", tmp_path),
        "curate_sample": lambda: training_service.curate_sample(
            "dataset", "import", CurationInput.model_construct(decision="reject")
        ),
        "dataset_status": lambda: training_service.dataset_status("dataset"),
        "seal_dataset": lambda: training_service.seal_dataset("dataset"),
        "preview_recipe": lambda: training_service.preview_recipe("dataset", recipe_request),
        "create_recipe": lambda: training_service.create_recipe("dataset", recipe_request),
        "preregister_recipe": lambda: training_service.preregister_recipe("recipe"),
        "export_dataset_evidence": lambda: training_service.export_dataset_evidence("dataset"),
        "export_recipe_evidence": lambda: training_service.export_recipe_evidence("recipe"),
    }
    with pytest.raises(TrainingServiceError) as caught:
        calls[operation]()
    assert caught.value.code == expected_code
    assert caught.value.__cause__ is None
    rendered = str(caught.value) + repr(caught.value)
    assert all(
        forbidden not in rendered
        for forbidden in (str(tmp_path), "/private", "SELECT", "stderr", "ffmpeg", "decoder")
    )
    with training_service.repository.database.connect() as connection:
        after = tuple(
            connection.execute(
                "SELECT (SELECT count(*) FROM training_datasets),"
                "(SELECT count(*) FROM training_recipes),(SELECT count(*) FROM artifacts)"
            ).fetchone()
        )
    after_files = tuple(
        sorted(path for path in training_service.artifacts.root.rglob("*") if path.is_file())
    )
    assert after == before
    assert after_files == before_files


def test_import_and_explicit_reject_publish_complete_immutable_ledgers(
    training_service: TrainingService, tmp_path: Path
) -> None:
    created = _new_dataset(training_service)
    media = tmp_path / "media"
    media.mkdir()
    Image.new("RGB", (8, 8), "red").save(media / "still.png")
    imported = training_service.import_media(
        created.dataset_id,
        media,
        suggestions=TrainingCurationSuggestions(caption="machine suggestion"),
    )
    assert imported.head_version == 2
    assert imported.imported[0].suggestions is not None
    rejected = training_service.curate_sample(
        imported.dataset_id,
        imported.imported[0].import_id,
        CurationInput(decision=CurationDecision.REJECT),
    )
    assert rejected.head_version == 3
    assert rejected.coverage.accepted_gold_stills == 0
    with pytest.raises(TrainingServiceError, match="stale_dataset"):
        training_service.dataset_status(imported.dataset_id)


def test_failed_duplicate_batch_leaves_head_and_import_ledger_unchanged(
    training_service: TrainingService, tmp_path: Path
) -> None:
    created = _new_dataset(training_service)
    media = tmp_path / "dupes"
    media.mkdir()
    Image.new("RGB", (8, 8), "blue").save(media / "a.png")
    (media / "b.png").write_bytes((media / "a.png").read_bytes())
    with pytest.raises(TrainingServiceError, match="duplicate_content"):
        training_service.import_media(created.dataset_id, media)
    status = training_service.dataset_status(created.lineage_id)
    assert status.head_version == 1 and status.imported == ()


def test_explicit_accept_uses_human_values_not_machine_suggestion(
    training_service: TrainingService, tmp_path: Path
) -> None:
    created = _new_dataset(training_service)
    media = tmp_path / "accepted"
    media.mkdir()
    Image.new("RGB", (8, 8), "green").save(media / "still.png")
    imported = training_service.import_media(
        created.dataset_id,
        media,
        suggestions=TrainingCurationSuggestions(caption="machine caption"),
    )
    accepted = training_service.curate_sample(
        imported.dataset_id,
        imported.imported[0].import_id,
        CurationInput(
            decision=CurationDecision.ACCEPT,
            sample_id="sample-human",
            source_tier=TrainingSourceTier.GOLD_STILL,
            source_lineage="human-lineage",
            caption="portrait of <hero>",
            split=DatasetSplit.TRAIN,
            annotation=TrainingAnnotation(
                canonical_outfit_id="outfit",
                head_angle="front",
                framing="portrait",
                motion_speed="slow",
                camera_motion="static",
                expression_change=False,
                partial_occlusion=False,
                lighting="ordinary",
                visible_outfit_details=True,
            ),
        ),
    )
    record = training_service.repository.get_dataset(accepted.dataset_id)
    assert record.draft.samples[0].caption == "portrait of <hero>"
    assert "machine" not in record.draft.samples[0].caption


def test_staged_mutation_rolls_back_artifact_import_and_head(
    training_service: TrainingService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _new_dataset(training_service)
    media = tmp_path / "mutated"
    media.mkdir()
    Image.new("RGB", (8, 8), "black").save(media / "still.png")
    real_stage = training_service.artifacts.stage

    def corrupt_stage(path: Path, media_type: str):
        staged = real_stage(path, media_type)
        staged.path.write_bytes(b"replaced")
        return staged

    monkeypatch.setattr(training_service.artifacts, "stage", corrupt_stage)
    with pytest.raises(TrainingServiceError, match="media_import_failed"):
        training_service.import_media(created.dataset_id, media)
    status = training_service.dataset_status(created.lineage_id)
    assert status.head_version == 1 and status.imported == ()


def test_dataset_evidence_is_deterministic_and_path_free(
    training_service: TrainingService,
) -> None:
    created = _new_dataset(training_service)
    first = training_service.export_dataset_evidence(created.dataset_id)
    assert first == training_service.export_dataset_evidence(created.dataset_id)
    assert str(training_service.source_root).encode() not in first


def test_base_lightning_source_and_runtime_drift_fail_path_free(
    training_service: TrainingService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_source = tmp_path / "base.bin"
    adapter_source = tmp_path / "adapter.bin"
    base_source.write_bytes(b"base-model")
    adapter_source.write_bytes(b"lightning")
    base = training_service.artifacts.commit(
        training_service.artifacts.stage(base_source, "application/octet-stream"), None
    )
    adapter = training_service.artifacts.commit(
        training_service.artifacts.stage(adapter_source, "application/octet-stream"), None
    )
    recipe = _recipe("sha256:" + "1" * 64).model_copy(
        update={
            "base_model_digest": f"sha256:{base.sha256}",
            "lightning_adapter_digests": {"lightning": f"sha256:{adapter.sha256}"},
            "source_code_digest": _source_digest(training_service.source_root),
            "runtime_identity": training_service.runtime.identity(),
        }
    )
    monkeypatch.setattr(training_service, "_provenance", lambda *_args, **_kwargs: ("", "", ""))

    base_path = training_service.artifacts.resolve(base.id)
    base_path.write_bytes(b"replaced")
    with pytest.raises(TrainingServiceError) as caught:
        training_service._verify_recipe_materials(recipe)
    assert caught.value.code == "material_drift" and str(base_path) not in str(caught.value)
    base_path.write_bytes(b"base-model")

    adapter_path = training_service.artifacts.resolve(adapter.id)
    adapter_path.write_bytes(b"replaced")
    with pytest.raises(TrainingServiceError, match="material_drift"):
        training_service._verify_recipe_materials(recipe)
    adapter_path.write_bytes(b"lightning")

    (training_service.source_root / "module.py").write_text("VERSION = 2")
    with pytest.raises(TrainingServiceError, match="source_drift"):
        training_service._verify_recipe_materials(recipe)
    recipe = recipe.model_copy(
        update={"source_code_digest": _source_digest(training_service.source_root)}
    )

    class ChangedRuntime:
        @staticmethod
        def identity():
            return {"runtime": "changed"}

    training_service.runtime = ChangedRuntime()
    with pytest.raises(TrainingServiceError, match="runtime_drift"):
        training_service._verify_recipe_materials(recipe)


def test_missing_or_unfinished_stage_provenance_fails_without_sensitive_cause(
    training_service: TrainingService,
) -> None:
    created = _new_dataset(training_service)
    with pytest.raises(TrainingServiceError) as caught:
        training_service._provenance(created.dataset_id, campaign_id="missing")
    assert caught.value.code == "prerequisite_missing"
    assert caught.value.__cause__ is None
    assert str(training_service.repository.database.path) not in str(caught.value)


def test_sealing_unready_dataset_is_side_effect_free(
    training_service: TrainingService,
) -> None:
    created = _new_dataset(training_service)
    with pytest.raises(TrainingServiceError, match="training_service_failed"):
        training_service.seal_dataset(created.dataset_id)
    assert training_service.dataset_status(created.lineage_id).head_version == 1
