import os
import shutil
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import video_lab.training_service as training_service_module
from lab_tests.test_training_dataset import _draft, _ready_samples, _recipe
from video_lab.training_dataset import (
    TrainingRecipeDatasetValidationError,
    freeze_training_dataset,
    validate_training_recipe_dataset,
)
from video_lab.training_models import CheckpointPlan, TrainingSourceTier
from video_lab.training_service import (
    RecipeInput,
    TrainingService,
    TrainingServiceError,
    _source_digest,
    discover_training_media,
)


def test_symlink_and_fifo_fail_path_free(tmp_path: Path) -> None:
    target = tmp_path / "secret.png"
    target.write_bytes(b"secret")
    link = tmp_path / "link.png"
    link.symlink_to(target)
    with pytest.raises(TrainingServiceError) as caught:
        discover_training_media(tmp_path)
    assert caught.value.code == "unsafe_path"
    assert str(tmp_path) not in str(caught.value)
    link.unlink()


@pytest.mark.parametrize("swap_ancestors", (False, True))
def test_source_snapshot_rejects_root_and_multi_ancestor_swaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap_ancestors: bool
) -> None:
    root = tmp_path / "source"
    for name in ("a", "b"):
        (root / name).mkdir(parents=True)
        (root / name / f"{name}.py").write_text(name)
    moved = tmp_path / "moved"
    original = training_service_module._snapshot_source_directory

    def mutate_after_snapshot(*args: object, **kwargs: object) -> int:
        total = original(*args, **kwargs)  # type: ignore[arg-type]
        if args[1] == "":
            if swap_ancestors:
                temporary = root / "temporary"
                (root / "a").rename(temporary)
                (root / "b").rename(root / "a")
                temporary.rename(root / "b")
            else:
                root.rename(moved)
                root.mkdir()
        return total

    monkeypatch.setattr(
        training_service_module, "_snapshot_source_directory", mutate_after_snapshot
    )
    with pytest.raises(TrainingServiceError, match="source_drift") as caught:
        training_service_module._source_snapshot_bytes(root)
    assert caught.value.__cause__ is None
    assert str(root) not in str(caught.value)
    os.mkfifo(tmp_path / "clip.mp4")
    with pytest.raises(TrainingServiceError, match="unsafe_path"):
        discover_training_media(tmp_path)


def test_source_snapshot_rejects_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-source.py"
    outside.write_text("secret")
    (tmp_path / "source.py").symlink_to(outside)
    with pytest.raises(TrainingServiceError, match="unsafe_source_root"):
        _source_digest(tmp_path)


def test_discovery_rejects_socket_and_symlink_root(tmp_path: Path) -> None:
    socket_root = Path("/tmp") / f"video-lab-{os.getpid()}"
    socket_root.mkdir(exist_ok=True)
    endpoint = socket_root / "m.mp4"
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(endpoint))
        with pytest.raises(TrainingServiceError, match="unsafe_path"):
            discover_training_media(socket_root)
    finally:
        server.close()
        shutil.rmtree(socket_root, ignore_errors=True)
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual)
    with pytest.raises(TrainingServiceError, match="unsafe_path"):
        discover_training_media(alias)


def _recipe_values() -> dict[str, object]:
    return {
        "stage_3_campaign_id": "campaign",
        "stage_3_result_id": "sha256:" + "9" * 64,
        "base_model_artifact_id": "base",
        "lightning_adapter_artifact_ids": {"lightning": "adapter"},
        "backend_id": "animegen-wan",
        "backend_config": {},
        "trainable_targets": ("transformer.blocks.0.attn",),
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "total_steps": 100,
        "gradient_accumulation_steps": 1,
        "source_tier_weights": {tier: 1.0 for tier in TrainingSourceTier},
        "checkpoint_plan": CheckpointPlan(),
        "calibration_steps": 4,
        "wall_clock_limit_seconds": 28800,
        "seed": 7,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (("trainable_targets", ()), ("total_steps", 3), ("wall_clock_limit_seconds", 1)),
)
def test_recipe_request_rejects_invalid_scientific_bounds(field: str, value: object) -> None:
    request = _recipe_values()
    request[field] = value
    with pytest.raises(ValidationError):
        RecipeInput.model_validate(request)


def test_source_evidence_is_deterministic_and_contains_no_path(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')")
    first = _source_digest(tmp_path)
    second = _source_digest(tmp_path)
    assert first == second
    assert str(tmp_path) not in first


def test_synthetic_effective_share_exact_half_is_valid_and_above_half_fails() -> None:
    frozen = freeze_training_dataset(_draft(_ready_samples()))
    exact = _recipe(frozen.dataset_digest).model_copy(
        update={
            "character_pack_digest": frozen.character_pack_digest,
            "source_tier_weights": {
                TrainingSourceTier.GOLD_STILL: 1.0,
                TrainingSourceTier.GOLD_CLIP: 1.0,
                TrainingSourceTier.SYNTHETIC_CLIP: 2.0,
            },
        }
    )
    evidence = validate_training_recipe_dataset(exact, frozen)
    assert evidence.synthetic_effective_share == 0.5
    above = exact.model_copy(
        update={
            "source_tier_weights": {
                **dict(exact.source_tier_weights),
                TrainingSourceTier.SYNTHETIC_CLIP: 2.0000000000000004,
            }
        }
    )
    with pytest.raises(TrainingRecipeDatasetValidationError, match="synthetic_clip_weight"):
        validate_training_recipe_dataset(above, frozen)


def _material_verification_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TrainingService:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VERSION = 1")
    service = object.__new__(TrainingService)
    service.source_root = source
    service.runtime = SimpleNamespace(identity=lambda: {"python": "3.12"})
    monkeypatch.setattr(service, "_artifact_for_digest", lambda _: None)
    monkeypatch.setattr(service, "_provenance", lambda *_args, **_kwargs: None)
    return service


def test_recipe_material_verification_rejects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _material_verification_service(tmp_path, monkeypatch)
    recipe = _recipe("sha256:" + "a" * 64)

    with pytest.raises(TrainingServiceError, match="source_drift"):
        service._verify_recipe_materials(recipe)


def test_recipe_material_verification_rejects_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _material_verification_service(tmp_path, monkeypatch)
    recipe = _recipe("sha256:" + "a" * 64).model_copy(
        update={
            "source_code_digest": _source_digest(service.source_root),
            "runtime_identity": {"python": "3.11"},
        }
    )

    with pytest.raises(TrainingServiceError, match="runtime_drift"):
        service._verify_recipe_materials(recipe)


def test_recipe_material_verification_preserves_material_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _material_verification_service(tmp_path, monkeypatch)
    recipe = _recipe("sha256:" + "a" * 64)

    def reject_material(_: str) -> None:
        raise TrainingServiceError("material_drift")

    monkeypatch.setattr(service, "_artifact_for_digest", reject_material)
    with pytest.raises(TrainingServiceError, match="material_drift"):
        service._verify_recipe_materials(recipe)
