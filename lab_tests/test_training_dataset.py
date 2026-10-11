from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

import pytest

from video_lab.training_dataset import (
    TrainingDatasetNotReady,
    TrainingRecipeDatasetValidationError,
    evaluate_training_dataset,
    freeze_training_dataset,
    validate_training_recipe_dataset,
)
from video_lab.training_models import (
    CheckpointPlan,
    CuratedEvidenceReferenceV1,
    DatasetSplit,
    FrozenTrainingDataset,
    TrainingAnnotation,
    TrainingDatasetDraft,
    TrainingMedia,
    TrainingRecipe,
    TrainingSample,
    TrainingSourceTier,
    training_dataset_id,
    training_recipe_id,
)


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _annotation(index: int, outfit: str = "jacket-v1") -> TrainingAnnotation:
    return TrainingAnnotation(
        canonical_outfit_id=outfit,
        head_angle=("front", "three_quarter", "profile")[index % 3],
        framing=("portrait", "full_body")[index % 2],
        motion_speed=("slow", "fast")[index % 2],
        camera_motion=("static", "moving")[index % 2],
        expression_change=index % 12 == 0,
        partial_occlusion=index == 1,
        lighting=("ordinary", "directional")[index % 2],
        visible_outfit_details=True,
    )


def _sample(
    index: int,
    tier: TrainingSourceTier,
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
    lineage: str | None = None,
    outfit: str = "jacket-v1",
    digest: str | None = None,
) -> TrainingSample:
    video = tier != TrainingSourceTier.GOLD_STILL
    sha256 = digest or f"sha256:{index:064x}"
    return TrainingSample(
        sample_id=f"sample-{index}",
        source_tier=tier,
        sample_digest=sha256,
        source_lineage=lineage or f"lineage-{index}",
        caption="<mirza> wearing the canonical jacket",
        split=split,
        media=TrainingMedia(
            sha256=sha256,
            media_type="video/mp4" if video else "image/png",
            byte_size=10,
            width=4,
            height=4,
            duration_seconds=2.0 if video else None,
            frame_count=16 if video else None,
            fps=8.0 if video else None,
        ),
        annotation=_annotation(index, outfit),
    )


def _draft(samples: Iterable[TrainingSample]) -> TrainingDatasetDraft:
    return TrainingDatasetDraft(
        character_id="mirza",
        character_pack_digest=_digest("pack"),
        canonical_outfit_id="jacket-v1",
        reserved_token="<mirza>",
        samples=tuple(samples),
    )


def _ready_samples() -> tuple[TrainingSample, ...]:
    stills = tuple(_sample(index, TrainingSourceTier.GOLD_STILL) for index in range(12))
    clips = tuple(_sample(index + 12, TrainingSourceTier.GOLD_CLIP) for index in range(8))
    clips += tuple(_sample(index + 20, TrainingSourceTier.SYNTHETIC_CLIP) for index in range(4))
    return stills + clips


def _recipe(
    dataset_digest: str,
    *,
    gold_clip_weight: float = 1.0,
    synthetic_clip_weight: float = 1.0,
) -> TrainingRecipe:
    return TrainingRecipe(
        schema_version="character-training-recipe.v1",
        dataset_digest=dataset_digest,
        character_pack_digest=_digest("pack"),
        stage_2_manifest_digest=_digest("manifest"),
        stage_2_evidence_digest=_digest("stage2"),
        stage_3_evidence_digest=_digest("stage3"),
        base_model_digest=_digest("base"),
        lightning_adapter_digests={"adapter": _digest("adapter")},
        source_code_digest=_digest("source"),
        runtime_identity={"python": "3.12"},
        backend_id="test-backend",
        backend_config={"precision": "bf16"},
        trainable_targets=("layers.1",),
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        optimizer="adamw",
        learning_rate=0.0001,
        total_steps=200,
        gradient_accumulation_steps=4,
        source_tier_weights={
            TrainingSourceTier.GOLD_STILL: 1.0,
            TrainingSourceTier.GOLD_CLIP: gold_clip_weight,
            TrainingSourceTier.SYNTHETIC_CLIP: synthetic_clip_weight,
        },
        checkpoint_plan=CheckpointPlan(),
        calibration_steps=10,
        wall_clock_limit_seconds=28800,
        seed=7,
    )


def _mutate_dataset_identity_field(
    draft: TrainingDatasetDraft, field_path: str
) -> TrainingDatasetDraft:
    dataset_values = {
        "schema_version": "changed-schema",
        "character_id": "other-character",
        "character_pack_digest": _digest("other-pack"),
        "canonical_outfit_id": "other-outfit",
        "reserved_token": "<other>",
        "source_tier_weights": {
            TrainingSourceTier.GOLD_STILL: 2.0,
            TrainingSourceTier.GOLD_CLIP: 1.0,
            TrainingSourceTier.SYNTHETIC_CLIP: 1.0,
        },
    }
    if field_path in dataset_values:
        return TrainingDatasetDraft.model_construct(
            **{**draft.__dict__, field_path: dataset_values[field_path]}
        )

    sample = draft.samples[0]
    sample_values: dict[str, object] = {
        "sample_id": "changed-sample",
        "source_tier": TrainingSourceTier.GOLD_CLIP,
        "sample_digest": _digest("changed-sample"),
        "source_lineage": "changed-lineage",
        "caption": "<mirza> changed caption",
        "split": DatasetSplit.VALIDATION,
        "accepted": False,
    }
    nested_path = field_path.removeprefix("sample.")
    if nested_path in sample_values:
        changed_sample = TrainingSample.model_construct(
            **{**sample.__dict__, nested_path: sample_values[nested_path]}
        )
    elif nested_path.startswith("media."):
        media_field = nested_path.removeprefix("media.")
        media_values: dict[str, object] = {
            "schema_version": "changed-media-schema",
            "sha256": _digest("changed-media"),
            "media_type": "image/jpeg",
            "byte_size": 11,
            "width": 5,
            "height": 6,
            "duration_seconds": 2.0,
            "frame_count": 2,
            "fps": 1.0,
        }
        changed_media = TrainingMedia.model_construct(
            **{**sample.media.__dict__, media_field: media_values[media_field]}
        )
        changed_sample = TrainingSample.model_construct(
            **{**sample.__dict__, "media": changed_media}
        )
    else:
        annotation_field = nested_path.removeprefix("annotation.")
        annotation_values: dict[str, object] = {
            "canonical_outfit_id": "other-outfit",
            "head_angle": "profile",
            "framing": "full_body",
            "motion_speed": "fast",
            "camera_motion": "moving",
            "expression_change": False,
            "partial_occlusion": True,
            "lighting": "directional",
            "visible_outfit_details": False,
        }
        changed_annotation = TrainingAnnotation.model_construct(
            **{
                **sample.annotation.__dict__,
                annotation_field: annotation_values[annotation_field],
            }
        )
        changed_sample = TrainingSample.model_construct(
            **{**sample.__dict__, "annotation": changed_annotation}
        )
    return TrainingDatasetDraft.model_construct(
        **{**draft.__dict__, "samples": (changed_sample, *draft.samples[1:])}
    )


def test_freeze_is_order_independent_and_binds_each_sample_contract_field() -> None:
    """Reordering samples must not change identity, but changing caption/split/lineage must."""
    samples = _ready_samples()
    frozen = freeze_training_dataset(_draft(samples))
    reordered = freeze_training_dataset(_draft(reversed(samples)))
    changed = samples[0].model_copy(update={"caption": "<mirza> changed caption"})
    changed_frozen = freeze_training_dataset(_draft((changed,) + samples[1:]))

    assert frozen.dataset_digest == reordered.dataset_digest
    assert frozen.dataset_digest != changed_frozen.dataset_digest


def test_curated_evidence_is_backward_compatible_bound_and_survives_freeze() -> None:
    draft = _draft(_ready_samples())
    legacy_payload = draft.model_dump(mode="json")
    for sample in legacy_payload["samples"]:
        sample.pop("curated_evidence")
    legacy_payload["samples"] = sorted(
        legacy_payload["samples"],
        key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    legacy_encoded = json.dumps(
        legacy_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    legacy_digest = "sha256:" + hashlib.sha256(legacy_encoded.encode()).hexdigest()
    assert training_dataset_id(draft) == legacy_digest

    reference = CuratedEvidenceReferenceV1(
        curated_dataset_digest=_digest("curated"),
        validation_report_digest=_digest("report"),
        sequence_id="sequence_a",
        validation_state="accepted",
    )
    changed_sample = draft.samples[0].model_copy(update={"curated_evidence": reference})
    enriched = draft.model_copy(update={"samples": (changed_sample, *draft.samples[1:])})
    frozen = freeze_training_dataset(enriched)

    assert frozen.dataset_digest != legacy_digest
    assert frozen.samples[0].curated_evidence == reference
    assert frozen.coverage == evaluate_training_dataset(draft)


@pytest.mark.parametrize(
    "field_path",
    [
        "schema_version",
        "character_id",
        "character_pack_digest",
        "canonical_outfit_id",
        "reserved_token",
        "source_tier_weights",
        "sample.sample_id",
        "sample.source_tier",
        "sample.sample_digest",
        "sample.source_lineage",
        "sample.caption",
        "sample.split",
        "sample.accepted",
        "sample.media.schema_version",
        "sample.media.sha256",
        "sample.media.media_type",
        "sample.media.byte_size",
        "sample.media.width",
        "sample.media.height",
        "sample.media.duration_seconds",
        "sample.media.frame_count",
        "sample.media.fps",
        "sample.annotation.canonical_outfit_id",
        "sample.annotation.head_angle",
        "sample.annotation.framing",
        "sample.annotation.motion_speed",
        "sample.annotation.camera_motion",
        "sample.annotation.expression_change",
        "sample.annotation.partial_occlusion",
        "sample.annotation.lighting",
        "sample.annotation.visible_outfit_details",
    ],
)
def test_dataset_id_changes_when_any_identity_field_changes(field_path: str) -> None:
    """Omitting one dataset or sample field would alias scientifically distinct inputs."""
    draft = _draft(_ready_samples())

    changed = _mutate_dataset_identity_field(draft, field_path)

    assert training_dataset_id(changed) != training_dataset_id(draft)


@pytest.mark.parametrize(
    ("changed_samples", "reason"),
    [
        (_ready_samples()[:11], "gold_still_count"),
        (_ready_samples()[:12] + _ready_samples()[12:19], "gold_clip_count"),
        (
            _ready_samples()
            + tuple(_sample(100 + index, TrainingSourceTier.GOLD_CLIP) for index in range(13)),
            "clip_count",
        ),
        (
            _ready_samples() + (_sample(100, TrainingSourceTier.SYNTHETIC_CLIP),) * 5,
            "synthetic_clip_weight",
        ),
    ],
)
def test_freeze_rejects_required_count_and_synthetic_weight_gates(
    changed_samples: tuple[TrainingSample, ...], reason: str
) -> None:
    """Relaxing accepted-data gates or the synthetic cap would corrupt training evidence."""
    with pytest.raises(TrainingDatasetNotReady) as raised:
        freeze_training_dataset(_draft(changed_samples))

    assert reason in raised.value.reason_codes


def test_freeze_uses_effective_source_tier_weights_for_synthetic_cap() -> None:
    """Using counts instead of the configured clip-sampling weights is a training bias bug."""
    draft = _draft(_ready_samples()).model_copy(
        update={
            "source_tier_weights": {
                TrainingSourceTier.GOLD_STILL: 1.0,
                TrainingSourceTier.GOLD_CLIP: 1.0,
                TrainingSourceTier.SYNTHETIC_CLIP: 3.0,
            }
        }
    )

    with pytest.raises(TrainingDatasetNotReady) as raised:
        freeze_training_dataset(draft)

    assert "synthetic_clip_weight" in raised.value.reason_codes


def test_freeze_rejects_one_ulp_above_exact_synthetic_half() -> None:
    """Float-rounding an exact above-half dataset share must not bypass readiness."""
    draft = _draft(_ready_samples()).model_copy(
        update={
            "source_tier_weights": {
                TrainingSourceTier.GOLD_STILL: 1.0,
                TrainingSourceTier.GOLD_CLIP: 1.0,
                TrainingSourceTier.SYNTHETIC_CLIP: math.nextafter(2.0, math.inf),
            }
        }
    )

    with pytest.raises(TrainingDatasetNotReady) as raised:
        freeze_training_dataset(draft)

    assert "synthetic_clip_weight" in raised.value.reason_codes


def test_freeze_rejects_duplicates_lineage_leakage_and_outfit_or_token_mismatch() -> None:
    """Allowing duplicate bytes, leakage, outfit mismatch, or tokenless captions is a bug."""
    samples = list(_ready_samples())
    duplicate = samples[0].model_copy(update={"sample_id": "duplicate", "source_lineage": "other"})
    leakage = samples[1].model_copy(
        update={"split": DatasetSplit.VALIDATION, "source_lineage": samples[0].source_lineage}
    )
    wrong_outfit = samples[2].model_copy(update={"annotation": _annotation(2, "other-outfit")})
    missing_token = samples[3].model_copy(update={"caption": "the canonical jacket"})

    with pytest.raises(TrainingDatasetNotReady) as raised:
        freeze_training_dataset(_draft([*samples, duplicate, leakage, wrong_outfit, missing_token]))

    assert {
        "duplicate_content",
        "lineage_split_leakage",
        "canonical_outfit",
        "reserved_token",
    } <= set(raised.value.reason_codes)


def test_evaluation_reports_every_required_coverage_dimension() -> None:
    """Dropping any required pose, motion, camera, lighting, or outfit dimension is a bug."""
    coverage = evaluate_training_dataset(_draft(_ready_samples()))

    assert coverage.accepted_gold_stills == 12
    assert coverage.accepted_clips == 12
    assert all(
        getattr(coverage, name)
        for name in (
            "frontal",
            "three_quarter",
            "profile",
            "portrait",
            "full_body",
            "slow_motion",
            "fast_motion",
            "static_camera",
            "moving_camera",
            "expression_change",
            "partial_occlusion",
            "ordinary_lighting",
            "directional_lighting",
            "visible_outfit_details",
        )
    )


@pytest.mark.parametrize(
    ("coverage_field", "annotation_update"),
    [
        ("frontal", {"head_angle": "profile"}),
        ("three_quarter", {"head_angle": "front"}),
        ("profile", {"head_angle": "front"}),
        ("portrait", {"framing": "full_body"}),
        ("full_body", {"framing": "portrait"}),
        ("slow_motion", {"motion_speed": "fast"}),
        ("fast_motion", {"motion_speed": "slow"}),
        ("static_camera", {"camera_motion": "moving"}),
        ("moving_camera", {"camera_motion": "static"}),
        ("expression_change", {"expression_change": False}),
        ("partial_occlusion", {"partial_occlusion": False}),
        ("ordinary_lighting", {"lighting": "directional"}),
        ("directional_lighting", {"lighting": "ordinary"}),
        ("visible_outfit_details", {"visible_outfit_details": False}),
    ],
)
def test_freeze_rejects_each_missing_coverage_dimension(
    coverage_field: str, annotation_update: dict[str, object]
) -> None:
    """Dropping any required curated dimension must keep the dataset unfrozen."""
    samples = tuple(
        sample.model_copy(
            update={"annotation": sample.annotation.model_copy(update=annotation_update)}
        )
        for sample in _ready_samples()
    )
    draft = _draft(samples)

    assert getattr(evaluate_training_dataset(draft), coverage_field) is False
    with pytest.raises(TrainingDatasetNotReady) as raised:
        freeze_training_dataset(draft)
    assert "coverage_incomplete" in raised.value.reason_codes


def test_freeze_rejects_temporal_coverage_supplied_only_by_stills() -> None:
    """Allowing still-only motion or expression labels would fabricate temporal coverage."""
    samples = list(_ready_samples())
    for index, sample in enumerate(samples):
        if sample.source_tier != TrainingSourceTier.GOLD_STILL:
            samples[index] = sample.model_copy(
                update={
                    "annotation": sample.annotation.model_copy(
                        update={
                            "motion_speed": "slow",
                            "camera_motion": "static",
                            "expression_change": False,
                        }
                    )
                }
            )

    with pytest.raises(TrainingDatasetNotReady) as raised:
        freeze_training_dataset(_draft(samples))

    assert "coverage_incomplete" in raised.value.reason_codes


@pytest.mark.parametrize(
    (
        "gold_clip_count",
        "synthetic_clip_count",
        "gold_clip_weight",
        "synthetic_clip_weight",
    ),
    [
        (8, 4, 1.0, 2.0),
        (8, 4, 10.0, 20.0),
        (9, 3, 1.0, 3.0),
        (8, 8, 1.0, 1.0),
    ],
)
def test_recipe_dataset_validation_accepts_exact_half_from_real_composition_at_any_scale(
    gold_clip_count: int,
    synthetic_clip_count: int,
    gold_clip_weight: float,
    synthetic_clip_weight: float,
) -> None:
    """Using raw or scale-sensitive tier weights would reject valid exact-half sampling."""
    samples = (
        tuple(_sample(index, TrainingSourceTier.GOLD_STILL) for index in range(12))
        + tuple(
            _sample(index + 12, TrainingSourceTier.GOLD_CLIP) for index in range(gold_clip_count)
        )
        + tuple(
            _sample(index + 12 + gold_clip_count, TrainingSourceTier.SYNTHETIC_CLIP)
            for index in range(synthetic_clip_count)
        )
    )
    dataset = freeze_training_dataset(_draft(samples))
    recipe = _recipe(
        dataset.dataset_digest,
        gold_clip_weight=gold_clip_weight,
        synthetic_clip_weight=synthetic_clip_weight,
    )

    evidence = validate_training_recipe_dataset(recipe, dataset)

    assert evidence.accepted_gold_clip_count == gold_clip_count
    assert evidence.accepted_synthetic_clip_count == synthetic_clip_count
    assert evidence.synthetic_effective_share == pytest.approx(0.5)
    assert evidence.recipe_digest == training_recipe_id(recipe)
    assert evidence.dataset_digest == dataset.dataset_digest


def test_recipe_dataset_validation_rejects_composition_share_above_half() -> None:
    """A recipe that gives synthetic clips over half of actual sampling must be rejected."""
    dataset = freeze_training_dataset(_draft(_ready_samples()))
    recipe = _recipe(
        dataset.dataset_digest,
        gold_clip_weight=1.0,
        synthetic_clip_weight=2.00001,
    )

    with pytest.raises(TrainingRecipeDatasetValidationError) as raised:
        validate_training_recipe_dataset(recipe, dataset)

    assert raised.value.code == "synthetic_clip_weight"


def test_recipe_dataset_validation_rejects_one_ulp_above_exact_half() -> None:
    """Rounding an exact above-half Decimal share to float 0.5 would bypass the cap."""
    dataset = freeze_training_dataset(_draft(_ready_samples()))
    recipe = _recipe(
        dataset.dataset_digest,
        gold_clip_weight=1.0,
        synthetic_clip_weight=math.nextafter(2.0, math.inf),
    )

    with pytest.raises(TrainingRecipeDatasetValidationError) as raised:
        validate_training_recipe_dataset(recipe, dataset)

    assert raised.value.code == "synthetic_clip_weight"


def test_recipe_dataset_validation_recomputes_order_independent_digest_from_samples() -> None:
    """Retaining a frozen digest after sample mutation must not validate forged evidence."""
    dataset = freeze_training_dataset(_draft(_ready_samples()))
    recipe = _recipe(dataset.dataset_digest)
    reordered = dataset.model_copy(update={"samples": tuple(reversed(dataset.samples))})
    changed_sample = dataset.samples[12].model_copy(
        update={"source_tier": TrainingSourceTier.SYNTHETIC_CLIP}
    )
    forged = dataset.model_copy(
        update={"samples": (*dataset.samples[:12], changed_sample, *dataset.samples[13:])}
    )

    assert validate_training_recipe_dataset(recipe, reordered).dataset_digest == (
        dataset.dataset_digest
    )
    with pytest.raises(TrainingRecipeDatasetValidationError) as raised:
        validate_training_recipe_dataset(recipe, forged)

    assert raised.value.code == "dataset_digest_mismatch"


def test_recipe_dataset_validation_recomputes_counts_instead_of_trusting_coverage() -> None:
    """Forged caller-supplied coverage counts must not weaken operative recipe validation."""
    dataset = freeze_training_dataset(_draft(_ready_samples()))
    forged_coverage = dataset.coverage.model_copy(
        update={
            "accepted_gold_clips": 12,
            "accepted_synthetic_clips": 0,
            "synthetic_effective_weight": 0.0,
        }
    )
    forged_dataset = dataset.model_copy(update={"coverage": forged_coverage})
    recipe = _recipe(
        dataset.dataset_digest,
        gold_clip_weight=1.0,
        synthetic_clip_weight=3.0,
    )

    with pytest.raises(TrainingRecipeDatasetValidationError) as raised:
        validate_training_recipe_dataset(recipe, forged_dataset)

    assert raised.value.code == "synthetic_clip_weight"


def test_frozen_dataset_digest_requires_exact_lowercase_sha256_format() -> None:
    """A malformed frozen dataset ID would break its binding to recipe evidence."""
    dataset = freeze_training_dataset(_draft(_ready_samples()))
    payload = dataset.model_dump(mode="python")
    payload["dataset_digest"] = "sha256:not-a-digest"

    with pytest.raises(ValueError):
        FrozenTrainingDataset.model_validate(payload)


@pytest.mark.parametrize(
    ("recipe_update", "code"),
    [
        ({"dataset_digest": _digest("other-dataset")}, "dataset_digest_mismatch"),
        ({"character_pack_digest": _digest("other-pack")}, "character_pack_digest_mismatch"),
    ],
)
def test_recipe_dataset_validation_binds_dataset_and_character_pack_digests(
    recipe_update: dict[str, object], code: str
) -> None:
    """Validating a recipe against unrelated frozen evidence would corrupt preregistration."""
    dataset = freeze_training_dataset(_draft(_ready_samples()))
    recipe = _recipe(dataset.dataset_digest).model_copy(update=recipe_update)

    with pytest.raises(TrainingRecipeDatasetValidationError) as raised:
        validate_training_recipe_dataset(recipe, dataset)

    assert raised.value.code == code
