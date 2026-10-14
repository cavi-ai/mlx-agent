"""Pure dataset readiness and freezing rules for character LoRA training."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

from video_lab.training_data import dataset_integrity_reasons
from video_lab.training_models import (
    DatasetCoverage,
    FrozenTrainingDataset,
    TrainingDatasetDraft,
    TrainingRecipe,
    TrainingRecipeValidationEvidence,
    TrainingSample,
    TrainingSourceTier,
    training_dataset_id,
    training_recipe_id,
)


class TrainingDatasetNotReady(ValueError):
    """A deliberately path-free readiness failure with stable machine reason codes."""

    def __init__(self, reason_codes: Iterable[str]) -> None:
        self.reason_codes = tuple(sorted(set(reason_codes)))
        super().__init__("training dataset is not ready: " + ", ".join(self.reason_codes))


class TrainingRecipeDatasetValidationError(ValueError):
    """A stable trainer-neutral recipe/dataset contract failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def evaluate_training_dataset(draft: TrainingDatasetDraft) -> DatasetCoverage:
    """Calculate accepted-data counts and the complete curated coverage matrix."""
    samples = tuple(sample for sample in draft.samples if sample.accepted)
    gold_stills = _count(samples, TrainingSourceTier.GOLD_STILL)
    gold_clips = _count(samples, TrainingSourceTier.GOLD_CLIP)
    synthetic_clips = _count(samples, TrainingSourceTier.SYNTHETIC_CLIP)
    clips = gold_clips + synthetic_clips
    synthetic_weight = float(
        _synthetic_effective_share(
            gold_clip_count=gold_clips,
            synthetic_clip_count=synthetic_clips,
            gold_clip_weight=draft.source_tier_weights[TrainingSourceTier.GOLD_CLIP],
            synthetic_clip_weight=draft.source_tier_weights[TrainingSourceTier.SYNTHETIC_CLIP],
        )
    )
    annotations = tuple(sample.annotation for sample in samples)
    clip_annotations = tuple(sample.annotation for sample in _clips(samples))
    return DatasetCoverage(
        accepted_gold_stills=gold_stills,
        accepted_gold_clips=gold_clips,
        accepted_synthetic_clips=synthetic_clips,
        accepted_clips=clips,
        synthetic_effective_weight=synthetic_weight,
        frontal=any(annotation.head_angle == "front" for annotation in annotations),
        three_quarter=any(annotation.head_angle == "three_quarter" for annotation in annotations),
        profile=any(annotation.head_angle == "profile" for annotation in annotations),
        portrait=any(annotation.framing == "portrait" for annotation in annotations),
        full_body=any(annotation.framing == "full_body" for annotation in annotations),
        slow_motion=any(annotation.motion_speed == "slow" for annotation in clip_annotations),
        fast_motion=any(annotation.motion_speed == "fast" for annotation in clip_annotations),
        static_camera=any(annotation.camera_motion == "static" for annotation in clip_annotations),
        moving_camera=any(annotation.camera_motion == "moving" for annotation in clip_annotations),
        expression_change=any(annotation.expression_change for annotation in clip_annotations),
        partial_occlusion=any(annotation.partial_occlusion for annotation in annotations),
        ordinary_lighting=any(annotation.lighting == "ordinary" for annotation in annotations),
        directional_lighting=any(
            annotation.lighting == "directional" for annotation in annotations
        ),
        visible_outfit_details=any(annotation.visible_outfit_details for annotation in annotations),
    )


def freeze_training_dataset(draft: TrainingDatasetDraft) -> FrozenTrainingDataset:
    """Freeze a ready draft under a canonical content digest, otherwise raise stable codes."""
    coverage = evaluate_training_dataset(draft)
    reasons = _readiness_reasons(draft, coverage)
    if reasons:
        raise TrainingDatasetNotReady(reasons)
    return FrozenTrainingDataset(
        **draft.model_dump(mode="python"),
        dataset_digest=training_dataset_id(draft),
        coverage=coverage,
    )


def validate_training_recipe_dataset(
    recipe: TrainingRecipe, dataset: FrozenTrainingDataset
) -> TrainingRecipeValidationEvidence:
    """Bind a recipe to frozen data and enforce its composition-weighted clip cap."""
    current_draft = TrainingDatasetDraft.model_validate(
        dataset.model_dump(mode="python", exclude={"coverage", "dataset_digest"})
    )
    if training_dataset_id(current_draft) != dataset.dataset_digest:
        raise TrainingRecipeDatasetValidationError("dataset_digest_mismatch")
    if recipe.dataset_digest != dataset.dataset_digest:
        raise TrainingRecipeDatasetValidationError("dataset_digest_mismatch")
    if recipe.character_pack_digest != dataset.character_pack_digest:
        raise TrainingRecipeDatasetValidationError("character_pack_digest_mismatch")

    accepted = tuple(sample for sample in dataset.samples if sample.accepted)
    gold_clip_count = _count(accepted, TrainingSourceTier.GOLD_CLIP)
    synthetic_clip_count = _count(accepted, TrainingSourceTier.SYNTHETIC_CLIP)
    gold_clip_weight = recipe.source_tier_weights[TrainingSourceTier.GOLD_CLIP]
    synthetic_clip_weight = recipe.source_tier_weights[TrainingSourceTier.SYNTHETIC_CLIP]
    effective_share = _synthetic_effective_share(
        gold_clip_count=gold_clip_count,
        synthetic_clip_count=synthetic_clip_count,
        gold_clip_weight=gold_clip_weight,
        synthetic_clip_weight=synthetic_clip_weight,
    )
    if effective_share > Decimal("0.5"):
        raise TrainingRecipeDatasetValidationError("synthetic_clip_weight")

    return TrainingRecipeValidationEvidence(
        recipe_digest=training_recipe_id(recipe),
        dataset_digest=dataset.dataset_digest,
        character_pack_digest=dataset.character_pack_digest,
        accepted_gold_clip_count=gold_clip_count,
        accepted_synthetic_clip_count=synthetic_clip_count,
        gold_clip_weight=gold_clip_weight,
        synthetic_clip_weight=synthetic_clip_weight,
        synthetic_effective_share=float(effective_share),
    )


def _readiness_reasons(draft: TrainingDatasetDraft, coverage: DatasetCoverage) -> set[str]:
    accepted = tuple(sample for sample in draft.samples if sample.accepted)
    reasons: set[str] = set()
    if not 12 <= coverage.accepted_gold_stills <= 20:
        reasons.add("gold_still_count")
    if not 12 <= coverage.accepted_clips <= 24:
        reasons.add("clip_count")
    if coverage.accepted_gold_clips < 8:
        reasons.add("gold_clip_count")
    if _synthetic_effective_share(
        gold_clip_count=_count(accepted, TrainingSourceTier.GOLD_CLIP),
        synthetic_clip_count=_count(accepted, TrainingSourceTier.SYNTHETIC_CLIP),
        gold_clip_weight=draft.source_tier_weights[TrainingSourceTier.GOLD_CLIP],
        synthetic_clip_weight=draft.source_tier_weights[TrainingSourceTier.SYNTHETIC_CLIP],
    ) > Decimal("0.5"):
        reasons.add("synthetic_clip_weight")
    reasons.update(dataset_integrity_reasons(accepted))
    if any(
        sample.annotation.canonical_outfit_id != draft.canonical_outfit_id for sample in accepted
    ):
        reasons.add("canonical_outfit")
    if any(draft.reserved_token not in sample.caption for sample in accepted):
        reasons.add("reserved_token")
    if any(
        sample.media.duration_seconds is None or not 2.0 <= sample.media.duration_seconds <= 5.0
        for sample in _clips(accepted)
    ):
        reasons.add("clip_duration")
    if not all(
        getattr(coverage, field)
        for field in (
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
    ):
        reasons.add("coverage_incomplete")
    return reasons


def _count(samples: tuple[TrainingSample, ...], tier: TrainingSourceTier) -> int:
    return sum(sample.source_tier == tier for sample in samples)


def _clips(samples: tuple[TrainingSample, ...]) -> tuple[TrainingSample, ...]:
    return tuple(
        sample for sample in samples if sample.source_tier != TrainingSourceTier.GOLD_STILL
    )


def _synthetic_effective_share(
    *,
    gold_clip_count: int,
    synthetic_clip_count: int,
    gold_clip_weight: float,
    synthetic_clip_weight: float,
) -> Decimal:
    weighted_gold = Decimal(gold_clip_count) * Decimal(str(gold_clip_weight))
    weighted_synthetic = Decimal(synthetic_clip_count) * Decimal(str(synthetic_clip_weight))
    total_clip_weight = weighted_gold + weighted_synthetic
    if total_clip_weight == 0:
        return Decimal(0)
    return weighted_synthetic / total_clip_weight
