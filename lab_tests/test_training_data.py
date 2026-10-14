from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError
from video_lab.training_data import (
    DatasetMedia,
    DatasetNotReady,
    DatasetPolicy,
    DatasetSample,
    FrozenDataset,
    TrainingDataset,
    freeze_dataset,
)


def sample(index: int = 1, **changes: object) -> DatasetSample:
    data = {
        "sample_id": f"sample-{index}",
        "source_lineage": f"source-{index}",
        "caption": "A landscape with drifting clouds",
        "split": "train",
        "media": {
            "sha256": "sha256:" + hashlib.sha256(str(index).encode()).hexdigest(),
            "media_type": "video/mp4",
            "byte_size": 10,
            "width": 16,
            "height": 16,
            "duration_seconds": 10.0,
            "frame_count": 80,
            "fps": 8.0,
        },
    }
    data.update(changes)
    return DatasetSample.model_validate(data)


def test_generic_dataset_freezes_without_character_or_campaign_fields() -> None:
    frozen = freeze_dataset(TrainingDataset(samples=(sample(),)))
    payload = frozen.model_dump(mode="json")
    assert payload["samples"][0]["caption"] == "A landscape with drifting clouds"
    assert payload["policy"]["max_synthetic_share"] is None
    assert set(payload) == {"schema_version", "samples", "policy", "dataset_digest"}
    assert FrozenDataset.model_validate_json(frozen.model_dump_json()) == frozen


def test_dataset_identity_is_order_independent_and_binds_policy() -> None:
    left = freeze_dataset(TrainingDataset(samples=(sample(1), sample(2))))
    right = freeze_dataset(TrainingDataset(samples=(sample(2), sample(1))))
    changed = freeze_dataset(
        TrainingDataset(
            samples=(sample(1), sample(2)), policy=DatasetPolicy(min_samples=2)
        )
    )
    assert left.dataset_digest == right.dataset_digest
    assert left.dataset_digest != changed.dataset_digest


@pytest.mark.parametrize(
    ("samples", "reason"),
    [
        ((), "sample_count"),
        ((sample(), sample()), "duplicate_sample_id"),
        ((sample(), sample(2, media=sample().media)), "duplicate_content"),
        (
            (sample(), sample(2, source_lineage="source-1", split="validation")),
            "lineage_split_leakage",
        ),
        ((sample(split="validation"),), "training_split_empty"),
    ],
)
def test_generic_integrity_rejects_invalid_datasets(samples, reason) -> None:
    with pytest.raises(DatasetNotReady) as error:
        freeze_dataset(TrainingDataset(samples=samples))
    assert reason in error.value.reason_codes


def test_duration_limits_are_selected_by_caller() -> None:
    freeze_dataset(TrainingDataset(samples=(sample(),)))
    with pytest.raises(DatasetNotReady) as error:
        freeze_dataset(
            TrainingDataset(
                samples=(sample(),), policy=DatasetPolicy(max_clip_seconds=5)
            )
        )
    assert error.value.reason_codes == ("clip_duration",)


def test_synthetic_share_uses_weights_and_has_no_default_cap() -> None:
    items = (sample(1, synthetic=True, weight=3), sample(2, weight=1))
    freeze_dataset(TrainingDataset(samples=items))
    with pytest.raises(DatasetNotReady) as error:
        freeze_dataset(
            TrainingDataset(
                samples=items, policy=DatasetPolicy(max_synthetic_share=0.5)
            )
        )
    assert error.value.reason_codes == ("synthetic_share",)
    freeze_dataset(
        TrainingDataset(samples=items, policy=DatasetPolicy(max_synthetic_share=0.75))
    )


@pytest.mark.parametrize("weight", [0, -1, float("inf"), float("nan")])
def test_weights_must_be_finite_and_positive(weight) -> None:
    with pytest.raises(ValidationError):
        sample(weight=weight)


@pytest.mark.parametrize(
    "changes",
    [
        {"min_clip_seconds": 5, "max_clip_seconds": 2},
        {"min_samples": 0},
        {"max_synthetic_share": float("nan")},
    ],
)
def test_invalid_policy_is_rejected(changes) -> None:
    with pytest.raises(ValidationError):
        DatasetPolicy(**changes)


def test_frozen_dataset_rejects_modified_caption_and_policy() -> None:
    frozen = freeze_dataset(TrainingDataset(samples=(sample(),)))
    payload = json.loads(frozen.model_dump_json())
    payload["samples"][0]["caption"] = "Changed caption"
    with pytest.raises(ValidationError, match="dataset_digest"):
        FrozenDataset.model_validate(payload)
    payload = json.loads(frozen.model_dump_json())
    payload["policy"]["min_samples"] = 2
    with pytest.raises(ValidationError):
        FrozenDataset.model_validate(payload)


def test_media_rejects_nonfinite_values_and_invalid_still_metadata() -> None:
    payload = sample().media.model_dump(mode="json")
    payload["fps"] = float("inf")
    with pytest.raises(ValidationError):
        DatasetMedia.model_validate(payload)
    payload["fps"] = 8
    payload["media_type"] = "image/png"
    with pytest.raises(ValidationError):
        DatasetMedia.model_validate(payload)


def test_snapshot_is_immutable() -> None:
    frozen = freeze_dataset(TrainingDataset(samples=(sample(),)))
    with pytest.raises(ValidationError):
        frozen.samples[0].caption = "Changed"


@pytest.mark.parametrize(
    "changes",
    [
        {"caption": "Different landscape"},
        {"weight": 2.0},
        {"synthetic": True},
        {"source_lineage": "other-source"},
        {"sample_id": "other-id"},
    ],
)
def test_dataset_identity_binds_sample_fields(changes) -> None:
    original = freeze_dataset(TrainingDataset(samples=(sample(),)))
    changed = freeze_dataset(TrainingDataset(samples=(sample(**changes),)))
    assert original.dataset_digest != changed.dataset_digest


def test_still_dataset_does_not_require_video_or_character_labels() -> None:
    media = DatasetMedia(
        sha256=sample().media.sha256,
        media_type="image/png",
        byte_size=10,
        width=16,
        height=16,
    )
    frozen = freeze_dataset(TrainingDataset(samples=(sample(media=media),)))
    assert frozen.samples[0].media.duration_seconds is None


def test_synthetic_cap_rejects_exact_share_above_half_at_large_weights() -> None:
    items = (
        sample(1, synthetic=True, weight=1e100),
        sample(2, weight=1e100),
        sample(3, synthetic=True, weight=1),
    )
    with pytest.raises(DatasetNotReady) as error:
        freeze_dataset(
            TrainingDataset(
                samples=items, policy=DatasetPolicy(max_synthetic_share=0.5)
            )
        )
    assert error.value.reason_codes == ("synthetic_share",)


def test_unknown_character_policy_is_not_accepted_as_generic_configuration() -> None:
    with pytest.raises(ValidationError):
        DatasetPolicy(canonical_outfit_id="outfit")
