from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import pytest

from video_lab.adaptive_study_models import AdaptiveOptimizerStep
from video_lab.domain import Manifest
from video_lab.preference_optimizer import PosteriorSummary
from video_lab.validation_models import (
    ValidationArchetype,
    ValidationCampaignDefinition,
    ValidationCandidates,
    ValidationIneligible,
    ValidationPanelInput,
    ValidationPrompt,
)
from video_lab.validation_panel import build_validation_panel, derive_validation_candidates


def terminal_step(
    *, grid: tuple[float, ...], means: tuple[float, ...], winner: float = 3.0
) -> AdaptiveOptimizerStep:
    return AdaptiveOptimizerStep.model_construct(
        posterior=PosteriorSummary(
            grid=grid,
            mean=means,
            covariance=tuple(tuple(0.0 for _ in grid) for _ in grid),
            argmax_probability=tuple(0.0 for _ in grid),
            recommendation=winner,
            concentration_interval=(grid[0], grid[-1]),
            concentration_probability=1.0,
            content_digest="sha256:controlled",
        )
    )


def test_candidate_derivation_uses_neighbors_and_best_distant() -> None:
    step = terminal_step(grid=(1.0, 2.0, 3.0, 4.0, 5.0), means=(0.0, 0.4, 1.0, 0.8, 0.9))

    candidates = derive_validation_candidates(step)

    assert candidates == ValidationCandidates(winner=3.0, lower=2.0, higher=4.0, distant=5.0)


def test_boundary_winner_is_ineligible() -> None:
    with pytest.raises(ValidationIneligible, match="interior"):
        derive_validation_candidates(
            terminal_step(grid=(1.0, 2.0, 3.0), means=(1.0, 0.2, 0.1), winner=1.0)
        )


def test_winner_without_a_distant_challenger_is_ineligible() -> None:
    with pytest.raises(ValidationIneligible, match="distant"):
        derive_validation_candidates(
            terminal_step(grid=(1.0, 2.0, 3.0), means=(0.1, 1.0, 0.2), winner=2.0)
        )


def test_winner_missing_from_frozen_grid_is_ineligible() -> None:
    with pytest.raises(ValidationIneligible, match="grid"):
        derive_validation_candidates(
            terminal_step(
                grid=(1.0, 2.0, 3.0, 4.0),
                means=(0.1, 0.2, 0.3, 0.4),
                winner=3.5,
            )
        )


def manifest() -> Manifest:
    return Manifest(
        model_id="org/model",
        model_revision="abc123",
        model_checksum="sha256:weights",
        adapter_id="diffusers-mps",
        adapter_version="0.1.0",
        runtime_versions={"python": "3.12.0"},
        prompt="a blue-haired character walking",
        negative_prompt="flicker",
        seed=42,
        width=512,
        height=288,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=1.0,
        scheduler="flow-match-euler",
        dtype="float16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="unified",
        hardware_fingerprint="apple-m5-max-128gb",
        os_fingerprint="macos-test",
    )


def definition(schedule_seed: int = 73021) -> ValidationCampaignDefinition:
    prompts = tuple(
        ValidationPrompt(
            archetype=archetype,
            prompt=f"character test {archetype.value}",
            negative_prompt="flicker",
            seeds=(101, 202, 303),
        )
        for archetype in ValidationArchetype
    )
    return ValidationCampaignDefinition(
        id="validation_1",
        stage_1_study_id="study_1",
        stage_1_evidence_digest="a" * 64,
        base_manifest=manifest(),
        candidates=ValidationCandidates(winner=3.0, lower=2.0, higher=4.0, distant=5.0),
        panel_input=ValidationPanelInput(prompts=prompts, schedule_seed=schedule_seed),
        created_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )


def test_panel_has_36_primary_and_four_hidden_repeats() -> None:
    panel = build_validation_panel(definition())

    assert len(panel.trials) == 40
    assert sum(not trial.hidden_repeat for trial in panel.trials) == 36
    assert sum(trial.hidden_repeat for trial in panel.trials) == 4
    assert Counter(trial.archetype for trial in panel.trials if trial.hidden_repeat) == {
        archetype: 1 for archetype in ValidationArchetype
    }
    assert all(trial.reuses_trial_id is None for trial in panel.trials if not trial.hidden_repeat)
    assert all(trial.reuses_trial_id is not None for trial in panel.trials if trial.hidden_repeat)


def test_panel_golden_vector_is_replay_stable_and_balanced() -> None:
    first = build_validation_panel(definition(schedule_seed=73021))
    second = build_validation_panel(definition(schedule_seed=73021))

    assert first == second
    assert first.content_digest == (
        "sha256:1c57ecc3262142380f131dc5869bde33598920bd500f6ef61a65c462b6b934f5"
    )
    assert (
        abs(
            sum(trial.winner_display_position == "left" for trial in first.trials)
            - sum(trial.winner_display_position == "right" for trial in first.trials)
        )
        <= 1
    )
    for challenger in first.challengers:
        trials = [trial for trial in first.trials if trial.challenger == challenger]
        assert (
            abs(
                sum(trial.winner_display_position == "left" for trial in trials)
                - sum(trial.winner_display_position == "right" for trial in trials)
            )
            <= 1
        )
