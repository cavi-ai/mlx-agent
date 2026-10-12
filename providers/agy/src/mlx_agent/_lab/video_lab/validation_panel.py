from __future__ import annotations

import hashlib
import random

from video_lab.adaptive_study_models import AdaptiveOptimizerStep
from video_lab.validation_models import (
    ValidationCampaignDefinition,
    ValidationCandidates,
    ValidationIneligible,
    ValidationPanel,
    ValidationTrial,
    validation_panel_digest,
)


def derive_validation_candidates(step: AdaptiveOptimizerStep) -> ValidationCandidates:
    grid = step.posterior.grid
    winner = step.posterior.recommendation
    try:
        index = grid.index(winner)
    except ValueError as error:
        raise ValidationIneligible(
            "Stage 1 winner must appear in the frozen candidate grid"
        ) from error
    if index == 0 or index == len(grid) - 1:
        raise ValidationIneligible("Stage 1 winner must be an interior candidate")
    excluded = {winner, grid[index - 1], grid[index + 1]}
    distant_candidates = tuple(value for value in grid if value not in excluded)
    if not distant_candidates:
        raise ValidationIneligible("Stage 1 winner must have a distant challenger")
    distant = min(
        distant_candidates,
        key=lambda value: (-step.posterior.mean[grid.index(value)], value),
    )
    return ValidationCandidates(
        winner=winner,
        lower=grid[index - 1],
        higher=grid[index + 1],
        distant=distant,
    )


def build_validation_panel(definition: ValidationCampaignDefinition) -> ValidationPanel:
    scheduler = random.Random(definition.panel_input.schedule_seed)
    primary_trials: list[ValidationTrial] = []
    repeat_sources: dict[str, list[ValidationTrial]] = {}
    for prompt in definition.panel_input.prompts:
        sources: list[ValidationTrial] = []
        for challenger in definition.candidates.challengers:
            for seed in prompt.seeds:
                trial = ValidationTrial(
                    id=(f"{definition.id}:primary:{prompt.archetype.value}:{challenger!r}:{seed}"),
                    ordinal=len(primary_trials) + 1,
                    archetype=prompt.archetype,
                    prompt=prompt.prompt,
                    negative_prompt=prompt.negative_prompt,
                    seed=seed,
                    challenger=challenger,
                    winner_display_position="left",
                )
                primary_trials.append(trial)
                sources.append(trial)
        repeat_sources[prompt.archetype.value] = sorted(
            sources, key=lambda trial: (trial.challenger, trial.seed)
        )

    repeat_trials: list[ValidationTrial] = []
    for prompt in definition.panel_input.prompts:
        source = repeat_sources[prompt.archetype.value][
            scheduler.randrange(len(repeat_sources[prompt.archetype.value]))
        ]
        repeat_trials.append(
            ValidationTrial(
                id=f"{definition.id}:repeat:{prompt.archetype.value}",
                ordinal=len(primary_trials) + len(repeat_trials) + 1,
                archetype=source.archetype,
                prompt=source.prompt,
                negative_prompt=source.negative_prompt,
                seed=source.seed,
                challenger=source.challenger,
                hidden_repeat=True,
                reuses_trial_id=source.id,
                winner_display_position="left",
            )
        )

    presentation = primary_trials + repeat_trials
    scheduler.shuffle(presentation)
    display_positions = _balanced_display_positions(
        presentation, _display_seed(definition.panel_input.schedule_seed)
    )
    trials = tuple(
        trial.model_copy(
            update={
                "ordinal": ordinal,
                "winner_display_position": display_positions[trial.id],
            }
        )
        for ordinal, trial in enumerate(presentation, start=1)
    )
    _validate_display_balance(trials)
    return ValidationPanel(
        campaign_id=definition.id,
        trials=trials,
        content_digest=validation_panel_digest(definition.id, trials),
    )


def _display_seed(schedule_seed: int) -> int:
    payload = f"balanced-validation-display.v1:{schedule_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _balanced_display_positions(trials: list[ValidationTrial], seed: int) -> dict[str, str]:
    generator = random.Random(seed)
    by_challenger: dict[float, list[ValidationTrial]] = {}
    for trial in trials:
        by_challenger.setdefault(trial.challenger, []).append(trial)
    positions: dict[str, str] = {}
    odd_challengers: list[float] = []
    for challenger, group in by_challenger.items():
        count = len(group)
        labels = ["left"] * (count // 2) + ["right"] * (count // 2)
        if count % 2:
            odd_challengers.append(challenger)
        generator.shuffle(labels)
        for trial, position in zip(group, labels, strict=True):
            positions[trial.id] = position
    required_left = len(trials) // 2
    additional_left = required_left - sum(position == "left" for position in positions.values())
    generator.shuffle(odd_challengers)
    for challenger in odd_challengers[:additional_left]:
        trial = next(trial for trial in by_challenger[challenger] if trial.id not in positions)
        positions[trial.id] = "left"
    for challenger in odd_challengers[additional_left:]:
        trial = next(trial for trial in by_challenger[challenger] if trial.id not in positions)
        positions[trial.id] = "right"
    return positions


def _validate_display_balance(trials: tuple[ValidationTrial, ...]) -> None:
    positions = [trial.winner_display_position for trial in trials]
    if abs(positions.count("left") - positions.count("right")) > 1:
        raise ValidationIneligible("validation display schedule is not globally balanced")
    for challenger in {trial.challenger for trial in trials}:
        challenger_positions = [
            trial.winner_display_position for trial in trials if trial.challenger == challenger
        ]
        if abs(challenger_positions.count("left") - challenger_positions.count("right")) > 1:
            raise ValidationIneligible("validation display schedule is not challenger balanced")
