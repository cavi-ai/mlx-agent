from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from video_lab.character_benchmark_models import (
    CharacterBenchmarkPanelInput,
    CharacterContextGroup,
    CharacterRecipePair,
)
from video_lab.character_benchmark_protocol import (
    DEFAULT_CHARACTER_CONTEXTS,
    build_character_panel,
    schedule_character_trials,
)
from video_lab.domain import canonical_json


def panel_input(*, schedule_seed: int = 73021) -> CharacterBenchmarkPanelInput:
    return CharacterBenchmarkPanelInput(
        contexts=DEFAULT_CHARACTER_CONTEXTS,
        seeds=(101, 202, 303),
        schedule_seed=schedule_seed,
    )


def recipes(*, identity_digest: str = "sha256:identity") -> CharacterRecipePair:
    return CharacterRecipePair(
        control_digest="sha256:control",
        identity_digest=identity_digest,
    )


def definition(
    *,
    pack_digest: str = "sha256:pack",
    panel: CharacterBenchmarkPanelInput | None = None,
    recipe_pair: CharacterRecipePair | None = None,
):
    return build_character_panel(
        character_pack_digest=pack_digest,
        stage_2_manifest_digest="sha256:manifest",
        stage_2_evidence_digest="sha256:evidence",
        recipes=recipe_pair or recipes(),
        panel_input=panel or panel_input(),
    )


def test_frozen_panel_has_exact_contexts_seeds_and_groups() -> None:
    benchmark = definition()

    assert [context.ordinal for context in benchmark.panel_input.contexts] == list(range(1, 13))
    assert [context.name.value for context in benchmark.panel_input.contexts] == [
        "neutral_face_portrait",
        "three_quarter_portrait",
        "profile_view",
        "full_body_neutral_pose",
        "full_body_action",
        "strong_expression",
        "camera_motion",
        "character_object_interaction",
        "strong_directional_lighting",
        "partial_facial_occlusion",
        "foreshortened_or_difficult_angle",
        "busy_environment_with_competing_subjects",
    ]
    assert Counter(context.group for context in benchmark.panel_input.contexts) == {
        CharacterContextGroup.CORE: 8,
        CharacterContextGroup.STRESS: 4,
    }
    assert benchmark.panel_input.seeds == (101, 202, 303)
    with pytest.raises(ValidationError):
        benchmark.model_copy(update={"unexpected": True})
    with pytest.raises(ValidationError):
        benchmark.panel_input.model_copy(update={"seeds": (101, 101, 303)})


def test_schedule_has_36_primaries_and_four_media_reusing_hidden_repeats() -> None:
    benchmark = definition()
    trials = schedule_character_trials(benchmark)
    primary = tuple(trial for trial in trials if not trial.hidden_repeat)
    repeats = tuple(trial for trial in trials if trial.hidden_repeat)
    by_id = {trial.id: trial for trial in trials}

    assert len(trials) == 40
    assert len(primary) == 36
    assert len(repeats) == 4
    assert Counter((trial.context.ordinal, trial.seed) for trial in primary) == {
        (context.ordinal, seed): 1
        for context in benchmark.panel_input.contexts
        for seed in benchmark.panel_input.seeds
    }
    assert all(
        tuple(spec.candidate for spec in trial.generation_specs) == ("a", "b") for trial in primary
    )
    assert all(
        trial.generation_specs[0].recipe_digest == benchmark.recipes.control_digest
        for trial in primary
    )
    assert all(
        trial.generation_specs[1].recipe_digest == benchmark.recipes.identity_digest
        for trial in primary
    )
    assert all(trial.generation_specs == () for trial in repeats)
    assert all(trial.reuses_trial_id in by_id for trial in repeats)
    assert all(
        not by_id[trial.reuses_trial_id].hidden_repeat for trial in repeats if trial.reuses_trial_id
    )
    assert len({trial.reuses_trial_id for trial in repeats}) == 4
    assert [trial.ordinal for trial in trials] == list(range(1, 41))


def test_repeat_selection_order_and_positions_are_deterministic_balanced_and_sealed() -> None:
    benchmark = definition()
    first = schedule_character_trials(benchmark)
    second = schedule_character_trials(benchmark)

    assert first == second
    assert canonical_json(
        {"trials": [trial.model_dump(mode="json") for trial in first]}
    ) == canonical_json({"trials": [trial.model_dump(mode="json") for trial in second]})
    primary = tuple(trial for trial in first if not trial.hidden_repeat)
    repeats = tuple(trial for trial in first if trial.hidden_repeat)
    assert Counter(trial.candidate_a_position for trial in primary) == {"left": 18, "right": 18}
    assert Counter(trial.candidate_a_position for trial in repeats) == {"left": 2, "right": 2}
    for context in benchmark.panel_input.contexts:
        positions = Counter(
            trial.candidate_a_position for trial in primary if trial.context == context
        )
        assert abs(positions["left"] - positions["right"]) == 1
    source_ids = [trial.reuses_trial_id or trial.id for trial in first]
    assert all(left != right for left, right in zip(source_ids, source_ids[1:], strict=False))
    assert all(trial.id.startswith("character_trial_") for trial in first)
    assert len({trial.id for trial in first}) == 40
    assert len({spec.id for trial in first for spec in trial.generation_specs}) == 72


def test_content_ids_change_with_pack_panel_or_recipe_inputs() -> None:
    base_panel = panel_input()
    changed_contexts = list(base_panel.contexts)
    changed_contexts[0] = changed_contexts[0].model_copy(update={"prompt": "changed prompt"})
    changed_panel = base_panel.model_copy(update={"contexts": tuple(changed_contexts)})
    benchmarks = (
        definition(),
        definition(pack_digest="sha256:changed-pack"),
        definition(panel=changed_panel),
        definition(recipe_pair=recipes(identity_digest="sha256:changed-identity")),
    )

    assert all(item.id.startswith("character_benchmark_") for item in benchmarks)
    assert len({item.id for item in benchmarks}) == len(benchmarks)
    schedules = tuple(schedule_character_trials(item) for item in benchmarks)
    assert len({schedule[0].id for schedule in schedules}) == len(schedules)
