from __future__ import annotations

import hashlib
import random
from typing import Any

from video_lab.character_benchmark_models import (
    Candidate,
    CandidatePosition,
    CharacterBenchmarkContext,
    CharacterBenchmarkDefinition,
    CharacterBenchmarkPanelInput,
    CharacterContextGroup,
    CharacterContextName,
    CharacterGenerationSpec,
    CharacterRecipePair,
    CharacterTrial,
)
from video_lab.domain import canonical_json

DEFAULT_CHARACTER_CONTEXTS = (
    CharacterBenchmarkContext(
        ordinal=1,
        name=CharacterContextName.NEUTRAL_FACE_PORTRAIT,
        group=CharacterContextGroup.CORE,
        prompt="Neutral face portrait.",
    ),
    CharacterBenchmarkContext(
        ordinal=2,
        name=CharacterContextName.THREE_QUARTER_PORTRAIT,
        group=CharacterContextGroup.CORE,
        prompt="Three-quarter portrait.",
    ),
    CharacterBenchmarkContext(
        ordinal=3,
        name=CharacterContextName.PROFILE_VIEW,
        group=CharacterContextGroup.CORE,
        prompt="Profile view.",
    ),
    CharacterBenchmarkContext(
        ordinal=4,
        name=CharacterContextName.FULL_BODY_NEUTRAL_POSE,
        group=CharacterContextGroup.CORE,
        prompt="Full-body neutral pose.",
    ),
    CharacterBenchmarkContext(
        ordinal=5,
        name=CharacterContextName.FULL_BODY_ACTION,
        group=CharacterContextGroup.CORE,
        prompt="Full-body action.",
    ),
    CharacterBenchmarkContext(
        ordinal=6,
        name=CharacterContextName.STRONG_EXPRESSION,
        group=CharacterContextGroup.CORE,
        prompt="Strong expression.",
    ),
    CharacterBenchmarkContext(
        ordinal=7,
        name=CharacterContextName.CAMERA_MOTION,
        group=CharacterContextGroup.CORE,
        prompt="Camera motion.",
    ),
    CharacterBenchmarkContext(
        ordinal=8,
        name=CharacterContextName.CHARACTER_OBJECT_INTERACTION,
        group=CharacterContextGroup.CORE,
        prompt="Character-object interaction.",
    ),
    CharacterBenchmarkContext(
        ordinal=9,
        name=CharacterContextName.STRONG_DIRECTIONAL_LIGHTING,
        group=CharacterContextGroup.STRESS,
        prompt="Strong directional lighting.",
    ),
    CharacterBenchmarkContext(
        ordinal=10,
        name=CharacterContextName.PARTIAL_FACIAL_OCCLUSION,
        group=CharacterContextGroup.STRESS,
        prompt="Partial facial occlusion.",
    ),
    CharacterBenchmarkContext(
        ordinal=11,
        name=CharacterContextName.FORESHORTENED_OR_DIFFICULT_ANGLE,
        group=CharacterContextGroup.STRESS,
        prompt="Foreshortened or difficult angle.",
    ),
    CharacterBenchmarkContext(
        ordinal=12,
        name=CharacterContextName.BUSY_ENVIRONMENT_WITH_COMPETING_SUBJECTS,
        group=CharacterContextGroup.STRESS,
        prompt="Busy environment with competing subjects.",
    ),
)


def build_character_panel(
    *,
    character_pack_digest: str,
    stage_2_manifest_digest: str,
    stage_2_evidence_digest: str,
    recipes: CharacterRecipePair,
    panel_input: CharacterBenchmarkPanelInput,
) -> CharacterBenchmarkDefinition:
    payload = {
        "schema_version": "character-benchmark.v1",
        "protocol_version": "character-panel.v1",
        "analysis_version": "character-dirichlet.v1",
        "character_pack_digest": character_pack_digest,
        "stage_2_manifest_digest": stage_2_manifest_digest,
        "stage_2_evidence_digest": stage_2_evidence_digest,
        "recipes": recipes.model_dump(mode="json"),
        "panel_input": panel_input.model_dump(mode="json"),
    }
    return CharacterBenchmarkDefinition(
        id=_content_id("character_benchmark", payload),
        character_pack_digest=character_pack_digest,
        stage_2_manifest_digest=stage_2_manifest_digest,
        stage_2_evidence_digest=stage_2_evidence_digest,
        recipes=recipes,
        panel_input=panel_input,
    )


def schedule_character_trials(
    definition: CharacterBenchmarkDefinition,
) -> tuple[CharacterTrial, ...]:
    primary = tuple(
        _primary_trial(definition, context, seed)
        for context in definition.panel_input.contexts
        for seed in definition.panel_input.seeds
    )
    repeats = _hidden_repeats(definition, primary)
    ordered = _schedule_order(definition, primary, repeats)
    positions = _balanced_candidate_a_positions(definition, ordered)
    return tuple(
        trial.model_copy(
            update={
                "ordinal": ordinal,
                "candidate_a_position": positions[trial.id],
            }
        )
        for ordinal, trial in enumerate(ordered, start=1)
    )


def _primary_trial(
    definition: CharacterBenchmarkDefinition,
    context: CharacterBenchmarkContext,
    seed: int,
) -> CharacterTrial:
    candidates: tuple[Candidate, Candidate] = ("a", "b")
    generation_specs = tuple(
        _generation_spec(definition, context, seed, candidate) for candidate in candidates
    )
    payload = {
        "benchmark_id": definition.id,
        "context": context.model_dump(mode="json"),
        "seed": seed,
        "hidden_repeat": False,
        "generation_spec_ids": [spec.id for spec in generation_specs],
    }
    return CharacterTrial(
        id=_content_id("character_trial", payload),
        benchmark_id=definition.id,
        ordinal=1,
        context=context,
        seed=seed,
        generation_specs=generation_specs,
        candidate_a_position="left",
    )


def _generation_spec(
    definition: CharacterBenchmarkDefinition,
    context: CharacterBenchmarkContext,
    seed: int,
    candidate: Candidate,
) -> CharacterGenerationSpec:
    recipe_digest = (
        definition.recipes.control_digest
        if candidate == "a"
        else definition.recipes.identity_digest
    )
    payload = {
        "benchmark_id": definition.id,
        "context": context.model_dump(mode="json"),
        "seed": seed,
        "candidate": candidate,
        "recipe_digest": recipe_digest,
    }
    return CharacterGenerationSpec(
        id=_content_id("character_generation", payload),
        benchmark_id=definition.id,
        context=context,
        seed=seed,
        candidate=candidate,
        recipe_digest=recipe_digest,
    )


def _hidden_repeats(
    definition: CharacterBenchmarkDefinition,
    primary: tuple[CharacterTrial, ...],
) -> tuple[CharacterTrial, ...]:
    generator = random.Random(_domain_seed("character-repeat-selection.v1", definition))
    selected_contexts = generator.sample(list(definition.panel_input.contexts), 4)
    sources = tuple(
        generator.choice(tuple(trial for trial in primary if trial.context == context))
        for context in selected_contexts
    )
    repeats: list[CharacterTrial] = []
    for source in sources:
        payload = {
            "benchmark_id": definition.id,
            "hidden_repeat": True,
            "reuses_trial_id": source.id,
            "context": source.context.model_dump(mode="json"),
            "seed": source.seed,
        }
        repeats.append(
            CharacterTrial(
                id=_content_id("character_trial", payload),
                benchmark_id=definition.id,
                ordinal=1,
                context=source.context,
                seed=source.seed,
                hidden_repeat=True,
                reuses_trial_id=source.id,
                generation_specs=(),
                candidate_a_position="left",
            )
        )
    return tuple(repeats)


def _schedule_order(
    definition: CharacterBenchmarkDefinition,
    primary: tuple[CharacterTrial, ...],
    repeats: tuple[CharacterTrial, ...],
) -> tuple[CharacterTrial, ...]:
    generator = random.Random(_domain_seed("character-order.v1", definition))
    ordered = list(primary)
    generator.shuffle(ordered)
    pending_repeats = list(repeats)
    generator.shuffle(pending_repeats)
    for repeat in pending_repeats:
        valid_positions = [
            position
            for position in range(len(ordered) + 1)
            if _can_insert_repeat(ordered, position, repeat)
        ]
        ordered.insert(generator.choice(valid_positions), repeat)
    return tuple(ordered)


def _can_insert_repeat(
    ordered: list[CharacterTrial], position: int, repeat: CharacterTrial
) -> bool:
    source_id = repeat.reuses_trial_id
    left_source = _source_id(ordered[position - 1]) if position > 0 else None
    right_source = _source_id(ordered[position]) if position < len(ordered) else None
    return source_id not in {left_source, right_source}


def _source_id(trial: CharacterTrial) -> str:
    return trial.reuses_trial_id or trial.id


def _balanced_candidate_a_positions(
    definition: CharacterBenchmarkDefinition,
    trials: tuple[CharacterTrial, ...],
) -> dict[str, CandidatePosition]:
    generator = random.Random(_domain_seed("character-display.v1", definition))
    positions: dict[str, CandidatePosition] = {}
    left: CandidatePosition = "left"
    right: CandidatePosition = "right"
    primary_by_context = [
        [trial for trial in trials if not trial.hidden_repeat and trial.context == context]
        for context in definition.panel_input.contexts
    ]
    extra_labels: list[CandidatePosition] = [left] * 6 + [right] * 6
    generator.shuffle(extra_labels)
    for group, extra_label in zip(primary_by_context, extra_labels, strict=True):
        labels = [left, right, extra_label]
        generator.shuffle(labels)
        for trial, label in zip(group, labels, strict=True):
            positions[trial.id] = label
    repeats = [trial for trial in trials if trial.hidden_repeat]
    repeat_labels: list[CandidatePosition] = [left, left, right, right]
    generator.shuffle(repeat_labels)
    for trial, label in zip(repeats, repeat_labels, strict=True):
        positions[trial.id] = label
    return positions


def _domain_seed(domain: str, definition: CharacterBenchmarkDefinition) -> int:
    payload = f"{domain}:{definition.panel_input.schedule_seed}:{definition.id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _content_id(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"
