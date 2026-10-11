from __future__ import annotations

from video_lab.character_benchmark_analysis import analyze_character_benchmark
from video_lab.character_benchmark_models import (
    BaselineDisposition,
    CharacterBenchmarkEvidence,
    CharacterBenchmarkPanelInput,
    CharacterRecipePair,
    IdentityChoice,
    IdentityDisposition,
    IdentityObservation,
    SecondaryChoice,
    SecondaryDimension,
    SecondaryObservation,
)
from video_lab.character_benchmark_protocol import (
    DEFAULT_CHARACTER_CONTEXTS,
    build_character_panel,
    schedule_character_trials,
)
from video_lab.domain import canonical_json


def complete_evidence(
    *,
    primary_choices: tuple[IdentityChoice, ...] | None = None,
    repeat_agreements: int = 4,
    secondary_choice: SecondaryChoice = SecondaryChoice.B,
) -> CharacterBenchmarkEvidence:
    definition = build_character_panel(
        character_pack_digest="sha256:pack",
        stage_2_manifest_digest="sha256:manifest",
        stage_2_evidence_digest="sha256:evidence",
        recipes=CharacterRecipePair(
            control_digest="sha256:control",
            identity_digest="sha256:identity",
        ),
        panel_input=CharacterBenchmarkPanelInput(
            contexts=DEFAULT_CHARACTER_CONTEXTS,
            seeds=(101, 202, 303),
            schedule_seed=73021,
        ),
    )
    trials = schedule_character_trials(definition)
    primary = tuple(
        sorted(
            (trial for trial in trials if not trial.hidden_repeat),
            key=lambda trial: (trial.context.ordinal, trial.seed),
        )
    )
    choices = primary_choices or (IdentityChoice.B,) * 36
    assert len(choices) == 36
    by_trial_id = {trial.id: choice for trial, choice in zip(primary, choices, strict=True)}
    repeats_seen = 0
    identity: list[IdentityObservation] = []
    for trial in trials:
        choice = (
            by_trial_id[trial.id] if not trial.hidden_repeat else by_trial_id[trial.reuses_trial_id]
        )
        if trial.hidden_repeat:
            if repeats_seen >= repeat_agreements:
                choice = IdentityChoice.A if choice is not IdentityChoice.A else IdentityChoice.B
            repeats_seen += 1
        identity.append(
            IdentityObservation(
                benchmark_id=definition.id,
                trial_id=trial.id,
                choice=choice,
            )
        )
    secondary = tuple(
        SecondaryObservation(
            benchmark_id=definition.id,
            trial_id=observation.trial_id,
            dimension=dimension,
            choice=secondary_choice,
        )
        for observation in identity
        if observation.choice is not IdentityChoice.NEITHER
        for dimension in SecondaryDimension
    )
    return CharacterBenchmarkEvidence(
        definition=definition,
        trials=trials,
        identity_observations=tuple(identity),
        secondary_observations=secondary,
    )


def test_identity_recipe_and_control_qualification_remain_identity_driven() -> None:
    identity_result = analyze_character_benchmark(complete_evidence())
    control_result = analyze_character_benchmark(
        complete_evidence(
            primary_choices=(IdentityChoice.A,) * 36,
            secondary_choice=SecondaryChoice.B,
        )
    )

    assert identity_result.baseline_disposition is BaselineDisposition.BASELINE_ESTABLISHED
    assert identity_result.identity_disposition is IdentityDisposition.IDENTITY_RECIPE_QUALIFIED
    assert identity_result.posterior.counts.b == 36
    assert identity_result.posterior.identity_recipe_superiority_probability == 1.0
    assert control_result.identity_disposition is IdentityDisposition.CONTROL_QUALIFIED
    assert control_result.posterior.counts.a == 36
    assert control_result.secondary.prompt_adherence.counts.b == 36


def test_four_way_posterior_is_seeded_versioned_and_byte_deterministic() -> None:
    choices = (
        (IdentityChoice.A,) * 10
        + (IdentityChoice.B,) * 12
        + (IdentityChoice.BOTH,) * 7
        + (IdentityChoice.NEITHER,) * 7
    )
    evidence = complete_evidence(primary_choices=choices)

    first = analyze_character_benchmark(evidence)
    second = analyze_character_benchmark(evidence)

    assert first.identity_disposition is IdentityDisposition.INCONCLUSIVE
    assert first.analysis_version == "character-dirichlet.v1"
    assert first.posterior.control_superiority_probability == 0.337799072265625
    assert first.posterior.identity_recipe_superiority_probability == 0.662200927734375
    assert first.posterior.counts.model_dump() == {"a": 10, "b": 12, "both": 7, "neither": 7}
    assert canonical_json(first) == canonical_json(second)


def test_both_and_neither_remain_distinct_identity_outcomes() -> None:
    tie_heavy = analyze_character_benchmark(
        complete_evidence(
            primary_choices=(IdentityChoice.A,) * 3
            + (IdentityChoice.B,) * 3
            + (IdentityChoice.BOTH,) * 30
        )
    )
    evidence = complete_evidence()
    changed_identity = tuple(
        observation.model_copy(update={"choice": IdentityChoice.NEITHER})
        if index < 20
        else observation
        for index, observation in enumerate(evidence.identity_observations)
    )
    identity_failure = analyze_character_benchmark(
        evidence.model_copy(
            update={
                "identity_observations": changed_identity,
            }
        )
    )

    assert tie_heavy.posterior.counts.both == 30
    assert tie_heavy.posterior.counts.neither == 0
    assert tie_heavy.both_rate > tie_heavy.neither_rate
    assert identity_failure.neither_rate == identity_failure.posterior.counts.neither / 36
    assert identity_failure.posterior.counts.both == 0
    assert identity_failure.secondary.prompt_adherence.observation_count < 36


def test_context_reversal_is_flagged_at_the_80_percent_threshold() -> None:
    choices = (IdentityChoice.A,) * 3 + (IdentityChoice.B,) * 33

    result = analyze_character_benchmark(complete_evidence(primary_choices=choices))

    assert result.identity_disposition is IdentityDisposition.IDENTITY_RECIPE_QUALIFIED
    assert len(result.context_reversals) == 1
    assert result.context_reversals[0].context.ordinal == 1
    assert result.context_reversals[0].disfavored_recipe == "control"
    assert result.context_reversals[0].superiority_probability >= 0.80


def test_repeats_only_measure_reliability_and_are_excluded_from_inference() -> None:
    evidence = complete_evidence(primary_choices=(IdentityChoice.A,) * 36)
    repeat_ids = {trial.id for trial in evidence.trials if trial.hidden_repeat}
    repeat_losses = tuple(
        observation.model_copy(update={"choice": IdentityChoice.B})
        if observation.trial_id in repeat_ids
        else observation
        for observation in evidence.identity_observations
    )

    result = analyze_character_benchmark(
        evidence.model_copy(update={"identity_observations": repeat_losses})
    )

    assert result.posterior.counts.a == 36
    assert result.posterior.counts.b == 0
    assert result.repeat_agreements == 0
    assert result.baseline_disposition is BaselineDisposition.EVALUATOR_UNRELIABLE


def test_repeat_identity_and_secondary_changes_leave_primary_projection_byte_identical() -> None:
    evidence = complete_evidence()
    repeat_ids = tuple(trial.id for trial in evidence.trials if trial.hidden_repeat)
    repeat_choices = dict(
        zip(
            repeat_ids,
            (
                IdentityChoice.A,
                IdentityChoice.B,
                IdentityChoice.BOTH,
                IdentityChoice.NEITHER,
            ),
            strict=True,
        )
    )
    changed = evidence.model_copy(
        update={
            "identity_observations": tuple(
                observation.model_copy(update={"choice": repeat_choices[observation.trial_id]})
                if observation.trial_id in repeat_choices
                else observation
                for observation in evidence.identity_observations
            ),
            "secondary_observations": tuple(
                observation.model_copy(update={"choice": SecondaryChoice.TIE})
                if observation.trial_id in repeat_choices
                else observation
                for observation in evidence.secondary_observations
            ),
        }
    )

    reliable = analyze_character_benchmark(evidence)
    unreliable = analyze_character_benchmark(changed)
    reliable_primary = reliable.model_dump(
        mode="json", exclude={"baseline_disposition", "repeat_agreements"}
    )
    unreliable_primary = unreliable.model_dump(
        mode="json", exclude={"baseline_disposition", "repeat_agreements"}
    )

    assert reliable.repeat_agreements == 4
    assert unreliable.repeat_agreements == 1
    assert reliable.baseline_disposition is BaselineDisposition.BASELINE_ESTABLISHED
    assert unreliable.baseline_disposition is BaselineDisposition.EVALUATOR_UNRELIABLE
    assert canonical_json(reliable_primary) == canonical_json(unreliable_primary)


def test_evaluator_unreliable_and_incomplete_evidence_are_distinct() -> None:
    unreliable = analyze_character_benchmark(complete_evidence(repeat_agreements=2))
    evidence = complete_evidence()
    primary_id = next(trial.id for trial in evidence.trials if not trial.hidden_repeat)
    incomplete = analyze_character_benchmark(
        evidence.model_copy(
            update={
                "identity_observations": tuple(
                    observation
                    for observation in evidence.identity_observations
                    if observation.trial_id != primary_id
                )
            }
        )
    )
    bad_provenance = analyze_character_benchmark(
        evidence.model_copy(update={"provenance_complete": False})
    )

    assert unreliable.baseline_disposition is BaselineDisposition.EVALUATOR_UNRELIABLE
    assert unreliable.repeat_agreements == 2
    assert incomplete.baseline_disposition is BaselineDisposition.INSUFFICIENT_EVIDENCE
    assert incomplete.identity_disposition is IdentityDisposition.INCONCLUSIVE
    assert bad_provenance.baseline_disposition is BaselineDisposition.INSUFFICIENT_EVIDENCE
