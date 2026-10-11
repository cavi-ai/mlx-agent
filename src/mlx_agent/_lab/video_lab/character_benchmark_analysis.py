from __future__ import annotations

from collections import Counter

import numpy as np

from video_lab.character_benchmark_models import (
    CHARACTER_POSTERIOR_DRAWS,
    CHARACTER_POSTERIOR_SEED,
    BaselineDisposition,
    CharacterBenchmarkEvidence,
    CharacterBenchmarkResult,
    CharacterContextGroup,
    ContextIdentityResult,
    ContextReversal,
    GroupIdentityResult,
    IdentityChoice,
    IdentityCounts,
    IdentityDisposition,
    IdentityObservation,
    IdentityPosterior,
    SecondaryChoice,
    SecondaryCounts,
    SecondaryDimension,
    SecondaryDimensionResult,
    SecondaryObservation,
    SecondarySummary,
)
from video_lab.character_benchmark_protocol import schedule_character_trials

_IDENTITY_QUALIFICATION_THRESHOLD = 0.90
_CONTEXT_REVERSAL_THRESHOLD = 0.80
_MINIMUM_REPEAT_AGREEMENTS = 3


def analyze_character_benchmark(
    evidence: CharacterBenchmarkEvidence,
) -> CharacterBenchmarkResult:
    complete = _has_complete_integrity(evidence)
    identity_by_trial = _identity_by_trial(evidence.identity_observations)
    primary_trials = tuple(trial for trial in evidence.trials if not trial.hidden_repeat)
    primary_observations = tuple(
        identity_by_trial[trial.id] for trial in primary_trials if trial.id in identity_by_trial
    )
    posterior = identity_posterior(_identity_counts(primary_observations))
    identity_disposition = (
        _identity_disposition(posterior) if complete else IdentityDisposition.INCONCLUSIVE
    )
    repeat_agreements = _repeat_agreements(evidence, identity_by_trial)
    if not complete:
        baseline_disposition = BaselineDisposition.INSUFFICIENT_EVIDENCE
    elif repeat_agreements < _MINIMUM_REPEAT_AGREEMENTS:
        baseline_disposition = BaselineDisposition.EVALUATOR_UNRELIABLE
    else:
        baseline_disposition = BaselineDisposition.BASELINE_ESTABLISHED

    context_results = _context_results(evidence, identity_by_trial)
    group_results = _group_results(context_results)
    primary_counts = posterior.counts
    denominator = primary_counts.total
    return CharacterBenchmarkResult(
        benchmark_id=evidence.definition.id,
        baseline_disposition=baseline_disposition,
        identity_disposition=identity_disposition,
        posterior=posterior,
        both_rate=primary_counts.both / denominator if denominator else 0.0,
        neither_rate=primary_counts.neither / denominator if denominator else 0.0,
        context_results=context_results,
        group_results=group_results,
        context_reversals=_context_reversals(context_results, identity_disposition),
        secondary=_secondary_summary(evidence, identity_by_trial),
        repeat_agreements=repeat_agreements,
        identity_break_contexts=tuple(
            item.context.name for item in context_results if item.posterior.counts.neither > 0
        ),
    )


def identity_posterior(counts: IdentityCounts) -> IdentityPosterior:
    samples = np.random.default_rng(CHARACTER_POSTERIOR_SEED).dirichlet(
        (counts.a + 1, counts.b + 1, counts.both + 1, counts.neither + 1),
        CHARACTER_POSTERIOR_DRAWS,
    )
    return IdentityPosterior(
        counts=counts,
        control_superiority_probability=float(np.mean(samples[:, 0] > samples[:, 1])),
        identity_recipe_superiority_probability=float(np.mean(samples[:, 1] > samples[:, 0])),
    )


def _has_complete_integrity(evidence: CharacterBenchmarkEvidence) -> bool:
    if not evidence.media_integrity_complete or not evidence.provenance_complete:
        return False
    expected_trials = schedule_character_trials(evidence.definition)
    if evidence.trials != expected_trials:
        return False
    if len(evidence.trials) != 40:
        return False
    if sum(not trial.hidden_repeat for trial in evidence.trials) != 36:
        return False
    if sum(trial.hidden_repeat for trial in evidence.trials) != 4:
        return False
    trial_ids = {trial.id for trial in evidence.trials}
    if len(trial_ids) != len(evidence.trials):
        return False
    observations = evidence.identity_observations
    if len(observations) != len(evidence.trials):
        return False
    if {observation.benchmark_id for observation in observations} != {evidence.definition.id}:
        return False
    return Counter(observation.trial_id for observation in observations) == Counter(trial_ids)


def _identity_by_trial(
    observations: tuple[IdentityObservation, ...],
) -> dict[str, IdentityObservation]:
    counts = Counter(observation.trial_id for observation in observations)
    return {
        observation.trial_id: observation
        for observation in observations
        if counts[observation.trial_id] == 1
    }


def _identity_counts(observations: tuple[IdentityObservation, ...]) -> IdentityCounts:
    counts = Counter(observation.choice for observation in observations)
    return IdentityCounts(
        a=counts[IdentityChoice.A],
        b=counts[IdentityChoice.B],
        both=counts[IdentityChoice.BOTH],
        neither=counts[IdentityChoice.NEITHER],
    )


def _identity_disposition(posterior: IdentityPosterior) -> IdentityDisposition:
    if posterior.identity_recipe_superiority_probability >= _IDENTITY_QUALIFICATION_THRESHOLD:
        return IdentityDisposition.IDENTITY_RECIPE_QUALIFIED
    if posterior.control_superiority_probability >= _IDENTITY_QUALIFICATION_THRESHOLD:
        return IdentityDisposition.CONTROL_QUALIFIED
    return IdentityDisposition.INCONCLUSIVE


def _repeat_agreements(
    evidence: CharacterBenchmarkEvidence,
    identity_by_trial: dict[str, IdentityObservation],
) -> int:
    agreements = 0
    for trial in evidence.trials:
        if not trial.hidden_repeat or trial.reuses_trial_id is None:
            continue
        repeat = identity_by_trial.get(trial.id)
        source = identity_by_trial.get(trial.reuses_trial_id)
        if repeat is not None and source is not None and repeat.choice is source.choice:
            agreements += 1
    return agreements


def _context_results(
    evidence: CharacterBenchmarkEvidence,
    identity_by_trial: dict[str, IdentityObservation],
) -> tuple[ContextIdentityResult, ...]:
    results: list[ContextIdentityResult] = []
    for context in evidence.definition.panel_input.contexts:
        observations = tuple(
            identity_by_trial[trial.id]
            for trial in evidence.trials
            if not trial.hidden_repeat
            and trial.context == context
            and trial.id in identity_by_trial
        )
        counts = _identity_counts(observations)
        results.append(
            ContextIdentityResult(
                context=context,
                posterior=identity_posterior(counts),
                both_rate=counts.both / counts.total if counts.total else 0.0,
                neither_rate=counts.neither / counts.total if counts.total else 0.0,
            )
        )
    return tuple(results)


def _group_results(
    contexts: tuple[ContextIdentityResult, ...],
) -> tuple[GroupIdentityResult, ...]:
    results: list[GroupIdentityResult] = []
    for group in CharacterContextGroup:
        grouped = tuple(item for item in contexts if item.context.group is group)
        counts = IdentityCounts(
            a=sum(item.posterior.counts.a for item in grouped),
            b=sum(item.posterior.counts.b for item in grouped),
            both=sum(item.posterior.counts.both for item in grouped),
            neither=sum(item.posterior.counts.neither for item in grouped),
        )
        results.append(
            GroupIdentityResult(
                group=group,
                posterior=identity_posterior(counts),
                both_rate=counts.both / counts.total if counts.total else 0.0,
                neither_rate=counts.neither / counts.total if counts.total else 0.0,
            )
        )
    return tuple(results)


def _context_reversals(
    contexts: tuple[ContextIdentityResult, ...],
    identity_disposition: IdentityDisposition,
) -> tuple[ContextReversal, ...]:
    if identity_disposition is IdentityDisposition.IDENTITY_RECIPE_QUALIFIED:
        return tuple(
            ContextReversal(
                context=item.context,
                disfavored_recipe="control",
                superiority_probability=item.posterior.control_superiority_probability,
            )
            for item in contexts
            if item.posterior.control_superiority_probability >= _CONTEXT_REVERSAL_THRESHOLD
        )
    if identity_disposition is IdentityDisposition.CONTROL_QUALIFIED:
        return tuple(
            ContextReversal(
                context=item.context,
                disfavored_recipe="identity",
                superiority_probability=item.posterior.identity_recipe_superiority_probability,
            )
            for item in contexts
            if item.posterior.identity_recipe_superiority_probability >= _CONTEXT_REVERSAL_THRESHOLD
        )
    return ()


def _secondary_summary(
    evidence: CharacterBenchmarkEvidence,
    identity_by_trial: dict[str, IdentityObservation],
) -> SecondarySummary:
    primary_trial_ids = {trial.id for trial in evidence.trials if not trial.hidden_repeat}
    eligible_ids = {
        trial_id
        for trial_id, observation in identity_by_trial.items()
        if trial_id in primary_trial_ids and observation.choice is not IdentityChoice.NEITHER
    }
    supplied = tuple(
        observation
        for observation in evidence.secondary_observations
        if observation.benchmark_id == evidence.definition.id
        and observation.trial_id in eligible_ids
    )
    key_counts = Counter((item.trial_id, item.dimension) for item in supplied)
    valid = tuple(item for item in supplied if key_counts[(item.trial_id, item.dimension)] == 1)
    by_dimension = {
        dimension: _secondary_dimension_result(
            tuple(item for item in valid if item.dimension is dimension)
        )
        for dimension in SecondaryDimension
    }
    return SecondarySummary(
        prompt_adherence=by_dimension[SecondaryDimension.PROMPT_ADHERENCE],
        motion_quality=by_dimension[SecondaryDimension.MOTION_QUALITY],
        temporal_consistency=by_dimension[SecondaryDimension.TEMPORAL_CONSISTENCY],
        visual_quality=by_dimension[SecondaryDimension.VISUAL_QUALITY],
    )


def _secondary_dimension_result(
    observations: tuple[SecondaryObservation, ...],
) -> SecondaryDimensionResult:
    choice_counts = Counter(item.choice for item in observations)
    counts = SecondaryCounts(
        a=choice_counts[SecondaryChoice.A],
        b=choice_counts[SecondaryChoice.B],
        tie=choice_counts[SecondaryChoice.TIE],
    )
    samples = np.random.default_rng(CHARACTER_POSTERIOR_SEED).dirichlet(
        (counts.a + 1, counts.b + 1, counts.tie + 1),
        CHARACTER_POSTERIOR_DRAWS,
    )
    return SecondaryDimensionResult(
        counts=counts,
        observation_count=counts.total,
        control_superiority_probability=float(np.mean(samples[:, 0] > samples[:, 1])),
        identity_recipe_superiority_probability=float(np.mean(samples[:, 1] > samples[:, 0])),
    )
