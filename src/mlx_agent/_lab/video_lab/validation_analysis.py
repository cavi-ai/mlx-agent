from __future__ import annotations

import numpy as np

from video_lab.validation_models import (
    ArchetypeReversal,
    ChallengerPosterior,
    OutcomeCounts,
    ValidationArchetype,
    ValidationDisposition,
    ValidationJudgment,
    ValidationOutcome,
    ValidationReason,
    ValidationResult,
)

_QUALIFICATION_THRESHOLD = 0.90
_REVERSAL_THRESHOLD = 0.80
_MINIMUM_REPEAT_AGREEMENTS = 3


def challenger_posterior(
    counts: OutcomeCounts, *, draws: int = 65536, seed: int = 73021
) -> ChallengerPosterior:
    if draws <= 0:
        raise ValueError("posterior draws must be positive")
    samples = np.random.default_rng(seed).dirichlet(
        (counts.wins + 1, counts.ties + 1, counts.losses + 1), draws
    )
    return ChallengerPosterior(
        counts=counts,
        superiority_probability=float(np.mean(samples[:, 0] > samples[:, 2])),
        rejection_probability=float(np.mean(samples[:, 2] > samples[:, 0])),
    )


def analyze_validation(
    judgments: tuple[ValidationJudgment, ...], repeat_agreements: int
) -> ValidationResult:
    if not 0 <= repeat_agreements <= 4:
        raise ValueError("repeat agreements must be between zero and four")
    campaign_id = _campaign_id(judgments)
    primary = tuple(judgment for judgment in judgments if not judgment.hidden_repeat)
    if not _has_complete_primary_evidence(primary):
        return ValidationResult(
            campaign_id=campaign_id,
            disposition=ValidationDisposition.INCONCLUSIVE,
            reason=ValidationReason.INCOMPLETE_PRIMARY_EVIDENCE,
            challengers=(),
            archetype_reversals=(),
            repeat_agreements=repeat_agreements,
        )
    challengers = tuple(
        challenger_posterior(_outcome_counts(group)).model_copy(update={"challenger": challenger})
        for challenger in sorted({judgment.challenger for judgment in primary})
        if (group := tuple(judgment for judgment in primary if judgment.challenger == challenger))
    )
    archetype_reversals = tuple(
        _archetype_reversal(
            archetype, tuple(judgment for judgment in primary if judgment.archetype is archetype)
        )
        for archetype in ValidationArchetype
        if any(judgment.archetype is archetype for judgment in primary)
    )
    disposition, reason = _decision(challengers, archetype_reversals, repeat_agreements)
    return ValidationResult(
        campaign_id=campaign_id,
        disposition=disposition,
        reason=reason,
        challengers=challengers,
        archetype_reversals=archetype_reversals,
        repeat_agreements=repeat_agreements,
    )


def _campaign_id(judgments: tuple[ValidationJudgment, ...]) -> str:
    campaign_ids = {judgment.campaign_id for judgment in judgments}
    if len(campaign_ids) != 1:
        raise ValueError("validation analysis requires judgments from exactly one campaign")
    return next(iter(campaign_ids))


def _outcome_counts(judgments: tuple[ValidationJudgment, ...]) -> OutcomeCounts:
    return OutcomeCounts(
        wins=sum(judgment.outcome is ValidationOutcome.WIN for judgment in judgments),
        ties=sum(judgment.outcome is ValidationOutcome.TIE for judgment in judgments),
        losses=sum(judgment.outcome is ValidationOutcome.LOSS for judgment in judgments),
    )


def _has_complete_primary_evidence(judgments: tuple[ValidationJudgment, ...]) -> bool:
    if len(judgments) != 36:
        return False
    challengers = {judgment.challenger for judgment in judgments}
    if len(challengers) != 3:
        return False
    if {judgment.archetype for judgment in judgments} != set(ValidationArchetype):
        return False
    contexts = {(judgment.challenger, judgment.archetype, judgment.seed) for judgment in judgments}
    if len(contexts) != len(judgments):
        return False
    if not all(
        sum(
            judgment.challenger == challenger and judgment.archetype is archetype
            for judgment in judgments
        )
        == 3
        for challenger in challengers
        for archetype in ValidationArchetype
    ):
        return False
    return all(
        len(
            {
                frozenset(
                    judgment.seed
                    for judgment in judgments
                    if judgment.challenger == challenger and judgment.archetype is archetype
                )
                for challenger in challengers
            }
        )
        == 1
        for archetype in ValidationArchetype
    )


def _archetype_reversal(
    archetype: ValidationArchetype, judgments: tuple[ValidationJudgment, ...]
) -> ArchetypeReversal:
    posterior = challenger_posterior(_outcome_counts(judgments))
    return ArchetypeReversal(
        archetype=archetype,
        counts=posterior.counts,
        reversal_probability=posterior.rejection_probability,
    )


def _decision(
    challengers: tuple[ChallengerPosterior, ...],
    archetype_reversals: tuple[ArchetypeReversal, ...],
    repeat_agreements: int,
) -> tuple[ValidationDisposition, ValidationReason]:
    if any(item.rejection_probability >= _QUALIFICATION_THRESHOLD for item in challengers):
        return ValidationDisposition.NOT_QUALIFIED, ValidationReason.CHALLENGER_REJECTION
    if any(item.reversal_probability >= _REVERSAL_THRESHOLD for item in archetype_reversals):
        return ValidationDisposition.NOT_QUALIFIED, ValidationReason.ARCHETYPE_REVERSAL
    if repeat_agreements < _MINIMUM_REPEAT_AGREEMENTS:
        return ValidationDisposition.INCONCLUSIVE, ValidationReason.REPEAT_RELIABILITY_FAILED
    if challengers and all(
        item.superiority_probability >= _QUALIFICATION_THRESHOLD for item in challengers
    ):
        return ValidationDisposition.QUALIFIED, ValidationReason.ALL_GATES_PASSED
    return ValidationDisposition.INCONCLUSIVE, ValidationReason.INSUFFICIENT_PRIMARY_EVIDENCE
