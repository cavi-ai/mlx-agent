from __future__ import annotations

from datetime import UTC, datetime

from video_lab.validation_analysis import analyze_validation
from video_lab.validation_models import (
    ValidationArchetype,
    ValidationChoice,
    ValidationDisposition,
    ValidationJudgment,
    ValidationReason,
)


def judgment(
    *,
    ordinal: int,
    archetype: ValidationArchetype,
    challenger: float,
    seed: int = 101,
    choice: ValidationChoice = ValidationChoice.A,
    hidden_repeat: bool = False,
) -> ValidationJudgment:
    return ValidationJudgment(
        campaign_id="validation_1",
        trial_id=f"trial_{ordinal}",
        ordinal=ordinal,
        archetype=archetype,
        challenger=challenger,
        seed=seed,
        hidden_repeat=hidden_repeat,
        winner_display_position="left",
        choice=choice,
        created_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )


def qualified_judgments() -> tuple[ValidationJudgment, ...]:
    return tuple(
        judgment(ordinal=ordinal, archetype=archetype, challenger=challenger, seed=seed)
        for ordinal, (archetype, challenger, seed) in enumerate(
            (
                (archetype, challenger, seed)
                for archetype in ValidationArchetype
                for challenger in (2.0, 4.0, 5.0)
                for seed in (101, 202, 303)
            ),
            start=1,
        )
    )


def test_analysis_qualifies_only_when_all_gates_pass() -> None:
    result = analyze_validation(qualified_judgments(), repeat_agreements=4)

    assert result.disposition is ValidationDisposition.QUALIFIED
    assert result.reason is ValidationReason.ALL_GATES_PASSED
    assert all(item.superiority_probability >= 0.90 for item in result.challengers)
    assert all(item.counts.wins == 12 for item in result.challengers)


def test_repeat_unreliability_is_inconclusive_not_rejected() -> None:
    result = analyze_validation(qualified_judgments(), repeat_agreements=2)

    assert result.disposition is ValidationDisposition.INCONCLUSIVE
    assert result.reason is ValidationReason.REPEAT_RELIABILITY_FAILED


def test_challenger_rejection_overrides_repeat_reliability() -> None:
    judgments = tuple(
        item.model_copy(update={"choice": ValidationChoice.B}) if item.challenger == 5.0 else item
        for item in qualified_judgments()
    )

    result = analyze_validation(judgments, repeat_agreements=2)

    assert result.disposition is ValidationDisposition.NOT_QUALIFIED
    assert result.reason is ValidationReason.CHALLENGER_REJECTION


def test_hidden_repeats_are_excluded_from_quality_inference() -> None:
    primary = qualified_judgments()
    repeat = judgment(
        ordinal=37,
        archetype=ValidationArchetype.PORTRAIT_MOTION,
        challenger=2.0,
        choice=ValidationChoice.B,
        hidden_repeat=True,
    )

    result = analyze_validation(primary + (repeat,), repeat_agreements=4)

    lower = next(item for item in result.challengers if item.challenger == 2.0)
    assert lower.counts.wins == 12
    assert lower.counts.losses == 0


def test_duplicated_twelve_row_primary_subset_is_inconclusive() -> None:
    duplicated_subset = tuple(
        judgment(
            ordinal=ordinal,
            archetype=ValidationArchetype.PORTRAIT_MOTION,
            challenger=2.0,
            seed=101,
        )
        for ordinal in range(1, 13)
    )

    result = analyze_validation(duplicated_subset, repeat_agreements=4)

    assert result.disposition is ValidationDisposition.INCONCLUSIVE
    assert result.reason is ValidationReason.INCOMPLETE_PRIMARY_EVIDENCE


def test_missing_or_duplicate_primary_context_is_inconclusive() -> None:
    complete = qualified_judgments()
    invalid_sets = (
        tuple(item for item in complete if item.challenger != 5.0),
        tuple(
            item
            for item in complete
            if item.archetype is not ValidationArchetype.ENVIRONMENT_INTERACTION
        ),
        tuple(item for item in complete if item.seed != 303),
        complete[:-1] + (complete[-1].model_copy(update={"seed": 101}),),
        tuple(
            item.model_copy(update={"seed": 404})
            if (
                item.challenger == 5.0
                and item.archetype is ValidationArchetype.PORTRAIT_MOTION
                and item.seed == 303
            )
            else item
            for item in complete
        ),
    )

    for judgments in invalid_sets:
        result = analyze_validation(judgments, repeat_agreements=4)

        assert result.disposition is ValidationDisposition.INCONCLUSIVE
        assert result.reason is ValidationReason.INCOMPLETE_PRIMARY_EVIDENCE
