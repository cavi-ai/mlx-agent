from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

import video_lab.preference_optimizer as optimizer
from video_lab.preference_optimizer import (
    AcquisitionScore,
    CandidatePair,
    OptimizerConfig,
    OptimizerDecision,
    PosteriorSummary,
    PreferenceObservation,
    PreferenceOutcome,
    SearchDisposition,
    StopReason,
    build_guidance_grid,
    fit_posterior,
    optimize_preference,
    outcome_probabilities,
)


def test_default_grid_and_initial_pair_match_approved_animegen_contract() -> None:
    config = OptimizerConfig()

    assert build_guidance_grid(config) == tuple(1.0 + 0.25 * index for index in range(17))
    assert config.initial_pair == CandidatePair(value_a=2.0, value_b=4.0)


def test_config_rejects_unrepresentable_bounds() -> None:
    with pytest.raises(ValidationError):
        OptimizerConfig(lower_bound=1.0, upper_bound=5.1, resolution=0.25)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("schema_version", "preference-optimizer-config.v2"),
        ("optimizer_version", "preference-gp-laplace.v2"),
        ("minimum_comparisons", 3),
        ("minimum_comparisons", 5),
        ("maximum_comparisons", 9),
        ("maximum_comparisons", 11),
    ),
)
def test_v1_config_rejects_unsupported_versions_and_comparison_budgets(
    field_name: str, value: str | int
) -> None:
    with pytest.raises(ValidationError):
        OptimizerConfig(**{field_name: value})


@pytest.mark.parametrize("resolution", (5e-324, 0.125))
def test_grid_validation_rejects_non_finite_or_excessive_cardinality(
    resolution: float,
) -> None:
    with pytest.raises(ValidationError, match="grid"):
        OptimizerConfig(resolution=resolution)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound"),
    ((0.0, 5.0), (1.0, 6.0)),
)
def test_config_rejects_bounds_outside_the_adapter_safe_range(
    lower_bound: float, upper_bound: float
) -> None:
    with pytest.raises(ValidationError, match="adapter-safe"):
        OptimizerConfig(lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.mark.parametrize("field_name", ("value_a", "value_b"))
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_candidate_pair_rejects_non_finite_values(field_name: str, value: float) -> None:
    values = {"value_a": 4.0, "value_b": 4.0}
    values[field_name] = value

    with pytest.raises(ValidationError, match="finite"):
        CandidatePair(**values)


@pytest.mark.parametrize(
    "field_name",
    ("lower_bound", "upper_bound", "resolution"),
)
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_config_rejects_non_finite_grid_values(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        OptimizerConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    (
        "kernel_amplitude",
        "kernel_length_scale",
        "jitter",
        "tie_threshold",
        "concentration_width",
        "information_threshold_bits",
    ),
)
def test_config_rejects_positive_infinity_for_all_floating_inputs(field_name: str) -> None:
    with pytest.raises(ValidationError, match="finite"):
        OptimizerConfig(**{field_name: math.inf})


@pytest.mark.parametrize(
    ("field_name", "value"),
    (("kernel_amplitude", math.nan), ("information_threshold_bits", -math.inf)),
)
def test_config_rejects_representative_other_non_finite_floating_inputs(
    field_name: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        OptimizerConfig(**{field_name: value})


def test_observation_requires_distinct_on_grid_values() -> None:
    config = OptimizerConfig()
    valid = PreferenceObservation(
        value_a=2.0,
        value_b=4.0,
        outcome=PreferenceOutcome.A,
    )

    assert valid.outcome is PreferenceOutcome.A
    with pytest.raises(ValidationError):
        PreferenceObservation(value_a=2.0, value_b=2.0, outcome=PreferenceOutcome.TIE)
    with pytest.raises(ValueError, match="grid"):
        config.validate_observations(
            (PreferenceObservation(value_a=2.1, value_b=4.0, outcome=PreferenceOutcome.B),)
        )


def test_ordered_logistic_probabilities_are_normalized_and_symmetric() -> None:
    config = OptimizerConfig()

    at_zero = outcome_probabilities(config, difference_mean=0.0, difference_variance=0.0)
    positive = outcome_probabilities(config, difference_mean=2.0, difference_variance=0.0)
    negative = outcome_probabilities(config, difference_mean=-2.0, difference_variance=0.0)

    assert sum(at_zero.values()) == pytest.approx(1.0)
    assert at_zero[PreferenceOutcome.A] == pytest.approx(at_zero[PreferenceOutcome.B])
    assert positive[PreferenceOutcome.A] == pytest.approx(negative[PreferenceOutcome.B])
    assert positive[PreferenceOutcome.A] > positive[PreferenceOutcome.TIE]


def test_posterior_moves_toward_repeated_preference_and_is_positive_semidefinite() -> None:
    config = OptimizerConfig(posterior_draws=512)
    observations = (
        PreferenceObservation(value_a=2.0, value_b=4.0, outcome=PreferenceOutcome.B),
        PreferenceObservation(value_a=2.5, value_b=4.0, outcome=PreferenceOutcome.B),
        PreferenceObservation(value_a=3.0, value_b=4.0, outcome=PreferenceOutcome.B),
    )

    posterior = fit_posterior(config, observations)
    covariance = np.asarray(posterior.covariance)

    assert (
        posterior.mean[build_guidance_grid(config).index(4.0)]
        > posterior.mean[build_guidance_grid(config).index(2.0)]
    )
    assert np.linalg.eigvalsh(covariance).min() >= -1e-8
    assert posterior.content_digest.startswith("sha256:")


def test_posterior_sampling_is_bitwise_replayable() -> None:
    config = OptimizerConfig(posterior_draws=1024)
    observations = (
        PreferenceObservation(value_a=2.0, value_b=4.0, outcome=PreferenceOutcome.B),
        PreferenceObservation(value_a=3.0, value_b=4.0, outcome=PreferenceOutcome.TIE),
    )

    first = fit_posterior(config, observations)
    second = fit_posterior(config, observations)

    assert first == second
    assert sum(first.argmax_probability) == pytest.approx(1.0)


def test_shortest_interval_uses_lower_bound_then_upper_bound_tie_breaking() -> None:
    from video_lab.preference_optimizer import shortest_mass_interval

    interval, mass = shortest_mass_interval(
        grid=(1.0, 1.25, 1.5, 1.75),
        probability=(0.45, 0.05, 0.45, 0.05),
        target_mass=0.50,
    )

    assert interval == (1.0, 1.25)
    assert mass == pytest.approx(0.50)


def test_empty_study_selects_approved_initial_pair() -> None:
    decision = optimize_preference(OptimizerConfig(posterior_draws=512), ())

    assert decision.selected_pair == CandidatePair(value_a=2.0, value_b=4.0)
    assert decision.stopping.reason is StopReason.CONTINUE
    assert decision.stopping.disposition is None
    assert decision.candidate_scores == ()


def test_acquisition_ledger_keeps_positive_information_repeated_pair() -> None:
    config = OptimizerConfig(posterior_draws=256)
    observations = (PreferenceObservation(value_a=2.0, value_b=4.0, outcome=PreferenceOutcome.B),)

    decision = optimize_preference(config, observations)
    repeated = next(
        score
        for score in decision.candidate_scores
        if score.pair == CandidatePair(value_a=2.0, value_b=4.0)
    )

    assert repeated.expected_information_bits > 0.0
    assert decision.selected_pair == decision.candidate_scores[0].pair
    assert tuple(score.expected_information_bits for score in decision.candidate_scores) == tuple(
        sorted(
            (score.expected_information_bits for score in decision.candidate_scores),
            reverse=True,
        )
    )


def _ten_distinct_observations() -> tuple[PreferenceObservation, ...]:
    values = build_guidance_grid(OptimizerConfig())
    return tuple(
        PreferenceObservation(
            value_a=values[index],
            value_b=values[index + 1],
            outcome=PreferenceOutcome.A,
        )
        for index in range(10)
    )


def _controlled_posterior(
    config: OptimizerConfig,
    *,
    interval: tuple[float, float] = (2.5, 3.0),
    mass: float = 0.95,
) -> PosteriorSummary:
    grid = build_guidance_grid(config)
    probability = tuple(1.0 if value == 2.75 else 0.0 for value in grid)
    return PosteriorSummary(
        grid=grid,
        mean=tuple(0.0 for _value in grid),
        covariance=tuple(tuple(0.0 for _column in grid) for _row in grid),
        argmax_probability=probability,
        recommendation=2.75,
        concentration_interval=interval,
        concentration_probability=mass,
        content_digest="sha256:controlled",
    )


def test_hard_cap_preserves_ledger_and_reports_budget_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OptimizerConfig(posterior_draws=256)
    observations = _ten_distinct_observations()
    monkeypatch.setattr(optimizer, "_expected_information_estimate_bits", lambda *args: 0.25)

    decision = optimize_preference(config, observations)

    assert decision.selected_pair is None
    assert decision.stopping.stop is True
    assert decision.stopping.reason is StopReason.SEARCH_BUDGET_EXHAUSTED
    assert decision.stopping.disposition is SearchDisposition.INCONCLUSIVE_WITHIN_BUDGET
    assert decision.stopping.maximum_information_bits == pytest.approx(0.25)
    assert decision.candidate_scores


def test_convergence_takes_precedence_exactly_at_hard_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OptimizerConfig(posterior_draws=256)
    observations = _ten_distinct_observations()
    posterior = _controlled_posterior(config)
    monkeypatch.setattr(optimizer, "_fit_posterior_with_draws", lambda *args: posterior)
    monkeypatch.setattr(optimizer, "_expected_information_estimate_bits", lambda *args: 0.0)

    decision = optimize_preference(config, observations)

    assert decision.selected_pair is None
    assert decision.stopping.reason is StopReason.SEARCH_CONVERGED
    assert decision.stopping.disposition is None
    assert decision.stopping.maximum_information_bits == 0.0
    assert decision.candidate_scores


def test_stopping_boundaries_are_applied_exactly() -> None:
    config = OptimizerConfig()
    posterior = _controlled_posterior(config)
    below_information_threshold = math.nextafter(config.information_threshold_bits, 0.0)

    assert optimizer._convergence_satisfied(
        config, posterior, comparison_count=4, maximum_information_bits=below_information_threshold
    )
    assert not optimizer._convergence_satisfied(
        config,
        posterior,
        comparison_count=4,
        maximum_information_bits=config.information_threshold_bits,
    )
    assert not optimizer._convergence_satisfied(
        config, posterior, comparison_count=3, maximum_information_bits=0.0
    )
    assert not optimizer._convergence_satisfied(
        config,
        posterior.model_copy(update={"concentration_probability": math.nextafter(0.9, 0.0)}),
        comparison_count=4,
        maximum_information_bits=0.0,
    )
    assert not optimizer._convergence_satisfied(
        config,
        posterior.model_copy(update={"concentration_interval": (2.5, 3.25)}),
        comparison_count=4,
        maximum_information_bits=0.0,
    )


def test_complete_interval_eligible_pair_set_is_independently_enumerated() -> None:
    config = OptimizerConfig(lower_bound=1.0, upper_bound=3.0, resolution=1.0)
    posterior = _controlled_posterior(config, interval=(2.0, 2.0))

    assert optimizer._eligible_pairs(config, posterior) == (
        CandidatePair(value_a=1.0, value_b=2.0),
        CandidatePair(value_a=2.0, value_b=3.0),
    )


def test_equal_scores_prefer_unseen_then_coordinate_order() -> None:
    repeated = {CandidatePair(value_a=1.0, value_b=2.0)}
    scores = (
        AcquisitionScore(
            pair=CandidatePair(value_a=1.0, value_b=2.0), expected_information_bits=0.25
        ),
        AcquisitionScore(
            pair=CandidatePair(value_a=2.0, value_b=3.0), expected_information_bits=0.25
        ),
        AcquisitionScore(
            pair=CandidatePair(value_a=1.0, value_b=3.0), expected_information_bits=0.25
        ),
    )

    ordered = optimizer._order_acquisition_scores(scores, repeated)

    assert tuple(item.pair for item in ordered) == (
        CandidatePair(value_a=1.0, value_b=3.0),
        CandidatePair(value_a=2.0, value_b=3.0),
        CandidatePair(value_a=1.0, value_b=2.0),
    )

    strictly_better_repeat = scores[0].model_copy(update={"expected_information_bits": 0.250001})
    assert optimizer._order_acquisition_scores((strictly_better_repeat, *scores[1:]), repeated)[
        0
    ].pair == CandidatePair(value_a=1.0, value_b=2.0)


def _synthetic_outcome(value_a: float, value_b: float, optimum: float) -> PreferenceOutcome:
    utility_a = -((value_a - optimum) ** 2)
    utility_b = -((value_b - optimum) ** 2)
    difference = utility_a - utility_b
    if abs(difference) <= 0.10:
        return PreferenceOutcome.TIE
    return PreferenceOutcome.A if difference > 0 else PreferenceOutcome.B


def _run_synthetic_study(
    config: OptimizerConfig, *, optimum: float | None
) -> tuple[tuple[PreferenceObservation, ...], OptimizerDecision]:
    observations: tuple[PreferenceObservation, ...] = ()
    while True:
        decision = optimize_preference(config, observations)
        if decision.stopping.stop:
            return observations, decision
        assert decision.selected_pair is not None
        outcome = (
            PreferenceOutcome.TIE
            if optimum is None
            else _synthetic_outcome(
                decision.selected_pair.value_a,
                decision.selected_pair.value_b,
                optimum,
            )
        )
        observations += (
            PreferenceObservation(
                value_a=decision.selected_pair.value_a,
                value_b=decision.selected_pair.value_b,
                outcome=outcome,
            ),
        )


def _run_surface_study(
    config: OptimizerConfig,
    *,
    utility: Callable[[float], float],
    noise: tuple[float, ...] = (0.0,),
) -> tuple[tuple[PreferenceObservation, ...], OptimizerDecision]:
    observations: tuple[PreferenceObservation, ...] = ()
    while True:
        decision = optimize_preference(config, observations)
        if decision.stopping.stop:
            return observations, decision
        assert decision.selected_pair is not None
        value_a = decision.selected_pair.value_a
        value_b = decision.selected_pair.value_b
        difference = utility(value_a) - utility(value_b) + noise[len(observations) % len(noise)]
        outcome = (
            PreferenceOutcome.TIE
            if abs(difference) <= 0.10
            else (PreferenceOutcome.A if difference > 0.0 else PreferenceOutcome.B)
        )
        observations += (PreferenceObservation(value_a=value_a, value_b=value_b, outcome=outcome),)


def test_unimodal_study_recommends_near_known_optimum_and_replays() -> None:
    config = OptimizerConfig(posterior_draws=512)

    observations, decision = _run_synthetic_study(config, optimum=2.75)
    replayed = optimize_preference(config, observations)

    assert abs(decision.posterior.recommendation - 2.75) <= config.resolution
    assert len(observations) <= config.maximum_comparisons
    assert replayed == decision


def test_noisy_unimodal_study_remains_bounded_and_replayable() -> None:
    config = OptimizerConfig(resolution=0.5, posterior_draws=256)

    observations, decision = _run_surface_study(
        config,
        utility=lambda value: -((value - 2.75) ** 2),
        noise=(0.18, -0.14, 0.09, -0.05),
    )

    assert abs(decision.posterior.recommendation - 2.75) <= 0.5
    assert len(observations) <= config.maximum_comparisons
    assert optimize_preference(config, observations) == decision


def test_multimodal_study_recommends_a_supported_mode() -> None:
    config = OptimizerConfig(resolution=0.5, posterior_draws=256)

    observations, decision = _run_surface_study(
        config,
        utility=lambda value: max(-((value - 2.0) ** 2), -((value - 4.0) ** 2)),
    )

    assert (
        min(
            abs(decision.posterior.recommendation - 2.0),
            abs(decision.posterior.recommendation - 4.0),
        )
        <= config.resolution
    )
    assert len(observations) <= config.maximum_comparisons


def test_flat_preference_never_manufactures_a_directional_winner() -> None:
    config = OptimizerConfig(posterior_draws=512)

    observations, decision = _run_synthetic_study(config, optimum=None)

    assert all(item.outcome is PreferenceOutcome.TIE for item in observations)
    assert len(observations) == config.maximum_comparisons
    assert decision.stopping.reason is StopReason.SEARCH_BUDGET_EXHAUSTED
    assert decision.stopping.disposition is SearchDisposition.INCONCLUSIVE_WITHIN_BUDGET
    assert decision.posterior.concentration_interval[0] <= decision.posterior.recommendation
    assert decision.posterior.recommendation <= decision.posterior.concentration_interval[1]


def test_expected_information_estimate_reuses_common_standard_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OptimizerConfig(posterior_draws=256)
    observations = (PreferenceObservation(value_a=2.0, value_b=4.0, outcome=PreferenceOutcome.B),)
    draws = optimizer._standard_normal_draws(config, observations)
    posterior = optimizer._fit_posterior_with_draws(config, observations, draws)
    seen_draws: list[np.ndarray] = []
    original_fit = optimizer._fit_posterior_with_draws

    def tracking_fit(
        inner_config: OptimizerConfig,
        inner_observations: tuple[PreferenceObservation, ...],
        standard_normal_draws: np.ndarray,
    ) -> PosteriorSummary:
        seen_draws.append(standard_normal_draws)
        return original_fit(inner_config, inner_observations, standard_normal_draws)

    monkeypatch.setattr(optimizer, "_fit_posterior_with_draws", tracking_fit)

    estimate = optimizer._expected_information_estimate_bits(
        config,
        observations,
        posterior,
        CandidatePair(value_a=2.0, value_b=4.0),
        draws,
    )

    assert estimate >= 0.0
    assert len(seen_draws) == len(PreferenceOutcome)
    assert all(item is draws for item in seen_draws)


def test_numerical_linear_algebra_failure_has_stable_public_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_solve(_left: np.ndarray, _right: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("controlled failure")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)

    with pytest.raises(RuntimeError, match="numerical failure"):
        fit_posterior(OptimizerConfig(posterior_draws=256), ())
