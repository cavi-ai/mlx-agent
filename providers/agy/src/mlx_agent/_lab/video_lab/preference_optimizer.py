from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from video_lab.domain import FrozenModel, canonical_json

FloatArray = npt.NDArray[np.float64]

OPTIMIZER_VERSION: Literal["preference-gp-laplace.v1"] = "preference-gp-laplace.v1"
OPTIMIZER_CONFIG_SCHEMA_VERSION: Literal["preference-optimizer-config.v1"] = (
    "preference-optimizer-config.v1"
)
_ADAPTER_SAFE_LOWER_BOUND = 1.0
_ADAPTER_SAFE_UPPER_BOUND = 5.0
# V1 is qualified on at most 17 points, bounding cubic GP work and pair enumeration.
_MAX_GUIDANCE_GRID_CARDINALITY = 17


class PreferenceOutcome(StrEnum):
    A = "a"
    B = "b"
    TIE = "no_perceptible_difference"


class StopReason(StrEnum):
    CONTINUE = "continue"
    SEARCH_CONVERGED = "search_converged"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    CANDIDATE_SPACE_EXHAUSTED = "candidate_space_exhausted"


class SearchDisposition(StrEnum):
    INCONCLUSIVE_WITHIN_BUDGET = "inconclusive_within_budget"


class CandidatePair(FrozenModel):
    value_a: float
    value_b: float

    @model_validator(mode="after")
    def distinct_values(self) -> Self:
        if not math.isfinite(self.value_a) or not math.isfinite(self.value_b):
            raise ValueError("candidate pair values must be finite")
        if self.value_a == self.value_b:
            raise ValueError("candidate pair values must differ")
        return self


class PreferenceObservation(CandidatePair):
    outcome: PreferenceOutcome


class OptimizerConfig(FrozenModel):
    schema_version: Literal["preference-optimizer-config.v1"] = OPTIMIZER_CONFIG_SCHEMA_VERSION
    optimizer_version: Literal["preference-gp-laplace.v1"] = OPTIMIZER_VERSION
    lower_bound: float = 1.0
    upper_bound: float = 5.0
    resolution: float = Field(default=0.25, gt=0)
    kernel_amplitude: float = Field(default=1.0, gt=0)
    kernel_length_scale: float = Field(default=0.75, gt=0)
    jitter: float = Field(default=1e-6, gt=0)
    tie_threshold: float = Field(default=0.35, gt=0)
    minimum_comparisons: Literal[4] = 4
    maximum_comparisons: Literal[10] = 10
    concentration_mass: float = Field(default=0.90, gt=0, lt=1)
    concentration_width: float = Field(default=0.5, ge=0)
    information_threshold_bits: float = Field(default=0.01, ge=0)
    posterior_draws: int = Field(default=4096, ge=256)

    @model_validator(mode="after")
    def valid_interval_and_budget(self) -> Self:
        if not all(
            math.isfinite(value)
            for value in (
                self.lower_bound,
                self.upper_bound,
                self.resolution,
                self.kernel_amplitude,
                self.kernel_length_scale,
                self.jitter,
                self.tie_threshold,
                self.concentration_mass,
                self.concentration_width,
                self.information_threshold_bits,
            )
        ):
            raise ValueError("optimizer floating-point inputs must be finite")
        if (
            self.lower_bound < _ADAPTER_SAFE_LOWER_BOUND
            or self.upper_bound > _ADAPTER_SAFE_UPPER_BOUND
        ):
            raise ValueError("optimizer bounds must stay within the adapter-safe range")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("optimizer bounds must form an exact positive grid")
        span_steps = (self.upper_bound - self.lower_bound) / self.resolution
        if not math.isfinite(span_steps):
            raise ValueError("optimizer grid cardinality must be finite and bounded")
        if span_steps > (_MAX_GUIDANCE_GRID_CARDINALITY - 1) + 1e-9:
            raise ValueError(
                "optimizer grid cardinality exceeds the v1 maximum of "
                f"{_MAX_GUIDANCE_GRID_CARDINALITY}"
            )
        rounded_steps = round(span_steps)
        if not math.isclose(span_steps, rounded_steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("optimizer bounds must form an exact positive grid")
        return self

    @property
    def initial_pair(self) -> CandidatePair:
        grid = build_guidance_grid(self)
        lower_target = self.lower_bound + 0.25 * (self.upper_bound - self.lower_bound)
        upper_target = self.lower_bound + 0.75 * (self.upper_bound - self.lower_bound)
        return CandidatePair(
            value_a=_nearest_grid_value(grid, lower_target),
            value_b=_nearest_grid_value(grid, upper_target),
        )

    def validate_observations(self, observations: tuple[PreferenceObservation, ...]) -> None:
        self.validate_observation_grid(observations)
        if len(observations) > self.maximum_comparisons:
            raise ValueError("preference observations exceed the comparison budget")

    def validate_observation_grid(self, observations: tuple[PreferenceObservation, ...]) -> None:
        allowed = set(build_guidance_grid(self))
        if any(item.value_a not in allowed or item.value_b not in allowed for item in observations):
            raise ValueError("preference observation value is outside the configured grid")


class PosteriorSummary(FrozenModel):
    grid: tuple[float, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    argmax_probability: tuple[float, ...]
    recommendation: float
    concentration_interval: tuple[float, float]
    concentration_probability: float
    content_digest: str


class AcquisitionScore(FrozenModel):
    pair: CandidatePair
    expected_information_bits: float = Field(ge=0)


class StoppingDecision(FrozenModel):
    stop: bool
    reason: StopReason
    disposition: SearchDisposition | None = None
    comparison_count: int = Field(ge=0)
    maximum_information_bits: float = Field(ge=0)


class OptimizerDecision(FrozenModel):
    posterior: PosteriorSummary
    candidate_scores: tuple[AcquisitionScore, ...]
    selected_pair: CandidatePair | None
    stopping: StoppingDecision


def build_guidance_grid(config: OptimizerConfig) -> tuple[float, ...]:
    steps = round((config.upper_bound - config.lower_bound) / config.resolution)
    return tuple(
        round(config.lower_bound + index * config.resolution, 12) for index in range(steps + 1)
    )


def _nearest_grid_value(grid: tuple[float, ...], target: float) -> float:
    return min(grid, key=lambda value: (abs(value - target), value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def outcome_probabilities(
    config: OptimizerConfig,
    *,
    difference_mean: float,
    difference_variance: float,
) -> dict[PreferenceOutcome, float]:
    scale = math.sqrt(1.0 + math.pi * max(0.0, difference_variance) / 8.0)
    mean = difference_mean / scale
    threshold = config.tie_threshold / scale
    probability_a = _sigmoid(mean - threshold)
    probability_b = _sigmoid(-mean - threshold)
    probability_tie = max(0.0, 1.0 - probability_a - probability_b)
    total = probability_a + probability_b + probability_tie
    return {
        PreferenceOutcome.A: probability_a / total,
        PreferenceOutcome.B: probability_b / total,
        PreferenceOutcome.TIE: probability_tie / total,
    }


def _observation_probability(
    config: OptimizerConfig, difference: float, outcome: PreferenceOutcome
) -> float:
    return outcome_probabilities(config, difference_mean=difference, difference_variance=0.0)[
        outcome
    ]


def _outcome_log_likelihood(
    config: OptimizerConfig, difference: float, outcome: PreferenceOutcome
) -> float:
    return math.log(max(_observation_probability(config, difference, outcome), 1e-15))


def _rbf_covariance(config: OptimizerConfig, grid: FloatArray) -> FloatArray:
    distances = grid[:, None] - grid[None, :]
    covariance = config.kernel_amplitude**2 * np.exp(
        -0.5 * distances**2 / config.kernel_length_scale**2
    )
    covariance += np.eye(len(grid), dtype=np.float64) * config.jitter
    return covariance


def _log_likelihood_derivatives(
    config: OptimizerConfig, difference: float, outcome: PreferenceOutcome
) -> tuple[float, float]:
    step = 1e-4
    center = _outcome_log_likelihood(config, difference, outcome)
    lower = _outcome_log_likelihood(config, difference - step, outcome)
    upper = _outcome_log_likelihood(config, difference + step, outcome)
    first = (upper - lower) / (2.0 * step)
    second = (upper - 2.0 * center + lower) / (step * step)
    return first, min(second, -1e-10)


def _map_utility(
    config: OptimizerConfig,
    grid: tuple[float, ...],
    observations: tuple[PreferenceObservation, ...],
    prior_inverse: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    indexes = {value: index for index, value in enumerate(grid)}
    utility = np.zeros(len(grid), dtype=np.float64)
    precision = prior_inverse.copy()
    for _iteration in range(80):
        gradient = -(prior_inverse @ utility)
        precision = prior_inverse.copy()
        for observation in observations:
            index_a = indexes[observation.value_a]
            index_b = indexes[observation.value_b]
            first, second = _log_likelihood_derivatives(
                config, utility[index_a] - utility[index_b], observation.outcome
            )
            direction = np.zeros(len(grid), dtype=np.float64)
            direction[index_a] = 1.0
            direction[index_b] = -1.0
            gradient += first * direction
            precision += -second * np.outer(direction, direction)
        step = np.linalg.solve(precision, gradient)
        candidate = utility + step
        if float(np.max(np.abs(candidate - utility))) < 1e-9:
            utility = candidate
            break
        utility = candidate
    else:
        raise RuntimeError("preference posterior MAP did not converge")
    return utility, precision


def _posterior_covariance(precision: FloatArray) -> FloatArray:
    return np.linalg.inv(precision)


def _sampling_seed(config: OptimizerConfig, observations: tuple[PreferenceObservation, ...]) -> int:
    payload = canonical_json(
        {
            "optimizer_version": config.optimizer_version,
            "config": config.model_dump(mode="json"),
            "observations": [item.model_dump(mode="json") for item in observations],
        }
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _standard_normal_draws(
    config: OptimizerConfig,
    observations: tuple[PreferenceObservation, ...],
) -> FloatArray:
    generator = np.random.default_rng(_sampling_seed(config, observations))
    return generator.standard_normal(
        size=(config.posterior_draws, len(build_guidance_grid(config)))
    )


def shortest_mass_interval(
    *,
    grid: tuple[float, ...],
    probability: tuple[float, ...],
    target_mass: float,
) -> tuple[tuple[float, float], float]:
    if len(grid) != len(probability) or not grid:
        raise ValueError("grid and probability must have equal non-zero length")
    candidates: list[tuple[float, float, float, float]] = []
    for lower in range(len(grid)):
        mass = 0.0
        for upper in range(lower, len(grid)):
            mass += probability[upper]
            if mass + 1e-12 >= target_mass:
                candidates.append((grid[upper] - grid[lower], grid[lower], grid[upper], mass))
                break
    if not candidates:
        return (grid[0], grid[-1]), float(sum(probability))
    _width, lower_value, upper_value, mass = min(candidates)
    return (lower_value, upper_value), mass


def _argmax_projection(
    config: OptimizerConfig,
    grid: tuple[float, ...],
    mean: FloatArray,
    covariance: FloatArray,
    standard_normal_draws: FloatArray,
) -> tuple[FloatArray, int, tuple[float, float], float]:
    stabilized = covariance + np.eye(len(grid), dtype=np.float64) * config.jitter
    factor = np.linalg.cholesky(stabilized)
    samples = mean + standard_normal_draws @ factor.T
    winners = np.argmax(samples, axis=1)
    counts = np.bincount(winners, minlength=len(grid)).astype(np.float64)
    probabilities = counts / float(config.posterior_draws)
    recommendation_index = int(np.argmax(probabilities))
    interval, mass = shortest_mass_interval(
        grid=grid,
        probability=tuple(float(value) for value in probabilities),
        target_mass=config.concentration_mass,
    )
    return probabilities, recommendation_index, interval, mass


def fit_posterior(
    config: OptimizerConfig,
    observations: tuple[PreferenceObservation, ...],
) -> PosteriorSummary:
    config.validate_observations(observations)
    standard_normal_draws = _standard_normal_draws(config, observations)
    return _fit_posterior_with_draws(config, observations, standard_normal_draws)


def _fit_posterior_with_draws(
    config: OptimizerConfig,
    observations: tuple[PreferenceObservation, ...],
    standard_normal_draws: FloatArray,
) -> PosteriorSummary:
    config.validate_observation_grid(observations)
    grid = build_guidance_grid(config)
    expected_shape = (config.posterior_draws, len(grid))
    if standard_normal_draws.shape != expected_shape:
        raise ValueError("standard-normal draws do not match the optimizer grid")
    try:
        grid_array = np.asarray(grid, dtype=np.float64)
        prior = _rbf_covariance(config, grid_array)
        prior_inverse = np.linalg.inv(prior)
        mean, precision = _map_utility(config, grid, observations, prior_inverse)
        covariance = _posterior_covariance(precision)
        probabilities, recommendation_index, interval, mass = _argmax_projection(
            config, grid, mean, covariance, standard_normal_draws
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError("preference posterior numerical failure") from error
    payload = {
        "optimizer_version": config.optimizer_version,
        "config": config.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in observations],
        "grid": list(grid),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
        "argmax_probability": probabilities.tolist(),
        "recommendation": grid[recommendation_index],
        "concentration_interval": list(interval),
        "concentration_probability": mass,
    }
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return PosteriorSummary(
        grid=grid,
        mean=tuple(float(value) for value in mean),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        argmax_probability=tuple(float(value) for value in probabilities),
        recommendation=grid[recommendation_index],
        concentration_interval=interval,
        concentration_probability=mass,
        content_digest=f"sha256:{digest}",
    )


def _entropy_bits(probability: tuple[float, ...]) -> float:
    return -sum(value * math.log2(value) for value in probability if value > 0.0)


def _canonical_pair(left: float, right: float) -> CandidatePair:
    return CandidatePair(value_a=min(left, right), value_b=max(left, right))


def _eligible_pairs(
    config: OptimizerConfig,
    posterior: PosteriorSummary,
) -> tuple[CandidatePair, ...]:
    lower, upper = posterior.concentration_interval
    values = build_guidance_grid(config)
    result: list[CandidatePair] = []
    for index, value_a in enumerate(values):
        for value_b in values[index + 1 :]:
            pair = CandidatePair(value_a=value_a, value_b=value_b)
            if lower <= value_a <= upper or lower <= value_b <= upper:
                result.append(pair)
    return tuple(result)


def _difference_moments(posterior: PosteriorSummary, pair: CandidatePair) -> tuple[float, float]:
    indexes = {value: index for index, value in enumerate(posterior.grid)}
    index_a = indexes[pair.value_a]
    index_b = indexes[pair.value_b]
    covariance = posterior.covariance
    mean = posterior.mean[index_a] - posterior.mean[index_b]
    variance = (
        covariance[index_a][index_a]
        + covariance[index_b][index_b]
        - 2.0 * covariance[index_a][index_b]
    )
    return mean, max(0.0, variance)


def _expected_information_estimate_bits(
    config: OptimizerConfig,
    observations: tuple[PreferenceObservation, ...],
    posterior: PosteriorSummary,
    pair: CandidatePair,
    standard_normal_draws: FloatArray,
) -> float:
    """Estimate EIG with version-pinned common random numbers for all outcomes."""
    mean, variance = _difference_moments(posterior, pair)
    predictive = outcome_probabilities(config, difference_mean=mean, difference_variance=variance)
    current_entropy = _entropy_bits(posterior.argmax_probability)
    expected_entropy = 0.0
    for outcome in PreferenceOutcome:
        hypothetical = observations + (
            PreferenceObservation(value_a=pair.value_a, value_b=pair.value_b, outcome=outcome),
        )
        updated = _fit_posterior_with_draws(config, hypothetical, standard_normal_draws)
        expected_entropy += predictive[outcome] * _entropy_bits(updated.argmax_probability)
    return max(0.0, current_entropy - expected_entropy)


def _order_acquisition_scores(
    scores: tuple[AcquisitionScore, ...], repeated: set[CandidatePair]
) -> tuple[AcquisitionScore, ...]:
    return tuple(
        sorted(
            scores,
            key=lambda score: (
                -score.expected_information_bits,
                score.pair in repeated,
                score.pair.value_a,
                score.pair.value_b,
            ),
        )
    )


def _candidate_scores(
    config: OptimizerConfig,
    observations: tuple[PreferenceObservation, ...],
    posterior: PosteriorSummary,
    standard_normal_draws: FloatArray,
) -> tuple[AcquisitionScore, ...]:
    repeated = {_canonical_pair(item.value_a, item.value_b) for item in observations}
    scores: list[AcquisitionScore] = []
    for pair in _eligible_pairs(config, posterior):
        estimate = _expected_information_estimate_bits(
            config, observations, posterior, pair, standard_normal_draws
        )
        if pair in repeated and estimate <= 0.0:
            continue
        scores.append(AcquisitionScore(pair=pair, expected_information_bits=estimate))
    return _order_acquisition_scores(tuple(scores), repeated)


def _convergence_satisfied(
    config: OptimizerConfig,
    posterior: PosteriorSummary,
    *,
    comparison_count: int,
    maximum_information_bits: float,
) -> bool:
    interval_width = posterior.concentration_interval[1] - posterior.concentration_interval[0]
    return (
        comparison_count >= config.minimum_comparisons
        and posterior.concentration_probability >= config.concentration_mass
        and interval_width <= config.concentration_width
        and maximum_information_bits < config.information_threshold_bits
    )


def optimize_preference(
    config: OptimizerConfig,
    observations: tuple[PreferenceObservation, ...],
) -> OptimizerDecision:
    config.validate_observations(observations)
    standard_normal_draws = _standard_normal_draws(config, observations)
    posterior = _fit_posterior_with_draws(config, observations, standard_normal_draws)
    if not observations:
        return OptimizerDecision(
            posterior=posterior,
            candidate_scores=(),
            selected_pair=config.initial_pair,
            stopping=StoppingDecision(
                stop=False,
                reason=StopReason.CONTINUE,
                comparison_count=0,
                maximum_information_bits=0.0,
            ),
        )
    ordered = _candidate_scores(config, observations, posterior, standard_normal_draws)
    maximum_information = ordered[0].expected_information_bits if ordered else 0.0
    converged = _convergence_satisfied(
        config,
        posterior,
        comparison_count=len(observations),
        maximum_information_bits=maximum_information,
    )
    at_budget = len(observations) >= config.maximum_comparisons
    exhausted = not ordered
    if converged:
        reason = StopReason.SEARCH_CONVERGED
        disposition = None
    elif at_budget:
        reason = StopReason.SEARCH_BUDGET_EXHAUSTED
        disposition = SearchDisposition.INCONCLUSIVE_WITHIN_BUDGET
    elif exhausted:
        reason = StopReason.CANDIDATE_SPACE_EXHAUSTED
        disposition = None
    else:
        reason = StopReason.CONTINUE
        disposition = None
    stop = converged or at_budget or exhausted
    return OptimizerDecision(
        posterior=posterior,
        candidate_scores=ordered,
        selected_pair=None if stop else ordered[0].pair,
        stopping=StoppingDecision(
            stop=stop,
            reason=reason,
            disposition=disposition,
            comparison_count=len(observations),
            maximum_information_bits=maximum_information,
        ),
    )
