"""Request-local bounds for verified prompt speculation; no runtime imports."""

from dataclasses import dataclass, field
import math

MAX_PROPOSAL_TOKENS = 32


@dataclass
class LookupPolicy:
    calibration: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    native_reason: str | None = None
    _window: int = 4
    _misses: int = 0
    _maximum: int = 8
    prefix_search_tokens: int = 3

    @classmethod
    def for_architecture(cls, architecture):
        policy = cls()
        if architecture == "applegpu_g17s":
            policy._window = policy._maximum = MAX_PROPOSAL_TOKENS
            policy.prefix_search_tokens = 8
        return policy

    @property
    def proposal_limit(self):
        return self._window if self.native_reason is None and len(self.calibration) == 2 else 0

    def observe(self, proposed, accepted, elapsed_seconds):
        if self.native_reason is not None:
            return
        if (type(proposed) is not int or type(accepted) is not int
                or not 0 <= accepted <= proposed <= self._maximum
                or type(elapsed_seconds) not in (int, float)
                or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0):
            self.native_reason = "invalid_observation"
            return
        if proposed == 0:
            if len(self.calibration) < 2:
                self.calibration.append(elapsed_seconds)
            return
        self.samples.append((elapsed_seconds, accepted + 1))
        self.samples[:] = self.samples[-4:]
        self._misses = self._misses + 1 if accepted == 0 else 0
        self._window = min(self._maximum, self._window + 2) if accepted == proposed else max(2, accepted + 1)
        if self._misses >= 2:
            self.native_reason = "low_acceptance"
        elif len(self.calibration) == 2 and elapsed_seconds > 2 * min(self.calibration) * (accepted + 1):
            # A single severe loss must not wait for a four-block window: short
            # replies can finish before that window ever fills.
            self.native_reason = "verification_cost"
        elif len(self.samples) == 4 and len(self.calibration) == 2:
            cost = sum(seconds for seconds, _ in self.samples) / sum(tokens for _, tokens in self.samples)
            if cost >= .9 * min(self.calibration):
                self.native_reason = "verification_cost"
