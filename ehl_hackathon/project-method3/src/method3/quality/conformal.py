from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SplitConformalCalibrator:
    """One-sided split-conformal calibration for a quality LOWER bound.

    Standard split-conformal recipe (Vovk et al.), one-sided variant: the
    nonconformity score is the signed overshoot `predicted - actual` (how much the
    base regressor over-predicted quality), not the absolute residual — an
    under-prediction never threatens the safety of a *lower* bound, so it shouldn't
    inflate the interval. Given n calibration scores, the finite-sample-corrected
    quantile index is `ceil((n+1)(1-alpha))`, 1-indexed into the sorted scores. If
    that index exceeds n, there is not enough calibration data to certify any finite
    bound at this alpha — `coverage_guaranteed` is False and `lower_bound` returns
    -inf (clamped to 0.0 by callers that know quality lives in [0,1]) rather than
    silently returning an uncalibrated number. This is deliberately more
    conservative than clamping to the max observed score, which some libraries do —
    an unearned bound is a worse bug than an unhelpful one.

    Guarantee (marginal, distribution-free, under calibration/test exchangeability):
    P(actual >= predicted - quantile) >= 1 - alpha.
    """
    alpha: float = 0.1
    quantile: float = float("inf")
    n_calibration: int = 0
    coverage_guaranteed: bool = False

    def fit(self, predicted: list[float], actual: list[float]) -> "SplitConformalCalibrator":
        if len(predicted) != len(actual):
            raise ValueError("predicted and actual must be the same length")
        if not predicted:
            raise ValueError("cannot calibrate on zero calibration examples")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        scores = sorted(p - a for p, a in zip(predicted, actual))
        n = len(scores)
        level = math.ceil((n + 1) * (1 - self.alpha))
        self.n_calibration = n
        if level > n:
            self.quantile = float("inf")
            self.coverage_guaranteed = False
        else:
            self.quantile = scores[level - 1]
            self.coverage_guaranteed = True
        return self

    def lower_bound(self, predicted: float, floor: float = 0.0) -> float:
        if self.quantile == float("inf"):
            return floor
        return max(floor, predicted - self.quantile)

    def to_dict(self) -> dict[str, object]:
        return {"alpha": self.alpha, "quantile": self.quantile, "n_calibration": self.n_calibration, "coverage_guaranteed": self.coverage_guaranteed}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SplitConformalCalibrator":
        return cls(float(payload["alpha"]), float(payload["quantile"]), int(payload["n_calibration"]), bool(payload["coverage_guaranteed"]))
