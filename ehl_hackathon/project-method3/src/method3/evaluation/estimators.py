from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class EstimatorResult:
    """A policy-value estimate with its uncertainty and the diagnostics needed to
    decide whether to believe it."""
    name: str
    value: float
    standard_error: float
    n: int
    n_matched: int              # samples where the logged action == the target action
    effective_sample_size: float
    max_weight: float
    clipped_fraction: float     # share of matched samples whose IPS term was suppressed

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        return (self.value - z * self.standard_error, self.value + z * self.standard_error)

    def to_dict(self) -> dict[str, object]:
        low, high = self.confidence_interval()
        return {
            "estimator": self.name, "value": self.value, "standard_error": self.standard_error,
            "ci95": [low, high], "n": self.n, "n_matched": self.n_matched,
            "effective_sample_size": self.effective_sample_size,
            "max_weight": self.max_weight, "clipped_fraction": self.clipped_fraction,
        }


@dataclass
class Sample:
    """One test trajectory reduced to what every estimator needs.

    q_target : Q̂(x, π(x))  — reward model's prediction for the action the TARGET policy takes
    q_logged : Q̂(x, a)     — reward model's prediction for the action that was LOGGED
    reward   : r            — the observed outcome (only meaningful when matched)
    propensity : p(a | x)   — logging-policy probability of the logged action
    matched  : 1{a == π(x)}
    """
    q_target: float
    q_logged: float
    reward: float
    propensity: float
    matched: bool


def _mean_and_se(contributions: list[float]) -> tuple[float, float]:
    """Sample mean and its standard error. The estimators below are all averages of
    per-sample contributions, so the SE of the mean is the honest uncertainty on the
    estimate — reporting a point value with no SE is how a high-variance IPS estimate
    gets mistaken for a precise one."""
    n = len(contributions)
    if n == 0:
        return 0.0, 0.0
    mean = sum(contributions) / n
    if n == 1:
        return mean, 0.0
    variance = sum((c - mean) ** 2 for c in contributions) / (n - 1)
    return mean, math.sqrt(variance / n)


def _weights(samples: list[Sample], clip: float) -> list[float]:
    return [(1.0 / max(clip, s.propensity)) if s.matched else 0.0 for s in samples]


def _diagnostics(weights: list[float], n_matched: int, clipped: int) -> tuple[float, float, float]:
    positive = [w for w in weights if w > 0]
    if not positive:
        return 0.0, 0.0, 0.0
    # Kish effective sample size on the importance weights.
    ess = sum(positive) ** 2 / sum(w * w for w in positive)
    return ess, max(positive), (clipped / n_matched if n_matched else 0.0)


def direct_method(samples: list[Sample]) -> EstimatorResult:
    """DM: trust the reward model entirely. Zero variance from importance weights,
    but fully inherits the reward model's bias."""
    contributions = [s.q_target for s in samples]
    value, se = _mean_and_se(contributions)
    n_matched = sum(1 for s in samples if s.matched)
    return EstimatorResult("direct_method", value, se, len(samples), n_matched, float(len(samples)), 0.0, 0.0)


def inverse_propensity(samples: list[Sample], clip: float = 0.05) -> EstimatorResult:
    """IPS: unbiased under correct propensities and positivity, but its variance
    explodes when some matched sample has a tiny propensity."""
    weights = _weights(samples, clip)
    contributions = [w * s.reward for w, s in zip(weights, samples)]
    value, se = _mean_and_se(contributions)
    n_matched = sum(1 for s in samples if s.matched)
    ess, max_w, _ = _diagnostics(weights, n_matched, 0)
    return EstimatorResult("ips", value, se, len(samples), n_matched, ess, max_w, 0.0)


def self_normalized_ips(samples: list[Sample], clip: float = 0.05) -> EstimatorResult:
    """SNIPS: divide by the sum of weights instead of n. Trades a small,
    asymptotically vanishing bias for a large variance reduction, and is
    scale-stable when the weights do not average to 1 (which they never exactly do
    in a finite sample). Standard practice alongside DR rather than instead of it."""
    weights = _weights(samples, clip)
    total_weight = sum(weights)
    n_matched = sum(1 for s in samples if s.matched)
    ess, max_w, _ = _diagnostics(weights, n_matched, 0)
    if total_weight <= 0:
        # No logged action ever agreed with the target policy: SNIPS is undefined.
        # Returning 0.0 silently would look like "the policy is terrible", so the
        # caller must read n_matched to see that this is "no evidence", not "bad".
        return EstimatorResult("snips", 0.0, 0.0, len(samples), 0, 0.0, 0.0, 0.0)
    value = sum(w * s.reward for w, s in zip(weights, samples)) / total_weight
    # Delta-method SE for a self-normalized ratio estimator.
    residual_contributions = [w * (s.reward - value) / (total_weight / len(samples)) for w, s in zip(weights, samples)]
    _, se = _mean_and_se(residual_contributions)
    return EstimatorResult("snips", value, se, len(samples), n_matched, ess, max_w, 0.0)


def doubly_robust(samples: list[Sample], clip: float = 0.05) -> EstimatorResult:
    """DR: DM plus an importance-weighted correction on the reward model's residual.
    Consistent if EITHER the propensity model OR the reward model is correct, and
    generally lower variance than plain IPS (Dudík, Langford & Li, ICML 2011)."""
    weights = _weights(samples, clip)
    contributions = [s.q_target + w * (s.reward - s.q_logged) for w, s in zip(weights, samples)]
    value, se = _mean_and_se(contributions)
    n_matched = sum(1 for s in samples if s.matched)
    ess, max_w, _ = _diagnostics(weights, n_matched, 0)
    return EstimatorResult("doubly_robust", value, se, len(samples), n_matched, ess, max_w, 0.0)


def switch_doubly_robust(samples: list[Sample], clip: float = 0.05, weight_threshold: float = 10.0) -> EstimatorResult:
    """SWITCH-DR: use the DR correction only while the importance weight is
    reasonable; above `weight_threshold`, fall back to the direct method for that
    sample. Extreme weights are exactly where DR's correction term does more harm
    than good, so switching them off buys a large variance reduction for a bounded,
    documented bias — the standard fix when overlap is weak (which dataset2's
    near-deterministic first era deliberately produces)."""
    weights = _weights(samples, clip)
    contributions = []
    clipped = 0
    kept_weights = []
    for weight, sample in zip(weights, samples):
        if weight > weight_threshold:
            contributions.append(sample.q_target)  # DM only for this sample
            clipped += 1
            kept_weights.append(0.0)
        else:
            contributions.append(sample.q_target + weight * (sample.reward - sample.q_logged))
            kept_weights.append(weight)
    value, se = _mean_and_se(contributions)
    n_matched = sum(1 for s in samples if s.matched)
    ess, max_w, clipped_fraction = _diagnostics(kept_weights, n_matched, clipped)
    return EstimatorResult("switch_dr", value, se, len(samples), n_matched, ess, max_w, clipped_fraction)


def all_estimators(samples: list[Sample], clip: float = 0.05, weight_threshold: float = 10.0) -> dict[str, EstimatorResult]:
    """Run the whole family. Agreement across estimators is evidence the estimate is
    real; a large spread (especially DM vs IPS) is evidence of propensity
    misspecification or weak overlap, and is the signal to distrust all of them."""
    return {
        "direct_method": direct_method(samples),
        "ips": inverse_propensity(samples, clip),
        "snips": self_normalized_ips(samples, clip),
        "doubly_robust": doubly_robust(samples, clip),
        "switch_dr": switch_doubly_robust(samples, clip, weight_threshold),
    }
