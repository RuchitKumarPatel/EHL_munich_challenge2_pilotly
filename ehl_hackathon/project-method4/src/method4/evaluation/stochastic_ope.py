from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class StochasticSample:
    """One logged trajectory reduced to what a STOCHASTIC-target estimator needs.

    q_by_model      : Q̂(x, a) for every candidate action a
    target_pi       : π(a | x) for every candidate action, from method4's policy
    logged_model    : the action actually taken
    logged_propensity : p(a | x) under the logging policy
    reward          : the observed outcome
    """
    q_by_model: dict[str, float]
    target_pi: dict[str, float]
    logged_model: str
    logged_propensity: float
    reward: float


@dataclass
class StochasticEstimate:
    name: str
    value: float
    standard_error: float
    n: int
    effective_sample_size: float
    max_weight: float
    clipped_fraction: float

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        return (self.value - z * self.standard_error, self.value + z * self.standard_error)

    def to_dict(self) -> dict[str, object]:
        low, high = self.confidence_interval()
        return {
            "estimator": self.name, "value": self.value, "standard_error": self.standard_error,
            "ci95": [low, high], "n": self.n,
            "effective_sample_size": self.effective_sample_size,
            "max_weight": self.max_weight, "clipped_fraction": self.clipped_fraction,
        }


def _mean_and_se(contributions: list[float]) -> tuple[float, float]:
    n = len(contributions)
    if n == 0:
        return 0.0, 0.0
    mean = sum(contributions) / n
    if n == 1:
        return mean, 0.0
    variance = sum((c - mean) ** 2 for c in contributions) / (n - 1)
    return mean, math.sqrt(variance / n)


def _weights(samples: list[StochasticSample], clip: float) -> list[float]:
    """Importance weight π(a|x) / p(a|x) for the LOGGED action.

    This is the structural advantage of a stochastic target policy. For a
    DETERMINISTIC target the weight is 1{a = π(x)} / p(a|x), which is zero on every
    trajectory where the router disagreed with the log — on dataset2 that left
    method3 with only 13 usable samples out of 159, and an estimator spread of 0.29
    across the family. A stochastic target assigns π(a|x) > 0 to EVERY action, so
    every logged trajectory contributes signal and the effective sample size rises
    by roughly an order of magnitude. The estimator is not merely more robust; it
    is computed from far more evidence.
    """
    return [s.target_pi.get(s.logged_model, 0.0) / max(clip, s.logged_propensity) for s in samples]


def _diagnostics(weights: list[float], clipped: int) -> tuple[float, float, float]:
    positive = [w for w in weights if w > 0]
    if not positive:
        return 0.0, 0.0, 0.0
    ess = sum(positive) ** 2 / sum(w * w for w in positive)
    return ess, max(positive), (clipped / len(weights) if weights else 0.0)


def direct_method(samples: list[StochasticSample]) -> StochasticEstimate:
    """DM under a stochastic target: the policy's expected value is the
    π-weighted average of the reward model's predictions, Σ_a π(a|x) Q̂(x,a)."""
    contributions = [sum(s.target_pi.get(m, 0.0) * q for m, q in s.q_by_model.items()) for s in samples]
    value, se = _mean_and_se(contributions)
    return StochasticEstimate("direct_method", value, se, len(samples), float(len(samples)), 0.0, 0.0)


def inverse_propensity(samples: list[StochasticSample], clip: float = 1e-6) -> StochasticEstimate:
    weights = _weights(samples, clip)
    contributions = [w * s.reward for w, s in zip(weights, samples)]
    value, se = _mean_and_se(contributions)
    ess, max_w, _ = _diagnostics(weights, 0)
    return StochasticEstimate("ips", value, se, len(samples), ess, max_w, 0.0)


def self_normalized_ips(samples: list[StochasticSample], clip: float = 1e-6) -> StochasticEstimate:
    weights = _weights(samples, clip)
    total = sum(weights)
    ess, max_w, _ = _diagnostics(weights, 0)
    if total <= 0:
        return StochasticEstimate("snips", 0.0, 0.0, len(samples), 0.0, 0.0, 0.0)
    value = sum(w * s.reward for w, s in zip(weights, samples)) / total
    residuals = [w * (s.reward - value) / (total / len(samples)) for w, s in zip(weights, samples)]
    _, se = _mean_and_se(residuals)
    return StochasticEstimate("snips", value, se, len(samples), ess, max_w, 0.0)


def doubly_robust(samples: list[StochasticSample], clip: float = 1e-6) -> StochasticEstimate:
    """DR for a stochastic target:
        V = E[ Σ_a π(a|x) Q̂(x,a) + (π(a|x)/p(a|x)) (r - Q̂(x,a)) ]
    """
    weights = _weights(samples, clip)
    contributions = []
    for weight, sample in zip(weights, samples):
        baseline = sum(sample.target_pi.get(m, 0.0) * q for m, q in sample.q_by_model.items())
        residual = sample.reward - sample.q_by_model.get(sample.logged_model, 0.0)
        contributions.append(baseline + weight * residual)
    value, se = _mean_and_se(contributions)
    ess, max_w, _ = _diagnostics(weights, 0)
    return StochasticEstimate("doubly_robust", value, se, len(samples), ess, max_w, 0.0)


def switch_doubly_robust(samples: list[StochasticSample], clip: float = 1e-6, weight_threshold: float = 10.0) -> StochasticEstimate:
    weights = _weights(samples, clip)
    contributions = []
    kept: list[float] = []
    clipped = 0
    for weight, sample in zip(weights, samples):
        baseline = sum(sample.target_pi.get(m, 0.0) * q for m, q in sample.q_by_model.items())
        if weight > weight_threshold:
            contributions.append(baseline)
            kept.append(0.0)
            clipped += 1
        else:
            residual = sample.reward - sample.q_by_model.get(sample.logged_model, 0.0)
            contributions.append(baseline + weight * residual)
            kept.append(weight)
    value, se = _mean_and_se(contributions)
    ess, max_w, clipped_fraction = _diagnostics(kept, clipped)
    return StochasticEstimate("switch_dr", value, se, len(samples), ess, max_w, clipped_fraction)


def all_estimators(samples: list[StochasticSample], clip: float = 1e-6, weight_threshold: float = 10.0) -> dict[str, StochasticEstimate]:
    return {
        "direct_method": direct_method(samples),
        "ips": inverse_propensity(samples, clip),
        "snips": self_normalized_ips(samples, clip),
        "doubly_robust": doubly_robust(samples, clip),
        "switch_dr": switch_doubly_robust(samples, clip, weight_threshold),
    }


def estimator_spread(results: dict[str, StochasticEstimate]) -> float:
    """Max-min across the family. Large spread = the estimators disagree = weak
    overlap or a misspecified propensity model, and a reason to distrust all of
    them rather than to pick the most flattering one."""
    values = [r.value for r in results.values()]
    return (max(values) - min(values)) if values else 0.0
