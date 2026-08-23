from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Sample:
    """One logged trajectory reduced to what a stochastic-target estimator needs."""
    q_by_model: dict[str, float]      # Q̂(x, a) for every candidate action
    target_pi: dict[str, float]       # π(a | x) under the policy being evaluated
    logged_model: str                 # the action actually taken
    logged_propensity: float          # p(a | x) under the logging policy
    reward: float                     # the observed outcome


@dataclass
class Result:
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


def _weights(samples: list[Sample], clip: float) -> list[float]:
    """Importance weight π(a|x)/p(a|x) on the logged action.

    A stochastic target assigns positive π to every action, so EVERY logged
    trajectory contributes signal. A deterministic target's weight is
    1{a = π(x)}/p(a|x), which is zero wherever the router disagreed with the log —
    measured on dataset2 that left project-method3 with 13 usable samples out of
    159. This is the structural reason a stochastic policy is easier to evaluate,
    not merely safer.
    """
    return [s.target_pi.get(s.logged_model, 0.0) / max(clip, s.logged_propensity) for s in samples]


def _diagnostics(weights: list[float], clipped: int) -> tuple[float, float, float]:
    positive = [w for w in weights if w > 0]
    if not positive:
        return 0.0, 0.0, 0.0
    ess = sum(positive) ** 2 / sum(w * w for w in positive)  # Kish effective sample size
    return ess, max(positive), (clipped / len(weights) if weights else 0.0)


def _baseline(sample: Sample) -> float:
    return sum(sample.target_pi.get(m, 0.0) * q for m, q in sample.q_by_model.items())


def direct_method(samples: list[Sample]) -> Result:
    contributions = [_baseline(s) for s in samples]
    value, se = _mean_and_se(contributions)
    return Result("direct_method", value, se, len(samples), float(len(samples)), 0.0, 0.0)


def inverse_propensity(samples: list[Sample], clip: float = 1e-6) -> Result:
    weights = _weights(samples, clip)
    value, se = _mean_and_se([w * s.reward for w, s in zip(weights, samples)])
    ess, max_w, _ = _diagnostics(weights, 0)
    return Result("ips", value, se, len(samples), ess, max_w, 0.0)


def self_normalized_ips(samples: list[Sample], clip: float = 1e-6) -> Result:
    weights = _weights(samples, clip)
    total = sum(weights)
    ess, max_w, _ = _diagnostics(weights, 0)
    if total <= 0:
        return Result("snips", 0.0, 0.0, len(samples), 0.0, 0.0, 0.0)
    value = sum(w * s.reward for w, s in zip(weights, samples)) / total
    residuals = [w * (s.reward - value) / (total / len(samples)) for w, s in zip(weights, samples)]
    _, se = _mean_and_se(residuals)
    return Result("snips", value, se, len(samples), ess, max_w, 0.0)


def doubly_robust(samples: list[Sample], clip: float = 1e-6) -> Result:
    """V = E[ Σ_a π(a|x)Q̂(x,a) + (π(a|x)/p(a|x))(r - Q̂(x,a)) ].
    Consistent if EITHER the propensity or the reward model is correct."""
    weights = _weights(samples, clip)
    contributions = [
        _baseline(s) + w * (s.reward - s.q_by_model.get(s.logged_model, 0.0))
        for w, s in zip(weights, samples)
    ]
    value, se = _mean_and_se(contributions)
    ess, max_w, _ = _diagnostics(weights, 0)
    return Result("doubly_robust", value, se, len(samples), ess, max_w, 0.0)


def switch_doubly_robust(samples: list[Sample], clip: float = 1e-6, weight_threshold: float = 10.0) -> Result:
    """Use the DR correction only while the weight is reasonable; above the
    threshold fall back to the direct method for that sample. Extreme weights are
    where DR's correction does more harm than good."""
    weights = _weights(samples, clip)
    contributions, kept, clipped = [], [], 0
    for weight, sample in zip(weights, samples):
        if weight > weight_threshold:
            contributions.append(_baseline(sample))
            kept.append(0.0)
            clipped += 1
        else:
            contributions.append(_baseline(sample) + weight * (sample.reward - sample.q_by_model.get(sample.logged_model, 0.0)))
            kept.append(weight)
    value, se = _mean_and_se(contributions)
    ess, max_w, clipped_fraction = _diagnostics(kept, clipped)
    return Result("switch_dr", value, se, len(samples), ess, max_w, clipped_fraction)


def all_estimators(samples: list[Sample], clip: float = 1e-6, weight_threshold: float = 10.0) -> dict[str, Result]:
    """The whole family. Agreement is evidence the estimate is real; a wide spread —
    especially DM vs IPS — signals propensity misspecification or weak overlap and
    is a reason to distrust all of them rather than to pick the most flattering."""
    return {
        "direct_method": direct_method(samples),
        "ips": inverse_propensity(samples, clip),
        "snips": self_normalized_ips(samples, clip),
        "doubly_robust": doubly_robust(samples, clip),
        "switch_dr": switch_doubly_robust(samples, clip, weight_threshold),
    }


def spread(results: dict[str, Result]) -> float:
    values = [r.value for r in results.values()]
    return (max(values) - min(values)) if values else 0.0
