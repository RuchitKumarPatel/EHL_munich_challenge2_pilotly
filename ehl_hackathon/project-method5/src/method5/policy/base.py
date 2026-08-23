from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from method5.bridge import OpeningContext


@dataclass
class ActionDistribution:
    """A routing decision as a distribution over models.

    Returning a distribution rather than a single model has two consequences that a
    deterministic router cannot have: the propensity of whatever action is taken is
    KNOWN EXACTLY rather than estimated (removing propensity misspecification as an
    error source in any later off-policy evaluation), and every action keeps
    positive probability so the logs stay valid for evaluating the policy's own
    successor.
    """
    probabilities: dict[str, float]
    reason: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("action distribution must contain at least one model")
        if any(p < 0 for p in self.probabilities.values()):
            raise ValueError("action probabilities must be non-negative")
        total = sum(self.probabilities.values())
        if total <= 0:
            raise ValueError("action distribution must have positive total mass")
        if abs(total - 1.0) > 1e-12:
            self.probabilities = {m: p / total for m, p in self.probabilities.items()}

    def probability_of(self, model: str) -> float:
        return self.probabilities.get(model, 0.0)

    def argmax(self) -> str:
        """Modal action. For reporting and deterministic-mode comparison only —
        never for generating logs, since collapsing to the mode is exactly the
        behavior the exploration floor exists to prevent."""
        return max(self.probabilities, key=lambda m: (self.probabilities[m], m))

    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.probabilities.values() if p > 0)

    def sample(self, rng) -> str:
        models = sorted(self.probabilities)  # sorted -> reproducible for a seeded rng
        return rng.choices(models, weights=[self.probabilities[m] for m in models], k=1)[0]


@runtime_checkable
class StochasticPolicy(Protocol):
    """A policy sees the OpeningContext ONLY — never a Trajectory.

    This is project-method3's pre-decision boundary, inherited unchanged. A routing
    decision must fire before the trajectory runs, so it cannot be allowed to read
    hindsight statistics (final call count, whether errors occurred). Enforcing it
    at the type level makes the leak structurally impossible rather than a matter of
    discipline. See tests/test_no_feature_leakage.py.
    """

    def action_distribution(self, opening: OpeningContext, candidate_models: list[str]) -> ActionDistribution:
        ...


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    """Numerically stable softmax (max subtracted before exponentiating)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        raise ValueError("no scores to normalize")
    top = max(scores.values())
    exponentials = {m: math.exp((s - top) / temperature) for m, s in scores.items()}
    total = sum(exponentials.values())
    return {m: v / total for m, v in exponentials.items()}


def mix_with_uniform(sharp: dict[str, float], all_models: list[str], epsilon: float) -> dict[str, float]:
    """Blend a sharp distribution with a uniform one over EVERY candidate.

    The uniform component spans all candidates — including ones the sharp component
    assigns no mass to — which is what guarantees strict positivity and therefore
    keeps every (context, action) cell observable in the resulting logs.
    """
    if not all_models:
        raise ValueError("no candidate models")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    uniform = 1.0 / len(all_models)
    return {m: (1.0 - epsilon) * sharp.get(m, 0.0) + epsilon * uniform for m in all_models}
