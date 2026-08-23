from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from method4.bridge import OpeningContext


@dataclass
class ActionDistribution:
    """A routing decision as a DISTRIBUTION over models, not a single choice.

    This is the central representational change in method4. method3 returned one
    model; method4 returns P(model | opening) and samples from it. Two consequences
    that a deterministic router cannot have:

      1. The propensity of whatever action gets taken is KNOWN EXACTLY, not
         estimated. Any future off-policy evaluation of a later router against
         method4's logs uses exact propensities, removing propensity
         misspecification as an error source entirely.
      2. Every action retains positive probability (see `floor`), so the logs
         method4 generates keep the whole action space observable. A deterministic
         policy would make 67% of (context, model) cells permanently unobservable
         on dataset2, which silently destroys the ability to evaluate its own
         successor.
    """
    probabilities: dict[str, float]
    reason: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("action distribution must contain at least one model")
        total = sum(self.probabilities.values())
        if total <= 0:
            raise ValueError("action distribution must have positive total mass")
        if any(p < 0 for p in self.probabilities.values()):
            raise ValueError("action probabilities must be non-negative")
        if abs(total - 1.0) > 1e-9:  # normalize defensively rather than trusting callers
            self.probabilities = {m: p / total for m, p in self.probabilities.items()}

    @property
    def support(self) -> set[str]:
        return {m for m, p in self.probabilities.items() if p > 0}

    def probability_of(self, model: str) -> float:
        return self.probabilities.get(model, 0.0)

    def argmax(self) -> str:
        """The modal action. Used for reporting and for a deterministic-mode
        comparison against method3 — never for generating logs, since collapsing to
        the mode is exactly the behavior method4 exists to avoid."""
        return max(self.probabilities, key=lambda m: (self.probabilities[m], m))

    def entropy(self) -> float:
        import math
        return -sum(p * math.log(p) for p in self.probabilities.values() if p > 0)

    def sample(self, rng) -> str:
        models = sorted(self.probabilities)  # sorted -> reproducible for a given rng
        weights = [self.probabilities[m] for m in models]
        return rng.choices(models, weights=weights, k=1)[0]


@runtime_checkable
class StochasticPolicy(Protocol):
    """Like method3's `Router`, a policy may see the OpeningContext ONLY — never a
    Trajectory. method4 inherits that pre-decision boundary unchanged; it only
    changes what the decision returns."""

    def action_distribution(self, opening: OpeningContext, candidate_models: list[str]) -> ActionDistribution:
        ...
