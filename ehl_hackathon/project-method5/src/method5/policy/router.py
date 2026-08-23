from __future__ import annotations

from method5.bridge import OpeningContext
from method5.policy.base import ActionDistribution, mix_with_uniform, softmax
from method5.policy.uncertainty import UncertaintyGate, signal_from_bounds
from method5.policy.value_ladder import ValueLadder


class AdaptiveRouter:
    """method5's policy: MVT ceiling, quality-driven selection, uncertainty-gated exploration.

    Pipeline for one opening context:

      1. Calibrated quality bounds per candidate (project-method3's KNN reward model
         + split-conformal calibrator, reused unchanged through the bridge).
      2. `ValueLadder` sets the ESCALATION CEILING — the costliest model whose every
         rung up the price ladder paid for itself against the measured habitat rate.
         Anything above it is ineligible. Price is consulted HERE and only here.
      3. Softmax over calibrated LOWER BOUNDS within the eligible band. Deliberately
         quality-only: the ladder has already settled the price question, and
         re-applying a price penalty inside the band double-counts cost — the bug
         that cost project-method4 ~0.05 quality by collapsing 89/159 decisions onto
         the single cheapest model.
      4. `UncertaintyGate` sets epsilon FOR THIS CONTEXT from the conformal interval
         width and the decision margin, and that epsilon mixes in a uniform
         distribution over every candidate. This is the one genuinely new mechanism:
         exploration is concentrated where it is informative instead of being spread
         at a flat rate, so the quality tax is paid only where something is actually
         learned.

    Conforms to `StochasticPolicy`: sees an OpeningContext only, never a Trajectory.
    """

    def __init__(
        self,
        reward_model,
        calibrators: dict,
        pricing: dict[str, dict[str, float]],
        value_ladder: ValueLadder,
        gate: UncertaintyGate | None = None,
        temperature: float = 0.05,
        minimum_support: float = 3.0,
    ) -> None:
        self.reward_model = reward_model
        self.calibrators = calibrators
        self.pricing = pricing
        self.value_ladder = value_ladder
        self.gate = gate or UncertaintyGate()
        self.temperature = temperature
        self.minimum_support = minimum_support

    def _bounds(self, opening: OpeningContext, model: str) -> tuple[float, float, float]:
        """(lower, upper, support) for one model.

        A model whose own calibrator cannot certify coverage falls back to the
        pooled `__global__` calibrator; if neither is certified the lower bound is
        0.0, so an uncertified model can never win the eligible band on a bound it
        has not earned. The upper bound mirrors the lower one about the point
        estimate, giving a symmetric interval whose WIDTH is what the gate consumes.
        """
        estimate = self.reward_model.predict(opening, model)
        calibrator = self.calibrators.get(model)
        pooled = self.calibrators.get("__global__")
        chosen = calibrator if (calibrator is not None and calibrator.coverage_guaranteed) else None
        if chosen is None and pooled is not None and pooled.coverage_guaranteed:
            chosen = pooled
        if chosen is None:
            return 0.0, 1.0, estimate.support   # no certified bound: maximally uncertain
        lower = chosen.lower_bound(estimate.mean)
        half_width = max(0.0, estimate.mean - lower)
        return lower, min(1.0, estimate.mean + half_width), estimate.support

    def action_distribution(self, opening: OpeningContext, candidate_models: list[str]) -> ActionDistribution:
        if not candidate_models:
            raise ValueError("no candidate models")

        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        support: dict[str, float] = {}
        price: dict[str, float] = {}
        for model in candidate_models:
            low, high, sup = self._bounds(opening, model)
            lower[model], upper[model], support[model] = low, high, sup
            price[model] = self.pricing[model]["input"]

        supported = {m for m in candidate_models if support[m] >= self.minimum_support}
        ladder_pool = supported or set(candidate_models)

        ceiling_model = self.value_ladder.ceiling({m: (lower[m], price[m]) for m in ladder_pool})
        ceiling_price = price[ceiling_model]
        eligible = {m for m in ladder_pool if price[m] <= ceiling_price}

        sharp = softmax({m: lower[m] for m in eligible}, self.temperature)

        signal = signal_from_bounds(lower, upper, eligible)
        epsilon = self.gate.epsilon(signal)
        probabilities = mix_with_uniform(sharp, candidate_models, epsilon)

        distribution = ActionDistribution(probabilities, reason=f"adaptive:ceiling={ceiling_model}")
        distribution.diagnostics = {
            "epsilon": epsilon,
            "interval_width": signal.interval_width,
            "decision_margin": signal.decision_margin,
            "ceiling_price": ceiling_price,
            "n_supported": float(len(supported)),
            "n_eligible": float(len(eligible)),
            "entropy": distribution.entropy(),
            "min_probability": min(probabilities.values()),
        }
        return distribution
