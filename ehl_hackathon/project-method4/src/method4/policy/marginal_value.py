from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarginalValueRule:
    """Charnov's Marginal Value Theorem (1976) as an escalation rule.

    A forager should leave its current patch when the MARGINAL intake rate there
    drops to the AVERAGE intake rate of the habitat. Translated to routing: keep
    using the cheaper model, and escalate to a costlier one only while the marginal
    quality bought per marginal dollar still beats what a dollar buys on average
    across the whole fleet.

        escalate m1 -> m2  iff  (Q(m2) - Q(m1)) / (c(m2) - c(m1))  >=  habitat_rate

    This replaces method3's hand-set `quality_floor = 0.5`, which was a magic number
    with two observed failure modes: on dataset1 every model cleared it, so the
    cascade silently collapsed to "always cheapest" and the escalation branch never
    executed; and it is expressed in quality units, so it cannot react to a price
    change at all (dataset2 moves prices 20% between eras).

    `habitat_rate` is not guessed — it is MEASURED from the logs as total realized
    quality over total realized spend, which makes the rule automatically:
      * budget-aware  (spend more per unit quality -> rate falls -> escalate less),
      * price-aware   (prices drop -> a dollar buys more -> rate rises),
      * self-calibrating to whatever fleet it is deployed on.

    Units: `habitat_rate` is quality-per-dollar. Costs must be passed in the same
    dollar units used to fit it (this project keeps method3's convention).
    """
    habitat_rate: float
    # Guards a degenerate comparison: two models whose costs are equal (or within
    # noise) have an undefined marginal rate. Treated as "no escalation justified",
    # since paying the same for either means the cheaper-tie should win on price.
    minimum_cost_gap: float = 1e-12

    @classmethod
    def fit(cls, qualities: list[float], costs: list[float]) -> "MarginalValueRule":
        """Estimate the habitat rate from logged outcomes: total quality / total cost.

        Aggregate ratio, deliberately NOT the mean of per-trajectory ratios — the
        latter lets a single near-zero-cost trajectory dominate with an enormous
        ratio. Total-over-total is the rate a forager actually experiences.
        """
        if len(qualities) != len(costs):
            raise ValueError("qualities and costs must be the same length")
        total_cost = sum(costs)
        if not qualities or total_cost <= 0:
            raise ValueError("cannot fit a habitat rate without positive total cost")
        return cls(habitat_rate=sum(qualities) / total_cost)

    def marginal_rate(self, quality_gain: float, cost_gap: float) -> float | None:
        """Quality bought per extra dollar. None when the cost gap is degenerate."""
        if cost_gap <= self.minimum_cost_gap:
            return None
        return quality_gain / cost_gap

    def should_escalate(self, base_quality: float, base_cost: float,
                        candidate_quality: float, candidate_cost: float) -> bool:
        """True iff moving from base to candidate is justified by the MVT rule."""
        rate = self.marginal_rate(candidate_quality - base_quality, candidate_cost - base_cost)
        if rate is None:
            return False
        return rate >= self.habitat_rate

    def best_escalation(self, options: dict[str, tuple[float, float]]) -> str:
        """Pick a model from {name: (quality, cost)} by walking the cost-ordered
        ladder and escalating one rung at a time while MVT still justifies it.

        Walking the ladder (rather than comparing every model to the cheapest) is
        what makes this Charnov's rule rather than a global argmax: each step must
        independently pay for itself, so a model is never selected on the strength
        of a large jump that is really several unjustified small ones bundled
        together. Ties in cost are broken toward higher quality.
        """
        if not options:
            raise ValueError("no options to choose from")
        ladder = sorted(options.items(), key=lambda item: (item[1][1], -item[1][0]))
        current_name, (current_quality, current_cost) = ladder[0]
        for name, (quality, cost) in ladder[1:]:
            if self.should_escalate(current_quality, current_cost, quality, cost):
                current_name, current_quality, current_cost = name, quality, cost
        return current_name

    def to_dict(self) -> dict[str, float]:
        return {"habitat_rate": self.habitat_rate, "minimum_cost_gap": self.minimum_cost_gap}

    @classmethod
    def from_dict(cls, payload: dict) -> "MarginalValueRule":
        return cls(float(payload["habitat_rate"]), float(payload.get("minimum_cost_gap", 1e-12)))
