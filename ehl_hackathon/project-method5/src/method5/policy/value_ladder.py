from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValueLadder:
    """Charnov's Marginal Value Theorem as an escalation ceiling.

    A forager leaves a patch when its marginal intake rate falls to the average
    rate of the habitat. Here: keep the cheaper model, and escalate only while the
    marginal quality bought per marginal dollar still beats what a dollar buys on
    average across the fleet.

    `habitat_rate` is MEASURED from logs (total realized quality / total realized
    spend), not guessed. That matters: it replaces a hand-set quality threshold,
    which had two observed failure modes — on dataset1 every model cleared the
    threshold so the cascade silently collapsed to always-cheapest, and being
    expressed in quality units it could not react to a price change at all.

    Carried over from project-method4 with one simplification: method4 also applied
    a price penalty *inside* the eligible band, which double-counted cost (costs
    span 50x while quality spans ~1.8x, so price dominated the score ~6:1 and
    collapsed 89/159 decisions onto the single cheapest model). The ladder sets the
    ceiling; selection inside the band is left to quality alone.
    """
    habitat_rate: float
    minimum_cost_gap: float = 1e-12

    @classmethod
    def fit(cls, qualities: list[float], costs: list[float]) -> "ValueLadder":
        """Aggregate ratio, deliberately not the mean of per-trajectory ratios — the
        latter lets one near-zero-cost trajectory dominate with an enormous ratio."""
        if len(qualities) != len(costs):
            raise ValueError("qualities and costs must be the same length")
        total_cost = sum(costs)
        if not qualities or total_cost <= 0:
            raise ValueError("cannot fit a habitat rate without positive total cost")
        return cls(habitat_rate=sum(qualities) / total_cost)

    def should_escalate(self, base_quality: float, base_cost: float,
                        candidate_quality: float, candidate_cost: float) -> bool:
        cost_gap = candidate_cost - base_cost
        if cost_gap <= self.minimum_cost_gap:
            return False
        return (candidate_quality - base_quality) / cost_gap >= self.habitat_rate

    def ceiling(self, options: dict[str, tuple[float, float]]) -> str:
        """Walk the cost-ordered ladder, escalating one rung at a time while MVT
        justifies it. Rung-by-rung (not best-vs-cheapest) is what makes this
        Charnov's rule: each step must independently pay for itself, so a model is
        never reached via one large jump that is really several unjustified small
        ones bundled together. Cost ties break toward higher quality."""
        if not options:
            raise ValueError("no options to choose from")
        ladder = sorted(options.items(), key=lambda item: (item[1][1], -item[1][0]))
        name, (quality, cost) = ladder[0]
        for candidate_name, (candidate_quality, candidate_cost) in ladder[1:]:
            if self.should_escalate(quality, cost, candidate_quality, candidate_cost):
                name, quality, cost = candidate_name, candidate_quality, candidate_cost
        return name

    def to_dict(self) -> dict[str, float]:
        return {"habitat_rate": self.habitat_rate, "minimum_cost_gap": self.minimum_cost_gap}

    @classmethod
    def from_dict(cls, payload: dict) -> "ValueLadder":
        return cls(float(payload["habitat_rate"]), float(payload.get("minimum_cost_gap", 1e-12)))
