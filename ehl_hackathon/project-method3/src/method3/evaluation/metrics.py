from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyMetrics:
    policy: str
    trajectories: int
    calls: int
    cost: float
    quality: float
    quality_ci: tuple[float, float]
    factual_coverage: float
    cached_tokens: float
    uncached_tokens: float
    switches: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy, "trajectories": self.trajectories, "calls": self.calls,
            "cost": self.cost, "quality": self.quality, "quality_ci": list(self.quality_ci),
            "factual_coverage": self.factual_coverage,
            "cached_tokens": self.cached_tokens, "uncached_tokens": self.uncached_tokens, "switches": self.switches,
        }
