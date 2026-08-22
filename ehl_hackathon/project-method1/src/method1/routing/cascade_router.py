from __future__ import annotations

from method1.data.schema import Trajectory

from .base import RouteDecision
from .difficulty_router import DifficultyRouter


class CascadeRouter:
    def __init__(self, router: DifficultyRouter, fallback_model: str, uncertainty_threshold: float = 0.55) -> None:
        self.router = router
        self.fallback_model = fallback_model
        self.uncertainty_threshold = uncertainty_threshold

    def route(self, trajectory: Trajectory) -> RouteDecision:
        decision = self.router.route(trajectory)
        if decision.uncertainty <= self.uncertainty_threshold:
            return decision
        cost = decision.costs[self.fallback_model]
        return RouteDecision(self.fallback_model, [self.fallback_model] * trajectory.n_calls, decision.scores.get(self.fallback_model, 0.5), cost, decision.uncertainty, decision.scores, decision.costs)

