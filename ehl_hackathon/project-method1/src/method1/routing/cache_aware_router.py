from __future__ import annotations

from method1.data.schema import Trajectory

from .base import RouteDecision
from .difficulty_router import DifficultyRouter
from .uncertainty import score_uncertainty


class CacheAwareRouter:
    def __init__(self, predictor: DifficultyRouter, quality_target: float | None = None, cost_weight: float = 0.15, uncertainty_threshold: float = 0.7) -> None:
        self.predictor = predictor
        self.quality_target = quality_target
        self.cost_weight = cost_weight
        self.uncertainty_threshold = uncertainty_threshold

    def route(self, trajectory: Trajectory) -> RouteDecision:
        scores = self.predictor.predict_scores(trajectory)
        costs = {model: self.predictor.cost_model.trajectory_cost(trajectory, [model] * trajectory.n_calls)[0] for model in self.predictor.models}
        candidates = [model for model in self.predictor.models if self.quality_target is None or scores[model] >= self.quality_target]
        if not candidates:
            candidates = list(self.predictor.models)
        scale = max(costs.values()) if costs else 1.0
        utility = {model: scores[model] - self.cost_weight * costs[model] / max(scale, 1e-9) for model in candidates}
        model = max(utility, key=utility.get)
        uncertainty = score_uncertainty(scores)
        if uncertainty >= self.uncertainty_threshold:
            model = max(candidates, key=lambda value: scores[value])
        return RouteDecision(model, [model] * trajectory.n_calls, scores[model], costs[model], uncertainty, scores, costs)

