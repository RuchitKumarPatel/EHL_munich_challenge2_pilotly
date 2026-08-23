from __future__ import annotations

from method1.data.schema import Trajectory
from method1.pricing.cost_model import CostModel

from .base import RouteDecision
from .uncertainty import score_uncertainty


MODEL_STRENGTH = {
    "claude-fable-5": 0.45,
    "claude-sonnet-5": 0.7,
    "claude-opus-4-8": 0.9,
    "claude-opus-5": 1.0,
    "gpt-5.6-sol": 0.45,
    "gpt-5.6-luna": 0.65,
    "gpt-5.6-terra": 0.85,
}


class StaticRouter:
    def __init__(self, model: str, cost_model: CostModel) -> None:
        self.model = model
        self.cost_model = cost_model

    def route(self, trajectory: Trajectory) -> RouteDecision:
        route = [self.model] * trajectory.n_calls
        cost, _ = self.cost_model.trajectory_cost(trajectory, route)
        strength = MODEL_STRENGTH.get(self.model, 0.5)
        return RouteDecision(self.model, route, strength, cost, 0.0, {self.model: strength}, {self.model: cost})


class LoggedRouter:
    def __init__(self, cost_model: CostModel) -> None:
        self.cost_model = cost_model

    def route(self, trajectory: Trajectory) -> RouteDecision:
        # Replay the real per-call model sequence rather than assuming one model for
        # the whole trajectory: trajectory.logged_model is "mixed" whenever the logged
        # session actually switched models mid-way, and pricing has no entry for the
        # literal string "mixed". Per-call replay also feeds CostModel's existing
        # switch-detection so a mid-session switch correctly loses cache credit.
        model = trajectory.logged_model
        route = [call.model for call in trajectory.calls]
        cost, _ = self.cost_model.trajectory_cost(trajectory, route)
        strengths = [MODEL_STRENGTH.get(m, 0.5) for m in route]
        score = sum(strengths) / len(strengths)
        return RouteDecision(model, route, score, cost, 0.0, {model: score}, {model: cost})


class CheapestRouter:
    def __init__(self, models: list[str], cost_model: CostModel) -> None:
        self.models = models
        self.cost_model = cost_model

    def route(self, trajectory: Trajectory) -> RouteDecision:
        options = {}
        for model in self.models:
            options[model] = self.cost_model.trajectory_cost(trajectory, [model] * trajectory.n_calls)[0]
        model = min(options, key=options.get)
        return RouteDecision(model, [model] * trajectory.n_calls, MODEL_STRENGTH.get(model, 0.5), options[model], score_uncertainty({key: MODEL_STRENGTH.get(key, 0.5) for key in self.models}), {key: MODEL_STRENGTH.get(key, 0.5) for key in self.models}, options)


class StrongestRouter:
    def __init__(self, models: list[str], cost_model: CostModel) -> None:
        self.models = models
        self.cost_model = cost_model

    def route(self, trajectory: Trajectory) -> RouteDecision:
        model = max(self.models, key=lambda value: MODEL_STRENGTH.get(value, 0.5))
        cost = self.cost_model.trajectory_cost(trajectory, [model] * trajectory.n_calls)[0]
        scores = {key: MODEL_STRENGTH.get(key, 0.5) for key in self.models}
        return RouteDecision(model, [model] * trajectory.n_calls, scores[model], cost, score_uncertainty(scores), scores, {key: self.cost_model.trajectory_cost(trajectory, [key] * trajectory.n_calls)[0] for key in self.models})

