from __future__ import annotations

from dataclasses import dataclass

from method3.data.schema import Trajectory
from method3.pricing.cost_model import CostModel

# Rough, hand-set capability ranking — same convention as project-method1's
# baselines.py, used only to pick the "strongest" reference baseline. Not learned,
# not used by the proposed CascadeRouter (which uses the fitted reward model instead).
MODEL_STRENGTH: dict[str, float] = {
    "claude-fable-5": 0.45, "claude-sonnet-5": 0.70, "claude-opus-4-8": 0.90, "claude-opus-5": 1.00,
    "gpt-5.6-sol": 0.45, "gpt-5.6-luna": 0.65, "gpt-5.6-terra": 0.85,
}


@dataclass
class BaselineDecision:
    model: str          # "mixed" for the logged baseline on a switched trajectory
    route: list[str]    # one model per call — the actual repricing input
    cost: float
    reason: str


class BaselineOracle:
    """Reference policies for the Pareto frontier: logged / cheapest / strongest /
    static. Deliberately NOT a `Router` — these need the FULL trajectory (to know its
    real token profile / call count) to be costed, which a real routing decision does
    not have available. They are honest oracle baselines for comparison, not claims
    of a deployable pre-decision policy; only CascadeRouter (routing/cascade_router.py)
    makes that claim, and only it is required to satisfy the Router protocol."""

    def __init__(self, candidate_models: list[str], cost_model: CostModel) -> None:
        self.candidate_models = candidate_models
        self.cost_model = cost_model

    def logged(self, trajectory: Trajectory) -> BaselineDecision:
        route = trajectory.per_call_models  # real per-call sequence; safe for "mixed"
        cost, _ = self.cost_model.trajectory_cost(trajectory, route)
        return BaselineDecision(trajectory.logged_model, route, cost, "logged_baseline")

    def cheapest(self, trajectory: Trajectory) -> BaselineDecision:
        options = {model: self.cost_model.trajectory_cost(trajectory, [model] * trajectory.n_calls)[0] for model in self.candidate_models}
        model = min(options, key=options.get)
        return BaselineDecision(model, [model] * trajectory.n_calls, options[model], "cheapest_baseline")

    def strongest(self, trajectory: Trajectory) -> BaselineDecision:
        model = max(self.candidate_models, key=lambda m: MODEL_STRENGTH.get(m, 0.5))
        cost, _ = self.cost_model.trajectory_cost(trajectory, [model] * trajectory.n_calls)
        return BaselineDecision(model, [model] * trajectory.n_calls, cost, "strongest_baseline")

    def static(self, trajectory: Trajectory, model: str) -> BaselineDecision:
        cost, _ = self.cost_model.trajectory_cost(trajectory, [model] * trajectory.n_calls)
        return BaselineDecision(model, [model] * trajectory.n_calls, cost, "static_baseline")
