from __future__ import annotations

from dataclasses import dataclass

from .cost import trajectory_cost, trajectory_cost_route
from .data import Trajectory
from .labels import observed_outcome
from .reward import CalibratedRewardModel, Estimate


@dataclass
class Decision:
    model: str
    cost: float
    estimate: Estimate
    factual: bool
    reason: str
    candidates: dict[str, dict[str, float]]


class SafeRouter:
    def __init__(self, reward: CalibratedRewardModel, prices: dict[str, tuple[float, float]], allowed_quality_loss: float = 0.02) -> None:
        self.reward = reward
        self.prices = prices
        self.allowed_quality_loss = allowed_quality_loss

    def route(self, trajectory: Trajectory, logged_model: str | None = None) -> Decision:
        baseline_model = logged_model if logged_model in self.reward.models else None
        baseline = self.reward.predict(trajectory, baseline_model) if baseline_model else None
        values = {}
        for model in self.reward.models:
            estimate = self.reward.predict(trajectory, model)
            cost = trajectory_cost(trajectory, model, self.prices)
            values[model] = {"mean": estimate.mean, "lower": estimate.lower, "uncertainty": estimate.uncertainty, "support": estimate.support, "distance": estimate.distance, "cost": cost}
        if baseline_model and baseline and baseline.support < self.reward.minimum_support:
            return Decision(baseline_model, values[baseline_model]["cost"], baseline, True, "insufficient_logged_support", values)
        eligible = []
        threshold = baseline.lower - self.allowed_quality_loss if baseline else 0.0
        for model, row in values.items():
            saves_cost = baseline is None or row["cost"] < values[baseline_model]["cost"] - 1e-12
            if row["support"] >= self.reward.minimum_support and row["lower"] >= threshold and saves_cost:
                eligible.append(model)
        if eligible:
            selected = min(eligible, key=lambda model: values[model]["cost"])
            estimate = self.reward.predict(trajectory, selected)
            factual = selected == logged_model
            reason = "supported_cheapest_route" if factual or selected != logged_model else "logged_fallback"
            return Decision(selected, values[selected]["cost"], estimate, factual, reason, values)
        if baseline_model:
            return Decision(baseline_model, values[baseline_model]["cost"], baseline, True, "insufficient_counterfactual_support", values)
        selected = min(self.reward.models, key=lambda model: values[model]["cost"])
        return Decision(selected, values[selected]["cost"], self.reward.predict(trajectory, selected), False, "no_logged_baseline", values)


class FixedRouter:
    def __init__(self, reward: CalibratedRewardModel, prices: dict[str, tuple[float, float]], mode: str,
                 ground_truth: dict[str, float] | None = None) -> None:
        self.reward = reward
        self.prices = prices
        self.mode = mode
        self.ground_truth = ground_truth

    def route(self, trajectory: Trajectory, logged_model: str | None = None) -> Decision:
        if self.mode == "logged" and logged_model in self.reward.models:
            model = logged_model
            reason = "logged_baseline"
        elif self.mode == "logged":
            # logged_model not a concrete model (e.g. "mixed": the logged session
            # actually switched models mid-way). Replay the real per-call trace instead
            # of a candidate model chosen by our own heuristic, and price the switch's
            # cache reset via trajectory_cost_route. Quality is the observed/ground-truth
            # outcome of what actually happened, not a reward-model prediction (there is
            # no single model to predict a reward for).
            route = [call.model for call in trajectory.calls]
            cost = trajectory_cost_route(trajectory, route, self.prices)
            score, confidence = observed_outcome(trajectory, self.ground_truth)
            candidates = {}
            for candidate in self.reward.models:
                est = self.reward.predict(trajectory, candidate)
                candidates[candidate] = {"mean": est.mean, "lower": est.lower, "uncertainty": est.uncertainty, "support": est.support, "distance": est.distance, "cost": trajectory_cost(trajectory, candidate, self.prices)}
            estimate = Estimate(mean=score, lower=max(0.0, score - (1.0 - confidence)), uncertainty=1.0 - confidence, support=1.0, distance=0.0)
            return Decision(logged_model or "mixed", cost, estimate, True, "logged_baseline_mixed", candidates)
        elif self.mode == "cheapest":
            model = min(self.reward.models, key=lambda item: trajectory_cost(trajectory, item, self.prices))
            reason = "cheapest_baseline"
        else:
            model = self.mode
            reason = "static_baseline"
        candidates = {}
        for candidate in self.reward.models:
            estimate = self.reward.predict(trajectory, candidate)
            candidates[candidate] = {"mean": estimate.mean, "lower": estimate.lower, "uncertainty": estimate.uncertainty, "support": estimate.support, "distance": estimate.distance, "cost": trajectory_cost(trajectory, candidate, self.prices)}
        estimate = self.reward.predict(trajectory, model)
        return Decision(model, trajectory_cost(trajectory, model, self.prices), estimate, model == logged_model, reason, candidates)
