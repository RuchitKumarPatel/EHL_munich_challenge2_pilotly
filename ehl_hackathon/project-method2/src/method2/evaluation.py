from __future__ import annotations

import json
import random
from pathlib import Path

from .cost import trajectory_cost
from .data import Trajectory
from .labels import observed_outcome
from .router import SafeRouter


def _interval(values: list[float], seed: int = 7) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    randomizer = random.Random(seed)
    samples = []
    for _ in range(500):
        samples.append(sum(values[randomizer.randrange(len(values))] for _ in values) / len(values))
    samples.sort()
    return sum(values) / len(values), samples[12], samples[487]


def evaluate(name: str, trajectories: list[Trajectory], router: SafeRouter, ground_truth: dict[str, float] | None = None) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    estimates = []
    lower_bounds = []
    factual_outcomes = []
    costs = []
    for trajectory in trajectories:
        decision = router.route(trajectory, trajectory.model)
        observed, confidence = observed_outcome(trajectory, ground_truth)
        factual = decision.model == trajectory.model
        rows.append({"trajectory": trajectory.key, "logged_model": trajectory.model, "selected_model": decision.model, "cost": decision.cost, "quality_estimate": decision.estimate.mean, "quality_lower_bound": decision.estimate.lower, "uncertainty": decision.estimate.uncertainty, "support": decision.estimate.support, "factual": factual, "observed_quality": observed if factual else None, "observed_confidence": confidence if factual else None, "reason": decision.reason})
        estimates.append(decision.estimate.mean)
        lower_bounds.append(decision.estimate.lower)
        costs.append(decision.cost)
        if factual:
            factual_outcomes.append(observed)
    quality_mean, quality_low, quality_high = _interval(estimates)
    lower_mean, _, _ = _interval(lower_bounds)
    observed_mean, observed_low, observed_high = _interval(factual_outcomes)
    return {"policy": name, "trajectories": len(trajectories), "cost": sum(costs), "quality_estimate": quality_mean, "quality_estimate_ci": [quality_low, quality_high], "quality_lower_bound": lower_mean, "factual_coverage": len(factual_outcomes) / max(1, len(trajectories)), "factual_observed_quality": observed_mean if factual_outcomes else None, "factual_observed_quality_ci": [observed_low, observed_high] if factual_outcomes else None}, rows


def write(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

