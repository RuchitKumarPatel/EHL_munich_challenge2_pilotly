from __future__ import annotations

import json
from pathlib import Path

from method1.data.schema import Trajectory
from method1.pricing.cost_model import CostModel
from method1.quality.outcome_model import composite_outcome
from method1.routing.base import Router

from .confidence_intervals import bootstrap_mean
from .doubly_robust import evaluate_doubly_robust
from .metrics import PolicyMetrics


def evaluate_policy(name: str, trajectories: list[Trajectory], router: Router, cost_model: CostModel, outcomes: dict[str, float] | None = None) -> tuple[PolicyMetrics, list[dict[str, object]]]:
    rows = []
    qualities = []
    for trajectory in trajectories:
        decision = router.route(trajectory)
        cost, cache = cost_model.trajectory_cost(trajectory, decision.route)
        observed_quality = (outcomes or {}).get(trajectory.key, composite_outcome(trajectory))
        quality = observed_quality if decision.model == trajectory.logged_model else decision.expected_quality
        qualities.append(quality)
        rows.append({"trajectory": trajectory.key, "logged_model": trajectory.logged_model, "model": decision.model, "cost": cost, "quality": quality, "quality_source": "observed_proxy" if decision.model == trajectory.logged_model else "counterfactual_model_prediction", "uncertainty": decision.uncertainty, **cache})
    mean_quality, _, _ = bootstrap_mean(qualities)
    metrics = PolicyMetrics(name, len(trajectories), sum(trajectory.n_calls for trajectory in trajectories), sum(float(row["cost"]) for row in rows), mean_quality, sum(float(row["uncertainty"]) for row in rows) / max(1, len(rows)), sum(float(row["cached_tokens"]) for row in rows), sum(float(row["uncached_tokens"]) for row in rows), sum(float(row["switches"]) for row in rows))
    return metrics, rows


def write_policy_report(path: str | Path, metrics: list[PolicyMetrics], rows: dict[str, list[dict[str, object]]]) -> None:
    payload = {"policies": [item.to_dict() for item in metrics], "rows": rows}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
