from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from method3.data.schema import Trajectory
from method3.pricing.cost_model import CostModel
from method3.quality.reward_model import KNNRewardModel
from method3.routing.base import Router
from method3.routing.cascade_router import CascadeRouter

from .confidence_intervals import bootstrap_mean
from .metrics import PolicyMetrics

Decide = Callable[[Trajectory], tuple[str, list[str]]]


def router_decide(router: Router, candidate_models: list[str]) -> Decide:
    """Adapt a `Router` (opening-context-only) into the `Decide` shape this module's
    evaluate_policy expects. This is the ONLY place a Router's decision is combined
    with the full trajectory — and only to reprice/grade what it chose, never to feed
    trajectory information back into the decision itself."""
    def decide(trajectory: Trajectory) -> tuple[str, list[str]]:
        decision = router.route(trajectory.opening, candidate_models)
        return decision.model, [decision.model] * trajectory.n_calls
    return decide


def evaluate_policy(
    name: str,
    trajectories: list[Trajectory],
    decide: Decide,
    cost_model: CostModel,
    reward_model: KNNRewardModel,
    outcomes: dict[str, float],
) -> tuple[PolicyMetrics, list[dict[str, object]]]:
    """Post-hoc grading of a policy's decisions against the real logged trajectories.
    `outcomes` must have full coverage (every trajectory graded — ground truth or
    structural proxy, see quality.labels.calibrated_outcome) since a factual route's
    quality is read straight from it; a non-factual (counterfactual) route's quality
    comes from `reward_model`'s prediction for the chosen model, keeping every
    policy — baselines and the proposed router alike — graded through the same
    quality signal instead of two incompatible conventions."""
    rows = []
    qualities = []
    for trajectory in trajectories:
        model, route = decide(trajectory)
        cost, diagnostics = cost_model.trajectory_cost(trajectory, route)
        factual = route == trajectory.per_call_models
        if factual:
            quality = outcomes.get(trajectory.key)
            if quality is None:
                raise KeyError(f"no outcome for factual trajectory {trajectory.key} — outcomes must cover every trajectory")
            quality_source = "observed"
        else:
            quality = reward_model.predict(trajectory.opening, model).mean
            quality_source = "counterfactual_reward_model"
        qualities.append(quality)
        rows.append({
            "trajectory": trajectory.key, "logged_model": trajectory.logged_model, "model": model,
            "cost": cost, "quality": quality, "quality_source": quality_source, "factual": factual,
            **diagnostics,
        })
    mean_quality, low, high = bootstrap_mean(qualities)
    metrics = PolicyMetrics(
        policy=name,
        trajectories=len(trajectories),
        calls=sum(t.n_calls for t in trajectories),
        cost=sum(float(row["cost"]) for row in rows),
        quality=mean_quality,
        quality_ci=(low, high),
        factual_coverage=sum(1 for row in rows if row["factual"]) / max(1, len(rows)),
        cached_tokens=sum(float(row["cached_tokens"]) for row in rows),
        uncached_tokens=sum(float(row["uncached_tokens"]) for row in rows),
        switches=sum(float(row["switches"]) for row in rows),
    )
    return metrics, rows


def write_policy_report(path: str | Path, metrics: list[PolicyMetrics], rows: dict[str, list[dict[str, object]]]) -> None:
    payload = {"policies": [item.to_dict() for item in metrics], "rows": rows}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
