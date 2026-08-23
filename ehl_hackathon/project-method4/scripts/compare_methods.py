#!/usr/bin/env python3
"""Head-to-head: method4 vs method3 vs the reference baselines on one test split.

Produces a single cost-quality frontier (CSV + non-overlapping-label PNG, reusing
method3's plotting) plus the sustainability metrics that are the actual point of
method4: does the policy keep its own successor evaluable, and how much usable
off-policy evidence does it leave behind?

Usage: python scripts/compare_methods.py <source> <method3_models> <method4_models> <out_dir>
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method4.bridge import (
    CostModel, KNNRewardModel, SplitConformalCalibrator, bootstrap_mean,
    calibrated_outcome, load_ground_truth, load_scenario_tags, load_trajectories,
    resolve_pricing, split_trajectories,
)
from method4.evaluation.coverage import action_space_coverage, deterministic_coverage
from method4.objective.geometric import geometric_mean_quality
from method4.policy.foraging_router import ForagingRouter
from method4.policy.marginal_value import MarginalValueRule
from method4.policy.response_threshold import ResponseThresholdTable

# method3's cascade router and Pareto plotting are reused directly rather than
# reimplemented, so the comparison is against the real method3 and the plot keeps
# the tested non-overlapping label placement.
from method3.routing.baselines import BaselineOracle  # noqa: E402
from method3.routing.cascade_router import CascadeRouter  # noqa: E402
from method3.visualization.pareto import write_pareto  # noqa: E402


def main() -> None:
    source = sys.argv[1]
    method3_models = Path(sys.argv[2])
    method4_models = Path(sys.argv[3])
    out_dir = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs" / "method4.json").read_text(encoding="utf-8"))

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    test = parts["test"] or trajectories
    ground_truth = load_ground_truth(source)
    outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in test}
    candidate_models = sorted({m for t in trajectories for m in t.per_call_models})
    pricing, guessed = resolve_pricing(source, candidate_models)
    cost_model = CostModel(pricing)

    reward_model = KNNRewardModel.load(method3_models / "reward_model.json")
    payload = json.loads((method3_models / "conformal_calibrators.json").read_text(encoding="utf-8"))
    calibrators = {m: SplitConformalCalibrator.from_dict(p) for m, p in payload["calibrators"].items()}
    marginal_value = MarginalValueRule.from_dict(json.loads((method4_models / "marginal_value.json").read_text(encoding="utf-8")))
    thresholds = ResponseThresholdTable.from_dict(json.loads((method4_models / "response_thresholds.json").read_text(encoding="utf-8")))

    oracle = BaselineOracle(candidate_models, cost_model)
    cascade = CascadeRouter(reward_model, calibrators, pricing, quality_floor=0.5, minimum_support=3.0)
    foraging = ForagingRouter(
        reward_model=reward_model, calibrators=calibrators, pricing=pricing,
        marginal_value=marginal_value, thresholds=thresholds,
        temperature=config["temperature"], exploration_floor=config["exploration_floor"],
        risk_aversion=config["risk_aversion"], minimum_support=config["minimum_support"],
        threshold_weight=config["threshold_weight"],
    )

    tags = load_scenario_tags(source)
    group_of = {k: f"{t.get('task_type','na')}|{t.get('difficulty','na')}" for k, t in tags.items()} if tags \
        else {t.key: "all" for t in test}

    def score(name: str, choose) -> tuple[dict, dict[str, str], dict[str, dict[str, float]] | None]:
        qualities, costs = [], []
        choices: dict[str, str] = {}
        distributions: dict[str, dict[str, float]] = {}
        for trajectory in test:
            model, route, distribution = choose(trajectory)
            cost, _ = cost_model.trajectory_cost(trajectory, route)
            factual = route == trajectory.per_call_models
            qualities.append(outcomes[trajectory.key] if factual
                             else reward_model.predict(trajectory.opening, model).mean)
            costs.append(cost)
            choices[trajectory.key] = model
            if distribution is not None:
                distributions[trajectory.key] = distribution
        mean, low, high = bootstrap_mean(qualities)
        row = {
            "policy": name, "trajectories": len(test), "calls": sum(t.n_calls for t in test),
            "cost": sum(costs), "quality": mean, "quality_ci": [low, high],
            "geometric_quality": geometric_mean_quality(qualities),
            "factual_coverage": sum(1 for t in test if choices[t.key] == t.logged_model) / len(test),
        }
        return row, choices, (distributions or None)

    rng = random.Random(config["sample_seed"])

    def baseline(fn):
        def choose(t):
            d = fn(t)
            return d.model, d.route, None
        return choose

    def cascade_choose(t):
        d = cascade.route(t.opening, candidate_models)
        return d.model, [d.model] * t.n_calls, None

    def foraging_choose(t):
        dist = foraging.action_distribution(t.opening, candidate_models)
        sampled = dist.sample(rng)
        return sampled, [sampled] * t.n_calls, dict(dist.probabilities)

    policies = [
        ("logged", baseline(oracle.logged)),
        ("cheapest", baseline(oracle.cheapest)),
        ("strongest", baseline(oracle.strongest)),
        ("method3_cascade", cascade_choose),
        ("method4_foraging", foraging_choose),
    ]

    metrics, sustainability = [], {}
    for name, choose in policies:
        row, choices, distributions = score(name, choose)
        metrics.append(row)
        coverage = (action_space_coverage(group_of, distributions, candidate_models) if distributions
                    else deterministic_coverage(group_of, choices, candidate_models))
        sustainability[name] = {
            "reachable_fraction": coverage["reachable_fraction"],
            "n_unreachable_cells": coverage["n_unreachable_cells"],
            "stochastic": distributions is not None,
        }

    frontier = write_pareto(
        metrics, out_dir / "pareto.csv", out_dir / "pareto.png",
        title="method4 vs method3 — held-out cost-quality frontier (dataset2)",
    )

    with (out_dir / "sustainability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "stochastic", "reachable_fraction", "n_unreachable_cells"])
        writer.writeheader()
        for name, row in sustainability.items():
            writer.writerow({"policy": name, **row})

    (out_dir / "comparison.json").write_text(json.dumps({
        "n_test": len(test), "guessed_prices": guessed, "config": config,
        "metrics": metrics, "sustainability": sustainability,
        "frontier": [r["policy"] for r in frontier],
    }, indent=2), encoding="utf-8")

    print(f"{'policy':<18}{'cost':>14}{'quality':>10}{'geo':>10}{'reach':>8}  frontier")
    print("-" * 68)
    frontier_names = {r["policy"] for r in frontier}
    for row in sorted(metrics, key=lambda r: r["cost"]):
        s = sustainability[row["policy"]]
        mark = "yes" if row["policy"] in frontier_names else ""
        print(f"{row['policy']:<18}{row['cost']:>14.8f}{row['quality']:>10.4f}"
              f"{row['geometric_quality']:>10.4f}{s['reachable_fraction']:>8.2f}  {mark}")
    print(f"\noutput={out_dir}")


if __name__ == "__main__":
    main()
