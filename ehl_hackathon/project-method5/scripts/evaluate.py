#!/usr/bin/env python3
"""Evaluate method5 on the test split and compare against method3, method4 and the baselines.

Reports the cost-quality frontier plus the properties method5 is actually claiming:
forward-looking action-space coverage, exploration targeting, off-policy estimator
agreement, and reward-model accuracy (kept separate from policy quality, since
policy quality is partly self-reported by the reward model and therefore cannot
compare two versions of the pipeline).

Usage: python scripts/evaluate.py <source> <method3_models> <method5_models> <out_dir> [config.json]
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method5.bridge import (
    CostModel, KNNRewardModel, LoggingPolicyModel, SplitConformalCalibrator,
    bootstrap_mean, calibrated_outcome, load_ground_truth, load_scenario_tags,
    load_trajectories, resolve_pricing, reward_model_accuracy, segment_report,
    split_trajectories,
)
from method5.evaluation.coverage import action_space_coverage, deterministic_coverage, exploration_efficiency
from method5.evaluation.ope import Sample, all_estimators, spread
from method5.labeling.priority import LabelCandidate, LabelTriage
from method5.policy.router import AdaptiveRouter
from method5.policy.uncertainty import UncertaintyGate, signal_from_bounds
from method5.policy.value_ladder import ValueLadder

# Reused directly so the comparison is against the real prior methods and the plot
# keeps method3's tested non-overlapping label placement.
from method3.routing.baselines import BaselineOracle  # noqa: E402
from method3.routing.cascade_router import CascadeRouter  # noqa: E402
from method3.visualization.pareto import write_pareto  # noqa: E402

DEFAULT_CONFIG = {"floor": 0.02, "ceiling": 0.30,
                  "softening": 0.05, "temperature": 0.02,
                  "minimum_support": 3.0, "sample_seed": 20260823, "label_budget": 40}


def load_config(path: Path | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path is not None and path.exists():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


def load_true_propensities(source: str) -> dict[str, dict[str, float]]:
    path = Path(source)
    for candidate in (path / "scenario_manifest.json", path.parent / "scenario_manifest.json"):
        if candidate.exists():
            rows = json.loads(candidate.read_text(encoding="utf-8"))
            return {r["key"]: r["true_propensity_all"] for r in rows if r.get("true_propensity_all")}
    return {}


def main() -> None:
    source = sys.argv[1]
    method3_models = Path(sys.argv[2])
    method5_models = Path(sys.argv[3])
    out_dir = Path(sys.argv[4])
    config = load_config(Path(sys.argv[5]) if len(sys.argv) > 5 else ROOT / "configs" / "method5.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    test = parts["test"] or trajectories
    ground_truth = load_ground_truth(source)
    outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in test}
    candidate_models = sorted({m for t in trajectories for m in t.per_call_models})
    pricing, guessed = resolve_pricing(source, candidate_models)
    if guessed:
        print(f"WARNING: guessed prices for {sorted(guessed)}", file=sys.stderr)
    cost_model = CostModel(pricing)

    reward_model = KNNRewardModel.load(method3_models / "reward_model.json")
    payload = json.loads((method3_models / "conformal_calibrators.json").read_text(encoding="utf-8"))
    calibrators = {m: SplitConformalCalibrator.from_dict(p) for m, p in payload["calibrators"].items()}
    ladder = ValueLadder.from_dict(json.loads((method5_models / "value_ladder.json").read_text(encoding="utf-8")))
    # reference_ratio comes from training (fit on the calibration split), never
    # from the config — it is calibrated data, not a tunable knob.
    gate_payload = json.loads((method5_models / "uncertainty_gate.json").read_text(encoding="utf-8"))
    gate = UncertaintyGate(floor=config["floor"], ceiling=config["ceiling"],
                           softening=config["softening"],
                           reference_ratio=float(gate_payload["reference_ratio"]))

    oracle = BaselineOracle(candidate_models, cost_model)
    cascade = CascadeRouter(reward_model, calibrators, pricing, quality_floor=0.5, minimum_support=3.0)
    adaptive = AdaptiveRouter(reward_model, calibrators, pricing, ladder, gate,
                              temperature=config["temperature"], minimum_support=config["minimum_support"])

    tags = load_scenario_tags(source)
    group_of = ({k: f"{t.get('task_type','na')}|{t.get('difficulty','na')}" for k, t in tags.items()}
                if tags else {t.key: "all" for t in test})

    rng = random.Random(config["sample_seed"])
    metrics, sustainability = [], {}
    method5_rows, method5_distributions = [], {}

    def evaluate(name: str, choose):
        qualities, costs = [], []
        choices, distributions, rows = {}, {}, []
        for trajectory in test:
            model, route, distribution = choose(trajectory)
            cost, diagnostics = cost_model.trajectory_cost(trajectory, route)
            factual = route == trajectory.per_call_models
            quality = outcomes[trajectory.key] if factual else reward_model.predict(trajectory.opening, model).mean
            qualities.append(quality)
            costs.append(cost)
            choices[trajectory.key] = model
            row = {"trajectory": trajectory.key, "logged_model": trajectory.logged_model,
                   "model": model, "cost": cost, "quality": quality, "factual": factual, **diagnostics}
            if distribution is not None:
                distributions[trajectory.key] = dict(distribution.probabilities)
                row.update({k: v for k, v in distribution.diagnostics.items()})
            rows.append(row)
        mean, low, high = bootstrap_mean(qualities)
        metrics.append({"policy": name, "trajectories": len(test), "calls": sum(t.n_calls for t in test),
                        "cost": sum(costs), "quality": mean, "quality_ci": [low, high],
                        "factual_coverage": sum(1 for r in rows if r["factual"]) / max(1, len(rows))})
        coverage = (action_space_coverage(group_of, distributions, candidate_models) if distributions
                    else deterministic_coverage(group_of, choices, candidate_models))
        sustainability[name] = {"stochastic": bool(distributions),
                                "reachable_fraction": coverage["reachable_fraction"],
                                "n_unreachable_cells": coverage["n_unreachable_cells"]}
        return rows, distributions

    def baseline(fn):
        return lambda t: (fn(t).model, fn(t).route, None)

    def cascade_choose(t):
        d = cascade.route(t.opening, candidate_models)
        return d.model, [d.model] * t.n_calls, None

    def adaptive_choose(t):
        d = adaptive.action_distribution(t.opening, candidate_models)
        sampled = d.sample(rng)
        return sampled, [sampled] * t.n_calls, d

    evaluate("logged", baseline(oracle.logged))
    evaluate("cheapest", baseline(oracle.cheapest))
    evaluate("strongest", baseline(oracle.strongest))
    evaluate("method3_cascade", cascade_choose)
    method5_rows, method5_distributions = evaluate("method5_adaptive", adaptive_choose)

    # --- stochastic-target OPE for method5 ---
    true_propensities = load_true_propensities(source)
    samples = []
    for trajectory in test:
        if trajectory.logged_model == "mixed" or trajectory.key not in outcomes:
            continue
        truth = true_propensities.get(trajectory.key)
        if not truth:
            continue
        opening = trajectory.opening
        samples.append(Sample(
            q_by_model={m: reward_model.predict(opening, m).mean for m in candidate_models},
            target_pi=method5_distributions[trajectory.key],
            logged_model=trajectory.logged_model,
            logged_propensity=truth.get(trajectory.logged_model, 1e-6),
            reward=outcomes[trajectory.key],
        ))
    estimators = all_estimators(samples)

    # --- exploration targeting: is epsilon going where the uncertainty is? ---
    efficiency = exploration_efficiency(
        [r["epsilon"] for r in method5_rows],
        [r["interval_width"] for r in method5_rows],
        [r["decision_margin"] for r in method5_rows],
        softening=config["softening"],
    )

    # --- labeling triage on the same signal ---
    triage = LabelTriage(gate=gate)
    label_candidates = []
    for trajectory in test:
        lower, upper = {}, {}
        for model in candidate_models:
            low, high, _ = adaptive._bounds(trajectory.opening, model)
            lower[model], upper[model] = low, high
        label_candidates.append(LabelCandidate(trajectory.key, signal_from_bounds(lower, upper),
                                               trajectory.key in ground_truth))
    labeling = triage.budget_efficiency(label_candidates, config["label_budget"])

    accuracy = reward_model_accuracy(test, reward_model, ground_truth)
    frontier = write_pareto(metrics, out_dir / "pareto.csv", out_dir / "pareto.png",
                            title="method5 vs method3 — held-out cost-quality frontier (dataset2)")

    with (out_dir / "sustainability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "stochastic", "reachable_fraction", "n_unreachable_cells"])
        writer.writeheader()
        for name, row in sustainability.items():
            writer.writerow({"policy": name, **row})

    (out_dir / "evaluation.json").write_text(json.dumps({
        "n_test": len(test), "config": config, "guessed_prices": guessed,
        "habitat_rate": ladder.habitat_rate,
        "metrics": metrics, "sustainability": sustainability,
        "frontier": [r["policy"] for r in frontier],
        "off_policy": {"estimators": {n: r.to_dict() for n, r in estimators.items()},
                       "spread": spread(estimators), "n_evaluable": len(samples)},
        "exploration_efficiency": efficiency,
        "labeling": labeling,
        "reward_model_accuracy": accuracy,
        "segmentation": segment_report(method5_rows, tags),
        "rows": method5_rows,
    }, indent=2), encoding="utf-8")

    frontier_names = {r["policy"] for r in frontier}
    print(f"{'policy':<18}{'cost':>14}{'quality':>10}{'reach':>8}  frontier")
    print("-" * 60)
    for row in sorted(metrics, key=lambda r: r["cost"]):
        s = sustainability[row["policy"]]
        print(f"{row['policy']:<18}{row['cost']:>14.8f}{row['quality']:>10.4f}"
              f"{s['reachable_fraction']:>8.2f}  {'yes' if row['policy'] in frontier_names else ''}")
    print(f"\nexploration: mean_eps={efficiency['mean_epsilon']:.3f} "
          f"range=[{efficiency['min_epsilon']:.3f},{efficiency['max_epsilon']:.3f}]")
    print(f"  corr(eps, signal)={efficiency['correlation_with_signal']:+.3f}  "
          f"corr(eps, margin)={efficiency['correlation_with_margin']:+.3f}  "
          f"corr(eps, width)={efficiency['correlation_with_width']:+.3f}")
    print(f"ope spread={spread(estimators):.4f} n_evaluable={len(samples)}")
    print(f"labeling ratio={labeling['ratio']:.2f}x")
    print(f"reward model MAE={accuracy['mae']:.4f} R2={accuracy['r_squared']:.3f}")
    print(f"\noutput={out_dir}")


if __name__ == "__main__":
    main()
