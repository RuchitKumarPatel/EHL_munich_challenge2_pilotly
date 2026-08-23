#!/usr/bin/env python3
"""Evaluate under a CHRONOLOGICAL split as well as the usual random split.

The random hash split mixes dataset2's eras, so a model is partly fit on the future
it is asked to predict. That flatters any method that cannot actually handle drift.
dataset2 drifts on purpose — the task mix shifts, prices move, a new model appears
only in the final era, and the logging policy itself changes — so the honest question
is "would this router have worked if deployed back then?", which only a chronological
split answers.

Both splits are reported side by side. The GAP between them is the result: a small
gap means the method is genuinely drift-robust; a large one means the random-split
number was optimistic.

Usage: python scripts/drift_eval.py <source> <method3_models> <method4_models> <out.json>
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method4.bridge import (
    CostModel, KNNRewardModel, SplitConformalCalibrator, bootstrap_mean,
    calibrated_outcome, load_ground_truth, load_trajectories, resolve_pricing,
    split_trajectories, split_trajectories_temporal,
)
from method4.evaluation.coverage import action_space_coverage, deterministic_coverage
from method4.objective.geometric import geometric_mean_by_segment, geometric_mean_quality
from method4.policy.foraging_router import ForagingRouter
from method4.policy.marginal_value import MarginalValueRule
from method4.policy.response_threshold import ResponseThresholdTable

from method3.routing.cascade_router import CascadeRouter  # noqa: E402


def load_timesteps(source: str) -> dict[str, float]:
    path = Path(source)
    for candidate in (path / "scenario_manifest.json", path.parent / "scenario_manifest.json"):
        if candidate.exists():
            rows = json.loads(candidate.read_text(encoding="utf-8"))
            return {r["key"]: float(r.get("timestep", 0)) for r in rows}
    return {}


def load_eras(source: str) -> dict[str, str]:
    path = Path(source)
    for candidate in (path / "scenario_manifest.json", path.parent / "scenario_manifest.json"):
        if candidate.exists():
            rows = json.loads(candidate.read_text(encoding="utf-8"))
            return {r["key"]: str(r.get("era", "na")) for r in rows}
    return {}


def main() -> None:
    source = sys.argv[1]
    method3_models = Path(sys.argv[2])
    method4_models = Path(sys.argv[3])
    output = Path(sys.argv[4])
    config = json.loads((ROOT / "configs" / "method4.json").read_text(encoding="utf-8"))

    trajectories = load_trajectories(source)
    ground_truth = load_ground_truth(source)
    timesteps = load_timesteps(source)
    eras = load_eras(source)
    candidate_models = sorted({m for t in trajectories for m in t.per_call_models})
    pricing, _ = resolve_pricing(source, candidate_models)
    cost_model = CostModel(pricing)

    reward_model = KNNRewardModel.load(method3_models / "reward_model.json")
    payload = json.loads((method3_models / "conformal_calibrators.json").read_text(encoding="utf-8"))
    calibrators = {m: SplitConformalCalibrator.from_dict(p) for m, p in payload["calibrators"].items()}
    marginal_value = MarginalValueRule.from_dict(json.loads((method4_models / "marginal_value.json").read_text(encoding="utf-8")))
    thresholds = ResponseThresholdTable.from_dict(json.loads((method4_models / "response_thresholds.json").read_text(encoding="utf-8")))

    cascade = CascadeRouter(reward_model, calibrators, pricing, quality_floor=0.5, minimum_support=3.0)
    foraging = ForagingRouter(
        reward_model=reward_model, calibrators=calibrators, pricing=pricing,
        marginal_value=marginal_value, thresholds=thresholds,
        temperature=config["temperature"], exploration_floor=config["exploration_floor"],
        risk_aversion=config["risk_aversion"], minimum_support=config["minimum_support"],
        threshold_weight=config["threshold_weight"],
    )

    splits = {
        "random": split_trajectories(trajectories),
        "temporal": split_trajectories_temporal(trajectories, timesteps),
    }

    report: dict[str, object] = {}
    for split_name, parts in splits.items():
        test = parts["test"]
        outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in test}
        group_of = {t.key: eras.get(t.key, "na") for t in test}
        split_report: dict[str, object] = {"n_test": len(test), "eras_in_test": sorted({eras.get(t.key, "na") for t in test})}

        for policy_name in ("method3_cascade", "method4_foraging"):
            rng = random.Random(config["sample_seed"])
            qualities, costs = [], []
            by_era: dict[str, list[float]] = {}
            choices: dict[str, str] = {}
            distributions: dict[str, dict[str, float]] = {}
            for trajectory in test:
                if policy_name == "method3_cascade":
                    model = cascade.route(trajectory.opening, candidate_models).model
                else:
                    distribution = foraging.action_distribution(trajectory.opening, candidate_models)
                    model = distribution.sample(rng)
                    distributions[trajectory.key] = dict(distribution.probabilities)
                route = [model] * trajectory.n_calls
                cost, _ = cost_model.trajectory_cost(trajectory, route)
                factual = route == trajectory.per_call_models
                quality = outcomes[trajectory.key] if factual else reward_model.predict(trajectory.opening, model).mean
                qualities.append(quality)
                costs.append(cost)
                choices[trajectory.key] = model
                by_era.setdefault(eras.get(trajectory.key, "na"), []).append(quality)

            mean, low, high = bootstrap_mean(qualities)
            coverage = (action_space_coverage(group_of, distributions, candidate_models) if distributions
                        else deterministic_coverage(group_of, choices, candidate_models))
            split_report[policy_name] = {
                "cost": sum(costs), "quality": mean, "quality_ci": [low, high],
                "geometric_quality": geometric_mean_quality(qualities),
                "by_era": geometric_mean_by_segment(by_era),
                "reachable_fraction": coverage["reachable_fraction"],
            }
        report[split_name] = split_report

    # The headline: how much of each method's random-split score survives a
    # chronological split.
    drift = {}
    for policy_name in ("method3_cascade", "method4_foraging"):
        random_q = report["random"][policy_name]["quality"]        # type: ignore[index]
        temporal_q = report["temporal"][policy_name]["quality"]    # type: ignore[index]
        drift[policy_name] = {
            "random_quality": random_q,
            "temporal_quality": temporal_q,
            "drift_gap": random_q - temporal_q,
            "retained_fraction": (temporal_q / random_q) if random_q else 0.0,
        }
    report["drift_summary"] = drift

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'split':<10}{'policy':<20}{'cost':>13}{'quality':>10}{'geo':>9}{'reach':>8}")
    print("-" * 70)
    for split_name in ("random", "temporal"):
        for policy_name in ("method3_cascade", "method4_foraging"):
            r = report[split_name][policy_name]  # type: ignore[index]
            print(f"{split_name:<10}{policy_name:<20}{r['cost']:>13.8f}{r['quality']:>10.4f}"
                  f"{r['geometric_quality']:>9.4f}{r['reachable_fraction']:>8.2f}")
    print("\ndrift (random -> temporal):")
    for policy_name, d in drift.items():
        print(f"  {policy_name:<20} gap={d['drift_gap']:+.4f}  retained={d['retained_fraction']:.3f}")
    print(f"\noutput={output}")


if __name__ == "__main__":
    main()
