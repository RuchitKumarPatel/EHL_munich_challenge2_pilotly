#!/usr/bin/env python3
"""Fit method4's policy components on the train split.

Reuses method3's reward model + conformal calibrators unchanged (via the bridge);
what is NEW here is the Marginal Value Theorem habitat rate and the response
threshold table.

Usage: python scripts/train_method4.py <source> <method3_models_dir> [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method4.bridge import (
    CostModel, KNNRewardModel, calibrated_outcome, load_ground_truth,
    load_trajectories, resolve_pricing, split_trajectories,
)
from method4.policy.foraging_router import context_bucket
from method4.policy.marginal_value import MarginalValueRule
from method4.policy.response_threshold import ResponseThresholdTable


def main() -> None:
    source = sys.argv[1]
    method3_models = Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    train = parts["train"] or trajectories
    ground_truth = load_ground_truth(source)
    outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in train}

    candidate_models = sorted({m for t in trajectories for m in t.per_call_models})
    pricing, guessed = resolve_pricing(source, candidate_models)
    if guessed:
        print(f"WARNING: guessed prices for {sorted(guessed)}", file=sys.stderr)
    cost_model = CostModel(pricing)

    # --- Pillar 2: the habitat rate, measured from what the logs actually paid ---
    # Cost of what REALLY happened (the logged per-call model sequence), paired with
    # the realized outcome. This is the "average intake rate of the habitat" in
    # Charnov's rule, and it is measured rather than assumed.
    qualities, costs = [], []
    for trajectory in train:
        if trajectory.key not in outcomes:
            continue
        cost, _ = cost_model.trajectory_cost(trajectory, trajectory.per_call_models)
        if cost > 0:
            qualities.append(outcomes[trajectory.key])
            costs.append(cost)
    marginal_value = MarginalValueRule.fit(qualities, costs)

    # --- Pillar 4: response thresholds, seeded from the reward model then moved by
    # observed outcomes on each (model, context bucket) pair ---
    reward_model = KNNRewardModel.load(method3_models / "reward_model.json")
    thresholds = ResponseThresholdTable()
    for trajectory in train:
        if trajectory.logged_model == "mixed" or trajectory.key not in outcomes:
            continue
        bucket = context_bucket(trajectory.opening)
        thresholds.update(trajectory.logged_model, bucket, performed=True, success=outcomes[trajectory.key])

    (out_dir / "marginal_value.json").write_text(json.dumps(marginal_value.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "response_thresholds.json").write_text(json.dumps(thresholds.to_dict(), indent=2), encoding="utf-8")

    buckets = sorted({context_bucket(t.opening) for t in train})
    print(f"train={len(train)} priced_trajectories={len(costs)}")
    print(f"habitat_rate={marginal_value.habitat_rate:.4f} quality-per-dollar")
    print(f"context_buckets={len(buckets)} threshold_entries={len(thresholds.thresholds)}")
    print(f"specialization_index={thresholds.specialization_index(buckets, candidate_models):.4f}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
