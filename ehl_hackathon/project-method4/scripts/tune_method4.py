#!/usr/bin/env python3
"""Select method4's policy hyperparameters on the CALIBRATION split.

Never touches the test split. Selection criterion is the geometric mean of realized
quality subject to a cost ceiling, which is the bet-hedging objective the policy is
built around rather than a metric invented for tuning.

Usage: python scripts/tune_method4.py <source> <method3_models> <method4_models> [out.json]
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method4.bridge import (
    CostModel, KNNRewardModel, SplitConformalCalibrator, calibrated_outcome,
    load_ground_truth, load_trajectories, resolve_pricing, split_trajectories,
)
from method4.objective.geometric import geometric_mean_quality
from method4.policy.foraging_router import ForagingRouter
from method4.policy.marginal_value import MarginalValueRule
from method4.policy.response_threshold import ResponseThresholdTable

# risk_aversion is the parameter that matters most and the one an initial run got
# wrong. MVT already fixes the justified price ceiling; re-applying a strong price
# penalty INSIDE that band double-counts cost. With costs spanning 50x and quality
# only ~1.8x, a risk_aversion of 1.0 let price dominate the score roughly 6:1 and
# collapsed 89/159 decisions onto the cheapest model. The grid spans that failure
# so the selection is made on evidence rather than by assertion.
GRID = {
    "risk_aversion": (0.0, 0.05, 0.15, 0.4, 1.0),
    "temperature": (0.05, 0.15, 0.35),
    "exploration_floor": (0.05, 0.08),
}


def main() -> None:
    source = sys.argv[1]
    method3_models = Path(sys.argv[2])
    method4_models = Path(sys.argv[3])
    output = Path(sys.argv[4]) if len(sys.argv) > 4 else method4_models / "tuning.json"

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    calibration = parts["calibration"]
    if not calibration:
        raise ValueError("empty calibration split — cannot tune without held-out data")
    ground_truth = load_ground_truth(source)
    outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in calibration}

    candidate_models = sorted({m for t in trajectories for m in t.per_call_models})
    pricing, _ = resolve_pricing(source, candidate_models)
    cost_model = CostModel(pricing)
    reward_model = KNNRewardModel.load(method3_models / "reward_model.json")
    payload = json.loads((method3_models / "conformal_calibrators.json").read_text(encoding="utf-8"))
    calibrators = {m: SplitConformalCalibrator.from_dict(p) for m, p in payload["calibrators"].items()}
    marginal_value = MarginalValueRule.from_dict(json.loads((method4_models / "marginal_value.json").read_text(encoding="utf-8")))
    thresholds = ResponseThresholdTable.from_dict(json.loads((method4_models / "response_thresholds.json").read_text(encoding="utf-8")))

    # Cost ceiling: the logged policy's own spend on this split. A configuration is
    # only admissible if it is no more expensive than what actually happened —
    # otherwise "better quality" is bought rather than earned.
    logged_cost = sum(cost_model.trajectory_cost(t, t.per_call_models)[0] for t in calibration)

    results = []
    for risk_aversion, temperature, floor in itertools.product(*GRID.values()):
        router = ForagingRouter(
            reward_model=reward_model, calibrators=calibrators, pricing=pricing,
            marginal_value=marginal_value, thresholds=thresholds,
            temperature=temperature, exploration_floor=floor,
            risk_aversion=risk_aversion, minimum_support=3.0,
        )
        rng = random.Random(12345)  # fixed across configs so comparisons are paired
        qualities, costs = [], []
        for trajectory in calibration:
            distribution = router.action_distribution(trajectory.opening, candidate_models)
            sampled = distribution.sample(rng)
            route = [sampled] * trajectory.n_calls
            cost, _ = cost_model.trajectory_cost(trajectory, route)
            factual = route == trajectory.per_call_models
            qualities.append(outcomes[trajectory.key] if factual else reward_model.predict(trajectory.opening, sampled).mean)
            costs.append(cost)
        total_cost = sum(costs)
        results.append({
            "risk_aversion": risk_aversion, "temperature": temperature, "exploration_floor": floor,
            "geometric_quality": geometric_mean_quality(qualities),
            "arithmetic_quality": sum(qualities) / len(qualities),
            "cost": total_cost,
            "within_logged_budget": total_cost <= logged_cost,
        })

    admissible = [r for r in results if r["within_logged_budget"]] or results
    best = max(admissible, key=lambda r: r["geometric_quality"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"logged_cost": logged_cost, "selected": best, "grid": results}, indent=2), encoding="utf-8")

    print(f"calibration trajectories = {len(calibration)}   logged budget = {logged_cost:.8f}")
    print("top configurations by geometric quality (within budget):")
    for row in sorted(admissible, key=lambda r: -r["geometric_quality"])[:6]:
        print(f"  ra={row['risk_aversion']:<5} T={row['temperature']:<5} floor={row['exploration_floor']:<5} "
              f"geo={row['geometric_quality']:.4f} arith={row['arithmetic_quality']:.4f} cost={row['cost']:.8f}")
    print(f"\nselected: {best}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
