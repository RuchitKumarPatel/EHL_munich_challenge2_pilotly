#!/usr/bin/env python3
"""Select method5's gate hyperparameters on the CALIBRATION split. Never test.

Criterion: mean quality subject to a cost ceiling set by the logged policy's own
spend. A configuration that simply buys quality by spending more is inadmissible —
otherwise "better" would just mean "more expensive".

Usage: python scripts/tune.py <source> <method3_models> <method5_models> [out.json]
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method5.bridge import (
    CostModel, KNNRewardModel, SplitConformalCalibrator, calibrated_outcome,
    load_ground_truth, load_trajectories, resolve_pricing, split_trajectories,
)
from method5.policy.router import AdaptiveRouter
from method5.policy.uncertainty import UncertaintyGate
from method5.policy.value_ladder import ValueLadder

GRID = {
    "ceiling": (0.10, 0.20, 0.30, 0.45),
    "temperature": (0.02, 0.05, 0.15),
}
FLOOR = 0.02   # fixed: the positivity guarantee, not a tuning knob


def main() -> None:
    source = sys.argv[1]
    method3_models = Path(sys.argv[2])
    method5_models = Path(sys.argv[3])
    output = Path(sys.argv[4]) if len(sys.argv) > 4 else method5_models / "tuning.json"

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
    ladder = ValueLadder.from_dict(json.loads((method5_models / "value_ladder.json").read_text(encoding="utf-8")))

    logged_cost = sum(cost_model.trajectory_cost(t, t.per_call_models)[0] for t in calibration)

    results = []
    gate_payload = json.loads((method5_models / "uncertainty_gate.json").read_text(encoding="utf-8"))
    reference_ratio = float(gate_payload["reference_ratio"])

    for ceiling, temperature in itertools.product(*GRID.values()):
        router = AdaptiveRouter(
            reward_model=reward_model, calibrators=calibrators, pricing=pricing,
            value_ladder=ladder,
            gate=UncertaintyGate(floor=FLOOR, ceiling=ceiling, reference_ratio=reference_ratio),
            temperature=temperature,
        )
        rng = random.Random(12345)   # fixed across configs -> paired comparison
        qualities, costs, epsilons = [], [], []
        for trajectory in calibration:
            distribution = router.action_distribution(trajectory.opening, candidate_models)
            sampled = distribution.sample(rng)
            route = [sampled] * trajectory.n_calls
            cost, _ = cost_model.trajectory_cost(trajectory, route)
            factual = route == trajectory.per_call_models
            qualities.append(outcomes[trajectory.key] if factual
                             else reward_model.predict(trajectory.opening, sampled).mean)
            costs.append(cost)
            epsilons.append(distribution.diagnostics["epsilon"])
        total_cost = sum(costs)
        results.append({
            "ceiling": ceiling, "temperature": temperature,
            "quality": sum(qualities) / len(qualities),
            "cost": total_cost,
            "mean_epsilon": sum(epsilons) / len(epsilons),
            "within_logged_budget": total_cost <= logged_cost,
        })

    admissible = [r for r in results if r["within_logged_budget"]] or results
    best = max(admissible, key=lambda r: r["quality"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"logged_cost": logged_cost, "floor": FLOOR,
                                  "selected": best, "grid": results}, indent=2), encoding="utf-8")

    print(f"calibration={len(calibration)}  logged budget={logged_cost:.8f}")
    for row in sorted(admissible, key=lambda r: -r["quality"])[:5]:
        print(f"  ceiling={row['ceiling']:<5} T={row['temperature']:<5} "
              f"q={row['quality']:.4f} cost={row['cost']:.8f} mean_eps={row['mean_epsilon']:.3f}")
    print(f"\nselected: {best}\noutput={output}")


if __name__ == "__main__":
    main()
