from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method2.cost import pricing
from method2.data import load_trajectories, split
from method2.evaluation import evaluate, write
from method2.ground_truth import load_ground_truth
from method2.reward import CalibratedRewardModel
from method2.router import FixedRouter, SafeRouter


def main() -> None:
    source = sys.argv[1]
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "reward_model.json"
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results" / "heldout_metrics.json"
    trajectories = load_trajectories(source)
    test = split(trajectories)["test"]
    reward = CalibratedRewardModel.load(model_path)
    prices = pricing(reward.models)
    ground_truth = load_ground_truth(source)
    policies = {"logged": FixedRouter(reward, prices, "logged", ground_truth), "cheapest": FixedRouter(reward, prices, "cheapest"), "method2": SafeRouter(reward, prices)}
    if "claude-opus-5" in reward.models:
        policies["method1_repriced_static_opus5"] = FixedRouter(reward, prices, "claude-opus-5")
    metrics = []
    rows = {}
    for name, router in policies.items():
        metric, policy_rows = evaluate(name, test, router, ground_truth)
        metrics.append(metric)
        rows[name] = policy_rows
    write(output, {"quality_metric": "model_specific_calibrated_lower_bound", "test_trajectories": len(test), "ground_truth_labels": len(ground_truth), "metrics": metrics, "rows": rows})
    print(f"evaluated={len(test)} policies={len(metrics)} ground_truth_labels={len(ground_truth)} output={output}")


if __name__ == "__main__":
    main()
