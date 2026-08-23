from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method1.data.loader import load_trajectories
from method1.data.split import split_trajectories
from method1.evaluation.doubly_robust import evaluate_doubly_robust
from method1.evaluation.matching import matched_outcomes
from method1.evaluation.offline_policy_eval import evaluate_policy
from method1.pricing import CostModel, ensure_models, load_pricing
from method1.quality.ground_truth import load_ground_truth
from method1.quality.outcome_model import calibrated_outcome
from method1.routing.baselines import CheapestRouter, LoggedRouter, StrongestRouter
from method1.routing.cache_aware_router import CacheAwareRouter
from method1.routing.difficulty_router import DifficultyRouter


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "router" / "model.json"
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results" / "policy_metrics.json"
    trajectories = load_trajectories(source)
    partitions = split_trajectories(trajectories)
    test = partitions["test"] or partitions["validation"] or trajectories
    models = sorted({call.model for trajectory in trajectories for call in trajectory.calls})
    pricing = ensure_models(load_pricing(), models)
    cost_model = CostModel(pricing)
    ground_truth = load_ground_truth(source)
    train_calibrated = {item.key: calibrated_outcome(item, ground_truth)[0] for item in (partitions["train"] or trajectories)}
    if model_path.exists():
        predictor = DifficultyRouter.load(model_path, cost_model)
    else:
        predictor = DifficultyRouter(models, cost_model).fit(partitions["train"] or trajectories, train_calibrated)
    proposed = CacheAwareRouter(predictor, quality_target=None, cost_weight=0.15)
    policies = {"logged": LoggedRouter(cost_model), "cheapest": CheapestRouter(models, cost_model), "strongest": StrongestRouter(models, cost_model), "method1": proposed}
    calibrated = {item.key: calibrated_outcome(item, ground_truth) for item in trajectories}
    outcomes = {key: value for key, (value, _) in calibrated.items()}
    ground_truth_count = sum(1 for _, kind in calibrated.values() if kind == "ground_truth")
    metrics = []
    rows = {}
    for name, policy in policies.items():
        result, policy_rows = evaluate_policy(name, test, policy, cost_model, outcomes)
        metrics.append(result.to_dict())
        rows[name] = policy_rows
    dr = evaluate_doubly_robust(test, proposed, outcomes, models)
    payload = {"split": "test", "n_test": len(test), "quality_type": "calibrated" if ground_truth_count else "proxy", "ground_truth_labels": ground_truth_count, "proxy_labels": len(trajectories) - ground_truth_count, "policies": metrics, "doubly_robust": dr, "matched_outcomes": matched_outcomes(test, outcomes), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"split": "test", "trajectories": len(test), "policies": metrics, "doubly_robust": dr}, indent=2))


if __name__ == "__main__":
    main()

