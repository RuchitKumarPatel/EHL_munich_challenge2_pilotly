from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method1.data.loader import load_trajectories
from method1.data.split import split_trajectories
from method1.evaluation.offline_policy_eval import evaluate_policy
from method1.pricing import CostModel, ensure_models, load_pricing
from method1.quality.outcome_model import composite_outcome
from method1.routing.baselines import CheapestRouter, LoggedRouter, StrongestRouter, StaticRouter
from method1.routing.cache_aware_router import CacheAwareRouter
from method1.routing.difficulty_router import DifficultyRouter
from method1.visualization.pareto import write_pareto


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results"
    trajectories = load_trajectories(source)
    test = split_trajectories(trajectories)["test"] or trajectories
    models = sorted({trajectory.logged_model for trajectory in trajectories if trajectory.logged_model != "mixed"})
    cost_model = CostModel(ensure_models(load_pricing(), models))
    train = split_trajectories(trajectories)["train"] or trajectories
    predictor = DifficultyRouter(models, cost_model).fit(train, {item.key: composite_outcome(item) for item in train})
    policies = {"logged": LoggedRouter(cost_model), "cheapest": CheapestRouter(models, cost_model), "strongest": StrongestRouter(models, cost_model), "method1": CacheAwareRouter(predictor, cost_weight=0.15)}
    for index, model in enumerate(models):
        policies[f"static_{index}_{model}"] = StaticRouter(model, cost_model)
    metrics = [evaluate_policy(name, test, policy, cost_model, {item.key: composite_outcome(item) for item in trajectories})[0] for name, policy in policies.items()]
    frontier = write_pareto(metrics, output / "pareto.csv", output / "pareto.png")
    print(f"policies={len(metrics)} frontier={len(frontier)} output={output}")


if __name__ == "__main__":
    main()
