#!/usr/bin/env python3
"""Evaluate logged / cheapest / strongest / method3 (cascade) on the test split:
per-policy cost-quality metrics, doubly-robust off-policy estimate for the cascade
router, and a per-scenario-tag segmentation breakdown (dataset1 manifest only).

Usage: python scripts/evaluate_router.py <source> [models_dir] [output.json] [config.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data import load_ground_truth, load_trajectories, split_trajectories
from method3.evaluation import (
    evaluate_off_policy, evaluate_policy, load_scenario_tags, router_decide,
    segment_report, write_policy_report,
)
from method3.propensity.calibration import score_propensity_model
from method3.pricing import CostModel, resolve_pricing
from method3.propensity.logging_policy import LoggingPolicyModel
from method3.quality.conformal import SplitConformalCalibrator
from method3.quality.labels import calibrated_outcome
from method3.quality.accuracy import reward_model_accuracy
from method3.quality.reward_model import KNNRewardModel
from method3.routing.baselines import BaselineOracle
from method3.routing.cascade_router import CascadeRouter

DEFAULT_CONFIG = {
    "quality_floor": 0.5,
    "minimum_support": 3.0,
    "propensity_clip": 0.05,
    "switch_weight_threshold": 10.0,
}


def load_config(path: Path | None) -> dict[str, float]:
    config = dict(DEFAULT_CONFIG)
    if path is not None and path.exists():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


def load_true_propensities(source: str) -> dict[str, dict[str, float]]:
    """dataset2's manifest records the real P(model | context) its logging policy
    used. Returns {} for datasets that do not (dataset1, the organizer's export)."""
    path = Path(source)
    for candidate in (path / "scenario_manifest.json", path.parent / "scenario_manifest.json"):
        if candidate.exists():
            rows = json.loads(candidate.read_text(encoding="utf-8"))
            return {row["key"]: row["true_propensity_all"] for row in rows if row.get("true_propensity_all")}
    return {}


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    models_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models"
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results" / "policy_metrics.json"
    config_path = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "configs" / "router.json"
    config = load_config(config_path)

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    test = parts["test"] or trajectories
    ground_truth = load_ground_truth(source)
    outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in test}
    ground_truth_count = sum(1 for t in test if t.key in ground_truth)

    candidate_models = sorted({model for t in trajectories for model in t.per_call_models})
    pricing, guessed_prices = resolve_pricing(source, candidate_models)
    cost_model = CostModel(pricing)

    reward_model = KNNRewardModel.load(models_dir / "reward_model.json")
    propensity_model = LoggingPolicyModel.load(models_dir / "propensity_model.json")
    calibrator_payload = json.loads((models_dir / "conformal_calibrators.json").read_text(encoding="utf-8"))
    calibrators = {model: SplitConformalCalibrator.from_dict(payload) for model, payload in calibrator_payload["calibrators"].items()}

    oracle = BaselineOracle(candidate_models, cost_model)
    cascade = CascadeRouter(reward_model, calibrators, pricing, quality_floor=config["quality_floor"], minimum_support=config["minimum_support"])

    def _decide(baseline_fn):
        def decide(t):
            decision = baseline_fn(t)
            return decision.model, decision.route
        return decide

    policies = {
        "logged": _decide(oracle.logged),
        "cheapest": _decide(oracle.cheapest),
        "strongest": _decide(oracle.strongest),
        "method3": router_decide(cascade, candidate_models),
    }

    metrics = []
    rows: dict[str, list[dict[str, object]]] = {}
    segmentation: dict[str, dict[str, object]] = {}
    tags = load_scenario_tags(source)
    for name, decide in policies.items():
        policy_metrics, policy_rows = evaluate_policy(name, test, decide, cost_model, reward_model, outcomes)
        metrics.append(policy_metrics.to_dict())
        rows[name] = policy_rows
        segmentation[name] = segment_report(policy_rows, tags)

    # Context groups for positivity detection: the cells within which "was this
    # action ever tried?" is a meaningful question. Falls back to a single global
    # group when no manifest is present.
    group_of = {
        key: f"{tag.get('task_type', 'na')}|{tag.get('difficulty', 'na')}"
        for key, tag in tags.items()
    } if tags else {t.key: "all" for t in test}

    off_policy = evaluate_off_policy(
        test, cascade, candidate_models, reward_model, propensity_model, outcomes,
        clip=config["propensity_clip"], weight_threshold=config.get("switch_weight_threshold", 10.0),
        group_of=group_of,
    )

    # dataset2 records the TRUE propensities its logging policy used, so the fitted
    # propensity model can be scored against ground truth rather than merely
    # assumed correct. Absent (dataset1, raw export) this reports n=0 and is skipped.
    # Reward-model accuracy is reported separately from policy quality: policy
    # quality is partly self-reported by the reward model, so it cannot be used
    # to compare two versions of the pipeline. See quality/accuracy.py.
    model_accuracy = reward_model_accuracy(test, reward_model, ground_truth)

    true_propensities = load_true_propensities(source)
    propensity_quality = score_propensity_model(test, propensity_model, true_propensities)

    payload = {
        "split": "test", "n_test": len(test),
        "ground_truth_labels": ground_truth_count, "proxy_labels": len(test) - ground_truth_count,
        "config": config,
        # Models whose price had to be guessed from their name rather than declared.
        # A wrong price silently corrupts every cost number and every price-based
        # routing comparison, so it is reported rather than buried.
        "guessed_prices": guessed_prices,
        "effective_pricing_per_mtok": {m: round(p["input"] * 1_000_000, 4) for m, p in sorted(pricing.items()) if m in candidate_models},
        "policies": metrics,
        "off_policy": off_policy,
        "propensity_model_quality": propensity_quality,
        "reward_model_accuracy": model_accuracy,
        "segmentation": segmentation, "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if guessed_prices:
        print(f"WARNING: price guessed from model name (not declared by the dataset) for: {sorted(guessed_prices)}", file=sys.stderr)
    print(json.dumps({
        "n_test": len(test),
        "guessed_prices": guessed_prices,
        "policies": metrics,
        "off_policy_estimators": {name: {k: v for k, v in result.items() if k in ("value", "standard_error", "ci95", "n_matched", "effective_sample_size", "clipped_fraction")}
                                  for name, result in off_policy["estimators"].items()},
        "estimator_spread": off_policy["estimator_spread"],
        "positivity": {k: v for k, v in off_policy.get("positivity", {}).items() if k != "trajectories_routed_into_unsupported_cells" and k != "unsupported_cells"},
        "propensity_model_quality": propensity_quality,
        "reward_model_accuracy": model_accuracy,
    }, indent=2))


if __name__ == "__main__":
    main()
