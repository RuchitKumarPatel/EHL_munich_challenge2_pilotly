#!/usr/bin/env python3
"""Evaluate method4 against method3 and the reference baselines on the test split.

Reports, for every policy: realized cost/quality, the stochastic-target OPE
estimator family, forward-looking action-space coverage (does this policy keep its
own successor evaluable?), and a labeling-priority summary.

Usage: python scripts/evaluate_method4.py <source> <method3_models> <method4_models> <out.json> [config.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method4.bridge import (
    CostModel, KNNRewardModel, PolicyMetrics, SplitConformalCalibrator, bootstrap_mean,
    calibrated_outcome, load_ground_truth, load_scenario_tags, load_trajectories,
    resolve_pricing, segment_report, split_trajectories,
)
from method4.evaluation.coverage import action_space_coverage, deterministic_coverage
from method4.evaluation.stochastic_ope import StochasticSample, all_estimators, estimator_spread
from method4.labeling.prioritized_replay import LabelCandidate, ReplayPriority
from method4.objective.geometric import geometric_mean_quality
from method4.policy.foraging_router import ForagingRouter, context_bucket
from method4.policy.marginal_value import MarginalValueRule
from method4.policy.response_threshold import ResponseThresholdTable

DEFAULT_CONFIG = {
    "temperature": 0.15,
    "exploration_floor": 0.08,
    "risk_aversion": 1.0,
    "minimum_support": 3.0,
    "threshold_weight": 0.35,
    "sample_seed": 20260823,
    "label_budget": 40,
}


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
    method4_models = Path(sys.argv[3])
    output = Path(sys.argv[4])
    config = load_config(Path(sys.argv[5]) if len(sys.argv) > 5 else ROOT / "configs" / "method4.json")

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
    calibrator_payload = json.loads((method3_models / "conformal_calibrators.json").read_text(encoding="utf-8"))
    calibrators = {m: SplitConformalCalibrator.from_dict(p) for m, p in calibrator_payload["calibrators"].items()}
    marginal_value = MarginalValueRule.from_dict(json.loads((method4_models / "marginal_value.json").read_text(encoding="utf-8")))
    thresholds = ResponseThresholdTable.from_dict(json.loads((method4_models / "response_thresholds.json").read_text(encoding="utf-8")))

    router = ForagingRouter(
        reward_model=reward_model, calibrators=calibrators, pricing=pricing,
        marginal_value=marginal_value, thresholds=thresholds,
        temperature=config["temperature"], exploration_floor=config["exploration_floor"],
        risk_aversion=config["risk_aversion"], minimum_support=config["minimum_support"],
        threshold_weight=config["threshold_weight"],
    )

    rng = random.Random(config["sample_seed"])
    distributions: dict[str, dict[str, float]] = {}
    rows = []
    qualities, costs = [], []
    for trajectory in test:
        distribution = router.action_distribution(trajectory.opening, candidate_models)
        distributions[trajectory.key] = dict(distribution.probabilities)
        sampled = distribution.sample(rng)
        route = [sampled] * trajectory.n_calls
        cost, diagnostics = cost_model.trajectory_cost(trajectory, route)
        factual = route == trajectory.per_call_models
        quality = outcomes[trajectory.key] if factual else reward_model.predict(trajectory.opening, sampled).mean
        qualities.append(quality)
        costs.append(cost)
        rows.append({
            "trajectory": trajectory.key, "logged_model": trajectory.logged_model, "model": sampled,
            "cost": cost, "quality": quality, "factual": factual,
            "modal_model": distribution.argmax(), "entropy": distribution.entropy(),
            "min_probability": min(distribution.probabilities.values()),
            "reason": distribution.reason, **diagnostics,
        })

    mean_quality, low, high = bootstrap_mean(qualities)
    method4_metrics = PolicyMetrics(
        policy="method4", trajectories=len(test), calls=sum(t.n_calls for t in test),
        cost=sum(costs), quality=mean_quality, quality_ci=(low, high),
        factual_coverage=sum(1 for r in rows if r["factual"]) / max(1, len(rows)),
        cached_tokens=sum(float(r["cached_tokens"]) for r in rows),
        uncached_tokens=sum(float(r["uncached_tokens"]) for r in rows),
        switches=sum(float(r["switches"]) for r in rows),
    ).to_dict()

    # --- stochastic-target OPE (uses every trajectory, not just agreements) ---
    true_propensities = load_true_propensities(source)
    samples = []
    for trajectory in test:
        if trajectory.logged_model == "mixed" or trajectory.key not in outcomes:
            continue
        truth = true_propensities.get(trajectory.key)
        if not truth:
            continue
        opening = trajectory.opening
        samples.append(StochasticSample(
            q_by_model={m: reward_model.predict(opening, m).mean for m in candidate_models},
            target_pi=distributions[trajectory.key],
            logged_model=trajectory.logged_model,
            logged_propensity=truth.get(trajectory.logged_model, 1e-6),
            reward=outcomes[trajectory.key],
        ))
    estimators = all_estimators(samples)

    # --- forward-looking coverage: method4 (stochastic) vs its own modal collapse ---
    tags = load_scenario_tags(source)
    group_of = {k: f"{t.get('task_type','na')}|{t.get('difficulty','na')}" for k, t in tags.items()} if tags \
        else {t.key: "all" for t in test}
    stochastic_coverage = action_space_coverage(group_of, distributions, candidate_models)
    modal_choices = {r["trajectory"]: r["modal_model"] for r in rows}
    modal_coverage = deterministic_coverage(group_of, modal_choices, candidate_models)

    # --- labeling triage ---
    priority = ReplayPriority()
    label_candidates = []
    for trajectory in test:
        distribution = distributions[trajectory.key]
        ordered = sorted(distribution.values(), reverse=True)
        margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        modal = max(distribution, key=lambda m: distribution[m])
        estimate = reward_model.predict(trajectory.opening, modal)
        calibrator = calibrators.get(modal) or calibrators.get("__global__")
        width = (estimate.mean - calibrator.lower_bound(estimate.mean)) if calibrator else 1.0
        label_candidates.append(LabelCandidate(trajectory.key, width, margin, trajectory.key in ground_truth))
    label_report = priority.expected_budget_saving(label_candidates, config["label_budget"])
    label_report["selected"] = priority.select(label_candidates, config["label_budget"])

    payload = {
        "n_test": len(test),
        "config": config,
        "guessed_prices": guessed,
        "habitat_rate": marginal_value.habitat_rate,
        "policy": method4_metrics,
        "geometric_mean_quality": geometric_mean_quality(qualities),
        "off_policy": {
            "estimators": {n: r.to_dict() for n, r in estimators.items()},
            "estimator_spread": estimator_spread(estimators),
            "n_evaluable": len(samples),
        },
        "coverage": {"stochastic": stochastic_coverage, "modal_collapse": modal_coverage},
        "labeling": label_report,
        "segmentation": segment_report(rows, tags),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({
        "n_test": len(test),
        "policy": method4_metrics,
        "geometric_mean_quality": payload["geometric_mean_quality"],
        "off_policy": {n: {k: v for k, v in r.to_dict().items() if k in ("value", "standard_error", "effective_sample_size")}
                       for n, r in estimators.items()},
        "estimator_spread": payload["off_policy"]["estimator_spread"],
        "n_evaluable": len(samples),
        "coverage_stochastic": stochastic_coverage["reachable_fraction"],
        "coverage_modal_collapse": modal_coverage["reachable_fraction"],
        "labeling_ratio": label_report["ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
