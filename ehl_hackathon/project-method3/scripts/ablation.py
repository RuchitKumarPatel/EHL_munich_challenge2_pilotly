#!/usr/bin/env python3
"""Ablate the two components that most differentiate method3 from method1/method2:
(1) conformal calibration (vs. gating on the raw predicted mean, no certified floor)
(2) ground-truth labels (vs. forcing the structural proxy everywhere, even when
    dataset1 ground truth is available).
Each of the 4 variants (2 label sources x 2 calibration modes) retrains its own
reward model and calibrators from scratch on the SAME split, since the label source
is itself the thing being ablated — there is no single pre-trained model to reuse.

Usage: python scripts/ablation.py <source> [output.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data import load_ground_truth, load_trajectories, split_trajectories
from method3.evaluation.offline_policy_eval import evaluate_policy, router_decide
from method3.pricing import CostModel, resolve_pricing
from method3.quality.conformal import SplitConformalCalibrator
from method3.quality.labels import calibrated_outcome, structural_proxy
from method3.quality.reward_model import KNNRewardModel
from method3.routing.cascade_router import CascadeRouter


class _NoFloorCalibrator(SplitConformalCalibrator):
    """Ablation stand-in: lower_bound == predicted mean, i.e. no calibration margin
    at all (the raw predicted mean is treated as if it were already a certified
    floor) — this is close to what a heuristic/uncalibrated router does."""
    def lower_bound(self, predicted: float, floor: float = 0.0) -> float:
        return max(floor, predicted)


def _reward_model_with_labels(train, models, labels) -> KNNRewardModel:
    return KNNRewardModel(models).fit(train, labels)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "ablation.json"

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    train, calibration, test = parts["train"] or trajectories, parts["calibration"], parts["test"] or trajectories
    ground_truth = load_ground_truth(source)
    candidate_models = sorted({model for t in trajectories for model in t.per_call_models})
    pricing, guessed_prices = resolve_pricing(source, candidate_models)
    cost_model = CostModel(pricing)

    outcomes_calibrated = {t.key: calibrated_outcome(t, ground_truth)[0] for t in trajectories}
    outcomes_proxy_only = {t.key: structural_proxy(t) for t in trajectories}

    # A single REFERENCE reward model, always trained on the best available labels,
    # is used to score every variant's counterfactual choices.
    #
    # Grading each variant with its OWN reward model is invalid and was producing a
    # nonsense result: the structural proxy sits on a systematically higher scale
    # than ground truth, so the proxy-trained variants scored ~0.87 — above every
    # other variant AND above the `strongest` baseline — purely because their grader
    # was inflated, not because they routed better. The ablation must isolate the
    # effect of each component on the ROUTING DECISION, which means holding the
    # measuring instrument fixed across variants.
    reference_reward_model = _reward_model_with_labels(
        train, candidate_models, {t.key: outcomes_calibrated[t.key] for t in train}
    )

    variants = {}
    for label_name, outcomes in (("ground_truth_calibrated", outcomes_calibrated), ("proxy_only", outcomes_proxy_only)):
        train_labels = {t.key: outcomes[t.key] for t in train}
        reward_model = _reward_model_with_labels(train, candidate_models, train_labels)

        per_model_pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in candidate_models}
        for t in calibration:
            if t.logged_model not in per_model_pairs:
                continue
            per_model_pairs[t.logged_model].append((reward_model.predict(t.opening, t.logged_model).mean, outcomes[t.key]))
        all_pairs = [pair for pairs in per_model_pairs.values() for pair in pairs]
        calibrators = {}
        for model, pairs in per_model_pairs.items():
            if pairs:
                calibrators[model] = SplitConformalCalibrator(alpha=0.1).fit([p for p, _ in pairs], [a for _, a in pairs])
        if all_pairs:
            calibrators["__global__"] = SplitConformalCalibrator(alpha=0.1).fit([p for p, _ in all_pairs], [a for _, a in all_pairs])

        for calibration_name, calibrator_set in (("conformal", calibrators), ("no_floor", {m: _NoFloorCalibrator(c.alpha, c.quantile, c.n_calibration, True) for m, c in calibrators.items()})):
            router = CascadeRouter(reward_model, calibrator_set, pricing, quality_floor=0.5, minimum_support=3.0)
            # `reward_model` (variant-specific) drives the ROUTING decision;
            # `reference_reward_model` (fixed) does the GRADING. Keeping those
            # separate is the whole point — see the note above.
            metrics, _ = evaluate_policy(
                f"{label_name}+{calibration_name}", test, router_decide(router, candidate_models),
                cost_model, reference_reward_model, {t.key: outcomes_calibrated[t.key] for t in test},
            )
            variants[f"{label_name}+{calibration_name}"] = metrics.to_dict()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(variants, indent=2), encoding="utf-8")
    print(json.dumps(variants, indent=2))


if __name__ == "__main__":
    main()
