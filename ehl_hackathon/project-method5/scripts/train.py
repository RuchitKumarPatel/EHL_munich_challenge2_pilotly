#!/usr/bin/env python3
"""Fit method5's policy components on the train split.

method5 reuses project-method3's reward model and conformal calibrators unchanged
(via the bridge). The only thing fit here is the MVT habitat rate — one scalar,
measured from what the logs actually paid.

Usage: python scripts/train.py <source> <method3_models_dir> [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method5.bridge import (
    CostModel, KNNRewardModel, SplitConformalCalibrator, calibrated_outcome,
    load_ground_truth, load_trajectories, resolve_pricing, split_trajectories,
)
from method5.policy.uncertainty import UncertaintyGate, signal_from_bounds
from method5.policy.value_ladder import ValueLadder


def main() -> None:
    source = sys.argv[1]
    method3_models = Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (method3_models / "reward_model.json").exists():
        raise FileNotFoundError(
            f"{method3_models}/reward_model.json not found — method5 reuses method3's "
            "reward model and calibrators; train those first."
        )

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    train = parts["train"] or trajectories
    ground_truth = load_ground_truth(source)
    outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in train}

    candidate_models = sorted({m for t in trajectories for m in t.per_call_models})
    pricing, guessed = resolve_pricing(source, candidate_models)
    if guessed:
        # A guessed price silently corrupts every cost number and every price-based
        # routing comparison, so it is surfaced rather than buried.
        print(f"WARNING: price guessed from model name for {sorted(guessed)}", file=sys.stderr)
    cost_model = CostModel(pricing)

    # Habitat rate: total realized quality over total realized spend, priced on the
    # REAL per-call model sequence (safe for mixed-model trajectories).
    qualities, costs = [], []
    for trajectory in train:
        if trajectory.key not in outcomes:
            continue
        cost, _ = cost_model.trajectory_cost(trajectory, trajectory.per_call_models)
        if cost > 0:
            qualities.append(outcomes[trajectory.key])
            costs.append(cost)
    ladder = ValueLadder.fit(qualities, costs)

    # Calibrate the uncertainty gate's reference scale on the CALIBRATION split —
    # held out from both the reward model's training data and from test. Without
    # this the gate saturates and goes flat (raw ratios on dataset2 span 12-60
    # against any fixed absolute scale), which defeats its entire purpose.
    reward_model = KNNRewardModel.load(method3_models / "reward_model.json")
    calibrator_payload = json.loads((method3_models / "conformal_calibrators.json").read_text(encoding="utf-8"))
    calibrators = {m: SplitConformalCalibrator.from_dict(p) for m, p in calibrator_payload["calibrators"].items()}
    pooled = calibrators.get("__global__")

    def bounds(opening, model):
        estimate = reward_model.predict(opening, model)
        calibrator = calibrators.get(model)
        chosen = calibrator if (calibrator is not None and calibrator.coverage_guaranteed) else None
        if chosen is None and pooled is not None and pooled.coverage_guaranteed:
            chosen = pooled
        if chosen is None:
            return 0.0, 1.0
        lower = chosen.lower_bound(estimate.mean)
        return lower, min(1.0, estimate.mean + max(0.0, estimate.mean - lower))

    gate_fit_split = parts["calibration"] or train
    signals = []
    for trajectory in gate_fit_split:
        lower, upper = {}, {}
        for model in candidate_models:
            lower[model], upper[model] = bounds(trajectory.opening, model)
        signals.append(signal_from_bounds(lower, upper))
    gate = UncertaintyGate().fit(signals)

    (out_dir / "value_ladder.json").write_text(json.dumps(ladder.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "uncertainty_gate.json").write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "train.metadata.json").write_text(json.dumps({
        "source": str(source), "method3_models": str(method3_models),
        "train_trajectories": len(train), "priced_trajectories": len(costs),
        "gate_fit_trajectories": len(gate_fit_split), "reference_ratio": gate.reference_ratio,
        "candidate_models": candidate_models, "guessed_prices": guessed,
    }, indent=2), encoding="utf-8")

    print(f"train={len(train)} priced={len(costs)} habitat_rate={ladder.habitat_rate:.4f} "
          f"gate_reference={gate.reference_ratio:.4f} (fit on {len(gate_fit_split)}) output={out_dir}")


if __name__ == "__main__":
    main()
