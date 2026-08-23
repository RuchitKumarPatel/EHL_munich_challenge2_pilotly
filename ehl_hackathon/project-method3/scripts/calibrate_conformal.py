#!/usr/bin/env python3
"""Fit a per-model split-conformal calibrator on the (held-out, never-trained-on)
calibration split — the quality-lower-bound guarantee CascadeRouter relies on.

Usage: python scripts/calibrate_conformal.py <source> <reward_model.json> [output.json] [alpha]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data import load_ground_truth, load_trajectories, split_trajectories
from method3.quality.conformal import SplitConformalCalibrator
from method3.quality.labels import calibrated_outcome
from method3.quality.reward_model import KNNRewardModel


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    reward_model_path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "models" / "reward_model.json")
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "models" / "conformal_calibrators.json"
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    calibration = parts["calibration"]
    if not calibration:
        raise ValueError(f"empty calibration split from {source} — conformal calibration requires held-out examples distinct from train")
    ground_truth = load_ground_truth(source)
    outcomes = {t.key: calibrated_outcome(t, ground_truth)[0] for t in calibration}
    reward_model = KNNRewardModel.load(reward_model_path)

    per_model: dict[str, list[tuple[float, float]]] = {model: [] for model in reward_model.models}
    for t in calibration:
        if t.logged_model not in per_model:
            continue
        predicted = reward_model.predict(t.opening, t.logged_model).mean
        per_model[t.logged_model].append((predicted, outcomes[t.key]))

    calibrators: dict[str, dict[str, object]] = {}
    for model, pairs in per_model.items():
        if not pairs:
            continue
        predicted = [p for p, _ in pairs]
        actual = [a for _, a in pairs]
        calibrators[model] = SplitConformalCalibrator(alpha=alpha).fit(predicted, actual).to_dict()

    # Pooled fallback calibrator across every model's calibration pairs together.
    # Per-model calibration is tighter/more model-specific when there's enough data
    # per model, but with a small calibration split (dataset1: ~32 trajectories over
    # 7 models) most per-model sets fall under the finite-sample threshold and
    # legitimately report coverage_guaranteed=False (see quality/conformal.py's
    # docstring — refusing an unearned bound is correct, not a bug). The pooled
    # calibrator has far more data and still gives a valid marginal guarantee, just
    # less locally-adaptive; CascadeRouter falls back to it under key "__global__"
    # when a candidate's own per-model calibrator isn't guaranteed.
    all_pairs = [pair for pairs in per_model.values() for pair in pairs]
    if all_pairs:
        predicted = [p for p, _ in all_pairs]
        actual = [a for _, a in all_pairs]
        calibrators["__global__"] = SplitConformalCalibrator(alpha=alpha).fit(predicted, actual).to_dict()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"alpha": alpha, "calibrators": calibrators}, indent=2), encoding="utf-8")
    per_model_guaranteed = sum(1 for model, c in calibrators.items() if model != "__global__" and c["coverage_guaranteed"])
    global_guaranteed = calibrators.get("__global__", {}).get("coverage_guaranteed", False)
    print(f"calibrated_models={len(per_model)} per_model_guaranteed={per_model_guaranteed} global_guaranteed={global_guaranteed} calibration_trajectories={len(calibration)} output={output}")


if __name__ == "__main__":
    main()
