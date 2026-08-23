#!/usr/bin/env python3
"""Fit the opening-features-only KNN reward model on the train split.

Usage: python scripts/train_reward_model.py <source> [output.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data import load_ground_truth, load_trajectories, split_trajectories
from method3.pricing import resolve_pricing
from method3.quality.cold_start import ColdStartPrior, infer_descriptor
from method3.quality.labels import calibrated_outcome
from method3.quality.reward_model import KNNRewardModel


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "reward_model.json"
    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    train = parts["train"] or trajectories
    if not train:
        raise ValueError(f"no trajectories loaded from {source}")
    models = sorted({model for t in trajectories for model in t.per_call_models})
    ground_truth = load_ground_truth(source)
    calibrated = {t.key: calibrated_outcome(t, ground_truth) for t in train}
    labels = {key: value for key, (value, _) in calibrated.items()}

    # Descriptor-based cold-start prior, fit on the SAME training split, so a model
    # with little or no direct history (e.g. one introduced late in the log) still
    # gets a defensible estimate instead of a flat 0.5 that permanently locks it
    # out of a support-gated router. See quality/cold_start.py.
    pricing, _guessed = resolve_pricing(source, models)
    descriptors = {m: infer_descriptor(m, pricing[m]["input"] * 1_000_000) for m in models}
    observed: dict[str, list[float]] = {m: [] for m in models}
    for trajectory in train:
        if trajectory.logged_model in observed and trajectory.key in labels:
            observed[trajectory.logged_model].append(labels[trajectory.key])
    cold_start = ColdStartPrior().fit(descriptors, observed)

    reward_model = KNNRewardModel(models, cold_start=cold_start).fit(train, labels)
    reward_model.save(output)
    ground_truth_count = sum(1 for _, kind in calibrated.values() if kind == "ground_truth")
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps({
        "source": str(source), "models": models, "train_trajectories": len(train),
        "ground_truth_labels": ground_truth_count, "proxy_labels": len(train) - ground_truth_count,
    }, indent=2), encoding="utf-8")
    print(f"trained={len(train)} models={len(models)} ground_truth_labels={ground_truth_count} output={output}")


if __name__ == "__main__":
    main()
