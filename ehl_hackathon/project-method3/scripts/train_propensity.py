#!/usr/bin/env python3
"""Fit the logging-policy propensity model on the train split (non-mixed trajectories
only — see propensity/logging_policy.py).

Usage: python scripts/train_propensity.py <source> [output.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data import load_trajectories, split_trajectories
from method3.propensity.logging_policy import LoggingPolicyModel


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "propensity_model.json"
    trajectories = load_trajectories(source)
    parts = split_trajectories(trajectories)
    train = parts["train"] or trajectories
    models = sorted({t.logged_model for t in trajectories if t.logged_model != "mixed"})
    if not models:
        raise ValueError(f"no non-mixed trajectories in {source} — cannot fit a logging-policy model")
    # Regularization strength is selected on the calibration split, never on test.
    # A fixed strength overfits badly enough to make the fitted propensities worse
    # than a context-free baseline — see LoggingPolicyModel.fit_selected.
    propensity, selection = LoggingPolicyModel.fit_selected(models, train, parts["calibration"])
    propensity.save(output)
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps({"source": str(source), "models": models, "selection": selection}, indent=2), encoding="utf-8")
    print(f"trained={sum(1 for t in train if t.logged_model != 'mixed')} models={len(models)} "
          f"selected_l2={selection['selected_l2']} output={output}")


if __name__ == "__main__":
    main()
