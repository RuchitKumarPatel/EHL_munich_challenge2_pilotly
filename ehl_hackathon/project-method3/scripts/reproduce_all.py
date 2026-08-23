#!/usr/bin/env python3
"""Full pipeline: train reward model -> train propensity model -> calibrate conformal
-> evaluate -> pareto frontier.

Usage: python scripts/reproduce_all.py <source> [run_dir]
run_dir defaults to models/ and results/ directly; pass a run_dir to write everything
under models/<run_dir>/ and results/<run_dir>/ instead (e.g. for a timestamped run).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    run_dir = sys.argv[2] if len(sys.argv) > 2 else ""
    models_dir = ROOT / "models" / run_dir if run_dir else ROOT / "models"
    results_dir = ROOT / "results" / run_dir if run_dir else ROOT / "results"

    run("train_reward_model.py", source, str(models_dir / "reward_model.json"))
    run("train_propensity.py", source, str(models_dir / "propensity_model.json"))
    run("calibrate_conformal.py", source, str(models_dir / "reward_model.json"), str(models_dir / "conformal_calibrators.json"))
    run("evaluate_router.py", source, str(models_dir), str(results_dir / "policy_metrics.json"))
    run("generate_pareto.py", str(results_dir / "policy_metrics.json"), str(results_dir / "pareto.csv"))
    print(f"reproduce_all complete: models={models_dir} results={results_dir}")


if __name__ == "__main__":
    main()
