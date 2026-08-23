#!/usr/bin/env python3
"""Full method4 pipeline: train -> tune (on calibration) -> evaluate -> compare.

Usage: python scripts/reproduce_all.py <source> <method3_models_dir> [run_tag]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def main() -> None:
    source = sys.argv[1]
    method3_models = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else time.strftime("%Y%m%d_%H%M%S")
    models = ROOT / "models" / tag
    results = ROOT / "results" / tag
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    run("train_method4.py", source, method3_models, str(models))
    run("tune_method4.py", source, method3_models, str(models), str(models / "tuning.json"))
    run("evaluate_method4.py", source, method3_models, str(models), str(results / "method4_metrics.json"))
    run("compare_methods.py", source, method3_models, str(models), str(results))
    print(f"\nreproduce_all complete: models={models} results={results}")


if __name__ == "__main__":
    main()
