from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    method2_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "heldout_metrics.json"
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "method1_vs_method2.csv"
    method2 = json.loads(method2_path.read_text(encoding="utf-8"))
    method1_metric = next(item for item in method2["metrics"] if item["policy"] == "method1_repriced_static_opus5")
    method2_metric = next(item for item in method2["metrics"] if item["policy"] == "method2")
    rows = [
        {"method": "method1_repriced_static_opus5", "cost": method1_metric["cost"], "quality": method1_metric["quality_lower_bound"], "quality_type": "calibrated_counterfactual_lower_bound", "test_trajectories": method2["test_trajectories"]},
        {"method": "method2_safe_router", "cost": method2_metric["cost"], "quality": method2_metric["quality_lower_bound"], "quality_type": "calibrated_counterfactual_lower_bound", "test_trajectories": method2["test_trajectories"]},
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={target}")


if __name__ == "__main__":
    main()
