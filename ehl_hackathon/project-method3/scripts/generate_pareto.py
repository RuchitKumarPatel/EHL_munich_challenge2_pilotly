#!/usr/bin/env python3
"""Usage: python scripts/generate_pareto.py [policy_metrics.json] [pareto.csv]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.visualization.pareto import aiq_summary, write_pareto


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "policy_metrics.json"
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "pareto.csv"
    payload = json.loads(source.read_text(encoding="utf-8"))
    frontier = write_pareto(payload["policies"], target, target.with_suffix(".png"))
    print(f"frontier={[row['policy'] for row in frontier]} output={target}")
    print(json.dumps(aiq_summary(payload["policies"]), indent=2))


if __name__ == "__main__":
    main()
