from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method2.pareto import write


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "heldout_metrics.json"
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "pareto.csv"
    payload = json.loads(source.read_text(encoding="utf-8"))
    frontier = write(payload["metrics"], target, target.with_suffix(".png"))
    print(f"frontier={len(frontier)} output={target}")


if __name__ == "__main__":
    main()
