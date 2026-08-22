from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method1.data.loader import load_trajectories
from method1.quality.deterministic_signals import trajectory_signals
from method1.quality.outcome_model import composite_outcome


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "processed" / "quality_labels.json"
    rows = []
    for trajectory in load_trajectories(source):
        rows.append({"trajectory": trajectory.key, "model": trajectory.logged_model, "quality": composite_outcome(trajectory), "signals": trajectory_signals(trajectory), "quality_type": "proxy"})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"labels={len(rows)} output={output}")


if __name__ == "__main__":
    main()

