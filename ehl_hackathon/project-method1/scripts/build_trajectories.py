from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method1.data.loader import load_trajectories


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "interim" / "trajectories.json"
    trajectories = load_trajectories(source)
    rows = [{"key": item.key, "logged_model": item.logged_model, "calls": item.n_calls, "estimated_tokens": item.estimated_tokens} for item in trajectories]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"trajectories={len(rows)} output={output}")


if __name__ == "__main__":
    main()

