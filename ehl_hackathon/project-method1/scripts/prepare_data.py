from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method1.data.loader import jsonl_paths, load_records, load_trajectories
from method1.data.split import split_trajectories


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "manifests" / "dataset.json"
    records = list(load_records(source))
    trajectories = load_trajectories(source)
    partitions = split_trajectories(trajectories)
    payload = {
        "source": str(source),
        "files": [str(path) for path in jsonl_paths(source)],
        "requests": len(records),
        "trajectories": len(trajectories),
        "models": sorted({record.model for record in records}),
        "partitions": {key: len(value) for key, value in partitions.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

