from __future__ import annotations

import json
from pathlib import Path


def load_ground_truth(source: str | Path) -> dict[str, float]:
    """Load dataset1-style scenario_manifest.json ground-truth quality labels, keyed
    by the same SHA-256 trajectory key data.schema.opening_key derives. Looks next to
    `source` (as a dir) and in its parent (source == an `export/` dir, manifest sits
    beside it). Returns {} when no manifest exists (e.g. the organizer's raw export) —
    a no-op fallback to the structural proxy everywhere this is used."""
    path = Path(source)
    for candidate in (path / "scenario_manifest.json", path.parent / "scenario_manifest.json"):
        if candidate.exists():
            rows = json.loads(candidate.read_text(encoding="utf-8"))
            return {row["key"]: float(row["ground_truth_quality"]) for row in rows if row.get("ground_truth_quality") is not None}
    return {}
