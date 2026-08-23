from __future__ import annotations

import json
from pathlib import Path

# Segmentation dimensions. Superset of dataset1's and dataset2's manifest fields —
# a row simply lacking a dimension contributes nothing to it, so one segmenter
# serves both datasets (and the organizer's raw export, which has no manifest at
# all and yields an empty report).
#
# `cache_break` is dataset1's name for a mid-trajectory model switch; dataset2
# splits that concept into the more precise `model_switch`, `context_compaction`,
# `tool_schema_change` and `tool_reorder`. Both names are listed so neither
# dataset silently loses its cache-related breakdown.
#
# Segmenting matters because an aggregate-only report is exactly what hid the
# mixed-model crash found earlier in this project: a single headline number was
# fine while one scenario class was completely broken.
_TAG_DIMENSIONS = (
    "task_type", "difficulty", "domain", "graded",
    "has_failure", "multi_turn", "branching",
    "cache_break", "model_switch", "context_compaction", "tool_schema_change", "tool_reorder",
    "era", "edge_case", "replicated",
)


def load_scenario_tags(source: str | Path) -> dict[str, dict[str, object]]:
    """Load dataset1-style scenario_manifest.json into {trajectory_key: {tag: value}}.
    Returns {} when no manifest exists (e.g. the organizer's raw export) — callers
    should treat that as "segmentation unavailable", not an error."""
    path = Path(source)
    for candidate in (path / "scenario_manifest.json", path.parent / "scenario_manifest.json"):
        if candidate.exists():
            rows = json.loads(candidate.read_text(encoding="utf-8"))
            tags = {}
            for row in rows:
                entry = {name: row.get(name) for name in _TAG_DIMENSIONS
                         if name not in {"graded", "replicated"} and name in row}
                entry["graded"] = row.get("ground_truth_quality") is not None
                if "replicate_group" in row:
                    entry["replicated"] = row.get("replicate_group") is not None
                tags[row["key"]] = entry
            return tags
    return {}


def segment_report(rows: list[dict[str, object]], tags: dict[str, dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    """Break a policy's per-trajectory evaluation rows down by each scenario_manifest
    tag dimension — catches exactly the class of failure an aggregate-only report
    hides (e.g. this session's mixed-model crash, invisible in a single headline
    number but obvious once segmented by cache_break)."""
    report: dict[str, dict[str, dict[str, float]]] = {name: {} for name in _TAG_DIMENSIONS}
    if not tags:
        return report
    for row in rows:
        row_tags = tags.get(str(row["trajectory"]))
        if row_tags is None:
            continue
        for dimension in _TAG_DIMENSIONS:
            if dimension not in row_tags:
                # This dataset simply does not carry that dimension (dataset1 has
                # `cache_break`, dataset2 has `model_switch`). Skip rather than
                # inventing a "None" bucket that would look like real data.
                continue
            value = str(row_tags.get(dimension))
            bucket = report[dimension].setdefault(value, {"n": 0.0, "quality_sum": 0.0, "cost_sum": 0.0})
            bucket["n"] += 1.0
            bucket["quality_sum"] += float(row["quality"])
            bucket["cost_sum"] += float(row["cost"])
    for dimension_buckets in report.values():
        for bucket in dimension_buckets.values():
            bucket["mean_quality"] = bucket["quality_sum"] / bucket["n"]
            bucket["mean_cost"] = bucket["cost_sum"] / bucket["n"]
            del bucket["quality_sum"]
            del bucket["cost_sum"]
    # Drop dimensions this dataset carries no data for, so the report shows only
    # breakdowns that actually exist.
    return {name: buckets for name, buckets in report.items() if buckets}
