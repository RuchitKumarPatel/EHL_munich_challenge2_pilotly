#!/usr/bin/env python3
"""Print a per-scenario breakdown from an evaluation's policy_metrics.json.

An aggregate cost/quality pair hides scenario-specific failure — the exact class of
bug that a single headline number kept invisible earlier in this project. This
renders the per-tag segmentation that evaluate_router.py already computed.

Usage: python scripts/segment_report.py <policy_metrics.json> [dimension ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    source = Path(sys.argv[1])
    wanted = sys.argv[2:]
    payload = json.loads(source.read_text(encoding="utf-8"))
    segmentation = payload.get("segmentation", {})
    if not segmentation:
        print("no segmentation available (dataset has no scenario_manifest.json)")
        return

    policies = list(segmentation)
    dimensions = wanted or sorted({d for p in policies for d in segmentation[p]})

    for dimension in dimensions:
        buckets = sorted({b for p in policies for b in segmentation[p].get(dimension, {})})
        if not buckets:
            continue
        print(f"\n=== {dimension} ===")
        header = f"{'bucket':<28}" + "".join(f"{p:>22}" for p in policies)
        print(header)
        print("-" * len(header))
        for bucket in buckets:
            cells = []
            for policy in policies:
                entry = segmentation[policy].get(dimension, {}).get(bucket)
                cells.append(f"{entry['mean_quality']:.3f} (n={int(entry['n'])})".rjust(22) if entry else "".rjust(22))
            print(f"{bucket:<28}" + "".join(cells))


if __name__ == "__main__":
    main()
