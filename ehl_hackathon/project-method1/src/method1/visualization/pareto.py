from __future__ import annotations

import csv
from pathlib import Path

from method1.evaluation.metrics import PolicyMetrics, pareto_frontier


def write_pareto(metrics: list[PolicyMetrics], csv_path: str | Path, png_path: str | Path | None = None) -> list[PolicyMetrics]:
    frontier = pareto_frontier(metrics)
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "trajectories", "calls", "cost", "quality", "uncertainty", "cacheable_tokens", "uncached_tokens", "switches", "frontier"])
        writer.writeheader()
        frontier_names = {item.policy for item in frontier}
        for item in metrics:
            row = item.to_dict()
            row["frontier"] = item.policy in frontier_names
            writer.writerow(row)
    if png_path is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return frontier
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter([item.cost for item in metrics], [item.quality for item in metrics])
        for item in metrics:
            axis.annotate(item.policy, (item.cost, item.quality), fontsize=8)
        axis.plot([item.cost for item in frontier], [item.quality for item in frontier], linewidth=2)
        axis.set_xlabel("Estimated cache-aware cost (USD)")
        axis.set_ylabel("Composite outcome proxy")
        axis.set_title("Cost-quality Pareto frontier")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_path, dpi=160)
        plt.close(figure)
    return frontier

