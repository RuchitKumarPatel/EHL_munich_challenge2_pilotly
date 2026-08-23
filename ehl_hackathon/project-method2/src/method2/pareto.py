from __future__ import annotations

import csv
from pathlib import Path


def write(metrics: list[dict[str, object]], output: str | Path, png: str | Path | None = None) -> list[dict[str, object]]:
    ordered = sorted(metrics, key=lambda item: float(item["cost"]))
    frontier = []
    best = -1.0
    for row in ordered:
        quality = float(row["quality_lower_bound"])
        if quality > best:
            frontier.append(row)
            best = quality
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "trajectories", "cost", "quality_estimate", "quality_lower_bound", "factual_coverage", "factual_observed_quality", "frontier"])
        writer.writeheader()
        names = {str(item["policy"]) for item in frontier}
        for row in metrics:
            writer.writerow({key: row.get(key) for key in writer.fieldnames[:-1]} | {"frontier": row["policy"] in names})
    if png is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return frontier
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter([float(row["cost"]) for row in metrics], [float(row["quality_lower_bound"]) for row in metrics])
        for row in metrics:
            axis.annotate(str(row["policy"]), (float(row["cost"]), float(row["quality_lower_bound"])), fontsize=8)
        axis.plot([float(row["cost"]) for row in frontier], [float(row["quality_lower_bound"]) for row in frontier])
        axis.set_xlabel("Estimated cache-aware cost (USD)")
        axis.set_ylabel("Calibrated quality lower bound")
        axis.set_title("Method 2 held-out Pareto frontier")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png, dpi=160)
        plt.close(figure)
    return frontier
