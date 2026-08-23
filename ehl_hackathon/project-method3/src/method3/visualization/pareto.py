from __future__ import annotations

import csv
import math
from pathlib import Path

from .label_layout import place_labels


def pareto_frontier(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    """Cost-ascending frontier: a policy is on it if no cheaper-or-equal policy
    reaches equal-or-higher quality."""
    ordered = sorted(metrics, key=lambda row: float(row["cost"]))
    frontier = []
    best_quality = -1.0
    for row in ordered:
        quality = float(row["quality"])
        if quality > best_quality:
            frontier.append(row)
            best_quality = quality
    return frontier


def write_pareto(metrics: list[dict[str, object]], output: str | Path, png: str | Path | None = None,
                 title: str = "method3 held-out cost-quality frontier") -> list[dict[str, object]]:
    frontier = pareto_frontier(metrics)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "trajectories", "calls", "cost", "quality", "factual_coverage", "quality_per_cost", "frontier"]
    frontier_names = {str(row["policy"]) for row in frontier}
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            cost = float(row["cost"])
            writer.writerow({
                "policy": row["policy"], "trajectories": row["trajectories"], "calls": row["calls"],
                "cost": cost, "quality": row["quality"], "factual_coverage": row["factual_coverage"],
                "quality_per_cost": (float(row["quality"]) / cost) if cost > 0 else float("inf"),
                "frontier": row["policy"] in frontier_names,
            })
    if png is not None:
        _plot(metrics, frontier, png, title)
    return frontier


def _plot(metrics: list[dict[str, object]], frontier: list[dict[str, object]], png: str | Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not metrics:
        return

    # Costs span orders of magnitude across policies, so a linear x-axis crushes
    # every cheap policy into the origin and makes the frontier unreadable. Plot
    # cost on a log axis and lay labels out in log space so the non-overlap
    # geometry matches what is actually rendered.
    costs = [max(float(row["cost"]), 1e-12) for row in metrics]
    qualities = [float(row["quality"]) for row in metrics]
    names = [str(row["policy"]) for row in metrics]
    log_costs = [math.log10(c) for c in costs]

    figure, axis = plt.subplots(figsize=(11, 6.5))

    x_low, x_high = min(log_costs), max(log_costs)
    x_pad = max((x_high - x_low) * 0.18, 0.35)
    x_limits = (x_low - x_pad, x_high + x_pad)
    y_low, y_high = min(qualities), max(qualities)
    y_pad = max((y_high - y_low) * 0.30, 0.02)
    y_limits = (y_low - y_pad, y_high + y_pad)

    frontier_names = {str(row["policy"]) for row in frontier}
    on_frontier = [name in frontier_names for name in names]

    if frontier:
        frontier_x = [math.log10(max(float(row["cost"]), 1e-12)) for row in frontier]
        frontier_y = [float(row["quality"]) for row in frontier]
        axis.plot(frontier_x, frontier_y, linewidth=1.6, alpha=0.55, zorder=1, label="Pareto frontier")

    axis.scatter(
        [x for x, f in zip(log_costs, on_frontier) if f],
        [y for y, f in zip(qualities, on_frontier) if f],
        s=95, zorder=3, marker="o", edgecolors="white", linewidths=1.2, label="on frontier",
    )
    # Only draw (and legend) the dominated series when something is actually
    # dominated — an empty legend entry reads as "there are dominated points you
    # cannot see", which is worse than no entry.
    dominated_x = [x for x, f in zip(log_costs, on_frontier) if not f]
    if dominated_x:
        axis.scatter(
            dominated_x,
            [y for y, f in zip(qualities, on_frontier) if not f],
            s=70, zorder=3, marker="s", alpha=0.75, edgecolors="white", linewidths=1.0, label="dominated",
        )

    # Estimate label extents in data coordinates so the placer works in the same
    # space as the rendered text.
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    font_size = 9
    char_width = 0.0088 * x_span * (font_size / 9.0)
    line_height = 0.055 * y_span * (font_size / 9.0)
    labels = [f"{name}\n${cost:.2e} · q={quality:.3f}" for name, cost, quality in zip(names, costs, qualities)]
    sizes = [
        (max(len(line) for line in label.split("\n")) * char_width, len(label.split("\n")) * line_height)
        for label in labels
    ]

    positions = place_labels(list(zip(log_costs, qualities)), sizes, x_limits, y_limits)

    for (x0, y0), (px, py), label, (width, height) in zip(positions, zip(log_costs, qualities), labels, sizes):
        centre_x, centre_y = x0 + width / 2.0, y0 + height / 2.0
        # Leader line, so a label pushed away from its marker is still unambiguous.
        axis.annotate(
            "", xy=(px, py), xytext=(centre_x, centre_y),
            arrowprops={"arrowstyle": "-", "linewidth": 0.7, "alpha": 0.45, "shrinkA": 2, "shrinkB": 6},
            zorder=2,
        )
        axis.text(
            centre_x, centre_y, label, fontsize=font_size, ha="center", va="center", zorder=4,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92, "linewidth": 0.7},
        )

    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    tick_values = sorted({round(v) for v in (x_limits[0], *log_costs, x_limits[1])})
    axis.set_xticks(tick_values)
    axis.set_xticklabels([f"$10^{{{int(v)}}}$" for v in tick_values])
    axis.set_xlabel("Estimated cost, USD (log scale)")
    axis.set_ylabel("Quality (bootstrap mean)")
    axis.set_title(title)
    axis.grid(alpha=0.22, linestyle=":")
    axis.legend(loc="lower right", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=170)
    plt.close(figure)


def aiq_summary(metrics: list[dict[str, object]]) -> dict[str, float]:
    """Quality-per-dollar per policy — an AIQ-*style* summary. RouterBench's AIQ
    integrates the cost-quality curve over a budget sweep; this is a simpler,
    precisely-defined stand-in with the same intent, not a reimplementation."""
    return {
        str(row["policy"]): (float(row["quality"]) / float(row["cost"]) if float(row["cost"]) > 0 else float("inf"))
        for row in metrics
    }
