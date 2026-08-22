from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PolicyMetrics:
    policy: str
    trajectories: int
    calls: int
    cost: float
    quality: float
    uncertainty: float
    cacheable_tokens: float
    uncached_tokens: float
    switches: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pareto_frontier(points: list[PolicyMetrics]) -> list[PolicyMetrics]:
    frontier = []
    for point in sorted(points, key=lambda item: item.cost):
        if not frontier or point.quality > frontier[-1].quality:
            frontier.append(point)
    return frontier


def group_costs(records: list[dict[str, Any]], key: str = "model") -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"trajectories": 0.0, "cost": 0.0, "quality": 0.0})
    for record in records:
        group = str(record[key])
        groups[group]["trajectories"] += 1
        groups[group]["cost"] += float(record["cost"])
        groups[group]["quality"] += float(record["quality"])
    for values in groups.values():
        values["mean_cost"] = values["cost"] / max(1.0, values["trajectories"])
        values["mean_quality"] = values["quality"] / max(1.0, values["trajectories"])
    return dict(groups)

