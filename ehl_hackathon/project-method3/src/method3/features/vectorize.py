from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Standardizer:
    """Zero-mean, unit-ish-scale standardization with a fixed feature-name schema
    (never inferred from a batch's keys — see opening_features.py's rationale) and a
    scale floor to guard divide-by-zero on a constant column (e.g. every training
    trajectory happens to have the same tool_count)."""
    names: tuple[str, ...]
    means: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    scale_floor: float = 1e-6

    def fit(self, rows: list[dict[str, float]]) -> "Standardizer":
        if not rows:
            raise ValueError("cannot fit a standardizer on zero rows")
        self.means = {name: sum(row.get(name, 0.0) for row in rows) / len(rows) for name in self.names}
        self.scales = {
            name: max(self.scale_floor, math.sqrt(sum((row.get(name, 0.0) - self.means[name]) ** 2 for row in rows) / len(rows)))
            for name in self.names
        }
        return self

    def transform(self, row: dict[str, float]) -> list[float]:
        if not self.means:
            raise RuntimeError("Standardizer.transform called before fit")
        return [(row.get(name, 0.0) - self.means[name]) / self.scales[name] for name in self.names]

    def to_dict(self) -> dict[str, object]:
        return {"names": list(self.names), "means": self.means, "scales": self.scales, "scale_floor": self.scale_floor}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Standardizer":
        instance = cls(tuple(payload["names"]), dict(payload["means"]), dict(payload["scales"]), float(payload["scale_floor"]))
        return instance


def euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / max(1, len(left)))
