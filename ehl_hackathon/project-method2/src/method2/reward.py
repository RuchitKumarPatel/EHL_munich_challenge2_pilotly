from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .data import Trajectory
from .features import distance, extract, vector


@dataclass
class Estimate:
    mean: float
    lower: float
    uncertainty: float
    support: float
    distance: float


class CalibratedRewardModel:
    def __init__(self, models: list[str], neighbors: int = 12, minimum_support: int = 3) -> None:
        self.models = models
        self.neighbors = neighbors
        self.minimum_support = minimum_support
        self.names: list[str] = []
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.examples: dict[str, list[tuple[list[float], float, float]]] = {model: [] for model in models}
        self.residual_bound = 0.25

    def fit(self, trajectories: list[Trajectory], labels: dict[str, tuple[float, float]], calibration: list[Trajectory]) -> "CalibratedRewardModel":
        rows = [extract(item) for item in trajectories]
        if not rows:
            raise ValueError("training trajectories are required")
        self.names = sorted(rows[0])
        self.means = {name: sum(row[name] for row in rows) / len(rows) for name in self.names}
        self.scales = {name: max(1.0, abs(self.means[name]) * 0.1, math.sqrt(sum((row[name] - self.means[name]) ** 2 for row in rows) / len(rows))) for name in self.names}
        self.examples = {model: [] for model in self.models}
        for trajectory in trajectories:
            if trajectory.model in self.examples and trajectory.key in labels:
                score, confidence = labels[trajectory.key]
                self.examples[trajectory.model].append((vector(extract(trajectory), self.names, self.means, self.scales), score, confidence))
        residuals = []
        for trajectory in calibration:
            if trajectory.key not in labels or trajectory.model not in self.examples or not self.examples[trajectory.model]:
                continue
            estimate = self.predict(trajectory, trajectory.model, include_bound=False)
            residuals.append(abs(labels[trajectory.key][0] - estimate.mean))
        if residuals:
            residuals.sort()
            self.residual_bound = residuals[min(len(residuals) - 1, int(math.ceil(0.9 * len(residuals))) - 1)]
        return self

    def predict(self, trajectory: Trajectory, model: str, include_bound: bool = True) -> Estimate:
        examples = self.examples.get(model, [])
        if not examples:
            return Estimate(0.0, 0.0, 1.0, 0.0, 1.0)
        current = vector(extract(trajectory), self.names, self.means, self.scales)
        nearest = sorted(((distance(current, item[0]), item[1], item[2]) for item in examples), key=lambda item: item[0])[: self.neighbors]
        weights = [1.0 / (0.05 + item[0]) for item in nearest]
        total = sum(weights)
        mean = sum(weight * item[1] * item[2] for weight, item in zip(weights, nearest)) / max(1e-9, sum(weight * item[2] for weight, item in zip(weights, nearest)))
        variance = sum(weight * (item[1] - mean) ** 2 for weight, item in zip(weights, nearest)) / max(1e-9, total)
        effective = total * total / max(1e-9, sum(weight * weight for weight in weights))
        avg_distance = sum(weight * item[0] for weight, item in zip(weights, nearest)) / max(1e-9, total)
        uncertainty = min(1.0, math.sqrt(variance) + self.residual_bound + 0.20 / math.sqrt(max(1.0, effective)) + 0.12 * avg_distance)
        lower = max(0.0, mean - uncertainty) if include_bound else mean
        return Estimate(max(0.0, min(1.0, mean)), lower, uncertainty, effective, avg_distance)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"models": self.models, "neighbors": self.neighbors, "minimum_support": self.minimum_support, "names": self.names, "means": self.means, "scales": self.scales, "examples": self.examples, "residual_bound": self.residual_bound}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedRewardModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(payload["models"], int(payload["neighbors"]), int(payload["minimum_support"]))
        model.names = payload["names"]
        model.means = {key: float(value) for key, value in payload["means"].items()}
        model.scales = {key: float(value) for key, value in payload["scales"].items()}
        model.examples = {key: [(list(vector_value), float(score), float(confidence)) for vector_value, score, confidence in values] for key, values in payload["examples"].items()}
        model.residual_bound = float(payload["residual_bound"])
        return model
