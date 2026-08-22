from __future__ import annotations

import json
import math
from pathlib import Path

from method1.data.schema import Trajectory
from method1.features.trajectory_features import extract_trajectory_features
from method1.pricing.cost_model import CostModel

from .base import RouteDecision
from .baselines import MODEL_STRENGTH
from .uncertainty import score_uncertainty


class DifficultyRouter:
    def __init__(self, models: list[str], cost_model: CostModel) -> None:
        self.models = models
        self.cost_model = cost_model
        self.feature_names: list[str] = []
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.global_weights: list[float] = []
        self.model_weights: dict[str, list[float]] = {}
        self.model_bias: dict[str, float] = {}

    def _vector(self, trajectory: Trajectory) -> list[float]:
        features = extract_trajectory_features(trajectory)
        if not self.feature_names:
            self.feature_names = sorted(features)
        return [1.0] + [(features.get(name, 0.0) - self.means.get(name, 0.0)) / self.scales.get(name, 1.0) for name in self.feature_names]

    @staticmethod
    def _fit_linear(vectors: list[list[float]], targets: list[float], steps: int = 700, learning_rate: float = 0.035, regularization: float = 0.02) -> list[float]:
        if not vectors:
            return []
        weights = [0.0] * len(vectors[0])
        for _ in range(steps):
            gradients = [0.0] * len(weights)
            for vector, target in zip(vectors, targets):
                prediction = sum(weight * value for weight, value in zip(weights, vector))
                error = prediction - target
                for index, value in enumerate(vector):
                    gradients[index] += error * value
            scale = 1.0 / max(1, len(vectors))
            for index in range(len(weights)):
                penalty = regularization * weights[index] if index else 0.0
                weights[index] -= learning_rate * (gradients[index] * scale + penalty)
        return weights

    def fit(self, trajectories: list[Trajectory], outcomes: dict[str, float]) -> "DifficultyRouter":
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        raw = [extract_trajectory_features(trajectory) for trajectory in trajectories]
        self.feature_names = sorted(raw[0])
        self.means = {name: sum(row.get(name, 0.0) for row in raw) / len(raw) for name in self.feature_names}
        self.scales = {name: max(1e-6, math.sqrt(sum((row.get(name, 0.0) - self.means[name]) ** 2 for row in raw) / len(raw))) for name in self.feature_names}
        vectors = [[1.0] + [(row.get(name, 0.0) - self.means[name]) / self.scales[name] for name in self.feature_names] for row in raw]
        targets = [max(0.0, min(1.0, outcomes.get(trajectory.key, 0.5))) for trajectory in trajectories]
        self.global_weights = self._fit_linear(vectors, targets)
        for model in self.models:
            subset = [(vector, target) for vector, target, trajectory in zip(vectors, targets, trajectories) if trajectory.logged_model == model]
            self.model_weights[model] = self._fit_linear([item[0] for item in subset], [item[1] for item in subset]) if len(subset) >= 3 else list(self.global_weights)
            self.model_bias[model] = sum(item[1] for item in subset) / len(subset) - sum(targets) / len(targets) if subset else 0.0
        return self

    def predict_scores(self, trajectory: Trajectory) -> dict[str, float]:
        vector = self._vector(trajectory)
        difficulty = max(0.0, min(1.0, (extract_trajectory_features(trajectory).get("trajectory_tokens", 0.0) / 50000.0)))
        scores = {}
        for model in self.models:
            weights = self.model_weights.get(model, self.global_weights)
            prediction = sum(weight * value for weight, value in zip(weights, vector)) if weights else MODEL_STRENGTH.get(model, 0.5)
            score = 0.5 + 0.45 * math.tanh(prediction - 0.5) + 0.08 * MODEL_STRENGTH.get(model, 0.5) - 0.04 * difficulty * (1.0 - MODEL_STRENGTH.get(model, 0.5)) + self.model_bias.get(model, 0.0)
            scores[model] = max(0.0, min(1.0, score))
        return scores

    def route(self, trajectory: Trajectory) -> RouteDecision:
        scores = self.predict_scores(trajectory)
        costs = {model: self.cost_model.trajectory_cost(trajectory, [model] * trajectory.n_calls)[0] for model in self.models}
        model = max(scores, key=scores.get)
        return RouteDecision(model, [model] * trajectory.n_calls, scores[model], costs[model], score_uncertainty(scores), scores, costs)

    def save(self, path: str | Path) -> None:
        payload = {"models": self.models, "feature_names": self.feature_names, "means": self.means, "scales": self.scales, "global_weights": self.global_weights, "model_weights": self.model_weights, "model_bias": self.model_bias}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, cost_model: CostModel) -> "DifficultyRouter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        router = cls(payload["models"], cost_model)
        router.feature_names = payload["feature_names"]
        router.means = {key: float(value) for key, value in payload["means"].items()}
        router.scales = {key: float(value) for key, value in payload["scales"].items()}
        router.global_weights = [float(value) for value in payload["global_weights"]]
        router.model_weights = {key: [float(value) for value in values] for key, values in payload["model_weights"].items()}
        router.model_bias = {key: float(value) for key, value in payload["model_bias"].items()}
        return router

