from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from method3.data.schema import OpeningContext, Trajectory
from method3.features.opening_features import OPENING_FEATURE_NAMES, extract_opening_features
from method3.features.vectorize import Standardizer, euclidean_distance
from method3.quality.cold_start import ColdStartPrior


@dataclass
class Estimate:
    mean: float
    support: float          # effective neighbor count (inverse-distance weighted)
    avg_distance: float
    is_prior: bool = False  # True when a descriptor-based cold-start prior contributed


class KNNRewardModel:
    """Point estimator for E[quality | opening_context, model], fit ONLY on
    OPENING features (see features/opening_features.py) — never on the full
    trajectory. This is the base regressor `quality.conformal.SplitConformalCalibrator`
    calibrates a lower bound on top of; it deliberately does not compute its own
    uncertainty/lower-bound (that's conformal.py's job, kept as one well-tested
    module instead of duplicated ad hoc formulas per router)."""

    def __init__(self, models: list[str], neighbors: int = 12, cold_start: "ColdStartPrior | None" = None) -> None:
        self.models = list(models)
        self.neighbors = neighbors
        self.standardizer = Standardizer(OPENING_FEATURE_NAMES)
        self.examples: dict[str, list[tuple[list[float], float]]] = {model: [] for model in self.models}
        self.cold_start = cold_start

    def fit(self, trajectories: list[Trajectory], labels: dict[str, float]) -> "KNNRewardModel":
        if not trajectories:
            raise ValueError("at least one trajectory is required to fit a reward model")
        rows = [extract_opening_features(t.opening) for t in trajectories]
        self.standardizer.fit(rows)
        self.examples = {model: [] for model in self.models}
        for trajectory, row in zip(trajectories, rows):
            if trajectory.logged_model not in self.examples or trajectory.key not in labels:
                continue  # mixed-model trajectories don't cleanly attribute quality to one model
            vector = self.standardizer.transform(row)
            self.examples[trajectory.logged_model].append((vector, labels[trajectory.key]))
        return self

    def predict(self, opening: OpeningContext, model: str) -> Estimate:
        examples = self.examples.get(model, [])
        if not examples:
            # No direct evidence for this model at all (e.g. a model introduced
            # after the training window). Fall back to the descriptor-based
            # cold-start prior if one was attached; otherwise stay neutral.
            # Support stays 0.0 either way, so a support-gated router still treats
            # this as weak evidence rather than a measurement.
            if self.cold_start is not None:
                return Estimate(mean=self.cold_start.prior_for(model), support=0.0, avg_distance=1.0, is_prior=True)
            return Estimate(mean=0.5, support=0.0, avg_distance=1.0, is_prior=False)
        vector = self.standardizer.transform(extract_opening_features(opening))
        neighbors = sorted(((euclidean_distance(vector, item[0]), item[1]) for item in examples), key=lambda pair: pair[0])[: self.neighbors]
        weights = [1.0 / (0.05 + distance) for distance, _ in neighbors]
        total_weight = sum(weights)
        mean = sum(w * score for w, (_, score) in zip(weights, neighbors)) / total_weight
        effective = total_weight * total_weight / sum(w * w for w in weights)
        avg_distance = sum(w * distance for w, (distance, _) in zip(weights, neighbors)) / total_weight
        is_prior = False
        if self.cold_start is not None:
            # Shrink thin direct evidence toward the descriptor prior; with plenty
            # of neighbors this is a no-op.
            blended = self.cold_start.blend(model, mean, effective)
            is_prior = abs(blended - mean) > 1e-12
            mean = blended
        return Estimate(mean=max(0.0, min(1.0, mean)), support=effective, avg_distance=avg_distance, is_prior=is_prior)

    def save(self, path: str | Path) -> None:
        payload = {
            "models": self.models,
            "neighbors": self.neighbors,
            "standardizer": self.standardizer.to_dict(),
            "examples": self.examples,
            "cold_start": self.cold_start.to_dict() if self.cold_start is not None else None,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KNNRewardModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cold_start = ColdStartPrior.from_dict(payload["cold_start"]) if payload.get("cold_start") else None
        model = cls(payload["models"], int(payload["neighbors"]), cold_start=cold_start)
        model.standardizer = Standardizer.from_dict(payload["standardizer"])
        model.examples = {key: [(list(vector), float(score)) for vector, score in values] for key, values in payload["examples"].items()}
        return model
