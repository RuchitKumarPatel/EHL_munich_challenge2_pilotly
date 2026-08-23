from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from method3.data.schema import OpeningContext, Trajectory
from method3.features.opening_features import OPENING_FEATURE_NAMES, extract_opening_features
from method3.features.vectorize import Standardizer


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(value - top) for value in logits]  # subtract max: numerically stable, no overflow
    total = sum(exps)
    return [value / total for value in exps]


@dataclass
class LoggingPolicyModel:
    """P(model chosen | opening context), fit on the actual historical assignment —
    used as the propensity score in doubly-robust off-policy evaluation.

    Why this exists (replacing a class-frequency propensity): a constant/global
    propensity implicitly assumes the logging policy assigned models uniformly at
    random regardless of context (MCAR). If the real logging policy was
    context-dependent (harder tasks routed to stronger models, say), a
    context-free propensity biases the IPS/DR correction term. This is a
    multinomial logistic regression (softmax) over OPENING features ONLY — the
    logging policy, whatever it was, could only have used information available
    before the trajectory ran, so this is not a leakage risk the way training the
    reward model on full-trajectory stats would be.

    Fit only on non-"mixed" trajectories (a switched trajectory has no single
    "the model the policy chose").
    """
    models: list[str]
    standardizer: Standardizer = None  # type: ignore[assignment]
    weights: dict[str, list[float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.standardizer is None:
            self.standardizer = Standardizer(OPENING_FEATURE_NAMES)
        if self.weights is None:
            self.weights = {model: [] for model in self.models}

    def fit(self, trajectories: list[Trajectory], steps: int = 800, learning_rate: float = 0.05, l2: float = 0.02) -> "LoggingPolicyModel":
        examples = [t for t in trajectories if t.logged_model in self.models]
        if not examples:
            raise ValueError("no non-mixed trajectories available to fit a logging-policy model")
        rows = [extract_opening_features(t.opening) for t in examples]
        self.standardizer.fit(rows)
        vectors = [[1.0] + self.standardizer.transform(row) for row in rows]
        dim = len(vectors[0])
        targets = [[1.0 if t.logged_model == model else 0.0 for model in self.models] for t in examples]

        weights = {model: [0.0] * dim for model in self.models}
        n = len(vectors)
        for _ in range(steps):
            gradients = {model: [0.0] * dim for model in self.models}
            for vector, target_row in zip(vectors, targets):
                logits = [sum(w * x for w, x in zip(weights[model], vector)) for model in self.models]
                probs = _softmax(logits)
                for model, prob, target in zip(self.models, probs, target_row):
                    error = prob - target
                    for index, x in enumerate(vector):
                        gradients[model][index] += error * x
            scale = 1.0 / n
            for model in self.models:
                for index in range(dim):
                    penalty = l2 * weights[model][index] if index else 0.0  # never regularize the bias term
                    weights[model][index] -= learning_rate * (gradients[model][index] * scale + penalty)
        self.weights = weights
        return self

    def held_out_log_loss(self, trajectories: list[Trajectory]) -> float:
        """Mean negative log-likelihood of the logged action on held-out data —
        the model-selection criterion used by `fit_selected`."""
        total = 0.0
        count = 0
        for trajectory in trajectories:
            if trajectory.logged_model not in self.models:
                continue
            probability = self.predict_proba(trajectory.opening).get(trajectory.logged_model, 1e-12)
            total += -math.log(max(probability, 1e-12))
            count += 1
        return total / count if count else float("inf")

    @classmethod
    def fit_selected(
        cls,
        models: list[str],
        train: list[Trajectory],
        validation: list[Trajectory],
        l2_grid: tuple[float, ...] = (0.02, 0.15, 0.6, 2.0),
        steps: int = 400,
        learning_rate: float = 0.10,
    ) -> tuple["LoggingPolicyModel", dict[str, object]]:
        """Fit at several regularization strengths and keep the one with the best
        held-out log-loss on `validation`.

        A fixed regularization strength is a real bug source here, not a tuning
        nicety: measured on dataset2, the previously hard-coded l2=0.02 OVERFIT
        badly enough that the fitted propensity model was worse than a context-free
        frequency baseline (test MAE 0.0635 vs 0.0405), which silently invalidates
        every importance-weighted estimate built on it. Training longer made it
        worse, not better. Selecting l2 on held-out data fixes it (MAE 0.0328 at
        l2=2.0) and, crucially, the validation criterion ranks the candidates in
        the same order as the true-propensity error — so the selection is honest
        and never consults the test split.
        """
        if not validation:
            # No held-out data to select on: fall back to the most regularized
            # option rather than the least, since overfitting is the failure mode
            # actually observed here.
            model = cls(models).fit(train, steps=steps, learning_rate=learning_rate, l2=max(l2_grid))
            return model, {"selected_l2": max(l2_grid), "selection": "no_validation_data_fallback"}
        scores = []
        best_model = None
        best_l2 = None
        best_loss = float("inf")
        for l2 in l2_grid:
            candidate = cls(models).fit(train, steps=steps, learning_rate=learning_rate, l2=l2)
            loss = candidate.held_out_log_loss(validation)
            scores.append({"l2": l2, "validation_log_loss": loss})
            if loss < best_loss:
                best_model, best_l2, best_loss = candidate, l2, loss
        return best_model, {"selected_l2": best_l2, "validation_log_loss": best_loss, "grid": scores}

    def predict_proba(self, opening: OpeningContext) -> dict[str, float]:
        vector = [1.0] + self.standardizer.transform(extract_opening_features(opening))
        logits = [sum(w * x for w, x in zip(self.weights[model], vector)) for model in self.models]
        probs = _softmax(logits)
        return dict(zip(self.models, probs))

    def propensity(self, opening: OpeningContext, model: str, clip: float = 0.05) -> float:
        if model not in self.models:
            return clip
        return max(clip, self.predict_proba(opening)[model])

    def to_dict(self) -> dict[str, object]:
        return {"models": self.models, "standardizer": self.standardizer.to_dict(), "weights": self.weights}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LoggingPolicyModel":
        instance = cls(list(payload["models"]))
        instance.standardizer = Standardizer.from_dict(payload["standardizer"])
        instance.weights = {key: list(value) for key, value in payload["weights"].items()}
        return instance

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LoggingPolicyModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
