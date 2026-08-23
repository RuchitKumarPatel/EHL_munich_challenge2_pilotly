from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResponseThresholdTable:
    """Task specialization via response thresholds (Bonabeau et al.; Theraulaz et al.).

    In an ant or bee colony there is no central allocator. Each worker has a
    per-task response threshold; it engages when the task stimulus exceeds its
    threshold, and — critically — PERFORMING a task lowers that worker's threshold
    for it while not performing raises it. Specialists emerge from this feedback
    alone, and the colony stays robust when workers are lost or conditions shift.

    The canonical response function is

        P(engage) = s^n / (s^n + theta^n)

    with stimulus `s`, threshold `theta`, and steepness `n` (n = 2 in the original
    model). Low threshold -> engages readily.

    Mapping to routing: the "workers" are models, the "task stimulus" is task
    difficulty, and the threshold is per (model, task_type). This is a
    self-maintaining alternative to a static skill estimate:

      * it degrades gracefully under drift (thresholds keep moving, so a model that
        stops performing well is progressively disengaged rather than being locked
        in by a frozen training-set estimate);
      * it gives a NEW model a principled entry path — start from a
        descriptor-derived prior threshold and let evidence move it — instead of
        method3's crude price-tier average;
      * it needs no retraining pass, only online updates.

    Thresholds are clamped to [min_threshold, max_threshold] so a model can never
    become permanently unreachable (threshold -> infinity) or permanently dominant
    (threshold -> 0). That clamp is what keeps this compatible with method4's
    positivity guarantee.
    """
    steepness: float = 2.0
    learning_rate: float = 0.06
    min_threshold: float = 0.05
    max_threshold: float = 0.95
    default_threshold: float = 0.5
    thresholds: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _key(model: str, task_type: str) -> str:
        return f"{model}||{task_type}"

    def threshold(self, model: str, task_type: str) -> float:
        return self.thresholds.get(self._key(model, task_type), self.default_threshold)

    def set_threshold(self, model: str, task_type: str, value: float) -> None:
        self.thresholds[self._key(model, task_type)] = self._clamp(value)

    def _clamp(self, value: float) -> float:
        return max(self.min_threshold, min(self.max_threshold, value))

    def engagement_probability(self, model: str, task_type: str, stimulus: float) -> float:
        """P(this model engages this task) under the Bonabeau response function."""
        stimulus = max(0.0, stimulus)
        theta = self.threshold(model, task_type)
        numerator = stimulus ** self.steepness
        denominator = numerator + theta ** self.steepness
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def update(self, model: str, task_type: str, performed: bool, success: float | None = None) -> float:
        """Move a threshold after an observed episode.

        `performed=True` with a `success` score in [0, 1] lowers the threshold in
        proportion to how well it went (good outcomes specialize the model into
        that task); a poor outcome raises it back. `performed=False` raises the
        threshold slightly — the biological "not doing the task makes you less
        inclined to" term, which is what prevents every model drifting toward zero
        threshold and re-collapsing into a single generalist.
        """
        current = self.threshold(model, task_type)
        if performed:
            quality = 0.5 if success is None else max(0.0, min(1.0, success))
            # success 1.0 -> full decrease; success 0.0 -> full increase
            delta = -self.learning_rate * (2.0 * quality - 1.0)
        else:
            delta = self.learning_rate * 0.25
        updated = self._clamp(current + delta)
        self.thresholds[self._key(model, task_type)] = updated
        return updated

    def seed_from_priors(self, priors: dict[tuple[str, str], float]) -> "ResponseThresholdTable":
        """Initialize thresholds from prior expected quality: better expected
        quality -> lower threshold (more eager to take the task)."""
        for (model, task_type), expected_quality in priors.items():
            self.set_threshold(model, task_type, 1.0 - max(0.0, min(1.0, expected_quality)))
        return self

    def specialization_index(self, task_types: list[str], models: list[str]) -> float:
        """0 = every model equally eager at everything (no division of labor);
        higher = genuine specialization. Mean over task types of the spread of
        engagement thresholds across models."""
        if not task_types or len(models) < 2:
            return 0.0
        spreads = []
        for task_type in task_types:
            values = [self.threshold(m, task_type) for m in models]
            mean = sum(values) / len(values)
            spreads.append((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)
        return sum(spreads) / len(spreads)

    def to_dict(self) -> dict[str, object]:
        return {
            "steepness": self.steepness, "learning_rate": self.learning_rate,
            "min_threshold": self.min_threshold, "max_threshold": self.max_threshold,
            "default_threshold": self.default_threshold, "thresholds": self.thresholds,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ResponseThresholdTable":
        return cls(
            steepness=float(payload["steepness"]), learning_rate=float(payload["learning_rate"]),
            min_threshold=float(payload["min_threshold"]), max_threshold=float(payload["max_threshold"]),
            default_threshold=float(payload["default_threshold"]),
            thresholds={k: float(v) for k, v in payload["thresholds"].items()},
        )
