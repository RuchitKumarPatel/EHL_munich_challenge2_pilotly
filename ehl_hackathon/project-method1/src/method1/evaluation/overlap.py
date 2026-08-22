from __future__ import annotations

from collections import Counter

from method1.data.schema import Trajectory


def support_report(trajectories: list[Trajectory], models: list[str]) -> dict[str, object]:
    counts = Counter(trajectory.logged_model for trajectory in trajectories)
    total = len(trajectories)
    probabilities = {model: counts.get(model, 0) / max(1, total) for model in models}
    effective_sample_size = 1.0 / sum(value * value for value in probabilities.values() if value > 0) if probabilities else 0.0
    return {"counts": dict(counts), "propensities": probabilities, "effective_sample_size": effective_sample_size, "supported_models": [model for model in models if counts.get(model, 0) > 0]}

