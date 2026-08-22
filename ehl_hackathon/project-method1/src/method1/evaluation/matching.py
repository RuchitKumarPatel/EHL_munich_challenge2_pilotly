from __future__ import annotations

from collections import defaultdict

from method1.data.schema import Trajectory
from method1.features.trajectory_features import extract_trajectory_features


def complexity_bin(trajectory: Trajectory) -> str:
    tokens = extract_trajectory_features(trajectory)["trajectory_tokens"]
    if tokens < 5000:
        return "small"
    if tokens < 15000:
        return "medium"
    return "large"


def matched_outcomes(trajectories: list[Trajectory], outcomes: dict[str, float]) -> dict[str, dict[str, float]]:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.logged_model != "mixed" and trajectory.key in outcomes:
            cells[(complexity_bin(trajectory), trajectory.logged_model)].append(outcomes[trajectory.key])
    output: dict[str, dict[str, float]] = {}
    for (bucket, model), values in sorted(cells.items()):
        output[f"{bucket}:{model}"] = {"n": float(len(values)), "mean_quality": sum(values) / len(values)}
    return output

