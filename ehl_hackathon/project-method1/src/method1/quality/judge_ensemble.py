from __future__ import annotations

from statistics import mean, pstdev

from method1.data.schema import Trajectory

from .judge_interface import Judge


class JudgeEnsemble:
    def __init__(self, judges: list[Judge]) -> None:
        if not judges:
            raise ValueError("at least one judge is required")
        self.judges = judges

    def score(self, trajectory: Trajectory) -> tuple[float, float]:
        values = [max(0.0, min(1.0, float(judge.score(trajectory)))) for judge in self.judges]
        return mean(values), pstdev(values) if len(values) > 1 else 0.0

