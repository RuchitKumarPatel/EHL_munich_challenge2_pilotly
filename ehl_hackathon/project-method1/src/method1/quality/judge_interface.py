from __future__ import annotations

from typing import Protocol

from method1.data.schema import Trajectory

from .deterministic_signals import trajectory_signals


class Judge(Protocol):
    def score(self, trajectory: Trajectory) -> float:
        ...


class HeuristicJudge:
    def score(self, trajectory: Trajectory) -> float:
        signals = trajectory_signals(trajectory)
        return 0.35 * signals["tool_success"] + 0.25 * signals["completion_proxy"] + 0.2 * signals["error_recovery"] + 0.2 * signals["non_repetition"]

