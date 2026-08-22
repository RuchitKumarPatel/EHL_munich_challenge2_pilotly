from __future__ import annotations

from method1.data.schema import Trajectory

from .deterministic_signals import trajectory_signals


def composite_outcome(trajectory: Trajectory) -> float:
    signals = trajectory_signals(trajectory)
    value = 0.35 * signals["tool_success"] + 0.25 * signals["completion_proxy"] + 0.2 * signals["error_recovery"] + 0.2 * signals["non_repetition"]
    return max(0.0, min(1.0, value))

