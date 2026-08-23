from __future__ import annotations

from method1.data.schema import Trajectory

from .deterministic_signals import trajectory_signals


def composite_outcome(trajectory: Trajectory) -> float:
    signals = trajectory_signals(trajectory)
    value = 0.35 * signals["tool_success"] + 0.25 * signals["completion_proxy"] + 0.2 * signals["error_recovery"] + 0.2 * signals["non_repetition"]
    return max(0.0, min(1.0, value))


def calibrated_outcome(trajectory: Trajectory, ground_truth: dict[str, float]) -> tuple[float, str]:
    """Prefer a recovered ground-truth label (dataset1's graded subset) over the
    structural proxy; falls back to composite_outcome when the trajectory isn't graded
    or no manifest was found (ground_truth == {})."""
    if trajectory.key in ground_truth:
        return ground_truth[trajectory.key], "ground_truth"
    return composite_outcome(trajectory), "proxy"

