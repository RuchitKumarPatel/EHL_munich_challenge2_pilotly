from __future__ import annotations

from method3.data.schema import Trajectory
from method3.features.diagnostic_features import extract_diagnostic_features


def structural_proxy(trajectory: Trajectory) -> float:
    """Heuristic quality label from post-hoc structural signals, used only when no
    ground-truth label exists. This is a LABEL (target), not a routing FEATURE — it
    is fine for a label to depend on the full trajectory outcome; that is ordinary
    supervised learning. Same signal family as project-method1's composite_outcome,
    rebuilt on this project's own diagnostic_features so there is one source of
    truth instead of two hand-tuned formulas drifting apart."""
    d = extract_diagnostic_features(trajectory)
    calls = max(1.0, d["function_calls"])
    tool_success = min(1.0, d["function_outputs"] / calls)
    completion = 1.0 if trajectory.n_calls >= 2 else 0.5
    error_recovery = max(0.0, 1.0 - d["error_markers"] / calls)
    non_repetition = max(0.0, 1.0 - d["duplicate_arg_calls"] / calls)
    value = 0.35 * tool_success + 0.25 * completion + 0.2 * error_recovery + 0.2 * non_repetition
    return max(0.0, min(1.0, value))


def calibrated_outcome(trajectory: Trajectory, ground_truth: dict[str, float]) -> tuple[float, str]:
    """Prefer a recovered ground-truth label (dataset1's graded subset) over the
    structural proxy; falls back to structural_proxy when ungraded or no manifest
    was found (ground_truth == {})."""
    if trajectory.key in ground_truth:
        return ground_truth[trajectory.key], "ground_truth"
    return structural_proxy(trajectory), "proxy"
