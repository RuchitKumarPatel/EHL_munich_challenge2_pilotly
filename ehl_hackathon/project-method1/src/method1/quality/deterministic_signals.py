from __future__ import annotations

from collections import Counter

from method1.data.schema import Trajectory


def trajectory_signals(trajectory: Trajectory) -> dict[str, float]:
    items = [item for call in trajectory.calls for item in call.input if isinstance(item, dict)]
    types = Counter(item.get("type", "missing") for item in items)
    calls = types.get("function_call", 0) + types.get("custom_tool_call", 0)
    outputs = types.get("function_call_output", 0) + types.get("custom_tool_call_output", 0)
    errors = sum(1 for item in items if "error" in str(item).lower() or "failed" in str(item).lower())
    repetitions = max(0, calls - len({str(item.get("arguments", item.get("input", ""))) for item in items if item.get("type") in {"function_call", "custom_tool_call"}}))
    tool_success = outputs / max(1, calls)
    completion = 1.0 if trajectory.n_calls >= 2 else 0.5
    recovery = 1.0 if errors == 0 else max(0.0, 1.0 - errors / max(1, calls))
    repetition = max(0.0, 1.0 - repetitions / max(1, calls))
    return {
        "tool_success": min(1.0, tool_success),
        "completion_proxy": completion,
        "error_recovery": recovery,
        "non_repetition": repetition,
        "calls": float(trajectory.n_calls),
        "errors": float(errors),
    }

