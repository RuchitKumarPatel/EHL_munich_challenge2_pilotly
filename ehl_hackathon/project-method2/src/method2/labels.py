from __future__ import annotations

from collections import Counter

from .data import Trajectory


def observed_outcome(trajectory: Trajectory, ground_truth: dict[str, float] | None = None) -> tuple[float, float]:
    if ground_truth and trajectory.key in ground_truth:
        return ground_truth[trajectory.key], 1.0
    items = [item for call in trajectory.calls for item in call.input if isinstance(item, dict)]
    counts = Counter(item.get("type", "message") for item in items)
    calls = counts["function_call"] + counts["custom_tool_call"]
    outputs = counts["function_call_output"] + counts["custom_tool_call_output"]
    failures = sum("error" in str(item).lower() or "failed" in str(item).lower() for item in items)
    arguments = [str(item.get("arguments", "")) for item in items if item.get("type") in {"function_call", "custom_tool_call"}]
    duplicates = max(0, len(arguments) - len(set(arguments)))
    completion = min(1.0, outputs / max(1, calls))
    recovery = max(0.0, 1.0 - failures / max(1, calls + 1))
    efficiency = max(0.0, 1.0 - duplicates / max(1, calls))
    score = 0.5 * completion + 0.3 * recovery + 0.2 * efficiency
    confidence = 0.4 + 0.35 * min(1.0, calls / 3) + 0.25 * completion
    return max(0.0, min(1.0, score)), max(0.0, min(1.0, confidence))

