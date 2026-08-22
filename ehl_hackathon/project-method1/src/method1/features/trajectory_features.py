from __future__ import annotations

from collections import Counter

from method1.data.schema import Trajectory

from .cache_features import extract_cache_features
from .text_features import extract_text_features
from .tool_features import extract_tool_features


def extract_trajectory_features(trajectory: Trajectory) -> dict[str, float]:
    first = trajectory.calls[0]
    item_types = Counter(item.get("type", "missing") for call in trajectory.calls for item in call.input if isinstance(item, dict))
    features = {}
    features.update(extract_text_features(first.input))
    features.update(extract_tool_features(first.input, first.tools))
    features.update(extract_cache_features(trajectory.calls))
    features.update({
        "trajectory_calls": float(trajectory.n_calls),
        "trajectory_tokens": float(trajectory.estimated_tokens),
        "average_call_tokens": float(trajectory.estimated_tokens / max(1, trajectory.n_calls)),
        "trajectory_function_calls": float(item_types.get("function_call", 0) + item_types.get("custom_tool_call", 0)),
        "trajectory_function_outputs": float(item_types.get("function_call_output", 0) + item_types.get("custom_tool_call_output", 0)),
        "trajectory_reasoning_items": float(item_types.get("reasoning", 0)),
        "trajectory_error_markers": float(sum(1 for call in trajectory.calls for item in call.input if isinstance(item, dict) and "error" in str(item).lower())),
    })
    return features

