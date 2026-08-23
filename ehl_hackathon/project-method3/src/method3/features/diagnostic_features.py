from __future__ import annotations

import json
from collections import Counter
from typing import Any

from method3.data.schema import Trajectory

# POST-HOC ONLY. Every one of these requires knowing how the trajectory actually
# unfolded (call count, whether errors happened, total tokens) — none of it exists
# before the trajectory runs. Legitimate uses: (1) the structural-proxy quality
# LABEL when no ground truth is available (a label may depend on the full outcome —
# that's supervised learning, not leakage), (2) cost/cache diagnostics and reporting.
# NEVER pass this into a routing decision — see features/opening_features.py and
# tests/test_no_feature_leakage.py.
DIAGNOSTIC_FEATURE_NAMES: tuple[str, ...] = (
    "n_calls", "total_tokens", "average_call_tokens",
    "function_calls", "function_outputs", "reasoning_items",
    "error_markers", "duplicate_arg_calls",
    "cacheable_tokens", "cacheable_fraction", "model_switches",
)


def _encoded(item: Any) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prefix_overlap_tokens(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> int:
    """Public: also used by pricing.cost_model for cache-aware repricing, so there is
    one implementation of "how much of the previous call's input this call reuses"."""
    shared = 0
    for left, right in zip(previous, current):
        left_text, right_text = _encoded(left), _encoded(right)
        if left_text != right_text:
            break
        shared += len(left_text)
    return shared // 4


def extract_diagnostic_features(trajectory: Trajectory) -> dict[str, float]:
    all_items = [item for call in trajectory.calls for item in call.input if isinstance(item, dict)]
    types = Counter(item.get("type", "message") for item in all_items)
    function_calls = types["function_call"] + types["custom_tool_call"]
    error_markers = sum(1 for item in all_items if "error" in str(item).lower() or "failed" in str(item).lower())
    call_args = [str(item.get("arguments", "")) for item in all_items if item.get("type") in {"function_call", "custom_tool_call"}]
    duplicate_arg_calls = max(0, len(call_args) - len(set(call_args)))
    overlaps = [prefix_overlap_tokens(left.input, right.input) for left, right in zip(trajectory.calls, trajectory.calls[1:])]
    total_tokens = trajectory.estimated_tokens
    switches = sum(1 for left, right in zip(trajectory.per_call_models, trajectory.per_call_models[1:]) if left != right)

    return {
        "n_calls": float(trajectory.n_calls),
        "total_tokens": float(total_tokens),
        "average_call_tokens": float(total_tokens / max(1, trajectory.n_calls)),
        "function_calls": float(function_calls),
        "function_outputs": float(types["function_call_output"] + types["custom_tool_call_output"]),
        "reasoning_items": float(types["reasoning"]),
        "error_markers": float(error_markers),
        "duplicate_arg_calls": float(duplicate_arg_calls),
        "cacheable_tokens": float(sum(overlaps)),
        "cacheable_fraction": float(sum(overlaps) / max(1, total_tokens)),
        "model_switches": float(switches),
    }
