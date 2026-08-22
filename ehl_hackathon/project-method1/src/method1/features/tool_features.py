from __future__ import annotations

from collections import Counter
from typing import Any


def extract_tool_features(items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, float]:
    types = Counter(item.get("type", "missing") for item in items if isinstance(item, dict))
    tool_names = Counter(tool.get("name", tool.get("type", "unknown")) for tool in tools if isinstance(tool, dict))
    return {
        "available_tools": float(len(tools)),
        "function_calls": float(types.get("function_call", 0) + types.get("custom_tool_call", 0)),
        "function_outputs": float(types.get("function_call_output", 0) + types.get("custom_tool_call_output", 0)),
        "reasoning_items": float(types.get("reasoning", 0)),
        "message_items": float(types.get("message", 0) + types.get("missing", 0)),
        "distinct_tool_names": float(len(tool_names)),
        "image_items": float(sum(1 for item in items if isinstance(item, dict) and item.get("type") == "input_image")),
    }

