from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from .data import Trajectory


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _texts(item)]
    return []


def _prefix_tokens(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    shared = 0
    for first, second in zip(left, right):
        first_text = json.dumps(first, sort_keys=True, separators=(",", ":"))
        second_text = json.dumps(second, sort_keys=True, separators=(",", ":"))
        if first_text != second_text:
            break
        shared += len(first_text)
    return shared // 4


def extract(trajectory: Trajectory) -> dict[str, float]:
    first = trajectory.calls[0]
    text = " ".join(_texts(first.input))
    words = re.findall(r"[a-zA-Z0-9_<>-]+", text.lower())
    all_items = [item for call in trajectory.calls for item in call.input if isinstance(item, dict)]
    types = Counter(item.get("type", "message") for item in all_items)
    overlaps = [_prefix_tokens(first_call.input, second_call.input) for first_call, second_call in zip(trajectory.calls, trajectory.calls[1:])]
    failures = sum("error" in str(item).lower() or "failed" in str(item).lower() for item in all_items)
    return {
        "prompt_tokens": float(first.tokens),
        "trajectory_tokens": float(sum(call.tokens for call in trajectory.calls)),
        "calls": float(len(trajectory.calls)),
        "tools": float(len(first.tools)),
        "text_words": float(len(words)),
        "unique_words": float(len(set(words))),
        "function_calls": float(types["function_call"] + types["custom_tool_call"]),
        "function_outputs": float(types["function_call_output"] + types["custom_tool_call_output"]),
        "reasoning": float(types["reasoning"]),
        "failures": float(failures),
        "cacheable_tokens": float(sum(overlaps)),
        "cacheable_fraction": float(sum(overlaps) / max(1, sum(call.tokens for call in trajectory.calls))),
        "has_image": float("input_image" in str(first.input)),
        "has_code": float("apply_patch" in text or "```" in text),
    }


def vector(features: dict[str, float], names: list[str], means: dict[str, float], scales: dict[str, float]) -> list[float]:
    return [(features.get(name, 0.0) - means[name]) / scales[name] for name in names]


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / max(1, len(left)))

