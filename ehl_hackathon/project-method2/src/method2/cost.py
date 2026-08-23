from __future__ import annotations

import json

from .data import Trajectory


DEFAULT_PRICING = {
    "claude-fable-5": (0.30, 0.03), "claude-sonnet-5": (3.0, 0.30), "claude-opus-4-8": (15.0, 1.50), "claude-opus-5": (15.0, 1.50), "gpt-5.6-sol": (0.30, 0.03), "gpt-5.6-luna": (1.0, 0.10), "gpt-5.6-terra": (2.0, 0.20)
}


def pricing(models: list[str]) -> dict[str, tuple[float, float]]:
    values = dict(DEFAULT_PRICING)
    for model in models:
        if model not in values:
            rate = 3.0 if "sonnet" in model or "luna" in model else 15.0 if "opus" in model or "terra" in model else 0.30
            values[model] = (rate, rate * 0.1)
    return values


def _overlap(previous: list[dict], current: list[dict]) -> int:
    total = 0
    for left, right in zip(previous, current):
        first = json.dumps(left, sort_keys=True, separators=(",", ":"))
        second = json.dumps(right, sort_keys=True, separators=(",", ":"))
        if first != second:
            break
        total += len(first)
    return total // 4


def trajectory_cost(trajectory: Trajectory, model: str, prices: dict[str, tuple[float, float]]) -> float:
    full, cached = prices[model]
    total = 0.0
    previous = None
    for call in trajectory.calls:
        overlap = 0 if previous is None else min(call.tokens, _overlap(previous.input, call.input))
        total += ((call.tokens - overlap) * full + overlap * cached) / 1_000_000
        previous = call
    return total


# Placeholder for the real-world overhead of establishing a fresh session with a new
# model (re-sending system/tool-schema context, cold start) on top of the cache-credit
# loss already modeled below. 256 tokens ~= one typical system-prompt block; no
# empirical measurement backs this number.
SWITCH_PENALTY_TOKENS = 256


def trajectory_cost_route(trajectory: Trajectory, route: list[str], prices: dict[str, tuple[float, float]],
                           switch_penalty_tokens: int = SWITCH_PENALTY_TOKENS) -> float:
    """Cost a per-call route (one model per call, not necessarily uniform) — needed to
    replay a trajectory whose *actual* logged model switched mid-session
    (`trajectory.model == "mixed"`). A switch resets the cache prefix (previous=None)
    and charges switch_penalty_tokens at the new model's input rate."""
    if len(route) != len(trajectory.calls):
        raise ValueError("route length must match trajectory calls")
    total = 0.0
    previous = None
    previous_model = None
    for call, model in zip(trajectory.calls, route):
        full, cached_rate = prices[model]
        if previous_model is not None and model != previous_model:
            previous = None
            total += switch_penalty_tokens * full / 1_000_000
        overlap = 0 if previous is None else min(call.tokens, _overlap(previous.input, call.input))
        total += ((call.tokens - overlap) * full + overlap * cached_rate) / 1_000_000
        previous = call
        previous_model = model
    return total

