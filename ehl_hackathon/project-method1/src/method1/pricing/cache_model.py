from __future__ import annotations

from method1.data.schema import RequestRecord
from method1.features.cache_features import common_prefix_tokens


def call_cost(previous: RequestRecord | None, current: RequestRecord, model: str, pricing: dict[str, dict[str, float]]) -> float:
    if model not in pricing:
        raise KeyError(f"no pricing for model {model}")
    tokens = current.estimated_tokens
    cached = common_prefix_tokens(previous, current) if previous is not None else 0
    cached = min(tokens, cached)
    rates = pricing[model]
    return ((tokens - cached) * rates["input"] + cached * rates["cached_input"]) / 1000

