from __future__ import annotations

from method1.data.schema import RequestRecord, Trajectory
from method1.features.cache_features import common_prefix_tokens

from .cache_model import call_cost


class CostModel:
    def __init__(self, pricing: dict[str, dict[str, float]], switch_penalty_tokens: int = 0) -> None:
        self.pricing = pricing
        self.switch_penalty_tokens = switch_penalty_tokens

    def trajectory_cost(self, trajectory: Trajectory, route: list[str]) -> tuple[float, dict[str, float]]:
        if len(route) != len(trajectory.calls):
            raise ValueError("route length must match trajectory calls")
        total = 0.0
        uncached = 0
        cached = 0
        switches = 0
        previous_call = None
        previous_model = None
        for call, model in zip(trajectory.calls, route):
            if model not in self.pricing:
                raise KeyError(f"no pricing for model {model}")
            if previous_model is not None and model != previous_model:
                switches += 1
                previous_call = None
                uncached += self.switch_penalty_tokens
            overlap = 0 if previous_call is None else min(call.estimated_tokens, common_prefix_tokens(previous_call, call))
            cached += overlap
            uncached += max(0, call.estimated_tokens - overlap)
            total += call_cost(previous_call, call, model, self.pricing)
            previous_call = call
            previous_model = model
        return total, {"uncached_tokens": float(uncached), "cached_tokens": float(cached), "switches": float(switches)}
