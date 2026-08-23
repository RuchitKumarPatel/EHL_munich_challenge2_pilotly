from __future__ import annotations

from method3.data.schema import Trajectory
from method3.features.diagnostic_features import prefix_overlap_tokens

# Losing the KV-cache prefix on a mid-trajectory model switch is modeled below
# (previous call reset to None -> next call's overlap is 0). This constant additionally
# charges the real-world overhead of establishing a fresh session with the new model
# (re-sending system/tool-schema context, cold start) — see project-method1's
# pricing/cost_model.py, fixed this session after this exact charge was found to be
# computed but never actually added to the dollar total. 256 tokens ~= one typical
# system-prompt block; no empirical measurement backs this number, it's a documented
# placeholder pending real switch-cost telemetry (see docs/limitations.md).
DEFAULT_SWITCH_PENALTY_TOKENS = 256


class CostModel:
    def __init__(self, pricing: dict[str, dict[str, float]], switch_penalty_tokens: int = DEFAULT_SWITCH_PENALTY_TOKENS) -> None:
        self.pricing = pricing
        self.switch_penalty_tokens = switch_penalty_tokens

    def trajectory_cost(self, trajectory: Trajectory, route: list[str]) -> tuple[float, dict[str, float]]:
        """Reprice `trajectory` (post-hoc: uses the full call sequence) as if `route`
        (one model per call) had handled it. This is NOT a routing decision — it's the
        evaluation-time repricing every baseline and the proposed router's chosen model
        get graded through. `route` may legitimately be the real per-call model
        sequence (for the "logged" baseline, mixed-model-safe) or a single model
        repeated (for a trajectory-level routing decision)."""
        if len(route) != len(trajectory.calls):
            raise ValueError(f"route length {len(route)} must match trajectory calls {len(trajectory.calls)}")
        unknown = sorted({model for model in route if model not in self.pricing})
        if unknown:
            raise KeyError(f"no pricing entry for model(s) {unknown} — call pricing.ensure_models() with the full candidate+logged model set first")

        total = 0.0
        uncached = 0
        cached = 0
        switches = 0
        previous_call = None
        previous_model: str | None = None
        for call, model in zip(trajectory.calls, route):
            if previous_model is not None and model != previous_model:
                switches += 1
                previous_call = None
                uncached += self.switch_penalty_tokens
                total += (self.switch_penalty_tokens * self.pricing[model]["input"]) / 1000
            overlap = 0 if previous_call is None else min(call.estimated_tokens, prefix_overlap_tokens(previous_call.input, call.input))
            cached += overlap
            uncached += max(0, call.estimated_tokens - overlap)
            rates = self.pricing[model]
            total += ((call.estimated_tokens - overlap) * rates["input"] + overlap * rates["cached_input"]) / 1000
            previous_call = call
            previous_model = model
        return total, {"uncached_tokens": float(uncached), "cached_tokens": float(cached), "switches": float(switches)}
