from __future__ import annotations

from method3.data.schema import OpeningContext
from method3.quality.conformal import SplitConformalCalibrator
from method3.quality.reward_model import KNNRewardModel
from method3.routing.base import RouteDecision

# RouteLLM-style win/quality predictor + FrugalGPT-style cascade, reinterpreted as a
# SINGLE-SHOT decision: predict whether the cheapest candidate's calibrated quality
# floor is met, choose it if so, escalate to a stronger candidate otherwise. A literal
# FrugalGPT cascade makes a real call to the cheap model and inspects its actual
# output before deciding to escalate; that needs live model access this project does
# not have, and — per "The Replay Gap" (arXiv:2608.08239) — a decision made mid-way
# through a trajectory cannot be honestly validated by statically replaying logged
# single-model data anyway. So this is a single upfront decision, matching what the
# historical trajectories actually are (one model, decided before the first call).


class CascadeRouter:
    """The proposed method3 policy. Conforms to the `Router` protocol: `route` sees
    only an OpeningContext, never a Trajectory — see routing/base.py and
    tests/test_no_feature_leakage.py."""

    def __init__(
        self,
        reward_model: KNNRewardModel,
        calibrators: dict[str, SplitConformalCalibrator],
        pricing: dict[str, dict[str, float]],
        quality_floor: float = 0.5,
        minimum_support: float = 3.0,
    ) -> None:
        self.reward_model = reward_model
        self.calibrators = calibrators
        self.pricing = pricing
        self.quality_floor = quality_floor
        self.minimum_support = minimum_support

    def _price_rate(self, model: str) -> float:
        return self.pricing[model]["input"]

    def route(self, opening: OpeningContext, candidate_models: list[str]) -> RouteDecision:
        global_calibrator = self.calibrators.get("__global__")
        candidates: dict[str, dict[str, float]] = {}
        for model in candidate_models:
            estimate = self.reward_model.predict(opening, model)
            calibrator = self.calibrators.get(model)
            if calibrator is not None and calibrator.coverage_guaranteed:
                lower = calibrator.lower_bound(estimate.mean)
            elif global_calibrator is not None and global_calibrator.coverage_guaranteed:
                # Per-model calibration set too small to certify anything (see
                # quality/conformal.py) — fall back to the pooled calibrator, which
                # has far more data and still gives a valid marginal guarantee.
                lower = global_calibrator.lower_bound(estimate.mean)
            else:
                lower = 0.0  # no certified floor at all: never counts as "safe"
            candidates[model] = {
                "mean": estimate.mean,
                "lower": lower,
                "support": estimate.support,
                "price_rate": self._price_rate(model),
            }

        eligible = [m for m, row in candidates.items() if row["support"] >= self.minimum_support and row["lower"] >= self.quality_floor]

        if eligible:
            selected = min(eligible, key=lambda m: candidates[m]["price_rate"])
            reason = "cascade_cheapest_safe"
        else:
            supported = [m for m, row in candidates.items() if row["support"] >= self.minimum_support]
            pool = supported or list(candidate_models)
            selected = max(pool, key=lambda m: candidates[m]["mean"])
            reason = "cascade_fallback_best_predicted" if supported else "cascade_fallback_no_support"

        row = candidates[selected]
        predicted_cost = opening.first_call_tokens * row["price_rate"] / 1000  # first-call proxy, not full-trajectory cost — see evaluation/offline_policy_eval.py for the real post-hoc cost
        return RouteDecision(
            model=selected,
            predicted_quality=row["mean"],
            quality_lower=row["lower"],
            quality_upper=min(1.0, row["mean"] + max(0.0, row["mean"] - row["lower"])),
            predicted_cost=predicted_cost,
            reason=reason,
            candidates=candidates,
        )
