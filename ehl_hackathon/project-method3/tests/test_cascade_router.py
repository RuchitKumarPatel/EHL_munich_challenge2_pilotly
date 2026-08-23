from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data.schema import Call, OpeningContext, Trajectory
from method3.quality.conformal import SplitConformalCalibrator
from method3.quality.reward_model import Estimate
from method3.routing.base import Router
from method3.routing.cascade_router import CascadeRouter

PRICING = {
    "cheap": {"input": 0.000001, "cached_input": 0.0000001},
    "mid": {"input": 0.000005, "cached_input": 0.0000005},
    "pricey": {"input": 0.00002, "cached_input": 0.000002},
}


class FakeRewardModel:
    """Test double: fixed per-model Estimate, no real fitting needed — isolates the
    cascade DECISION LOGIC from the KNN regressor's behavior."""
    def __init__(self, table: dict[str, Estimate]) -> None:
        self.table = table

    def predict(self, opening: OpeningContext, model: str) -> Estimate:
        return self.table[model]


def _opening() -> OpeningContext:
    history = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
    ]
    call = Call("cheap", history, [], "test", 0)
    return Trajectory("k", [call]).opening


def _guaranteed_calibrator(quantile: float) -> SplitConformalCalibrator:
    return SplitConformalCalibrator(alpha=0.1, quantile=quantile, n_calibration=50, coverage_guaranteed=True)


def _unguaranteed_calibrator() -> SplitConformalCalibrator:
    return SplitConformalCalibrator(alpha=0.1, quantile=float("inf"), n_calibration=1, coverage_guaranteed=False)


def test_conforms_to_router_protocol():
    router = CascadeRouter(FakeRewardModel({"cheap": Estimate(0.9, 10, 0.1)}), {}, PRICING)
    assert isinstance(router, Router)


def test_picks_cheapest_among_eligible():
    reward = FakeRewardModel({
        "cheap": Estimate(mean=0.85, support=10, avg_distance=0.1),
        "mid": Estimate(mean=0.95, support=10, avg_distance=0.1),
        "pricey": Estimate(mean=0.97, support=10, avg_distance=0.1),
    })
    calibrators = {m: _guaranteed_calibrator(0.05) for m in PRICING}  # lower ~= mean - 0.05, all clear a 0.5 floor
    router = CascadeRouter(reward, calibrators, PRICING, quality_floor=0.5, minimum_support=3.0)
    decision = router.route(_opening(), list(PRICING))
    assert decision.model == "cheap"
    assert decision.reason == "cascade_cheapest_safe"


def test_escalates_when_cheapest_does_not_clear_floor():
    reward = FakeRewardModel({
        "cheap": Estimate(mean=0.3, support=10, avg_distance=0.1),   # too low quality
        "mid": Estimate(mean=0.9, support=10, avg_distance=0.1),
        "pricey": Estimate(mean=0.95, support=10, avg_distance=0.1),
    })
    calibrators = {m: _guaranteed_calibrator(0.05) for m in PRICING}
    router = CascadeRouter(reward, calibrators, PRICING, quality_floor=0.7, minimum_support=3.0)
    decision = router.route(_opening(), list(PRICING))
    assert decision.model == "mid"  # cheapest of {mid, pricey} that clears 0.7
    assert decision.reason == "cascade_cheapest_safe"


def test_falls_back_to_best_predicted_when_nothing_clears_floor():
    reward = FakeRewardModel({
        "cheap": Estimate(mean=0.5, support=10, avg_distance=0.1),
        "mid": Estimate(mean=0.6, support=10, avg_distance=0.1),
        "pricey": Estimate(mean=0.55, support=10, avg_distance=0.1),
    })
    calibrators = {m: _guaranteed_calibrator(0.5) for m in PRICING}  # huge quantile: nobody clears 0.9 floor
    router = CascadeRouter(reward, calibrators, PRICING, quality_floor=0.9, minimum_support=3.0)
    decision = router.route(_opening(), list(PRICING))
    assert decision.model == "mid"  # highest predicted mean among supported candidates
    assert decision.reason == "cascade_fallback_best_predicted"


def test_falls_back_to_no_support_pool_when_nothing_has_support():
    reward = FakeRewardModel({
        "cheap": Estimate(mean=0.4, support=0.5, avg_distance=1.0),
        "mid": Estimate(mean=0.8, support=0.5, avg_distance=1.0),
    })
    calibrators = {m: _guaranteed_calibrator(0.01) for m in ("cheap", "mid")}
    router = CascadeRouter(reward, calibrators, PRICING, quality_floor=0.5, minimum_support=3.0)
    decision = router.route(_opening(), ["cheap", "mid"])
    assert decision.model == "mid"
    assert decision.reason == "cascade_fallback_no_support"


def test_uncalibrated_model_can_never_be_eligible_even_with_high_mean():
    reward = FakeRewardModel({
        "cheap": Estimate(mean=0.99, support=10, avg_distance=0.1),  # high mean, NOT calibrated
        "mid": Estimate(mean=0.8, support=10, avg_distance=0.1),
    })
    calibrators = {"cheap": _unguaranteed_calibrator(), "mid": _guaranteed_calibrator(0.05)}
    router = CascadeRouter(reward, calibrators, PRICING, quality_floor=0.5, minimum_support=3.0)
    decision = router.route(_opening(), ["cheap", "mid"])
    assert decision.model == "mid"
    assert decision.candidates["cheap"]["lower"] == 0.0


def test_falls_back_to_pooled_global_calibrator_when_per_model_not_guaranteed():
    reward = FakeRewardModel({
        "cheap": Estimate(mean=0.9, support=10, avg_distance=0.1),
        "mid": Estimate(mean=0.6, support=10, avg_distance=0.1),
    })
    calibrators = {
        "cheap": _unguaranteed_calibrator(),   # too little per-model calibration data
        "mid": _guaranteed_calibrator(0.05),
        "__global__": _guaranteed_calibrator(0.1),  # pooled: more data, still guaranteed
    }
    router = CascadeRouter(reward, calibrators, PRICING, quality_floor=0.7, minimum_support=3.0)
    decision = router.route(_opening(), ["cheap", "mid"])
    # "cheap" has no per-model guarantee but the pooled calibrator gives it
    # lower = 0.9 - 0.1 = 0.8, which clears the 0.7 floor and is cheaper than "mid"
    assert decision.candidates["cheap"]["lower"] == 0.8
    assert decision.model == "cheap"
    assert decision.reason == "cascade_cheapest_safe"


def test_missing_calibrator_treated_as_uncalibrated_not_a_crash():
    reward = FakeRewardModel({"cheap": Estimate(mean=0.9, support=10, avg_distance=0.1)})
    router = CascadeRouter(reward, {}, PRICING, quality_floor=0.5, minimum_support=3.0)
    decision = router.route(_opening(), ["cheap"])
    assert decision.candidates["cheap"]["lower"] == 0.0
    assert decision.reason == "cascade_fallback_best_predicted"
