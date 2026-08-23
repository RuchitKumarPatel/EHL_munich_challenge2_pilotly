from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data.schema import Call, Trajectory
from method3.pricing.cost_model import CostModel
from method3.routing.baselines import BaselineOracle

PRICING = {
    "cheap": {"input": 0.000001, "cached_input": 0.0000001},
    "mid": {"input": 0.000005, "cached_input": 0.0000005},
    "pricey": {"input": 0.00002, "cached_input": 0.000002},
}


def _trajectory(models: list[str]) -> Trajectory:
    history: list[dict] = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
    ]
    calls = []
    for i, model in enumerate(models):
        calls.append(Call(model, [dict(item) for item in history], [], "test", i))
        call_id = f"c{i}"
        history.append({"type": "function_call", "name": "run_shell", "arguments": f"{{\"cmd\": \"{i}\"}}", "call_id": call_id})
        history.append({"type": "function_call_output", "call_id": call_id, "output": "ok"})
    return Trajectory("k", calls)


def _oracle() -> BaselineOracle:
    return BaselineOracle(list(PRICING), CostModel(PRICING))


def test_logged_baseline_on_mixed_trajectory_does_not_crash():
    # Regression test for the exact crash found this session.
    trajectory = _trajectory(["cheap", "cheap", "pricey"])
    decision = _oracle().logged(trajectory)
    assert decision.model == "mixed"
    assert decision.route == ["cheap", "cheap", "pricey"]
    assert decision.cost > 0


def test_logged_baseline_on_pure_trajectory():
    trajectory = _trajectory(["mid", "mid"])
    decision = _oracle().logged(trajectory)
    assert decision.model == "mid"
    assert decision.route == ["mid", "mid"]


def test_cheapest_picks_lowest_cost_candidate():
    trajectory = _trajectory(["mid", "mid", "mid"])
    decision = _oracle().cheapest(trajectory)
    assert decision.model == "cheap"


def test_strongest_picks_highest_strength_candidate():
    trajectory = _trajectory(["cheap"])
    oracle = BaselineOracle(["cheap", "pricey"], CostModel(PRICING))
    decision = oracle.strongest(trajectory)
    # neither "cheap" nor "pricey" is in MODEL_STRENGTH, both default to 0.5 —
    # exercise the tie-break path without asserting a specific winner, just that it
    # doesn't crash and returns one of the two candidates.
    assert decision.model in {"cheap", "pricey"}


def test_static_baseline_uses_requested_model_regardless_of_logged_model():
    trajectory = _trajectory(["cheap", "cheap"])
    decision = _oracle().static(trajectory, "pricey")
    assert decision.model == "pricey"
    assert decision.route == ["pricey", "pricey"]
