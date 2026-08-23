from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, Trajectory
from method3.pricing.cost_model import CostModel

PRICING = {
    "cheap": {"input": 0.000001, "cached_input": 0.0000001},
    "pricey": {"input": 0.00001, "cached_input": 0.000001},
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
        history.append({"type": "function_call_output", "call_id": call_id, "output": "ok " * 50})
    return Trajectory("k", calls)


def test_no_switch_costs_less_than_a_route_with_a_switch():
    trajectory = _trajectory(["cheap"] * 4)
    model = CostModel(PRICING, switch_penalty_tokens=256)
    no_switch_cost, no_switch_diag = model.trajectory_cost(trajectory, ["cheap"] * 4)
    with_switch_cost, with_switch_diag = model.trajectory_cost(trajectory, ["cheap", "cheap", "pricey", "pricey"])
    assert no_switch_diag["switches"] == 0.0
    assert with_switch_diag["switches"] == 1.0
    assert with_switch_cost > no_switch_cost


def test_switch_penalty_is_actually_charged_in_dollars():
    # Regression test for the exact bug fixed this session: switch_penalty_tokens was
    # added to the reported uncached_tokens count but never multiplied into `total`.
    trajectory = _trajectory(["cheap", "pricey"])
    zero_penalty = CostModel(PRICING, switch_penalty_tokens=0)
    real_penalty = CostModel(PRICING, switch_penalty_tokens=1000)
    zero_cost, _ = zero_penalty.trajectory_cost(trajectory, ["cheap", "pricey"])
    real_cost, _ = real_penalty.trajectory_cost(trajectory, ["cheap", "pricey"])
    expected_extra = 1000 * PRICING["pricey"]["input"] / 1000
    assert abs((real_cost - zero_cost) - expected_extra) < 1e-12


def test_mixed_model_trajectory_replay_does_not_crash():
    # Regression test for the exact crash found this session (LoggedRouter /
    # FixedRouter on trajectory.logged_model == "mixed").
    trajectory = _trajectory(["cheap", "cheap", "pricey"])
    model = CostModel(PRICING)
    route = trajectory.per_call_models  # the real per-call sequence, not "mixed"
    cost, diagnostics = model.trajectory_cost(trajectory, route)
    assert cost > 0
    assert diagnostics["switches"] == 1.0


def test_route_length_mismatch_raises_clear_error():
    trajectory = _trajectory(["cheap"] * 3)
    model = CostModel(PRICING)
    with pytest.raises(ValueError):
        model.trajectory_cost(trajectory, ["cheap", "cheap"])


def test_unknown_model_raises_clear_error_not_bare_keyerror():
    trajectory = _trajectory(["cheap"])
    model = CostModel(PRICING)
    with pytest.raises(KeyError, match="no pricing entry"):
        model.trajectory_cost(trajectory, ["nonexistent-model"])


def test_cache_reset_on_switch_makes_next_call_fully_uncached():
    trajectory = _trajectory(["cheap", "cheap", "cheap"])
    model = CostModel(PRICING, switch_penalty_tokens=0)
    _, no_switch = model.trajectory_cost(trajectory, ["cheap", "cheap", "cheap"])
    _, with_switch = model.trajectory_cost(trajectory, ["cheap", "pricey", "cheap"])
    # switching at call 1 and again at call 2 should reduce total cached tokens
    # relative to never switching (each switch kills the running prefix credit)
    assert with_switch["cached_tokens"] < no_switch["cached_tokens"]
