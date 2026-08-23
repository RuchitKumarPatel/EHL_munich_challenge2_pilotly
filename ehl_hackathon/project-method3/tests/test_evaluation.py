from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, Trajectory
from method3.evaluation.offline_policy_eval import evaluate_policy
from method3.pricing.cost_model import CostModel
from method3.quality.reward_model import Estimate

PRICING = {
    "cheap": {"input": 0.000001, "cached_input": 0.0000001},
    "pricey": {"input": 0.00002, "cached_input": 0.000002},
}


class FakeRewardModel:
    def predict(self, opening, model):
        return Estimate(mean=0.42, support=10, avg_distance=0.1)


def _trajectory(key: str, models: list[str]) -> Trajectory:
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
    return Trajectory(key, calls)


def test_factual_decision_uses_outcome_not_reward_model():
    trajectory = _trajectory("t1", ["cheap", "cheap"])
    outcomes = {"t1": 0.9}

    def decide(t):
        return t.logged_model, t.per_call_models  # exactly replays what happened

    metrics, rows = evaluate_policy("logged", [trajectory], decide, CostModel(PRICING), FakeRewardModel(), outcomes)
    assert rows[0]["factual"] is True
    assert rows[0]["quality"] == 0.9  # from outcomes, NOT the fake reward model's 0.42
    assert rows[0]["quality_source"] == "observed"


def test_counterfactual_decision_uses_reward_model():
    trajectory = _trajectory("t2", ["cheap", "cheap"])
    outcomes = {"t2": 0.9}

    def decide(t):
        return "pricey", ["pricey"] * t.n_calls  # different model than logged

    metrics, rows = evaluate_policy("static_pricey", [trajectory], decide, CostModel(PRICING), FakeRewardModel(), outcomes)
    assert rows[0]["factual"] is False
    assert rows[0]["quality"] == 0.42  # from the fake reward model
    assert rows[0]["quality_source"] == "counterfactual_reward_model"


def test_mixed_trajectory_logged_replay_is_factual():
    trajectory = _trajectory("t3", ["cheap", "pricey"])  # mixed
    outcomes = {"t3": 0.7}

    def decide(t):
        return t.logged_model, t.per_call_models

    metrics, rows = evaluate_policy("logged", [trajectory], decide, CostModel(PRICING), FakeRewardModel(), outcomes)
    assert rows[0]["factual"] is True
    assert rows[0]["model"] == "mixed"
    assert rows[0]["quality"] == 0.7
    assert metrics.switches == 1.0


def test_missing_outcome_for_factual_trajectory_raises_not_silently_wrong():
    trajectory = _trajectory("t4", ["cheap"])

    def decide(t):
        return t.logged_model, t.per_call_models

    with pytest.raises(KeyError):
        evaluate_policy("logged", [trajectory], decide, CostModel(PRICING), FakeRewardModel(), {})


def test_metrics_aggregate_across_multiple_trajectories():
    trajectories = [_trajectory("a", ["cheap"]), _trajectory("b", ["cheap", "cheap"])]
    outcomes = {"a": 0.8, "b": 0.6}

    def decide(t):
        return t.logged_model, t.per_call_models

    metrics, rows = evaluate_policy("logged", trajectories, decide, CostModel(PRICING), FakeRewardModel(), outcomes)
    assert metrics.trajectories == 2
    assert metrics.calls == 3
    assert metrics.factual_coverage == 1.0
    assert len(rows) == 2
