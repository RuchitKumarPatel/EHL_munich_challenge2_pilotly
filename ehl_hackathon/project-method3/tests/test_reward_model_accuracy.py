from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, Trajectory
from method3.quality.accuracy import reward_model_accuracy


class FakeRewardModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, opening, model):
        class E:
            mean = self.prediction
        return E()


def _trajectory(key: str, model: str) -> Trajectory:
    history = [
        {"role": "system", "content": "s"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": key}]},
    ]
    return Trajectory(key, [Call(model, history, [], "f", 0)])


def _mixed(key: str) -> Trajectory:
    history = [
        {"role": "system", "content": "s"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": key}]},
    ]
    return Trajectory(key, [Call("a", history, [], "f", 0),
                            Call("b", history + [{"type": "message", "role": "assistant", "content": []}], [], "f", 1)])


def test_perfect_predictions_give_zero_error():
    trajectories = [_trajectory("t1", "a"), _trajectory("t2", "a")]
    gt = {"t1": 0.5, "t2": 0.5}
    report = reward_model_accuracy(trajectories, FakeRewardModel(0.5), gt)
    assert report["mae"] == pytest.approx(0.0)
    assert report["rmse"] == pytest.approx(0.0)
    assert report["bias"] == pytest.approx(0.0)


def test_bias_sign_flags_systematic_optimism():
    trajectories = [_trajectory("t1", "a")]
    report = reward_model_accuracy(trajectories, FakeRewardModel(0.9), {"t1": 0.5})
    assert report["bias"] > 0        # predicting higher than reality
    report_low = reward_model_accuracy(trajectories, FakeRewardModel(0.1), {"t1": 0.5})
    assert report_low["bias"] < 0


def test_mixed_model_trajectories_are_excluded():
    trajectories = [_trajectory("t1", "a"), _mixed("m1")]
    report = reward_model_accuracy(trajectories, FakeRewardModel(0.5), {"t1": 0.5, "m1": 0.5})
    assert report["n"] == 1


def test_ungraded_trajectories_are_excluded():
    trajectories = [_trajectory("t1", "a"), _trajectory("t2", "a")]
    report = reward_model_accuracy(trajectories, FakeRewardModel(0.5), {"t1": 0.5})
    assert report["n"] == 1


def test_no_scorable_trajectories_is_reported_not_crashing():
    report = reward_model_accuracy([], FakeRewardModel(0.5), {})
    assert report["n"] == 0
    assert "note" in report


def test_constant_predictor_does_not_beat_the_mean_baseline():
    # A model that always predicts the same value cannot beat predicting the mean.
    trajectories = [_trajectory(f"t{i}", "a") for i in range(6)]
    gt = {f"t{i}": v for i, v in enumerate([0.2, 0.4, 0.6, 0.8, 0.5, 0.3])}
    report = reward_model_accuracy(trajectories, FakeRewardModel(0.9), gt)
    assert report["beats_mean_baseline"] is False
    assert report["r_squared"] < 0.0


def test_zero_variance_ground_truth_does_not_divide_by_zero():
    trajectories = [_trajectory(f"t{i}", "a") for i in range(3)]
    gt = {f"t{i}": 0.5 for i in range(3)}
    report = reward_model_accuracy(trajectories, FakeRewardModel(0.5), gt)
    assert report["r_squared"] == 0.0
