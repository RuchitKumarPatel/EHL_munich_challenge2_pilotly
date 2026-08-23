from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, Trajectory
from method3.propensity.logging_policy import LoggingPolicyModel


def _trajectory(key: str, model: str, text: str) -> Trajectory:
    history = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    ]
    call = Call(model, history, [], "test", 0)
    return Trajectory(key, [call])


def _mixed(key: str, text: str) -> Trajectory:
    history = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    ]
    calls = [Call("model-a", history, [], "test", 0), Call("model-b", history + [{"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "x"}]}], [], "test", 1)]
    return Trajectory(key, calls)


def _separable_dataset(rng: random.Random, n_per_class: int = 60):
    trajectories = []
    for i in range(n_per_class):
        pad = " filler" * rng.randint(0, 5)
        trajectories.append(_trajectory(f"a{i}", "model-a", "alpha task" + pad))
        trajectories.append(_trajectory(f"b{i}", "model-b", "beta task" + pad))
    rng.shuffle(trajectories)
    return trajectories


def test_probabilities_sum_to_one():
    rng = random.Random(0)
    data = _separable_dataset(rng)
    model = LoggingPolicyModel(["model-a", "model-b"]).fit(data, steps=300)
    probs = model.predict_proba(data[0].opening)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_recovers_separable_logging_pattern():
    rng = random.Random(1)
    data = _separable_dataset(rng, n_per_class=80)
    model = LoggingPolicyModel(["model-a", "model-b"]).fit(data, steps=600)
    alpha_probs = model.predict_proba(_trajectory("t", "model-a", "alpha task").opening)
    beta_probs = model.predict_proba(_trajectory("t", "model-b", "beta task").opening)
    assert alpha_probs["model-a"] > 0.8
    assert beta_probs["model-b"] > 0.8


def test_mixed_trajectories_excluded_from_fit():
    rng = random.Random(2)
    data = _separable_dataset(rng, n_per_class=30) + [_mixed("m0", "alpha mixed"), _mixed("m1", "beta mixed")]
    # must not raise, and must still fit on the 60 clean examples
    model = LoggingPolicyModel(["model-a", "model-b"]).fit(data, steps=200)
    assert model.weights["model-a"]


def test_fit_raises_when_only_mixed_trajectories_present():
    data = [_mixed("m0", "x"), _mixed("m1", "y")]
    with pytest.raises(ValueError):
        LoggingPolicyModel(["model-a", "model-b"]).fit(data)


def test_clip_floor_applied():
    rng = random.Random(3)
    data = _separable_dataset(rng, n_per_class=80)
    model = LoggingPolicyModel(["model-a", "model-b"]).fit(data, steps=600)
    # ask for propensity of the model that's near-certainly NOT chosen here
    p = model.propensity(_trajectory("t", "model-a", "alpha task").opening, "model-b", clip=0.1)
    assert p >= 0.1


def test_unknown_model_returns_clip():
    rng = random.Random(4)
    data = _separable_dataset(rng, n_per_class=20)
    model = LoggingPolicyModel(["model-a", "model-b"]).fit(data, steps=200)
    assert model.propensity(data[0].opening, "model-z", clip=0.07) == 0.07


def test_save_load_round_trip_is_bit_exact(tmp_path):
    rng = random.Random(5)
    data = _separable_dataset(rng, n_per_class=20)
    model = LoggingPolicyModel(["model-a", "model-b"]).fit(data, steps=200)
    path = tmp_path / "propensity.json"
    model.save(path)
    restored = LoggingPolicyModel.load(path)
    for t in data[:5]:
        assert model.predict_proba(t.opening) == restored.predict_proba(t.opening)


def test_fit_selected_picks_a_grid_value_and_reports_it():
    rng = random.Random(20)
    data = _separable_dataset(rng, n_per_class=40)
    train, validation = data[:60], data[60:]
    model, selection = LoggingPolicyModel.fit_selected(
        ["model-a", "model-b"], train, validation, l2_grid=(0.02, 0.5), steps=120
    )
    assert selection["selected_l2"] in (0.02, 0.5)
    assert len(selection["grid"]) == 2
    # the reported best loss really is the minimum over the grid
    assert selection["validation_log_loss"] == min(g["validation_log_loss"] for g in selection["grid"])
    assert abs(sum(model.predict_proba(train[0].opening).values()) - 1.0) < 1e-9


def test_fit_selected_falls_back_to_strongest_regularization_without_validation():
    rng = random.Random(21)
    data = _separable_dataset(rng, n_per_class=20)
    model, selection = LoggingPolicyModel.fit_selected(
        ["model-a", "model-b"], data, [], l2_grid=(0.02, 0.5, 3.0), steps=80
    )
    # Overfitting is the observed failure mode, so no-validation must not default
    # to the LEAST regularized option.
    assert selection["selected_l2"] == 3.0
    assert model.weights["model-a"]


def test_held_out_log_loss_is_finite_and_positive():
    rng = random.Random(22)
    data = _separable_dataset(rng, n_per_class=25)
    model = LoggingPolicyModel(["model-a", "model-b"]).fit(data[:30], steps=100)
    loss = model.held_out_log_loss(data[30:])
    assert 0.0 < loss < 50.0


def test_held_out_log_loss_ignores_unknown_models():
    rng = random.Random(23)
    data = _separable_dataset(rng, n_per_class=20)
    model = LoggingPolicyModel(["model-a"]).fit([t for t in data if t.logged_model == "model-a"], steps=80)
    # trajectories logged with model-b are skipped rather than crashing
    assert model.held_out_log_loss(data) < float("inf")
