from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, Trajectory
from method3.quality.reward_model import KNNRewardModel


def _single_model_trajectory(key: str, model: str, task_text: str, n_calls: int = 3) -> Trajectory:
    history: list[dict] = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": task_text}]},
    ]
    tools = [{"type": "function", "name": "run_shell", "description": "d", "parameters": {}, "strict": False}]
    calls = []
    for i in range(n_calls):
        calls.append(Call(model, [dict(item) for item in history], tools, "test", i))
        call_id = f"{key}-{i}"
        history.append({"type": "function_call", "name": "run_shell", "arguments": f"{{\"cmd\": \"{i}\"}}", "call_id": call_id})
        history.append({"type": "function_call_output", "call_id": call_id, "output": "ok"})
    return Trajectory(key, calls)


def _mixed_model_trajectory(key: str, task_text: str) -> Trajectory:
    t = _single_model_trajectory(key, "model-a", task_text, n_calls=4)
    # flip the model on the later calls so trajectory.logged_model == "mixed"
    mixed_calls = [t.calls[0], t.calls[1]] + [Call("model-b", c.input, c.tools, c.source_file, c.source_line) for c in t.calls[2:]]
    return Trajectory(key, mixed_calls)


def _fit_model():
    trajectories = [_single_model_trajectory(f"a{i}", "model-a", f"task {i} alpha") for i in range(5)]
    trajectories += [_single_model_trajectory(f"b{i}", "model-b", f"task {i} beta") for i in range(5)]
    trajectories.append(_mixed_model_trajectory("mixed0", "task mixed"))
    labels = {t.key: 0.8 if t.logged_model == "model-a" else 0.4 for t in trajectories}
    model = KNNRewardModel(["model-a", "model-b"]).fit(trajectories, labels)
    return model, trajectories, labels


def test_mixed_model_trajectories_excluded_from_examples():
    model, _, _ = _fit_model()
    assert len(model.examples["model-a"]) == 5
    assert len(model.examples["model-b"]) == 5


def test_predict_reflects_which_model_examples_are_used():
    model, trajectories, _ = _fit_model()
    opening = trajectories[0].opening  # "task 0 alpha", model-a-flavored
    est_a = model.predict(opening, "model-a")
    est_b = model.predict(opening, "model-b")
    assert est_a.mean > est_b.mean  # model-a examples were labeled higher quality


def test_predict_unknown_model_returns_neutral_zero_support():
    model, _, _ = _fit_model()
    est = model.predict(_single_model_trajectory("x", "model-c", "whatever").opening, "model-c")
    assert est.support == 0.0
    assert est.mean == 0.5


def test_fit_rejects_empty_trajectory_list():
    with pytest.raises(ValueError):
        KNNRewardModel(["model-a"]).fit([], {})


def test_save_load_round_trip_is_bit_exact(tmp_path):
    model, trajectories, _ = _fit_model()
    path = tmp_path / "reward.json"
    model.save(path)
    restored = KNNRewardModel.load(path)
    for opening_source in trajectories[:3]:
        for candidate in ("model-a", "model-b"):
            before = model.predict(opening_source.opening, candidate)
            after = restored.predict(opening_source.opening, candidate)
            assert before == after


def test_save_creates_missing_parent_directory(tmp_path):
    model, _, _ = _fit_model()
    nested = tmp_path / "nested" / "dir" / "reward.json"
    model.save(nested)
    assert nested.exists()
    KNNRewardModel.load(nested)  # must not raise
