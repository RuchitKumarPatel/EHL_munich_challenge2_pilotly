from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, Trajectory
from method3.data.split import split_trajectories


def _trajectory(key: str) -> Trajectory:
    history = [{"role": "system", "content": key}, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": key}]}]
    return Trajectory(key, [Call("m", history, [], "f", 0)])


def test_split_partitions_are_disjoint_and_cover_all():
    trajectories = [_trajectory(f"t{i}") for i in range(200)]
    parts = split_trajectories(trajectories)
    all_keys = {t.key for t in parts["train"]} | {t.key for t in parts["calibration"]} | {t.key for t in parts["test"]}
    assert len(all_keys) == 200
    train_keys = {t.key for t in parts["train"]}
    cal_keys = {t.key for t in parts["calibration"]}
    test_keys = {t.key for t in parts["test"]}
    assert train_keys.isdisjoint(cal_keys)
    assert train_keys.isdisjoint(test_keys)
    assert cal_keys.isdisjoint(test_keys)


def test_split_is_deterministic():
    trajectories = [_trajectory(f"t{i}") for i in range(50)]
    a = split_trajectories(trajectories)
    b = split_trajectories(trajectories)
    assert [t.key for t in a["train"]] == [t.key for t in b["train"]]


def test_split_rejects_invalid_fractions():
    trajectories = [_trajectory("t0")]
    with pytest.raises(ValueError):
        split_trajectories(trajectories, train_fraction=0.7, calibration_fraction=0.4)
    with pytest.raises(ValueError):
        split_trajectories(trajectories, train_fraction=0.0, calibration_fraction=0.2)
