from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, Trajectory


def test_call_from_mapping_rejects_missing_key():
    with pytest.raises(ValueError):
        Call.from_mapping({"model": "m", "input": []}, "f", 1)


def test_call_from_mapping_rejects_extra_key():
    with pytest.raises(ValueError):
        Call.from_mapping({"model": "m", "input": [], "tools": [], "extra": 1}, "f", 1)


def test_call_from_mapping_rejects_empty_model():
    with pytest.raises(ValueError):
        Call.from_mapping({"model": "", "input": [], "tools": []}, "f", 1)


def test_trajectory_rejects_empty_calls():
    with pytest.raises(ValueError):
        Trajectory("k", [])


def test_logged_model_single_vs_mixed():
    history = [{"role": "system", "content": "s"}, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "t"}]}]
    same = Trajectory("k", [Call("a", history, [], "f", 0), Call("a", history, [], "f", 1)])
    assert same.logged_model == "a"
    mixed = Trajectory("k", [Call("a", history, [], "f", 0), Call("b", history, [], "f", 1)])
    assert mixed.logged_model == "mixed"
