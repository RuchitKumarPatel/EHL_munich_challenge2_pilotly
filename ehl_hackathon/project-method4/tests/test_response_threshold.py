from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method4.policy.response_threshold import ResponseThresholdTable


def test_lower_threshold_means_higher_engagement():
    table = ResponseThresholdTable()
    table.set_threshold("eager", "code", 0.2)
    table.set_threshold("reluctant", "code", 0.8)
    assert table.engagement_probability("eager", "code", 0.5) > table.engagement_probability("reluctant", "code", 0.5)


def test_engagement_rises_monotonically_with_stimulus():
    table = ResponseThresholdTable()
    table.set_threshold("m", "code", 0.5)
    values = [table.engagement_probability("m", "code", s) for s in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values)


def test_engagement_is_half_when_stimulus_equals_threshold():
    table = ResponseThresholdTable()
    table.set_threshold("m", "code", 0.4)
    assert table.engagement_probability("m", "code", 0.4) == pytest.approx(0.5)


def test_zero_stimulus_gives_zero_engagement():
    table = ResponseThresholdTable()
    assert table.engagement_probability("m", "code", 0.0) == 0.0


def test_negative_stimulus_is_clamped_not_crashing():
    table = ResponseThresholdTable()
    assert table.engagement_probability("m", "code", -5.0) == 0.0


def test_success_specializes_a_model_into_a_task():
    table = ResponseThresholdTable()
    before = table.threshold("m", "code")
    for _ in range(10):
        table.update("m", "code", performed=True, success=1.0)
    assert table.threshold("m", "code") < before


def test_failure_disengages_a_model_from_a_task():
    table = ResponseThresholdTable()
    before = table.threshold("m", "code")
    for _ in range(10):
        table.update("m", "code", performed=True, success=0.0)
    assert table.threshold("m", "code") > before


def test_not_performing_raises_the_threshold():
    table = ResponseThresholdTable()
    before = table.threshold("m", "code")
    table.update("m", "code", performed=False)
    assert table.threshold("m", "code") > before


def test_thresholds_are_clamped_and_never_become_unreachable():
    # The clamp is what keeps this compatible with method4's positivity guarantee:
    # no model can be driven to a threshold that makes it permanently unselectable.
    table = ResponseThresholdTable(min_threshold=0.05, max_threshold=0.95)
    for _ in range(500):
        table.update("m", "code", performed=True, success=1.0)
    assert table.threshold("m", "code") >= 0.05
    for _ in range(500):
        table.update("m", "code", performed=True, success=0.0)
    assert table.threshold("m", "code") <= 0.95


def test_specialization_emerges_from_differential_success():
    # Two models, two tasks, each good at one. Division of labor should appear
    # with no central assignment — the defining property of the biological model.
    table = ResponseThresholdTable()
    for _ in range(25):
        table.update("coder", "code", performed=True, success=0.95)
        table.update("coder", "prose", performed=True, success=0.15)
        table.update("writer", "code", performed=True, success=0.15)
        table.update("writer", "prose", performed=True, success=0.95)
    assert table.threshold("coder", "code") < table.threshold("writer", "code")
    assert table.threshold("writer", "prose") < table.threshold("coder", "prose")
    assert table.specialization_index(["code", "prose"], ["coder", "writer"]) > 0.1


def test_specialization_index_is_zero_without_differentiation():
    table = ResponseThresholdTable()
    assert table.specialization_index(["code"], ["a", "b"]) == 0.0


def test_specialization_index_degenerate_inputs():
    table = ResponseThresholdTable()
    assert table.specialization_index([], ["a", "b"]) == 0.0
    assert table.specialization_index(["code"], ["a"]) == 0.0


def test_seed_from_priors_maps_quality_to_threshold():
    table = ResponseThresholdTable().seed_from_priors({("good", "code"): 0.9, ("bad", "code"): 0.2})
    assert table.threshold("good", "code") < table.threshold("bad", "code")


def test_seed_from_priors_clamps_out_of_range_quality():
    table = ResponseThresholdTable().seed_from_priors({("x", "t"): 5.0, ("y", "t"): -3.0})
    assert table.min_threshold <= table.threshold("x", "t") <= table.max_threshold
    assert table.min_threshold <= table.threshold("y", "t") <= table.max_threshold


def test_round_trip_serialization():
    table = ResponseThresholdTable()
    table.update("m", "code", performed=True, success=0.9)
    restored = ResponseThresholdTable.from_dict(table.to_dict())
    assert restored.thresholds == table.thresholds
    assert restored.engagement_probability("m", "code", 0.5) == table.engagement_probability("m", "code", 0.5)


def test_unknown_pair_uses_the_default_threshold():
    table = ResponseThresholdTable(default_threshold=0.42)
    assert table.threshold("never-seen", "never-seen") == 0.42
