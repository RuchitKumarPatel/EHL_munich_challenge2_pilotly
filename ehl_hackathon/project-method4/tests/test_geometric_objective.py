from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method4.objective.geometric import geometric_mean_by_segment, geometric_mean_quality


def test_geometric_equals_arithmetic_when_all_values_identical():
    assert geometric_mean_quality([0.8] * 10) == pytest.approx(0.8)


def test_geometric_is_below_arithmetic_when_values_vary():
    # The defining AM-GM property, and the whole reason bet-hedging prefers it.
    values = [0.4, 1.0]
    assert geometric_mean_quality(values) < sum(values) / len(values)


def test_geometric_separates_two_policies_with_identical_arithmetic_mean():
    # Both average 0.7, so an arithmetic objective is indifferent between them.
    # The geometric objective is not, and prefers the steady one — which is exactly
    # the discrimination bet-hedging buys and the reason method4 reports it.
    steady = [0.7, 0.7, 0.7, 0.7]
    swingy = [0.2, 1.0, 0.2, 1.4]
    assert sum(steady) / 4 == pytest.approx(sum(swingy) / 4)
    assert geometric_mean_quality(steady) > geometric_mean_quality(swingy)


def test_zero_quality_does_not_produce_negative_infinity():
    assert math.isfinite(geometric_mean_quality([0.0, 0.8]))


def test_empty_input_is_zero():
    assert geometric_mean_quality([]) == 0.0


def test_segment_report_contains_each_segment_and_both_aggregates():
    report = geometric_mean_by_segment({"era1": [0.8, 0.9], "era2": [0.4, 0.6]})
    assert report["segment::era1"] == pytest.approx(0.85)
    assert report["segment::era2"] == pytest.approx(0.5)
    assert "arithmetic_across_segments" in report
    assert "geometric_across_segments" in report


def test_variance_penalty_is_non_negative_and_zero_for_uniform_segments():
    uneven = geometric_mean_by_segment({"a": [0.95], "b": [0.15]})
    even = geometric_mean_by_segment({"a": [0.6], "b": [0.6]})
    assert uneven["variance_penalty"] > even["variance_penalty"]
    assert even["variance_penalty"] == pytest.approx(0.0, abs=1e-9)
    assert uneven["variance_penalty"] >= 0.0


def test_uneven_policy_scores_worse_across_segments_than_uniform_one():
    # A router that is excellent in one era and poor in another is fragile to
    # which era it actually lands in; the geometric criterion says so.
    uniform = geometric_mean_by_segment({"e1": [0.7], "e2": [0.7], "e3": [0.7]})
    lopsided = geometric_mean_by_segment({"e1": [1.0], "e2": [1.0], "e3": [0.1]})
    assert lopsided["arithmetic_across_segments"] > uniform["arithmetic_across_segments"]
    assert lopsided["geometric_across_segments"] < uniform["geometric_across_segments"]


def test_empty_segments_are_safe():
    assert geometric_mean_by_segment({})["geometric_across_segments"] == 0.0
    report = geometric_mean_by_segment({"a": []})
    assert math.isfinite(report["geometric_across_segments"])
