from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method4.policy.marginal_value import MarginalValueRule


def test_fit_uses_aggregate_ratio_not_mean_of_ratios():
    # One near-free trajectory would make mean-of-ratios explode; total/total is stable.
    rule = MarginalValueRule.fit([1.0, 1.0], [1e-9, 1.0])
    assert rule.habitat_rate == pytest.approx(2.0 / (1.0 + 1e-9), rel=1e-6)


def test_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        MarginalValueRule.fit([], [])
    with pytest.raises(ValueError):
        MarginalValueRule.fit([1.0], [0.0])
    with pytest.raises(ValueError):
        MarginalValueRule.fit([1.0, 2.0], [1.0])


def test_escalates_when_marginal_rate_beats_habitat_rate():
    rule = MarginalValueRule(habitat_rate=1.0)
    # +0.5 quality for +0.1 cost -> marginal rate 5.0 > 1.0
    assert rule.should_escalate(0.4, 0.1, 0.9, 0.2)


def test_refuses_escalation_when_marginal_rate_is_below_habitat_rate():
    rule = MarginalValueRule(habitat_rate=10.0)
    # +0.5 quality for +0.1 cost -> rate 5.0 < 10.0
    assert not rule.should_escalate(0.4, 0.1, 0.9, 0.2)


def test_never_escalates_for_zero_or_negative_cost_gap():
    rule = MarginalValueRule(habitat_rate=0.0)
    assert not rule.should_escalate(0.1, 1.0, 0.9, 1.0)   # equal cost
    assert not rule.should_escalate(0.1, 1.0, 0.9, 0.5)   # cheaper candidate
    assert rule.marginal_rate(0.5, 0.0) is None


def test_price_awareness_a_cheaper_fleet_justifies_more_escalation():
    # Same quality ladder; when the habitat rate is low (dollars buy little on
    # average) escalation is easier to justify than when it is high.
    options = {"cheap": (0.60, 1.0), "pricey": (0.80, 2.0)}   # marginal rate 0.2
    assert MarginalValueRule(habitat_rate=0.1).best_escalation(options) == "pricey"
    assert MarginalValueRule(habitat_rate=0.5).best_escalation(options) == "cheap"


def test_best_escalation_walks_the_ladder_one_rung_at_a_time():
    # cheap->mid is justified (0.2/1.0 = 0.2 >= 0.15); mid->top is not (0.02/1.0 = 0.02).
    options = {"cheap": (0.50, 1.0), "mid": (0.70, 2.0), "top": (0.72, 3.0)}
    assert MarginalValueRule(habitat_rate=0.15).best_escalation(options) == "mid"


def test_ladder_walk_rejects_a_bundled_unjustified_jump():
    # A global argmax on rate-vs-cheapest would pick "top" ((0.75-0.50)/2 = 0.125),
    # but each individual rung must pay for itself: cheap->mid is 0.20 (ok at 0.15),
    # mid->top is 0.05 (not ok), so the ladder correctly stops at "mid".
    options = {"cheap": (0.50, 1.0), "mid": (0.70, 2.0), "top": (0.75, 3.0)}
    rule = MarginalValueRule(habitat_rate=0.15)
    assert rule.best_escalation(options) == "mid"


def test_single_option_is_returned():
    assert MarginalValueRule(habitat_rate=1.0).best_escalation({"only": (0.5, 1.0)}) == "only"


def test_empty_options_raise():
    with pytest.raises(ValueError):
        MarginalValueRule(habitat_rate=1.0).best_escalation({})


def test_cost_tie_breaks_toward_higher_quality():
    options = {"a": (0.40, 1.0), "b": (0.60, 1.0)}
    # equal cost -> no escalation is possible, so the ladder's first (best-quality
    # at that price) entry must win
    assert MarginalValueRule(habitat_rate=1.0).best_escalation(options) == "b"


def test_round_trip_serialization():
    rule = MarginalValueRule(habitat_rate=3.25)
    restored = MarginalValueRule.from_dict(rule.to_dict())
    assert restored.habitat_rate == rule.habitat_rate
    assert restored.minimum_cost_gap == rule.minimum_cost_gap
