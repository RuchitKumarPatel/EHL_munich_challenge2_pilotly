from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method4.evaluation.coverage import action_space_coverage, deterministic_coverage
from method4.labeling.prioritized_replay import LabelCandidate, ReplayPriority


# ---------------- prioritized replay ----------------

def test_wide_interval_outranks_narrow_at_equal_margin():
    p = ReplayPriority()
    wide = LabelCandidate("wide", interval_width=0.5, decision_margin=0.1)
    narrow = LabelCandidate("narrow", interval_width=0.05, decision_margin=0.1)
    assert p.priority(wide) > p.priority(narrow)


def test_near_tie_outranks_blowout_at_equal_uncertainty():
    # Decision-relevance term: a label that could flip the choice is worth more
    # than an equally uncertain one that could not.
    p = ReplayPriority()
    tie = LabelCandidate("tie", interval_width=0.3, decision_margin=0.001)
    blowout = LabelCandidate("blowout", interval_width=0.3, decision_margin=0.9)
    assert p.priority(tie) > p.priority(blowout)


def test_already_labeled_has_zero_priority_and_is_never_selected():
    p = ReplayPriority()
    done = LabelCandidate("done", interval_width=0.9, decision_margin=0.0, already_labeled=True)
    assert p.priority(done) == 0.0
    assert p.select([done], budget=5) == []


def test_select_respects_budget_and_orders_by_priority():
    p = ReplayPriority()
    candidates = [
        LabelCandidate("low", 0.05, 0.5),
        LabelCandidate("high", 0.60, 0.01),
        LabelCandidate("mid", 0.30, 0.10),
    ]
    assert p.select(candidates, budget=2) == ["high", "mid"]


def test_zero_budget_selects_nothing_and_negative_raises():
    p = ReplayPriority()
    assert p.select([LabelCandidate("a", 0.5, 0.1)], budget=0) == []
    with pytest.raises(ValueError):
        p.select([], budget=-1)


def test_ranking_is_deterministic_under_ties():
    p = ReplayPriority()
    candidates = [LabelCandidate("b", 0.3, 0.1), LabelCandidate("a", 0.3, 0.1)]
    assert p.rank(candidates) == p.rank(candidates)
    assert [k for k, _ in p.rank(candidates)] == ["a", "b"]   # tie broken by key


def test_zero_margin_does_not_divide_by_zero():
    p = ReplayPriority()
    import math
    assert math.isfinite(p.priority(LabelCandidate("x", 0.5, 0.0)))


def test_prioritized_budget_beats_uniform_spending():
    p = ReplayPriority()
    candidates = [LabelCandidate(f"lo{i}", 0.01, 0.9) for i in range(50)]
    candidates += [LabelCandidate(f"hi{i}", 0.8, 0.001) for i in range(5)]
    report = p.expected_budget_saving(candidates, budget=5)
    assert report["ratio"] > 3.0     # same budget, several times the useful signal


def test_expected_budget_saving_degenerate_inputs():
    p = ReplayPriority()
    assert p.expected_budget_saving([], budget=5)["ratio"] == 1.0
    assert p.expected_budget_saving([LabelCandidate("a", 0.5, 0.1)], budget=0)["ratio"] == 1.0


# ---------------- coverage ----------------

def test_stochastic_policy_keeps_the_whole_action_space_reachable():
    models = ["a", "b", "c", "d"]
    group_of = {f"t{i}": f"g{i % 3}" for i in range(30)}
    # every action retains mass, as method4's exploration floor guarantees
    distributions = {k: {"a": 0.70, "b": 0.15, "c": 0.10, "d": 0.05} for k in group_of}
    report = action_space_coverage(group_of, distributions, models)
    assert report["reachable_fraction"] == 1.0
    assert report["n_unreachable_cells"] == 0
    assert report["min_action_probability"] > 0


def test_deterministic_policy_leaves_most_of_the_action_space_unreachable():
    models = ["a", "b", "c", "d"]
    group_of = {f"t{i}": f"g{i % 3}" for i in range(30)}
    choices = {k: "a" for k in group_of}          # always the same model
    report = deterministic_coverage(group_of, choices, models)
    assert report["reachable_fraction"] == pytest.approx(0.25)
    assert report["n_unreachable_cells"] == 9     # 3 groups x 3 unused models


def test_stochastic_strictly_dominates_deterministic_on_coverage():
    models = ["a", "b", "c"]
    group_of = {f"t{i}": "g0" for i in range(10)}
    det = deterministic_coverage(group_of, {k: "a" for k in group_of}, models)
    sto = action_space_coverage(group_of, {k: {"a": 0.8, "b": 0.15, "c": 0.05} for k in group_of}, models)
    assert sto["reachable_fraction"] > det["reachable_fraction"]


def test_probability_below_threshold_does_not_count_as_observable():
    models = ["a", "b"]
    group_of = {"t0": "g0"}
    report = action_space_coverage(group_of, {"t0": {"a": 0.9999999, "b": 1e-9}}, models, observability_threshold=1e-4)
    assert report["n_unreachable_cells"] == 1


def test_entropy_reported_and_zero_for_degenerate_policy():
    models = ["a", "b"]
    group_of = {"t0": "g0"}
    sharp = action_space_coverage(group_of, {"t0": {"a": 1.0}}, models)
    assert sharp["mean_policy_entropy"] == pytest.approx(0.0)


def test_coverage_rejects_empty_model_list():
    with pytest.raises(ValueError):
        action_space_coverage({"t": "g"}, {"t": {"a": 1.0}}, [])


def test_contexts_without_a_group_are_ignored_not_crashing():
    models = ["a", "b"]
    report = action_space_coverage({"t0": "g0"}, {"t0": {"a": 0.5, "b": 0.5}, "unknown": {"a": 1.0}}, models)
    assert report["reachable_fraction"] == 1.0
