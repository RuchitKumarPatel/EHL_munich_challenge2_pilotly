from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method4.policy.base import ActionDistribution
from method4.policy.probability_matching import (
    kelly_scores, make_distribution, probability_match, softmax,
)


# ---------------- softmax ----------------

def test_softmax_sums_to_one_and_is_order_preserving():
    p = softmax({"a": 1.0, "b": 2.0, "c": 0.0}, temperature=1.0)
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert p["b"] > p["a"] > p["c"]


def test_softmax_is_numerically_stable_for_huge_scores():
    p = softmax({"a": 1e6, "b": 1e6 - 1.0}, temperature=1.0)
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert all(math.isfinite(v) for v in p.values())


def test_softmax_temperature_controls_sharpness():
    sharp = softmax({"a": 1.0, "b": 0.0}, temperature=0.05)
    flat = softmax({"a": 1.0, "b": 0.0}, temperature=10.0)
    assert sharp["a"] > flat["a"]


def test_softmax_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        softmax({"a": 1.0}, temperature=0.0)


# ---------------- probability matching ----------------

def test_every_action_keeps_positive_probability():
    # The core positivity guarantee method4 exists to provide.
    p = probability_match({"a": 10.0, "b": -10.0, "c": -50.0}, temperature=0.1, exploration_floor=0.1)
    assert all(v > 0 for v in p.values())
    assert abs(sum(p.values()) - 1.0) < 1e-12


def test_exploration_floor_lower_bounds_every_probability():
    floor = 0.12
    scores = {m: (100.0 if m == "a" else -100.0) for m in ("a", "b", "c", "d")}
    p = probability_match(scores, temperature=0.01, exploration_floor=floor)
    assert min(p.values()) >= floor / len(scores) - 1e-12


def test_zero_floor_can_starve_actions_showing_the_floor_matters():
    scores = {m: (100.0 if m == "a" else -100.0) for m in ("a", "b")}
    p = probability_match(scores, temperature=0.01, exploration_floor=0.0)
    assert p["b"] < 1e-9   # effectively unobservable without the floor


def test_ineligible_models_still_receive_floor_mass():
    p = probability_match({"a": 1.0, "b": 1.0}, temperature=0.1, exploration_floor=0.2, eligible={"a"})
    assert p["b"] > 0
    assert p["a"] > p["b"]


def test_empty_eligible_set_falls_back_to_all_models():
    p = probability_match({"a": 1.0, "b": 0.0}, temperature=0.1, exploration_floor=0.05, eligible=set())
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert p["a"] > p["b"]


def test_rejects_invalid_floor_and_empty_scores():
    with pytest.raises(ValueError):
        probability_match({"a": 1.0}, exploration_floor=1.0)
    with pytest.raises(ValueError):
        probability_match({}, exploration_floor=0.1)


def test_mass_concentrates_on_the_best_eligible_model():
    p = probability_match({"a": 5.0, "b": 0.0, "c": -5.0}, temperature=0.2, exploration_floor=0.05)
    assert p["a"] > 0.7


# ---------------- Kelly / geometric scoring ----------------

def test_kelly_prefers_cheaper_at_equal_quality():
    s = kelly_scores({"cheap": 0.8, "pricey": 0.8}, {"cheap": 1.0, "pricey": 10.0})
    assert s["cheap"] > s["pricey"]


def test_kelly_prefers_higher_quality_at_equal_price():
    s = kelly_scores({"good": 0.9, "bad": 0.4}, {"good": 1.0, "bad": 1.0})
    assert s["good"] > s["bad"]


def test_kelly_is_log_scaled_not_linear():
    # Doubling price should cost a constant amount of score regardless of level,
    # which is the defining property of the log objective.
    s = kelly_scores({"a": 0.8, "b": 0.8, "c": 0.8}, {"a": 1.0, "b": 2.0, "c": 4.0})
    assert abs((s["a"] - s["b"]) - (s["b"] - s["c"])) < 1e-9


def test_kelly_risk_aversion_scales_price_penalty():
    low = kelly_scores({"a": 0.8, "b": 0.8}, {"a": 1.0, "b": 10.0}, risk_aversion=0.5)
    high = kelly_scores({"a": 0.8, "b": 0.8}, {"a": 1.0, "b": 10.0}, risk_aversion=2.0)
    assert (high["a"] - high["b"]) > (low["a"] - low["b"])


def test_kelly_handles_zero_quality_and_zero_price_without_crashing():
    s = kelly_scores({"a": 0.0}, {"a": 0.0})
    assert math.isfinite(s["a"])


# ---------------- ActionDistribution ----------------

def test_distribution_normalizes_and_reports_support():
    d = ActionDistribution({"a": 2.0, "b": 2.0})
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-12
    assert d.support == {"a", "b"}


def test_distribution_rejects_degenerate_input():
    with pytest.raises(ValueError):
        ActionDistribution({})
    with pytest.raises(ValueError):
        ActionDistribution({"a": 0.0})
    with pytest.raises(ValueError):
        ActionDistribution({"a": -1.0, "b": 2.0})


def test_sampling_frequencies_match_the_distribution():
    d = ActionDistribution({"a": 0.7, "b": 0.3})
    rng = random.Random(0)
    draws = [d.sample(rng) for _ in range(20000)]
    assert abs(draws.count("a") / len(draws) - 0.7) < 0.02


def test_sampling_is_reproducible_for_a_seeded_rng():
    d = ActionDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
    first = [d.sample(random.Random(7)) for _ in range(1)]
    second = [d.sample(random.Random(7)) for _ in range(1)]
    assert first == second


def test_entropy_is_maximal_for_uniform_and_zero_for_degenerate():
    uniform = ActionDistribution({"a": 0.5, "b": 0.5})
    nearly_certain = ActionDistribution({"a": 1.0 - 1e-12, "b": 1e-12})
    assert uniform.entropy() == pytest.approx(math.log(2), rel=1e-9)
    assert nearly_certain.entropy() < 1e-9


def test_argmax_is_deterministic_under_ties():
    d = ActionDistribution({"a": 0.5, "b": 0.5})
    assert d.argmax() == d.argmax()


def test_make_distribution_reports_diagnostics():
    d = make_distribution({"a": 0.8, "b": 0.5}, {"a": 1.0, "b": 5.0}, eligible={"a", "b"})
    assert d.diagnostics["min_probability"] > 0
    assert d.diagnostics["entropy"] > 0
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-12
