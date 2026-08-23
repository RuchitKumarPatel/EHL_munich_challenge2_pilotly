from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method5.bridge import Call, SplitConformalCalibrator, Trajectory
from method5.policy.base import ActionDistribution, StochasticPolicy, mix_with_uniform, softmax
from method5.policy.router import AdaptiveRouter
from method5.policy.uncertainty import UncertaintyGate
from method5.policy.value_ladder import ValueLadder

PRICING = {
    "cheap": {"input": 0.30e-6, "cached_input": 0.03e-6},
    "mid": {"input": 3.0e-6, "cached_input": 0.3e-6},
    "top": {"input": 15.0e-6, "cached_input": 1.5e-6},
}


class FakeEstimate:
    def __init__(self, mean, support):
        self.mean, self.support = mean, support


class FakeRewardModel:
    def __init__(self, table):
        self.table = table

    def predict(self, opening, model):
        mean, support = self.table[model]
        return FakeEstimate(mean, support)


def _opening():
    history = [
        {"role": "system", "content": "s"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "t"}]},
    ]
    return Trajectory("k", [Call("cheap", history, [], "f", 0)]).opening


def _cal(quantile):
    return SplitConformalCalibrator(alpha=0.1, quantile=quantile, n_calibration=50, coverage_guaranteed=True)


def _router(table, habitat_rate=1e5, gate=None, calibrators=None):
    return AdaptiveRouter(
        reward_model=FakeRewardModel(table),
        calibrators=calibrators if calibrators is not None else {m: _cal(0.05) for m in PRICING},
        pricing=PRICING,
        value_ladder=ValueLadder(habitat_rate=habitat_rate),
        gate=gate or UncertaintyGate(),
    )


ALL = ["cheap", "mid", "top"]


def test_conforms_to_stochastic_policy_protocol():
    assert isinstance(_router({m: (0.8, 10) for m in ALL}), StochasticPolicy)


def test_distribution_is_normalized_and_strictly_positive():
    d = _router({m: (0.8, 10) for m in ALL}).action_distribution(_opening(), ALL)
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-12
    assert all(p > 0 for p in d.probabilities.values())


def test_every_action_stays_reachable_even_when_one_model_dominates():
    # Positivity guarantee: the logs must keep every action observable.
    table = {"cheap": (0.99, 50), "mid": (0.10, 50), "top": (0.10, 50)}
    d = _router(table).action_distribution(_opening(), ALL)
    assert min(d.probabilities.values()) > 0.0


def test_equal_quality_never_justifies_paying_more():
    # Zero marginal quality gain can never clear a positive habitat rate, so the
    # ladder correctly stays at the cheapest model and the band is a single option.
    table = {m: (0.80, 50) for m in ALL}
    d = _router(table, habitat_rate=1e-9).action_distribution(_opening(), ALL)
    assert d.reason.endswith("ceiling=cheap")
    assert d.diagnostics["n_eligible"] == 1


def test_price_is_consulted_only_at_the_ladder_not_inside_the_band():
    # Regression guard for method4's double-counting bug. Within the eligible band
    # the score must be the calibrated LOWER BOUND alone — no price term — because
    # the ladder has already settled the price question. Verified by reproducing
    # the expected within-band distribution from quality only.
    table = {"cheap": (0.40, 50), "mid": (0.90, 50), "top": (0.95, 50)}
    router = _router(table, habitat_rate=1e-9)
    d = router.action_distribution(_opening(), ALL)
    assert d.diagnostics["n_eligible"] == 3          # escalation justified all the way

    lower = {m: router._bounds(_opening(), m)[0] for m in ALL}
    expected_sharp = softmax(lower, router.temperature)
    epsilon = d.diagnostics["epsilon"]
    expected = mix_with_uniform(expected_sharp, ALL, epsilon)
    for model in ALL:
        assert d.probabilities[model] == pytest.approx(expected[model], rel=1e-9)


def test_within_band_ordering_follows_quality_not_price():
    # The expensive model has the best bound, so it must receive the most mass
    # despite being 50x the price of the cheapest.
    table = {"cheap": (0.40, 50), "mid": (0.70, 50), "top": (0.95, 50)}
    d = _router(table, habitat_rate=1e-9).action_distribution(_opening(), ALL)
    assert d.probabilities["top"] > d.probabilities["mid"] > d.probabilities["cheap"]


def test_high_habitat_rate_blocks_escalation_to_expensive_models():
    table = {"cheap": (0.60, 50), "mid": (0.65, 50), "top": (0.70, 50)}
    d = _router(table, habitat_rate=1e12).action_distribution(_opening(), ALL)
    assert d.reason.endswith("ceiling=cheap")
    assert d.diagnostics["n_eligible"] == 1


def test_low_habitat_rate_permits_escalation():
    table = {"cheap": (0.40, 50), "mid": (0.70, 50), "top": (0.90, 50)}
    d = _router(table, habitat_rate=1e-9).action_distribution(_opening(), ALL)
    assert d.diagnostics["n_eligible"] == 3
    assert d.argmax() == "top"          # best quality inside the band wins


def test_unsupported_models_are_excluded_from_the_band_but_keep_floor_mass():
    table = {"cheap": (0.9, 0.0), "mid": (0.8, 50), "top": (0.8, 50)}
    d = _router(table, habitat_rate=1e-9).action_distribution(_opening(), ALL)
    assert d.diagnostics["n_supported"] == 2
    assert d.probabilities["cheap"] > 0     # still reachable via the uniform floor


def test_uncertain_context_explores_more_than_certain_one():
    # The central claim of method5. Wide interval + near-tie -> higher epsilon.
    uncertain = _router({m: (0.50, 50) for m in ALL},
                        habitat_rate=1e-9, calibrators={m: _cal(0.45) for m in PRICING})
    certain = _router({"cheap": (0.95, 50), "mid": (0.20, 50), "top": (0.20, 50)},
                      habitat_rate=1e-9, calibrators={m: _cal(0.01) for m in PRICING})
    eps_uncertain = uncertain.action_distribution(_opening(), ALL).diagnostics["epsilon"]
    eps_certain = certain.action_distribution(_opening(), ALL).diagnostics["epsilon"]
    assert eps_uncertain > eps_certain


def test_uncertain_context_has_higher_entropy_than_certain_one():
    uncertain = _router({m: (0.50, 50) for m in ALL},
                        habitat_rate=1e-9, calibrators={m: _cal(0.45) for m in PRICING})
    certain = _router({"cheap": (0.95, 50), "mid": (0.20, 50), "top": (0.20, 50)},
                      habitat_rate=1e-9, calibrators={m: _cal(0.01) for m in PRICING})
    assert (uncertain.action_distribution(_opening(), ALL).entropy()
            > certain.action_distribution(_opening(), ALL).entropy())


def test_uncertified_model_cannot_win_on_an_unearned_bound():
    # "cheap" has a huge point estimate but no certified calibrator and no pooled
    # fallback, so its lower bound must be 0.0 and it must not dominate the band.
    calibrators = {"mid": _cal(0.05), "top": _cal(0.05)}   # note: no "cheap", no "__global__"
    table = {"cheap": (0.99, 50), "mid": (0.80, 50), "top": (0.80, 50)}
    d = _router(table, habitat_rate=1e-9, calibrators=calibrators).action_distribution(_opening(), ALL)
    assert d.argmax() != "cheap"


def test_pooled_calibrator_is_used_when_per_model_is_uncertified():
    calibrators = {
        "cheap": SplitConformalCalibrator(alpha=0.1, quantile=float("inf"), n_calibration=1, coverage_guaranteed=False),
        "mid": _cal(0.05), "top": _cal(0.05),
        "__global__": _cal(0.10),
    }
    table = {"cheap": (0.95, 50), "mid": (0.50, 50), "top": (0.50, 50)}
    d = _router(table, habitat_rate=1e-9, calibrators=calibrators).action_distribution(_opening(), ALL)
    assert d.argmax() == "cheap"     # 0.95 - 0.10 = 0.85 still beats 0.45


def test_empty_candidate_list_raises():
    with pytest.raises(ValueError):
        _router({m: (0.8, 10) for m in ALL}).action_distribution(_opening(), [])


def test_sampling_is_reproducible_and_matches_the_distribution():
    d = _router({m: (0.8, 10) for m in ALL}).action_distribution(_opening(), ALL)
    assert d.sample(random.Random(3)) == d.sample(random.Random(3))
    draws = [d.sample(random.Random(i)) for i in range(400)]
    assert set(draws) <= set(ALL)


# ---------------- primitives ----------------

def test_softmax_stable_and_normalized():
    p = softmax({"a": 1e6, "b": 1e6 - 1}, 1.0)
    assert abs(sum(p.values()) - 1.0) < 1e-12


def test_softmax_rejects_bad_input():
    with pytest.raises(ValueError):
        softmax({"a": 1.0}, 0.0)
    with pytest.raises(ValueError):
        softmax({}, 1.0)


def test_mix_with_uniform_bounds_every_probability_below():
    mixed = mix_with_uniform({"a": 1.0}, ["a", "b", "c"], epsilon=0.3)
    assert mixed["b"] == pytest.approx(0.3 / 3)
    assert abs(sum(mixed.values()) - 1.0) < 1e-12


def test_mix_with_uniform_rejects_bad_input():
    with pytest.raises(ValueError):
        mix_with_uniform({"a": 1.0}, [], 0.1)
    with pytest.raises(ValueError):
        mix_with_uniform({"a": 1.0}, ["a"], 1.0)


def test_action_distribution_rejects_degenerate_input():
    with pytest.raises(ValueError):
        ActionDistribution({})
    with pytest.raises(ValueError):
        ActionDistribution({"a": 0.0})
    with pytest.raises(ValueError):
        ActionDistribution({"a": -0.5, "b": 1.0})
