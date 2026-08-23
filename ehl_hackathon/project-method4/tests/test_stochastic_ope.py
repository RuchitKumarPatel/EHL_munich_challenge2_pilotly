from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method4.evaluation.stochastic_ope import (
    StochasticSample, all_estimators, direct_method, doubly_robust,
    estimator_spread, inverse_propensity, self_normalized_ips, switch_doubly_robust,
)

MODELS = ["a", "b", "c", "d"]
TRUE_Q = {"a": 0.9, "b": 0.6, "c": 0.4, "d": 0.2}


def _samples(rng: random.Random, n: int, target_pi: dict[str, float], q_hat: dict[str, float] | None = None,
             logging_pi: dict[str, float] | None = None) -> list[StochasticSample]:
    logging_pi = logging_pi or {m: 1.0 / len(MODELS) for m in MODELS}
    q_hat = q_hat if q_hat is not None else dict(TRUE_Q)
    out = []
    for _ in range(n):
        logged = rng.choices(MODELS, weights=[logging_pi[m] for m in MODELS], k=1)[0]
        reward = max(0.0, min(1.0, rng.gauss(TRUE_Q[logged], 0.05)))
        out.append(StochasticSample(dict(q_hat), dict(target_pi), logged, logging_pi[logged], reward))
    return out


def _true_value(target_pi: dict[str, float]) -> float:
    return sum(target_pi[m] * TRUE_Q[m] for m in MODELS)


def test_all_estimators_recover_the_true_policy_value():
    rng = random.Random(0)
    target = {"a": 0.7, "b": 0.2, "c": 0.07, "d": 0.03}
    samples = _samples(rng, 6000, target)
    truth = _true_value(target)
    for name, result in all_estimators(samples).items():
        assert abs(result.value - truth) < 0.03, f"{name} off: {result.value} vs {truth}"


def test_dr_consistent_when_reward_model_is_badly_wrong():
    rng = random.Random(1)
    target = {"a": 0.6, "b": 0.3, "c": 0.05, "d": 0.05}
    samples = _samples(rng, 8000, target, q_hat={m: 0.05 for m in MODELS})
    assert abs(doubly_robust(samples).value - _true_value(target)) < 0.04


def test_dm_is_wrong_when_reward_model_is_wrong_showing_dr_adds_value():
    rng = random.Random(2)
    target = {"a": 0.6, "b": 0.3, "c": 0.05, "d": 0.05}
    samples = _samples(rng, 3000, target, q_hat={m: 0.05 for m in MODELS})
    truth = _true_value(target)
    assert abs(direct_method(samples).value - truth) > 0.3      # DM inherits the bias
    assert abs(doubly_robust(samples).value - truth) < 0.05     # DR corrects it


def test_stochastic_target_uses_every_sample_unlike_a_deterministic_one():
    # The central claim behind method4's design. Asserted on the actual mechanism —
    # how many logged trajectories contribute a non-zero importance weight at all —
    # rather than on an arbitrary ESS ratio. A deterministic target contributes only
    # on trajectories where it happens to agree with the log (~1/|A| of them); a
    # stochastic target contributes on every single one.
    rng = random.Random(3)
    n = 2000
    deterministic = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    stochastic = {"a": 0.7, "b": 0.15, "c": 0.1, "d": 0.05}
    det_samples = _samples(rng, n, deterministic)
    sto_samples = _samples(rng, n, stochastic)

    det_contributing = sum(1 for s in det_samples if s.target_pi.get(s.logged_model, 0.0) > 0)
    sto_contributing = sum(1 for s in sto_samples if s.target_pi.get(s.logged_model, 0.0) > 0)
    assert sto_contributing == n                 # every trajectory carries signal
    assert det_contributing < 0.4 * n            # only the agreements do
    # and the effective sample size is correspondingly higher
    assert doubly_robust(sto_samples).effective_sample_size > doubly_robust(det_samples).effective_sample_size


def test_deterministic_target_has_zero_weight_on_disagreements():
    target = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    samples = [StochasticSample(dict(TRUE_Q), dict(target), "b", 0.25, 0.6)]
    # logged "b" but target puts zero mass there -> no IPS contribution at all
    assert inverse_propensity(samples).value == 0.0


def test_snips_more_stable_than_ips_under_distorted_propensities():
    rng = random.Random(4)
    target = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    samples = _samples(rng, 3000, target)
    distorted = [StochasticSample(s.q_by_model, s.target_pi, s.logged_model, s.logged_propensity * 0.5, s.reward) for s in samples]
    truth = _true_value(target)
    assert abs(self_normalized_ips(distorted).value - truth) < abs(inverse_propensity(distorted).value - truth)


def test_switch_dr_suppresses_extreme_weights_and_lowers_variance():
    rng = random.Random(5)
    target = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    samples = _samples(rng, 800, target)
    samples.append(StochasticSample(dict(TRUE_Q), dict(target), "a", 1e-7, 1.0))
    dr = doubly_robust(samples)
    switch = switch_doubly_robust(samples, weight_threshold=10.0)
    assert switch.clipped_fraction > 0
    assert switch.standard_error < dr.standard_error


def test_switch_dr_matches_dr_when_no_weight_exceeds_threshold():
    rng = random.Random(6)
    target = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    samples = _samples(rng, 500, target)
    dr = doubly_robust(samples)
    switch = switch_doubly_robust(samples, weight_threshold=1e9)
    assert abs(dr.value - switch.value) < 1e-12
    assert switch.clipped_fraction == 0.0


def test_spread_is_small_when_estimators_agree():
    rng = random.Random(7)
    target = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    assert estimator_spread(all_estimators(_samples(rng, 5000, target))) < 0.05


def test_standard_error_shrinks_with_more_data():
    rng = random.Random(8)
    target = {"a": 0.6, "b": 0.2, "c": 0.15, "d": 0.05}
    small = doubly_robust(_samples(rng, 200, target))
    large = doubly_robust(_samples(rng, 5000, target))
    assert large.standard_error < small.standard_error


def test_uniform_target_recovers_the_average_quality():
    rng = random.Random(9)
    uniform = {m: 0.25 for m in MODELS}
    samples = _samples(rng, 6000, uniform)
    assert abs(doubly_robust(samples).value - _true_value(uniform)) < 0.03


def test_empty_input_is_safe():
    for result in all_estimators([]).values():
        assert result.n == 0
        assert result.value == 0.0


def test_confidence_interval_brackets_value():
    rng = random.Random(10)
    result = doubly_robust(_samples(rng, 1000, {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}))
    low, high = result.confidence_interval()
    assert low <= result.value <= high
