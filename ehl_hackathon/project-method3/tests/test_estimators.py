from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.evaluation.estimators import (
    Sample, all_estimators, direct_method, doubly_robust, inverse_propensity,
    self_normalized_ips, switch_doubly_robust,
)


def _uniform_samples(rng: random.Random, n: int, true_value: float, n_actions: int = 4) -> list[Sample]:
    """Logging picks uniformly among n_actions; the target policy picks action 0.
    Reward for the target action is `true_value` in expectation, so every unbiased
    estimator should recover `true_value`."""
    propensity = 1.0 / n_actions
    samples = []
    for _ in range(n):
        matched = rng.random() < propensity
        reward = max(0.0, min(1.0, rng.gauss(true_value, 0.1))) if matched else max(0.0, min(1.0, rng.gauss(0.4, 0.1)))
        samples.append(Sample(q_target=true_value, q_logged=(true_value if matched else 0.4), reward=reward, propensity=propensity, matched=matched))
    return samples


def test_ips_recovers_true_value_under_correct_propensity():
    rng = random.Random(0)
    samples = _uniform_samples(rng, 4000, true_value=0.8)
    result = inverse_propensity(samples, clip=0.01)
    assert abs(result.value - 0.8) < 0.05


def test_dr_recovers_true_value_and_beats_ips_variance():
    rng = random.Random(1)
    samples = _uniform_samples(rng, 4000, true_value=0.8)
    ips = inverse_propensity(samples, clip=0.01)
    dr = doubly_robust(samples, clip=0.01)
    assert abs(dr.value - 0.8) < 0.05
    # DR uses the reward model to absorb most of the signal, leaving a small
    # residual to importance-weight -> materially lower variance than plain IPS.
    assert dr.standard_error < ips.standard_error


def test_dr_is_consistent_when_reward_model_is_wrong_but_propensity_right():
    rng = random.Random(2)
    samples = _uniform_samples(rng, 5000, true_value=0.8)
    broken = [Sample(q_target=0.1, q_logged=0.1, reward=s.reward, propensity=s.propensity, matched=s.matched) for s in samples]
    result = doubly_robust(broken, clip=0.01)
    # Reward model badly wrong (0.1 vs 0.8) but propensities correct -> DR still lands near truth.
    assert abs(result.value - 0.8) < 0.06


def test_dr_is_consistent_when_propensity_wrong_but_reward_model_right():
    rng = random.Random(3)
    samples = _uniform_samples(rng, 3000, true_value=0.8)
    # Propensity badly misspecified, but q_target is exactly right.
    skewed = [Sample(q_target=0.8, q_logged=s.q_logged, reward=s.reward, propensity=0.9, matched=s.matched) for s in samples]
    result = doubly_robust(skewed, clip=0.01)
    assert abs(result.value - 0.8) < 0.06


def test_snips_is_stable_when_weights_do_not_average_to_one():
    rng = random.Random(4)
    samples = _uniform_samples(rng, 2000, true_value=0.8)
    # Understate every propensity: IPS inflates, SNIPS renormalizes it away.
    distorted = [Sample(s.q_target, s.q_logged, s.reward, s.propensity * 0.5, s.matched) for s in samples]
    ips = inverse_propensity(distorted, clip=0.001)
    snips = self_normalized_ips(distorted, clip=0.001)
    assert abs(snips.value - 0.8) < abs(ips.value - 0.8)


def test_switch_dr_suppresses_extreme_weights():
    rng = random.Random(5)
    samples = _uniform_samples(rng, 500, true_value=0.8)
    # One pathological low-propensity matched sample.
    samples.append(Sample(q_target=0.8, q_logged=0.8, reward=1.0, propensity=1e-6, matched=True))
    dr = doubly_robust(samples, clip=1e-6)
    switch = switch_doubly_robust(samples, clip=1e-6, weight_threshold=10.0)
    assert switch.clipped_fraction > 0
    assert switch.standard_error < dr.standard_error


def test_switch_dr_equals_dr_when_nothing_exceeds_threshold():
    rng = random.Random(6)
    samples = _uniform_samples(rng, 500, true_value=0.7)
    dr = doubly_robust(samples, clip=0.25)          # weights capped at 4
    switch = switch_doubly_robust(samples, clip=0.25, weight_threshold=1000.0)
    assert abs(dr.value - switch.value) < 1e-12
    assert switch.clipped_fraction == 0.0


def test_no_matched_samples_is_reported_not_silently_zero():
    samples = [Sample(q_target=0.9, q_logged=0.3, reward=0.0, propensity=0.5, matched=False) for _ in range(50)]
    results = all_estimators(samples)
    assert results["snips"].n_matched == 0
    assert results["ips"].n_matched == 0
    # DM still works (it needs no matched samples at all) and reports the model's view.
    assert abs(results["direct_method"].value - 0.9) < 1e-9
    # DR degenerates to DM when nothing matches, which is the correct behavior.
    assert abs(results["doubly_robust"].value - 0.9) < 1e-9


def test_direct_method_has_no_importance_weight_variance():
    samples = [Sample(q_target=0.6, q_logged=0.6, reward=0.9, propensity=0.001, matched=True) for _ in range(20)]
    result = direct_method(samples)
    assert result.value == 0.6
    assert result.standard_error == 0.0
    assert result.max_weight == 0.0


def test_standard_error_shrinks_with_sample_size():
    rng = random.Random(7)
    small = doubly_robust(_uniform_samples(rng, 200, 0.8), clip=0.01)
    large = doubly_robust(_uniform_samples(rng, 4000, 0.8), clip=0.01)
    assert large.standard_error < small.standard_error


def test_confidence_interval_brackets_the_value():
    rng = random.Random(8)
    result = doubly_robust(_uniform_samples(rng, 1000, 0.8), clip=0.01)
    low, high = result.confidence_interval()
    assert low <= result.value <= high


def test_empty_sample_list_does_not_crash():
    results = all_estimators([])
    for result in results.values():
        assert result.n == 0
        assert result.value == 0.0
