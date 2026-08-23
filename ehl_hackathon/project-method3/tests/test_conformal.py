from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.quality.conformal import SplitConformalCalibrator


def test_perfect_predictions_give_zero_quantile():
    predicted = actual = [0.5] * 20
    cal = SplitConformalCalibrator(alpha=0.1).fit(list(predicted), list(actual))
    assert cal.quantile == 0.0
    assert cal.lower_bound(0.5) == 0.5


def test_constant_bias_recovered_exactly():
    actual = [0.3, 0.5, 0.7, 0.4, 0.6, 0.2, 0.8, 0.5, 0.6, 0.4]
    bias = 0.1
    predicted = [a + bias for a in actual]
    cal = SplitConformalCalibrator(alpha=0.1).fit(predicted, actual)
    assert cal.coverage_guaranteed
    # lower_bound(predicted_i) should recover actual_i (up to which order statistic
    # got selected) — check it never OVERSHOOTS actual on this exact calibration set.
    for p, a in zip(predicted, actual):
        assert cal.lower_bound(p) <= a + 1e-9


def test_insufficient_calibration_data_is_not_guaranteed():
    cal = SplitConformalCalibrator(alpha=0.1).fit([0.9], [0.5])
    assert not cal.coverage_guaranteed
    assert cal.quantile == float("inf")
    assert cal.lower_bound(0.9) == 0.0
    assert cal.lower_bound(0.9, floor=0.2) == 0.2


def test_empirical_marginal_coverage_close_to_target():
    rng = random.Random(1234)
    alpha = 0.1
    # predicted = actual + noise; noise is exchangeable between calibration and test
    # by construction (same generative process), which is exactly conformal's
    # required assumption.
    def sample(n):
        actual = [rng.uniform(0.2, 0.9) for _ in range(n)]
        predicted = [a + rng.gauss(0.0, 0.08) for a in actual]
        return predicted, actual

    cal_predicted, cal_actual = sample(3000)
    cal = SplitConformalCalibrator(alpha=alpha).fit(cal_predicted, cal_actual)
    assert cal.coverage_guaranteed

    test_predicted, test_actual = sample(3000)
    covered = sum(1 for p, a in zip(test_predicted, test_actual) if a >= cal.lower_bound(p) - 1e-12)
    coverage = covered / len(test_predicted)
    # Target 90% coverage; with 3000 exchangeable test points the empirical rate
    # should land close (generous band to keep this non-flaky).
    assert 0.85 <= coverage <= 0.96


def test_exact_threshold_uses_max_score():
    # n=9, alpha=0.1 -> level = ceil(10*0.9) = 9 == n -> guaranteed, quantile is the
    # single largest (worst-case) score in the calibration set.
    scores_actual = [0.1 * i for i in range(9)]
    predicted = [a + i for i, a in enumerate(scores_actual)]  # increasing overshoot
    cal = SplitConformalCalibrator(alpha=0.1).fit(predicted, scores_actual)
    assert cal.coverage_guaranteed
    expected_scores = sorted(p - a for p, a in zip(predicted, scores_actual))
    assert cal.quantile == expected_scores[-1]


def test_round_trip_serialization():
    cal = SplitConformalCalibrator(alpha=0.05).fit([0.1, 0.5, 0.9, 0.3, 0.7], [0.05, 0.4, 0.8, 0.2, 0.6])
    restored = SplitConformalCalibrator.from_dict(cal.to_dict())
    assert restored.alpha == cal.alpha
    assert restored.quantile == cal.quantile
    assert restored.n_calibration == cal.n_calibration
    assert restored.coverage_guaranteed == cal.coverage_guaranteed


def test_rejects_mismatched_lengths_and_empty_input():
    import pytest
    with pytest.raises(ValueError):
        SplitConformalCalibrator().fit([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        SplitConformalCalibrator().fit([], [])
