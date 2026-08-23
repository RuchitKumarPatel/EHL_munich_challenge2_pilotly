from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method5.policy.uncertainty import UncertaintyGate, UncertaintySignal, signal_from_bounds


def test_high_uncertainty_and_near_tie_explores_more_than_certain_and_clear():
    gate = UncertaintyGate()
    uncertain = UncertaintySignal(interval_width=0.30, decision_margin=0.001)
    certain = UncertaintySignal(interval_width=0.01, decision_margin=0.40)
    assert gate.epsilon(uncertain) > gate.epsilon(certain)


def test_epsilon_always_within_floor_and_ceiling():
    gate = UncertaintyGate(floor=0.02, ceiling=0.30)
    extremes = [
        UncertaintySignal(0.0, 0.0), UncertaintySignal(0.0, 10.0),
        UncertaintySignal(100.0, 0.0), UncertaintySignal(100.0, 100.0),
        UncertaintySignal(-5.0, -5.0),   # negatives clamped, not crashing
    ]
    for signal in extremes:
        eps = gate.epsilon(signal)
        assert 0.02 <= eps <= 0.30


def test_floor_is_strictly_positive_preserving_positivity():
    # The floor is what keeps every action reachable, so the logs stay valid for
    # evaluating the policy's own successor. A zero floor recreates method3's
    # failure mode (72% of context-model cells permanently unobservable).
    gate = UncertaintyGate()
    perfectly_certain = UncertaintySignal(interval_width=0.0, decision_margin=1e9)
    assert gate.epsilon(perfectly_certain) >= gate.floor > 0.0


def test_epsilon_increases_monotonically_with_interval_width():
    gate = UncertaintyGate()
    values = [gate.epsilon(UncertaintySignal(w, 0.05)) for w in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)]
    assert values == sorted(values)


def test_epsilon_decreases_monotonically_with_decision_margin():
    gate = UncertaintyGate()
    values = [gate.epsilon(UncertaintySignal(0.2, m)) for m in (0.0, 0.05, 0.1, 0.3, 0.8)]
    assert values == sorted(values, reverse=True)


def test_zero_margin_does_not_divide_by_zero():
    import math
    gate = UncertaintyGate()
    assert math.isfinite(gate.epsilon(UncertaintySignal(0.5, 0.0)))


def test_zero_width_gives_the_floor_regardless_of_margin():
    gate = UncertaintyGate()
    for margin in (0.0, 0.5, 5.0):
        assert gate.epsilon(UncertaintySignal(0.0, margin)) == pytest.approx(gate.floor)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        UncertaintyGate(floor=0.0)                      # zero floor breaks positivity
    with pytest.raises(ValueError):
        UncertaintyGate(floor=0.4, ceiling=0.2)         # floor above ceiling
    with pytest.raises(ValueError):
        UncertaintyGate(ceiling=1.0)                    # must stay below 1
    with pytest.raises(ValueError):
        UncertaintyGate(softening=0.0)
    with pytest.raises(ValueError):
        UncertaintyGate(reference_ratio=0.0)


def test_round_trip_serialization():
    gate = UncertaintyGate(floor=0.03, ceiling=0.25, softening=0.07, reference_ratio=3.0)
    restored = UncertaintyGate.from_dict(gate.to_dict())
    signal = UncertaintySignal(0.2, 0.1)
    assert restored.epsilon(signal) == gate.epsilon(signal)


# ---------------- signal construction ----------------

def test_signal_uses_the_leading_model_interval_and_runner_up_gap():
    lower = {"a": 0.80, "b": 0.60, "c": 0.20}
    upper = {"a": 0.95, "b": 0.99, "c": 0.99}
    signal = signal_from_bounds(lower, upper)
    assert signal.interval_width == pytest.approx(0.15)   # leader "a": 0.95 - 0.80
    assert signal.decision_margin == pytest.approx(0.20)  # 0.80 - 0.60


def test_signal_respects_the_eligible_set():
    lower = {"a": 0.90, "b": 0.60, "c": 0.55}
    upper = {"a": 0.95, "b": 0.80, "c": 0.70}
    signal = signal_from_bounds(lower, upper, eligible={"b", "c"})
    assert signal.interval_width == pytest.approx(0.20)   # leader among eligible is "b"
    assert signal.decision_margin == pytest.approx(0.05)


def test_single_eligible_model_means_no_decision_uncertainty():
    signal = signal_from_bounds({"a": 0.8}, {"a": 0.9})
    assert signal.decision_margin == 1.0
    # and therefore exploration collapses toward the floor
    assert UncertaintyGate().epsilon(signal) < 0.10


def test_empty_eligible_set_falls_back_to_all_models():
    lower = {"a": 0.8, "b": 0.5}
    upper = {"a": 0.9, "b": 0.6}
    signal = signal_from_bounds(lower, upper, eligible=set())
    assert signal.decision_margin == pytest.approx(0.30)


def test_missing_upper_bound_yields_zero_width_not_a_crash():
    signal = signal_from_bounds({"a": 0.8, "b": 0.5}, {})
    assert signal.interval_width == 0.0


def test_signal_is_deterministic_under_ties():
    lower = {"a": 0.5, "b": 0.5}
    upper = {"a": 0.7, "b": 0.9}
    first = signal_from_bounds(lower, upper)
    second = signal_from_bounds(lower, upper)
    assert first == second
    assert first.decision_margin == pytest.approx(0.0)


# ---------------- self-calibration ----------------

def test_fit_sets_reference_to_the_median_raw_ratio():
    gate = UncertaintyGate()
    signals = [UncertaintySignal(w, 0.05) for w in (0.1, 0.2, 0.3, 0.4, 0.5)]
    gate.fit(signals)
    expected = sorted(gate.raw_ratio(s) for s in signals)[2]
    assert gate.reference_ratio == pytest.approx(expected)


def test_fitted_gate_puts_the_median_context_at_the_midpoint():
    gate = UncertaintyGate(floor=0.02, ceiling=0.30)
    signals = [UncertaintySignal(w, 0.05) for w in (0.1, 0.2, 0.3, 0.4, 0.5)]
    gate.fit(signals)
    median_signal = UncertaintySignal(0.3, 0.05)
    assert gate.epsilon(median_signal) == pytest.approx((0.02 + 0.30) / 2, rel=1e-9)


def test_fitting_prevents_the_saturation_that_flattened_the_gate():
    # Regression guard for a real defect: with an absolute scale far below the
    # observed widths, every context squashed to ~0.95 and epsilon varied only in
    # the third decimal, making the gate effectively flat. After fitting, the same
    # signals must spread across a usable fraction of the floor-ceiling range.
    gate = UncertaintyGate(floor=0.02, ceiling=0.30)
    signals = [UncertaintySignal(w, m) for w, m in
               [(0.33, 0.01), (0.38, 0.09), (0.39, 0.15), (0.48, 0.21), (0.52, 0.33)]]
    gate.fit(signals)
    epsilons = [gate.epsilon(s) for s in signals]
    assert (max(epsilons) - min(epsilons)) > 0.3 * (gate.ceiling - gate.floor)


def test_fitted_gate_correlates_positively_with_uncertainty():
    # The property that failed before self-calibration: exploration must RISE with
    # interval width, not stay flat or invert.
    gate = UncertaintyGate(floor=0.02, ceiling=0.30)
    signals = [UncertaintySignal(w, 0.10) for w in (0.05, 0.15, 0.25, 0.35, 0.45)]
    gate.fit(signals)
    epsilons = [gate.epsilon(s) for s in signals]
    assert epsilons == sorted(epsilons)
    assert epsilons[-1] > epsilons[0] * 1.5


def test_fit_on_empty_or_degenerate_signals_keeps_a_valid_reference():
    gate = UncertaintyGate()
    before = gate.reference_ratio
    gate.fit([])
    assert gate.reference_ratio == before > 0
    gate.fit([UncertaintySignal(0.0, 1.0)])          # all-zero ratios
    assert gate.reference_ratio > 0


def test_fit_is_robust_to_a_heavy_tailed_outlier():
    # Median, not mean: one near-zero margin produces an enormous ratio that would
    # drag a mean far above the bulk and re-introduce saturation.
    gate_median = UncertaintyGate().fit(
        [UncertaintySignal(0.2, 0.1)] * 10 + [UncertaintySignal(0.9, 0.0)]
    )
    plain = UncertaintyGate().raw_ratio(UncertaintySignal(0.2, 0.1))
    assert gate_median.reference_ratio == pytest.approx(plain)
