from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method5.evaluation.coverage import action_space_coverage, deterministic_coverage, exploration_efficiency
from method5.evaluation.ope import (
    Sample, all_estimators, direct_method, doubly_robust, inverse_propensity,
    self_normalized_ips, spread, switch_doubly_robust,
)
from method5.labeling.priority import LabelCandidate, LabelTriage
from method5.policy.uncertainty import UncertaintyGate, UncertaintySignal

MODELS = ["a", "b", "c", "d"]
TRUE_Q = {"a": 0.9, "b": 0.6, "c": 0.4, "d": 0.2}


def _samples(rng, n, target_pi, q_hat=None, logging_pi=None):
    logging_pi = logging_pi or {m: 0.25 for m in MODELS}
    q_hat = dict(TRUE_Q) if q_hat is None else q_hat
    out = []
    for _ in range(n):
        logged = rng.choices(MODELS, weights=[logging_pi[m] for m in MODELS], k=1)[0]
        reward = max(0.0, min(1.0, rng.gauss(TRUE_Q[logged], 0.05)))
        out.append(Sample(dict(q_hat), dict(target_pi), logged, logging_pi[logged], reward))
    return out


def _truth(target_pi):
    return sum(target_pi[m] * TRUE_Q[m] for m in MODELS)


# ---------------- OPE ----------------

def test_every_estimator_recovers_the_true_policy_value():
    rng = random.Random(0)
    target = {"a": 0.7, "b": 0.2, "c": 0.07, "d": 0.03}
    samples = _samples(rng, 6000, target)
    for name, result in all_estimators(samples).items():
        assert abs(result.value - _truth(target)) < 0.03, name


def test_dr_survives_a_badly_wrong_reward_model():
    rng = random.Random(1)
    target = {"a": 0.6, "b": 0.3, "c": 0.05, "d": 0.05}
    samples = _samples(rng, 8000, target, q_hat={m: 0.05 for m in MODELS})
    assert abs(doubly_robust(samples).value - _truth(target)) < 0.04
    # while the direct method, which trusts it entirely, does not
    assert abs(direct_method(samples).value - _truth(target)) > 0.3


def test_dr_survives_a_badly_wrong_propensity():
    rng = random.Random(2)
    target = {"a": 0.6, "b": 0.3, "c": 0.05, "d": 0.05}
    samples = _samples(rng, 3000, target)
    skewed = [Sample(s.q_by_model, s.target_pi, s.logged_model, 0.9, s.reward) for s in samples]
    assert abs(doubly_robust(skewed).value - _truth(target)) < 0.06


def test_stochastic_target_uses_every_sample_a_deterministic_one_does_not():
    rng = random.Random(3)
    n = 2000
    deterministic = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    stochastic = {"a": 0.7, "b": 0.15, "c": 0.1, "d": 0.05}
    det = _samples(rng, n, deterministic)
    sto = _samples(rng, n, stochastic)
    assert sum(1 for s in sto if s.target_pi[s.logged_model] > 0) == n
    assert sum(1 for s in det if s.target_pi[s.logged_model] > 0) < 0.4 * n
    assert doubly_robust(sto).effective_sample_size > doubly_robust(det).effective_sample_size


def test_snips_beats_ips_under_distorted_propensities():
    rng = random.Random(4)
    target = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    samples = _samples(rng, 3000, target)
    distorted = [Sample(s.q_by_model, s.target_pi, s.logged_model, s.logged_propensity * 0.5, s.reward) for s in samples]
    truth = _truth(target)
    assert abs(self_normalized_ips(distorted).value - truth) < abs(inverse_propensity(distorted).value - truth)


def test_switch_dr_suppresses_extreme_weights():
    rng = random.Random(5)
    target = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    samples = _samples(rng, 800, target)
    samples.append(Sample(dict(TRUE_Q), dict(target), "a", 1e-9, 1.0))
    assert switch_doubly_robust(samples, weight_threshold=10.0).standard_error < doubly_robust(samples).standard_error


def test_switch_dr_equals_dr_below_threshold():
    rng = random.Random(6)
    target = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    samples = _samples(rng, 500, target)
    assert switch_doubly_robust(samples, weight_threshold=1e9).value == pytest.approx(doubly_robust(samples).value)


def test_spread_small_when_estimators_agree():
    rng = random.Random(7)
    assert spread(all_estimators(_samples(rng, 5000, {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}))) < 0.05


def test_empty_samples_are_safe():
    for result in all_estimators([]).values():
        assert result.n == 0 and result.value == 0.0


# ---------------- coverage ----------------

def test_stochastic_keeps_everything_reachable_deterministic_does_not():
    group_of = {f"t{i}": f"g{i % 3}" for i in range(30)}
    stochastic = action_space_coverage(group_of, {k: {"a": 0.7, "b": 0.15, "c": 0.1, "d": 0.05} for k in group_of}, MODELS)
    deterministic = deterministic_coverage(group_of, {k: "a" for k in group_of}, MODELS)
    assert stochastic["reachable_fraction"] == 1.0 and stochastic["n_unreachable_cells"] == 0
    assert deterministic["reachable_fraction"] == pytest.approx(0.25)
    assert deterministic["n_unreachable_cells"] == 9


def test_probability_below_threshold_is_not_observable():
    report = action_space_coverage({"t0": "g0"}, {"t0": {"a": 1 - 1e-9, "b": 1e-9}}, ["a", "b"])
    assert report["n_unreachable_cells"] == 1


def test_coverage_rejects_empty_model_list():
    with pytest.raises(ValueError):
        action_space_coverage({"t": "g"}, {"t": {"a": 1.0}}, [])


# ---------------- exploration efficiency ----------------

def test_targeted_exploration_tracks_the_signal_flat_does_not():
    widths = [0.01, 0.05, 0.10, 0.20, 0.40]
    margins = [0.30, 0.20, 0.10, 0.05, 0.01]
    raw = [w / (0.05 + m) for w, m in zip(widths, margins)]
    targeted = [0.02 + 0.28 * (r / (1.0 + r)) for r in raw]
    flat = [0.10] * 5
    # Epsilon is a MONOTONE but NONLINEAR (saturating) function of the signal, so
    # Pearson correlation is strong rather than 1.0 — monotonicity is the property
    # that actually matters, and is asserted directly alongside it.
    assert exploration_efficiency(targeted, widths, margins)["correlation_with_signal"] > 0.8
    assert [t for _, t in sorted(zip(raw, targeted))] == sorted(targeted)
    assert exploration_efficiency(flat, widths, margins)["correlation_with_signal"] == 0.0


def test_all_three_correlations_are_reported():
    # Correlating against width alone is misleading when width and margin are
    # entangled; the report must expose which term is driving the gate.
    widths = [0.30, 0.35, 0.40, 0.45]
    margins = [0.02, 0.10, 0.20, 0.30]
    epsilons = [0.20, 0.12, 0.07, 0.04]
    report = exploration_efficiency(epsilons, widths, margins)
    assert report["correlation_with_margin"] < -0.9      # explores where decision is close
    assert report["correlation_with_width"] < 0          # entanglement artifact, not a defect
    assert report["correlation_with_signal"] > 0.8       # tracks the real signal
    assert report["epsilon_range"] == pytest.approx(0.16)


def test_exploration_efficiency_degenerate_inputs():
    assert exploration_efficiency([], [], [])["n"] == 0
    assert exploration_efficiency([0.1], [0.1, 0.2], [0.1, 0.2])["n"] == 0
    assert exploration_efficiency([0.1], [0.1], [0.1, 0.2])["n"] == 0


# ---------------- labeling ----------------

def _triage():
    return LabelTriage(gate=UncertaintyGate())


def test_uncertain_near_tie_outranks_certain_blowout():
    triage = _triage()
    uncertain = LabelCandidate("u", UncertaintySignal(0.40, 0.001))
    certain = LabelCandidate("c", UncertaintySignal(0.02, 0.50))
    assert triage.priority(uncertain) > triage.priority(certain)


def test_already_labeled_scores_zero_and_is_never_selected():
    triage = _triage()
    done = LabelCandidate("done", UncertaintySignal(0.9, 0.0), already_labeled=True)
    assert triage.priority(done) == 0.0
    assert triage.select([done], budget=5) == []


def test_prioritized_budget_beats_uniform():
    triage = _triage()
    candidates = [LabelCandidate(f"lo{i}", UncertaintySignal(0.01, 0.9)) for i in range(50)]
    candidates += [LabelCandidate(f"hi{i}", UncertaintySignal(0.8, 0.001)) for i in range(5)]
    assert triage.budget_efficiency(candidates, budget=5)["ratio"] > 3.0


def test_ranking_is_deterministic_and_budget_respected():
    triage = _triage()
    candidates = [LabelCandidate("b", UncertaintySignal(0.3, 0.1)), LabelCandidate("a", UncertaintySignal(0.3, 0.1))]
    assert triage.rank(candidates) == triage.rank(candidates)
    assert [k for k, _ in triage.rank(candidates)] == ["a", "b"]
    assert len(triage.select(candidates, budget=1)) == 1


def test_negative_budget_raises_and_zero_selects_nothing():
    triage = _triage()
    with pytest.raises(ValueError):
        triage.select([], budget=-1)
    assert triage.select([LabelCandidate("a", UncertaintySignal(0.5, 0.1))], budget=0) == []


def test_labeling_and_exploration_share_one_signal():
    # The same raw ratio drives both, so a context worth exploring is exactly a
    # context worth labeling. Guards against the two drifting apart.
    gate = UncertaintyGate()
    triage = LabelTriage(gate=gate)
    signal = UncertaintySignal(0.3, 0.02)
    assert triage.priority(LabelCandidate("x", signal)) == gate.raw_ratio(signal)
