from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data.schema import Call, Trajectory
from method3.evaluation.overlap import positivity_report
from method3.quality.cold_start import ColdStartPrior, infer_descriptor
from method3.quality.reward_model import KNNRewardModel


# ---------------- cold start ----------------

def _descriptors():
    return {
        "cheap-a": infer_descriptor("cheap-a", 0.30),
        "cheap-b": infer_descriptor("cheap-b", 0.30),
        "frontier-a": infer_descriptor("frontier-a", 15.0),
        "frontier-new": infer_descriptor("frontier-new", 15.0),
    }


def test_infer_descriptor_tiers():
    assert infer_descriptor("x", 0.3).tier == "cheap"
    assert infer_descriptor("x", 3.0).tier == "mid"
    assert infer_descriptor("x", 15.0).tier == "frontier"


def test_prior_uses_tier_evidence_for_a_model_with_no_history():
    prior = ColdStartPrior().fit(_descriptors(), {
        "cheap-a": [0.4, 0.45, 0.5],
        "cheap-b": [0.42, 0.44],
        "frontier-a": [0.9, 0.92, 0.88],
        "frontier-new": [],           # never seen
    })
    # frontier-new has no history but shares the "frontier" tier with frontier-a,
    # so its prior should come from that tier, not from a flat 0.5.
    assert prior.prior_for("frontier-new") > 0.8


def test_prior_falls_back_to_global_for_unknown_descriptor():
    prior = ColdStartPrior().fit(_descriptors(), {"cheap-a": [0.5, 0.5]})
    assert prior.prior_for("totally-unknown-model") == prior.global_quality


def test_blend_returns_prior_with_no_support_and_direct_with_high_support():
    prior = ColdStartPrior().fit(_descriptors(), {"frontier-a": [0.9, 0.9, 0.9]})
    assert prior.blend("frontier-new", direct_mean=0.2, direct_support=0.0) == prior.prior_for("frontier-new")
    high = prior.blend("frontier-new", direct_mean=0.2, direct_support=1000.0)
    assert abs(high - 0.2) < 0.01


def test_blend_is_monotone_in_support():
    prior = ColdStartPrior().fit(_descriptors(), {"frontier-a": [0.9, 0.9]})
    low = prior.blend("frontier-new", 0.2, 1.0)
    mid = prior.blend("frontier-new", 0.2, 10.0)
    high = prior.blend("frontier-new", 0.2, 100.0)
    assert low > mid > high  # more direct evidence -> closer to the direct 0.2


def test_cold_start_round_trips():
    prior = ColdStartPrior().fit(_descriptors(), {"cheap-a": [0.4], "frontier-a": [0.9]})
    restored = ColdStartPrior.from_dict(prior.to_dict())
    assert restored.prior_for("frontier-new") == prior.prior_for("frontier-new")
    assert restored.global_quality == prior.global_quality


def _trajectory(key: str, model: str, text: str) -> Trajectory:
    history = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    ]
    return Trajectory(key, [Call(model, history, [], "f", 0)])


def test_reward_model_uses_prior_for_unseen_model_and_flags_it():
    trajectories = [_trajectory(f"t{i}", "frontier-a", f"task {i}") for i in range(6)]
    labels = {t.key: 0.9 for t in trajectories}
    prior = ColdStartPrior().fit(_descriptors(), {"frontier-a": [0.9] * 6})
    model = KNNRewardModel(["frontier-a", "frontier-new"], cold_start=prior).fit(trajectories, labels)
    estimate = model.predict(trajectories[0].opening, "frontier-new")
    assert estimate.is_prior is True
    assert estimate.support == 0.0          # still weak evidence, so support gating holds
    assert estimate.mean > 0.8              # but no longer a useless flat 0.5


def test_reward_model_without_prior_stays_neutral_for_unseen_model():
    trajectories = [_trajectory(f"t{i}", "frontier-a", f"task {i}") for i in range(4)]
    labels = {t.key: 0.9 for t in trajectories}
    model = KNNRewardModel(["frontier-a", "frontier-new"]).fit(trajectories, labels)
    estimate = model.predict(trajectories[0].opening, "frontier-new")
    assert estimate.mean == 0.5
    assert estimate.is_prior is False


def test_reward_model_with_prior_round_trips(tmp_path):
    trajectories = [_trajectory(f"t{i}", "frontier-a", f"task {i}") for i in range(5)]
    labels = {t.key: 0.9 for t in trajectories}
    prior = ColdStartPrior().fit(_descriptors(), {"frontier-a": [0.9] * 5})
    model = KNNRewardModel(["frontier-a", "frontier-new"], cold_start=prior).fit(trajectories, labels)
    path = tmp_path / "rm.json"
    model.save(path)
    restored = KNNRewardModel.load(path)
    assert restored.cold_start is not None
    before = model.predict(trajectories[0].opening, "frontier-new")
    after = restored.predict(trajectories[0].opening, "frontier-new")
    assert before == after


# ---------------- positivity / overlap ----------------

def test_positivity_report_detects_a_never_logged_action():
    # "cheap-a" is never logged in group "g1", so routing there is unsupported.
    trajectories = [_trajectory(f"a{i}", "frontier-a", "x") for i in range(5)]
    group_of = {t.key: "g1" for t in trajectories}
    target_of = {t.key: "cheap-a" for t in trajectories}   # target routes to the unseen action
    report = positivity_report(trajectories, ["cheap-a", "frontier-a"], group_of, target_of)
    assert report["n_affected_trajectories"] == 5
    assert report["affected_fraction"] == 1.0
    assert any(c["model"] == "cheap-a" and c["group"] == "g1" for c in report["unsupported_cells"])


def test_positivity_report_clean_when_every_action_is_observed():
    trajectories = [_trajectory("a", "cheap-a", "x"), _trajectory("b", "frontier-a", "x2")]
    group_of = {t.key: "g1" for t in trajectories}
    target_of = {t.key: "cheap-a" for t in trajectories}
    report = positivity_report(trajectories, ["cheap-a", "frontier-a"], group_of, target_of)
    assert report["n_affected_trajectories"] == 0
    assert report["n_unsupported_cells"] == 0


def test_positivity_report_ignores_mixed_model_trajectories():
    history = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "t"}]},
    ]
    mixed = Trajectory("m", [Call("cheap-a", history, [], "f", 0), Call("frontier-a", history + [{"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "z"}]}], [], "f", 1)])
    group_of = {"m": "g1"}
    target_of = {"m": "cheap-a"}
    report = positivity_report([mixed], ["cheap-a", "frontier-a"], group_of, target_of)
    # the mixed trajectory contributes no observation, so both actions look unsupported
    assert report["n_unsupported_cells"] == 2
