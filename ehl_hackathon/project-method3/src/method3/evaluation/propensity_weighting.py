from __future__ import annotations

from method3.data.schema import Trajectory
from method3.propensity.logging_policy import LoggingPolicyModel
from method3.quality.reward_model import KNNRewardModel
from method3.routing.base import Router

from .estimators import Sample, all_estimators
from .overlap import positivity_report, support_report


def build_samples(
    trajectories: list[Trajectory],
    router: Router,
    candidate_models: list[str],
    reward_model: KNNRewardModel,
    propensity_model: LoggingPolicyModel,
    outcomes: dict[str, float],
) -> tuple[list[Sample], dict[str, str]]:
    """Reduce each test trajectory to the quantities every OPE estimator needs.

    Only meaningful for a true `Router` (opening-context-only decision).
    Mixed-model trajectories are skipped: contextual-bandit OPE needs a single
    well-defined logged action, and a trajectory that switched models mid-way has
    none.
    """
    samples: list[Sample] = []
    target_model_of: dict[str, str] = {}
    for trajectory in trajectories:
        if trajectory.logged_model == "mixed" or trajectory.key not in outcomes:
            continue
        opening = trajectory.opening
        decision = router.route(opening, candidate_models)
        target = decision.model
        target_model_of[trajectory.key] = target
        samples.append(Sample(
            q_target=reward_model.predict(opening, target).mean,
            q_logged=reward_model.predict(opening, trajectory.logged_model).mean,
            reward=outcomes[trajectory.key],
            propensity=propensity_model.propensity(opening, trajectory.logged_model, clip=1e-6),
            matched=(target == trajectory.logged_model),
        ))
    return samples, target_model_of


def evaluate_off_policy(
    trajectories: list[Trajectory],
    router: Router,
    candidate_models: list[str],
    reward_model: KNNRewardModel,
    propensity_model: LoggingPolicyModel,
    outcomes: dict[str, float],
    clip: float = 0.05,
    weight_threshold: float = 10.0,
    group_of: dict[str, str] | None = None,
) -> dict[str, object]:
    """Off-policy value of `router` under the whole estimator family (DM, IPS,
    SNIPS, DR, SWITCH-DR), each with a standard error and overlap diagnostics.

    Reporting a family rather than a single number is the point: these estimators
    have different bias/variance failure modes, so agreement is evidence the
    estimate is real and a large spread — especially DM vs IPS — is evidence of
    propensity misspecification or weak overlap, i.e. a signal to distrust all of
    them. A single DR number hides exactly that.
    """
    samples, target_model_of = build_samples(trajectories, router, candidate_models, reward_model, propensity_model, outcomes)
    results = all_estimators(samples, clip=clip, weight_threshold=weight_threshold)

    values = [r.value for r in results.values() if r.n_matched > 0 or r.name == "direct_method"]
    spread = (max(values) - min(values)) if values else 0.0

    payload: dict[str, object] = {
        "estimators": {name: result.to_dict() for name, result in results.items()},
        "estimator_spread": spread,
        "n_evaluable": len(samples),
        "n_matched": results["direct_method"].n_matched,
        "support": support_report(trajectories, propensity_model, candidate_models, clip=clip),
    }
    if group_of is not None:
        payload["positivity"] = positivity_report(trajectories, candidate_models, group_of, target_model_of)
    return payload


# Backwards-compatible thin wrapper: the previous single-estimator entry point.
def evaluate_doubly_robust(
    trajectories: list[Trajectory],
    router: Router,
    candidate_models: list[str],
    reward_model: KNNRewardModel,
    propensity_model: LoggingPolicyModel,
    outcomes: dict[str, float],
    clip: float = 0.05,
) -> dict[str, object]:
    payload = evaluate_off_policy(trajectories, router, candidate_models, reward_model, propensity_model, outcomes, clip=clip)
    estimators = payload["estimators"]  # type: ignore[index]
    return {
        "estimate": estimators["doubly_robust"]["value"],
        "direct_estimate": estimators["direct_method"]["value"],
        "n": payload["n_evaluable"],
        "propensity_floor": clip,
        "support": payload["support"],
    }
