from __future__ import annotations

import math

from method3.data.schema import Trajectory
from method3.propensity.logging_policy import LoggingPolicyModel


def score_propensity_model(
    trajectories: list[Trajectory],
    propensity_model: LoggingPolicyModel,
    true_propensities: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Score ESTIMATED propensities against the TRUE ones.

    Normally impossible — a real log never reveals the probabilities its logging
    policy actually used, so a propensity model's quality can only be argued for
    indirectly. dataset2 records them per trajectory, which turns "is the
    propensity model any good?" into a measurable question, and therefore turns
    the doubly-robust estimate's central assumption into a checkable one rather
    than an article of faith.

    Reports, over trajectories with a recorded truth:
      * mean absolute error and RMSE on the logged action's probability;
      * mean total-variation distance between the full estimated and true
        action distributions (the quantity that actually governs importance-weight
        error, not just the logged action's marginal);
      * log-loss of the estimated distribution on the logged action;
      * the same for a context-free frequency baseline, so the fitted model has to
        beat the naive thing dataset1 could not distinguish it from.
    """
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    total_variations: list[float] = []
    log_losses: list[float] = []
    baseline_absolute_errors: list[float] = []
    baseline_log_losses: list[float] = []

    # Context-free baseline: the empirical marginal frequency of each model.
    counts: dict[str, int] = {}
    considered: list[Trajectory] = []
    for trajectory in trajectories:
        if trajectory.logged_model == "mixed" or trajectory.key not in true_propensities:
            continue
        considered.append(trajectory)
        counts[trajectory.logged_model] = counts.get(trajectory.logged_model, 0) + 1
    total = sum(counts.values())
    baseline = {model: count / total for model, count in counts.items()} if total else {}

    for trajectory in considered:
        truth = true_propensities[trajectory.key]
        estimated = propensity_model.predict_proba(trajectory.opening)
        logged = trajectory.logged_model

        estimated_logged = estimated.get(logged, 0.0)
        true_logged = truth.get(logged, 0.0)
        absolute_errors.append(abs(estimated_logged - true_logged))
        squared_errors.append((estimated_logged - true_logged) ** 2)

        models = set(estimated) | set(truth)
        total_variations.append(0.5 * sum(abs(estimated.get(m, 0.0) - truth.get(m, 0.0)) for m in models))
        log_losses.append(-math.log(max(estimated_logged, 1e-12)))

        baseline_logged = baseline.get(logged, 1e-12)
        baseline_absolute_errors.append(abs(baseline_logged - true_logged))
        baseline_log_losses.append(-math.log(max(baseline_logged, 1e-12)))

    n = len(absolute_errors)
    if n == 0:
        return {"n": 0, "note": "no trajectories with recorded true propensities"}

    fitted_mae = sum(absolute_errors) / n
    baseline_mae = sum(baseline_absolute_errors) / n
    fitted_log_loss = sum(log_losses) / n
    baseline_log_loss = sum(baseline_log_losses) / n
    return {
        "n": n,
        "mae_logged_action": fitted_mae,
        "rmse_logged_action": math.sqrt(sum(squared_errors) / n),
        "mean_total_variation": sum(total_variations) / n,
        "log_loss": fitted_log_loss,
        "baseline_mae_logged_action": baseline_mae,
        "baseline_log_loss": baseline_log_loss,
        "mae_improvement_over_baseline": baseline_mae - fitted_mae,
        "log_loss_improvement_over_baseline": baseline_log_loss - fitted_log_loss,
        "beats_context_free_baseline": fitted_mae < baseline_mae,
    }
