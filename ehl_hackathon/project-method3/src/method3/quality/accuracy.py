from __future__ import annotations

import math

from method3.data.schema import Trajectory


def reward_model_accuracy(
    trajectories: list[Trajectory],
    reward_model,
    ground_truth: dict[str, float],
) -> dict[str, object]:
    """Score the reward model's predictions directly against known outcomes.

    This exists because a policy's reported `quality` is NOT a clean yardstick for
    comparing two versions of the pipeline. Quality on a counterfactual route is
    supplied by the reward model itself, so changing the reward model changes the
    reported quality of every policy — including baselines whose routing did not
    change at all. Observed while improving the opening features: `cheapest` always
    picks the same model, yet its reported quality moved 0.769 -> 0.753 purely
    because the reward model became less optimistic. Reading that as "the change
    made things worse" would have been exactly backwards — measured here, the same
    change improved held-out prediction error by 10%.

    Only FACTUAL trajectories are scored (the logged model ran, and its outcome is
    known), because that is the only place a prediction can be checked against
    reality. Mixed-model trajectories are excluded: there is no single action whose
    quality is being predicted.
    """
    errors: list[float] = []
    predictions: list[float] = []
    actuals: list[float] = []
    for trajectory in trajectories:
        if trajectory.logged_model == "mixed" or trajectory.key not in ground_truth:
            continue
        predicted = reward_model.predict(trajectory.opening, trajectory.logged_model).mean
        actual = ground_truth[trajectory.key]
        errors.append(abs(predicted - actual))
        predictions.append(predicted)
        actuals.append(actual)

    n = len(errors)
    if n == 0:
        return {"n": 0, "note": "no factual trajectories with ground truth"}

    mae = sum(errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mean_actual = sum(actuals) / n
    # Baseline: predicting the training-set mean for everything. A model that cannot
    # beat this has learned nothing context-specific, however good its MAE looks.
    baseline_mae = sum(abs(a - mean_actual) for a in actuals) / n
    total_variance = sum((a - mean_actual) ** 2 for a in actuals)
    residual = sum((p - a) ** 2 for p, a in zip(predictions, actuals))
    r_squared = 1.0 - (residual / total_variance) if total_variance > 0 else 0.0
    bias = sum(p - a for p, a in zip(predictions, actuals)) / n
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,                       # >0 = systematically optimistic
        "r_squared": r_squared,
        "baseline_mae_predict_mean": baseline_mae,
        "beats_mean_baseline": mae < baseline_mae,
    }
