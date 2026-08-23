from __future__ import annotations

import collections
from collections import Counter, defaultdict

from method3.data.schema import Trajectory
from method3.propensity.logging_policy import LoggingPolicyModel


def support_report(trajectories: list[Trajectory], propensity_model: LoggingPolicyModel, candidate_models: list[str], clip: float = 0.05) -> dict[str, object]:
    """Overlap/support diagnostics for the importance-weighted estimators: per-model
    average fitted propensity plus a Kish effective-sample-size summary on the
    inverse-propensity weights (low relative to n means the correction rides on a
    handful of trajectories)."""
    non_mixed = [t for t in trajectories if t.logged_model != "mixed"]
    propensities: dict[str, list[float]] = {model: [] for model in candidate_models}
    for trajectory in non_mixed:
        opening = trajectory.opening
        for model in candidate_models:
            propensities[model].append(propensity_model.propensity(opening, model, clip=clip))
    average_propensity = {model: (sum(values) / len(values) if values else clip) for model, values in propensities.items()}
    logged_propensities = [propensity_model.propensity(t.opening, t.logged_model, clip=clip) for t in non_mixed]
    weights = [1.0 / p for p in logged_propensities]
    effective_sample_size = (sum(weights) ** 2 / sum(w * w for w in weights)) if weights else 0.0
    return {
        "n_non_mixed": len(non_mixed),
        "average_propensity": average_propensity,
        "effective_sample_size": effective_sample_size,
        "propensity_floor": clip,
    }


def positivity_report(
    trajectories: list[Trajectory],
    candidate_models: list[str],
    group_of: dict[str, str],
    target_model_of: dict[str, str],
    minimum_observations: int = 1,
) -> dict[str, object]:
    """Detect POSITIVITY (overlap) violations empirically from the logged data.

    Point identification of a policy value requires that every action the target
    policy takes has nonzero logging probability in that context. Where it does
    not, no amount of importance weighting recovers the answer — the data contain
    no information about that region, and any estimate there is pure reward-model
    extrapolation. An estimator that silently extrapolates looks confident and is
    wrong, so this reports the violation instead.

    `group_of` maps trajectory key -> a context group label (e.g. task type, or
    task type x difficulty). `target_model_of` maps trajectory key -> the model the
    target policy would choose. A (group, target model) pair that was NEVER logged
    is an unsupported cell; trajectories routed into one are flagged.
    """
    observed: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for trajectory in trajectories:
        if trajectory.logged_model == "mixed":
            continue
        group = group_of.get(trajectory.key)
        if group is None:
            continue
        observed[group][trajectory.logged_model] += 1

    # Enumerate cells over every group that APPEARS, not just groups that have at
    # least one usable observation. A group whose trajectories were all mixed-model
    # contributes no observations at all, so it would vanish from the cell listing
    # while still (correctly) flagging affected trajectories — an inconsistent
    # report that hides the most severe case, a context with no support whatsoever.
    all_groups = {group for key, group in group_of.items() if group is not None}
    all_groups.update(observed)

    unsupported_cells = []
    for group in sorted(all_groups):
        counts = observed.get(group, collections.Counter())
        for model in candidate_models:
            if counts.get(model, 0) < minimum_observations:
                unsupported_cells.append({"group": group, "model": model, "observations": counts.get(model, 0)})

    affected = []
    for trajectory in trajectories:
        group = group_of.get(trajectory.key)
        target = target_model_of.get(trajectory.key)
        if group is None or target is None:
            continue
        if observed.get(group, collections.Counter()).get(target, 0) < minimum_observations:
            affected.append({"trajectory": trajectory.key, "group": group, "target_model": target})

    n_considered = sum(1 for t in trajectories if group_of.get(t.key) is not None and target_model_of.get(t.key) is not None)
    return {
        "groups": len(observed),
        "candidate_models": len(candidate_models),
        "unsupported_cells": unsupported_cells,
        "n_unsupported_cells": len(unsupported_cells),
        "trajectories_routed_into_unsupported_cells": affected,
        "n_affected_trajectories": len(affected),
        "affected_fraction": (len(affected) / n_considered) if n_considered else 0.0,
        "minimum_observations": minimum_observations,
    }
