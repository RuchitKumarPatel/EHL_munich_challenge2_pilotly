from __future__ import annotations

import collections


def action_space_coverage(
    group_of: dict[str, str],
    distributions: dict[str, dict[str, float]],
    candidate_models: list[str],
    observability_threshold: float = 1e-4,
) -> dict[str, object]:
    """Measure whether the logs a policy WOULD generate keep the action space
    observable — the property method4 exists to guarantee.

    This is the forward-looking counterpart to method3's `positivity_report`, which
    can only diagnose overlap failures already present in historical data. A policy
    is not just consuming logs; it is producing the logs its own successor will be
    evaluated on. A deterministic policy makes 67% of dataset2's (context group,
    model) cells permanently unobservable, so next year's router cannot be evaluated
    against this year's data at all. That is a self-inflicted, compounding failure
    that no estimator can repair after the fact.

    A cell (group, model) counts as observable if some context in that group assigns
    the model probability above `observability_threshold`. `reachable_fraction` is
    the headline: 1.0 means every action remains evaluable everywhere.
    """
    if not candidate_models:
        raise ValueError("no candidate models")
    per_group: dict[str, set[str]] = collections.defaultdict(set)
    min_probability = 1.0
    entropies: list[float] = []
    for key, distribution in distributions.items():
        group = group_of.get(key)
        if group is None:
            continue
        for model, probability in distribution.items():
            if probability > observability_threshold:
                per_group[group].add(model)
        if distribution:
            min_probability = min(min_probability, min(distribution.values()))
            entropies.append(_entropy(distribution))

    groups = sorted(set(group_of.values()))
    total_cells = len(groups) * len(candidate_models)
    reachable = sum(len(per_group.get(group, set())) for group in groups)
    unreachable = [
        {"group": group, "model": model}
        for group in groups for model in candidate_models
        if model not in per_group.get(group, set())
    ]
    return {
        "groups": len(groups),
        "candidate_models": len(candidate_models),
        "total_cells": total_cells,
        "reachable_cells": reachable,
        "reachable_fraction": (reachable / total_cells) if total_cells else 0.0,
        "unreachable_cells": unreachable[:50],
        "n_unreachable_cells": len(unreachable),
        "min_action_probability": min_probability if distributions else 0.0,
        "mean_policy_entropy": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "observability_threshold": observability_threshold,
    }


def _entropy(distribution: dict[str, float]) -> float:
    import math
    return -sum(p * math.log(p) for p in distribution.values() if p > 0)


def deterministic_coverage(
    group_of: dict[str, str],
    choices: dict[str, str],
    candidate_models: list[str],
) -> dict[str, object]:
    """Same measurement for a deterministic policy, so the two can be compared
    on identical terms. Each context contributes exactly one reachable action."""
    distributions = {key: {model: 1.0} for key, model in choices.items()}
    return action_space_coverage(group_of, distributions, candidate_models)
