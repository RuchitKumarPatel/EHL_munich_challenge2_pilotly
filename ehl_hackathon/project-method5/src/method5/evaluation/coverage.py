from __future__ import annotations

import collections
import math


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in distribution.values() if p > 0)


def action_space_coverage(
    group_of: dict[str, str],
    distributions: dict[str, dict[str, float]],
    candidate_models: list[str],
    observability_threshold: float = 1e-4,
) -> dict[str, object]:
    """Will the logs THIS policy produces keep the action space observable?

    A policy is not only consuming logs; it is producing the logs its own successor
    will be evaluated on. A deterministic policy makes most (context, model) cells
    permanently unobservable — measured on dataset2, project-method3 would leave 104
    of 144 cells never logged — so next year's router cannot be evaluated against
    this year's data at all. No estimator can repair that after the fact, which is
    why it is measured here as a first-class property rather than assumed.

    `reachable_fraction` is the headline: 1.0 means every action stays evaluable in
    every context group.
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
        "n_unreachable_cells": len(unreachable),
        "unreachable_cells": unreachable[:50],
        "min_action_probability": min_probability if distributions else 0.0,
        "mean_policy_entropy": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "observability_threshold": observability_threshold,
    }


def deterministic_coverage(
    group_of: dict[str, str],
    choices: dict[str, str],
    candidate_models: list[str],
) -> dict[str, object]:
    """The same measurement for a deterministic policy, so the two are comparable
    on identical terms: each context contributes exactly one reachable action."""
    return action_space_coverage(group_of, {k: {m: 1.0} for k, m in choices.items()}, candidate_models)


def _correlation(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    mean_a, mean_b = sum(a) / n, sum(b) / n
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / n
    var_a = sum((x - mean_a) ** 2 for x in a) / n
    var_b = sum((y - mean_b) ** 2 for y in b) / n
    denominator = math.sqrt(var_a * var_b)
    return (covariance / denominator) if denominator > 0 else 0.0


def exploration_efficiency(
    epsilons: list[float],
    interval_widths: list[float],
    decision_margins: list[float],
    softening: float = 0.05,
) -> dict[str, float]:
    """Is exploration going where it is worth spending?

    method5's claim over a flat rate is that the SAME average exploration budget is
    concentrated where the estimate is both unsure AND the decision is close. That
    is a ratio of two terms, so correlating epsilon against ONE of them is
    misleading — measured on dataset2, epsilon correlates -0.39 with interval width
    alone, which looks like a failure but is not: width and margin are entangled
    (+0.72) and margin has roughly four times the relative spread, so margin is the
    dominant term. Against the signal the gate actually uses, the correlation is
    +0.99, and against margin it is -0.89 (explore exactly where the decision is
    closest), which is the intended behavior.

    All three are therefore reported. `correlation_with_signal` is the one that
    says whether the gate is doing what it was built to do; the other two say which
    term is driving it on this particular dataset. A FLAT policy scores 0.0 on all
    three by construction, which is the comparison that matters.
    """
    n = len(epsilons)
    if n == 0 or n != len(interval_widths) or n != len(decision_margins):
        return {"n": 0, "mean_epsilon": 0.0, "correlation_with_signal": 0.0,
                "correlation_with_width": 0.0, "correlation_with_margin": 0.0}
    raw = [w / (softening + max(0.0, m)) for w, m in zip(interval_widths, decision_margins)]
    return {
        "n": n,
        "mean_epsilon": sum(epsilons) / n,
        "min_epsilon": min(epsilons),
        "max_epsilon": max(epsilons),
        "epsilon_range": max(epsilons) - min(epsilons),
        "correlation_with_signal": _correlation(epsilons, raw),
        "correlation_with_width": _correlation(epsilons, interval_widths),
        "correlation_with_margin": _correlation(epsilons, decision_margins),
    }
