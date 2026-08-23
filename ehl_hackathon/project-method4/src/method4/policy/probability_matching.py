from __future__ import annotations

import math

from method4.policy.base import ActionDistribution


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    """Numerically stable softmax. Temperature -> 0 approaches argmax; larger
    temperature flattens toward uniform."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    top = max(scores.values())
    exponentials = {m: math.exp((s - top) / temperature) for m, s in scores.items()}
    total = sum(exponentials.values())
    return {m: v / total for m, v in exponentials.items()}


def probability_match(
    scores: dict[str, float],
    temperature: float = 0.15,
    exploration_floor: float = 0.08,
    eligible: set[str] | None = None,
) -> dict[str, float]:
    """Probability matching with a guaranteed exploration floor.

    Honeybees and starlings do not deterministically pick the best-known option;
    they match choice probability to their reinforcement history. That is
    ecologically optimal precisely when reward rates are unknown or non-stationary
    — which is this problem: the per-(model, task) skill structure is estimated
    from a small sample, and dataset2's environment genuinely drifts across eras.
    The same rule is Thompson sampling / probability matching in the bandit
    literature.

    Two layers, and the second is the load-bearing one:

      * `temperature` sets how sharply mass concentrates on high-scoring models
        among the ELIGIBLE set (those that passed the MVT/quality gate). Low
        temperature keeps almost all traffic on the best choice, so the quality
        cost of being stochastic stays small.
      * `exploration_floor` (epsilon) mixes in a uniform distribution over ALL
        candidate models — including ineligible ones. This is what guarantees
        strict positivity: every action keeps probability >= floor/|A|, so no
        (context, action) cell can become unobservable, and the logs this policy
        produces remain valid for evaluating its own successor. Without it, a
        deterministic policy makes 67% of dataset2's action space permanently
        unobservable.

    `eligible` restricts the sharp (softmax) component; the uniform floor always
    spans every candidate. Passing None treats every scored model as eligible.
    """
    if not scores:
        raise ValueError("no scored models")
    if not 0.0 <= exploration_floor < 1.0:
        raise ValueError("exploration_floor must be in [0, 1)")

    models = list(scores)
    eligible_models = [m for m in models if eligible is None or m in eligible]
    if not eligible_models:
        # Nothing cleared the gate: fall back to matching over everything rather
        # than returning an empty distribution.
        eligible_models = models

    sharp = softmax({m: scores[m] for m in eligible_models}, temperature)
    uniform_mass = 1.0 / len(models)
    return {
        m: (1.0 - exploration_floor) * sharp.get(m, 0.0) + exploration_floor * uniform_mass
        for m in models
    }


def kelly_scores(
    quality_lower: dict[str, float],
    price: dict[str, float],
    risk_aversion: float = 1.0,
) -> dict[str, float]:
    """Score models on LOG quality-per-dollar rather than linear.

    Bet-hedging theory (seed dormancy, Kelly betting): under environmental
    variance, what compounds is the GEOMETRIC mean, and maximizing it means
    maximizing E[log(outcome)], not E[outcome]. A linear score is indifferent
    between a certain 0.8 and a coin flip over {0.4, 1.2}; a log score is not, and
    correctly prefers the certain option. Since dataset2 drifts across eras and the
    router must survive all of them, geometric-mean optimality is the right
    objective, and it is also what makes the exploration floor above principled
    rather than merely defensive.

    `risk_aversion` scales how strongly price is penalized. The +1e-12 guards
    log(0) for a free or unpriced model.
    """
    scores = {}
    for model, lower in quality_lower.items():
        cost = max(price.get(model, 0.0), 1e-12)
        scores[model] = math.log(max(lower, 1e-6)) - risk_aversion * math.log(cost)
    return scores


def make_distribution(
    quality_lower: dict[str, float],
    price: dict[str, float],
    eligible: set[str] | None = None,
    temperature: float = 0.15,
    exploration_floor: float = 0.08,
    risk_aversion: float = 1.0,
    reason: str = "probability_matching",
) -> ActionDistribution:
    scores = kelly_scores(quality_lower, price, risk_aversion)
    probabilities = probability_match(scores, temperature, exploration_floor, eligible)
    distribution = ActionDistribution(probabilities, reason=reason)
    distribution.diagnostics = {
        "entropy": distribution.entropy(),
        "min_probability": min(probabilities.values()),
        "n_eligible": float(len(eligible) if eligible is not None else len(scores)),
    }
    return distribution
