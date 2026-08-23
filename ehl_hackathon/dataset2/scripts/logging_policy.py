"""The historical logging policies that produced dataset2's trajectories.

dataset1 chose the logged model with `rng.choice(MODELS)` — uniform at random,
independent of context. That is MCAR, and it has two bad consequences verified
empirically before writing this:
  * a fitted propensity model has literally nothing to learn (every context maps
    to the same uniform distribution), so project-method3's LoggingPolicyModel
    cannot demonstrate any advantage over a context-free frequency propensity;
  * inverse-propensity / doubly-robust correction can never be shown to matter,
    because there is no selection bias to correct.

Real logs come from context-dependent policies that change over time. dataset2
therefore has three ERAS with genuinely different policies, which gives:
  * real, learnable propensity structure (selection bias to correct);
  * a temporal axis for distribution-shift / drift testing;
  * varying overlap quality — the near-deterministic era has weak overlap and the
    exploration era has strong overlap, so an estimator's sensitivity to
    positivity can actually be measured. Overlap failure is what makes
    importance-weighted estimators blow up (see "Logging Policy Design for
    Off-Policy Evaluation", arXiv:2605.15108, on the reward-coverage tradeoff,
    and arXiv:2603.21485 on deterministic-logging bias).

Plus one DELIBERATE, documented positivity violation (see FORBIDDEN_PAIRS) so a
methodology's overlap diagnostics have a known-planted failure to detect.
"""
from __future__ import annotations

import math
import random

from skill_matrix import MODEL_PRICE, expected_quality

# Eras are ordered; each covers a contiguous slice of the (synthetic) timeline.
# `share` values are relative weights, normalized by the generator.
ERAS = (
    {
        "name": "heuristic_2025h1",
        "share": 0.34,
        # Near-deterministic hand-written rule: hard work to the frontier model,
        # easy work to whatever is cheap. Concentrated -> WEAK OVERLAP on purpose.
        "temperature": 0.35,
        "epsilon": 0.03,
        "pool": ("claude-fable-5", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-5"),
        "cost_pressure": 0.0,
    },
    {
        "name": "explore_2025h2",
        "share": 0.40,
        # An explicit exploration period (an A/B / bandit rollout). High epsilon
        # -> STRONG OVERLAP, so this era alone can support clean OPE.
        "temperature": 1.20,
        "epsilon": 0.35,
        "pool": ("claude-fable-5", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-5"),
        "cost_pressure": 0.0,
    },
    {
        "name": "cost_push_2026h1",
        "share": 0.26,
        # A cost-cutting mandate skews selection toward cheap models, and a NEW
        # model (gpt-5.7-nova) enters the pool. Any router trained on earlier eras
        # has never seen nova -> tests generalization to an unseen model, and the
        # shifted mix tests robustness to logging-policy drift.
        "temperature": 0.70,
        "epsilon": 0.15,
        "pool": ("claude-fable-5", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-5", "gpt-5.7-nova"),
        "cost_pressure": 1.6,
    },
)

# Planted positivity/overlap violation: this (task_type, difficulty) cell NEVER
# logs this model, in any era. A target policy that wants to route here has zero
# support, and no amount of importance weighting can recover it — the data simply
# contain no information. A good methodology should DETECT and report this rather
# than silently extrapolating. Recorded in the manifest as `positivity_violation`.
FORBIDDEN_PAIRS = {
    ("summarization_extraction", "easy"): "claude-opus-5",
}


def _utility(model: str, task_type: str, difficulty: str, cost_pressure: float) -> float:
    """What the historical operators were (approximately) optimizing: quality,
    penalized by log-price when under cost pressure. Deliberately NOT the same
    objective a good router would use — the logging policy is a flawed heuristic,
    which is exactly what makes correcting for it worthwhile."""
    quality = expected_quality(model, task_type, difficulty)
    price_penalty = cost_pressure * 0.08 * math.log10(MODEL_PRICE[model] / 0.30 + 1.0)
    # The heuristic operators over-trusted difficulty as a proxy and under-used
    # task_type, so this leans on a difficulty/price interaction rather than the
    # true per-task skill structure.
    difficulty_pull = {"easy": -0.10, "medium": 0.0, "hard": 0.14}[difficulty] * math.log10(MODEL_PRICE[model] / 0.30 + 1.0)
    return quality + difficulty_pull - price_penalty


def allowed_models(era: dict, task_type: str, difficulty: str) -> list[str]:
    forbidden = FORBIDDEN_PAIRS.get((task_type, difficulty))
    return [m for m in era["pool"] if m != forbidden]


def action_probabilities(era: dict, task_type: str, difficulty: str) -> dict[str, float]:
    """The true propensity distribution P(model | context, era). The generator
    samples from exactly this and records it in the manifest, so a methodology's
    *estimated* propensities can be scored against the real ones — something no
    real-world dataset can offer and dataset1 could not either."""
    pool = allowed_models(era, task_type, difficulty)
    utilities = {m: _utility(m, task_type, difficulty, era["cost_pressure"]) for m in pool}
    top = max(utilities.values())
    exps = {m: math.exp((u - top) / era["temperature"]) for m, u in utilities.items()}
    total = sum(exps.values())
    greedy = {m: v / total for m, v in exps.items()}
    epsilon = era["epsilon"]
    uniform = 1.0 / len(pool)
    return {m: (1.0 - epsilon) * greedy[m] + epsilon * uniform for m in pool}


def sample_model(rng: random.Random, era: dict, task_type: str, difficulty: str) -> tuple[str, dict[str, float]]:
    probabilities = action_probabilities(era, task_type, difficulty)
    models = list(probabilities)
    weights = [probabilities[m] for m in models]
    chosen = rng.choices(models, weights=weights, k=1)[0]
    return chosen, probabilities
