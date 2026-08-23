"""The latent ground-truth structure dataset2 is generated from.

This module is the single source of truth for *why* routing can beat both
"always cheapest" and "always strongest" on dataset2 — and it is the thing
dataset1 was missing entirely. In dataset1, `ground_truth_quality` was drawn
from difficulty alone and never depended on which model ran the task, so no
model was genuinely better at anything and no router could beat picking the
cheapest model. Verified empirically before writing this: dataset1's
quality-by-model spread is pure noise from difficulty mix.

Model complementarity is the entire premise of routing (see the routing
survey, arXiv:2603.04445, and "When Does Combining Language Models Help?",
arXiv:2606.27288): no single model is best on every input, and a well-matched
cheap specialist routinely beats a large generalist on a task the generalist
was never tuned for. SKILL encodes exactly that.

Read as: SKILL[model][task_type] = latent competence in [0, 1] on a
*medium*-difficulty instance of that task type. DIFFICULTY_SENSITIVITY says
how fast a model falls off as difficulty rises (weak/cheap models fall off
faster — this is what makes difficulty-aware routing pay).
"""
from __future__ import annotations

# Price tiers ($/1M input tokens), consistent with the pricing tables all three
# project-method* codebases already use.
MODEL_PRICE = {
    "claude-fable-5": 0.30,
    "gpt-5.6-sol": 0.30,
    "gpt-5.6-luna": 1.00,
    "gpt-5.6-terra": 2.00,
    "claude-sonnet-5": 3.00,
    "claude-opus-4-8": 15.00,
    "claude-opus-5": 15.00,
    # Introduced only partway through the log (see ERAS) — an "unseen at train
    # time" model, so routers can be tested on generalization to a model that
    # was not in the historical pool for most of the data.
    "gpt-5.7-nova": 4.00,
}

TASK_TYPES = (
    "qa_reasoning",
    "code_generation",
    "summarization_extraction",
    "multi_step_planning",
    "creative_writing",
    "data_analysis",          # deliberately rare + late-appearing: OOD-ish task type
)

DIFFICULTIES = ("easy", "medium", "hard")

# The complementarity structure. Deliberately NOT monotone in price:
#   * claude-fable-5 is a CHEAP SPECIALIST at summarization_extraction (0.88) —
#     it beats claude-opus-5 (0.83) there at 1/50th the price. Routing that task
#     to the cheap model is a genuine, learnable win.
#   * claude-sonnet-5 beats both opus models at code_generation (0.89 vs 0.86/0.87)
#     at 1/5th the price — the "well-matched smaller model" case from the literature.
#   * gpt-5.6-terra is the multi_step_planning specialist (0.87) over pricier models.
#   * claude-opus-5 is the best generalist and clearly best at hard qa_reasoning,
#     so "always cheapest" genuinely loses quality on reasoning-heavy work.
# An oracle router therefore beats BOTH "always cheapest" AND "always strongest",
# which is the property that makes this dataset able to discriminate routers at all.
SKILL = {
    "claude-fable-5":   {"qa_reasoning": 0.52, "code_generation": 0.44, "summarization_extraction": 0.88, "multi_step_planning": 0.46, "creative_writing": 0.58, "data_analysis": 0.50},
    "gpt-5.6-sol":      {"qa_reasoning": 0.55, "code_generation": 0.48, "summarization_extraction": 0.70, "multi_step_planning": 0.50, "creative_writing": 0.84, "data_analysis": 0.54},
    "gpt-5.6-luna":     {"qa_reasoning": 0.74, "code_generation": 0.63, "summarization_extraction": 0.75, "multi_step_planning": 0.66, "creative_writing": 0.72, "data_analysis": 0.68},
    "gpt-5.6-terra":    {"qa_reasoning": 0.76, "code_generation": 0.70, "summarization_extraction": 0.74, "multi_step_planning": 0.87, "creative_writing": 0.70, "data_analysis": 0.78},
    "claude-sonnet-5":  {"qa_reasoning": 0.80, "code_generation": 0.89, "summarization_extraction": 0.78, "multi_step_planning": 0.76, "creative_writing": 0.75, "data_analysis": 0.80},
    "claude-opus-4-8":  {"qa_reasoning": 0.87, "code_generation": 0.86, "summarization_extraction": 0.81, "multi_step_planning": 0.84, "creative_writing": 0.82, "data_analysis": 0.86},
    "claude-opus-5":    {"qa_reasoning": 0.92, "code_generation": 0.87, "summarization_extraction": 0.83, "multi_step_planning": 0.88, "creative_writing": 0.85, "data_analysis": 0.90},
    "gpt-5.7-nova":     {"qa_reasoning": 0.85, "code_generation": 0.82, "summarization_extraction": 0.80, "multi_step_planning": 0.83, "creative_writing": 0.80, "data_analysis": 0.88},
}

# How much a model degrades from easy -> hard. Cheap/weak models fall off a cliff;
# frontier models degrade gently. This is what makes *difficulty-aware* routing pay
# on top of *task-aware* routing.
DIFFICULTY_SENSITIVITY = {
    "claude-fable-5": 0.30, "gpt-5.6-sol": 0.28, "gpt-5.6-luna": 0.20, "gpt-5.6-terra": 0.17,
    "claude-sonnet-5": 0.14, "claude-opus-4-8": 0.09, "claude-opus-5": 0.07, "gpt-5.7-nova": 0.10,
}

# Asymmetric on purpose: the easy bonus is damped so easy instances do not all
# saturate at the ceiling (which would erase every quality difference between
# models on easy work and make that third of the dataset undiscriminating),
# while the hard penalty is full-strength so the weak/cheap models genuinely
# fall apart on hard work.
_DIFFICULTY_OFFSET = {"easy": -0.45, "medium": 0.0, "hard": 1.15}

# Per-model noise scale on a single observed outcome. Two runs of the SAME
# (model, task) do NOT give the same score — this is the "single-draw label noise"
# that "How Much of the Routing Gap Is Real?" (arXiv:2607.03436) shows inflates
# apparent router-to-oracle gaps. dataset2 emits explicit replicate groups so a
# methodology can separate reproducible specialist advantage from this noise.
OUTCOME_NOISE = 0.06


def expected_quality(model: str, task_type: str, difficulty: str) -> float:
    """Latent E[quality] for this (model, task_type, difficulty) — noise-free."""
    base = SKILL[model][task_type]
    penalty = DIFFICULTY_SENSITIVITY[model] * _DIFFICULTY_OFFSET[difficulty]
    return max(0.02, min(0.97, base - penalty))


def oracle_choice(task_type: str, difficulty: str, pool: list[str]) -> str:
    """Highest expected quality in `pool` (quality-only oracle, ignores price)."""
    return max(pool, key=lambda m: expected_quality(m, task_type, difficulty))


def cost_aware_oracle_choice(task_type: str, difficulty: str, pool: list[str], quality_floor: float) -> str | None:
    """Cheapest model clearing `quality_floor` in expectation — the decision a
    perfect cost-aware router would make. Returns None when nothing clears it."""
    eligible = [m for m in pool if expected_quality(m, task_type, difficulty) >= quality_floor]
    if not eligible:
        return None
    return min(eligible, key=lambda m: MODEL_PRICE[m])
