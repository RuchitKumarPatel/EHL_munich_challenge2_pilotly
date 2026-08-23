# project-method4

A routing policy built from four mechanisms that biology already solved, aimed at
the one shortcoming method3 could not fix from inside its own design: **a
deterministic router destroys the data its successor needs.**

```
python -m pytest tests/ -v                                   # 89 tests
python scripts/reproduce_all.py ../dataset2/export ../project-method3/models/dataset2_20260823_023229
```

## The problem method4 exists to solve

method3 is a good router and it stays on the frontier. But it is deterministic, and
a deterministic policy only ever logs one action per context. Measured on dataset2:
if method3 were deployed, **only 33% of (context group, model) cells would ever
appear in its logs** — 67% of the action space becomes permanently unobservable, so
next year's router cannot be evaluated against this year's data. The symptom is
already visible: only **13 of 159** test trajectories matched method3's choice, so
its whole importance-weighted estimator family rode on 13 samples and spread 0.29.

That is a compounding, self-inflicted failure no estimator can repair after the
fact, and the fix has to be in the policy, not the evaluator.

## Results on dataset2 (159 held-out trajectories)

| policy | cost | quality | geometric | action-space reach | frontier |
|---|---|---|---|---|---|
| cheapest | $1.11e-04 | 0.7687 | 0.7634 | 0.12 | ✅ |
| **method4_foraging** | **$7.91e-04** | **0.8224** | **0.8208** | **1.00** | ✅ |
| method3_cascade | $9.88e-04 | 0.8271 | 0.8263 | 0.33 | ✅ |
| logged | $2.76e-03 | 0.8213 | 0.8052 | 0.62 | ✗ dominated |
| strongest | $5.53e-03 | 0.8505 | 0.8495 | 0.12 | ✅ |

method4 sits on the frontier at **20% lower cost than method3** for a 0.005 quality
difference (confidence intervals overlap: [0.814, 0.831] vs [0.822, 0.833]), and it
**dominates the logged policy outright** — 3.5x cheaper *and* higher quality.

The decisive differences are not on that table's first two columns:

| | method3 | method4 |
|---|---|---|
| action-space reach (successor evaluable?) | 0.33 | **1.00** |
| trajectories usable for off-policy evaluation | 13 / 159 | **145 / 159** |
| estimator family spread | 0.29 | **0.21** |
| labeling efficiency vs uniform spend | 1.0x | **3.5x** |

## The four mechanisms

**1. Probability matching — honeybees and starlings** (`policy/probability_matching.py`).
Bees don't pick the best-known flower; they match choice probability to reinforcement
history, which is ecologically optimal precisely when reward rates are unknown or
shifting. The router returns a *distribution*, not a choice. Two consequences: the
propensity of whatever action is taken is **known exactly** rather than estimated,
and an exploration floor keeps every action strictly positive so no cell can become
unobservable. This is what moves reach from 0.33 to 1.00 and usable OPE samples from
13 to 145.

**2. Marginal Value Theorem — Charnov 1976** (`policy/marginal_value.py`).
A forager leaves a patch when its marginal intake rate falls to the habitat average.
Here: escalate to a costlier model only while the marginal quality bought per
marginal dollar beats the fleet-wide average — a rate **measured from the logs**
(60697 quality/$ on dataset2), replacing method3's hand-set `quality_floor = 0.5`,
a magic number that on dataset1 silently collapsed the cascade to always-cheapest.
Because it is a rate, it reacts automatically to price changes and budget pressure.

**3. Response thresholds — ant and bee division of labor** (`policy/response_threshold.py`).
Colony workers have per-task thresholds; performing a task lowers yours for it, not
performing raises it, and specialists emerge with no central allocator. Models get
per-(model, context-bucket) thresholds under the canonical Bonabeau response
function `s^n/(s^n + θ^n)`. Thresholds are clamped so no model can become permanently
unreachable — keeping this compatible with the positivity guarantee.

**4. Bet-hedging — seed dormancy and the Kelly criterion** (`objective/geometric.py`).
Under environmental variance what compounds is the *geometric* mean, so a strategy
can rationally accept a lower arithmetic mean for lower variance. dataset2 drifts
across eras by construction, so geometric-mean quality is reported alongside the
arithmetic one; the gap between them is itself the fragility signal. This is also
what turns the exploration floor from a defensive hack into the principled optimum.

**Plus, from the hippocampus** (`labeling/prioritized_replay.py`): sharp-wave ripples
replay *surprising and rewarding* episodes preferentially. The same triage is applied
to the scarce labeling budget — prioritize trajectories where the conformal interval
is wide **and** the decision is near-tied (a label that could actually flip the
choice), which captures **3.5x** the decision-relevant uncertainty per label versus
spending uniformly.

## What was deliberately *not* built

Predictive-coding-style **per-step escalation** — only propagate the surprise upward —
is elegant and is genuinely how cortex saves energy. It is skipped because
["The Replay Gap"](https://arxiv.org/html/2608.08239) shows mid-trajectory model
switching cannot be honestly evaluated by replaying static logs, so it would make the
system *less* falsifiable. Parked until live rollout infrastructure exists.

## Structure

- `bridge.py` — the single, documented import point for method3. method4 changes the
  policy and the evaluation of stochastic policies; it reuses method3's data,
  feature, conformal-calibration and pricing layers unchanged rather than forking
  them (that fork is exactly what made method1 and method2 drift apart).
- `policy/` — `base.py` (ActionDistribution + StochasticPolicy),
  `probability_matching.py`, `marginal_value.py`, `response_threshold.py`,
  `foraging_router.py` (the composite policy).
- `objective/geometric.py` — bet-hedging objective.
- `labeling/prioritized_replay.py` — label-budget triage.
- `evaluation/stochastic_ope.py` — DM / IPS / SNIPS / DR / SWITCH-DR for a
  **stochastic** target policy; `coverage.py` — forward-looking action-space reach.
- `scripts/` — train, tune (calibration split only), evaluate, compare, reproduce_all.
- `tests/` — 89 tests.

## A bug worth recording

The first working version routed 89/159 trajectories to the single cheapest model
and lost 0.05 quality. Cause: **price was counted twice** — the MVT ladder already
fixes the justified price ceiling, and the Kelly score then re-optimized for price
*inside* that band. With costs spanning 50x and quality only ~1.8x, price dominated
the score ~6:1. The fix is conceptual, not cosmetic: once MVT says the patch is worth
exploiting, selection *within* it should be driven by quality. `risk_aversion`
dropped 1.0 → 0.15, chosen by `scripts/tune_method4.py` on the **calibration split,
never on test**, and quality recovered 0.773 → 0.822.
