# project-method5

The consolidation. One new idea, two knobs removed, everything else inherited from
what measurably worked.

```
python -m pytest tests/ -v      # 67 tests
python scripts/reproduce_all.py ../dataset2/export ../project-method3/models/dataset2_20260823_045709
```

## Results on dataset2 (159 held-out)

| policy | cost | quality | action-space reach | frontier |
|---|---|---|---|---|
| cheapest | $1.11e-04 | 0.7528 | 0.12 | ✅ |
| method3_cascade | $9.69e-04 | 0.8108 | 0.28 | ✅ |
| **method5_adaptive** | **$1.06e-03** | **0.8236** | **1.00** | ✅ |
| logged | $2.76e-03 | 0.8213 | 0.62 | ✗ dominated |
| strongest | $5.53e-03 | 0.8484 | 0.12 | ✅ |

method5 **dominates the logged policy** (higher quality at 2.6x lower cost) and beats
method3 by **+0.013 quality** while being the only policy with **full action-space
reach**. It does not beat `strongest` on raw quality — it reaches 97% of it for 19%
of the cost.

## The one new idea: uncertainty-gated exploration

method4 explored at a **flat rate everywhere**, which is wasteful in both directions:
it spends budget in contexts where the conformal bound is already tight and the
winner obvious (pure quality tax, zero information), and under-explores contexts
where the estimate is genuinely uncertain. method5 makes the exploration rate a
function of uncertainty already being computed:

```
raw    = interval_width / (softening + decision_margin)
eps(x) = floor + (ceiling - floor) * raw / (reference + raw)
```

Grounded in [information-directed sampling](https://arxiv.org/pdf/1403.5556)
(Russo & Van Roy — spend exploration where information gained per unit regret is
favorable) and [uncertainty-driven adaptive exploration](https://arxiv.org/html/2509.03219v2)
(ADEU: concentrating exploration in high-uncertainty states reached good policies in
4,515 episodes vs UCB's 16,800).

**Measured**: `corr(eps, signal) = +0.985`, `corr(eps, decision_margin) = -0.886` —
exploration goes precisely where the decision is closest. A flat policy scores 0.0 on
both by construction.

**The unification**: the same signal drives label triage (`labeling/priority.py`).
One estimated quantity answers both "how much should I explore here?" and "is this
worth paying to label?", because they are the same question — where is the estimate
both unsure and consequential. Measured labeling efficiency: **2.36x** vs uniform.

## Lineage — what came from where

| Source | Inherited |
|---|---|
| method1 | Doubly-robust OPE structure |
| method2 | Never-degrade-below-logged stance |
| method3 | **Type-enforced pre-decision boundary**, split-conformal calibration, real switch-cost pricing, reward-model-accuracy metric, tool-identity features (+10.4% MAE) |
| method4 | Stochastic policy with **exact** propensities, MVT habitat rate, coverage metric, labeling triage |

**Removed, on evidence:** response thresholds (measured specialization only 0.07 — not
paying for its complexity); the flat `exploration_floor` (now derived); and
`risk_aversion` (MVT already sets the price ceiling — it was double-counting, the bug
that cost method4 0.05 quality). Net: **fewer knobs than method4**.

## Two bugs found and fixed during development

**1. The gate was flat.** With a fixed absolute `width_scale=0.10` against observed
interval widths of 0.33–0.52, raw ratios spanned 12–60 and the squash saturated at
0.92–0.98. Epsilon varied only between 0.094 and 0.099 — the gate was doing *nothing*,
exactly the behavior it exists to replace, and the exploration/uncertainty correlation
came out negative. Fixed by **self-calibrating** `reference_ratio` to the median raw
ratio on the calibration split, which both restores discrimination (epsilon now spans
0.038–0.068) and removes a dataset-specific magic number. The median, not the mean —
raw ratios are heavy-tailed and one near-zero margin would drag a mean back into
saturation.

**2. The efficiency metric was misleading.** It correlated epsilon against interval
*width* alone while the gate uses a *ratio* of width to margin. That reported −0.39
and looked like failure. It is not: width and margin are entangled (+0.72) and margin
has ~4x the relative spread, so margin is the dominant term. All three correlations
are now reported so the reader sees which term drives the gate.

## Structure

- `bridge.py` — single documented reuse point for method3's data, feature, conformal
  and pricing layers (120 tests behind them). Not forked: that fork is what made
  method1 and method2 drift apart.
- `policy/` — `uncertainty.py` (the gate), `value_ladder.py` (MVT ceiling),
  `router.py` (composite), `base.py` (ActionDistribution + protocol).
- `evaluation/` — `ope.py` (stochastic-target DM/IPS/SNIPS/DR/SWITCH-DR),
  `coverage.py` (forward-looking reach + exploration targeting).
- `labeling/priority.py` — same signal, reused.
- `scripts/` — train, tune (**calibration split only, never test**), evaluate,
  reproduce_all.

## Honest caveats

`ope spread = 0.45` — the estimator family still disagrees widely, because dataset2's
first era is near-deterministic by construction and importance weights there stay
large. Trust `switch_dr` and `direct_method`; read plain IPS as a variance diagnostic.
The realized cost/quality figures above are direct measurements on the test split and
are the more reliable numbers.

dataset2 is synthetic with a generating function written for this project, and
method5 was tuned against it. Real-data validation remains impossible at the
organizer export's n=25. See `docs/limitations.md`.
