---
name: 42-partial-identification
description: Produce interval bounds on the friction effect instead of a point estimate. Invoke whenever any claim about routing's effect on quality/friction is about to be made.
---

# 42-partial-identification

The friction effect of routing is NOT point-identified in this export. This skill produces bounds
and the width that makes them honest. Writes into `results/estimates.json` and `results/claims.json`.

## 1. Preconditions
- `00-premise-freeze` read; F4 binding (Kish ESS per arm = n_a exactly for all nine arms).
- `results/labels.jsonl` (`y_fric` positives = 259), `results/routes.jsonl`, strata from 40 exist.
- 41-estimate-offpolicy has published the overlap diagnostic.

## 2. Procedure
1. State the identification problem in one line: routing is not randomized, arms do not overlap
   within strata for most runs, and the outcome is unobserved on the counterfactual arm.
2. Manski-style worst-case bounds: within each stratum, impute the unobserved counterfactual
   outcome at 0 for the lower bound and 1 for the upper bound; aggregate by stratum weight.
   Report the resulting interval. It will be wide; report it anyway.
3. Assumption-indexed narrowing: report the bound under a stated monotonicity assumption
   (routing does not reduce friction) and under a bounded-selection assumption. Every narrowed
   interval carries its assumption in the same sentence.
4. Power: report the MDE of 11.4pp on the BEST-powered arm pair. Any effect smaller than that is
   not detectable in this export, whatever the point estimate would have been.
5. Sensitivity: show how the bound moves with the 4 `y_proc` nulls and the 789 unresolvable
   tool outputs (10,422 total minus 9,633 resolvable = 789).
6. Write bounds as `[lo, hi]` pairs with the assumption name attached. Never write a midpoint.
7. `python -m router.bounds` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- Base rate: `y_fric` positives = 259/1000 = 0.259.
- Kish ESS per arm = n_a exactly (331/281/113/112/71/69/20/2/1) is restated as the reason bounds
  replace weights.
- MDE = 11.4pp on the best-powered arm pair is stated with every interval.
- Resolvability inputs: 10,422 tool outputs, 9,633 resolvable (92.4%), 474 errors, 789 unresolved;
  `y_proc` {0:737, 1:241, 2:18, null:4}.
- Every interval in `results/estimates.json` is a `[lo, hi]` pair with a named assumption; zero
  midpoints, zero bare point estimates.
- Every quoted bound exists in `results/claims.json` under a stable key.

## 4. Banned moves
- A friction POINT estimate — MDE is 11.4pp on the best-powered arm pair; the export cannot
  resolve smaller effects.
- IPS / DR / SNIPS — Kish ESS per arm equals n_a exactly.
- Reporting the midpoint of a bound as "the estimate" — a midpoint is a point estimate wearing
  an interval.
- Narrowing a bound without naming the assumption in the same sentence.
- An LLM judge over the final assistant message to fill counterfactual outcomes — 623/762 say
  "done", only 8 admit any problem.
- Text-keyword error detection to fill the 789 unresolvable outputs — precision 2.8% to 49%.
- Bootstrap CIs presented as identification — resampling narrows sampling error, not the
  identification gap.

## 5. Postconditions
- Bounds written to `results/estimates.json`, mirrored into `results/claims.json`.
- `91-verify-number` re-runs; it fails any prose that states a friction effect without an interval.
- 43 and 95 consume these bounds.

## 6. Escalate when
- A bound comes back degenerate (lo == hi) — that means the imputation collapsed and the code is
  wrong, not that the effect is identified.
- Someone (judge, teammate, slide) demands a single friction number. Give the interval and the
  11.4pp MDE; do not supply the point.
- Resolvability drops below 92.4%, widening the unresolved share past what the sensitivity
  analysis covers.
