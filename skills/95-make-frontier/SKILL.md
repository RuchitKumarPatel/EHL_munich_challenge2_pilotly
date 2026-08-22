---
name: 95-make-frontier
description: Build the headline cost-quality frontier — a band, not a curve of points — from routes, bounds and the rho sweep. Invoke last, after both gates are green, when the presentation chart is needed.
---

# 95-make-frontier

The headline artifact: cost on one axis, friction on the other, drawn as a BAND because the
friction side is only partially identified. Writes `results/metrics.json` and the frontier chart.

## 1. Preconditions
- `90-leak-audit` green and `91-verify-number` green. `92-data-safety` green.
- `results/routes.jsonl`, `results/estimates.json` (with bounds from 42) exist.
- The price sheet from 20-price-and-cache is named; rho sweep computed.
- matplotlib 3.11.1 and numpy 2.5.2 are available. sklearn/scipy are not and must not be used.

## 2. Procedure
1. Sweep the policy threshold tau over the routable set (675 trajectories, 197,551,096 gross tok,
   59.0% of the bill). At each tau record: share routed, estimated cost, and the friction BOUND.
2. Cost axis: use `cost_routed_usd` from `results/routes.jsonl` at the reference rho = 0.83
   (multiplier 0.329), and draw the rho = 1.00 (0.192) and rho = 0.55 (0.555) curves as the cost
   uncertainty band. Label the assumed sheet on the axis.
3. Friction axis: plot the `[lo, hi]` interval from 42-partial-identification at each tau as a
   shaded band. Never plot a midpoint line through it.
4. Annotate the 11.4pp MDE as a reference bar on the friction axis — it shows which vertical
   distances on this chart are meaningless.
5. Annotate coverage honestly: the frontier speaks for 59.0% of the bill; 41.0% is excluded by
   gate (gpt lane 245 = 32.3%, non-admissible 23 = 7.7%, already-cheapest 71 = 7.0%,
   pre-treatment images 9 = 1.7%).
6. Put the 14.79x correction on the chart or in its caption: naive 22,631,879 vs reconstructed
   334,729,910 tokens. It is the single most load-bearing fact behind the cost axis.
7. Write every plotted number into `results/claims.json`; save the chart under `results/`.
8. `python -m router.frontier` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- Routable set on the chart = 675/1000 = 197,551,096 gross tok = 59.0% of 334,729,910.
- Exclusions annotated: 245 (32.3%), 23 (7.7%), 71 (7.0%), 9 (1.7%); the five shares sum to 100.0.
- Cost band drawn at multipliers 0.192 (rho 1.00), 0.329 (rho 0.83, reference), 0.555 (rho 0.55);
  0.272 (rho 0.90) available in `results/metrics.json`.
- Friction drawn as `[lo, hi]` bands; 0 midpoint lines; 0 unlabelled points.
- 11.4pp MDE marked on the friction axis.
- 14.79x (22,631,879 -> 334,729,910) stated in chart or caption.
- Every axis carrying dollars names the assumed price sheet; every token count says estimated.
- `91-verify-number` exits 0 on the chart's labels and caption.

## 4. Banned moves
- A single point instead of a frontier — the challenge's headline artifact is the frontier, and
  the friction axis is not point-identified (MDE 11.4pp).
- A midpoint line through the friction band — a midpoint is a point estimate in disguise.
- A cost axis at one rho with no band — the sweep spans 0.192 to 0.555.
- A cost axis built on `naive_tok` — 22,631,879 is 1/14.79 of the reconstructed gross.
- Dollars without the assumed sheet — model ids are anonymized.
- Plotting a "saving" over the whole export — the policy covers 59.0% of the bill, not 100%.
- Any label containing a `PII_*` token or an export snippet (92-data-safety).
- sklearn / scipy for smoothing or fitting the frontier — not installed.

## 5. Postconditions
- `results/metrics.json` and the frontier chart written under `results/` (gitignored).
- `91-verify-number` re-runs over the chart labels and caption; `92-data-safety` re-runs over the
  written files.
- The presentation and submission steps (`skills/make-presentation`, `skills/prepare-submission`)
  consume this artifact.

## 6. Escalate when
- The friction band is so wide it covers the full [0,1] range at every tau — report that as the
  finding (the export cannot resolve the quality side) rather than narrowing it silently.
- Any gate is red. Do not draw a chart from a red pipeline.
- Someone asks for the chart to show one number for the slide. The claim is one sentence; the
  chart stays a band.
