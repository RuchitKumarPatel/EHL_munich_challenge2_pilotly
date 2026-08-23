# Limitations

These are the honest gaps, in order of how much they should temper trust in the
headline numbers.

## The importance-weighted estimators are running on almost no evidence

On dataset2's 159-trajectory test split, the cascade router agrees with the logged
model on only **13** trajectories. Every importance-weighted estimator (IPS, SNIPS,
DR, SWITCH-DR) draws its entire correction term from those 13 samples, and the
estimator family spreads across **0.606 (IPS) to 0.897 (SNIPS)** — a 0.29 range.
That spread is the honest signal, and it is why the family is reported rather than
a single DR number: when DM and IPS disagree this much, no member of the family
should be trusted to three decimal places. The per-policy quality figures in
`policy_metrics.json` (which are direct measurements on the test split, not
importance-weighted) are the more reliable numbers here.

This is intrinsic to evaluating a *deterministic* target policy against a logging
policy that mostly disagrees with it — not a fixable implementation defect. It
would improve with a logging policy that explored more (dataset2's
`explore_2025h2` era is exactly that, and is the slice worth evaluating on when
overlap matters).

## Most test contexts have no logged support for the chosen action

`positivity_report` flags that **81%** of test trajectories are routed into
`(task_type × difficulty, model)` cells that were never logged — 63 unsupported
cells out of 144. Part of that is dataset2's deliberately planted positivity
violation, but most is simple sparsity: 8 models × 18 context groups needs far more
than 454 training trajectories to cover. Estimates for those trajectories are pure
reward-model extrapolation. The report surfaces this instead of silently
extrapolating, but surfacing is not solving.

## method3 does not beat the strongest model on quality

On dataset2: `strongest` reaches 0.851 quality at $5.53e-03; method3 reaches 0.827
at $9.88e-04. method3 **dominates the logged policy** (better quality *and* 2.8x
cheaper) and is 5.6x cheaper than `strongest` for 0.024 less quality — but if
quality is the only objective and budget is unbounded, `strongest` still wins.
That is the correct result, not a defect: the cost-quality frontier has more than
one point on it and method3 claims a favorable point on it, not global dominance.

## Model prices must be declared, not guessed

`pricing.resolve_pricing` prefers prices the dataset declares. The name-based
fallback in `ensure_models` is a last resort and now *reports* every model it had
to guess for, because a guessed price is a correctness bug, not an approximation:
`gpt-5.7-nova` matched none of the name rules and silently fell through to the
cheapest tier ($0.30/Mtok when the dataset says $4.00/Mtok), so the router believed
a mid-priced model was the cheapest available and sent it **100% of traffic**.
Check `guessed_prices` in the evaluation output — if it is non-empty, treat every
cost number and every price-based routing decision as unverified.

## Per-model conformal calibration is data-starved on dataset1

dataset1's calibration split (~20% of 150 trajectories, further divided across 7
candidate models) leaves roughly 4–5 calibration examples per model — well under
the ~9–10 needed for a 90%-coverage (`alpha=0.1`) finite-sample guarantee. Every
per-model calibrator on dataset1 currently reports `coverage_guaranteed=False`; the
pooled (`"__global__"`) calibrator, fit across all ~32 calibration pairs together,
does reach the threshold and is what `CascadeRouter` actually uses in practice. This
is the calibrator being honest about its own data requirements, not a bug — but it
means the quality bound is currently a marginal, non-model-specific one rather than
the tighter per-model bound the design intends. More calibration volume (a larger
dataset1, or a larger calibration fraction) is the direct fix; see the earlier
session decision to prioritize dataset/label work before further method work.

## Cascade escalation is invisible at the default quality floor

At `quality_floor=0.5` (`configs/router.json`), the cheapest candidate model clears
the floor on essentially every dataset1 trajectory, so `CascadeRouter` collapses to
"always pick cheapest" — the escalation branch is real and tested
(`tests/test_cascade_router.py`, and empirically confirmed with `configs/router_strict.json`
at `quality_floor=0.85`, where it diverges from both the cheapest and logged
baselines), but the *default* config doesn't exercise it on this dataset. This
reflects the actual quality distribution of dataset1's models (fairly clustered),
not a routing-logic defect.

## Doubly-robust OPE is only as good as the fitted propensity and reward model

`LoggingPolicyModel` is a linear (single-layer softmax) model over opening
features; if the real historical logging policy depended on something more complex
than what those features capture, the propensity — and therefore the DR
correction — will be biased. The `direct_estimate` is reported alongside the DR
`estimate` specifically so a large gap between them is visible as a warning sign,
and `support_report`'s effective-sample-size number flags when the correction is
riding on a handful of trajectories. Neither of those diagnostics *fixes*
misspecification, they only surface it.

## Structural proxy label, where ground truth isn't available

Trajectories outside dataset1's graded subset (or any trajectory from the
organizer's raw export, which carries no ground truth at all) fall back to
`quality.labels.structural_proxy` — a heuristic over call-success/error/repetition
signals, not a measurement of actual task quality. This is the same category of
limitation the original problem statement calls out ("the final output of each
trajectory is missing... evaluation remains imperfect") and is not something this
project claims to have solved, only to have made the provenance of every label
explicit (`ground_truth` vs `proxy`, tagged per row) rather than silently blended.

## Per-step routing is out of scope, not solved

See `docs/methodology.md` §2 and `docs/literature.md`'s entry on "The Replay Gap."
`CascadeRouter` makes one decision per trajectory. A per-step router that adapts
mid-trajectory is a real technique in the literature (MTRouter, Router-R1) with real
reported gains — but those results come from evaluation against *live* environments,
which this project doesn't have access to. Building a per-step router against this
project's static logged data would produce numbers that look plausible and aren't
trustworthy, which is worse than not building it.

## The switch-penalty constant is a placeholder

`pricing/cost_model.py`'s `DEFAULT_SWITCH_PENALTY_TOKENS = 256` (re-establishing a
session with a new model after a mid-trajectory switch) is not backed by any
measurement — it's a documented, clearly-labeled assumption pending real
switch-cost telemetry (see the KV-cache cross-model-transfer literature cited in
`docs/literature.md`, which confirms this cost is real and nonzero but doesn't give
a task-specific number this project could plug in).

## Small evaluation sets generally

Even with dataset1's 150 trajectories (vs. the organizer's 25), a 31-trajectory test
split is small enough that bootstrap confidence intervals on any given policy's
quality routinely overlap with its neighbors on the Pareto frontier — treat
cost/quality *rankings* as directional, not as statistically separated results,
unless a segmentation breakdown or the CI bounds say otherwise for a specific claim.
