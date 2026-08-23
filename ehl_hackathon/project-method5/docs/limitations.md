# Limitations

## method5 does not beat `strongest` on raw quality

0.8236 vs 0.8484. It reaches 97% of the strongest model's quality for 19% of its
cost, and it dominates the logged policy outright. But if quality is the only
objective and budget is unbounded, always-strongest still wins. That is the correct
result and this project does not claim otherwise.

## The estimator family still disagrees badly

`spread = 0.45` across DM / IPS / SNIPS / DR / SWITCH-DR. dataset2's first era uses a
near-deterministic logging policy by construction, so importance weights there remain
large no matter how well the target policy is specified. The realized cost/quality
numbers in the README are direct measurements on the test split — not
importance-weighted — and are the trustworthy figures. Read plain IPS as a variance
diagnostic rather than an estimate.

## Exploration is driven by margin, not width, on this dataset

`corr(eps, decision_margin) = -0.886` but `corr(eps, interval_width) = -0.392`. The
gate is working (`corr(eps, signal) = +0.985`), but on dataset2 the interval widths
barely vary (relative spread 0.13) while margins vary a lot (0.52), so the decision
margin dominates. On a dataset with more heterogeneous conformal widths the balance
would shift. Do not generalize "method5 explores where the interval is wide" from
this run — it explores where the *ratio* is high, and here that means where the
decision is close.

## The exploration tax is real, just better targeted

Exploration still costs quality; method5 only spends it more selectively. On a
genuinely stationary workload that will never be retrained, any exploration is pure
loss and a deterministic router is the better choice. The case for method5 rests on
the system being retrained against its own logs, where the coverage guarantee
(reach 1.00 vs method3's 0.28) is what keeps that possible at all.

## Hyperparameters are tuned, and tuning is a dependency

`ceiling` and `temperature` come from a grid search on the calibration split;
`reference_ratio` is fit there too. The selection never touches test, and the budget
criterion is the logged policy's own spend. But a different dataset needs a re-fit,
and the first shipped defaults were badly wrong (see the README's bug notes) — treat
these as dataset2-specific, not universal.

## Everything method3's limitations already said

method5 reuses method3's data, feature, conformal and pricing layers through the
bridge, so it inherits every caveat in `project-method3/docs/limitations.md`: label
sparsity (~48% of dataset2 has no ground truth), the unmeasured switch-penalty
constant, per-trajectory rather than per-step routing, and the fact that dataset2 is
synthetic with a known generating function.

## Per-step routing remains out of scope

Deliberately, on evidence: ["The Replay Gap"](https://arxiv.org/html/2608.08239)
shows a mid-trajectory model swap rewrites 61–94% of subsequent actions relative to a
same-model control, so static replay scores a different world than the swap would
produce. Building it here would yield plausible, untrustworthy numbers.

## The closed loop

dataset2 was designed for this project, and method4/method5 were then designed and
tuned against dataset2. That is a closed loop, and a fifth method tuned on the same
synthetic data risks confirming its own generating function back to itself. The
strongest remaining work is not a method6 — it is labels, better features, and
validation on real data.
