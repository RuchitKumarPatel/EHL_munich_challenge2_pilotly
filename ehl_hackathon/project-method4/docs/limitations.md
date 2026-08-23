# Limitations

## method4 does not beat method3 on raw quality

0.8224 vs 0.8271, with overlapping confidence intervals ([0.814, 0.831] vs
[0.822, 0.833]). method4's case is that it reaches statistically indistinguishable
quality at 20% lower cost **and** keeps the system evaluable (reach 1.00 vs 0.33,
145 usable OPE samples vs 13). If the only objective is single-shot quality on a
system that will never be retrained, method3 is the simpler choice and this project
does not claim otherwise.

## Stochasticity buys sustainability by spending a little quality

The exploration floor deliberately routes ~5% of traffic away from the best-known
model. That is a real, permanent quality tax, and it is the price of positivity. The
tax is bounded and tunable (`exploration_floor`), and the bet-hedging argument says
it is worth paying under drift — but on a genuinely stationary workload with no
retraining, it is pure loss.

## The estimator family still disagrees

Spread is 0.21 (IPS 0.677 vs SNIPS 0.889) — much better than method3's 0.29 and on
10x the evidence, but not small. The logging policy in dataset2's first era is
near-deterministic by construction, so importance weights there remain large. Read
`switch_dr` and `direct_method` as the trustworthy members and treat plain IPS as a
variance diagnostic rather than an estimate.

## Response-threshold specialization is weak on dataset2

The specialization index is ~0.07. Thresholds key on a *proxy* for task type (the
declared tool signature plus a prompt-size tercile), because the true task type is a
manifest label a pre-decision router may not read. The proxy separates task families
only partially, so the mechanism contributes less than the other three pillars here.
An earlier version keyed on the dominant hashed-bag-of-words bucket and produced only
**3** buckets across 454 trajectories (filler text dominated every hash); the tool
signature gives 14. Better context features would raise this further.

## Hyperparameters are tuned, and tuning is a dependency

`risk_aversion`, `temperature` and `exploration_floor` come from a grid search on the
calibration split (`scripts/tune_method4.py`). The selection never touches test, and
the criterion is the geometric-mean objective under a cost ceiling set by the logged
policy's own spend. But a different dataset needs a re-tune, and the default that
shipped first was badly wrong (see the README's bug note) — so treat the defaults as
dataset2-specific, not universal.

## Everything method3's limitations already said

method4 reuses method3's data, feature, conformal and pricing layers, so it inherits
every caveat in `project-method3/docs/limitations.md` — label sparsity (~48% of
dataset2 has no ground truth), the unmeasured switch-penalty constant, per-trajectory
rather than per-step routing, and the fact that dataset2 is synthetic with a known
generating function.

## Per-step routing remains out of scope

Deliberately, on evidence: ["The Replay Gap"](https://arxiv.org/html/2608.08239)
shows a mid-trajectory model swap rewrites 61–94% of subsequent actions relative to a
same-model control, so static replay scores a different world than the swap would
produce. Building it here would produce plausible, untrustworthy numbers.
