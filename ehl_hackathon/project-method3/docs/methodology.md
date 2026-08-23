# Methodology

## 1. The pre-decision/post-hoc boundary

A routing decision must fire before the trajectory it's routing has happened. Both
project-method1 (`extract_trajectory_features`) and project-method2
(`features.extract`) train and predict using features that include the trajectory's
total call count, total tokens, and whether an error occurred anywhere in it — all
only knowable after the fact. Both `Router.route()`-equivalent interfaces in those
projects also take the whole already-loaded `Trajectory`, not just its opening, so
the leak is in the type signature, not just feature selection.

This project fixes that at the type level. `data/schema.py`'s `OpeningContext` is
built from `trajectory.calls[0]` only and holds no reference to any later call —
`OpeningContext.from_first_call()` is the only constructor, and there is no method
on it that reaches back into the trajectory. `routing/base.py`'s `Router.route`
takes `(opening: OpeningContext, candidate_models: list[str])`. A conforming router
cannot leak later-call information even by accident, because it's never in scope.

`features/opening_features.py` (pre-decision: text stats, tool schema, a hashed
bag-of-words block for lexical signal) and `features/diagnostic_features.py`
(post-hoc: call count, error markers, cache overlap) are two separate modules with
disjoint field-name sets, checked by `tests/test_no_feature_leakage.py`.

It is legitimate for a training LABEL to depend on the full trajectory outcome —
that's ordinary supervised learning, not leakage. `quality/labels.py`'s
`structural_proxy` is built on `diagnostic_features` for exactly this reason; the
distinction the whole project turns on is features (decision inputs) vs. labels
(training targets), not "may this touch the full trajectory."

## 2. Why route once, per trajectory, not per call

"The Replay Gap" (arXiv:2608.08239) forks live SWE-bench trajectories at controlled
points and shows that swapping models mid-trajectory rewrites 61–94% of everything
that happens after the swap relative to a same-model control — static replay
evaluation of a mid-trajectory switch is provably scoring a different world than the
one the switch would actually produce. This project has no live model access, so a
per-step router's decisions cannot be honestly validated here at all. `CascadeRouter`
therefore makes one decision, before the first call, for the whole trajectory —
exactly matching how the historical single-model trajectories were generated, so
replaying the chosen model's own trajectory is not a divergence-prone counterfactual.
Per-step live routing (MTRouter-style) is a legitimate direction but requires live
rollout infrastructure this project doesn't have; see `docs/limitations.md`.

## 3. Quality estimation: KNN point estimate + split-conformal lower bound

`quality/reward_model.py`'s `KNNRewardModel` predicts `E[quality | opening, model]`
via inverse-distance-weighted KNN over standardized opening features, fit
per-candidate-model on trajectories whose `logged_model` matches (mixed-model
trajectories are excluded — they don't cleanly attribute an outcome to one model).

`quality/conformal.py`'s `SplitConformalCalibrator` wraps that point estimate with a
distribution-free lower bound: given a held-out calibration set (never used to fit
the reward model — see the 3-way split below), the one-sided nonconformity score is
the signed overshoot `predicted - actual`, and the finite-sample-corrected quantile
at level `ceil((n+1)(1-alpha))` is subtracted from a new prediction. This replaces
method2's `residual_bound` (an uncorrected 90th-percentile heuristic, same bound
applied everywhere regardless of local difficulty) and method1's ad hoc
`0.5 + 0.45*tanh(...)` uncertainty formula with one well-tested, literature-standard
mechanism (Vovk et al.'s split conformal), used identically for every candidate
model instead of a bespoke formula per router.

Guarantee: `P(actual >= predicted - quantile) >= 1 - alpha`, marginally, under
calibration/test exchangeability. When a candidate model's own calibration set is
too small to certify anything at the target alpha (`ceil((n+1)(1-alpha)) > n` —
common with dataset1's 7-way-split ~32-trajectory calibration set, see
`docs/limitations.md`), the calibrator honestly reports `coverage_guaranteed=False`
rather than silently returning an uncalibrated number. `scripts/calibrate_conformal.py`
also fits one pooled calibrator across every model's calibration pairs
(`"__global__"`); `CascadeRouter` falls back to it when a candidate's own per-model
calibrator isn't guaranteed — still a valid marginal guarantee, just less
locally-adaptive, and the difference between "no floor" and "a usable pooled floor"
is a real, tested behavior change (`tests/test_cascade_router.py`).

## 4. Routing decision: single-shot difficulty-gated cascade

RouteLLM's win-predictor and FrugalGPT's cascade both inform this, adapted for the
offline-only, single-decision-point constraint above: `CascadeRouter.route` computes
a calibrated lower bound for every candidate model, and among those that clear both
a minimum-support threshold (enough nearby training examples to trust the KNN
estimate) and a configured quality floor, picks the one with the cheapest price
rate. If nothing clears the floor, it falls back to the highest predicted mean among
supported candidates (not the literal cheapest — matching method2's `SafeRouter`
intent, but gated by a real calibrated bound instead of a heuristic one). A literal
FrugalGPT cascade makes a real call to the cheap model and inspects its output before
escalating; this project has no live model access, so this is a *predicted* cascade
— one decision, made from the calibrated estimate, not from an observed cheap-model
result.

## 5. Off-policy evaluation: fitted propensity, doubly robust

method1's doubly-robust estimator is structurally sound but its propensity is a
context-free class frequency — implicitly assuming the historical logging policy
assigned models uniformly at random regardless of context. If the real logging
policy was context-dependent, that biases the IPS/DR correction term.
`propensity/logging_policy.py`'s `LoggingPolicyModel` fits a proper multinomial
logistic regression (pure-Python gradient descent, softmax over opening features —
the same pre-decision-only inputs, since the logging policy, whatever it was, also
could only have used information available before the trajectory ran) predicting
`P(model chosen | opening)`, used as the propensity in
`evaluation/propensity_weighting.py`'s doubly-robust estimate.

DR estimation is only meaningful for a true `Router` (opening-context-only
decision) — the oracle baselines (`routing/baselines.py`) need the full trajectory
to be costed and are documented as reference points, not causal policies subject to
this kind of evaluation.

## 6. Split design

`data/split.py` does a deterministic hash-based 3-way split: train / calibration /
test. Calibration is distinct from train specifically because split-conformal
validity requires the calibration set to be exchangeable with test and NOT used to
fit the reward model — merging it into train would silently invalidate the coverage
guarantee.

## 7. Segmented evaluation

`evaluation/segmentation.py` breaks every policy's results down by dataset1's
scenario_manifest.json tags (task_type, difficulty, domain, has_failure,
cache_break, multi_turn, branching, graded). An aggregate-only report is exactly
what hid this session's mixed-model-trajectory crash in method1/method2 — a report
segmented by `cache_break` would have surfaced it immediately.

## 8. Zero new dependencies

Everything above (KNN, split conformal, softmax regression, doubly-robust
estimation) is implemented in pure Python stdlib. matplotlib is optional (Pareto
plot only). This matches project-method1/2's convention and, more importantly,
keeps the parts of this project making the strongest correctness claims small
enough to read and test end-to-end rather than trusting an opaque library.
