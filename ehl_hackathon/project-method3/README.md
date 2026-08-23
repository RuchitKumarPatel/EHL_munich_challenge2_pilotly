# project-method3

A calibrated cascade router for agent trajectories. Chooses which model handles an
entire trajectory, before it runs, using only information available at that point —
and reports a distribution-free calibrated lower bound on the quality of that choice,
not a heuristic uncertainty score.

Built after auditing project-method1 and project-method2 and finding two load-bearing
issues in both: a feature-leakage bug (routing decisions were trained/predicted using
full-trajectory hindsight statistics — total call count, whether errors occurred —
that can't be known before the trajectory runs) and ad hoc, uncorrected uncertainty
heuristics standing in for a real quality guarantee. See `docs/methodology.md` for
the full writeup and `docs/literature.md` for what this design borrows from (and
deliberately doesn't attempt) in the routing literature.

## Quickstart

```
pip install -r requirements.txt   # matplotlib only, for the Pareto plot
python -m pytest tests/ -v        # 58 tests, ~5s
python scripts/reproduce_all.py ../dataset1/export
```

Also runs unmodified against the organizer's raw export
(`../viktor-tumai-starter/viktor-tumai-starter/export`) — no ground truth or
scenario tags there, so it falls back to the structural-proxy label and skips
segmentation, exactly as designed.

## The one invariant that matters most

A routing decision (`routing/base.py`'s `Router.route`) receives an `OpeningContext`
— system + first user message + tool schema — and nothing else. It cannot see how
many calls the trajectory ends up taking, whether an error occurs, or any other
fact only knowable in hindsight, because `OpeningContext` doesn't hold a reference to
that information at all. This is enforced by type, not by convention, and is
regression-tested in `tests/test_no_feature_leakage.py`.

Everything that legitimately needs the full trajectory — training LABELS (a label
may depend on the full outcome; that's supervised learning, not leakage), cost
accounting, and post-hoc evaluation — lives in a separate, clearly-named module
(`features/diagnostic_features.py`) and is never passed to a router.

## What's in here

- `src/method3/data/` — schema, loader, train/calibration/test split, dataset1
  ground-truth manifest loader.
- `src/method3/features/` — the pre-decision/post-hoc split (see above).
- `src/method3/quality/` — structural-proxy + ground-truth labels, a KNN reward
  model (opening features only), and split-conformal calibration for the quality
  lower bound.
- `src/method3/propensity/` — a fitted (not frequency-based) logging-policy model
  for honest doubly-robust off-policy evaluation.
- `src/method3/pricing/` — cache-aware cost model with a real (not just reported)
  model-switch penalty.
- `src/method3/routing/` — `CascadeRouter` (the proposed policy) and
  `BaselineOracle` (logged/cheapest/strongest reference points — explicitly
  documented as oracle baselines, not deployable policies, since they need the full
  trajectory to be costed).
- `src/method3/evaluation/` — policy grading, doubly-robust OPE, per-scenario-tag
  segmentation (dataset1 manifest tags), bootstrap CIs.
- `scripts/` — one script per pipeline stage; `reproduce_all.py` runs the lot.
- `tests/` — 58 tests, several written specifically as regressions for bugs found
  and fixed in method1/method2 this session (mixed-model-trajectory crashes, the
  switch-penalty-computed-but-never-charged bug) and for this project's own central
  claim (no feature leakage).

## Scope decision worth knowing about

Routing decisions are made once, per trajectory, before it runs — not per call
mid-trajectory. This isn't a limitation of ambition; per-step model switching can't
be honestly evaluated from logged single-model data. See "The Replay Gap"
(arXiv:2608.08239) in `docs/literature.md` and `docs/limitations.md`.
