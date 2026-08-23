# Evaluation protocol

1. **Split**: deterministic hash-based train (60%) / calibration (20%) / test (20%)
   on trajectory key (`data/split.py`). Calibration is never merged into train — see
   `docs/methodology.md` §6.
2. **Labels**: `quality.labels.calibrated_outcome` — ground truth
   (dataset1's `scenario_manifest.json`) where available, structural proxy
   otherwise, provenance tagged per trajectory.
3. **Reward model**: `KNNRewardModel` fit on train, opening features only, one
   example set per non-mixed candidate model.
4. **Conformal calibration**: per-model + pooled (`"__global__"`) split-conformal
   calibrators fit on the calibration split's (predicted, actual) pairs.
5. **Propensity**: `LoggingPolicyModel` fit on train, opening features only,
   non-mixed trajectories.
6. **Policies evaluated on test**: `logged` / `cheapest` / `strongest` (oracle
   baselines, `routing/baselines.py`) and `method3` (`CascadeRouter`, the proposed
   policy — the only one of these that is a `Router` in the pre-decision-only
   sense).
7. **Grading**: `evaluate_policy` — a policy's chosen route is graded factually
   (from the label) when it matches the trajectory's real per-call model sequence,
   counterfactually (from the reward model's prediction) otherwise. Cost is always
   the post-hoc repriced cost of the chosen route via `pricing.CostModel`, which
   includes the cache-reset switch penalty.
8. **Off-policy estimate**: `evaluate_doubly_robust` for `method3` only (the oracle
   baselines aren't causal policies in the sense DR requires — see
   `docs/methodology.md` §5), reporting both the DR estimate and the unweighted
   direct estimate side by side, plus a support/overlap diagnostic.
9. **Segmentation**: every policy's per-trajectory rows broken down by dataset1's
   scenario tags (task_type, difficulty, domain, has_failure, cache_break,
   multi_turn, branching, graded) — `evaluation/segmentation.py`. Empty (`{}`) when
   no manifest exists (e.g. the organizer's raw export).
10. **Reporting**: cost-quality Pareto frontier + a quality-per-dollar summary
    (`visualization/pareto.py` — explicitly NOT a reimplementation of RouterBench's
    formal AIQ metric, see that module's docstring) with bootstrap 95% CIs on every
    quality number.

Run the whole thing: `python scripts/reproduce_all.py <source>`. Component-level
ablation (conformal vs. no calibration margin; ground-truth vs. proxy-only labels):
`python scripts/ablation.py <source>`.
