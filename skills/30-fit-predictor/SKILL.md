---
name: 30-fit-predictor
description: Fit the pre-treatment friction predictor p_fric on results/features.npz with numpy only. Invoke after features and labels, before designing the routing policy.
---

# 30-fit-predictor

Fits `p_fric(x)` — probability of friction from PRE-TREATMENT features only — with a numpy
logistic regression. Its output is consumed by 31-design-policy as the `p_fric` field of
`results/routes.jsonl`.

## 1. Preconditions
- `00-premise-freeze` read (F4: no learnable logging policy; this predicts the OUTCOME, it is
  not a propensity model and must never be used as one).
- `results/features.npz` + `results/feature_manifest.json` exist, every column `pre_treatment: true`.
- `results/labels.jsonl` exists; `y_fric` has 259 positives out of 1000.
- `90-leak-audit` green.

## 2. Procedure
1. Load `X`, `cols`, `idx`; align `y_fric` by `idx`. Assert the manifest and `cols` agree in
   order and length.
2. Standardize columns with train-fold statistics only. Store the fold means/stds.
3. Fit L2-regularized logistic regression by gradient descent / Newton steps in numpy.
   No sklearn, no scipy — neither is installed and neither may be added.
4. Evaluate out-of-fold: K-fold CV with folds blocked by job key from `results/jobkey.jsonl`,
   so runs of the same cron path never straddle a fold. Report OOF AUC and OOF Brier score
   against the base rate 259/1000 = 0.259.
5. Report per-column sign stability across folds: a coefficient is sign-stable iff its sign is
   identical in every fold. Only sign-stable columns may be cited as structure in the writeup.
6. Emit `p_fric` per `idx` from the out-of-fold predictions, never from an in-sample refit.
7. `python -m router.predictor` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- 1000 OOF predictions, `idx` ascending 0..999, all in [0, 1], all finite.
- Base rate reproduces `y_fric` positives = 259 (0.259).
- Every input column has `pre_treatment: true` in the manifest; feature count matches
  `len(cols)`; the pre/post token split behind them is 12,114,376 / 10,517,503 = 53.5% / 46.5%.
- Folds are job-blocked against the 354 distinct paths / 152 fully-literal jobs from jobkey.
- OOF Brier is reported against the 0.259 base rate; an OOF AUC at or below 0.5 is reported as
  "no learnable signal", not hidden.
- `p_fric` is out-of-fold everywhere; no in-sample score is written to `results/routes.jsonl`.

## 4. Banned moves
- sklearn / scipy — not installed and must not be installed.
- ANY feature from `input[first_user_msg+1:]` — 46.5% of tokens are model-generated (collider).
- `empty_final` / `final_text_length` — 238/238 empty finals are gpt; perfect family leak.
- System-prompt hash — 871 distinct over 1000 rows; it memorizes rows.
- Random (non-job-blocked) folds — 152 fully-literal jobs cover 374 runs and one job covers 134;
  random folds leak the job across the split.
- Using `p_fric` as a propensity / IPS weight — Kish ESS per arm equals n_a exactly; there is no
  logging policy to model.
- Reporting an in-sample AUC as if it were performance.

## 5. Postconditions
- `p_fric` per `idx` available to 31-design-policy; sign-stable column list available to it too.
- `90-leak-audit` re-runs over the used column set.
- 31 unblocks.

## 6. Escalate when
- OOF AUC is at or below 0.5 — say so and route on cost structure alone; do not add
  post-treatment features to rescue it.
- Any column with `pre_treatment: false` reaches the design matrix.
- A fold contains fewer than ~10 positives, making the fold's estimate meaningless (259 positives
  over K folds).
