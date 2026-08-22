---
name: 40-build-strata
description: Build the comparison strata (job key x pre-treatment size band) used by every off-policy estimate. Invoke after jobkey, features and policy, before estimation, bounds or the family contrast.
---

# 40-build-strata

Defines the strata inside which two arms are comparable, and reports how much of the export
actually sits in a stratum containing more than one arm. Feeds `results/estimates.json`.

## 1. Preconditions
- `00-premise-freeze` read (F4: strata replace weighting; there is no propensity model).
- `results/jobkey.jsonl`, `results/features.npz`, `results/recon.jsonl`, `results/routes.jsonl` exist.
- `90-leak-audit` green: strata are built from pre-treatment variables only.

## 2. Procedure
1. Primary stratifier: the literal cron path from `results/jobkey.jsonl`. Only fully-literal jobs
   qualify — 152 jobs covering 374 runs.
2. Secondary stratifier: a coarse `pre_tok` band (from recon). Use few, wide bands; a stratum
   with one run per arm carries no information.
3. Mark each stratum with the arms present and the families present. Report:
   strata with >= 2 arms (28 jobs, 245 runs) and strata spanning both families (23 jobs, 229 runs).
4. Flag the concentration: `crons/heartbeat/` alone is n = 134 across 7 arms, i.e. 134 of the 229
   both-family runs. Any pooled contrast must state this.
5. Report the unstratifiable remainder: 216 runs with no cron path, 410 runs whose path contains
   a `PII_` placeholder, and literal-path runs whose job has only one arm.
6. Emit per-stratum counts (arms x n) into `results/estimates.json`; every quoted number also
   goes into `results/claims.json`.
7. `python -m router.strata` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- 784/1000 runs carry a literal `Cron path:`; 216 have none; 410 have a `PII_` placeholder in
  the path; 354 distinct paths.
- 152 fully-literal jobs covering 374 runs.
- 28 of those span >= 2 arms (245 runs); 23 span BOTH families (229 runs).
- `crons/heartbeat/` is the largest clean job: n = 134 across 7 arms, i.e. 134 of the 229.
- Every stratum is built from pre-treatment variables only (jobkey + `pre_tok`).
- Arm totals across all strata plus the remainder reproduce 331/281/113/112/71/69/20/2/1.

## 4. Banned moves
- The `# === Cron memory: PII_URL_N` field as the stratifier — 763/763 PII placeholders, 18
  distinct values, `PII_URL_3` appears 306 times; it merges unrelated jobs.
- ANY `PII_*` token as a join key — renumbered per request.
- `group_trajectories()` — 953 collision groups; n is 1000.
- Stratifying on anything from `input[first_user_msg+1:]` — 46.5% of tokens are post-treatment.
- IPS / DR weights inside or across strata — Kish ESS per arm equals n_a exactly.
- Pooling the 23 both-family jobs without disclosing that 134/229 runs are one job.
- Slicing bands so fine that strata reach n = 1 per arm, then reporting the mean as an estimate.

## 5. Postconditions
- Stratum table in `results/estimates.json`; every quoted count in `results/claims.json`.
- `91-verify-number` re-runs.
- 41, 42, 43 unblock.

## 6. Escalate when
- Both-family coverage drops below 23 jobs / 229 runs, or `crons/heartbeat/` below n = 134.
- Any stratum used for a headline claim has fewer than 2 arms, or fewer than ~10 runs per arm
  (the MDE is already 11.4pp on the best-powered pair).
- A stratifier you want is only computable after the first user message.
