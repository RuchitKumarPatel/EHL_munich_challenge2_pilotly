---
name: 43-family-contrast
description: Compare claude vs gpt friction on the clean cross-family jobs, with the concentration caveat attached. Invoke when a family-level claim is about to be made.
---

# 43-family-contrast

The only within-job comparison this export supports across families: the 23 fully-literal cron
jobs that were served by both families. Writes into `results/estimates.json` / `results/claims.json`.

## 1. Preconditions
- `00-premise-freeze` read; `results/jobkey.jsonl`, `results/labels.jsonl`, strata from 40 exist.
- The contrast is restricted to fully-literal job keys — byte-exact path match, no `PII_` token.

## 2. Procedure
1. Select the 23 fully-literal jobs that span BOTH families (229 runs).
2. Within each job, tabulate friction (`y_fric`) by family. Pool across the 23 jobs.
3. Report the pooled rates: claude 33/157 = 21.0%, gpt 21/72 = 29.2%.
4. Attach the concentration caveat in the SAME sentence: 134 of the 229 runs are the single job
   `crons/heartbeat/`. The pooled contrast is largely one job.
5. Report the leave-one-job-out version: recompute the pooled rates with `crons/heartbeat/`
   removed, so the reader sees how much of the gap survives.
6. Present the difference as a BOUND with the 11.4pp MDE stated, not as a point estimate. At
   n = 157 vs 72 the gap is not resolvable.
7. Note that family is perfectly separated by the toolset (245 gpt / 755 claude, 0 exceptions),
   so a "family effect" here is confounded with a tool-contract difference. Say so.
8. `python -m router.family` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- 23 fully-literal jobs span both families, covering 229 runs (of 152 fully-literal jobs / 374 runs).
- Pooled friction: claude 33/157 = 21.0%; gpt 21/72 = 29.2%.
- 33 + 21 = 54 friction runs; 157 + 72 = 229 runs.
- 134 of the 229 runs are `crons/heartbeat/` (n = 134 across 7 arms).
- The 11.4pp MDE is quoted with the difference; the difference is reported as an interval.
- Family split of the whole export restated: 245 gpt / 755 claude, 0 exceptions.

## 4. Banned moves
- Reporting "gpt is 8.2pp worse" as a finding — MDE is 11.4pp on the best-powered arm pair; this
  pair is smaller still.
- Pooling without disclosing that 134/229 runs are one job.
- Using the `# === Cron memory: PII_URL_N` field to widen the job set — 763/763 PII placeholders,
  18 distinct values, `PII_URL_3` appears 306 times.
- ANY `PII_*` token as a join key — renumbered per request.
- `group_trajectories()` to build the job set — 953 collision groups; n is 1000.
- `empty_final` / `final_text_length` in this contrast — 238/238 empty finals are gpt; it would
  manufacture the family difference outright.
- Text-keyword error detection for the friction counts — precision 2.8% to 49%.
- An LLM judge over the final message — 623/762 say "done", 8 admit any problem.

## 5. Postconditions
- Contrast table and its caveat in `results/estimates.json`, mirrored into `results/claims.json`.
- `91-verify-number` re-runs and fails any family sentence lacking the concentration caveat.
- 95-make-frontier may cite the bound, never a point.

## 6. Escalate when
- The both-family job count drops below 23 or the run count below 229 — the contrast loses its
  only evidence.
- Removing `crons/heartbeat/` flips the sign of the gap. Then report both and claim nothing.
- Someone wants this framed as "family X is better". It is confounded with the tool contract
  (perfect 245/755 separation); refuse that framing.
