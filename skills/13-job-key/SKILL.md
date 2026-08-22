---
name: 13-job-key
description: Derive the recurring-job key (literal cron path) per trajectory and write results/jobkey.jsonl. Invoke after recon, before strata, family contrast or any cross-arm comparison.
---

# 13-job-key

The only usable cross-row join key in this export is the literal `Cron path:` string in the
system message. Writes `results/jobkey.jsonl`.

## 1. Preconditions
- `00-premise-freeze` read; `results/recon.jsonl` exists.
- The key must be pre-treatment: it is read from the system message, before `input[1]`.

## 2. Procedure
1. For each line, scan the system message for a literal `Cron path:` line. Capture the path
   verbatim. If absent, `cron_path = null`.
2. `literal` = True iff the captured path contains NO `PII_` placeholder token.
3. `bucket` = `"none"` when `cron_path` is null; `"pii"` when the path contains a `PII_`
   placeholder; `"literal"` otherwise.
4. Group runs by the exact literal path string. A job is "fully literal" iff every run carrying
   that path is `literal`. Report: number of distinct paths, number of fully-literal jobs and the
   runs they cover, how many of those span >= 2 arms and how many span BOTH families.
5. Write one object per line: `idx, cron_path, literal, bucket`. Nothing else.
6. `python -m router.jobkey` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- 784/1000 runs carry a literal `Cron path:` marker; 216 runs have no cron path.
- 354 distinct paths.
- 152 fully-literal jobs covering 374 runs.
- Of those 152: 28 span >= 2 arms (245 runs); 23 span BOTH families (229 runs).
- 410 runs have a `PII_` placeholder inside the path.
- Largest clean job: `crons/heartbeat/`, n = 134 across 7 arms.
- 1000 lines out, `idx` ascending 0..999.

## 4. Banned moves
- The `# === Cron memory: PII_URL_N` field as a job key — 763/763 values are PII placeholders,
  only 18 distinct, and `PII_URL_3` appears 306 times. It groups unrelated runs.
- ANY `PII_*` token as a cross-row join key — placeholders are renumbered per request, so the
  same entity gets different numbers in different rows and different entities collide.
- `group_trajectories()` as a grouping fallback — 953 collision groups; n is 1000.
- System-prompt hash as a job key — 871 distinct over 1000 rows, an identifier not a group.
- Fuzzy / normalized path matching to raise the 152 count — the join must be byte-exact or the
  cross-family contrast in 43 is not clean.
- Printing a captured path containing a `PII_` token into any user-facing output.

## 5. Postconditions
- `results/jobkey.jsonl` written, 1000 lines.
- `92-data-safety` re-runs (no PII placeholder may reach a committed or published file).
- 12 (block C features), 40 (strata) and 43 (family contrast) unblock.

## 6. Escalate when
- The literal-path count moves off 784, or distinct paths off 354 — the system-prompt template
  changed and every group-based estimate must be redone.
- The 23 both-family jobs drop below 23, or `crons/heartbeat/` drops below n = 134: the family
  contrast in 43 loses its only pooled evidence.
- No fully-literal job spans both families. Then 43-family-contrast has nothing to report and
  must say so rather than substituting a PII-based key.
