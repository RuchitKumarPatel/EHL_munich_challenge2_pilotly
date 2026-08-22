---
name: 90-leak-audit
description: Gate. Prove no post-treatment, label or identifier field reached the features, the predictor or the policy. Runs after every artifact write and before any result is quoted.
---

# 90-leak-audit

The leak gate. It re-runs after every artifact write in this pipeline. A red gate blocks
41, 42, 43 and 95 — no number may be quoted while it is red.

## 1. Preconditions
- `results/feature_manifest.json` and `results/features.npz` exist.
- `results/recon.jsonl` exists (supplies `pre_tok` and the cut index per line).
- If `results/routes.jsonl` exists, its gate fields are audited too.

## 2. Procedure
1. Cut check. For every column in `results/feature_manifest.json`, assert `pre_treatment: true`
   and confirm it is computable from `items[:first_user_msg_index+1]` + `tools` alone. Any column
   needing a later item fails the gate.
2. Token-split check. Recompute pre-treatment vs post-treatment tokens over the export and assert
   12,114,376 vs 10,517,503 = 53.5% / 46.5%. A drift means the cut index moved.
3. Label check. Assert no column name matches a field of `results/labels.jsonl`
   (`n_calls, n_obs, n_err, n_unknown, trailing_err, y_fric, y_proc, y_rate, n_kill`), and that
   no column is a monotone function of one.
4. Artifact check. Assert `empty_final` and `final_text_length` appear nowhere in the manifest,
   the predictor's column set, or `routes.jsonl` gates. 238/238 empty finals are gpt.
5. Identifier check. Flag any column with more than ~500 distinct values on 1000 rows. The
   system-prompt hash has 871 distinct values and must be absent.
6. PII check. Assert no column name or value derives from a `PII_*` token.
7. Policy check. Assert every gate in `routes.jsonl` (`family, vision, admissible, sign_stable,
   tau_pass`) is computable pre-treatment, and that `p_fric` is out-of-fold.
8. `python -m router.leakaudit` prints PASS/FAIL per check and exits non-zero on any FAIL.

## 3. Acceptance
- Pre-treatment tokens 12,114,376 vs post-treatment 10,517,503 = 53.5% / 46.5%.
- 100% of manifest columns have `pre_treatment: true`; 0 post-cut columns.
- 0 label-derived columns; 0 occurrences of `empty_final` / `final_text_length` anywhere.
- 0 columns with identifier cardinality (the 871-distinct system-prompt hash is absent).
- 0 `PII_*`-derived columns or join keys.
- Policy gates: family reproduces 245 gpt / 755 claude; vision gate flags exactly 9; routable set
  675/1000 = 59.0% of gross.
- Exit code 0.

## 4. Banned moves
- ANY feature from `input[first_user_msg+1:]` — 46.5% of tokens are model-generated (collider).
- `empty_final` / `final_text_length` — 238/238 empty finals are gpt, a serialization artifact
  encoding family perfectly.
- System-prompt hash — 871 distinct over 1000 rows; an identifier.
- Any `PII_*` token as a feature or join key — renumbered per request.
- The `# === Cron memory: PII_URL_N` field — 763/763 placeholders, 18 distinct, `PII_URL_3` x306.
- Waiving a FAIL because "the effect is small" — the gate is binary.
- Quoting any result while the gate is red.

## 5. Postconditions
- Gate status recorded in `results/claims.json` under a stable key.
- Green unblocks 41, 42, 43, 95. Red blocks all four and blocks 91-verify-number.
- Re-runs automatically after 10, 11, 12, 13, 30 and 31 write.

## 6. Escalate when
- A check fails and the fix would require changing the cut index or the manifest contract.
- A column is defensible on argument but fails a check on measurement. Measurement wins; escalate
  rather than adding an exception list.
- The token split drifts from 12,114,376 / 10,517,503 for any reason.
