---
name: 12-extract-features
description: Build the pre-treatment feature matrix results/features.npz plus feature_manifest.json. Invoke after recon and jobkey, before fitting any predictor or designing a policy.
---

# 12-extract-features

Extracts features from the pre-treatment slice ONLY — `input[:first_user_msg_index+1]` plus the
tools block. Writes `results/features.npz` and `results/feature_manifest.json`.

## 1. Preconditions
- `00-premise-freeze` read; `results/recon.jsonl` and `results/jobkey.jsonl` exist.
- The cut point is fixed: index of the first `role == 'user'` item (index 1 in 1000/1000 lines).
  Everything after it is model-generated and therefore a collider.

## 2. Procedure
1. For each line take `items[:first_user_msg_index+1]` and `tools`. Nothing else is readable.
2. Block A — request shape: `pre_tok`, `tools_tok`, system-message length, user-message length,
   log transforms, counts of structural markers in the user message.
3. Block B — toolset: number of tools, presence indicators for individual tool names.
   Family is derived here (245 gpt / 755 claude) and is a GATE variable, not a free feature.
4. Block C — job context: features from `results/jobkey.jsonl` (`bucket`, `literal`, path depth,
   path-prefix indicators). The cron path lives in the system message, i.e. pre-treatment.
5. Include a `has_pre_image` indicator: an `input_image` part inside the pre-treatment slice
   (9 trajectories). It gates routing in 31-design-policy.
6. Assemble `X` (n, d) float64, `cols` (d,) unicode, `idx` (n,) int64 — `idx` ascending 0..999.
   Save with `numpy.savez` to `results/features.npz`.
7. Write `results/feature_manifest.json`:
   `{"cols":[{"name":str,"pre_treatment":bool,"block":"A"|"B"|"C"}, ...]}`, one entry per column,
   same order as `cols`. Every entry must have `pre_treatment: true`.
8. Report the token split: sum of pre-treatment tokens vs post-treatment tokens across the export.
9. `python -m router.features` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- Pre-treatment tokens 12,114,376 vs post-treatment 10,517,503 = 53.5% / 46.5%
  (these sum to 22,631,879, the naive total from recon).
- `X.shape[0]` = 1000, `len(cols) == X.shape[1] == len(manifest["cols"])`.
- `idx` ascending 0..999; `X` finite everywhere (no NaN, no inf).
- Every manifest entry has `pre_treatment: true`; every entry's `block` is one of A/B/C.
- Family feature reproduces 245 gpt / 755 claude; `has_pre_image` sums to 9.

## 4. Banned moves
- ANY feature computed from `input[first_user_msg+1:]` — 46.5% of all tokens are model-generated
  there; conditioning on them is conditioning on a collider of the treatment.
- `empty_final` / `final_text_length` — 238/238 empty finals are gpt; it encodes family perfectly
  as a serialization artifact.
- System-prompt hash as a feature — 871 distinct values over 1000 rows; it is an identifier.
- Any PII_* token as a feature value or join key — renumbered per request.
- The `# === Cron memory: PII_URL_N` field — 763/763 are placeholders, 18 distinct values,
  `PII_URL_3` alone appears 306 times.
- Label fields from `results/labels.jsonl` (`n_err`, `y_fric`, `trailing_err`, ...) — outcome.
- sklearn / scipy transformers — not installed; stdlib + numpy only.

## 5. Postconditions
- `results/features.npz` and `results/feature_manifest.json` written.
- `90-leak-audit` re-runs and must confirm: 0 columns sourced post-cut, 0 label columns,
  0 identifier-cardinality columns.
- 30, 31, 40 unblock.

## 6. Escalate when
- The pre/post token split moves off 12,114,376 / 10,517,503 — the cut point changed.
- Any column would need a value from after the first user message to be computable.
- A column reaches near-1000 distinct values on 1000 rows (identifier smell, like the 871-distinct
  system-prompt hash). Drop it rather than hashing it smaller.
