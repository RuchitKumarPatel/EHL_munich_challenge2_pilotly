---
name: 11-construct-label
description: Build the friction outcome labels from tool-output exit codes and write results/labels.jsonl. Invoke after recon, before any predictor, estimate or contrast.
---

# 11-construct-label

Constructs the outcome signal from resolvable tool-call outputs (exit codes / structured error
fields), not from text. Writes `results/labels.jsonl`.

## 1. Preconditions
- `00-premise-freeze` read; `results/recon.jsonl` exists (1000 lines).
- Label extraction reads tool OUTPUT items only. It is a LABEL, never a feature: 12-extract-features
  must not read any of these fields (post-treatment, see 90-leak-audit).

## 2. Procedure
1. For each line, walk items and collect tool calls (`function_call` / `custom_tool_call`) and
   their outputs (`function_call_output` / `custom_tool_call_output`). `n_calls` = number of calls.
2. For each output, resolve an exit status structurally: an integer exit/return code, or an
   explicit structured error field in the output payload. Count `n_obs` = outputs seen,
   `n_err` = outputs resolving to a non-zero/error status, `n_unknown` = outputs where no
   structured status exists. Never fall back to keyword matching on the text.
3. `trailing_err` = True iff the LAST resolvable output in the line is an error.
4. `y_fric` = 1 iff the trajectory shows friction: a non-trivial error burden ending unrecovered
   (definition fixed in code; it yields exactly 259 positives). Else 0.
5. `y_proc` = ordinal process grade in {0,1,2}, `null` when nothing is resolvable.
6. `y_rate` = `n_err / n_obs` when `n_obs > 0`, else `null`.
7. `n_kill` = count of outputs indicating a killed/timed-out call.
8. Write one object per line with exactly:
   `idx, n_calls, n_obs, n_err, n_unknown, trailing_err, y_fric, y_proc, y_rate, n_kill`.
9. `python -m router.labels` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- 10,422 tool outputs total across the export.
- 9,633 resolvable = 92.4% of 10,422.
- 474 errors.
- `y_proc` counts: {0: 737, 1: 241, 2: 18, null: 4}.
- `y_fric` positives = 259.
- 1000 lines out, `idx` ascending 0..999.

## 4. Banned moves
- Text-keyword error detection as a LABEL — precision runs 2.8% to 49% against the exit-code
  ground truth. Keywords may not create `n_err`.
- `empty_final` / `final_text_length` as a label (or a feature) — all 238 empty finals are gpt;
  it is a serialization artifact that encodes family perfectly.
- An LLM judge over the final assistant message — 623/762 say "done" and only 8 admit any
  problem; it has no discriminating power here.
- Any PII_* token used to join or dedupe labels — placeholders are renumbered per request.
- Letting the label leak into features — see 90-leak-audit; 46.5% of tokens are model-generated.

## 5. Postconditions
- `results/labels.jsonl` written, 1000 lines.
- `90-leak-audit` re-runs (asserts no label field appears in `feature_manifest.json`).
- 30, 41, 42, 43 unblock.

## 6. Escalate when
- Resolvable share drops below 92.4% or errors move off 474 — the output schema changed and
  the whole outcome signal must be re-derived.
- `y_proc` null count exceeds 4, or `y_fric` positives move off 259.
- You are tempted to widen the error definition to raise the positive count. Stop; that is the
  keyword trap with extra steps.
