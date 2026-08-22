---
name: 91-verify-number
description: Gate. Every number in any prose, slide, README or answer must exist in results/claims.json under a stable key and carry its estimate/assumption label. Invoke before emitting any user-facing text containing a number.
---

# 91-verify-number

The number gate. Nothing numeric leaves this repo unless it traces to `results/claims.json`.

## 1. Preconditions
- `90-leak-audit` green.
- `results/claims.json` exists. The artifacts backing it exist: `recon.jsonl`, `labels.jsonl`,
  `jobkey.jsonl`, `features.npz`, `routes.jsonl`, `estimates.json`, `metrics.json`.

## 2. Procedure
1. Extract every numeric literal from the text about to be emitted (prose, slide, chart label,
   commit-free README draft, chat answer).
2. For each, find the stable key in `results/claims.json` holding the same value. No key -> the
   number does not ship. Adding the key means re-running the module that produces it, not typing
   the value into the JSON.
3. Assert every token count is labelled an ESTIMATE. The export has no `usage` field; the
   estimator is `tok(x) = len(json.dumps(x)) // 4` everywhere.
4. Assert every dollar figure names its assumed price sheet inline — model ids are anonymized,
   no public sheet applies.
5. Assert every cost figure states its rho, or shows the sweep 1.00/0.90/0.83/0.55 ->
   0.192/0.272/0.329/0.555.
6. Assert no friction effect appears as a point: it is an interval with a named assumption, plus
   the 11.4pp MDE.
7. Assert n is stated as 1000 wherever sample size appears; 953 is a defect value and fails.
8. `python -m router.verify <textfile>` prints, per number, key/value/label status and exits
   non-zero on the first unbacked number.

## 3. Acceptance
- 100% of numeric literals in the checked text resolve to a stable key in `results/claims.json`.
- Anchor values present and matching: n = 1000; turns 10,845; gross 334,729,910;
  gross ex-tools 290,009,114; tools re-billed 44,720,796; cache read 308,074,571 (92.0%);
  cache write 26,655,339 (8.0%); naive 22,631,879; ratio 14.79x.
- Labels: 10,422 outputs, 9,633 resolvable (92.4%), 474 errors, `y_proc` {0:737,1:241,2:18,null:4},
  `y_fric` 259.
- Jobkey: 784 literal, 354 paths, 152 jobs / 374 runs, 28 multi-arm (245 runs), 23 both-family
  (229 runs), 216 no path, 410 PII paths, `crons/heartbeat/` n = 134.
- Features: 12,114,376 / 10,517,503 = 53.5% / 46.5%.
- Policy: 675/1000, 197,551,096 tok, 59.0%; exclusions 245 (32.3%), 23 (7.7%), 71 (7.0%), 9 (1.7%).
- Family contrast: claude 33/157 = 21.0%, gpt 21/72 = 29.2%, with the 134/229 caveat.
- Exit code 0.

## 4. Banned moves
- Typing a number into `results/claims.json` by hand to make the gate pass — the key must be
  written by the module that computes it.
- Rounding a contract integer (e.g. 334.7M in place of 334,729,910) in a checked artifact.
- Quoting a token count without the word estimated — the export has no `usage` field.
- Quoting a dollar figure without the assumed sheet — ids are anonymized.
- Quoting n = 953 — that is `group_trajectories()`'s collision count; n is 1000.
- Quoting a friction point estimate — MDE 11.4pp; intervals only.
- Quoting a single-rho saving as the headline without the sweep.

## 5. Postconditions
- Gate result recorded; green permits the text to be emitted.
- Re-runs after 20, 31, 40, 41, 42, 43 and 95 write.
- Red blocks 95-make-frontier and any submission or presentation step.

## 6. Escalate when
- A needed number has no producing module. Write the module; do not hand-fill the key.
- Two artifacts disagree on the same quantity — stop everything, that is a pipeline defect.
- A deadline pressures you to ship an unbacked number. Ship the text without the number instead.
