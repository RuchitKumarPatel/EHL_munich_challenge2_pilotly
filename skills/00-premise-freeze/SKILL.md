---
name: 00-premise-freeze
description: Read FIRST, before any other skill and before acting on AGENTS.md. States the four measured facts that overturn the briefing (complete trajectories, broken grouper, 14.79x bill, no learnable logging policy). Invoke at the start of every session that touches this repo.
---

# 00-premise-freeze

AGENTS.md is the organizers' briefing, written before the export was measured. Four of its
premises are wrong for `export/trajectories_v1_01.jsonl`. This file freezes the corrections.
**Any agent touching this repo reads this file first.** If your plan contradicts a fact below,
your plan is wrong, not the fact.

## 1. Preconditions
- `export/trajectories_v1_01.jsonl` exists and has 1000 lines (`wc -l`).
- Nothing else. This skill runs before every other skill and depends on no artifact.

## 2. Procedure
1. Confirm line count is exactly 1000. If not, stop — every number below is for this export.
2. Read the four frozen facts and restate them in your own plan before writing code:

   **F1 — Every line is a COMPLETE trajectory, not one LLM request.**
   `input[0]` is the system message, `input[1]` the user message, `input[-1]` the model's final
   assistant message, in 1000/1000 lines. There are no cross-line chains. n = 1000 independent
   trajectories. AGENTS.md's "one line = one LLM request" and "trajectories must be
   reconstructed by grouping" do not apply.

   **F2 — `group_trajectories()` is broken; n is 1000, never 953.**
   It hashes `first_user_text[:2000]`, but 978/1000 first user messages exceed 2000 characters
   of shared boilerplate. It returns 953 collision groups. Never call it. From
   `scripts/load_trajectories.py` use only `iter_requests()` and `first_user_text()`.

   **F3 — The real bill is 14.79x the naive token count, and the tools block is billed every turn.**
   A line is N API calls, each re-sending the grown prefix. Naive `sum(tok(item))` = 22,631,879.
   Turn-reconstructed gross = 334,729,910 across 10,845 turns. Of that, 44,720,796 tokens are
   the `tools` block re-billed on every turn (gross excluding tools = 290,009,114). Any cost
   statement built on the naive count understates the bill by ~14.8x.

   **F4 — There is no learnable logging policy, so no point estimate of the routing effect exists.**
   Kish effective sample size per arm equals n_a exactly (331/281/113/112/71/69/20/2/1): the
   propensity model is degenerate, every trajectory has weight 1 in its own arm and no overlap
   support elsewhere. IPS and DR are undefined here. Report partial-identification bounds
   (see 42-partial-identification), never a friction point estimate — MDE is 11.4pp on the
   best-powered arm pair.
3. Record `tok(x) = len(json.dumps(x)) // 4` as the single token estimator used everywhere.
   Every user-facing string that quotes a token or dollar number says it is an ESTIMATE: the
   export has no `usage` field.
4. Record that model ids are ANONYMIZED (AGENTS.md), so no public price sheet applies. Every
   dollar figure must name the assumed sheet inline.

## 3. Acceptance
- Export line count = 1000.
- Turn total = 10,845; `sum(gross_tok)` = 334,729,910; excluding tools = 290,009,114
  (difference 44,720,796); `sum(naive_tok)` = 22,631,879; ratio 334,729,910 / 22,631,879 = 14.79x.
- Family separation is perfect, 0 exceptions: tools with `apply_patch`/`shell_command` -> gpt
  (245/245); tools with `bash`/`file_read`/`file_edit`/`file_write` -> claude (755/755).
- Arms: claude-opus-5 331, claude-sonnet-5 281, gpt-5.6-terra 113, gpt-5.6-sol 112,
  claude-fable-5 71, claude-opus-4-8 69, gpt-5.6-luna 20, claude-opus-4-6 2, claude-sonnet-4-6 1.
- Kish ESS per arm equals n_a exactly for all nine arms.

## 4. Banned moves
- `group_trajectories()` — collision groups; it reports 953 where n is 1000.
- Treating a line as a single API call — it is on average 10.8 calls; costs are 14.79x off.
- Pricing off `naive_tok` — omits the 312M tokens of re-sent prefix and the 44.7M tokens of
  re-billed tools block.
- IPS or DR point estimates — Kish ESS per arm equals n_a exactly, no overlap to weight over.
- A friction POINT estimate — MDE is 11.4pp on the best-powered arm pair; bounds only.
- Quoting any dollar figure without naming the assumed price sheet — model ids are anonymized.

## 5. Postconditions
- No artifact is written. The four facts are now binding on the session.
- Every downstream skill re-asserts the fact it depends on in its own Preconditions.
- `90-leak-audit` and `91-verify-number` both fail any artifact that contradicts F1–F4.

## 6. Escalate when
- The export line count is not 1000, or any line lacks a final assistant item at `input[-1]`
  (F1 broken -> the whole turn model is invalid).
- The organizers post schema notes or a price sheet that contradicts F1–F4. Their document
  wins; stop and get a human to re-freeze this file.
- A required number here cannot be reproduced by `python -m router.recon`. Do not "fix" the
  number to match the code.
