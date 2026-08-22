---
name: 10-reconstruct-turns
description: Turn each trajectory line into its N billed API calls and write results/recon.jsonl. Invoke before any cost, feature or policy work — every downstream number depends on it.
---

# 10-reconstruct-turns

Computes, per trajectory, the turn boundaries and the billed prefix of each turn.
Writes `results/recon.jsonl`. This is the root artifact: costs, features and policy all read it.

## 1. Preconditions
- `00-premise-freeze` read. F3 binding: a line is N calls, not one.
- `export/trajectories_v1_01.jsonl` present, 1000 lines.
- `results/` exists (gitignored; safe to write at runtime).

## 2. Procedure
1. Iterate lines with `iter_requests()` only. Keep file order; `idx` = 0-based line number.
2. Family from the tools block: names containing `apply_patch` or `shell_command` -> `gpt`;
   names containing `bash`/`file_read`/`file_edit`/`file_write` -> `claude`. Assert exactly
   one match per line. Check: 245 gpt / 755 claude.
3. Mark each item model-produced iff
   `item.get('role') == 'assistant'` OR `item.get('type') in {'reasoning','function_call','custom_tool_call'}`.
4. A turn boundary opens at the START of every maximal run of model-produced items.
   `turn_cuts` = the sorted list of those start indices. `n_turns = len(turn_cuts)`.
5. `tok(x) = len(json.dumps(x)) // 4`. `tools_tok = tok(tools)`.
   For turn starting at index i: `prefix_tokens[k] = tools_tok + sum(tok(items[:i]))`.
   Compute the item-token prefix sums once per line; do not re-serialize per turn.
6. `gross_tok = sum(prefix_tokens)`.
   `cache_read_tok = sum(prefix_tokens[:-1])`; `cache_write_tok = gross_tok - cache_read_tok`.
7. `pre_tok = tools_tok + sum(tok(items[:first_user_msg_index+1]))` where first_user_msg_index
   is the index of the first item with `role == 'user'` (index 1 in 1000/1000 lines).
8. `naive_tok = sum(tok(it) for it in items)` — the starter-kit count, kept for the 14.79x ratio.
9. Write one JSON object per line, file order, with exactly the contract keys:
   `idx, model, family, n_turns, turn_cuts, prefix_tokens, tools_tok, pre_tok, gross_tok,
   cache_read_tok, cache_write_tok, naive_tok`.
10. `python -m router.recon` prints the acceptance block below, expected vs actual, and exits
    non-zero on any mismatch.

## 3. Acceptance
- `sum(n_turns)` = 10,845.
- `sum(gross_tok)` = 334,729,910.
- Gross EXCLUDING the tools block = 290,009,114; difference = 44,720,796.
- `sum(cache_read_tok)` = 308,074,571 (92.0% of gross).
- `sum(cache_write_tok)` = 26,655,339 (8.0% of gross).
- `sum(naive_tok)` = 22,631,879.
- Turns per line by arm: opus-5 10.36, sonnet-5 11.83, sol 10.85, terra 8.51, fable-5 11.72,
  opus-4-8 10.29, luna 17.30.
- Spend concentration on `gross_tok`: top10 17.7%, top50 40.5%, top100 54.7%, top500 90.2%.
- 1000 lines out; family counts 755 claude / 245 gpt; `len(prefix_tokens) == n_turns` every line.

## 4. Banned moves
- `group_trajectories()` — 953 collision groups; n is 1000 (premise F2).
- Any other tokenizer or `tok` definition — every acceptance number above is `len(json.dumps(x)) // 4`.
- Excluding the tools block from the per-turn prefix — it is 44,720,796 tokens of the bill.
- Charging the tools block once per line — that is the 290,009,114 figure, not the bill.
- Treating `naive_tok` as cost — it is 22,631,879, i.e. 1/14.79 of the real gross.
- Deriving turn boundaries from text markers or item counts instead of the model-produced-run
  rule — the rule above is what reproduces 10,845.

## 5. Postconditions
- `results/recon.jsonl` written, 1000 lines, key `idx` ascending 0..999.
- `90-leak-audit` and `91-verify-number` re-run.
- 11, 12, 13, 20, 30, 31, 40, 41, 42, 43, 95 all unblock.

## 6. Escalate when
- Any acceptance number is off by more than 0. These are exact integer contracts; a 1-token
  drift means the tok rule or the boundary rule changed. Do not adjust the expected value.
- A line has zero model-produced items (n_turns = 0), or family detection matches both or
  neither toolset. Both would falsify premise F1/F3.
