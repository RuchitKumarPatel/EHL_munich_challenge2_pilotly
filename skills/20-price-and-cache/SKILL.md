---
name: 20-price-and-cache
description: Turn reconstructed turn prefixes into cache-aware cost under a NAMED price sheet, and compute the effective multiplier at several cache-hit rates. Invoke after recon, before any dollar figure is written or spoken.
---

# 20-price-and-cache

Prices the reconstructed bill. Every output of this skill is an ESTIMATE under an ASSUMED price
sheet — model ids are anonymized (AGENTS.md), so no public sheet applies. Feeds
`results/estimates.json` and `results/claims.json`.

## 1. Preconditions
- `00-premise-freeze` read (F3: gross is 334,729,910, not 22,631,879).
- `results/recon.jsonl` exists with `gross_tok`, `cache_read_tok`, `cache_write_tok` per line.
- A named price sheet is chosen and recorded as a string, e.g.
  `"assumed sheet: <name>, cache read 0.10x input, cache write 1.25x input"`.

## 2. Procedure
1. Fix the cache multipliers: cache read = 0.10x the uncached input rate, cache write = 1.25x.
   Record both numbers next to the sheet name.
2. Per line, cost = `rate(model) * (0.10 * cache_read_tok + 1.25 * cache_write_tok)`. Sum for
   the logged bill. Also compute the fully-uncached counterfactual `rate(model) * gross_tok`.
3. Effective multiplier = cache-aware cost / fully-uncached cost, at a given cache-hit rate rho
   (rho = share of `cache_read_tok` that actually hits). Compute for rho in {1.00, 0.90, 0.83, 0.55}.
   rho = 0.83 is the reference: it is the rate at which the sheet's own break-even sits.
4. Report the observed prefix split from recon: 92.0% read / 8.0% write. Note that this is the
   IDEAL split (every prefix hits); rho scales it down.
5. Write every number into `results/claims.json` under stable keys before quoting it anywhere.
6. Attach the sheet name and the word "estimated" to every dollar string you emit.
7. `python -m router.pricing` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- Prefix split from recon: `cache_read_tok` 308,074,571 = 92.0%, `cache_write_tok` 26,655,339
  = 8.0%, of `gross_tok` 334,729,910.
- Effective multiplier vs fully-uncached, read 0.10x / write 1.25x:
  rho = 1.00 -> 0.192; rho = 0.90 -> 0.272; rho = 0.83 -> 0.329; rho = 0.55 -> 0.555.
- Gross / naive = 334,729,910 / 22,631,879 = 14.79x — the number a naive cost model misses.
- Tools block re-billed across turns = 44,720,796 tokens (gross minus 290,009,114).
- Every emitted dollar figure carries the sheet name and the word "estimated".

## 4. Banned moves
- Quoting a dollar figure without naming the assumed sheet — ids are anonymized, no public
  sheet applies, and an unnamed number reads as measured.
- Pricing `naive_tok` — 22,631,879 is 1/14.79 of the reconstructed gross.
- Charging the tools block once per trajectory — it is 44,720,796 tokens across turns.
- Assuming rho = 1.00 as the headline — that is the 0.192 best case; the reference is
  rho = 0.83 -> 0.329, and the reader must see the rho sweep.
- Calling any token count "measured" — the export has no `usage` field; `tok(x) =
  len(json.dumps(x)) // 4` everywhere.
- Applying claude cache multipliers to gpt lines without saying the sheet assumes both.

## 5. Postconditions
- Cost fields land in `results/estimates.json`; every quoted number exists in
  `results/claims.json` under a stable key.
- `91-verify-number` re-runs and fails any number in prose that is absent from claims.json.
- 31-design-policy and 95-make-frontier unblock.

## 6. Escalate when
- The multiplier sweep does not reproduce 0.192 / 0.272 / 0.329 / 0.555 at
  rho = 1.00 / 0.90 / 0.83 / 0.55.
- The organizers post a real price sheet — re-run everything and re-label every figure; do not
  keep old dollars alongside new ones.
- Someone asks for a single headline dollar saving with no rho stated. The answer is a range
  across the rho sweep, not a point.
