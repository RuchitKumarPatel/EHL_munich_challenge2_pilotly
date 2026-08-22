---
name: 41-estimate-offpolicy
description: Produce the off-policy COST estimate for the designed route and the diagnostic that shows why no outcome point estimate exists. Invoke after strata and policy; hand outcome claims to 42-partial-identification.
---

# 41-estimate-offpolicy

Estimates what the routed policy would have COST, and documents the overlap failure that makes
an off-policy OUTCOME point estimate impossible. Writes `results/estimates.json`.

## 1. Preconditions
- `00-premise-freeze` read. F4 is the operative fact: Kish ESS per arm equals n_a exactly
  (331/281/113/112/71/69/20/2/1), so the logging policy is degenerate.
- `results/routes.jsonl`, `results/recon.jsonl`, strata from 40-build-strata exist.
- The price sheet from 20-price-and-cache is named and fixed.

## 2. Procedure
1. Cost side. Sum `cost_logged_usd` and `cost_routed_usd` over the 675 routable trajectories
   (197,551,096 gross tok, 59.0% of the bill). This is a DETERMINISTIC re-pricing of observed
   token prefixes under a stated sheet — not a causal estimate. Label it that way.
2. Report the saving at each rho in the sweep {1.00, 0.90, 0.83, 0.55}, mapping to effective
   multipliers 0.192 / 0.272 / 0.329 / 0.555. Never quote one rho alone.
3. Cache caveat: a model switch resets the prefix cache, so a routed trajectory's first turn is a
   cache write. Recompute the routed split rather than reusing the logged 92.0% / 8.0%.
4. Overlap diagnostic. Compute Kish ESS per arm and show it equals n_a for all nine arms.
   Publish this table — it is the evidence that IPS/DR are undefined here.
5. Outcome side. Do NOT compute one here. Emit `y_fric` bounds via 42-partial-identification and
   reference them from `results/estimates.json`.
6. Write every quoted number to `results/claims.json` under a stable key.
7. `python -m router.offpolicy` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- Routable set 675/1000 = 197,551,096 gross tok = 59.0% of `sum(gross_tok)` 334,729,910.
- Kish ESS per arm equals n_a exactly: 331, 281, 113, 112, 71, 69, 20, 2, 1.
- Effective multiplier sweep reproduces rho 1.00 -> 0.192, 0.90 -> 0.272, 0.83 -> 0.329,
  0.55 -> 0.555.
- The 14.79x correction is stated: gross 334,729,910 vs naive 22,631,879.
- No field in `results/estimates.json` contains an IPS, SNIPS, DR or DM point estimate of friction.
- Every dollar figure names the assumed sheet and the word "estimated".

## 4. Banned moves
- IPS, SNIPS, DR or any weighted point estimate — Kish ESS per arm equals n_a exactly; there is
  no overlap to weight over.
- A friction POINT estimate — MDE is 11.4pp on the best-powered arm pair; bounds only (skill 42).
- Presenting the re-pricing as a causal saving — it is arithmetic on observed prefixes.
- Pricing `naive_tok` — 22,631,879 is 1/14.79 of the reconstructed gross.
- Reusing the logged 92.0%/8.0% cache split for the routed policy — a switch resets the cache.
- Counting savings on trajectories with `supported = false` in `results/routes.jsonl`.
- An LLM judge over the final message as the outcome — 623/762 say "done", 8 admit any problem.

## 5. Postconditions
- `results/estimates.json` written; every number mirrored into `results/claims.json`.
- `91-verify-number` re-runs and fails any prose number absent from claims.json.
- 42, 43, 95 unblock.

## 6. Escalate when
- Any Kish ESS differs from n_a — that would mean a learnable logging policy exists and the whole
  estimation strategy changes; verify before believing it.
- The routable set or its 59.0% share moves.
- Someone requires a single causal saving number. The honest answer is a rho-indexed range plus
  the bounds from 42; refuse the point estimate.
