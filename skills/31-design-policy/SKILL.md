---
name: 31-design-policy
description: Define the routing policy, apply its gates, and write results/routes.jsonl with logged vs routed cost per trajectory. Invoke after pricing and the predictor, before off-policy estimation or the frontier.
---

# 31-design-policy

Applies the gate chain that defines the routable set, then prices logged vs routed cost per
trajectory. Writes `results/routes.jsonl`.

## 1. Preconditions
- `00-premise-freeze` read (F4: no point estimate of the effect; this skill produces the POLICY
  and its cost, not its causal benefit).
- `results/recon.jsonl`, `results/features.npz`, `results/jobkey.jsonl` exist; `p_fric` from
  30-fit-predictor available; the price sheet from 20-price-and-cache is named and fixed.
- `90-leak-audit` green: no gate may read a post-treatment field.

## 2. Procedure
1. Gate 1 — family. Only the claude lane is routable. The gpt lane (245 trajectories) has no
   cheaper in-family sibling in this export; excluded.
2. Gate 2 — admissible logged arm. The logged arm must be one this policy is willing to move
   from. 23 trajectories sit on non-admissible arms (opus-4-6 n=2, sonnet-4-6 n=1, and the rest);
   excluded.
3. Gate 3 — no pre-treatment image. 9 trajectories carry an `input_image` part in
   `items[:first_user_msg+1]`; excluded (the cheap sibling's vision behaviour is unobserved here).
4. Gate 4 — not already cheapest. 71 trajectories already run on claude-fable-5; excluded.
5. Survivors = the routable set. Route each to the cheap sibling unless `p_fric` exceeds the
   threshold tau; record `tau_pass`. `route` = the chosen arm, `changed` = `route != logged`.
6. `supported` = True iff the trajectory's job key has at least one observed run on the routed
   arm (from `results/jobkey.jsonl`). Unsupported routes may be counted in cost but never in
   any outcome claim.
7. `cost_logged_usd` and `cost_routed_usd` from 20-price-and-cache, same sheet, same rho, both
   labelled estimated.
8. Write per line: `idx, logged, route, changed, p_fric,
   gates{family, vision, admissible[], sign_stable[], tau_pass}, cost_logged_usd,
   cost_routed_usd, supported`.
9. `python -m router.policy` prints acceptance, expected vs actual, non-zero on mismatch.

## 3. Acceptance
- Routable set = 675/1000 trajectories = 197,551,096 gross tok = 59.0% of the bill.
- Exclusions, by count and by share of gross: gpt lane 245 (32.3%); non-admissible logged arm
  23 (7.7%); already-cheapest (fable-5) 71 (7.0%); pre-treatment images 9 (1.7%).
- 245 + 23 + 71 + 9 + 675 = 1000; 32.3 + 7.7 + 7.0 + 1.7 + 59.0 = 100.0.
- 1000 lines out, `idx` ascending 0..999; every excluded trajectory has `changed = false`.
- `gates.sign_stable` lists only columns sign-stable across all folds in 30-fit-predictor.
- Every dollar field names the assumed sheet in `results/claims.json`.

## 4. Banned moves
- Routing on any feature from `input[first_user_msg+1:]` — 46.5% of tokens are model-generated;
  the policy would read the model's own behaviour.
- Routing across families — family is perfectly separated by the toolset (245/755, 0 exceptions);
  a cross-family route changes the tool contract, not just the price.
- Routing a trajectory with a pre-treatment image (9 of them) — no observed vision behaviour on
  the cheap sibling.
- Routing trajectories already on claude-fable-5 (71) — there is nothing cheaper to move to.
- Claiming a saving on unsupported routes — `supported = false` means no observed run of that
  job on that arm.
- Reporting a friction delta here — that is 42-partial-identification's job, and it is bounds only.
- IPS/DR reweighting to "correct" the routed cost — Kish ESS per arm equals n_a exactly.

## 5. Postconditions
- `results/routes.jsonl` written, 1000 lines.
- `90-leak-audit` and `91-verify-number` re-run.
- 40, 41, 42, 95 unblock.

## 6. Escalate when
- The routable set moves off 675 / 197,551,096 tok / 59.0% — a gate changed and every downstream
  saving claim is stale.
- The four exclusion counts do not sum with the routable set to 1000.
- Someone asks to drop a gate to raise coverage. Coverage is 59.0% of the bill by construction;
  raising it means routing cases the export cannot support.
