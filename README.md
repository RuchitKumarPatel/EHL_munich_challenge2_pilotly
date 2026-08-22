# Viktor Challenge Router

This repo contains our router and offline evaluation for the Viktor / TUM.ai agent-routing challenge.

The main result is a **cache-aware segment router with an explicit switch penalty**. It uses a gated trajectory-level router to propose a cheaper model, then decides per call whether switching is still worth it after accounting for prefix-cache loss.

## Headline Result

On the reconstructed export currently in `export/`:

| Policy | Estimated Cost | Cost Delta | Burden Delta | Changed Trajectories |
|---|---:|---:|---:|---:|
| Original logged routing | `$76.93` | `0.00%` | `0.0000` | `0` |
| Gated router `lambda=0.20` | `$56.04` | `-27.15%` | `-0.0089` | `340` |
| Cache-aware segment router | `$55.79` | `-27.48%` | `-0.0089` | `315` |
| Classifier + cache segment | `$55.16` | `-28.30%` | `-0.0070` | `377` |

Presentation claim:

> Cache-aware segment routing reduces estimated cost by **27.5%** while keeping the continuation-burden proxy non-worse.

The classifier experiment gives a slightly lower offline cost, but it is less defensible as the main policy because it learns proxy labels from the gated router and matches those labels only `48.67%` of the time. Use it as an extension, not the headline claim.

## Policy Summary

The final recommended policy is:

```text
1. Reconstruct trajectories from the JSONL export.
2. Look only at the first request of each trajectory.
3. Extract initial features:
   prompt/template words, task type, token bucket, tool signature,
   complex-tool bucket, image flag, and logged model family.
4. Use cross-fitted neighbor estimates to propose a cheaper same-family model.
5. Apply quality guards:
   burden mean must be non-worse,
   burden UCB must be non-worse,
   expected cost must improve.
6. Run a cache-aware segment optimizer:
   for each call, keep the logged model or use the proposed model.
7. Charge a 2,000-token switch penalty whenever the segment route changes model.
8. Switch only when cache-aware savings survive that penalty.
```

Why this matters: provider prefix caching couples calls inside a trajectory. A route that looks cheap per call can become expensive if it switches models and loses cached-prefix value.

## Reproduce

Use Python 3.10+; the main pipeline uses only the standard library.

```bash
# 1. Sanity-check and reconstruct trajectories
python3 scripts/load_trajectories.py export/

# 2. Generate gated trajectory-level proposals
python3 scripts/template_bandit_router.py export \
  --selection-mode gated-score \
  --same-family \
  --min-similarity 0.30 \
  --lambdas 0,0.05,0.1,0.2,0.35,0.5,0.8,1.2 \
  --output-prefix current_gated_sim030

# 3. Export lambda=0.20 trajectory/call details
python3 scripts/export_gated_trajectories.py export \
  --routes results/current_gated_sim030_routes.jsonl \
  --lambda-value 0.2 \
  --out-prefix results/current_gated_sim030_lambda02

# 4. Apply cache-aware segment router with switch penalty
python3 scripts/cache_segment_router.py export \
  --routes results/current_gated_sim030_routes.jsonl \
  --lambda-value 0.2 \
  --switch-penalty-tokens 2000 \
  --out-prefix results/cache_segment_router

# 5. Optional: classifier proposal experiment
python3 scripts/classifier_router.py export \
  --routes results/current_gated_sim030_routes.jsonl \
  --lambda-value 0.2 \
  --switch-penalty-tokens 2000 \
  --out-prefix results/classifier_router

# 6. Generate presentation charts
python3 scripts/plot_policy_comparison.py
```

## Important Outputs

| Path | Purpose |
|---|---|
| `results/three_way_policy_comparison.svg` | Original vs gated vs cache-segment recommendation |
| `results/all_policy_comparison.svg` | Adds classifier experiment to the comparison |
| `results/cache_segment_router_aggregate.csv` | Main aggregate result for recommended policy |
| `results/cache_segment_router_summary.csv` | Trajectory-level segment-routing results |
| `results/cache_segment_router_calls.csv` | Call-level cache/switch/cost details |
| `results/classifier_router_aggregate.csv` | Optional classifier experiment aggregate |
| `scripts/pricing.json` | Pricing assumptions used by the cost model |

## Evaluation Notes

- There is no `usage` field in the export, so token counts are estimates.
- Output cost is not included because final outputs are missing and output token counts are not logged.
- Prefix-cache reuse is inferred from item-level shared prefixes.
- A model switch resets cache in the cost model.
- Quality is not directly observed. We use a **continuation-burden proxy** based on later calls, tool-error text, and wait/timeout signals.
- All router quality estimates are cross-fitted to avoid evaluating a trajectory on its own neighborhood.

## Files

| Path | What |
|---|---|
| `scripts/load_trajectories.py` | Reconstructs trajectories from JSONL requests |
| `scripts/cost_model.py` | Cache-aware cost model using estimated tokens |
| `scripts/template_bandit_router.py` | First-request neighbor/gated router |
| `scripts/export_gated_trajectories.py` | Exports trajectory and call-level details |
| `scripts/cache_segment_router.py` | Recommended cache-aware segment router |
| `scripts/classifier_router.py` | Optional cross-fitted classifier proposal experiment |
| `scripts/plot_policy_comparison.py` | Generates SVG/CSV comparison charts |

## Dataset Rule

The dataset is challenge-use only. Do not upload or redistribute the export files.
