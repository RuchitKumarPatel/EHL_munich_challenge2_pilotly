"""Viktor Challenge router — offline cost/friction router over 1000 logged agent trajectories.

Pipeline order (each module runs as `python -m router.<name>` and prints its acceptance
check; `make all` runs the whole chain in this order):

    1. recon     reconstruct turns and the cache split       -> results/recon.jsonl
    2. labels    tool-output outcomes and friction labels     -> results/labels.jsonl
    3. jobkey    literal cron-path job key                    -> results/jobkey.jsonl
    4. features  pre-treatment feature matrix                 -> results/features.npz
                                                              -> results/feature_manifest.json
    5. costs     cache-aware price model, effective multipliers
    6. gates     admissibility gates (family, vision, sign stability)
    7. model     friction predictor, pre-treatment features only
    8. ope       off-policy evaluation — bounds, never point estimates
    9. figs      figures, including the cost-quality frontier
   10. report    -> results/metrics.json, results/claims.json, the written report

Stages 1-3 are independent of each other and depend only on export/. Stage 4 consumes 1-3,
stage 5 consumes 1, stages 6-7 consume 4 (and 2 for the label), stage 8 consumes 5-7, and
stages 9-10 consume everything.

Artifact field names and the pinned acceptance numbers are in docs/CONTRACTS.md — that file,
not this docstring, is the contract. Standing constraints: stdlib + numpy + matplotlib only
(sklearn and scipy are not installed and are banned), no network, no API keys, no global state.
All token counts are ESTIMATES: tok(x) = len(json.dumps(x)) // 4; the export has no usage field.
"""
