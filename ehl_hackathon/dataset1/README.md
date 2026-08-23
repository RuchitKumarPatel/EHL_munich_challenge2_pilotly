# dataset1

A synthetic router-evaluation dataset that covers the scenario variations
missing from the organizer's redacted export (`viktor-tumai-starter/export/trajectories_v1_00.jsonl`),
while staying schema-identical to it so both `project-method1` and
`project-method2` can load it with zero code changes.

## Layout

- `export/trajectories_v1_00.jsonl` — raw requests. Each line is exactly
  `{"model": str, "input": [...], "tools": [...]}`, matching
  `method1.data.schema.RequestRecord` and `method2.data.load_trajectories`
  strictly (no extra keys allowed). One line per LLM call; each call's
  `input` contains the full growing history, same convention as the real
  export — trajectories are reconstructed by grouping on the opening
  system+first-user messages and ordering by history length.
- `scenario_manifest.json` — one entry per trajectory with the rich labels
  the raw schema can't carry (task type, difficulty, domain, failure/
  cache-break/branching/multi-turn flags, synthetic ground-truth quality,
  per-call latency). Keyed by `key` = the exact SHA-256 hash both pipelines'
  loaders derive from the raw export, so it joins directly onto
  `load_trajectories(...)` output without touching either pipeline's code.
- `manifest.json` — summary counts (like `project-method1/data/manifests/dataset.json`).
- `scripts/generate_dataset1.py` — the generator; rerun with `--seed` to
  reproduce, or `--n-per-combo` to scale up.

## Scale

150 tasks / 1203 requests, hash-split (same split fn as both methods) into
**87 train / 32 validation / 31 test** trajectories — vs. 18/4/3 in the
original 25-trajectory export. A 3-trajectory test set can't support any
meaningful cost-quality comparison; this one can.

## Coverage vs. the gaps identified in the original dataset

| Gap in original export | How dataset1 covers it |
|---|---|
| 1 task template for all 25 trajectories | 5 task types × verb/tool templates: `qa_reasoning`, `code_generation`, `summarization_extraction`, `multi_step_planning`, `creative_writing` (30 each) |
| No difficulty gradient (only call-count varied) | 3 explicit difficulty tiers (`easy`/`medium`/`hard`, 50 each) controlling call count, filler size, and a target quality center |
| 1 tool declared/used (`run_shell`; `file_read` never called) | 5 tool defs (`run_shell`, `file_read`, `web_search`, `code_exec`, `retrieval`); each task type uses a 2–3 tool subset actually invoked in the trace |
| 0 failures across all 153 calls | 51/150 trajectories inject a `step-N error: ...` output plus an identical-argument retry (drives the `error_recovery`/`non_repetition` signals both methods already compute) |
| No cache-break / mid-session model switch | 22/150 trajectories switch the logged `model` partway through (reconstructed as `logged_model == "mixed"` by both loaders) — the scenario the problem statement calls out explicitly |
| No branching / conditional remediation | 34/150 trajectories emit a `step-N warn: ...` output followed by 2–4 extra remediation calls |
| No multi-turn user interaction | 41/150 trajectories insert a mid-session user follow-up/clarification message |
| No recoverable outcome signal at all | 31/150 trajectories ("graded") end with an `assistant` message containing `quality_score=X.XX`, mirrored in the manifest as `ground_truth_quality`; the other 119 stay ungraded (`null`), preserving the honest off-policy-evaluation problem for the rest |
| 1 boilerplate domain/context (`<PERSON_A>`/`<COMPANY_A>`) | 5 domains (engineering/finance/support/sales/data_ops) × 5 project-name variants, 30 trajectories per domain |
| No timing/latency data | `latency_ms_per_call` + `total_latency_ms` per trajectory in the manifest, synthesized from token size × a per-model speed factor + jitter |
| 3-trajectory test split | 31-trajectory test split (see Scale above) |

## Validated against both pipelines

```
project-method1: load_trajectories('dataset1/export') -> 150 trajectories,
  22 with logged_model == "mixed", split 87/32/31, manifest keys match
  reconstructed trajectory keys exactly (0 missing, 0 extra).
project-method2: load_trajectories('dataset1/export') -> 150 trajectories,
  split 87/32/31 (train/calibration/test).
project-method1/scripts/build_trajectories.py and create_quality_labels.py
  both run against dataset1/export unmodified.
```

## Regenerating

```
cd dataset1
python scripts/generate_dataset1.py --out . --n-per-combo 2 --seed 13
```

Content is synthetic filler (no PII, no real task semantics) — do not draw
task-quality conclusions from the literal text, only from the structural
signals (call count, tool mix, failures, branching, cache overlap, model
identity) the way both methods' feature extractors already do.
