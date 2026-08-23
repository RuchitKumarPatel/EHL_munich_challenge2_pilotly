# dataset2

A router-evaluation corpus built so that a routing methodology's claims are
actually **falsifiable**. dataset1 could not distinguish a real router from
"always pick the cheapest model"; dataset2 can, and it says why.

Schema-identical to the organizer's export and to dataset1 — every JSONL line is
exactly `{"model", "input", "tools"}` — so `project-method1`, `project-method2`
and `project-method3` all load it with zero code changes (verified: all three
reconstruct the same 756 trajectories, 60 mixed-model, identical splits).

```
python scripts/generate_dataset2.py --out .. --seed 20260823   # regenerate
python scripts/validate_dataset2.py ..                          # 42 invariants, exit 0
```

## Why dataset1 needed replacing

Two flaws were confirmed **empirically** against dataset1 before any of this was
designed. They are not stylistic complaints; each one structurally prevented a
routing method from being evaluated at all.

**1. Zero model complementarity.** dataset1 drew `ground_truth_quality` from
*difficulty alone* — the model that ran the task never entered the formula. So no
model was genuinely better at anything, and no router could beat picking the
cheapest model, because there was nothing to exploit. This is exactly what
happened: project-method3 on dataset1 produced a policy **byte-identical to
`cheapest`** (same cost `0.00002268`, same quality `0.8431`). The dataset also
reported the incoherent ordering `cheapest (0.843) > strongest (0.830)`.

**2. Uniform-random logging policy.** dataset1 chose the logged model with
`rng.choice(MODELS)` — independent of context, i.e. MCAR. A fitted propensity
model has literally nothing to learn from that, so project-method3's
`LoggingPolicyModel` and its doubly-robust correction could never demonstrate any
advantage over a context-free frequency baseline. There was no selection bias to
correct.

Model complementarity is the entire premise of routing (routing survey
arXiv:2603.04445; "When Does Combining Language Models Help?" arXiv:2606.27288):
no single model is best on every input, and a well-matched cheap specialist
routinely beats a frontier generalist on its specialty. dataset1 had none of it.

## The result

| | dataset1 (31 test) | dataset2 (159 test) |
|---|---|---|
| `cheapest` quality | 0.8431 | 0.7786 |
| `method3` quality | 0.8431 — *identical to cheapest* | **0.8174** |
| `strongest` quality | 0.8304 (< cheapest — incoherent) | 0.8511 |
| method3 vs cheapest | same cost, **same quality** | same cost, **+0.039 quality**, CIs disjoint |
| per-model conformal calibrators certified | **0 / 7** | **7 / 8** |
| Pareto frontier points | 1 | 4 |

On dataset2, method3 reaches the *same cost* as `cheapest` while scoring `0.8174`
vs `0.7786`, with non-overlapping bootstrap CIs (`[0.816, 0.819]` vs
`[0.764, 0.793]`) — it routes between *equally cheap* models by task type,
exploiting the designed complementarity. That is a real, measurable routing win
that dataset1 was structurally incapable of showing.

## What's in dataset2

**Scale.** 756 trajectories / 4562 requests (dataset1: 150 / 1203; organizer: 25 /
153). 2026 practice guidance is ≥500 cases before trusting aggregate metrics.
Test split is 159 trajectories, which is also what fixed the calibration-data
starvation that left every per-model conformal calibrator uncertified on dataset1.

**1. Model complementarity** (`scripts/skill_matrix.py`) — a latent
per-`(model, task_type)` skill matrix, deliberately **non-monotone in price**:
- `claude-fable-5` (cheapest) is the *summarization specialist* and beats
  `claude-opus-5` there at 1/50th the price;
- `claude-sonnet-5` beats both Opus models at code generation at 1/5th the price;
- `gpt-5.6-terra` owns multi-step planning; `gpt-5.6-sol` owns creative writing;
- `claude-opus-5` remains the best generalist and clearly wins hard reasoning.

Weak models also degrade far faster with difficulty, so *difficulty*-aware routing
pays on top of *task*-aware routing. An oracle router beats **both** extremes on
quality **and** cost (measured: routed `0.863` @ price `8.38` vs strongest `0.859`
@ `15.00`).

**2. Context-dependent logging policy across three eras**
(`scripts/logging_policy.py`), replacing dataset1's uniform choice:
- `heuristic_2025h1` — near-deterministic hand rule. **Weak overlap** on purpose
  (min propensity `0.028`).
- `explore_2025h2` — an exploration/bandit period. **Strong overlap**
  (min propensity `0.110`), so this era alone supports clean OPE.
- `cost_push_2026h1` — cost-cutting mandate skews to cheap models, prices drop
  20%, and a new model enters the pool.

This gives real selection bias to correct, plus a way to measure an estimator's
sensitivity to positivity (cf. "Logging Policy Design for OPE" arXiv:2605.15108
on the reward–coverage tradeoff; arXiv:2603.21485 on deterministic-logging bias).

**3. True propensities recorded per trajectory** (`true_propensity`,
`true_propensity_all`) — an *estimated* propensity can be scored against the real
one. No real-world dataset can offer this, and dataset1 did not either.

**4. A planted positivity violation** — `claude-opus-5` is never logged for
`(summarization_extraction, easy)` in any era. A method should **detect and report**
that this cell has no support rather than silently extrapolating into it. Flagged
per row as `positivity_violation`.

**5. Temporal order and drift** — `timestep` and `era_index` on every row; the task
mix genuinely shifts (`data_analysis` goes 2% → 28% across eras).

**6. An unseen model** — `gpt-5.7-nova` appears **only in the final era**, so a
router trained on earlier data is tested on generalization to a model that was
never in its historical pool.

**7. An OOD domain** — `legal` appears only in the final era.

**8. Replicate groups** — 30 groups × 4 runs of an *identical* spec, so
reproducible specialist advantage can be separated from single-draw label noise
(cf. arXiv:2607.03436, which shows this noise inflates apparent router-to-oracle
gaps).

**9. Real cache breaks** —
- `context_compaction` (53): history is compacted mid-session, genuinely
  rewriting the prefix so cache credit is measurably lost (reuse drops from 1.00
  to ~0.6–0.8 at the compaction step);
- `tool_schema_change` (50) and `tool_reorder` (46): the documented
  production cache killers;
- `model_switch` (60): mid-trajectory model change.

**10. Heavy-tailed distributions** — log-normal call counts (1 / median 5 / max 40)
instead of dataset1's flat uniform, plus skewed model usage.

**11. Edge cases** — 6 each of `single_call`, `no_tools`, `oversized_prompt`,
`unicode_heavy`, `long_horizon`, `all_failure`.

**12. Honest label sparsity** — only ~52% of trajectories carry a recoverable
`ground_truth_quality`; the rest preserve the original challenge's "final output
is missing" problem rather than pretending it away.

## Two subtleties worth knowing

**The opening is frozen; cache breaks happen after it.** All three loaders group
requests into trajectories by hashing the opening system + first-user messages, so
mutating those mid-session would silently split one trajectory into several.
Cache-breaking scenarios therefore mutate history *after* the opening, or mutate
the `tools` block — which is what actually breaks caches in production anyway.

**History length stays monotone.** The export's premise is that each request's
input contains every item of the previous one, and all three loaders order a
trajectory's calls by serialized length on that basis. Naive compaction *shrinks*
history and would scramble reconstructed call order — a bug caught during
development. The compaction summary is padded so the prefix genuinely changes
(cache lost) while total length still grows (ordering premise preserved). The
validator checks this directly.

## Files

- `export/trajectories_v1_00.jsonl` — the corpus.
- `scenario_manifest.json` — per-trajectory labels, scenario flags, true
  propensities, counterfactual expected quality for every candidate model, oracle
  choices, latency, replicate bookkeeping. Keyed by the same SHA-256 all three
  loaders derive, so it joins directly onto their output.
- `manifest.json` — coverage summary.
- `scripts/skill_matrix.py` — the latent ground truth (complementarity).
- `scripts/logging_policy.py` — the three-era logging policies.
- `scripts/generate_dataset2.py` — generator.
- `scripts/validate_dataset2.py` — **42 invariants**, exit 0 = all hold.

## A note on the validator

Two of its checks initially failed, and both turned out to be bugs *in the checks*,
not in the data — worth recording because the second one is the exact trap the
dataset exists to pose:

- The compaction check used an absolute reuse threshold. But the frozen
  system+user opening is itself a large share of tokens, so even total
  post-opening invalidation leaves high retained reuse. Corrected to detect the
  drop off the 1.00 full-reuse baseline.
- The complementarity check pooled quality per model **across difficulties**. That
  is confounded: the logging policy sends hard work to frontier models, so each
  model faces a different difficulty mix. Pooled means made `claude-opus-5` look
  like the summarization winner even though `claude-fable-5` beats it at both easy
  (0.96 vs 0.85) and medium. Corrected to compare within
  `(task_type, difficulty)` cells. **A method evaluated on dataset2 must make the
  same adjustment** — that confounding is deliberate.
