# POSTMORTEM — fine-tuning a pretrained text classifier for the router

Branch `text-classifier-finetune`. Outcome: **negative, adopted nothing**. `main` is unchanged.
The verdict and its numbers live in ADR-008; this file records what went *wrong on the way there*,
so the next person does not pay for it twice.

Total cost: ~3 h of CPU (51.5 min frozen encoding, 39 min fine-tune round 1, 89.6 min round 2)
plus the environment fights below.

Ordered by how much time each one cost, worst first.

---

## 1. The arm head silently learned nothing, and the null result looked real

**Symptom.** Round 1 finished cleanly and reported `Delta_hat(sonnet-opus): mean -0.0001,
sd 0.0000`. Read at face value that says *"the data contains no arm-dependent signal"* — a
publishable-sounding conclusion, and the wrong one.

**Cause.** The friction head was `nn.Sequential(nn.Linear(768+16,128), GELU, nn.Linear(128,1))`
fed `cat([h, arm_emb])`. The arm was 16 of 784 input dims, randomly initialised at scale 0.02,
competing with a 768-dim text representation. The optimiser ignored it outright:
`max|p̂_sonnet − p̂_opus| = 1.6e-4` across every held-out row.

**How it was caught.** Not by the loss — by checking whether the target was learnable at all
*before* believing the null. The claude-cron population has sonnet − opus = **+5.4pp** unadjusted
and **+6.8pp** stratum-adjusted (11 strata). A model reporting a 0.0pp arm effect against a
+6.8pp design-based contrast is broken, not informative.

**Fix.** Give the arm its own unmixed path:

```
friction_logit = text_mlp(h)  +  arm_intercept[a]  +  (proj(h) · arm_loading[a])
                 main effect     arm main effect      explicit text x arm interaction
```

with `arm_loading` initialised at scale 0.5, not 0.02. After the fix `Delta_hat` has
mean +0.104, sd 0.114 — still noise (`corr` with the real stratum gap = **+0.042**), but now
*measurably* noise instead of degenerate.

**Rule.** A head whose output does not vary with an input is not evidence about that input.
Before reporting any null, print the spread of the prediction and compare it against a
design-based estimate of the same quantity.

---

## 2. Undertrained heads are indistinguishable from absent signal, if you only look at R²

**Symptom.** Round 1 spend head: held-out R² = +0.091. Easy to read as "the transformer cannot
predict spend".

**Cause.** Underfit, not absence of signal. `LR=2e-5` uniformly (trunk *and* heads), 2 epochs,
116 optimiser steps, batch 4.

**How it was caught.** Comparing the *spread* of the prediction to the spread of the target:
prediction sd was **30%** of target sd (0.172 vs 0.577) with corr 0.344 — the classic signature of
a head collapsed toward the mean. R² alone does not separate "no signal" from "not yet trained".

**Fix.** Separate parameter groups (trunk 3e-5, heads 1e-3), 5 epochs, `OneCycleLR` with
per-group `max_lr`. R² moved 0.091 → 0.150 for a 4x compute increase — **still underfit**, which is
itself the honest finding: the spend result is compute-limited and should be labelled as such.
The Δ result is *not* compute-limited; it is data-limited (44 training positives).

---

## 3. Weights were unreachable, and the library lied about why

Three separate failures stacked:

| step | symptom | cause | fix |
|---|---|---|---|
| `pip` | `No module named pip` in `.venv` | venv built without pip | `python -m ensurepip --upgrade` → pip 26.1.2 |
| `huggingface.co` | HTTP **000** (PyPI was **200**) | host blocked on this network | mirror: `hf-mirror.com` serves `config.json` *and* `model.safetensors`, both 200 |
| `snapshot_download` | `LocalEntryNotFoundError`, then `FileMetadataError: Distant resource does not seem to be on huggingface.co` — **with `HF_ENDPOINT` correctly set** (`constants.ENDPOINT` confirmed as the mirror) | (a) `huggingface_hub` 1.28 routes through Xet storage; (b) even with `HF_HUB_DISABLE_XET=1` it follows an internal redirect to `cdn-lfs.huggingface.co`, which is **also** blocked | bypass the library — `curl` each file from the mirror, then `from_pretrained(local_dir)` |

The error message names `huggingface.co` even when the endpoint is set to something else, which
sends you looking for a config bug that is not there. Verify with `curl` against the mirror
before debugging your own configuration.

```
for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -sL -o "$MODELDIR/$f" "https://hf-mirror.com/answerdotai/ModernBERT-base/resolve/main/$f"
done
```

---

## 4. The CPU budget decided the experiment design, not the science

Measured on this laptop, 12 threads, no GPU, ModernBERT-base (149M):

| shape | forward | forward+backward |
|---|---|---|
| L=512, B=8 | 10.1 s (0.8 row/s) | 31.9 s (**0.25 row/s**) |
| L=1024, B=8 | 26.8 s (0.3 row/s) | 99.9 s (0.08 row/s) |
| L=2048, B=4 | 29.2 s (0.14 row/s) | 125.9 s (0.03 row/s) |

Consequences that are *design compromises, not findings*:

- Fine-tune ran at **L=512**. The user message alone is **6771 tokens** on a typical row, so the
  model saw ~7.5% of the input (head 255 + tail 255 tokens). A GPU run at L=4096 is a different
  experiment and may well beat the hashed n-grams on the cost axis.
- Only **one** grouped split was fine-tuned (train 229 / test 329). Everywhere else in this repo
  the rule is 24 splits, because a single split is a lottery. The effect sizes here were large
  (2.5x on the money), but a single split remains the weakest part of the evidence.
- Frozen encoding and fine-tuning had to be serialised — both saturate all 12 threads, so running
  them concurrently halves each.

---

## 5. The pre-treatment text is very nearly a row ID

**Risk.** 871 distinct system-prompt hashes over 1000 rows (ADR-006). Redaction placeholders
carry a per-request serial, so the same entity is numbered differently in every row it appears
in -- two different serials of the same placeholder family are one entity, not two. Raw
un-redacted vendor identifiers (opaque mixed-alphanumeric workspace, user and bot ids) also
survive inside the memory content; they are quoted nowhere in this repo, because each one that
was checked occurs in `export/` and is therefore licensed data rather than an invented example.
Any model fed the raw text memorises rows instead of learning the task.

**Mitigation applied to every text pipeline here:**

```
PII_([A-Z]+)_\d+   ->  pii\1        # collapse the per-request numbering
\b[A-Za-z0-9]{8,}\b (mixed alnum) -> <id>
\d+                ->  #
lowercase
```

plus grouped CV by literal cron path everywhere, so a job never straddles a train/test boundary.

---

## 6. Frozen masked-LM pooling is a bad baseline and it is not the model's fault

Mean-pooled frozen ModernBERT-base (L=1024) scored spend R² **0.367** / capture@100 **78.2%**
against **0.566** / **92.2%** for hashed normalised word 1-2-grams. Blending toward the transformer
degraded capture monotonically.

ModernBERT-base is a **masked LM**, not a sentence encoder — its mean-pooled representation was
never trained for linear separability. And the routing signal here is the *presence of specific
tokens* (`silent`, `nothing`, `empty` → cheap; `search`, `fresh`, `review`, `sources` → expensive),
which bag-of-words reads directly and a 768-dim average destroys. If a frozen embedding baseline
is wanted, use a model trained for embeddings; do not read "frozen BERT is weak" as
"transformers are weak here".

---

## 7. The cross-cutting trap: nuisance parameters fitted on the full sample

Not specific to this branch, but it bit **three times** in one session and it is the single
highest-yield thing to remember.

| mechanism | with full-sample nuisances | refit honestly per split |
|---|---|---|
| per-pair risk weights δ | **+11.6% to +21.8%** money at matched risk | **−45% to −56%** |
| per-cell Δ selection | looked competitive | richer in **0/24** splits |
| fine-tuned Δ̂ weighting | — | **−54%** of the money |

The per-pair case is the clearest illustration: `opus-4-8 → sonnet-5` has a design-based risk of
**+0.0419 ± 0.0453** on a half sample — the standard deviation exceeds the mean. Ranking by
`saving / δ̂` then divides by a divisor that is mostly noise, and pulls in exactly the moves that
should have been left alone.

**Rule.** Anything fitted on outcomes — cell rates, pair constants, calibration scales, a
fine-tuned network — must be refit *inside* the split it is evaluated on. Reusing a full-sample
constant is not a small approximation; it reverses the sign of the conclusion.

---

## 8. `results/` was rebuilt underneath a running analysis

Mid-session an unrelated `make all` (PID 579498) deleted and regenerated `results/`, removing
`routes.jsonl` while the analysis harness had it open. The harness died with `FileNotFoundError`.

**Fix.** Snapshot everything the analysis needs into the scratchpad once, and read the snapshot —
never the live `results/` tree — for the duration of an experiment. `results/` is generated and
is not a stable interface.

---

## What would actually be worth trying next

Only one of these is about the model:

1. **GPU, L=4096, more epochs, several grouped splits.** The spend head is demonstrably underfit;
   this is the one open question. It targets the **cost** axis, where headroom still exists
   (the current predictor captures ~93% of oracle dollars).
2. **An embedding-trained encoder** rather than a masked LM, if a frozen baseline is wanted.
3. **Nothing on the risk axis.** Within the claude lane the arm effect is +6.1pp against an MDE of
   8.2pp. Five mechanisms have now failed out-of-sample there — per-cell Δ, per-pair δ,
   job-history features, hashed-n-gram p̂, and a fine-tuned Δ̂ head. The constraint is the
   sample, not the model class, and more capacity makes it worse rather than better: the
   fine-tuned head invented a Δ̂ range of [−0.253, +0.638] where the real stratum gap spans
   [+0.046, +0.157].
