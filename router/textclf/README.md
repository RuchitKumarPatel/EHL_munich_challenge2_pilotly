# textclf — pretrained text classifier for the router (branch: text-classifier-finetune)

Answers one question: **does a real pretrained transformer, fine-tuned on the pre-treatment
text, beat the hand-built features on either axis of the router?**

Axes, and the answer measured on this export:

| axis | what it would buy | verdict |
|---|---|---|
| cost (`spend` head) | better `ŝ` ⇒ more money at IDENTICAL risk | **no** — hashed n-grams win |
| risk (`Δ` head)     | arm-aware decision weighting             | **no** — Δ̂ is noise, and using it costs 54% of the money |

## Files
- `encode.py`   frozen ModernBERT-base mean-pooled embeddings of the pre-treatment slice (L=1024)
- `finetune.py` multi-task fine-tune: one trunk, spend head (arm-FREE) + friction head (arm-CONDITIONED)
- `probe.py`    frozen-embedding probe — the floor the fine-tune has to beat
- `evalpool.py` cron-restricted routing evaluation; **same friction estimand as the rest of the repo**

## What is legal to feed it (ADR-001)
`tools` + `input[0]` + `input[1]`, nothing else. The spend head never sees the arm: the arm is
the treatment and is unknown at routing time. The friction head is arm-conditioned on purpose —
that is the only way to read Δ̂ = p̂(sonnet) − p̂(logged) off a single forward pass.

## Reproducing

These four modules READ a scratch directory; they do not build it. Point them at
one with `TEXTCLF_SCRATCH`, which every module resolves before it imports
`transformers` and refuses to guess:

```
export TEXTCLF_SCRATCH=/path/to/scratchpad
MAXLEN=512 EPOCHS=5 BS=4 LR=3e-5 HEAD_LR=1e-3 SPLIT=0 .venv/bin/python router/textclf/finetune.py
```

### What that directory has to contain

Everything below is derived from the licensed export, so **none of it is in this
repository and none of it can be** — see rule 1 in `AGENTS.md`. This list is the
only record of what the run needed. `tests/test_textclf_repro.py` extracts the
paths from the source on every run and fails when this section drifts from them.

| path | what it is | who reads it |
|---|---|---|
| `models/modernbert-base` | the five weight files fetched below | `encode.py`, `finetune.py` |
| `X.npy`, `cols.json` | the 35-column tabular feature matrix and its column names | `evalpool.py`, `probe.py` |
| `Xtext.npy` | hashed n-gram features — the champion `probe.py` has to beat | `probe.py` |
| `tc/emb_modernbert_L1024.npy` | frozen embeddings, written by `encode.py` | `probe.py` |
| `contrast.py` | the modelled-both-arms estimand | `evalpool.py` |

`contrast.py` is itself the head of a chain that is **not** in this repo either:
`contrast.py` → `oof.py` → `mech.py` → `harness.py`, and `harness.py` reads
`flat.json` and `costmat.json` from the same directory. `harness.py` bottoms out
in this repo's own `router.ope`, `router.strata` and `router.pricing`, so the
estimand is the repo's; the four scratch modules are the plumbing that fed it.

**Honest limit:** that plumbing was never brought into the repo, so this path is
reproducible only by someone who still has the scratch directory. What IS
reproducible from the repo alone is the result the postmortem turns on — see
`docs/POSTMORTEM-textclf.md`, which states the measured verdict and its evidence.
Written outputs (`tc/textlen.json`, `tc/spend_bert.npy`, `tc/spend_hash.npy`,
`tc/ft_split*_L*.json`) land in the same directory and are not preconditions.

### Weights

Weights are NOT on huggingface.co from this network (`000`); the mirror serves them:

```
for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -sL -o "$TEXTCLF_SCRATCH/models/modernbert-base/$f" "https://hf-mirror.com/answerdotai/ModernBERT-base/resolve/main/$f"
done
```

`huggingface_hub.snapshot_download` does NOT work even with `HF_ENDPOINT` set: it follows an
internal redirect to `cdn-lfs.huggingface.co`, which is also blocked. Fetch the files with curl.

## Cost, measured on this laptop (12 CPU threads, no GPU)
- frozen encoding, 1000 rows @ L=1024: **51.5 min**
- fine-tune, 229 rows x 5 epochs @ L=512: **82.6 min** + 7 min inference

## The trap that cost a whole run
First revision put the arm in as 16 of 784 input dims to a 128-unit MLP. The model ignored it
outright: `max|p̂_sonnet − p̂_opus| = 1.6e-4`, Δ̂ sd = 0.0000. The head now gives the arm its own
unmixed path — text main effect + per-arm intercept + explicit text x arm interaction. After the
fix Δ̂ sd = 0.1143. It is still noise, but it is now *measurably* noise rather than degenerate.
