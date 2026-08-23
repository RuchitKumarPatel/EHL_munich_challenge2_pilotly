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
Weights are NOT on huggingface.co from this network (`000`); the mirror serves them:

```
for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -sL -o "$MODELDIR/$f" "https://hf-mirror.com/answerdotai/ModernBERT-base/resolve/main/$f"
done
MAXLEN=512 EPOCHS=5 BS=4 LR=3e-5 HEAD_LR=1e-3 SPLIT=0 .venv/bin/python router/textclf/finetune.py
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
