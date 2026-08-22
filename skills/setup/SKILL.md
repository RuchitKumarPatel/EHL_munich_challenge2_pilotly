---
name: setup
description: Set up everything needed to participate in the Viktor Challenge — environment, dataset, sanity checks, first baseline.
---

# /setup — get a team from zero to a running baseline

Do these steps in order. Announce each result in one line.

## 1. Environment
- Confirm Python 3.10+ (`python --version`). Everything here is standard library;
  `matplotlib` is optional (only for the frontier PNG).
- No GPU, no API keys needed.

## 2. Dataset
- **No dataset link yet?** `python scripts/make_synthetic_sample.py` writes a shape-identical
  synthetic sample to `./export/`. Build your pipeline on it.
- **Real dataset** — download part 01 directly (link valid through Sunday; more parts, if any, arrive in the challenge Discord):
  `curl -L -o trajectories_v1_01.jsonl.tar.gz '<PRESIGNED_URL_FROM_CHALLENGE_DISCORD>'`
  1. Extract into `./export/`: `mkdir -p export && tar xzf trajectories_v1_01.jsonl.tar.gz -C export/`
     (scripts read any `export/*.jsonl`; the archive also contains the LICENSE — read it).
  2. Challenge use only, no redistribution.
     **Never commit `export/` or the archives to git** (both are in `.gitignore` — keep them there).

## 3. Sanity checks
- `python scripts/load_trajectories.py export/` — prints request counts, per-model
  distribution, reconstructed trajectory stats, and a schema check.
- Open one line yourself and confirm: exactly `model`, `input`, `tools` — **no `output`,
  no `usage`**. Models are anonymized (families `claude-opus/-sonnet/-fable`, `gpt-5.6-*`,
  across generations — ~9 ids); PII shows as named placeholders
  (`<PERSON_ROBERT>`, `<PROJECT_NAME>`, `<COMPANY_A>`); images are redacted placeholders.

## 4. First baseline (the honest floor)
- `python scripts/baseline_router.py export/` — reconstructs trajectories, routes early
  small-prompt calls to a cheaper sibling, prices cache-aware on estimated tokens.
  Writes `results/routes.jsonl`.
- `python scripts/plot_frontier.py results/routes.jsonl` — frontier CSV (+ PNG if matplotlib).

## 5. Where the points are
Beating the baseline is expected. The judged depth: a constructed outcome signal (earlier
outputs are recoverable from later inputs), an honest off-policy estimate with named failure
modes, and cache-aware pricing on clearly-labeled estimated tokens. See AGENTS.md → "The twist".
