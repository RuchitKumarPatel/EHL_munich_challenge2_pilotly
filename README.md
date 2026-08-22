# Viktor Challenge Starter — Build the Router

Starter kit for the **Viktor Challenge** at the TUM.ai hackathon (Munich, 22–23 Aug 2026).
From real LLM-request logs, build a router that picks the right model for every call —
then prove it works, even though the log shows only the model that ran, and no outputs or token counts.

## Quick start (5 minutes)

```bash
# 1. No dataset yet? Generate a synthetic sample with the same shape:
python scripts/make_synthetic_sample.py            # writes ./export/

# 2. Got the real dataset links (shipped at kickoff)? Then instead: the export ships
#    as trajectories_v1_<index>.jsonl.tar.gz archives — download, verify the posted
#    SHA-256, then:  mkdir -p export && tar xzf trajectories_v1_01.jsonl.tar.gz -C export/

# 3. Sanity-check the export, reconstruct trajectories, print stats:
python scripts/load_trajectories.py export/

# 4. Run the baseline heuristic router + cache-aware cost report:
python scripts/baseline_router.py export/

# 5. Turn results into a cost–quality frontier CSV (+ PNG if matplotlib is installed):
python scripts/plot_frontier.py results/routes.jsonl

# 6. Build the local explorer site and open it:
python scripts/build_site_data.py export/         # writes site/data.js
xdg-open site/index.html                          # macOS: open site/index.html
```

Python 3.10+, standard library only (matplotlib optional for the PNG).

## The explorer site

`site/index.html` is a self-contained local viewer for the export — no build step, no server,
no network. It reads `site/data.js`, which `scripts/build_site_data.py` regenerates from
`export/` and `results/routes.jsonl`. Tool arguments and outputs are unwrapped from their JSON
encoding on the way in, so the item timeline reads as text rather than one escaped line; pass
`--pretty` to indent `data.js` itself when you want to read or grep it by hand.

Four tabs:

- **Übersicht** — requests, cost and token distributions per model, tool usage, and a
  diagnostic panel on how much trajectory structure the reconstruction actually recovers
- **Trajektorien** — filter and search all reconstructed trajectories; drill into the call
  sequence and the full item timeline of the sampled ones
- **Cache-Trap** — shared-prefix share per call position, and what switching models at call *i*
  costs on a sampled trajectory
- **Frontier** — the cost–quality curve over adoption share, with its placeholder-quality
  weakness stated on the chart

The site embeds redacted excerpts of the dataset. `site/data.js` is gitignored — keep it local,
do not publish or upload the rendered page.

## Using a coding agent

Point Claude Code / Codex / Cursor / opencode at this repo — `AGENTS.md` briefs your agent.
In Claude Code you also get slash commands:

- `/setup` — set up everything needed to participate
- `/make-presentation` — build a Viktor-branded presentation of your solution
- `/prepare-submission` — package your solution into a formal submission

## What's here

| Path | What |
|---|---|
| `AGENTS.md` | Agent briefing: dataset shape, the cache trap, judging, starter ideas |
| `skills/` | The three guided workflows above (plain Markdown, readable by humans too) |
| `scripts/` | Loader + trajectory reconstruction, baseline router, cache-aware cost model (estimated tokens), frontier plot, synthetic sample, site data builder |
| `site/` | Local explorer site (`index.html`, plus the gitignored `data.js` it reads) |
| `templates/presentation.html` | Self-contained branded slide template |

## Rules that matter

- **License:** challenge use only — no redistribution of the dataset. Full terms ship with the download.
- No GPU or API keys needed. Judge-model rescoring is allowed (credits announced at kickoff).
- Questions → the challenge Discord; the Viktor team answers there all weekend.
