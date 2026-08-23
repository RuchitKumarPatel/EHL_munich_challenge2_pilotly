# CONTRACTS — the file every agent on this project reads first

This is the shared brief for the Viktor Challenge router. The two sections below are
transcribed **verbatim** from the team brief. Field names and numbers here are binding: other
agents' modules are written against them. If your code disagrees with this file, your code is
wrong until a human says otherwise. Do not edit a number here to make a test pass.

Repo: `/home/frans/Projekte/EHL_munich_challenge2_pilotly`
Python: `.venv/bin/python` (3.13.14). numpy 2.5.2 and matplotlib 3.11.1 ARE available.
sklearn and scipy are NOT installed and must NOT be used or installed. stdlib + numpy +
matplotlib only.

---

## ARTIFACT CONTRACTS (agents depend on these exactly — do not change field names)

`results/recon.jsonl` — one JSON object per line, in file order, key `"idx"` = 0-based line number:

```
{"idx": int, "model": str, "family": "claude"|"gpt", "n_turns": int,
 "turn_cuts": [int], "prefix_tokens": [int], "tools_tok": int, "pre_tok": int,
 "gross_tok": int, "cache_read_tok": int, "cache_write_tok": int, "naive_tok": int}
```

`pre_tok` = `tools_tok + sum(tok(items[:first_user_msg_index+1]))`
`naive_tok` = `sum(tok(it) for it in items)`   (what the starter kit counts)

`results/labels.jsonl`:

```
{"idx": int, "n_calls": int, "n_obs": int, "n_err": int, "n_unknown": int,
 "trailing_err": bool, "y_fric": 0|1, "y_proc": 0|1|2|null, "y_rate": float|null, "n_kill": int}
```

`results/jobkey.jsonl`:

```
{"idx": int, "cron_path": str|null, "literal": bool, "bucket": "literal"|"pii"|"none"}
```

`results/features.npz`: arrays `"X"` (n,d) float64, `"cols"` (d,) unicode, `"idx"` (n,) int64.
plus `results/feature_manifest.json`:
`{"cols":[{"name":str,"pre_treatment":bool,"block":"A"|"B"|"C"}]}`

`results/routes.jsonl`:

```
{"idx": int, "logged": str, "route": str, "changed": bool, "p_fric": float,
 "gates": {"family":str,"vision":bool,"admissible":[str],"sign_stable":[str],"tau_pass":bool},
 "cost_logged_usd": float, "cost_routed_usd": float, "supported": bool}
```

`results/estimates.json`, `results/metrics.json`, `results/claims.json`: free-form dicts, but
every number quoted anywhere must exist in `claims.json` under a stable key.

That contract is enforced in two places, and between them they cover everything a reader sees:

* `results/NUMBERS.md` at GENERATION time — every numeral in it is interpolated through
  `router/report.py`'s `_Q.__call__`, which raises `KeyError` on a key that is not in
  `claims.json`. Nothing rescans it afterwards because nothing can get into it unkeyed.
* the git-tracked user-facing artifacts at CHECK time — `python -m router.verify` reads
  `README.md`, `router/console.html`, `presentation.html`, `templates/presentation.html` and
  `site/index.html`, pulls every numeral out of their prose, and exits 1 on any that no claim
  stands behind (0 clean, 2 when `claims.json` has not been built, and 2 when one of those five
  files is absent — a surface it cannot read is BLOCKED, never skipped into a PASS, see ADR-018).
  See ADR-009 for what it can and cannot prove.

---

## VERIFIED ACCEPTANCE NUMBERS (these are ground truth — your module must reproduce them)

**recon:** `n_turns` total = 10,845; `sum(gross_tok)` = 334,729,910; the same sum EXCLUDING the
tools block = 290,009,114 (difference 44,720,796); `sum(cache_read_tok)` = 308,074,571 (92.0%);
`sum(cache_write_tok)` = 26,655,339 (8.0%); `sum(naive_tok)` = 22,631,879.
turns/line by arm: opus-5 10.36, sonnet-5 11.83, sol 10.85, terra 8.51, fable-5 11.72,
opus-4-8 10.29, luna 17.30.
spend concentration on `gross_tok`: top10 17.7%, top50 40.5%, top100 54.7%, top500 90.2%.

**labels:** 10,422 tool outputs total; 9,633 resolvable (92.4%); 474 errors;
`y_proc` counts `{0:737, 1:241, 2:18, null:4}`; `y_fric` positives = 259.

**jobkey:** 784/1000 runs carry a literal `"Cron path:"`; 354 distinct paths; 152 fully-literal
jobs covering 374 runs; of those 28 span >=2 arms (245 runs) and 23 span BOTH families (229
runs); 216 runs have no cron path; 410 runs have a `PII_` placeholder inside the path.
Largest clean job: `crons/heartbeat/` n=134 across 7 arms.

**features:** pre-treatment tokens 12,114,376 vs post-treatment 10,517,503 = 53.5% / 46.5%.

**costs:** effective multiplier vs fully-uncached, read 0.10x / write 1.25x:
rho=1.00 -> 0.192, rho=0.90 -> 0.272, rho=0.83 -> 0.329, rho=0.55 -> 0.555.

**policy:** routable set = 675/1000 trajectories = 197,551,096 gross tok = 59.0% of the bill
(claude lane, admissible logged arm, no pre-treatment image, not already on fable-5).
Excluded: gpt lane 245 (32.3%), non-admissible logged arm 23 (7.7%), already-cheapest 71 (7.0%),
pre-treatment images 9 (1.7%).

**family:** pooled over the 23 clean cross-family jobs: claude 33/157 = 21.0% friction,
gpt 21/72 = 29.2%; 134 of the 229 runs are the single job `crons/heartbeat/`.

---

## THE DATA (measured facts — do NOT re-derive, build on these)

`export/trajectories_v1_01.jsonl`, 1000 lines. Each line = `{model, input, tools}` and is a
COMPLETE trajectory: `input[0]` is the system message, `input[1]` the user message, `input[-1]`
the model's final assistant message — 1000/1000. There is no `output` field and no `usage`
field; the final answer is the last item of `input`. There are NO cross-line chains:
n = 1000 independent trajectories.

`scripts/load_trajectories.py`'s `group_trajectories()` is **BROKEN** (hashes
`first_user_text[:2000]` but 978/1000 first user messages exceed 2000 chars of shared
boilerplate, giving 953 collision groups). **NEVER call it.** Use `iter_requests()` and
`first_user_text()` only.

Token estimate everywhere: `tok(x) = len(json.dumps(x)) // 4`. All token counts are ESTIMATES;
the export has no usage field. Any user-facing string must say so.

Model families (perfect separation, 0 exceptions):

| tools contain | family | n |
| --- | --- | --- |
| `apply_patch` / `shell_command` | gpt | 245/245 |
| `bash` / `file_read` / `file_edit` / `file_write` | claude | 755/755 |

Arms: claude-opus-5 331, claude-sonnet-5 281, gpt-5.6-terra 113, gpt-5.6-sol 112,
claude-fable-5 71, claude-opus-4-8 69, gpt-5.6-luna 20, claude-opus-4-6 2, claude-sonnet-4-6 1.

Model ids are ANONYMIZED per `AGENTS.md`; no public price sheet applies. Every dollar figure
must name its assumed sheet.

## TURN RECONSTRUCTION (the core correction — every cost depends on it)

A line is not one API call. It is N calls, each re-sending the grown prefix.

Rule: a turn boundary opens at the START of every maximal run of model-produced items, where a
model-produced item is `(item.get('role') == 'assistant')` OR
`(item.get('type') in {'reasoning','function_call','custom_tool_call'})`.

Billed prefix of the turn starting at item index `i` = `tools_tok + sum(tok(items[:i]))`.
Cache split per line: `P = [billed prefix of each turn]`; `cache_read = sum(P[:-1])`;
`cache_write = sum(P) - cache_read`.

---

## BANNED, each with the measurement that killed it

| banned | the measurement that killed it |
| --- | --- |
| `group_trajectories()` | collision groups; n is 1000, not 953 |
| the `# === Cron memory: PII_URL_N` field as a job key | 763/763 are PII placeholders, only 18 distinct values, `PII_URL_3` appears 306 times |
| any `PII_*` token as a cross-row join key | placeholders are renumbered per request |
| text-keyword error detection as a LABEL | precision 2.8% to 49% vs exit-code ground truth |
| `empty_final` / `final_text_length` as label or feature | 238/238 empty finals are gpt — a serialization artifact that encodes family perfectly |
| an LLM judge over the final assistant message | 623/762 say "done", 8 admit any problem |
| system-prompt hash as a feature | 871 distinct over 1000 rows — it is an identifier |
| ANY feature computed from `input[first_user_msg+1:]` | 46.5% of tokens are model-generated (collider) |
| sklearn / scipy | not installed |
| IPS or DR point estimates | Kish ESS per arm equals `n_a` exactly (331/281/113/112/71/69/20/2/1) |
| a friction POINT estimate | MDE is 11.4pp on the best-powered arm pair; report bounds only |

## HARD RULES

- NEVER modify, delete or move: `scripts/`, `export/`, `site/`, `README.md`, `.gitignore`,
  `AGENTS.md`, `CLAUDE.md`. Other people are working in those.
  (`.claude/settings.json` blocks writes under `scripts/`, `export/`, `site/` at the tool level.)
- NEVER run `git add`, `git commit`, or any git write command.
- `export/` is READ-ONLY and proprietary. Never copy its contents into any file you write, never
  print more than a ~60-character redacted snippet, never upload anything anywhere.
- Write ONLY the files listed as yours. Another agent owns every other file.
- Scratch work goes in `/tmp`. `results/` is generated output (gitignored) — writing there at
  runtime is fine.

## CODE STYLE

Small, readable, stdlib-first. Every module gets a docstring stating what it computes, which
artifact it writes, and its acceptance numbers. Every public function gets a one-line docstring.
Modules must be runnable as `python -m router.<name>` printing their acceptance check, AND
importable. No global state. No network. No API keys.
