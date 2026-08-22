# COST LEVERS — what was tried against the prefix bill, and what survived

A record of every intervention measured against the reconstructed bill, so nobody spends
another session re-deriving a dead end. Six were tested. **Two work, four do not**, and three
of the four fail for the same reason.

Numbers are cost units under the assumed sheet (cache read 0.10x input, cache write 1.25x
input), against the baseline `sum(gross_tok)` = 334,729,910 est. tok / 64,126,631 cost units.
Everything is an ESTIMATE: `tok(x) = len(json.dumps(x)) // 4`, the export has no `usage` field.

Status values: **SHIPPED** (in the pipeline, keys in `claims.json`) | **MEASURED** (numbers
exist, not wired in) | **DEAD** (measured, does not work) | **IMPOSSIBLE** (cannot be built).

---

## The one rule that explains every result

> **What is fixed BEFORE the first turn is free to change.**
> **What is changed DURING the trajectory costs the cache.**

Prompt caching is a prefix hash. Anything that edits bytes already in the cached prefix
invalidates everything after the edit, and the re-written suffix is billed at 1.25x instead of
0.10x — **a factor of 12.5**. An intervention therefore has to save more than 92% of the tokens
it touches merely to break even on a single edit.

Every DEAD entry below reaches into a running trajectory. Both surviving levers are decided at
admission and never touched again. This is the same architecture ADR-001 arrives at from
causality; it happens to also be the only one the cache tolerates.

---

## Summary

| # | Lever | Result | Status |
|---|-------|--------|--------|
| 1 | Cross-model KV cache translation | cannot be built at all | **IMPOSSIBLE** |
| 2 | Active cache pre-warming | ≤ 0.07% — a latency tool, not a cost tool | **DEAD** |
| 3 | Terse internal mode / low effort | −1.71% ceiling; one extra turn costs 5.4x that | **DEAD** |
| 4 | Context editing (`clear_tool_uses`) | −31.9% tokens but **+9.5% to +71.9% cost** | **DEAD** |
| 5 | Tools cache breakpoint + 1h TTL | −5.02% to −6.98%, depends on unknown traffic density | **MEASURED** |
| 6 | Tool description truncation | **−6.12%** at 300 tok/tool, zero misses, no leak | **SHIPPED** |

---

## 1. Cross-model KV cache translation — IMPOSSIBLE

**The idea.** Carry the cached prefix across a model switch so mid-trajectory routing does not
pay a full re-prefill.

**Two independent blockers, either sufficient.**

*Research.* The frontier stops at within-family transfer. NVIDIA (arXiv 2608.03893) restricts
itself explicitly — *"we ... restrict the present study to within-family transfer"*, all six
evaluated pairs matched-KV by construction with a shared tokenizer — and names the router case
open: *"Whether the closed-form approach extends across families ... is open."* DroidSpeak
(NSDI'26) requires *"the same foundational model"* and leaves cross-foundation sharing to future
work. Even inside a family it is unreliable: DroidSpeak's Insight 1 is *"reusing the whole KV
cache between models leads to a huge loss in accuracy"* (>50% on HotpotQA), and NVIDIA's
training-free mapper drops 2 of 6 pairs to 41.6–44.2% of standalone accuracy.

*Provider surface.* There is no cache object to translate. Anthropic holds KV representations
in memory only; a grep of the prompt-caching docs and the full Messages API reference for
`cache_id` / `cache_name` / `export` / `download` returns nothing. The only cache surface is
the usage counters. Gateways (OpenRouter, LiteLLM, Portkey) operate on HTTP and can re-send
tokens, never move a KV entry. OpenRouter states it directly: *"Changing the model always
starts a different cache lineage."*

**Do not revisit** unless a provider ships an addressable cache object. Nothing in the research
literature changes the second blocker.

**Related trap:** Cache-to-Cache (arXiv 2510.03215) looks like learned cache translation and is
not — it *adds* a second full prefill (137 ms vs 27 ms, ~5x more) and is end-to-end slower than
the receiver alone. Its reported speedup comes entirely from replacing text-to-text decoding.

---

## 2. Active cache pre-warming — DEAD

**The idea.** Fire a `max_tokens: 0` request to write the cache before real traffic arrives.

**Why it fails.** Pre-warming does not eliminate a write, it *moves* one. Anthropic's own
guidance: it *"trades a cache-write charge now for lower TTFT on the next real request"* — a
latency tool. Once the tools block is shared (lever 5), the entire remaining write mass is
44,716 cost units, i.e. **0.07%** of the bill. Driving that to zero is not a lever.

Break-even for keep-alive pings, if ever needed: 25.0 cold starts/day at 5-min TTL, 1.3/day at
1h. But traffic refreshes the cache for free, so with continuous traffic pings are pure waste,
and with sparse traffic `ttl: "1h"` buys the same coverage for +0.04% one-off instead of
121,219 cost units/day.

**Do not build ping infrastructure.** Set the TTL instead.

---

## 3. Terse internal mode ("caveman" for model output) — DEAD

**The idea.** Have the agent think and act tersely, switching to normal prose only for the final
user-facing answer. Everything the model emits enters the history and is re-billed.

**Why it fails.** The premise is right — items are re-billed 10–15x — but it points at the
smallest slice. Decomposition of the bill by item type, with re-billing (sums exactly to
`gross_tok`):

| class | % of bill | avg re-bills |
|---|---|---|
| tool_output | 34.2% | 14.9x |
| system | 29.4% | 10.8x |
| tools_block | 13.4% | 10.9x |
| user | 11.8% | 12.2x |
| tool_call_args | 8.0% | 14.4x |
| **assistant_prose** | **2.4%** | 14.4x |
| reasoning | 0.7% | 12.9x |

The agent barely talks. Compressing prose 70% saves 1.71% of gross tokens. Against that: an
extra turn costs a full prefix, ~30,865 est. tok, versus ~5,714 saved per trajectory at maximum
compression — **ratio 5.4x**. If terser reasoning causes one extra turn in more than **18.5%**
of trajectories, the intervention loses money. The corpus already shows this mechanism from the
other direction: cheaper arms need +2.88 turns on the same job.

Not off-policy evaluable either — no trajectory in the export runs terse, so the cost side is
computable and the friction side is not.

The official knob for this trade-off is `output_config.effort` (documented to produce *"fewer
and more-consolidated tool calls, less preamble, and terser confirmations"*), which reaches
tool_call_args too. It is on the cache invalidation list, so it must be set at admission.

---

## 4. Context editing (`clear_tool_uses_20250919`) — DEAD, and instructively so

**The idea.** tool_output is 34.2% of the bill at 14.9x re-billing. Delete old tool results from
the context and that mass goes away.

**It does — and the bill goes up anyway.** Sweep over how many turns of tool output to keep:

| keep | Δ tokens | Δ cost | clears |
|---|---|---|---|
| 1 | −31.9% | **+9.5%** | 9,845 |
| 2 | −29.6% | +27.3% | 8,848 |
| 4 | −25.5% | +47.0% | 7,113 |
| 12 | −14.4% | **+71.9%** | 3,749 |
| 24 | −6.9% | +64.1% | 1,741 |

**No interval beats doing nothing.** Each clear rewrites the suffix from the edit point; write
share goes from 8.0% to 18–25%. The curve is non-monotone: the optimum is at the aggressive
edge, where prefixes stay small enough that the rewrites are cheap, and it is *still* worse than
the baseline.

**What this is really about:** context editing exists for the context WINDOW, not the bill. If a
trajectory would overflow, clearing is mandatory and this table is irrelevant. As a cost lever
it is the mirror image of the starter-kit error in ADR-007 — that one undercounted the bill 14.79x
by ignoring re-billing; this one would have claimed a 32% saving by ignoring the cache.

Visual write-up of this result: the Sägezahn artifact (charts of the divergence, the
per-turn prefix, and the read/write split).

---

## 5. Tools cache breakpoint + 1-hour TTL — MEASURED, blocked on Q3

**The idea.** The tools block renders first, before `system`. Put a `cache_control` breakpoint on
the last tool definition and it becomes its own cache entry — and there are only **9 distinct
tools blocks across 1000 trajectories**, against 871 distinct system prompts. One entry serves
many trajectories instead of each writing its own.

This is sound: the documented invalidation hierarchy has three tiers, and a system-prompt change
invalidates system and messages but **not** the tools cache.

**Measured effect** — but it depends on how densely traffic arrives, and the export has **no
timing** (AGENTS.md). Under an assumed uniform 1000 trajectories/day:

| configuration | Δ cost |
|---|---|
| breakpoint on tools, 5-min TTL | −5.02% |
| breakpoint on tools, 1h TTL | −6.98% |
| + tool diet | −9.91% |

Sensitivity to the unknown timeframe (tool tokens written per period):

| spread | 5-min TTL | 1h TTL |
|---|---|---|
| 1h burst | 69,590 | 35,773 |
| 1 day | 1,205,327 | 125,316 |
| 1 week | 4,109,537 | 299,724 |
| 1 month | 4,109,537 | 4,109,537 |

At week-scale spread the 5-minute TTL is worth exactly nothing — every trajectory writes its own
block. The 1h TTL moves that cliff by roughly 12x, for +0.04% one-off on the write.

**This is a new open question, on par with Q1/Q2 and unlike them actionable by the operator:**
does the production system already place the breakpoint ahead of the system prompt? If yes, this
lever is already banked and our cost basis is ~7% too high. If no, it is two lines of config.
Not yet added to `docs/OPEN-QUESTIONS.md`.

---

## 6. Tool description truncation — SHIPPED

**The finding.** The median trajectory DEFINES 12 tools and CALLS 2. Tool definitions carried but
never called are **7.32% of the bill** (24,491,657 token-turns). Five tools are never called
anywhere in the export.

But the dominant term is size, not membership:

| tool | size | defined in | called in | wasted | % of bill |
|---|---|---|---|---|---|
| `coworker_send_slack_message` | **1,878 tok** | 929 | 383 (41%) | 5,886,966 | **1.76%** |
| `wait_for_background_work` | 552 | 992 | 186 (19%) | 3,849,456 | 1.15% |
| `submit_draft` | 285 | 1000 | 14 (1.4%) | 3,007,218 | 0.90% |
| `send_message_to_thread` | 279 | 992 | 11 (1.1%) | 2,952,975 | 0.88% |
| `view_image` | 223 | 1000 | 39 (3.9%) | 2,069,217 | 0.62% |
| `bash` | 182 | 755 | 741 (98%) | 7,826 | 0.00% |

One tool is ten times the size of `bash` and wastes more than the entire terse-mode lever.

**Three interventions were priced** (`router/tooling.py`, `make tooling`):

| | mechanism | best | miss rate | leak-free |
|---|---|---|---|---|
| A OMIT | ship fewer tools | −3.54% (history) | **19.0%** | yes |
| A OMIT | core-set fitted | −12.39% | 52.1% | **no** |
| **B TRUNCATE** | **cap descriptions** | **−6.12% @ 300 tok** | **0** | **yes** |
| C DEFER | `defer_loading` + tool search | −9.65% | 0 | no |
| B+C | both | −12.00% | 0 | no |

**Only B is quotable without an asterisk.** A token cap is a constant, not a set fitted on this
corpus, and no tool is removed — the failure mode is a worse-described tool, not a missing one.
Everything keyed `.upper_bound` in `claims.json` is fitted on the same 1000 trajectories it is
scored on and would do worse deployed.

**Why OMIT loses.** The oracle that knows exactly which tools each run will call saves 9.14%.
Any implementable approximation gives that back as safety margin: the leak-free job-history
policy saves 3.54% while leaving 19.0% of trajectories short a tool. Prior runs of a `cron_path`
are only defined for recurring jobs; first runs have no history.

**What OMIT failure looks like, empirically.** Two trajectories call a tool absent from their own
array: idx 8 calls `shell_command` — a gpt-lane tool — from a claude-lane trajectory, and idx 8
and 120 call `failed_tool`, the harness placeholder for an unresolved call. That is the outage an
omit policy manufactures on purpose.

**Not measured:** whether a 300-token description still drives correct tool use. The export
contains no truncated descriptions, so the cost side is computable and the quality side is not
— the same shape as ADR-005, except without even a bound.

---

## Methodological traps hit while measuring this

Both produced wrong numbers that looked plausible. Check for them in any new lever.

1. **`tok(array)` is not `sum(tok(item))`.** `json.dumps` adds brackets and commas, and `tok()`
   floor-divides once per item instead of once per array. Reconstructing the tools block
   per-tool came out **105,101 est. tokens light** corpus-wide, so every delta was measured
   against the wrong ruler. `router.tooling` carries the difference as `frame`, and
   `tests/test_tooling.py` asserts the do-nothing policy reproduces recon to the token.

2. **Averaging tool mass across turns understates the write.** The final prefix is the cache
   write at 1.25x; the rest are reads at 0.10x. Any intervention where the tool mass grows over
   turns (deferred loading) must be priced per turn. Averaging made DEFER look like −10.30%
   when it is −9.57%, and B+C −11.56% when it is −12.00%.

---

## Still open

- **system: 29.4% of the bill**, mean 9,088 est. tok, re-billed 10.8x. The second largest item
  class and untouched. It is fixed before the first turn, so it falls on the right side of the
  rule. Not measured.
- **Turn count is superlinear in cost.** gross/turn rises from 16,703 (0–4 turns) to 52,867
  (60–64); twelve trajectories carry 16.1% of the bill. But they are not loops — identical
  consecutive tool calls run 0.1–1.3%, and exactly one trajectory has ≥5. A turn cap would cut
  real work, not waste. `results/metrics.json` has an empty `predictor/workload_head` slot;
  predicting expected turn count at admission would identify the twelve.
- **Q3 for the organizers** (see lever 5): is the cache breakpoint already ahead of the system
  prompt in production?
