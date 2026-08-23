# REVIEW LOG

One dated section per review turn. Records what was looked at, what was found, and — just as
importantly — what was checked and found clean, so a later turn does not re-derive the same pass.

---

## 2026-08-23 — turn 4, first review

**Scope.** `last_review_sha` was empty, so this is the first review and it scoped to
`main~10..main` (21 files, 1789 insertions) plus the one unmerged branch with unique commits
(`text-classifier-finetune`, 1 ahead). The other four local branches are 0 ahead of `main`.

The diff is **entirely loop infrastructure** — `loop/run.sh`, `loop/backlog.py`,
`loop/bootstrap.sh`, `loop/config.env`, the six turn prompts, `.entire/` config and the ignore
file. `router/` and `tests/` are untouched in this range. Consequences, stated so the next
reviewer does not repeat the reasoning: no `leak-hunter` was dispatched (ADR-001's spine —
`router/features.py`, the feature manifest, any cross-trajectory join — is not in the diff), and
no `estimator-critic` was dispatched (the diff makes no new quantitative claim).

**Method.** `code-review` skill over the range; `python-reviewer` on `loop/backlog.py`;
`security-reviewer` on the five shell/config files. Every agent finding below was re-verified by
hand before filing.

### Found and filed

| id | severity | finding |
|----|----------|---------|
| i0007 | critical | `refs/entire/checkpoints/*` — 34 refs, all already on `origin` — carry session transcripts holding **67 distinct concrete redaction serials**. Outside every gate: the ignore file does not apply to refs, `_repo_files()` enumerates tracked files only, and `push_all_branches` exempts `entire/*` by name because the pre-push hook pushes them. |
| i0008 | critical | `loop/run.sh:193` splices `state.json`'s `cost_usd` into a `python -c` source string. Any turn can write that file; the supervisor re-evaluates it every iteration, outside any tool-permission boundary. |
| i0009 | high | `loop/run.sh:154` scans the checked-out tree, `loop/run.sh:161-183` pushes every ref under `refs/heads`. A branch that is not the current checkout is pushed unscanned. |
| i0010 | high | `loop/bootstrap.sh:100` pushes every branch with no data-safety call anywhere in the file. Commit `8257642` added the gate to `run.sh` only. Its `make test` also runs *before* `make all`, so the two content-scanning suite classes skip on an empty output directory rather than fail. |
| i0011 | high | `CONCRETE_PII` (`tests/test_data_safety.py:267`) requires a trailing `_<digits>`; a serial written with the digits after another separator evades it. One reached `main` and survived the turn-1 cleanup. Fixed in `c05e078`. |
| i0012 | high | `loop/backlog.py:101-104` returns `deck` before testing the review cadence, so `review` never fires when `deck_every` divides `review_every`. Measured over turns 1..40: `(5,10)`, `(4,4)`, `(5,5)`, `(3,6)` never fire once. Current `(5,4)` works by luck. Same line: `deck_every=0` raises `ZeroDivisionError`. |
| i0013 | medium | `docs/DATA-SAFETY-DEBT.md:61` and `loop/README.md:57` claim nothing leaves the machine past a red suite. Three push paths bypass it, plus i0007. |
| i0014 | medium | `TEAM_MEMBERS` is empty; `templates/presentation.html:34` will render a `FILL` placeholder on the title slide. Needs a human — blocked. |
| i0015 | low | `loop/backlog.py:57-66` — unhandled `JSONDecodeError` becomes an empty string that `run.sh` trusts as a scalar. |
| i0016 | low | `loop/run.sh:180` skips branches against remote refs only ever fetched for `main`. |

### Acted on directly

Two things, both narrow, both recorded so they can be undone:

1. **`c05e078`** removed a concrete redaction serial from `loop/JOURNAL.md`. Committed alone, per
   the standing rule that dataset content in a tracked file comes out immediately. The sentence's
   point — that the merge gate caught two serials — never needed the value.
2. **The `entire` pre-push hook was renamed** to `.git/hooks/pre-push.disabled-by-review-turn4`,
   because the loop pushes again at the end of every turn and each push shipped more transcript.
   This is local, untracked and reversible in one `mv`; see
   `.git/hooks/README-pre-push-disabled.txt`. Branch pushes are unaffected and still gated.
   **Do not restore it until i0007 is resolved.**

Nothing else was fixed. Every other finding goes through an implement turn so it gets a test.

### Left for a human, deliberately

i0006 (filed by a concurrent session at 01:35) and i0007 both need a force-push or a remote ref
deletion on a shared remote that teammates hold branches on. That is irreversible and
outward-facing; an unattended turn must not attempt it. i0014 needs names only a human knows.

### Checked and clean — do not re-derive

- **`make test`: 82 tests, OK.** Including all of `tests/test_data_safety.py`.
- **Numbers traceability.** The only project number quoted anywhere in the diff is `11.4 pp`
  (`loop/prompts/deck.md:55`, `loop/prompts/review.md:38`); `refusal.mde_best_powered_arm_pair.pp`
  exists in `claims.json` with that value. No untraceable figure in the range.
- **Whole tracked tree swept for concrete serials.** After `c05e078` the only ones left are
  `PII_URL_3`, `PII_SERVICE_1`, `PII_PROJECT_12` — all three in `DOCUMENTARY` with justifications,
  allowlist still at 4 of its cap of 8.
- **`loop/logs/` is untracked** and ignored; it holds serials, and correctly never reaches git.
- **The broad stage-everything call in `commit_stragglers`** was suspected and cleared: the ignore
  file covers the export directory, the generated artifacts, `submission/`, `*.zip` and `*.tar.gz`,
  so it cannot stage the dataset. The residual risk is a *derivative* quoting it, which is exactly
  what the data-safety suite exists to catch.
- **`_write_json` (`loop/backlog.py:43-54`)** is a correct atomic write — same-directory `mkstemp`,
  `os.replace` after close, tmp unlinked on `BaseException`. A killed turn cannot truncate state.
  Reviewed and explicitly approved; no finding.
- **Credential handling.** `usage_pct`/`quota_reset_epoch` read `~/.claude.json` but emit only a
  derived integer. No path writes its content to a log, prompt, journal or commit.
- **Retry bounding.** The rate-limit retry is capped at 3 attempts, any single wait at 7200s, and
  `sleep_for_quota` polls the kill switch and the deadline every 60s. Cannot spin forever.
- **Kill switch is cooperative by design** — checked between turns, not during one. Documented in
  `loop/stop.sh`. Not filed as a defect, but note it means setting `.loop-stop` mid-turn still
  allows one more push cycle to run.

## 2026-08-23 — turn 8 (review)

**Scope.** `last_review_sha` was `c05e078`. Two surfaces:

- `c05e078..main` — 6 files, +411/-7. Docs, the backlog, and one new code-bearing
  artifact: `presentation.html` (88 lines), built by turn 5 and never reviewed.
- `main...idea/integrate-turn7` — 27 files, +2898/-156, **unmerged**. The six
  turn-6/7 idea branches integrated into one merge candidate (i0017). Reviewed
  here rather than after landing, because a finding on an unmerged branch is
  cheaper.

Read-only worktree at `/tmp/wt-int`. Ran the `code-review` skill over the branch
and dispatched `python-reviewer`, `security-reviewer` and `estimator-critic`.
`leak-hunter` was **not** dispatched, deliberately: the diff touches neither
`router/features.py` nor the feature manifest nor any cross-trajectory join, and
`router/textclf/` is a frozen negative result with no path into the pipeline.

### Findings filed

| id | severity | what |
|----|----------|------|
| i0021 | **highest** | The deck's headline `+4.59 pp` is `upper(A) − upper(B)` and bounds nothing; the comparison it frames is settled by reroute volume, not routing quality. |
| i0022 | high | The per-ref data-safety scan fails open — any git error becomes "branch clean". |
| i0023 | high | Branch names reach `git push` / `ls-tree` / `cat-file` as bare arguments; an option-shaped ref executes code. **Pre-existing on `main`**, not a branch regression. |
| i0018 | high | The new opaque-token detector's 16-char floor misses the 11-char identifiers its own docstring says it exists to catch. |
| i0019 | medium | `router.verify` reports PASS when a listed artifact is absent. |
| i0026 | medium | `loop/README.md:86` contradicts the i0005 fix shipping on the same branch. |
| i0024 | low | `tests/test_push_gate.py` defines a class below its `__main__` guard. |
| i0025 | low | `router.verify` prose stripper: unclosed markdown fence, `>` inside a quoted attribute, fixed 2-char percent lookahead. |
| i0020 | low | `router/textclf` hardcodes a dead `/tmp` session-uuid scratch path. |

### The three that were measured, not asserted

**i0021 — the deck headline.** Verified two ways, independently of the agent that
raised it. `router/ope.py:594` is typed `-> Tuple[None, None, float, dict]` and
returns `(None, None, hi, diag)`: `non_inferiority_bound` produces a one-sided
**upper** bound only, and both frontier legs come from it, so their difference
bounds the true gap in neither direction. Separately, the identified delta is
`Σ W·1[switched]·(m − y)` with `m, y ∈ [0,1]` and spend weights summing to 1, so
`|delta| ≤` the switched spend share `= 0.035888 = 3.5888 pp` — while the hull at
matched spend sits at `−3.8322 pp`. **The gated router cannot reach the hull by
arithmetic**, however well it routes. The slide is measuring that we moved 3.6% of
spend. That is a better and more defensible finding than the one on the slide, and
it is the one the deck should make.

**i0022 — the scan fails open.** No crafted input needed:
`DATA_SAFETY_SCAN_REF=refs/heads/no-such-branch-xyz .venv/bin/python -m unittest
tests.test_data_safety` exits **0** with `OK (skipped=6)`. `_repo_files()` returns
`None` on any git failure, every content test turns that into `SkipTest`, and
`unittest` exits 0 when everything skips. `ds_scan_ref` reads only that exit code.

**i0018 — the detector misses its own incident.** All five tokens removed by commit
`af4e78b` occur in the export (1, 2, 7, 7025 and 9828 occurrences, counted never
printed). The three raw identifiers are **11 characters**; `OPAQUE_RE` requires
16–64, so it matches none of the five. Measured the cost of lowering the floor so
the implement turn does not have to: repo-wide candidates go 27 (floor 16) → 63
(11) → 93 (8), against a single streamed pass over the 101 MB export, so runtime is
unchanged. The 16-char floor buys nothing.

### Looked at and found clean

- **Data safety of the whole diff.** Every changed file scanned for opaque tokens
  and for mixed-alphanumeric runs; the only hits were English words, model ids, a
  Claude Code session uuid and the AWS test fixture. No new export content.
- **The pre-existing documentary placeholders** (`tests/test_data_safety.py`,
  `router/demo.py`, `docs/CONTRACTS.md`, ADR-005). Re-counted against the export:
  still nonzero, still inside `DOCUMENTARY` with justifications, **unchanged by
  this diff**. Adjudicated by the turn-4 review; not relitigated.
- **`presentation.html` provenance.** All 41 numerals across the five artifacts bind
  to a claims key; 0 orphans; the copies on `main` and on the branch are identical.
  README binds 0 not through a bug but because its 15 numerals are 10 inside code
  fences and 5 legitimately in `NOT_A_CLAIM` (hackathon dates, quick-start minutes,
  Python floor) — checked directly.
- **The deck's discipline about bounds.** Outside i0021 it is careful and correct:
  "bounds, never points", tokens named as estimates, dollars as an assumption,
  friction as process not quality, the MDE and the absent point estimate stated on
  the closing slide.
- **`torch` isolation.** Nothing outside `router/textclf/` references it; `make all`
  and `make test` never import `transformers`. i0004's acceptance criterion is met.
- **The turn-4 fix to the `cost_usd` → `python -c` interpolation (i0008).** Complete,
  not merely moved: every `python -c` in `loop/*.sh` is single-quoted with values
  passed as argv, `grep -n '\-c "' loop/*.sh` is empty repo-wide, and every `$(( ))`
  site consumes an internally-generated or pre-coerced value. Its regression test
  proves the payload is live under the old form before proving it inert under the
  new one — a real negative control.
- **`plan_turn` cadence (i0012).** The review/deck tie-break and deferral hold over
  the full 12×12 grid.
- **German locale handling in `router.verify`.** `"11,4 pp"` → `refusal.mde_best_powered_arm_pair.pp`,
  `"1000"` → `corpus.n_trajectories`, no thousands-comma misread; locale declared
  per artifact, never sniffed.
- **`loop/backlog.py`** has no `eval`, `exec`, `subprocess` or `shell=True` anywhere.

### Note for the next review turn

The `security-reviewer` reported a mid-run system reminder about bypass-permissions
mode as a probable prompt injection. It is **not** — it is this harness's genuine
reminder, and it reaches subagents too. Recorded so nobody spends a turn chasing it.

---

## 2026-08-23 — turn 12, quality and security

**Scope.** `last_review_sha` was `9680073` (the carry-forward before turn 8), so `main`'s new
material is `9680073..main`: the turn-9 integration merge — already reviewed at turn 8 *as a
branch*, so not re-derived — plus **`ba96b34`, the turn-10 deck rebuild, which no review had
seen**: `presentation.html`, `router/report.py`, `router/figs.py`. Second half of the scope is the
unmerged work, and there is a lot of it: **eight** turn-11/12 idea branches, all eight already
merged into `idea/integrate-turn12` (24 files, +3244/−151 vs `main`). Reviewed the integration
branch rather than the eight, in a throwaway worktree with `.venv`, the export and the generated
artifacts symlinked in.

**Gate re-run independently on the merge candidate**, not taken from its own notes: `make test`
**230 OK** (`main` is 140), `router.gates` **GREEN** 4 pass / 2 warn / 0 fail, `router.verify`
**PASS** on 44 numerals across 5 artifacts, per-ref data-safety scan green with receipt
`files=133 ran=32 skipped=0 failed=0`. ADR headings run 001..022 with no duplicate number.

**Seven findings filed, all accepted** (none needed `evaluate`: each either defeats a control or
is a proven correctness gap). In priority order:

| id | weight | finding |
|----|--------|---------|
| i0032 | **highest** | The deck's *replacement* headline is overstated the way the number it replaced was. `weights("spend")` is **estimated tokens** (`router/ope.py:171`), so "3.6% of est. spend" is a token share printed beside a dollar-denominated chart — the same 119 trajectories are **9.8%** of the `assumed_default` bill. And "no policy at this tau" is not established. |
| i0033 | high | The push gate runs `-m tests.test_data_safety` **from the working tree** every turn, and the receipt only checks the shape of a printed line. A ten-line stub that prints a well-formed receipt makes every branch that cycle scan clean — proven end to end against a real remote. |
| i0034 | high | `router.verify` reads a model identifier's version suffix as a signed numeral: **six of the nine arm names orphan**, three bind to an unrelated claim. Turn 15 is a deck turn. |
| i0035 | medium | Three ways the prose gate narrows itself and still prints PASS: quote pairing across tag boundaries, the percent lookahead crossing blanked markup, an unclosed fence exiting 0. |
| i0036 | medium | `_check_backlog` validates that `notes` is a list, never a note's shape — `render` then raises `KeyError`, exit 1 with a traceback, where the branch's own ADR-019 and tests promise exit 3 and a sentence. |
| i0038 | medium | Two branches merged cleanly and now **three passages disagree** about whether the opaque-token detector's 16-character floor is still live; `docs/DATA-SAFETY-DEBT.md` contradicts itself 63 lines apart. The i0026 class, reappearing through a clean merge. |
| i0037 | low | All three `backlog_run` call sites redirect stdout to `/dev/null`, swallowing the helper's own failure warning. |

**Bookkeeping corrected on `main`, with evidence.** `i0015`, `i0020` and `i0025` read `proposed`
with `branch: null` on `main` while built and green on their branches — the `i0031` staleness
again. Corrected, and a merge instruction pinned as a note on `i0022`: **merge
`idea/integrate-turn12`, not the four branches `main`'s queue names**, and resolve the
`backlog.json` conflict as a **union**, because `main` has no `i0027`..`i0031` and none of
`i0032`..`i0038`. `plan_turn` on the corrected queue returns 13 merge, 14 merge, 15 deck, 16 review.

### Looked at and found clean

- **Data safety of the whole scope.** 32 opaque-looking candidates extracted from the branch
  diff, every one counted against the export: **all zero**. The `Entire-Checkpoint:` ULIDs in
  `main`'s commit messages: also zero. No dataset content anywhere in the diff; nothing removed.
- **The i0021 fix (turn 10) holds and was re-checked at source.** The retired key is named for
  what it is, `4.59` survives only as a claims-table row, the chart's arrow labels two levels
  instead of a distance, and `router/console.html` — the German artifact, easy to forget — carries
  no gap framing at all. The *direction* of turn 10's reframing is right; i0032 is about how far
  it was pushed, not about undoing it.
- **i0022 is genuinely fixed**, verified by re-running the original repro rather than reading the
  fix: `DATA_SAFETY_SCAN_REF` at an unresolvable ref now exits 1 with `files=0 … failed=3`, where
  it used to print `OK (skipped=6)` and exit 0.
- **i0023 is genuinely fixed** (confirmed by the security reviewer against a hostile ref in a
  throwaway repo): the name is refused before it reaches a git argument list, the push is pinned
  to the scanned sha, and clean sibling branches in the same cycle still push.
- **The 7 tests that do not run under `SCAN_REF`** were checked one by one rather than assumed:
  all are the generated-artifact scanners and the git-history regression fixture, none of which a
  ref can carry. Legitimate, not an inert control.
- **`router/figs.py`'s `switched_spend_share`** reproduces from `routes.jsonl` and `recon.jsonl`
  by hand to the digits the OPE path reports, and the new frontier CSV column has no consumer that
  reads by position.
- **`loop/backlog.py`'s fail-closed store, `next_id`, `backlog_scalar`**, the `TYPE`/`TURN`
  handling in `run.sh` (no path traversal — `TYPE` only reaches `PROMPT_FILE` after matching a
  fixed enum), and the single-quoted `python -c` rule from ADR-012: all still hold.

### Note for the next review turn

Two agents this turn had side effects worth knowing about. The `code-review` skill ran
`git checkout --` on the shared review worktree and silently discarded this turn's uncommitted
backlog edits, which had to be re-filed — **commit backlog changes before dispatching anything
that may clean a worktree.** And the `security-reviewer`'s output tripped the harness's
instruction-shaped-content filter; its findings were still sound and were reproduced
independently before being filed.
