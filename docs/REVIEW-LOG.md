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
