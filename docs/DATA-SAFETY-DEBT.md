# DATA-SAFETY DEBT — export content that reached `origin` and is still in history

**Status: OPEN, deliberately deferred.** Recorded 2026-08-23 ~01:35 during the overnight loop.
The working tree is clean and `tests/test_data_safety.py` is green; what remains is history that
has already been pushed. Cleaning it needs a rewrite of shared branches, which is a daytime
decision with the team, not a 2am one.

Do not quote any of the affected tokens into a file while fixing this — that is how the second
of the two incidents below happened.

---

## What is in history

### 1. `docs/POSTMORTEM-textclf.md` — raw un-redacted identifiers

**Where.** Commit `24e7469` on branch `text-classifier-finetune`, section 5, one paragraph.
Pushed to `origin` by `loop/bootstrap.sh`, which pushed every branch before any gate existed.

**What.** Three opaque vendor-style identifiers, quoted verbatim to illustrate that raw ids
survive inside memory content. Each appears in `export/` — occurrence counts 1, 7 and 2. They are
therefore slices of the licensed dataset, not invented examples. The same paragraph also names two
redaction placeholder serials.

**Why it matters.** `AGENTS.md`: challenge use only, no redistribution. A push is redistribution.

**Working tree, turn 7.** The paragraph was rewritten on `text-classifier-finetune` to describe the
shape of both problems instead of quoting either -- no identifier and no serial survives in the file.
`tests.test_data_safety` passes on that branch. This closes the *tree*, not the *history*: commit
`24e7469` is still on `origin` and still carries the original paragraph, so item 1 stays open here
until `i0006` is executed by a human.

### 2. Loop bookkeeping — placeholder serials

**Where.** Commit `5082319` on `main`, which introduced them; removed again in `8257642`. Both
are on `origin`, so both are in history.

**What.** Turn 1's merge gate correctly refused the textclf branch and wrote its reason into the
backlog note and the journal — quoting the two placeholder serials it had just found. The finding
was right; recording it that way reproduced the problem in three more files
(`docs/IDEAS.md`, `loop/JOURNAL.md`, `loop/state/backlog.json`).

---

## Severity

Lower than it sounds, and worth stating plainly rather than either dramatising or waving away:

- The repository is **private** (`isPrivate: true`), and every member already holds the dataset
  under the same licence. This is not a public leak.
- The quantity is three identifiers and two placeholder serials, not a data dump.
- It is nonetheless a breach of the repo's own stated rule, and of `AGENTS.md`.

## What the fix requires

A history rewrite of `main` and `text-classifier-finetune`, then a force-push. Both are shared:
`origin` also carries `dev-rkp` and `router-cache-segment-policy`, so at least one teammate has
branches that may be based on the affected commits. Coordinate before rewriting; a force-push that
strands someone's branch costs more than the debt does.

The minimal alternative, if a rewrite is judged not worth it: fix the content going forward
(backlog `i0004` already carries the exact edit for the postmortem) and accept the history, on the
record, in this file.

## What already stops a recurrence

- `loop/run.sh::push_all_branches` runs `tests/test_data_safety.py` and pushes **nothing** when it
  is red. Nothing leaves the machine past a red suite.
- `loop/prompts/_common.md` hard rule 2: no export content in any tracked file, and `grep -c` any
  concrete-looking token against `export/` before writing it down — counts only.
- Backlog `i0005` closes the gap that let the raw ids through: the suite matches placeholder
  *shapes* by regex, and nothing currently catches a raw vendor id.
