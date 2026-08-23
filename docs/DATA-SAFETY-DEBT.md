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

**Detection, turn 12.** Until now nothing would have caught this class a second time. The
opaque-token detector added for `i0005` needed a token of 16 characters and these are 11, so it
matched none of the five tokens commit `af4e78b` removed — measured, not argued. It now uses a
second, compact window (8–15 characters, no separator, two shape rejections) and catches all
three raw identifiers; the two placeholder serials were already covered by the shape detector. See
ADR-016 for why the single-window floor could not simply be lowered. This incident is now a
regression fixture rather than a docstring citation:
`tests.test_data_safety.OpaqueDetectorCatchesTheIncidentItWasWrittenFor` reads both sides of
`af4e78b` out of history at test time and fails if any token that commit removed stops being
detected. It writes no token into any tracked file and prints masked shapes only.

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

- `loop/push_gate.sh::push_all_branches` — sourced by both `loop/run.sh` and `loop/bootstrap.sh`,
  which no longer push on their own — scans the working tree once and then scans **each branch
  against its own tree** (`DATA_SAFETY_SCAN_REF`) before pushing that branch. A branch that fails
  is skipped and named; the clean ones still go. Until ADR-011 this said "nothing leaves the
  machine past a red suite", which was false three ways: the scan read whatever was checked out
  while every branch was pushed, `bootstrap.sh` had no gate at all, and the two paths below were
  never covered.
- **Not covered, on the record.** (a) `refs/entire/checkpoints/*` — exempt by name, pushed by the
  `entire` CLI itself; see `i0007` and the 36 refs already on `origin`. (b) The hand-written
  `git push` in `loop/prompts/merge.md` and `loop/prompts/deck.md`; those prompts are told to run
  the suite first, which is an instruction to a model rather than a gate.
- `loop/prompts/_common.md` hard rule 2: no export content in any tracked file, and `grep -c` any
  concrete-looking token against `export/` before writing it down — counts only.
- Both detector gaps that let this through are now closed (ADR-010, backlog `i0005` and
  `i0011`). `RepoCarriesNoOpaqueExportToken` catches a raw vendor id by asking the export whether
  it knows the token, so it needs no shape; and the concrete-placeholder pattern now accepts any
  attached separator, not only a trailing `_<digits>`, which is what the shorthand on line 3 of
  `loop/JOURNAL.md` exploited. Both were confirmed against a seeded leak, and both report a
  masked shape rather than the value.
- **One of those two closures is narrower than it reads, and this is the honest statement of it.**
  The opaque-token extractor requires a token of 16 characters, and the three identifiers whose
  removal it was written for are 11 — so it matches none of them and would not have caught the
  night-one incident. Backlog `i0018` lowers the floor; until it lands, the shape-free check
  covers longer ids only and `_common.md`'s manual `grep -c` rule is still doing the work for
  short ones.
- **Neither detector could be trusted to have run, and that is fixed (ADR-015, backlog `i0022`).**
  Every content check skipped when it could not enumerate files, and a suite in which everything
  skips exits 0, so `DATA_SAFETY_SCAN_REF` pointed at an unresolvable ref printed `OK (skipped=6)`
  and the push gate pushed that branch having scanned nothing. A missing `export/` made the
  raw-identifier check inert the same way. Both are now red under the gate, and the gate checks a
  receipt naming how many files were enumerated and how many tests ran rather than reading the
  exit code alone.
- Neither closes the *history* half of this file. The commits listed above still carry what they
  carry; a detector added today does not reach them.
