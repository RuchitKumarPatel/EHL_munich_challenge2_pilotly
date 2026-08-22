## Your job this turn: REVIEW — quality and security on everything that landed

Every other turn type has an author's incentive: get the thing built, get it merged. This turn is
the only one whose value is entirely in what it stops.

### 1. Scope yourself to what is new

```bash
.venv/bin/python -c "import json;print(json.load(open('loop/state/state.json')).get('last_review_sha') or '')"
git log --oneline <that-sha>..main
git diff --stat <that-sha>..main
```

If `last_review_sha` is empty, this is the first review: scope to `git diff main~10..main` or the
whole of `router/` if that is smaller, and say so in the log.

Also look at branches that have **not** merged yet — `loop/backlog.py list --json` gives you the
`branch` of every `implemented` or `blocked` idea. A finding on an unmerged branch is cheaper to
fix than one that has already landed.

If the diff contains no code and no docs, say so and finish. A review turn with nothing to review
should cost almost nothing; do not manufacture findings to justify the turn.

### 2. Quality

Run the `code-review` skill over the scope. Then dispatch the `everything-claude-code:python-reviewer`
agent on the changed modules under `router/`.

Weight findings by what this project is actually judged on. In descending order:

1. **A number that does not trace to `results/claims.json`.** The repo's own rule is that a figure
   with no stable key is not ours. A slide or a doc quoting an untraceable number is the single
   most damaging class of defect here.
2. **A post-treatment feature used as a routing input.** ADR-001 is the spine of the whole design.
   If anything in the diff touches `router/features.py`, the feature manifest, or a cross-trajectory
   join, dispatch `leak-hunter` — that is what it is for.
3. **An estimate presented as a point where the project only has a bound.** `results/NUMBERS.md`
   is explicit: the MDE is 11.4 pp, there is no point estimate, and there will not be one.
   Dispatch `estimator-critic` on any new quantitative claim.
4. Ordinary correctness, dead code, and duplication.

### 3. Security and data safety

Run the `security-review` skill, and dispatch `everything-claude-code:security-reviewer` over the
same scope.

Then the part that is specific to this repo and matters more than generic OWASP findings:

```bash
make test    # test_data_safety.py is in here
```

`tests/test_data_safety.py` catches placeholder tokens — `PII_PERSON_N`, `<ENTITY_FOO_N>` —
through two regexes. **It does not catch raw un-redacted identifiers copied out of the export.**
That gap is not hypothetical: on the loop's first night, `docs/POSTMORTEM-textclf.md` quoted three
real Slack-style ids straight from `export/`, the placeholder test caught the placeholders beside
them, and the raw ids only came to light because a human looked. Grep the diff for anything that
looks like an opaque vendor id — long runs of uppercase and digits — and check it against the
export before deciding it is invented:

```bash
grep -oE '\b[A-Z][A-Z0-9]{8,}\b' <changed-file> | sort -u
grep -c '<the-token>' export/*.jsonl     # count only, never print export content
```

Anything with a nonzero count is a slice of the proprietary dataset. `AGENTS.md`: challenge use
only, no redistribution. It comes out, and the file gets a note saying what it was replaced with.

Closing that gap in `tests/test_data_safety.py` is itself a legitimate backlog entry if it is
still open.

### 4. Record findings as work, not as prose

Findings go into the backlog so the normal machinery fixes them:

```bash
P=.venv/bin/python
# ordinary finding -- goes through the evaluate turn like any other idea
$P loop/backlog.py add --kind code --title "<the defect>" --why "<file:line, what breaks, how it was found>"

# security or data-safety finding -- skips evaluation, there is nothing to judge
$P loop/backlog.py add --kind code --title "<the defect>" --why "<file:line, evidence>"
$P loop/backlog.py set --id iNNNN --status accepted --note "SECURITY/DATA-SAFETY. Fix: <exact change>. Success = <the test that will prove it>"
```

Be strict about that split. Routing a real data-safety finding through `evaluate` wastes a turn
deciding something that is not a judgement call. Routing a style opinion straight to `accepted`
burns an implement turn on nothing.

Then append a dated section to `docs/REVIEW-LOG.md`: the scope reviewed, what was found, what was
filed, and — importantly — what you looked at and found clean. A review log that only lists
problems cannot tell you next week what was already checked.

### 5. Close out

```bash
.venv/bin/python loop/backlog.py mark-review --sha "$(git rev-parse main)"
.venv/bin/python loop/backlog.py render
git add docs/REVIEW-LOG.md docs/IDEAS.md loop/state/backlog.json loop/state/state.json
git commit -m "review: <n> findings across <scope>"
```

Do not fix anything yourself this turn, with one exception: if you find dataset content in a
tracked file, remove it immediately and commit that on its own. Everything else goes through the
implement turn, so that a fix gets tests and a gate rather than a reviewer's confidence.
