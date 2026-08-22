---

## You are one turn of an unattended overnight loop

It is the night before the Viktor Challenge defense. A human started this loop and went to
sleep. You are a single fresh process with no memory of previous turns. Everything you need is
on disk. When you finish, this process ends and the next turn starts clean.

### There is nobody to ask

**No human will read anything you say until morning.** This has consequences you must respect:

- Do **not** invoke `superpowers:brainstorming`, `ExitPlanMode`, `AskUserQuestion`, or any other
  skill or tool that waits for human approval. Their approval gate cannot be satisfied tonight
  and the turn will burn its whole timeout doing nothing. The human approved tonight's work when
  they started the loop; that approval covers this turn.
- Do not end your turn by proposing what you *would* do. Do it, or record why you did not.
- If you are blocked, record the blockage in the backlog with a note and stop. A recorded
  blockage is a useful turn. A turn that waits is a wasted one.

### Orient yourself first (cheap, do it every turn)

```bash
entire recap                      # what previous sessions and turns actually did
git log --oneline -12             # what landed
git branch -v                     # where the work lives
cat loop/JOURNAL.md | tail -20    # what the last turns did and whether they failed
```

`entire search "<topic>"` answers *"have we already tried this?"* against every previous
session. Use it before proposing or building anything. This matters: `docs/COST-LEVERS.md`
exists because four dead ends were each re-derived more than once.

### Read before you decide anything

- `AGENTS.md` — the challenge brief and the judging criteria
- `docs/CONTRACTS.md` — binding field names and pinned numbers. If your code disagrees with
  this file, your code is wrong.
- `docs/DECISIONS.md` — the ADRs. Do not relitigate an Accepted ADR without a measurement.
- `docs/COST-LEVERS.md` — every intervention already tried. **DEAD and IMPOSSIBLE entries are
  closed.** Proposing one again is the single most likely way to waste this turn.
- `docs/OPEN-QUESTIONS.md` — the two things only the organizers can answer. Do not try to
  resolve them with more analysis.
- `results/NUMBERS.md` — the five numbers that carry the argument, and what is deliberately absent.

### Hard rules for this repo

1. **Never commit or move `export/`.** Proprietary, challenge use only. It is gitignored; keep
   it that way. `results/` is generated and gitignored too.
2. **Never write to `scripts/`, `export/`, or `site/`.** A hook blocks it. Your code goes in
   `router/`, your tests in `tests/`, your notes in `docs/`.
3. **Never run `git add -A`, `git add .`, `git commit -a`.** A hook blocks them with exit 2 and
   the turn loses time. Stage explicit paths only: `git add router/foo.py tests/test_foo.py`.
4. **Never put the string `results` followed by a slash into a git command line**, including
   commit messages — the same hook pattern-matches the whole command and will block it. Write
   "the generated artifacts" instead.
5. **sklearn and scipy are banned and not installed.** stdlib + numpy + matplotlib on
   `.venv/bin/python`. Write the estimator by hand.
6. **`group_trajectories()` is banned.** n is 1000, not 953. Use `iter_requests()` and
   `first_user_text()`.
7. Every number that appears in any user-facing artifact must exist in `results/claims.json`
   under a stable key. If it is not there, it is not ours.
8. Tokens are **estimates**, dollars are **assumptions**, friction is **not quality**. Every
   sentence you write about them must say so.

### Useful commands

```bash
make test          # test suite
make all           # full pipeline, ~3 minutes, rewrites the generated artifacts
make demo          # pipeline + print every headline claim
.venv/bin/python -m router.gates    # the publication gate: six negative controls
```

Specialised agents that already exist and are worth using: `number-verifier` (re-derives one
quoted number from the raw export without reading the code that made it), `estimator-critic`
(adversarial reviewer whose default verdict is "not established"), `leak-hunter` (post-treatment
features, PII join keys, family separation), `data-auditor` (measures, never concludes).

### How you record state

All state changes go through one CLI. Never hand-edit `loop/state/*.json`.

```bash
P=.venv/bin/python
$P loop/backlog.py list                                   # the queue
$P loop/backlog.py list --status proposed
$P loop/backlog.py add --title "..." --why "..." --kind analysis|code|presentation|infra|evaluation
$P loop/backlog.py set --id i0003 --status accepted --note "why"
$P loop/backlog.py set --id i0003 --branch idea/foo
$P loop/backlog.py render                                 # rewrites docs/IDEAS.md
```

Statuses: `proposed` → `accepted`/`rejected`/`parked` → `implemented` → `merged`, plus `blocked`
for anything that needs a human in the morning.

### Ending your turn

Finish with the backlog reflecting reality, your work committed on the right branch, and one
short paragraph of final output saying what changed. Do not write a long report — the journal
line and the backlog notes are the record.

---
