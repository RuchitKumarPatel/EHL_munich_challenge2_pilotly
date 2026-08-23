## Your job this turn: IMPLEMENT — build exactly one accepted idea

### 1. Pick and brief yourself

Take the `accepted` idea whose note names the strongest measurement. Its note is your brief: it
contains the plan and the success criterion the evaluate turn committed to. Build that, not a
larger version of it.

### 2. Branch

If the idea already names a `branch` in the backlog, **work on that branch** — the point is to
make existing work better across iterations, not to fork a second copy of it:

```bash
git checkout <existing-branch>
```

Otherwise create one from `main`:

```bash
git checkout main
git checkout -b idea/<short-kebab-slug>
.venv/bin/python loop/backlog.py set --id iNNNN --branch idea/<short-kebab-slug>
```

Always branch new work from `main`. Never start a *new* idea on top of another idea's branch —
the merge turn relies on every new branch being a clean descendant of `main`.

### 3. Build it test-first

The repo has a real test suite (`tests/`) and a real pipeline (`make all`, ~3 min). Follow the
repo's conventions rather than inventing new ones: read a neighbouring module in `router/`
before writing a new one and match its structure, its docstring style, and how it writes into
the generated artifacts.

- Write the failing test first, in `tests/`. Watch it fail. Then make it pass.
- stdlib + numpy + matplotlib only. sklearn and scipy are banned and not installed.
- New numbers must be written into `results/claims.json` through `router/report.py` under a
  stable key, not printed and transcribed.
- If you touch `router/features.py`, the feature manifest, or any cross-trajectory join,
  dispatch the `leak-hunter` agent before you consider it done.

### 4. The gate — all four must pass

```bash
make test
make all
.venv/bin/python -m router.gates
```

and, if you introduced or changed a headline number, dispatch `number-verifier` on it. Its
verdict must be MATCH.

### 5. Commit and record honestly

Stage explicit paths only. No `git add -A`, no `git add .`, no `git commit -a` — a hook blocks
those with exit 2. Do not put the string `results` followed by a slash anywhere in the command
line, commit message included; the same hook matches the whole command.

```bash
git add router/<files> tests/<files> docs/<files>
git commit -m "<what changed and why, in the repo's voice>"
```

Then, honestly:

```bash
P=.venv/bin/python
# everything green:
$P loop/backlog.py set --id iNNNN --status implemented --note "<what was measured, and the number>"
# tried it, does not work:
$P loop/backlog.py set --id iNNNN --status rejected --note "<what failed, with the evidence>"
# half built, needs a human:
$P loop/backlog.py set --id iNNNN --status blocked --note "<exactly where it stands and what is needed>"
```

**Mark `implemented` only when all four checks passed.** The merge turn trusts this flag to move
`main`. A false `implemented` is the one failure mode that can damage the branch the team
presents from. If the idea turned out not to work, `rejected` with the evidence is a genuinely
good outcome for the night — `docs/COST-LEVERS.md` is made of exactly those.

Finally `loop/backlog.py render` and commit `docs/IDEAS.md` and `loop/state/backlog.json`.
