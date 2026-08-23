## Your job this turn: MERGE — move one verified idea onto main

`main` is the branch the team presents from. It only moves when the evidence says it should.

### 1. Re-verify on the idea branch — do not trust the flag

Take the oldest `implemented` idea and check out its branch (the `branch` field in the backlog).
Run the gate again yourself:

```bash
make test
make all
.venv/bin/python -m router.gates
```

The implement turn already claimed these pass. Verify anyway: it ran in a different process with
a different working tree state, and the whole point of this turn is to be the second opinion.
If any check fails, set the idea to `blocked` with a note naming the failing check, and stop.
Do not fix it here — that is the next implement turn's job.

### 2. Check that main is safe to move

```bash
git fetch origin main
git rev-parse main origin/main
git merge-base --is-ancestor main origin/main
```

This is a **shared team repository**. If `origin/main` has commits that local `main` does not,
and local `main` also has commits `origin/main` does not, the branches have diverged: a
teammate pushed. In that case do **not** merge. Set the idea to `blocked` with a note saying
main diverged, and stop. A human resolves it in the morning.

If local `main` is merely behind, fast-forward it first, then re-run the gate in step 1 against
the updated base before merging.

### 3. Merge

```bash
git checkout main
git merge --no-ff <branch> -m "<what this idea added, and the number that shows it worked>"
```

`--no-ff` on purpose: each idea stays a legible unit in the history rather than dissolving into
a flat line.

Not every branch is an `idea/<slug>` branch the loop created. Some pre-date the loop and sit on
an older base, so the merge can conflict. If it does:

```bash
git merge --abort
.venv/bin/python loop/backlog.py set --id iNNNN --status blocked --note "conflicts with main in <files>; needs a human"
```

Do not resolve a non-trivial conflict unattended at 03:00. An aborted merge loses nothing — the
branch is pushed and intact.

### 4. Verify on main, and undo if it broke

```bash
make test
make all
.venv/bin/python -m router.gates
```

If anything fails on the merged result, undo the merge and record it:

```bash
git reset --hard HEAD~1
.venv/bin/python loop/backlog.py set --id iNNNN --status blocked --note "merged clean but <check> failed on main; merge reverted"
```

A reverted merge is a correct outcome. A broken `main` at 08:00 is not.

### 5. Push and record

```bash
git push origin main
git push origin idea/<slug>
.venv/bin/python loop/backlog.py set --id iNNNN --status merged --note "on main as <short sha>"
.venv/bin/python loop/backlog.py render
```

Commit `docs/IDEAS.md` and `loop/state/backlog.json` with explicit paths, then push again.

If more than one idea is `implemented`, merge only the oldest this turn. The next merge turn
takes the next one, and it will branch-check against a main that has already moved.
