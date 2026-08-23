## Your job this turn: EVALUATE — judge one proposed idea properly

Ideas are cheap. This turn is the filter that stops the loop from spending a 40-minute
implement turn on something that cannot work.

### 1. Pick

Take the **oldest** idea with status `proposed` (lowest id). That one gets the deep treatment.

You may additionally reject-on-sight any *other* `proposed` idea that is plainly a duplicate of
an existing entry or is listed DEAD/IMPOSSIBLE in `docs/COST-LEVERS.md`. Give each a one-line
note. Do not spend real effort on them — that is what their own evaluate turn is for.

### 2. Judge the picked idea against four questions

**Is it new?** `entire search` it. Check `docs/COST-LEVERS.md` and every backlog entry including
`rejected` ones. A rediscovered dead end is the most expensive kind of idea.

**Would it survive an adversarial reviewer?** If the idea makes or depends on a quantitative or
causal claim, dispatch the `estimator-critic` agent against it. That agent's default verdict is
"not established" — take that seriously rather than arguing it down. If the idea touches
`router/features.py`, the feature manifest, or any join across trajectories, also dispatch
`leak-hunter` on the *plan*, before code exists.

**Does it move a judging criterion?** Name which one, and name the measurement that would show
it worked. "Makes the analysis better" is not an answer. "Raises out-of-fold AUPRC above the
0.3475 single-column baseline, measured under job-blocked folds" is.

**Can it be finished and verified in one 40-minute turn?** Be honest. Something that needs
`make all` plus a new estimator plus a new figure probably cannot. If it is too big but still
good, the answer is `parked`, not `rejected`.

### 3. Probe cheaply — but do not implement

You may read code, inspect the export, and run small throwaway scripts under `/tmp` to test
feasibility. Use `data-auditor` when the question is "how many / what share / does this hold in
the data". Do **not** write anything into `router/` this turn, and do **not** create a branch.
If you find yourself implementing, stop and mark the idea `accepted` instead.

### 4. Decide and record

Exactly one of:

```bash
P=.venv/bin/python
$P loop/backlog.py set --id iNNNN --status accepted --note "serves <criterion>; success = <measurement>; plan = <2-3 sentences>"
$P loop/backlog.py set --id iNNNN --status rejected --note "<the specific reason, with the evidence>"
$P loop/backlog.py set --id iNNNN --status parked   --note "<why it is good but too big for one turn>"
```

An `accepted` note is the brief the implement turn will work from, so it must contain the plan
and the success measurement. A vague accept produces a wasted implement turn.

Then `loop/backlog.py render` and commit `docs/IDEAS.md` and `loop/state/backlog.json` with
explicit paths.
