# The overnight loop

An unattended agent loop that keeps improving this project while nobody is watching, and keeps
the slide set current while it does. One turn = one fresh `claude -p` process. Nothing is
carried in context between turns; the state lives in git, in `loop/state/`, and in `entire`'s
checkpoints.

That is the whole design argument: a turn can hang, crash, or go off the rails without taking
the night with it. The supervisor notices, writes a journal line, and starts the next one clean.

## Run it

```bash
# 1. fill in the team name so the deck does not ship [FILL] markers
$EDITOR loop/config.env

# 2. start
nohup loop/run.sh > loop/logs/supervisor.log 2>&1 &

# 3. check on it any time
loop/status.sh

# 4. stop it politely (finishes the running turn), or hard
loop/stop.sh
loop/stop.sh --now
```

Restart after a stop: `rm .loop-stop` and start again. The turn counter, the backlog and the
branches all carry over — a restart resumes, it does not reset.

## What a night looks like

The supervisor picks the turn type from the queue. No model discretion is involved; the rule is
in `loop/backlog.py::plan_turn`:

```
turn % DECK_EVERY == 0    ->  deck        rebuild the slides from claims.json
turn % REVIEW_EVERY == 0  ->  review      quality + security over what landed since the last one
any implemented           ->  merge       second-opinion the gate, then move main
any accepted              ->  implement   build it test-first on idea/<slug>
any proposed              ->  evaluate    judge one idea properly, accept/reject/park
otherwise                 ->  council     refill the queue
```

The two cadences come first for the same reason: a slide rebuild or a security pass that only
runs when the queue happens to be empty is one that never runs on the busiest night, which is
exactly the night it matters. `merge` outranks `implement` so `main` keeps moving, which keeps
every idea branch a clean descendant of `main` and every merge a simple one. `council` is last
because it is by far the most expensive turn type — it only fires when the queue is genuinely dry.

A `review` turn files what it finds as backlog entries rather than fixing anything, so every fix
still goes through `implement` and gets a test and a gate. The one exception is dataset content in
a tracked file, which it removes on the spot.

## What the data-safety gate covers, and what it does not

`loop/push_gate.sh` is the one implementation of "may this ref leave this machine".
`loop/run.sh` and `loop/bootstrap.sh` both source it; neither contains a `git push` of its own.
It scans the working tree once, and then scans **each branch against its own tree** before
pushing that branch — a branch is not trusted because some other branch was clean when it was
checked out. A branch that fails is skipped and named in the log; the clean ones still go.

This is not theoretical caution. On the loop's first night, `bootstrap.sh` pushed every branch
before any check ran, and one of them carried a postmortem quoting three raw identifiers straight
out of `export/` — to a shared repository, for a dataset licensed challenge-use-only. A push
cannot be taken back.

**Two push paths still bypass it, and saying so is the point of this section.**

- `refs/entire/checkpoints/*` are exempt by name and the `entire` CLI pushes them itself. Backlog
  `i0007`: 36 such refs are already on `origin` and carry session transcript. `.gitignore` does
  not apply to refs and the suite never enumerates them. Containment is
  `entire configure --local --skip-push-sessions`; removing what is already there needs a human.
- `loop/prompts/merge.md` and `loop/prompts/deck.md` push by hand. They are told to run the
  suite first, but that is an instruction to a model, not a gate.

The suite catches the placeholder shapes (`PII_*_<n>`, `<ENTITY_*_<n>>`). Raw un-redacted ids are
covered by backlog `i0005`; until that lands, the review turn greps for them by hand and
`_common.md` tells every turn to `grep -c` any concrete-looking token against `export/` before
writing it down.

## Branches

- `main` — the integration branch and the branch the team presents from. It moves only through
  a merge turn, and only after `make test`, `make all` and `router.gates` all pass **again** on
  the merged result. If they do not, the merge is reverted and the idea is marked `blocked`.
- `idea/<slug>` — one per idea, always branched from `main`. Kept after merging, so the history
  of a rejected or blocked attempt survives.

`origin` is a **shared team repository**. Before every turn the supervisor fetches `main`. If a
teammate has pushed and the branches have genuinely diverged, merging is suppressed for the rest
of the night: the loop keeps generating, evaluating and implementing on idea branches, and
leaves `main` alone for a human to resolve. Being behind is fine and fast-forwards silently;
being diverged stops the loop from touching shared history.

## What the loop spends

This machine authenticates with an **OAuth subscription** (Claude Max 20×), not an API key, and
extra usage is disabled. Two consequences worth being precise about:

- **Nothing is billed per token.** The `total_cost_usd` each turn reports is a notional
  API-equivalent figure. It is kept as a ceiling because it is the best available proxy for how
  much quota a turn ate, but no invoice exists behind it.
- **The limit is a wall, not a meter.** With extra usage off, exhausting the quota makes calls
  fail rather than cost money. So the loop cannot run up a bill — the worst case is that it
  stops early.

There are two windows, and they behave differently:

| Window | Behaviour | Handled by |
|---|---|---|
| **5-hour** | refills on its own | not a stopping condition — `sleep_for_quota` waits for the reset and retries the same turn, up to 3 times per turn |
| **7-day** | does not refill before the defense | `LOOP_MAX_SEVEN_DAY_PCT` stops the loop and leaves you the remainder |

Hitting the 5-hour limit therefore pauses the loop, it does not end it. The turn counter does
not advance across a retry, so nothing is lost or double-counted.

## Stopping conditions

Whichever hits first, from `loop/config.env`: wall clock (`LOOP_HOURS`), turn count
(`LOOP_MAX_TURNS`), the notional effort ceiling (`LOOP_BUDGET_USD`), the weekly quota guard
(`LOOP_MAX_SEVEN_DAY_PCT`), the `.loop-stop` kill switch, or three consecutive turns that changed
neither the queue nor any branch. Each turn also has its own `timeout` — INT first so the result
json still gets flushed, SIGKILL 60s later — so a hung turn costs one slot rather than the night.

A seventh condition is the one you hope never fires: `loop/state/state.json` not yielding a
number for `cost_usd`. The supervisor stops rather than continue, because it cannot know what it
has spent. That file is written by the turn agents, so the supervisor treats it as untrusted
input throughout — every `python -c` in `run.sh` is single-quoted and takes its values from
argv, and `tests/test_supervisor_input.py` fails the build if one is written any other way.
See ADR-012.

## Ultracode

`ULTRACODE_<TURNTYPE>=1` in `loop/config.env` makes that turn type author and run a multi-agent
workflow for every substantive task instead of working solo. It is per turn type on purpose:
breadth is worth a multiple of the quota on a judgement call, and worth nothing on a
deterministic gate. The shipped default has it on for `evaluate` only. See the comments in
`config.env` for the reasoning per type.

## What to read in the morning

| File | What it tells you |
|---|---|
| `loop/JOURNAL.md` | one line per turn: type, time, ok / timed out / rc, cost |
| `docs/IDEAS.md` | the backlog rendered — what was proposed, judged, built, merged, and why |
| `docs/REVIEW-LOG.md` | each review turn's scope, findings, and what it checked and found clean |
| `presentation.html` | the current deck, rebuilt every `DECK_EVERY` turns |
| `git log --oneline main` | what actually landed |
| `loop/logs/turn-NNN-*.json` | one turn's full result, including its final message |
| `loop/logs/turn-NNN-*.prompt.md` | the exact prompt that turn was given |
| `entire recap` | the same night from the session store, with more detail |

Anything with status `blocked` is waiting for you specifically.

## Editing the loop's behaviour

The prompts in `loop/prompts/` are the behaviour. `_common.md` is prepended to every turn and
holds the repo's hard rules and the orientation ritual; the five turn files hold one job each.
Change a prompt and the next turn picks it up — no restart needed.

`loop/backlog.py` is the only writer of `loop/state/*.json`. Never hand-edit those files while
the loop is running.

## The one thing this loop cannot do

It cannot answer `docs/OPEN-QUESTIONS.md` Q1 or Q2. Those need a message from the organizers,
not more analysis. Every prompt says so, because an agent left alone with an unanswerable
question will otherwise spend a whole turn on it.
