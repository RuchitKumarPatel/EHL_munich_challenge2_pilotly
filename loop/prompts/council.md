## Your job this turn: COUNCIL — refill the idea queue

The backlog is dry. Generate the next batch of candidate improvements.

### 1. Establish what is actually weak right now

Do not brainstorm in the abstract. Read the project's own record of its weaknesses first:

- `results/NUMBERS.md` — read all five headline numbers, and especially the sections
  "What is deliberately absent" and the arithmetic corrections. Number 5 says the router does
  not yet beat randomising between two arms. That is the largest open scientific problem in the
  project; ideas that attack it are worth more than ideas that polish what already works.
- `docs/COST-LEVERS.md` — the DEAD and IMPOSSIBLE list. Anything here is closed.
- `docs/OPEN-QUESTIONS.md` — Q1 and Q2 are for the organizers. Do not propose analysis that
  claims to resolve them.
- The judging criteria in `AGENTS.md`: routing insight, evaluation rigor, presentation clarity —
  plus the special prize for the best off-policy-evaluation insight.

### 2. Run the council

Invoke the `council` skill to deliberate on this question:

> Given the judging criteria and the measured weaknesses above, what are the highest-value
> changes we can make to this project in the remaining hours before the defense? Weigh routing
> insight, evaluation rigor and presentation clarity against what one agent can actually finish
> and verify in a 40-minute working turn.

Give the council the concrete context: the five headline numbers, the DEAD lever list, the two
open questions, and the constraint that anything proposed must be buildable and verifiable
tonight with stdlib + numpy + matplotlib.

If the `council` skill fails or is unavailable, fall back to dispatching four parallel agents
with distinct lenses — routing insight, evaluation rigor, presentation clarity, adversarial
reviewer — and synthesise their answers yourself.

### 3. Filter hard before recording anything

For each candidate the council produces, check in this order and drop it on the first failure:

1. `entire search "<the idea>"` — already tried in a previous session?
2. Is it in `docs/COST-LEVERS.md` as DEAD or IMPOSSIBLE?
3. Does it contradict an Accepted ADR in `docs/DECISIONS.md` without bringing a measurement?
4. Does it try to answer Q1 or Q2 from `docs/OPEN-QUESTIONS.md` with more analysis?
5. Is it a duplicate of something already in the backlog (any status, including `rejected`)?
6. Could one agent build **and verify** it in about 40 minutes?

### 4. Record

Add **4 to 6** survivors with `loop/backlog.py add`. Each `--why` must name which judging
criterion it serves and what measurement would show it worked. A `--why` that reads like
marketing copy is a bad entry; rewrite it.

Then `loop/backlog.py render`, and commit `docs/IDEAS.md` and `loop/state/backlog.json` with
explicit paths.

If fewer than 3 candidates survive the filter, that is a finding, not a failure — record the
survivors, and add one note explaining what the council converged on that was already closed.
