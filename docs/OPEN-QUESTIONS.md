# OPEN QUESTIONS — the two unknowns that block a headline number

Both of these are questions for the **organizers**, not for more analysis. No amount of work on
this export resolves either. Until they are answered, the answers below are what the
presentation must say — and saying them clearly is scored, not penalized ("Name where the
estimate can fail — that is scored, not penalized", `AGENTS.md`).

Status: **both OPEN** as of the last update to this file.

---

## Q1 — Are the model ids real names or pseudonyms, and is any tier order published?

**Why it blocks everything.** `AGENTS.md` states the `model` field is an **anonymized** id:
"real names hidden, tier order not published". The nine ids — `claude-opus-5`,
`claude-sonnet-5`, `claude-fable-5`, `claude-opus-4-8`, `gpt-5.6-sol`, `gpt-5.6-terra`,
`gpt-5.6-luna`, `claude-opus-4-6`, `claude-sonnet-4-6` — *look* like a tier ladder, and the
naming convention of the real vendors invites the reading that `opus > sonnet > fable` in both
capability and price. That reading is an assumption, not a fact from this dataset.

**What breaks if the ids are pseudonyms.** The SIGN of every saving is undetermined. A route
from `claude-opus-5` to `claude-fable-5` is a saving only if `fable` is the cheaper id. If the
mapping is permuted — and a deliberate anonymization has every reason to permute it — the same
route is a *cost increase* of the same magnitude. The arithmetic, the cache model and the
friction bounds are all unaffected; only the direction of the conclusion flips. A headline of
the form "we cut spend by X%" is therefore not derivable from this export alone.

**What we can still say without the answer.** Everything sign-free:

- the size of the routable pool (675/1000 trajectories = 197,551,096 est. gross tokens = 59.0%
  of the bill) — this is a *fraction of spend that a router can touch*, independent of price order
- the cache decomposition (92.0% read / 8.0% write) and the effective multipliers
  (rho=1.00 -> 0.192x, rho=0.55 -> 0.555x vs fully-uncached)
- the friction bounds per arm pair, stated as bounds
- the shape of the cost-quality frontier as a function of an *assumed* price ratio, plotted over
  a range of ratios rather than at one point

**What would resolve it.** One line from the organizers: either the id -> real-model mapping, or
a statement of the intended price/tier order among the anonymized ids, or an explicit "the ids
are permuted, do not infer order". Any of the three unblocks the sign.

**Our stance until then.** Report savings as a function of an assumed price ratio, always name
the assumption inline, and show the frontier over a swept ratio rather than a single point. Do
not print a single headline dollar figure without the sheet named next to it.

---

## Q2 — Is there any price sheet for the `gpt-5.6-*` ids?

**Why it blocks a third of the bill.** `AGENTS.md` says pricing is an assumption and points at
`scripts/pricing.json` "if the organizers post prices". The gpt lane is 245/1000 trajectories
and **32.3% of estimated gross tokens**. With no sheet for `gpt-5.6-sol` / `-terra` / `-luna`,
that third of the bill has no dollar value at all — not an uncertain one, an absent one.

**What breaks.** Two things, and they are different:

1. *Pricing the gpt lane at all.* We cannot state what the gpt trajectories cost, so we cannot
   state a total bill, so we cannot state a percentage saving against a total bill. Every
   money figure we publish is therefore scoped to the claude lane unless a gpt sheet appears.
2. *Cross-family routing.* The design already routes **within family only** (see
   `docs/DECISIONS.md`), so the missing gpt sheet does not block the router itself. But it does
   block any claim of the form "move work from the gpt lane to the claude lane saves money" —
   and the cross-family friction contrast we do have is thin anyway: pooled over the 23 clean
   cross-family jobs, claude 33/157 = 21.0% friction vs gpt 21/72 = 29.2%, with 134 of those 229
   runs being the single job `crons/heartbeat/`. That is one job, not a population.

**What we can still say without the answer.** Token-denominated results for the whole export,
and dollar-denominated results for the claude lane under a named sheet. The 32.3% gpt share is
reported as an explicitly unpriced remainder, with its token mass shown, rather than silently
excluded or silently priced at a guess.

**What would resolve it.** A populated `scripts/pricing.json` covering the gpt ids, or an
organizer statement that no sheet exists for them.

**Our stance until then.** Every dollar figure names its assumed sheet in the same sentence.
The gpt lane is shown as a labelled unpriced block on the frontier chart, never dropped from the
denominator without a note.

---

## Standing caveat that is NOT an open question

All token counts are ESTIMATES: `tok(x) = len(json.dumps(x)) // 4`. The export has no `usage`
field, so no tokenizer disagreement can be resolved against ground truth here. This is settled,
not open — but every user-facing string must still say "estimated".
