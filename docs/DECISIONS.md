# DECISIONS — architecture decision record

One entry per decision that is expensive to revisit. Each states the decision, why, and the
**measurement** that forced it. A decision without a measurement behind it does not belong here.

Numbers cited are pinned in `docs/CONTRACTS.md`. Open items live in `docs/OPEN-QUESTIONS.md`.

Status values: **Accepted** | Superseded by ADR-nnn | Proposed.

---

## ADR-001 — Route once, at admission

**Status:** Accepted

**Decision.** The router picks a model **once per trajectory, at admission** — before the first
assistant token exists — using only the system message, the first user message, and the tool
block (`input[:first_user_msg_index+1]` plus `tools`). It never re-routes mid-trajectory.

**Why.** Anything later is post-treatment. Once the model is chosen, every subsequent item in
the trace is a *consequence* of that choice, so conditioning on it opens a collider path and the
resulting estimate answers no causal question. Routing at admission is also the only decision
point that actually exists in the logged data: `AGENTS.md` states one model serves all calls of
a trajectory, so mid-trajectory switching is not observed anywhere in the export and could not
be evaluated off-policy even if we wanted it.

**The measurement.** 46.5% of estimated token mass lives after the first user message
(post-treatment 10,517,503 vs pre-treatment 12,114,376 = 46.5% / 53.5%). Nearly half the trace
is model-generated and therefore unusable as a routing input. Every column in
`results/feature_manifest.json` carries a `pre_treatment` flag for exactly this reason, and the
`leak-hunter` agent exists to enforce it.

**Consequence.** `n_turns`, `n_calls`, `n_err`, `trailing_err`, `gross_tok`, `cache_read_tok`
and everything derived from tool outputs are legitimate **outcomes** and legitimate **cost
accounting**, and are never inputs to the routing decision.

---

## ADR-002 — Route within model family only

**Status:** Accepted

**Decision.** The router may move a trajectory between arms **inside** the claude family. It
never proposes a claude -> gpt or gpt -> claude switch.

**Why.** Family is not a free choice in this data — it is determined by the toolset the
trajectory was admitted with, and the two families are given *different tools*. Comparing across
them compares two different jobs, not two models on one job. On top of that, the gpt lane has no
price sheet (`docs/OPEN-QUESTIONS.md` Q2), so a cross-family recommendation could not be priced
even if it were identified.

**The measurement.** Family is perfectly separated by the tool block, 0 exceptions:
`apply_patch`/`shell_command` -> gpt 245/245, `bash`/`file_read`/`file_edit`/`file_write` ->
claude 755/755. Any feature that recovers the toolset recovers the treatment exactly, so a
cross-family model is circular by construction. The cross-family evidence we do have is one job
wearing a crowd's clothing: pooled over the 23 clean cross-family jobs, claude 33/157 = 21.0%
friction vs gpt 21/72 = 29.2%, but **134 of those 229 runs are the single job
`crons/heartbeat/`**.

**Consequence.** The routable pool is 675/1000 trajectories = 197,551,096 est. gross tokens =
59.0% of the bill. The gpt lane (245 runs, 32.3% of the bill) is reported as out of scope with
its token mass shown, not silently dropped.

---

## ADR-003 — A deterministic router, not an LLM judge

**Status:** Accepted

**Decision.** The routing rule is deterministic code over the pre-treatment feature matrix. No
LLM is called at routing time, and no LLM produces the outcome label.

**Why.** An LLM in the loop is unauditable, unreproducible offline, needs an API key
(`AGENTS.md` requires the whole thing run offline with no keys), and — decisively — the LLM
signals we tested do not measure what they claim to. A rule we can print is also a rule a judge
can check, which is the first judging criterion: "real, nameable structure in the traces beats a
black box."

**The measurement.** Two independent failures. (a) An LLM judge over the final assistant message
says "done" in 623/762 cases and admits any problem in 8 — it is a politeness detector, not an
outcome label. (b) Text-keyword error detection scores 2.8%–49% precision against exit-code
ground truth, depending on the keyword set. The label we use instead is constructed from tool
exit codes: 10,422 tool outputs, 9,633 resolvable (92.4%), 474 errors, 259 `y_fric` positives.

**Consequence.** The outcome is `y_fric` / `y_proc` from `results/labels.jsonl`. `empty_final`
and `final_text_length` are banned outright — 238/238 empty finals are gpt, a serialization
artifact that encodes family perfectly.

---

## ADR-004 — Knapsack under a budget, not a probability threshold

**Status:** Accepted

**Decision.** Adoption is chosen as a **knapsack**: rank admissible trajectories by predicted
saving per unit of predicted friction risk and take them until the risk budget is spent. We do
not route "every trajectory whose `p_fric` is below tau".

**Why.** A fixed threshold spends the risk budget wherever the predictor happens to be
confident, which is uncorrelated with where the money is. Spend in this export is extremely
concentrated, so a threshold rule burns most of its risk allowance on trajectories that cannot
move the bill. A knapsack also produces a *curve* rather than a point, and the headline artifact
the challenge asks for is "the cost-quality frontier, not a single point".

**The measurement.** Spend concentration on `gross_tok`: top 10 trajectories = 17.7% of the
bill, top 50 = 40.5%, top 100 = 54.7%, top 500 = 90.2%. Half the bill is 100 trajectories.
Meanwhile the exclusions are cheap to state and mostly small: gpt lane 245 runs (32.3%),
non-admissible logged arm 23 runs (7.7%), already on the cheapest arm 71 runs (7.0%),
pre-treatment images 9 runs (1.7%).

**Consequence.** `results/routes.jsonl` carries `p_fric` and both costs per row so the frontier
can be re-swept at any budget without refitting. `tau_pass` survives in the `gates` object as a
diagnostic, not as the adoption rule.

---

## ADR-005 — Report bounds, never point estimates

**Status:** Accepted

**Decision.** Every off-policy quantity ships as an interval or a partial-identification bound.
No IPS estimate, no doubly-robust estimate, no friction point estimate anywhere — not in
`claims.json`, not on a slide.

**Why.** Reweighting buys nothing here and the study is underpowered for the contrast we care
about. Publishing a point estimate would imply a precision the design cannot deliver, and the
challenge scores honesty about that explicitly.

**The measurement.** Two, and each is sufficient on its own. (a) The Kish effective sample size
per arm equals `n_a` **exactly** (331/281/113/112/71/69/20/2/1) — the propensity weights are
degenerate, so IPS and DR reduce to the naive within-arm means while inheriting all their
variance-inflation caveats. There is no reweighting gain to be had. (b) The minimum detectable
effect on the **best-powered** arm pair is 11.4pp; the friction differences we are chasing are
smaller than that. A point estimate finer than 11.4pp is noise with a decimal point.

**Consequence.** `results/estimates.json` carries bounds. Any number quoted anywhere must exist
in `results/claims.json` under a stable key, so a reader can trace a slide back to its source.
The `estimator-critic` agent defaults to "not established" for the same reason.

---

## ADR-006 — The job key is the literal cron path, and nothing else

**Status:** Accepted

**Decision.** Runs are clustered by `cron_path` from `results/jobkey.jsonl`, and **only rows
with `literal: true` may be joined to each other**. Rows with `bucket` `"pii"` or `"none"` are
never joined.

**Why.** We need a recurrence key to compare arms on the same job. Every other candidate key in
this export is either an identifier (so it groups nothing) or a redaction placeholder (so it
groups the wrong things).

**The measurement.** The tempting key — the `# === Cron memory: PII_URL_N` field — is 763/763
placeholders over only **18 distinct values**, with `PII_URL_3` alone appearing 306 times. It
looks like a job id and is not one; placeholders are renumbered per request, so no `PII_*` token
joins across rows. In the other direction, the system-prompt hash has 871 distinct values over
1000 rows — a row id, not a feature. And `scripts/load_trajectories.py::group_trajectories()`
is broken outright: it hashes `first_user_text[:2000]`, but 978/1000 first user messages exceed
2000 characters of shared boilerplate, so it returns 953 collision groups for 1000 independent
trajectories. n is 1000.

**Consequence.** 784/1000 runs carry a literal `"Cron path:"` across 354 distinct paths; 152
fully-literal jobs cover 374 runs; 28 of those span >=2 arms (245 runs) and 23 span both
families (229 runs). 216 runs have no cron path and 410 have a `PII_` placeholder inside the
path — those are excluded from every job-level comparison and the exclusion is reported.

---

## ADR-007 — stdlib + numpy + matplotlib, and turn reconstruction from scratch

**Status:** Accepted

**Decision.** The whole pipeline runs on `.venv/bin/python` with stdlib, numpy and matplotlib.
sklearn and scipy are banned. Trajectory structure is reconstructed by `router/recon.py`, not by
the starter kit's grouping.

**Why.** Reproducibility on a laptop with no network and no keys, and because the starter kit's
two central abstractions are both wrong for this export.

**The measurement.** (a) sklearn and scipy are not installed and installing them would make the
pipeline unreproducible for the rest of the team; every estimator we need is a few lines by hand
(AUC from ranks is six). `.claude/settings.json` warns on any attempt to install them. (b) The
starter kit counts `naive_tok` = 22,631,879 estimated tokens. The correct billed figure, once
each turn's re-sent prefix is counted, is `gross_tok` = 334,729,910 across 10,845 turns — about
**14.8x** the naive number, split 92.0% cache read (308,074,571) / 8.0% cache write
(26,655,339). Building the cost story on the naive count would have been wrong by more than an
order of magnitude.

**Consequence.** `router/recon.py` owns the turn-boundary rule (a boundary opens at the start of
every maximal run of model-produced items) and every downstream cost number derives from it.
`group_trajectories()` is banned; `.claude/settings.json` warns on any file that writes its name.

---

## ADR-008 — A pretrained transformer is allowed on this branch, and it does not help

**Status:** Accepted (branch `text-classifier-finetune`). Supersedes ADR-007 **for this branch only**;
`main` keeps stdlib + numpy + matplotlib.

**Decision.** `torch` (CPU wheel) and `transformers` are installed on this branch, and
`answerdotai/ModernBERT-base` is fine-tuned on the pre-treatment text. The result is reported as a
**negative**: the router on `main` does not change.

**Why.** ADR-007 bans sklearn and scipy for reproducibility, and `AGENTS.md` requires the pipeline to
run offline on a laptop with no GPU and no API keys. Neither forbids a package: the `AGENTS.md`
constraint is about *runnability*, and a CPU fine-tune of a 149M-parameter encoder satisfies it.
The guard in `.claude/settings.json` matches only `sklearn|scipy|scikit-learn` and does not fire on
`torch`. The question was worth the compute because "a better text model would find the signal" is
the first objection any judge raises against a hand-built feature matrix, and the honest way to
close it is to run it.

**The measurement.** Three findings, each sufficient on its own.

(a) *Frozen representation is worse than hashed n-grams.* Cluster-CV by literal cron path, spend
target `log1p(effective billed tokens)`: ModernBERT-base mean-pooled at L=1024 gets OOF R² 0.367 and
captures 78.2% of the oracle dollars at k=100, against 0.566 / 92.2% for hashed normalised word
1-2-grams and 0.510 / 87.0% for the 35 tabular columns. Blending toward the transformer degrades
capture monotonically (92.2% → 78.2%). ModernBERT is a masked LM, not a sentence encoder, and the
routing signal is the presence of specific tokens (`silent`, `nothing`, `search`) that bag-of-words
reads directly.

(b) *Fine-tuning does not close the gap.* 5 epochs, L=512 head+tail, 229 training rows grouped by
cron path, trunk LR 3e-5 / head LR 1e-3: held-out spend R² **0.150**, friction AUC **0.625** — against
0.566 and 0.688 for the hashed n-grams on the same population. The spend head is still underfit
(prediction sd is 30% of target sd after round 1, and the R² only moves 0.091 → 0.150 across a 4x
compute increase).

(c) *The arm-conditioned Δ head learns noise.* After the architecture fix that gave the arm its own
unmixed path, Δ̂ = p̂(sonnet) − p̂(opus) has mean +0.104 and sd 0.114 on held-out rows — but
`corr(Δ̂, train-half stratum gap) = +0.042`, and Δ̂ ranges over [−0.253, +0.638] where the real
stratum gap ranges over [+0.046, +0.157]. It invents roughly 4x more heterogeneity than exists.
Using it as a decision weight *destroys* value: at a matched held-out friction bound of 0.008, the
best spend predictor alone yields $22.80 while the same predictor divided by (1 + 5·Δ̂) yields
**$10.51** — a 54% loss from the Δ term alone.

**Consequence.** This is the fifth risk-side mechanism to fail out-of-sample (per-cell Δ, per-pair δ,
job-history features, hashed-n-gram p̂, fine-tuned Δ̂), and the first tested with a real pretrained
model. The constraint is not representational: within the claude lane the arm effect is +6.1pp
against an MDE of 8.2pp, so the sampling noise of any selection signal exceeds the signal. The
router therefore stays what it is on `main` — rank by predicted spend, take top-k, target
`claude-sonnet-5` — and no model-class change is pending. Reproduction notes, the mirror workaround
for the blocked `huggingface.co`, and the measured CPU costs are in `router/textclf/README.md`.

## ADR-009 — The prose gate finds orphans, and does not pretend to find wrong keys

**Status:** Accepted

*(ADR-008 is the text-classifier negative result, which is still on `text-classifier-finetune`.
This one is numbered 009 so the two do not collide when that branch merges.)*

**Decision.** `python -m router.verify` enforces the "every quoted number is in `claims.json`"
contract on the five git-tracked user-facing artifacts. It reports ORPHANS — numerals no claim
has the value of — and it deliberately does not assert that a bound numeral is bound to the
RIGHT key.

**Why the weaker claim.** "Resolve every numeric literal to a claims key" sounds stronger and is
in fact vacuous. 28 claims carry small-int values 0..12 (`corpus.n_arms` = 9, `model.cv.folds` =
5, `basis.chars_per_token` = 4, ...), so README step numbers and a `[0, 1]` range in the deck all
"resolve" by coincidence while carrying no claim at all. Value matching cannot separate a claim
numeral from a structural one, so the gate does not claim it can. Measured while building it:
perturbing the console's MDE from `11,4` to `11,7` is NOT caught, because 11.7 happens to be
`recon.turns.per_line.claude-fable-5`. The negative control in `tests/test_verify.py` uses `11,9`
and says why in a comment.

**The two parser rules, both load-bearing.** (a) LOCALE IS DECLARED, never sniffed:
`router/console.html` is German, so `11,4` there is 11.4 (`refusal.mde_best_powered_arm_pair.pp`)
while `10,845` in the English deck is 10845 (`recon.turns.total`). A single heuristic over the
digits would have to get one of them wrong. (b) PERCENT: shares live in `claims.json` as
fractions and are quoted as percents, so a numeral followed by `%` may also bind to claim/100 —
`41.0%` is `policy.refused.gross_share` = 0.4098. Only when followed by `%`, or the tolerance
would be loose enough to bind anything. A literal binds when it is a correct rounding of the
claim at the precision it was written with.

**Scope.** Prose only. `<script>` and `<style>` bodies, HTML comments, fenced code blocks and
inline code spans are blanked before extraction — they hold chart geometry, row data and shell
commands, and every literal in them that matched a claim matched a coincidental 0, 1 or 2.
Structural numerals that survive that (the hackathon dates, `Python 3.10+`, panel numbers, gate
ordinals, the 95% level) live in a capped `NOT_A_CLAIM` allowlist, one justification each, in the
idiom `tests/test_data_safety.py` already uses for documentary placeholders.

**Consequence.** `make test` is the gate (`tests/test_verify.py`); `make verify` is the readable
report of what bound to what. It is deliberately NOT part of `make all` — `all` rebuilds the
claims table, and a deck half-way through an edit would wedge the pipeline for the wrong reason.

## ADR-010 — A leak detector is tested against the pattern, not against the repo

**Status:** Accepted

**Decision.** `tests/test_data_safety.py` gains two things: the concrete-placeholder pattern now
accepts any *attached* separator between a prefix and its serial and normalises what it finds
back to the underscore form, and a new shape-free check extracts opaque-looking identifiers from
every repo file and fails on any the export also contains. Both are backed by tests that
exercise the detector directly rather than by the corpus tests alone.

**Why.** A corpus test passes for two different reasons — the repo is clean, or the detector is
blind — and it cannot tell you which. Three raw identifiers and, separately, a placeholder
serial written in a slash shorthand all reached a shared repository while this suite was green.
The suite was not lying; it was answering a narrower question than everyone read it as
answering.

**The measurement.** (a) The old pattern required a trailing `_<digits>`; the shorthand that sat
on `main` from commit 5082319 to c05e078 put the digits after a slash, and the value it encoded
occurs in the export in the thousands. The generalised pattern matches all eight attached
separators tested and normalises them to one token, so the DOCUMENTARY allowlist stays at 4 of
its cap of 8. (b) Whitespace is excluded from the separator class, and that was measured, not
assumed: with whitespace allowed the file failed on its own docstring, where "…, 3 of them"
reads as a serial. (c) The opaque-token extractor yields 24 candidates across 114 repo files
once `data:…;base64,` payloads are stripped — without the strip, `presentation.html`'s inlined
577 KB PNG alone shatters into ~9,000 fragments of the same shape. Exactly one of the 24 occurs
in the export, and it is an arm identifier from `router.pricing.OBSERVED_ARMS`; exempting the
arm table structurally rather than by allowlist leaves 23 candidates and 0 hits. (d) Both corpus checks were confirmed by
seeding a leak into an untracked scratch file: each failed, naming the file and a masked shape.

**Consequence.** Failure messages print a masked shape — letters to `a`/`A`, digits to `#` —
and never the token, because the token is the leak. The exemption list tracks the arm table, so
adding an arm cannot be mistaken for widening a safety carve-out. What this still does not
cover: an identifier shorter than 16 characters, or one that appears in the export only in a
form the repo rewrote.

## ADR-011 — Every push path goes through one gate, and it scans the ref it is pushing

**Status:** Accepted

**Decision.** `loop/push_gate.sh` is the single implementation of "may this ref leave this
machine". `loop/run.sh` and `loop/bootstrap.sh` both source it and neither contains a `git push`
of its own. The gate scans the working tree once, then scans **each branch against its own tree**
via `DATA_SAFETY_SCAN_REF` before pushing that branch. A branch that fails is skipped and named;
the clean ones still go.

**Why.** The gate scanned whatever was checked out and then pushed every ref under `refs/heads`.
A branch an earlier turn updated and moved off was pushed having never been scanned — which is
the exact incident the gate was written to prevent. `loop/bootstrap.sh` had no gate at all: the
gate was added to `run.sh` in 8257642 and never backported, so re-running bootstrap reproduced
the night-one leak verbatim. Both were found by the turn-4 security review (`i0009`, `i0010`).

**The measurement.** Against a throwaway repository with a real local remote and three branches,
one of which carries a seeded marker: the marked branch does not appear on the remote, the other
two do, and the refusal is logged by name — `tests/test_push_gate.py`, four assertions on what
actually landed rather than on what the function printed. Mutating the per-ref check to a no-op
kills three of the four. The ref-awareness itself is tested in both directions against this
repository: a clean ref passes, and a `git commit-tree` commit carrying a concrete placeholder
fails and names the offending path. That probe commit is deliberately left dangling, so no push
path can ever see it. `make test` 93 OK.

**Consequence.** Two claims in the docs were false and are now restated rather than deleted:
`loop/README.md` and `docs/DATA-SAFETY-DEBT.md` both said nothing leaves the machine past a red
suite. Three paths bypassed it. The remaining uncovered paths — `entire/*` checkpoint refs
(`i0007`) and the hand-written pushes in the merge and deck prompts — are now named in both
files instead of being implied away. A false safety claim is worse than no claim: it is what let
this gate's own predecessors stop looking.

## ADR-012 — The supervisor treats its own state file as untrusted input

**Status:** Accepted

**Decision.** No `python -c` in `loop/run.sh` may be written with a double-quoted argument.
Every one of them is single-quoted, so the shell cannot splice anything into it, and every value
it needs travels as `argv`. When `loop/state/state.json` does not yield a number for `cost_usd`,
the loop STOPS rather than continue past a guard it could not evaluate.

**Why.** `run.sh` is the one process in this project that runs outside every tool-permission
boundary — it is the thing that launches the sandboxed turn, so nothing sandboxes it. The budget
check built its Python by interpolation:

    "$PY" -c "import sys;sys.exit(0 if float('$spent') >= float('$LOOP_BUDGET_USD') else 1)"

`$spent` is the `cost_usd` field printed straight out of a file that every turn agent rewrites
while running with `bypassPermissions`. A turn that wrote a crafted string into its own cost
field would have that string executed as Python by the supervisor — and `stop_reason` is the
first thing the loop body calls, so it would be re-executed at the top of every later iteration.

**The measurement.** `tests/test_supervisor_input.py` runs the same probe payload down both
paths. Through the old interpolated form it executes and writes its sentinel file; through
`stop_reason` as it now stands the sentinel is absent and the loop stops with
`unreadable cost_usd`. Before the fix 4 of the file's 9 tests failed; after it, 9 pass and the
suite is 91 (was 82).

**Consequence.** Five readers were converted, not just the one that was exploitable:
`state_field`, `usage_pct`, `quota_reset_epoch`, `is_rate_limited` and the per-turn `COST`
extractor. Two behavioural changes came with it. A new `state_int` coerces the turn counter,
because `[ "$turn" -ge 40 ]` on a non-integer is a shell error and an erroring guard reads as
"do not stop" — the max-turns limit was switchable off by writing a string. And the budget check
is now three-valued (over / under / unreadable) and fails closed on the third, which is the only
one of the three that a corrupt state file can reach by accident.

## ADR-013 — Review outranks the deck on a tie, and the deck is deferred rather than dropped

**Status:** Accepted

**Decision.** `loop/backlog.py::plan_turn` tests the review cadence before the deck cadence. A
turn that satisfies both runs the `review`; the deck runs on the following turn instead. Both
cadences share one guard (`_cadence_due`), so `0` — or any value `<= 0` — disables either of them
identically.

**Why.** The periodic quality-and-security pass is the loop's only control that looks at what the
loop itself is doing. Silently disabling it is a strictly worse failure than showing an
out-of-date slide, and the previous ordering disabled it silently.

**The measurement.** With the deck check first, a review that lost a tie waited for its next
multiple — which, whenever `DECK_EVERY` divides `REVIEW_EVERY`, is also a deck turn, so it lost
again forever. Over turns 1..40 with a non-empty queue, `review` fired **zero** times for
`(5,10)`, `(4,4)`, `(5,5)` and `(3,6)`. The shipped `(5,4)` did fire, but lost turn 20 — it worked
by coincidence of the two numbers being coprime-ish, not by design. Separately, `DECK_EVERY=0`
raised `ZeroDivisionError` inside a command substitution `run.sh` does not exit-check, so the
supervisor saw an empty `TYPE` and logged `unknown turn type`.

The naive fix — swap the two checks — was written, measured and rejected in the same turn: it
moves the starvation onto the deck, which then fires **zero** times for `(4,4)` and `(5,5)`. The
deferral clause is what makes the trade one-sided. Over the full grid `DECK_EVERY` 1..12 ×
`REVIEW_EVERY` 1..12, turns 1..40: **0 review multiples missed**, and deck staleness bounded at
`DECK_EVERY + 1` turns for every pair with `REVIEW_EVERY >= 2`.

**Consequence.** `tests/test_backlog_plan.py` (14 tests) pins both halves — review is never
missed, and the deck is never starved by the rule that protects review. The bound is asserted
across the whole grid, not just the five pairs that happened to be tried.

---

## ADR-014 — A difference of two one-sided upper bounds is not a gap, and nothing may print it as one

**Status:** Accepted

**Decision.** `router.ope.non_inferiority_bound` returns `(None, None, hi, diag)` — a ONE-SIDED
UPPER bound and nothing else. Both legs of the frontier comparison are built from it, so
`upper(router) - upper(mixture)` bounds the true difference in neither direction. Three
consequences, all now enforced in code rather than in a reviewer's memory:

1. The claims key is named for the arithmetic it holds:
   `frontier.gated_router.upper_bound_minus_hull_upper_bound_pp`. It was
   `frontier.gated_router.gap_above_hull_pp`, which invited every reader to quote it as the gap.
   The value is unchanged; only the name is, so no prose can quote it without the key name
   contradicting the sentence.
2. `results/NUMBERS.md` #5 states both levels and then says **do not subtract them**. The chart
   annotation does the same: the double-headed arrow is labelled with the two bounds, not with
   the distance between them.
3. Neither `presentation.html` nor the chart quotes the difference anywhere.

**What replaces it — the arithmetic ceiling.** The identified delta is
`sum(W * 1[switched] * (m - y))` with `m, y` in `[0, 1]` and spend weights summing to 1, so its
magnitude cannot exceed the share of spend the policy moves. The gated router moves
`ope.routed.switched.spend_share` = 3.6% of est. spend, and the hull at matched spend sits at
-3.83 pp. No policy at this tau can reach that hull however well it routes. This is both true and
strictly stronger than the discarded claim: it reframes the negative result from "our routing rule
is bad" to "we set tau too conservatively for the frontier test to be informative", which is a
statement about a knob we chose rather than about a rule the data rejected.

`policy_point` now carries `switched_spend_share` on every plotted point, and it is a column of
`results/frontier.csv`, so the ceiling is a property of each point rather than a sentence about
one of them. Cross-check: the value computed there (0.035888) equals
`ope.routed.switched.spend_share` computed on the independent OPE path.

**Also corrected.** The comparator at the router's spend is not "one cheap arm". The hull is
non-decreasing and convex over the admissible single-arm policies, so a hull value strictly
between two vertices is an interpolation — that is, a RANDOMISATION. At $375.45 it lies between
`claude-sonnet-5` ($172.47) and `claude-opus-5` ($686.55), so the thing the router loses to is a
mixture with substantial weight on the most expensive policy on the chart.

**What was NOT done, deliberately.** No real gap number was produced. Getting one requires a joint
job-clustered bootstrap with both policies evaluated inside the same replicate, emitted through
`router/report.py` under its own key. Pasting a number computed in a scratch session would
reintroduce exactly the defect this ADR closes. Until that exists, the project quotes no gap.

**Do not** reach for "4.59 pp is below the 11.4 pp MDE, so it is not significant."
`refusal.mde_best_powered_arm_pair.pp` is an unweighted arm-pair rate contrast — a different
estimand on a different weighting. Asserting it would replace one untraceable comparison with
another.

## ADR-015 — A data-safety check that cannot run is RED, and the gate checks a receipt

**Status.** Accepted (turn 11).

**Context.** `python -m unittest` exits 0 when every test skips. Every content check in
`tests/test_data_safety.py` skipped when it could not enumerate repo files, and
`RepoCarriesNoOpaqueExportToken` skipped outright when `export/` was absent. `loop/push_gate.sh`
read only the exit code. So:

```
DATA_SAFETY_SCAN_REF=refs/heads/no-such-branch python -m unittest tests.test_data_safety
→ OK (skipped=6)    exit 0
```

and the gate pushed that branch having enumerated zero files. No exotic input was needed — a ref
deleted at the remote between enumeration and scan, a dropped object, or a name `git` refuses to
parse all land in the same place. On a machine without the 101 MB export the headline
raw-identifier detector was inert and green for the same reason. This is the `i0010` fail-open
class reappearing inside the gate written to close it.

**Decision.** Two changes, and both are needed — either alone leaves the hole open.

1. **The suite fails closed when the gate is asking.** `DATA_SAFETY_STRICT=1` (implied by
   `DATA_SAFETY_SCAN_REF`) turns every environmental excuse into a failure: a git error while
   enumerating raises, an enumeration that yields zero files raises, and an absent or empty
   `export/` raises. Outside the gate they are still skips, so `make test` stays green on a
   laptop without the export. The distinction is the point: "git is unavailable" is plausible on
   a developer machine and is never plausible for a ref the gate just read out of `for-each-ref`.

2. **The gate stops trusting the exit code.** The suite prints one machine-readable line —
   `DATA-SAFETY-RECEIPT ref=… files=N ran=N skipped=N failed=N` — and `ds_receipt_ok` refuses the
   branch when the line is absent, when `files` is 0, when `ran` is 0, or when `failed` is not 0.
   A green exit code with nothing behind it no longer reads as a clean branch. The receipt is
   printed only by the gate runner in the module's `__main__`, so the gate invokes
   `python -m tests.test_data_safety`, not `python -m unittest`.

**Consequences.** The ref scan now runs only the four ref-aware classes; the two working-tree
classes cannot travel through a ref and re-walked the export once per branch for nothing. Measured
on this repo, a per-ref scan goes 21 tests / 5.0 s to 16 tests / 4.6 s — a small saving, and
honestly so: the opaque-token detector's own streamed pass over the export dominates and is
genuinely per-ref, because the candidate token set differs per branch.

**What this does not fix.** The 16-character floor in `RepoCarriesNoOpaqueExportToken`
(backlog `i0018`) — a detector that now provably runs still cannot match the 11-character ids it
was written for. And branch names still reach `git push` as bare arguments (backlog `i0023`).
Both are recorded rather than implied away; `loop/README.md` and `docs/DATA-SAFETY-DEBT.md` name
the first as a live limit in the same commit as this ADR, because a safety sentence that is wrong
in the reassuring direction is what let the previous review turns stop looking.

**Evidence.** `tests/test_push_gate.py::TheSuiteFailsClosedWhenItCannotRun` drives the real suite
at an unresolvable ref and at an emptied export, both directions, and asserts the receipt.
`TheGateRefusesAScanThatDidNotRun` drives the real `push_all_branches` against a real local
remote with three stub scanners — no receipt, `files=0`, `ran=0` — and asserts on what LANDED on
the remote, with the honest scanner as the negative control. Against the pre-fix code, 9 of those
10 cases are red; the one that passes is the control asserting a missing export still only skips
outside the gate.
---

## ADR-016 — The opaque-token detector uses two windows, because one floor cannot serve both

**Status:** Accepted

**The defect (backlog i0018).** `RepoCarriesNoOpaqueExportToken` was written to catch the class of
leak that a shape-matching detector cannot see: a raw, un-redacted identifier copied out of a
trajectory. Its docstring cites the three such identifiers removed in commit `af4e78b` as the
incident it prevents. It matched none of them. They are 11 characters long; its window started at
16. The class was green and detected nothing of its stated class — the same failure mode as a
detector that skips instead of failing, and worse than having no detector, because a green control
is what lets the next reviewer stop looking.

**The obvious fix was measured and rejected.** Lowering the single window's floor from 16 to 8
takes the repo from 27 candidates to 88, and **18 of those 88 occur in the export** — model
identifiers, feature-column names, dates, all colliding with the corpus by coincidence. The corpus
test would be permanently red for reasons that are not leaks, which is how a control gets muted.

**Decision — two windows with different rules.**

| window | length | charset | extra rejections |
|---|---|---|---|
| long | 16–64 | `[A-Za-z0-9_-]` | — |
| compact | 8–15 | `[A-Za-z0-9]` (no separator) | `word1024` / `1024word`; `<digits>x<digits>` |

The separator exclusion is the whole trick and it was not guessed: **every one of the 18 false
positives carries a `-` or a `_`**. A raw identifier is not hyphenated, so excluding separators
below 16 costs no real detection and drops the false positives from 18 to 1. The two remaining
rejections are shape rules, measured against this tree:

- `word1024` / `1024word` is a name with a size attached, not an identifier. Deliberately narrow —
  it fires only on a *single* letter/digit boundary, so `L1024tokens` stays a candidate. Widening
  it to any letter/digit mixture would discard exactly the shape the compact window exists for.
- `<digits>x<digits>` is a pixel dimension. `1920x1080` sits in `loop/JOURNAL.md` **and** in the
  export, by pure coincidence, and it was the last false positive left.

Both are shape rules rather than allowlist entries, so neither can silently grow the way
`DOCUMENTARY` can — which is the same reasoning as ADR-010.

**Measured after the change.** 36 candidates over 126 files, 0 of which the export knows; runtime
unchanged at ~3 s, because the export scan is one streamed pass regardless of set size. `make
test` 151 OK, `make all` green (301 claims), `router.gates` GREEN 4 pass / 2 warn / 0 fail.

**The test that keeps this honest.** `OpaqueDetectorCatchesTheIncidentItWasWrittenFor` replays the
incident instead of asserting that it is handled: it reads both sides of `af4e78b` out of history,
takes every token that commit removed, keeps the ones the export actually knows (5 of 5), and
asserts each is caught by one of this file's detectors — two by the placeholder shapes, three by
the opaque windows. Before this change it failed naming three masked shapes. No token is written
into any tracked file: the fixture is the commit, and failures print masked shapes only, per
ADR-010.

**Known limit, recorded rather than fixed.** The compact window still requires both a letter and a
digit, so a purely alphabetic identifier of any length is invisible to it. Nothing in this corpus
has that shape, so lowering that bar now would buy false positives and no detection; the day one
appears, this is the line to revisit.
## ADR-017 — A branch name is untrusted input, and the push is pinned to the sha that was scanned

**Status:** Accepted (turn 12)

**Context.** `loop/push_gate.sh` took every name out of `git for-each-ref` and handed it
straight to `git rev-parse` and `git push` as a bare argument, with no `--` separator
anywhere; `tests/test_data_safety.py` did the same with `DATA_SAFETY_SCAN_REF` for
`git ls-tree` and `git cat-file`. The reflex objection is that a branch name is not
attacker input. In this repo it is: every turn runs an agent with `bypassPermissions` and
unrestricted bash, and while `git branch` and `git checkout -b` refuse an option-shaped
name, `git update-ref` does not.

Measured on git 2.55, in a throwaway repository, before anything was changed:

- `git update-ref 'refs/heads/--receive-pack=/tmp/x/pwn.sh' HEAD` is accepted, and
  `for-each-ref` lists the ref.
- `git rev-parse` echoes that name back and exits 0, so the gate's `|| continue` after
  `rev-parse` did not catch it.
- `git push -q origin '--receive-pack=/tmp/x/pwn.sh'` parses it as an **option** and spawns
  the named program locally, as the operator's user. The payload ran. The push then does
  whatever `push.default` says, so it is not a no-op either — in the gate's exact call it
  only appeared harmless because the throwaway repo's `main` had no upstream. With an
  upstream configured, as `main` has here, the payload fires.
- The same name makes `git ls-tree` exit 129, which the per-ref scan reads as an
  infrastructure error rather than as a leak — the fail-open path ADR-015 is about.

Two of the four shapes the review asked for turn out to be unreachable end to end:
`update-ref` refuses a name containing a space or a glob character, and `for-each-ref`
warns and ignores a ref file written by hand. They are still exercised against the
validator, because the validator must not depend on that staying true.

**Decision.**

1. `ds_ref_name_ok` validates every name before it becomes a git argument. It is an
   **allowlist** — `[A-Za-z0-9._/-]`, no leading dash — with `git check-ref-format` behind
   it as a second gate, not as the first one. check-ref-format answers a different question
   and returns 0 for `refs/heads/--receive-pack=x`; that was measured too. A name that fails
   stops that branch loudly and marks the cycle refused; the clean branches still push.
2. The branch is resolved **once**, with `--end-of-options`, to a commit, and everything
   downstream uses the sha: the scan is `DATA_SAFETY_SCAN_REF=<sha>` and the push is
   `git push -- <sha>:refs/heads/<name>`. This closes the option shape and the
   scan-then-push window in the same move — the gate used to scan a *name* and then push
   whatever that name pointed at by the time the push ran.
3. `tests/test_data_safety.py` refuses an option-shaped or whitespace-carrying
   `DATA_SAFETY_SCAN_REF` at import rather than scanning nothing, and its git calls carry
   separators: `--end-of-options` for `cat-file`, a trailing `--` for `ls-tree`.
   `ls-tree` does **not** accept `--end-of-options` before the tree-ish — it silently
   returns nothing, which is precisely the failure being defended against, so the
   separator goes where a pathspec would.

**Consequences.** A ref name that is not an ordinary branch name in this repo is never
pushed. The cost is that a legitimately exotic name would also be refused; the negative
control in `tests/test_push_gate.py` pins the six shapes this repo actually uses, so a
validator that refuses everything — and therefore silently stops backing the night's work
up — cannot pass.

**Scope.** Pre-existing debt, not a regression: `main` carried the identical bare-argument
form at `loop/run.sh:182` and `loop/bootstrap.sh:100` before the gate was extracted. The
supervisors' other git calls take `$INTEGRATION_BRANCH` and `$BASE_BRANCH`, which come from
`loop/config.env` and not from `for-each-ref`; they are out of scope here and untouched.

**Evidence.** `tests/test_push_gate.py::BranchNamesAreValidatedBeforeTheyReachGit` (7 tests)
and `::ScanRefIsNotHandedToGitAsAnOption` (3). The refusal test is paired with a **positive
control** that the payload does fire when the name is a bare argument, without which the
refusal would prove only that the throwaway repo was broken. Mutation-tested: neutering the
character class fails 8 of the 21, and pushing by name instead of by scanned sha fails 1.
`make test` 150 OK (main is 140), `make all` green (301 claims), `router.gates` GREEN
4 pass / 2 warn / 0 fail, `router.verify` PASS on 44 numerals.
---

## ADR-018 — An artifact the prose gate cannot read is BLOCKED, never skipped into a PASS

**Status:** Accepted

**Decision.** `router/verify.ARTIFACTS` carries a fourth field, `required`. A required artifact
that is not on disk goes to `Report.missing`, `Report.ok` is false whenever that list is
non-empty, and `python -m router.verify` exits **2 — BLOCKED**, the same code an absent claims
table returns. All five declared artifacts are git-tracked and shipped, so all five are
`required=True`; `tests/test_verify.py` pins that, so marking one optional to quiet a red gate
breaks a test rather than passing silently.

**Why.** The gate's surface is a declaration, and a declaration that narrows itself when a file
moves is worse than no declaration: a narrowed gate prints the same green banner as a complete
one. `presentation.html` carries 36 of the 44 bound numerals and only arrived at the repo root at
turn 5, so any later deck turn that renamed or relocated it would have switched the prose contract
off for the entire deck while `make verify` still read PASS.

**The measurement.** On the tree as shipped, `python -m router.verify` binds **44 numerals across
5 artifacts** and exits 0. Deleting `presentation.html` and `README.md` from a scratch copy
previously printed `PASS - 5 numeral(s) across 3 artifact(s)` and exited **0**; it now names both
paths as `REQUIRED and absent from this tree`, prints the BLOCKED block, and exits **2**. That
before-state is the turn-8 review's reproduction, re-run here rather than taken from the note.

**Two things the exit contract had to get right.** BLOCKED dominates a partial orphan list: when
the run cannot read its whole surface, the orphans it did find are an incomplete answer, so the
code is 2 and not 1. But it does not swallow them — `print_report` emits the orphan block as well,
and a test drops one artifact while perturbing a numeral in another and asserts both appear.

**Consequence.** `router.verify` now distinguishes three outcomes rather than two: clean (0), a
number that does not trace (1), and a surface it could not read (2). The `skipped` list exists for
a genuinely optional artifact and is empty today; nothing may be added to it without an entry
here saying why that file stopped being shipped.

---

## ADR-019 — The state store refuses an unreadable file, and the supervisor checks the exit code

**Status:** Accepted

**Context.** `loop/backlog.py` is the only writer of `loop/state/`, and every turn agent — running
with bypassPermissions — is a writer of that same directory by way of this CLI. It read both files
with a bare `json.loads` and indexed `i["id"]` and `i["status"]` with no `.get`. `loop/run.sh`
captures its stdout by command substitution under `set -uo pipefail` (no `-e`), which **discards
the exit code**. So a half-written file produced a traceback, the traceback produced an empty
scalar, and the loop stopped with:

    unknown turn type '' -- stopping

The right failure, reported as the wrong cause. Measured on the pre-fix code: `turn --bump`,
`plan` and `list` each exit 1 with empty stdout on a truncated state file, on a truncated backlog
file, and on an item missing its `status` key. This is the same shape as ADR-012 (the supervisor
trusting its own state file) and the same fail-open class as ADR-015 (a check that cannot run
reading as a pass).

**Decision, two halves that only work together.**

1. *The store fails closed and by name.* A file that exists and does not parse, or parses to the
   wrong shape, raises `StateError` and exits `3` — distinct from argparse's `2` — with a sentence
   on **stderr** and **nothing on stdout**, because the supervisor cannot tell an empty answer from
   an absent one. `_check_backlog` validates every field the module later indexes without a
   `.get`: item is an object, `id` matches `i\d+` and is unique, `status` is one of `STATUSES`,
   `title` is a string, `notes` is a list.
2. *The supervisor checks the code.* Every capture of the CLI goes through `backlog_scalar`, which
   returns non-zero and prints nothing on a failed **or empty** read, and logs the CLI's own
   message; the writes go through `backlog_run`. `TURN`, `TYPE` and the queue block of the turn
   prompt each stop the night with the real cause in the log and in the journal.

**The tempting fix is the bug.** Defaulting a corrupt `state.json` — the obvious "be robust" move —
sets `cost_usd` to `0`, and `stop_reason`'s ceiling reads exactly that field. One corrupt file
would buy the night an unlimited budget. So: **absent is not corrupt.** A missing file defaults (the
first turn of a night has none); a file that exists and does not parse is an error and is never
rewritten from defaults.

**Also closed.** `next_id` used to swallow a non-numeric id and restart at `i0001` — the id such a
file is most likely to already hold, so the recovery path minted a duplicate. Validation makes the
`try/except` unnecessary, and the ordering total.

**The durable guard is textual.** `loop/run.sh` may contain no raw `$("${BACKLOG[@]}" ...)` capture;
only the two guards' `"$@"` pass-through is exempt, and the test pins that there are exactly two.
A behavioural test cannot hold when someone adds a seventh call site next week; this one does.

**Evidence.** Both halves mutation-controlled: reverting `loop/backlog.py` to its previous version
fails 11 of the 19 tests, and turning off the empty-output check in `backlog_scalar` fails 1 more.
`make test` 159 OK (was 140), `make all` green (301 claims), `router.gates` GREEN 4 pass / 2 warn /
0 fail, `router.verify` PASS on 44 numerals. No claim value moved: this is loop infrastructure and
quotes no export number.
