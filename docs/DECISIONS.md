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
