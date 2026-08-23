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
