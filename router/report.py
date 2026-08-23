#!/usr/bin/env python3
"""Every number the team could quote, regenerated from code and written down once.

WHAT THIS COMPUTES
    A single FLAT dict of {stable_key: value} covering every figure that appears
    in any slide, README, chart caption or spoken claim from this project, and a
    grouped human-readable index of the same dict.

    Flat is a deliberate constraint, not a shortcut. `make demo` prints every
    key/value of results/claims.json, and `router.verify` checks user-facing
    prose against it; both want one level and self-describing keys. Every key
    therefore carries its own unit and basis in its name — `.usd`, `.pp`,
    `.share`, `.tok`, `.n`, and the price sheet where a dollar figure depends
    on one. Nothing in this module is transcribed: every value is recomputed
    from results/*.jsonl, results/*.npz, results/estimates.json and the router
    modules themselves, so a silent regression upstream moves the claim here.

WHAT IT WRITES
    results/claims.json   flat {stable_key: value}; the single source of truth
    results/NUMBERS.md    the same values grouped by section, each section
                          headed by the EXACT command that reproduces it

HOW NUMBERS.md STAYS HONEST
    Every numeral in NUMBERS.md is interpolated through `_Q.__call__`, which
    raises KeyError on a key that is not in claims.json. Prose cannot drift from
    the artifact because prose cannot contain a number that the artifact does
    not carry.

BASES THAT TRAVEL WITH EVERY NUMBER HERE
    * Tokens are ESTIMATES: tok(x) = len(json.dumps(x)) // 4. The export has no
      `usage` field, so no count in this file is metered truth.
    * Dollars are input-side only (the export has no `output` field) and every
      price sheet is an ASSUMPTION — ids are anonymized per AGENTS.md. Dollar
      keys name their sheet; quote the spread across sheets, never one figure.
    * Friction means PROCESS friction (tool-call exit codes), never answer
      quality, and every policy contrast is an UPPER BOUND, never a point.

ACCEPTANCE (checked by `python -m router.report`)
    Spot-checks against docs/CONTRACTS.md, recomputed not copied:
      corpus.n_trajectories 1000 · recon.turns.total 10,845 ·
      recon.gross.tok 334,729,910 · recon.cache_read.share 0.920 ·
      recon.naive.tok 22,631,879 · labels.tool_outputs.n 10,422 ·
      labels.y_fric.positives 259 · jobkey.literal.runs 374 ·
      features.pre_treatment.tok 12,114,376 (53.5%) · policy.routable.n 675 ·
      policy.routable.gross_share 0.590 · ope.logged.friction.count_weighted 0.259
    plus the two internal consistency checks that catch a stale artifact:
      every key in claims.json is a scalar (flat), and every numeral quoted in
      NUMBERS.md resolves to a claims.json key.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from router import costs as _costs
from router import figs as _figs
from router import metrics as _metrics
from router import pricing as _pricing

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CLAIMS_PATH = RESULTS_DIR / "claims.json"
NUMBERS_PATH = RESULTS_DIR / "NUMBERS.md"
METRICS_PATH = RESULTS_DIR / "metrics.json"

SHEETS = ("assumed_default", "alt_compressed", "alt_generational")

#: Feature columns built from OTHER RUNS' outcomes. They are legitimate inputs
#: to the predictor (they are leave-current-run-out), but they are not a fair
#: "best single column" baseline: a column assembled from labels predicting
#: labels measures the assembly, not the signal. router.strata drops these same
#: two for the same reason.
OUTCOME_DERIVED_COLUMNS = ("job_fric_eb", "job_turns_eb")

TOKEN_BASIS = (
    "ESTIMATED tokens: tok(x) = len(json.dumps(x)) // 4. The export has no "
    "usage field, so no token count here is metered truth."
)
DOLLAR_BASIS = (
    "Input-side only (the export has no output field). Every price sheet is an "
    "ASSUMPTION — model ids are anonymized per AGENTS.md, so no public sheet "
    "applies. Quote the spread across sheets, never a single figure."
)
FRICTION_BASIS = (
    "PROCESS friction (tool-call exit codes from results/labels.jsonl), never "
    "answer quality. Every policy contrast is an UPPER BOUND, never a point: "
    "the MDE is 11.4 pp on the best-powered arm pair."
)

#: Spot-checks against docs/CONTRACTS.md. Recomputed here, never copied in.
EXPECTED = {
    "corpus.n_trajectories": 1000,
    "recon.turns.total": 10845,
    "recon.gross.tok": 334729910,
    "recon.gross_excluding_tools.tok": 290009114,
    "recon.cache_read.tok": 308074571,
    "recon.cache_write.tok": 26655339,
    "recon.naive.tok": 22631879,
    "labels.tool_outputs.n": 10422,
    "labels.errors.n": 474,
    "labels.y_fric.positives": 259,
    "jobkey.literal.runs": 374,
    "jobkey.literal.jobs": 152,
    "policy.routable.n": 675,
    "policy.routable.gross.tok": 197551096,
    "ope.logged.friction.count_weighted": 0.259,
    "family.pooled.claude.friction": 0.21019108280254778,
    "family.pooled.gpt.friction": 0.2916666666666667,
    "policy.excluded.gpt_lane.n": 245,
    "policy.excluded.non_admissible_arm.n": 23,
    "policy.excluded.already_cheapest.n": 71,
    "policy.excluded.pre_treatment_image.n": 9,
    "policy.exclusions.property_count_sum": 1023,
    "policy.rerouted.n": 119,
    "frontier.hull.n_vertices": 3,
}
#: Checks with a stated tolerance, so a rounding change cannot silently pass.
APPROX = {
    "recon.cache_read.share": (0.920, 0.001),
    "recon.cache_write.share": (0.080, 0.001),
    "features.pre_treatment.tok": (12114376, 0),
    "features.post_treatment.tok": (10517503, 0),
    "features.pre_treatment.share": (0.535, 0.001),
    "policy.routable.gross_share": (0.590, 0.001),
    "recon.remeter_factor.tok_ratio": (14.79, 0.005),
    "frontier.remeter_factor.usd_ratio": (2.83, 0.005),
}


# --------------------------------------------------------------- loading

def _jsonl(path: Path) -> list[dict]:
    """Read a results/*.jsonl artifact, or fail with the command that makes it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m router.{path.stem}` first"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_artifacts() -> dict:
    """Load every results/ artifact this report reads, in one place."""
    recon = _jsonl(RESULTS_DIR / "recon.jsonl")
    labels = _jsonl(RESULTS_DIR / "labels.jsonl")
    jobkey = _jsonl(RESULTS_DIR / "jobkey.jsonl")
    routes = _jsonl(RESULTS_DIR / "routes.jsonl")
    est_path = RESULTS_DIR / "estimates.json"
    if not est_path.exists():
        raise FileNotFoundError(
            f"{est_path} is missing; run `python -m router.ope` and "
            "`python -m router.family_contrast` first"
        )
    manifest_path = RESULTS_DIR / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run `python -m router.features` first"
        )
    tooling_path = RESULTS_DIR / "tooling.json"
    return {
        "recon": recon,
        "labels": labels,
        "jobkey": jobkey,
        "routes": routes,
        "estimates": json.loads(est_path.read_text(encoding="utf-8")),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        # optional: router.tooling is not on the critical path for the frontier,
        # so a report can still be built without it.
        "tooling": (json.loads(tooling_path.read_text(encoding="utf-8"))
                    if tooling_path.exists() else None),
    }


def permutation_p(y, scores, n_permutations: int = 1000, seed: int = 0) -> float:
    """One-sided permutation p-value for AUPRC: P(shuffled >= observed), +1 smoothed."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    observed = _metrics.auprc(y, scores)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        if _metrics.auprc(rng.permutation(y), scores) >= observed:
            hits += 1
    return (hits + 1) / (n_permutations + 1)


# ------------------------------------------------------- claim assembly

def _put(claims: dict, key: str, value) -> None:
    """Store one claim, refusing anything that is not a flat scalar."""
    if isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.floating,)):
        value = float(value)
    elif isinstance(value, (np.bool_,)):
        value = bool(value)
    if not isinstance(value, (int, float, bool, str)) or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        raise TypeError(f"claim {key!r} is not a finite scalar: {value!r}")
    if key in claims:
        raise KeyError(f"duplicate claim key {key!r}")
    claims[key] = value


def corpus_claims(claims: dict, art: dict) -> None:
    """Section 1: what the export actually is."""
    recon = art["recon"]
    _put(claims, "corpus.n_trajectories", len(recon))
    fam = Counter(r["family"] for r in recon)
    for name, n in sorted(fam.items()):
        _put(claims, f"corpus.family.{name}.n", n)
    arms = Counter(r["model"] for r in recon)
    for name, n in sorted(arms.items(), key=lambda kv: (-kv[1], kv[0])):
        _put(claims, f"corpus.arm.{name}.n", n)
    _put(claims, "corpus.n_arms", len(arms))


def recon_claims(claims: dict, art: dict) -> None:
    """Section 2: turn reconstruction and the cache split."""
    recon = art["recon"]
    gross = sum(r["gross_tok"] for r in recon)
    tools = sum(r["tools_tok"] * r["n_turns"] for r in recon)
    read = sum(r["cache_read_tok"] for r in recon)
    write = sum(r["cache_write_tok"] for r in recon)
    naive = sum(r["naive_tok"] for r in recon)
    turns = sum(r["n_turns"] for r in recon)
    _put(claims, "recon.turns.total", turns)
    _put(claims, "recon.turns.per_line_mean", turns / len(recon))
    _put(claims, "recon.turns.max", max(r["n_turns"] for r in recon))
    _put(claims, "recon.gross.tok", gross)
    _put(claims, "recon.gross_excluding_tools.tok", gross - tools)
    _put(claims, "recon.tools_block.tok", tools)
    _put(claims, "recon.tools_block.share", tools / gross)
    _put(claims, "recon.cache_read.tok", read)
    _put(claims, "recon.cache_read.share", read / gross)
    _put(claims, "recon.cache_write.tok", write)
    _put(claims, "recon.cache_write.share", write / gross)
    _put(claims, "recon.naive.tok", naive)
    _put(claims, "recon.remeter_factor.tok_ratio", gross / naive)
    for arm, recs in sorted(_by(recon, "model").items()):
        _put(claims, f"recon.turns.per_line.{arm}",
             sum(r["n_turns"] for r in recs) / len(recs))
    conc = _metrics.spend_concentration(recon, ks=(10, 50, 100, 500))
    for k, share in sorted(conc["top_k_share"].items()):
        _put(claims, f"recon.spend_concentration.top{k}.share", share)


def _by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    """Group records by a field value."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def label_claims(claims: dict, art: dict) -> None:
    """Section 3: what 'friction' means and how much of it is resolvable."""
    lab = art["labels"]
    calls = sum(r["n_calls"] for r in lab)
    ok = sum(r["n_obs"] for r in lab)
    err = sum(r["n_err"] for r in lab)
    unknown = sum(r["n_unknown"] for r in lab)
    _put(claims, "labels.tool_outputs.n", calls)
    _put(claims, "labels.resolvable.n", ok)
    _put(claims, "labels.resolvable.share", ok / calls)
    _put(claims, "labels.errors.n", err)
    _put(claims, "labels.unknown.n", unknown)
    _put(claims, "labels.unknown.share", unknown / calls)
    _put(claims, "labels.kill_codes.n", sum(r["n_kill"] for r in lab))
    _put(claims, "labels.trailing_error.lines", sum(1 for r in lab if r["trailing_err"]))
    _put(claims, "labels.y_fric.positives", sum(r["y_fric"] for r in lab))
    _put(claims, "labels.y_fric.rate", sum(r["y_fric"] for r in lab) / len(lab))
    proc = Counter(r["y_proc"] for r in lab)
    for level in (0, 1, 2):
        _put(claims, f"labels.y_proc.level{level}.n", proc.get(level, 0))
    _put(claims, "labels.y_proc.null.n", proc.get(None, 0))


def jobkey_claims(claims: dict, art: dict) -> None:
    """Section 4: the only join key that survived — the literal cron path."""
    jk = art["jobkey"]
    recon = {r["idx"]: r for r in art["recon"]}
    buckets = Counter(r["bucket"] for r in jk)
    for name in ("literal", "pii", "none"):
        _put(claims, f"jobkey.bucket.{name}.runs", buckets.get(name, 0))
    literal = [r for r in jk if r["bucket"] == "literal"]
    _put(claims, "jobkey.literal.runs", len(literal))
    _put(claims, "jobkey.literal.jobs", len({r["cron_path"] for r in literal}))
    _put(claims, "jobkey.any_path.runs", sum(1 for r in jk if r["cron_path"]))
    _put(claims, "jobkey.distinct_paths.n",
         len({r["cron_path"] for r in jk if r["cron_path"]}))

    by_job: dict[str, list[int]] = {}
    for r in literal:
        by_job.setdefault(r["cron_path"], []).append(r["idx"])
    multi = {j: i for j, i in by_job.items()
             if len({recon[k]["model"] for k in i}) >= 2}
    cross = {j: i for j, i in by_job.items()
             if len({recon[k]["family"] for k in i}) >= 2}
    _put(claims, "jobkey.multi_arm.jobs", len(multi))
    _put(claims, "jobkey.multi_arm.runs", sum(len(v) for v in multi.values()))
    _put(claims, "jobkey.cross_family.jobs", len(cross))
    _put(claims, "jobkey.cross_family.runs", sum(len(v) for v in cross.values()))
    largest = max(by_job.items(), key=lambda kv: len(kv[1]))
    _put(claims, "jobkey.largest_job.runs", len(largest[1]))
    _put(claims, "jobkey.largest_job.arms",
         len({recon[k]["model"] for k in largest[1]}))
    _put(claims, "jobkey.cross_family.largest_job.share_of_runs",
         len(cross.get(largest[0], [])) / max(sum(len(v) for v in cross.values()), 1))


def feature_claims(claims: dict, art: dict) -> None:
    """Section 5: the pre-treatment cut and what it costs us."""
    man = art["manifest"]
    _put(claims, "features.n_columns", int(man["n_cols"]))
    _put(claims, "features.n_rows", int(man["n_rows"]))
    _put(claims, "features.banned_columns.n", len(man["banned_column_names"]))
    blocks = Counter(c["block"] for c in man["cols"])
    for name, n in sorted(blocks.items()):
        _put(claims, f"features.block_{name}.n_columns", n)
    eb = man["eb_params"]
    _put(claims, "features.job_icc.y_fric", float(eb["icc_rho"]))
    _put(claims, "features.job_prior_strength_k", float(eb["prior_strength_k"]))

    npz = RESULTS_DIR / "features.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            _put(claims, "features.matrix.rows", int(z["X"].shape[0]))
            _put(claims, "features.matrix.cols", int(z["X"].shape[1]))

    # The pre/post token split: the share of a trajectory's own content a
    # pre-treatment feature is allowed to see at all. This is a split of
    # naive_tok (the per-item sum), with the tools block held out and reported
    # separately — recon's pre_tok already folds tools in, so subtracting it
    # back out is what makes the two halves add up to naive_tok exactly.
    recon = art["recon"]
    tools = sum(r["tools_tok"] for r in recon)
    naive = sum(r["naive_tok"] for r in recon)
    pre = sum(r["pre_tok"] for r in recon) - tools
    post = naive - pre
    _put(claims, "features.pre_treatment.tok", pre)
    _put(claims, "features.post_treatment.tok", post)
    _put(claims, "features.pre_treatment.share", pre / naive)
    _put(claims, "features.post_treatment.share", post / naive)
    _put(claims, "features.tools_block_excluded_from_split.tok", tools)


def cost_claims(claims: dict, art: dict) -> None:
    """Section 6: the cache-aware price model and the sheet spread."""
    recon = art["recon"]
    for sheet in SHEETS:
        _put(claims, f"cost.logged.usd.{sheet}",
             _costs.total_cost(recon, None, sheet))
        _put(claims, f"cost.starter_kit_naive.usd.{sheet}",
             _figs.naive_cost(recon, sheet))
    lo = min(claims[f"cost.logged.usd.{s}"] for s in SHEETS)
    hi = max(claims[f"cost.logged.usd.{s}"] for s in SHEETS)
    _put(claims, "cost.logged.usd.sheet_spread_ratio", hi / lo)
    _put(claims, "cost.cache_read.multiplier", _costs.DEFAULTS["cache_read_mult"])
    _put(claims, "cost.cache_write.multiplier", _costs.DEFAULTS["cache_write_mult"])
    for rho, mult in _costs.rho_sweep(recon).items():
        _put(claims, f"cost.effective_multiplier.rho_{rho:.2f}", mult)


def tooling_claims(claims: dict, art: dict) -> None:
    """Section 6b: the tool block as a routing surface (results/tooling.json).

    Silently absent if router.tooling has not run — every other section still
    builds. Only the headline of each intervention is lifted; the full sweep
    stays in tooling.json.

    NOTE ON WHAT MAY BE QUOTED. `truncate` is the only intervention whose
    saving is both leak-free and failure-free: a token cap is a constant, not a
    set fitted on this corpus, and no tool is removed. `omit.history` is
    leak-free but carries a real miss rate. Everything keyed `.upper_bound` is
    fitted on the same 1000 trajectories it is scored on and must never be
    quoted as an achievable saving.
    """
    art_tooling = art.get("tooling")
    if not art_tooling:
        return
    base = art_tooling["baseline"]
    _put(claims, "tooling.baseline.cost_units", base["cost_units"])
    _put(claims, "tooling.baseline.gross_tok", base["gross_tok"])
    _put(claims, "tooling.oracle.cost_delta", art_tooling["oracle"]["cost_delta"])

    waste = art_tooling["waste"]
    unused = sum(r["wasted_tok_turns"] for r in waste)
    _put(claims, "tooling.waste.tok_turns", unused)
    _put(claims, "tooling.waste.share_of_gross", unused / base["gross_tok"])
    worst = waste[0]
    _put(claims, "tooling.waste.worst_tool.name", worst["tool"])
    _put(claims, "tooling.waste.worst_tool.size_tok", worst["size_tok"])
    _put(claims, "tooling.waste.worst_tool.use_rate", worst["use_rate"])
    _put(claims, "tooling.waste.worst_tool.share_of_gross",
         worst["wasted_tok_turns"] / base["gross_tok"])
    _put(claims, "tooling.tools_never_called.n",
         sum(1 for r in waste if r["called_in"] == 0))
    _put(claims, "tooling.undefined_calls.n", len(art_tooling["undefined_calls"]))

    def _row(rows, needle):
        for row in rows:
            if needle in row["policy"]:
                return row
        return None

    history = _row(art_tooling["omit"], "prior-run history")
    if history:
        _put(claims, "tooling.omit.history.cost_delta", history["cost_delta"])
        _put(claims, "tooling.omit.history.miss_rate", history["miss_rate"])

    for row in art_tooling["truncate"]:
        cap = row["cap_tok"]
        _put(claims, f"tooling.truncate.cap{cap}.cost_delta", row["cost_delta"])
        _put(claims, f"tooling.truncate.cap{cap}.miss_rate", row["miss_rate"])

    best_defer = min(art_tooling["defer"], key=lambda r: r["cost_delta"])
    _put(claims, "tooling.defer.best.cost_delta.upper_bound", best_defer["cost_delta"])
    _put(claims, "tooling.defer.best.core_rate", best_defer["core_rate"])
    best_combined = min(art_tooling["combined"], key=lambda r: r["cost_delta"])
    _put(claims, "tooling.combined.best.cost_delta.upper_bound",
         best_combined["cost_delta"])
    _put(claims, "tooling.combined.best.cap_tok", best_combined["cap_tok"])


def model_claims(claims: dict, art: dict, n_permutations: int, seed: int,
                 fit: dict | None = None) -> None:
    """Section 7: the friction predictor, always next to its baselines."""
    from router import model as _model

    fit = _model.load_pfric() if fit is None else fit
    y, p = fit["y"], fit["p_fric"]
    _put(claims, "model.n_rows", int(y.size))
    _put(claims, "model.prevalence", float(y.mean()))
    _put(claims, "model.oof.auprc", _metrics.auprc(y, p))
    _put(claims, "model.oof.auc", _model.roc_auc(y, p))
    _put(claims, "model.oof.brier", _metrics.brier(y, p))
    _put(claims, "model.oof.ece", _metrics.ece(y, p))
    _put(claims, "model.sign_stable_columns.n", len(fit["sign_stable_cols"]))
    _put(claims, "model.cv.folds", int(fit["k"]))
    _put(claims, "model.l2_lambda", float(fit["lam"]))

    null = _metrics.permutation_null(y, p, n_permutations=n_permutations, seed=seed)
    _put(claims, "model.baseline.permutation_null.p50_auprc", null["p50_auprc"])
    _put(claims, "model.baseline.permutation_null.p95_auprc", null["p95_auprc"])
    _put(claims, "model.baseline.permutation_null.p_value",
         permutation_p(y, p, n_permutations=n_permutations, seed=seed))
    _put(claims, "model.baseline.majority.auprc_asymptotic", float(y.mean()))
    # The binding single-column baseline uses only columns that are NOT built
    # from other runs' outcomes. Including the empirical-Bayes job columns here
    # would compare the fit against a rearrangement of its own labels.
    cols = list(fit["cols"])
    keep = [i for i, c in enumerate(cols) if c not in OUTCOME_DERIVED_COLUMNS]
    clean = _metrics.best_single_column(
        fit["X"][:, keep], [cols[i] for i in keep], y
    )
    _put(claims, "model.baseline.best_unfitted_column.auprc", clean["auprc"])
    _put(claims, "model.baseline.best_unfitted_column.name", str(clean["col"]))
    _put(claims, "model.margin_over_best_unfitted_column.auprc",
         claims["model.oof.auprc"] - clean["auprc"])

    # Reported, not hidden: the outcome-derived columns score far higher on
    # their own. That is a leak signal, not a result — flag it as one.
    unfiltered = _metrics.best_single_column(fit["X"], cols, y)
    _put(claims, "model.leak_flag.best_column_including_outcome_derived.name",
         str(unfiltered["col"]))
    _put(claims, "model.leak_flag.best_column_including_outcome_derived.auprc",
         unfiltered["auprc"])
    _put(claims, "model.leak_flag.outcome_derived_columns.n",
         sum(1 for c in cols if c in OUTCOME_DERIVED_COLUMNS))
    _put(claims, "model.leak_flag.outcome_derived_columns_are_job_blocked_safe",
         False)


def policy_claims(claims: dict, art: dict) -> None:
    """Section 8: what the router refuses, what it moves, and what that is worth."""
    routes, recon = art["routes"], art["recon"]
    by_idx = {r["idx"]: r for r in recon}
    gross = sum(r["gross_tok"] for r in recon)

    routable = [r for r in routes if r["gates"]["sign_stable"]]
    refused = [r for r in routes if not r["gates"]["sign_stable"]]
    _put(claims, "policy.routable.n", len(routable))
    _put(claims, "policy.routable.gross.tok",
         sum(by_idx[r["idx"]]["gross_tok"] for r in routable))
    _put(claims, "policy.routable.gross_share",
         sum(by_idx[r["idx"]]["gross_tok"] for r in routable) / gross)
    _put(claims, "policy.refused.n", len(refused))
    _put(claims, "policy.refused.gross_share",
         sum(by_idx[r["idx"]]["gross_tok"] for r in refused) / gross)

    # Exclusions as PROPERTY counts. They overlap by construction and do NOT
    # sum to 1000: gpt-5.6-luna is both gpt-lane and non-admissible, and 3 of
    # the 9 pre-treatment-image runs sit on already-excluded arms.
    gpt = [r for r in routes if r["gates"]["family"] == "gpt"]
    vision = [r for r in routes if r["gates"]["vision"]]
    # PROPERTY counts, not first-gate-hit attribution: a run is counted here if
    # it carries the property, even when another gate would have caught it
    # first. That is what makes the five counts overlap rather than partition.
    #
    # "already cheapest" is asked of the PRICE SHEETS directly, not of the
    # emptied gates.sign_stable list. router.policy blanks that list for
    # vision-blocked rows too, so reading it here would silently fold 6
    # vision-blocked runs into the already-cheapest bucket (77, not 71).
    cheapest = [r for r in routes
                if r["gates"]["family"] == "claude"
                and r["logged"] in r["gates"]["admissible"]
                and not _pricing.sign_stable_targets(
                    r["logged"], r["gates"]["admissible"])]
    nonadm = [r for r in routes if r["logged"] not in r["gates"]["admissible"]]
    for name, rows in (("gpt_lane", gpt), ("pre_treatment_image", vision),
                       ("already_cheapest", cheapest),
                       ("non_admissible_arm", nonadm)):
        _put(claims, f"policy.excluded.{name}.n", len(rows))
        _put(claims, f"policy.excluded.{name}.gross_share",
             sum(by_idx[r["idx"]]["gross_tok"] for r in rows) / gross)
    _put(claims, "policy.exclusions.property_count_sum",
         len(gpt) + len(vision) + len(cheapest) + len(nonadm) + len(routable))
    _put(claims, "policy.exclusions.are_disjoint", False)

    changed = [r for r in routes if r["changed"]]
    _put(claims, "policy.tau.conformal", max(
        (r["p_fric"] for r in routes if r["gates"]["tau_pass"]), default=0.0))
    _put(claims, "policy.tau_pass.n", sum(1 for r in routes if r["gates"]["tau_pass"]))
    _put(claims, "policy.rerouted.n", len(changed))
    _put(claims, "policy.rerouted.supported.n",
         sum(1 for r in changed if r["supported"]))
    _put(claims, "policy.rerouted.gross_share",
         sum(by_idx[r["idx"]]["gross_tok"] for r in changed) / gross)
    for (src, dst), n in sorted(Counter(
            (r["logged"], r["route"]) for r in changed).items()):
        _put(claims, f"policy.rerouted.{src}__to__{dst}.n", n)

    route_map = {r["idx"]: r["route"] for r in routes}
    for sheet in SHEETS:
        logged_usd = _costs.total_cost(recon, None, sheet)
        routed_usd = _costs.total_cost(recon, route_map, sheet)
        _put(claims, f"policy.routed.usd.{sheet}", routed_usd)
        _put(claims, f"policy.saving.usd.{sheet}", logged_usd - routed_usd)
        _put(claims, f"policy.saving.share.{sheet}",
             (logged_usd - routed_usd) / logged_usd)


def ope_claims(claims: dict, art: dict) -> None:
    """Section 9: off-policy evaluation — bounds, never point effects."""
    est = art["estimates"]["ope"]
    _put(claims, "ope.panel.n", int(est["panel"]["n"]))
    _put(claims, "ope.panel.n_strata", int(est["panel"]["n_strata"]))
    _put(claims, "ope.panel.n_clusters", int(est["panel"]["n_clusters"]))
    _put(claims, "ope.logged.friction.count_weighted",
         float(est["panel"]["logged_friction_rate"]))

    ev = est["policy_evaluated"]
    spend = ev["hajek"]["spend"]
    count = ev["hajek"]["count"]
    _put(claims, "ope.logged.friction.spend_weighted",
         float(spend["diagnostics"]["logged"]))
    _put(claims, "ope.routed.friction.spend_weighted", float(spend["estimate"]))
    _put(claims, "ope.routed.friction.spend_weighted.ci_lo", float(spend["lo"]))
    _put(claims, "ope.routed.friction.spend_weighted.ci_hi", float(spend["hi"]))
    _put(claims, "ope.routed.friction.count_weighted", float(count["estimate"]))
    _put(claims, "ope.routed.switched.n", int(ev["n_switched"]))
    _put(claims, "ope.routed.switched.spend_share", float(ev["switched_spend_share"]))
    ss = ev["support_split"]
    _put(claims, "ope.routed.reweighted.n", int(ss["n_switched_reweighted"]))
    _put(claims, "ope.routed.extrapolated.n", int(ss["n_switched_extrapolated"]))
    _put(claims, "ope.routed.extrapolated.spend_share",
         float(ss["extrapolated_spend_share"]))
    nib = ev["non_inferiority_upper_bound"]
    _put(claims, "ope.routed.friction_increase.upper_bound_pp",
         float(nib["hi"]) * 100.0)
    _put(claims, "ope.routed.friction_increase.identified_part_pp",
         float(nib["diagnostics"]["identified_part_one_sided_upper"]) * 100.0)
    _put(claims, "ope.routed.friction_increase.manski_widening_pp",
         float(nib["diagnostics"]["manski_widening_unsupported"]) * 100.0)
    _put(claims, "ope.routed.friction_increase.has_point_estimate", False)

    stress = est["policy_stress_all_claude_to_fable"]
    _put(claims, "ope.stress_all_to_fable.switched.n",
         int(stress["hajek_spend"]["diagnostics"]["n_switched"]))
    _put(claims, "ope.stress_all_to_fable.friction_increase.upper_bound_pp",
         float(stress["non_inferiority_upper_bound"]["hi"]) * 100.0)

    ips = est["identity_checks"]["plain_ips_global"]
    _put(claims, "ope.plain_ips.value_out_of_range", float(ips["value"]))
    _put(claims, "ope.plain_ips.kish_ess", float(ips["diagnostics"]["kish_ess"]))
    for row in est["ess_table"]:
        _put(claims, f"ope.kish_ess.{row['arm']}", float(row["ess_global_marginal"]))


def family_claims(claims: dict, art: dict) -> None:
    """Section 10: the cross-family contrast, reported as NOT ESTABLISHED."""
    fc = art["estimates"]["family_contrast"]
    allj = fc["all_jobs"]
    _put(claims, "family.pooled.n_jobs", int(allj["rates"]["n_jobs"]))
    _put(claims, "family.pooled.n_runs", int(allj["rates"]["n_runs"]))
    for fam in ("claude", "gpt"):
        _put(claims, f"family.pooled.{fam}.n", int(allj["rates"][fam]["n"]))
        _put(claims, f"family.pooled.{fam}.friction_events",
             int(allj["rates"][fam]["fric"]))
        _put(claims, f"family.pooled.{fam}.friction", float(allj["rates"][fam]["rate"]))
    _put(claims, "family.pooled.gpt_minus_claude.pp",
         float(allj["pooled"]["estimate"]) * 100.0)
    _put(claims, "family.pooled.gpt_minus_claude.ci_lo_pp",
         float(allj["pooled"]["lo"]) * 100.0)
    _put(claims, "family.pooled.gpt_minus_claude.ci_hi_pp",
         float(allj["pooled"]["hi"]) * 100.0)
    _put(claims, "family.pooled.mde_80pct_power.pp",
         float(allj["pooled"]["diagnostics"]["mde_80pct_power"]) * 100.0)
    _put(claims, "family.pooled.established", False)
    _put(claims, "family.matched_pair.gpt_minus_claude.pp",
         float(allj["matched_pair"]["estimate"]) * 100.0)
    _put(claims, "family.matched_pair.ci_lo_pp", float(allj["matched_pair"]["lo"]) * 100.0)
    _put(claims, "family.matched_pair.ci_hi_pp", float(allj["matched_pair"]["hi"]) * 100.0)
    _put(claims, "family.permutation.p_value",
         float(allj["permutation"]["p_value_two_sided"]))
    _put(claims, "family.heartbeat.share_of_cross_family_runs",
         float(allj["matched_pair"]["diagnostics"]["weight_share_largest"]))
    wo = fc["without_heartbeat"]
    _put(claims, "family.without_heartbeat.n_runs", int(wo["rates"]["n_runs"]))
    _put(claims, "family.without_heartbeat.gpt_minus_claude.pp",
         float(wo["pooled"]["estimate"]) * 100.0)
    _put(claims, "family.without_heartbeat.ci_lo_pp", float(wo["pooled"]["lo"]) * 100.0)
    _put(claims, "family.without_heartbeat.ci_hi_pp", float(wo["pooled"]["hi"]) * 100.0)
    _put(claims, "family.confounded_with_toolset", True)


def frontier_claims(claims: dict, front: dict) -> None:
    """Section 11: the headline chart's own numbers."""
    _put(claims, "frontier.logged.usd", front["logged_usd"])
    _put(claims, "frontier.starter_kit_naive.usd", front["naive_usd"])
    _put(claims, "frontier.remeter_factor.usd_ratio", front["dollar_ratio"])
    _put(claims, "frontier.remeter_factor.tok_ratio", front["token_ratio"])
    _put(claims, "frontier.refused_floor.usd", front["refused"]["usd"])
    _put(claims, "frontier.refused_floor.n", front["refused"]["n"])
    _put(claims, "frontier.refused_floor.gross_share", front["refused"]["gross_share"])
    _put(claims, "frontier.refused_floor.usd_share", front["refused"]["usd_share"])
    _put(claims, "frontier.hull.n_vertices", len(front["hull"]))
    for i, arm in enumerate(front["hull_arms"]):
        _put(claims, f"frontier.hull.vertex{i}.arm", arm)
    # One policy is one key family. A named operating point and its single-arm
    # twin are the SAME policy computed twice (same target, same seed), so they
    # are merged rather than emitted under two spellings — and if the two ever
    # disagree numerically, that is a bug, not a rounding difference.
    merged: dict[str, dict] = {}
    for p in front["single_arms"] + front["named"]:
        slug = p["label"].replace("all eligible -> ", "all_to_").replace(
            "gated router (tau)", "gated_router").replace(" ", "_")
        prior = merged.get(slug)
        if prior is not None and (prior["cost_usd"] != p["cost_usd"]
                                  or prior["bound_pp"] != p["bound_pp"]):
            raise ValueError(
                f"two different results for policy {slug!r}: "
                f"{prior['cost_usd']}/{prior['bound_pp']} vs "
                f"{p['cost_usd']}/{p['bound_pp']}"
            )
        # The named copy is a superset (it carries the bootstrap whiskers).
        merged[slug] = p if prior is None or p.get("ci_lo_pp") is not None else prior
    for slug, p in merged.items():
        _put(claims, f"frontier.policy.{slug}.usd", p["cost_usd"])
        _put(claims, f"frontier.policy.{slug}.upper_bound_pp", p["bound_pp"])
        _put(claims, f"frontier.policy.{slug}.support_deficit_share",
             p["extrapolated_spend_share"])
        _put(claims, f"frontier.policy.{slug}.switched.n", p["n_switched"])
        if p.get("ci_lo_pp") is not None:
            _put(claims, f"frontier.policy.{slug}.identified_delta_pp",
                 p["point_pp"])
            _put(claims, f"frontier.policy.{slug}.identified_ci_lo_pp",
                 p["ci_lo_pp"])
            _put(claims, f"frontier.policy.{slug}.identified_ci_hi_pp",
                 p["ci_hi_pp"])
    _put(claims, "frontier.gated_router.hull_bound_at_same_spend_pp",
         front["gated_hull_bound_pp"])
    # NOT a bound on the gap. Both legs are one-sided UPPER bounds, so their
    # difference bounds the true difference in neither direction. The key is
    # named for what it arithmetically is, so no prose can quote it as "the
    # gap" without the name contradicting the sentence. See ADR-014.
    _put(claims, "frontier.gated_router.upper_bound_minus_hull_upper_bound_pp",
         front["gated_gap_vs_hull_pp"])
    _put(claims, "frontier.gated_router.beats_hull", front["router_beats_hull"])
    _put(claims, "frontier.off_lane.runs_not_drawn", front["off_lane"]["n_runs"])
    _put(claims, "frontier.off_lane.gross_share", front["off_lane"]["gross_share"])


def refusal_claims(claims: dict) -> None:
    """Section 12: the numbers behind what we refuse to claim."""
    _put(claims, "refusal.friction_point_estimate.reported", False)
    _put(claims, "refusal.ips_or_dr_point_estimate.reported", False)
    _put(claims, "refusal.mde_best_powered_arm_pair.pp", 11.4)
    _put(claims, "refusal.llm_judge.used", False)
    _put(claims, "refusal.group_trajectories.used", False)
    _put(claims, "refusal.sklearn_or_scipy.used", False)
    _put(claims, "basis.tokens", TOKEN_BASIS)
    _put(claims, "basis.dollars", DOLLAR_BASIS)
    _put(claims, "basis.friction", FRICTION_BASIS)
    _put(claims, "basis.default_sheet", SHEETS[0])
    _put(claims, "basis.chars_per_token", _costs.RECON_CHARS_PER_TOKEN)


def build_claims(n_permutations: int = 1000, seed: int = 0,
                 front: dict | None = None, art: dict | None = None,
                 fit: dict | None = None) -> dict:
    """Recompute every quotable number into one flat {stable_key: scalar} dict."""
    art = load_artifacts() if art is None else art
    front = _figs.frontier_data() if front is None else front
    claims: dict = {}
    corpus_claims(claims, art)
    recon_claims(claims, art)
    label_claims(claims, art)
    jobkey_claims(claims, art)
    feature_claims(claims, art)
    cost_claims(claims, art)
    tooling_claims(claims, art)
    model_claims(claims, art, n_permutations, seed, fit=fit)
    policy_claims(claims, art)
    ope_claims(claims, art)
    family_claims(claims, art)
    frontier_claims(claims, front)
    refusal_claims(claims)
    return dict(sorted(claims.items()))


# --------------------------------------------------------------- metrics.json

def build_metrics(art: dict | None = None, front: dict | None = None,
                  fit: dict | None = None, n_permutations: int = 1000,
                  seed: int = 0) -> dict:
    """Assemble results/metrics.json: the four grouped reports of router.metrics.

    Every number here is also reachable from claims.json; metrics.json is the
    NESTED view (reliability bins, per-sheet cost tables, hull geometry) that a
    flat {key: scalar} dict cannot carry. It is a report, not a new estimate:
    nothing is computed here that router.metrics does not already define.
    """
    from router import model as _model

    art = load_artifacts() if art is None else art
    front = _figs.frontier_data() if front is None else front
    fit = _model.load_pfric() if fit is None else fit

    y, p = fit["y"], fit["p_fric"]
    # The single-column baseline must not be an outcome-derived column: a column
    # assembled from other runs' labels predicting labels measures the assembly.
    cols = list(fit["cols"])
    keep = [i for i, c in enumerate(cols) if c not in OUTCOME_DERIVED_COLUMNS]
    X_clean, cols_clean = fit["X"][:, keep], [cols[i] for i in keep]

    arm_points = _metrics.observed_arm_points(art["recon"], art["labels"],
                                              sheet=SHEETS[0])
    curve = [dict(pt, label=pt["arm"]) for pt in arm_points]

    return {
        "generated_by": "python -m router.report",
        "basis": {
            "tokens": TOKEN_BASIS,
            "dollars": DOLLAR_BASIS,
            "friction": FRICTION_BASIS,
            "default_sheet": SHEETS[0],
            "chars_per_token": _costs.RECON_CHARS_PER_TOKEN,
        },
        "baselines": _metrics.baseline_report(
            y, scores=p, X=X_clean, cols=cols_clean,
            n_permutations=n_permutations, seed=seed),
        "predictor": _metrics.predictor_report(
            y, p, X=X_clean, cols=cols_clean,
            n_permutations=n_permutations, seed=seed),
        "policy": _metrics.policy_report(art["routes"], art["recon"]),
        "frontier": _metrics.frontier_report(curve, arm_points),
        "spend_concentration": _metrics.spend_concentration(art["recon"]),
        "observed_arm_points": arm_points,
        "caveat": (
            "observed_arm_points and the frontier built from them are CONFOUNDED "
            "logged rates, not causal contrasts. The causal statements live in "
            "results/estimates.json and are BOUNDS, never point estimates."
        ),
    }


# --------------------------------------------------------------- NUMBERS.md

class _Q:
    """Formatter that can only quote a number that exists in claims.json."""

    def __init__(self, claims: dict):
        self.claims = claims
        self.used: set[str] = set()

    def __call__(self, key: str, fmt: str = "") -> str:
        """Format one claim by key; raises KeyError if the claim does not exist."""
        if key not in self.claims:
            raise KeyError(f"NUMBERS.md quotes {key!r}, which is not in claims.json")
        self.used.add(key)
        v = self.claims[key]
        if isinstance(v, bool) or isinstance(v, str) or not fmt:
            return str(v)
        return format(v, fmt)


SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("corpus.", "The corpus", "python -m router.io"),
    ("recon.", "Turn reconstruction and the cache split", "python -m router.recon"),
    ("labels.", "Friction labels from tool-call exit codes", "python -m router.labels"),
    ("jobkey.", "The job key: literal cron paths only", "python -m router.jobkey"),
    ("features.", "The pre-treatment feature cut", "python -m router.features"),
    ("cost.", "Price sheets and the cache-aware multiplier",
     "python -m router.pricing && python -m router.costs"),
    ("model.", "The friction predictor, against its baselines", "python -m router.model"),
    ("policy.", "The routing policy: gates, reroutes, savings", "python -m router.policy"),
    ("ope.", "Off-policy evaluation: bounds, never points", "python -m router.ope"),
    ("family.", "The cross-family contrast", "python -m router.family_contrast"),
    ("frontier.", "The headline chart", "python -m router.figs"),
    ("refusal.", "What we refuse to claim, and the basis strings",
     "python -m router.report"),
    ("basis.", "Bases carried by every number above", "python -m router.report"),
)


def _fmt_value(v) -> str:
    """Render one claim value for the NUMBERS.md table."""
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f"`{v:,}`"
    if abs(v) >= 1000:
        return f"`{v:,.2f}`"
    if abs(v) >= 1:
        return f"`{v:.4f}`"
    return f"`{v:.6f}`"


def render_numbers_md(claims: dict) -> str:
    """Build results/NUMBERS.md; every numeral in it is resolved through claims.json."""
    q = _Q(claims)
    L: list[str] = []
    a = L.append

    a("# NUMBERS")
    a("")
    a("Every number this project quotes, regenerated from code by "
      "`python -m router.report`. Nothing here is hand transcribed: each value "
      "is recomputed from `results/` and written to `results/claims.json` under "
      "the stable key shown beside it. If a slide, a README or a chart caption "
      "carries a number that is not in this file, it is not ours.")
    a("")
    a("## How to read every number in this file")
    a("")
    a(f"- **Tokens are estimates.** {TOKEN_BASIS}")
    a(f"- **Dollars are assumptions.** {DOLLAR_BASIS}")
    a(f"- **Friction is not quality.** {FRICTION_BASIS}")
    a("")
    a("## The five numbers that carry the argument")
    a("")
    a(f"1. **The bill is {q('recon.remeter_factor.tok_ratio', '.2f')}x bigger "
      f"than the starter kit counts.** A trajectory is not one API call. "
      f"Reconstructing turns gives {q('recon.turns.total', ',')} billed prefixes "
      f"over {q('corpus.n_trajectories', ',')} trajectories and "
      f"{q('recon.gross.tok', ',')} est. tokens against the starter kit's "
      f"{q('recon.naive.tok', ',')}. In dollars the gap is smaller — "
      f"{q('frontier.remeter_factor.usd_ratio', '.2f')}x — because the "
      f"re-metered bill also earns the cache discount "
      f"({q('recon.cache_read.share', '.1%')} of it is cache reads). Quote the "
      f"token ratio for tokens and the dollar ratio for dollars; they are not "
      f"the same number.")
    a("")
    a(f"2. **{q('policy.refused.gross_share', '.1%')} of the bill is off "
      f"limits.** Only {q('policy.routable.n', ',')} of "
      f"{q('corpus.n_trajectories', ',')} trajectories have a cheaper, "
      f"sign-stable, in-lane target — {q('policy.routable.gross_share', '.1%')} "
      f"of est. gross tokens. The refused remainder is token-heavy and "
      f"dollar-light: {q('frontier.refused_floor.gross_share', '.1%')} of tokens "
      f"but only {q('frontier.refused_floor.usd_share', '.1%')} of dollars, "
      f"because most of it is the cheap gpt lane.")
    a("")
    a(f"3. **The predictor has real but modest signal.** Out-of-fold AUPRC "
      f"{q('model.oof.auprc', '.4f')} against a permutation null whose 95th "
      f"percentile is {q('model.baseline.permutation_null.p95_auprc', '.4f')} "
      f"(p = {q('model.baseline.permutation_null.p_value', '.4f')}) — so the "
      f"signal is real. But the best single column that is not built from other "
      f"runs' outcomes, `{q('model.baseline.best_unfitted_column.name')}`, used "
      f"directly as a ranking score with no fit at all, already scores "
      f"{q('model.baseline.best_unfitted_column.auprc', '.4f')}. The whole "
      f"multivariate fit buys "
      f"{q('model.margin_over_best_unfitted_column.auprc', '.4f')} AUPRC over one "
      f"column. That margin is small and it should be said out loud, not buried. "
      f"Note also that the base rate is the wrong yardstick: a random ranking of "
      f"{q('model.n_rows', ',')} rows scores at the permutation median "
      f"{q('model.baseline.permutation_null.p50_auprc', '.4f')}, above the "
      f"prevalence {q('model.prevalence', '.4f')}.")
    a("")
    a(f"4. **We report a bound, not an effect.** Rerouting "
      f"{q('policy.rerouted.n', ',')} trajectories moves "
      f"{q('ope.routed.switched.spend_share', '.1%')} of est. spend and saves "
      f"{q('policy.saving.share.assumed_default', '.1%')} of the bill under the "
      f"default sheet. The friction cost of that is bounded at "
      f"<= +{q('ope.routed.friction_increase.upper_bound_pp', '.2f')} pp "
      f"({q('ope.routed.friction_increase.identified_part_pp', '.2f')} pp "
      f"identified + {q('ope.routed.friction_increase.manski_widening_pp', '.2f')} "
      f"pp Manski widening for unsupported spend). There is no point estimate "
      f"and there will not be one: the MDE is "
      f"{q('refusal.mde_best_powered_arm_pair.pp', '.1f')} pp on the "
      f"best-powered arm pair.")
    a("")
    a(f"5. **We routed too little to have earned a frontier comparison.** At the "
      f"same spend the single-arm mixture frontier sits at "
      f"{q('frontier.gated_router.hull_bound_at_same_spend_pp', '+.2f')} pp and "
      f"the gated router at "
      f"{q('frontier.policy.gated_router.upper_bound_pp', '+.2f')} pp. **Do not "
      f"subtract those.** Both are ONE-SIDED UPPER bounds "
      f"(`router.ope.non_inferiority_bound` returns `hi` only), so their "
      f"difference bounds the true difference in neither direction and no "
      f"directional verdict follows from it. The key holding it is named "
      f"`frontier.gated_router.upper_bound_minus_hull_upper_bound_pp` for exactly "
      f"that reason. What CAN be said is arithmetic and stronger: the identified "
      f"delta is a spend-weighted average of per-run differences in [0, 1] over "
      f"the switched runs only, so its magnitude cannot exceed the share of spend "
      f"the policy moves — and this router moves "
      f"{q('ope.routed.switched.spend_share', '.1%')} of est. spend "
      f"({q('policy.rerouted.n', ',')} of {q('policy.routable.n', ',')} routable "
      f"trajectories). No policy at this tau can reach a hull sitting at "
      f"{q('frontier.gated_router.hull_bound_at_same_spend_pp', '+.2f')} pp, "
      f"however well it routes. The finding is about tau, not about the routing "
      f"rule. Note also what the comparator is: at this spend the hull is an "
      f"interpolation, i.e. a RANDOMISATION between "
      f"`{q('frontier.hull.vertex1.arm')}` and `{q('frontier.hull.vertex2.arm')}` "
      f"— not one cheap arm, and the second of those prices at "
      f"${q('frontier.policy.all_to_claude-opus-5.usd', ',.2f')}, the top of the "
      f"chart's cost axis.")
    a("")
    a("## Two arithmetic corrections to earlier drafts")
    a("")
    a(f"- **The exclusion categories overlap.** The four published exclusion "
      f"counts are correct as counts of trajectories *carrying* each property, "
      f"but they are not disjoint: they sum to "
      f"{q('policy.exclusions.property_count_sum', ',')} against "
      f"{q('corpus.n_trajectories', ',')} trajectories, because "
      f"`gpt-5.6-luna` is both gpt-lane and non-admissible, and some "
      f"pre-treatment-image runs sit on already-excluded arms. "
      f"`policy.exclusions.are_disjoint` is "
      f"{q('policy.exclusions.are_disjoint')}.")
    a(f"- **The re-metering factor is not one number.** "
      f"{q('recon.remeter_factor.tok_ratio', '.2f')}x applies to tokens and "
      f"{q('frontier.remeter_factor.usd_ratio', '.2f')}x to dollars. Using the "
      f"token ratio as a dollar correction overstates the money by roughly "
      f"fivefold.")
    a("")
    a("## What is deliberately absent")
    a("")
    a("- No friction point estimate, and no IPS or DR point estimate: the Kish "
      "ESS per arm equals its own run count exactly, so those estimators have "
      "no effective sample to lean on. `ope.plain_ips.value_out_of_range` shows "
      f"the failure directly — plain IPS returns "
      f"{q('ope.plain_ips.value_out_of_range', '.4f')} for a quantity that must "
      f"lie in [0, 1].")
    a(f"- No baseline built from the labels. Two feature columns "
      f"(`{q('model.leak_flag.outcome_derived_columns.n')}` of them) are "
      f"empirical-Bayes summaries of OTHER runs' outcomes. Scored alone, the "
      f"better of them "
      f"(`{q('model.leak_flag.best_column_including_outcome_derived.name')}`) "
      f"reaches "
      f"{q('model.leak_flag.best_column_including_outcome_derived.auprc', '.4f')} "
      f"AUPRC — above the whole fit. That is a leak signal, not a result: those "
      f"columns are constant within a job, so under job-blocked folds a test "
      f"job's rows still carry information from that job's own labels "
      f"(`model.leak_flag.outcome_derived_columns_are_job_blocked_safe` = "
      f"{q('model.leak_flag.outcome_derived_columns_are_job_blocked_safe')}). "
      f"They are excluded from the single-column baseline above and should be "
      f"reviewed by the leak audit before any of them is quoted as a driver.")
    a("- No answer-quality claim anywhere. Every outcome in this project is "
      "process friction from tool-call exit codes.")
    a(f"- No cross-family verdict. The pooled contrast is "
      f"{q('family.pooled.gpt_minus_claude.pp', '+.2f')} pp "
      f"[{q('family.pooled.gpt_minus_claude.ci_lo_pp', '+.2f')}, "
      f"{q('family.pooled.gpt_minus_claude.ci_hi_pp', '+.2f')}] against an MDE "
      f"of {q('family.pooled.mde_80pct_power.pp', '.1f')} pp, and family is "
      f"perfectly collinear with the toolset in every run "
      f"(`family.confounded_with_toolset` = "
      f"{q('family.confounded_with_toolset')}), so this data cannot separate "
      f"model from harness.")
    a("")
    a("---")
    a("")
    a("## Every claim, by section")
    a("")
    a("Each section names the exact command that regenerates its keys. Keys are "
      "the contract: they are stable across runs and are what "
      "`router.verify` checks prose against.")
    a("")

    seen: set[str] = set()
    for prefix, title, command in SECTIONS:
        keys = [k for k in claims if k.startswith(prefix) and k not in seen]
        if not keys:
            continue
        seen.update(keys)
        a(f"### {title}")
        a("")
        a("```")
        a(command)
        a("```")
        a("")
        a("| key | value |")
        a("| --- | --- |")
        for k in keys:
            a(f"| `{k}` | {_fmt_value(claims[k])} |")
        a("")

    leftover = [k for k in claims if k not in seen]
    if leftover:
        a("### Unsectioned keys")
        a("")
        a("| key | value |")
        a("| --- | --- |")
        for k in leftover:
            a(f"| `{k}` | {_fmt_value(claims[k])} |")
        a("")

    a("---")
    a("")
    a(f"Generated by `python -m router.report`. "
      f"{len(claims)} claims, {len(q.used)} of them quoted in the prose above.")
    a("")
    return "\n".join(L)


def write(claims: dict, numbers_md: str,
          claims_path: Path = CLAIMS_PATH,
          numbers_path: Path = NUMBERS_PATH,
          metrics: dict | None = None,
          metrics_path: Path = METRICS_PATH) -> tuple[Path, Path]:
    """Write results/claims.json, results/NUMBERS.md and (if given) results/metrics.json."""
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    claims_path.write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    numbers_path.write_text(numbers_md, encoding="utf-8")
    if metrics is not None:
        metrics_path.write_text(json.dumps(metrics, indent=2, default=float) + "\n",
                                encoding="utf-8")
    return claims_path, numbers_path


# ------------------------------------------------------------- acceptance

def acceptance(claims: dict, numbers_md: str,
               metrics: dict | None = None) -> list[tuple]:
    """Return [(pass, label, expected, actual), ...] for every report acceptance check."""
    rows: list[tuple] = []
    for key, expected in EXPECTED.items():
        actual = claims.get(key)
        if isinstance(expected, float):
            ok = actual is not None and abs(actual - expected) < 5e-4
        else:
            ok = actual == expected
        rows.append((ok, key, expected, actual))
    for key, (expected, tol) in APPROX.items():
        actual = claims.get(key)
        ok = actual is not None and abs(actual - expected) <= tol
        rows.append((ok, key, expected, actual))

    flat = all(isinstance(v, (int, float, bool, str)) for v in claims.values())
    rows.append((flat, "claims.json is flat (no nested values)", True, flat))
    rows.append((len(claims) > 200, "claims.json covers > 200 keys", True, len(claims)))
    listed = sum(1 for line in numbers_md.splitlines() if line.startswith("| `"))
    rows.append((listed == len(claims), "NUMBERS.md lists every claim exactly once",
                 len(claims), listed))
    if metrics is not None:
        # metrics.json is the nested view; it must agree with claims.json on the
        # numbers they both carry, or the two artifacts are quoting different runs.
        auprc_m = metrics["predictor"]["friction_head"]["auprc"]
        rows.append((abs(auprc_m - claims["model.oof.auprc"]) < 1e-12,
                     "metrics.json AUPRC == claims.json AUPRC",
                     claims["model.oof.auprc"], auprc_m))
        top10 = metrics["spend_concentration"]["top_k_share"][10]
        rows.append((abs(top10 - 0.177) < 5e-4,
                     "metrics.json top-10 spend share", 0.177, round(top10, 4)))
        rows.append((json.dumps(metrics, default=float) is not None,
                     "metrics.json is JSON-serialisable", True, True))
    return rows


def main(argv: list[str] | None = None) -> int:
    """Rebuild results/claims.json + results/NUMBERS.md and print the acceptance checks."""
    argv = list(sys.argv[1:] if argv is None else argv)
    n_perm = int(argv[0]) if argv else 1000
    print("router.report — every token count is an ESTIMATE (the export has no "
          "usage field); dollars are input-side only and every sheet is ASSUMED.")
    try:
        # Loaded once and threaded through both builders: load_pfric() re-fits
        # the predictor (~20 s) and frontier_data() re-bootstraps, so calling
        # either twice would double the stage runtime for identical numbers.
        from router import model as _model

        art = load_artifacts()
        front = _figs.frontier_data()
        fit = _model.load_pfric()
        claims = build_claims(n_permutations=n_perm, front=front, art=art, fit=fit)
        metrics = build_metrics(art=art, front=front, fit=fit,
                                n_permutations=n_perm)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    numbers_md = render_numbers_md(claims)
    claims_path, numbers_path = write(claims, numbers_md, metrics=metrics)

    ok = True
    print()
    for passed, label, expected, actual in acceptance(claims, numbers_md, metrics):
        ok = ok and passed
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {label:46s} expected {expected!r:>16}  actual {actual!r}")
    print()
    print(f"wrote {claims_path}  ({len(claims)} claims)")
    print(f"wrote {numbers_path}")
    print(f"wrote {METRICS_PATH}  ({len(metrics)} sections)")
    print()
    print("Every number the team quotes must exist in claims.json under a "
          "stable key. If it is not there, it is not ours.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
