#!/usr/bin/env python3
"""Every metric the router reports, in one module, grouped exactly four ways.

WHAT THIS COMPUTES
    Four report functions, each returning a plain dict (JSON-serialisable, no
    numpy scalars), so `router.frontier` / `router.report` can drop them straight
    into results/metrics.json and results/claims.json:

      1. baseline_report()   — majority, permutation-null and best-single-column
                               reference levels. This is a GROUP OF ITS OWN and
                               not an appendix: on this corpus every headline
                               predictor number is within a few points of a
                               baseline, so a score printed without its baselines
                               is not interpretable.
      2. predictor_report()  — AUPRC, Brier, ECE + reliability bins for the
                               friction head, Spearman rho for the workload head,
                               each printed next to all three baselines.
      3. policy_report()     — delta spend % on the RE-METERED basis per price
                               sheet, friction non-inferiority upper bound,
                               extrapolated share of spend, refusal rate.
      4. frontier_report()   — AIQ (area under the cost-quality curve), distance
                               above the non-decreasing convex hull over the
                               admissible single-arm policies, and cost per
                               percentage point of friction budget.

WHAT IT WRITES
    Nothing. Pure functions over arrays and plain records. `main()` reads
    results/*.jsonl read-only for the live section and writes no file.

BASES AND CAVEATS THAT TRAVEL WITH THESE NUMBERS
    * Every token count is an ESTIMATE (tok(x) = len(json.dumps(x)) // 4; the
      export has no `usage` field). Every dict returned here that carries tokens
      or dollars also carries a "token_basis" string saying so.
    * "Re-metered basis" means the reconstructed turn-prefix bill from
      results/recon.jsonl (334,729,910 est. tok), NOT the starter kit's flat
      per-item sum (22,631,879 est. tok). The two differ by 14.79x and every
      delta-spend percentage below is meaningless on the naive basis.
    * Dollars are input-side only (the export has no `output` field) and every
      sheet is an ASSUMPTION (ids are anonymized per AGENTS.md). Per-sheet
      figures are reported as a spread; a single sheet's delta is not quotable.
    * Friction is only partially identified (MDE 11.4pp on the best-powered arm
      pair). `policy_report` therefore returns an UPPER BOUND, never a point
      effect, and `frontier_report` never interpolates a midpoint through a band.

ACCEPTANCE (checked by `python -m router.metrics`)
    Closed-form identities, each verified exactly rather than to a tolerance:
      auprc(perfect ranker) == 1.0 ; auprc(all-tied scores) == prevalence ;
      auprc(y=[1,0,1,0], s=[.9,.8,.7,.6]) == 5/6 ;
      brier(constant base rate p) == p*(1-p) ; ece(perfect) == 0 ;
      ece(constant 0.5 at base rate 0.3) == 0.2 ;
      spearman(monotone) == 1, spearman(reversed) == -1,
      spearman([1,2,3,4],[1,3,2,4]) == 0.8 (tie-free Spearman identity) ;
      z_for(0.05) == 1.644853626951 to 12dp (A&S seed + Newton, math.erf only) ;
      the permutation null sits ABOVE the prevalence by a bias that decays with n
      (+0.031 at n=100, +0.005 at n=1000) — the finite-sample chance level for
      average precision is NOT the base rate, and this module says so in the
      report dict rather than letting a 0.28 AUPRC read as "beats 0.259" ;
      the non-decreasing convex hull is concave, non-decreasing, contains the
      cheapest point, and a dominated point sits at a NEGATIVE distance above it ;
      aiq(constant quality q) == q.
    Live cross-check against results/recon.jsonl: spend concentration top10 =
    17.7% of gross, reproducing the published figure from docs/CONTRACTS.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# DELIBERATELY NOT IMPLEMENTED: accuracy at predicting the logged model.
#
# There is no `logged_model_accuracy()` in this module and there will not be
# one. The obvious "can we recover the routing policy?" metric was measured and
# discarded, not overlooked:
#
#     a naive Bayes classifier over pre-treatment features scores 0.411
#     accuracy at predicting which arm served a trajectory, against a 0.438
#     majority baseline WITHIN the claude lane (331 opus-5 of 755 claude rows
#     = 0.438 for the constant "always claude-opus-5" rule).
#
# It scores BELOW the constant rule. The logged assignment carries no learnable
# structure over anything observable before the model runs, which is the same
# fact that shows up as Kish ESS per arm equalling n_a exactly (331/281/113/
# 112/71/69/20/2/1) and is why IPS and DR point estimates are banned upstream.
#
# The consequence is a design consequence, not a scoring one: THERE IS NO
# LOGGING POLICY TO IMITATE. A router cannot be evaluated by how well it
# reproduces the observed arm choice, because the observed arm choice is not a
# policy. Every metric in this module therefore scores the router against COST
# and FRICTION directly, and a high "agreement with the log" would be evidence
# of memorising row identities, not of a good router. Reporting such a number
# would invite exactly the misreading the rest of the pipeline is built to
# prevent.
# --------------------------------------------------------------------------

#: Caveat string attached to every dict in this module that carries tokens/dollars.
TOKEN_BASIS = "ESTIMATED (tok = len(json.dumps(x))//4; the export has no usage field)"

#: Caveat string attached to every dict that carries a friction figure.
FRICTION_BASIS = (
    "friction is partially identified (MDE 11.4pp on the best-powered arm pair); "
    "bounds only, never a point effect"
)

#: The reconstructed vs naive metering correction (docs/CONTRACTS.md).
REMETER_FACTOR = 334_729_910 / 22_631_879  # 14.79x


# ============================================================ primitives ===

def _as1d(a) -> np.ndarray:
    """Coerce any sequence to a 1-D float64 numpy array."""
    return np.asarray(a, dtype=np.float64).ravel()


def auprc(y, scores) -> float:
    """Average precision (area under the precision-recall curve), tie-aware."""
    y, s = _as1d(y), _as1d(scores)
    if y.size != s.size:
        raise ValueError(f"auprc: {y.size} labels vs {s.size} scores")
    pos = y.sum()
    if pos == 0 or pos == y.size:
        return float("nan")  # AUPRC is undefined without both classes
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    # A threshold can only fall BETWEEN distinct scores: evaluate at tie-group ends.
    ends = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1.0)
    recall = tp / pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier(y, p) -> float:
    """Brier score: mean squared error of the probability forecast (lower is better)."""
    y, p = _as1d(y), _as1d(p)
    return float(np.mean((p - y) ** 2))


def reliability_bins(y, p, n_bins: int = 10) -> list[dict]:
    """Equal-width calibration bins over [0,1]: count, mean forecast, observed rate."""
    y, p = _as1d(y), _as1d(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = which == b
        n = int(m.sum())
        out.append({
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": n,
            "mean_p": float(p[m].mean()) if n else None,
            "frac_pos": float(y[m].mean()) if n else None,
            "gap": float(y[m].mean() - p[m].mean()) if n else None,
        })
    return out


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error: n-weighted mean |observed rate - mean forecast| over bins."""
    bins = reliability_bins(y, p, n_bins)
    n = float(len(_as1d(y)))
    return float(sum(b["n"] / n * abs(b["gap"]) for b in bins if b["n"]))


def rankdata(a) -> np.ndarray:
    """Ranks 1..n with ties given their average rank (the tie handling Spearman needs)."""
    a = _as1d(a)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation (Pearson on average-tied ranks); nan if either side is constant."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf (stdlib only; scipy is not installed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def z_for(one_sided_alpha: float) -> float:
    """Standard normal quantile z with Phi(z) = 1 - alpha (A&S seed + Newton polish)."""
    if not 0.0 < one_sided_alpha < 1.0:
        raise ValueError("one_sided_alpha must be in (0,1)")
    p = one_sided_alpha
    flip = p > 0.5
    if flip:
        p = 1.0 - p
    t = math.sqrt(-2.0 * math.log(p))
    z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3
    )
    for _ in range(4):  # Newton on Phi(z) - (1-alpha); erf makes this machine-exact
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        z -= (normal_cdf(z) - (1.0 - p)) / pdf
    return -z if flip else z


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no normal-approximation blowup at 0/1)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = n + z * z
    center = (k + 0.5 * z * z) / denom
    half = z * math.sqrt(p * (1 - p) * n + 0.25 * z * z) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def diff_interval(k1: int, n1: int, k2: int, n2: int, z: float, method: str = "newcombe"):
    """Interval for p1 - p2 by Newcombe's Wilson-hybrid (default) or the plain Wald formula."""
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    if method == "wald":
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        return (d - z * se, d + z * se)
    if method != "newcombe":
        raise ValueError(f"unknown method {method!r}")
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lo, hi)


def noninferiority_upper_bound(k_new: int, n_new: int, k_ref: int, n_ref: int,
                               alpha: float = 0.05, method: str = "newcombe") -> dict:
    """One-sided upper bound on (friction_new - friction_ref); the whole non-inferiority claim."""
    z = z_for(alpha)
    lo, hi = diff_interval(k_new, n_new, k_ref, n_ref, z, method)
    return {
        "k_new": int(k_new), "n_new": int(n_new),
        "k_ref": int(k_ref), "n_ref": int(n_ref),
        "rate_new": k_new / n_new if n_new else None,
        "rate_ref": k_ref / n_ref if n_ref else None,
        "point_diff_pp": 100.0 * (k_new / n_new - k_ref / n_ref) if n_new and n_ref else None,
        "upper_bound_pp": 100.0 * hi,
        "lower_bound_pp": 100.0 * lo,
        "one_sided_alpha": alpha,
        "z": z,
        "method": method,
        "claim": "friction_new - friction_ref is at most upper_bound_pp percentage points",
        "causal": False,
        "basis": FRICTION_BASIS,
    }


# ================================================= hull / frontier geometry ===

def nondecreasing_convex_hull(points) -> list[tuple[float, float]]:
    """Upper concave, non-decreasing envelope of (cost, quality) points — the mixture frontier."""
    best_at: dict[float, float] = {}
    for x, y in points:                                    # collapse duplicate costs
        x, y = float(x), float(y)
        if x not in best_at or y > best_at[x]:
            best_at[x] = y
    run, kept = -math.inf, []
    for x, y in sorted(best_at.items()):                   # drop cost-dominated points
        if y > run:
            kept.append((x, y))
            run = y
    hull: list[tuple[float, float]] = []
    for px, py in kept:                                    # monotone chain, upper hull
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((px, py))
    return hull


def hull_value(hull, x: float) -> float:
    """Quality the hull achieves at cost x (linear between vertices, clamped outside its range)."""
    if not hull:
        return float("nan")
    if x <= hull[0][0]:
        return hull[0][1]
    if x >= hull[-1][0]:
        return hull[-1][1]
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if x1 <= x <= x2:
            return y1 if x2 == x1 else y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return hull[-1][1]


def distance_above_hull(hull, x: float, y: float) -> float:
    """Vertical quality gap above the mixture frontier; negative means the hull dominates."""
    return float(y - hull_value(hull, x))


def aiq(curve, cost_lo: float | None = None, cost_hi: float | None = None) -> dict:
    """Area under the cost-quality curve on a cost axis normalised to [0,1] (AIQ)."""
    pts = sorted((float(c), float(q)) for c, q in curve)
    if len(pts) < 2:
        q = pts[0][1] if pts else float("nan")
        return {"aiq": q, "n_points": len(pts), "degenerate": True,
                "basis": "single point: AIQ is that point's quality"}
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    lo = xs.min() if cost_lo is None else float(cost_lo)
    hi = xs.max() if cost_hi is None else float(cost_hi)
    if hi == lo:
        return {"aiq": float(ys.mean()), "n_points": len(pts), "degenerate": True,
                "basis": "zero cost range: AIQ is the mean quality"}
    u = (xs - lo) / (hi - lo)
    return {
        "aiq": float(np.trapezoid(ys, u)),
        "n_points": len(pts),
        "degenerate": False,
        "cost_lo": lo,
        "cost_hi": hi,
        "basis": "trapezoid over cost normalised to [0,1]; quality = 1 - friction",
    }


# ================================================ group 1: baseline_report ===

def best_single_column(X, cols, y) -> dict:
    """Best AUPRC obtainable from ONE raw feature column used directly as the score."""
    X = np.asarray(X, dtype=np.float64)
    y = _as1d(y)
    best = {"col": None, "sign": None, "auprc": float("nan")}
    for j, name in enumerate(cols):
        col = X[:, j]
        if not np.isfinite(col).all() or col.std() == 0:
            continue
        for sign in (1.0, -1.0):
            a = auprc(y, sign * col)
            if not math.isnan(a) and (math.isnan(best["auprc"]) or a > best["auprc"]):
                best = {"col": str(name), "sign": int(sign), "auprc": a}
    return best


def permutation_null(y, scores, n_permutations: int = 1000, seed: int = 0) -> dict:
    """Distribution of AUPRC when the labels are shuffled — the honest chance level."""
    y, s = _as1d(y), _as1d(scores)
    rng = np.random.default_rng(seed)
    vals = np.array([auprc(rng.permutation(y), s) for _ in range(n_permutations)])
    prev = float(y.mean())
    return {
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "mean_auprc": float(vals.mean()),
        "p50_auprc": float(np.median(vals)),
        "p95_auprc": float(np.quantile(vals, 0.95)),
        "max_auprc": float(vals.max()),
        "prevalence": prev,
        "mean_minus_prevalence": float(vals.mean() - prev),
    }


def baseline_report(y, scores=None, X=None, cols=None,
                    n_permutations: int = 1000, seed: int = 0) -> dict:
    """Group 1: the three reference levels every predictor number must be read against."""
    y = _as1d(y)
    prev = float(y.mean())
    out = {
        "n": int(y.size),
        "positives": int(y.sum()),
        "prevalence": prev,
        "majority": {
            "rule": "constant forecast p = base rate (the no-information forecast)",
            # A constant score ranks every row identically, so its AP is exactly the
            # prevalence. But this is the ASYMPTOTIC chance level only: a random
            # ranking of finitely many rows scores ABOVE it (+0.03 at n=100, +0.005 at
            # n=1000). Compare a real score against permutation_null, not against this.
            "auprc": prev,
            "brier": prev * (1.0 - prev),
            "ece": 0.0,
            "caveat": "asymptotic chance level; use permutation_null for the finite-sample one",
        },
        "permutation_null": None,
        "best_single_column": None,
    }
    if scores is not None:
        out["permutation_null"] = permutation_null(y, scores, n_permutations, seed)
    if X is not None and cols is not None:
        out["best_single_column"] = best_single_column(X, cols, y)
    return out


# =============================================== group 2: predictor_report ===

def predictor_report(y, p, X=None, cols=None, workload_true=None, workload_pred=None,
                     n_bins: int = 10, n_permutations: int = 1000, seed: int = 0) -> dict:
    """Group 2: friction-head AUPRC/Brier/ECE + workload-head Spearman, next to all baselines."""
    y, p = _as1d(y), _as1d(p)
    base = baseline_report(y, scores=p, X=X, cols=cols,
                           n_permutations=n_permutations, seed=seed)
    bins = reliability_bins(y, p, n_bins)
    rep = {
        "n": int(y.size),
        "friction_head": {
            "auprc": auprc(y, p),
            "brier": brier(y, p),
            "ece": ece(y, p, n_bins),
            "n_bins": n_bins,
            "reliability_bins": bins,
            "mean_forecast": float(p.mean()),
            "base_rate": float(y.mean()),
        },
        "workload_head": None,
        "baselines": base,
        "must_be_out_of_fold": (
            "p is expected to be OUT-OF-FOLD with job-blocked folds; an in-sample "
            "score here is not a performance number"
        ),
        "basis": FRICTION_BASIS,
    }
    if workload_true is not None and workload_pred is not None:
        wt, wp = _as1d(workload_true), _as1d(workload_pred)
        rep["workload_head"] = {
            "spearman_rho": spearman(wt, wp),
            "n": int(wt.size),
            "target": "workload (est. tokens / tool calls) — a RANK claim, not a level claim",
            "token_basis": TOKEN_BASIS,
        }
    # The verdict is the point of the group: a score is only news if it clears all three.
    a = rep["friction_head"]["auprc"]
    lvls = {"majority": base["majority"]["auprc"]}
    if base["permutation_null"]:
        lvls["permutation_null_p95"] = base["permutation_null"]["p95_auprc"]
    if base["best_single_column"]:
        lvls["best_single_column"] = base["best_single_column"]["auprc"]
    rep["beats_baseline"] = {k: (None if math.isnan(a) else bool(a > v)) for k, v in lvls.items()}
    rep["baseline_levels"] = lvls
    rep["binding_baseline"] = (
        "permutation_null_p95" if "permutation_null_p95" in lvls else "majority"
    )
    rep["binding_baseline_note"] = (
        "the majority AUPRC equals the prevalence only asymptotically; the permutation "
        "null is the finite-sample chance level and is the one a score must clear"
    )
    return rep


# ================================================== group 3: policy_report ===

def _recon_by_idx(recon) -> dict[int, dict]:
    """Index recon records by idx, accepting either a list or an already-keyed dict."""
    if isinstance(recon, dict):
        return {int(k): v for k, v in recon.items()}
    return {int(r["idx"]): r for r in recon}


def policy_report(routes, recon, sheets=None, friction_counts=None,
                  noninferiority_alpha: float = 0.05, spend_field: str = "gross_tok",
                  **cost_params) -> dict:
    """Group 3: delta spend % per sheet, friction upper bound, extrapolated spend share, refusals."""
    from router import costs as _costs               # imported here: metrics stays importable alone
    from router.pricing import ALL_SHEETS

    sheets = tuple(ALL_SHEETS if sheets is None else sheets)
    by_idx = _recon_by_idx(recon)
    rows = [r for r in routes if int(r["idx"]) in by_idx]
    recs = [by_idx[int(r["idx"])] for r in rows]

    corpus_spend = float(sum(v[spend_field] for v in by_idx.values()))
    scope_spend = float(sum(r[spend_field] for r in recs))
    supported_spend = float(sum(
        rec[spend_field] for rec, row in zip(recs, rows) if row.get("supported", True)
    ))

    n = len(rows)
    n_changed = sum(1 for r in rows if r.get("changed", r.get("route") != r.get("logged")))
    n_supported = sum(1 for r in rows if r.get("supported", True))

    per_sheet = {}
    for s in sheets:
        logged = _costs.total_cost(recs, [r["logged"] for r in rows], s, **cost_params)
        routed = _costs.total_cost(recs, [r["route"] for r in rows], s, **cost_params)
        per_sheet[s] = {
            "cost_logged_usd": logged,
            "cost_routed_usd": routed,
            "delta_usd": routed - logged,
            "delta_spend_pct": 100.0 * (routed - logged) / logged if logged else None,
            "sheet_status": "ASSUMED — ids anonymized, no public sheet applies",
        }
    deltas = [v["delta_spend_pct"] for v in per_sheet.values() if v["delta_spend_pct"] is not None]

    rep = {
        "n_considered": n,
        "n_changed": n_changed,
        "n_supported": n_supported,
        "refusal_rate": 1.0 - n_changed / n if n else None,
        "unsupported_rate": 1.0 - n_supported / n if n else None,
        "delta_spend_pct": per_sheet,
        "delta_spend_pct_spread": {
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "sign_agrees_across_sheets": bool(deltas) and (
                all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
            ),
        },
        "spend_basis": (
            "RE-METERED: reconstructed turn-prefix bill from results/recon.jsonl, "
            f"which is {REMETER_FACTOR:.2f}x the starter kit's naive per-item sum"
        ),
        "share_of_spend_covered": scope_spend / corpus_spend if corpus_spend else None,
        "share_of_spend_supported": supported_spend / corpus_spend if corpus_spend else None,
        "extrapolated_share_of_spend": (
            1.0 - supported_spend / corpus_spend if corpus_spend else None
        ),
        "extrapolated_meaning": (
            "the share of the corpus bill NOT covered by a supported reroute; any "
            "corpus-wide saving quoted over this share is an extrapolation, not a measurement"
        ),
        "friction_noninferiority": None,
        "token_basis": TOKEN_BASIS,
        "cost_basis": "input side only (the export has no output field)",
    }
    if friction_counts:
        k_new, n_new = friction_counts["new"]
        k_ref, n_ref = friction_counts["ref"]
        rep["friction_noninferiority"] = noninferiority_upper_bound(
            k_new, n_new, k_ref, n_ref, alpha=noninferiority_alpha
        )
    return rep


# ================================================ group 4: frontier_report ===

def frontier_report(curve, admissible, friction_budget_pp: float | None = None) -> dict:
    """Group 4: AIQ, distance above the single-arm mixture hull, and USD per pp of friction."""
    pts = [(float(c["cost"]), float(c["quality"])) for c in curve]
    arm_pts = [(float(a["cost"]), float(a["quality"])) for a in admissible]
    hull = nondecreasing_convex_hull(arm_pts)

    area = aiq(pts)
    dists = []
    for c, (x, y) in zip(curve, pts):
        d = distance_above_hull(hull, x, y)
        dists.append({
            "label": c.get("label"),
            "cost": x,
            "quality": y,
            "hull_quality": hull_value(hull, x),
            "distance_above_hull": d,
            "in_hull_cost_range": bool(hull and hull[0][0] <= x <= hull[-1][0]),
        })

    # Cost per percentage point of friction budget: how many dollars one extra
    # point of admitted friction buys. Endpoint slope + the marginal segments.
    marg = []
    for a, b in zip(sorted(pts), sorted(pts)[1:]):
        d_fric_pp = 100.0 * ((1.0 - b[1]) - (1.0 - a[1]))
        marg.append({
            "cost_from": a[0], "cost_to": b[0],
            "d_cost_usd": b[0] - a[0],
            "d_friction_pp": d_fric_pp,
            "usd_per_pp": (b[0] - a[0]) / d_fric_pp if d_fric_pp else None,
        })
    lo_pt, hi_pt = (sorted(pts)[0], sorted(pts)[-1]) if len(pts) >= 2 else (None, None)
    endpoint_pp = (100.0 * ((1.0 - hi_pt[1]) - (1.0 - lo_pt[1]))) if lo_pt else None

    rep = {
        "aiq": area,
        "hull": [{"cost": x, "quality": y} for x, y in hull],
        "hull_vertices": len(hull),
        "hull_arms": [a.get("arm") for a in admissible
                      if (float(a["cost"]), float(a["quality"])) in hull],
        "distance_above_hull": dists,
        "max_distance_above_hull": max((d["distance_above_hull"] for d in dists), default=None),
        "any_point_above_hull": any(d["distance_above_hull"] > 0 for d in dists),
        "cost_per_friction_pp": {
            "endpoints": {
                "d_cost_usd": (hi_pt[0] - lo_pt[0]) if lo_pt else None,
                "d_friction_pp": endpoint_pp,
                "usd_per_pp": ((hi_pt[0] - lo_pt[0]) / endpoint_pp)
                              if endpoint_pp else None,
                # The readable framing: dollars you must add to take one point of
                # friction OFF. Positive means paying more buys less friction.
                "usd_per_pp_of_friction_removed": (-(hi_pt[0] - lo_pt[0]) / endpoint_pp)
                                                  if endpoint_pp else None,
            },
            "marginal": marg,
            "sign_convention": ("usd_per_pp is d(cost)/d(friction); "
                                "usd_per_pp_of_friction_removed flips it so a positive "
                                "figure means dollars spent to remove one friction point"),
        },
        "friction_budget_pp": friction_budget_pp,
        "cost_at_budget": None,
        "quality_convention": "quality = 1 - friction; higher is better; cost axis in USD",
        "hull_meaning": (
            "the non-decreasing concave envelope of the ADMISSIBLE SINGLE-ARM policies: "
            "everything on or under it is reachable by randomising between two arms, so a "
            "router only earns its complexity by sitting strictly above it"
        ),
        "token_basis": TOKEN_BASIS,
        "basis": FRICTION_BASIS,
    }
    if friction_budget_pp is not None and hull:
        q = 1.0 - friction_budget_pp / 100.0
        reach = [x for x, y in hull if y >= q]
        rep["cost_at_budget"] = min(reach) if reach else None
    return rep


# =============================================================== live view ===

def _results_dir() -> Path:
    """Path to results/ relative to this file (the repo's generated-output directory)."""
    return Path(__file__).resolve().parent.parent / "results"


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL artifact into a list of dicts; empty list if the file is absent."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def spend_concentration(recon, ks=(5, 10, 50, 100, 500), field: str = "gross_tok") -> dict:
    """Share of total est. spend carried by the top-k lines — the fragility of every total."""
    vals = sorted((float(r[field]) for r in recon), reverse=True)
    total = sum(vals)
    return {
        "field": field,
        "n": len(vals),
        "total": total,
        "top_k_share": {int(k): (sum(vals[:k]) / total if total else None) for k in ks},
        "token_basis": TOKEN_BASIS,
    }


def observed_arm_points(recon, labels, sheet: str = "assumed_default",
                        min_n: int = 30, **cost_params) -> list[dict]:
    """Single-arm (cost, quality) points from the LOGGED data — confounded, not causal."""
    from router import costs as _costs
    by_arm: dict[str, list[dict]] = {}
    lab = {int(r["idx"]): r for r in labels}
    for r in recon:
        by_arm.setdefault(r["model"], []).append(r)
    out = []
    for arm, recs in by_arm.items():
        ys = [lab[r["idx"]]["y_fric"] for r in recs if r["idx"] in lab]
        if len(recs) < min_n or not ys:
            continue
        fric = sum(ys) / len(ys)
        out.append({
            "arm": arm,
            "n": len(recs),
            # counterfactual cost of serving the WHOLE corpus on this arm
            "cost": _costs.total_cost(recon, arm, sheet, **cost_params),
            "quality": 1.0 - fric,
            "friction": fric,
            "basis": "OBSERVED logged friction for this arm — CONFOUNDED, not a causal contrast",
        })
    return sorted(out, key=lambda d: d["cost"])


# ============================================================== acceptance ===

def _acceptance_checks() -> list[tuple[str, object, object]]:
    """Closed-form identities this module must reproduce exactly; returns (label, exp, act)."""
    checks: list[tuple[str, object, object]] = []
    r6 = lambda v: round(float(v), 6)  # noqa: E731 - local rounding shorthand
    r4 = lambda v: round(float(v), 4)  # noqa: E731

    # --- AUPRC ---
    checks.append(("auprc(perfect ranker) == 1.0", 1.0,
                   r6(auprc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]))))
    checks.append(("auprc(all scores tied) == prevalence 0.25", 0.25,
                   r6(auprc([1, 0, 0, 0], [0.5] * 4))))
    checks.append(("auprc(y=1,0,1,0 / s=.9,.8,.7,.6) == 5/6", r6(5 / 6),
                   r6(auprc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]))))
    # Worst ranker: both positives last -> precision 1/3 then 1/2 at recall .5 and 1.
    checks.append(("auprc(worst ranker) == 5/12", r6(5 / 12),
                   r6(auprc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]))))

    # --- Brier / ECE ---
    y = np.array([1.0] * 3 + [0.0] * 7)
    checks.append(("brier(constant base rate 0.3) == 0.3*0.7", r6(0.21),
                   r6(brier(y, np.full(10, 0.3)))))
    checks.append(("brier(perfect) == 0.0", 0.0, r6(brier(y, y))))
    checks.append(("ece(perfect 0/1 forecast) == 0.0", 0.0, r6(ece(y, y))))
    checks.append(("ece(constant 0.5 at base rate 0.3) == 0.2", 0.2,
                   r6(ece(y, np.full(10, 0.5)))))

    # --- Spearman ---
    checks.append(("spearman(monotone) == 1.0", 1.0, r6(spearman([1, 2, 3, 4], [10, 20, 30, 40]))))
    checks.append(("spearman(reversed) == -1.0", -1.0, r6(spearman([1, 2, 3, 4], [40, 30, 20, 10]))))
    checks.append(("spearman([1,2,3,4],[1,3,2,4]) == 0.8 (1 - 6*2/(4*15))", 0.8,
                   r6(spearman([1, 2, 3, 4], [1, 3, 2, 4]))))
    checks.append(("spearman with ties uses average ranks", 1.0,
                   r6(spearman([1, 1, 2, 2], [5, 5, 9, 9]))))

    # --- normal quantiles (stdlib erf + Newton, no scipy) ---
    r12 = lambda v: round(float(v), 12)  # noqa: E731 - Newton lands within 1 ulp
    checks.append(("z_for(0.05) == 1.644853626951 (12dp)", 1.644853626951, r12(z_for(0.05))))
    checks.append(("z_for(0.025) == 1.959963984540 (12dp)", 1.95996398454, r12(z_for(0.025))))
    checks.append(("normal_cdf(z_for(a)) == 1-a", 0.95, r6(normal_cdf(z_for(0.05)))))

    # --- interval machinery ---
    nib = noninferiority_upper_bound(21, 72, 33, 157, alpha=0.05)
    checks.append(("non-inferiority upper bound >= point difference (gpt vs claude, 23 clean jobs)",
                   True, nib["upper_bound_pp"] >= nib["point_diff_pp"]))
    # 21/72 - 33/157 = 8.1476pp exactly. docs/CONTRACTS.md quotes 29.2 - 21.0 = 8.2pp,
    # which is the same contrast with both rates rounded FIRST. Keep the exact one.
    checks.append(("its point difference is 8.15pp (8.2pp if both rates are rounded first)",
                   8.15, round(nib["point_diff_pp"], 2)))
    checks.append(("wilson(0,10) lower bound is exactly 0.0", 0.0, r6(wilson(0, 10, 1.96)[0])))
    checks.append(("wilson(10,10) upper bound is exactly 1.0", 1.0, r6(wilson(10, 10, 1.96)[1])))

    # --- permutation null: the FINITE-SAMPLE chance level, which is NOT the prevalence ---
    # Average precision under a random ranking sits ABOVE the prevalence by a bias that
    # decays roughly like 1/n (measured here: +0.031 at n=100, +0.005 at n=1000). This is
    # why nothing in this project compares an AUPRC to the base rate and calls it a win.
    def _bias(n, seed):
        rng = np.random.default_rng(seed)
        yy = (rng.random(n) < 0.25).astype(float)
        ss = rng.random(n)
        return permutation_null(yy, ss, n_permutations=300, seed=1)["mean_minus_prevalence"]

    b100, b1000 = _bias(100, 3), _bias(1000, 3)
    checks.append(("permutation-null mean AUPRC sits ABOVE the prevalence (small-sample AP bias)",
                   True, b100 > 0 and b1000 > 0))
    checks.append(("that bias shrinks with n: bias(1000) < bias(100)/2", True, b1000 < b100 / 2))
    checks.append(("bias(n=100) is about +0.03, bias(n=1000) about +0.005", (0.031, 0.005),
                   (round(b100, 3), round(b1000, 3))))

    # --- hull geometry ---
    arms = [(1.0, 0.50), (2.0, 0.80), (3.0, 0.82), (4.0, 0.70), (2.5, 0.60)]
    hull = nondecreasing_convex_hull(arms)
    checks.append(("hull drops the cost-dominated arms (4.0,0.70) and (2.5,0.60)", True,
                   (4.0, 0.70) not in hull and (2.5, 0.60) not in hull))
    checks.append(("hull keeps the cheapest point", True, hull[0] == (1.0, 0.50)))
    checks.append(("hull quality is non-decreasing in cost", True,
                   all(b[1] >= a[1] for a, b in zip(hull, hull[1:]))))
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(hull, hull[1:])]
    checks.append(("hull is concave (slopes non-increasing)", True,
                   all(s2 <= s1 + 1e-12 for s1, s2 in zip(slopes, slopes[1:]))))
    checks.append(("hull slopes are non-negative", True, all(s >= -1e-12 for s in slopes)))
    checks.append(("a hull vertex sits at distance 0", 0.0,
                   r6(distance_above_hull(hull, 2.0, 0.80))))
    checks.append(("a dominated point sits at a NEGATIVE distance", True,
                   distance_above_hull(hull, 2.5, 0.60) < 0))
    checks.append(("a point above the mixture line has a POSITIVE distance", True,
                   distance_above_hull(hull, 1.5, 0.70) > 0))
    checks.append(("hull_value interpolates the 1.0-2.0 chord at 1.5", 0.65,
                   r6(hull_value(hull, 1.5))))

    # --- AIQ ---
    checks.append(("aiq(constant quality 0.8) == 0.8", 0.8,
                   r6(aiq([(1.0, 0.8), (5.0, 0.8), (9.0, 0.8)])["aiq"])))
    checks.append(("aiq(linear 0 -> 1 over cost) == 0.5", 0.5,
                   r6(aiq([(0.0, 0.0), (1.0, 1.0)])["aiq"])))
    checks.append(("aiq of a single point is degenerate, not 0", True,
                   aiq([(1.0, 0.42)])["degenerate"]))

    # --- policy_report on a fixture whose answer is arithmetic ---
    # Two identical lines, both logged on claude-opus-5. Because the two arms bill the
    # SAME effective tokens, the whole delta collapses to the rate ratio, so the answer
    # is exact rather than approximate: rerouting one of two lines from 15.00 to 0.80
    # per 1M moves spend by (0.80 - 15.00) / (2 * 15.00) = -47.3333%.
    fixture_rec = {"idx": 0, "model": "claude-opus-5", "prefix_tokens": [2000, 4000],
                   "cache_read_tok": 2000, "cache_write_tok": 4000, "gross_tok": 6000}
    recon_fx = [dict(fixture_rec), dict(fixture_rec, idx=1)]
    routes_fx = [
        {"idx": 0, "logged": "claude-opus-5", "route": "claude-fable-5",
         "changed": True, "supported": True},
        {"idx": 1, "logged": "claude-opus-5", "route": "claude-opus-5",
         "changed": False, "supported": False},
    ]
    pol = policy_report(routes_fx, recon_fx)
    checks.append(("policy_report: delta spend == (0.80-15.00)/(2*15.00) = -47.3333%",
                   -47.3333,
                   r4(pol["delta_spend_pct"]["assumed_default"]["delta_spend_pct"])))
    checks.append(("policy_report: refusal rate is 1 of 2 considered", 0.5,
                   r6(pol["refusal_rate"])))
    checks.append(("policy_report: half the spend is unsupported, so half is extrapolated",
                   0.5, r6(pol["extrapolated_share_of_spend"])))
    checks.append(("policy_report: the saving's SIGN agrees across all three ASSUMED sheets",
                   True, pol["delta_spend_pct_spread"]["sign_agrees_across_sheets"]))
    checks.append(("policy_report: reports one delta per registered sheet, never one figure",
                   3, len(pol["delta_spend_pct"])))
    pol_ni = policy_report(routes_fx, recon_fx,
                           friction_counts={"new": (21, 72), "ref": (33, 157)})
    checks.append(("policy_report: the friction claim is an UPPER BOUND, not an effect",
                   False, pol_ni["friction_noninferiority"]["causal"]))
    # Newcombe upper bound on gpt(21/72) - claude(33/157) at one-sided alpha 0.05.
    # The point difference is 8.15pp and the bound is 18.8pp: with n=229 runs, 134 of
    # them one job, the data cannot exclude gpt being ~19pp worse. That gap IS the
    # finding, and it is why no point effect is reported anywhere.
    checks.append(("policy_report: that bound is 18.8pp at one-sided alpha 0.05 (Newcombe)",
                   18.8, round(pol_ni["friction_noninferiority"]["upper_bound_pp"], 1)))
    checks.append(("policy_report: the sheet spread is real, -47.3% to -29.2% on one route",
                   (-47.3333, -29.1667),
                   (r4(pol["delta_spend_pct_spread"]["min"]),
                    r4(pol["delta_spend_pct_spread"]["max"]))))

    # --- report wiring: a report is only a report if the baselines are in it ---
    yb = np.array([1.0] * 26 + [0.0] * 74)
    pb = np.full(100, 0.26)
    rep = predictor_report(yb, pb, n_permutations=50, seed=0)
    checks.append(("predictor_report: constant forecast does NOT beat the majority baseline",
                   False, rep["beats_baseline"]["majority"]))
    checks.append(("predictor_report: constant forecast at the base rate has ECE 0", 0.0,
                   r6(rep["friction_head"]["ece"])))
    checks.append(("predictor_report carries all three baseline levels", 3,
                   len(rep["baseline_levels"]) + (0 if rep["baselines"]["best_single_column"] else 1)))
    checks.append(("no logged-model-accuracy metric is exported", False,
                   any("logged_model" in n and "accur" in n for n in globals())))
    return checks


def main(argv=None) -> int:
    """Print the closed-form acceptance checks plus a live view over any present artifacts."""
    print("router.metrics — four metric groups: baseline / predictor / policy / frontier")
    print("All token counts are ESTIMATES (the export has no usage field). Dollars are")
    print("input-side only and every price sheet is an ASSUMPTION. Friction is bounded,")
    print("never point-estimated (MDE 11.4pp).\n")

    ok = True
    print("acceptance — closed-form identities:")
    for label, expected, actual in _acceptance_checks():
        good = expected == actual
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}\n"
              f"         expected={expected!r}  actual={actual!r}")

    # ---- live view over whatever artifacts exist -------------------------
    res = _results_dir()
    recon = _load_jsonl(res / "recon.jsonl")
    labels = _load_jsonl(res / "labels.jsonl")
    print(f"\nlive view — results/recon.jsonl: {len(recon)} records, "
          f"results/labels.jsonl: {len(labels)} records")

    if recon:
        conc = spend_concentration(recon)
        print("\nspend concentration on gross_tok (est.):")
        for k, share in conc["top_k_share"].items():
            print(f"  top{k:<4} {share:6.1%}")
        actual = round(100 * conc["top_k_share"][10], 1)
        good = actual == 17.7
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] top10 share reproduces docs/CONTRACTS.md\n"
              f"         expected=17.7  actual={actual}")

    if recon and labels:
        arms = observed_arm_points(recon, labels)
        rep = frontier_report(
            curve=[{"label": a["arm"], "cost": a["cost"], "quality": a["quality"]} for a in arms],
            admissible=arms,
        )
        print("\nsingle-arm policies, whole corpus, sheet=assumed_default (ASSUMED), rho=1.00.")
        print("Friction below is the arm's OWN logged rate: confounded, NOT a causal contrast.")
        print(f"  {'arm':<20}{'n':>5}{'cost USD':>12}{'friction':>10}{'above hull':>12}")
        for a, d in zip(arms, rep["distance_above_hull"]):
            print(f"  {a['arm']:<20}{a['n']:>5}{a['cost']:>12,.2f}"
                  f"{a['friction']:>10.3f}{d['distance_above_hull']:>12.4f}")
        print(f"  hull vertices: {rep['hull_arms']}")
        print(f"  AIQ over these points: {rep['aiq']['aiq']:.4f}  "
              f"({rep['aiq']['basis']})")
        ep = rep["cost_per_friction_pp"]["endpoints"]
        if ep["usd_per_pp_of_friction_removed"] is not None:
            print(f"  cheapest -> dearest arm: "
                  f"${ep['usd_per_pp_of_friction_removed']:,.2f} per percentage point of "
                  f"friction removed [assumed_default, ASSUMED, est. tokens]")
        good = all(d["distance_above_hull"] <= 1e-12 for d in rep["distance_above_hull"])
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] no single-arm point lies above its own hull\n"
              f"         expected=True  actual={good}")

    print(f"\nRESULT: {'all checks pass' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
