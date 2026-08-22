#!/usr/bin/env python3
"""Off-policy evaluation for the Viktor router — stratified Hajek, bounds, no point claims.

WHAT THIS COMPUTES
    The value (friction rate) of a counterfactual routing policy, estimated from
    a log in which exactly one arm ran per trajectory.

    Design decisions, each forced by a measurement:

    1. NO PROPENSITY MODEL.  The propensity is the marginal within-stratum arm
       frequency e(a|s) = n_{s,a} / n_s.  A fitted 9-arm logit on pre-treatment
       features scores 0.3290 accuracy against a 0.3310 majority baseline —
       NEGATIVE lift — and fitting it destroys 40-75% of the Kish ESS.  A
       propensity model here buys variance and nothing else.

    2. SELF-NORMALISED (Hajek), never plain IPS.  Under the marginal propensity
       the weights are constant inside a cell, so self-normalisation cancels
       them and the cell estimate is the cell mean.  That is a feature: it is
       what keeps the estimate inside [0, 1].  ``plain_ips`` is implemented and
       CALLED in main() precisely to show it break — on the identity policy it
       returns a "friction rate" above 1, which is impossible, and its weight
       mass concentrates on a handful of lines.

    3. SWITCH ON SUPPORT.  In a stratum where the target arm has >= 3
       observations the reweighted (Hajek) cell mean is used.  Below that the
       line falls back to the direct method (shrinkage + calibration) and its
       spend is counted into an explicitly reported EXTRAPOLATED SHARE.  The
       extrapolated share is reported next to every number; it is not hidden
       inside the interval.

    4. BOUNDS, NOT POINTS, for the decision claim.  ``non_inferiority_bound``
       returns an UPPER bound on the friction increase and deliberately returns
       ``estimate is None`` — the MDE on the best-powered arm pair is 11.4pp, so
       a friction point estimate would be theatre.  The bound is the identified
       part's one-sided bootstrap quantile PLUS a worst-case (Manski) widening
       over the unsupported spend, where the target arm is assumed to fail on
       every line we cannot observe it on.

    5. CLUSTER BOOTSTRAP.  Resampling is over CLUSTERS = literal cron job paths
       (runs of the same recurring job are not independent); rows without a
       literal path are their own singleton cluster — PII_ placeholders are
       renumbered per request and may never be a join key.  Every nuisance
       (shrinkage prior, calibration constant, stratum arm frequencies) is
       REFIT inside each replicate; nothing is frozen from the full sample.

    Every public estimator returns ``(estimate, lo, hi, diagnostics)``.  Never a
    bare float.

ARTIFACT WRITTEN
    results/estimates.json, section "ope" (merged; other sections preserved).

ACCEPTANCE NUMBERS (printed by ``python -m router.ope``)
    Kish ESS under the marginal propensity equals n_a EXACTLY for all 9 arms:
      331 / 281 / 113 / 112 / 71 / 69 / 20 / 2 / 1  (this is why IPS adds nothing)
    Count-weighted stratified Hajek on the IDENTITY policy reproduces the logged
      friction rate exactly: 259 / 1000 = 0.259
    Plain IPS on the SAME identity policy returns a value > 1 — an impossible
      friction rate — with its weight mass concentrated on a few lines
    The direct method is calibrated: mean prediction over logged rows == 0.259
    2000 clustered bootstrap replicates over 778 clusters

    ALL TOKEN AND SPEND FIGURES ARE ESTIMATES: tok(x) = len(json.dumps(x)) // 4;
    the export carries no usage field.  Friction is the y_fric label from
    results/labels.jsonl (exit-code grounded), not a text heuristic.

Run:     .venv/bin/python -m router.ope
Import:  load_panel, kish_ess, arm_ess_table, hajek_stratified, direct_method,
         plain_ips, bootstrap_clustered, non_inferiority_bound, policy_route_to
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from router.strata import (
    ESTIMATE_CAVEAT,
    RESULTS_DIR,
    SUPPORT_MIN_OBS,
    build_strata,
    load_labels,
    update_estimates,
)

ROUTES_PATH = RESULTS_DIR / "routes.jsonl"

#: Shrinkage pseudo-counts: arm rate toward the global rate, cell rate toward the arm rate.
K_ARM = 20.0
K_CELL = 5.0

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 12345

#: Arms too thin to be a routing target at all (they are supported in zero strata).
UNROUTABLE_ARMS = ("claude-opus-4-6", "claude-sonnet-4-6")

EXPECTED_ESS = {
    "claude-opus-5": 331,
    "claude-sonnet-5": 281,
    "gpt-5.6-terra": 113,
    "gpt-5.6-sol": 112,
    "claude-fable-5": 71,
    "claude-opus-4-8": 69,
    "gpt-5.6-luna": 20,
    "claude-opus-4-6": 2,
    "claude-sonnet-4-6": 1,
}
EXPECTED_LOGGED_RATE = 0.259


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Panel:
    """Arrays for one row per trajectory: outcome, arm, stratum, cluster, spend."""

    idx: np.ndarray
    y: np.ndarray
    arm_code: np.ndarray
    stratum_code: np.ndarray
    cluster_code: np.ndarray
    spend: np.ndarray
    arms: List[str]
    strata: List[str]
    clusters: List[str]
    family: List[str]

    @property
    def n(self) -> int:
        """Number of trajectories in the panel."""
        return int(self.idx.size)

    @property
    def n_arms(self) -> int:
        """Number of distinct logged arms."""
        return len(self.arms)

    def weights(self, mode: str = "spend") -> np.ndarray:
        """Return the per-row aggregation weight: estimated gross tokens, or 1 per line."""
        if mode == "spend":
            return self.spend.astype(np.float64)
        if mode == "count":
            return np.ones(self.n, dtype=np.float64)
        raise ValueError("weight mode must be 'spend' or 'count'")


def load_panel(records: Sequence[dict] | None = None, labels: Dict[int, dict] | None = None) -> Panel:
    """Build the estimation panel from results/recon.jsonl, labels.jsonl and the strata."""
    if records is None:
        records, _meta = build_strata()
    labels = load_labels() if labels is None else labels
    records = [r for r in sorted(records, key=lambda r: r["idx"]) if r["idx"] in labels]

    arms = sorted({r["model"] for r in records})
    strata = sorted({r["stratum"] for r in records})
    clusters = sorted({r["cluster"] for r in records})
    a_of = {a: i for i, a in enumerate(arms)}
    s_of = {s: i for i, s in enumerate(strata)}
    c_of = {c: i for i, c in enumerate(clusters)}
    return Panel(
        idx=np.array([r["idx"] for r in records], dtype=np.int64),
        y=np.array([float(labels[r["idx"]]["y_fric"]) for r in records], dtype=np.float64),
        arm_code=np.array([a_of[r["model"]] for r in records], dtype=np.int64),
        stratum_code=np.array([s_of[r["stratum"]] for r in records], dtype=np.int64),
        cluster_code=np.array([c_of[r["cluster"]] for r in records], dtype=np.int64),
        spend=np.array([float(r["gross_tok"]) for r in records], dtype=np.float64),
        arms=arms,
        strata=strata,
        clusters=clusters,
        family=[r["family"] for r in records],
    )


# --------------------------------------------------------------------------- #
# effective sample size
# --------------------------------------------------------------------------- #
def kish_ess(weights: Sequence[float] | np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum(w^2); 0.0 when all weights vanish."""
    w = np.asarray(weights, dtype=np.float64)
    denom = float((w * w).sum())
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


def arm_ess_table(panel: Panel) -> List[dict]:
    """Per arm: n_a, Kish ESS under the global marginal propensity (== n_a), and under strata."""
    n = panel.n
    table: List[dict] = []
    n_s = np.bincount(panel.stratum_code, minlength=len(panel.strata)).astype(np.float64)
    for a, arm in enumerate(panel.arms):
        rows = panel.arm_code == a
        n_a = int(rows.sum())
        # global marginal propensity e(a) = n_a / n -> every weight is n / n_a
        w_global = np.full(n_a, n / n_a, dtype=np.float64)
        # within-stratum marginal propensity e(a|s) = n_{s,a} / n_s
        s_rows = panel.stratum_code[rows]
        n_sa = np.bincount(s_rows, minlength=len(panel.strata)).astype(np.float64)
        w_strat = n_s[s_rows] / n_sa[s_rows]
        table.append(
            {
                "arm": arm,
                "n": n_a,
                "ess_global_marginal": kish_ess(w_global),
                "ess_within_stratum": kish_ess(w_strat),
                "ess_loss_pct": 100.0 * (1.0 - kish_ess(w_strat) / n_a),
                "n_strata_present": int((n_sa > 0).sum()),
            }
        )
    return sorted(table, key=lambda r: -r["n"])


def print_ess_table(table: Sequence[dict]) -> None:
    """Print the per-arm ESS table (global marginal ESS must equal n_a exactly)."""
    print()
    print("PER-ARM KISH ESS — the reason no propensity is fitted")
    print("  %-18s %6s %14s %14s %9s %7s" % ("arm", "n", "ESS(marginal)", "ESS(stratum)", "loss", "strata"))
    for row in table:
        print(
            "  %-18s %6d %14.2f %14.2f %8.1f%% %7d"
            % (
                row["arm"],
                row["n"],
                row["ess_global_marginal"],
                row["ess_within_stratum"],
                row["ess_loss_pct"],
                row["n_strata_present"],
            )
        )
    print("  ESS(marginal) == n exactly: under a marginal propensity every weight in an arm is")
    print("  identical, so importance weighting adds NO information — it only adds variance.")


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def policy_identity(panel: Panel) -> np.ndarray:
    """Target policy = whatever actually ran (used for the exact-reproduction checks)."""
    return panel.arm_code.copy()


def policy_route_to(
    panel: Panel,
    arm: str,
    restrict_family: bool = True,
    exclude_arms: Sequence[str] = UNROUTABLE_ARMS,
) -> np.ndarray:
    """Route every eligible line to `arm`, leaving the rest on their logged arm."""
    if arm not in panel.arms:
        raise ValueError("unknown arm %r" % arm)
    target = panel.arms.index(arm)
    fam = {panel.arms[a]: f for a, f in zip(panel.arm_code, panel.family)}
    want_family = fam[arm]
    out = panel.arm_code.copy()
    for i in range(panel.n):
        logged = panel.arms[int(panel.arm_code[i])]
        if logged == arm or logged in exclude_arms:
            continue
        if restrict_family and panel.family[i] != want_family:
            continue
        out[i] = target
    return out


def load_routes_policy(panel: Panel, path: Path | str = ROUTES_PATH) -> np.ndarray | None:
    """Load results/routes.jsonl as a target policy, or None if it does not match the contract."""
    path = Path(path)
    if not path.exists():
        return None
    by_idx: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "idx" not in rec or not isinstance(rec.get("route"), str):
                return None  # not the contracted schema — refuse rather than guess
            by_idx[int(rec["idx"])] = rec["route"]
    out = panel.arm_code.copy()
    for i, ix in enumerate(panel.idx):
        arm = by_idx.get(int(ix))
        if arm is None or arm not in panel.arms:
            return None
        out[i] = panel.arms.index(arm)
    return out


# --------------------------------------------------------------------------- #
# the core estimator (vectorised so the bootstrap can refit everything)
# --------------------------------------------------------------------------- #
def _calibration_scale(pred: np.ndarray, y: np.ndarray, mult: np.ndarray) -> float:
    """Find c so the multiplicity-weighted mean of clip(c*pred, 0, 1) matches the observed rate."""
    target = float((mult * y).sum() / mult.sum())
    if pred.size == 0 or target <= 0:
        return 1.0
    lo, hi = 0.0, 50.0

    def mean_at(c: float) -> float:
        return float((mult * np.clip(c * pred, 0.0, 1.0)).sum() / mult.sum())

    if mean_at(hi) < target:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _core(
    panel: Panel,
    target: np.ndarray,
    mult: np.ndarray,
    weight: np.ndarray,
    min_obs: int,
    k_arm: float,
    k_cell: float,
    observed_when_unswitched: bool = True,
) -> dict:
    """One estimate of the policy value; every nuisance is refit from `mult` alone.

    On a line the policy does NOT reroute, the potential outcome under the policy is the
    logged outcome, so y_i is used directly (``observed_when_unswitched``).  Substituting a
    cell mean there would let modelling error on the 96% of lines the policy never touches
    dominate the contrast — under spend weighting that error does not cancel (the spend-
    weighted logged friction rate is 0.61, the count-weighted one 0.26).  Passing False
    reconstructs every line from the cell means instead; that is the mode the identity-policy
    acceptance check uses, because it must reproduce 259/1000 exactly.
    """
    n_strata, n_arms = len(panel.strata), panel.n_arms
    cell = panel.stratum_code * n_arms + panel.arm_code
    tcell = panel.stratum_code * n_arms + target
    size = n_strata * n_arms

    n_sa = np.bincount(cell, weights=mult, minlength=size)
    y_sa = np.bincount(cell, weights=mult * panel.y, minlength=size)
    n_a = np.bincount(panel.arm_code, weights=mult, minlength=n_arms)
    y_a = np.bincount(panel.arm_code, weights=mult * panel.y, minlength=n_arms)

    total_m = float(mult.sum())
    g = float((mult * panel.y).sum() / total_m) if total_m else 0.0
    arm_rate = (y_a + k_arm * g) / (n_a + k_arm)                       # nuisance 1: shrinkage prior
    cell_rate = (y_sa + k_cell * np.tile(arm_rate, n_strata)) / (n_sa + k_cell)

    c = _calibration_scale(cell_rate[cell], panel.y, mult)             # nuisance 2: calibration
    direct = np.clip(c * cell_rate[tcell], 0.0, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        hajek = np.where(n_sa > 0, y_sa / np.maximum(n_sa, 1e-12), 0.0)  # nuisance 3: stratum freqs
    switched = target != panel.arm_code
    supported = n_sa[tcell] >= min_obs
    m_i = np.where(supported, hajek[tcell], direct)
    if observed_when_unswitched:
        m_i = np.where(switched, m_i, panel.y)
        supported = supported | (~switched)

    W = weight * mult
    tw = float(W.sum())
    if tw <= 0:
        return {"value": float("nan"), "logged": float("nan"), "delta": float("nan")}
    value = float((W * m_i).sum() / tw)
    logged = float((W * panel.y).sum() / tw)
    sup_w = float(W[supported].sum())
    return {
        "value": value,
        "logged": logged,
        "delta": value - logged,
        "delta_identified": float((W * supported * (m_i - panel.y)).sum() / tw),
        "worst_case_unsupported": float((W * (~supported) * (1.0 - panel.y)).sum() / tw),
        "supported_share": sup_w / tw,
        "extrapolated_share": 1.0 - sup_w / tw,
        "calibration_scale": c,
        "direct_only_value": float((W * direct).sum() / tw),
        "n_switched": int(switched.sum()),
        "switched_weight_share": float(W[switched].sum() / tw),
        "observed_when_unswitched": bool(observed_when_unswitched),
    }


# --------------------------------------------------------------------------- #
# public estimators — all return (estimate, lo, hi, diagnostics)
# --------------------------------------------------------------------------- #
def hajek_stratified(
    panel: Panel,
    target: np.ndarray,
    weight: str = "spend",
    min_obs: int = SUPPORT_MIN_OBS,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    k_arm: float = K_ARM,
    k_cell: float = K_CELL,
    observed_when_unswitched: bool = True,
) -> Tuple[float, float, float, dict]:
    """Stratified self-normalised (Hajek) policy value, with a clustered bootstrap interval."""
    w = panel.weights(weight)
    mult = np.ones(panel.n, dtype=np.float64)
    point = _core(panel, target, mult, w, min_obs, k_arm, k_cell, observed_when_unswitched)
    lo, hi, boot = (float("nan"), float("nan"), {})
    if n_boot:
        _e, lo, hi, boot = bootstrap_clustered(
            panel, target, weight=weight, min_obs=min_obs, n_boot=n_boot, alpha=alpha,
            seed=seed, k_arm=k_arm, k_cell=k_cell, stat="value",
            observed_when_unswitched=observed_when_unswitched,
        )
    diag = dict(point)
    diag.update(
        {
            "estimator": "stratified Hajek (self-normalised), marginal within-stratum propensity",
            "weight": weight,
            "min_obs_for_support": min_obs,
            "n_boot": n_boot,
            "alpha": alpha,
            "bootstrap": boot,
            "caveat": ESTIMATE_CAVEAT,
        }
    )
    return point["value"], lo, hi, diag


def direct_method(
    panel: Panel,
    target: np.ndarray,
    weight: str = "spend",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    k_arm: float = K_ARM,
    k_cell: float = K_CELL,
) -> Tuple[float, float, float, dict]:
    """Pure outcome-model (shrinkage + calibration) policy value — used only where support fails."""
    w = panel.weights(weight)
    mult = np.ones(panel.n, dtype=np.float64)
    # min_obs above n forces every row onto the direct prediction
    point = _core(panel, target, mult, w, panel.n + 1, k_arm, k_cell, False)
    lo, hi, boot = (float("nan"), float("nan"), {})
    if n_boot:
        _e, lo, hi, boot = bootstrap_clustered(
            panel, target, weight=weight, min_obs=panel.n + 1, n_boot=n_boot, alpha=alpha,
            seed=seed, k_arm=k_arm, k_cell=k_cell, stat="value", observed_when_unswitched=False,
        )
    diag = dict(point)
    diag.update(
        {
            "estimator": "direct method: cell rate shrunk to arm rate to global rate, then calibrated",
            "weight": weight,
            "k_arm": k_arm,
            "k_cell": k_cell,
            "extrapolation": "100% — every row is modelled, none is reweighted",
            "caveat": ESTIMATE_CAVEAT,
        }
    )
    return point["value"], lo, hi, diag


def plain_ips(
    panel: Panel,
    target: np.ndarray,
    weight: str = "count",
    propensity: str = "global",
    top_k: int = 5,
) -> Tuple[float, float, float, dict]:
    """Plain (un-normalised) IPS — computed to be SHOWN BREAKING, with weight concentration."""
    n_arms = panel.n_arms
    if propensity == "global":
        n_a = np.bincount(panel.arm_code, minlength=n_arms).astype(np.float64)
        e = n_a[panel.arm_code] / panel.n
    elif propensity == "stratum":
        n_s = np.bincount(panel.stratum_code, minlength=len(panel.strata)).astype(np.float64)
        cell = panel.stratum_code * n_arms + panel.arm_code
        n_sa = np.bincount(cell, minlength=len(panel.strata) * n_arms).astype(np.float64)
        e = n_sa[cell] / n_s[panel.stratum_code]
    else:
        raise ValueError("propensity must be 'global' or 'stratum'")

    match = (panel.arm_code == target).astype(np.float64)
    base = panel.weights(weight)
    base = base / base.mean()
    w = base * match / np.maximum(e, 1e-12)
    total = float(w.sum())
    value = float((w * panel.y).sum() / panel.n)
    top = np.sort(w)[::-1][:top_k]
    diag = {
        "estimator": "plain IPS (NOT self-normalised) — reported as a failure demonstration",
        "propensity": propensity,
        "weight": weight,
        "top%d_weight_share" % top_k: float(top.sum() / total) if total else float("nan"),
        "kish_ess": kish_ess(w[w > 0]),
        "max_weight": float(w.max()),
        "coverage_share_of_rows": float((w > 0).mean()),
        "out_of_range": bool(value < 0.0 or value > 1.0),
        "why_it_breaks": (
            "a friction rate must lie in [0,1]; un-normalised IPS is not constrained to, and its "
            "mass sits on a few lines, so the estimate is a property of those lines, not the policy"
        ),
        "caveat": ESTIMATE_CAVEAT,
    }
    return value, float("nan"), float("nan"), diag


def support_split(
    panel: Panel, target: np.ndarray, min_obs: int = SUPPORT_MIN_OBS, weight: str = "spend"
) -> dict:
    """Split the rerouted lines into reweighted (>= min_obs observations) vs extrapolated."""
    n_arms = panel.n_arms
    cell = panel.stratum_code * n_arms + panel.arm_code
    tcell = panel.stratum_code * n_arms + target
    n_sa = np.bincount(cell, minlength=len(panel.strata) * n_arms).astype(np.float64)
    switched = target != panel.arm_code
    supported = n_sa[tcell] >= min_obs
    w = panel.weights(weight)
    tw = float(w.sum())
    return {
        "min_obs": min_obs,
        "n_switched": int(switched.sum()),
        "n_switched_reweighted": int((switched & supported).sum()),
        "n_switched_extrapolated": int((switched & ~supported).sum()),
        "switched_spend_share": float(w[switched].sum() / tw),
        "reweighted_spend_share": float(w[switched & supported].sum() / tw),
        "extrapolated_spend_share": float(w[switched & ~supported].sum() / tw),
        "weight": weight,
        "caveat": ESTIMATE_CAVEAT,
    }


def bootstrap_clustered(
    panel: Panel,
    target: np.ndarray,
    weight: str = "spend",
    min_obs: int = SUPPORT_MIN_OBS,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    k_arm: float = K_ARM,
    k_cell: float = K_CELL,
    stat: str = "value",
    observed_when_unswitched: bool = True,
) -> Tuple[float, float, float, dict]:
    """Cluster bootstrap over literal job paths, refitting EVERY nuisance in each replicate."""
    w = panel.weights(weight)
    n_clusters = len(panel.clusters)
    rng = np.random.default_rng(seed)
    point = _core(
        panel, target, np.ones(panel.n), w, min_obs, k_arm, k_cell, observed_when_unswitched
    )[stat]

    draws = np.empty(n_boot, dtype=np.float64)
    extra = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
        mult = counts[panel.cluster_code].astype(np.float64)
        if mult.sum() <= 0:
            draws[b] = np.nan
            extra[b] = np.nan
            continue
        res = _core(panel, target, mult, w, min_obs, k_arm, k_cell, observed_when_unswitched)
        draws[b] = res[stat]
        extra[b] = res.get("extrapolated_share", np.nan)

    good = draws[np.isfinite(draws)]
    lo = float(np.quantile(good, alpha / 2)) if good.size else float("nan")
    hi = float(np.quantile(good, 1 - alpha / 2)) if good.size else float("nan")
    diag = {
        "stat": stat,
        "n_boot": n_boot,
        "n_clusters": n_clusters,
        "cluster": "literal cron job path; rows without one are singleton clusters",
        "refit_per_replicate": ["shrinkage prior", "calibration constant", "stratum arm frequencies"],
        "alpha": alpha,
        "seed": seed,
        "replicates_used": int(good.size),
        "boot_sd": float(good.std(ddof=1)) if good.size > 1 else float("nan"),
        "one_sided_upper": float(np.quantile(good, 1 - alpha)) if good.size else float("nan"),
        "extrapolated_share_mean": float(np.nanmean(extra)) if extra.size else float("nan"),
        "caveat": ESTIMATE_CAVEAT,
    }
    return float(point), lo, hi, diag


def non_inferiority_bound(
    panel: Panel,
    target: np.ndarray,
    weight: str = "spend",
    min_obs: int = SUPPORT_MIN_OBS,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Tuple[None, None, float, dict]:
    """UPPER bound on the friction increase of the routed policy — deliberately never a point."""
    w = panel.weights(weight)
    point = _core(panel, target, np.ones(panel.n), w, min_obs, K_ARM, K_CELL)
    _e, _lo, _hi, boot = bootstrap_clustered(
        panel, target, weight=weight, min_obs=min_obs, n_boot=n_boot, alpha=alpha,
        seed=seed, stat="delta_identified",
    )
    identified_hi = boot["one_sided_upper"]
    manski = point["worst_case_unsupported"]
    hi = identified_hi + manski
    diag = {
        "estimand": "upper bound on E[friction under route] - E[friction logged], spend-weighted",
        "identified_part_one_sided_upper": identified_hi,
        "manski_widening_unsupported": manski,
        "extrapolated_spend_share": point["extrapolated_share"],
        "alpha": alpha,
        "n_boot": n_boot,
        "point_estimate": None,
        "why_no_point": (
            "MDE is 11.4pp on the best-powered arm pair; a point estimate of a friction "
            "difference at this n would be noise dressed as a decision"
        ),
        "worst_case_assumption": (
            "on every line whose stratum has < %d observations of the target arm, the target "
            "arm is assumed to hit friction (y=1)" % min_obs
        ),
        "caveat": ESTIMATE_CAVEAT,
    }
    return None, None, float(hi), diag


# --------------------------------------------------------------------------- #
# acceptance + main
# --------------------------------------------------------------------------- #
def calibration_check(panel: Panel, k_arm: float = K_ARM, k_cell: float = K_CELL) -> Tuple[float, float]:
    """Return (mean calibrated prediction on the logged arms, observed rate) — they must match."""
    res = _core(
        panel, panel.arm_code.copy(), np.ones(panel.n), np.ones(panel.n), panel.n + 1, k_arm, k_cell, False
    )
    return res["direct_only_value"], float(panel.y.mean())


def acceptance(panel: Panel, ess: Sequence[dict], logged_hajek: float, ips_value: float) -> List[tuple]:
    """Return [(pass, label, expected, actual), ...] for every ope acceptance number."""
    rows: List[tuple] = []
    for row in ess:
        expected = EXPECTED_ESS.get(row["arm"])
        actual = row["ess_global_marginal"]
        rows.append(
            (
                expected is not None and abs(actual - expected) < 1e-9,
                "ESS(marginal) %s == n" % row["arm"],
                expected,
                round(actual, 6),
            )
        )
    rows.append(
        (
            abs(logged_hajek - EXPECTED_LOGGED_RATE) < 1e-12,
            "identity-policy Hajek == logged rate",
            EXPECTED_LOGGED_RATE,
            round(logged_hajek, 12),
        )
    )
    rows.append(
        (
            ips_value > 1.0,
            "plain IPS on identity policy is out of range",
            "> 1.0 (impossible rate)",
            round(ips_value, 4),
        )
    )
    pred, obs = calibration_check(panel)
    rows.append(
        (
            abs(pred - obs) < 1e-6,
            "direct method calibrated to the logged rate",
            round(obs, 6),
            round(pred, 6),
        )
    )
    rows.append(
        (
            len(panel.clusters) == 778,
            "bootstrap clusters (152 literal jobs + 626 solo)",
            778,
            len(panel.clusters),
        )
    )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full OPE report: ESS table, IPS failure, Hajek, bounds; print acceptance."""
    records, meta = build_strata()
    panel = load_panel(records)

    print("router.ope — stratified Hajek off-policy evaluation")
    print("  %s" % ESTIMATE_CAVEAT)
    print(
        "  panel: n=%d, %d arms, %d strata, %d bootstrap clusters (feature source: %s)"
        % (panel.n, panel.n_arms, len(panel.strata), len(panel.clusters), meta["feature_source"])
    )
    print(
        "  NO propensity model is fitted: a 9-arm logit on pre-treatment features scores 0.3290\n"
        "  against a 0.3310 majority baseline (negative lift) and costs 40-75% of the Kish ESS."
    )

    ess = arm_ess_table(panel)
    print_ess_table(ess)

    identity = policy_identity(panel)
    logged_hajek, _lo, _hi, _d = hajek_stratified(
        panel, identity, weight="count", min_obs=1, n_boot=0, observed_when_unswitched=False
    )
    ips_value, _, _, ips_diag = plain_ips(panel, identity, weight="count", propensity="global")
    ips_strat, _, _, ips_strat_diag = plain_ips(panel, identity, weight="spend", propensity="stratum")

    print()
    print("IDENTITY-POLICY SANITY (target policy = the logged arm)")
    print("  stratified Hajek, count-weighted : %.6f   (== observed 259/1000, exactly)" % logged_hajek)
    print(
        "  plain IPS, global propensity     : %.4f   OUT OF RANGE -> %s"
        % (ips_value, "BREAKS" if ips_diag["out_of_range"] else "in range")
    )
    print(
        "  plain IPS, stratum propensity    : %.4f   top5 weight share %.1f%%, Kish ESS %.1f of %d"
        % (
            ips_strat,
            100 * ips_strat_diag["top5_weight_share"],
            ips_strat_diag["kish_ess"],
            panel.n,
        )
    )
    print("  -> %s" % ips_diag["why_it_breaks"])

    # headline counterfactual: results/routes.jsonl when it matches the contract, else the
    # reference policy "every eligible claude line onto the cheapest claude arm".
    target = load_routes_policy(panel)
    if target is None:
        target = policy_route_to(panel, "claude-fable-5")
        policy_name = "reference: claude lane -> claude-fable-5"
        policy_source = "built in (results/routes.jsonl absent or not contract-conforming)"
    else:
        policy_name = "results/routes.jsonl"
        policy_source = str(ROUTES_PATH)
    n_switched = int((target != panel.arm_code).sum())
    switched_spend = float(panel.spend[target != panel.arm_code].sum() / panel.spend.sum())
    print()
    print("COUNTERFACTUAL POLICY: %s" % policy_name)
    print("  source: %s" % policy_source)
    print("  %d of %d lines rerouted, %.1f%% of estimated spend"
          % (n_switched, panel.n, 100 * switched_spend))
    print("  routed-to arms: %s" % ", ".join(
        "%s x%d" % (panel.arms[a], c)
        for a, c in sorted(
            ((a, int(((target == a) & (panel.arm_code != a)).sum())) for a in range(panel.n_arms)),
            key=lambda kv: -kv[1],
        )
        if c
    ))

    results = {}
    for weight in ("spend", "count"):
        est, lo, hi, diag = hajek_stratified(panel, target, weight=weight)
        results[weight] = {"estimate": est, "lo": lo, "hi": hi, "diagnostics": diag}
        print(
            "  Hajek [%s-weighted] value %.4f  95%% CI [%.4f, %.4f]  logged %.4f  extrapolated spend %.1f%%"
            % (weight, est, lo, hi, diag["logged"], 100 * diag["extrapolated_share"])
        )

    split = support_split(panel, target)
    print(
        "  SWITCH SUPPORT: %d rerouted lines -> %d reweighted (>= %d obs of the target in the\n"
        "  stratum), %d EXTRAPOLATED via the direct method = %.1f%% of estimated spend"
        % (
            split["n_switched"],
            split["n_switched_reweighted"],
            split["min_obs"],
            split["n_switched_extrapolated"],
            100 * split["extrapolated_spend_share"],
        )
    )

    dm_est, dm_lo, dm_hi, dm_diag = direct_method(panel, target, weight="spend", n_boot=500)
    print(
        "  direct method (all rows modelled) %.4f  95%% CI [%.4f, %.4f]  calibration scale %.3f"
        % (dm_est, dm_lo, dm_hi, dm_diag["calibration_scale"])
    )

    _p, _l, ni_hi, ni_diag = non_inferiority_bound(panel, target)
    print()
    print("NON-INFERIORITY BOUND (upper bound on the friction INCREASE; there is no point estimate)")
    print(
        "  <= %+.1f pp = identified %.1f pp + worst-case %.1f pp over the %.1f%% extrapolated spend"
        % (
            100 * ni_hi,
            100 * ni_diag["identified_part_one_sided_upper"],
            100 * ni_diag["manski_widening_unsupported"],
            100 * ni_diag["extrapolated_spend_share"],
        )
    )
    print("  %s" % ni_diag["why_no_point"])

    # stress case: the aggressive end of the frontier — every eligible claude line on fable
    stress = policy_route_to(panel, "claude-fable-5")
    st_split = support_split(panel, stress)
    st_est, st_lo, st_hi, st_diag = hajek_stratified(panel, stress, weight="spend")
    _sp, _sl, st_ni, st_ni_diag = non_inferiority_bound(panel, stress)
    print()
    print("STRESS POLICY: every eligible claude line -> claude-fable-5 (%d lines, %.1f%% of spend)"
          % (st_split["n_switched"], 100 * st_split["switched_spend_share"]))
    print(
        "  Hajek [spend-weighted] value %.4f  95%% CI [%.4f, %.4f]  logged %.4f  extrapolated %.1f%%"
        % (st_est, st_lo, st_hi, st_diag["logged"], 100 * st_split["extrapolated_spend_share"])
    )
    print("  non-inferiority upper bound on the friction increase: <= %+.1f pp" % (100 * st_ni))

    payload = {
        "policy": {"name": policy_name, "source": policy_source},
        "panel": {
            "n": panel.n,
            "arms": panel.arms,
            "n_strata": len(panel.strata),
            "n_clusters": len(panel.clusters),
            "logged_friction_rate": logged_hajek,
        },
        "ess_table": ess,
        "identity_checks": {
            "hajek_count_weighted": logged_hajek,
            "plain_ips_global": {"value": ips_value, "diagnostics": ips_diag},
            "plain_ips_stratum": {"value": ips_strat, "diagnostics": ips_strat_diag},
        },
        "policy_evaluated": {
            "n_switched": n_switched,
            "switched_spend_share": switched_spend,
            "hajek": results,
            "direct_method": {"estimate": dm_est, "lo": dm_lo, "hi": dm_hi, "diagnostics": dm_diag},
            "support_split": split,
            "non_inferiority_upper_bound": {"estimate": None, "hi": ni_hi, "diagnostics": ni_diag},
        },
        "policy_stress_all_claude_to_fable": {
            "support_split": st_split,
            "hajek_spend": {"estimate": st_est, "lo": st_lo, "hi": st_hi, "diagnostics": st_diag},
            "non_inferiority_upper_bound": {
                "estimate": None,
                "hi": st_ni,
                "diagnostics": st_ni_diag,
            },
        },
        "caveat": ESTIMATE_CAVEAT,
    }
    update_estimates("ope", payload)
    print("\n  wrote section 'ope' into %s/estimates.json" % RESULTS_DIR)

    rows = acceptance(panel, ess, logged_hajek, ips_value)
    print("\nrouter.ope acceptance")
    ok = True
    for good, label, expected, actual in rows:
        ok = ok and bool(good)
        print("  [%s] %-42s expected %-22s actual %s" % ("PASS" if good else "FAIL", label, expected, actual))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
