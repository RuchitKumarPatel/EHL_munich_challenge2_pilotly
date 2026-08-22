#!/usr/bin/env python3
"""The routing policy: an inspectable gate chain, a friction-budgeted knapsack, and split conformal tau.

WHAT THIS COMPUTES
    For every one of the 1000 trajectories, which arm the policy would have
    used, and why -- with the "why" written into the record rather than left in
    the code. Writes results/routes.jsonl.

THE GATE CHAIN, IN THIS ORDER (each gate is one function; each leaves a trace)
    1  FAMILY LANE      read from the tools block, never chosen. bash/file_read/
       file_edit/file_write => claude (755); apply_patch/shell_command => gpt
       (245). Perfect separation, 0 exceptions. A cross-family reroute changes
       the tool contract, not just the price, so only in-lane targets exist.
    2  VISION GATE      img_pre > 0 => keep the logged arm. 9 trajectories carry
       an input_image part inside items[:first_user_msg+1]; the cheap sibling's
       vision behaviour is unobserved in this export.
    3  ADMISSIBLE ARMS  drop any arm with n <= 20 logged runs: claude-opus-4-6
       (n=2), claude-sonnet-4-6 (n=1), gpt-5.6-luna (n=20) -- 23 trajectories
       carry one of these as their LOGGED arm. An arm with 2 observed runs
       supports neither a target nor a source claim.
    4  SIGN-STABLE CHEAPER TARGETS  router.pricing.sign_stable_targets: a
       target survives only if it is cheaper than the logged arm under ALL
       THREE assumed sheets. Model ids are anonymized (AGENTS.md) and no public
       sheet applies, so a target that is cheaper under one sheet and dearer
       under another is not a saving, it is a coin flip. This is what empties
       the gpt lane (terra and sol price identically on every sheet) and what
       excludes the 71 runs already on claude-fable-5 (nothing below them).
    5  TAU GATE         calibrated p_fric <= tau => downgrade to the cheapest
       surviving target; otherwise keep the logged arm. p_fric comes from
       router.model, out-of-fold and isotonic-calibrated -- thresholding a raw
       score would make tau and the conformal guarantee meaningless.

THEN THE OPTIMISER
    `knapsack_route` selects by DESCENDING savings / p_fric until a friction
    budget binds. A plain risk threshold optimises the wrong quantity here: the
    top 10 trajectories are 17.7% of the bill, the top 100 are 54.7% and the
    top 500 are 90.2%. Under a flat tau, a thousand cheap low-risk trajectories
    can consume the entire tolerable friction while saving almost nothing,
    while the handful of enormous ones that actually carry the bill sit just
    above tau and never move. Ranking by saving-per-unit-expected-friction is
    the fractional-knapsack ordering for exactly that objective.

AND THE GUARANTEE
    `conformal_tau(cal_scores, cal_y, alpha)` is split conformal on the
    calibration positives: tau is the floor(alpha*(n_pos+1))-th smallest score
    among calibration trajectories that actually were frictional. By
    exchangeability, a genuinely frictional new trajectory is routed with
    probability at most alpha -- distribution-free, no model assumption. The
    implied bound on the realised friction count among routed trajectories is
    alpha * (number of frictional trajectories), i.e. at most alpha * 259 here.
    It is a bound on the MISS RATE among frictional cases; it is not a point
    estimate of the friction the policy causes. That estimate does not exist:
    the MDE on the best-powered arm pair is 11.4pp, which is why
    42-partial-identification reports bounds only.

WHAT IT WRITES
    results/routes.jsonl, 1000 lines, idx ascending:
      {"idx", "logged", "route", "changed", "p_fric",
       "gates": {"family", "vision", "admissible", "sign_stable", "tau_pass"},
       "cost_logged_usd", "cost_routed_usd", "supported"}
    `gates.admissible` is the ARM list surviving gate 3 for this row and
    `gates.sign_stable` is the ARM list surviving gate 4 -- the two candidate
    sets, so a single decision is fully explainable from its own record. (The
    model's sign-stable FEATURE COLUMNS are a different object; router.model
    prints them and `main()` echoes them here.)

    Every dollar figure is input-side only (the export has no `output` field),
    built on ESTIMATED tokens (len(json.dumps(x))//4 -- no `usage` field), and
    carries its ASSUMED price sheet. The sheet is an assumption, not a quote.

ACCEPTANCE (printed by `python -m router.policy`)
    routable set 675/1000 = 197,551,096 gross est. tok = 59.0% of the bill.
    Exclusions, as counts of trajectories carrying the property and as a share
    of gross est. tokens: gpt lane 245 (32.3%), non-admissible logged arm 23
    (7.7%), already-cheapest 71 (7.0%), pre-treatment image 9 (1.7%).
    These four are NOT disjoint -- gpt-5.6-luna's 20 runs are both gpt and
    non-admissible, and 3 of the 9 image runs sit on fable-5 or sonnet-4-6 --
    so the counts sum to 348 and the shares to 48.7%, not to 325 / 41.0%. The
    disjoint first-gate-hit attribution is printed underneath, and it does sum.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from router import model as rmodel
from router.costs import line_cost
from router.pricing import ALL_SHEETS, format_usd, rate_in, sign_stable_targets

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
ROUTES_PATH = os.path.join(RESULTS_DIR, "routes.jsonl")

#: An arm needs more than this many logged runs to be admissible as a source or a target.
MIN_ARM_RUNS = 20

#: The price sheet used for every dollar field written to results/routes.jsonl.
#: ASSUMED -- ids are anonymized and no public sheet applies. See router.pricing.
DEFAULT_SHEET = "assumed_default"

#: Split-conformal miss-rate level: at most this share of genuinely frictional
#: trajectories may be routed onto a cheaper arm.
DEFAULT_ALPHA = 0.10

#: Measured acceptance numbers (verified against results/recon.jsonl).
EXPECTED = {
    "n": 1000,
    "routable_n": 675,
    "routable_gross": 197_551_096,
    "routable_share_pct": 59.0,
    "gpt_n": 245, "gpt_share_pct": 32.3,
    "nonadmissible_n": 23, "nonadmissible_share_pct": 7.7,
    "cheapest_n": 71, "cheapest_share_pct": 7.0,
    "image_n": 9, "image_share_pct": 1.7,
}


# --------------------------------------------------------------------------
# gate 1 -- family lane
# --------------------------------------------------------------------------

def gate_family(recon_rec: dict) -> str:
    """Gate 1: return the family lane read from the tools block ('claude' or 'gpt'); never chosen."""
    return recon_rec["family"]


# --------------------------------------------------------------------------
# gate 2 -- vision
# --------------------------------------------------------------------------

def gate_vision(img_pre: int) -> bool:
    """Gate 2: True when a PRE-TREATMENT image blocks rerouting (img_pre > 0 => keep logged)."""
    return int(img_pre) > 0


# --------------------------------------------------------------------------
# gate 3 -- admissible arms
# --------------------------------------------------------------------------

def arm_counts(recon: Sequence[dict]) -> Dict[str, int]:
    """Logged run count per arm, measured from the recon records."""
    out: Dict[str, int] = {}
    for r in recon:
        out[r["model"]] = out.get(r["model"], 0) + 1
    return out


def admissible_arms(counts: Dict[str, int], min_runs: int = MIN_ARM_RUNS) -> List[str]:
    """Gate 3: arms with strictly more than `min_runs` logged runs, alphabetical."""
    return sorted(a for a, n in counts.items() if n > min_runs)


def gate_admissible(logged: str, family: str, counts: Dict[str, int],
                    min_runs: int = MIN_ARM_RUNS) -> Tuple[bool, List[str]]:
    """Gate 3: (logged arm admissible?, in-lane admissible candidate arms)."""
    ok = counts.get(logged, 0) > min_runs
    lane = "gpt" if family == "gpt" else "claude"
    cands = [a for a in admissible_arms(counts, min_runs)
             if (a.startswith("gpt") if lane == "gpt" else a.startswith("claude"))]
    return ok, cands


# --------------------------------------------------------------------------
# gate 4 -- sign-stable cheaper targets
# --------------------------------------------------------------------------

def gate_sign_stable(logged: str, candidates: Sequence[str], sheets=ALL_SHEETS) -> List[str]:
    """Gate 4: candidates cheaper than `logged` under EVERY assumed sheet (cheapest last)."""
    surviving = sign_stable_targets(logged, [c for c in candidates if c != logged], sheets)
    return sorted(surviving, key=lambda a: -rate_in(a, DEFAULT_SHEET))


def cheapest_target(targets: Sequence[str], sheet: str = DEFAULT_SHEET) -> str | None:
    """The cheapest arm among the sign-stable survivors under the named sheet."""
    if not targets:
        return None
    return min(targets, key=lambda a: rate_in(a, sheet))


# --------------------------------------------------------------------------
# gate 5 -- tau, and the conformal calibration of tau
# --------------------------------------------------------------------------

def gate_tau(p_fric: float, tau: float) -> bool:
    """Gate 5: True when the CALIBRATED friction probability is at or below tau."""
    return float(p_fric) <= float(tau)


def conformal_tau(cal_scores, cal_y, alpha: float = DEFAULT_ALPHA) -> float:
    """Split-conformal tau: a frictional trajectory is routed with probability <= alpha.

    Nonconformity score = the calibrated p_fric. tau is the
    floor(alpha*(n_pos+1))-th smallest score among the CALIBRATION POSITIVES
    (the trajectories that actually were frictional). A new trajectory is
    routed iff its score <= tau, so by exchangeability of the positives a
    genuinely frictional one clears the gate with probability at most alpha.
    Returns -inf (route nothing) when alpha is too small for the calibration
    set to certify any threshold -- the honest answer, not a guessed tau.
    """
    s = np.asarray(cal_scores, dtype=np.float64).ravel()
    y = np.asarray(cal_y, dtype=np.float64).ravel()
    pos = np.sort(s[y == 1])
    n_pos = pos.size
    if n_pos == 0:
        return float("-inf")
    k = int(np.floor(alpha * (n_pos + 1)))
    if k < 1:
        return float("-inf")
    # TIE-SAFE. The conformal p-value of a score t is (1 + #{positives <= t}) /
    # (n_pos + 1), so t is admissible iff #{positives <= t} <= k - 1. Taking the
    # k-th order statistic directly would break the guarantee whenever the
    # calibrator ties many positives at one value, which isotonic calibrators do
    # by construction. We take the LARGEST admissible observed score instead.
    admissible = [t for t in np.unique(s) if int(np.count_nonzero(pos <= t)) <= k - 1]
    if not admissible:
        return float("-inf")
    return float(max(admissible))


# --------------------------------------------------------------------------
# the knapsack
# --------------------------------------------------------------------------

def knapsack_route(records: Sequence[dict], p_fric, savings, budget: float) -> Dict[str, object]:
    """Select routes by descending savings/p_fric until the friction budget binds.

    `records` are the routable candidates (each needs an "idx"); `p_fric` and
    `savings` are same-length sequences, or dicts keyed by idx. `budget` is the
    tolerated EXPECTED number of frictional routed trajectories, i.e. the
    ceiling on sum(p_fric) over the selected set.

    Why ratio-ordered rather than threshold-ordered: gross spend on this corpus
    is extremely concentrated (top 10 = 17.7%, top 100 = 54.7%, top 500 = 90.2%
    of 334.7M est. tokens). A flat tau spends the friction budget on whichever
    trajectories happen to be safe, not on the ones that carry the bill; this
    ordering is the fractional-knapsack optimum for saving per unit of expected
    friction, and it is exact whenever the budget binds mid-item.
    """
    recs = list(records)
    idxs = [r["idx"] for r in recs]
    p = np.array([p_fric[i] for i in idxs] if isinstance(p_fric, dict)
                 else list(p_fric), dtype=np.float64)
    s = np.array([savings[i] for i in idxs] if isinstance(savings, dict)
                 else list(savings), dtype=np.float64)
    if p.size != len(recs) or s.size != len(recs):
        raise ValueError("p_fric and savings must align with records")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p > 0, s / np.where(p > 0, p, 1.0), np.inf)
    ratio = np.where(s <= 0, -np.inf, ratio)  # never spend budget on a non-saving

    order = np.argsort(-ratio, kind="mergesort")
    selected: List[int] = []
    spent = 0.0
    saved = 0.0
    binding = False
    for j in order:
        if not np.isfinite(ratio[j]) and ratio[j] < 0:
            continue
        if spent + p[j] > budget + 1e-12:
            binding = True
            continue
        selected.append(int(idxs[j]))
        spent += float(p[j])
        saved += float(s[j])
    return {
        "selected": sorted(selected),
        "n_selected": len(selected),
        "expected_friction": spent,
        "budget": float(budget),
        "budget_binding": binding,
        "savings": saved,
        "order": [int(idxs[j]) for j in order],
    }


# --------------------------------------------------------------------------
# the whole chain, per trajectory
# --------------------------------------------------------------------------

def support_map(jobkeys: Dict[int, dict], recon: Dict[int, dict]) -> Dict[str, set]:
    """{literal cron path: set of arms observed on it} -- the support test for a reroute."""
    out: Dict[str, set] = {}
    for idx, jk in jobkeys.items():
        if jk.get("literal") and jk.get("cron_path"):
            out.setdefault(jk["cron_path"], set()).add(recon[idx]["model"])
    return out


def route_line(rec: dict, p_fric: float, img_pre: int, counts: Dict[str, int],
               tau: float, sheet: str = DEFAULT_SHEET, sheets=ALL_SHEETS,
               force_keep: bool = False, **cost_kw) -> dict:
    """Run the five gates on one recon record and return its results/routes.jsonl entry."""
    logged = rec["model"]
    family = gate_family(rec)                                   # gate 1
    vision_blocks = gate_vision(img_pre)                        # gate 2
    logged_ok, cands = gate_admissible(logged, family, counts)  # gate 3
    stable = [] if not logged_ok else gate_sign_stable(logged, cands, sheets)  # gate 4
    if vision_blocks:
        stable = []
    tau_pass = gate_tau(p_fric, tau)                            # gate 5

    target = cheapest_target(stable, sheet)
    route = target if (target is not None and tau_pass and not force_keep) else logged
    return {
        "idx": rec["idx"],
        "logged": logged,
        "route": route,
        "changed": route != logged,
        "p_fric": float(p_fric),
        "gates": {
            "family": family,
            "vision": bool(vision_blocks),
            "admissible": cands if logged_ok else [],
            "sign_stable": stable,
            "tau_pass": bool(tau_pass),
        },
        "cost_logged_usd": round(line_cost(rec, logged, sheet, **cost_kw), 6),
        "cost_routed_usd": round(line_cost(rec, route, sheet, **cost_kw), 6),
        "supported": True,
    }


def routable_mask(recon: Sequence[dict], img_pre: Dict[int, int],
                  counts: Dict[str, int], sheets=ALL_SHEETS) -> np.ndarray:
    """Boolean mask of trajectories surviving gates 1-4 (tau not yet applied)."""
    out = []
    for r in recon:
        fam = gate_family(r)
        if gate_vision(img_pre.get(r["idx"], 0)):
            out.append(False)
            continue
        ok, cands = gate_admissible(r["model"], fam, counts)
        out.append(bool(ok and gate_sign_stable(r["model"], cands, sheets)))
    return np.array(out, dtype=bool)


def route_all(tau: float | None = None, alpha: float = DEFAULT_ALPHA,
              budget: float | None = None, sheet: str = DEFAULT_SHEET,
              out_path: str = ROUTES_PATH, write: bool = True,
              fit: Dict[str, object] | None = None, **cost_kw) -> Dict[str, object]:
    """Run the gate chain over all 1000 trajectories and write results/routes.jsonl.

    `tau` defaults to `conformal_tau` at level `alpha`. When `budget` is given,
    the knapsack additionally trims the tau-passing set down to that expected
    friction budget, and any trajectory the knapsack drops is written back with
    its logged arm and `changed = false`.
    """
    fit = rmodel.load_pfric() if fit is None else fit
    idx = np.asarray(fit["idx"], dtype=int)
    p = np.asarray(fit["p_fric"], dtype=np.float64)
    y = np.asarray(fit["y"], dtype=np.float64)
    img_pre = fit["img_pre"]

    recon_map = rmodel.load_recon()
    recon = [recon_map[int(i)] for i in idx]
    counts = arm_counts(list(recon_map.values()))
    jobkeys = rmodel.load_jobkeys()
    support = support_map(jobkeys, recon_map)

    if tau is None:
        tau = conformal_tau(p, y, alpha)

    rows = [route_line(r, float(pi), int(img_pre.get(int(r["idx"]), 0)), counts,
                       tau, sheet=sheet, **cost_kw)
            for r, pi in zip(recon, p)]

    knap = None
    if budget is not None:
        cand = [row for row in rows if row["changed"]]
        keep = set(knapsack_route(
            cand,
            {row["idx"]: row["p_fric"] for row in cand},
            {row["idx"]: row["cost_logged_usd"] - row["cost_routed_usd"] for row in cand},
            budget,
        )["selected"])
        knap = {"budget": float(budget), "kept": len(keep), "candidates": len(cand)}
        for row in rows:
            if row["changed"] and row["idx"] not in keep:
                row["route"] = row["logged"]
                row["changed"] = False
                row["cost_routed_usd"] = row["cost_logged_usd"]
                row["gates"]["tau_pass"] = False

    for row in rows:
        if not row["changed"]:
            continue
        jk = jobkeys.get(row["idx"]) or {}
        path = jk.get("cron_path") if jk.get("literal") else None
        row["supported"] = bool(path and row["route"] in support.get(path, set()))

    if write:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    mask = routable_mask(recon, img_pre, counts)
    return {
        "rows": rows, "tau": float(tau), "alpha": float(alpha), "sheet": sheet,
        "routable_mask": mask, "recon": recon, "counts": counts,
        "knapsack": knap, "fit": fit, "path": out_path if write else None,
    }


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------

def exclusion_table(recon: Sequence[dict], img_pre: Dict[int, int],
                    counts: Dict[str, int]) -> Tuple[List[tuple], List[tuple]]:
    """(property counts/shares, disjoint first-gate-hit counts/shares) over gross est. tokens."""
    total = sum(r["gross_tok"] for r in recon)
    nonadm = {a for a, n in counts.items() if n <= MIN_ARM_RUNS}

    def block(pred):
        sel = [r for r in recon if pred(r)]
        g = sum(r["gross_tok"] for r in sel)
        return len(sel), g, 100.0 * g / total

    props = [
        ("gpt lane", lambda r: r["family"] == "gpt"),
        ("non-admissible logged arm", lambda r: r["model"] in nonadm),
        ("already cheapest (fable-5)", lambda r: r["model"] == "claude-fable-5"),
        ("pre-treatment image", lambda r: img_pre.get(r["idx"], 0) > 0),
    ]
    prop_rows = [(name, *block(pred)) for name, pred in props]

    def first_hit(r):
        if r["family"] == "gpt":
            return "gpt lane"
        if img_pre.get(r["idx"], 0) > 0:
            return "pre-treatment image"
        if r["model"] in nonadm:
            return "non-admissible logged arm"
        if r["model"] == "claude-fable-5":
            return "already cheapest (fable-5)"
        return "ROUTABLE"
    order = ["gpt lane", "pre-treatment image", "non-admissible logged arm",
             "already cheapest (fable-5)", "ROUTABLE"]
    disjoint = [(name, *block(lambda r, n=name: first_hit(r) == n)) for name in order]
    return prop_rows, disjoint


def main(argv=None) -> int:
    """Run the gate chain, write results/routes.jsonl, print every acceptance number."""
    print("router.policy -- gate chain, friction-budgeted knapsack, split-conformal tau")
    print("Dollars are INPUT-SIDE ONLY (no `output` field in the export), built on")
    print("ESTIMATED tokens (len(json.dumps(x))//4 -- no `usage` field), under an ASSUMED")
    print(f"price sheet ({DEFAULT_SHEET}). Model ids are anonymized; no public sheet applies.\n")

    fit = rmodel.load_pfric()
    print(f"p_fric source: {fit['source']} (out-of-fold, isotonic-calibrated, k={fit['k']})")
    print("sign-stable FEATURE columns from router.model "
          f"({len(fit['sign_stable_cols'])}/{len(fit['cols'])}):")
    print("  " + (", ".join(fit["sign_stable_cols"]) or "(none)") + "\n")

    res = route_all(alpha=DEFAULT_ALPHA, fit=fit)
    rows, recon, counts = res["rows"], res["recon"], res["counts"]
    img_pre = fit["img_pre"]
    total = sum(r["gross_tok"] for r in recon)

    nonadm = sorted(a for a, n in counts.items() if n <= MIN_ARM_RUNS)
    print(f"gate 3 -- arms with n <= {MIN_ARM_RUNS} logged runs are non-admissible: "
          + ", ".join(f"{a} (n={counts[a]})" for a in nonadm) + "\n")

    props, disjoint = exclusion_table(recon, img_pre, counts)
    mask = res["routable_mask"]
    r_n = int(mask.sum())
    r_g = int(sum(r["gross_tok"] for r, m in zip(recon, mask) if m))

    print("routable set and exclusions (share = % of 334,729,910 gross ESTIMATED tokens):")
    print(f"  {'category':<34}{'runs':>7}{'gross est. tok':>18}{'share':>9}")
    print(f"  {'ROUTABLE (gates 1-4 pass)':<34}{r_n:>7}{r_g:>18,}{100.0 * r_g / total:>8.1f}%")
    for name, n, g, s in props:
        print(f"  {'excluded: ' + name:<34}{n:>7}{g:>18,}{s:>8.1f}%")
    print("  (these four OVERLAP: luna is both gpt and non-admissible, and 3 of the 9")
    print("   image runs sit on fable-5 / sonnet-4-6. Disjoint attribution:)")
    print(f"  {'category (first gate hit)':<34}{'runs':>7}{'gross est. tok':>18}{'share':>9}")
    for name, n, g, s in disjoint:
        print(f"  {name:<34}{n:>7}{g:>18,}{s:>8.1f}%")
    print(f"  {'TOTAL':<34}{sum(n for _, n, _, _ in disjoint):>7}"
          f"{sum(g for _, _, g, _ in disjoint):>18,}"
          f"{sum(s for _, _, _, s in disjoint):>8.1f}%\n")

    # --- tau, conformal, and the knapsack -------------------------------
    tau = res["tau"]
    changed = [r for r in rows if r["changed"]]
    y = np.asarray(fit["y"])
    n_pos = int(y.sum())
    print(f"gate 5 -- split-conformal tau at alpha={DEFAULT_ALPHA:.2f}: tau = {tau:.4f}")
    print(f"  guarantee: a genuinely frictional trajectory clears the gate with probability")
    print(f"  <= {DEFAULT_ALPHA:.2f}; implied ceiling on frictional routed runs "
          f"= {DEFAULT_ALPHA:.2f} x {n_pos} = {DEFAULT_ALPHA * n_pos:.1f}.")
    print(f"  It is a MISS-RATE bound, not an estimate of friction caused. No point")
    print(f"  estimate exists here: the MDE on the best-powered arm pair is 11.4pp.\n")

    saved = sum(r["cost_logged_usd"] - r["cost_routed_usd"] for r in rows)
    logged_usd = sum(r["cost_logged_usd"] for r in rows)
    print(f"tau-gated route: {len(changed)} of {r_n} routable trajectories move "
          f"({100.0 * len(changed) / max(1, r_n):.1f}% of the routable set)")
    print(f"  expected frictional routed runs = sum p_fric over moved = "
          f"{sum(r['p_fric'] for r in changed):.1f}")
    print(f"  logged  {format_usd(logged_usd, DEFAULT_SHEET, compact=True)}")
    print(f"  saved   {format_usd(saved, DEFAULT_SHEET, compact=True)}"
          f"   ({100.0 * saved / logged_usd:.1f}% of the input-side bill)")
    print(f"  supported (job observed on the routed arm): "
          f"{sum(1 for r in changed if r['supported'])}/{len(changed)}\n")

    # the knapsack sees EVERY routable trajectory, tau-passing or not, and ranks
    # them by the saving each would deliver if it were moved to its cheapest
    # sign-stable target.
    by_idx = {r["idx"]: r for r in recon}
    cand = [r for r in rows if r["gates"]["sign_stable"]]
    potential = {
        r["idx"]: r["cost_logged_usd"]
        - line_cost(by_idx[r["idx"]], cheapest_target(r["gates"]["sign_stable"]), DEFAULT_SHEET)
        for r in cand
    }
    knap_budget = DEFAULT_ALPHA * n_pos
    knap = knapsack_route(cand, {r["idx"]: r["p_fric"] for r in cand}, potential, knap_budget)
    print(f"knapsack over the whole routable set at friction budget {knap_budget:.1f}:")
    print(f"  selects {knap['n_selected']}/{len(cand)} by descending savings/p_fric; "
          f"expected friction {knap['expected_friction']:.1f}; budget binds: {knap['budget_binding']}")
    print(f"  savings {format_usd(knap['savings'], DEFAULT_SHEET, compact=True)}")
    print("  (vs the flat-tau set above -- the ratio ordering buys more saving per unit")
    print("   of expected friction because the top 100 runs are 54.7% of the bill.)\n")

    # ------------------------------------------------------------- checks
    ok = True

    def check(label, expected, actual):
        nonlocal ok
        good = expected == actual
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label:<48} expected {expected!r:>16}  actual {actual!r:>16}")

    prop = {name: (n, g, s) for name, n, g, s in props}
    print("acceptance:")
    check("lines written", EXPECTED["n"], len(rows))
    check("idx ascending 0..999", True, [r["idx"] for r in rows] == list(range(1000)))
    check("routable runs", EXPECTED["routable_n"], r_n)
    check("routable gross est. tok", EXPECTED["routable_gross"], r_g)
    check("routable share of bill", EXPECTED["routable_share_pct"], round(100.0 * r_g / total, 1))
    for key, name in (("gpt", "gpt lane"), ("nonadmissible", "non-admissible logged arm"),
                      ("cheapest", "already cheapest (fable-5)"), ("image", "pre-treatment image")):
        check(f"excluded: {name} (runs)", EXPECTED[f"{key}_n"], prop[name][0])
        check(f"excluded: {name} (share)", EXPECTED[f"{key}_share_pct"], round(prop[name][2], 1))
    check("disjoint attribution sums to n", EXPECTED["n"], sum(n for _, n, _, _ in disjoint))
    check("disjoint attribution sums to the whole bill", total,
          sum(g for _, _, g, _ in disjoint))
    check("no excluded trajectory changed arm", 0,
          sum(1 for r, m in zip(rows, mask) if r["changed"] and not m))
    check("no cross-family route", 0,
          sum(1 for r in rows if r["changed"] and r["gates"]["family"] != "claude"))
    check("no routed trajectory carries a pre-treatment image", 0,
          sum(1 for r in rows if r["changed"] and r["gates"]["vision"]))
    check("every routed target is sign-stable cheaper", True,
          all(r["route"] in r["gates"]["sign_stable"] for r in rows if r["changed"]))
    check("cost_routed <= cost_logged on every changed row", True,
          all(r["cost_routed_usd"] <= r["cost_logged_usd"] + 1e-9 for r in rows if r["changed"]))
    check("unchanged rows have equal logged/routed cost", True,
          all(r["cost_routed_usd"] == r["cost_logged_usd"] for r in rows if not r["changed"]))
    check("conformal tau routes <= alpha of the frictional runs", True,
          bool(sum(1 for r, yy in zip(rows, y) if r["changed"] and yy == 1)
               <= DEFAULT_ALPHA * n_pos + 1))
    check("knapsack respects its friction budget", True,
          bool(knap["expected_friction"] <= knap_budget + 1e-9))

    print(f"\nwrote {res['path']}")
    print("RESULT: " + ("all checks pass" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
