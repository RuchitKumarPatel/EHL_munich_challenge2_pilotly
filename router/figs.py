#!/usr/bin/env python3
"""The single headline chart: cost against a friction UPPER BOUND, never a point.

WHAT THIS COMPUTES
    One panel, one question: if you move spend off the logged arms, how much
    process friction could you be buying, at worst?

      X  re-metered cache-aware spend in USD, log scale. The re-metered basis is
         the reconstructed turn-prefix bill from results/recon.jsonl
         (334,729,910 est. tok), not the starter kit's flat per-item sum
         (22,631,879 est. tok). A ghost tick marks where the starter kit's
         dollar figure lands, and the logged operating point is marked.
      Y  the 95% ONE-SIDED UPPER BOUND on the spend-weighted friction increase
         over the logged policy, in percentage points. Never a point estimate:
         the MDE is 11.4pp on the best-powered arm pair, so a point difference
         at this n would be noise dressed as a decision. y = 0 is logged parity.
         The axis measures PROCESS FRICTION, NOT ANSWER QUALITY.

    On top of that: the non-decreasing convex hull over the admissible
    single-arm policies (dashed — everything on or under it is reachable by
    randomising between two arms, so a router only earns its complexity by
    sitting below it on this axis); job-clustered bootstrap whiskers on three
    named operating points; a shaded region carrying the refused share of
    spend; and the support-deficiency share as a hatched rug on its own axis.

WHY THE SUPPORT DEFICIT IS A RUG AND NOT A BAND
    The share of spend whose target arm was never observed in its stratum is a
    dimensionless fraction of dollars. Drawn as a vertical band on the friction
    axis it would be read as a confidence interval, which it is not — it is a
    statement about which counterfactuals the data can speak to at all. It gets
    its own hatched axis, its own scale and its own units.

WHAT IT WRITES
    results/frontier.png   the chart (matplotlib Agg, no display required)
    results/frontier.csv   every plotted point, with the basis on every row

ACCEPTANCE (checked by `python -m router.figs`)
    Reproduced from the artifacts, not transcribed:
      logged re-metered spend                    $412.95  [assumed_default]
      starter-kit naive spend                    $146.14  [assumed_default]
        -> the token bases differ by 14.79x but the DOLLAR figures differ by
           2.83x, because the re-metered bill also earns the cache discount.
           Quoting 14.79x as a dollar correction would be wrong.
      refused spend       325 lines, 41.0% of est. gross tokens, $56.72
        -> the refused share is token-heavy and dollar-light (13.7% of $),
           because most of it is the cheap gpt lane.
      hull vertices       claude-fable-5 -> claude-sonnet-5 -> claude-opus-5
      gated router        $375.45, bound <= +0.76pp, 119 reroutes
      all-eligible -> claude-fable-5   $78.22, bound <= +6.65pp, 681 reroutes
      and the finding the chart exists to show: the gated router does NOT sit
      below the single-arm mixture frontier.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from router import costs as _costs  # noqa: E402
from router import metrics as _metrics  # noqa: E402
from router import ope as _ope  # noqa: E402
from router import pricing as _pricing  # noqa: E402

# Dollar signs are everywhere on this chart; without this a "$57 ... $" pair is
# parsed as TeX and the render dies. Disable mathtext for all text, globally.
matplotlib.rcParams["text.parse_math"] = False
matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RECON_PATH = RESULTS_DIR / "recon.jsonl"
ROUTES_PATH = RESULTS_DIR / "routes.jsonl"
PNG_PATH = RESULTS_DIR / "frontier.png"
CSV_PATH = RESULTS_DIR / "frontier.csv"

DEFAULT_SHEET = "assumed_default"
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 12345

#: Brand palette. Violet on white, peach as the single accent.
VIOLET = "#6748FD"
PEACH = "#FFBD9E"
INK = "#1B1240"
MUTED = "#7A719E"
GRIDC = "#E7E3F7"
GHOST = "#B9B2D4"

#: Arms with enough logged runs to be an admissible single-arm policy.
#: Matches router.policy.MIN_ARM_RUNS; claude-opus-4-6 (n=2) and
#: claude-sonnet-4-6 (n=1) are excluded and are supported in ZERO strata.
MIN_ARM_RUNS = 20

TOKEN_BASIS = (
    "ESTIMATED tokens: tok(x) = len(json.dumps(x)) // 4. The export has no "
    "usage field and no output field, so dollars are input-side only."
)
SHEET_BASIS = (
    "Every price sheet is an ASSUMPTION — model ids are anonymized per "
    "AGENTS.md and no public sheet applies. Quote the spread, not one figure."
)
FRICTION_BASIS = (
    "PROCESS FRICTION (y_fric from tool-output exit codes), NOT answer quality. "
    "Values are 95% one-sided UPPER BOUNDS on the increase over logged, never "
    "point estimates."
)

#: Acceptance ground truth, all reproduced from the artifacts at run time.
EXPECTED = {
    "logged_usd": 412.95,
    "naive_usd": 146.14,
    "dollar_ratio": 2.83,
    "token_ratio": 14.79,
    "refused_n": 325,
    "refused_gross_share": 0.410,
    "refused_usd": 56.72,
    "hull_arms": ["claude-fable-5", "claude-sonnet-5", "claude-opus-5"],
    "gated_usd": 375.45,
    "gated_bound_pp": 0.76,
    "gated_switched": 119,
    "fable_usd": 78.22,
    "named_labels": ["gated router (tau)", "all eligible -> claude-sonnet-5",
                     "all eligible -> claude-fable-5"],
    "fable_bound_pp": 6.65,
    "fable_switched": 681,
    "router_beats_hull": False,
    "routable_family": ["claude"],
    "off_lane_runs": 245,
}


# ------------------------------------------------------------------ loading

def load_recon(path: Path | str = RECON_PATH) -> list[dict]:
    """Read results/recon.jsonl into a list of records in file order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m router.recon` first")
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_routes(path: Path | str = ROUTES_PATH) -> list[dict]:
    """Read results/routes.jsonl into a list of contract-shaped route records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m router.policy` first")
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if rows and ("idx" not in rows[0] or not isinstance(rows[0].get("route"), str)):
        raise ValueError(f"{path} does not match the routes.jsonl contract")
    return rows


def naive_cost(recon: list[dict], sheet: str = DEFAULT_SHEET) -> float:
    """USD if you bill the starter kit's flat per-item token sum with no cache discount."""
    return sum(
        r["naive_tok"] * _pricing.rate_in(r["model"], sheet) / 1e6 for r in recon
    )


def refused_spend(recon: list[dict], routes: list[dict],
                  sheet: str = DEFAULT_SHEET, **cost_kw) -> dict:
    """The spend no router can move: lines with no cheaper sign-stable in-lane target."""
    by_idx = {r["idx"]: r for r in recon}
    refused = [by_idx[r["idx"]] for r in routes if not r["gates"]["sign_stable"]]
    gross_total = sum(r["gross_tok"] for r in recon)
    usd_total = _costs.total_cost(recon, None, sheet, **cost_kw)
    usd = _costs.total_cost(refused, None, sheet, **cost_kw)
    return {
        "n": len(refused),
        "gross_tok": sum(r["gross_tok"] for r in refused),
        "gross_share": sum(r["gross_tok"] for r in refused) / gross_total,
        "usd": usd,
        "usd_share": usd / usd_total if usd_total else 0.0,
        "sheet": sheet,
        "meaning": (
            "lines the router refuses: gpt lane, non-admissible logged arm, "
            "already on the cheapest arm, or a pre-treatment image. Their cost "
            "is a floor under every policy on this chart."
        ),
    }


# ------------------------------------------------------ policies and points

def admissible_arms(panel, min_runs: int = MIN_ARM_RUNS) -> list[str]:
    """Arms with at least `min_runs` logged trajectories — the single-arm policy set."""
    counts = np.bincount(panel.arm_code, minlength=len(panel.arms))
    return [a for a, c in zip(panel.arms, counts) if c >= min_runs]


def target_to_route(panel, target: np.ndarray) -> dict[int, str]:
    """Turn an ope target-arm code vector into the {idx: arm} map router.costs wants."""
    return {int(i): panel.arms[int(c)] for i, c in zip(panel.idx, target)}


def policy_point(label: str, kind: str, panel, target: np.ndarray,
                 recon: list[dict], sheet: str = DEFAULT_SHEET,
                 n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                 whisker: bool = False, arm: str | None = None,
                 **cost_kw) -> dict:
    """One plotted policy: its re-metered cost, its friction upper bound, its support deficit."""
    route = target_to_route(panel, target)
    cost = _costs.total_cost(recon, route, sheet, **cost_kw)
    _e, _lo, hi, diag = _ope.non_inferiority_bound(
        panel, target, n_boot=n_boot, seed=seed
    )
    point = {
        "label": label,
        "kind": kind,
        "arm": arm or "",
        "cost_usd": float(cost),
        "sheet": sheet,
        "bound_pp": float(hi) * 100.0,
        "manski_pp": float(diag["manski_widening_unsupported"]) * 100.0,
        "identified_upper_pp": float(diag["identified_part_one_sided_upper"]) * 100.0,
        "extrapolated_spend_share": float(diag["extrapolated_spend_share"]),
        # The ARITHMETIC CEILING on this point's identified delta. The delta is a
        # spend-weighted mean of per-run differences in [0, 1] taken over switched
        # runs only, so |delta| can never exceed the share of spend the policy
        # moves. A policy that moves little spend cannot reach a distant hull
        # however well it routes. See ADR-014.
        "switched_spend_share": float(
            panel.weights("spend")[target != panel.arm_code].sum()
            / panel.weights("spend").sum()
        ),
        "n_switched": int((target != panel.arm_code).sum()),
        "ci_lo_pp": None,
        "ci_hi_pp": None,
        "point_pp": None,
    }
    if whisker:
        pt, lo, hi2, _bd = _ope.bootstrap_clustered(
            panel, target, stat="delta_identified", n_boot=n_boot, seed=seed
        )
        point["point_pp"] = float(pt) * 100.0
        point["ci_lo_pp"] = float(lo) * 100.0
        point["ci_hi_pp"] = float(hi2) * 100.0
    return point


def frontier_data(recon: list[dict] | None = None, routes: list[dict] | None = None,
                  sheet: str = DEFAULT_SHEET, n_boot: int = DEFAULT_N_BOOT,
                  seed: int = DEFAULT_SEED, **cost_kw) -> dict:
    """Everything the chart and the CSV need, computed once from the artifacts."""
    recon = load_recon() if recon is None else recon
    routes = load_routes() if routes is None else routes
    panel = _ope.load_panel()

    logged_usd = _costs.total_cost(recon, None, sheet, **cost_kw)
    logged = {
        "label": "logged (as served)", "kind": "logged", "arm": "",
        "cost_usd": float(logged_usd), "sheet": sheet,
        "bound_pp": 0.0, "manski_pp": 0.0, "identified_upper_pp": 0.0,
        "extrapolated_spend_share": 0.0, "switched_spend_share": 0.0,
        "n_switched": 0,
        "ci_lo_pp": None, "ci_hi_pp": None, "point_pp": 0.0,
    }

    # --- the admissible single-arm policies: the mixture frontier ------------
    arms = admissible_arms(panel)
    singles = []
    for arm in arms:
        tgt = _ope.policy_route_to(panel, arm)
        singles.append(policy_point(
            f"all eligible -> {arm}", "single_arm", panel, tgt, recon,
            sheet=sheet, n_boot=n_boot, seed=seed, arm=arm, **cost_kw
        ))
    singles.sort(key=lambda p: p["cost_usd"])

    # Hull in (cost, quality) space with quality = -bound_pp: lower bound is
    # better, so negating turns the shared geometry helper the right way up.
    hull_q = _metrics.nondecreasing_convex_hull(
        [(p["cost_usd"], -p["bound_pp"]) for p in singles]
    )
    hull = [{"cost_usd": x, "bound_pp": -q} for x, q in hull_q]
    hull_costs = {round(h["cost_usd"], 6) for h in hull}
    for p in singles:
        p["on_hull"] = round(p["cost_usd"], 6) in hull_costs and any(
            abs(-h["bound_pp"] - (-p["bound_pp"])) < 1e-9
            and abs(h["cost_usd"] - p["cost_usd"]) < 1e-9 for h in hull
        )
    hull_arms = [p["arm"] for p in singles if p["on_hull"]]

    # --- the three named operating points, with whiskers ---------------------
    gated_target = _ope.load_routes_policy(panel)
    if gated_target is None:
        raise ValueError(
            "results/routes.jsonl was rejected by router.ope.load_routes_policy; "
            "the gated router point cannot be drawn from a policy we cannot read"
        )
    named = [policy_point(
        "gated router (tau)", "operating", panel, gated_target, recon,
        sheet=sheet, n_boot=n_boot, seed=seed, whisker=True, **cost_kw
    )]
    # The label must be the SAME string the single-arm point uses, or the two
    # spellings become two claim keys for one policy in results/claims.json.
    for arm in ("claude-sonnet-5", "claude-fable-5"):
        tgt = _ope.policy_route_to(panel, arm)
        named.append(policy_point(
            f"all eligible -> {arm}", "operating", panel, tgt, recon,
            sheet=sheet, n_boot=n_boot, seed=seed, whisker=True, arm=arm,
            **cost_kw
        ))

    # The only lane the router can move is the one with a cheaper sign-stable
    # in-lane target. The other lane's single-arm policies are still computed
    # and still written to the CSV, but they are not drawn: they are not
    # policies this router can propose, and under this sheet all of that lane's
    # arms price identically, so none of them moves the bill at all.
    routable_family = sorted({r["gates"]["family"] for r in routes
                              if r["gates"]["sign_stable"]})
    fam_of = {p["arm"]: ("gpt" if p["arm"].startswith("gpt") else "claude")
              for p in singles}
    for p in singles:
        p["family"] = fam_of[p["arm"]]
        p["plotted"] = p["family"] in routable_family
    off_lane = [p for p in singles if not p["plotted"]]
    off_idx = {r["idx"] for r in routes
               if r["gates"]["family"] not in routable_family}
    off_lane_summary = {
        "family": sorted({p["family"] for p in off_lane}),
        "n_arms": len(off_lane),
        "n_runs": len(off_idx),
        "gross_share": (sum(r["gross_tok"] for r in recon if r["idx"] in off_idx)
                        / sum(r["gross_tok"] for r in recon)),
    }

    named_costs = {round(p["cost_usd"], 6) for p in named}
    plotted_singles = [p for p in singles if p["plotted"]
                       and (round(p["cost_usd"], 6) not in named_costs or not p["arm"])]

    refused = refused_spend(recon, routes, sheet=sheet, **cost_kw)
    naive = naive_cost(recon, sheet)
    gross_tok = sum(r["gross_tok"] for r in recon)
    naive_tok = sum(r["naive_tok"] for r in recon)

    gated = named[0]
    hull_at_gated = -_metrics.hull_value(hull_q, gated["cost_usd"])
    return {
        "logged": logged,
        "single_arms": singles,
        "plotted_single_arms": plotted_singles,
        "hull": hull,
        "hull_arms": hull_arms,
        "routable_family": routable_family,
        "off_lane": off_lane_summary,
        "named": named,
        "refused": refused,
        "naive_usd": float(naive),
        "logged_usd": float(logged_usd),
        "dollar_ratio": float(logged_usd / naive) if naive else float("nan"),
        "gross_tok": int(gross_tok),
        "naive_tok": int(naive_tok),
        "token_ratio": float(gross_tok / naive_tok) if naive_tok else float("nan"),
        "gated_hull_bound_pp": float(hull_at_gated),
        "gated_gap_vs_hull_pp": float(gated["bound_pp"] - hull_at_gated),
        "router_beats_hull": bool(gated["bound_pp"] < hull_at_gated),
        "sheet": sheet,
        "n_boot": n_boot,
        "seed": seed,
        "token_basis": TOKEN_BASIS,
        "sheet_basis": SHEET_BASIS,
        "friction_basis": FRICTION_BASIS,
    }


# ----------------------------------------------------------------- the CSV

CSV_COLUMNS = (
    "label", "kind", "arm", "cost_usd", "sheet", "bound_pp",
    "identified_upper_pp", "manski_pp", "point_pp", "ci_lo_pp", "ci_hi_pp",
    "extrapolated_spend_share", "switched_spend_share", "n_switched",
    "on_hull", "plotted", "basis",
)


def write_csv(data: dict, path: Path | str = CSV_PATH) -> Path:
    """Write every plotted point to results/frontier.csv, basis carried on each row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    basis = (
        "cost = re-metered cache-aware est. USD, input-side only, "
        f"sheet={data['sheet']} (ASSUMED); bound_pp = 95% one-sided UPPER bound "
        "on the spend-weighted PROCESS-FRICTION increase over logged, in pp, "
        "never a point"
    )
    rows = [data["logged"]] + list(data["single_arms"]) + list(data["named"])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in CSV_COLUMNS}
            out["on_hull"] = bool(r.get("on_hull", False))
            out["plotted"] = bool(r.get("plotted", True))
            out["basis"] = basis
            writer.writerow(out)
    return path


# ------------------------------------------------------------- the drawing

def _style_axes(ax) -> None:
    """Apply the shared projector-readable axis styling."""
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK, labelsize=12, length=4, width=1.0)


def _label_groups(points, key_round: int = 6):
    """Group markers that share an x position, so their labels can be staggered not stacked."""
    groups: dict[float, list] = {}
    for p in points:
        groups.setdefault(round(p["cost_usd"], key_round), []).append(p)
    return groups


def _stagger(ys: list[float], min_gap: float,
             ceiling: float | None = None) -> list[float]:
    """Push label anchors apart top-down so no two sit closer than min_gap in data units."""
    out: list[float] = []
    for y in sorted(ys, reverse=True):
        if ceiling is not None and y > ceiling:
            y = ceiling
        if out and (out[-1] - y) < min_gap:
            y = out[-1] - min_gap
        out.append(y)
    return out


def render(data: dict, path: Path | str = PNG_PATH) -> Path:
    """Draw and save the single headline chart to results/frontier.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    singles = data["plotted_single_arms"]
    named = data["named"]
    logged = data["logged"]
    hull = data["hull"]
    refused = data["refused"]
    gated = named[0]

    all_pts = ([logged] + [p for p in data["single_arms"] if p["plotted"]]
               + named)
    lo_y = min([p["bound_pp"] for p in all_pts]
               + [p["ci_lo_pp"] for p in named if p["ci_lo_pp"] is not None])
    hi_y = max([p["bound_pp"] for p in all_pts]
               + [p["ci_hi_pp"] for p in named if p["ci_hi_pp"] is not None])
    span = hi_y - lo_y
    ylim = (lo_y - 0.22 * span, hi_y + 0.20 * span)
    xlim = (refused["usd"] * 0.70, max(p["cost_usd"] for p in all_pts) * 1.55)

    fig = plt.figure(figsize=(15.0, 9.2), dpi=170, facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[7.0, 1.15], hspace=0.10,
        left=0.078, right=0.975, top=0.775, bottom=0.225,
    )
    ax = fig.add_subplot(gs[0])
    rug = fig.add_subplot(gs[1], sharex=ax)

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _style_axes(ax)
    ax.grid(True, which="major", axis="both", color=GRIDC, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    # --- the refused floor: spend no router can move -------------------------
    # Rotated inside its own band so it can never collide with the plot body.
    ax.axvspan(xlim[0], refused["usd"], color=PEACH, alpha=0.45, zorder=1, lw=0)
    ax.axvline(refused["usd"], color="#D97B49", lw=1.5, alpha=0.95, zorder=2)
    ax.text(
        (xlim[0] * refused["usd"]) ** 0.5, ylim[0] + 0.5 * (ylim[1] - ylim[0]),
        f"REFUSED FLOOR  ${refused['usd']:,.0f}  ·  {refused['n']} lines\n"
        f"{refused['gross_share']:.1%} of est. tokens, "
        f"{refused['usd_share']:.1%} of est. $",
        rotation=90, ha="center", va="center", fontsize=10.5, linespacing=1.6,
        color="#8A4B23", zorder=6,
    )

    # --- logged parity ------------------------------------------------------
    ax.axhline(0.0, color=INK, lw=1.3, ls=(0, (6, 4)), alpha=0.5, zorder=3)
    ax.text(
        xlim[1] * 0.99, 0.012 * (ylim[1] - ylim[0]),
        "logged parity", ha="right", va="bottom", fontsize=11.5,
        color=INK, alpha=0.7, zorder=6,
    )

    # --- the mixture hull ----------------------------------------------------
    # Drawn as the TRUE linear mixture set (dense sample in linear cost), which
    # is why it curves on a log axis. Straight segments here would be a lie.
    hull_q = [(h["cost_usd"], -h["bound_pp"]) for h in hull]
    hx = np.linspace(hull_q[0][0], hull_q[-1][0], 400)
    hy = np.array([-_metrics.hull_value(hull_q, x) for x in hx])
    ax.plot(hx, hy, color=VIOLET, lw=2.4, ls=(0, (7, 4)), alpha=0.9, zorder=4)

    # --- single-arm policies, labels staggered within each shared x ----------
    for xpos, group in _label_groups(singles).items():
        group = sorted(group, key=lambda p: p["bound_pp"], reverse=True)
        ceiling = (0.035 * (ylim[1] - ylim[0])
                   if abs(xpos - logged["cost_usd"]) < 1e-6 else None)
        anchors = _stagger([p["bound_pp"] for p in group],
                           0.058 * (ylim[1] - ylim[0]), ceiling)
        for p, ay in zip(group, anchors):
            ax.plot(p["cost_usd"], p["bound_pp"], marker="o", ms=10,
                    mfc="white", mec=VIOLET, mew=2.0, zorder=5)
            ax.annotate(
                p["arm"], xy=(p["cost_usd"], p["bound_pp"]),
                xytext=(xpos * 1.085, ay), textcoords="data",
                ha="left", va="center", fontsize=11, color=MUTED, zorder=6,
                arrowprops=dict(arrowstyle="-", color=GHOST, lw=0.9,
                                shrinkA=1, shrinkB=3),
            )

    # --- the three named operating points, with job-clustered whiskers -------
    label_offsets = {"gated router (tau)": (0, 58)}
    for p in named:
        lo, hi = p["ci_lo_pp"], p["ci_hi_pp"]
        ax.plot([p["cost_usd"]] * 2, [lo, hi], color=VIOLET, lw=2.6,
                alpha=0.38, solid_capstyle="butt", zorder=5)
        for cap in (lo, hi):
            ax.plot([p["cost_usd"]], [cap], marker="_", ms=13, mew=2.6,
                    color=VIOLET, alpha=0.38, zorder=5)
        ax.plot(p["cost_usd"], p["bound_pp"], marker="D", ms=12,
                mfc=VIOLET, mec="white", mew=1.6, zorder=7)
        ax.annotate(
            f"{p['label']}\n<= {p['bound_pp']:+.2f} pp   ${p['cost_usd']:,.0f}",
            (p["cost_usd"], p["bound_pp"]),
            textcoords="offset points",
            xytext=label_offsets.get(p["label"], (0, 26)), ha="center",
            fontsize=12, color=INK, fontweight="bold", linespacing=1.4, zorder=7,
        )

    # --- the logged point ----------------------------------------------------
    ax.plot(logged["cost_usd"], 0.0, marker="*", ms=26, mfc=PEACH,
            mec=INK, mew=1.5, zorder=8)
    ax.annotate(
        f"LOGGED  ${logged['cost_usd']:,.0f}",
        (logged["cost_usd"], 0.0), textcoords="offset points",
        xytext=(20, -4), ha="left", fontsize=12.5, color=INK,
        fontweight="bold", zorder=8,
    )

    # --- the two levels the chart exists to show, drawn geometrically --------
    # Both endpoints are ONE-SIDED UPPER bounds (router/ope.py non_inferiority_bound
    # returns hi only), so their difference bounds the true difference in neither
    # direction. The arrow shows the two levels; it deliberately does not label
    # the distance between them as a gap. See ADR-014.
    hull_at_gated = data["gated_hull_bound_pp"]
    ax.annotate(
        "", xy=(gated["cost_usd"], gated["bound_pp"]),
        xytext=(gated["cost_usd"], hull_at_gated), textcoords="data",
        arrowprops=dict(arrowstyle="<|-|>", color=VIOLET, lw=1.8,
                        shrinkA=1, shrinkB=1), zorder=6,
    )
    ax.annotate(
        f"router bound {gated['bound_pp']:+.2f} pp\n"
        f"mixture bound {hull_at_gated:+.2f} pp\n"
        "both one-sided UPPER bounds",
        xy=(gated["cost_usd"], 0.5 * (gated["bound_pp"] + hull_at_gated)),
        xytext=(-18, -58), textcoords="offset points", ha="right", va="center",
        fontsize=11.5, color=VIOLET, linespacing=1.5, zorder=6,
    )

    # --- what is deliberately not drawn --------------------------------------
    off = data["off_lane"]
    ax.text(
        0.995, 0.035,
        f"NOT DRAWN: the {'/'.join(off['family'])} lane ({off['n_runs']} runs, "
        f"{off['gross_share']:.1%} of est. tokens) is refused at the family gate, and "
        f"under this sheet all {off['n_arms']} of its arms price identically,\n"
        "so no single-arm policy there moves the bill. Those points are computed and "
        "kept in results/frontier.csv with plotted=False.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        color=MUTED, linespacing=1.6, zorder=6,
    )

    # --- the ghost tick ------------------------------------------------------
    for target in (ax, rug):
        target.axvline(data["naive_usd"], color=GHOST, lw=1.5, ls=(0, (2, 3)),
                       alpha=0.95, zorder=2)

    ax.set_ylabel(
        "friction non-inferiority UPPER bound, pp above logged\n"
        "PROCESS FRICTION, NOT ANSWER QUALITY",
        fontsize=13, color=INK, labelpad=12,
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _p: "0" if abs(v) < 1e-9 else f"{v:+.0f}"
    ))
    plt.setp(ax.get_xticklabels(), visible=False)

    # --- the support-deficiency rug -----------------------------------------
    _style_axes(rug)
    rug_points = [p for p in data["single_arms"] if p["plotted"]] + named
    deficits = [p["extrapolated_spend_share"] * 100.0 for p in rug_points]
    rug.set_ylim(0, max(deficits) * 1.55)
    rug.grid(True, axis="y", color=GRIDC, linewidth=1.0, zorder=0)
    rug.set_axisbelow(True)
    for p in rug_points:
        x = p["cost_usd"]
        rug.bar(x, p["extrapolated_spend_share"] * 100.0, width=x * 0.075,
                color="white", edgecolor=VIOLET, hatch="////", linewidth=1.2,
                zorder=3)
    rug.set_ylabel("support deficit\n(% of est. spend)", fontsize=10.5,
                   color=INK, labelpad=12)
    rug.text(
        0.5, 0.93,
        "hatched, its own scale — share of est. spend whose target arm was never "
        "observed in its stratum. A dimensionless fraction of dollars, NOT a "
        "confidence interval.",
        transform=rug.transAxes, ha="center", va="top", fontsize=10,
        color=MUTED, zorder=6,
    )

    ticks = [50, 100, 200, 400, 700]
    rug.set_xticks(sorted(ticks + [data["naive_usd"]]))
    rug.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _p: f"${v:,.0f}")
    )
    rug.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for tick_label in rug.get_xticklabels():
        if tick_label.get_position()[0] == data["naive_usd"]:
            tick_label.set_color(GHOST)
            tick_label.set_fontstyle("italic")
    rug.set_xlabel(
        "re-metered cache-aware spend, est. USD (log scale)  ·  "
        f"sheet = {data['sheet']}, ASSUMED  ·  input-side only\n"
        f"the grey ghost tick at ${data['naive_usd']:,.0f} is what the starter kit "
        f"counts: flat per-item tokens, no cache split — {data['token_ratio']:.2f}x "
        f"fewer tokens, but only {data['dollar_ratio']:.2f}x fewer dollars",
        fontsize=12, color=INK, labelpad=10, linespacing=1.7,
    )

    # --- titles and the caveats that travel with every number ----------------
    fig.text(
        0.078, 0.982,
        "What moving spend off the logged arms could cost you, at worst",
        fontsize=23, color=INK, fontweight="bold", ha="left", va="top",
    )
    fig.text(
        0.078, 0.936,
        "n = 1000 trajectories. Every marker is a 95% ONE-SIDED UPPER BOUND on the "
        "spend-weighted process-friction increase over the logged policy, never a point\n"
        "estimate (MDE is 11.4 pp on the best-powered arm pair). Whiskers are 95% "
        "job-clustered bootstrap CIs on the IDENTIFIED part only; each marker adds the\n"
        "Manski widening for unsupported spend, so it sits above its own whisker.",
        fontsize=11.5, color=MUTED, ha="left", va="top", linespacing=1.65,
    )
    verdict = "BELOW" if data["router_beats_hull"] else "ABOVE"
    fig.text(
        0.078, 0.856,
        f"FINDING: the gated router sits {verdict} the single-arm mixture frontier "
        f"— {gated['bound_pp']:+.2f} pp against {hull_at_gated:+.2f} pp at the same spend.\n"
        "Do NOT subtract those: both are one-sided UPPER bounds. It moves only "
        f"{gated['switched_spend_share']:.1%} of est. spend, which caps its delta.",
        fontsize=12.5, color=VIOLET, ha="left", va="top", fontweight="bold",
        linespacing=1.5,
    )
    fig.text(
        0.078, 0.030, TOKEN_BASIS + "\n" + SHEET_BASIS,
        fontsize=9.5, color=MUTED, ha="left", va="center", linespacing=1.6,
    )

    handles = [
        plt.Line2D([], [], color=VIOLET, lw=2.4, ls=(0, (7, 4)),
                   label="mixture frontier over the admissible in-lane "
                         "single-arm policies"),
        plt.Line2D([], [], marker="o", ls="none", ms=10, mfc="white",
                   mec=VIOLET, mew=2.0, label="admissible in-lane single-arm policy"),
        plt.Line2D([], [], marker="D", ls="none", ms=11, mfc=VIOLET,
                   mec="white", mew=1.4, label="named operating point"),
        plt.Line2D([], [], marker="*", ls="none", ms=18, mfc=PEACH, mec=INK,
                   mew=1.2, label="logged policy, as served"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.526, 0.072),
        ncol=4, fontsize=11, frameon=False, handletextpad=0.9,
        columnspacing=2.0,
    )

    fig.savefig(path, facecolor="white", dpi=170)
    plt.close(fig)
    return path


# ------------------------------------------------------------- acceptance

def acceptance(data: dict) -> list[tuple]:
    """Return [(pass, label, expected, actual), ...] for every figs acceptance number."""
    named = {p["label"]: p for p in data["named"]}
    gated = named["gated router (tau)"]
    fable = named["all eligible -> claude-fable-5"]
    rows = [
        (abs(data["logged_usd"] - EXPECTED["logged_usd"]) < 0.01,
         "logged re-metered spend USD", EXPECTED["logged_usd"],
         round(data["logged_usd"], 2)),
        (abs(data["naive_usd"] - EXPECTED["naive_usd"]) < 0.01,
         "starter-kit naive spend USD", EXPECTED["naive_usd"],
         round(data["naive_usd"], 2)),
        (abs(data["dollar_ratio"] - EXPECTED["dollar_ratio"]) < 0.005,
         "dollar ratio re-metered/naive", EXPECTED["dollar_ratio"],
         round(data["dollar_ratio"], 2)),
        (abs(data["token_ratio"] - EXPECTED["token_ratio"]) < 0.005,
         "token ratio gross/naive", EXPECTED["token_ratio"],
         round(data["token_ratio"], 2)),
        (data["refused"]["n"] == EXPECTED["refused_n"],
         "refused lines", EXPECTED["refused_n"], data["refused"]["n"]),
        (abs(data["refused"]["gross_share"] - EXPECTED["refused_gross_share"]) < 0.001,
         "refused share of est. gross tokens", EXPECTED["refused_gross_share"],
         round(data["refused"]["gross_share"], 3)),
        (abs(data["refused"]["usd"] - EXPECTED["refused_usd"]) < 0.01,
         "refused spend USD", EXPECTED["refused_usd"],
         round(data["refused"]["usd"], 2)),
        (data["hull_arms"] == EXPECTED["hull_arms"],
         "hull vertices", EXPECTED["hull_arms"], data["hull_arms"]),
        (abs(gated["cost_usd"] - EXPECTED["gated_usd"]) < 0.01,
         "gated router spend USD", EXPECTED["gated_usd"],
         round(gated["cost_usd"], 2)),
        (abs(gated["bound_pp"] - EXPECTED["gated_bound_pp"]) < 0.01,
         "gated router friction upper bound pp", EXPECTED["gated_bound_pp"],
         round(gated["bound_pp"], 2)),
        (gated["n_switched"] == EXPECTED["gated_switched"],
         "gated router reroutes", EXPECTED["gated_switched"], gated["n_switched"]),
        (abs(fable["cost_usd"] - EXPECTED["fable_usd"]) < 0.01,
         "all-eligible->fable spend USD", EXPECTED["fable_usd"],
         round(fable["cost_usd"], 2)),
        (abs(fable["bound_pp"] - EXPECTED["fable_bound_pp"]) < 0.01,
         "all-eligible->fable upper bound pp", EXPECTED["fable_bound_pp"],
         round(fable["bound_pp"], 2)),
        (fable["n_switched"] == EXPECTED["fable_switched"],
         "all-eligible->fable reroutes", EXPECTED["fable_switched"],
         fable["n_switched"]),
        ([p["label"] for p in data["named"]] == EXPECTED["named_labels"],
         "named operating points", EXPECTED["named_labels"],
         [p["label"] for p in data["named"]]),
        (data["routable_family"] == EXPECTED["routable_family"],
         "lane the router can move", EXPECTED["routable_family"],
         data["routable_family"]),
        (data["off_lane"]["n_runs"] == EXPECTED["off_lane_runs"],
         "off-lane runs not drawn", EXPECTED["off_lane_runs"],
         data["off_lane"]["n_runs"]),
        (data["router_beats_hull"] == EXPECTED["router_beats_hull"],
         "gated router sits below the mixture frontier",
         EXPECTED["router_beats_hull"], data["router_beats_hull"]),
    ]
    return rows


def main(argv: list[str] | None = None) -> int:
    """Rebuild results/frontier.png + frontier.csv and print the acceptance checks."""
    argv = list(sys.argv[1:] if argv is None else argv)
    sheet = argv[0] if argv else DEFAULT_SHEET
    print("router.figs — all token counts are ESTIMATES (the export has no "
          "usage field); dollars are input-side only and every sheet is ASSUMED.")
    try:
        data = frontier_data(sheet=sheet)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    csv_path = write_csv(data)
    png_path = render(data)

    ok = True
    print()
    for passed, label, expected, actual in acceptance(data):
        ok = ok and passed
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {label:46s} expected {expected!r:>42}  actual {actual!r}")
    print()
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print()
    print("READ THE CHART THIS WAY: the y axis is PROCESS FRICTION, not answer "
          "quality, and every marker is an UPPER BOUND, not an effect.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
