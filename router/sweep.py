"""The conformal knob, swept: what alpha actually buys, and what it cannot buy.

    results/sweep.json   every swept operating point, machine-readable
    results/sweep.png    the two-panel figure

WHY THIS MODULE EXISTS

`results/NUMBERS.md` #5 and the deck both say the same thing about the shipped
router: at its spend the single-arm mixture hull reaches a better friction bound,
so the routing logic has not earned its complexity. That is true and it is
measured, but on its own it is a verdict without a mechanism -- a reader cannot
tell whether the gate chain is *wrong* or merely *tight*.

It is tight, and that is NOT the explanation. `router.policy` calibrates its
threshold by split conformal at `DEFAULT_ALPHA = 0.10`, a statement about how
often a frictional trajectory may be routed, not a statement about spend. At
that level the gate lets 119 of 1000 trajectories through and moves 3.6% of the
bill. ADR-014 records the arithmetic that follows: the identified delta is a
spend-weighted mean of per-run differences in [0, 1] taken over switched runs
only, so its magnitude cannot exceed the share of spend the policy moves. A
policy moving 3.6% of spend cannot reach a hull 3.83 pp away however well it
ranks -- so at the shipped alpha the miss is not evidence about the ranking.

THE MEASUREMENT THIS MODULE ADDS. Sweep alpha and that excuse expires. By
alpha=0.15 the policy moves 6.06 pp of spend against a hull 3.48 pp away: the
ceiling has stopped binding. It keeps not binding all the way to alpha=0.50,
where the policy moves 23.66 pp and spends $226.61 instead of $412.95. The
router does not get under the hull at any of them. Its bound gets WORSE as the
gate opens (0.76 -> 2.12 pp) while the hull it is chasing gets closer.

So the honest reading is the opposite of the one the deck shipped with: the
conformal threshold is not what limits this router. The ranking is -- and
`model.margin_over_best_unfitted_column.auprc` already said so, at 0.0685 AUPRC
over one raw column with no fit at all. Loosening tau buys spend, not skill.

Each swept point is re-priced through the same `figs.policy_point` the headline
chart uses, so a swept point and a plotted point are the same computation and
cannot disagree.

WHAT IT DELIBERATELY DOES NOT DO

It does not produce a gap number between the router and the hull at any alpha.
ADR-014 forbids that: both curves are one-sided UPPER bounds and their
difference bounds nothing in either direction. The figure plots both levels and
labels them; it never draws the distance between them as a quantity. The one
comparison that IS licensed -- whether the router's bound sits below the hull's
at the same spend -- is a boolean, and it is reported as one.

Everything here is an ESTIMATE: tokens are `len(json.dumps(x)) // 4`, dollars
rest on an assumed sheet, and friction is tool-call exit codes, never answer
quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import figs as _figs
from . import metrics as _metrics
from . import ope as _ope
from . import policy as _policy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSON_PATH = RESULTS_DIR / "sweep.json"
PNG_PATH = RESULTS_DIR / "sweep.png"

#: The grid. Starts below the shipped 0.10 so the figure shows the gate closing
#: as well as opening, and stops at 0.50 because alpha is the permitted
#: probability that a frictional trajectory is routed -- past a coin flip the
#: conformal guarantee is not a guarantee anyone would ship.
ALPHA_GRID = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)

SHIPPED_ALPHA = _policy.DEFAULT_ALPHA

#: Pinned so a refactor that silently changes the sweep fails loudly. These are
#: the shipped operating point, which must reproduce `figs` exactly.
EXPECTED = {
    "shipped_alpha": 0.10,
    "shipped_tau": 0.1708,
    "shipped_switched": 119,
    "shipped_usd": 375.45,
    "shipped_bound_pp": 0.76,
    "shipped_beats_hull": False,
}


def _target_from_rows(panel, rows) -> np.ndarray:
    """Turn one `route_all` result into an ope target vector.

    Mirrors `ope.load_routes_policy` but reads the rows in memory, so a sweep
    never writes over `results/routes.jsonl` -- the artifact the rest of the
    pipeline is pinned to.
    """
    by_idx = {int(r["idx"]): r["route"] for r in rows}
    out = panel.arm_code.copy()
    for i, ix in enumerate(panel.idx):
        arm = by_idx.get(int(ix))
        if arm is None or arm not in panel.arms:
            raise ValueError(
                f"row {ix} routes to {arm!r}, which is not an arm in the panel"
            )
        out[i] = panel.arms.index(arm)
    return out


def sweep_data(alphas=ALPHA_GRID, sheet: str = _figs.DEFAULT_SHEET,
               n_boot: int = _figs.DEFAULT_N_BOOT,
               seed: int = _figs.DEFAULT_SEED) -> dict:
    """Re-run the gate chain at every alpha and price each result like the chart does."""
    panel = _ope.load_panel()
    recon = _figs.load_recon()

    # The single-arm hull, computed by the same code path the headline chart
    # uses. Recomputing it here rather than reading frontier.csv keeps the two
    # figures from ever disagreeing about where the hull is.
    front = _figs.frontier_data(recon=recon, sheet=sheet, n_boot=n_boot, seed=seed)
    hull_q = [(h["cost_usd"], -h["bound_pp"]) for h in front["hull"]]

    points = []
    for alpha in alphas:
        # write=False is load-bearing, not tidiness. `route_all` defaults to
        # writing results/routes.jsonl, so a sweep left at the default overwrites
        # the pipeline's routing artifact eleven times and leaves the LAST alpha
        # behind. Measured: it did, and `make report` caught it -- policy.rerouted.n
        # came back 497 (alpha=0.50) instead of 119.
        res = _policy.route_all(alpha=float(alpha), write=False)
        target = _target_from_rows(panel, res["rows"])
        pt = _figs.policy_point(
            f"gated router (alpha={alpha:.2f})", "sweep", panel, target, recon,
            sheet=sheet, n_boot=n_boot, seed=seed,
        )
        hull_here = -_metrics.hull_value(hull_q, pt["cost_usd"])
        pt.update({
            "alpha": float(alpha),
            "tau": float(res["tau"]),
            "hull_bound_pp": float(hull_here),
            # A boolean, not a distance. ADR-014: the two levels are one-sided
            # upper bounds and their difference is not a quantity we may name.
            "beats_hull": bool(pt["bound_pp"] < hull_here),
            # ADR-014's arithmetic ceiling: |identified delta| <= moved spend.
            # Printed in pp so it sits on the same axis as the bounds.
            "delta_ceiling_pp": float(pt["switched_spend_share"] * 100.0),
            "is_shipped": abs(float(alpha) - SHIPPED_ALPHA) < 1e-9,
        })
        # Does ADR-014's ceiling still EXPLAIN the miss at this alpha? Only while
        # the policy moves less spend than the hull sits away. Once it moves
        # more, "we did not move enough spend" stops being an available excuse
        # and the miss has to be attributed to the ranking instead.
        pt["ceiling_binds"] = bool(pt["delta_ceiling_pp"] < abs(pt["hull_bound_pp"]))
        points.append(pt)

    shipped = next(p for p in points if p["is_shipped"])
    crossing = next((p for p in points if p["beats_hull"]), None)
    unbinds = next((p for p in points if not p["ceiling_binds"]), None)

    return {
        "alphas": [float(a) for a in alphas],
        "points": points,
        "hull": front["hull"],
        "hull_arms": front["hull_arms"],
        "singles": front["plotted_single_arms"],
        "shipped": shipped,
        "crossing": crossing,
        "any_alpha_beats_hull": crossing is not None,
        # The result this module was built to find. If no alpha beats the hull
        # AND the ceiling stops binding partway up the grid, then "our tau is too
        # tight" is not the explanation for the miss -- above `ceiling_unbinds_at`
        # the policy moves more than enough spend and still does not get there.
        "ceiling_unbinds_at": unbinds,
        "ceiling_binds_everywhere": unbinds is None,
        "sheet": sheet,
        "n_boot": n_boot,
        "seed": seed,
        "token_basis": _figs.TOKEN_BASIS,
        "sheet_basis": _figs.SHEET_BASIS,
        "friction_basis": _figs.FRICTION_BASIS,
    }


def write_json(data: dict, path: Path | str = JSON_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
VIOLET = "#6748FD"
NAVY = "#150079"
PEACH = "#FFBD9E"
INK = "#1a1a24"
MUTED = "#6b6b7d"


def render(data: dict, path: Path | str = PNG_PATH) -> Path:
    """Two panels: where the swept router sits against the hull, and why it sits there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = data["points"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- panel 1: cost vs friction upper bound, router path over the hull ----
    hull = data["hull"]
    hx = [h["cost_usd"] for h in hull]
    hy = [h["bound_pp"] for h in hull]
    ax1.plot(hx, hy, color=MUTED, lw=1.6, ls="--", zorder=2,
             label="single-arm mixture hull")
    for s in data["singles"]:
        ax1.scatter([s["cost_usd"]], [s["bound_pp"]], s=38, color=MUTED,
                    zorder=3, marker="s")
        ax1.annotate(s["arm"], (s["cost_usd"], s["bound_pp"]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=7.5, color=MUTED)

    rx = [p["cost_usd"] for p in pts]
    ry = [p["bound_pp"] for p in pts]
    ax1.plot(rx, ry, color=VIOLET, lw=2.0, marker="o", ms=4.5, zorder=4,
             label="gated router, alpha swept")
    ship = data["shipped"]
    ax1.scatter([ship["cost_usd"]], [ship["bound_pp"]], s=150, facecolor="none",
                edgecolor=NAVY, lw=2.2, zorder=5)
    ax1.annotate(f"shipped\nalpha={ship['alpha']:.2f}",
                 (ship["cost_usd"], ship["bound_pp"]),
                 textcoords="offset points", xytext=(10, -22),
                 fontsize=8.5, color=NAVY, fontweight="bold")
    for p in pts:
        if p["alpha"] in (data["alphas"][0], data["alphas"][-1]):
            ax1.annotate(f"a={p['alpha']:.2f}", (p["cost_usd"], p["bound_pp"]),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=7.5, color=VIOLET)

    ax1.axhline(0.0, color=INK, lw=0.7, alpha=0.35)
    ax1.set_xlabel("est. input-token spend, cache-aware, assumed sheet (USD)")
    ax1.set_ylabel("friction upper bound (pp)   lower is better")
    ax1.set_title("The router never gets under the hull, at any alpha", fontsize=11,
                  color=INK, loc="left", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right", frameon=False)

    # ---- panel 2: the mechanism -- moved spend caps the reachable delta ------
    ax = [p["alpha"] for p in pts]
    ceil = [p["delta_ceiling_pp"] for p in pts]
    need = [abs(p["hull_bound_pp"]) for p in pts]
    ax2.plot(ax, ceil, color=VIOLET, lw=2.0, marker="o", ms=4.5,
             label="moved spend = ceiling on |identified delta| (pp)")
    ax2.plot(ax, need, color=PEACH, lw=2.0, marker="s", ms=4.0,
             label="|hull bound| at that spend (pp)")
    ax2.axvline(SHIPPED_ALPHA, color=NAVY, lw=1.4, ls=":", alpha=0.8)
    ax2.annotate("shipped", (SHIPPED_ALPHA, max(need) * 0.92),
                 textcoords="offset points", xytext=(5, 0),
                 fontsize=8.5, color=NAVY, fontweight="bold")

    # Where the two curves cross is the whole point of this panel: to the left,
    # "we did not move enough spend" explains the miss; to the right it does not,
    # and panel 1 shows the miss continuing anyway.
    unb = data.get("ceiling_unbinds_at")
    if unb is not None:
        ax2.axvspan(unb["alpha"], max(ax), color=VIOLET, alpha=0.055, zorder=0)
        ax2.annotate(
            f"from alpha={unb['alpha']:.2f} the policy moves MORE than enough\n"
            f"spend to reach the hull — and panel 1 shows it still does not",
            xy=(unb["alpha"], unb["delta_ceiling_pp"]),
            xytext=(unb["alpha"] + 0.03, max(ceil) * 0.55),
            fontsize=8, color=INK,
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.0},
        )
    ax2.set_xlabel("conformal alpha (permitted P[frictional run is routed])")
    ax2.set_ylabel("pp")
    ax2.set_title("And not because it moves too little spend", fontsize=11,
                  color=INK, loc="left", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", frameon=False)

    for a in (ax1, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(True, color="#e8e8ef", lw=0.7, zorder=0)
        a.set_axisbelow(True)

    fig.suptitle(
        "Both curves are one-sided UPPER bounds. Their difference is not a gap "
        "and is never drawn (ADR-014).",
        fontsize=8.5, color=MUTED, y=0.022,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", dpi=170)
    plt.close(fig)
    return path


def acceptance(data: dict) -> list[tuple]:
    """[(pass, label, expected, actual), ...] — the shipped point must match figs exactly."""
    s = data["shipped"]
    return [
        (abs(s["alpha"] - EXPECTED["shipped_alpha"]) < 1e-9,
         "shipped alpha", EXPECTED["shipped_alpha"], s["alpha"]),
        (abs(s["tau"] - EXPECTED["shipped_tau"]) < 0.001,
         "shipped conformal tau", EXPECTED["shipped_tau"], round(s["tau"], 4)),
        (s["n_switched"] == EXPECTED["shipped_switched"],
         "shipped reroutes", EXPECTED["shipped_switched"], s["n_switched"]),
        (abs(s["cost_usd"] - EXPECTED["shipped_usd"]) < 0.01,
         "shipped spend USD", EXPECTED["shipped_usd"], round(s["cost_usd"], 2)),
        (abs(s["bound_pp"] - EXPECTED["shipped_bound_pp"]) < 0.01,
         "shipped friction upper bound pp", EXPECTED["shipped_bound_pp"],
         round(s["bound_pp"], 2)),
        (s["beats_hull"] == EXPECTED["shipped_beats_hull"],
         "shipped point sits below the hull", EXPECTED["shipped_beats_hull"],
         s["beats_hull"]),
        (data["ceiling_binds_everywhere"] is False,
         "ADR-014 ceiling stops binding somewhere on the grid", False,
         data["ceiling_binds_everywhere"]),
        (data["any_alpha_beats_hull"] is False,
         "no swept alpha puts the router below the hull", False,
         data["any_alpha_beats_hull"]),
        # Monotonicity is the sanity check on the whole sweep: a looser gate must
        # never route strictly fewer trajectories. If this fails the sweep is not
        # measuring the knob it claims to measure.
        (all(a["n_switched"] <= b["n_switched"]
             for a, b in zip(data["points"], data["points"][1:])),
         "reroutes are non-decreasing in alpha", True,
         [p["n_switched"] for p in data["points"]]),
    ]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sheet = argv[0] if argv else _figs.DEFAULT_SHEET
    print("router.sweep — tokens are ESTIMATES, dollars rest on an ASSUMED sheet, "
          "friction is tool-call exit codes and not answer quality.")
    try:
        data = sweep_data(sheet=sheet)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    json_path = write_json(data)
    png_path = render(data)

    print()
    print(f"{'alpha':>6}{'tau':>9}{'reroutes':>10}{'moved spend':>13}"
          f"{'USD':>10}{'bound pp':>10}{'hull pp':>10}{'below hull':>12}")
    for p in data["points"]:
        print(f"{p['alpha']:>6.2f}{p['tau']:>9.4f}{p['n_switched']:>10}"
              f"{p['switched_spend_share'] * 100:>12.2f}%{p['cost_usd']:>10.2f}"
              f"{p['bound_pp']:>10.2f}{p['hull_bound_pp']:>10.2f}"
              f"{p['beats_hull']!s:>12}")

    unb = data.get("ceiling_unbinds_at")
    if unb is not None:
        print(f"FINDING: ADR-014's arithmetic ceiling stops binding at alpha="
              f"{unb['alpha']:.2f} — from there the policy moves "
              f"{unb['delta_ceiling_pp']:.2f} pp of spend against a hull "
              f"{abs(unb['hull_bound_pp']):.2f} pp away, so 'we did not move enough "
              f"spend' no longer explains the miss.")
    else:
        print("FINDING: the arithmetic ceiling binds at every alpha on this grid.")

    print()
    if data["any_alpha_beats_hull"]:
        c = data["crossing"]
        print(f"FINDING: the gate first sits below the mixture hull at alpha="
              f"{c['alpha']:.2f} (tau={c['tau']:.4f}), moving "
              f"{c['switched_spend_share'] * 100:.1f}% of spend.")
    else:
        print("FINDING: no alpha on this grid puts the gated router below the "
              "single-arm mixture hull — including the alphas where the ceiling "
              "no longer binds. The threshold is therefore not what limits this "
              "router; the ranking is.")

    ok = True
    print()
    for passed, label, expected, actual in acceptance(data):
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label}: expected {expected}, got {actual}")

    print()
    print(f"wrote {json_path.relative_to(RESULTS_DIR.parent)}")
    print(f"wrote {png_path.relative_to(RESULTS_DIR.parent)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
