#!/usr/bin/env python3
"""The router console: a local HTTP server over the pipeline. `python -m router.app`.

WHAT THIS COMPUTES
    Nothing of its own. It is the seam between the browser and the modules that
    already produce every number: router.outcome for the composite axis,
    router.figs for a policy's cost and its friction bound, router.policy for the
    gate chain, router.serve for the fold ensemble behind the live demo. Every
    figure the console renders comes back through one of those, unrounded, with
    the basis string the module attached to it.

WHAT IT SERVES
    GET  /                  router/console.html -- the console
    GET  /explorer          site/index.html     -- the organizers' raw-trajectory viewer
    GET  /api/bootstrap     the static half: presets, component base rates, lanes,
                            incumbent priors, price sheets, and the caveat list
    POST /api/frontier      {preset|weights, sheet, alpha, n_boot} -> the frontier
    POST /api/route         {task, lane, trigger, ...} -> the live selection demo

OFFLINE AND LOCAL BY CONSTRUCTION
    Binds 127.0.0.1 only. No network call, no API key, no CDN, no webfont:
    router/console.html inlines its own CSS and draws its charts in hand-written
    SVG, so the console works with the laptop's wifi off -- which is the state a
    demo machine is in exactly when it matters.

    The console lives in router/ rather than site/ on purpose: site/ is the
    organizers' shipped explorer and is not ours to change. We serve it read-only
    at /explorer and keep our own page beside our own code.

PROPRIETARY DATA NEVER CROSSES THE SEAM
    The export is challenge-use only. No endpoint returns trajectory text, a
    trajectory id, a job key, a cron path or a PII_ / <ENTITY_> placeholder. What
    crosses is: aggregate counts, estimated token medians, dollar totals, friction
    bounds, arm names and the user's OWN typed task echoed back through the
    feature extractor. `assert_clean()` enforces that on every response body rather
    than trusting each handler, and `--selftest` checks it against the live payloads.

FIRST REQUEST IS SLOW ON PURPOSE
    /api/route needs the fold ensemble, which is a real refit (about 9 s). It is
    built once at startup, before the port opens, so the first click in the browser
    is fast and nobody demos against a half-warm process.

ACCEPTANCE (printed by `python -m router.app --selftest`)
    /api/bootstrap, /api/frontier and /api/route all answer without a network call;
    the frontier under preset repo_baseline reproduces results/frontier.csv:
    logged $412.95, all-eligible->fable $78.22 at bound +6.65 pp, gated router
    $375.45 at +0.76 pp;
    every response body passes assert_clean();
    two different tasks return two different selection distributions.

CAVEAT CARRIED ON EVERY PAYLOAD
    Token counts are ESTIMATES (tok(x) = len(json.dumps(x)) // 4; no `usage` field).
    Dollars are input-side only under an ASSUMED sheet. The outcome axis is PROCESS
    friction from tool-output exit codes, never answer quality.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from router import figs as _figs
from router import model as rmodel
from router import ope as _ope
from router import outcome as _outcome
from router import policy as rpolicy
from router import serve as _serve
from router.costs import total_cost
from router.pricing import ALL_SHEETS

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = REPO_ROOT / "site"
RESULTS_DIR = REPO_ROOT / "results"

#: Our page. It sits next to the module that serves it, not in site/, which holds
#: the organizers' explorer and is read-only for us.
CONSOLE_HTML = Path(__file__).resolve().parent / "console.html"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: Bootstrap replicates for an interactive frontier redraw. router.figs uses 2000
#: for the published chart; at 8 policies that is 20 s, which is not a slider.
#: The console asks for this by default and offers the published setting explicitly.
INTERACTIVE_N_BOOT = 400
PUBLISHED_N_BOOT = 2000

#: Caveats shown in the console's footer, verbatim from the modules that own them.
CAVEATS = [
    "All token counts are ESTIMATES: tok(x) = len(json.dumps(x)) // 4. The export has no `usage` field.",
    "All dollar amounts are INPUT-SIDE (the export has no `output` field) and name their ASSUMED price sheet.",
    "Model IDs are anonymized (AGENTS.md), so no public price sheet applies. Gate 4 therefore requires a target to be cheaper under ALL three sheets.",
    "The Y axis is PROCESS friction from tool exit codes, never answer quality. This export has no quality label.",
    "The composite is a DECLARED weighting, not a measured quantity. Always report the weights with the number.",
    "Every policy point is a 95% UPPER BOUND on the friction increase, never a point estimate: the MDE is 11.4 pp.",
    "The spread in the live demo is OUR uncertainty (5 folds + tau bootstrap). The router itself does not roll dice.",
]


# --------------------------------------------------------------------------- #
# data-safety gate on every response
# --------------------------------------------------------------------------- #
#: Redaction placeholders and identifiers that must never leave the process.
#: `<ENTITY_N>` and `PII_KIND_N` are the export's own redaction tokens; a cron path
#: or a job key would identify a customer's schedule even without them.
#: `<PERSON_ROBERT>` and `<COMPANY_A>` carry no trailing digit, so a pattern anchored
#: on `_\d+>` would pass the most common placeholders straight through. Match the
#: whole `<UPPER_SNAKE>` shape instead.
_FORBIDDEN = (
    re.compile(r"<[A-Z][A-Z0-9_]*>"),
    re.compile(r"\bPII_[A-Z]+_\d+"),
    re.compile(r"\bcron_path\b"),
    re.compile(r"\bjob_key\b"),
)


def assert_clean(payload: Any) -> Any:
    """Refuse to serve a body carrying a redaction placeholder or a job identifier.

    The endpoints are written not to leak, but "written not to" is not a control.
    This runs on the serialised body of every response, so a future handler that
    starts echoing a trajectory field fails loudly here instead of quietly shipping
    proprietary data to a browser.
    """
    blob = json.dumps(payload, ensure_ascii=False)
    for pattern in _FORBIDDEN:
        hit = pattern.search(blob)
        if hit:
            raise ValueError(
                f"response body carries a forbidden token {hit.group(0)!r} — "
                "refusing to serve it"
            )
    return payload


# --------------------------------------------------------------------------- #
# the frontier
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}
_FRONTIER_CACHE: dict[tuple, dict] = {}


def state() -> dict[str, Any]:
    """Load the artifacts the console needs, once, and keep them for the process."""
    with _LOCK:
        if not _STATE:
            _STATE["recon"] = _figs.load_recon()
            _STATE["ensemble"] = _serve.get_ensemble()
            _STATE["labels"] = _outcome.load_labels()
        return _STATE


def _fit_from_ensemble(ens: _serve.Ensemble) -> dict:
    """The dict router.policy.route_all expects, filled from the already-built ensemble.

    Passing this in is what lets the console move alpha without paying for a second
    refit: route_all only reads idx / p_fric / y / img_pre off it, and all four are
    the same objects the acceptance check already matched against routes.jsonl.
    """
    return {
        "idx": ens.idx,
        "p_fric": ens.oof_p,
        "y": ens.y,
        "img_pre": ens._img_pre(),
    }


def _target_vector(panel, rows: list[dict]) -> np.ndarray:
    """Map a route_all row list onto the panel's arm-code vector."""
    by_idx = {int(r["idx"]): r["route"] for r in rows}
    arm_of = {a: i for i, a in enumerate(panel.arms)}
    out = panel.arm_code.copy()
    for i, ix in enumerate(panel.idx):
        arm = by_idx.get(int(ix))
        if arm is None or arm not in arm_of:
            raise ValueError(f"route for idx {int(ix)} is not an arm in the panel")
        out[i] = arm_of[arm]
    return out


def _point(label: str, kind: str, panel, target, recon, logged_badness: float,
           sheet: str, n_boot: int, arm: str | None = None) -> dict:
    """One chart point: cost on x, the composite outcome FLOOR on y, Manski band on top.

    The y value is deliberately a floor, not a point. router.ope refuses to return a
    point estimate of a friction difference at this sample size (MDE 11.4 pp), and
    inverting a refusal into a plotted dot would be worse than plotting nothing. So:

        outcome_floor      = 1 - (logged badness + the 95% upper bound on the increase)
        outcome_identified = 1 - (logged badness + the identified part of that bound)

    Both are lower bounds on the composite outcome proxy. The gap between them IS the
    Manski widening -- the price of trajectories whose target arm was never observed
    in their stratum. A tall band means the estimate leans on extrapolation; the rug
    value `extrapolated_spend_share` says how much spend that covers.
    """
    p = _figs.policy_point(label, kind, panel, target, recon, sheet=sheet,
                           n_boot=n_boot, arm=arm)
    bound = p["bound_pp"] / 100.0
    identified = p["identified_upper_pp"] / 100.0
    return {
        "label": label,
        "kind": kind,
        "arm": arm or "",
        "cost_usd": p["cost_usd"],
        "bound_pp": p["bound_pp"],
        "manski_pp": p["manski_pp"],
        "identified_upper_pp": p["identified_upper_pp"],
        "outcome_floor": 1.0 - (logged_badness + bound),
        "outcome_identified": 1.0 - (logged_badness + identified),
        "extrapolated_spend_share": p["extrapolated_spend_share"],
        "n_switched": p["n_switched"],
        "sheet": sheet,
    }


def frontier(preset: str | None = None, weights: dict | None = None,
             sheet: str = rpolicy.DEFAULT_SHEET, alpha: float = rpolicy.DEFAULT_ALPHA,
             n_boot: int = INTERACTIVE_N_BOOT) -> dict:
    """Every point the chart draws, under one declared weighting and one price sheet."""
    w = _outcome.resolve_weights(preset, weights)
    key = (tuple(sorted(w.items())), sheet, round(float(alpha), 6), int(n_boot))
    with _LOCK:
        if key in _FRONTIER_CACHE:
            return _FRONTIER_CACHE[key]

    st = state()
    recon, ens = st["recon"], st["ensemble"]
    panel = _outcome.composite_panel(w, labels=st["labels"])
    logged_badness = _outcome.logged_level(panel, "spend")

    points = [{
        "label": "logged (as delivered)", "kind": "logged", "arm": "",
        "cost_usd": total_cost(recon, None, sheet),
        "bound_pp": 0.0, "manski_pp": 0.0, "identified_upper_pp": 0.0,
        "outcome_floor": 1.0 - logged_badness,
        "outcome_identified": 1.0 - logged_badness,
        "extrapolated_spend_share": 0.0, "n_switched": 0, "sheet": sheet,
    }]

    for arm in _figs.admissible_arms(panel):
        points.append(_point(f"all admissible → {arm}", "single_arm", panel,
                             _ope.policy_route_to(panel, arm), recon,
                             logged_badness, sheet, n_boot, arm=arm))

    run = rpolicy.route_all(alpha=alpha, sheet=sheet, write=False,
                            fit=_fit_from_ensemble(ens))
    gated = _point(f"gated router (tau, alpha={alpha:g})", "operating", panel,
                   _target_vector(panel, run["rows"]), recon,
                   logged_badness, sheet, n_boot)
    gated["tau"] = run["tau"]
    gated["n_changed"] = sum(1 for r in run["rows"] if r["changed"])
    points.append(gated)

    # The hull is taken over the SINGLE-ARM points only: a mixture of two single-arm
    # policies is itself reachable, so the segment between them is a real option and
    # the gated router has to beat it, not merely beat the arms it sits between.
    singles = [p for p in points if p["kind"] == "single_arm"]
    hull = _figs._metrics.nondecreasing_convex_hull(
        [(p["cost_usd"], p["outcome_floor"]) for p in singles]
    )
    hull_pts = [{"cost_usd": x, "outcome_floor": q} for x, q in hull]
    for p in singles:
        p["on_hull"] = any(abs(h["cost_usd"] - p["cost_usd"]) < 1e-9
                           and abs(h["outcome_floor"] - p["outcome_floor"]) < 1e-12
                           for h in hull_pts)

    described = _outcome.describe(w)
    described["caveat"] = (
        "identical to the published binary y_fric—every number matches results/"
        if described["is_repo_baseline"] else
        "a DECLARED weighting, not a measured quality label—report the weights with the number"
    )
    out = {
        "points": points,
        "hull": hull_pts,
        "logged_badness": logged_badness,
        "outcome": described,
        "sheet": sheet,
        "alpha": alpha,
        "n_boot": n_boot,
        "axis": {
            "x": "estimated cost USD (cache-aware, input-side, ASSUMED sheet)",
            "y": "Composite Outcome Proxy — LOWER BOUND (1 − badness − 95% upper bound on the increase)",
        },
    }
    with _LOCK:
        _FRONTIER_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# the live demo
# --------------------------------------------------------------------------- #
def route_task(task: str, lane: str = "claude", trigger: str = "cron",
               headers: list[str] | None = None, ctx_lines: int = 10,
               n_threads: int = 0, images: int = 0,
               has_job_key: bool = False, job_run_index: float = 0.0,
               job_fric_eb: float | None = None,
               incumbent: str | None = None,
               alpha: float = rpolicy.DEFAULT_ALPHA,
               sheet: str = rpolicy.DEFAULT_SHEET,
               n_draws: int = _serve.DEFAULT_DRAWS) -> dict:
    """Score one typed task and return the selection distribution plus its reasoning."""
    ens = state()["ensemble"]
    if lane not in ("claude", "gpt"):
        raise ValueError("lane must be 'claude' or 'gpt'")
    req = _serve.synthesize_request(
        task, lane=lane, trigger=trigger, headers=tuple(headers or ()),
        ctx_lines=ctx_lines, n_threads=n_threads, images=images,
        lane_med=ens.lane_medians,
    )
    x = _serve.feature_row(req, ens, job_fric_eb=job_fric_eb,
                           job_run_index=job_run_index, has_job_key=has_job_key)
    dist = _serve.selection_distribution(ens, x, lane, incumbent=incumbent,
                                         alpha=alpha, n_draws=n_draws,
                                         img_pre=images, sheet=sheet)
    # The incumbent the console explains and draws the gate chain for. It must be
    # the one carrying the most logged traffic, not whichever arm sorts first:
    # the cheapest arm is alphabetically early AND the one gate 4 always refuses
    # (nothing is below it), so an alphabetical pick shows the least informative
    # chain in the lane every single time.
    lead = incumbent or max(dist["per_incumbent"],
                            key=lambda a: dist["per_incumbent"][a]["share_of_lane"])
    dist["lead_incumbent"] = lead
    dist["explain"] = _serve.explain(ens, x, lead)
    dist["features"] = {c: float(v) for c, v in zip(ens.cols, x)}

    # This server is the UI seam, and the UI speaks English. The router modules
    # keep their English strings (they are the repo's code language and are
    # printed by every acceptance check); the German wording is attached here so
    # neither side has to know about the other's audience.
    dist["ui"] = {
        "incumbent_mode": ("incumbent model fixed to " + incumbent if incumbent else
                           "averaged over the lane's logged model mix"),
        "basis": ("The spread comes from the 5 fold predictors and the conformal-tau "
                  "bootstrap—the router itself is deterministic. p_fric is PROCESS "
                  "friction from tool exit codes, never answer quality."),
        "explain_basis": ("Contribution = standardized feature value × fitted slope, "
                          "averaged over the 5 folds. This explains the SCORE that the "
                          "isotonic calibrator maps to p_fric. The mapping is monotonic, "
                          "so rank carries over but magnitude does not."),
    }

    dist["support"] = support_check(ens, x)
    return dist


#: A binary feature value is "rare" below this share of training rows. Not
#: out-of-range -- 13 of 1000 logged runs carry no `# === Thread info ===` block --
#: but worth naming, because a task built out of several rare choices at once is
#: a combination the predictor has little to say about.
RARE_SHARE = 0.05


def support_check(ens: _serve.Ensemble, x: np.ndarray) -> dict:
    """Is this live row inside the cloud the predictor was fit on?

    Binary and continuous columns need different questions, and asking one question
    for both is how a perfectly ordinary task gets flagged as an extrapolation.
    `hdr_thread_info` is 1 on 987 of 1000 rows, so its standard deviation is 0.11 and
    ANY row without that header scores |z| = 8.7 -- while 13 real logged rows sit at
    exactly that value. A z-score measures distance in a spread; on a near-constant
    indicator there is no spread to measure.

    So: continuous columns are judged by |z| against the training moments; binary
    columns are judged by whether the value occurs in training at all, and reported
    as RARE rather than out-of-range when it occurs but seldom.
    """
    X, cols, *_ = rmodel.load_design()
    continuous = set(_serve.rfeatures.CONTINUOUS_COLS)
    col_at = {c: i for i, c in enumerate(cols)}

    worst_z, worst_col = 0.0, None
    rare: list[dict] = []
    unseen: list[str] = []
    for c in ens.cols:
        j = col_at[c]
        v = float(x[j])
        column = X[:, j]
        if c in continuous:
            sd = float(column.std()) or 1.0
            z = abs(v - float(column.mean())) / sd
            if z > worst_z:
                worst_z, worst_col = z, c
            continue
        share = float(np.mean(np.isclose(column, v)))
        if share == 0.0:
            unseen.append(c)
        elif share < RARE_SHARE:
            rare.append({"col": c, "value": v, "share": share})

    in_range = worst_z < 4.0 and not unseen
    if unseen:
        names = ", ".join(unseen)
        note = (f"A category in this task NEVER occurs in the export ({names})—"
                "the distribution is pure extrapolation.")
    elif not in_range:
        note = (f"'{worst_col}' is {worst_z:.1f} standard deviations from the corpus—"
                "read this as extrapolation, not a decision.")
    elif rare:
        names = ", ".join(r["col"] for r in rare)
        note = (f"Inside the training cloud, but with {len(rare)} rare feature values "
                f"({names})—the folds spread more widely here.")
    else:
        note = "Typical task: all features lie in the densely populated part of the corpus."
    return {
        "in_range": bool(in_range),
        "max_abs_z": worst_z,
        "max_abs_z_col": worst_col,
        "rare": rare,
        "never_seen": unseen,
        "note": note,
        "basis": ("continuous columns use |z| against the training moments; binary "
                  "columns use the value's frequency in the export—a z-score on an "
                  "almost-constant indicator measures nothing."),
    }


def bootstrap_payload() -> dict:
    """The static half of the console: everything that does not depend on a knob."""
    ens = state()["ensemble"]
    labels = state()["labels"]
    return {
        "components": _outcome.component_base_rates(labels),
        "component_order": list(_outcome.COMPONENTS),
        "presets": {k: _outcome.resolve_weights(k) for k in _outcome.WEIGHT_PRESETS},
        "preset_notes": _outcome.PRESET_NOTES,
        "default_preset": _outcome.DEFAULT_PRESET,
        "sheets": list(ALL_SHEETS),
        "default_sheet": rpolicy.DEFAULT_SHEET,
        "default_alpha": rpolicy.DEFAULT_ALPHA,
        "lanes": {
            lane: {
                "incumbents": _serve.incumbent_prior(ens, lane),
                "medians": ens.lane_medians.get(lane, {}),
                "tools": sorted(_serve.GPT_TOOLS if lane == "gpt" else _serve.CLAUDE_TOOLS),
            }
            for lane in ("claude", "gpt")
        },
        "triggers": list(_serve.rfeatures.TRIGGER_CATEGORIES),
        "header_blocks": [n for n, _ in _serve.rfeatures.HEADER_BLOCKS],
        "prices": _serve.price_table(ens),
        "arm_counts": dict(sorted(ens.counts.items())),
        "tau_point": ens.tau(),
        "n_trajectories": int(ens.idx.size),
        "n_boot": {"interactive": INTERACTIVE_N_BOOT, "published": PUBLISHED_N_BOOT},
        "caveats": CAVEATS,
    }


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #
_STATIC = {
    "/": (CONSOLE_HTML, "text/html; charset=utf-8"),
    "/explorer": (SITE_DIR / "index.html", "text/html; charset=utf-8"),
    "/data.js": (SITE_DIR / "data.js", "application/javascript; charset=utf-8"),
}


class Handler(BaseHTTPRequestHandler):
    """Routes three GETs and two POSTs; everything else is a 404."""

    server_version = "ViktorRouterConsole/1.0"

    def log_message(self, fmt, *args):
        """One compact line per request; the default logs the whole request line twice."""
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    # -- helpers -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(assert_clean(payload), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    # -- verbs -------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/bootstrap":
            self._guarded(bootstrap_payload)
            return
        static = _STATIC.get(path)
        if static is None:
            self._json({"error": f"no route for {path}"}, 404)
            return
        file, ctype = static
        if not file.exists():
            self._json({"error": f"{file.name} is missing from site/"}, 404)
            return
        self._send(200, file.read_bytes(), ctype)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/frontier":
            body = self._read_json()
            self._guarded(lambda: frontier(
                preset=body.get("preset"), weights=body.get("weights"),
                sheet=body.get("sheet", rpolicy.DEFAULT_SHEET),
                alpha=float(body.get("alpha", rpolicy.DEFAULT_ALPHA)),
                n_boot=int(body.get("n_boot", INTERACTIVE_N_BOOT)),
            ))
        elif path == "/api/route":
            body = self._read_json()
            self._guarded(lambda: route_task(
                task=body.get("task", ""), lane=body.get("lane", "claude"),
                trigger=body.get("trigger", "cron"), headers=body.get("headers"),
                ctx_lines=int(body.get("ctx_lines", 10)),
                n_threads=int(body.get("n_threads", 0)),
                images=int(body.get("images", 0)),
                has_job_key=bool(body.get("has_job_key", False)),
                job_run_index=float(body.get("job_run_index", 0)),
                job_fric_eb=(None if body.get("job_fric_eb") is None
                             else float(body["job_fric_eb"])),
                incumbent=body.get("incumbent") or None,
                alpha=float(body.get("alpha", rpolicy.DEFAULT_ALPHA)),
                sheet=body.get("sheet", rpolicy.DEFAULT_SHEET),
                n_draws=int(body.get("n_draws", _serve.DEFAULT_DRAWS)),
            ))
        else:
            self._json({"error": f"no route for {path}"}, 404)

    def _guarded(self, fn) -> None:
        """Run a handler; turn a bad request into a 400 and a real fault into a 500.

        The message is returned verbatim because every raise in this package is a
        sentence written for a person -- an unknown preset, an arm with no logged
        runs, a weight vector of zeros. Swallowing them into "internal error" would
        make the console undebuggable at the one moment it is on a projector.
        """
        try:
            self._json(fn())
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 - a demo server must not die on one bad request
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


# --------------------------------------------------------------------------- #
# selftest + main
# --------------------------------------------------------------------------- #
def acceptance() -> list[tuple]:
    """Return [(pass, label, expected, actual), ...] over the live payloads."""
    rows: list[tuple] = []

    boot = bootstrap_payload()
    rows.append((boot["n_trajectories"] == 1000, "bootstrap: trajectories", 1000,
                 boot["n_trajectories"]))
    rows.append((abs(boot["tau_point"] - 0.170833) < 1e-6, "bootstrap: tau",
                 0.170833, round(boot["tau_point"], 6)))

    # the published frontier, reproduced through the console's own code path
    f = frontier(preset="repo_baseline", n_boot=PUBLISHED_N_BOOT)
    by = {p["label"]: p for p in f["points"]}
    logged = by["logged (as delivered)"]
    fable = by["all admissible → claude-fable-5"]
    gated = next(p for p in f["points"] if p["kind"] == "operating")
    rows.append((abs(logged["cost_usd"] - 412.9544292) < 1e-4,
                 "frontier: logged cost USD", 412.9544, round(logged["cost_usd"], 4)))
    rows.append((abs(fable["cost_usd"] - 78.22111108) < 1e-4,
                 "frontier: all->fable cost USD", 78.2211, round(fable["cost_usd"], 4)))
    rows.append((abs(fable["bound_pp"] - 6.650461790857802) < 1e-4,
                 "frontier: all->fable bound pp", 6.6505, round(fable["bound_pp"], 4)))
    rows.append((abs(gated["cost_usd"] - 375.45284212) < 1e-4,
                 "frontier: gated router cost USD", 375.4528, round(gated["cost_usd"], 4)))
    rows.append((abs(gated["bound_pp"] - 0.7596729064382319) < 1e-4,
                 "frontier: gated router bound pp", 0.7597, round(gated["bound_pp"], 4)))
    rows.append((gated["n_changed"] == 119, "frontier: rerouted trajectories", 119,
                 gated["n_changed"]))

    # the composite axis moves the ranking, which is the reason it is a slider
    fb = frontier(preset="balanced", n_boot=INTERACTIVE_N_BOOT)
    moved = abs(fb["logged_badness"] - f["logged_badness"]) > 0.1
    rows.append((moved, "composite preset changes the logged level", True, moved))

    # the live demo
    a = route_task("Post the daily standup reminder.", ctx_lines=2, n_draws=200)
    b = route_task("Investigate the failed nightly ETL, read the logs, patch the "
                   "config and report back." * 8, trigger="slack_mention",
                   headers=["hdr_thread_info", "hdr_channel_thread"],
                   ctx_lines=45, n_threads=9, n_draws=200)
    arms = set(a["distribution"]) | set(b["distribution"])
    gap = max(abs(a["distribution"].get(k, 0.0) - b["distribution"].get(k, 0.0))
              for k in arms)
    rows.append((gap > 0.05, "two tasks give two distributions", "> 0.05", round(gap, 4)))
    rows.append((a["support"]["in_range"], "live task inside the training range",
                 True, a["support"]["in_range"]))

    gpt = route_task("Refactor the payment module.", lane="gpt", n_draws=200)
    lane_ok = all(k.startswith("gpt") for k in gpt["distribution"])
    rows.append((lane_ok, "gpt lane stays in the gpt lane", True, lane_ok))

    # the data-safety gate is live on every body
    for name, payload in (("bootstrap", boot), ("frontier", f), ("route", a)):
        try:
            assert_clean(payload)
            ok = True
        except ValueError:
            ok = False
        rows.append((ok, f"{name} payload passes assert_clean", True, ok))

    # ... and it actually fires when there is something to catch
    try:
        assert_clean({"leak": "<PERSON_ROBERT> was here"})
        fired = False
    except ValueError:
        fired = True
    rows.append((fired, "assert_clean rejects a redaction placeholder", True, fired))
    return rows


def main(argv: list[str] | None = None) -> int:
    """Serve the console, or run the acceptance checks with --selftest."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--selftest", action="store_true",
                    help="run every acceptance check and exit, without opening a port")
    a = ap.parse_args(argv)

    missing = [p.name for p in (RESULTS_DIR / "routes.jsonl",
                                RESULTS_DIR / "features.npz",
                                RESULTS_DIR / "labels.jsonl") if not p.exists()]
    if missing:
        print(f"missing artifacts: {', '.join(missing)} — run `make all` first",
              file=sys.stderr)
        return 2

    print("warming the fold ensemble (a real refit — about 9 s) ...", file=sys.stderr)
    state()

    if a.selftest:
        rows = acceptance()
        failed = 0
        for ok, label, expected, actual in rows:
            failed += 0 if ok else 1
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {label:<46} expected={expected!r:<14} actual={actual!r}")
        print()
        print("  Token counts are ESTIMATES; dollars are input-side only under an ASSUMED sheet.")
        return 1 if failed else 0

    if not CONSOLE_HTML.exists():
        print(f"missing {CONSOLE_HTML}", file=sys.stderr)
        return 2

    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"router console on http://{a.host}:{a.port}/  (Ctrl-C to stop)", file=sys.stderr)
    print(f"  raw-trajectory explorer on http://{a.host}:{a.port}/explorer", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
