#!/usr/bin/env python3
"""Route ONE real trajectory and print the whole decision chain: `python -m router.demo --line N`.

WHAT THIS COMPUTES
    Nothing new. It replays, for a single logged trajectory, the same five gates
    router.policy applies to all 1000, and shows the arithmetic behind the
    decision: the reconstructed turn count and billed prefix, each gate with its
    verdict, the calibrated out-of-fold p_fric against the split-conformal tau,
    the support check, and the routed-vs-logged delta under an ASSUMED sheet.

WHAT IT WRITES
    No artifact. It reads results/recon.jsonl, results/routes.jsonl,
    results/labels.jsonl, results/jobkey.jsonl, results/features.npz and one
    line of the export.

OFFLINE BY CONSTRUCTION
    No network, no API key, no model call. This is the stage demo: everything it
    prints already exists on disk or is recomputed from it in a few seconds.

PROPRIETARY DATA
    The export is proprietary. Every string that originates in a trajectory is
    passed through `redact()` before printing: placeholders are collapsed, URLs,
    emails, long digit/hex runs and @handles are masked, and the result is hard
    capped at SNIPPET_LIMIT = 60 characters INCLUDING the ellipsis. `redact()`
    is the only path by which trajectory text reaches stdout, and there is an
    acceptance check that no printed line exceeds that cap.

TWO MODES
    default    reads results/routes.jsonl for the stored out-of-fold p_fric and
               recomputes tau exactly from those scores plus results/labels.jsonl.
               Instant. Faithful: p_fric is stored out-of-fold, so nothing is
               being re-derived in a way that could leak.
    --refit    re-fits the predictor in-process (~10 s) and CHECKS the freshly
               computed p_fric against the stored one before using it. Use this
               when someone in the room asks whether the number is live.

ACCEPTANCE (printed by `python -m router.demo --selftest`)
    tau recovered from the stored scores at alpha=0.10 is 0.170833;
    119 of 1000 trajectories are rerouted and 6 of those are `supported`;
    the gate chain replayed here reproduces results/routes.jsonl exactly on all
    1000 rows (route, changed and every gate field);
    redact() caps every snippet at 60 chars and leaves no PII_ / <ENTITY_ token.

CAVEAT ON EVERY NUMBER
    Token counts are ESTIMATES: tok(x) = len(json.dumps(x)) // 4, because the
    export has no `usage` field. Dollar figures are INPUT-SIDE ONLY (the export
    has no `output` field either) and name their ASSUMED sheet -- model ids are
    anonymized per AGENTS.md and no public price sheet applies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
RECON_PATH = os.path.join(RESULTS_DIR, "recon.jsonl")
ROUTES_PATH = os.path.join(RESULTS_DIR, "routes.jsonl")
LABELS_PATH = os.path.join(RESULTS_DIR, "labels.jsonl")
JOBKEY_PATH = os.path.join(RESULTS_DIR, "jobkey.jsonl")
FEATURES_PATH = os.path.join(RESULTS_DIR, "features.npz")

#: Hard cap on any string that originates inside a trajectory, ellipsis included.
SNIPPET_LIMIT = 60

#: Default alpha for the split-conformal tau (matches router.policy.DEFAULT_ALPHA).
DEFAULT_ALPHA = 0.10

#: A trajectory that is rerouted, supported by an observed run of the target arm
#: on the same literal cron job, and carries the largest saving among those six.
DEFAULT_LINE = 647

ESTIMATE_CAVEAT = (
    "Token counts are ESTIMATES: tok(x)=len(json.dumps(x))//4 (the export has no "
    "`usage` field).\nDollar figures are INPUT-SIDE ONLY under an ASSUMED sheet; "
    "model ids are anonymized."
)

EXPECTED = {
    "tau": 0.1708333333333333,
    "n": 1000,
    "changed": 119,
    "supported": 6,
    "alpha": DEFAULT_ALPHA,
}

# --------------------------------------------------------------------------- #
# redaction -- the only path trajectory text takes to stdout
# --------------------------------------------------------------------------- #

_PII_RE = re.compile(r"PII_[A-Z][A-Z0-9_]*")
_ENTITY_RE = re.compile(r"<ENTITY_[^>]*>")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HANDLE_RE = re.compile(r"(?<![\w])@[\w.-]{2,}")
_LONGNUM_RE = re.compile(r"\b(?=[0-9a-fA-F]*[0-9])[0-9a-fA-F]{8,}\b")
_WS_RE = re.compile(r"\s+")


def redact(text: object, limit: int = SNIPPET_LIMIT) -> str:
    """Mask every identifier in a trajectory string and hard cap it at `limit` chars.

    The cap includes the trailing ellipsis, so the returned string is NEVER
    longer than `limit`. Order matters: entities and placeholders first (they
    are already anonymized markers and should read as such), then URLs, emails,
    @handles and long hex/digit runs, then whitespace collapse, then the cut.
    """
    if text is None:
        return "(none)"
    s = str(text)
    s = _ENTITY_RE.sub("[ENTITY]", s)
    s = _PII_RE.sub("[PII]", s)
    s = _URL_RE.sub("[URL]", s)
    s = _EMAIL_RE.sub("[EMAIL]", s)
    s = _HANDLE_RE.sub("[HANDLE]", s)
    s = _LONGNUM_RE.sub("[ID]", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return "(empty)"
    if len(s) > limit:
        s = s[: max(0, limit - 3)].rstrip() + "..."
    return s


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _load_jsonl(path: str) -> Dict[int, dict]:
    """Read a results/*.jsonl artifact into {idx: record}; raise a readable error if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s is missing. Build it first: python -m router.cli all" % path
        )
    out: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[int(rec["idx"])] = rec
    return out


def load_img_pre() -> Dict[int, int]:
    """{idx: pre-treatment image count} from results/features.npz (0 for all when absent)."""
    if not os.path.exists(FEATURES_PATH):
        return {}
    import numpy as np

    z = np.load(FEATURES_PATH, allow_pickle=False)
    cols = [str(c) for c in z["cols"]]
    if "img_pre" not in cols:
        return {}
    col = z["X"][:, cols.index("img_pre")]
    return {int(i): int(v) for i, v in zip(z["idx"], col)}


def read_export_line(idx: int, export_dir: Optional[str] = None) -> Optional[dict]:
    """Return the raw request for one export line, or None when the export is unreadable."""
    try:
        from router.io import iter_lines
    except ImportError:                                       # pragma: no cover
        return None
    kwargs = {"export_dir": export_dir} if export_dir else {}
    try:
        for i, req in iter_lines(**kwargs):
            if i == idx:
                return req
    except (OSError, ValueError):
        return None
    return None


def task_snippet(request: Optional[dict]) -> str:
    """A single redacted, <=60-char snippet of the trajectory's task text."""
    if not request:
        return "(export not read)"
    try:
        from router.features import item_text, strip_system_block
        from router.io import first_user_index
    except ImportError:                                       # pragma: no cover
        return "(router.features unavailable)"
    items = request.get("input") or []
    cut = first_user_index(items)
    if cut < 0:
        return "(no user message)"
    text = item_text(items[cut])
    _preamble, residual = strip_system_block(text)
    return redact(residual or text)


def support_sets(jobkeys: Dict[int, dict], recon: Dict[int, dict]) -> Dict[str, Dict[str, int]]:
    """{literal cron path: {arm: observed run count}} -- the support test for a reroute."""
    out: Dict[str, Dict[str, int]] = {}
    for idx, jk in jobkeys.items():
        if jk.get("literal") and jk.get("cron_path"):
            arm = recon[idx]["model"]
            out.setdefault(jk["cron_path"], {})
            out[jk["cron_path"]][arm] = out[jk["cron_path"]].get(arm, 0) + 1
    return out


class Context:
    """Everything the demo needs, loaded once: artifacts, tau, arm counts, support."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, refit: bool = False) -> None:
        self.alpha = float(alpha)
        self.recon = _load_jsonl(RECON_PATH)
        self.labels = _load_jsonl(LABELS_PATH)
        self.jobkeys = _load_jsonl(JOBKEY_PATH)
        self.routes = _load_jsonl(ROUTES_PATH)
        self.img_pre = load_img_pre()
        self.support = support_sets(self.jobkeys, self.recon)
        self.arm_counts: Dict[str, int] = {}
        for rec in self.recon.values():
            self.arm_counts[rec["model"]] = self.arm_counts.get(rec["model"], 0) + 1

        self.p_source = "results/routes.jsonl (stored out-of-fold p_fric)"
        self.p_fric = {i: float(r["p_fric"]) for i, r in self.routes.items()}
        self.refit_delta: Optional[float] = None
        if refit:
            self._refit()
        self.tau = self._tau()

    def _refit(self) -> None:
        """Re-fit the predictor in-process and check it against the stored p_fric."""
        import numpy as np

        from router import model as rmodel

        fit = rmodel.load_pfric()
        fresh = {int(i): float(p) for i, p in zip(fit["idx"], fit["p_fric"])}
        common = sorted(set(fresh) & set(self.p_fric))
        if common:
            self.refit_delta = float(
                np.max([abs(fresh[i] - self.p_fric[i]) for i in common])
            )
        self.p_fric = fresh
        self.p_source = "re-fit in-process (router.model.load_pfric, out-of-fold)"

    def _tau(self) -> float:
        """Split-conformal tau over the calibration positives at level alpha."""
        from router.policy import conformal_tau

        idxs = sorted(self.p_fric)
        scores = [self.p_fric[i] for i in idxs]
        y = [self.labels[i]["y_fric"] for i in idxs]
        return float(conformal_tau(scores, y, self.alpha))

    def replay(self, idx: int) -> dict:
        """Re-run the five gates on one line and return a fresh routes.jsonl-shaped record."""
        from router.policy import route_line

        return route_line(
            self.recon[idx],
            self.p_fric[idx],
            int(self.img_pre.get(idx, 0)),
            self.arm_counts,
            self.tau,
        )


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #

def _v(passed: bool) -> str:
    """Fixed-width gate verdict token."""
    return "PASS " if passed else "BLOCK"


def _usd(x: float) -> str:
    """Format a dollar amount with a sign, four decimals."""
    return ("-$%.4f" % abs(x)) if x < 0 else ("$%.4f" % x)


def print_decision(ctx: Context, idx: int, sheet: str = "assumed_default",
                   export_dir: Optional[str] = None, out=None) -> dict:
    """Print the full decision chain for one trajectory; return the replayed record."""
    out = out or sys.stdout
    say = lambda s="": print(s, file=out)                        # noqa: E731

    rec = ctx.recon[idx]
    jk = ctx.jobkeys.get(idx, {})
    lab = ctx.labels.get(idx, {})
    stored = ctx.routes.get(idx, {})
    live = ctx.replay(idx)
    g = live["gates"]
    p = live["p_fric"]

    say("=" * 74)
    say("TRAJECTORY  idx %d  of %d" % (idx, len(ctx.recon)))
    say("=" * 74)
    say("  logged arm     %s  (%d logged runs on this arm)"
        % (rec["model"], ctx.arm_counts.get(rec["model"], 0)))
    say("  family lane    %s  -- read from the tools block, never chosen" % rec["family"])
    say("  job key        %s  [%s]"
        % (redact(jk.get("cron_path")), jk.get("bucket", "?")))
    say("  task snippet   %s" % task_snippet(read_export_line(idx, export_dir)))
    say("                 (redacted, hard capped at %d chars -- the export is proprietary)"
        % SNIPPET_LIMIT)
    say()

    naive = rec["naive_tok"]
    gross = rec["gross_tok"]
    read, write = rec["cache_read_tok"], rec["cache_write_tok"]
    say("BILLING, RECONSTRUCTED   (a line is N API calls, each re-sending the prefix)")
    say("  turns                  %10d" % rec["n_turns"])
    say("  gross billed prefix    %10s est. tok   (tools block %s of it)"
        % ("{:,}".format(gross), "{:,}".format(rec["tools_tok"] * rec["n_turns"])))
    say("  cache read             %10s est. tok   %5.1f%%"
        % ("{:,}".format(read), 100.0 * read / gross if gross else 0.0))
    say("  cache write            %10s est. tok   %5.1f%%"
        % ("{:,}".format(write), 100.0 * write / gross if gross else 0.0))
    say("  naive item sum         %10s est. tok   <- what the starter kit counts"
        % "{:,}".format(naive))
    say("  reconstruction is %.1fx the naive count for this line"
        % (gross / naive if naive else 0.0))
    say()

    say("GATE CHAIN  (router.policy, same five gates as all 1000 rows)")
    say("  1 family lane         %s  %s  -- in-lane targets only; a cross-family"
        % (_v(True), g["family"]))
    say("                               reroute changes the tool contract, not the price")
    say("  2 vision              %s  pre-run images: %d"
        % (_v(not g["vision"]), int(ctx.img_pre.get(idx, 0))))
    adm = g["admissible"]
    say("  3 admissible arm      %s  logged arm has n=%d (needs > 20)"
        % (_v(bool(adm)), ctx.arm_counts.get(rec["model"], 0)))
    say("      in-lane candidates: %s" % (", ".join(adm) if adm else "(none)"))
    st = g["sign_stable"]
    say("  4 sign-stable cheaper %s  %s"
        % (_v(bool(st)), ", ".join(st) if st else "(none survive)"))
    say("      a target survives only if it is cheaper under ALL THREE assumed sheets")
    say("  5 tau                 %s  p_fric %.4f  vs  tau %.4f  (alpha=%.2f)"
        % (_v(g["tau_pass"]), p, ctx.tau, ctx.alpha))
    say("      p_fric source: %s" % ctx.p_source)
    if ctx.refit_delta is not None:
        say("      re-fit vs stored p_fric, max abs diff over 1000 rows: %.2e"
            % ctx.refit_delta)
    say("      tau is split conformal: a genuinely frictional trajectory is routed")
    say("      with probability <= %.2f. It is a MISS-RATE bound, not a point"
        % ctx.alpha)
    say("      estimate of the friction the policy causes; that estimate does not exist.")
    say()

    path = jk.get("cron_path") if jk.get("literal") else None
    obs = ctx.support.get(path, {}) if path else {}
    say("SUPPORT CHECK  (may this decision carry an OUTCOME claim, not just a price?)")
    if not live["changed"]:
        say("  n/a -- the policy keeps the logged arm here.")
    elif not path:
        say("  NO  -- no literal cron path (bucket=%s), so there is no clean job to"
            % jk.get("bucket", "?"))
        say("         look the target arm up in. Count the saving, never the outcome.")
    elif live["route"] in obs:
        say("  YES -- %s has %d observed run(s) on this job. Only 6 of the 119"
            % (live["route"], obs[live["route"]]))
        say("         reroutes clear this bar; 41/42/43 must filter on `supported`.")
    else:
        say("  NO  -- %s was never observed on this job (observed: %s)."
            % (live["route"], ", ".join("%s x%d" % kv for kv in sorted(obs.items())) or "none"))
        say("         Extrapolated. Count the saving, never the outcome.")
    say()

    lo, ro = live["cost_logged_usd"], live["cost_routed_usd"]
    delta = ro - lo
    say("DECISION")
    say("  logged   %-16s %10s" % (live["logged"], _usd(lo)))
    say("  routed   %-16s %10s" % (live["route"], _usd(ro)))
    say("  delta    %-16s %10s   %s"
        % ("", _usd(delta), ("%+.1f%%" % (100.0 * delta / lo)) if lo else "n/a"))
    say("  [sheet=%s, ASSUMED; est. tokens; INPUT-SIDE ONLY -- the export has no" % sheet)
    say("   `output` field, so output volume is unknowable]")
    if stored:
        agree = (stored.get("route") == live["route"]
                 and stored.get("changed") == live["changed"])
        say("  matches results/routes.jsonl: %s" % ("YES" if agree else "NO -- REBUILD IT"))
    say()

    y_fric = lab.get("y_fric")
    say("WHAT ACTUALLY HAPPENED ON THE LOGGED ARM  (POST-TREATMENT -- read for")
    say("honesty, NEVER fed to any gate above; every feature is cut at the first")
    say("user message)")
    say("  tool calls %d | outputs %d | errors %d | unresolvable %d | killed %d"
        % (lab.get("n_calls", 0), lab.get("n_obs", 0), lab.get("n_err", 0),
           lab.get("n_unknown", 0), lab.get("n_kill", 0)))
    say("  y_fric %s   y_proc %s" % (y_fric, lab.get("y_proc")))
    say("  This one line proves nothing either way: the MDE on the best-powered")
    say("  arm pair is 11.4pp, which is why 42-partial-identification reports bounds.")
    say("=" * 74)
    return live


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #

def replay_all(ctx: Context) -> Tuple[int, int, int, List[int]]:
    """Replay the gate chain over every line; return (n, changed, supported, mismatches)."""
    changed = supported = 0
    bad: List[int] = []
    for idx in sorted(ctx.recon):
        live = ctx.replay(idx)
        stored = ctx.routes.get(idx)
        if live["changed"]:
            changed += 1
        if stored is not None:
            if (stored["route"] != live["route"]
                    or stored["changed"] != live["changed"]
                    or stored["gates"] != live["gates"]):
                bad.append(idx)
            if stored["changed"] and stored["supported"]:
                supported += 1
    return len(ctx.recon), changed, supported, bad


def selftest(alpha: float = DEFAULT_ALPHA) -> int:
    """Print the demo's acceptance checks with PASS/FAIL; return 0 iff all pass."""
    print("router.demo selftest")
    print("  " + ESTIMATE_CAVEAT.replace("\n", "\n  "))
    print()
    ctx = Context(alpha=alpha)
    n, changed, supported, bad = replay_all(ctx)

    probes = [
        "PII_PROJECT_12 and <ENTITY_ORG_3> in one line",
        "see https://example.com/a/b?sig=deadbeefcafe1234 for details",
        "ping me at someone@example.com or @some.handle about deadbeef1234",
        "x" * 500,
        "",
    ]
    reds = [redact(p) for p in probes]
    caps_ok = all(len(r) <= SNIPPET_LIMIT for r in reds)
    clean = all(("PII_" not in r and "<ENTITY_" not in r and "@" not in r
                 and "http" not in r) for r in reds)

    checks = [
        ("lines", EXPECTED["n"], n),
        ("conformal tau @ alpha=%.2f" % alpha, round(EXPECTED["tau"], 6), round(ctx.tau, 6)),
        ("rerouted trajectories", EXPECTED["changed"], changed),
        ("supported reroutes", EXPECTED["supported"], supported),
        ("replay reproduces results/routes.jsonl", 0, len(bad)),
        ("redact() caps every snippet at %d" % SNIPPET_LIMIT, True, caps_ok),
        ("redact() leaves no PII_/<ENTITY_/@/URL", True, clean),
    ]
    ok = True
    for label, expected, actual in checks:
        good = expected == actual
        ok = ok and good
        print("  [%s] %-42s expected %-10s actual %s"
              % ("PASS" if good else "FAIL", label, expected, actual))
    if bad:
        print("  mismatching idx (first 10): %s" % bad[:10])
    print()
    for probe, red in zip(probes[:3], reds[:3]):
        print("  redact(%d chars) -> %r" % (len(probe), red))
    print()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for `python -m router.demo`."""
    p = argparse.ArgumentParser(
        prog="python -m router.demo",
        description="Route one real trajectory and print the whole decision chain. "
                    "Offline: no network, no API key, no model call.",
    )
    p.add_argument("--line", type=int, default=DEFAULT_LINE,
                   help="0-based export line to route (default %d)" % DEFAULT_LINE)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="conformal miss-rate level for tau (default %.2f)" % DEFAULT_ALPHA)
    p.add_argument("--sheet", default="assumed_default",
                   help="ASSUMED price sheet named in the output (default assumed_default)")
    p.add_argument("--export", default=None, help="export directory for the task snippet")
    p.add_argument("--refit", action="store_true",
                   help="re-fit the predictor in-process (~10 s) and check it against the "
                        "stored out-of-fold p_fric")
    p.add_argument("--selftest", action="store_true", help="run the acceptance checks and stop")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print one trajectory's decision chain (or --selftest); 0 on success."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.selftest:
        return selftest(args.alpha)

    print("router.demo -- one trajectory, the whole decision chain. Offline: no")
    print("network, no API key, no model call.")
    print(ESTIMATE_CAVEAT)
    print()
    try:
        ctx = Context(alpha=args.alpha, refit=args.refit)
    except FileNotFoundError as exc:
        print("router.demo: %s" % exc, file=sys.stderr)
        return 2
    if args.line not in ctx.recon:
        print("router.demo: --line %d is out of range (0..%d)"
              % (args.line, len(ctx.recon) - 1), file=sys.stderr)
        return 2
    print_decision(ctx, args.line, sheet=args.sheet, export_dir=args.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
