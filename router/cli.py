#!/usr/bin/env python3
"""One entry point for the whole router pipeline: `python -m router.cli <stage>`.

WHAT THIS COMPUTES
    Nothing. It is a runner. Every number in this project is produced by a stage
    module; this file only decides which stage runs, in what order, with which
    export directory and which ASSUMED price sheet, and prints one summary line
    per stage. It writes no artifact of its own.

WHAT IT WRITES
    No artifact. Each stage writes its own (results/recon.jsonl, results/labels.jsonl,
    results/jobkey.jsonl, results/features.npz + results/feature_manifest.json,
    results/routes.jsonl, results/estimates.json, ...). See docs/CONTRACTS.md.

STAGES, IN DEPENDENCY ORDER
    recon labels jobkey features costs model policy gates strata ope family
    figs report

    This is the true dependency order and the Makefile's `all` now matches it.
    `gates` (router.gates -> router.controls) reads results/features.npz,
    results/labels.jsonl, results/recon.jsonl AND results/routes.jsonl, so it can
    only run once `policy` has written routes. `all` therefore runs
    model -> policy -> gates, and stops at the first FAILING GATE so that no
    downstream estimate is produced behind a red control.

    `strata` and `family` merge their own sections into results/estimates.json,
    and `report` refuses to run without the "family_contrast" section — so they
    are stages of `all`, not optional extras.

STAGE -> MODULE RESOLUTION
    Several stages have more than one candidate module because the repo grew
    under several owners with divergent names (skill 30 says `router.predictor`,
    the Makefile says `router.model`; skill 90 says `router.leakaudit`, the gate
    that exists is `router.controls`). Each stage carries an ordered candidate
    list and the FIRST IMPORTABLE one wins; the chosen module is printed, so a
    rename can never silently change which code ran. A stage with no importable
    candidate is reported MISSING, not skipped silently, and makes the run exit 2.

ACCEPTANCE (printed by `python -m router.cli --selftest`)
    13 stages declared, in dependency order, with no duplicate stage names;
    every stage resolves to an importable module or is reported MISSING;
    `--sheet` is rejected unless it is a registered sheet in router.pricing;
    `--export` is rejected unless the directory holds trajectories_*.jsonl;
    the stage table's declared export mechanism is verified per stage after
    import, and any stage that cannot honour a non-default --export is reported
    NOT HONOURED rather than run against the wrong data.

CAVEAT CARRIED EVERYWHERE
    All token counts downstream are ESTIMATES (tok(x) = len(json.dumps(x)) // 4);
    the export has no `usage` field. Every dollar figure is input-side only and
    names its ASSUMED sheet.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DEFAULT_EXPORT = os.path.join(REPO_ROOT, "export")

#: The caveat every user-facing surface in this project must carry.
ESTIMATE_CAVEAT = (
    "token counts are ESTIMATES (tok(x)=len(json.dumps(x))//4); the export has no "
    "`usage` field. Dollar figures are input-side only under an ASSUMED sheet."
)

# --------------------------------------------------------------------------- #
# how --export reaches a stage
# --------------------------------------------------------------------------- #
# "io"      the stage's export path is a def-time default taken from
#           router.io.DEFAULT_EXPORT_DIR. Patching router.io BEFORE the stage
#           module is first imported therefore reaches it. Verified by
#           inspecting the entry function's default after import.
# "attr"    the stage's main() reads a module-level constant at CALL time.
#           Patching the module attribute after import reaches it. Verified by
#           reading the attribute back.
# "argv"    the stage's main(argv) takes the export directory positionally.
# "none"    the stage never opens the export (it reads results/ only).
# "no"      the stage binds its export path at import time in a way this runner
#           cannot reach. A non-default --export is refused for that stage.
EXPORT_MODES = ("io", "attr", "argv", "none", "no")


class Stage:
    """One pipeline stage: its candidate modules, how flags reach it, and its cost."""

    def __init__(
        self,
        name: str,
        candidates: Sequence[str],
        summary: str,
        export_mode: str,
        export_probe: Optional[Tuple[str, str]] = None,
        export_attrs: Sequence[str] = (),
        sheet_attr: Optional[str] = None,
        is_gate: bool = False,
        slow_note: str = "",
    ) -> None:
        if export_mode not in EXPORT_MODES:
            raise ValueError("unknown export mode %r" % (export_mode,))
        self.name = name
        self.candidates = tuple(candidates)
        self.summary = summary
        self.export_mode = export_mode
        self.export_probe = export_probe
        self.export_attrs = tuple(export_attrs)
        self.sheet_attr = sheet_attr
        self.is_gate = is_gate
        self.slow_note = slow_note


#: The pipeline, in DEPENDENCY order. Order is load-bearing: `all` walks this list.
STAGES: Tuple[Stage, ...] = (
    Stage("recon", ("router.recon",),
          "reconstruct turns and the cache split -> results/recon.jsonl",
          export_mode="io", export_probe=("build", "export_dir")),
    Stage("labels", ("router.labels",),
          "tool-output outcomes and friction labels -> results/labels.jsonl",
          export_mode="argv", export_attrs=("EXPORT_DIR",)),
    Stage("jobkey", ("router.jobkey",),
          "literal cron-path job key -> results/jobkey.jsonl",
          export_mode="attr", export_attrs=("EXPORT_DIR",)),
    Stage("features", ("router.features",),
          "pre-treatment design matrix -> results/features.npz + feature_manifest.json",
          export_mode="io", export_probe=("build", "export_dir")),
    Stage("costs", ("router.costs",),
          "cache-aware price model and the effective multiplier sweep",
          export_mode="none", sheet_attr=None),
    Stage("model", ("router.model", "router.predictor"),
          "out-of-fold friction predictor on pre-treatment features only",
          export_mode="io", export_probe=("fallback_design", "export_dir"),
          slow_note="~13 s"),
    Stage("policy", ("router.policy",),
          "five-gate chain + conformal tau -> results/routes.jsonl",
          export_mode="io", sheet_attr="DEFAULT_SHEET", slow_note="~5 s"),
    Stage("gates", ("router.gates", "router.leakaudit", "router.controls"),
          "publication gate: leak, post-treatment, balance and weight controls",
          export_mode="none", is_gate=True, slow_note="~3 s"),
    Stage("strata", ("router.strata",),
          "stratum table + per-arm support deficit -> estimates.json['strata']",
          export_mode="attr", export_attrs=("EXPORT_DIR",), slow_note="~3 s"),
    Stage("ope", ("router.ope",),
          "off-policy evaluation: ESS, Hajek, bounds -> results/estimates.json",
          export_mode="no", slow_note="~17 s"),
    Stage("family", ("router.family_contrast",),
          "cross-family BUNDLE contrast -> estimates.json['family_contrast']",
          export_mode="no", slow_note="~4 s"),
    Stage("figs", ("router.figs", "router.frontier"),
          "figures, including the cost-quality frontier",
          export_mode="none", slow_note="~41 s"),
    Stage("report", ("router.report",),
          "results/metrics.json + results/claims.json + the written report",
          export_mode="none"),
)

STAGE_NAMES: Tuple[str, ...] = tuple(s.name for s in STAGES)
STAGE_BY_NAME: Dict[str, Stage] = {s.name: s for s in STAGES}


# --------------------------------------------------------------------------- #
# flag validation
# --------------------------------------------------------------------------- #

def resolve_export(path: Optional[str]) -> str:
    """Return an absolute, validated export directory (defaults to <repo>/export)."""
    resolved = os.path.abspath(path or DEFAULT_EXPORT)
    if not os.path.isdir(resolved):
        raise ValueError("--export %s is not a directory" % resolved)
    names = [n for n in os.listdir(resolved)
             if n.startswith("trajectories_") and n.endswith(".jsonl")]
    if not names:
        raise ValueError("--export %s holds no trajectories_*.jsonl file" % resolved)
    return resolved


def resolve_sheet(name: Optional[str]) -> str:
    """Return a validated ASSUMED price-sheet name registered in router.pricing."""
    pricing = importlib.import_module("router.pricing")
    known = list(pricing.sheet_names())
    chosen = "assumed_default" if name is None else name
    if chosen not in known:
        raise ValueError("--sheet %r is not registered; known sheets: %s"
                         % (chosen, ", ".join(known)))
    return chosen


def patch_io_export(export_dir: str) -> bool:
    """Patch router.io.DEFAULT_EXPORT_DIR before any stage module imports it."""
    io_mod = importlib.import_module("router.io")
    io_mod.DEFAULT_EXPORT_DIR = export_dir
    return True


# --------------------------------------------------------------------------- #
# stage resolution and execution
# --------------------------------------------------------------------------- #

def resolve_module(stage: Stage) -> Tuple[Optional[object], Optional[str], List[str]]:
    """Import the first available candidate module; return (module, name, tried).

    An ImportError means "this candidate does not exist here" and moves on to the
    next name. Any OTHER exception means the module exists but is broken at import
    time; that is a real defect and is re-raised with the module named, rather
    than silently falling through to a different implementation.
    """
    tried: List[str] = []
    for cand in stage.candidates:
        tried.append(cand)
        try:
            return importlib.import_module(cand), cand, tried
        except ImportError:
            continue
        except Exception as exc:                     # noqa: BLE001 - name the culprit
            raise RuntimeError("%s failed at import time: %s: %s"
                               % (cand, type(exc).__name__, exc)) from exc
    return None, None, tried


def apply_export(stage: Stage, mod: object, export_dir: str,
                 is_default: bool) -> Tuple[str, List[str]]:
    """Point one imported stage module at `export_dir`; return (verdict, argv extras).

    Verdict is one of "default", "honoured", "argv", "n/a", "NOT HONOURED".
    Never guesses: a stage whose export path this runner cannot reach is
    reported NOT HONOURED so the caller can refuse to run it against the
    wrong data rather than producing numbers from the default export.
    """
    if is_default:
        return "default", []
    if stage.export_mode == "none":
        return "n/a", []
    for attr in stage.export_attrs:
        if hasattr(mod, attr):
            setattr(mod, attr, type(getattr(mod, attr))(export_dir))
    if stage.export_mode == "argv":
        return "argv", [export_dir]
    if stage.export_mode == "no":
        return "NOT HONOURED", []
    if stage.export_mode == "attr":
        ok = all(str(getattr(mod, a, "")) == export_dir for a in stage.export_attrs)
        return ("honoured" if ok else "NOT HONOURED"), []
    # export_mode == "io": verify the def-time default actually moved.
    if stage.export_probe is None:
        return "honoured", []
    import inspect

    fn_name, param = stage.export_probe
    fn = getattr(mod, fn_name, None)
    if fn is None:
        return "NOT HONOURED", []
    default = inspect.signature(fn).parameters[param].default
    return ("honoured" if os.path.abspath(str(default)) == export_dir
            else "NOT HONOURED"), []


def apply_sheet(stage: Stage, mod: object, sheet: str) -> str:
    """Point one stage module at an ASSUMED price sheet; return the verdict string."""
    if stage.sheet_attr is None:
        return "n/a"
    if not hasattr(mod, stage.sheet_attr):
        return "NOT HONOURED"
    setattr(mod, stage.sheet_attr, sheet)
    return "honoured"


def call_main(main: Callable, extra_argv: Sequence[str], stage_name: str):
    """Call a stage's main() with an EXPLICIT argv when it takes one.

    Never let a stage fall through to ``sys.argv``: this runner's own flags
    (`router.cli costs --sheet ...`) would otherwise be re-parsed by the stage
    and read as, say, a recon path. Stages whose main() takes no argument are
    called with none, and are refused an argv rather than being handed one
    silently.
    """
    import inspect

    params = [
        pr for pr in inspect.signature(main).parameters.values()
        if pr.kind in (pr.POSITIONAL_ONLY, pr.POSITIONAL_OR_KEYWORD, pr.VAR_POSITIONAL)
    ]
    if params:
        return main(list(extra_argv))
    if extra_argv:
        raise TypeError(
            "stage %s: %s.main() takes no argv, so %r cannot be passed to it"
            % (stage_name, getattr(main, "__module__", "?"), list(extra_argv))
        )
    return main()


def call_gate_fast(mod: object) -> int:
    """Run the publication gate at reduced permutations; INDICATIVE, not licensing."""
    print("  --fast: 5 permutations instead of 50. INDICATIVE ONLY -- this pass does")
    print("  NOT license a claim. Re-run without --fast before quoting any number.")
    print("  (measured 2026-08-22: the full gate runs in ~3 s, so --fast buys almost")
    print("   nothing here. It exists for the case where the control set grows.)")
    report = mod.run_all(n_permutations=5, verbose=True)
    verdict = report.get("verdict")
    return 0 if verdict not in ("FAIL", "BLOCKED") else (1 if verdict == "FAIL" else 2)


def run_stage(stage: Stage, export_dir: str, sheet: str, is_default_export: bool,
              fast: bool = False, dry_run: bool = False) -> Dict[str, object]:
    """Run one stage and return its result record (never raises on stage failure)."""
    started = time.time()
    rec: Dict[str, object] = {
        "stage": stage.name, "module": None, "status": "MISSING", "code": None,
        "seconds": 0.0, "export": "default", "sheet": "n/a", "note": "",
        "is_gate": stage.is_gate,
    }
    try:
        mod, name, tried = resolve_module(stage)
    except RuntimeError as exc:
        rec["status"] = "ERROR"
        rec["note"] = str(exc)
        return rec
    if mod is None:
        rec["note"] = "no importable module among: " + ", ".join(tried)
        return rec
    rec["module"] = name

    export_verdict, extra_argv = apply_export(stage, mod, export_dir, is_default_export)
    rec["export"] = export_verdict
    rec["sheet"] = apply_sheet(stage, mod, sheet)

    if export_verdict == "NOT HONOURED":
        rec["status"] = "REFUSED"
        rec["note"] = ("%s binds its export path at import time; it would read the "
                       "DEFAULT export, so it was not run" % name)
        return rec

    if dry_run:
        rec["status"] = "DRY"
        rec["note"] = "would run %s.main(%r)" % (name, extra_argv)
        return rec

    main = getattr(mod, "main", None)
    if main is None:
        rec["status"] = "MISSING"
        rec["note"] = "%s has no main()" % name
        return rec

    print("-" * 78)
    print("[%s] %s -- %s%s" % (stage.name, name, stage.summary,
                               ("  (" + stage.slow_note + ")") if stage.slow_note else ""))
    print("-" * 78)
    try:
        if stage.is_gate and fast and hasattr(mod, "run_all"):
            code = call_gate_fast(mod)
        else:
            code = call_main(main, extra_argv, stage.name)
    except SystemExit as exc:                       # a stage that calls sys.exit()
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:                        # noqa: BLE001 - report, never crash the run
        rec["seconds"] = time.time() - started
        rec["status"] = "ERROR"
        rec["code"] = None
        rec["note"] = "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()
        return rec

    rec["seconds"] = time.time() - started
    rec["code"] = 0 if code is None else int(code)
    rec["status"] = "OK" if rec["code"] == 0 else "FAIL"
    return rec


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def summary_line(rec: Dict[str, object]) -> str:
    """One fixed-width summary line for a finished stage."""
    gate = "GATE" if rec["is_gate"] else "    "
    code = "-" if rec["code"] is None else str(rec["code"])
    flags = []
    if rec["export"] not in ("default",):
        flags.append("export=%s" % rec["export"])
    if rec["sheet"] not in ("n/a",):
        flags.append("sheet=%s" % rec["sheet"])
    tail = "  ".join(x for x in (" ".join(flags), str(rec["note"])) if x)
    return "  [%-7s] %-9s %s %-18s %7.1fs  exit %-3s %s" % (
        rec["status"], rec["stage"], gate, rec["module"] or "-",
        float(rec["seconds"]), code, tail,
    )


def exit_code(records: Sequence[Dict[str, object]]) -> int:
    """0 all clean; 1 a stage FAILED or ERRORED; 2 a stage was MISSING or REFUSED."""
    if any(r["status"] in ("FAIL", "ERROR") for r in records):
        return 1
    if any(r["status"] in ("MISSING", "REFUSED") for r in records):
        return 2
    return 0


def print_stage_table() -> None:
    """Print the declared pipeline: stage, candidate modules, dependency position."""
    print("router.cli stages (dependency order; `all` walks this list)")
    for i, s in enumerate(STAGES, 1):
        gate = " GATE" if s.is_gate else "     "
        print("  %2d.%s %-9s %-42s %s"
              % (i, gate, s.name, " | ".join(s.candidates), s.summary))
    print()
    print("  " + ESTIMATE_CAVEAT)


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #

def selftest() -> int:
    """Check the stage table's own invariants and print PASS/FAIL; 0 iff all pass."""
    checks: List[Tuple[str, object, object]] = []
    checks.append(("stage count", 13, len(STAGES)))
    checks.append(("no duplicate stage names", len(STAGES), len(set(STAGE_NAMES))))
    checks.append(("model before policy", True,
                   STAGE_NAMES.index("model") < STAGE_NAMES.index("policy")))
    checks.append(("policy before gates", True,
                   STAGE_NAMES.index("policy") < STAGE_NAMES.index("gates")))
    checks.append(("gates before ope", True,
                   STAGE_NAMES.index("gates") < STAGE_NAMES.index("ope")))
    checks.append(("features after recon/labels/jobkey", True,
                   STAGE_NAMES.index("features") > STAGE_NAMES.index("jobkey")))
    checks.append(("every export mode declared", True,
                   all(s.export_mode in EXPORT_MODES for s in STAGES)))

    checks.append(("every stage declares a candidate module", len(STAGES),
                   sum(1 for s in STAGES if s.candidates)))

    # Module availability is REPORTED, never asserted: router/figs.py and
    # router/report.py belong to other owners and may legitimately not exist
    # yet. A missing module shows up as MISSING in a real run and makes that
    # run exit 2; it is not a defect in this stage table.
    resolved = {}
    for s in STAGES:
        try:
            resolved[s.name] = resolve_module(s)[1]
        except RuntimeError as exc:
            resolved[s.name] = "BROKEN (%s)" % exc

    try:
        resolve_sheet("definitely-not-a-sheet")
        sheet_guard = False
    except ValueError:
        sheet_guard = True
    checks.append(("--sheet rejects an unregistered sheet", True, sheet_guard))

    try:
        resolve_export(os.path.join(REPO_ROOT, "router"))
        export_guard = False
    except ValueError:
        export_guard = True
    checks.append(("--export rejects a dir with no trajectories", True, export_guard))

    print("router.cli selftest")
    ok = True
    for label, expected, actual in checks:
        good = expected == actual
        ok = ok and good
        print("  [%s] %-42s expected %-6s actual %s"
              % ("PASS" if good else "FAIL", label, expected, actual))
    print()
    print("  stage -> module resolution (reported, not asserted):")
    for name, mod in resolved.items():
        print("    %-9s -> %s" % (name, mod or "MISSING (no candidate importable)"))
    missing = [n for n, m in resolved.items() if not m]
    if missing:
        print("  NOTE: %d stage(s) have no module yet: %s. A real run reports them"
              % (len(missing), ", ".join(missing)))
        print("        MISSING and exits 2; it never silently skips them.")
    print()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for `python -m router.cli`."""
    p = argparse.ArgumentParser(
        prog="python -m router.cli",
        description="Run one router pipeline stage, or the whole pipeline with `all`.",
        epilog=ESTIMATE_CAVEAT,
    )
    p.add_argument("stage", nargs="?", choices=list(STAGE_NAMES) + ["all"],
                   help="stage to run, or `all` for the whole pipeline")
    p.add_argument("--export", default=None,
                   help="export directory (default <repo>/export)")
    p.add_argument("--sheet", default=None,
                   help="ASSUMED price sheet name (default assumed_default)")
    p.add_argument("--fast", action="store_true",
                   help="run the gate at reduced permutations: INDICATIVE, not licensing")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve every stage and print what would run, then stop")
    p.add_argument("--list", action="store_true", help="print the stage table and stop")
    p.add_argument("--selftest", action="store_true",
                   help="check the stage table's invariants and stop")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the requested stage (or `all`); return 0 clean, 1 on failure, 2 on missing."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.list:
        print_stage_table()
        return 0
    if args.selftest:
        return selftest()
    if args.stage is None:
        build_parser().print_help()
        return 2

    try:
        export_dir = resolve_export(args.export)
        sheet = resolve_sheet(args.sheet)
    except ValueError as exc:
        print("router.cli: %s" % exc, file=sys.stderr)
        return 2

    is_default_export = export_dir == os.path.abspath(DEFAULT_EXPORT)
    if not is_default_export:
        patch_io_export(export_dir)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("router.cli -- %s" % ("full pipeline" if args.stage == "all"
                                else "stage %s" % args.stage))
    print("  export %s%s" % (export_dir, "" if is_default_export else "  (NON-DEFAULT)"))
    print("  sheet  %s  (ASSUMED -- model ids are anonymized, no public sheet applies)" % sheet)
    print("  " + ESTIMATE_CAVEAT)
    if not is_default_export:
        print()
        print("  WARNING: --export is NON-DEFAULT but every stage still writes into")
        print("           %s. Artifacts built from the default export" % RESULTS_DIR)
        print("           WILL BE OVERWRITTEN. Point RESULTS elsewhere or rebuild after.")
    print()

    todo = list(STAGES) if args.stage == "all" else [STAGE_BY_NAME[args.stage]]
    records: List[Dict[str, object]] = []
    stopped: Optional[str] = None

    for stage in todo:
        rec = run_stage(stage, export_dir, sheet, is_default_export,
                        fast=args.fast, dry_run=args.dry_run)
        records.append(rec)
        if args.stage == "all" and stage.is_gate and rec["status"] in ("FAIL", "ERROR"):
            stopped = stage.name
            break

    print()
    print("=" * 78)
    print("SUMMARY  (%d stage%s)" % (len(records), "" if len(records) == 1 else "s"))
    for rec in records:
        print(summary_line(rec))
    if stopped:
        remaining = [s.name for s in todo[len(records):]]
        print()
        print("  STOPPED at the failing gate `%s`. Not run: %s"
              % (stopped, ", ".join(remaining) or "(none)"))
        print("  A red gate means no downstream estimate may be quoted.")
    print("=" * 78)
    return exit_code(records)


if __name__ == "__main__":
    raise SystemExit(main())
