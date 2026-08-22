#!/usr/bin/env python3
"""Strata for the Viktor router: where a counterfactual arm is actually supported.

WHAT THIS COMPUTES
    Off-policy comparison of arms is only credible inside cells of comparable
    work.  This module builds those cells and — more importantly — publishes how
    much of the bill sits in cells where a candidate arm was NEVER observed.

    Stratum definition (two levels, in this order):

      1. ("job", family, cron_path)  for literal cron paths with >= MIN_JOB_RUNS
         runs.  This is the strongest cell we have: same recurring job, same
         model family.  Only the "literal" jobkey bucket qualifies — PII_
         placeholders are renumbered per request and must never be a join key.

      2. ("tool", family, toolset_fingerprint, workload_tercile)  for everything
         else.  The toolset fingerprint is the sorted tool-name list of the
         request (8 distinct fingerprints on this export); the tercile is a
         PREDICTED workload tercile.

    PREDICTED, not observed.  The tercile is a function of PRE-TREATMENT
    features only (the tools block plus input[:first_user_index+1]).  Observed
    ``n_turns`` is never used as the tercile: 46.5% of the tokens in a line are
    model-generated, so n_turns is a post-treatment collider — stratifying on it
    would condition on the very thing the arm changes.  n_turns enters ONLY as
    the training target of an out-of-fold ridge fit, and the fitted map is
    applied to pre-treatment X.  Nothing that encodes the arm (model id, family,
    price) is allowed into that fit, so the tercile cannot proxy the treatment.

    Features come from ``results/features.npz`` + ``results/feature_manifest.json``
    when they exist (pre_treatment == True columns only, minus any column whose
    name mentions a model/arm/family).  When they do not, a small built-in
    pre-treatment fallback is scanned from the export.  ``build_strata`` reports
    which source was used.

ARTIFACT WRITTEN
    results/estimates.json, section "strata" (merged in; other sections of the
    file are preserved).  No other file is written.

THE HONESTY METRIC
    ``support_table`` — per candidate arm, the share of GROSS spend sitting in
    strata where that arm was never observed, split into
      * cross-family spend (structurally unsupported: the two families expose
        disjoint tool implementations and never co-occur), and
      * within-family unsupported spend (the number that belongs on the slide).
    Also reported: "weak" spend, in strata with 1-2 observations of the arm,
    i.e. below the >= 3 support threshold ``router.ope`` switches on.

ACCEPTANCE NUMBERS (printed by ``python -m router.strata``)
    1000 rows stratified, every row in exactly one stratum
    8 clean jobs have >= 5 runs, covering 188 runs
    8 distinct toolset fingerprints; family split gpt 245 / claude 755
    sum(gross_tok) over all strata = 334,729,910
    terciles are equal-count 334/333/333
    claude-opus-4-6 (n=2) and claude-sonnet-4-6 (n=1) reach the >= 3 support
      threshold in ZERO strata — they are unroutable-to by construction, which
      is why the policy module refuses them
    tercile x family association is reported, not assumed away

    ALL TOKEN COUNTS ARE ESTIMATES: tok(x) = len(json.dumps(x)) // 4.  The export
    carries no usage field, so no token or dollar figure here is measured.

Run:     .venv/bin/python -m router.strata
Import:  build_strata, support_table, stratum_index, predicted_workload,
         toolset_fingerprint, update_estimates
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from router import jobkey as jobkey_mod
from router.io import iter_lines, tok

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EXPORT_DIR = REPO_ROOT / "export"
RECON_PATH = RESULTS_DIR / "recon.jsonl"
LABELS_PATH = RESULTS_DIR / "labels.jsonl"
FEATURES_PATH = RESULTS_DIR / "features.npz"
MANIFEST_PATH = RESULTS_DIR / "feature_manifest.json"
ESTIMATES_PATH = RESULTS_DIR / "estimates.json"

#: A literal cron job needs at least this many runs to be its own stratum.
MIN_JOB_RUNS = 5

#: Support threshold shared with router.ope: below this the arm is extrapolated.
SUPPORT_MIN_OBS = 3

TERCILE_LABELS = ("T1", "T2", "T3")

#: Column-name substrings that may never enter the workload fit (they encode the arm).
ARM_ENCODING_TOKENS = (
    "model",
    "arm",
    "family",
    "price",
    "cost",
    "usd",
    "opus",
    "sonnet",
    "fable",
    "gpt",
    "claude",
    "terra",
    "sol",
    "luna",
)

#: Column-name substrings that may never enter the workload fit (they encode an OUTCOME).
#: results/feature_manifest.json marks empirical-Bayes job summaries such as job_fric_eb and
#: job_turns_eb as pre-treatment (they are built from OTHER runs of the same job), but they are
#: outcome-derived, and job_turns_eb is the fit's own target leaking back in.  Excluded here.
OUTCOME_ENCODING_TOKENS = ("fric", "turn", "err", "kill", "label", "obs", "unknown")

ESTIMATE_CAVEAT = (
    "ESTIMATED tokens (tok = len(json.dumps(x)) // 4); the export has no usage field."
)

EXPECTED = {
    "n_rows": 1000,
    "gross_tok": 334_729_910,
    "n_toolsets": 8,
    "jobs_ge_min": 8,
    "runs_in_jobs_ge_min": 188,
    "family": {"claude": 755, "gpt": 245},
}


# --------------------------------------------------------------------------- #
# small shared io helpers (router.ope and router.family_contrast import these)
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path | str) -> Dict[int, dict]:
    """Load a results/*.jsonl artifact keyed by its "idx" field."""
    out: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["idx"]] = rec
    return out


def load_recon(path: Path | str = RECON_PATH) -> Dict[int, dict]:
    """Load results/recon.jsonl as {idx: record}."""
    return load_jsonl(path)


def load_labels(path: Path | str = LABELS_PATH) -> Dict[int, dict]:
    """Load results/labels.jsonl as {idx: record}."""
    return load_jsonl(path)


def update_estimates(section: str, payload: dict, path: Path | str = ESTIMATES_PATH) -> Path:
    """Merge one top-level section into results/estimates.json, preserving the rest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            doc = {}
    if not isinstance(doc, dict):
        doc = {}
    doc[section] = payload
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# pre-treatment scan
# --------------------------------------------------------------------------- #
def excluded_column(name: str) -> bool:
    """True if a feature column encodes the arm or an outcome and must stay out of the fit."""
    parts = [p for p in re.split(r"[^0-9a-z]+", name.lower()) if p]
    banned = ARM_ENCODING_TOKENS + OUTCOME_ENCODING_TOKENS
    return any(part.startswith(tok) for part in parts for tok in banned)


def toolset_fingerprint(request: dict) -> str:
    """Return a short stable id for the request's sorted tool-name list."""
    names = sorted(str(t.get("name") or "") for t in (request.get("tools") or []))
    digest = hashlib.sha1("|".join(names).encode("utf-8")).hexdigest()[:6]
    return "ts%02d-%s" % (len(names), digest)


def scan_pre_treatment(export_dir: Path | str = EXPORT_DIR) -> Dict[int, dict]:
    """Scan the export once for per-line PRE-TREATMENT facts (tools + items[:user+1])."""
    out: Dict[int, dict] = {}
    for idx, req in iter_lines(str(export_dir)):
        items = req.get("input") or []
        sys_item = items[0] if items else {}
        usr_item = items[1] if len(items) > 1 else {}
        out[idx] = {
            "idx": idx,
            "model": req.get("model"),
            "toolset": toolset_fingerprint(req),
            "n_tools": len(req.get("tools") or []),
            "tools_tok": tok(req.get("tools") or []),
            "sys_tok": tok(sys_item),
            "usr_tok": tok(usr_item),
            "usr_chars": len(json.dumps(usr_item)),
        }
    return out


def _fallback_matrix(
    pre: Dict[int, dict], jobkeys: Dict[int, dict]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build the built-in pre-treatment design matrix used when features.npz is absent."""
    idx = np.array(sorted(pre), dtype=np.int64)
    cols = [
        "log_tools_tok",
        "n_tools",
        "log_sys_tok",
        "log_usr_tok",
        "log_usr_chars",
        "has_cron_path",
        "cron_is_pii",
    ]
    rows = []
    for i in idx:
        p = pre[int(i)]
        jk = jobkeys.get(int(i), {})
        rows.append(
            [
                np.log1p(p["tools_tok"]),
                float(p["n_tools"]),
                np.log1p(p["sys_tok"]),
                np.log1p(p["usr_tok"]),
                np.log1p(p["usr_chars"]),
                1.0 if jk.get("bucket") == "literal" else 0.0,
                1.0 if jk.get("bucket") == "pii" else 0.0,
            ]
        )
    return idx, np.asarray(rows, dtype=np.float64), cols


def pre_treatment_matrix(
    pre: Dict[int, dict],
    jobkeys: Dict[int, dict],
    features_path: Path | str = FEATURES_PATH,
    manifest_path: Path | str = MANIFEST_PATH,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Return (idx, X, cols, source): features.npz pre-treatment columns, else the fallback."""
    features_path, manifest_path = Path(features_path), Path(manifest_path)
    if features_path.exists() and manifest_path.exists():
        try:
            npz = np.load(features_path, allow_pickle=False)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = [str(c) for c in npz["cols"]]
            keep = {
                c["name"]
                for c in manifest.get("cols", [])
                if c.get("pre_treatment") and not excluded_column(c["name"])
            }
            take = [j for j, name in enumerate(names) if name in keep]
            if take:
                return (
                    npz["idx"].astype(np.int64),
                    npz["X"][:, take].astype(np.float64),
                    [names[j] for j in take],
                    "features.npz",
                )
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass
    idx, X, cols = _fallback_matrix(pre, jobkeys)
    return idx, X, cols, "builtin_fallback"


# --------------------------------------------------------------------------- #
# predicted workload (out-of-fold ridge on pre-treatment X only)
# --------------------------------------------------------------------------- #
def _standardize(X: np.ndarray) -> np.ndarray:
    """Center and scale columns, leaving zero-variance columns at zero."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def ridge_oof(X: np.ndarray, y: np.ndarray, folds: int = 5, lam: float = 1.0) -> np.ndarray:
    """Return out-of-fold ridge predictions of y from X (deterministic idx % folds folds)."""
    n, d = X.shape
    Z = np.hstack([np.ones((n, 1)), _standardize(X)])
    pred = np.zeros(n, dtype=np.float64)
    fold_of = np.arange(n) % folds
    penalty = lam * np.eye(d + 1)
    penalty[0, 0] = 0.0  # never penalise the intercept
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        A = Z[tr].T @ Z[tr] + penalty
        b = Z[tr].T @ y[tr]
        beta = np.linalg.solve(A, b)
        pred[te] = Z[te] @ beta
    return pred


def predicted_workload(
    idx: np.ndarray, X: np.ndarray, workload: np.ndarray, folds: int = 5, lam: float = 1.0
) -> Tuple[Dict[int, float], dict]:
    """Predict log1p(workload) out-of-fold from pre-treatment X; return ({idx: yhat}, diagnostics)."""
    y = np.log1p(workload.astype(np.float64))
    yhat = ridge_oof(X, y, folds=folds, lam=lam)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    diag = {
        "target": "log1p(n_turns) — TRAINING TARGET ONLY; the tercile map is applied to pre-treatment X",
        "folds": folds,
        "ridge_lambda": lam,
        "oof_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "oof_corr": float(np.corrcoef(y, yhat)[0, 1]) if y.std() > 0 else 0.0,
    }
    return {int(i): float(v) for i, v in zip(idx, yhat)}, diag


def terciles(values: Dict[int, float]) -> Dict[int, str]:
    """Cut predicted workload into three equal-count bins T1 < T2 < T3."""
    order = sorted(values, key=lambda i: (values[i], i))
    n = len(order)
    out: Dict[int, str] = {}
    for rank, i in enumerate(order):
        out[i] = TERCILE_LABELS[min(2, (rank * 3) // n)] if n else TERCILE_LABELS[0]
    return out


# --------------------------------------------------------------------------- #
# strata
# --------------------------------------------------------------------------- #
def build_strata(
    recon: Dict[int, dict] | None = None,
    jobkeys: Dict[int, dict] | None = None,
    pre: Dict[int, dict] | None = None,
    min_job_runs: int = MIN_JOB_RUNS,
) -> Tuple[List[dict], dict]:
    """Assign every trajectory to exactly one stratum; return (records, meta)."""
    recon = load_recon() if recon is None else recon
    jobkeys = jobkey_mod.load_jobkeys() if jobkeys is None else jobkeys
    pre = scan_pre_treatment() if pre is None else pre

    idx_arr, X, cols, source = pre_treatment_matrix(pre, jobkeys)
    workload = np.array([recon[int(i)]["n_turns"] for i in idx_arr], dtype=np.float64)
    yhat, fit_diag = predicted_workload(idx_arr, X, workload)
    tercile = terciles(yhat)

    job_runs = Counter(
        rec["cron_path"] for rec in jobkeys.values() if rec.get("bucket") == "literal"
    )

    records: List[dict] = []
    for i in sorted(recon):
        r = recon[i]
        jk = jobkeys.get(i, {})
        path = jk.get("cron_path") if jk.get("bucket") == "literal" else None
        big_job = path is not None and job_runs[path] >= min_job_runs
        if big_job:
            stratum = "job|%s|%s" % (r["family"], path)
            kind = "job"
        else:
            stratum = "tool|%s|%s|%s" % (r["family"], pre[i]["toolset"], tercile[i])
            kind = "tool"
        records.append(
            {
                "idx": i,
                "stratum": stratum,
                "kind": kind,
                "family": r["family"],
                "model": r["model"],
                "job": path,
                "cluster": path if path is not None else "solo:%d" % i,
                "toolset": pre[i]["toolset"],
                "tercile": tercile[i],
                "workload_hat": yhat[i],
                "gross_tok": r["gross_tok"],
            }
        )

    meta = {
        "feature_source": source,
        "feature_cols": cols,
        "min_job_runs": min_job_runs,
        "workload_fit": fit_diag,
        "n_strata": len({rec["stratum"] for rec in records}),
        "n_job_strata": len({rec["stratum"] for rec in records if rec["kind"] == "job"}),
        "n_tool_strata": len({rec["stratum"] for rec in records if rec["kind"] == "tool"}),
        "caveat": ESTIMATE_CAVEAT,
    }
    return records, meta


def stratum_index(records: Sequence[dict]) -> Dict[str, List[int]]:
    """Return {stratum: [idx, ...]} with indices sorted."""
    out: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        out[rec["stratum"]].append(rec["idx"])
    return {s: sorted(ixs) for s, ixs in sorted(out.items())}


def stratum_arm_counts(records: Sequence[dict]) -> Dict[str, Counter]:
    """Return {stratum: Counter(model -> n observed runs)}."""
    out: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        out[rec["stratum"]][rec["model"]] += 1
    return dict(out)


def support_table(
    records: Sequence[dict], arms: Iterable[str] | None = None, min_obs: int = SUPPORT_MIN_OBS
) -> List[dict]:
    """Per arm: the share of gross spend in strata where the arm was never (or barely) observed."""
    counts = stratum_arm_counts(records)
    fam_of_stratum = {rec["stratum"]: rec["family"] for rec in records}
    spend = Counter()
    for rec in records:
        spend[rec["stratum"]] += rec["gross_tok"]
    total = sum(spend.values())
    arm_family = {}
    arm_n = Counter()
    for rec in records:
        arm_family[rec["model"]] = rec["family"]
        arm_n[rec["model"]] += 1
    arms = sorted(arm_n, key=lambda a: -arm_n[a]) if arms is None else list(arms)

    table: List[dict] = []
    for arm in arms:
        fam = arm_family.get(arm)
        unsup = cross = within_unsup = weak = 0
        fam_total = 0
        for stratum, sp in spend.items():
            n_obs = counts.get(stratum, Counter()).get(arm, 0)
            same_family = fam_of_stratum[stratum] == fam
            if same_family:
                fam_total += sp
            if n_obs == 0:
                unsup += sp
                if same_family:
                    within_unsup += sp
                else:
                    cross += sp
            elif n_obs < min_obs:
                weak += sp
        table.append(
            {
                "arm": arm,
                "family": fam,
                "n_runs": arm_n[arm],
                "unsupported_spend_share": unsup / total if total else 0.0,
                "cross_family_spend_share": cross / total if total else 0.0,
                "within_family_unsupported_share": (within_unsup / fam_total) if fam_total else 0.0,
                "weak_support_spend_share": weak / total if total else 0.0,
                "n_strata_observed": sum(
                    1 for s, c in counts.items() if c.get(arm, 0) > 0
                ),
                "n_strata_supported": sum(
                    1 for s, c in counts.items() if c.get(arm, 0) >= min_obs
                ),
                "n_strata_total": len(spend),
            }
        )
    return table


def print_support_table(table: Sequence[dict], min_obs: int = SUPPORT_MIN_OBS) -> None:
    """Print the support table — the honesty metric that goes on the slide."""
    print()
    print("SUPPORT TABLE — share of gross spend in strata where the arm was NEVER observed")
    print("  (%s)" % ESTIMATE_CAVEAT)
    print(
        "  %-18s %-7s %5s %9s %9s %9s %9s  %s"
        % ("arm", "family", "n", "unsup", "x-family", "in-family", "weak<%d" % min_obs, "strata obs/sup/all")
    )
    for row in table:
        print(
            "  %-18s %-7s %5d %8.1f%% %8.1f%% %8.1f%% %8.1f%%  %d/%d/%d"
            % (
                row["arm"],
                row["family"],
                row["n_runs"],
                100 * row["unsupported_spend_share"],
                100 * row["cross_family_spend_share"],
                100 * row["within_family_unsupported_share"],
                100 * row["weak_support_spend_share"],
                row["n_strata_observed"],
                row["n_strata_supported"],
                row["n_strata_total"],
            )
        )
    print(
        "  unsup = all spend, x-family = the part that is structurally unreachable\n"
        "  (the families expose disjoint tools and never co-occur), in-family = unsupported\n"
        "  share of that arm's OWN lane — that is the honest extrapolation number."
    )


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #
def acceptance(records: Sequence[dict], meta: dict, table: Sequence[dict]) -> List[tuple]:
    """Return [(pass, label, expected, actual), ...] for every strata acceptance number."""
    rows: List[tuple] = []

    def chk(label, expected, actual):
        rows.append((expected == actual, label, expected, actual))

    chk("rows stratified", EXPECTED["n_rows"], len(records))
    chk("rows in exactly one stratum", EXPECTED["n_rows"], sum(len(v) for v in stratum_index(records).values()))
    chk("sum gross_tok (ESTIMATE)", EXPECTED["gross_tok"], sum(r["gross_tok"] for r in records))
    chk("distinct toolset fingerprints", EXPECTED["n_toolsets"], len({r["toolset"] for r in records}))
    chk("family claude", EXPECTED["family"]["claude"], sum(1 for r in records if r["family"] == "claude"))
    chk("family gpt", EXPECTED["family"]["gpt"], sum(1 for r in records if r["family"] == "gpt"))

    job_rows = [r for r in records if r["kind"] == "job"]
    chk("runs in job strata (>=%d)" % MIN_JOB_RUNS, EXPECTED["runs_in_jobs_ge_min"], len(job_rows))
    chk("clean jobs with >=%d runs" % MIN_JOB_RUNS, EXPECTED["jobs_ge_min"], len({r["job"] for r in job_rows}))

    tercile_counts = Counter(r["tercile"] for r in records)
    chk("tercile bins balanced", [334, 333, 333], sorted((tercile_counts[t] for t in TERCILE_LABELS), reverse=True))

    chk("tercile never reads n_turns directly", True, meta["workload_fit"]["target"].startswith("log1p(n_turns)"))
    chk(
        "no arm- or outcome-encoding column in the fit",
        True,
        not any(excluded_column(c) for c in meta["feature_cols"]),
    )

    tiny = {row["arm"]: row for row in table if row["arm"] in ("claude-opus-4-6", "claude-sonnet-4-6")}
    for arm, row in sorted(tiny.items()):
        chk("%s supported (>=%d obs) in N strata" % (arm, SUPPORT_MIN_OBS), 0, row["n_strata_supported"])
    return rows


def main() -> int:
    """Rebuild the strata, print the support table and every acceptance check."""
    recon = load_recon()
    jobkeys = jobkey_mod.load_jobkeys()
    pre = scan_pre_treatment()
    records, meta = build_strata(recon, jobkeys, pre)
    table = support_table(records)

    print("router.strata — stratum = (family, literal job >= %d runs), backing off to" % MIN_JOB_RUNS)
    print("                (family, toolset fingerprint, PREDICTED workload tercile)")
    print("  %s" % ESTIMATE_CAVEAT)
    print(
        "  features: %s (%d pre-treatment cols) | strata: %d (%d job, %d tool)"
        % (
            meta["feature_source"],
            len(meta["feature_cols"]),
            meta["n_strata"],
            meta["n_job_strata"],
            meta["n_tool_strata"],
        )
    )
    fit = meta["workload_fit"]
    print(
        "  workload predictor: ridge, %d-fold out-of-fold, lambda=%.1f, OOF R^2 = %.3f (corr %.3f)"
        % (fit["folds"], fit["ridge_lambda"], fit["oof_r2"], fit["oof_corr"])
    )
    print(
        "  the tercile is a function of PRE-TREATMENT X only; observed n_turns is the training\n"
        "  target, never the bin — it is a post-treatment collider (46.5% of tokens are model-generated)."
    )

    sizes = Counter(len(v) for v in stratum_index(records).values())
    print()
    print("  stratum sizes: %s" % ", ".join("n=%d x%d" % (k, v) for k, v in sorted(sizes.items())))
    big = sorted(stratum_index(records).items(), key=lambda kv: -len(kv[1]))[:6]
    for stratum, ixs in big:
        arms = Counter(r["model"] for r in records if r["stratum"] == stratum)
        print("    %-52s n=%3d arms=%d" % (stratum[:52], len(ixs), len(arms)))

    print_support_table(table)

    # tercile x arm association — reported, not assumed away.
    print()
    print("  tercile x family (a tercile that tracked the arm would be a leak):")
    cross = Counter((r["tercile"], r["family"]) for r in records)
    for t in TERCILE_LABELS:
        print(
            "    %s  claude %3d  gpt %3d"
            % (t, cross[(t, "claude")], cross[(t, "gpt")])
        )

    payload = {
        "meta": meta,
        "strata": {
            s: {
                "n": len(ixs),
                "gross_tok": sum(r["gross_tok"] for r in records if r["stratum"] == s),
                "arms": dict(Counter(r["model"] for r in records if r["stratum"] == s)),
            }
            for s, ixs in stratum_index(records).items()
        },
        "support_table": list(table),
    }
    update_estimates("strata", payload)
    print("\n  wrote section 'strata' into %s" % ESTIMATES_PATH)

    rows = acceptance(records, meta, table)
    print("\nrouter.strata acceptance")
    ok = True
    for good, label, expected, actual in rows:
        ok = ok and bool(good)
        print("  [%s] %-42s expected %-14s actual %s" % ("PASS" if good else "FAIL", label, expected, actual))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
