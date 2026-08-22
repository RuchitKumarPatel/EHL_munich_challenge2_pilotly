#!/usr/bin/env python3
"""Cross-family contrast for the Viktor router — a BUNDLE comparison, matched on job.

WHAT THIS COMPUTES
    The friction difference between the two model families, restricted to the 23
    literal cron jobs that were served by BOTH families (229 runs).  Matching on
    the job is what makes the comparison worth anything: the same recurring task,
    the same harness prompt, different family.

    Two disclosures are printed with EVERY result.  They are not footnotes; they
    are the finding:

    (1) THIS IS A MODEL+HARNESS BUNDLE CONTRAST, NOT A MODEL CONTRAST.
        The two families expose disjoint tool implementations — claude lines
        carry bash / file_read / file_edit / file_write, gpt lines carry
        shell_command / apply_patch — with 0 exceptions in 1000 runs and no
        co-occurrence anywhere in the export.  Family and toolset are perfectly
        collinear, so NO amount of this data separates "the model is better"
        from "the tool surface is better".  Any number below is about the bundle.

    (2) 134 OF THE 229 RUNS ARE ONE JOB (crons/heartbeat/), 58.5%.
        A pooled cross-family number is therefore mostly one recurring job plus a
        22-job tail whose second-largest member has n=14.  Every contrast is
        reported twice: with crons/heartbeat/ and without it.

    Inference: cluster bootstrap over JOBS (2000 replicates), plus a within-job
    permutation test that reshuffles the family label inside each job.  The
    permutation test is conditionally valid only if the family that served a run
    was as-good-as-random within the job — which is an assumption, not a fact:
    the export carries no assignment mechanism.

ARTIFACT WRITTEN
    results/estimates.json, section "family_contrast" (merged; the rest of the
    file is preserved).

ACCEPTANCE NUMBERS (printed by ``python -m router.family_contrast``)
    23 clean cross-family jobs, 229 runs
    pooled claude 33/157 = 21.0% friction
    pooled gpt    21/72  = 29.2% friction
    crons/heartbeat/ contributes 134 of the 229 runs
    the pooled gap is smaller than the MDE at this n — the contrast is NOT
      established, and is reported as an interval, never as a point claim

    Friction is the exit-code-grounded y_fric label from results/labels.jsonl.
    Token/spend figures elsewhere in this project are ESTIMATES; this module
    reports run counts and rates, which are exact.

Run:     .venv/bin/python -m router.family_contrast
Import:  build_runs, build_pairs, pooled_rates, pooled_contrast,
         matched_pair_contrast, permutation_pvalue, leave_one_out, mde
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from router import jobkey as jobkey_mod
from router.strata import RESULTS_DIR, load_labels, update_estimates

HEARTBEAT = "crons/heartbeat/"
DEFAULT_N_BOOT = 2000
DEFAULT_N_PERM = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 20260822

#: Normal quantiles, hardcoded because scipy is not installed (and must not be).
Z_975 = 1.959963985
Z_80 = 0.841621234

DISCLOSURES = (
    "BUNDLE, NOT MODEL: the families expose disjoint tool implementations "
    "(bash/file_read/file_edit/file_write vs shell_command/apply_patch), perfectly "
    "collinear with the family in 1000/1000 runs and never co-occurring. This contrast "
    "is model+harness bundle vs model+harness bundle; no amount of this data separates them.",
    "ONE JOB DOMINATES: 134 of the 229 cross-family runs (58.5%) are the single job "
    "crons/heartbeat/. Every contrast below is reported WITH and WITHOUT it.",
)

EXPECTED = {
    "n_jobs": 23,
    "n_runs": 229,
    "claude_fric": 33,
    "claude_n": 157,
    "gpt_fric": 21,
    "gpt_n": 72,
    "heartbeat_runs": 134,
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_runs(
    jobkeys: Dict[int, dict] | None = None,
    arms: Dict[int, dict] | None = None,
    labels: Dict[int, dict] | None = None,
) -> List[dict]:
    """Return one record per run on a clean cross-family job: job, family, model, y_fric."""
    jobkeys = jobkey_mod.load_jobkeys() if jobkeys is None else jobkeys
    arms = jobkey_mod.load_arms() if arms is None else arms
    labels = load_labels() if labels is None else labels
    jobs = jobkey_mod.cross_family_jobs(jobkeys, arms)
    runs: List[dict] = []
    for job, ixs in sorted(jobs.items()):
        for i in ixs:
            runs.append(
                {
                    "idx": i,
                    "job": job,
                    "family": arms[i]["family"],
                    "model": arms[i]["model"],
                    "y": int(labels[i]["y_fric"]),
                }
            )
    return runs


def build_pairs(runs: Sequence[dict]) -> List[dict]:
    """Collapse runs into one matched pair per job: counts and friction rates per family."""
    by_job: Dict[str, dict] = {}
    for r in runs:
        cell = by_job.setdefault(r["job"], {"job": r["job"], "n_c": 0, "y_c": 0, "n_g": 0, "y_g": 0})
        if r["family"] == "claude":
            cell["n_c"] += 1
            cell["y_c"] += r["y"]
        else:
            cell["n_g"] += 1
            cell["y_g"] += r["y"]
    pairs = []
    for cell in by_job.values():
        p_c = cell["y_c"] / cell["n_c"] if cell["n_c"] else float("nan")
        p_g = cell["y_g"] / cell["n_g"] if cell["n_g"] else float("nan")
        pairs.append(
            {
                **cell,
                "n": cell["n_c"] + cell["n_g"],
                "p_c": p_c,
                "p_g": p_g,
                "diff": p_g - p_c,
                "mh_weight": (cell["n_c"] * cell["n_g"]) / (cell["n_c"] + cell["n_g"]),
            }
        )
    return sorted(pairs, key=lambda p: (-p["n"], p["job"]))


def drop_job(items: Sequence[dict], job: str) -> List[dict]:
    """Return the runs or pairs with one job removed (used for the heartbeat sensitivity)."""
    return [it for it in items if it["job"] != job]


# --------------------------------------------------------------------------- #
# estimators — all return (estimate, lo, hi, diagnostics)
# --------------------------------------------------------------------------- #
def pooled_rates(runs: Sequence[dict]) -> dict:
    """Pooled friction counts and rates per family over the given runs."""
    out = {}
    for fam in ("claude", "gpt"):
        rows = [r for r in runs if r["family"] == fam]
        n = len(rows)
        k = sum(r["y"] for r in rows)
        out[fam] = {"n": n, "fric": k, "rate": (k / n) if n else float("nan")}
    out["n_jobs"] = len({r["job"] for r in runs})
    out["n_runs"] = len(runs)
    return out


def mde(n1: int, n2: int, p: float, z_alpha: float = Z_975, z_power: float = Z_80) -> float:
    """Minimum detectable difference in proportions at these n's (two-sided 5%, 80% power)."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return (z_alpha + z_power) * float(np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))


def _job_bootstrap(
    runs: Sequence[dict], stat, n_boot: int, alpha: float, seed: int
) -> Tuple[float, float, dict]:
    """Cluster bootstrap over jobs; returns (lo, hi, diagnostics) for the given statistic."""
    jobs = sorted({r["job"] for r in runs})
    by_job = {j: [r for r in runs if r["job"] == j] for j in jobs}
    rng = np.random.default_rng(seed)
    single = len(jobs) < 2
    # With one job there is no between-job variation to resample, so a job bootstrap would
    # report a zero-width interval.  Resample RUNS instead, preserving each family's count.
    fam_rows = {f: [r for r in runs if r["family"] == f] for f in ("claude", "gpt")}
    draws = []
    for _ in range(n_boot):
        sample: List[dict] = []
        if single:
            for rows in fam_rows.values():
                if rows:
                    pick = rng.integers(0, len(rows), size=len(rows))
                    sample.extend(rows[k] for k in pick)
        else:
            pick = rng.integers(0, len(jobs), size=len(jobs))
            for j in pick:
                sample.extend(by_job[jobs[j]])
        val = stat(sample)
        if np.isfinite(val):
            draws.append(val)
    arr = np.asarray(draws, dtype=np.float64)
    lo = float(np.quantile(arr, alpha / 2)) if arr.size else float("nan")
    hi = float(np.quantile(arr, 1 - alpha / 2)) if arr.size else float("nan")
    return lo, hi, {
        "n_boot": n_boot,
        "n_clusters": len(jobs),
        "cluster": (
            "run within the single job, family sizes preserved (no between-job variation exists)"
            if single
            else "literal cron job path"
        ),
        "replicates_used": int(arr.size),
        "boot_sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "alpha": alpha,
        "seed": seed,
    }


def _pooled_diff(runs: Sequence[dict]) -> float:
    """gpt friction rate minus claude friction rate, pooled over runs."""
    r = pooled_rates(runs)
    if not r["claude"]["n"] or not r["gpt"]["n"]:
        return float("nan")
    return r["gpt"]["rate"] - r["claude"]["rate"]


def _mh_diff(runs: Sequence[dict]) -> float:
    """Matched-pair (Mantel-Haenszel weighted) mean of the per-job gpt-minus-claude difference."""
    pairs = [p for p in build_pairs(runs) if p["n_c"] and p["n_g"]]
    if not pairs:
        return float("nan")
    w = np.array([p["mh_weight"] for p in pairs], dtype=np.float64)
    d = np.array([p["diff"] for p in pairs], dtype=np.float64)
    return float((w * d).sum() / w.sum())


def pooled_contrast(
    runs: Sequence[dict], n_boot: int = DEFAULT_N_BOOT, alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float, dict]:
    """Pooled gpt-minus-claude friction difference with a job-clustered bootstrap interval."""
    rates = pooled_rates(runs)
    est = _pooled_diff(runs)
    lo, hi, boot = _job_bootstrap(runs, _pooled_diff, n_boot, alpha, seed)
    pooled_p = (rates["claude"]["fric"] + rates["gpt"]["fric"]) / max(rates["n_runs"], 1)
    diag = {
        "estimator": "pooled difference in friction rates, jobs matched by construction",
        "rates": rates,
        "mde_80pct_power": mde(rates["claude"]["n"], rates["gpt"]["n"], pooled_p),
        "established": bool(abs(est) > mde(rates["claude"]["n"], rates["gpt"]["n"], pooled_p)),
        "bootstrap": boot,
        "disclosures": list(DISCLOSURES),
    }
    return est, lo, hi, diag


def matched_pair_contrast(
    runs: Sequence[dict], n_boot: int = DEFAULT_N_BOOT, alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float, dict]:
    """Mantel-Haenszel matched-pair contrast across jobs, with a job-clustered bootstrap."""
    est = _mh_diff(runs)
    lo, hi, boot = _job_bootstrap(runs, _mh_diff, n_boot, alpha, seed)
    pairs = build_pairs(runs)
    diag = {
        "estimator": "per-job difference, Mantel-Haenszel weights n_c*n_g/(n_c+n_g)",
        "n_pairs": len(pairs),
        "largest_pair": {"job": pairs[0]["job"], "n": pairs[0]["n"]} if pairs else None,
        "weight_share_largest": (
            pairs[0]["mh_weight"] / sum(p["mh_weight"] for p in pairs) if pairs else float("nan")
        ),
        "bootstrap": boot,
        "disclosures": list(DISCLOSURES),
    }
    return est, lo, hi, diag


def permutation_pvalue(
    runs: Sequence[dict], n_perm: int = DEFAULT_N_PERM, seed: int = DEFAULT_SEED
) -> Tuple[float, float, float, dict]:
    """Two-sided within-job permutation test of the family label; returns (obs, nan, nan, diag)."""
    jobs = sorted({r["job"] for r in runs})
    obs = _pooled_diff(runs)
    rng = np.random.default_rng(seed)
    blocks = []
    for j in jobs:
        rows = [r for r in runs if r["job"] == j]
        y = np.array([r["y"] for r in rows], dtype=np.float64)
        n_g = sum(1 for r in rows if r["family"] == "gpt")
        blocks.append((y, n_g))
    n_c_tot = sum(len(y) - n_g for y, n_g in blocks)
    n_g_tot = sum(n_g for _y, n_g in blocks)
    hits = 0
    for _ in range(n_perm):
        s_g = s_c = 0.0
        for y, n_g in blocks:
            perm = rng.permutation(y)
            s_g += perm[:n_g].sum()
            s_c += perm[n_g:].sum()
        diff = s_g / n_g_tot - s_c / n_c_tot
        if abs(diff) >= abs(obs) - 1e-12:
            hits += 1
    p = (hits + 1) / (n_perm + 1)
    diag = {
        "test": "within-job permutation of the family label (family assignment reshuffled inside each job)",
        "p_value_two_sided": p,
        "n_perm": n_perm,
        "observed_diff": obs,
        "validity": (
            "conditionally valid ONLY if the serving family was as-good-as-random within a job; "
            "the export records no assignment mechanism, so this is an assumption, not a fact"
        ),
        "disclosures": list(DISCLOSURES),
    }
    return obs, float("nan"), float("nan"), diag


def leave_one_out(runs: Sequence[dict]) -> List[dict]:
    """Per job: the pooled contrast recomputed with that job removed (influence check)."""
    jobs = sorted({r["job"] for r in runs})
    full = _pooled_diff(runs)
    out = []
    for j in jobs:
        rest = drop_job(runs, j)
        d = _pooled_diff(rest)
        out.append(
            {
                "job": j,
                "n_dropped": sum(1 for r in runs if r["job"] == j),
                "diff_without": d,
                "influence": full - d,
            }
        )
    return sorted(out, key=lambda r: -abs(r["influence"]))


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def print_disclosures() -> None:
    """Print the two mandatory disclosures — they accompany every result in this module."""
    for i, text in enumerate(DISCLOSURES, start=1):
        print("  DISCLOSURE %d: %s" % (i, text))


def print_pair_table(pairs: Sequence[dict]) -> None:
    """Print the per-job matched-pair table, largest job first."""
    print()
    print("  %-52s %5s %10s %10s %8s" % ("job", "n", "claude", "gpt", "gpt-claude"))
    for p in pairs:
        print(
            "  %-52s %5d %4d/%-5d %4d/%-5d %+8.1f pp"
            % (
                (p["job"][:50] + "..") if len(p["job"]) > 52 else p["job"],
                p["n"],
                p["y_c"],
                p["n_c"],
                p["y_g"],
                p["n_g"],
                100 * p["diff"],
            )
        )


def _report(label: str, runs: Sequence[dict], n_boot: int, seed: int) -> dict:
    """Print and return the full contrast bundle for one run set."""
    rates = pooled_rates(runs)
    est, lo, hi, diag = pooled_contrast(runs, n_boot=n_boot, seed=seed)
    mh, mh_lo, mh_hi, mh_diag = matched_pair_contrast(runs, n_boot=n_boot, seed=seed)
    _o, _l, _h, perm = permutation_pvalue(runs, n_perm=DEFAULT_N_PERM, seed=seed)
    print()
    print("%s — %d jobs, %d runs" % (label, rates["n_jobs"], rates["n_runs"]))
    print(
        "  claude %d/%-4d = %5.1f%%   gpt %d/%-4d = %5.1f%%"
        % (
            rates["claude"]["fric"],
            rates["claude"]["n"],
            100 * rates["claude"]["rate"],
            rates["gpt"]["fric"],
            rates["gpt"]["n"],
            100 * rates["gpt"]["rate"],
        )
    )
    unit = "jobs" if diag["bootstrap"]["n_clusters"] > 1 else "runs (single job — see cluster note)"
    print(
        "  pooled contrast      %+5.1f pp  95%% CI [%+5.1f, %+5.1f]  (bootstrap over %d %s)"
        % (100 * est, 100 * lo, 100 * hi, diag["bootstrap"]["n_clusters"] if diag["bootstrap"]["n_clusters"] > 1 else len(runs), unit)
    )
    print(
        "  matched-pair (MH)    %+5.1f pp  95%% CI [%+5.1f, %+5.1f]  largest job holds %.0f%% of the weight"
        % (100 * mh, 100 * mh_lo, 100 * mh_hi, 100 * mh_diag["weight_share_largest"])
    )
    print(
        "  MDE at 80%% power     %5.1f pp  -> the gap is %s at this n"
        % (100 * diag["mde_80pct_power"], "ESTABLISHED" if diag["established"] else "NOT ESTABLISHED")
    )
    print("  within-job permutation p = %.3f (%s)" % (perm["p_value_two_sided"], "two-sided"))
    print_disclosures()
    return {
        "rates": rates,
        "pooled": {"estimate": est, "lo": lo, "hi": hi, "diagnostics": diag},
        "matched_pair": {"estimate": mh, "lo": mh_lo, "hi": mh_hi, "diagnostics": mh_diag},
        "permutation": perm,
    }


# --------------------------------------------------------------------------- #
# acceptance + main
# --------------------------------------------------------------------------- #
def acceptance(runs: Sequence[dict]) -> List[tuple]:
    """Return [(pass, label, expected, actual), ...] for every family-contrast acceptance number."""
    rates = pooled_rates(runs)
    rows: List[tuple] = []

    def chk(label, expected, actual):
        rows.append((expected == actual, label, expected, actual))

    chk("clean cross-family jobs", EXPECTED["n_jobs"], rates["n_jobs"])
    chk("cross-family runs", EXPECTED["n_runs"], rates["n_runs"])
    chk("claude friction count", EXPECTED["claude_fric"], rates["claude"]["fric"])
    chk("claude runs", EXPECTED["claude_n"], rates["claude"]["n"])
    chk("gpt friction count", EXPECTED["gpt_fric"], rates["gpt"]["fric"])
    chk("gpt runs", EXPECTED["gpt_n"], rates["gpt"]["n"])
    chk("claude friction rate", "21.0%", "%.1f%%" % (100 * rates["claude"]["rate"]))
    chk("gpt friction rate", "29.2%", "%.1f%%" % (100 * rates["gpt"]["rate"]))
    chk(
        "crons/heartbeat/ runs",
        EXPECTED["heartbeat_runs"],
        sum(1 for r in runs if r["job"] == HEARTBEAT),
    )
    pooled_p = (rates["claude"]["fric"] + rates["gpt"]["fric"]) / rates["n_runs"]
    gap = abs(rates["gpt"]["rate"] - rates["claude"]["rate"])
    m = mde(rates["claude"]["n"], rates["gpt"]["n"], pooled_p)
    rows.append((gap < m, "pooled gap is below the MDE (not established)", "gap < MDE", "%.1f pp < %.1f pp" % (100 * gap, 100 * m)))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Print the cross-family bundle contrast with and without crons/heartbeat/, plus acceptance."""
    runs = build_runs()
    pairs = build_pairs(runs)

    print("router.family_contrast — cross-family BUNDLE contrast, matched on literal cron job")
    print("  outcome: y_fric from results/labels.jsonl (exit-code grounded, not a text heuristic)")
    print_disclosures()
    print_pair_table(pairs)

    full = _report("ALL 23 CROSS-FAMILY JOBS", runs, DEFAULT_N_BOOT, DEFAULT_SEED)
    without = _report(
        "WITHOUT %s" % HEARTBEAT, drop_job(runs, HEARTBEAT), DEFAULT_N_BOOT, DEFAULT_SEED
    )
    hb_only = _report(
        "%s ONLY" % HEARTBEAT, [r for r in runs if r["job"] == HEARTBEAT], DEFAULT_N_BOOT, DEFAULT_SEED
    )

    loo = leave_one_out(runs)
    print()
    print("  leave-one-job-out influence on the pooled contrast (top 5):")
    for row in loo[:5]:
        print(
            "    drop %-46s (n=%3d) -> %+5.1f pp   shift %+5.1f pp"
            % (row["job"][:46], row["n_dropped"], 100 * row["diff_without"], 100 * row["influence"])
        )

    payload = {
        "disclosures": list(DISCLOSURES),
        "pairs": pairs,
        "all_jobs": full,
        "without_heartbeat": without,
        "heartbeat_only": hb_only,
        "leave_one_out": loo,
        "outcome": "y_fric (results/labels.jsonl)",
    }
    update_estimates("family_contrast", payload)
    print("\n  wrote section 'family_contrast' into %s/estimates.json" % RESULTS_DIR)

    rows = acceptance(runs)
    print("\nrouter.family_contrast acceptance")
    ok = True
    for good, label, expected, actual in rows:
        ok = ok and bool(good)
        print("  [%s] %-42s expected %-14s actual %s" % ("PASS" if good else "FAIL", label, expected, actual))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
