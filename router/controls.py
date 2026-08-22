#!/usr/bin/env python3
"""THE PUBLICATION GATE. Six negative controls that must survive before anything is claimed.

WHAT THIS COMPUTES
    `run_all()` runs six controls over the pipeline's artifacts, prints a banner, and
    returns PASS or FAIL. `main()` exits non-zero on FAIL so the Makefile can depend
    on it. Nothing downstream — no chart, no estimate, no slide — is licensed by a
    red banner.

      1 permutation null      shuffle y, REFIT THE WHOLE PIPELINE, score out-of-fold.
                              The shuffled AUPRC must sit at the finite-sample chance
                              level. If a shuffled label still beats the majority
                              baseline, the pipeline leaks and every number it
                              produced is void. Note the whole pipeline is permuted,
                              feature selection and standardisation included: those
                              are exactly where a leak hides, and permuting only the
                              final fit would miss it.
      2 random-feature        pure-noise columns must score at chance through the same
                              pipeline; and when a noise column is dropped in beside
                              the real ones, it must not rank among the strong
                              coefficients or hold its sign across folds.
      3 no post-treatment     every column in results/feature_manifest.json carries
                              pre_treatment: true, the manifest and features.npz agree
                              in order and length, and no banned column name appears.
      4 SMD balance           standardized mean differences on the PRE-TREATMENT
                              covariates for each arm pair inside each stratum;
                              every |SMD| >= 0.1 is flagged. This is observational
                              data with no randomisation, so imbalance is expected and
                              is reported as WARN, not FAIL — but a flagged pair is a
                              pair whose contrast must be reported as BOUNDED, never
                              as adjusted-away.
      5 weight concentration  share of sum(w) carried by the top 5 lines. The spend
                              weights are radically skewed (top 10 lines = 17.7% of
                              the bill), so a total that rides on five rows is a
                              total with no standard error worth printing.
      6 growth-ratio OOS      fit g_a = sum(est)/sum(pre) on a random 70% of arm a's
                              rows, then predict the held-out 30%'s total.

WHY CHECK 6 IS A HOLDOUT AND NOT THE RATIO IDENTITY
    The tempting check — "does sum(pre_i * g_a) equal sum(est_i)?" — CANNOT FAIL and
    validates nothing. With g_a = sum(est)/sum(pre) taken over arm a's OWN rows,
    sum(pre_i * g_a) = g_a * sum(pre_i) = sum(est_i) by construction: it is the
    definition of g_a rearranged, an algebraic tautology that would return 0.000 error
    on random noise, on shuffled labels, and on a bug that silently zeroed half the
    corpus. A control that passes on garbage is not a control. Check 6 therefore fits
    g_a on one subset and is scored on a DISJOINT one, where the answer is not
    guaranteed and does in fact carry real error.

WHAT IT WRITES
    Nothing. This module reads results/ and prints. It is a gate, not a producer.

DEPENDENCIES AND DEGRADED MODES
    Reads results/recon.jsonl, labels.jsonl, jobkey.jsonl, features.npz +
    feature_manifest.json, routes.jsonl. A missing artifact makes the checks that need
    it BLOCKED, never PASS. Exit code 2 means BLOCKED (cannot certify), 1 means FAIL
    (certified broken), 0 means the gate is green.

    The reference logistic fit used by controls 1 and 2 lives HERE, not in
    router.predictor, and that is deliberate: a control that shares its estimator with
    the thing it is controlling cannot detect a bug in that estimator. Pass
    `fit_fn=` to additionally run the controls through the real predictor.

ACCEPTANCE (checked by `python -m router.controls --selftest`)
    The gate is run twice over synthetic fixtures whose answer is known:
      * a CLEAN fixture raises no FAIL on any of the six checks;
      * a POISONED fixture — full-data label-driven feature selection wrapped around
        the CV, a post-treatment column declared in the manifest, one row carrying
        most of the weight, and an arm whose growth ratio is unstable — raises FAIL on
        checks 1, 2, 3, 5 and 6 and flags imbalance on check 4.
    A gate that cannot fail is worth exactly as much as the tautology in check 6, so
    the acceptance test is that it DOES fail on the poisoned fixture.

    Live figures reproduced against the real artifacts (`python -m router.controls`):
      spend concentration on gross_tok: top-1 3.4%, top-5 12.1%, top-10 17.7%,
      top-50 40.5%, top-100 54.7%, top-500 90.2% — the last four are checked against
      docs/CONTRACTS.md and FAIL the gate on any disagreement, so a regression in
      results/recon.jsonl cannot pass through this module unnoticed. Kish ESS of the
      spend weights is 164.0 of 1000 lines.
      Growth-ratio holdout, 70/30, 400 splits, seed 0, median |relative error|:
      opus-5 0.141, terra 0.160, opus-4-8 0.241, sonnet-5 0.248, fable-5 0.283,
      sol 0.341. A counterfactual total built by rescaling pre-treatment tokens by
      g_a therefore carries roughly 14%-34% error. The in-sample ratio identity
      reports 0.000 on the same data, which is the point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from router.metrics import auprc

#: Ordered names of the six controls, as they appear in the banner.
CHECKS = (
    "1 permutation-null",
    "2 random-feature",
    "3 no-post-treatment",
    "4 smd-balance",
    "5 weight-concentration",
    "6 growth-ratio-oos",
)

#: Column-name fragments that must never reach a design matrix. Each is banned by a
#: measurement, not by taste (see docs/CONTRACTS.md).
BANNED_FEATURE_FRAGMENTS = (
    "empty_final",        # 238/238 empty finals are gpt: encodes family perfectly
    "final_text",         # same artifact
    "sysprompt_hash",     # 871 distinct over 1000 rows: an identifier
    "system_prompt_hash",
    "n_obs", "n_err", "n_unknown", "trailing_err", "n_kill",   # read tool OUTPUTS
    "y_fric", "y_proc", "y_rate",                              # the labels themselves
    "n_turns", "n_calls",                                      # model-produced volume
    "gross_tok", "cache_read", "cache_write", "naive_tok",     # post-treatment tokens
)

#: |SMD| at or above this is flagged as imbalance (the conventional 0.1 threshold).
SMD_FLAG = 0.10

#: Check 5 fails when the top 5 lines carry at least this share of sum(w).
TOP5_FAIL_SHARE = 0.25

#: Check 6 bands on the median absolute relative error of the held-out total.
GROWTH_WARN, GROWTH_FAIL = 0.25, 0.50

#: Checks 1 and 2 fail when a shuffled/noise fit clears prevalence by this much.
CHANCE_MARGIN = 0.02


# ============================================== reference estimator (ours) ===

def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 30) -> np.ndarray:
    """L2-penalised logistic regression by IRLS; returns [intercept, coefs] (numpy only)."""
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    beta = np.zeros(d + 1)
    ridge = l2 * np.eye(d + 1)
    ridge[0, 0] = 0.0                                   # never penalise the intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30.0, 30.0)))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        grad = Xb.T @ (y - p) - ridge @ beta
        hess = (Xb.T * w) @ Xb + ridge
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def group_folds(groups, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Fold assignment that keeps every row of a group together (job-blocked CV)."""
    groups = np.asarray([str(g) for g in groups])
    uniq, sizes = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    uniq, sizes = uniq[order], sizes[order]
    order = np.argsort(-sizes, kind="mergesort")        # largest group first, greedy fill
    load = np.zeros(k)
    assign: dict[str, int] = {}
    for i in order:
        f = int(np.argmin(load))
        assign[uniq[i]] = f
        load[f] += sizes[i]
    fold_of = np.array([assign[g] for g in groups])
    return [np.flatnonzero(fold_of == f) for f in range(k)]


def oof_scores(X, y, groups=None, k: int = 5, l2: float = 1.0, seed: int = 0) -> np.ndarray:
    """Out-of-fold predicted probabilities; standardisation uses TRAIN-fold statistics only."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = X.shape[0]
    groups = np.arange(n) if groups is None else groups
    out = np.full(n, np.nan)
    for te in group_folds(groups, k=k, seed=seed):
        tr = np.setdiff1d(np.arange(n), te)
        if tr.size == 0 or te.size == 0 or len(np.unique(y[tr])) < 2:
            out[te] = float(y[tr].mean()) if tr.size else float(y.mean())
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd == 0, 1.0, sd)
        beta = logistic_fit((X[tr] - mu) / sd, y[tr], l2=l2)
        z = np.c_[np.ones(te.size), (X[te] - mu) / sd] @ beta
        out[te] = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return out


def _default_fit_fn(X, y, groups, seed: int = 0) -> np.ndarray:
    """The reference pipeline the controls permute: job-blocked CV, train-fold standardisation."""
    return oof_scores(X, y, groups=groups, seed=seed)


# ================================================================= loading ===

def _results_dir(results_dir=None) -> Path:
    """Resolve the results/ directory (repo-relative by default)."""
    if results_dir:
        return Path(results_dir)
    return Path(__file__).resolve().parent.parent / "results"


def _jsonl(path: Path) -> list[dict]:
    """Read a JSONL artifact, or return [] when the file is absent."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bundle(results_dir=None) -> dict:
    """Collect every artifact the gate can use; absent ones become None, never a guess."""
    res = _results_dir(results_dir)
    recon = _jsonl(res / "recon.jsonl")
    labels = _jsonl(res / "labels.jsonl")
    jobkey = _jsonl(res / "jobkey.jsonl")
    routes = _jsonl(res / "routes.jsonl")
    bundle: dict = {
        "results_dir": str(res),
        "recon": recon or None,
        "labels": {int(r["idx"]): r for r in labels} or None,
        "jobkey": {int(r["idx"]): r for r in jobkey} or None,
        "routes": routes or None,
        "routes_conforming": bool(routes) and all("idx" in r for r in routes),
        "X": None, "cols": None, "idx": None, "manifest": None,
        "manifest_path": str(res / "feature_manifest.json"),
        "features_path": str(res / "features.npz"),
    }
    npz = res / "features.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            bundle["X"] = np.asarray(z["X"], dtype=np.float64)
            bundle["cols"] = [str(c) for c in z["cols"]]
            bundle["idx"] = np.asarray(z["idx"], dtype=np.int64)
    man = res / "feature_manifest.json"
    if man.exists():
        bundle["manifest"] = json.loads(man.read_text())

    # Vectors the checks share, derived only from artifacts that exist.
    if recon and bundle["labels"]:
        order = [r["idx"] for r in recon]
        bundle["order"] = np.asarray(order, dtype=np.int64)
        bundle["y"] = np.asarray([bundle["labels"][i]["y_fric"] for i in order], dtype=np.float64)
        bundle["arm"] = np.asarray([r["model"] for r in recon])
        bundle["w"] = np.asarray([float(r["gross_tok"]) for r in recon])
        bundle["pre"] = np.asarray([float(r["pre_tok"]) for r in recon])
        bundle["est"] = np.asarray([float(r["gross_tok"]) for r in recon])
        # FALLBACK pre-treatment covariates: everything here is fixed before the model
        # runs (system + user prompt size, toolset size). Used only when features.npz
        # is absent, and labelled as such in the report.
        tools = np.asarray([float(r["tools_tok"]) for r in recon])
        bundle["fallback_cov"] = np.c_[bundle["pre"], tools, bundle["pre"] - tools]
        bundle["fallback_cov_cols"] = ["pre_tok", "tools_tok", "prompt_tok"]
    if bundle["jobkey"] and recon:
        bundle["stratum"] = np.asarray([
            bundle["jobkey"].get(r["idx"], {}).get("cron_path")
            if bundle["jobkey"].get(r["idx"], {}).get("literal") else None
            for r in recon
        ], dtype=object)
    return bundle


def _covariates(bundle) -> tuple[np.ndarray | None, list[str], str]:
    """Pre-treatment design matrix, ROW-ALIGNED to the label vector by idx (never by file order)."""
    if bundle.get("X") is not None and bundle.get("cols"):
        X = bundle["X"]
        order, fidx = bundle.get("order"), bundle.get("idx")
        if order is not None and fidx is not None:
            # features.npz and recon.jsonl happen to share an order today; joining on
            # idx anyway means a future reordering of either artifact silently aligns
            # instead of silently mismatching every row against the wrong label.
            pos = {int(i): k for k, i in enumerate(fidx)}
            missing = [int(i) for i in order if int(i) not in pos]
            if missing:
                raise KeyError(f"features.npz is missing {len(missing)} idx values, "
                               f"first {missing[:3]}")
            X = X[[pos[int(i)] for i in order]]
        return X, list(bundle["cols"]), "results/features.npz (joined on idx)"
    if bundle.get("fallback_cov") is not None:
        return (bundle["fallback_cov"], list(bundle["fallback_cov_cols"]),
                "FALLBACK pre-treatment covariates from recon (features.npz absent)")
    return None, [], "none"


def _res(name, verdict, headline, **detail) -> dict:
    """Build one check result record."""
    return {"check": name, "verdict": verdict, "headline": headline, **detail}


# ============================================================== the checks ===

def check_permutation_null(bundle, n_permutations: int = 50, seed: int = 0,
                           fit_fn=None, margin: float = CHANCE_MARGIN) -> dict:
    """Control 1: shuffle y, refit the whole pipeline, and demand a chance-level AUPRC."""
    name = CHECKS[0]
    X, cols, src = _covariates(bundle)
    y = bundle.get("y")
    if X is None or y is None:
        return _res(name, "BLOCKED", "no design matrix and/or labels available",
                    needs="results/features.npz (or recon+labels) and results/labels.jsonl")
    fit = fit_fn or _default_fit_fn
    groups = bundle.get("stratum")
    groups = np.array([g if g else f"_solo_{i}" for i, g in enumerate(groups)]) \
        if groups is not None else np.arange(len(y))

    real = float(auprc(y, fit(X, y, groups, seed)))
    prev = float(y.mean())
    rng = np.random.default_rng(seed)
    shuffled = np.array([
        float(auprc(yp, fit(X, yp, groups, seed)))
        for yp in (rng.permutation(y) for _ in range(n_permutations))
    ])
    med = float(np.median(shuffled))
    p95 = float(np.quantile(shuffled, 0.95))
    pval = float((shuffled >= real).mean())
    leaks = med > prev + margin
    return _res(
        name, "FAIL" if leaks else "PASS",
        f"shuffled-label OOF AUPRC median {med:.4f} vs majority baseline {prev:.4f} "
        f"(margin {margin}); real {real:.4f}, permutation p = {pval:.3f}",
        design_matrix=src, n_features=len(cols), n=int(y.size),
        real_auprc=real, prevalence=prev, shuffled_median=med, shuffled_p95=p95,
        shuffled_mean=float(shuffled.mean()), n_permutations=int(n_permutations),
        permutation_p_value=pval,
        note=("the WHOLE pipeline is refit under each shuffle, so a label-driven "
              "preprocessing step (selection, target encoding, full-data scaling) "
              "shows up here rather than hiding behind the final fit"),
    )


def check_random_feature(bundle, n_repeats: int = 10, seed: int = 0,
                         fit_fn=None, margin: float = CHANCE_MARGIN) -> dict:
    """Control 2: pure noise must score at chance, and a planted noise column must stay weak."""
    name = CHECKS[1]
    y = bundle.get("y")
    if y is None:
        return _res(name, "BLOCKED", "no labels available", needs="results/labels.jsonl")
    fit = fit_fn or _default_fit_fn
    n = int(y.size)
    groups = np.arange(n)
    rng = np.random.default_rng(seed + 991)
    noise_auprc = np.array([
        float(auprc(y, fit(rng.standard_normal((n, 4)), y, groups, seed)))
        for _ in range(n_repeats)
    ])
    prev = float(y.mean())
    med = float(np.median(noise_auprc))
    fails = med > prev + margin

    planted = None
    X, cols, src = _covariates(bundle)
    if X is not None and X.shape[1] >= 1:
        Xn = np.c_[X, rng.standard_normal(n)]
        ranks, signs = [], []
        for te in group_folds(groups, k=5, seed=seed):
            tr = np.setdiff1d(np.arange(n), te)
            mu, sd = Xn[tr].mean(0), Xn[tr].std(0)
            sd = np.where(sd == 0, 1.0, sd)
            beta = logistic_fit((Xn[tr] - mu) / sd, y[tr])[1:]
            mag = np.abs(beta)
            ranks.append(float((mag < mag[-1]).mean()))     # share of real cols it beats
            signs.append(float(np.sign(beta[-1])))
        planted = {
            "mean_rank_percentile": float(np.mean(ranks)),
            "sign_stable_across_folds": bool(len(set(signs)) == 1),
            "n_real_columns": len(cols),
        }
        if planted["mean_rank_percentile"] > 0.75 and planted["sign_stable_across_folds"]:
            fails = True

    return _res(
        name, "FAIL" if fails else "PASS",
        f"noise-only OOF AUPRC median {med:.4f} vs majority baseline {prev:.4f} "
        f"(margin {margin})" + (
            f"; planted noise column beats {planted['mean_rank_percentile']:.0%} of real "
            f"columns, sign-stable={planted['sign_stable_across_folds']}" if planted else ""),
        design_matrix=src, noise_median_auprc=med, prevalence=prev,
        noise_max_auprc=float(noise_auprc.max()), n_repeats=int(n_repeats),
        planted_noise_column=planted,
    )


def check_no_post_treatment(bundle) -> dict:
    """Control 3: assert every declared feature column is pre-treatment and un-banned."""
    name = CHECKS[2]
    man = bundle.get("manifest")
    if man is None:
        return _res(name, "BLOCKED", "results/feature_manifest.json is absent",
                    needs=bundle.get("manifest_path"),
                    note="a missing manifest is never a pass: unlabelled columns are unaudited")
    entries = man.get("cols", [])
    post = [c.get("name") for c in entries if not c.get("pre_treatment", False)]
    missing_flag = [c.get("name") for c in entries if "pre_treatment" not in c]
    banned = sorted({
        c.get("name") for c in entries
        for frag in BANNED_FEATURE_FRAGMENTS
        if frag in str(c.get("name", ""))
    })
    npz_cols = bundle.get("cols")
    align = None
    if npz_cols is not None:
        man_names = [c.get("name") for c in entries]
        align = {
            "manifest_len": len(man_names),
            "npz_len": len(npz_cols),
            "same_order": man_names == list(npz_cols),
        }
    bad = bool(post or banned or missing_flag or (align and not align["same_order"]))
    return _res(
        name, "FAIL" if bad else "PASS",
        f"{len(entries)} declared columns; {len(post)} post-treatment, {len(banned)} banned, "
        f"{len(missing_flag)} missing the pre_treatment flag"
        + ("" if align is None else
           f"; manifest/npz aligned={align['same_order']}"),
        n_columns=len(entries), post_treatment_columns=post, banned_columns=banned,
        columns_missing_flag=missing_flag, alignment=align,
        rule=("46.5% of the corpus's tokens are model-generated; any column touching "
              "input[first_user_msg+1:] is a collider, not a feature"),
    )


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference between two samples, pooled-SD convention."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled_sd = float(np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))))
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0


def check_smd(bundle, min_arm_n: int = 5, strict: bool = False, top: int = 10) -> dict:
    """Control 4: |SMD| on pre-treatment covariates for every arm pair inside every stratum."""
    name = CHECKS[3]
    X, cols, src = _covariates(bundle)
    arms, strata = bundle.get("arm"), bundle.get("stratum")
    if X is None or arms is None:
        return _res(name, "BLOCKED", "no covariates and/or arm labels available",
                    needs="results/recon.jsonl (+ features.npz for the full covariate set)")
    if strata is None:
        strata = np.array(["_all"] * len(arms), dtype=object)

    rows, n_cmp, n_flag = [], 0, 0
    for st in sorted({s for s in strata if s}):
        m = np.array([s == st for s in strata])
        present = [a for a in sorted(set(arms[m])) if (arms[m] == a).sum() >= min_arm_n]
        for i, a1 in enumerate(present):
            for a2 in present[i + 1:]:
                a_rows, b_rows = m & (arms == a1), m & (arms == a2)
                for j, col in enumerate(cols):
                    v = smd(X[a_rows, j], X[b_rows, j])
                    if np.isnan(v):
                        continue
                    n_cmp += 1
                    if abs(v) >= SMD_FLAG:
                        n_flag += 1
                        rows.append({"stratum": st, "arm_a": a1, "arm_b": a2, "column": col,
                                     "smd": v, "n_a": int(a_rows.sum()), "n_b": int(b_rows.sum())})
    rows.sort(key=lambda r: -abs(r["smd"]))
    if n_cmp == 0:
        verdict, head = "BLOCKED", "no stratum contained two arms with enough rows to compare"
    elif n_flag == 0:
        verdict, head = "PASS", f"0 of {n_cmp} arm-pair x covariate comparisons reach |SMD| 0.1"
    else:
        verdict = "FAIL" if strict else "WARN"
        head = (f"{n_flag} of {n_cmp} comparisons reach |SMD| >= {SMD_FLAG} "
                f"(worst {rows[0]['smd']:+.2f}) — imbalanced, so these contrasts are BOUNDED, "
                f"not adjustable")
    return _res(
        name, verdict, head, covariates=src, n_comparisons=n_cmp, n_flagged=n_flag,
        flagged_share=(n_flag / n_cmp if n_cmp else None),
        worst=rows[:top], threshold=SMD_FLAG, min_arm_n=min_arm_n, strict=strict,
        note=("this is observational data with no randomisation: imbalance is the "
              "expected finding, and its consequence is that no arm contrast may be "
              "reported as a point effect"),
    )


def check_weight_concentration(bundle, top_k: int = 5,
                               fail_share: float = TOP5_FAIL_SHARE) -> dict:
    """Control 5: how much of sum(w) rides on the top 5 lines (spend weights are the estimand)."""
    name = CHECKS[4]
    w = bundle.get("w")
    if w is None:
        return _res(name, "BLOCKED", "no weights available", needs="results/recon.jsonl")
    order = np.sort(np.asarray(w, dtype=np.float64))[::-1]
    total = float(order.sum())
    shares = {int(k): (float(order[:k].sum()) / total if total else None)
              for k in (1, top_k, 10, 50, 100, 500) if k <= order.size}
    top = shares.get(top_k)
    ess = float(total ** 2 / np.sum(order ** 2)) if total else 0.0
    bad = top is not None and top >= fail_share

    # Cross-check against the published concentration figures. This is free here and
    # turns control 5 into a regression test on results/recon.jsonl itself: if the
    # reconstruction drifts, the gate goes red before any chart is drawn.
    crosscheck = None
    if order.size == 1000:
        expected = {10: 17.7, 50: 40.5, 100: 54.7, 500: 90.2}
        actual = {k: round(100 * shares[k], 1) for k in expected}
        crosscheck = {"source": "docs/CONTRACTS.md", "expected_pct": expected,
                      "actual_pct": actual, "agrees": actual == expected}
        if not crosscheck["agrees"]:
            bad = True

    return _res(
        name, "FAIL" if bad else "PASS",
        f"top-{top_k} lines carry {top:.1%} of sum(w) (fail at {fail_share:.0%}); "
        f"top-10 {shares.get(10, float('nan')):.1%}, Kish ESS {ess:.1f} of {order.size} lines",
        weights="gross_tok (est. tokens) — the spend basis every cost claim is weighted by",
        n=int(order.size), top_k_share=shares, kish_ess=ess, fail_share=fail_share,
        contracts_crosscheck=crosscheck,
        token_basis="ESTIMATED (the export has no usage field)",
    )


def check_growth_ratio_oos(bundle, min_n: int = 30, n_splits: int = 400,
                           train_frac: float = 0.70, seed: int = 0) -> dict:
    """Control 6: fit g_a on 70% of an arm's rows, score the total on the held-out 30%."""
    name = CHECKS[5]
    pre, est, arms = bundle.get("pre"), bundle.get("est"), bundle.get("arm")
    if pre is None or est is None or arms is None:
        return _res(name, "BLOCKED", "no pre/est token vectors available",
                    needs="results/recon.jsonl")
    rng = np.random.default_rng(seed)
    per_arm, worst = {}, 0.0
    for a in sorted(set(arms)):
        m = arms == a
        n = int(m.sum())
        if n < min_n:
            continue
        p, e = pre[m], est[m]
        n_tr = max(1, int(round(train_frac * n)))
        errs = np.empty(n_splits)
        for s in range(n_splits):
            perm = rng.permutation(n)
            tr, te = perm[:n_tr], perm[n_tr:]
            g = e[tr].sum() / p[tr].sum()
            errs[s] = (p[te].sum() * g - e[te].sum()) / e[te].sum()
        abs_err = np.abs(errs)
        per_arm[a] = {
            "n": n,
            "g_full": float(e.sum() / p.sum()),
            "median_abs_rel_err": float(np.median(abs_err)),
            "mean_abs_rel_err": float(abs_err.mean()),
            "p90_abs_rel_err": float(np.quantile(abs_err, 0.90)),
            "max_abs_rel_err": float(abs_err.max()),
            "median_signed_rel_err": float(np.median(errs)),
        }
        worst = max(worst, per_arm[a]["median_abs_rel_err"])
    if not per_arm:
        return _res(name, "BLOCKED", f"no arm has >= {min_n} rows", min_n=min_n)
    verdict = "FAIL" if worst > GROWTH_FAIL else ("WARN" if worst > GROWTH_WARN else "PASS")
    return _res(
        name, verdict,
        f"held-out total from g_a carries a median |relative error| of "
        f"{min(v['median_abs_rel_err'] for v in per_arm.values()):.3f}-{worst:.3f} "
        f"across {len(per_arm)} arms (warn > {GROWTH_WARN}, fail > {GROWTH_FAIL})",
        per_arm=per_arm, n_splits=int(n_splits), train_frac=train_frac, seed=int(seed),
        not_a_check=("sum(pre_i * g_a) == sum(est_i) in sample is an ALGEBRAIC IDENTITY "
                     "and is deliberately not used: it returns 0.000 error on any input"),
        consequence=("any counterfactual total built by rescaling pre-treatment tokens by "
                     "g_a inherits this error; quote it as a band, not a figure"),
    )


# ================================================================ run_all ====

def run_all(bundle=None, results_dir=None, n_permutations: int = 50, seed: int = 0,
            fit_fn=None, strict_smd: bool = False, verbose: bool = True) -> dict:
    """Run all six controls, print the banner, and return PASS/FAIL for the whole gate."""
    if bundle is None:
        bundle = load_bundle(results_dir)
    results = [
        check_permutation_null(bundle, n_permutations=n_permutations, seed=seed, fit_fn=fit_fn),
        check_random_feature(bundle, seed=seed, fit_fn=fit_fn),
        check_no_post_treatment(bundle),
        check_smd(bundle, strict=strict_smd),
        check_weight_concentration(bundle),
        check_growth_ratio_oos(bundle, seed=seed),
    ]
    counts = {v: sum(1 for r in results if r["verdict"] == v)
              for v in ("PASS", "WARN", "FAIL", "BLOCKED")}
    verdict = "FAIL" if counts["FAIL"] else ("BLOCKED" if counts["BLOCKED"] else "PASS")
    report = {"verdict": verdict, "counts": counts, "checks": results,
              "results_dir": bundle.get("results_dir")}
    if verbose:
        print_banner(report)
    return report


def print_banner(report: dict) -> None:
    """Print the gate banner: one line per control, then the verdict in a box."""
    print("=" * 78)
    print("ROUTER PUBLICATION GATE — six negative controls")
    print("=" * 78)
    for r in report["checks"]:
        print(f"[{r['verdict']:<7}] {r['check']:<24} {r['headline']}")
    c = report["counts"]
    verdict = report["verdict"]
    msg = {
        "PASS": "GATE GREEN — the controls held; downstream claims are licensed to proceed.",
        "FAIL": "GATE RED — a control failed. NOTHING downstream stands until it is fixed.",
        "BLOCKED": "GATE BLOCKED — a prerequisite artifact is missing; the gate CANNOT certify.",
    }[verdict]
    print("-" * 78)
    print(f"  {c['PASS']} pass / {c['WARN']} warn / {c['FAIL']} fail / {c['BLOCKED']} blocked")
    print(f"  {msg}")
    if c["WARN"]:
        print("  A WARN is not a licence to ignore it: it names a claim that must be")
        print("  reported as bounded rather than as a point estimate.")
    print("=" * 78)


# ============================================================== self-test ====

def synthetic_bundle(poison: bool = False, n: int = 600, seed: int = 0) -> dict:
    """A fixture with a known answer: clean, or poisoned in five specific, detectable ways."""
    rng = np.random.default_rng(seed)
    d = 8
    X = rng.standard_normal((n, d))
    logit = -1.2 + 0.9 * X[:, 0] - 0.7 * X[:, 1]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float64)
    arms = np.array(["arm_a"] * (n // 2) + ["arm_b"] * (n - n // 2))
    stratum = np.array([f"job_{i % 12}" for i in range(n)], dtype=object)
    cols = [f"f{i}" for i in range(d)]
    pre = rng.lognormal(9.0, 0.5, n)
    est = pre * np.where(arms == "arm_a", 16.0, 20.0) * rng.lognormal(0.0, 0.20, n)
    w = est.copy()
    manifest = {"cols": [{"name": c, "pre_treatment": True, "block": "A"} for c in cols]}

    if poison:
        # (3) a post-treatment column is declared, and one carries a banned name
        manifest["cols"][2] = {"name": "n_turns", "pre_treatment": False, "block": "B"}
        # (4) arm_b is systematically bigger inside every stratum
        X[arms == "arm_b", 0] += 1.5
        # (5) one line carries most of the spend
        w = w.copy()
        w[0] = w.sum() * 1.5
        # (6) arm_b's growth ratio is wildly unstable row to row
        est = est.copy()
        est[arms == "arm_b"] *= rng.lognormal(0.0, 3.0, int((arms == "arm_b").sum()))

    return {
        "results_dir": "<synthetic>", "recon": None, "labels": None, "jobkey": None,
        "routes": None, "routes_conforming": False,
        "X": X, "cols": cols, "idx": np.arange(n), "manifest": manifest,
        "manifest_path": "<synthetic>", "features_path": "<synthetic>",
        "y": y, "arm": arms, "stratum": stratum, "w": w, "pre": pre, "est": est,
        "fallback_cov": X, "fallback_cov_cols": cols,
    }


def leaky_fit_fn(X, y, groups, seed: int = 0) -> np.ndarray:
    """A pipeline that selects features against the FULL-DATA labels before cross-validating."""
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal((X.shape[0], 300))
    corr = np.array([abs(np.corrcoef(noise[:, j], y)[0, 1]) for j in range(noise.shape[1])])
    keep = np.argsort(-corr)[:5]                     # <- the leak: chosen using all of y
    return oof_scores(noise[:, keep], y, groups=groups, seed=seed)


def selftest(verbose: bool = True) -> int:
    """Run the gate on a clean and a poisoned fixture and assert it distinguishes them."""
    print("router.controls SELFTEST — a gate that cannot fail certifies nothing.\n")

    print("--- CLEAN fixture (expect: no FAIL) " + "-" * 40)
    clean = run_all(synthetic_bundle(poison=False), n_permutations=20, verbose=verbose)
    print("\n--- POISONED fixture (expect: FAIL on 1,2,3,5,6; WARN on 4) " + "-" * 16)
    dirty_bundle = synthetic_bundle(poison=True)
    dirty = run_all(dirty_bundle, n_permutations=20, fit_fn=leaky_fit_fn, verbose=verbose)

    cv = {r["check"]: r["verdict"] for r in clean["checks"]}
    dv = {r["check"]: r["verdict"] for r in dirty["checks"]}
    checks = [
        ("clean fixture: gate is not RED", "PASS", clean["verdict"]),
        ("clean 1 permutation-null", "PASS", cv[CHECKS[0]]),
        ("clean 2 random-feature", "PASS", cv[CHECKS[1]]),
        ("clean 3 no-post-treatment", "PASS", cv[CHECKS[2]]),
        ("clean 5 weight-concentration", "PASS", cv[CHECKS[4]]),
        ("clean 6 growth-ratio-oos", "PASS", cv[CHECKS[5]]),
        ("poisoned fixture: gate is RED", "FAIL", dirty["verdict"]),
        ("poisoned 1 catches label-driven feature selection", "FAIL", dv[CHECKS[0]]),
        ("poisoned 2 catches noise scoring above chance", "FAIL", dv[CHECKS[1]]),
        ("poisoned 3 catches the post-treatment column", "FAIL", dv[CHECKS[2]]),
        ("poisoned 4 flags the arm imbalance", "WARN", dv[CHECKS[3]]),
        ("poisoned 5 catches the dominant line", "FAIL", dv[CHECKS[4]]),
        ("poisoned 6 catches the unstable growth ratio", "FAIL", dv[CHECKS[5]]),
        ("poisoned 4 becomes FAIL under strict_smd", "FAIL",
         check_smd(dirty_bundle, strict=True)["verdict"]),
        ("check 6 rejects the tautology: in-sample identity error is exactly 0",
         0.0, round(_tautology_error(dirty_bundle), 12)),
    ]
    ok = True
    print("\nacceptance — the gate distinguishes clean from poisoned:")
    for label, expected, actual in checks:
        good = expected == actual
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}\n"
              f"         expected={expected!r}  actual={actual!r}")
    print(f"\nRESULT: {'all checks pass' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


def _tautology_error(bundle) -> float:
    """Demonstrate the banned in-sample identity: sum(pre*g_a) - sum(est) is exactly 0."""
    pre, est, arms = bundle["pre"], bundle["est"], bundle["arm"]
    err = 0.0
    for a in sorted(set(arms)):
        m = arms == a
        g = est[m].sum() / pre[m].sum()
        err += abs((pre[m] * g).sum() - est[m].sum()) / est[m].sum()
    return float(err)


def main(argv=None) -> int:
    """Run the gate (or --selftest); exit 0 green, 1 on FAIL, 2 when a prerequisite is missing."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    results_dir = next((a for a in argv if not a.startswith("-")), None)
    print("router.controls — publication gate. Token counts are ESTIMATES (the export")
    print("has no usage field); friction is bounded, never point-estimated.\n")
    report = run_all(results_dir=results_dir, strict_smd="--strict-smd" in argv)
    if report["verdict"] == "FAIL":
        return 1
    if report["verdict"] == "BLOCKED":
        print("\nRun `python -m router.controls --selftest` to verify the gate itself,")
        print("then rebuild the missing artifacts and re-run.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
