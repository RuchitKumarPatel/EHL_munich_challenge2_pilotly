#!/usr/bin/env python3
"""Pre-treatment friction predictor: ridge logistic IRLS + isotonic calibration, numpy only.

WHAT THIS COMPUTES
    p_fric(x) -- the probability that a trajectory ends up frictional
    (results/labels.jsonl, y_fric, 259/1000 positives) -- from PRE-TREATMENT
    features only, i.e. from items[:first_user_msg+1] and the tools block.
    Everything after the first user message is 46.5% of the token volume and is
    model-generated: conditioning on it is conditioning on a collider, so no
    column here may come from there.

    The fit is a ridge (L2) logistic regression solved by IRLS / Newton, with
    PER-ARM INTERCEPTS and COMMON SLOPES ACROSS ARMS. That is a power decision,
    not a modelling preference: the minimum detectable friction difference on
    the best-powered arm pair in this export is 11.4pp. Arm x feature
    interactions would fit one slope per arm on arms with 331, 281, 113, 112,
    71, 69, 20, 2 and 1 runs -- manufacturing arm-specific structure the data
    cannot support, and handing the policy a per-arm signal that is mostly
    noise. Arms get a level (an intercept) because their base rates can differ;
    they do not get a shape.

    Raw scores are then CALIBRATED BEFORE ANY THRESHOLDING with pool-adjacent-
    violators isotonic regression, fitted on an inner split of the training
    fold. router.policy thresholds p_fric against tau; thresholding an
    uncalibrated score makes tau meaningless and makes the conformal guarantee
    in router.policy.conformal_tau incoherent.

WHAT IT WRITES
    Nothing by default. `fit_predict_oof` returns out-of-fold calibrated
    probabilities in memory; `router.policy` imports and calls it. `main()`
    prints the metric table and writes nothing.

FOLDS
    K-fold blocked on the JOB KEY from results/jobkey.jsonl (the literal
    "Cron path:" string, 354 distinct paths / 152 fully-literal jobs / 374
    runs, largest job n=134). Runs of one cron job are near-duplicates; random
    folds would put them on both sides of the split and inflate every metric.
    Runs with no literal path get a singleton group and may be split freely.

REPORTED, ALWAYS AGAINST THREE BASELINES
    AUPRC (the headline: 25.9% positives, so AUC alone flatters a useless
    model), Brier, ECE with reliability bins, plus AUC for reference. The three
    baselines printed alongside are: (1) MAJORITY -- the constant base rate;
    (2) PERMUTATION NULL -- the identical pipeline on shuffled labels, which is
    the honest zero for AUPRC because a ranker that ties everything still
    scores the base rate; (3) SINGLE BEST COLUMN -- the best univariate model
    under the same folds, so "the multivariate fit earns its complexity" has to
    be demonstrated rather than assumed.

FEATURES
    Prefers results/features.npz + results/feature_manifest.json (owned by
    router.features). If that artifact is absent this module falls back to its
    own small pre-treatment design matrix built from the export, and says so
    loudly on every run. The fallback exists so the predictor and the policy
    are runnable standalone; it is not a substitute for the real feature set.

ACCEPTANCE (printed by `python -m router.model`)
    n = 1000 out-of-fold predictions, idx ascending 0..999, all finite, all in
    [0,1]; base rate y_fric = 259/1000 = 0.259; folds job-blocked so no cron
    path straddles a fold (0 straddling groups); the largest job (n=134) lands
    wholly inside one fold; all token counts feeding the features are
    ESTIMATES (len(json.dumps(x))//4 -- the export has no `usage` field).
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from router.io import DEFAULT_EXPORT_DIR, first_user_index, iter_lines, tok

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FEATURES_PATH = os.path.join(RESULTS_DIR, "features.npz")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "feature_manifest.json")
LABELS_PATH = os.path.join(RESULTS_DIR, "labels.jsonl")
JOBKEY_PATH = os.path.join(RESULTS_DIR, "jobkey.jsonl")
RECON_PATH = os.path.join(RESULTS_DIR, "recon.jsonl")

#: Default fit hyper-parameters. lam is the L2 penalty on the SLOPES only.
DEFAULT_LAM = 4.0
DEFAULT_K = 5
DEFAULT_MAX_ITER = 60

#: Measured, from results/labels.jsonl.
EXPECTED = {"n": 1000, "y_fric_positives": 259, "largest_job_n": 134}


# --------------------------------------------------------------------------
# 1. the solver
# --------------------------------------------------------------------------

def ridge_logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = DEFAULT_LAM,
    max_iter: int = DEFAULT_MAX_ITER,
    penalize: np.ndarray | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """Fit L2-penalised logistic regression by IRLS/Newton; returns the weight vector.

    `penalize` is a 0/1 mask over columns saying which weights the L2 term
    touches. `design_matrix` builds X as [per-arm intercepts | common slopes]
    and returns a mask that penalises the slopes and leaves the per-arm
    intercepts free -- per-arm levels, common shape (see the module docstring
    for why arm x feature interactions are refused).

    Objective  -loglik(w) + lam * sum_j penalize_j * w_j^2
    Newton step  w <- w + solve(X'WX + 2*lam*P, X'(y-p) - 2*lam*P w)
    with step halving on any step that does not decrease the objective.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X has {n} rows but y has {y.shape[0]}")
    P = np.ones(d) if penalize is None else np.asarray(penalize, dtype=np.float64).ravel()
    if P.shape[0] != d:
        raise ValueError(f"penalize has {P.shape[0]} entries for {d} columns")

    def objective(w: np.ndarray) -> float:
        z = X @ w
        # log(1+exp(z)) computed stably
        nll = float(np.sum(np.logaddexp(0.0, z) - y * z))
        return nll + lam * float(np.sum(P * w * w))

    w = np.zeros(d)
    obj = objective(w)
    for _ in range(int(max_iter)):
        z = X @ w
        p = _sigmoid(z)
        wgt = np.clip(p * (1.0 - p), 1e-10, None)
        grad = X.T @ (y - p) - 2.0 * lam * P * w
        H = (X * wgt[:, None]).T @ X + np.diag(2.0 * lam * P)
        # jitter keeps H invertible when an arm dummy is empty in this fold
        H[np.diag_indices_from(H)] += 1e-8
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        for _ in range(30):
            cand = w + t * step
            cand_obj = objective(cand)
            if cand_obj <= obj:
                break
            t *= 0.5
        else:
            break
        if abs(obj - cand_obj) < tol * (1.0 + abs(obj)):
            w, obj = cand, cand_obj
            break
        w, obj = cand, cand_obj
    return w


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def predict_score(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Raw (uncalibrated) logistic probability X @ w pushed through the sigmoid."""
    return _sigmoid(np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64))


def design_matrix(Xs: np.ndarray, arm_codes: np.ndarray, n_arms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build [per-arm intercepts | common slopes] and the L2 mask that spares the intercepts."""
    Xs = np.asarray(Xs, dtype=np.float64)
    n = Xs.shape[0]
    D = np.zeros((n, n_arms))
    D[np.arange(n), np.asarray(arm_codes, dtype=int)] = 1.0
    X = np.hstack([D, Xs])
    penalize = np.concatenate([np.zeros(n_arms), np.ones(Xs.shape[1])])
    return X, penalize


# --------------------------------------------------------------------------
# 2. calibration
# --------------------------------------------------------------------------

class IsotonicCalibrator:
    """Step-function map from raw score to calibrated probability (from PAV blocks)."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

    def __call__(self, scores) -> np.ndarray:
        """Calibrate raw scores; constant-extrapolates outside the calibration range."""
        s = np.atleast_1d(np.asarray(scores, dtype=np.float64))
        if self.x.size == 0:
            return np.full(s.shape, float(self.y.mean()) if self.y.size else 0.0)
        out = np.interp(s, self.x, self.y, left=self.y[0], right=self.y[-1])
        return np.clip(out, 0.0, 1.0)


def isotonic_pav(scores: np.ndarray, y: np.ndarray,
                 prior_weight: float = 2.0, prior: float | None = None) -> IsotonicCalibrator:
    """Fit a monotone score -> probability calibrator by pool-adjacent-violators.

    `prior_weight` adds that many pseudo-observations at the base rate to each
    pooled block AFTER the PAV run. Without it, a block whose calibration rows
    happen to contain zero positives is assigned probability EXACTLY 0, and a
    downstream threshold or conformal statement over such a block is
    meaningless -- on this corpus the unsmoothed fit put 134 rows at p = 0.0
    while 36 of them (26.9%) were in fact frictional. Monotonicity is restored
    with a cumulative max, because shrinking unequal blocks toward a common
    base rate can otherwise invert two adjacent ones.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    if s.size == 0:
        return IsotonicCalibrator(np.array([]), np.array([]))
    base = float(t.mean()) if prior is None else float(prior)
    order = np.argsort(s, kind="mergesort")
    s, t = s[order], t[order]

    # PAV over (value, weight) blocks
    vals: List[float] = []
    wts: List[float] = []
    ends: List[int] = []
    for i in range(s.size):
        vals.append(float(t[i]))
        wts.append(1.0)
        ends.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            e2 = ends.pop()
            v1, w1 = vals.pop(), wts.pop()
            ends.pop()
            w = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / w)
            wts.append(w)
            ends.append(e2)

    # smooth each pooled block toward the base rate, then re-impose monotonicity
    smoothed = [
        (v * w + prior_weight * base) / (w + prior_weight) if prior_weight > 0 else v
        for v, w in zip(vals, wts)
    ]
    smoothed = list(np.maximum.accumulate(np.asarray(smoothed, dtype=np.float64)))

    # expand blocks back to per-sample fitted values, then keep one knot per
    # distinct score (last wins, which is the right-continuous step function)
    fitted = np.empty(s.size)
    start = 0
    for v, w in zip(smoothed, wts):
        stop = start + int(round(w))
        fitted[start:stop] = v
        start = stop

    xs, ys = [], []
    for i in range(s.size):
        if i + 1 == s.size or s[i + 1] != s[i]:
            xs.append(s[i])
            ys.append(fitted[i])
    return IsotonicCalibrator(np.array(xs), np.array(ys))


# --------------------------------------------------------------------------
# 3. folds
# --------------------------------------------------------------------------

def grouped_kfold(groups: Sequence, k: int = DEFAULT_K, seed: int = 0) -> List[np.ndarray]:
    """Assign rows to k folds by GROUP label so one job never straddles a fold.

    Greedy largest-group-first bin packing onto the currently smallest fold:
    with one job at n=134 out of 1000, random group assignment routinely leaves
    a fold with almost no positives.
    """
    groups = list(groups)
    n = len(groups)
    if k < 2:
        raise ValueError("k must be at least 2")
    members: Dict[object, List[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    rng = np.random.default_rng(seed)
    keys = list(members)
    order = sorted(
        range(len(keys)),
        key=lambda j: (-len(members[keys[j]]), float(rng.random())),
    )
    fold_of = np.empty(n, dtype=int)
    sizes = np.zeros(k, dtype=int)
    for j in order:
        f = int(np.argmin(sizes))
        idxs = members[keys[j]]
        fold_of[idxs] = f
        sizes[f] += len(idxs)
    return [np.flatnonzero(fold_of == f) for f in range(k)]


# --------------------------------------------------------------------------
# 4. out-of-fold fit
# --------------------------------------------------------------------------

def _standardize(train: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score `train` and apply the TRAIN mean/std to `other` (no test statistics leak)."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (train - mu) / sd, (other - mu) / sd


def fit_predict_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    arms: Sequence[str] | None = None,
    lam: float = DEFAULT_LAM,
    k: int = DEFAULT_K,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 0,
    calibrate: bool = True,
) -> Dict[str, object]:
    """Out-of-fold calibrated p_fric plus per-fold weights and slope sign stability."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    arm_list = ["_all"] * n if arms is None else list(arms)
    arm_names = sorted(set(arm_list))
    code = {a: i for i, a in enumerate(arm_names)}
    arm_codes = np.array([code[a] for a in arm_list], dtype=int)

    folds = grouped_kfold(groups, k=k, seed=seed)
    oof_raw = np.full(n, np.nan)
    oof_cal = np.full(n, np.nan)
    slopes = np.zeros((k, d))

    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test, assume_unique=False)
        Xtr, Xte = _standardize(X[train], X[test])
        Dtr, pen = design_matrix(Xtr, arm_codes[train], len(arm_names))
        Dte, _ = design_matrix(Xte, arm_codes[test], len(arm_names))
        w = ridge_logistic_irls(Dtr, y[train], lam=lam, max_iter=max_iter, penalize=pen)
        slopes[f] = w[len(arm_names):]
        oof_raw[test] = predict_score(Dte, w)

        if not calibrate:
            oof_cal[test] = oof_raw[test]
            continue

        # inner grouped split of the TRAINING fold only -> honest calibration
        inner_groups = [groups[i] for i in train]
        inner = grouped_kfold(inner_groups, k=min(4, max(2, k - 1)), seed=seed + 101)
        cal_scores = np.full(train.size, np.nan)
        for itest in inner:
            itr = np.setdiff1d(np.arange(train.size), itest)
            A, B = _standardize(X[train][itr], X[train][itest])
            DA, penA = design_matrix(A, arm_codes[train][itr], len(arm_names))
            DB, _ = design_matrix(B, arm_codes[train][itest], len(arm_names))
            wi = ridge_logistic_irls(DA, y[train][itr], lam=lam, max_iter=max_iter, penalize=penA)
            cal_scores[itest] = predict_score(DB, wi)
        cal = isotonic_pav(cal_scores, y[train])
        oof_cal[test] = cal(oof_raw[test])

    sign_stable = [
        bool(np.all(np.sign(slopes[:, j]) == np.sign(slopes[0, j])) and abs(slopes[0, j]) > 1e-8)
        for j in range(d)
    ]
    return {
        "p": oof_cal,
        "raw": oof_raw,
        "folds": folds,
        "slopes": slopes,
        "sign_stable": np.array(sign_stable),
        "arm_names": arm_names,
        "lam": lam,
        "k": k,
    }


# --------------------------------------------------------------------------
# 5. metrics
# --------------------------------------------------------------------------

def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC AUC with tie correction; 0.5 when either class is empty."""
    y = np.asarray(y).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=np.float64)
    sp = p[order]
    i = 0
    while i < sp.size:
        j = i
        while j + 1 < sp.size and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (area under precision-recall, step interpolation)."""
    y = np.asarray(y).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    n1 = int((y == 1).sum())
    if n1 == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    ap = 0.0
    tp = 0
    seen = 0
    i = 0
    while i < ys.size:
        j = i
        while j + 1 < ys.size and ps[j + 1] == ps[i]:
            j += 1
        block = ys[i:j + 1]
        tp += int(block.sum())
        seen = j + 1
        # every positive in a tied block gets the same precision (tp/seen)
        ap += (tp / seen) * float(block.sum())
        i = j + 1
    return float(ap / n1)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between the predicted probability and the 0/1 label."""
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> List[dict]:
    """Equal-width reliability table: one row per bin with n, mean predicted, observed rate."""
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(m.sum())
        rows.append({
            "lo": float(lo), "hi": float(hi), "n": cnt,
            "mean_pred": float(p[m].mean()) if cnt else float("nan"),
            "obs_rate": float(y[m].mean()) if cnt else float("nan"),
        })
    return rows


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: n-weighted mean |observed - predicted| over the bins."""
    rows = reliability_bins(y, p, n_bins)
    n = sum(r["n"] for r in rows)
    if n == 0:
        return float("nan")
    return float(sum(r["n"] * abs(r["obs_rate"] - r["mean_pred"]) for r in rows if r["n"]) / n)


def score_block(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """All four headline metrics for one prediction vector."""
    return {"auprc": auprc(y, p), "auc": roc_auc(y, p),
            "brier": brier(y, p), "ece": ece(y, p, n_bins)}


# --------------------------------------------------------------------------
# 6. data loading (features.npz preferred, fallback design matrix otherwise)
# --------------------------------------------------------------------------

def load_labels(path: str = LABELS_PATH) -> Dict[int, dict]:
    """Read results/labels.jsonl into {idx: record}."""
    with open(path, "r", encoding="utf-8") as fh:
        return {r["idx"]: r for r in (json.loads(l) for l in fh if l.strip())}


def load_jobkeys(path: str = JOBKEY_PATH) -> Dict[int, dict]:
    """Read results/jobkey.jsonl into {idx: record}."""
    with open(path, "r", encoding="utf-8") as fh:
        return {r["idx"]: r for r in (json.loads(l) for l in fh if l.strip())}


def load_recon(path: str = RECON_PATH) -> Dict[int, dict]:
    """Read results/recon.jsonl into {idx: record}."""
    with open(path, "r", encoding="utf-8") as fh:
        return {r["idx"]: r for r in (json.loads(l) for l in fh if l.strip())}


def group_labels(idxs: Sequence[int], jobkeys: Dict[int, dict] | None = None) -> List[str]:
    """Fold group per idx: the literal cron path, or a per-row singleton when there is none."""
    jobkeys = load_jobkeys() if jobkeys is None else jobkeys
    out = []
    for i in idxs:
        rec = jobkeys.get(int(i))
        if rec and rec.get("literal") and rec.get("cron_path"):
            out.append("job:" + rec["cron_path"])
        else:
            out.append("solo:%d" % int(i))
    return out


_URL_RE = re.compile(r"https?://")
_PII_RE = re.compile(r"PII_[A-Z]+_\d+")
_ENTITY_RE = re.compile(r"<[A-Z_]+_\d+>")

#: Columns of the FALLBACK design matrix. All are computed from
#: items[:first_user_msg+1] only -- system message + first user message.
FALLBACK_COLS = [
    "log_sys_tok", "log_user_tok", "log_pre_tok", "user_n_lines", "user_n_bullets",
    "user_n_fences", "user_n_urls", "user_n_pii", "user_n_entity", "user_n_qmarks",
    "user_digit_ratio", "user_upper_ratio", "sys_n_headers", "img_pre", "has_cron_path",
]


def _pre_text(items: List[dict], lo: int, hi: int) -> str:
    """Concatenate the input_text parts of items[lo:hi] (pre-treatment slice only)."""
    buf = []
    for it in items[lo:hi]:
        c = it.get("content")
        if isinstance(c, str):
            buf.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    buf.append(part["text"])
    return "\n".join(buf)


def _count_images(items: List[dict], lo: int, hi: int) -> int:
    """Number of input_image parts in items[lo:hi]."""
    n = 0
    for it in items[lo:hi]:
        c = it.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "input_image")
    return n


def fallback_design(export_dir: str = DEFAULT_EXPORT_DIR,
                    jobkeys: Dict[int, dict] | None = None
                    ) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], Dict[int, int]]:
    """Build the standalone PRE-TREATMENT design matrix; returns (X, cols, idx, arms, img_pre).

    Used only when results/features.npz is missing. Every column reads
    items[:first_user_msg+1] and nothing else, so no post-treatment (model-
    generated) text can reach the fit. The tools block is deliberately NOT a
    column: its size is a perfect proxy for the family lane, and the lane is a
    gate in router.policy, not a learned effect.
    """
    jobkeys = load_jobkeys() if jobkeys is None else jobkeys
    rows, idxs, arms = [], [], []
    img_pre: Dict[int, int] = {}
    for idx, req in iter_lines(export_dir):
        items = req["input"]
        fu = first_user_index(items)
        hi = fu + 1
        sys_txt = _pre_text(items, 0, min(1, hi))
        usr_txt = _pre_text(items, min(1, hi), hi)
        sys_tok = sum(tok(it) for it in items[:min(1, hi)])
        usr_tok = sum(tok(it) for it in items[min(1, hi):hi])
        n_img = _count_images(items, 0, hi)
        img_pre[idx] = n_img
        chars = max(1, len(usr_txt))
        jk = jobkeys.get(idx) or {}
        rows.append([
            math.log1p(sys_tok),
            math.log1p(usr_tok),
            math.log1p(sys_tok + usr_tok),
            usr_txt.count("\n"),
            len(re.findall(r"(?m)^\s*[-*]\s", usr_txt)),
            usr_txt.count("```"),
            len(_URL_RE.findall(usr_txt)),
            len(_PII_RE.findall(usr_txt)),
            len(_ENTITY_RE.findall(usr_txt)),
            usr_txt.count("?"),
            sum(c.isdigit() for c in usr_txt) / chars,
            sum(c.isupper() for c in usr_txt) / chars,
            len(re.findall(r"(?m)^#+\s", sys_txt)),
            n_img,
            1.0 if jk.get("literal") else 0.0,
        ])
        idxs.append(idx)
        arms.append(req["model"])
    return (np.asarray(rows, dtype=np.float64), list(FALLBACK_COLS),
            np.asarray(idxs, dtype=np.int64), arms, img_pre)


def load_design(features_path: str = FEATURES_PATH, manifest_path: str = MANIFEST_PATH
                ) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], Dict[int, int], str]:
    """Load the design matrix; returns (X, cols, idx, arms, img_pre, source).

    Prefers results/features.npz (+ feature_manifest.json, whose columns must
    all be pre_treatment). Falls back to `fallback_design` and reports which
    source was used so no caller can mistake one for the other.
    """
    jobkeys = load_jobkeys()
    if os.path.exists(features_path):
        z = np.load(features_path, allow_pickle=False)
        X = np.asarray(z["X"], dtype=np.float64)
        cols = [str(c) for c in z["cols"]]
        idx = np.asarray(z["idx"], dtype=np.int64)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                man = json.load(fh)
            names = [c["name"] for c in man["cols"]]
            if names != cols:
                raise ValueError("feature_manifest.json column order disagrees with features.npz")
            bad = [c["name"] for c in man["cols"] if not c.get("pre_treatment")]
            if bad:
                raise ValueError(f"post-treatment columns in the design matrix: {bad}")
        recon = load_recon()
        arms = [recon[int(i)]["model"] for i in idx]
        _, _, _, _, img_pre = fallback_design(jobkeys=jobkeys)
        return X, cols, idx, arms, img_pre, "results/features.npz"
    X, cols, idx, arms, img_pre = fallback_design(jobkeys=jobkeys)
    return X, cols, idx, arms, img_pre, "FALLBACK (router.model.fallback_design)"


def load_pfric(lam: float = DEFAULT_LAM, k: int = DEFAULT_K, seed: int = 0
               ) -> Dict[str, object]:
    """Fit end to end and return {idx, p_fric, y, arms, groups, cols, sign_stable_cols, ...}."""
    X, cols, idx, arms, img_pre, source = load_design()
    labels = load_labels()
    y = np.array([labels[int(i)]["y_fric"] for i in idx], dtype=np.float64)
    jobkeys = load_jobkeys()
    groups = group_labels(idx, jobkeys)
    fit = fit_predict_oof(X, y, groups, arms=arms, lam=lam, k=k, seed=seed)
    stable = [c for c, s in zip(cols, fit["sign_stable"]) if s]
    return {
        "idx": idx, "y": y, "arms": arms, "groups": groups, "cols": cols,
        "X": X, "p_fric": fit["p"], "raw": fit["raw"], "folds": fit["folds"],
        "slopes": fit["slopes"], "sign_stable_cols": stable,
        "img_pre": img_pre, "source": source, "lam": lam, "k": k, "seed": seed,
    }


# --------------------------------------------------------------------------
# 7. baselines
# --------------------------------------------------------------------------

def baseline_majority(y: np.ndarray, folds: Iterable[np.ndarray]) -> np.ndarray:
    """Constant predictor: every row gets its TRAINING fold's base rate."""
    y = np.asarray(y, dtype=np.float64)
    p = np.empty_like(y)
    n = y.size
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        p[test] = y[train].mean()
    return p


def baseline_permutation(X, y, groups, arms, lam, k, seed, n_rep: int = 10) -> Dict[str, float]:
    """Run the identical OOF pipeline on shuffled labels; returns mean metrics over repeats."""
    rng = np.random.default_rng(seed + 7)
    acc = {"auprc": [], "auc": [], "brier": [], "ece": []}
    for r in range(n_rep):
        yp = np.asarray(y, dtype=np.float64).copy()
        rng.shuffle(yp)
        fit = fit_predict_oof(X, yp, groups, arms=arms, lam=lam, k=k, seed=seed)
        m = score_block(yp, fit["p"])
        for key in acc:
            acc[key].append(m[key])
    return {key: float(np.mean(v)) for key, v in acc.items()}


def baseline_best_column(X, y, groups, arms, cols, lam, k, seed) -> Tuple[str, Dict[str, float]]:
    """Best single column under the same job-blocked folds, scored by OOF AUPRC."""
    best_name, best_m, best_score = None, None, -1.0
    for j, name in enumerate(cols):
        fit = fit_predict_oof(X[:, [j]], y, groups, arms=arms, lam=lam, k=k, seed=seed)
        m = score_block(y, fit["p"])
        if m["auprc"] > best_score:
            best_name, best_m, best_score = name, m, m["auprc"]
    return best_name, best_m


# --------------------------------------------------------------------------
# 8. acceptance
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    """Fit p_fric out-of-fold and print the metric table, the baselines and acceptance."""
    print("router.model -- pre-treatment friction predictor (ridge logistic IRLS + PAV)")
    print("All token counts feeding these features are ESTIMATES (len(json.dumps(x))//4);")
    print("the export has no `usage` field. p_fric is OUT-OF-FOLD everywhere.\n")

    d = load_pfric()
    idx, y, p, cols = d["idx"], d["y"], d["p_fric"], d["cols"]
    print(f"design matrix source: {d['source']}")
    if d["source"].startswith("FALLBACK"):
        print("  !! results/features.npz is ABSENT. This is router.model's own standalone")
        print("     pre-treatment matrix, not the project feature set. Re-run once")
        print("     router.features lands; the metrics below will change.")
    print(f"  shape {d['X'].shape[0]} x {d['X'].shape[1]}  |  arms {len(set(d['arms']))}"
          f"  |  folds k={d['k']} blocked on job key  |  lam={d['lam']}\n")

    base = float(y.mean())
    m = score_block(y, p)
    maj = score_block(y, baseline_majority(y, d["folds"]))
    perm = baseline_permutation(d["X"], y, d["groups"], d["arms"], d["lam"], d["k"], d["seed"])
    bcol, bm = baseline_best_column(d["X"], y, d["groups"], d["arms"], cols, d["lam"], d["k"], d["seed"])

    print("out-of-fold metrics vs three baselines (n=1000, positives=%d, base rate=%.3f)"
          % (int(y.sum()), base))
    hdr = f"  {'':<26}{'AUPRC':>9}{'AUC':>9}{'Brier':>9}{'ECE':>9}"
    print(hdr)
    print(f"  {'ridge logistic + PAV':<26}{m['auprc']:>9.4f}{m['auc']:>9.4f}{m['brier']:>9.4f}{m['ece']:>9.4f}")
    print(f"  {'baseline: majority':<26}{maj['auprc']:>9.4f}{maj['auc']:>9.4f}{maj['brier']:>9.4f}{maj['ece']:>9.4f}")
    print(f"  {'baseline: permutation null':<26}{perm['auprc']:>9.4f}{perm['auc']:>9.4f}{perm['brier']:>9.4f}{perm['ece']:>9.4f}")
    print(f"  {'baseline: best 1 column':<26}{bm['auprc']:>9.4f}{bm['auc']:>9.4f}{bm['brier']:>9.4f}{bm['ece']:>9.4f}"
          f"   ({bcol})")
    print()

    print("reliability bins (calibrated OOF p_fric):")
    print(f"  {'bin':>12}{'n':>7}{'mean pred':>12}{'observed':>11}")
    for r in reliability_bins(y, p):
        if not r["n"]:
            continue
        print(f"  {r['lo']:>5.2f}-{r['hi']:<5.2f}{r['n']:>7}{r['mean_pred']:>12.4f}{r['obs_rate']:>11.4f}")
    print()

    print(f"sign-stable slope columns across all {d['k']} folds "
          f"({len(d['sign_stable_cols'])}/{len(cols)}):")
    print("  " + (", ".join(d["sign_stable_cols"]) if d["sign_stable_cols"] else "(none)"))
    if m["auc"] <= 0.5:
        print("\n  OOF AUC <= 0.5: NO LEARNABLE PRE-TREATMENT SIGNAL. Route on cost")
        print("  structure alone; do not add post-treatment features to rescue this.")
    print()

    # ------------------------------------------------------------- checks
    ok = True

    def check(label, expected, actual):
        nonlocal ok
        good = expected == actual
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label:<52} expected {expected!r:>12}  actual {actual!r:>12}")

    jobkeys = load_jobkeys()
    folds = d["folds"]
    fold_of = {}
    for f, test in enumerate(folds):
        for i in test:
            fold_of[int(i)] = f
    straddle = 0
    sizes = {}
    for g in set(d["groups"]):
        rows = [i for i, gg in enumerate(d["groups"]) if gg == g]
        sizes[g] = len(rows)
        if len({fold_of[i] for i in rows}) > 1:
            straddle += 1
    biggest = max(sizes, key=lambda g: sizes[g])

    print("acceptance:")
    check("n out-of-fold predictions", EXPECTED["n"], int(p.size))
    check("idx ascending 0..999", True, bool(np.array_equal(idx, np.arange(EXPECTED["n"]))))
    check("all p_fric finite", True, bool(np.all(np.isfinite(p))))
    check("all p_fric in [0,1]", True, bool(np.all((p >= 0.0) & (p <= 1.0))))
    check("y_fric positives", EXPECTED["y_fric_positives"], int(y.sum()))
    check("base rate", 0.259, round(base, 3))
    check("job groups straddling a fold", 0, straddle)
    check("largest job size (crons/heartbeat/)", EXPECTED["largest_job_n"], sizes[biggest])
    check("largest job lands in exactly one fold", 1,
          len({fold_of[i] for i, g in enumerate(d["groups"]) if g == biggest}))
    check("every fold has >= 10 positives", True,
          bool(all(y[t].sum() >= 10 for t in folds)))
    check("calibrated Brier <= majority Brier", True, bool(m["brier"] <= maj["brier"] + 1e-9))
    check("AUPRC beats the permutation null", True, bool(m["auprc"] > perm["auprc"]))

    print("\nRESULT: " + ("all checks pass" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
