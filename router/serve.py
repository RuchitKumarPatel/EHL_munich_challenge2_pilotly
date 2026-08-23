#!/usr/bin/env python3
"""The deployable router: a fold ensemble that scores a task nobody logged.

WHAT THIS COMPUTES
    Two things the offline pipeline never needed.

    1  A DEPLOYABLE predictor. router.model fits five grouped folds and keeps only
       the out-of-fold predictions -- enough to evaluate, useless to serve, because
       the standardisation, the weight vector and the isotonic calibrator are thrown
       away at the end of each fold. `build_ensemble()` refits the same five folds
       with the same lam/k/seed and KEEPS those three objects per fold, so a task
       that is not in the export can be scored by every fold predictor.

    2  A SELECTION DISTRIBUTION over arms. router.policy is deterministic given
       (features, arm, tau): one task, one answer. The honest source of spread is
       the two things that were estimated rather than observed -- WHICH predictor
       (the five folds disagree) and WHERE tau falls (split conformal on a finite
       calibration set). `selection_distribution()` draws (fold, tau) pairs, runs
       the real five-gate chain on each draw, and reports the share of draws that
       land on each arm. That share is a statement about OUR uncertainty, not about
       any randomness in the router: the router does not flip coins.

WHY THE INCUMBENT IS MARGINALISED
    Gate 1 reads the family lane off the tools block, and gates 3-4 need an
    INCUMBENT arm -- the router only ever downgrades from something. A typed task
    has no incumbent, so by default we average over the arms actually served in
    that lane, weighted by their logged share (`incumbent_prior()`). Pinning an
    incumbent is the other supported mode, and the console offers both. The prior
    is a description of this export's traffic mix, not a claim about anyone else's.

WHAT IT WRITES
    No artifact. It reads results/features.npz, results/labels.jsonl,
    results/recon.jsonl, results/jobkey.jsonl and results/routes.jsonl, and builds
    everything else in memory. The ensemble is rebuilt on demand (about 12 s) and
    cached for the process lifetime.

THE SYNTHETIC BRIEFING IS SCHEMA, NOT DATA
    `synthesize_request()` builds a request whose shape the feature extractor
    recognises: the `<system>...</system>` block, the `Triggered by:` line, the
    `# === ... ===` header markers and the tool names. Every one of those strings
    is a constant already committed in router/features.py and router/io.py. No text
    from the export is copied into it, and the console never receives trajectory
    text. Sizes (system tokens, tool tokens, context lines) default to the corpus
    MEDIAN for the chosen lane -- a number, not a quote.

    The synthesized request is then passed through the REAL `router.features.extract`,
    not a reimplementation, so the console's feature vector cannot drift from the
    one the 1000 logged rows were scored with.

ACCEPTANCE (printed by `python -m router.serve`)
    the ensemble's out-of-fold p reproduces results/routes.jsonl p_fric on all
    1000 rows to 1e-12 -- the fold artifacts are the same fit, not a lookalike;
    the conformal tau recomputed from those scores is 0.170833 at alpha=0.10;
    replaying the gate chain through `route_features` reproduces routes.jsonl's
    route and changed fields on all 1000 rows;
    every selection distribution sums to 1 and is supported only on arms that
    survive gates 1-4 for some incumbent in the lane;
    two tasks that differ only in their lane never share an arm in their support.

CAVEAT ON EVERY NUMBER
    Token counts are ESTIMATES (tok(x) = len(json.dumps(x)) // 4; the export has
    no `usage` field). Dollar figures are input-side only under an ASSUMED sheet.
    p_fric is PROCESS friction from tool-output exit codes, never answer quality.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from router import features as rfeatures
from router import model as rmodel
from router import policy as rpolicy
from router.io import CLAUDE_TOOLS, GPT_TOOLS
from router.pricing import ALL_SHEETS, rate_in

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
ROUTES_PATH = RESULTS_DIR / "routes.jsonl"

#: Draws taken when the console asks for a selection distribution. Each draw is
#: one (fold, bootstrap tau) pair run through the real gate chain.
DEFAULT_DRAWS = 400

#: Bootstrap replicates behind the tau distribution. Resampled by JOB, not by row.
DEFAULT_TAU_BOOT = 200

#: Acceptance numbers, measured against results/routes.jsonl.
EXPECTED = {
    "n": 1000,
    "tau_alpha_10": 0.170833,
    "changed": 119,
}


# --------------------------------------------------------------------------- #
# 1. the fold ensemble
# --------------------------------------------------------------------------- #
@dataclass
class FoldPredictor:
    """One fold's complete, serving-ready predictor: standardiser, weights, calibrator."""

    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    arm_names: list[str]
    calibrator: object

    def p_fric(self, x: np.ndarray, arm: str) -> float:
        """Calibrated friction probability for one feature row under one incumbent arm.

        An arm this fold never saw has no intercept, which is a question the fold
        cannot answer -- we refuse rather than silently substituting arm 0.
        """
        if arm not in self.arm_names:
            raise KeyError(f"fold has no intercept for arm {arm!r}")
        xs = (np.asarray(x, dtype=np.float64).reshape(1, -1) - self.mean) / self.std
        codes = np.array([self.arm_names.index(arm)], dtype=int)
        D, _ = rmodel.design_matrix(xs, codes, len(self.arm_names))
        raw = rmodel.predict_score(D, self.w)
        return float(np.asarray(self.calibrator(raw)).ravel()[0])


@dataclass
class Ensemble:
    """The five fold predictors plus everything the gate chain needs to run live."""

    folds: list[FoldPredictor]
    cols: list[str]
    idx: np.ndarray
    arms: list[str]
    y: np.ndarray
    groups: list[str]
    oof_p: np.ndarray
    fold_index: list[np.ndarray]
    counts: dict[str, int]
    lam: float
    k: int
    seed: int
    lane_medians: dict[str, dict[str, float]] = field(default_factory=dict)
    neutral_recur: dict[str, float] = field(default_factory=dict)

    # -- tau ---------------------------------------------------------------
    def tau(self, alpha: float = rpolicy.DEFAULT_ALPHA) -> float:
        """The point tau results/routes.jsonl was cut at, recomputed from the stored scores."""
        return rpolicy.conformal_tau(self.oof_p, self.y, alpha)

    def tau_draws(self, alpha: float = rpolicy.DEFAULT_ALPHA,
                  n_boot: int = DEFAULT_TAU_BOOT, seed: int = 0) -> np.ndarray:
        """Bootstrap distribution of the conformal tau, resampled by JOB.

        tau is an order statistic of the calibration positives, so its sampling
        error is exactly the uncertainty the console should show. Resampling rows
        would understate it: one cron job contributes up to 134 correlated rows.
        Draws where the resample certifies no threshold come back as -inf, which
        routes nothing -- they are kept, not filtered, because refusing to route
        IS one of the outcomes.
        """
        rng = np.random.default_rng(seed)
        by_group: dict[str, list[int]] = {}
        for i, g in enumerate(self.groups):
            by_group.setdefault(g, []).append(i)
        keys = list(by_group)
        out = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            pick = rng.integers(0, len(keys), size=len(keys))
            rows = np.concatenate([by_group[keys[j]] for j in pick])
            out[b] = rpolicy.conformal_tau(self.oof_p[rows], self.y[rows], alpha)
        return out

    # -- verification ------------------------------------------------------
    def verify_against_routes(self, path: Path | str = ROUTES_PATH) -> dict[str, object]:
        """Check the refit reproduces the stored p_fric, and the chain the stored routes."""
        path = Path(path)
        if not path.exists():
            return {"checked": False, "reason": f"{path} missing — run `make policy`"}
        stored = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    stored[int(r["idx"])] = r
        p_gap = max(abs(stored[int(i)]["p_fric"] - float(p))
                    for i, p in zip(self.idx, self.oof_p))
        tau = self.tau()
        img_pre = self._img_pre()
        recon = rmodel.load_recon()
        mismatch = 0
        for i, p in zip(self.idx, self.oof_p):
            i = int(i)
            row = rpolicy.route_line(recon[i], float(p), int(img_pre.get(i, 0)),
                                     self.counts, tau)
            if row["route"] != stored[i]["route"] or row["changed"] != stored[i]["changed"]:
                mismatch += 1
        return {
            "checked": True,
            "n": len(stored),
            "max_p_fric_gap": float(p_gap),
            "tau": float(tau),
            "route_mismatches": mismatch,
            "changed": sum(1 for r in stored.values() if r["changed"]),
        }

    def _img_pre(self) -> dict[int, int]:
        """Pre-treatment image counts per idx, from the same source router.model uses."""
        if not hasattr(self, "_img_cache"):
            _X, _c, _i, _a, img_pre, _s = rmodel.load_design()
            object.__setattr__(self, "_img_cache", img_pre)
        return self._img_cache


def build_ensemble(lam: float = rmodel.DEFAULT_LAM, k: int = rmodel.DEFAULT_K,
                   seed: int = 0) -> Ensemble:
    """Refit the K grouped folds, keeping every artifact needed to serve a new row.

    This is `router.model.fit_predict_oof` with nothing thrown away. The fold
    split, the standardisation, the ridge solve and the inner calibration split
    are the same calls with the same seeds, which is what makes the acceptance
    check (`verify_against_routes`) a real check rather than a tautology.
    """
    X, cols, idx, arms, _img_pre, _source = rmodel.load_design()
    labels = rmodel.load_labels()
    y = np.array([labels[int(i)]["y_fric"] for i in idx], dtype=np.float64)
    jobkeys = rmodel.load_jobkeys()
    groups = rmodel.group_labels(idx, jobkeys)

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    arm_names = sorted(set(arms))
    code = {a: i for i, a in enumerate(arm_names)}
    arm_codes = np.array([code[a] for a in arms], dtype=int)

    folds = rmodel.grouped_kfold(groups, k=k, seed=seed)
    oof = np.full(n, np.nan)
    predictors: list[FoldPredictor] = []

    for _f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test, assume_unique=False)
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xtr, Xte = (X[train] - mu) / sd, (X[test] - mu) / sd
        Dtr, pen = rmodel.design_matrix(Xtr, arm_codes[train], len(arm_names))
        Dte, _ = rmodel.design_matrix(Xte, arm_codes[test], len(arm_names))
        w = rmodel.ridge_logistic_irls(Dtr, y[train], lam=lam,
                                       max_iter=rmodel.DEFAULT_MAX_ITER, penalize=pen)
        raw_te = rmodel.predict_score(Dte, w)

        # inner grouped split of the TRAINING fold only -> honest calibration,
        # identical to router.model.fit_predict_oof
        inner_groups = [groups[i] for i in train]
        inner = rmodel.grouped_kfold(inner_groups, k=min(4, max(2, k - 1)), seed=seed + 101)
        cal_scores = np.full(train.size, np.nan)
        for itest in inner:
            itr = np.setdiff1d(np.arange(train.size), itest)
            A, B = rmodel._standardize(X[train][itr], X[train][itest])
            DA, penA = rmodel.design_matrix(A, arm_codes[train][itr], len(arm_names))
            DB, _ = rmodel.design_matrix(B, arm_codes[train][itest], len(arm_names))
            wi = rmodel.ridge_logistic_irls(DA, y[train][itr], lam=lam,
                                            max_iter=rmodel.DEFAULT_MAX_ITER, penalize=penA)
            cal_scores[itest] = rmodel.predict_score(DB, wi)
        calibrator = rmodel.isotonic_pav(cal_scores, y[train])
        oof[test] = calibrator(raw_te)

        predictors.append(FoldPredictor(mean=mu, std=sd, w=w,
                                        arm_names=list(arm_names), calibrator=calibrator))

    recon_map = rmodel.load_recon()
    return Ensemble(
        folds=predictors, cols=list(cols), idx=np.asarray(idx, dtype=int),
        arms=list(arms), y=y, groups=list(groups), oof_p=oof, fold_index=folds,
        counts=rpolicy.arm_counts(list(recon_map.values())),
        lam=lam, k=k, seed=seed,
        lane_medians=lane_medians(X, cols, recon_map),
        neutral_recur=neutral_recurrence(X, cols),
    )


_CACHE: dict[tuple, Ensemble] = {}


def get_ensemble(lam: float = rmodel.DEFAULT_LAM, k: int = rmodel.DEFAULT_K,
                 seed: int = 0) -> Ensemble:
    """Process-lifetime cache around `build_ensemble` — the fit takes about 12 s."""
    key = (lam, k, seed)
    if key not in _CACHE:
        _CACHE[key] = build_ensemble(lam, k, seed)
    return _CACHE[key]


# --------------------------------------------------------------------------- #
# 2. the lane, its incumbents and its typical shape
# --------------------------------------------------------------------------- #
#: Feature columns the console exposes as shape sliders, defaulted to the lane median.
SHAPE_COLS: tuple[str, ...] = ("prerun_tok", "log_sys_tok", "n_tools", "log_tools_tok")


def lane_medians(X: np.ndarray, cols: Sequence[str],
                 recon_map: dict[int, dict] | None = None) -> dict[str, dict[str, float]]:
    """Per-lane medians of the SHAPE feature columns — the console's slider defaults.

    Taken off the built feature matrix rather than off recon.jsonl, so a default
    is by construction a value the predictor was trained on. A synthetic request
    assembled from these medians lands in the middle of the training cloud instead
    of many standard deviations outside it, which is where an unpadded two-line
    prompt lands and where a linear model's extrapolation means nothing.

    These are AGGREGATES (medians of estimated token counts), never text.
    """
    recon_map = rmodel.load_recon() if recon_map is None else recon_map
    col_at = {c: i for i, c in enumerate(cols)}
    is_gpt = X[:, col_at["is_gpt"]] > 0.5
    out: dict[str, dict[str, float]] = {}
    for lane, mask in (("claude", ~is_gpt), ("gpt", is_gpt)):
        if not mask.any():
            continue
        med = {c: float(np.median(X[mask, col_at[c]])) for c in SHAPE_COLS}
        med["n"] = int(mask.sum())
        med["ctx_lines"] = float(np.median(X[mask, col_at["ctx_lines"]]))
        med["n_threads"] = float(np.median(X[mask, col_at["n_threads"]]))
        out[lane] = med
    return out


def neutral_recurrence(X: np.ndarray, cols: Sequence[str]) -> dict[str, float]:
    """Block B for a task with no job history — the GLOBAL PRIOR, not zeros.

    `router.features.NEUTRAL_RECUR` is a placeholder that `recurrence_features()`
    overwrites: a real row with no literal cron path gets job_fric_eb = the
    leave-one-out global friction rate (about 0.26) and job_turns_eb = the
    leave-one-out mean log1p(n_turns) (about 2.09). Feeding the raw zeros to a
    live scorer instead puts job_fric_eb 24 standard deviations below the training
    mean -- that one column then dominates the linear score and every task comes
    back with the same p_fric. We read the prior straight off the rows that carry
    it (has_job_key == 0), so the live default IS the value those rows were fit on.
    """
    col_at = {c: i for i, c in enumerate(cols)}
    no_key = X[:, col_at["has_job_key"]] < 0.5
    if not no_key.any():  # pragma: no cover - every export has such rows
        raise ValueError("no rows without a job key: cannot read the Block-B prior")
    return {
        "has_job_key": 0.0,
        "job_run_index": 0.0,
        "job_fric_eb": float(np.mean(X[no_key, col_at["job_fric_eb"]])),
        "job_turns_eb": float(np.mean(X[no_key, col_at["job_turns_eb"]])),
    }


def incumbent_prior(ens: Ensemble, lane: str) -> dict[str, float]:
    """Logged share of each arm within a lane — the default weighting over incumbents.

    Only arms that pass gate 3 (more than MIN_ARM_RUNS logged runs) can be an
    incumbent the router reasons about, so the thin arms are dropped here rather
    than carried as a target the policy would refuse anyway.
    """
    prefix = "gpt" if lane == "gpt" else "claude"
    lane_counts = {a: n for a, n in ens.counts.items()
                   if a.startswith(prefix) and n > rpolicy.MIN_ARM_RUNS}
    total = sum(lane_counts.values())
    if not total:
        return {}
    return {a: n / total for a, n in sorted(lane_counts.items())}


# --------------------------------------------------------------------------- #
# 3. synthesizing a request the feature extractor recognises
# --------------------------------------------------------------------------- #
#: Filler token used to pad a synthetic item up to a requested estimated size.
#: `tok()` is len(json.dumps(x)) // 4, so four filler characters buy one token.
_PAD_UNIT = "x"


def _pad_to(base: str, target_tok: int) -> str:
    """Extend `base` with filler until tok() of the enclosing item reaches target_tok."""
    need = max(0, int(target_tok) * 4 - len(json.dumps(base)))
    return base + (_PAD_UNIT * need)


def lane_tools(lane: str, n_tools: int, tools_tok: int) -> list[dict]:
    """A synthetic tools array in the right lane, padded to a requested size.

    The names are the ones router.io separates the families on and nothing else;
    gate 1 reads this array, so the lane is a property of the tool contract, never
    a choice the router makes.
    """
    names = sorted(GPT_TOOLS if lane == "gpt" else CLAUDE_TOOLS)
    picked = [names[i % len(names)] if i < len(names) else f"tool_{i}"
              for i in range(max(1, int(n_tools)))]
    tools = [{"type": "function", "name": nm, "description": "", "parameters": {}}
             for nm in picked]
    slack = max(0, int(tools_tok) * 4 - len(json.dumps(tools)))
    if slack:
        tools[0]["description"] = _PAD_UNIT * slack
    return tools


def synthesize_request(task_text: str, lane: str = "claude", trigger: str = "cron",
                       headers: Sequence[str] = (), ctx_lines: int = 0,
                       n_threads: int = 0, images: int = 0,
                       prerun_tok: int | None = None, sys_tok: int | None = None,
                       tools_tok: int | None = None, n_tools: int | None = None,
                       lane_med: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Build a pre-treatment request view for a typed task.

    Every marker written here is a constant from router.features / router.io:
    the `<system>` wrapper, the `Triggered by:` line, the `# === ... ===` header
    blocks, the `\\n[` context markers and the `thread_ts` handles. Nothing is
    copied from the export.

    THE BRIEFING CARRIES THE BULK, NOT THE SYSTEM ITEM. In this corpus the
    pre-treatment prefix sits mostly inside the first USER message -- the harness
    briefing, its context blocks and its pasted threads -- with the system item a
    distant second. Padding the system item instead leaves log_user_tok about five
    standard deviations low, and the scorer reads that as a task unlike anything
    it was fit on. The filler goes into the briefing block for the same reason the
    defaults are lane medians: to land inside the training cloud, not beside it.
    """
    if lane_med is None:
        X, cols, *_ = rmodel.load_design()
        lane_med = lane_medians(X, cols)
    med = lane_med.get(lane, {})
    prerun_tok = int(med.get("prerun_tok", 2500)) if prerun_tok is None else int(prerun_tok)
    sys_tok = (round(float(np.expm1(med.get("log_sys_tok", 9.0)))) if sys_tok is None
               else int(sys_tok))
    tools_tok = (round(float(np.expm1(med.get("log_tools_tok", 8.3)))) if tools_tok is None
                 else int(tools_tok))
    n_tools = int(med.get("n_tools", 12)) if n_tools is None else int(n_tools)

    tools = lane_tools(lane, n_tools, tools_tok)

    briefing = ["<system>", f"Triggered by: `{trigger}`" if trigger != "absent" else ""]
    known = dict(rfeatures.HEADER_BLOCKS)
    for name in headers:
        if name in known:
            briefing.append(known[name])
    briefing.extend("\n[context]" for _ in range(max(0, int(ctx_lines))))
    briefing.extend("thread_ts" for _ in range(max(0, int(n_threads))))
    # pad INSIDE the block so the filler counts toward prerun_tok, then close it
    body = "\n".join(b for b in briefing if b)
    block = _pad_to(body, prerun_tok - len("</system>") // 4) + "\n</system>"

    user_parts: list[dict] = [{"type": "input_text", "text": block + "\n" + (task_text or "")}]
    for _ in range(max(0, int(images))):
        user_parts.append({"type": "input_image", "image_url": "[base64 image redacted]"})
    user_item = {"type": "message", "role": "user", "content": user_parts}

    sys_text = "You are an agent."
    shell = {"type": "message", "role": "system",
             "content": [{"type": "input_text", "text": sys_text}]}
    pad = max(0, sys_tok - len(json.dumps(shell)) // 4)
    sys_item = {"type": "message", "role": "system",
                "content": [{"type": "input_text", "text": sys_text + _PAD_UNIT * (pad * 4)}]}
    return {"input": [sys_item, user_item], "tools": tools}


def feature_row(req: dict, ens: Ensemble, job_fric_eb: float | None = None,
                job_run_index: float = 0.0, has_job_key: bool = False) -> np.ndarray:
    """Feature vector for a synthetic request, in the ensemble's column order.

    Block B (recurrence) has no history for a task nobody has run, so it defaults to
    the ensemble's GLOBAL PRIOR (`neutral_recurrence`) -- the same value a logged row
    without a literal cron path carries -- not to the raw zeros in
    router.features.NEUTRAL_RECUR, which sit far outside the training range. The
    console exposes the override as "this is the Nth run of a known cron job, which
    has failed X% of the time", because that is the one feature block a live caller
    genuinely might know and the extractor cannot infer from the text.
    """
    recur = dict(ens.neutral_recur)
    if has_job_key:
        recur["has_job_key"] = 1.0
        recur["job_run_index"] = float(job_run_index)
        if job_fric_eb is not None:
            recur["job_fric_eb"] = float(job_fric_eb)
    feats = rfeatures.extract(idx=-1, req=req, jobkey_record=None, recur=recur, toolset_rank=0)
    return np.array([float(feats[c]) for c in ens.cols], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 4. routing one feature row
# --------------------------------------------------------------------------- #
def route_features(ens: Ensemble, arm: str, p_fric: float, tau: float, lane: str,
                   img_pre: int = 0, sheet: str = rpolicy.DEFAULT_SHEET) -> dict:
    """Run the real five-gate chain on a live row. No reimplementation, no shortcut."""
    rec = {"idx": -1, "model": arm, "family": lane}
    vision = rpolicy.gate_vision(img_pre)
    logged_ok, cands = rpolicy.gate_admissible(arm, lane, ens.counts)
    stable = [] if (not logged_ok or vision) else rpolicy.gate_sign_stable(
        arm, cands, ALL_SHEETS)
    tau_pass = rpolicy.gate_tau(p_fric, tau)
    target = rpolicy.cheapest_target(stable, sheet)
    route = target if (target is not None and tau_pass) else arm
    return {
        "logged": arm, "route": route, "changed": route != arm,
        "p_fric": float(p_fric), "tau": float(tau),
        "gates": {
            "family": lane, "vision": bool(vision),
            "admissible": cands if logged_ok else [], "sign_stable": stable,
            "tau_pass": bool(tau_pass), "target": target,
        },
        "_rec": rec,
    }


#: Features whose PARTIAL slope points the opposite way to their MARGINAL relation
#: with friction. Naming them in the payload is the only honest way to ship a
#: slider whose direction would otherwise read as a bug. Measured on this export.
SUPPRESSED = {
    "job_fric_eb": (
        "counterintuitive by design, not by accident: on its own this feature "
        "correlates POSITIVELY with friction (r = +0.10 on the 374 rows that carry a "
        "job key), but its partial slope is sign-stably NEGATIVE across all five folds. "
        "It is collinear with job_turns_eb at r = 0.69, and job_turns_eb is the stronger "
        "marginal predictor (r = +0.22). Conditioned on how long a job runs, a job that "
        "has failed more often is the SAFER one to downgrade. Do not read this slider as "
        "a causal claim."
    ),
}


def explain(ens: Ensemble, x: np.ndarray, arm: str, top_k: int = 8) -> dict:
    """Per-feature contribution to the score for one task, averaged over the folds.

    The contribution is the standardised value times the fitted slope, which is what
    the linear score actually sums. It answers "why did THIS task move" in the units
    the model works in, rather than showing a slope table that is the same for every
    task. Fold spread is reported per feature so a contribution that only one fold
    believes is visible as such.
    """
    n_arms = len(ens.folds[0].arm_names)
    contrib = np.zeros((len(ens.folds), len(ens.cols)))
    for f, fold in enumerate(ens.folds):
        z = (np.asarray(x, dtype=np.float64) - fold.mean) / fold.std
        contrib[f] = z * fold.w[n_arms:]
    mean = contrib.mean(axis=0)
    order = np.argsort(-np.abs(mean))[:top_k]
    return {
        "arm": arm,
        "top": [
            {
                "col": ens.cols[j],
                "value": float(x[j]),
                "contribution": float(mean[j]),
                "lo": float(contrib[:, j].min()),
                "hi": float(contrib[:, j].max()),
                "sign_stable": bool(np.all(np.sign(contrib[:, j]) == np.sign(contrib[0, j]))),
                "caveat": SUPPRESSED.get(ens.cols[j]),
            }
            for j in order
        ],
        "basis": (
            "contribution = standardised feature value x fitted slope, averaged over the "
            "5 folds. It explains the SCORE, which the isotonic calibrator then maps to "
            "p_fric — the mapping is monotone, so the ordering carries over but the "
            "magnitudes do not."
        ),
    }


# --------------------------------------------------------------------------- #
# 5. the selection distribution
# --------------------------------------------------------------------------- #
def selection_distribution(ens: Ensemble, x: np.ndarray, lane: str,
                           incumbent: str | None = None,
                           alpha: float = rpolicy.DEFAULT_ALPHA,
                           n_draws: int = DEFAULT_DRAWS,
                           img_pre: int = 0,
                           sheet: str = rpolicy.DEFAULT_SHEET,
                           seed: int = 0) -> dict:
    """P(arm is the routed arm) over draws of (fold predictor, bootstrap tau).

    The two sources of spread are the only two things here that were estimated:
    which of the five fold predictors you believe, and where the conformal tau
    falls under resampling of the calibration jobs. Everything else -- the lane,
    the admissible set, the sign-stable targets, the price comparison -- is read
    off the data and is identical in every draw. A distribution that is a point
    mass therefore means the gates decided, not that the model was certain.
    """
    prior = ({incumbent: 1.0} if incumbent else incumbent_prior(ens, lane))
    if not prior:
        raise ValueError(f"no admissible incumbent arm in lane {lane!r}")
    for arm in prior:
        if arm not in ens.folds[0].arm_names:
            raise ValueError(f"arm {arm!r} was never logged in this export")

    taus = ens.tau_draws(alpha=alpha, n_boot=max(1, n_draws // len(ens.folds)), seed=seed)
    tau_point = ens.tau(alpha)

    # p_fric depends on (fold, incumbent) only — 5 x |arms| evaluations, not n_draws
    p_by = {(f, a): ens.folds[f].p_fric(x, a)
            for f in range(len(ens.folds)) for a in prior}

    mass: dict[str, float] = {}
    tau_pass_mass = 0.0
    n = 0
    per_incumbent: dict[str, dict] = {}
    for arm, share in prior.items():
        arm_mass: dict[str, float] = {}
        for f in range(len(ens.folds)):
            for t in taus:
                r = route_features(ens, arm, p_by[(f, arm)], float(t), lane,
                                   img_pre=img_pre, sheet=sheet)
                arm_mass[r["route"]] = arm_mass.get(r["route"], 0.0) + 1.0
                tau_pass_mass += share * (1.0 if r["gates"]["tau_pass"] else 0.0)
                n += 1
        total = sum(arm_mass.values())
        ps = [p_by[(f, arm)] for f in range(len(ens.folds))]
        chain = route_features(ens, arm, float(np.mean(ps)), tau_point, lane,
                               img_pre=img_pre, sheet=sheet)
        per_incumbent[arm] = {
            "share_of_lane": share,
            "p_fric_mean": float(np.mean(ps)),
            "p_fric_folds": [float(v) for v in ps],
            "p_fric_lo": float(np.min(ps)),
            "p_fric_hi": float(np.max(ps)),
            "distribution": {k: v / total for k, v in sorted(arm_mass.items())},
            "gates": chain["gates"],
            "route_at_point_tau": chain["route"],
        }
        for k, v in arm_mass.items():
            mass[k] = mass.get(k, 0.0) + share * v / total

    total = sum(mass.values()) or 1.0
    dist = {k: v / total for k, v in sorted(mass.items(), key=lambda kv: -kv[1])}
    n_draws_used = len(ens.folds) * len(taus)
    return {
        "lane": lane,
        "distribution": dist,
        "per_incumbent": per_incumbent,
        "incumbent_mode": "pinned" if incumbent else "marginalised over the lane's logged mix",
        "tau_point": float(tau_point),
        "tau_lo": float(np.percentile(taus[np.isfinite(taus)], 5)) if np.any(np.isfinite(taus)) else None,
        "tau_hi": float(np.percentile(taus[np.isfinite(taus)], 95)) if np.any(np.isfinite(taus)) else None,
        "tau_never_certified_share": float(np.mean(~np.isfinite(taus))),
        "n_draws": n_draws_used,
        "alpha": float(alpha),
        "sheet": sheet,
        "p_tau_pass": tau_pass_mass / n if n else 0.0,
        "basis": (
            "spread comes from the 5 fold predictors and the bootstrap of the conformal "
            "tau — the router itself is deterministic. p_fric is PROCESS friction from "
            "tool-output exit codes, never answer quality."
        ),
    }


def price_table(ens: Ensemble, sheets: Sequence[str] = ALL_SHEETS) -> dict[str, dict[str, float]]:
    """Input rate per 1M ESTIMATED tokens for every logged arm, under each ASSUMED sheet.

    All three sheets are returned, not just the default, because gate 4 only accepts
    a target that is cheaper under every one of them -- the console cannot show why
    an arm was refused while displaying a single column of prices.
    """
    return {arm: {s: float(rate_in(arm, s)) for s in sheets}
            for arm in sorted(ens.counts)}


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #
def acceptance(ens: Ensemble | None = None) -> list[tuple]:
    """Return [(pass, label, expected, actual), ...] for every acceptance number."""
    ens = get_ensemble() if ens is None else ens
    rows: list[tuple] = []

    v = ens.verify_against_routes()
    if not v.get("checked"):
        return [(False, "results/routes.jsonl readable", "present", v.get("reason"))]

    rows.append((v["n"] == EXPECTED["n"], "rows in routes.jsonl", EXPECTED["n"], v["n"]))
    rows.append((v["max_p_fric_gap"] < 1e-12,
                 "refit reproduces stored p_fric", "< 1e-12",
                 f"{v['max_p_fric_gap']:.3e}"))
    rows.append((abs(v["tau"] - EXPECTED["tau_alpha_10"]) < 1e-6,
                 "conformal tau at alpha=0.10", EXPECTED["tau_alpha_10"], round(v["tau"], 6)))
    rows.append((v["route_mismatches"] == 0,
                 "gate chain replays routes.jsonl exactly", 0, v["route_mismatches"]))
    rows.append((v["changed"] == EXPECTED["changed"], "rerouted trajectories",
                 EXPECTED["changed"], v["changed"]))

    # a live task in each lane: distributions sum to 1 and stay in their lane
    supports = {}
    for lane in ("claude", "gpt"):
        req = synthesize_request("Summarise yesterday's incidents and post the digest.",
                                 lane=lane, lane_med=ens.lane_medians)
        x = feature_row(req, ens)
        d = selection_distribution(ens, x, lane, n_draws=50)
        s = sum(d["distribution"].values())
        rows.append((abs(s - 1.0) < 1e-9, f"{lane} lane: distribution sums to 1", 1.0, round(s, 12)))
        prefix = "gpt" if lane == "gpt" else "claude"
        in_lane = all(a.startswith(prefix) for a in d["distribution"])
        rows.append((in_lane, f"{lane} lane: support stays in lane", True, in_lane))
        supports[lane] = set(d["distribution"])

    disjoint = not (supports["claude"] & supports["gpt"])
    rows.append((disjoint, "the two lanes share no arm", True, disjoint))

    # The console's whole point is that the answer depends on the task. If two
    # very different tasks come back with the same distribution, the live demo is
    # showing the incumbent prior and nothing else -- which is what a Block-B
    # feature pinned outside the training range silently produces. Pin it.
    quiet = synthesize_request("Post the daily standup reminder.", lane="claude",
                               trigger="cron", headers=("hdr_thread_info",),
                               ctx_lines=2, n_threads=0, lane_med=ens.lane_medians)
    busy = synthesize_request("Investigate the failed nightly ETL, read the logs, "
                              "patch the config and report back." * 8, lane="claude",
                              trigger="slack_mention",
                              headers=("hdr_thread_info", "hdr_channel_thread"),
                              ctx_lines=45, n_threads=9, lane_med=ens.lane_medians)
    dq = selection_distribution(ens, feature_row(quiet, ens), "claude", n_draws=200)
    db = selection_distribution(ens, feature_row(busy, ens), "claude", n_draws=200)
    arms = set(dq["distribution"]) | set(db["distribution"])
    gap = max(abs(dq["distribution"].get(a, 0.0) - db["distribution"].get(a, 0.0)) for a in arms)
    rows.append((gap > 0.05, "two different tasks give different distributions",
                 "> 0.05 max gap", round(gap, 4)))

    # And the feature vector must land inside the cloud the predictor was fit on,
    # or the linear extrapolation behind that spread means nothing.
    X, _cols, *_ = rmodel.load_design()
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = np.abs((feature_row(quiet, ens) - mu) / sd)
    rows.append((z.max() < 4.0, "synthetic row stays inside the training range",
                 "max |z| < 4", round(float(z.max()), 2)))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the ensemble, print what it reproduces, and run the acceptance table."""
    print("building the fold ensemble (refit of router.model, nothing discarded) ...")
    ens = get_ensemble()
    v = ens.verify_against_routes()
    print(f"  folds={len(ens.folds)}  cols={len(ens.cols)}  arms={len(ens.folds[0].arm_names)}")
    print(f"  max |p_refit - p_stored| = {v['max_p_fric_gap']:.3e} over {v['n']} rows")
    print(f"  tau(alpha=0.10) = {v['tau']:.6f}   rerouted = {v['changed']}")
    print()
    for lane in ("claude", "gpt"):
        med = ens.lane_medians.get(lane, {})
        prior = incumbent_prior(ens, lane)
        print(f"  lane {lane:<7} n={int(med.get('n', 0)):<4} median prerun_tok={med.get('prerun_tok', 0):.0f} "
              f"n_tools={med.get('n_tools', 0):.0f} ctx_lines={med.get('ctx_lines', 0):.0f}")
        print(f"  {'':<12} incumbents: " +
              ", ".join(f"{a} {s:.1%}" for a, s in prior.items()))
    print()

    rows = acceptance(ens)
    failed = 0
    for ok, label, expected, actual in rows:
        mark = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"  [{mark}] {label:<46} expected={expected!r:<12} actual={actual!r}")
    print()
    print("  Token counts are ESTIMATES (len(json.dumps(x))//4); the export has no `usage` field.")
    print("  The selection spread is OUR uncertainty (folds + tau bootstrap), not router randomness.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
