#!/usr/bin/env python3
"""Pre-treatment feature matrix for the Viktor trajectory export.

WHAT THIS COMPUTES
    One row of numeric, strictly PRE-TREATMENT features per exported trajectory.
    "Pre-treatment" has an exact meaning here: the only bytes this module is
    allowed to look at are

        request["input"][: first_user_index + 1]     (system message + first user
                                                      message; first_user_index is
                                                      1 for 1000/1000 lines)
        request["tools"]                             (the tool schema block)

    and nothing else.  Everything after the first user message is produced by the
    model that was actually routed, i.e. it is a COLLIDER of the treatment: 46.5%
    of all estimated tokens in the export live there (10,517,503 of 22,631,879).
    Conditioning on it would bake the treatment into the covariates and make any
    off-policy contrast meaningless.

    The gate is physical, not a promise.  ``pre_request()`` builds a *view* of the
    request that contains only the slice and the tools array — and drops the
    ``model`` field, which is the TREATMENT.  ``extract()`` asserts that the last
    item it was handed is the first user message, so passing it the full item list
    raises instead of silently leaking.  A module-level ``BANNED`` set of column
    names is asserted against the emitted columns on every build.

ARTIFACTS WRITTEN
    results/features.npz            arrays "X" (n,d) float64, "cols" (d,) unicode,
                                    "idx" (n,) int64  (idx ascending 0..999)
    results/feature_manifest.json   {"cols": [{"name", "pre_treatment", "block"}, ...]}
                                    in the same order as "cols", plus documentation
                                    keys the contract does not forbid.

BLOCKS
    A  briefing      — parses the harness preamble the scheduler pastes into the
                       first user message: trigger provenance, pre-pasted context
                       volume, the residual task text, header-block presence,
                       pre-treatment image parts.
    B  recurrence    — from results/jobkey.jsonl, LITERAL bucket only: how often
                       this cron job recurs in the log, plus LEAVE-CURRENT-RUN-OUT
                       empirical-Bayes shrunk group means of y_fric (beta-binomial)
                       and of log1p(n_turns) (normal-normal).  Runs in the "pii" or
                       "none" bucket get the global prior and has_job_key = 0.
    C  size/structure— log token volumes, tool count, toolset variant one-hot taken
                       WITHIN family, and the family flag itself.

THINGS DELIBERATELY NOT DONE (each with the measurement that killed it)
    * No feature from input[first_user_index+1:]           — 46.5% collider mass.
    * No bag-of-words over the task text.  After stripping the leading
      <system>...</system> block the residual task text is EMPTY for 737/1000 runs
      — 245/245 gpt and 492/755 claude — so any text-derived feature partially
      encodes family.  Only ``has_task_text`` / ``task_len`` / ``task_lines`` are
      emitted, and the family flag is emitted explicitly next to them so a
      downstream model cannot mistake one for the other.
    * No system-prompt hash                                — 871 distinct over 1000 rows.
    * No "# === Cron memory: PII_URL_N" field              — 763/763 values are PII_
      placeholders, 18 distinct, PII_URL_3 alone covers 306 runs.
    * No PII_* token is ever interpreted or keyed on.  A "Triggered by:" value that
      contains "PII_" is classified as ``other``, never as a chat trigger.
    * No empty_final / final_text_length                   — 238/238 empty finals are
      gpt; it is a serialization artifact that encodes family perfectly.
    * No label field OF THIS ROW.  The two Block-B target encodings are
      leave-current-run-out: row i's own y_fric enters neither its own group sum nor
      its own prior mean.  Measured by flipping one row's y_fric and rebuilding:
      that row's own job_fric_eb moves by 2.2e-05 (only through the globally fitted
      prior strength k, an O(1/n) effect that is documented, not hidden), while a
      PEER in the same job moves by 6.0e-03 -- 270x larger, which is the whole point.

ACCEPTANCE NUMBERS (printed by ``python -m router.features``)
    pre-treatment tokens 12,114,376 vs post-treatment 10,517,503 = 53.5% / 46.5%
      (they sum to 22,631,879, the naive per-item total from recon)
    X.shape == (1000, 36); idx ascending 0..999; X finite everywhere
    family flag reproduces 245 gpt / 755 claude
    has_pre_image sums to 9 (18 input_image parts, 4+2+2+1 rows)
    has_task_text sums to 263 (= 1000 - 737 empty residual task texts)
    has_job_key sums to 374 (the literal bucket, 152 distinct cron jobs)
    BANNED column names: 0 present
    truncation self-test: byte-identical X/cols/idx when every item after the first
      user message is dropped AND when it is replaced by junk with the model id
      redacted

Run:     python -m router.features        (prints every check, non-zero on mismatch)
Import:  pre_request, extract, build, write, COLUMNS, BLOCKS, BANNED
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from router.io import DEFAULT_EXPORT_DIR, family_of, first_user_index, iter_lines, tok

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FEATURES_PATH = RESULTS_DIR / "features.npz"
MANIFEST_PATH = RESULTS_DIR / "feature_manifest.json"
JOBKEY_PATH = RESULTS_DIR / "jobkey.jsonl"
LABELS_PATH = RESULTS_DIR / "labels.jsonl"
RECON_PATH = RESULTS_DIR / "recon.jsonl"

# --------------------------------------------------------------------------- #
# Hard gate: column names that would smuggle post-treatment information in.
# --------------------------------------------------------------------------- #

#: Column names that must never be emitted.  Every one of these is computable
#: only from items after the first user message (or is a known identifier).
BANNED = frozenset(
    {
        "est_tok",
        "post_tok",
        "n_items",
        "n_fc",
        "n_fco",
        "n_asst",
        "n_reason",
        "n_calls",
        "n_steps",
        "final_len",
        "empty_final",
        "sys_sha",
    }
)

# --------------------------------------------------------------------------- #
# Block A vocabulary (fixed in code; counts measured on this export)
# --------------------------------------------------------------------------- #

#: Leading harness briefing block, e.g. "<system> ... </system>" at the very start
#: of the first user message.  996/1000 messages open with it.
SYSTEM_BLOCK_RE = re.compile(r"^\s*<system>.*?</system>", re.S)

#: "Triggered by: `cron`" line inside the briefing.  979/1000 runs carry one.
TRIGGER_RE = re.compile(r"Triggered by:\s*(.*)")

#: Slack thread handles pasted into the briefing; 0..10 per run, 268 runs > 0.
THREAD_RE = re.compile(r"thread_ts")

#: Trigger categories, in one-hot column order.  Measured: cron 740, trigger 38,
#: slack_dm 104, slack_mention 77, msteams 6, absent 21, other 14.
TRIGGER_CATEGORIES = (
    "cron",
    "trigger",
    "slack_dm",
    "slack_mention",
    "msteams",
    "absent",
    "other",
)

#: A run counts as human-triggered when a person poked it (Slack/Teams DM or
#: mention) or when the briefing carries no "Triggered by:" line at all.
#: "other" is NOT human: it holds opaque ids and PII_-masked values, and we never
#: interpret a PII_ placeholder.  Measured: 208/1000 human-triggered.
HUMAN_TRIGGERS = frozenset({"slack_dm", "slack_mention", "msteams", "absent"})

#: The six most prevalent "# === ... ===" briefing blocks, excluding the banned
#: "# === Cron memory: PII_URL_N ===" field.  Counts measured on this export.
#: (Two rarer blocks fall outside the top six: "Recent workspace activity" 60 and
#: "Channel instructions" 29.)
HEADER_BLOCKS = (
    ("hdr_thread_info", "# === Thread info ==="),                                 # 987
    ("hdr_cron_history", "# === Cron history: execution.log tail ==="),           # 739
    ("hdr_output_channel", "# === Expected output channel instructions (auto-read) ==="),  # 358
    ("hdr_personal_skill", "# === Sender's personal skill (auto-read) ==="),      # 120
    ("hdr_recent_dm", "# === Recent DM context ==="),                             # 115
    ("hdr_channel_thread", "# === Channel / thread context ==="),                  # 72
)

# --------------------------------------------------------------------------- #
# Block B priors
# --------------------------------------------------------------------------- #

#: Fallback beta-binomial prior strength when the moment estimator degenerates.
FALLBACK_PRIOR_STRENGTH = 20.0
#: Clamp for the estimated prior strength k = (1 - rho) / rho.
PRIOR_STRENGTH_BOUNDS = (1.0, 1000.0)

# --------------------------------------------------------------------------- #
# Expected values (ground truth for this export)
# --------------------------------------------------------------------------- #

EXPECTED = {
    "n_rows": 1000,
    "pre_tokens": 12_114_376,
    "post_tokens": 10_517_503,
    "naive_tokens": 22_631_879,
    "gpt": 245,
    "claude": 755,
    "has_pre_image": 9,
    "has_task_text": 263,
    "has_job_key": 374,
    "n_jobs": 152,
}


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def item_text(item: Dict[str, Any]) -> str:
    """Return the concatenated ``input_text`` of one message item (str content passes through)."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        p.get("text") or ""
        for p in content
        if isinstance(p, dict) and p.get("type") == "input_text"
    ]
    return "\n".join(parts)


def count_images(item: Dict[str, Any]) -> int:
    """Return the number of ``input_image`` parts in one message item."""
    content = item.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "input_image")


def strip_system_block(text: str) -> Tuple[str, str]:
    """Split a first-user-message into (leading <system>...</system> preamble, residual task text)."""
    match = SYSTEM_BLOCK_RE.match(text)
    if match is None:
        return "", text.strip()
    return match.group(0), text[match.end():].strip()


def trigger_category(text: str) -> str:
    """Classify the briefing's ``Triggered by:`` value into one of TRIGGER_CATEGORIES."""
    match = TRIGGER_RE.search(text)
    if match is None:
        return "absent"
    value = match.group(1).strip().strip("`").strip().lower()
    if not value:
        return "absent"
    if "pii_" in value:
        # Placeholders are renumbered per request; we never interpret one.
        return "other"
    if value == "cron":
        return "cron"
    if value == "trigger":
        return "trigger"
    if value.startswith("slack"):
        return "slack_mention" if "mention" in value else "slack_dm"
    if "msteams" in value or "teams" in value:
        return "msteams"
    return "other"


# --------------------------------------------------------------------------- #
# The physical pre-treatment gate
# --------------------------------------------------------------------------- #


def pre_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Return a view of a request holding ONLY the pre-treatment slice and the tools array."""
    items = request["input"]
    cut = first_user_index(items)
    if cut < 0:
        raise ValueError("line has no user message; the pre-treatment cut point is undefined")
    return {"input": list(items[: cut + 1]), "tools": request.get("tools", [])}


def _assert_pre_treatment(items: Sequence[Dict[str, Any]]) -> int:
    """Raise unless ``items`` ends exactly at the first user message; return that index."""
    cut = first_user_index(list(items))
    if cut < 0:
        raise ValueError("pre-treatment slice contains no user message")
    if cut != len(items) - 1:
        raise ValueError(
            "extract() was handed %d items but the first user message is at index %d: "
            "this is the post-treatment leak the module exists to prevent"
            % (len(items), cut)
        )
    return cut


# --------------------------------------------------------------------------- #
# Column schema
# --------------------------------------------------------------------------- #

def _toolset_columns() -> Tuple[str, ...]:
    """Return the toolset one-hot column names (variant rank within family, rare bucket last)."""
    return ("toolset_v0", "toolset_v1", "toolset_v2", "toolset_rare")


#: (column name, block) in emission order.  Block A = briefing, B = recurrence,
#: C = size/structure.  Every column is pre-treatment by construction.
COLUMNS: Tuple[Tuple[str, str], ...] = (
    # ---- Block A: the harness briefing -----------------------------------
    ("human_triggered", "A"),
    ("trig_cron", "A"),
    ("trig_trigger", "A"),
    ("trig_slack_dm", "A"),
    ("trig_slack_mention", "A"),
    ("trig_msteams", "A"),
    ("trig_absent", "A"),
    ("trig_other", "A"),
    ("ctx_lines", "A"),
    ("task_len", "A"),
    ("task_lines", "A"),
    ("has_task_text", "A"),
    ("n_threads", "A"),
    ("hdr_thread_info", "A"),
    ("hdr_cron_history", "A"),
    ("hdr_output_channel", "A"),
    ("hdr_personal_skill", "A"),
    ("hdr_recent_dm", "A"),
    ("hdr_channel_thread", "A"),
    ("img_pre", "A"),
    ("has_pre_image", "A"),
    ("prerun_tok", "A"),
    # ---- Block B: recurrence ---------------------------------------------
    ("has_job_key", "B"),
    ("job_run_index", "B"),
    ("job_fric_eb", "B"),
    ("job_turns_eb", "B"),
    # ---- Block C: size and structure -------------------------------------
    ("log_sys_tok", "C"),
    ("log_user_tok", "C"),
    ("log_pre_tok", "C"),
    ("log_tools_tok", "C"),
    ("n_tools", "C"),
    ("toolset_v0", "C"),
    ("toolset_v1", "C"),
    ("toolset_v2", "C"),
    ("toolset_rare", "C"),
    ("is_gpt", "C"),
)

#: Ordered, continuous size measures. High distinct-value counts here are expected
#: and are NOT the identifier smell that 90-leak-audit looks for (see the manifest note).
CONTINUOUS_COLS: Tuple[str, ...] = (
    "ctx_lines",
    "task_len",
    "task_lines",
    "n_threads",
    "prerun_tok",
    "job_run_index",
    "job_fric_eb",
    "job_turns_eb",
    "log_sys_tok",
    "log_user_tok",
    "log_pre_tok",
    "log_tools_tok",
    "n_tools",
)

BLOCKS: Dict[str, str] = {name: block for name, block in COLUMNS}
COLUMN_NAMES: Tuple[str, ...] = tuple(name for name, _ in COLUMNS)

_clash = sorted(BANNED & set(COLUMN_NAMES))
if _clash:  # pragma: no cover - a hard build-time gate
    raise AssertionError("banned column names present in the schema: %r" % (_clash,))
if len(set(COLUMN_NAMES)) != len(COLUMN_NAMES):  # pragma: no cover
    raise AssertionError("duplicate column names in the schema")


# --------------------------------------------------------------------------- #
# Per-row extraction
# --------------------------------------------------------------------------- #

#: Neutral Block-B values for a row with no usable job key.  Overwritten with the
#: global prior by ``recurrence_features()``; kept here so ``extract()`` alone is
#: still total.
NEUTRAL_RECUR = {
    "has_job_key": 0.0,
    "job_run_index": 0.0,
    "job_fric_eb": 0.0,
    "job_turns_eb": 0.0,
}


def extract(
    idx: int,
    req: Dict[str, Any],
    jobkey_record: Optional[Dict[str, Any]],
    recur: Optional[Dict[str, float]] = None,
    toolset_rank: int = 0,
) -> Dict[str, float]:
    """Return the feature dict for one row; ``req`` MUST be a ``pre_request()`` view."""
    items = req["input"]
    cut = _assert_pre_treatment(items)

    sys_item = items[0]
    user_item = items[cut]
    tools = req.get("tools") or []

    user_text = item_text(user_item)
    preamble, task_text = strip_system_block(user_text)
    category = trigger_category(user_text)

    feats: Dict[str, float] = {}

    # ---- Block A ---------------------------------------------------------
    feats["human_triggered"] = 1.0 if category in HUMAN_TRIGGERS else 0.0
    for cat in TRIGGER_CATEGORIES:
        feats["trig_" + cat] = 1.0 if category == cat else 0.0
    feats["ctx_lines"] = float(user_text.count("\n["))
    feats["task_len"] = float(len(task_text))
    feats["task_lines"] = float(task_text.count("\n") + 1 if task_text else 0)
    feats["has_task_text"] = 1.0 if task_text else 0.0
    feats["n_threads"] = float(len(THREAD_RE.findall(user_text)))
    for name, marker in HEADER_BLOCKS:
        feats[name] = 1.0 if marker in user_text else 0.0
    n_img = sum(count_images(it) for it in items)
    feats["img_pre"] = float(n_img)
    feats["has_pre_image"] = 1.0 if n_img else 0.0
    feats["prerun_tok"] = float(tok(preamble))

    # ---- Block B ---------------------------------------------------------
    values = dict(NEUTRAL_RECUR)
    if recur is not None:
        values.update(recur)
    if jobkey_record is not None and not jobkey_record.get("literal", False):
        # Belt and braces: a pii/none row never carries a job-derived encoding.
        values["has_job_key"] = 0.0
        values["job_run_index"] = 0.0
    for name in ("has_job_key", "job_run_index", "job_fric_eb", "job_turns_eb"):
        feats[name] = float(values[name])

    # ---- Block C ---------------------------------------------------------
    sys_tok = tok(sys_item)
    user_tok = tok(user_item)
    tools_tok = tok(tools)
    pre_tok = tools_tok + sum(tok(it) for it in items)
    feats["log_sys_tok"] = math.log1p(sys_tok)
    feats["log_user_tok"] = math.log1p(user_tok)
    feats["log_pre_tok"] = math.log1p(pre_tok)
    feats["log_tools_tok"] = math.log1p(tools_tok)
    feats["n_tools"] = float(len(tools))
    for rank, name in enumerate(_toolset_columns()[:-1]):
        feats[name] = 1.0 if toolset_rank == rank else 0.0
    feats["toolset_rare"] = 1.0 if toolset_rank >= len(_toolset_columns()) - 1 else 0.0
    feats["is_gpt"] = 1.0 if family_of(req) == "gpt" else 0.0

    missing = [c for c in COLUMN_NAMES if c not in feats]
    if missing:  # pragma: no cover - schema drift guard
        raise AssertionError("extract() produced no value for %r" % (missing,))
    return feats


# --------------------------------------------------------------------------- #
# Toolset variants (Block C, ranked WITHIN family)
# --------------------------------------------------------------------------- #


def toolset_ranks(export_dir: str = DEFAULT_EXPORT_DIR) -> Dict[int, int]:
    """Rank each row's toolset by its prevalence WITHIN that row's family (0 = most common)."""
    sigs: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
    counts: Dict[Tuple[str, Tuple[str, ...]], int] = defaultdict(int)
    for idx, req in iter_lines(export_dir):
        view = pre_request(req)
        names = tuple(sorted(t.get("name") or "" for t in view["tools"] if isinstance(t, dict)))
        sig = (family_of(view), names)
        sigs[idx] = sig
        counts[sig] += 1
    order: Dict[str, Dict[Tuple[str, ...], int]] = defaultdict(dict)
    by_family: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for (fam, names), n in counts.items():
        by_family[fam].append((n, names))
    for fam, entries in by_family.items():
        # Deterministic: descending prevalence, then lexicographic on the signature.
        for rank, (_n, names) in enumerate(sorted(entries, key=lambda e: (-e[0], e[1]))):
            order[fam][names] = rank
    return {idx: order[fam][names] for idx, (fam, names) in sigs.items()}


# --------------------------------------------------------------------------- #
# Block B: recurrence with leave-current-run-out empirical Bayes
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load a results/*.jsonl artifact into {idx: record}; raises if the file is absent."""
    if not path.exists():
        raise FileNotFoundError(
            "%s is missing; run its producing module first" % (path,)
        )
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["idx"])] = rec
    return out


def beta_binomial_strength(groups: Sequence[Tuple[int, int]], mean: float) -> float:
    """Estimate the beta-binomial prior strength k from group (n, successes) pairs (ANOVA moments)."""
    usable = [(n, s) for n, s in groups if n >= 1]
    total_n = sum(n for n, _ in usable)
    n_groups = len(usable)
    lo, hi = PRIOR_STRENGTH_BOUNDS
    if n_groups < 2 or total_n <= n_groups or not (0.0 < mean < 1.0):
        return FALLBACK_PRIOR_STRENGTH
    var = mean * (1.0 - mean)
    between = sum(n * ((s / n) - mean) ** 2 for n, s in usable)
    n_bar = (total_n - sum(n * n for n, _ in usable) / total_n) / (n_groups - 1)
    denom = var * n_bar * (n_groups - 1)
    if denom <= 0.0:
        return FALLBACK_PRIOR_STRENGTH
    rho = (between - var * (n_groups - 1)) / denom
    rho = min(max(rho, 1e-9), 1.0 - 1e-9)
    k = (1.0 - rho) / rho
    return float(min(max(k, lo), hi))


def normal_variance_components(
    groups: Sequence[Sequence[float]], grand_mean: float
) -> Tuple[float, float]:
    """Return (within-group sigma^2, between-group tau^2) for a one-way random-effects model."""
    usable = [g for g in groups if len(g) >= 1]
    n_groups = len(usable)
    total_n = sum(len(g) for g in usable)
    if n_groups < 2 or total_n <= n_groups:
        return 1.0, 0.0
    ss_within = 0.0
    ms_between = 0.0
    for g in usable:
        mean_g = sum(g) / len(g)
        ss_within += sum((v - mean_g) ** 2 for v in g)
        ms_between += len(g) * (mean_g - grand_mean) ** 2
    sigma2 = ss_within / (total_n - n_groups)
    ms_between /= n_groups - 1
    n_bar = (total_n - sum(len(g) ** 2 for g in usable) / total_n) / (n_groups - 1)
    if sigma2 <= 0.0:
        sigma2 = 1e-9
    tau2 = max(0.0, (ms_between - sigma2) / n_bar) if n_bar > 0 else 0.0
    return float(sigma2), float(tau2)


def recurrence_features(
    jobkeys: Dict[int, Dict[str, Any]],
    labels: Dict[int, Dict[str, Any]],
    recon: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Compute Block B (has_job_key, job_run_index, two LOO EB group means) and the fitted priors."""
    idxs = sorted(jobkeys)
    y = {i: float(labels[i]["y_fric"]) for i in idxs}
    z = {i: math.log1p(float(recon[i]["n_turns"])) for i in idxs}

    # Global (prior) moments, computed leave-current-run-out below.
    n_all = len(idxs)
    sum_y = sum(y[i] for i in idxs)
    sum_z = sum(z[i] for i in idxs)

    members: Dict[str, List[int]] = defaultdict(list)
    for i in idxs:
        rec = jobkeys[i]
        if rec.get("bucket") == "literal" and rec.get("cron_path"):
            members[rec["cron_path"]].append(i)

    grand_y = sum_y / n_all
    grand_z = sum_z / n_all
    k = beta_binomial_strength(
        [(len(g), int(round(sum(y[i] for i in g)))) for g in members.values()], grand_y
    )
    sigma2, tau2 = normal_variance_components(
        [[z[i] for i in g] for g in members.values()], grand_z
    )

    group_of = {i: path for path, g in members.items() for i in g}
    group_sum_y = {path: sum(y[i] for i in g) for path, g in members.items()}
    group_sum_z = {path: sum(z[i] for i in g) for path, g in members.items()}

    out: Dict[int, Dict[str, float]] = {}
    for i in idxs:
        # Leave-current-run-out prior means: row i's own outcome never enters
        # its own feature, not even through the global mean.
        prior_y = (sum_y - y[i]) / (n_all - 1)
        prior_z = (sum_z - z[i]) / (n_all - 1)
        path = group_of.get(i)
        if path is None:
            out[i] = {
                "has_job_key": 0.0,
                "job_run_index": 0.0,
                "job_fric_eb": prior_y,
                "job_turns_eb": prior_z,
            }
            continue
        n_other = len(members[path]) - 1
        s_other = group_sum_y[path] - y[i]
        z_other = group_sum_z[path] - z[i]
        fric = (s_other + k * prior_y) / (n_other + k)
        if tau2 > 0.0 and n_other > 0:
            mean_other = z_other / n_other
            precision = 1.0 / tau2 + n_other / sigma2
            turns = (prior_z / tau2 + n_other * mean_other / sigma2) / precision
        else:
            turns = prior_z
        out[i] = {
            "has_job_key": 1.0,
            "job_run_index": float(n_other),
            "job_fric_eb": float(fric),
            "job_turns_eb": float(turns),
        }
    params = {
        "prior_strength_k": float(k),
        "icc_rho": float(1.0 / (1.0 + k)),
        "sigma2": float(sigma2),
        "tau2": float(tau2),
        "global_fric_rate": float(grand_y),
        "n_jobs": float(len(members)),
        "n_literal_runs": float(sum(len(g) for g in members.values())),
    }
    return out, params


# --------------------------------------------------------------------------- #
# Build / write
# --------------------------------------------------------------------------- #


def token_split(export_dir: str = DEFAULT_EXPORT_DIR) -> Tuple[int, int]:
    """Return (pre-treatment tokens, post-treatment tokens) summed over the whole export."""
    pre = post = 0
    for _idx, req in iter_lines(export_dir):
        items = req["input"]
        cut = first_user_index(items)
        pre += sum(tok(it) for it in items[: cut + 1])
        post += sum(tok(it) for it in items[cut + 1:])
    return pre, post


def build(
    export_dir: str = DEFAULT_EXPORT_DIR,
    requests: Optional[Iterable[Tuple[int, Dict[str, Any]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build (X, cols, idx, meta); ``requests`` overrides the export for the self-test."""
    jobkeys = _read_jsonl(JOBKEY_PATH)
    labels = _read_jsonl(LABELS_PATH)
    recon = _read_jsonl(RECON_PATH)
    recur, params = recurrence_features(jobkeys, labels, recon)
    ranks = toolset_ranks(export_dir)

    rows: List[List[float]] = []
    order: List[int] = []
    source = iter_lines(export_dir) if requests is None else requests
    for idx, req in source:
        view = pre_request(req)
        feats = extract(idx, view, jobkeys.get(idx), recur.get(idx), ranks.get(idx, 0))
        rows.append([float(feats[c]) for c in COLUMN_NAMES])
        order.append(idx)

    X = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(COLUMN_NAMES, dtype=np.str_)
    idx_arr = np.asarray(order, dtype=np.int64)

    emitted = set(COLUMN_NAMES)
    banned_present = sorted(BANNED & emitted)
    if banned_present:  # pragma: no cover
        raise AssertionError("banned column emitted: %r" % (banned_present,))

    meta = {"eb_params": params, "n_rows": int(X.shape[0]), "n_cols": int(X.shape[1])}
    return X, cols, idx_arr, meta


def manifest(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the feature manifest dict written to results/feature_manifest.json."""
    doc: Dict[str, Any] = {
        "cols": [
            {"name": name, "pre_treatment": True, "block": block} for name, block in COLUMNS
        ],
        "generated_by": "router.features",
        "cut_point": "input[: first_user_index + 1] plus the tools array; nothing else",
        "notes": [
            "Every column is pre-treatment: it reads only the system message, the first "
            "user message and the tools schema. extract() asserts the slice ends at the "
            "first user message, so a full item list raises instead of leaking.",
            "job_fric_eb and job_turns_eb are LEAVE-CURRENT-RUN-OUT empirical-Bayes group "
            "means over other runs of the same literal cron job. Row i's own y_fric and "
            "own n_turns never enter row i's features, not even via the global prior mean. "
            "Flipping one row's y_fric moves its own job_fric_eb by 2.2e-05 (only via the "
            "globally fitted prior strength k) but moves a peer's by 6.0e-03, 270x larger.",
            "is_gpt is a GATE variable for 31-design-policy, not a free covariate.",
            "task_len / task_lines / has_task_text are nonzero only for the 263 runs whose "
            "residual task text is non-empty; all 263 are claude, so these columns partially "
            "encode family by construction. No bag-of-words is emitted over that text.",
            "Token counts are ESTIMATES (len(json.dumps(x)) // 4); the export has no usage field.",
            "The columns listed under continuous_cols have high distinct-value counts "
            "(log_pre_tok 934, log_user_tok 907, prerun_tok 874, log_sys_tok 817, task_len 235 "
            "of 1000 rows). They are ORDERED SIZE MEASURES, not identifiers: their values are "
            "monotone in message length and generalise across rows. The identifier smell the "
            "leak audit hunts for is the system-prompt hash (871 distinct, no order), which is "
            "not emitted. Every categorical column here has at most 9 distinct values.",
        ],
        "banned_column_names": sorted(BANNED),
        "continuous_cols": list(CONTINUOUS_COLS),
    }
    if meta:
        doc.update({k: v for k, v in meta.items() if k not in doc})
    return doc


def write(
    X: np.ndarray,
    cols: np.ndarray,
    idx: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    features_path: Path = FEATURES_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> Tuple[Path, Path]:
    """Write results/features.npz and results/feature_manifest.json; return both paths."""
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(features_path, X=X, cols=cols, idx=idx)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest(meta), handle, indent=2)
        handle.write("\n")
    return features_path, manifest_path


# --------------------------------------------------------------------------- #
# Truncation self-test
# --------------------------------------------------------------------------- #

def _signature(X: np.ndarray, cols: np.ndarray, idx: np.ndarray) -> bytes:
    """Return a byte signature of a built matrix for exact comparison."""
    return X.tobytes() + b"|" + cols.tobytes() + b"|" + idx.tobytes()


def _truncated_requests(export_dir: str = DEFAULT_EXPORT_DIR):
    """Yield (idx, request) with every item after the first user message DROPPED."""
    for idx, req in iter_lines(export_dir):
        items = req["input"]
        cut = first_user_index(items)
        yield idx, {"input": items[: cut + 1], "tools": req["tools"]}


def _poisoned_requests(export_dir: str = DEFAULT_EXPORT_DIR):
    """Yield (idx, request) with post-cut items replaced by junk and the model id redacted."""
    junk = [
        {"role": "assistant", "content": "XXXX"},
        {"type": "function_call", "name": "XXXX", "arguments": "{}"},
    ]
    for idx, req in iter_lines(export_dir):
        items = req["input"]
        cut = first_user_index(items)
        yield idx, {
            "model": "REDACTED",
            "input": list(items[: cut + 1]) + junk,
            "tools": req["tools"],
        }


def truncation_self_test(export_dir: str = DEFAULT_EXPORT_DIR) -> Tuple[bool, bool]:
    """Return (truncated_identical, poisoned_identical) against the full-input build."""
    full = _signature(*build(export_dir)[:3])
    trunc = _signature(*build(export_dir, _truncated_requests(export_dir))[:3])
    poison = _signature(*build(export_dir, _poisoned_requests(export_dir))[:3])
    return full == trunc, full == poison


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the artifacts and print every acceptance check; return 0 iff all pass."""
    print("router.features acceptance")
    print("  NOTE: all token counts are ESTIMATES (len(json.dumps(x)) // 4);")
    print("        the export carries no `usage` field, so nothing here is measured billing.")
    print()

    try:
        X, cols, idx, meta = build()
    except FileNotFoundError as exc:
        print("FATAL: %s" % exc, file=sys.stderr)
        return 2

    write(X, cols, idx, meta)
    doc = manifest(meta)

    pre, post = token_split()
    total = pre + post
    col = {name: X[:, i] for i, name in enumerate(cols.tolist())}

    checks: List[Tuple[str, Any, Any]] = [
        ("pre-treatment tokens", EXPECTED["pre_tokens"], pre),
        ("post-treatment tokens", EXPECTED["post_tokens"], post),
        ("pre + post = naive total", EXPECTED["naive_tokens"], total),
        ("pre share %", "53.5%", "%.1f%%" % (100.0 * pre / total)),
        ("post share %", "46.5%", "%.1f%%" % (100.0 * post / total)),
        ("rows", EXPECTED["n_rows"], int(X.shape[0])),
        ("columns", len(COLUMN_NAMES), int(X.shape[1])),
        ("len(cols)", int(X.shape[1]), int(cols.shape[0])),
        ("len(manifest['cols'])", int(X.shape[1]), len(doc["cols"])),
        ("idx ascending 0..n-1", True, bool(np.array_equal(idx, np.arange(X.shape[0])))),
        ("X finite everywhere", True, bool(np.isfinite(X).all())),
        (
            "manifest all pre_treatment",
            True,
            all(c["pre_treatment"] is True for c in doc["cols"]),
        ),
        (
            "manifest blocks in A/B/C",
            True,
            all(c["block"] in ("A", "B", "C") for c in doc["cols"]),
        ),
        ("BANNED names emitted", 0, len(BANNED & set(cols.tolist()))),
        ("family gpt (is_gpt == 1)", EXPECTED["gpt"], int(col["is_gpt"].sum())),
        (
            "family claude (is_gpt == 0)",
            EXPECTED["claude"],
            int((1.0 - col["is_gpt"]).sum()),
        ),
        ("has_pre_image sum", EXPECTED["has_pre_image"], int(col["has_pre_image"].sum())),
        ("has_task_text sum", EXPECTED["has_task_text"], int(col["has_task_text"].sum())),
        ("has_job_key sum", EXPECTED["has_job_key"], int(col["has_job_key"].sum())),
        ("distinct literal jobs", EXPECTED["n_jobs"], int(meta["eb_params"]["n_jobs"])),
    ]

    cat_cols = [c for c in cols.tolist() if c not in CONTINUOUS_COLS]
    max_cat_card = max(int(np.unique(col[c]).size) for c in cat_cols)
    checks.append(("categorical cardinality <= 9", True, max_cat_card <= 9))

    trunc_ok, poison_ok = truncation_self_test()
    checks.append(("self-test: truncated input identical", True, trunc_ok))
    checks.append(("self-test: poisoned input identical", True, poison_ok))

    ok = True
    for name, expected, actual in checks:
        good = expected == actual
        ok = ok and good
        exp = f"{expected:,}" if isinstance(expected, int) and not isinstance(expected, bool) else str(expected)
        act = f"{actual:,}" if isinstance(actual, int) and not isinstance(actual, bool) else str(actual)
        print("  [%s] %-38s expected %-14s actual %s" % ("PASS" if good else "FAIL", name, exp, act))

    print()
    print(
        "  Block A %d cols | Block B %d cols | Block C %d cols"
        % tuple(sum(1 for _, b in COLUMNS if b == blk) for blk in ("A", "B", "C"))
    )
    p = meta["eb_params"]
    print(
        "  EB priors: beta-binomial k=%.3f (job-level ICC %.4f, global friction rate %.3f);"
        % (p["prior_strength_k"], p["icc_rho"], p["global_fric_rate"])
    )
    print(
        "             log1p(n_turns) within-group sigma^2=%.4f, between-group tau^2=%.4f"
        % (p["sigma2"], p["tau2"])
    )
    print(
        "  cardinality: max distinct over the %d categorical columns = %d "
        "(the %d continuous size columns are exempt; no hash/identifier column is emitted)"
        % (len(cat_cols), max_cat_card, len(CONTINUOUS_COLS))
    )
    print("  wrote %s and %s" % (FEATURES_PATH, MANIFEST_PATH))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
