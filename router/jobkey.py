#!/usr/bin/env python3
"""Recurrence key for the Viktor trajectory export: the literal cron job path.

WHAT THIS COMPUTES
    Each of the 1000 exported runs is an independent trajectory (there are no
    trajectory ids and no cross-line chains).  To compare arms we need to know
    which runs are *repeats of the same job*.  The only honest key in the data
    is the literal cron path that the harness prints into the first user
    message, e.g.

        Cron path: `crons/heartbeat/`

    extracted with  Cron path:\\s*`?([A-Za-z0-9_\\-/\\.]+)  and then bucketed:

        "literal" — a path was found and contains NO 'PII_' substring
        "pii"     — a path was found but carries a PII_ placeholder
                    (we report it, we never key on it)
        "none"    — no cron path in the first user message at all

    Only the "literal" bucket is ever used as a join key.  Everything downstream
    (the cross-family friction comparison in particular) is restricted to it.

WHY NOT THE OBVIOUS FIELD — the banned alternative
    The first user message also contains a line of the form

        # === Cron memory: PII_URL_3

    It looks like a job identifier and it is not.  Measured on this export:
    763 runs carry that field, 763/763 of the values are PII_ placeholders,
    there are only 18 distinct values in total, and a single value (PII_URL_3)
    covers 306 runs.  Worse, PII_ placeholders are renumbered per request, so
    the same token does not denote the same entity across rows.  Grouping on it
    would fuse hundreds of unrelated jobs into one bogus "job" and manufacture
    cross-arm comparisons that do not exist.  THE '# === Cron memory:' FIELD
    MUST NEVER BE USED AS A KEY — nor may any other PII_* token.
    This module replaces it.

    (Equally banned and equally not used here:
     scripts/load_trajectories.py's group_trajectories(), which hashes only the
     first 2000 chars of the first user message.  978/1000 first user messages
     exceed 2000 chars of shared boilerplate, so it collapses n=1000 into 953
     collision groups.  We call iter_requests() and first_user_text() only.)

ARTIFACT WRITTEN
    results/jobkey.jsonl — one object per export line, in file order:
        {"idx": int, "cron_path": str|null, "literal": bool,
         "bucket": "literal"|"pii"|"none"}

ACCEPTANCE NUMBERS (ground truth for this export)
    784/1000 runs carry a literal "Cron path:" line
    354 distinct paths
    152 fully-literal jobs covering 374 runs
      of those, 28 span >= 2 arms  (245 runs)
      and       23 span BOTH families (229 runs)
    216 runs have no cron path
    410 runs have a PII_ placeholder inside the path
    largest clean job: crons/heartbeat/  n=134 across 7 arms

    Note 134 of the 229 cross-family runs are that single job, so any
    cross-family contrast built on this key is really one job plus a tail.

Run:     python -m router.jobkey       (prints every check with PASS/FAIL)
Import:  extract_path, bucket_of, load_jobkeys, clean_jobs,
         multi_arm_jobs, cross_family_jobs, build_records, write_jobkeys
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO_ROOT / "export"
JOBKEY_PATH = REPO_ROOT / "results" / "jobkey.jsonl"
RECON_PATH = REPO_ROOT / "results" / "recon.jsonl"

CRON_PATH_RE = re.compile(r"Cron path:\s*`?([A-Za-z0-9_\-/\.]+)")

#: tools that separate the two model families with 0 exceptions on this export
GPT_TOOL_NAMES = frozenset({"apply_patch", "shell_command"})
CLAUDE_TOOL_NAMES = frozenset({"bash", "file_read", "file_edit", "file_write"})

EXPECTED = {
    "n_runs": 1000,
    "n_with_path": 784,
    "n_distinct_paths": 354,
    "n_clean_jobs": 152,
    "n_clean_runs": 374,
    "n_multi_arm_jobs": 28,
    "n_multi_arm_runs": 245,
    "n_cross_family_jobs": 23,
    "n_cross_family_runs": 229,
    "n_bucket_none": 216,
    "n_bucket_pii": 410,
    "largest_clean_job": "crons/heartbeat/",
    "largest_clean_job_n": 134,
    "largest_clean_job_arms": 7,
}


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def extract_path(text: str) -> str | None:
    """Return the literal cron path in `text`, or None if there is no Cron path line."""
    if not text:
        return None
    m = CRON_PATH_RE.search(text)
    return m.group(1) if m else None


def bucket_of(path: str | None) -> str:
    """Classify a cron path into exactly one of 'literal', 'pii', 'none'."""
    if path is None:
        return "none"
    return "pii" if "PII_" in path else "literal"


def family_of(request: dict) -> str:
    """Return 'gpt' or 'claude' from the request's tool block (perfect separation)."""
    names = set()
    for tool in request.get("tools") or []:
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if name:
            names.add(name)
    if names & GPT_TOOL_NAMES:
        return "gpt"
    if names & CLAUDE_TOOL_NAMES:
        return "claude"
    return "unknown"


# --------------------------------------------------------------------------- #
# building / io
# --------------------------------------------------------------------------- #
def _organizer_loader():
    """Import the organizers' loader and return (iter_requests, first_user_text)."""
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from load_trajectories import first_user_text, iter_requests

    return iter_requests, first_user_text


def _iter_export(export_dir: Path | str):
    """Yield (idx, request) over the export, in file order (never group_trajectories)."""
    iter_requests, _ = _organizer_loader()
    for idx, (_chunk, _lineno, request) in enumerate(iter_requests(str(export_dir))):
        yield idx, request


def build_records(export_dir: Path | str = EXPORT_DIR) -> list[dict]:
    """Scan the export and return one jobkey record per line, in file order."""
    _, first_user_text = _organizer_loader()
    records = []
    for idx, request in _iter_export(export_dir):
        path = extract_path(first_user_text(request))
        bucket = bucket_of(path)
        records.append(
            {"idx": idx, "cron_path": path, "literal": bucket == "literal", "bucket": bucket}
        )
    return records


def scan_arms(export_dir: Path | str = EXPORT_DIR) -> dict[int, dict]:
    """Return {idx: {'model':..., 'family':...}} by scanning the export."""
    return {
        idx: {"model": request["model"], "family": family_of(request)}
        for idx, request in _iter_export(export_dir)
    }


def load_arms(recon_path: Path | str = RECON_PATH, export_dir: Path | str = EXPORT_DIR) -> dict[int, dict]:
    """Return {idx: {'model','family'}}, reading results/recon.jsonl if present else the export."""
    recon_path = Path(recon_path)
    if recon_path.exists():
        arms = {}
        with open(recon_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    arms[rec["idx"]] = {"model": rec["model"], "family": rec["family"]}
        if arms:
            return arms
    return scan_arms(export_dir)


def write_jobkeys(records: list[dict], out_path: Path | str = JOBKEY_PATH) -> Path:
    """Write the jobkey records to results/jobkey.jsonl and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return out_path


def load_jobkeys(path: Path | str = JOBKEY_PATH) -> dict[int, dict]:
    """Load results/jobkey.jsonl as {idx: record}; builds it from the export if missing."""
    path = Path(path)
    if not path.exists():
        records = build_records()
        write_jobkeys(records, path)
        return {r["idx"]: r for r in records}
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["idx"]] = rec
    return out


# --------------------------------------------------------------------------- #
# job views (literal bucket only)
# --------------------------------------------------------------------------- #
def clean_jobs(jobkeys: dict[int, dict] | None = None) -> dict[str, list[int]]:
    """Return {cron_path: [idx,...]} for the literal bucket only, indices sorted."""
    jobkeys = load_jobkeys() if jobkeys is None else jobkeys
    jobs: dict[str, list[int]] = defaultdict(list)
    for idx, rec in jobkeys.items():
        if rec["bucket"] == "literal":
            jobs[rec["cron_path"]].append(idx)
    return {path: sorted(ixs) for path, ixs in sorted(jobs.items())}


def multi_arm_jobs(
    jobkeys: dict[int, dict] | None = None, arms: dict[int, dict] | None = None
) -> dict[str, list[int]]:
    """Clean jobs whose runs were served by >= 2 distinct model ids."""
    arms = load_arms() if arms is None else arms
    return {
        path: ixs
        for path, ixs in clean_jobs(jobkeys).items()
        if len({arms[i]["model"] for i in ixs}) >= 2
    }


def cross_family_jobs(
    jobkeys: dict[int, dict] | None = None, arms: dict[int, dict] | None = None
) -> dict[str, list[int]]:
    """Clean jobs whose runs span BOTH model families (claude and gpt)."""
    arms = load_arms() if arms is None else arms
    return {
        path: ixs
        for path, ixs in clean_jobs(jobkeys).items()
        if len({arms[i]["family"] for i in ixs}) >= 2
    }


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #
def _check(rows: list, label: str, expected, actual) -> bool:
    ok = expected == actual
    rows.append((ok, label, expected, actual))
    return ok


def acceptance(jobkeys: dict[int, dict], arms: dict[int, dict]) -> list[tuple]:
    """Return [(pass, label, expected, actual), ...] for every jobkey acceptance number."""
    rows: list[tuple] = []
    buckets = Counter(r["bucket"] for r in jobkeys.values())
    paths = Counter(r["cron_path"] for r in jobkeys.values() if r["cron_path"])
    jobs = clean_jobs(jobkeys)
    multi = multi_arm_jobs(jobkeys, arms)
    cross = cross_family_jobs(jobkeys, arms)
    biggest = max(jobs.items(), key=lambda kv: len(kv[1]))

    E = EXPECTED
    _check(rows, "runs total", E["n_runs"], len(jobkeys))
    _check(rows, 'runs with a literal "Cron path:" line', E["n_with_path"], sum(paths.values()))
    _check(rows, "distinct cron paths", E["n_distinct_paths"], len(paths))
    _check(rows, "fully-literal jobs", E["n_clean_jobs"], len(jobs))
    _check(rows, "runs in fully-literal jobs", E["n_clean_runs"], sum(len(v) for v in jobs.values()))
    _check(rows, "clean jobs spanning >=2 arms", E["n_multi_arm_jobs"], len(multi))
    _check(rows, "runs in those multi-arm jobs", E["n_multi_arm_runs"], sum(len(v) for v in multi.values()))
    _check(rows, "clean jobs spanning BOTH families", E["n_cross_family_jobs"], len(cross))
    _check(rows, "runs in those cross-family jobs", E["n_cross_family_runs"], sum(len(v) for v in cross.values()))
    _check(rows, "runs with no cron path (bucket=none)", E["n_bucket_none"], buckets["none"])
    _check(rows, "runs with a PII_ path (bucket=pii)", E["n_bucket_pii"], buckets["pii"])
    _check(rows, "largest clean job", E["largest_clean_job"], biggest[0])
    _check(rows, "largest clean job, n runs", E["largest_clean_job_n"], len(biggest[1]))
    _check(rows, "largest clean job, distinct arms", E["largest_clean_job_arms"],
           len({arms[i]["model"] for i in biggest[1]}))
    _check(rows, "buckets partition the runs", E["n_runs"],
           buckets["literal"] + buckets["pii"] + buckets["none"])
    _check(rows, "no PII_ token survives into a job key", 0,
           sum(1 for p in jobs if "PII_" in p))
    return rows


def main() -> int:
    """Rebuild results/jobkey.jsonl, print every acceptance check and the cross-family table."""
    records = build_records(EXPORT_DIR)
    out = write_jobkeys(records, JOBKEY_PATH)
    jobkeys = {r["idx"]: r for r in records}
    arms = scan_arms(EXPORT_DIR)

    print(f"router.jobkey — wrote {out} ({len(records)} records)")
    print("key = literal cron path from the first user message; the "
          "'# === Cron memory: PII_URL_N' field is never used (763/763 PII, 18 distinct values).")
    print()
    print("ACCEPTANCE")
    rows = acceptance(jobkeys, arms)
    width = max(len(r[1]) for r in rows)
    for ok, label, exp, act in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}  expected={exp}  actual={act}")
    n_fail = sum(1 for r in rows if not r[0])
    print(f"  {len(rows) - n_fail}/{len(rows)} checks passed")
    print()

    cross = cross_family_jobs(jobkeys, arms)
    print("CLEAN CROSS-FAMILY JOBS (literal key, runs on both families)")
    print(f"  {'n':>4} {'claude':>7} {'gpt':>5} {'arms':>5}  path")
    for path, ixs in sorted(cross.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fam = Counter(arms[i]["family"] for i in ixs)
        n_arms = len({arms[i]["model"] for i in ixs})
        print(f"  {len(ixs):>4} {fam['claude']:>7} {fam['gpt']:>5} {n_arms:>5}  {path}")
    tot = sum(len(v) for v in cross.values())
    big = max(cross.items(), key=lambda kv: len(kv[1]))
    print(f"  {len(cross)} jobs, {tot} runs; {len(big[1])} of them ({len(big[1]) / tot:.1%}) "
          f"are the single job {big[0]} — treat the pooled contrast accordingly.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
