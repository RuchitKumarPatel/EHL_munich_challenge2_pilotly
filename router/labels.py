"""Constructed outcome signal: friction labels from tool-output exit codes.

WHAT THIS COMPUTES
    For every trajectory (= one line of the export, which is one COMPLETE run, not one
    API call) this module pairs every tool call to its tool output *within that line*,
    resolves each output to a structural status ('ok' | 'err' | 'unknown') from exit
    codes and structured error fields ONLY, and reduces the per-line status sequence to
    the outcome labels used by every downstream estimator.

    Status resolution is structural on purpose. Text-keyword error detection was measured
    against this exit-code ground truth and scores 2.8%-49.3% precision; see
    banned_keyword_crosstab(), which is a *printed regression test* kept in this module so
    that nobody re-adds keyword detection at hour 30 of the build.

    'unknown' outputs are EXCLUDED from the denominator and are NEVER imputed as success.
    Imputing them would bias the label against Slack / upload / read-style tools that
    never emit an exit code (787 of 10,422 outputs, concentrated in exactly those tools).

ARTIFACT WRITTEN
    results/labels.jsonl -- one JSON object per input line, in file order:
      {"idx": int, "n_calls": int, "n_obs": int, "n_err": int, "n_unknown": int,
       "trailing_err": bool, "y_fric": 0|1, "y_proc": 0|1|2|null,
       "y_rate": float|null, "n_kill": int}

ACCEPTANCE NUMBERS (verified against export/trajectories_v1_01.jsonl, n=1000 lines)
    lines out                1000, idx ascending 0..999
    tool-call/output pairs   10,422  (10,123 function_call + 299 custom_tool_call)
    resolvable outputs        9,635 = 92.4%   [brief quotes 9,633; see NOTE below]
    errors                      474
    y_proc counts            {0: 737, 1: 241, 2: 18, None: 4}
    y_fric positives            259
    trailing_err lines           18
    n_kill total                 84
    keyword precision vs exit-code truth: 2.8% (rate-limit) .. 49.3% (traceback)

    NOTE on the two-output gap: 474 errors and 9,633 resolvable cannot both hold. Pairing
    is a bijection here (every one of the 10,422 outputs is consumed exactly once), so the
    status counts are a property of the output items themselves: 9,161 ok + 474 err + 787
    unknown = 10,422. Holding the brief's 474 errors fixed forces 9,635 resolvable. The
    two schemes that would yield 9,633 (first-wins / last-wins dict pairing over the 14
    duplicated call_ids) break the error count instead (475 / 473). Every other acceptance
    number -- pairs, errors, y_proc, y_fric, keyword precision range -- reproduces exactly,
    so the count is reported as NEAR (+2) rather than silently reshaped to match.

Runnable:   python -m router.labels        (prints acceptance, non-zero exit on FAIL)
Importable: tool_status, pair_tool_calls, label_line, build_labels, banned_keyword_crosstab
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "export"
RESULTS_DIR = REPO_ROOT / "results"
LABELS_PATH = RESULTS_DIR / "labels.jsonl"

CALL_TYPES = ("function_call", "custom_tool_call")
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")

#: Exit codes that mean the call was killed rather than merely unsuccessful:
#: 124/125 GNU timeout, 128+n signal conventions, 130 SIGINT, 137 SIGKILL, 143 SIGTERM.
KILL_CODES = frozenset({124, 125, 128, 129, 130, 137, 143})

#: Rule 4 of tool_status: a plain-text exit line, anchored per line.
EXIT_CODE_RE = re.compile(r"^Exit code:\s*(-?\d+)", re.MULTILINE)

#: BANNED as a label source. Kept only for banned_keyword_crosstab()'s regression print.
BANNED_KEYWORD_PATTERNS = {
    "rate_limit": r"rate.?limit|429",
    "timeout": r"timed? ?out|TimeoutError",
    "permission": r"Permission denied|EACCES|403|Unauthorized",
    "not_found": r"No such file|not found|ENOENT",
    "traceback": r"Traceback \(most recent call last\)",
}

EXPECTED = {
    "n_lines": 1000,
    "n_pairs": 10422,
    "n_function_call_pairs": 10123,
    "n_custom_tool_call_pairs": 299,
    "n_obs": 9633,  # brief's figure; this module measures 9635 -- see module docstring
    "n_err": 474,
    "y_proc": {"0": 737, "1": 241, "2": 18, "null": 4},
    "y_fric": 259,
}


# --------------------------------------------------------------------------- reading


def iter_lines(export_dir=EXPORT_DIR):
    """Yield (idx, request) for every non-blank line of every export chunk, in file order."""
    chunks = sorted(Path(export_dir).glob("*.jsonl"))
    if not chunks:
        raise FileNotFoundError(f"no *.jsonl chunks found in {export_dir}")
    idx = 0
    for path in chunks:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    yield idx, json.loads(line)
                    idx += 1


# ------------------------------------------------------------------- status resolution


def _as_dict(output):
    """Return the tool output parsed as a dict, or None if it is not a JSON object."""
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return None
    try:
        parsed = json.loads(output)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_text(output):
    """Return the raw text of a tool output for regex rules."""
    return output if isinstance(output, str) else json.dumps(output)


def _coerce_code(value):
    """Return an exit code as int, or None if it is not integer-like."""
    if isinstance(value, bool) or not isinstance(value, (int, str, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_output(output):
    """Resolve ONE tool output to (status, exit_code|None) in strict rule priority order."""
    payload = _as_dict(output)
    if payload is not None:
        # 1. structured exit code wins over everything else.
        if "exit_code" in payload:
            code = _coerce_code(payload["exit_code"])
            if code is None:
                return "err", None  # present but not an integer -> not a clean exit
            return ("ok" if code == 0 else "err"), code
        # 2. explicit non-empty string error field.
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return "err", None
        # 3. explicit boolean success flag.
        success = payload.get("success")
        if success is False:
            return "err", None
        if success is True:
            return "ok", None
    # 4. plain-text exit line.
    match = EXIT_CODE_RE.search(_as_text(output))
    if match:
        code = int(match.group(1))
        return ("ok" if code == 0 else "err"), code
    # 5. no structural status exists. NOT imputed as success.
    return "unknown", None


def tool_status(output):
    """Resolve ONE tool output to 'ok', 'err' or 'unknown' (never imputes success)."""
    return resolve_output(output)[0]


def exit_code_of(output):
    """Return the structural exit code of ONE tool output, or None if it carries none."""
    return resolve_output(output)[1]


def is_kill_code(code):
    """True iff an exit code denotes a killed / timed-out call."""
    return code is not None and code in KILL_CODES


# ------------------------------------------------------------------------- pairing


def pair_tool_calls(items):
    """Pair each tool call to its output by call_id WITHIN one line, in call order.

    call_id is a per-request counter (only 234 distinct strings across the whole corpus),
    so it is meaningless across lines and is never used as a join key. A call_id can also
    repeat inside a single line, so the k-th call with a given id takes the k-th output
    with that id -- a dict keyed on call_id would silently mispair those.
    """
    outputs = defaultdict(list)
    calls = []
    for item in items:
        kind = item.get("type")
        if kind in CALL_TYPES:
            calls.append(item)
        elif kind in OUTPUT_TYPES:
            outputs[item["call_id"]].append(item)
    seen = Counter()
    pairs = []
    for call in calls:
        cid = call["call_id"]
        rank = seen[cid]
        seen[cid] += 1
        bucket = outputs.get(cid, ())
        if rank < len(bucket):
            pairs.append((call, bucket[rank]))
    return pairs


# -------------------------------------------------------------------------- labelling


def label_line(idx, request):
    """Build the label record for one trajectory line."""
    items = request.get("input", [])
    calls = [it for it in items if it.get("type") in CALL_TYPES]
    pairs = pair_tool_calls(items)

    statuses = []
    n_kill = 0
    for _call, out in pairs:
        status, code = resolve_output(out["output"])
        statuses.append(status)
        if status == "err" and is_kill_code(code):
            n_kill += 1

    resolvable = [s for s in statuses if s != "unknown"]
    n_obs = len(resolvable)
    n_err = sum(1 for s in resolvable if s == "err")
    n_unknown = len(statuses) - n_obs
    trailing_err = bool(resolvable) and resolvable[-1] == "err"

    if trailing_err:
        y_proc = 2
    elif n_err > 0:
        y_proc = 1
    elif n_obs > 0:
        y_proc = 0
    else:
        y_proc = None

    return {
        "idx": idx,
        "n_calls": len(calls),
        "n_obs": n_obs,
        "n_err": n_err,
        "n_unknown": n_unknown,
        "trailing_err": trailing_err,
        "y_fric": 1 if n_err > 0 else 0,
        "y_proc": y_proc,
        "y_rate": (n_err / n_obs) if n_obs > 0 else None,
        "n_kill": n_kill,
    }


def build_labels(export_dir=EXPORT_DIR):
    """Label every trajectory in the export, in file order."""
    return [label_line(idx, req) for idx, req in iter_lines(export_dir)]


def write_labels(rows, path=LABELS_PATH):
    """Write label records to results/labels.jsonl, one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


# ------------------------------------------------- banned-keyword regression test


def banned_keyword_crosstab(export_dir=EXPORT_DIR, patterns=None):
    """Cross-tab banned text keywords against exit-code ground truth (regression test)."""
    patterns = patterns or BANNED_KEYWORD_PATTERNS
    compiled = {name: re.compile(pat, re.IGNORECASE) for name, pat in patterns.items()}
    table = {name: Counter() for name in compiled}
    for _idx, req in iter_lines(export_dir):
        for _call, out in pair_tool_calls(req.get("input", [])):
            status = tool_status(out["output"])
            if status == "unknown":
                continue  # never in the denominator, here neither
            text = _as_text(out["output"])
            truth = status == "err"
            for name, rx in compiled.items():
                table[name][(bool(rx.search(text)), truth)] += 1
    report = {}
    for name, cell in table.items():
        tp, fp = cell[(True, True)], cell[(True, False)]
        fn, tn = cell[(False, True)], cell[(False, False)]
        report[name] = {
            "pattern": patterns[name],
            "hits": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "recall": (tp / (tp + fn)) if (tp + fn) else None,
        }
    return report


def print_banned_keyword_crosstab(report):
    """Print the keyword-vs-exit-code cross-tab and why keywords stay banned."""
    print("BANNED-KEYWORD REGRESSION TEST (keywords vs exit-code ground truth)")
    print(f"  {'pattern':<12} {'hits':>6} {'true err':>9} {'false err':>10} "
          f"{'precision':>10} {'recall':>8}")
    for name, row in report.items():
        prec = "n/a" if row["precision"] is None else f"{row['precision'] * 100:.1f}%"
        rec = "n/a" if row["recall"] is None else f"{row['recall'] * 100:.1f}%"
        print(f"  {name:<12} {row['hits']:>6} {row['tp']:>9} {row['fp']:>10} "
              f"{prec:>10} {rec:>8}")
    precs = [r["precision"] for r in report.values() if r["precision"] is not None]
    if precs:
        print(f"  -> precision spans {min(precs) * 100:.1f}%..{max(precs) * 100:.1f}%; "
              f"keyword hits are majority NON-errors. Keywords may never create n_err.")


# ------------------------------------------------------------------------ acceptance


def _check(name, expected, actual, results, near=False):
    """Record and print one acceptance check as PASS / NEAR / FAIL."""
    ok = expected == actual
    tag = "PASS" if ok else ("NEAR" if near else "FAIL")
    print(f"  [{tag}] {name:<34} expected={expected!s:<38} actual={actual!s}")
    results.append((name, ok, tag))
    return ok


def acceptance(rows, pair_kinds, report):
    """Print every label acceptance check with PASS/FAIL; return True iff none FAILed."""
    results = []
    n_obs = sum(r["n_obs"] for r in rows)
    n_err = sum(r["n_err"] for r in rows)
    n_unknown = sum(r["n_unknown"] for r in rows)
    n_pairs = n_obs + n_unknown
    proc = Counter("null" if r["y_proc"] is None else str(r["y_proc"]) for r in rows)

    print("LABEL ACCEPTANCE  (token/status counts are structural, not usage-reported)")
    _check("lines written", EXPECTED["n_lines"], len(rows), results)
    _check("idx ascending 0..n-1", True,
           [r["idx"] for r in rows] == list(range(len(rows))), results)
    _check("tool call/output pairs", EXPECTED["n_pairs"], n_pairs, results)
    _check("  function_call pairs", EXPECTED["n_function_call_pairs"],
           pair_kinds["function_call"], results)
    _check("  custom_tool_call pairs", EXPECTED["n_custom_tool_call_pairs"],
           pair_kinds["custom_tool_call"], results)
    _check("resolvable outputs", EXPECTED["n_obs"], n_obs, results, near=True)
    _check("resolvable share of pairs", "92.4%", f"{100 * n_obs / n_pairs:.1f}%", results)
    _check("errors", EXPECTED["n_err"], n_err, results)
    _check("y_proc counts", EXPECTED["y_proc"], dict(sorted(proc.items())), results)
    _check("y_fric positives", EXPECTED["y_fric"], sum(r["y_fric"] for r in rows), results)
    _check("y_fric == (n_err>0)", True,
           all(r["y_fric"] == (1 if r["n_err"] > 0 else 0) for r in rows), results)
    _check("y_rate None iff n_obs==0", True,
           all((r["y_rate"] is None) == (r["n_obs"] == 0) for r in rows), results)
    _check("every call paired", sum(r["n_calls"] for r in rows), n_pairs, results)
    _check("y_proc==2 iff trailing_err", True,
           all((r["y_proc"] == 2) == r["trailing_err"] for r in rows), results)

    print(f"  [INFO] unknown outputs            {n_unknown} "
          f"({100 * n_unknown / n_pairs:.1f}% of pairs, excluded from the denominator)")
    print(f"  [INFO] trailing_err lines         {sum(1 for r in rows if r['trailing_err'])}")
    print(f"  [INFO] killed/timed-out calls     {sum(r['n_kill'] for r in rows)} "
          f"(exit codes in {sorted(KILL_CODES)})")
    print()
    print_banned_keyword_crosstab(report)

    failed = [n for n, ok, tag in results if tag == "FAIL"]
    near = [n for n, ok, tag in results if tag == "NEAR"]
    print()
    if near:
        print(f"NEAR (documented, see module docstring): {', '.join(near)}")
    print("RESULT: " + ("all label checks PASS" if not failed
                        else f"{len(failed)} FAILED: {', '.join(failed)}"))
    return not failed


def main(argv=None):
    """Build labels, write results/labels.jsonl and print the acceptance report."""
    argv = sys.argv[1:] if argv is None else argv
    export_dir = Path(argv[0]) if argv else EXPORT_DIR

    rows = []
    pair_kinds = Counter()
    for idx, req in iter_lines(export_dir):
        rows.append(label_line(idx, req))
        for call, _out in pair_tool_calls(req.get("input", [])):
            pair_kinds[call["type"]] += 1

    path = write_labels(rows)
    print(f"wrote {path} ({len(rows)} lines)")
    report = banned_keyword_crosstab(export_dir)
    return 0 if acceptance(rows, pair_kinds, report) else 1


if __name__ == "__main__":
    sys.exit(main())
