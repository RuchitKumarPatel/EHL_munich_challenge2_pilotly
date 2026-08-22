"""Turn segmentation and cache decomposition — the cost foundation of the project.

WHAT IT COMPUTES
A JSONL line in the export is NOT one API call. It is a whole trajectory, and the
provider billed it as N calls, each of which re-sent the entire grown prefix. The
starter kit counts ``sum(tok(item) for item in input)`` (22.6M tokens); the real
billed volume is ~15x that, because the prefix is re-sent every turn.

  turn boundary  = the START of every maximal run of model-produced items, where an
                   item is model-produced iff  item["role"] == "assistant"  OR
                   item["type"] in {"reasoning", "function_call", "custom_tool_call"}.
  billed prefix of the turn starting at index i
                 = tools_tok + sum(tok(items[:i]))     # per-item tok, summed
  cache split    = with P the list of billed prefixes in order,
                   cache_read  = sum(P[:-1])           # every prefix but the last was
                                                       # already on the wire before
                   cache_write = sum(P) - cache_read

THE ONE DETAIL THAT MUST NEVER REGRESS
The tools block is billed on EVERY turn, not once per line. It is 44,720,796 tokens
— 13.4% of the entire bill. ``main()`` therefore also prints the excluding-tools
total so that gap stays visible forever.

All token counts are ESTIMATES (``len(json.dumps(x)) // 4``); the export has no
``usage`` field. Any user-facing string built on these numbers must say so.

ARTIFACT
  results/recon.jsonl — one object per export line, in file order:
    {"idx", "model", "family", "n_turns", "turn_cuts", "prefix_tokens", "tools_tok",
     "pre_tok", "gross_tok", "cache_read_tok", "cache_write_tok", "naive_tok"}

ACCEPTANCE (all verified, printed by ``python -m router.recon``)
  n_turns total                        10,845
  sum(gross_tok)                  334,729,910
  sum(gross_tok) excluding tools  290,009,114   (gap 44,720,796)
  sum(cache_read_tok)             308,074,571   (92.0%)
  sum(cache_write_tok)             26,655,339   ( 8.0%)
  sum(naive_tok)                   22,631,879
  turns/line by arm: opus-5 10.36, sonnet-5 11.83, sol 10.85, terra 8.51,
                     fable-5 11.72, opus-4-8 10.29, luna 17.30
  gross_tok concentration: top10 17.7%, top50 40.5%, top100 54.7%, top500 90.2%
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from router.io import (
    DEFAULT_EXPORT_DIR,
    family_of,
    first_user_index,
    iter_lines,
    tok,
)

#: Item ``type`` values that are produced by the model rather than the harness.
MODEL_ITEM_TYPES = frozenset({"reasoning", "function_call", "custom_tool_call"})

#: Default output path for the artifact this module owns.
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "recon.jsonl"
)


def is_model_item(item: Dict[str, Any]) -> bool:
    """Return True if the item was produced by the model (assistant message, reasoning, or a tool call)."""
    return item.get("role") == "assistant" or item.get("type") in MODEL_ITEM_TYPES


def segment_turns(items: List[Dict[str, Any]]) -> List[int]:
    """Return the start index of each turn: the first item of every maximal run of model-produced items."""
    cuts: List[int] = []
    previous_was_model = False
    for i, item in enumerate(items):
        current = is_model_item(item)
        if current and not previous_was_model:
            cuts.append(i)
        previous_was_model = current
    return cuts


def billed_prefixes(items: List[Dict[str, Any]], tools_tok: int) -> List[int]:
    """Return the billed input size of every turn: tools_tok plus the per-item token sum of the prefix before it."""
    item_toks = [tok(it) for it in items]
    running = 0
    running_by_index = [0] * (len(items) + 1)
    for i, t in enumerate(item_toks):
        running += t
        running_by_index[i + 1] = running
    return [tools_tok + running_by_index[i] for i in segment_turns(items)]


def recon_line(idx: int, req: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct one export line into the results/recon.jsonl record described in the module docstring."""
    items = req["input"]
    item_toks = [tok(it) for it in items]
    tools_tok = tok(req.get("tools", []))

    cuts = segment_turns(items)
    prefix_sums = [0] * (len(items) + 1)
    running = 0
    for i, t in enumerate(item_toks):
        running += t
        prefix_sums[i + 1] = running

    prefix_tokens = [tools_tok + prefix_sums[i] for i in cuts]
    gross = sum(prefix_tokens)
    cache_read = sum(prefix_tokens[:-1])

    fu = first_user_index(items)
    pre_tok = tools_tok + (prefix_sums[fu + 1] if fu >= 0 else 0)

    return {
        "idx": idx,
        "model": req["model"],
        "family": family_of(req),
        "n_turns": len(cuts),
        "turn_cuts": cuts,
        "prefix_tokens": prefix_tokens,
        "tools_tok": tools_tok,
        "pre_tok": pre_tok,
        "gross_tok": gross,
        "cache_read_tok": cache_read,
        "cache_write_tok": gross - cache_read,
        "naive_tok": prefix_sums[-1],
    }


def build(export_dir: str = DEFAULT_EXPORT_DIR) -> List[Dict[str, Any]]:
    """Reconstruct every export line and return the records in file order."""
    return [recon_line(idx, req) for idx, req in iter_lines(export_dir)]


def write(records: List[Dict[str, Any]], out_path: str = DEFAULT_OUT) -> str:
    """Write the records to results/recon.jsonl, one compact JSON object per line."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return out_path


def _checks(records: List[Dict[str, Any]]) -> List[tuple]:
    """Build the (name, expected, actual, ok) acceptance rows from the reconstructed records."""
    n_turns = sum(r["n_turns"] for r in records)
    gross = sum(r["gross_tok"] for r in records)
    excl_tools = sum(r["gross_tok"] - r["tools_tok"] * r["n_turns"] for r in records)
    cache_read = sum(r["cache_read_tok"] for r in records)
    cache_write = sum(r["cache_write_tok"] for r in records)
    naive = sum(r["naive_tok"] for r in records)

    rows = [
        ("lines", 1000, len(records), None),
        ("n_turns total", 10_845, n_turns, None),
        ("sum(gross_tok)", 334_729_910, gross, None),
        ("sum(gross_tok) excl tools", 290_009_114, excl_tools, None),
        ("tools-block gap", 44_720_796, gross - excl_tools, None),
        ("sum(cache_read_tok)", 308_074_571, cache_read, None),
        ("sum(cache_write_tok)", 26_655_339, cache_write, None),
        ("sum(naive_tok)", 22_631_879, naive, None),
    ]
    out = [(n, f"{e:,}", f"{a:,}", e == a) for n, e, a, _ in rows]

    turns: Dict[str, int] = {}
    lines: Dict[str, int] = {}
    for r in records:
        turns[r["model"]] = turns.get(r["model"], 0) + r["n_turns"]
        lines[r["model"]] = lines.get(r["model"], 0) + 1
    expected_tpl = {
        "claude-opus-5": 10.36,
        "claude-sonnet-5": 11.83,
        "gpt-5.6-sol": 10.85,
        "gpt-5.6-terra": 8.51,
        "claude-fable-5": 11.72,
        "claude-opus-4-8": 10.29,
        "gpt-5.6-luna": 17.30,
    }
    for model, exp in expected_tpl.items():
        act = round(turns[model] / lines[model], 2)
        out.append(("turns/line %s" % model, "%.2f" % exp, "%.2f" % act, abs(act - exp) < 5e-3))

    ordered = sorted((r["gross_tok"] for r in records), reverse=True)
    for k, exp in ((10, 17.7), (50, 40.5), (100, 54.7), (500, 90.2)):
        act = round(100.0 * sum(ordered[:k]) / gross, 1)
        out.append(("gross share top%d" % k, "%.1f%%" % exp, "%.1f%%" % act, abs(act - exp) < 5e-2))

    read_share = round(100.0 * cache_read / gross, 1)
    out.append(("cache_read share", "92.0%", "%.1f%%" % read_share, abs(read_share - 92.0) < 5e-2))
    write_share = round(100.0 * cache_write / gross, 1)
    out.append(("cache_write share", "8.0%", "%.1f%%" % write_share, abs(write_share - 8.0) < 5e-2))
    return out


def main() -> None:
    """Rebuild results/recon.jsonl and print every recon acceptance number with PASS/FAIL."""
    records = build()
    path = write(records)
    print("router.recon -> %s" % path)
    print("all token counts are ESTIMATES (len(json.dumps(x))//4); the export has no usage field")
    print("")
    rows = _checks(records)
    ok = True
    for name, expected, actual, good in rows:
        ok = ok and good
        print(
            "  [%s] %-32s expected %14s  actual %14s"
            % ("PASS" if good else "FAIL", name, expected, actual)
        )
    print("")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
