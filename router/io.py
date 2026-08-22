"""Thin, dependency-free loader for the Viktor trajectory export.

Computes nothing analytical and writes no artifact. It exists so that every other
module in ``router`` reads the export the same way and estimates tokens the same
way. Deliberately tiny: four public functions, no state, no caching, no I/O beyond
reading the JSONL files.

Notes that the rest of the project depends on:
  * ``scripts/load_trajectories.py:group_trajectories()`` is BROKEN (it hashes only
    the first 2000 chars of the first user message, which is shared boilerplate for
    978/1000 rows). It is never used here. Each JSONL line is already a COMPLETE,
    independent trajectory: ``input[0]`` is the system message, ``input[1]`` the user
    message, ``input[-1]`` the model's final assistant message. n = 1000.
  * ``tok()`` is an ESTIMATE (``len(json.dumps(x)) // 4``). The export carries no
    ``usage`` field, so no token count anywhere in this project is measured.

Acceptance numbers (checked by ``python -m router.io``):
  1000 lines; family split gpt 245 / claude 755 with 0 unresolved;
  first_user_index == 1 for 1000/1000 lines; sum of per-item tok == 22,631,879.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Tuple

#: Tool names that appear only in the gpt-family toolset.
GPT_TOOLS = frozenset({"apply_patch", "shell_command"})

#: Tool names that appear only in the claude-family toolset.
CLAUDE_TOOLS = frozenset({"bash", "file_read", "file_edit", "file_write"})

#: Default location of the export relative to the repository root.
DEFAULT_EXPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export"
)


def export_files(export_dir: str = DEFAULT_EXPORT_DIR) -> List[str]:
    """Return the export's trajectory JSONL paths in deterministic (sorted) order."""
    names = sorted(
        n
        for n in os.listdir(export_dir)
        if n.startswith("trajectories_v1_") and n.endswith(".jsonl")
    )
    return [os.path.join(export_dir, n) for n in names]


def iter_lines(export_dir: str = DEFAULT_EXPORT_DIR) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield ``(idx, request)`` for every export line, idx being the 0-based line number."""
    idx = 0
    for path in export_files(export_dir):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield idx, json.loads(line)
                idx += 1


def tok(obj: Any) -> int:
    """Estimate billed tokens for any JSON-serialisable object as ``len(json.dumps(obj)) // 4``."""
    return len(json.dumps(obj)) // 4


def family_of(request: Dict[str, Any]) -> str:
    """Return ``"gpt"`` or ``"claude"`` from the request's tool names (perfect separation, 0 exceptions)."""
    names = {t.get("name") for t in request.get("tools", []) if isinstance(t, dict)}
    if names & GPT_TOOLS:
        return "gpt"
    if names & CLAUDE_TOOLS:
        return "claude"
    raise ValueError("tools array matches neither family: %r" % (sorted(n for n in names if n),))


def first_user_index(items: List[Dict[str, Any]]) -> int:
    """Return the index of the first ``role == "user"`` item, or -1 if the line has none."""
    for i, item in enumerate(items):
        if isinstance(item, dict) and item.get("role") == "user":
            return i
    return -1


def main() -> None:
    """Print the loader's acceptance checks with PASS/FAIL."""
    n = 0
    fam = {"gpt": 0, "claude": 0}
    first_user_is_one = 0
    naive = 0
    for _idx, req in iter_lines():
        n += 1
        fam[family_of(req)] += 1
        items = req["input"]
        if first_user_index(items) == 1:
            first_user_is_one += 1
        naive += sum(tok(it) for it in items)

    checks = [
        ("lines", 1000, n),
        ("family gpt", 245, fam["gpt"]),
        ("family claude", 755, fam["claude"]),
        ("first_user_index == 1", 1000, first_user_is_one),
        ("sum per-item tok (naive)", 22_631_879, naive),
    ]
    print("router.io acceptance")
    ok = True
    for name, expected, actual in checks:
        good = expected == actual
        ok = ok and good
        print(
            "  [%s] %-26s expected %12s  actual %12s"
            % ("PASS" if good else "FAIL", name, f"{expected:,}", f"{actual:,}")
        )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
