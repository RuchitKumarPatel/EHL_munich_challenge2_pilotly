#!/usr/bin/env python3
"""The tool block as a routing surface: what the prefix carries, and what it costs.

WHAT THIS COMPUTES
    `router.recon` reports one number for the tool block -- `tools_tok`, the size
    of the `tools` array -- and bills it once per turn, because every turn
    re-sends the whole prefix. Corpus-wide that is 44,720,796 est. tokens, 13.4%
    of the reconstructed bill, for text the model mostly never uses: the median
    trajectory DEFINES 12 tools and CALLS 2.

    This module opens that number up. It reads the per-tool definitions out of
    the export, joins them to which tools each trajectory actually called, and
    prices three interventions that all sit BEFORE the first turn -- so unlike
    context editing they never invalidate a cached prefix mid-flight:

      A  OMIT      ship only the tools the job is expected to need.
                   Cheap, and the only one that can FAIL: a tool that was
                   dropped and then needed is an outage, not an inefficiency.
      B  TRUNCATE  cap every tool DESCRIPTION at a token budget. No tool is
                   removed, so the failure mode is a worse-described tool, not
                   a missing one. Miss rate is 0 by construction.
      C  DEFER     `defer_loading: true` + a tool-search tool: rare tools are
                   findable but absent from the prefix until first use. Miss
                   rate is 0 because the agent can load what it lacks.

    B and C compose. Both are configuration, not model choice, so none of this
    depends on the price ORDER of the anonymized arms (`docs/OPEN-QUESTIONS.md`
    Q1) or on a gpt sheet (Q2): fewer tokens is fewer tokens in either lane.

THE COST MODEL, AND WHY IT IS PER-TURN
    Billing follows `router.costs`: a trajectory's prefixes are
    `prefix_tokens`, of which the last is the cache WRITE and the rest are
    cache READS, at 1.25x and 0.10x. An intervention changes the tool block, so
    it changes every prefix:

        new_prefix[k] = prefix_tokens[k] - tools_tok + tool_tokens_at_turn(k)

    Interventions A and B are flat -- `tool_tokens_at_turn` is constant, so the
    whole trajectory shifts down by a fixed amount. Intervention C is NOT: a
    deferred tool enters the prefix at its first call and stays. Averaging the
    tool mass across turns would put too little of it in the final prefix, which
    is the expensive one, and would overstate C's saving. Hence per-turn.

WHAT IT WRITES
    results/tooling.json  -- the full sweep, plus the per-tool waste table.
    Nothing else. `router.report` lifts the headline keys into claims.json.

TWO HONEST LIMITS -- READ BEFORE QUOTING ANY NUMBER FROM HERE
    1. The CORE sets and the "expected to need" sets are computed from the same
       1000 trajectories they are then scored on. Corpus-wide use rates are not
       available to a router at admission time; a deployed policy would estimate
       them from history and do WORSE. Every saving here is therefore an UPPER
       BOUND on what the intervention delivers, and the miss rates are LOWER
       bounds. Only the job-history policy is leak-free (it reads strictly
       earlier runs of the same `cron_path`), and it is reported separately.
    2. C's cost is understated. A tool-search call consumes a turn's tool-call
       budget and returns a result that enters the history; only the search
       TOOL's own definition (SEARCH_TOOL_TOK) is priced here, not the searches.

ACCEPTANCE (checked by `python -m router.tooling`)
    Tool block re-billed across turns = 44,720,796 est. tok = 13.4% of gross
    334,729,910. Distinct tool blocks (byte-exact) = 9. Tools defined per
    trajectory: median 12. Tools called per trajectory: median 2. Five tools are
    never called anywhere in the export.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from router.io import DEFAULT_EXPORT_DIR, iter_lines, tok

#: Billing multipliers. Same assumption as router.costs DEFAULTS.
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25

#: Item types that carry a model-issued tool call.
CALL_TYPES = frozenset({"function_call", "custom_tool_call"})

#: Estimated size of the tool-search tool's own definition, carried every turn
#: under intervention C. A guess in the size class of the smaller real tools
#: (bash is 182); it is a COST added to C, so a wrong value here makes C look
#: better or worse than it is -- `main()` prints C's sensitivity to it.
SEARCH_TOOL_TOK = 150

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ACCEPTANCE = {
    "tools_block_tok": 44_720_796,
    "gross_tok": 334_729_910,
    "n_distinct_tool_blocks": 9,
    "median_tools_defined": 12,
    "median_tools_called": 2,
    "n_tools_never_called": 5,
}


# --------------------------------------------------------------------------- load


def load_tool_records(export_dir: str = DEFAULT_EXPORT_DIR) -> list[dict]:
    """One record per trajectory: tool sizes, which were called, and when first called.

    ``first_turn[name]`` is the 0-based index of the turn during which ``name``
    was first called, derived from ``turn_cuts``: the turn of item ``j`` is the
    number of cuts at or below ``j``. Needed only by intervention C.
    """
    recon = load_recon()
    out = []
    for idx, req in iter_lines(export_dir):
        rec = recon[idx]
        cuts = rec["turn_cuts"]
        sizes = {
            t["name"]: tok(t)
            for t in req.get("tools", [])
            if isinstance(t, dict) and t.get("name")
        }
        # recon's tools_tok is tok(whole array), which is NOT sum(tok(item)):
        # json.dumps adds the brackets and the separating commas, and tok()'s
        # floor division truncates once per item instead of once per array.
        # Corpus-wide the gap is 105,101 tok. Carry it as a constant frame so
        # the all-tools policy reproduces recon exactly; removing a tool saves
        # its own tokens, not the comma in front of it.
        frame = rec["tools_tok"] - sum(sizes.values())
        called: set[str] = set()
        first_turn: dict[str, int] = {}
        for j, item in enumerate(req.get("input", [])):
            if not isinstance(item, dict) or item.get("type") not in CALL_TYPES:
                continue
            name = item.get("name")
            if not name:
                continue
            called.add(name)
            if name not in first_turn:
                first_turn[name] = sum(1 for c in cuts if c <= j)
        out.append(
            {
                "idx": idx,
                "sizes": sizes,
                "called": called,
                "first_turn": first_turn,
                "frame": frame,
                "block_hash": _block_hash(req.get("tools", [])),
            }
        )
    return out


def load_recon(path: Path | None = None) -> dict[int, dict]:
    """Return ``{idx: recon_record}`` from results/recon.jsonl."""
    path = path or RESULTS_DIR / "recon.jsonl"
    with open(path, encoding="utf-8") as handle:
        return {r["idx"]: r for r in map(json.loads, handle) if r}


def load_jobkeys(path: Path | None = None) -> dict[int, dict]:
    """Return ``{idx: jobkey_record}`` from results/jobkey.jsonl."""
    path = path or RESULTS_DIR / "jobkey.jsonl"
    with open(path, encoding="utf-8") as handle:
        return {r["idx"]: r for r in map(json.loads, handle) if r}


def _block_hash(tools) -> str:
    """Stable identity of a tools array as sent (order-sensitive, like the cache key)."""
    import hashlib

    return hashlib.sha256(json.dumps(tools).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- billing


def bill(records: list[dict], recon: dict[int, dict], tool_tokens_at_turn) -> float:
    """Cost units for the corpus under a policy.

    ``tool_tokens_at_turn(record, k)`` returns the tool tokens present in the
    prefix of turn ``k`` (0-based) for that trajectory. The rest of the prefix
    is whatever ``recon`` measured minus the original tool block, so the two
    stay consistent by construction.
    """
    read = write = 0.0
    for rec in records:
        r = recon[rec["idx"]]
        original = sum(rec["sizes"].values()) + rec["frame"]
        prefixes = [
            p - original + tool_tokens_at_turn(rec, k)
            for k, p in enumerate(r["prefix_tokens"])
        ]
        write += prefixes[-1]
        read += sum(prefixes[:-1])
    return CACHE_READ_MULT * read + CACHE_WRITE_MULT * write


def gross_tokens(records: list[dict], recon: dict[int, dict], tool_tokens_at_turn) -> int:
    """Total sent tokens under a policy -- the number a naive model would optimise."""
    total = 0
    for rec in records:
        r = recon[rec["idx"]]
        original = sum(rec["sizes"].values()) + rec["frame"]
        total += sum(
            p - original + tool_tokens_at_turn(rec, k)
            for k, p in enumerate(r["prefix_tokens"])
        )
    return int(total)


def misses(records: list[dict], keep_for) -> tuple[int, int]:
    """``(trajectories missing >=1 tool, total missing tool-trajectory pairs)``.

    Only tools the trajectory actually HAD can be missed: the gap is measured
    against ``called & defined``, not ``called``. Two trajectories call a name
    that was never in their tools array at all (see ``undefined_calls``); no
    policy can be blamed for those, and counting them would put a floor of 2
    under every row including "change nothing".
    """
    n_traj = n_pairs = 0
    for rec in records:
        removable = rec["called"] & set(rec["sizes"])
        gap = removable - keep_for(rec)
        if gap:
            n_traj += 1
            n_pairs += len(gap)
    return n_traj, n_pairs


def undefined_calls(records: list[dict]) -> list[dict]:
    """Trajectories that called a tool name absent from their own tools array.

    Two exist in this export, and they are worth naming because they are the
    only direct evidence of what an omit-policy MISS looks like in practice:
    idx 8 calls ``shell_command`` -- a gpt-lane tool -- from a claude-lane
    trajectory, and both idx 8 and idx 120 call ``failed_tool``, the harness's
    placeholder for a call that did not resolve. A model asked for something it
    did not have; that is exactly the failure intervention A manufactures on
    purpose.
    """
    return [
        {"idx": rec["idx"], "tools": sorted(rec["called"] - set(rec["sizes"]))}
        for rec in records
        if rec["called"] - set(rec["sizes"])
    ]


# --------------------------------------------------------------------------- policies


def flat(keep_for, cap: int | None = None):
    """Policy where the tool block is the same at every turn (interventions A and B)."""

    def at_turn(rec, _k):
        keep = keep_for(rec)
        return rec["frame"] + sum(
            _capped(size, cap) for name, size in rec["sizes"].items() if name in keep
        )

    return at_turn


def deferred(core_for, cap: int | None = None, search_tok: int = SEARCH_TOOL_TOK):
    """Intervention C: core tools always present, others enter at their first call."""

    def at_turn(rec, k):
        core = core_for(rec)
        total = search_tok + rec["frame"]
        for name, size in rec["sizes"].items():
            loaded = name in core or (
                name in rec["called"] and rec["first_turn"].get(name, 0) <= k
            )
            if loaded:
                total += _capped(size, cap)
        return total

    return at_turn


def _capped(size: int, cap: int | None) -> int:
    return size if cap is None else min(size, cap)


def use_rates(records: list[dict]) -> dict[str, tuple[int, int, int]]:
    """``{tool: (defined_in, called_in, size)}`` over the corpus. LEAKY -- see module docstring."""
    defined, called, size = Counter(), Counter(), {}
    for rec in records:
        for name, sz in rec["sizes"].items():
            defined[name] += 1
            size[name] = sz
        for name in rec["called"]:
            called[name] += 1
    return {n: (defined[n], called[n], size[n]) for n in defined}


def core_at(records: list[dict], rate: float) -> set[str]:
    """Tools whose corpus-wide call rate clears ``rate``. LEAKY -- upper bound only."""
    return {
        name
        for name, (d, c, _) in use_rates(records).items()
        if d and c / d >= rate
    }


def prior_run_tools(records: list[dict], jobkeys: dict[int, dict]) -> dict[int, set[str]]:
    """``{idx: tools this cron_path called in STRICTLY EARLIER runs}``. Leak-free.

    Only rows with ``literal: true`` may be joined (ADR-006): a ``PII_``
    placeholder inside the path is renumbered per request and groups the wrong
    rows. Trajectories without a literal path get no history and are absent.
    """
    by_job: dict[str, list[int]] = defaultdict(list)
    for rec in records:
        key = jobkeys.get(rec["idx"])
        if key and key.get("literal"):
            by_job[key["cron_path"]].append(rec["idx"])

    called = {rec["idx"]: rec["called"] for rec in records}
    history: dict[int, set[str]] = {}
    for indices in by_job.values():
        seen: set[str] = set()
        for idx in sorted(indices):
            history[idx] = set(seen)
            seen |= called[idx]
    return history


def waste_table(records: list[dict], recon: dict[int, dict]) -> list[dict]:
    """Per tool: token-turns carried in trajectories that never called it."""
    wasted, carried = Counter(), Counter()
    for rec in records:
        turns = recon[rec["idx"]]["n_turns"]
        for name, size in rec["sizes"].items():
            carried[name] += size * turns
            if name not in rec["called"]:
                wasted[name] += size * turns
    rates = use_rates(records)
    rows = []
    for name, (d, c, size) in rates.items():
        rows.append(
            {
                "tool": name,
                "size_tok": size,
                "defined_in": d,
                "called_in": c,
                "use_rate": c / d,
                "carried_tok_turns": carried[name],
                "wasted_tok_turns": wasted[name],
            }
        )
    rows.sort(key=lambda r: -r["wasted_tok_turns"])
    return rows


# --------------------------------------------------------------------------- sweep


CAP_GRID = (1200, 800, 600, 400, 300, 200, 150)
CORE_GRID = (0.90, 0.50, 0.35, 0.25, 0.15, 0.05)
COMBINED_GRID = ((0.50, 400), (0.50, 300), (0.35, 400), (0.35, 300), (0.25, 300))


def sweep(records: list[dict], recon: dict[int, dict], jobkeys: dict[int, dict]) -> dict:
    """Every policy, priced. Returns the dict written to results/tooling.json."""
    everything = flat(lambda rec: set(rec["sizes"]))
    base_cost = bill(records, recon, everything)
    base_gross = gross_tokens(records, recon, everything)
    history = prior_run_tools(records, jobkeys)

    def row(label, at_turn, keep_for, extra=None):
        cost = bill(records, recon, at_turn)
        gross = gross_tokens(records, recon, at_turn)
        miss_traj, miss_pairs = misses(records, keep_for) if keep_for else (0, 0)
        out = {
            "policy": label,
            "cost_units": cost,
            "cost_delta": cost / base_cost - 1,
            "gross_tok": gross,
            "gross_delta": gross / base_gross - 1,
            "miss_trajectories": miss_traj,
            "miss_rate": miss_traj / len(records),
            "miss_pairs": miss_pairs,
        }
        out.update(extra or {})
        return out

    result = {
        "basis": (
            "ESTIMATED tokens, tok(x) = len(json.dumps(x)) // 4. Cost units are "
            f"read {CACHE_READ_MULT}x + write {CACHE_WRITE_MULT}x applied to recon's "
            "prefix split -- NOT dollars: model ids are anonymized and no public "
            "price sheet applies."
        ),
        "leak_warning": (
            "CORE sets and omit policies use corpus-wide call rates, which a router "
            "does not have at admission. Savings are UPPER BOUNDS; miss rates are "
            "LOWER bounds. Only 'omit: prior-run history' is leak-free."
        ),
        "baseline": {"cost_units": base_cost, "gross_tok": base_gross},
        "oracle": row(
            "oracle: only tools this run called (knows the future)",
            flat(lambda rec: rec["called"]),
            lambda rec: rec["called"],
        ),
        "omit": [],
        "truncate": [],
        "defer": [],
        "combined": [],
        "waste": waste_table(records, recon),
        "undefined_calls": undefined_calls(records),
    }

    # A -- omit, leak-free variant first
    def hist_keep(rec):
        return history.get(rec["idx"], set(rec["sizes"])) & set(rec["sizes"])

    result["omit"].append(
        row("omit: prior-run history only (leak-free)", flat(hist_keep), hist_keep)
    )
    for rate in CORE_GRID:
        core = core_at(records, rate)

        def keep(rec, core=core):
            return (core | history.get(rec["idx"], set())) & set(rec["sizes"])

        result["omit"].append(
            row(f"omit: core>={rate:.0%} + history", flat(keep), keep,
                {"core_rate": rate, "core_size": len(core)})
        )

    # B -- truncate every description, nothing removed
    def keep_all(rec):
        return set(rec["sizes"])

    for cap in CAP_GRID:
        over = sum(
            1
            for name, (_, _, size) in use_rates(records).items()
            if size > cap
        )
        result["truncate"].append(
            row(f"truncate: cap {cap} tok/tool", flat(keep_all, cap), None,
                {"cap_tok": cap, "tools_over_cap": over})
        )

    # C -- defer everything outside core; nothing can be missing
    for rate in CORE_GRID:
        core = core_at(records, rate)
        result["defer"].append(
            row(f"defer: core>={rate:.0%}, rest tool-search",
                deferred(lambda rec, core=core: core & set(rec["sizes"])), None,
                {"core_rate": rate, "core_size": len(core)})
        )

    # B + C
    for rate, cap in COMBINED_GRID:
        core = core_at(records, rate)
        result["combined"].append(
            row(f"defer core>={rate:.0%} + cap {cap} tok",
                deferred(lambda rec, core=core: core & set(rec["sizes"]), cap), None,
                {"core_rate": rate, "cap_tok": cap, "core_size": len(core)})
        )

    return result


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    """Print the acceptance checks and the full policy sweep; write results/tooling.json."""
    argv = list(sys.argv[1:] if argv is None else argv)
    export = argv[0] if argv else DEFAULT_EXPORT_DIR

    recon_path = RESULTS_DIR / "recon.jsonl"
    if not recon_path.exists():
        print(f"router.tooling: no recon at {recon_path} -- run `make recon` first")
        return 2

    recon = load_recon()
    jobkeys = load_jobkeys() if (RESULTS_DIR / "jobkey.jsonl").exists() else {}
    records = load_tool_records(export)

    print(f"router.tooling — {len(records)} trajectories from {export}")
    print("All token counts are ESTIMATES (the export has no usage field). Cost units")
    print("are read/write multipliers, NOT dollars: no public sheet applies.\n")

    ok = True

    def check(label, expected, actual):
        nonlocal ok
        good = expected == actual
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: expected={expected!r} actual={actual!r}")

    defined_counts = [len(r["sizes"]) for r in records]
    called_counts = [len(r["called"]) for r in records]
    rates = use_rates(records)
    tools_block = sum(
        (sum(r["sizes"].values()) + r["frame"]) * recon[r["idx"]]["n_turns"]
        for r in records
    )
    gross = sum(recon[r["idx"]]["gross_tok"] for r in records)

    print("acceptance:")
    check("tool block re-billed across turns", ACCEPTANCE["tools_block_tok"], tools_block)
    check("gross_tok", ACCEPTANCE["gross_tok"], gross)
    check("distinct tool blocks", ACCEPTANCE["n_distinct_tool_blocks"],
          len({r["block_hash"] for r in records}))
    check("median tools DEFINED", ACCEPTANCE["median_tools_defined"],
          int(statistics.median(defined_counts)))
    check("median tools CALLED", ACCEPTANCE["median_tools_called"],
          int(statistics.median(called_counts)))
    check("tools never called anywhere", ACCEPTANCE["n_tools_never_called"],
          sum(1 for _, (_, c, _) in rates.items() if c == 0))
    print(f"\n  tool block is {tools_block / gross:.1%} of the reconstructed bill")
    for row in undefined_calls(records):
        print(f"  note: idx {row['idx']} called {row['tools']} — never in its own tools array")

    result = sweep(records, recon, jobkeys)
    base = result["baseline"]["cost_units"]

    print("\nper-tool waste — carried every turn, never called (top 8 by wasted tok-turns):")
    print(f"  {'tool':32s}{'tok':>6s}{'def':>6s}{'called':>8s}{'rate':>8s}{'wasted':>14s}{'% bill':>9s}")
    for r in result["waste"][:8]:
        print(f"  {r['tool']:32s}{r['size_tok']:6d}{r['defined_in']:6d}"
              f"{r['called_in']:8d}{r['use_rate']:8.1%}"
              f"{r['wasted_tok_turns']:14,}{r['wasted_tok_turns'] / gross:9.2%}")

    def table(title, rows, show_miss):
        print(f"\n{title}")
        head = f"  {'policy':46s}{'cost units':>13s}{'Δ cost':>9s}{'Δ tok':>9s}"
        print(head + (f"{'miss rate':>11s}" if show_miss else ""))
        for r in rows:
            line = (f"  {r['policy']:46s}{r['cost_units']:13,.0f}"
                    f"{r['cost_delta']:9.2%}{r['gross_delta']:9.2%}")
            if show_miss:
                line += f"{r['miss_rate']:11.1%}"
            print(line)

    print(f"\nbaseline (all tools, full descriptions): {base:,.0f} cost units")
    table("A — OMIT: ship fewer tools (can fail)", [result["oracle"]] + result["omit"], True)
    table("B — TRUNCATE: cap descriptions, drop nothing (cannot fail)", result["truncate"], False)
    table("C — DEFER: tool search for the rest (cannot fail)", result["defer"], False)
    table("B+C — combined", result["combined"], False)

    print(f"\nC's sensitivity to SEARCH_TOOL_TOK={SEARCH_TOOL_TOK} "
          "(its own definition, carried every turn):")
    core = core_at(records, 0.50)
    for guess in (0, 150, 400, 800):
        cost = bill(records, recon,
                    deferred(lambda rec, core=core: core & set(rec["sizes"]),
                             search_tok=guess))
        print(f"  search tool {guess:4d} tok -> defer core>=50% costs {cost:13,.0f}"
              f"  ({cost / base - 1:+.2%})")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "tooling.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nwrote {out}")
    print("\nREMINDER: savings above are UPPER BOUNDS -- core sets are fitted on the same")
    print("1000 trajectories they are scored on. Only 'omit: prior-run history' is leak-free.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
