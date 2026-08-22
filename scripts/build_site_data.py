#!/usr/bin/env python3
"""Build the aggregated data file the local viewer site reads.

Reads the redacted export (and results/routes.jsonl if present) and writes
`site/data.js` as `window.VIKTOR_DATA = {...}` — a .js assignment rather than
JSON so `site/index.html` opens straight off the filesystem (fetch() is blocked
on file://, a script tag is not).

Only aggregates plus a small sample of trajectories go in, with item text
truncated. The export itself is challenge-use-only: site/data.js is gitignored
and the site is never published.

Usage: python scripts/build_site_data.py [export_dir] [--out site/data.js]
"""
import json, sys, argparse, hashlib, math
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from load_trajectories import iter_requests, group_trajectories, est_tokens, first_user_text
from cost_model import (load_pricing, shared_prefix_tokens, trajectory_cost, logged_route)
from baseline_router import cheap_for

PREVIEW_CHARS = 1400        # per item in a sampled trajectory
LIST_PREVIEW_CHARS = 220    # per trajectory in the browser list
MAX_SAMPLES = 80            # trajectories kept with full item detail

# ---------------------------------------------------------------- item helpers

def item_kind(item):
    """One readable label per Responses-format input item."""
    t = item.get("type")
    if t == "message" or (t is None and "role" in item):
        return f"message:{item.get('role', '?')}"
    return t or "unknown"

def humanize(value, indent=""):
    """Render a decoded JSON value as indented text with real newlines.

    Tool arguments and outputs arrive as JSON strings inside a JSON line, so a
    raw dump shows escaped newlines and one endless line. This unwraps one level
    into something a person can read in the viewer.
    """
    pad = indent + "  "
    if isinstance(value, str):
        return indent + value if indent else value
    if isinstance(value, dict):
        if not value:
            return indent + "{}"
        out = []
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{indent}{k}:")
                out.append(humanize(v, pad))
            elif isinstance(v, str) and ("\n" in v or len(v) > 70):
                out.append(f"{indent}{k}:")
                out.extend(pad + ln for ln in v.split("\n"))
            elif isinstance(v, str):
                out.append(f"{indent}{k}: {v}")
            else:
                out.append(f"{indent}{k}: {json.dumps(v, ensure_ascii=False)}")
        return "\n".join(out)
    if isinstance(value, list):
        if not value:
            return indent + "[]"
        return "\n".join(f"{indent}- " + humanize(x, pad).lstrip() for x in value)
    return indent + json.dumps(value, ensure_ascii=False)

def maybe_json(s):
    """Pretty-print a JSON-encoded string; leave anything else untouched."""
    if not isinstance(s, str):
        return humanize(s)
    head = s.lstrip()[:1]
    if head not in ("{", "["):
        return s
    try:
        return humanize(json.loads(s))
    except ValueError:
        return s

def item_name(item):
    """Tool name for call items — shown next to the kind badge, not inline."""
    return item.get("name") if item.get("type") in (
        "function_call", "custom_tool_call") else None

def item_text(item):
    """Flatten an input item to display text."""
    t = item.get("type")
    if t in ("function_call", "custom_tool_call"):
        return maybe_json(item.get("arguments") or item.get("input") or "")
    if t in ("function_call_output", "custom_tool_call_output"):
        return maybe_json(item.get("output"))
    if t == "reasoning":
        return f"[reasoning, summary items: {len(item.get('summary') or [])}]"
    c = item.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if p.get("type") in ("input_text", "output_text", "text"):
                parts.append(p.get("text", ""))
            elif p.get("type") == "input_image":
                parts.append("[image]")
            else:
                parts.append(f"[{p.get('type')}]")
        return " ".join(parts)
    return ""

def trunc(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n] + f" … [+{len(s) - n} chars]"

def tool_calls_in(req):
    return [i.get("name", "?") for i in req["input"]
            if i.get("type") in ("function_call", "custom_tool_call")]

# ---------------------------------------------------------------- aggregations

def histogram(values, n_bins=24):
    """Log-spaced bins — token counts span orders of magnitude."""
    vals = [v for v in values if v > 0]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(vals)}]
    l0, l1 = math.log10(lo), math.log10(hi)
    edges = [10 ** (l0 + (l1 - l0) * i / n_bins) for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in vals:
        i = min(int((math.log10(v) - l0) / (l1 - l0) * n_bins), n_bins - 1)
        counts[i] += 1
    return [{"lo": round(edges[i]), "hi": round(edges[i + 1]), "count": counts[i]}
            for i in range(n_bins)]

def build_cache_profile(groups):
    """Per call position: how much of the input is a prefix shared with the
    previous call (the share a provider cache could serve). Estimated from
    item-level overlap — the export has no usage.cached_tokens."""
    by_pos = defaultdict(list)
    for calls in groups.values():
        for i in range(1, len(calls)):
            inp = est_tokens(calls[i]["input"])
            if inp <= 0:
                continue
            shared = min(shared_prefix_tokens(calls[i - 1], calls[i]), inp)
            by_pos[i].append(shared / inp)
    out = []
    for pos in sorted(by_pos):
        fr = sorted(by_pos[pos])
        out.append({"pos": pos, "n": len(fr),
                    "p25": round(fr[len(fr) // 4], 4),
                    "median": round(fr[len(fr) // 2], 4),
                    "p75": round(fr[(3 * len(fr)) // 4], 4)})
    return out

def switch_penalty(calls, pricing):
    """Cost of the logged single-model route vs. switching to the cheap sibling
    at call i (and staying there). Shows what a mid-task switch costs in cache
    resets — the numbers are estimates, like everything else here."""
    logged = logged_route(calls)
    base, _ = trajectory_cost(calls, logged, pricing)
    rows = []
    for i in range(1, len(calls)):
        route = logged[:i] + [cheap_for(logged[0])] * (len(calls) - i)
        usd, _ = trajectory_cost(calls, route, pricing)
        rows.append({"switch_at": i, "cost_usd": round(usd, 6),
                     "delta_pct": round((usd / base - 1) * 100, 2) if base else 0.0})
    return {"baseline_usd": round(base, 6), "cheap_model": cheap_for(logged[0]), "switches": rows}

# ---------------------------------------------------------------- main build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default="export")
    ap.add_argument("--out", default="site/data.js")
    ap.add_argument("--routes", default="results/routes.jsonl")
    ap.add_argument("--pretty", action="store_true",
                    help="indent the emitted JSON (bigger file, greppable by hand)")
    a = ap.parse_args()

    pricing = load_pricing()
    reqs, chunks = [], Counter()
    for name, _, r in iter_requests(a.export):
        reqs.append(r)
        chunks[name] += 1
    groups = group_trajectories(reqs)

    # --- per-model rollup -------------------------------------------------
    model_req = Counter(r["model"] for r in reqs)
    model_tok = Counter()
    for r in reqs:
        model_tok[r["model"]] += est_tokens(r["input"])
    model_traj, model_cost = Counter(), Counter()
    for calls in groups.values():
        model_traj[calls[0]["model"]] += 1
        usd, _ = trajectory_cost(calls, logged_route(calls), pricing)
        model_cost[calls[0]["model"]] += usd
    models = sorted(
        ({"model": m, "requests": model_req[m], "trajectories": model_traj[m],
          "est_tokens": model_tok[m], "cost_usd": round(model_cost[m], 4)}
         for m in model_req),
        key=lambda d: -d["requests"])

    # --- grouping diagnostics --------------------------------------------
    # The loader keys on the first user text. Keying on the leading item (the
    # system prompt) is a coarser alternative; comparing the two says how much
    # structure the reconstruction actually recovers.
    sys_groups = Counter()
    for r in reqs:
        head = r["input"][0] if r["input"] else {}
        sys_groups[hashlib.sha1(json.dumps(head, sort_keys=True)[:4000].encode()).hexdigest()[:12]] += 1
    sizes = Counter(len(v) for v in groups.values())
    # A genuine consecutive call pair shares its opening items. Pairs with zero
    # overlap are a strong sign the grouping key merged two distinct tasks.
    pairs = pairs_shared = 0
    for calls in groups.values():
        for i in range(1, len(calls)):
            pairs += 1
            if shared_prefix_tokens(calls[i - 1], calls[i]) > 0:
                pairs_shared += 1
    diagnostics = {
        "n_requests": len(reqs),
        "n_trajectories": len(groups),
        "n_singletons": sizes.get(1, 0),
        "max_calls": max(sizes) if sizes else 0,
        "n_system_prompt_variants": len(sys_groups),
        "largest_system_prompt_group": max(sys_groups.values()) if sys_groups else 0,
        "mixed_model_trajectories": sum(
            1 for v in groups.values() if len({r["model"] for r in v}) > 1),
        "consecutive_pairs": pairs,
        "consecutive_pairs_with_shared_prefix": pairs_shared,
    }

    # --- distributions ----------------------------------------------------
    calls_per_traj = [{"n_calls": k, "count": v} for k, v in sorted(sizes.items())]
    token_hist = histogram([est_tokens(r["input"]) for r in reqs])

    tool_available, tool_used = Counter(), Counter()
    for r in reqs:
        for t in r.get("tools") or []:
            tool_available[t.get("name", "?")] += 1
    # Count tool calls once per trajectory, not once per request: every request's
    # input repeats the whole history, so summing over requests double-counts.
    # The last call's input is the union of the trajectory's items.
    for calls in groups.values():
        tool_used.update(tool_calls_in(calls[-1]))
    tools = sorted(
        ({"name": n, "available_in_requests": tool_available.get(n, 0), "calls": tool_used.get(n, 0)}
         for n in set(tool_available) | set(tool_used)),
        key=lambda d: (-d["calls"], -d["available_in_requests"]))

    # --- trajectory list (all of them, metadata only) ---------------------
    traj_list = []
    for key, calls in groups.items():
        last = calls[-1]
        used = Counter(tool_calls_in(last))
        traj_list.append({
            "tool_calls": dict(used.most_common()),
            "id": key,
            "model": calls[0]["model"],
            "models": sorted({c["model"] for c in calls}),
            "n_calls": len(calls),
            "est_tokens": sum(est_tokens(c["input"]) for c in calls),
            "max_input_tokens": est_tokens(last["input"]),
            "n_tool_calls": len(tool_calls_in(last)),
            "n_tools_available": len(last.get("tools") or []),
            "preview": trunc(first_user_text(calls[0]).strip(), LIST_PREVIEW_CHARS),
        })
    traj_list.sort(key=lambda d: (-d["n_calls"], -d["est_tokens"]))

    # --- sampled detail ---------------------------------------------------
    # Every multi-call trajectory first (that is where the structure is), then
    # the largest singletons, capped at MAX_SAMPLES.
    multi = [t for t in traj_list if t["n_calls"] > 1][:MAX_SAMPLES]
    rest = [t for t in traj_list if t["n_calls"] == 1][:max(0, MAX_SAMPLES - len(multi))]
    samples = {}
    for meta in multi + rest:
        calls = groups[meta["id"]]
        last = calls[-1]
        # Each call's input is a prefix of the next; store the union timeline once
        # and record where each call ended (n_items).
        items = [{"kind": item_kind(it), "name": item_name(it),
                  "chars": len(json.dumps(it)),
                  "text": trunc(item_text(it), PREVIEW_CHARS)} for it in last["input"]]
        call_rows = []
        for i, c in enumerate(calls):
            inp = est_tokens(c["input"])
            shared = min(shared_prefix_tokens(calls[i - 1], c), inp) if i else 0
            call_rows.append({"model": c["model"], "n_items": len(c["input"]),
                              "est_tokens": inp, "shared_prefix_tokens": shared,
                              "uncached_tokens": inp - shared})
        samples[meta["id"]] = {
            "items": items,
            "calls": call_rows,
            "tools": [t.get("name", "?") for t in (last.get("tools") or [])],
            "cache": switch_penalty(calls, pricing) if len(calls) > 1 else None,
        }

    # --- routing / frontier ----------------------------------------------
    routes_path = Path(a.routes)
    frontier, routes_summary = [], None
    if routes_path.exists():
        recs = [json.loads(l) for l in routes_path.open() if l.strip()]
        total_calls = sum(r["n_calls"] for r in recs) or 1
        by_cost = sorted(recs, key=lambda r: r["cost_logged_usd"])
        for step in range(11):
            frac = step / 10
            adopt = {r["trajectory"] for r in by_cost[: round(frac * len(recs))]}
            cost = sum(r["cost_routed_usd"] if r["trajectory"] in adopt else r["cost_logged_usd"]
                       for r in recs)
            kept = sum((sum(1 for m in r["route"] if m == r["logged_model"]) / len(r["route"])
                        if r["trajectory"] in adopt else 1.0) * r["n_calls"] for r in recs)
            frontier.append({"adopt_frac": frac, "cost_usd": round(cost, 4),
                             "quality_placeholder": round(kept / total_calls, 4)})
        # Attach per-trajectory routing costs so the browser can filter on them.
        by_id = {r["trajectory"]: r for r in recs}
        for t in traj_list:
            r = by_id.get(t["id"])
            if r:
                t["cost_logged_usd"] = r["cost_logged_usd"]
                t["cost_routed_usd"] = r["cost_routed_usd"]
                # `logged_model` in routes.jsonl is the first call's model only, so
                # comparing the route against it mislabels mixed-model groups.
                # A changed cost is the reliable signal that the route differs.
                t["rerouted"] = abs(r["cost_routed_usd"] - r["cost_logged_usd"]) > 1e-9
        routes_summary = {
            "n_trajectories": len(recs),
            "cost_logged_usd": round(sum(r["cost_logged_usd"] for r in recs), 4),
            "cost_routed_usd": round(sum(r["cost_routed_usd"] for r in recs), 4),
            "rerouted_trajectories": sum(
                1 for r in recs
                if abs(r["cost_routed_usd"] - r["cost_logged_usd"]) > 1e-9),
        }

    data = {
        "meta": {
            "chunks": [{"name": n, "requests": c} for n, c in sorted(chunks.items())],
            "est_tokens_total": sum(model_tok.values()),
            "pricing": pricing,
            "preview_chars": PREVIEW_CHARS,
            "n_samples": len(samples),
            "caveats": [
                "Alle Token-Zahlen sind SCHÄTZUNGEN (serialisierte Zeichen / 4) — der Export hat kein `usage`-Feld.",
                "Output-Tokens fehlen im Export und sind in keiner Kostenzahl enthalten.",
                "Modell-ids sind anonymisiert; die Preise sind eine ANNAHME (scripts/cost_model.py).",
                "Der gecachte Anteil ist aus Item-Prefix-Overlap ABGELEITET, nicht gemessen.",
                "Qualität auf dem Frontier ist ein PLATZHALTER (Anteil Calls auf dem geloggten Modell).",
            ],
        },
        "diagnostics": diagnostics,
        "models": models,
        "calls_per_traj": calls_per_traj,
        "token_hist": token_hist,
        "tools": tools,
        "cache_profile": build_cache_profile(groups),
        "trajectories": traj_list,
        "samples": samples,
        "frontier": frontier,
        "routes_summary": routes_summary,
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, ensure_ascii=False, indent=2) if a.pretty
               else json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out.write_text("window.VIKTOR_DATA = " + payload + ";\n", encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  requests={diagnostics['n_requests']} trajectories={diagnostics['n_trajectories']} "
          f"samples={len(samples)} models={len(models)}")

if __name__ == "__main__":
    main()
