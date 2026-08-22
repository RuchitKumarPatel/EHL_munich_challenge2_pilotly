#!/usr/bin/env python3
"""Cache-aware segment router with an explicit model-switch penalty.

Input: route proposals from `template_bandit_router.py`.

The gated router proposes one candidate model per trajectory. This script turns
that into a per-call segment route by choosing, for each call, either:

* the logged model for that call; or
* the gated candidate model for the trajectory.

The dynamic program minimizes estimated input cost plus a switch penalty. Cache
reuse is only credited when two consecutive routed calls use the same model.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import load_pricing, price_of
from template_bandit_router import load_observations, realized_logged_cost, route_cost_from_profile


def call_cost(tokens, shared, model, prev_model, pricing):
    cached = min(shared, tokens) if prev_model == model else 0
    uncached = tokens - cached
    uncached_price, cached_price, _ = price_of(model, pricing)
    return (uncached * uncached_price + cached * cached_price) / 1e6, cached, uncached


def switch_penalty_cost(model, penalty_tokens, pricing):
    uncached_price, _, _ = price_of(model, pricing)
    return penalty_tokens * uncached_price / 1e6


def optimize_segment_route(obs, proposed_model, pricing, switch_penalty_tokens):
    """Return the cheapest route over {logged call model, proposed trajectory model}."""
    if not obs.calls:
        return [], 0.0, []

    states = []
    for i, logged_model in enumerate(obs.logged_route):
        candidates = sorted({logged_model, proposed_model})
        states.append(candidates)

    dp = {}
    back = {}
    for model in states[0]:
        cost, cached, uncached = call_cost(obs.token_counts[0], obs.shared_counts[0], model, None, pricing)
        dp[(0, model)] = cost
        back[(0, model)] = (None, {"cost": cost, "cached": cached, "uncached": uncached, "switch_penalty": 0.0})

    for i in range(1, len(states)):
        next_dp = {}
        for model in states[i]:
            best = None
            for prev_model in states[i - 1]:
                prev_cost = dp[(i - 1, prev_model)]
                cost, cached, uncached = call_cost(obs.token_counts[i], obs.shared_counts[i], model, prev_model, pricing)
                penalty = switch_penalty_cost(model, switch_penalty_tokens, pricing) if model != prev_model else 0.0
                total = prev_cost + cost + penalty
                candidate = (
                    total,
                    prev_model,
                    {"cost": cost, "cached": cached, "uncached": uncached, "switch_penalty": penalty},
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
            next_dp[(i, model)] = best[0]
            back[(i, model)] = (best[1], best[2])
        dp.update(next_dp)

    last_i = len(states) - 1
    final_model = min(states[-1], key=lambda model: dp[(last_i, model)])
    route = [final_model]
    details = []
    model = final_model
    for i in range(last_i, -1, -1):
        prev_model, detail = back[(i, model)]
        details.append(detail)
        if prev_model is not None:
            route.append(prev_model)
        model = prev_model
    route.reverse()
    details.reverse()
    return route, dp[(last_i, final_model)], details


def read_proposals(path, lambda_value):
    proposals = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if float(row["lambda"]) == lambda_value:
                proposals[row["trajectory"]] = row
    return proposals


def mean(rows, key):
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", nargs="?", default=".")
    parser.add_argument("--routes", default="results/current_gated_sim030_routes.jsonl")
    parser.add_argument("--lambda-value", type=float, default=0.2)
    parser.add_argument("--switch-penalty-tokens", type=int, default=2_000)
    parser.add_argument("--out-prefix", default="results/cache_segment_router")
    args = parser.parse_args()

    pricing = load_pricing()
    observations = {obs.key: obs for obs in load_observations(args.export_dir)}
    proposals = read_proposals(Path(args.routes), args.lambda_value)
    if not proposals:
        raise SystemExit(f"no proposals found for lambda={args.lambda_value} in {args.routes}")

    summary_rows = []
    call_rows = []
    for key, proposal in sorted(proposals.items()):
        obs = observations[key]
        proposed_model = proposal["chosen_model"]
        segment_route, segment_cost_with_penalty, details = optimize_segment_route(
            obs, proposed_model, pricing, args.switch_penalty_tokens
        )
        original_cost = realized_logged_cost(obs, pricing)
        gated_cost = route_cost_from_profile(proposed_model, obs.token_counts, obs.shared_counts, pricing)
        switch_penalty_total = sum(d["switch_penalty"] for d in details)
        segment_cost = segment_cost_with_penalty - switch_penalty_total
        switch_count = sum(1 for i in range(1, len(segment_route)) if segment_route[i] != segment_route[i - 1])
        changed_calls = sum(1 for logged, routed in zip(obs.logged_route, segment_route) if logged != routed)
        adopted_fraction = changed_calls / max(1, len(segment_route))
        logged_burden = float(proposal["logged_est_burden"])
        gated_burden = float(proposal["chosen_est_burden"])
        segment_burden = logged_burden + adopted_fraction * (gated_burden - logged_burden)

        summary_rows.append(
            {
                "trajectory": key,
                "n_calls": len(obs.calls),
                "task_type": proposal["task_type"],
                "token_bucket": proposal["token_bucket"],
                "logged_route": " -> ".join(obs.logged_route),
                "gated_model": proposed_model,
                "segment_route": " -> ".join(segment_route),
                "changed_calls": changed_calls,
                "switches": switch_count,
                "original_cost_usd": round(original_cost, 9),
                "gated_cost_usd": round(gated_cost, 9),
                "segment_cost_usd": round(segment_cost, 9),
                "segment_cost_with_penalty_usd": round(segment_cost_with_penalty, 9),
                "switch_penalty_usd": round(switch_penalty_total, 9),
                "segment_saved_vs_original_usd": round(original_cost - segment_cost_with_penalty, 9),
                "logged_est_burden": logged_burden,
                "gated_est_burden": gated_burden,
                "segment_est_burden": segment_burden,
                "segment_burden_delta": segment_burden - logged_burden,
                "chosen_ess": proposal["chosen_ess"],
                "neighbor_count": proposal["neighbor_count"],
            }
        )
        for i, (model, detail) in enumerate(zip(segment_route, details)):
            call_rows.append(
                {
                    "trajectory": key,
                    "call_index": i,
                    "logged_model": obs.logged_route[i],
                    "gated_model": proposed_model,
                    "segment_model": model,
                    "input_tokens_est": obs.token_counts[i],
                    "shared_prefix_tokens_est": obs.shared_counts[i],
                    "segment_cached_tokens_est": detail["cached"],
                    "segment_uncached_tokens_est": detail["uncached"],
                    "segment_call_cost_usd": round(detail["cost"], 9),
                    "switch_penalty_usd": round(detail["switch_penalty"], 9),
                }
            )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    calls_path = Path(f"{args.out_prefix}_calls.csv")
    jsonl_path = Path(f"{args.out_prefix}.jsonl")
    write_csv(summary_path, summary_rows)
    write_csv(calls_path, call_rows)
    with jsonl_path.open("w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")

    total_original = sum(r["original_cost_usd"] for r in summary_rows)
    total_gated = sum(r["gated_cost_usd"] for r in summary_rows)
    total_segment = sum(r["segment_cost_with_penalty_usd"] for r in summary_rows)
    total_penalty = sum(r["switch_penalty_usd"] for r in summary_rows)
    segment_changed = sum(1 for r in summary_rows if r["changed_calls"])
    model_mix = Counter(row["segment_route"] for row in summary_rows)

    aggregate = [
        {
            "policy": "original_logged",
            "cost_usd": round(total_original, 6),
            "cost_delta_pct": 0.0,
            "burden_delta": 0.0,
            "changed_trajectories": 0,
            "switch_penalty_usd": 0.0,
        },
        {
            "policy": f"gated_router_lambda_{args.lambda_value:.2f}",
            "cost_usd": round(total_gated, 6),
            "cost_delta_pct": round((total_gated / total_original - 1) * 100, 4),
            "burden_delta": round(mean(summary_rows, "gated_est_burden") - mean(summary_rows, "logged_est_burden"), 5),
            "changed_trajectories": sum(1 for r in summary_rows if r["logged_route"] != r["gated_model"]),
            "switch_penalty_usd": 0.0,
        },
        {
            "policy": f"cache_segment_switch_penalty_{args.switch_penalty_tokens}_tokens",
            "cost_usd": round(total_segment, 6),
            "cost_delta_pct": round((total_segment / total_original - 1) * 100, 4),
            "burden_delta": round(mean(summary_rows, "segment_burden_delta"), 5),
            "changed_trajectories": segment_changed,
            "switch_penalty_usd": round(total_penalty, 6),
        },
    ]
    aggregate_path = Path(f"{args.out_prefix}_aggregate.csv")
    write_csv(aggregate_path, aggregate)

    print(f"wrote {summary_path}")
    print(f"wrote {calls_path}")
    print(f"wrote {jsonl_path}")
    print(f"wrote {aggregate_path}")
    print(f"original_cost_usd={total_original:.6f}")
    print(f"gated_cost_usd={total_gated:.6f}")
    print(f"segment_cost_with_penalty_usd={total_segment:.6f}")
    print(f"switch_penalty_usd={total_penalty:.6f}")
    print(f"segment_changed_trajectories={segment_changed}")
    print(f"segment_route_patterns={len(model_mix)}")


if __name__ == "__main__":
    main()
