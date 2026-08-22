#!/usr/bin/env python3
"""Export trajectory-level details for the gated broad router."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import load_pricing, price_of, shared_prefix_tokens
from template_bandit_router import load_observations, route_cost_from_profile, realized_logged_cost


def call_cost(tokens, shared, model, prev_model, pricing):
    cached = min(shared, tokens) if prev_model == model else 0
    uncached = tokens - cached
    uncached_price, cached_price, _ = price_of(model, pricing)
    return (uncached * uncached_price + cached * cached_price) / 1e6, cached, uncached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", nargs="?", default=".")
    parser.add_argument("--routes", default="results/template_bandit_gated_sim030_routes.jsonl")
    parser.add_argument("--lambda-value", type=float, default=0.2)
    parser.add_argument("--out-prefix", default="results/gated_broad_trajectories")
    args = parser.parse_args()

    pricing = load_pricing()
    observations = {obs.key: obs for obs in load_observations(args.export_dir)}
    route_rows = []
    with open(args.routes) as f:
        for line in f:
            rec = json.loads(line)
            if float(rec["lambda"]) == args.lambda_value:
                route_rows.append(rec)

    route_rows.sort(key=lambda r: r["logged_cost_usd"] - r["routed_cost_usd"], reverse=True)
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    calls_path = Path(f"{args.out_prefix}_calls.csv")
    jsonl_path = Path(f"{args.out_prefix}.jsonl")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_records = []
    call_records = []
    with jsonl_path.open("w") as jf:
        for rec in route_rows:
            obs = observations[rec["trajectory"]]
            chosen = rec["chosen_model"]
            original_route = list(obs.logged_route)
            new_route = [chosen for _ in obs.calls]
            call_details = []
            for i, tokens in enumerate(obs.token_counts):
                original_prev = original_route[i - 1] if i > 0 else None
                new_prev = new_route[i - 1] if i > 0 else None
                original_cost, original_cached, original_uncached = call_cost(
                    tokens, obs.shared_counts[i], original_route[i], original_prev, pricing
                )
                new_cost, new_cached, new_uncached = call_cost(
                    tokens, obs.shared_counts[i], new_route[i], new_prev, pricing
                )
                call_row = {
                    "trajectory": obs.key,
                    "call_index": i,
                    "original_model": original_route[i],
                    "new_model": new_route[i],
                    "input_tokens_est": tokens,
                    "shared_prefix_tokens_est": obs.shared_counts[i],
                    "original_cached_tokens_est": original_cached,
                    "new_cached_tokens_est": new_cached,
                    "original_uncached_tokens_est": original_uncached,
                    "new_uncached_tokens_est": new_uncached,
                    "original_call_cost_usd": round(original_cost, 9),
                    "new_call_cost_usd": round(new_cost, 9),
                    "call_cost_saved_usd": round(original_cost - new_cost, 9),
                    "input_items": len(obs.calls[i]["input"]),
                }
                call_records.append(call_row)
                call_details.append(call_row)

            original_cost = realized_logged_cost(obs, pricing)
            new_cost = route_cost_from_profile(chosen, obs.token_counts, obs.shared_counts, pricing)
            summary = {
                "trajectory": obs.key,
                "n_calls": len(obs.calls),
                "task_type": rec["task_type"],
                "token_bucket": rec["token_bucket"],
                "first_tokens": rec["first_tokens"],
                "original_route": " -> ".join(original_route),
                "new_route": " -> ".join(new_route),
                "original_model": rec["logged_model"],
                "new_policy_model": chosen,
                "changed": rec["logged_model"] != chosen,
                "supported": rec["supported"],
                "reason": rec["reason"],
                "original_cost_usd": round(original_cost, 9),
                "new_policy_cost_usd": round(new_cost, 9),
                "cost_saved_usd": round(original_cost - new_cost, 9),
                "cost_reduction_pct": round((1 - new_cost / original_cost) * 100, 4) if original_cost else 0.0,
                "logged_est_burden": rec["logged_est_burden"],
                "new_est_burden": rec["chosen_est_burden"],
                "burden_delta": rec["chosen_est_burden"] - rec["logged_est_burden"],
                "chosen_ess": rec["chosen_ess"],
                "neighbor_count": rec["neighbor_count"],
            }
            summary_records.append(summary)
            jf.write(json.dumps({**summary, "calls": call_details}) + "\n")

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0]))
        writer.writeheader()
        writer.writerows(summary_records)
    with calls_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(call_records[0]))
        writer.writeheader()
        writer.writerows(call_records)

    total_original = sum(r["original_cost_usd"] for r in summary_records)
    total_new = sum(r["new_policy_cost_usd"] for r in summary_records)
    print(f"wrote {summary_path}")
    print(f"wrote {calls_path}")
    print(f"wrote {jsonl_path}")
    print(f"trajectories={len(summary_records)} changed={sum(r['changed'] for r in summary_records)}")
    print(f"original_cost_usd={total_original:.6f}")
    print(f"new_policy_cost_usd={total_new:.6f}")
    print(f"saved_usd={total_original - total_new:.6f}")
    print(f"cost_reduction={(1 - total_new / total_original) * 100:.2f}%")


if __name__ == "__main__":
    main()
