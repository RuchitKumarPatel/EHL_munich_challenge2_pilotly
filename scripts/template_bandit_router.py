#!/usr/bin/env python3
"""Template-conditioned router with honest-ish offline replay.

This router uses only the first request of a reconstructed trajectory to choose
one model for the whole trajectory. It then compares that route to the logged
data with two separate measurements:

1. cache-aware replay cost on the fixed historical trajectory;
2. cross-fitted neighborhood estimates of a continuation-burden proxy.

The proxy is not ground-truth quality. It uses signals visible in the logged
trajectory after the first call: extra calls, tool-error text, and timeout/wait
language. Lower burden is better. Counterfactual quality is estimated only from
similar histories in the training folds, so support and effective sample size
are reported with every policy point.

Usage:
  python3 scripts/template_bandit_router.py .
  python3 scripts/template_bandit_router.py export --lambdas 0,0.1,0.25,0.5,1
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from cost_model import load_pricing, logged_route, price_of, shared_prefix_tokens
from load_trajectories import est_tokens, first_user_text, group_key, iter_requests


ERROR_RE = re.compile(
    r"error|failed|failure|traceback|exception|non-zero|exit_code[^0-9-]*-[0-9]|exit_code[^0-9]*[1-9]",
    re.IGNORECASE,
)
WAIT_RE = re.compile(r"timeout|timed out|wait_for_background_work|still running|poll", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[a-z0-9_]+")
COMPLEX_TOOL_TERMS = ("bash", "code", "search", "patch", "edit", "write")


@dataclass(frozen=True)
class Observation:
    key: str
    calls: list
    logged_model: str
    logged_route: list
    first_tokens: int
    token_counts: tuple
    shared_counts: tuple
    total_tokens: int
    uncached_tokens_sticky: int
    features: dict
    burden: float


def stable_hash_int(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def family(model):
    if model.startswith("claude"):
        return "claude"
    if model.startswith("gpt"):
        return "gpt"
    return model.split("-", 1)[0]


def dominant_model(route):
    return Counter(route).most_common(1)[0][0]


def text_parts(item):
    parts = []
    if "arguments" in item:
        parts.append(str(item.get("arguments", "")))
    if "output" in item:
        parts.append(str(item.get("output", "")))
    content = item.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
    return parts


def text_of_input(items):
    return "\n".join(part for item in items for part in text_parts(item))


def normalize_text(text):
    text = PLACEHOLDER_RE.sub(" placeholder ", text.lower())
    return WORD_RE.findall(text)


def ngrams(words, n):
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def task_type(text):
    low = text.lower()
    if "cron" in low or "queue" in low or "monitor" in low or "slack" in low:
        return "automation"
    if "apply_patch" in low or "bash" in low or "script" in low or "code" in low:
        return "coding"
    if "image" in low or "screenshot" in low:
        return "vision"
    if "summar" in low or "draft" in low or "write" in low:
        return "writing"
    return "general"


def token_bucket(tokens):
    if tokens < 4_000:
        return "tiny"
    if tokens < 12_000:
        return "short"
    if tokens < 30_000:
        return "medium"
    return "long"


def tool_name(tool):
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(tool.get("name") or tool.get("type") or "")


def complex_tool_count(tool_names):
    return sum(any(term in name.lower() for term in COMPLEX_TOOL_TERMS) for name in tool_names)


def complex_tool_bucket(count):
    if count == 0:
        return "none"
    if count <= 2:
        return "some"
    return "many"


def initial_features(first_call):
    opening = first_user_text(first_call)
    words = normalize_text(opening)[:220]
    tools = sorted({tool_name(t) for t in first_call.get("tools", []) if tool_name(t)})
    complex_tools = complex_tool_count(tools)
    has_image = any(
        isinstance(item.get("content"), list)
        and any(isinstance(part, dict) and part.get("type") == "input_image" for part in item["content"])
        for item in first_call["input"]
    )
    return {
        "opening": opening,
        "words": set(words),
        "bigrams": ngrams(words, 2),
        "prefix12": " ".join(words[:12]),
        "prefix24": " ".join(words[:24]),
        "task_type": task_type(opening),
        "first_tokens": est_tokens(first_call["input"]),
        "token_bucket": token_bucket(est_tokens(first_call["input"])),
        "tool_names": set(tools),
        "tool_signature": "|".join(tools),
        "tool_count": len(tools),
        "complex_tool_count": complex_tools,
        "complex_tool_bucket": complex_tool_bucket(complex_tools),
        "has_complex_tool": complex_tools > 0,
        "has_image": has_image,
    }


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a, b):
    word_sim = jaccard(a["words"], b["words"])
    bigram_sim = jaccard(a["bigrams"], b["bigrams"])
    same_prefix = 1.0 if a["prefix24"] and a["prefix24"] == b["prefix24"] else 0.0
    if not same_prefix and a["prefix12"] and a["prefix12"] == b["prefix12"]:
        same_prefix = 0.6
    tool_sim = jaccard(a["tool_names"], b["tool_names"])
    task_sim = 1.0 if a["task_type"] == b["task_type"] else 0.0
    bucket_sim = 1.0 if a["token_bucket"] == b["token_bucket"] else 0.3
    complex_tool_sim = 1.0 if a["complex_tool_bucket"] == b["complex_tool_bucket"] else 0.0
    image_sim = 1.0 if a["has_image"] == b["has_image"] else 0.0
    return (
        0.36 * word_sim
        + 0.21 * bigram_sim
        + 0.17 * same_prefix
        + 0.10 * tool_sim
        + 0.06 * complex_tool_sim
        + 0.05 * task_sim
        + 0.03 * bucket_sim
        + 0.02 * image_sim
    )


def sticky_uncached_tokens(calls):
    if not calls:
        return 0
    total = est_tokens(calls[0]["input"])
    for idx in range(1, len(calls)):
        total += max(0, est_tokens(calls[idx]["input"]) - shared_prefix_tokens(calls[idx - 1], calls[idx]))
    return total


def burden_proxy(calls):
    """Lower is better. This is a continuation burden proxy, not true quality."""
    if len(calls) == 1:
        return 0.0
    later_text = text_of_input([item for call in calls[1:] for item in call["input"]])
    extra_call_penalty = min((len(calls) - 1) / 4.0, 1.0)
    error_penalty = min(len(ERROR_RE.findall(later_text)) / 3.0, 1.0)
    wait_penalty = 1.0 if WAIT_RE.search(later_text) else 0.0
    return min(1.0, 0.55 * extra_call_penalty + 0.30 * error_penalty + 0.15 * wait_penalty)


def route_cost_from_profile(model, token_counts, shared_counts, pricing):
    cost = 0.0
    for idx, tokens in enumerate(token_counts):
        cached = min(shared_counts[idx], tokens) if idx > 0 else 0
        uncached = tokens - cached
        uncached_price, cached_price, _ = price_of(model, pricing)
        cost += (uncached * uncached_price + cached * cached_price) / 1e6
    return cost


def realized_logged_cost(obs, pricing):
    cost = 0.0
    for idx, model in enumerate(obs.logged_route):
        tokens = obs.token_counts[idx]
        cached = 0
        if idx > 0 and model == obs.logged_route[idx - 1]:
            cached = min(obs.shared_counts[idx], tokens)
        uncached = tokens - cached
        uncached_price, cached_price, _ = price_of(model, pricing)
        cost += (uncached * uncached_price + cached * cached_price) / 1e6
    return cost


def load_observations(export_dir):
    rows = [(chunk, line_no, req) for chunk, line_no, req in iter_requests(export_dir)]
    groups = defaultdict(list)
    first_seen = {}
    for pos, (chunk, line_no, req) in enumerate(rows):
        key = group_key(req)
        groups[key].append(req)
        first_seen.setdefault(key, (chunk, line_no, pos))
    observations = []
    for key in sorted(groups, key=lambda k: first_seen[k]):
        calls = sorted(groups[key], key=lambda r: len(r["input"]))
        route = logged_route(calls)
        token_counts = tuple(est_tokens(call["input"]) for call in calls)
        shared_counts = tuple([0] + [shared_prefix_tokens(calls[idx - 1], calls[idx]) for idx in range(1, len(calls))])
        observations.append(
            Observation(
                key=key,
                calls=calls,
                logged_model=dominant_model(route),
                logged_route=route,
                first_tokens=token_counts[0],
                token_counts=token_counts,
                shared_counts=shared_counts,
                total_tokens=sum(token_counts),
                uncached_tokens_sticky=sticky_uncached_tokens(calls),
                features=initial_features(calls[0]),
                burden=burden_proxy(calls),
            )
        )
    return observations


def weighted_stats(values):
    total_w = sum(w for _, w in values)
    if total_w <= 0:
        return None
    mean = sum(v * w for v, w in values) / total_w
    var = sum(w * (v - mean) ** 2 for v, w in values) / total_w
    ess = total_w * total_w / max(1e-12, sum(w * w for _, w in values))
    return mean, var, ess, total_w


def neighborhood(test_obs, train, top_k, min_similarity):
    scored = []
    for obs in train:
        sim = similarity(test_obs.features, obs.features)
        if sim >= min_similarity:
            scored.append((sim, obs))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def expected_cost(model, test_obs, neighbors, pricing):
    if not neighbors:
        token_counts = test_obs.token_counts
        shared_counts = test_obs.shared_counts
    else:
        ratios_total = []
        ratios_uncached = []
        for sim, obs in neighbors:
            base = max(1, obs.first_tokens)
            weight = sim * sim
            ratios_total.append((obs.total_tokens / base, weight))
            ratios_uncached.append((obs.uncached_tokens_sticky / base, weight))
        total_stats = weighted_stats(ratios_total)
        uncached_stats = weighted_stats(ratios_uncached)
        total_ratio = total_stats[0] if total_stats else 1.0
        uncached_ratio = uncached_stats[0] if uncached_stats else 1.0
        estimated_total = max(test_obs.first_tokens, test_obs.first_tokens * total_ratio)
        estimated_uncached = min(estimated_total, max(test_obs.first_tokens, test_obs.first_tokens * uncached_ratio))
        token_counts = (round(estimated_uncached), round(estimated_total - estimated_uncached))
        shared_counts = (0, round(estimated_total - estimated_uncached))
    return route_cost_from_profile(model, token_counts, shared_counts, pricing)


def model_estimates(test_obs, train, models, pricing, top_k, min_similarity, min_ess, confidence):
    neighbors = neighborhood(test_obs, train, top_k, min_similarity)
    all_neighbor_count = len(neighbors)
    estimates = {}
    for model in models:
        same_model = [(obs.burden, sim * sim) for sim, obs in neighbors if obs.logged_model == model]
        stats = weighted_stats(same_model)
        if not stats:
            continue
        mean, var, ess, total_weight = stats
        if ess < min_ess:
            continue
        se = math.sqrt(max(var, 0.01) / ess)
        cost = expected_cost(model, test_obs, neighbors, pricing)
        estimates[model] = {
            "burden_mean": mean,
            "burden_ucb": min(1.0, mean + confidence * se),
            "burden_se": se,
            "ess": ess,
            "weight": total_weight,
            "neighbor_count": all_neighbor_count,
            "expected_cost": cost,
            "family": family(model),
        }
    return estimates


def fallback_estimate(model, train, test_obs, pricing):
    same = [obs.burden for obs in train if obs.logged_model == model]
    if not same:
        same = [obs.burden for obs in train]
    mean = sum(same) / len(same)
    var = sum((x - mean) ** 2 for x in same) / max(1, len(same))
    ess = len(same)
    se = math.sqrt(max(var, 0.01) / max(1, ess))
    return {
        "burden_mean": mean,
        "burden_ucb": min(1.0, mean + 1.96 * se),
        "burden_se": se,
        "ess": ess,
        "weight": float(ess),
        "neighbor_count": 0,
        "expected_cost": expected_cost(model, test_obs, [], pricing),
        "family": family(model),
    }


def estimate_for_observation(test_obs, train, models, pricing, args):
    estimates = model_estimates(
        test_obs,
        train,
        models,
        pricing,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        min_ess=args.min_ess,
        confidence=args.confidence,
    )
    logged_estimate = estimates.get(test_obs.logged_model) or fallback_estimate(
        test_obs.logged_model, train, test_obs, pricing
    )
    return estimates, logged_estimate


def allowed_family(model, test_obs, args):
    return not args.same_family or family(model) == family(test_obs.logged_model)


def choose_safe_cheapest(test_obs, estimates, logged_estimate, args):
    admissible = []
    for model, est in estimates.items():
        if not allowed_family(model, test_obs, args):
            continue
        if est["burden_ucb"] > logged_estimate["burden_ucb"] + args.ucb_margin:
            continue
        if est["burden_mean"] > logged_estimate["burden_mean"] + args.mean_margin:
            continue
        admissible.append((est["expected_cost"], model, est))
    if not admissible:
        return test_obs.logged_model, logged_estimate, logged_estimate, False, "quality_guard"
    _, model, est = min(admissible, key=lambda row: (row[0], row[1]))
    return model, est, logged_estimate, True, "safe_cheapest"


def choose_scored_model(test_obs, estimates, logged_estimate, lam, args):
    cheapest_supported = min(est["expected_cost"] for est in estimates.values())
    most_expensive_supported = max(est["expected_cost"] for est in estimates.values())
    cost_span = max(1e-12, most_expensive_supported - cheapest_supported)

    best_model = test_obs.logged_model
    best_estimate = logged_estimate
    best_score = float("inf")
    for model, est in estimates.items():
        if not allowed_family(model, test_obs, args):
            continue
        normalized_cost = (est["expected_cost"] - cheapest_supported) / cost_span
        score = est["burden_ucb"] + lam * normalized_cost
        if score < best_score:
            best_score = score
            best_model = model
            best_estimate = est
    if best_score == float("inf"):
        return test_obs.logged_model, logged_estimate, logged_estimate, False, "family_guard"
    return best_model, best_estimate, logged_estimate, True, "supported"


def choose_gated_score(test_obs, estimates, logged_estimate, lam, args):
    model, chosen_est, _, supported, reason = choose_scored_model(test_obs, estimates, logged_estimate, lam, args)
    if not supported:
        return model, chosen_est, logged_estimate, supported, reason
    if chosen_est["burden_ucb"] > logged_estimate["burden_ucb"] + args.ucb_margin:
        return test_obs.logged_model, logged_estimate, logged_estimate, False, "ucb_quality_guard"
    if chosen_est["burden_mean"] > logged_estimate["burden_mean"] + args.mean_margin:
        return test_obs.logged_model, logged_estimate, logged_estimate, False, "mean_quality_guard"
    min_expected = logged_estimate["expected_cost"] * (1.0 - args.min_expected_savings)
    if chosen_est["expected_cost"] >= min_expected:
        return test_obs.logged_model, logged_estimate, logged_estimate, False, "expected_cost_guard"
    return model, chosen_est, logged_estimate, True, "gated_score"


def choose_from_estimates(test_obs, estimates, logged_estimate, lam, args):
    if not estimates:
        return test_obs.logged_model, logged_estimate, logged_estimate, False, "no_supported_neighbor"

    if args.selection_mode == "safe-cheapest":
        return choose_safe_cheapest(test_obs, estimates, logged_estimate, args)
    if args.selection_mode == "gated-score":
        return choose_gated_score(test_obs, estimates, logged_estimate, lam, args)
    return choose_scored_model(test_obs, estimates, logged_estimate, lam, args)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key, weight_key=None):
    if not rows:
        return 0.0
    if weight_key is None:
        return sum(r[key] for r in rows) / len(rows)
    total_w = sum(r[weight_key] for r in rows)
    return sum(r[key] * r[weight_key] for r in rows) / max(1e-12, total_w)


def fold_id(obs, folds):
    return stable_hash_int(obs.key) % folds


def simulate(observations, pricing, lambdas, args):
    models = sorted({obs.logged_model for obs in observations})
    logged_cost_total = sum(realized_logged_cost(obs, pricing) for obs in observations)
    logged_burden_observed = sum(obs.burden for obs in observations) / len(observations)
    prepared = {}
    for fold in range(args.folds):
        train = [obs for obs in observations if fold_id(obs, args.folds) != fold]
        test = [obs for obs in observations if fold_id(obs, args.folds) == fold]
        for obs in test:
            estimates, logged_estimate = estimate_for_observation(obs, train, models, pricing, args)
            prepared[obs.key] = {
                "fold": fold,
                "estimates": estimates,
                "logged_estimate": logged_estimate,
            }

    details = []
    summaries = []

    for lam in lambdas:
        rows = []
        for obs in observations:
            prep = prepared[obs.key]
            chosen, chosen_est, logged_est, supported, reason = choose_from_estimates(
                obs, prep["estimates"], prep["logged_estimate"], lam, args
            )
            routed_cost = route_cost_from_profile(chosen, obs.token_counts, obs.shared_counts, pricing)
            logged_cost = realized_logged_cost(obs, pricing)
            rows.append(
                {
                    "lambda": lam,
                    "trajectory": obs.key,
                    "fold": prep["fold"],
                    "n_calls": len(obs.calls),
                    "logged_model": obs.logged_model,
                    "chosen_model": chosen,
                    "supported": supported,
                    "reason": reason,
                    "match_logged": chosen == obs.logged_model,
                    "task_type": obs.features["task_type"],
                    "first_tokens": obs.first_tokens,
                    "token_bucket": obs.features["token_bucket"],
                    "neighbor_count": chosen_est["neighbor_count"],
                    "chosen_ess": chosen_est["ess"],
                    "chosen_est_burden": chosen_est["burden_mean"],
                    "chosen_ucb_burden": chosen_est["burden_ucb"],
                    "logged_est_burden": logged_est["burden_mean"],
                    "logged_ucb_burden": logged_est["burden_ucb"],
                    "observed_logged_burden": obs.burden,
                    "logged_cost_usd": logged_cost,
                    "routed_cost_usd": routed_cost,
                    "cost_delta_usd": routed_cost - logged_cost,
                }
            )
        routed_cost_total = sum(r["routed_cost_usd"] for r in rows)
        chosen_counts = Counter(r["chosen_model"] for r in rows)
        summaries.append(
            {
                "lambda": lam,
                "trajectories": len(rows),
                "calls": sum(r["n_calls"] for r in rows),
                "logged_cost_usd": round(logged_cost_total, 6),
                "routed_replay_cost_usd": round(routed_cost_total, 6),
                "replay_cost_delta": round(routed_cost_total / logged_cost_total - 1, 4),
                "logged_observed_burden": round(logged_burden_observed, 5),
                "logged_xfit_est_burden": round(mean(rows, "logged_est_burden"), 5),
                "routed_xfit_est_burden": round(mean(rows, "chosen_est_burden"), 5),
                "xfit_burden_delta": round(mean(rows, "chosen_est_burden") - mean(rows, "logged_est_burden"), 5),
                "routed_xfit_ucb_burden": round(mean(rows, "chosen_ucb_burden"), 5),
                "logged_match_rate": round(mean(rows, "match_logged"), 4),
                "supported_rate": round(mean(rows, "supported"), 4),
                "avg_effective_n": round(mean(rows, "chosen_ess"), 3),
                "avg_neighbors": round(mean(rows, "neighbor_count"), 3),
                "chosen_models": json.dumps(dict(sorted(chosen_counts.items()))),
            }
        )
        details.extend(rows)
    return summaries, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", nargs="?", default="export")
    parser.add_argument("--lambdas", default="0,0.05,0.1,0.2,0.35,0.5,0.8,1.2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--min-similarity", type=float, default=0.16)
    parser.add_argument("--min-ess", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--same-family", action="store_true", help="Only route inside the logged model family.")
    parser.add_argument("--selection-mode", choices=("score", "safe-cheapest", "gated-score"), default="score")
    parser.add_argument("--ucb-margin", type=float, default=0.0)
    parser.add_argument("--mean-margin", type=float, default=0.0)
    parser.add_argument("--min-expected-savings", type=float, default=0.0)
    parser.add_argument("--output-prefix", default="template_bandit")
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    pricing = load_pricing()
    observations = load_observations(args.export_dir)
    if not observations:
        raise SystemExit(f"no trajectories found in {args.export_dir}")

    Path("results").mkdir(exist_ok=True)
    summaries, details = simulate(observations, pricing, lambdas, args)

    summary_path = Path(f"results/{args.output_prefix}_summary.csv")
    detail_path = Path(f"results/{args.output_prefix}_routes.jsonl")
    write_csv(summary_path, summaries)
    with detail_path.open("w") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")

    print("Template-conditioned sticky router")
    print(f"trajectories={len(observations)} folds={args.folds}")
    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")
    for row in summaries:
        print(row)
    print("NOTE: cost is cache-aware fixed-trajectory replay on estimated input tokens.")
    print("NOTE: burden is a cross-fitted continuation proxy, not measured quality.")


if __name__ == "__main__":
    main()
