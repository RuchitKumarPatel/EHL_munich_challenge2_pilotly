#!/usr/bin/env python3
"""Cross-fitted classifier proposal router.

This script tests whether a simple learned classifier can replace the
neighbor-gated proposal step. The final route is still produced by the
cache-aware segment optimizer with a switch penalty.

Training target:
  The gated router's chosen model at a selected lambda.

Features:
  Only first-request features: task type, token bucket, tool signature, complex
  tool bucket, image flag, logged family/model, and opening prompt words.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cache_segment_router import optimize_segment_route, write_csv
from cost_model import load_pricing
from template_bandit_router import family, load_observations, realized_logged_cost, route_cost_from_profile


def read_proposals(path, lambda_value):
    proposals = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if float(row["lambda"]) == lambda_value:
                proposals[row["trajectory"]] = row
    return proposals


def feature_tokens(obs):
    features = obs.features
    tokens = [
        f"logged_model={obs.logged_model}",
        f"logged_family={family(obs.logged_model)}",
        f"task={features['task_type']}",
        f"token_bucket={features['token_bucket']}",
        f"complex_bucket={features['complex_tool_bucket']}",
        f"has_complex={features['has_complex_tool']}",
        f"has_image={features['has_image']}",
        f"tool_count={min(features['tool_count'], 20)}",
        f"complex_tool_count={min(features['complex_tool_count'], 10)}",
    ]
    for tool in sorted(features["tool_names"]):
        tokens.append(f"tool={tool}")
    for word in sorted(features["words"]):
        tokens.append(f"word={word}")
    for bigram in sorted(features["bigrams"]):
        tokens.append(f"bigram={bigram}")
    return tokens


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.label_counts = Counter()
        self.feature_counts = defaultdict(Counter)
        self.total_features = Counter()
        self.vocab = set()

    def fit(self, rows):
        for label, features in rows:
            self.label_counts[label] += 1
            for feature in features:
                self.feature_counts[label][feature] += 1
                self.total_features[label] += 1
                self.vocab.add(feature)

    def predict(self, features, allowed_labels):
        labels = [label for label in self.label_counts if label in allowed_labels]
        if not labels:
            return None, {}
        total_docs = sum(self.label_counts.values())
        vocab_size = max(1, len(self.vocab))
        scores = {}
        feature_bag = Counter(features)
        for label in labels:
            score = math.log(self.label_counts[label] / total_docs)
            denom = self.total_features[label] + self.alpha * vocab_size
            counts = self.feature_counts[label]
            for feature, value in feature_bag.items():
                score += value * math.log((counts[feature] + self.alpha) / denom)
            scores[label] = score
        best = max(scores, key=scores.get)
        return best, scores


def proposal_stats(train_rows):
    by_label = defaultdict(list)
    for row in train_rows:
        by_label[row["chosen_model"]].append(float(row["chosen_est_burden"]))
    return {
        label: sum(values) / len(values)
        for label, values in by_label.items()
    }


def mean(rows, key):
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def evaluate_classifier(observations, proposals, pricing, switch_penalty_tokens, alpha):
    by_fold = defaultdict(list)
    for key, row in proposals.items():
        by_fold[int(row["fold"])].append((key, row))

    summary_rows = []
    call_rows = []
    for fold in sorted(by_fold):
        train = [(key, row) for other_fold, rows in by_fold.items() if other_fold != fold for key, row in rows]
        test = by_fold[fold]
        train_examples = [
            (row["chosen_model"], feature_tokens(observations[key]))
            for key, row in train
            if key in observations
        ]
        model = MultinomialNB(alpha=alpha)
        model.fit(train_examples)
        burden_by_label = proposal_stats([row for _, row in train])

        for key, proposal in test:
            obs = observations[key]
            allowed = {
                label
                for label in model.label_counts
                if family(label) == family(obs.logged_model)
            }
            predicted, scores = model.predict(feature_tokens(obs), allowed)
            if predicted is None:
                predicted = obs.logged_model
            segment_route, segment_cost_with_penalty, details = optimize_segment_route(
                obs, predicted, pricing, switch_penalty_tokens
            )

            original_cost = realized_logged_cost(obs, pricing)
            classifier_sticky_cost = route_cost_from_profile(predicted, obs.token_counts, obs.shared_counts, pricing)
            switch_penalty_total = sum(d["switch_penalty"] for d in details)
            changed_calls = sum(1 for logged, routed in zip(obs.logged_route, segment_route) if logged != routed)
            adopted_fraction = changed_calls / max(1, len(segment_route))
            logged_burden = float(proposal["logged_est_burden"])
            predicted_burden = burden_by_label.get(predicted, logged_burden)
            segment_burden = logged_burden + adopted_fraction * (predicted_burden - logged_burden)

            summary_rows.append(
                {
                    "trajectory": key,
                    "fold": fold,
                    "n_calls": len(obs.calls),
                    "task_type": proposal["task_type"],
                    "token_bucket": proposal["token_bucket"],
                    "logged_route": " -> ".join(obs.logged_route),
                    "gated_model_label": proposal["chosen_model"],
                    "classifier_model": predicted,
                    "segment_route": " -> ".join(segment_route),
                    "classifier_matches_gated_label": predicted == proposal["chosen_model"],
                    "classifier_matches_logged": predicted == obs.logged_model,
                    "changed_calls": changed_calls,
                    "switches": sum(1 for i in range(1, len(segment_route)) if segment_route[i] != segment_route[i - 1]),
                    "original_cost_usd": round(original_cost, 9),
                    "classifier_sticky_cost_usd": round(classifier_sticky_cost, 9),
                    "segment_cost_with_penalty_usd": round(segment_cost_with_penalty, 9),
                    "switch_penalty_usd": round(switch_penalty_total, 9),
                    "logged_est_burden": logged_burden,
                    "classifier_est_burden": predicted_burden,
                    "segment_est_burden": segment_burden,
                    "segment_burden_delta": segment_burden - logged_burden,
                    "score_margin": round(
                        sorted(scores.values(), reverse=True)[0] - sorted(scores.values(), reverse=True)[1],
                        6,
                    )
                    if len(scores) > 1
                    else "",
                }
            )
            for i, (routed_model, detail) in enumerate(zip(segment_route, details)):
                call_rows.append(
                    {
                        "trajectory": key,
                        "call_index": i,
                        "logged_model": obs.logged_route[i],
                        "classifier_model": predicted,
                        "segment_model": routed_model,
                        "input_tokens_est": obs.token_counts[i],
                        "shared_prefix_tokens_est": obs.shared_counts[i],
                        "segment_cached_tokens_est": detail["cached"],
                        "segment_uncached_tokens_est": detail["uncached"],
                        "segment_call_cost_usd": round(detail["cost"], 9),
                        "switch_penalty_usd": round(detail["switch_penalty"], 9),
                    }
                )
    return summary_rows, call_rows


def aggregate(summary_rows):
    total_original = sum(float(r["original_cost_usd"]) for r in summary_rows)
    total_classifier = sum(float(r["classifier_sticky_cost_usd"]) for r in summary_rows)
    total_segment = sum(float(r["segment_cost_with_penalty_usd"]) for r in summary_rows)
    total_penalty = sum(float(r["switch_penalty_usd"]) for r in summary_rows)
    return [
        {
            "policy": "original_logged",
            "cost_usd": round(total_original, 6),
            "cost_delta_pct": 0.0,
            "burden_delta": 0.0,
            "changed_trajectories": 0,
            "switch_penalty_usd": 0.0,
            "classifier_label_match_pct": "",
        },
        {
            "policy": "classifier_sticky_route",
            "cost_usd": round(total_classifier, 6),
            "cost_delta_pct": round((total_classifier / total_original - 1) * 100, 4),
            "burden_delta": round(mean(summary_rows, "classifier_est_burden") - mean(summary_rows, "logged_est_burden"), 5),
            "changed_trajectories": sum(1 for r in summary_rows if not r["classifier_matches_logged"]),
            "switch_penalty_usd": 0.0,
            "classifier_label_match_pct": round(mean(summary_rows, "classifier_matches_gated_label") * 100, 2),
        },
        {
            "policy": "classifier_cache_segment_switch_penalty_2000_tokens",
            "cost_usd": round(total_segment, 6),
            "cost_delta_pct": round((total_segment / total_original - 1) * 100, 4),
            "burden_delta": round(mean(summary_rows, "segment_burden_delta"), 5),
            "changed_trajectories": sum(1 for r in summary_rows if int(r["changed_calls"]) > 0),
            "switch_penalty_usd": round(total_penalty, 6),
            "classifier_label_match_pct": round(mean(summary_rows, "classifier_matches_gated_label") * 100, 2),
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", nargs="?", default="export")
    parser.add_argument("--routes", default="results/current_gated_sim030_routes.jsonl")
    parser.add_argument("--lambda-value", type=float, default=0.2)
    parser.add_argument("--switch-penalty-tokens", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--out-prefix", default="results/classifier_router")
    args = parser.parse_args()

    pricing = load_pricing()
    observations = {obs.key: obs for obs in load_observations(args.export_dir)}
    proposals = read_proposals(Path(args.routes), args.lambda_value)
    if not proposals:
        raise SystemExit(f"no proposals found for lambda={args.lambda_value} in {args.routes}")

    summary_rows, call_rows = evaluate_classifier(
        observations, proposals, pricing, args.switch_penalty_tokens, args.alpha
    )
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    calls_path = Path(f"{args.out_prefix}_calls.csv")
    aggregate_path = Path(f"{args.out_prefix}_aggregate.csv")
    jsonl_path = Path(f"{args.out_prefix}.jsonl")
    write_csv(summary_path, summary_rows)
    write_csv(calls_path, call_rows)
    write_csv(aggregate_path, aggregate(summary_rows))
    with jsonl_path.open("w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {summary_path}")
    print(f"wrote {calls_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {jsonl_path}")
    for row in aggregate(summary_rows):
        print(row)


if __name__ == "__main__":
    main()
