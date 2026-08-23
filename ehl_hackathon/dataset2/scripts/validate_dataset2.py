#!/usr/bin/env python3
"""Validate dataset2 against every invariant it is supposed to satisfy.

This is not a smoke test — it is the thing that makes "no bugs remain" a checkable
claim rather than an assertion. It re-derives trajectories the same way the three
project-method* loaders do and cross-checks the result against the manifest, then
verifies the statistical properties dataset2 exists to provide (complementarity,
propensity structure, planted positivity violation, real cache breaks, drift).

Exit code 0 = all checks pass. Non-zero = at least one invariant is violated.

Usage: python validate_dataset2.py [dataset_dir]
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

FAILURES: list[str] = []
PASSES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        PASSES.append(label)
    else:
        FAILURES.append(f"{label}{(' — ' + detail) if detail else ''}")


# --------------------------------------------------------------------------
# Reconstruction, mirroring the project-method* loaders exactly
# --------------------------------------------------------------------------

def opening_key_of_request(request: dict) -> str:
    opening = []
    for item in request["input"]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role in {"system", "user"}:
            opening.append(item)
        if role == "user":
            break
    encoded = json.dumps(opening, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def estimated_tokens(value) -> int:
    return max(1, len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) // 4)


def prefix_overlap_tokens(previous: list, current: list) -> int:
    shared = 0
    for left, right in zip(previous, current):
        left_text = json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        right_text = json.dumps(right, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if left_text != right_text:
            break
        shared += len(left_text)
    return shared // 4


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    export = root / "export"
    manifest_rows = json.loads((root / "scenario_manifest.json").read_text(encoding="utf-8"))
    manifest = {row["key"]: row for row in manifest_rows}

    requests = []
    for path in sorted(export.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    requests.append((path.name, line_number, json.loads(line)))

    # ---------------- schema ----------------
    bad_schema = [f"{f}:{n}" for f, n, r in requests if set(r) != {"model", "input", "tools"}]
    check(not bad_schema, "schema: every line has exactly {model, input, tools}", f"{len(bad_schema)} bad, e.g. {bad_schema[:3]}")
    bad_types = [f"{f}:{n}" for f, n, r in requests
                 if not isinstance(r.get("model"), str) or not r["model"]
                 or not isinstance(r.get("input"), list) or not isinstance(r.get("tools"), list)]
    check(not bad_types, "schema: field types valid", f"{len(bad_types)} bad")

    # ---------------- manifest key integrity ----------------
    check(len(manifest) == len(manifest_rows), "manifest: keys are unique",
          f"{len(manifest_rows)} rows -> {len(manifest)} distinct keys")

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for _, _, request in requests:
        grouped[opening_key_of_request(request)].append(request)

    check(set(grouped) == set(manifest), "reconstruction: trajectory keys == manifest keys",
          f"only-in-export={len(set(grouped) - set(manifest))} only-in-manifest={len(set(manifest) - set(grouped))}")

    # ---------------- per-trajectory checks ----------------
    call_count_mismatch = []
    opening_unstable = []
    order_not_monotone = []
    model_mismatch = []
    compaction_no_break = []
    switch_not_mixed = []

    for key, calls in grouped.items():
        row = manifest.get(key)
        if row is None:
            continue
        ordered = sorted(calls, key=lambda c: estimated_tokens(c["input"]))

        if len(ordered) != row["n_calls"]:
            call_count_mismatch.append((key[:12], len(ordered), row["n_calls"]))

        # every call of a trajectory must yield the same opening key
        if len({opening_key_of_request(c) for c in ordered}) != 1:
            opening_unstable.append(key[:12])

        # the export premise: history length grows monotonically across calls
        lengths = [estimated_tokens(c["input"]) for c in ordered]
        if any(b <= a for a, b in zip(lengths, lengths[1:])):
            order_not_monotone.append(key[:12])

        models = [c["model"] for c in ordered]
        if row["model_switch"]:
            if len(set(models)) < 2:
                switch_not_mixed.append(key[:12])
        else:
            if set(models) != {row["logged_model"]}:
                model_mismatch.append((key[:12], sorted(set(models)), row["logged_model"]))

        # A compacted trajectory must show a real prefix break somewhere.
        # Non-compaction steps reuse the previous input entirely (ratio == 1.0),
        # so ANY ratio meaningfully below 1.0 is a genuine cache break. An
        # absolute threshold would be wrong here: the frozen system+user opening
        # is itself a large share of a trajectory's tokens, so even a total
        # post-opening invalidation leaves a high retained fraction (measured:
        # ratios land around 0.6-0.8 at the compaction step). What matters is
        # that reuse drops off the 1.0 baseline, forcing real reprocessing.
        if row["context_compaction"] and len(ordered) >= 2:
            overlaps = [prefix_overlap_tokens(a["input"], b["input"]) for a, b in zip(ordered, ordered[1:])]
            full = [estimated_tokens(a["input"]) for a in ordered[:-1]]
            ratios = [o / f for o, f in zip(overlaps, full) if f > 0]
            if not ratios or min(ratios) >= 0.95:
                compaction_no_break.append(key[:12])

    check(not call_count_mismatch, "reconstruction: manifest n_calls matches reconstructed call count",
          f"{len(call_count_mismatch)} mismatched, e.g. {call_count_mismatch[:3]}")
    check(not opening_unstable, "reconstruction: opening key stable within each trajectory",
          f"{len(opening_unstable)} unstable")
    check(not order_not_monotone, "premise: serialized history length strictly increases across calls",
          f"{len(order_not_monotone)} non-monotone, e.g. {order_not_monotone[:5]}")
    check(not model_mismatch, "manifest: logged_model matches non-switched trajectories",
          f"{len(model_mismatch)} mismatched, e.g. {model_mismatch[:3]}")
    check(not switch_not_mixed, "scenario: model_switch trajectories really use >1 model",
          f"{len(switch_not_mixed)} bad")
    check(not compaction_no_break, "scenario: context_compaction really breaks the cached prefix",
          f"{len(compaction_no_break)} showed no break")

    # ---------------- label integrity ----------------
    graded = [r for r in manifest_rows if r["graded"]]
    ungraded = [r for r in manifest_rows if not r["graded"]]
    check(all(r["ground_truth_quality"] is not None for r in graded), "labels: graded rows carry ground_truth_quality")
    check(all(r["ground_truth_quality"] is None for r in ungraded), "labels: ungraded rows carry no ground_truth_quality")
    check(all(0.0 <= r["ground_truth_quality"] <= 1.0 for r in graded), "labels: quality within [0, 1]")
    numeric_ok = all(math.isfinite(r["ground_truth_quality"]) for r in graded) and \
        all(math.isfinite(v) for r in manifest_rows for v in r["counterfactual_expected_quality"].values())
    check(numeric_ok, "labels: no NaN/inf anywhere in quality fields")

    # ---------------- propensity integrity ----------------
    bad_sum = [r["key"][:12] for r in manifest_rows if abs(sum(r["true_propensity_all"].values()) - 1.0) > 1e-3]
    check(not bad_sum, "propensity: true_propensity_all sums to 1", f"{len(bad_sum)} bad")
    bad_self = [r["key"][:12] for r in manifest_rows
                if abs(r["true_propensity"] - r["true_propensity_all"].get(r["logged_model"], -1)) > 1e-6]
    check(not bad_self, "propensity: recorded propensity matches the logged model's entry", f"{len(bad_self)} bad")
    all_positive = all(p > 0 for r in manifest_rows for p in r["true_propensity_all"].values())
    check(all_positive, "propensity: every in-pool action has nonzero probability")

    # ---------------- planted positivity violation ----------------
    violated = [r for r in manifest_rows if r["positivity_violation"]]
    check(bool(violated), "overlap: positivity-violation cell is populated")
    leaked = [r["key"][:12] for r in violated if r["logged_model"] == r["positivity_violation"]]
    check(not leaked, "overlap: forbidden model never actually logged in its cell", f"{len(leaked)} leaks")
    in_pool = [r["key"][:12] for r in violated if r["positivity_violation"] in r["true_propensity_all"]]
    check(not in_pool, "overlap: forbidden model absent from the propensity support", f"{len(in_pool)} present")

    # ---------------- model complementarity (the core property) ----------------
    oracles = {r["oracle_model_quality_only"] for r in manifest_rows}
    check(len(oracles) >= 3, "complementarity: >=3 distinct models are the quality-oracle somewhere",
          f"oracles={sorted(oracles)}")

    # Measured from the emitted labels, not from the generator's intent.
    #
    # Compared WITHIN (task_type, difficulty) cells, never pooled across
    # difficulty. Pooling would be confounded: the logging policy sends hard work
    # to frontier models and easy work to cheap ones, so each model faces a
    # different difficulty mix and raw per-model means compare different
    # populations. (Confirmed while building: pooled means make opus-5 look like
    # the summarization winner even though fable-5 beats it at both easy and
    # medium.) That confounding is a deliberate property of dataset2 — the point
    # is that a method must adjust for it — so the validator must not fall into
    # the very trap the dataset is designed to pose.
    cell = collections.defaultdict(list)
    for r in graded:
        if not r["model_switch"]:
            cell[(r["task_type"], r["difficulty"], r["logged_model"])].append(r["ground_truth_quality"])
    cell_means = {k: statistics.mean(v) for k, v in cell.items() if len(v) >= 3}

    winners_by_cell = {}
    for (task_type, difficulty, model), mean_quality in cell_means.items():
        slot = (task_type, difficulty)
        if slot not in winners_by_cell or mean_quality > winners_by_cell[slot][0]:
            winners_by_cell[slot] = (mean_quality, model)
    distinct_winners = {model for _, model in winners_by_cell.values()}
    check(len(distinct_winners) >= 3,
          "complementarity: >=3 distinct models observably win some (task_type, difficulty) cell",
          f"winners={sorted(distinct_winners)}")

    task_level_winners = collections.defaultdict(set)
    for (task_type, _difficulty), (_q, model) in winners_by_cell.items():
        task_level_winners[task_type].add(model)
    check(len({tuple(sorted(v)) for v in task_level_winners.values()}) >= 2,
          "complementarity: observed winners differ across task types",
          f"{ {k: sorted(v) for k, v in task_level_winners.items()} }")

    cheap_models = {"claude-fable-5", "gpt-5.6-sol"}
    cheap_specialist_cells = [
        slot for slot, (_q, model) in winners_by_cell.items()
        if slot[0] == "summarization_extraction" and slot[1] in {"easy", "medium"} and model in cheap_models
    ]
    check(bool(cheap_specialist_cells),
          "complementarity: a cheap model observably wins easy/medium summarization_extraction",
          f"summarization winners={ {s: winners_by_cell[s][1] for s in winners_by_cell if s[0] == 'summarization_extraction'} }")

    # ---------------- logging policy is context-dependent ----------------
    by_difficulty = collections.defaultdict(collections.Counter)
    for r in manifest_rows:
        by_difficulty[r["difficulty"]][r["logged_model"]] += 1

    def share(difficulty: str, model: str) -> float:
        total = sum(by_difficulty[difficulty].values())
        return by_difficulty[difficulty][model] / total if total else 0.0

    frontier_easy = share("easy", "claude-opus-5") + share("easy", "claude-opus-4-8")
    frontier_hard = share("hard", "claude-opus-5") + share("hard", "claude-opus-4-8")
    check(frontier_hard > frontier_easy * 1.25,
          "logging policy: frontier-model share genuinely rises with difficulty (not MCAR)",
          f"easy={frontier_easy:.3f} hard={frontier_hard:.3f}")

    # ---------------- temporal drift / OOD ----------------
    era_task = collections.defaultdict(collections.Counter)
    for r in manifest_rows:
        era_task[r["era_index"]][r["task_type"]] += 1

    def task_share(era_index: int, task_type: str) -> float:
        total = sum(era_task[era_index].values())
        return era_task[era_index][task_type] / total if total else 0.0

    check(task_share(2, "data_analysis") > task_share(0, "data_analysis") * 2.5,
          "drift: task mix genuinely shifts across eras",
          f"era0={task_share(0, 'data_analysis'):.3f} era2={task_share(2, 'data_analysis'):.3f}")

    nova_eras = {r["era_index"] for r in manifest_rows if r["logged_model"] == "gpt-5.7-nova"}
    check(nova_eras == {2}, "drift: gpt-5.7-nova appears only in the final era", f"eras={sorted(nova_eras)}")

    legal_eras = {r["era_index"] for r in manifest_rows if r["domain"] == "legal"}
    check(legal_eras == {2}, "drift: the OOD 'legal' domain appears only in the final era", f"eras={sorted(legal_eras)}")

    # ---------------- replicate groups ----------------
    groups = collections.defaultdict(list)
    for r in manifest_rows:
        if r["replicate_group"]:
            groups[r["replicate_group"]].append(r)
    check(bool(groups), "replicates: replicate groups exist")
    consistent_spec = all(
        len({(x["task_type"], x["difficulty"], x["logged_model"]) for x in rows}) == 1
        for rows in groups.values()
    )
    check(consistent_spec, "replicates: each group shares one (task_type, difficulty, model) spec")
    varying = [rows for rows in groups.values() if len({x["ground_truth_quality"] for x in rows}) > 1]
    check(len(varying) >= 0.8 * len(groups),
          "replicates: repeated runs of the same spec give varying scores (single-draw noise)",
          f"{len(varying)}/{len(groups)} vary")

    # ---------------- edge cases ----------------
    edge_counts = collections.Counter(r["edge_case"] for r in manifest_rows if r["edge_case"])
    for expected in ("single_call", "no_tools", "oversized_prompt", "unicode_heavy", "long_horizon", "all_failure"):
        check(edge_counts.get(expected, 0) > 0, f"edge case present: {expected}")
    singles = [r for r in manifest_rows if r["edge_case"] == "single_call"]
    check(all(r["n_calls"] == 1 for r in singles), "edge case: single_call really has one call")
    no_tools = [r for r in manifest_rows if r["edge_case"] == "no_tools"]
    check(all(r["tool_set"] == [] for r in no_tools), "edge case: no_tools really declares no tools")

    # unicode survives the JSONL round trip
    unicode_keys = {r["key"] for r in manifest_rows if r["edge_case"] == "unicode_heavy"}
    unicode_ok = False
    for key in unicode_keys:
        text = json.dumps(grouped[key][0]["input"], ensure_ascii=False)
        if any(ch in text for ch in ("é", "日", "Ω", "🚀", "Ü")):
            unicode_ok = True
            break
    check(unicode_ok, "edge case: unicode content survives the JSONL round trip")

    # ---------------- heavy tail ----------------
    call_counts = sorted(r["n_calls"] for r in manifest_rows)
    median = call_counts[len(call_counts) // 2]
    check(call_counts[-1] >= 6 * median, "distribution: trajectory length is heavy-tailed",
          f"median={median} max={call_counts[-1]}")

    # ---------------- scale ----------------
    check(len(manifest_rows) >= 500, "scale: >=500 trajectories (2026 guidance for trusting aggregates)",
          f"n={len(manifest_rows)}")

    # ---------------- report ----------------
    print(f"PASSED {len(PASSES)}")
    for label in PASSES:
        print(f"  [ok]   {label}")
    if FAILURES:
        print(f"\nFAILED {len(FAILURES)}")
        for label in FAILURES:
            print(f"  [FAIL] {label}")
        return 1
    print("\nAll invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
