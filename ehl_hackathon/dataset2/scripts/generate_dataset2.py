#!/usr/bin/env python3
"""Generate dataset2 — a router-evaluation corpus built to make a routing
methodology's claims actually falsifiable.

Schema contract (identical to the organizer's export and to dataset1, so
project-method1 / project-method2 / project-method3 all load it unchanged):
every JSONL line is exactly {"model": str, "input": [...], "tools": [...]}.
No extra top-level keys are permitted, so all scenario labels live in a
companion `scenario_manifest.json` keyed by the SHA-256 of the opening
system+first-user messages — the same key all three projects' loaders derive.

What dataset2 adds over dataset1 (each verified empirically as a real gap):

  1. MODEL COMPLEMENTARITY (scripts/skill_matrix.py). dataset1 drew quality from
     difficulty alone, so no model was better at anything and no router could
     beat "always cheapest". dataset2 has a latent per-(model, task_type) skill
     matrix where cheap specialists genuinely beat frontier models on their
     specialty. An oracle router beats BOTH extremes on quality AND cost.
  2. CONTEXT-DEPENDENT LOGGING POLICY across three eras
     (scripts/logging_policy.py), replacing dataset1's uniform-random choice, so
     propensity models have real selection bias to learn and correct.
  3. TRUE PROPENSITIES recorded per trajectory, so an estimated propensity can be
     scored against the real one — impossible with real-world data.
  4. A PLANTED POSITIVITY VIOLATION for overlap diagnostics to detect.
  5. TEMPORAL ORDER + DRIFT (eras, task-mix shift, a model introduced late).
  6. AN UNSEEN MODEL (gpt-5.7-nova) appearing only in the final era.
  7. REPLICATE GROUPS: the same spec run several times, so reproducible skill can
     be separated from single-draw label noise (cf. arXiv:2607.03436).
  8. REAL CACHE BREAKS: context compaction that actually rewrites the prefix (so
     cost models measurably lose cache credit), plus mid-session tool-schema
     changes and tool reordering — the documented real-world cache killers.
  9. HEAVY-TAILED trajectory lengths and model usage, instead of uniform.
 10. EDGE CASES: single-call, empty tool list, oversized prompt, unicode,
     very-long-horizon, and all-failure trajectories.
 11. SCALE: ~600 trajectories (2026 guidance is >=500 before trusting aggregates).
 12. A HELD-OUT-BY-CONSTRUCTION OOD slice (a domain and a task type that appear
     only late), for genuine distribution-shift testing.

Usage: python generate_dataset2.py --out .. --seed 20260823
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from logging_policy import ERAS, FORBIDDEN_PAIRS, sample_model
from skill_matrix import (
    DIFFICULTIES, MODEL_PRICE, OUTCOME_NOISE, TASK_TYPES,
    cost_aware_oracle_choice, expected_quality, oracle_choice,
)

# --------------------------------------------------------------------------
# Static vocabulary
# --------------------------------------------------------------------------

TOOL_DEFS = {
    "run_shell": {"type": "function", "name": "run_shell", "description": "Run a shell command",
                  "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}, "strict": False},
    "file_read": {"type": "function", "name": "file_read", "description": "Read a file",
                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "strict": False},
    "file_write": {"type": "function", "name": "file_write", "description": "Write a file",
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, "strict": False},
    "web_search": {"type": "function", "name": "web_search", "description": "Search the web",
                   "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, "strict": False},
    "code_exec": {"type": "function", "name": "code_exec", "description": "Execute a code snippet",
                  "parameters": {"type": "object", "properties": {"language": {"type": "string"}, "code": {"type": "string"}}, "required": ["code"]}, "strict": False},
    "retrieval": {"type": "function", "name": "retrieval", "description": "Retrieve a document from the knowledge base",
                  "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}, "strict": False},
    "sql_query": {"type": "function", "name": "sql_query", "description": "Run a read-only SQL query",
                  "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}, "strict": False},
}

TASK_TOOLS = {
    "qa_reasoning": ["retrieval", "file_read"],
    "code_generation": ["file_read", "file_write", "code_exec", "run_shell"],
    "summarization_extraction": ["file_read", "retrieval"],
    "multi_step_planning": ["run_shell", "web_search", "retrieval"],
    "creative_writing": ["web_search", "file_read"],
    "data_analysis": ["sql_query", "code_exec", "retrieval"],
}

TASK_VERB = {
    "qa_reasoning": "answer the multi-step question",
    "code_generation": "implement and verify the code change",
    "summarization_extraction": "summarize and extract the key fields",
    "multi_step_planning": "plan and execute the multi-step operation",
    "creative_writing": "draft the requested content",
    "data_analysis": "analyze the dataset and report the findings",
}

# The last domain is introduced only in the final era -> a genuine OOD slice.
DOMAINS = (
    {"name": "engineering", "person": "<PERSON_A>", "company": "<COMPANY_A>", "late_only": False},
    {"name": "finance", "person": "<PERSON_B>", "company": "<COMPANY_B>", "late_only": False},
    {"name": "support", "person": "<PERSON_C>", "company": "<COMPANY_C>", "late_only": False},
    {"name": "sales", "person": "<PERSON_D>", "company": "<COMPANY_D>", "late_only": False},
    {"name": "data_ops", "person": "<PERSON_E>", "company": "<COMPANY_E>", "late_only": False},
    {"name": "legal", "person": "<PERSON_F>", "company": "<COMPANY_F>", "late_only": True},
)

# Non-ASCII content, so tokenizer/encoding assumptions get exercised rather than
# only ever seeing ASCII filler.
UNICODE_SNIPPETS = ("naïve café", "日本語テキスト", "Ωμέγα ±3σ", "emoji 🚀 payload", "Ünicöde ﬁle")

MODEL_SPEED_MS_PER_KTOK = {
    "claude-fable-5": 250, "gpt-5.6-sol": 450, "gpt-5.6-luna": 300, "gpt-5.6-terra": 700,
    "claude-sonnet-5": 500, "claude-opus-4-8": 850, "claude-opus-5": 900, "gpt-5.7-nova": 400,
}

# Per-era price multipliers: prices genuinely move over time, so a cost model that
# assumes one static price table can be shown to be wrong (cf. TwinRouterBench's
# "dynamic" track, where model availability and prices change).
ERA_PRICE_MULTIPLIER = {"heuristic_2025h1": 1.0, "explore_2025h2": 1.0, "cost_push_2026h1": 0.80}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def msg(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def opening_key(system_item: dict, user_item: dict) -> str:
    """Must match project-method1's data.loader._opening_context,
    project-method2's data._key, and project-method3's data.schema.opening_key."""
    encoded = json.dumps([system_item, user_item], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tool_arguments(rng: random.Random, tool_name: str, step: int) -> dict:
    if tool_name == "run_shell":
        return {"cmd": f"step-{step} " + "x" * rng.randint(10, 200)}
    if tool_name == "file_read":
        return {"path": f"/data/step-{step}-" + "f" * rng.randint(4, 40)}
    if tool_name == "file_write":
        return {"path": f"/out/step-{step}.txt", "content": "c" * rng.randint(20, 300)}
    if tool_name == "web_search":
        return {"query": f"step-{step} query " + "q" * rng.randint(4, 40)}
    if tool_name == "code_exec":
        return {"language": "python", "code": f"# step-{step}\n" + "pass\n" * rng.randint(1, 5)}
    if tool_name == "retrieval":
        return {"doc_id": f"doc-{step}-" + "d" * rng.randint(4, 20)}
    if tool_name == "sql_query":
        return {"sql": f"SELECT col_{step} FROM t WHERE id > {rng.randint(1, 999)}"}
    raise ValueError(f"unknown tool {tool_name}")


def heavy_tailed_calls(rng: random.Random, difficulty: str) -> int:
    """Log-normal-ish call counts: mostly short, with a genuine long tail.
    dataset1 used a flat uniform range, which under-represents the long-horizon
    trajectories where cache economics and cost accumulation actually matter."""
    centre = {"easy": 1.1, "medium": 1.6, "hard": 2.1}[difficulty]
    value = rng.lognormvariate(centre, 0.55)
    return max(2, min(40, int(round(value))))


# --------------------------------------------------------------------------
# Trajectory construction
# --------------------------------------------------------------------------

def build_trajectory(rng: random.Random, spec: dict) -> tuple[list[dict], dict]:
    """Return (requests, manifest_entry) for one trajectory.

    INVARIANT: the opening system item and first user item are byte-identical in
    every request of this trajectory. All three loaders group requests into
    trajectories by hashing exactly those two items, so mutating them mid-session
    would silently split one trajectory into several. Cache-breaking scenarios
    therefore mutate history AFTER the opening (context compaction) or mutate the
    `tools` block — which is what actually breaks caches in production anyway.
    """
    task_id = spec["task_id"]
    task_type = spec["task_type"]
    difficulty = spec["difficulty"]
    domain = spec["domain"]
    model = spec["model"]
    era = spec["era"]
    edge_case = spec.get("edge_case")

    # ---- opening (frozen for the whole trajectory) ----
    project = f"<PROJECT_{rng.randint(1, 9)}>"
    system_item = {
        "role": "system",
        "content": (f"You are an autonomous agent. Task context {task_id}. "
                    f"Domain: {domain['name']}. Era: {era['name']}. "
                    f"User is {domain['person']} at {domain['company']}."),
    }
    filler_words = {"easy": (20, 90), "medium": (60, 220), "hard": (150, 420)}[difficulty]
    if edge_case == "oversized_prompt":
        filler_words = (3000, 4200)
    user_text = (f"Task {task_id}: {TASK_VERB[task_type]} for {domain['person']} on {project}. "
                 + "filler " * rng.randint(*filler_words))
    if edge_case == "unicode_heavy":
        user_text += " " + " ".join(rng.choice(UNICODE_SNIPPETS) for _ in range(rng.randint(4, 12)))
    user_item = msg("user", user_text)
    if edge_case != "no_tools" and rng.random() < 0.25:
        user_item["content"].append({"type": "input_image", "image_url": "data:image/png;base64,[base64 image redacted]", "detail": "auto"})

    key = opening_key(system_item, user_item)
    history: list[dict] = [system_item, user_item]

    # ---- tools ----
    tool_names = list(TASK_TOOLS[task_type])
    tools = [] if edge_case == "no_tools" else [TOOL_DEFS[name] for name in tool_names]

    # ---- scenario flags ----
    if edge_case == "single_call":
        n_calls = 1
    elif edge_case == "long_horizon":
        n_calls = rng.randint(28, 40)
    else:
        n_calls = heavy_tailed_calls(rng, difficulty)

    all_failure = edge_case == "all_failure"
    failure_rate = {"easy": 0.06, "medium": 0.16, "hard": 0.32}[difficulty]
    has_failure = all_failure or (rng.random() < failure_rate and n_calls >= 3)
    branching = rng.random() < 0.20 and n_calls >= 4
    multi_turn = rng.random() < 0.25 and n_calls >= 3
    # A real cache break: history is compacted mid-session, rewriting the prefix.
    context_compaction = rng.random() < 0.14 and n_calls >= 6
    # Tool schema mutated mid-session (the documented production cache killer).
    tool_schema_change = rng.random() < 0.10 and n_calls >= 4 and tools
    tool_reorder = rng.random() < 0.08 and n_calls >= 3 and len(tools) > 1
    # Mid-trajectory model switch.
    model_switch = rng.random() < 0.13 and n_calls >= 4

    switch_step = rng.randint(2, n_calls - 2) if model_switch else None
    switch_model = rng.choice([m for m in era["pool"] if m != model]) if model_switch else None
    compaction_step = rng.randint(4, n_calls - 2) if context_compaction else None
    tool_change_step = rng.randint(2, n_calls - 2) if tool_schema_change else None
    reorder_step = rng.randint(1, max(1, n_calls - 2)) if tool_reorder else None
    clarify_step = rng.randint(1, max(1, n_calls - 2)) if multi_turn else None
    warn_step = rng.randint(1, max(1, n_calls - 3)) if branching else None
    fail_steps = set()
    if has_failure:
        n_fail = n_calls - 1 if all_failure else rng.randint(1, max(1, min(3, n_calls - 2)))
        fail_steps = set(rng.sample(range(0, max(1, n_calls - 1)), min(n_fail, max(1, n_calls - 1))))

    # ---- ground-truth outcome from the latent skill matrix ----
    # The model that actually ran it drives quality (this is the complementarity
    # dataset1 lacked). Failures/branching depress the realized score.
    base_quality = expected_quality(model, task_type, difficulty)
    if model_switch:
        # A switched trajectory is a blend, slightly penalized for the disruption.
        base_quality = 0.5 * base_quality + 0.5 * expected_quality(switch_model, task_type, difficulty) - 0.04
    realized = base_quality + rng.gauss(0.0, OUTCOME_NOISE)
    if has_failure:
        realized -= 0.10 * len(fail_steps)
    if branching:
        realized -= 0.03
    realized = max(0.0, min(1.0, realized))

    # Only a subset carries a recoverable ground-truth label; the rest keep the
    # honest "no final output" problem the original challenge describes.
    graded = rng.random() < 0.45 or spec.get("force_graded", False)

    # ---- emit calls ----
    requests: list[dict] = []
    latencies: list[float] = []
    current_tools = list(tools)
    repeat_args = None
    era_multiplier = ERA_PRICE_MULTIPLIER[era["name"]]

    for i in range(n_calls):
        active_model = model if switch_step is None or i < switch_step else switch_model

        if tool_change_step is not None and i == tool_change_step and current_tools:
            # Mutate a tool's schema in place: same tool name, changed parameters.
            mutated = json.loads(json.dumps(current_tools[0]))
            mutated["parameters"]["properties"]["verbosity"] = {"type": "string"}
            mutated["description"] = mutated["description"] + " (v2)"
            current_tools = [mutated] + current_tools[1:]
        if reorder_step is not None and i == reorder_step and len(current_tools) > 1:
            current_tools = current_tools[1:] + current_tools[:1]

        if compaction_step is not None and i == compaction_step and len(history) > 4:
            # Replace the middle of the history with a compacted summary. Indices
            # 0 and 1 (system, user) are never touched, so the grouping key holds,
            # but the item at index 2 genuinely changes, so every cached token
            # after the opening is really invalidated — a measurable cache break.
            #
            # The export's stated premise is that each request's input contains
            # every item of the previous one, and ALL THREE project-method*
            # loaders order a trajectory's calls by serialized length on the
            # strength of that. A naive compaction shrinks the history and would
            # silently scramble reconstructed call order. So the summary is padded
            # until the compacted history is still comfortably longer than the
            # previous request: the prefix changes (cache lost) while length stays
            # monotone (ordering premise preserved).
            head, tail = history[:2], history[-2:]
            previous_length = len(json.dumps(history, ensure_ascii=False, separators=(",", ":")))
            summary_text = f"[context compacted at step {i}] " + "summary " * rng.randint(20, 80)
            candidate = head + [msg("assistant", summary_text)] + tail
            # Margin absorbs the small differences between the loaders'
            # serialization variants (sort_keys on/off).
            required = previous_length * 1.05 + 64
            while len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) <= required:
                summary_text += "summary " * 40
                candidate = head + [msg("assistant", summary_text)] + tail
            history = candidate

        requests.append({
            "model": active_model,
            "input": [json.loads(json.dumps(item)) for item in history],
            "tools": [json.loads(json.dumps(t)) for t in current_tools],
        })
        approx_tokens = max(1, len(json.dumps(history)) // 4)
        latencies.append(round(approx_tokens * (MODEL_SPEED_MS_PER_KTOK[active_model] / 1000.0) * rng.uniform(0.8, 1.3), 1))

        if active_model.startswith("gpt"):
            history.append({"type": "reasoning", "id": f"rs_{task_id}_{i}", "summary": []})

        if current_tools:
            tool_name = tool_names[i % len(tool_names)]
            call_id = f"toolu_{task_id}_{i}"
            arguments = repeat_args if repeat_args is not None else tool_arguments(rng, tool_name, i)
            repeat_args = None
            history.append({"type": "function_call", "name": tool_name, "arguments": json.dumps(arguments), "call_id": call_id})
            if i in fail_steps:
                history.append({"type": "function_call_output", "call_id": call_id,
                                "output": f"step-{i} error: {tool_name} failed. " + "log " * rng.randint(5, 60)})
                repeat_args = arguments  # retry with identical args -> repetition signal
            elif i == warn_step:
                history.append({"type": "function_call_output", "call_id": call_id,
                                "output": f"step-{i} warn: unexpected state, remediation required. " + "log " * rng.randint(5, 60)})
            else:
                history.append({"type": "function_call_output", "call_id": call_id,
                                "output": f"step-{i} ok. " + "log " * rng.randint(10, 320)})
        else:
            history.append(msg("assistant", f"step-{i} reasoning without tools. " + "text " * rng.randint(10, 80)))

        if i == clarify_step:
            history.append(msg("user", f"Follow-up on task {task_id}: clarifying scope for {project}. " + "filler " * rng.randint(5, 50)))

        if i == n_calls - 2 and n_calls >= 2:
            if graded:
                history.append(msg("assistant", f"Outcome: task {task_id} complete. quality_score={realized:.3f}"))
            else:
                history.append(msg("assistant", "Work completed and verified."))

    pool = list(era["pool"])
    entry = {
        "key": key,
        "task_id": task_id,
        "task_type": task_type,
        "difficulty": difficulty,
        "domain": domain["name"],
        "era": era["name"],
        "era_index": spec["era_index"],
        "timestep": spec["timestep"],
        "logged_model": model,
        "n_calls": n_calls,
        "tool_set": [] if edge_case == "no_tools" else tool_names,
        # --- outcome / label ---
        "ground_truth_quality": round(realized, 4) if graded else None,
        "latent_expected_quality": round(base_quality, 4),
        "outcome_noise_scale": OUTCOME_NOISE,
        # --- scenario flags ---
        "has_failure": has_failure,
        "n_failures": len(fail_steps),
        "branching": branching,
        "multi_turn": multi_turn,
        "model_switch": model_switch,
        "switch_step": switch_step,
        "switch_model": switch_model,
        "context_compaction": context_compaction,
        "compaction_step": compaction_step,
        "tool_schema_change": tool_schema_change,
        "tool_reorder": tool_reorder,
        "edge_case": edge_case,
        "graded": graded,
        # --- causal / OPE ground truth (impossible to have in real data) ---
        "true_propensity": round(spec["propensities"][model], 6),
        "true_propensity_all": {m: round(p, 6) for m, p in spec["propensities"].items()},
        "candidate_pool": pool,
        "positivity_violation": FORBIDDEN_PAIRS.get((task_type, difficulty)),
        "oracle_model_quality_only": oracle_choice(task_type, difficulty, pool),
        "oracle_model_cost_aware_070": cost_aware_oracle_choice(task_type, difficulty, pool, 0.70),
        "counterfactual_expected_quality": {m: round(expected_quality(m, task_type, difficulty), 4) for m in pool},
        # --- cost / latency ---
        "era_price_multiplier": era_multiplier,
        "effective_price_per_mtok": {m: round(MODEL_PRICE[m] * era_multiplier, 4) for m in pool},
        "latency_ms_per_call": latencies,
        "total_latency_ms": round(sum(latencies), 1),
        # --- replicate bookkeeping ---
        "replicate_group": spec.get("replicate_group"),
        "replicate_index": spec.get("replicate_index"),
    }
    return requests, entry


# --------------------------------------------------------------------------
# Corpus assembly
# --------------------------------------------------------------------------

def choose_task_type(rng: random.Random, era_index: int) -> str:
    """Task mix DRIFTS across eras — data_analysis is near-absent early and common
    late, so a router trained on early data faces a genuinely shifted test mix."""
    weights = []
    for task_type in TASK_TYPES:
        if task_type == "data_analysis":
            weights.append([0.02, 0.06, 0.28][era_index])
        elif task_type == "code_generation":
            weights.append([0.28, 0.24, 0.16][era_index])
        else:
            weights.append(0.18)
    return rng.choices(list(TASK_TYPES), weights=weights, k=1)[0]


def choose_domain(rng: random.Random, era_index: int) -> dict:
    available = [d for d in DOMAINS if not d["late_only"] or era_index == 2]
    return rng.choice(available)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="..")
    parser.add_argument("--n-tasks", type=int, default=600)
    parser.add_argument("--replicate-groups", type=int, default=30)
    parser.add_argument("--replicates-per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "export").mkdir(parents=True, exist_ok=True)

    era_shares = [era["share"] for era in ERAS]
    total_share = sum(era_shares)
    era_counts = [int(round(args.n_tasks * share / total_share)) for share in era_shares]
    era_counts[-1] = args.n_tasks - sum(era_counts[:-1])

    specs: list[dict] = []
    task_id = 1
    timestep = 0

    # --- main corpus ---
    for era_index, (era, count) in enumerate(zip(ERAS, era_counts)):
        for _ in range(count):
            task_type = choose_task_type(rng, era_index)
            difficulty = rng.choices(DIFFICULTIES, weights=[0.38, 0.40, 0.22], k=1)[0]
            domain = choose_domain(rng, era_index)
            model, propensities = sample_model(rng, era, task_type, difficulty)
            specs.append({
                "task_id": task_id, "task_type": task_type, "difficulty": difficulty,
                "domain": domain, "era": era, "era_index": era_index, "timestep": timestep,
                "model": model, "propensities": propensities, "edge_case": None,
            })
            task_id += 1
            timestep += 1

    # --- replicate groups: identical spec, different draws ---
    # Lets a methodology separate reproducible specialist advantage from
    # single-draw label noise (arXiv:2607.03436). Forced graded so the replicate
    # scores are actually observable.
    for group in range(args.replicate_groups):
        era_index = rng.randrange(len(ERAS))
        era = ERAS[era_index]
        task_type = choose_task_type(rng, era_index)
        difficulty = rng.choice(DIFFICULTIES)
        domain = choose_domain(rng, era_index)
        model, propensities = sample_model(rng, era, task_type, difficulty)
        for replicate_index in range(args.replicates_per_group):
            specs.append({
                "task_id": task_id, "task_type": task_type, "difficulty": difficulty,
                "domain": domain, "era": era, "era_index": era_index, "timestep": timestep,
                "model": model, "propensities": propensities, "edge_case": None,
                "replicate_group": f"rep-{group}", "replicate_index": replicate_index,
                "force_graded": True,
            })
            task_id += 1
            timestep += 1

    # --- edge cases: a few of each, spread across eras ---
    edge_cases = ["single_call", "no_tools", "oversized_prompt", "unicode_heavy", "long_horizon", "all_failure"]
    for edge_case in edge_cases:
        for _ in range(6):
            era_index = rng.randrange(len(ERAS))
            era = ERAS[era_index]
            task_type = choose_task_type(rng, era_index)
            difficulty = rng.choice(DIFFICULTIES)
            domain = choose_domain(rng, era_index)
            model, propensities = sample_model(rng, era, task_type, difficulty)
            specs.append({
                "task_id": task_id, "task_type": task_type, "difficulty": difficulty,
                "domain": domain, "era": era, "era_index": era_index, "timestep": timestep,
                "model": model, "propensities": propensities, "edge_case": edge_case,
            })
            task_id += 1
            timestep += 1

    # --- build ---
    lines: list[dict] = []
    manifest: list[dict] = []
    for spec in specs:
        requests, entry = build_trajectory(rng, spec)
        lines.extend(requests)
        manifest.append(entry)

    # Shuffle: real chunked exports give no ordering guarantee, and any pipeline
    # that accidentally depends on line order should break loudly here rather
    # than silently on the real data. (Temporal order lives in the manifest's
    # `timestep`, which is where a drift-aware method should read it from.)
    rng.shuffle(lines)

    export_path = out / "export" / "trajectories_v1_00.jsonl"
    with export_path.open("w", encoding="utf-8") as handle:
        for request in lines:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    (out / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = build_summary(manifest, lines, args.seed)
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(manifest)} trajectories, {len(lines)} requests -> {export_path}")
    print(json.dumps(summary["coverage"], indent=2))


def build_summary(manifest: list[dict], lines: list[dict], seed: int) -> dict:
    def count(predicate) -> int:
        return sum(1 for entry in manifest if predicate(entry))

    def distribution(field: str) -> dict:
        out: dict = {}
        for entry in manifest:
            out[str(entry[field])] = out.get(str(entry[field]), 0) + 1
        return dict(sorted(out.items()))

    call_counts = sorted(entry["n_calls"] for entry in manifest)
    return {
        "source": "dataset2/scripts/generate_dataset2.py",
        "seed": seed,
        "trajectories": len(manifest),
        "requests": len(lines),
        "models": sorted({entry["logged_model"] for entry in manifest}
                         | {entry["switch_model"] for entry in manifest if entry["switch_model"]}),
        "coverage": {
            "task_type": distribution("task_type"),
            "difficulty": distribution("difficulty"),
            "domain": distribution("domain"),
            "era": distribution("era"),
            "logged_model": distribution("logged_model"),
            "edge_case": distribution("edge_case"),
            "graded": count(lambda e: e["graded"]),
            "has_failure": count(lambda e: e["has_failure"]),
            "branching": count(lambda e: e["branching"]),
            "multi_turn": count(lambda e: e["multi_turn"]),
            "model_switch": count(lambda e: e["model_switch"]),
            "context_compaction": count(lambda e: e["context_compaction"]),
            "tool_schema_change": count(lambda e: e["tool_schema_change"]),
            "tool_reorder": count(lambda e: e["tool_reorder"]),
            "replicated": count(lambda e: e["replicate_group"] is not None),
            "in_positivity_violation_cell": count(lambda e: e["positivity_violation"] is not None),
            "calls_min_median_max": [call_counts[0], call_counts[len(call_counts) // 2], call_counts[-1]],
        },
    }


if __name__ == "__main__":
    main()
