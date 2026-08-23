#!/usr/bin/env python3
"""Generate dataset1: a synthetic router-eval export covering the scenario
variations that the organizer's redacted export (trajectories_v1_00.jsonl)
does not, while staying schema-identical to it.

Schema contract (unchanged, enforced by both project-method1's
RequestRecord.from_mapping and project-method2's data.load_trajectories):
each JSONL line is exactly {"model": str, "input": [...], "tools": [...]}.
No extra top-level keys are allowed, so all the rich scenario labels this
generator knows about (task type, difficulty, domain, failure/cache-break/
branching flags, synthetic ground-truth quality, per-call latency) are
written to a companion `scenario_manifest.json`, keyed by the exact SHA-256
of the opening system+first-user messages -- the same key both pipelines'
trajectory-grouping code derives from the raw export. Join on that key.

Usage: python generate_dataset1.py --out .. --n-per-combo 2 --seed 13
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-opus-4-8",
          "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]

# ms per 1000 estimated-tokens, used only to synthesize a latency signal
# (the real export has none -- this fills identified gap #10).
MODEL_SPEED = {
    "claude-opus-5": 900, "claude-opus-4-8": 850, "claude-sonnet-5": 500,
    "claude-fable-5": 250, "gpt-5.6-terra": 700, "gpt-5.6-sol": 450, "gpt-5.6-luna": 300,
}

TOOL_DEFS = {
    "run_shell": {"type": "function", "name": "run_shell", "description": "Run a shell command",
                  "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
                  "strict": False},
    "file_read": {"type": "function", "name": "file_read", "description": "Read a file",
                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                  "strict": False},
    "web_search": {"type": "function", "name": "web_search", "description": "Search the web",
                   "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                   "strict": False},
    "code_exec": {"type": "function", "name": "code_exec", "description": "Execute a code snippet",
                  "parameters": {"type": "object", "properties": {"language": {"type": "string"}, "code": {"type": "string"}}, "required": ["code"]},
                  "strict": False},
    "retrieval": {"type": "function", "name": "retrieval", "description": "Retrieve a document from the knowledge base",
                  "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
                  "strict": False},
}

# Gap #1: task-type diversity (was: single template for every trajectory)
TASK_TYPES = {
    "qa_reasoning": {"verb": "answer the multi-step question", "tools": ["retrieval", "file_read"]},
    "code_generation": {"verb": "implement and verify the code change", "tools": ["run_shell", "file_read", "code_exec"]},
    "summarization_extraction": {"verb": "summarize and extract key fields from the documents", "tools": ["file_read", "retrieval"]},
    "multi_step_planning": {"verb": "plan and execute the multi-step operation", "tools": ["run_shell", "web_search", "retrieval"]},
    "creative_writing": {"verb": "draft the requested content", "tools": ["web_search", "file_read"]},
}

# Gap #2: a real difficulty gradient (was: call-count as the only proxy)
DIFFICULTIES = {
    "easy":   {"n_calls": (3, 5),  "filler": (20, 80),   "error_prob": 0.05, "quality_center": 0.85},
    "medium": {"n_calls": (5, 9),  "filler": (60, 200),  "error_prob": 0.15, "quality_center": 0.65},
    "hard":   {"n_calls": (8, 14), "filler": (150, 400), "error_prob": 0.30, "quality_center": 0.45},
}

# Gap #9: domain/context variety (was: one boilerplate <PERSON_A>/<COMPANY_A>)
DOMAINS = [
    {"name": "engineering", "person": "<PERSON_A>", "company": "<COMPANY_A>"},
    {"name": "finance",     "person": "<PERSON_B>", "company": "<COMPANY_B>"},
    {"name": "support",     "person": "<PERSON_C>", "company": "<COMPANY_C>"},
    {"name": "sales",       "person": "<PERSON_D>", "company": "<COMPANY_D>"},
    {"name": "data_ops",    "person": "<PERSON_E>", "company": "<COMPANY_E>"},
]

CACHE_BREAK_PROB = 0.15   # gap #5: mid-session model switch (cache invalidation)
MULTI_TURN_PROB = 0.25    # gap #7: follow-up / clarification turns
BRANCHING_PROB = 0.20     # gap #6: conditional remediation branch on a "warn" output
GROUND_TRUTH_PROB = 0.20  # gap #8: a graded subset with a recoverable outcome signal


def msg(role, text):
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def opening_key(system_item, user_item):
    """Must match method1.data.trajectory_builder._opening_context /
    method2.data._key: sha256 of [system, user] dumped with sort_keys, compact separators."""
    encoded = json.dumps([system_item, user_item], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tool_args(rng, tool_name, step):
    if tool_name == "run_shell":
        return {"cmd": f"step-{step} " + "x" * rng.randint(10, 200)}
    if tool_name == "file_read":
        return {"path": f"/data/step-{step}-" + "f" * rng.randint(4, 40)}
    if tool_name == "web_search":
        return {"query": f"step-{step} query " + "q" * rng.randint(4, 40)}
    if tool_name == "code_exec":
        return {"language": "python", "code": f"# step-{step}\n" + "pass\n" * rng.randint(1, 5)}
    if tool_name == "retrieval":
        return {"doc_id": f"doc-{step}-" + "d" * rng.randint(4, 20)}
    raise ValueError(tool_name)


def make_task(rng, task_id, task_type, difficulty, domain, model):
    diff = DIFFICULTIES[difficulty]
    tool_names = TASK_TYPES[task_type]["tools"]
    tools = [TOOL_DEFS[name] for name in tool_names]

    project = f"<PROJECT_NAME_{rng.randint(1, 5)}>"
    system_item = {"role": "system", "content": (
        f"You are an autonomous agent. Task context {task_id}. Domain: {domain['name']}. "
        f"User is {domain['person']} at {domain['company']}.")}
    user_text = (f"Task {task_id}: {TASK_TYPES[task_type]['verb']} for {domain['person']} on {project}. "
                 + "filler " * rng.randint(*diff["filler"]))
    user_item = msg("user", user_text)
    if rng.random() < 0.3:
        user_item["content"].append({"type": "input_image",
                                      "image_url": "data:image/png;base64,[base64 image redacted]", "detail": "auto"})
    key = opening_key(system_item, user_item)
    history = [system_item, user_item]

    has_failure = rng.random() < diff["error_prob"] * 2  # scenario-level flag, denser than per-call prob
    cache_break = rng.random() < CACHE_BREAK_PROB
    multi_turn = rng.random() < MULTI_TURN_PROB
    branching = rng.random() < BRANCHING_PROB
    graded = rng.random() < GROUND_TRUTH_PROB

    n_calls = rng.randint(*diff["n_calls"])
    if branching:
        n_calls += rng.randint(2, 4)

    fail_step = rng.randint(1, max(1, n_calls - 2)) if has_failure else None
    warn_step = rng.randint(1, max(1, n_calls - 3)) if branching else None
    clarify_step = rng.randint(1, max(1, n_calls - 2)) if multi_turn else None

    switch_step = None
    switch_model = None
    if cache_break and n_calls >= 4:
        switch_step = rng.randint(2, n_calls - 2)
        switch_model = rng.choice([m for m in MODELS if m != model])

    quality_center = diff["quality_center"]
    if has_failure:
        quality_center -= 0.15
    if branching:
        quality_center -= 0.05
    ground_truth_quality = None
    if graded:
        ground_truth_quality = max(0.0, min(1.0, rng.gauss(quality_center, 0.08)))

    requests = []
    latency_ms = []
    repeat_args = None  # set on a failed call so the retry reuses identical arguments

    for i in range(n_calls):
        active_model = model if switch_step is None or i < switch_step else switch_model
        requests.append({"model": active_model, "input": [json.loads(json.dumps(m)) for m in history], "tools": tools})
        tokens = max(1, len(json.dumps(history)) // 4)
        latency_ms.append(round(tokens * (MODEL_SPEED[active_model] / 1000.0) * rng.uniform(0.85, 1.2), 1))

        if active_model.startswith("gpt"):
            history.append({"type": "reasoning", "id": f"rs_{task_id}_{i}", "summary": []})

        tool_name = tool_names[i % len(tool_names)]
        call_id = f"toolu_{task_id}_{i}"
        if repeat_args is not None:
            args = repeat_args
            repeat_args = None
        else:
            args = tool_args(rng, tool_name, i)
        history.append({"type": "function_call", "name": tool_name, "arguments": json.dumps(args), "call_id": call_id})

        if i == fail_step:
            history.append({"type": "function_call_output", "call_id": call_id,
                             "output": f"step-{i} error: failed to complete {tool_name}. " + "log " * rng.randint(5, 50)})
            repeat_args = args  # next call retries with identical arguments (repetition signal)
        elif i == warn_step:
            history.append({"type": "function_call_output", "call_id": call_id,
                             "output": f"step-{i} warn: unexpected state, remediation required. " + "log " * rng.randint(5, 50)})
        else:
            history.append({"type": "function_call_output", "call_id": call_id,
                             "output": f"step-{i} ok. " + "log " * rng.randint(10, 300)})

        if i == clarify_step:
            history.append(msg("user", f"Follow-up on task {task_id}: clarifying scope for {project}. "
                                        + "filler " * rng.randint(5, 40)))

        if i == n_calls - 2:
            if graded:
                history.append(msg("assistant", f"Outcome: task {task_id} completed. quality_score={ground_truth_quality:.2f}"))
            else:
                history.append(msg("assistant", "Both moved and verified."))

    manifest_entry = {
        "key": key, "task_id": task_id, "task_type": task_type, "difficulty": difficulty,
        "domain": domain["name"], "logged_model": model, "n_calls": n_calls,
        "tool_set": tool_names, "has_failure": has_failure, "fail_step": fail_step,
        "cache_break": cache_break, "switch_step": switch_step, "switch_model": switch_model,
        "multi_turn": multi_turn, "clarify_step": clarify_step,
        "branching": branching, "warn_step": warn_step,
        "ground_truth_quality": ground_truth_quality,
        "latency_ms_per_call": latency_ms, "total_latency_ms": round(sum(latency_ms), 1),
    }
    return requests, manifest_entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--n-per-combo", type=int, default=2,
                     help="tasks generated per (task_type, difficulty, domain) combo")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "export").mkdir(parents=True, exist_ok=True)

    lines, manifest = [], []
    task_id = 1
    combos = [(tt, d, dom) for tt in TASK_TYPES for d in DIFFICULTIES for dom in DOMAINS]
    for task_type, difficulty, domain in combos:
        for _ in range(a.n_per_combo):
            model = rng.choice(MODELS)
            reqs, entry = make_task(rng, task_id, task_type, difficulty, domain, model)
            lines += reqs
            manifest.append(entry)
            task_id += 1

    rng.shuffle(lines)  # no ordering guarantee, matching the real export's behavior
    with open(out / "export" / "trajectories_v1_00.jsonl", "w", encoding="utf-8") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")

    with open(out / "scenario_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    def count(pred):
        return sum(1 for m in manifest if pred(m))

    summary = {
        "source": "dataset1/scripts/generate_dataset1.py",
        "seed": a.seed,
        "tasks": len(manifest),
        "requests": len(lines),
        "models": sorted({m["logged_model"] for m in manifest} | {m["switch_model"] for m in manifest if m["switch_model"]}),
        "coverage": {
            "task_type": {k: count(lambda m, k=k: m["task_type"] == k) for k in TASK_TYPES},
            "difficulty": {k: count(lambda m, k=k: m["difficulty"] == k) for k in DIFFICULTIES},
            "domain": {d["name"]: count(lambda m, n=d["name"]: m["domain"] == n) for d in DOMAINS},
            "has_failure": count(lambda m: m["has_failure"]),
            "cache_break": count(lambda m: m["cache_break"]),
            "multi_turn": count(lambda m: m["multi_turn"]),
            "branching": count(lambda m: m["branching"]),
            "graded_ground_truth": count(lambda m: m["ground_truth_quality"] is not None),
        },
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(manifest)} tasks, {len(lines)} requests -> {out}/export/trajectories_v1_00.jsonl")
    print(json.dumps(summary["coverage"], indent=2))


if __name__ == "__main__":
    main()
