from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Any

from method3.data.schema import OpeningContext

# Number of hashed bag-of-words buckets (the "hashing trick": Weinberger et al. 2009)
# added below. Aggregate stats alone (char/word counts, binary markers) are blind to
# WHICH words appear — "alpha task" and "beta task" produce identical aggregate
# features. Hashed word counts give the reward/propensity models real lexical signal
# (domain, task-type verb, placeholder tokens like <PERSON_A> vs <PERSON_B>) without a
# growing vocabulary table or a new dependency. zlib.crc32 (not builtin `hash()`,
# which is randomized per-process via PYTHONHASHSEED) so bucket assignment is
# identical between the process that trains a model and any later process that loads
# and predicts with it.
_HASH_BUCKETS = 16

# Separate hashed space for DECLARED TOOL NAMES. The tool set is by far the most
# discriminative pre-decision signal available: it is chosen by the caller, sits in
# the request itself, and in practice separates task families almost cleanly (a
# code task ships file_write/code_exec, an analysis task ships sql_query). Measured
# in project-method4, bucketing on tool signature produced 14 distinct context
# buckets where the lexical hash produced only 3. Aggregate counts
# (`tool_count`, `distinct_tool_names`) throw that identity away, so tool names get
# their own hashed features here.
_TOOL_HASH_BUCKETS = 8

# Fixed, explicit schema — every extractor call returns exactly these keys, always,
# regardless of what's present in a given trajectory. A dynamically-derived key set
# (e.g. "whatever keys happen to be in the first training row") is a latent bug: if a
# later batch's first row is missing a key the train-time and predict-time feature
# vectors silently misalign. Fixed names close that off entirely.
OPENING_FEATURE_NAMES: tuple[str, ...] = (
    "prompt_tokens", "tool_count", "distinct_tool_names",
    "text_chars", "text_words", "text_unique_words",
    "has_code_marker", "has_question", "has_error_word", "has_image",
    "system_present", "user_content_items",
    "type_token_ratio", "max_token_repetition",
) + tuple(f"text_hash_{i}" for i in range(_HASH_BUCKETS)) \
  + tuple(f"tool_hash_{i}" for i in range(_TOOL_HASH_BUCKETS))


def _hash_bucket(word: str, buckets: int = _HASH_BUCKETS) -> int:
    return zlib.crc32(word.encode("utf-8")) % buckets


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_strings(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_strings(item))
        return out
    return []


def extract_opening_features(opening: OpeningContext) -> dict[str, float]:
    """Features computable strictly from the opening context (system + first user
    message + declared tool schema) — the only information available before a
    routing decision fires. NEVER extend this to read anything beyond `opening`;
    that is exactly the leakage bug this module exists to prevent (see
    tests/test_no_feature_leakage.py and docs/methodology.md)."""
    text = " ".join(_strings(list(opening.opening_items)))
    words = re.findall(r"[A-Za-z0-9_<>-]+", text.lower())
    counts = Counter(words)
    tool_names = {tool.get("name", tool.get("type", "unknown")) for tool in opening.tools if isinstance(tool, dict)}
    system_present = any(isinstance(item, dict) and item.get("role") == "system" for item in opening.opening_items)
    user_item = next((item for item in opening.opening_items if isinstance(item, dict) and item.get("role") == "user"), None)
    user_content = user_item.get("content") if user_item else None
    user_content_items = len(user_content) if isinstance(user_content, list) else (1 if user_content else 0)

    # Hash over the DISTINCT vocabulary, not the raw token stream.
    #
    # Counting every occurrence lets a token repeated hundreds of times dominate its
    # bucket and swamp the content words that actually identify the task. On
    # dataset2 that is exactly what happened: the padding token appears up to ~400
    # times per prompt, so the count-based hash was effectively a noisy measure of
    # prompt length (which `prompt_tokens` already reports) rather than of topic.
    # Presence-weighting each distinct token once keeps the lexical identity signal
    # and drops the length signal that was drowning it.
    #
    # A sub-linear count term is kept alongside so genuine emphasis is not lost
    # entirely, but it cannot dominate: log1p of the count, averaged over the
    # distinct tokens in the bucket.
    bucket_types = [0] * _HASH_BUCKETS
    bucket_weight = [0.0] * _HASH_BUCKETS
    for word, count in counts.items():
        bucket = _hash_bucket(word)
        bucket_types[bucket] += 1
        bucket_weight[bucket] += math.log1p(count)
    hash_features = {
        f"text_hash_{i}": (bucket_weight[i] / bucket_types[i]) if bucket_types[i] else 0.0
        for i in range(_HASH_BUCKETS)
    }

    # Declared tool identity, hashed into its own space (see _TOOL_HASH_BUCKETS).
    tool_hash_counts = [0] * _TOOL_HASH_BUCKETS
    for name in tool_names:
        tool_hash_counts[_hash_bucket(str(name), _TOOL_HASH_BUCKETS)] += 1
    tool_hash_features = {f"tool_hash_{i}": float(c) for i, c in enumerate(tool_hash_counts)}

    # Lexical-diversity signals: filler-padded prompts have a very low type/token
    # ratio and a very high single-token repetition count, so these separate
    # "long because it is genuinely complex" from "long because it is padded" —
    # a distinction the raw length features cannot make.
    total_words = len(words)
    type_token_ratio = (len(counts) / total_words) if total_words else 0.0
    max_token_repetition = float(max(counts.values())) if counts else 0.0

    return {
        "type_token_ratio": type_token_ratio,
        "max_token_repetition": math.log1p(max_token_repetition),
        **tool_hash_features,
        "prompt_tokens": float(opening.first_call_tokens),
        "tool_count": float(len(opening.tools)),
        "distinct_tool_names": float(len(tool_names)),
        "text_chars": float(len(text)),
        "text_words": float(len(words)),
        "text_unique_words": float(len(counts)),
        "has_code_marker": float("```" in text or "apply_patch" in text),
        "has_question": float("?" in text),
        "has_error_word": float(any(word in counts for word in ("error", "failed", "failure", "exception"))),
        "has_image": float("input_image" in text or any("input_image" in str(item) for item in opening.opening_items)),
        "system_present": float(system_present),
        "user_content_items": float(user_content_items),
        **hash_features,
    }
