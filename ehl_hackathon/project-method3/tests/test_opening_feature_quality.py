from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method3.data.schema import Call, Trajectory
from method3.features.opening_features import OPENING_FEATURE_NAMES, extract_opening_features

TOOLS_CODE = [
    {"type": "function", "name": "code_exec", "parameters": {}},
    {"type": "function", "name": "file_write", "parameters": {}},
]
TOOLS_SQL = [
    {"type": "function", "name": "sql_query", "parameters": {}},
    {"type": "function", "name": "retrieval", "parameters": {}},
]


def _opening(text: str, tools: list[dict]) -> object:
    history = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    ]
    return Trajectory("k", [Call("m", history, tools, "f", 0)]).opening


def test_tool_identity_is_represented_not_just_counted():
    # Two openings with the SAME number of tools but different tools must produce
    # different features. Aggregate counts alone cannot tell a code task from an
    # analysis task, which is what capped specialization before tool hashing.
    code = extract_opening_features(_opening("do the thing", TOOLS_CODE))
    sql = extract_opening_features(_opening("do the thing", TOOLS_SQL))
    assert code["tool_count"] == sql["tool_count"]
    assert code["distinct_tool_names"] == sql["distinct_tool_names"]
    assert code != sql          # but the hashed tool identity separates them
    tool_hashes = [n for n in OPENING_FEATURE_NAMES if n.startswith("tool_hash_")]
    assert any(code[n] != sql[n] for n in tool_hashes)


def test_repeated_filler_does_not_dominate_the_lexical_hash():
    # Regression guard for the real defect: counting every occurrence let a token
    # repeated hundreds of times swamp its bucket, turning the lexical hash into a
    # noisy proxy for prompt length. Presence-weighting fixes it, so adding more
    # padding must not keep inflating the hash features without bound.
    short = extract_opening_features(_opening("alpha beta " + "filler " * 10, TOOLS_CODE))
    long = extract_opening_features(_opening("alpha beta " + "filler " * 400, TOOLS_CODE))
    hashes = [n for n in OPENING_FEATURE_NAMES if n.startswith("text_hash_")]
    # 40x more padding must not move the lexical features anywhere near 40x.
    for name in hashes:
        if short[name] > 0:
            assert long[name] / short[name] < 3.0


def test_content_words_survive_heavy_padding():
    # The distinguishing content ("alpha" vs "gamma") must still change the
    # features even when buried in padding.
    a = extract_opening_features(_opening("alpha " + "filler " * 300, TOOLS_CODE))
    b = extract_opening_features(_opening("gamma " + "filler " * 300, TOOLS_CODE))
    hashes = [n for n in OPENING_FEATURE_NAMES if n.startswith("text_hash_")]
    assert any(a[n] != b[n] for n in hashes)


def test_type_token_ratio_separates_padded_from_dense_text():
    padded = extract_opening_features(_opening("filler " * 200, TOOLS_CODE))
    dense = extract_opening_features(_opening(" ".join(f"word{i}" for i in range(200)), TOOLS_CODE))
    assert dense["type_token_ratio"] > padded["type_token_ratio"]
    assert padded["max_token_repetition"] > dense["max_token_repetition"]


def test_feature_schema_is_complete_and_stable():
    features = extract_opening_features(_opening("x", TOOLS_CODE))
    assert set(features) == set(OPENING_FEATURE_NAMES)
    assert len(OPENING_FEATURE_NAMES) == len(set(OPENING_FEATURE_NAMES))


def test_no_tools_is_handled():
    features = extract_opening_features(_opening("x", []))
    assert features["tool_count"] == 0.0
    assert all(features[n] == 0.0 for n in OPENING_FEATURE_NAMES if n.startswith("tool_hash_"))


def test_empty_content_yields_finite_complete_features():
    # There is no such thing as a zero-token opening: the extractor walks every
    # value in the opening items, so the structural strings ("message", "user",
    # "input_text") are always present even when all content is empty. That is a
    # long-standing quirk rather than a defect — those tokens are constant across
    # every opening, so they carry no signal and the standardizer's scale floor
    # handles the resulting zero-variance columns. What matters here is that empty
    # content produces finite, complete features rather than a crash or a NaN.
    import math
    history = [
        {"role": "system", "content": ""},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": ""}]},
    ]
    opening = Trajectory("k", [Call("m", history, TOOLS_CODE, "f", 0)]).opening
    features = extract_opening_features(opening)
    assert set(features) == set(OPENING_FEATURE_NAMES)
    assert all(math.isfinite(v) for v in features.values())
    assert 0.0 <= features["type_token_ratio"] <= 1.0


def test_empty_user_text_still_reflects_the_system_message():
    features = extract_opening_features(_opening("", TOOLS_CODE))
    assert features["text_words"] > 0          # the system message is still text
    assert features["type_token_ratio"] > 0.0


def test_features_are_deterministic_across_calls():
    # crc32 (not builtin hash) so bucketing is stable across processes too.
    first = extract_opening_features(_opening("alpha beta gamma", TOOLS_CODE))
    second = extract_opening_features(_opening("alpha beta gamma", TOOLS_CODE))
    assert first == second
