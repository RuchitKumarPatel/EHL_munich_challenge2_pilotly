from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_strings(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(_strings(item))
        return output
    return []


def extract_text_features(items: list[dict[str, Any]]) -> dict[str, float]:
    text = " ".join(_strings(items))
    words = re.findall(r"[A-Za-z0-9_<>-]+", text.lower())
    counts = Counter(words)
    return {
        "text_chars": float(len(text)),
        "text_words": float(len(words)),
        "text_unique_words": float(len(counts)),
        "text_lines": float(text.count("\n") + 1 if text else 0),
        "has_code_marker": float("```" in text or "apply_patch" in text),
        "has_error_word": float(any(word in counts for word in ("error", "failed", "failure", "exception"))),
        "has_question": float("?" in text),
    }

