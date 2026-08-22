from __future__ import annotations

import json

from method1.data.schema import RequestRecord


def _encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def common_prefix_tokens(previous: RequestRecord, current: RequestRecord) -> int:
    left = [_encoded(item) for item in previous.input]
    right = [_encoded(item) for item in current.input]
    common = 0
    for first, second in zip(left, right):
        if first != second:
            break
        common += len(first) + 1
    return common // 4


def extract_cache_features(calls: list[RequestRecord]) -> dict[str, float]:
    if len(calls) < 2:
        return {"prefix_overlap_tokens": 0.0, "cacheable_fraction": 0.0}
    overlaps = [common_prefix_tokens(left, right) for left, right in zip(calls, calls[1:])]
    total = sum(call.estimated_tokens for call in calls)
    return {
        "prefix_overlap_tokens": float(sum(overlaps)),
        "cacheable_fraction": float(sum(overlaps) / max(1, total)),
        "max_prefix_overlap_tokens": float(max(overlaps)),
    }

