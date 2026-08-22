from __future__ import annotations

import math


def score_uncertainty(scores: dict[str, float]) -> float:
    if not scores:
        return 1.0
    values = sorted(scores.values(), reverse=True)
    if len(values) == 1:
        return 0.5
    margin = max(0.0, min(1.0, values[0] - values[1]))
    return 1.0 - margin


def entropy_uncertainty(scores: dict[str, float]) -> float:
    if not scores:
        return 1.0
    values = [max(0.000001, value) for value in scores.values()]
    total = sum(values)
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return entropy / max(1.0, math.log(len(values)))

