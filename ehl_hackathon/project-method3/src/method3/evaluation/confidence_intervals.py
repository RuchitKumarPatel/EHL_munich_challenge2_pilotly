from __future__ import annotations

import random


def bootstrap_mean(values: list[float], resamples: int = 500, seed: int = 7) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) via a fixed-seed nonparametric bootstrap. Deterministic
    across runs (fixed seed) so a reported number doesn't wobble between reproductions."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    samples = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    samples.sort()
    low_index = max(0, int(0.025 * resamples))
    high_index = min(resamples - 1, int(0.975 * resamples))
    return sum(values) / n, samples[low_index], samples[high_index]
