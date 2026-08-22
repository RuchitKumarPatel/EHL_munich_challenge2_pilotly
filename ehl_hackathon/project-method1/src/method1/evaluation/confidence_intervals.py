from __future__ import annotations

import random


def bootstrap_mean(values: list[float], repetitions: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(repetitions):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return sum(values) / len(values), lower, upper

