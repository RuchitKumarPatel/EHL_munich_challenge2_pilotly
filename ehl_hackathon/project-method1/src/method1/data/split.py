from __future__ import annotations

import hashlib
from collections import defaultdict

from .schema import Trajectory


def split_trajectories(trajectories: list[Trajectory], train_fraction: float = 0.6, validation_fraction: float = 0.2) -> dict[str, list[Trajectory]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must be positive and leave a test partition")
    buckets = defaultdict(list)
    for trajectory in trajectories:
        value = int(hashlib.sha256(trajectory.key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if value < train_fraction:
            bucket = "train"
        elif value < train_fraction + validation_fraction:
            bucket = "validation"
        else:
            bucket = "test"
        buckets[bucket].append(trajectory)
    return {key: sorted(buckets[key], key=lambda item: item.key) for key in ("train", "validation", "test")}

