from __future__ import annotations

import hashlib
from collections import defaultdict

from .schema import Trajectory


def split_trajectories_temporal(
    trajectories: list[Trajectory],
    order_of: dict[str, float],
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, list[Trajectory]]:
    """Chronological train / calibration / test split, ordered by `order_of`
    (trajectory key -> timestep).

    A random split is the wrong evaluation when the log spans a real distribution
    shift: it puts later trajectories in train and earlier ones in test, so the
    model is partly fit on the future it is being asked to predict, and the score
    flatters any method that cannot actually handle drift. dataset2 contains
    deliberate drift — the task mix moves across eras, a new model appears only in
    the last one, and the logging policy itself changes — so the honest question
    "would this router have worked if deployed back then?" needs a chronological
    split.

    NOTE the tension with split-conformal: conformal coverage requires the
    calibration and test sets to be exchangeable, which a chronological split
    deliberately breaks. That is the point — the resulting coverage gap MEASURES
    the drift rather than hiding it. Read a temporal-split conformal bound as a
    drift diagnostic, not as a guarantee.

    Trajectories with no entry in `order_of` are placed last (treated as most
    recent) rather than dropped, so nothing silently disappears from evaluation.
    """
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1 or train_fraction + calibration_fraction >= 1:
        raise ValueError("fractions must be positive and leave a nonempty test partition")
    missing_order = float("inf")
    ordered = sorted(trajectories, key=lambda t: (order_of.get(t.key, missing_order), t.key))
    n = len(ordered)
    train_end = int(n * train_fraction)
    calibration_end = train_end + int(n * calibration_fraction)
    return {
        "train": ordered[:train_end],
        "calibration": ordered[train_end:calibration_end],
        "test": ordered[calibration_end:],
    }


def split_trajectories(trajectories: list[Trajectory], train_fraction: float = 0.6, calibration_fraction: float = 0.2) -> dict[str, list[Trajectory]]:
    """Hash-based 3-way split: train / calibration / test. The calibration split is
    distinct from train and reserved for split-conformal calibration (quality.conformal)
    — conformal validity requires calibration examples exchangeable with test examples
    and NOT used to fit the underlying reward model, so this must never be merged into
    train. Deterministic on trajectory.key (stable across reruns/model changes), same
    algorithm as project-method1's data.split.split_trajectories and
    project-method2's data.split."""
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1 or train_fraction + calibration_fraction >= 1:
        raise ValueError("fractions must be positive and leave a nonempty test partition")
    buckets: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        value = int(hashlib.sha256(trajectory.key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if value < train_fraction:
            bucket = "train"
        elif value < train_fraction + calibration_fraction:
            bucket = "calibration"
        else:
            bucket = "test"
        buckets[bucket].append(trajectory)
    return {key: sorted(buckets[key], key=lambda item: item.key) for key in ("train", "calibration", "test")}
