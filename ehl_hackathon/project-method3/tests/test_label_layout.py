from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.visualization.label_layout import Box, place_labels


def _boxes(points, sizes, positions):
    return [Box(x0, y0, x0 + w, y0 + h) for (x0, y0), (w, h) in zip(positions, sizes)]


def _any_overlap(boxes) -> bool:
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a.overlaps(b):
                return True
    return False


def test_no_overlap_for_well_separated_points():
    points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
    sizes = [(0.12, 0.06)] * 3
    positions = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
    assert not _any_overlap(_boxes(points, sizes, positions))


def test_no_overlap_for_identical_stacked_points():
    # The hard case: several policies land on essentially the same cost/quality.
    points = [(0.5, 0.5)] * 5
    sizes = [(0.14, 0.06)] * 5
    positions = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
    assert not _any_overlap(_boxes(points, sizes, positions))


def test_no_overlap_for_tight_cluster():
    points = [(0.50, 0.50), (0.51, 0.505), (0.49, 0.495), (0.505, 0.49), (0.495, 0.51)]
    sizes = [(0.13, 0.055)] * 5
    positions = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
    assert not _any_overlap(_boxes(points, sizes, positions))


def test_no_overlap_across_random_layouts():
    rng = random.Random(11)
    for _ in range(40):
        n = rng.randint(2, 9)
        points = [(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)) for _ in range(n)]
        sizes = [(rng.uniform(0.08, 0.16), rng.uniform(0.04, 0.07)) for _ in range(n)]
        positions = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
        assert not _any_overlap(_boxes(points, sizes, positions))


def test_every_point_gets_exactly_one_label():
    points = [(0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8)]
    sizes = [(0.1, 0.05)] * 4
    positions = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
    assert len(positions) == len(points)


def test_labels_do_not_cover_their_own_markers():
    points = [(0.3, 0.3), (0.7, 0.7)]
    sizes = [(0.1, 0.05)] * 2
    positions = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
    for (x0, y0), (px, py), (w, h) in zip(positions, points, sizes):
        box = Box(x0, y0, x0 + w, y0 + h)
        inside = box.x0 <= px <= box.x1 and box.y0 <= py <= box.y1
        assert not inside


def test_placement_is_deterministic():
    points = [(0.5, 0.5), (0.52, 0.5), (0.5, 0.52)]
    sizes = [(0.12, 0.06)] * 3
    first = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
    second = place_labels(points, sizes, (0.0, 1.0), (0.0, 1.0))
    assert first == second


def test_single_point_is_handled():
    positions = place_labels([(0.5, 0.5)], [(0.1, 0.05)], (0.0, 1.0), (0.0, 1.0))
    assert len(positions) == 1


def test_empty_input_returns_empty():
    assert place_labels([], [], (0.0, 1.0), (0.0, 1.0)) == []


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        place_labels([(0.1, 0.1)], [(0.1, 0.05), (0.1, 0.05)], (0.0, 1.0), (0.0, 1.0))


def test_box_overlap_semantics():
    a = Box(0.0, 0.0, 1.0, 1.0)
    assert a.overlaps(Box(0.5, 0.5, 1.5, 1.5))
    assert not a.overlaps(Box(1.0, 0.0, 2.0, 1.0))   # touching edges do not overlap
    assert not a.overlaps(Box(2.0, 2.0, 3.0, 3.0))
