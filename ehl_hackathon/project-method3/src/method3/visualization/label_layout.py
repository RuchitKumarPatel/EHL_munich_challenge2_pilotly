from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Box:
    """An axis-aligned label box in data coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: "Box", pad_x: float = 0.0, pad_y: float = 0.0) -> bool:
        return not (
            self.x1 + pad_x <= other.x0
            or other.x1 + pad_x <= self.x0
            or self.y1 + pad_y <= other.y0
            or other.y1 + pad_y <= self.y0
        )


def place_labels(
    points: list[tuple[float, float]],
    sizes: list[tuple[float, float]],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    candidate_offsets: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Choose a non-overlapping position for each label.

    Deterministic greedy placement: for each point in turn, try a ring of candidate
    offsets and take the first that collides with neither an already-placed label
    nor any data marker, and that stays inside the axes. If every candidate
    collides, take the one with the smallest total overlap area so a label is never
    dropped — a missing label is worse than a slightly crowded one.

    Written rather than pulled in (adjustText is the usual choice) to keep this
    project's zero-new-dependency rule, and because a deterministic placer can be
    unit-tested for the actual property that matters: no two label boxes overlap.

    `points` are marker positions and `sizes` are (width, height) per label, both in
    DATA coordinates. Returns one (x, y) bottom-left label position per point.
    """
    if len(points) != len(sizes):
        raise ValueError("points and sizes must be the same length")
    if not points:
        return []

    x_span = max(x_limits[1] - x_limits[0], 1e-12)
    y_span = max(y_limits[1] - y_limits[0], 1e-12)
    step_x = 0.022 * x_span
    step_y = 0.030 * y_span

    if candidate_offsets is None:
        candidate_offsets = []
        # Rings of increasing radius; within a ring, prefer right/up placements
        # first because they read most naturally next to a scatter marker.
        for ring in (1, 2, 3, 4):
            for dx, dy in ((1, 0.4), (1, -0.4), (0.2, 1), (0.2, -1), (-1, 0.4), (-1, -0.4), (1, 1.2), (-1, 1.2), (1, -1.2), (-1, -1.2)):
                candidate_offsets.append((dx * ring * step_x, dy * ring * step_y))

    # Data markers are obstacles too, so labels do not sit on top of the dots.
    marker_half_x = 0.010 * x_span
    marker_half_y = 0.014 * y_span
    marker_boxes = [Box(px - marker_half_x, py - marker_half_y, px + marker_half_x, py + marker_half_y) for px, py in points]

    placed: list[Box] = []
    positions: list[tuple[float, float]] = []

    for index, ((px, py), (width, height)) in enumerate(zip(points, sizes)):
        best_position = None
        best_penalty = None
        for offset_x, offset_y in candidate_offsets:
            x0 = px + offset_x if offset_x >= 0 else px + offset_x - width
            y0 = py + offset_y - height / 2.0
            box = Box(x0, y0, x0 + width, y0 + height)

            outside = (box.x0 < x_limits[0] or box.x1 > x_limits[1] or box.y0 < y_limits[0] or box.y1 > y_limits[1])
            penalty = 0.0
            if outside:
                penalty += 10.0
            for other in placed:
                penalty += _overlap_area(box, other) / (x_span * y_span)
            for other_index, marker in enumerate(marker_boxes):
                if other_index == index:
                    continue
                penalty += _overlap_area(box, marker) / (x_span * y_span)
            # Its own marker must not be covered either.
            penalty += _overlap_area(box, marker_boxes[index]) / (x_span * y_span)

            if penalty == 0.0:
                best_position, best_penalty = (x0, y0), 0.0
                break
            if best_penalty is None or penalty < best_penalty:
                best_position, best_penalty = (x0, y0), penalty

        x0, y0 = best_position  # type: ignore[misc]
        placed.append(Box(x0, y0, x0 + width, y0 + height))
        positions.append((x0, y0))

    return positions


def _overlap_area(a: Box, b: Box) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height
