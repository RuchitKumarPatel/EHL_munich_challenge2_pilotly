from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Call, Trajectory, opening_key


def jsonl_paths(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl"))


def load_calls(source: str | Path) -> Iterator[Call]:
    paths = jsonl_paths(source)
    if not paths:
        raise FileNotFoundError(f"no JSONL files found in {source}")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"request must be an object at {path}:{line_number}")
                yield Call.from_mapping(value, str(path), line_number)


def group_trajectories(calls: Iterable[Call]) -> list[Trajectory]:
    grouped: dict[str, list[Call]] = {}
    for call in calls:
        grouped.setdefault(opening_key(call), []).append(call)
    trajectories = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda call: (call.estimated_tokens, call.source_file, call.source_line))
        trajectories.append(Trajectory(key, ordered))
    return sorted(trajectories, key=lambda trajectory: trajectory.key)


def load_trajectories(source: str | Path) -> list[Trajectory]:
    return group_trajectories(load_calls(source))
