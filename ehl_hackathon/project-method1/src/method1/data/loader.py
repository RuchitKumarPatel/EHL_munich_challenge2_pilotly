from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import RequestRecord, Trajectory


def jsonl_paths(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl"))


def load_records(source: str | Path) -> Iterator[RequestRecord]:
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
                yield RequestRecord.from_mapping(value, str(path), line_number)


def _opening_context(record: RequestRecord) -> str:
    opening = []
    for item in record.input:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role in {"system", "user"}:
            opening.append(item)
        if role == "user":
            break
    encoded = json.dumps(opening, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def group_trajectories(records: Iterable[RequestRecord]) -> list[Trajectory]:
    grouped: dict[str, list[RequestRecord]] = {}
    for record in records:
        grouped.setdefault(_opening_context(record), []).append(record)
    trajectories = []
    for key, calls in grouped.items():
        ordered = sorted(calls, key=lambda call: (call.estimated_tokens, call.source_file, call.source_line))
        trajectories.append(Trajectory(key, ordered))
    return sorted(trajectories, key=lambda trajectory: trajectory.key)


def load_trajectories(source: str | Path) -> list[Trajectory]:
    return group_trajectories(load_records(source))

