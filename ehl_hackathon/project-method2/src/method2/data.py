from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def token_estimate(value: Any) -> int:
    return max(1, len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) // 4)


@dataclass(frozen=True)
class Call:
    model: str
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    source: str
    line: int

    @property
    def tokens(self) -> int:
        return token_estimate(self.input)


@dataclass
class Trajectory:
    key: str
    calls: list[Call]

    @property
    def model(self) -> str:
        models = {call.model for call in self.calls}
        return next(iter(models)) if len(models) == 1 else "mixed"


def _key(call: Call) -> str:
    opening = []
    for item in call.input:
        if isinstance(item, dict) and item.get("role") in {"system", "user"}:
            opening.append(item)
            if item.get("role") == "user":
                break
    text = json.dumps(opening, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_trajectories(source: str | Path) -> list[Trajectory]:
    path = Path(source)
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files found in {source}")
    grouped: dict[str, list[Call]] = {}
    for file in paths:
        with file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != {"model", "input", "tools"}:
                    raise ValueError(f"schema mismatch at {file}:{line_number}")
                if not isinstance(row["model"], str) or not isinstance(row["input"], list) or not isinstance(row["tools"], list):
                    raise ValueError(f"invalid types at {file}:{line_number}")
                call = Call(row["model"], row["input"], row["tools"], str(file), line_number)
                grouped.setdefault(_key(call), []).append(call)
    output = []
    for key, calls in grouped.items():
        output.append(Trajectory(key, sorted(calls, key=lambda item: (item.tokens, item.source, item.line))))
    return sorted(output, key=lambda item: item.key)


def split(trajectories: list[Trajectory]) -> dict[str, list[Trajectory]]:
    buckets = {"train": [], "calibration": [], "test": []}
    for trajectory in trajectories:
        value = int(hashlib.sha256(trajectory.key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        buckets["train" if value < 0.6 else "calibration" if value < 0.8 else "test"].append(trajectory)
    if not buckets["test"]:
        buckets["test"] = list(trajectories)
    return buckets
