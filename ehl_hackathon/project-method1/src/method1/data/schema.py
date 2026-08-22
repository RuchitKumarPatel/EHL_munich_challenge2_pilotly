from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


def estimate_tokens(value: Any) -> int:
    return max(1, len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) // 4)


@dataclass(frozen=True)
class RequestRecord:
    model: str
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    source_file: str
    source_line: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any], source_file: str, source_line: int) -> "RequestRecord":
        expected = {"model", "input", "tools"}
        actual = set(value)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(f"schema mismatch at {source_file}:{source_line}: missing={sorted(missing)} extra={sorted(extra)}")
        if not isinstance(value["model"], str) or not value["model"]:
            raise ValueError(f"invalid model at {source_file}:{source_line}")
        if not isinstance(value["input"], list) or not isinstance(value["tools"], list):
            raise ValueError(f"input and tools must be lists at {source_file}:{source_line}")
        return cls(value["model"], value["input"], value["tools"], source_file, source_line)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.input)


@dataclass
class Trajectory:
    key: str
    calls: list[RequestRecord]

    @property
    def logged_model(self) -> str:
        models = {call.model for call in self.calls}
        return sorted(models)[0] if len(models) == 1 else "mixed"

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def estimated_tokens(self) -> int:
        return sum(call.estimated_tokens for call in self.calls)

