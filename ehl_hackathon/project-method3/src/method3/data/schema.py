from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Crude token estimate: serialized chars / 4. No usage field exists anywhere in
    this pipeline's inputs; every token number derived from this is an estimate."""
    return max(1, len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) // 4)


@dataclass(frozen=True)
class Call:
    """One logged LLM request. `input` is the full growing history up to and
    including this call (Responses-format item list), matching the organizer's
    export and dataset1's schema exactly."""
    model: str
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    source_file: str
    source_line: int

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.input)

    @classmethod
    def from_mapping(cls, value: dict[str, Any], source_file: str, source_line: int) -> "Call":
        expected = {"model", "input", "tools"}
        actual = set(value)
        if actual != expected:
            raise ValueError(f"schema mismatch at {source_file}:{source_line}: expected keys {sorted(expected)}, got {sorted(actual)}")
        if not isinstance(value["model"], str) or not value["model"]:
            raise ValueError(f"invalid model at {source_file}:{source_line}")
        if not isinstance(value["input"], list) or not isinstance(value["tools"], list):
            raise ValueError(f"input and tools must be lists at {source_file}:{source_line}")
        return cls(value["model"], value["input"], value["tools"], source_file, source_line)


@dataclass(frozen=True)
class OpeningContext:
    """The ONLY view of a trajectory a routing decision is allowed to see: the
    opening system+first-user messages and the declared tool schema of the first
    call. Deliberately holds no reference to any later call, so a router literally
    cannot read ahead — this is the pre-decision/post-hoc boundary enforced by type,
    not by convention. See tests/test_no_feature_leakage.py.

    `key` is the SHA-256 of [system_item, user_item] dumped with sort_keys and
    compact separators — identical to project-method1's
    data.trajectory_builder._opening_context and project-method2's data._key, so a
    dataset1 scenario_manifest.json (or anything else keyed that way) joins directly.
    """
    key: str
    opening_items: tuple[dict[str, Any], ...]  # the [system, user] items only
    tools: tuple[dict[str, Any], ...]
    first_call_tokens: int

    @classmethod
    def from_first_call(cls, first_call: Call, key: str) -> "OpeningContext":
        opening: list[dict[str, Any]] = []
        for item in first_call.input:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role in {"system", "user"}:
                opening.append(item)
            if role == "user":
                break
        return cls(key=key, opening_items=tuple(opening), tools=tuple(first_call.tools), first_call_tokens=first_call.estimated_tokens)


def opening_key(first_call: Call) -> str:
    """Standalone key derivation, usable before an OpeningContext exists (e.g. while
    grouping raw Call records into trajectories)."""
    import hashlib
    opening: list[dict[str, Any]] = []
    for item in first_call.input:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role in {"system", "user"}:
            opening.append(item)
        if role == "user":
            break
    encoded = json.dumps(opening, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class Trajectory:
    """A reconstructed trajectory: every logged Call sharing the same opening
    context, ordered by history length (= call order, since each call's `input` is
    the full growing history). Post-hoc-only object — carries every call, including
    ones that happen after any routing decision would fire. Never pass this directly
    to a router; pass `trajectory.opening` instead."""
    key: str
    calls: list[Call] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError(f"trajectory {self.key} has no calls")

    @property
    def opening(self) -> OpeningContext:
        return OpeningContext.from_first_call(self.calls[0], self.key)

    @property
    def logged_model(self) -> str:
        models = {call.model for call in self.calls}
        return sorted(models)[0] if len(models) == 1 else "mixed"

    @property
    def per_call_models(self) -> list[str]:
        """The real per-call model sequence — use this (not `logged_model`) whenever
        pricing/replaying what actually happened, since `logged_model` collapses to
        "mixed" for a switched trajectory and is not itself a priceable model."""
        return [call.model for call in self.calls]

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def estimated_tokens(self) -> int:
        return sum(call.estimated_tokens for call in self.calls)
