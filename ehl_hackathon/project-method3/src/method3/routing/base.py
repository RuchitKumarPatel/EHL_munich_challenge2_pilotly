from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from method3.data.schema import OpeningContext


@dataclass
class RouteDecision:
    model: str
    predicted_quality: float
    quality_lower: float
    quality_upper: float
    predicted_cost: float
    reason: str
    candidates: dict[str, dict[str, float]] = field(default_factory=dict)


@runtime_checkable
class Router(Protocol):
    """A routing decision may see the OpeningContext only — never a Trajectory. This
    is the pre-decision/post-hoc boundary enforced at the type level: there is no
    later-call information reachable from `opening`, so a conforming implementation
    cannot leak it even by accident. See tests/test_no_feature_leakage.py and
    docs/methodology.md. `candidate_models` is the pool of models this decision may
    choose among (not necessarily "models seen in the training data")."""

    def route(self, opening: OpeningContext, candidate_models: list[str]) -> RouteDecision:
        ...
