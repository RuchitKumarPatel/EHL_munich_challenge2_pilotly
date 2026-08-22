from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from method1.data.schema import Trajectory


@dataclass
class RouteDecision:
    model: str
    route: list[str]
    expected_quality: float
    expected_cost: float
    uncertainty: float
    scores: dict[str, float]
    costs: dict[str, float]


class Router(Protocol):
    def route(self, trajectory: Trajectory) -> RouteDecision:
        ...

