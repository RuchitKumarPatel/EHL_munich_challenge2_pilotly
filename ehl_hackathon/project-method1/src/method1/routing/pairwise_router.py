from __future__ import annotations

from method1.data.schema import Trajectory

from .base import RouteDecision
from .difficulty_router import DifficultyRouter


class PairwiseRouter:
    def __init__(self, difficulty_router: DifficultyRouter) -> None:
        self.difficulty_router = difficulty_router

    def route(self, trajectory: Trajectory) -> RouteDecision:
        return self.difficulty_router.route(trajectory)

