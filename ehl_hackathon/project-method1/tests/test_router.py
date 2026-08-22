import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method1.data.schema import RequestRecord, Trajectory
from method1.pricing import CostModel
from method1.routing.difficulty_router import DifficultyRouter


class RouterTests(unittest.TestCase):
    def test_router_returns_one_model_for_trajectory(self):
        calls = [RequestRecord("cheap", [{"role": "user", "content": "task"}], [], "x", 1)]
        trajectory = Trajectory("x", calls)
        cost_model = CostModel({"cheap": {"input": 0.1, "cached_input": 0.01}, "strong": {"input": 1.0, "cached_input": 0.1}})
        router = DifficultyRouter(["cheap", "strong"], cost_model).fit([trajectory], {"x": 0.8})
        decision = router.route(trajectory)
        self.assertEqual(len(decision.route), 1)
        self.assertIn(decision.model, {"cheap", "strong"})

