import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method1.data.schema import RequestRecord, Trajectory
from method1.pricing import CostModel


class CacheModelTests(unittest.TestCase):
    def test_repeated_prefix_is_cheaper(self):
        first = RequestRecord("model", [{"role": "user", "content": "hello"}], [], "x", 1)
        second = RequestRecord("model", [{"role": "user", "content": "hello"}, {"type": "function_call", "arguments": "x"}], [], "x", 2)
        pricing = {"model": {"input": 1.0, "cached_input": 0.1}}
        cost_model = CostModel(pricing)
        cost, details = cost_model.trajectory_cost(Trajectory("x", [first, second]), ["model", "model"])
        self.assertGreater(details["cached_tokens"], 0)
        self.assertGreater(cost, 0)
