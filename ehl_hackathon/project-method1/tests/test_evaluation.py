import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method1.data.schema import RequestRecord, Trajectory
from method1.evaluation.metrics import PolicyMetrics, pareto_frontier


class EvaluationTests(unittest.TestCase):
    def test_frontier_keeps_improvements(self):
        points = [PolicyMetrics("a", 1, 1, 1.0, 0.5, 0.1, 0, 1, 0), PolicyMetrics("b", 1, 1, 2.0, 0.8, 0.1, 0, 1, 0), PolicyMetrics("c", 1, 1, 3.0, 0.7, 0.1, 0, 1, 0)]
        self.assertEqual([item.policy for item in pareto_frontier(points)], ["a", "b"])

