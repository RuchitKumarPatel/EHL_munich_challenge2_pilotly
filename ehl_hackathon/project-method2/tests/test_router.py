import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method2.cost import pricing
from method2.data import Call, Trajectory
from method2.reward import CalibratedRewardModel
from method2.router import SafeRouter


def build(key, model, text):
    return Trajectory(key, [Call(model, [{"role": "user", "content": text}], [], "test", 1)])


class RouterTests(unittest.TestCase):
    def test_unsupported_candidate_falls_back_to_logged_model(self):
        train = [build("1" * 64, "logged", "task one"), build("2" * 64, "logged", "task two"), build("3" * 64, "logged", "task three")]
        labels = {item.key: (0.8, 1.0) for item in train}
        reward = CalibratedRewardModel(["logged", "other"], minimum_support=3).fit(train, labels, [])
        decision = SafeRouter(reward, pricing(["logged", "other"])).route(train[0], "logged")
        self.assertEqual(decision.model, "logged")
        self.assertIn(decision.reason, {"insufficient_logged_support", "insufficient_counterfactual_support"})
