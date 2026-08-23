import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method2.data import Call, Trajectory
from method2.reward import CalibratedRewardModel


def trajectory(key, model, text):
    return Trajectory(key, [Call(model, [{"role": "user", "content": text}], [], "test", 1)])


class RewardTests(unittest.TestCase):
    def test_prediction_has_bounded_lower_confidence_value(self):
        items = [trajectory("1" * 64, "a", "small task"), trajectory("2" * 64, "a", "small code task"), trajectory("3" * 64, "a", "small analysis task")]
        labels = {item.key: (0.8, 1.0) for item in items}
        model = CalibratedRewardModel(["a"], minimum_support=2).fit(items, labels, [])
        estimate = model.predict(items[0], "a")
        self.assertLessEqual(estimate.lower, estimate.mean)
        self.assertLessEqual(estimate.mean, 1.0)
        self.assertGreaterEqual(estimate.lower, 0.0)

