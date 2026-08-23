from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method2.data import load_trajectories, split
from method2.ground_truth import load_ground_truth
from method2.labels import observed_outcome
from method2.reward import CalibratedRewardModel


def main() -> None:
    source = sys.argv[1]
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "reward_model.json"
    trajectories = load_trajectories(source)
    parts = split(trajectories)
    models = sorted({item.model for item in trajectories if item.model != "mixed"})
    ground_truth = load_ground_truth(source)
    labels = {item.key: observed_outcome(item, ground_truth) for item in trajectories}
    model = CalibratedRewardModel(models).fit(parts["train"] or trajectories, labels, parts["calibration"])
    model.save(target)
    print(f"trained={len(parts['train'])} calibration={len(parts['calibration'])} models={len(models)} ground_truth_labels={len(ground_truth)} output={target}")


if __name__ == "__main__":
    main()

