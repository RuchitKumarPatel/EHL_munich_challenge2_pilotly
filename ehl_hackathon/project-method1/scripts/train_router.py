from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method1.data.loader import load_trajectories
from method1.data.split import split_trajectories
from method1.pricing import CostModel, ensure_models, load_pricing
from method1.quality.outcome_model import composite_outcome
from method1.routing.difficulty_router import DifficultyRouter


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "router" / "model.json"
    trajectories = load_trajectories(source)
    models = sorted({trajectory.logged_model for trajectory in trajectories if trajectory.logged_model != "mixed"})
    pricing = ensure_models(load_pricing(), models)
    train = split_trajectories(trajectories)["train"] or trajectories
    router = DifficultyRouter(models, CostModel(pricing))
    outcomes = {trajectory.key: composite_outcome(trajectory) for trajectory in train}
    router.fit(train, outcomes)
    router.save(target)
    metadata = target.with_suffix(".metadata.json")
    metadata.write_text(json.dumps({"source": str(source), "models": models, "train_trajectories": len(train), "quality_type": "proxy"}, indent=2), encoding="utf-8")
    print(f"trained={len(train)} models={len(models)} output={target}")


if __name__ == "__main__":
    main()

