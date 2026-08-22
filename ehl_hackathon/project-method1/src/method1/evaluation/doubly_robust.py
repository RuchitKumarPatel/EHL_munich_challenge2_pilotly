from __future__ import annotations

from collections import Counter

from method1.data.schema import Trajectory
from method1.evaluation.overlap import support_report
from method1.routing.base import Router


def evaluate_doubly_robust(trajectories: list[Trajectory], router: Router, outcomes: dict[str, float], models: list[str], clip: float = 0.05) -> dict[str, object]:
    counts = Counter(trajectory.logged_model for trajectory in trajectories)
    total = max(1, len(trajectories))
    propensity = {model: max(clip, counts.get(model, 0) / total) for model in models}
    estimates = []
    direct = []
    for trajectory in trajectories:
        logged = trajectory.logged_model
        if logged == "mixed" or trajectory.key not in outcomes:
            continue
        decision = router.route(trajectory)
        target = decision.model
        model_prediction = decision.scores.get(target, 0.5)
        logged_prediction = decision.scores.get(logged, 0.5)
        correction = (outcomes[trajectory.key] - logged_prediction) / propensity.get(logged, clip) if target == logged else 0.0
        estimates.append(model_prediction + correction)
        direct.append(model_prediction)
    report = support_report(trajectories, models)
    return {"estimate": sum(estimates) / len(estimates) if estimates else 0.0, "direct_estimate": sum(direct) / len(direct) if direct else 0.0, "n": len(estimates), "propensity_floor": clip, "support": report}

