from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method5.bridge import Call, OpeningContext, Trajectory, extract_opening_features
from method5.policy.base import StochasticPolicy
from method5.policy.router import AdaptiveRouter


def _trajectory(n_calls: int, include_error: bool) -> Trajectory:
    history: list[dict] = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
    ]
    tools = [{"type": "function", "name": "run_shell", "parameters": {}}]
    calls = []
    for i in range(n_calls):
        calls.append(Call("m", [dict(item) for item in history], tools, "test", i))
        call_id = f"c{i}"
        history.append({"type": "function_call", "name": "run_shell", "arguments": f'{{"cmd":"{i}"}}', "call_id": call_id})
        output = "error: boom" if (include_error and i == 0) else "ok"
        history.append({"type": "function_call_output", "call_id": call_id, "output": output})
    return Trajectory("k", calls)


def test_policy_protocol_takes_opening_context_not_trajectory():
    # The pre-decision boundary enforced at the type level: a routing decision
    # fires BEFORE the trajectory runs, so it must not be able to read hindsight
    # (final call count, whether errors occurred). Inherited from project-method3.
    annotation = list(inspect.signature(StochasticPolicy.action_distribution).parameters.values())[1].annotation
    assert annotation is OpeningContext or annotation == "OpeningContext"


def test_router_signature_takes_opening_context():
    annotation = list(inspect.signature(AdaptiveRouter.action_distribution).parameters.values())[1].annotation
    assert annotation is OpeningContext or annotation == "OpeningContext"


def test_openings_identical_when_only_the_future_differs():
    # Two trajectories sharing an opening but diverging wildly afterwards (call
    # count, whether an error occurs) must be indistinguishable to a router.
    short = _trajectory(2, include_error=False)
    long = _trajectory(9, include_error=True)
    assert short.opening == long.opening
    assert extract_opening_features(short.opening) == extract_opening_features(long.opening)


def test_opening_context_exposes_nothing_that_reaches_later_calls():
    opening = _trajectory(4, include_error=True).opening
    forbidden = {"calls", "trajectory", "n_calls", "per_call_models", "estimated_tokens"}
    assert forbidden.isdisjoint(set(vars(opening)))


def test_router_decision_is_invariant_to_what_happens_later():
    # End-to-end version of the guarantee: the same opening must produce the same
    # distribution regardless of how the trajectory turned out.
    from method5.bridge import SplitConformalCalibrator
    from method5.policy.uncertainty import UncertaintyGate
    from method5.policy.value_ladder import ValueLadder

    class FakeEstimate:
        mean, support = 0.8, 50.0

    class FakeRewardModel:
        def predict(self, opening, model):
            return FakeEstimate()

    pricing = {"a": {"input": 1e-6}, "b": {"input": 5e-6}}
    calibrators = {m: SplitConformalCalibrator(0.1, 0.05, 50, True) for m in pricing}
    router = AdaptiveRouter(FakeRewardModel(), calibrators, pricing,
                            ValueLadder(habitat_rate=1.0), UncertaintyGate())
    short = router.action_distribution(_trajectory(2, False).opening, ["a", "b"])
    long = router.action_distribution(_trajectory(9, True).opening, ["a", "b"])
    assert short.probabilities == long.probabilities
