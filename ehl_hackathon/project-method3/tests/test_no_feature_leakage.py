from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.data.schema import Call, OpeningContext, Trajectory
from method3.features.diagnostic_features import DIAGNOSTIC_FEATURE_NAMES, extract_diagnostic_features
from method3.features.opening_features import OPENING_FEATURE_NAMES, extract_opening_features
from method3.routing.base import Router


def _trajectory(n_calls: int, include_error: bool) -> Trajectory:
    history: list[dict] = [
        {"role": "system", "content": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
    ]
    tools = [{"type": "function", "name": "run_shell", "description": "d", "parameters": {}, "strict": False}]
    calls = []
    for i in range(n_calls):
        calls.append(Call("model-a", [dict(item) for item in history], tools, "test", i))
        call_id = f"c{i}"
        history.append({"type": "function_call", "name": "run_shell", "arguments": f"{{\"cmd\": \"{i}\"}}", "call_id": call_id})
        # Note: this iteration's output is appended AFTER this call's snapshot is
        # captured, so it only becomes visible in a LATER call's input — put the
        # error early (i == 0) so it's actually observable by any subsequent call.
        output = "error: boom" if (include_error and i == 0) else "ok"
        history.append({"type": "function_call_output", "call_id": call_id, "output": output})
    return Trajectory("k", calls)


def test_opening_and_diagnostic_feature_names_disjoint():
    # The two schemas must not share a single field name. Sharing a name is exactly
    # how a post-hoc statistic sneaks into a routing-feature vector unnoticed: a
    # future edit that "helpfully" merges the two dicts, or a copy-paste into the
    # wrong module, would silently reintroduce the leak this project exists to fix.
    assert set(OPENING_FEATURE_NAMES).isdisjoint(set(DIAGNOSTIC_FEATURE_NAMES))


def test_opening_context_cannot_see_past_first_call():
    short = _trajectory(n_calls=2, include_error=False)
    long = _trajectory(n_calls=9, include_error=True)
    # Two trajectories that share the same opening (system+first-user) but differ
    # wildly in what happens afterwards (call count, whether an error occurs) must
    # produce an IDENTICAL OpeningContext and therefore identical opening features.
    # If they don't, opening_features is reading something from later calls.
    assert short.opening == long.opening
    assert extract_opening_features(short.opening) == extract_opening_features(long.opening)


def test_opening_features_do_not_change_with_trajectory_length():
    for n in (1, 3, 10):
        trajectory = _trajectory(n_calls=n, include_error=(n == 10))
        features = extract_opening_features(trajectory.opening)
        assert features["prompt_tokens"] == extract_opening_features(_trajectory(1, False).opening)["prompt_tokens"]


def test_diagnostic_features_do_change_with_trajectory_length():
    # Sanity check the negative: diagnostics SHOULD reflect the full trajectory,
    # otherwise this test suite couldn't tell a real leak from a feature extractor
    # that's simply inert.
    short = extract_diagnostic_features(_trajectory(2, include_error=False))
    long = extract_diagnostic_features(_trajectory(9, include_error=True))
    assert short["n_calls"] != long["n_calls"]
    assert short["error_markers"] != long["error_markers"]


def test_opening_context_has_no_calls_attribute():
    # Structural guard: OpeningContext must not expose anything that could be used
    # to reach the rest of the trajectory (e.g. a `calls` or `trajectory` field).
    opening = _trajectory(3, False).opening
    forbidden = {"calls", "trajectory", "n_calls", "per_call_models", "estimated_tokens"}
    assert forbidden.isdisjoint(set(vars(opening)))


def test_router_protocol_signature_takes_opening_context_only():
    import inspect
    signature = inspect.signature(Router.route)
    params = list(signature.parameters.values())
    # First real parameter after `self` must be annotated OpeningContext, not
    # Trajectory — this is the type-level enforcement of the pre-decision boundary.
    annotation = params[1].annotation
    assert annotation is OpeningContext or annotation == "OpeningContext"
