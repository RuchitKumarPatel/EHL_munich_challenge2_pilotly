"""Single documented reuse point for the layers method5 does not change.

method5 changes exactly one thing about the POLICY — how much to explore, and
where — plus the reporting around it. Everything upstream of that decision is
already implemented and heavily tested in project-method3 (120 tests), including
fixes for real bugs found during this project's development: mixed-model replay,
the model-switch penalty that was computed but never charged, silently guessed
model prices, and the opening-feature rework that improved reward-model accuracy
by 10.4% MAE.

Reimplementing any of that here would duplicate ~1500 lines and reintroduce the
drift that made project-method1 and project-method2 disagree with each other.
Routing every such import through this one module keeps the coupling visible in
a single place instead of scattered across the codebase.

The sys.path insert is deliberate, not a packaging shortcut: the repo keeps each
project as a sibling directory with its own `src/`, none pip-installed. Failing
loudly here beats an obscure ImportError deep inside a script.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_METHOD3_SRC = _REPO / "project-method3" / "src"

if not _METHOD3_SRC.is_dir():
    raise ImportError(
        f"project-method3 not found at {_METHOD3_SRC}. method5 reuses its data, feature, "
        "conformal-calibration and pricing layers; the projects must sit side by side."
    )
if str(_METHOD3_SRC) not in sys.path:
    sys.path.insert(0, str(_METHOD3_SRC))

from method3.data import (  # noqa: E402
    Call, OpeningContext, Trajectory,
    load_ground_truth, load_trajectories, split_trajectories, split_trajectories_temporal,
)
from method3.evaluation import (  # noqa: E402
    PolicyMetrics, bootstrap_mean, load_scenario_tags, segment_report,
)
from method3.features import OPENING_FEATURE_NAMES, extract_opening_features  # noqa: E402
from method3.pricing import CostModel, resolve_pricing  # noqa: E402
from method3.propensity.logging_policy import LoggingPolicyModel  # noqa: E402
from method3.quality import (  # noqa: E402
    Estimate, KNNRewardModel, SplitConformalCalibrator,
    calibrated_outcome, reward_model_accuracy, structural_proxy,
)

__all__ = [
    "Call", "OpeningContext", "Trajectory",
    "load_ground_truth", "load_trajectories", "split_trajectories", "split_trajectories_temporal",
    "PolicyMetrics", "bootstrap_mean", "load_scenario_tags", "segment_report",
    "OPENING_FEATURE_NAMES", "extract_opening_features",
    "CostModel", "resolve_pricing", "LoggingPolicyModel",
    "Estimate", "KNNRewardModel", "SplitConformalCalibrator",
    "calibrated_outcome", "reward_model_accuracy", "structural_proxy",
]
