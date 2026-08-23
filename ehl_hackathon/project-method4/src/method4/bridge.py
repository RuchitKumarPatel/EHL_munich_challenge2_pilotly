"""Explicit, single-point dependency on project-method3.

method4 changes the ROUTING POLICY and the EVALUATION of stochastic policies. It
does not change how trajectories are parsed, how quality is calibrated, how cost is
priced, or how opening features are extracted — all of which are already implemented
and well tested in project-method3 (104 tests), including several fixes for real
bugs found there (mixed-model replay, the switch penalty that was computed but never
charged, silently guessed model prices).

Re-implementing that here would duplicate ~1000 lines and invite exactly the drift
that made project-method1 and project-method2 disagree. Vendoring a copy would do the
same more slowly. So method4 imports method3 as a library, through this one module,
so the coupling is visible in a single place rather than scattered across imports.

The path insert is deliberate rather than a packaging shortcut: the repo keeps each
project as a sibling directory with its own `src/`, and neither is pip-installed.
Failing loudly here (rather than an obscure ImportError deep in a script) is the
point of routing every method3 import through this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

_METHOD3_SRC = Path(__file__).resolve().parents[3] / "project-method3" / "src"

if not _METHOD3_SRC.is_dir():
    raise ImportError(
        f"project-method3 not found at {_METHOD3_SRC}. method4 reuses method3's data, "
        "feature, quality-calibration and pricing layers; both projects must sit side by side."
    )
if str(_METHOD3_SRC) not in sys.path:
    sys.path.insert(0, str(_METHOD3_SRC))

from method3.data import (  # noqa: E402
    Call, OpeningContext, Trajectory,
    load_ground_truth, load_trajectories, split_trajectories, split_trajectories_temporal,
)
from method3.evaluation import (  # noqa: E402
    PolicyMetrics, bootstrap_mean, evaluate_policy, load_scenario_tags, segment_report,
)
from method3.features import OPENING_FEATURE_NAMES, extract_opening_features  # noqa: E402
from method3.pricing import CostModel, resolve_pricing  # noqa: E402
from method3.quality import (  # noqa: E402
    ColdStartPrior, Estimate, KNNRewardModel, SplitConformalCalibrator,
    calibrated_outcome, infer_descriptor, structural_proxy,
)

__all__ = [
    "Call", "OpeningContext", "Trajectory",
    "load_ground_truth", "load_trajectories", "split_trajectories", "split_trajectories_temporal",
    "PolicyMetrics", "bootstrap_mean", "evaluate_policy", "load_scenario_tags", "segment_report",
    "OPENING_FEATURE_NAMES", "extract_opening_features",
    "CostModel", "resolve_pricing",
    "ColdStartPrior", "Estimate", "KNNRewardModel", "SplitConformalCalibrator",
    "calibrated_outcome", "infer_descriptor", "structural_proxy",
]
