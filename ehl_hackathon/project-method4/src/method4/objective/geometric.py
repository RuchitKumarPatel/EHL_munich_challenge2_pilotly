from __future__ import annotations

import math


def geometric_mean_quality(qualities: list[float], floor: float = 1e-6) -> float:
    """Geometric mean of per-trajectory quality: exp(mean(log q)).

    Bet-hedging theory (seed dormancy, Kelly betting) says that what compounds
    across a varying environment is the GEOMETRIC mean, not the arithmetic one, and
    that a strategy maximizing geometric-mean fitness can rationally accept a lower
    arithmetic mean in exchange for lower variance. dataset2 drifts deliberately
    across eras, and a router has to survive all of them, so the geometric mean is
    the honest headline for "how will this hold up over time".

    It is reported ALONGSIDE the arithmetic mean rather than replacing it: the two
    agree when outcomes are stable and diverge exactly when variance is high, and
    that divergence is itself the signal worth seeing. `floor` guards log(0).
    """
    if not qualities:
        return 0.0
    return math.exp(sum(math.log(max(q, floor)) for q in qualities) / len(qualities))


def geometric_mean_by_segment(segment_qualities: dict[str, list[float]], floor: float = 1e-6) -> dict[str, float]:
    """Per-segment arithmetic mean folded into a cross-segment geometric mean.

    This is the bet-hedging criterion applied across environments (eras, task
    types): a policy that is excellent in one era and terrible in another scores
    far worse here than one that is uniformly decent, which is the correct
    preference for a router expected to keep working as the environment moves.
    """
    if not segment_qualities:
        return {"geometric_across_segments": 0.0}
    per_segment = {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in segment_qualities.items()
    }
    means = [max(v, floor) for v in per_segment.values()]
    across = math.exp(sum(math.log(v) for v in means) / len(means))
    arithmetic = sum(per_segment.values()) / len(per_segment)
    return {
        **{f"segment::{k}": v for k, v in sorted(per_segment.items())},
        "arithmetic_across_segments": arithmetic,
        "geometric_across_segments": across,
        # Always >= 0 by AM-GM; large values mean the policy is unevenly good and
        # therefore fragile to which environment it actually lands in.
        "variance_penalty": arithmetic - across,
    }
