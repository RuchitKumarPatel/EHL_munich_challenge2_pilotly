from __future__ import annotations

from dataclasses import dataclass

from method5.policy.uncertainty import UncertaintyGate, UncertaintySignal


@dataclass
class LabelCandidate:
    trajectory_key: str
    signal: UncertaintySignal
    already_labeled: bool = False


@dataclass
class LabelTriage:
    """Spend a scarce labeling budget where it changes decisions.

    Deliberately reuses the SAME `UncertaintyGate.raw_ratio` that sets the
    exploration rate. One estimated quantity — calibrated interval width over
    decision margin — answers both "how much should I explore here?" and "is this
    worth paying to label?", because they are the same question: where is the
    current estimate both unsure AND consequential.

    This matters because labels are the measured binding constraint, not model
    capacity: an ablation showed that replacing ground-truth labels with the
    structural proxy collapses routing quality from 0.824 to 0.779 — all the way
    down to always-cheapest — while roughly half of dataset2 carries no ground-truth
    label at all. Spending a fixed budget uniformly wastes most of it on
    trajectories whose routing decision was never in doubt.
    """
    gate: UncertaintyGate

    def priority(self, candidate: LabelCandidate) -> float:
        if candidate.already_labeled:
            return 0.0        # re-labeling buys nothing
        return self.gate.raw_ratio(candidate.signal)

    def rank(self, candidates: list[LabelCandidate]) -> list[tuple[str, float]]:
        """Highest priority first; ties broken by key so the queue is reproducible."""
        return sorted(
            ((c.trajectory_key, self.priority(c)) for c in candidates),
            key=lambda item: (-item[1], item[0]),
        )

    def select(self, candidates: list[LabelCandidate], budget: int) -> list[str]:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        return [key for key, score in self.rank(candidates)[:budget] if score > 0.0]

    def budget_efficiency(self, candidates: list[LabelCandidate], budget: int) -> dict[str, float]:
        """Decision-relevant uncertainty captured by a prioritized budget versus the
        same budget spent uniformly at random. Ratio 3.0 means three times the
        useful signal per label. The uniform expectation is computed exactly (mean
        priority x budget) rather than simulated, keeping this deterministic."""
        ranked = self.rank(candidates)
        scores = [score for _, score in ranked]
        if not scores or budget <= 0:
            return {"prioritized": 0.0, "uniform_expected": 0.0, "ratio": 1.0, "n": len(scores)}
        take = min(budget, len(scores))
        prioritized = sum(scores[:take])
        uniform_expected = (sum(scores) / len(scores)) * take
        return {
            "prioritized": prioritized,
            "uniform_expected": uniform_expected,
            "ratio": (prioritized / uniform_expected) if uniform_expected > 0 else 1.0,
            "n": len(scores),
        }
