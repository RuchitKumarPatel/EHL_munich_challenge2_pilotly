from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabelCandidate:
    trajectory_key: str
    interval_width: float     # conformal upper - lower for the routed model: how unsure we are
    decision_margin: float    # score gap between the best and runner-up model
    already_labeled: bool = False


@dataclass
class ReplayPriority:
    """Triage a scarce labeling budget the way the hippocampus triages consolidation.

    The brain does not replay experience uniformly during rest. Sharp-wave ripples
    preferentially reactivate episodes that were SURPRISING or REWARDING, and that
    selective replay is what gets consolidated. The engineering analogue is
    prioritized experience replay, and the same logic applies to deciding which
    trajectories are worth paying to label.

    This matters because labels are the binding constraint, not model capacity:
    method3's ablation showed that swapping ground-truth labels for the structural
    proxy collapses the router from 0.824 to 0.779 — all the way down to
    always-cheapest — and roughly half of dataset2 carries no ground-truth label at
    all. Spending a fixed labeling budget uniformly wastes most of it on
    trajectories whose routing decision was never in doubt.

    Priority combines two terms, both required:

      * `interval_width` — the calibrated uncertainty on the routed model. Wide
        interval = the model does not know. This is the "surprise" term.
      * `decision_margin` — how close the top two candidates were. A near-tie means
        a label here could actually FLIP the routing decision; a blowout means it
        cannot, however uncertain the estimate is. This is the decision-relevance
        term, and it is what keeps the criterion from degenerating into "label
        whatever is noisiest", which buys uncertainty reduction in regions that do
        not change any action.

        priority = interval_width / (margin_softening + decision_margin)

    Already-labeled trajectories score 0: re-labeling buys nothing.
    """
    margin_softening: float = 0.02

    def priority(self, candidate: LabelCandidate) -> float:
        if candidate.already_labeled:
            return 0.0
        width = max(0.0, candidate.interval_width)
        margin = max(0.0, candidate.decision_margin)
        return width / (self.margin_softening + margin)

    def rank(self, candidates: list[LabelCandidate]) -> list[tuple[str, float]]:
        """All candidates, highest priority first. Ties broken by key for
        determinism (a non-deterministic labeling queue is unreproducible)."""
        scored = [(c.trajectory_key, self.priority(c)) for c in candidates]
        return sorted(scored, key=lambda item: (-item[1], item[0]))

    def select(self, candidates: list[LabelCandidate], budget: int) -> list[str]:
        """The `budget` trajectories worth labeling first. Zero-priority items
        (already labeled) are never selected even if the budget allows."""
        if budget < 0:
            raise ValueError("budget must be non-negative")
        return [key for key, score in self.rank(candidates)[:budget] if score > 0.0]

    def expected_budget_saving(self, candidates: list[LabelCandidate], budget: int) -> dict[str, float]:
        """How much of the total decision-relevant uncertainty a prioritized budget
        captures versus spending the same budget uniformly at random.

        Reported as a ratio so it is directly interpretable: 3.0 means the same
        number of labels buys three times the decision-relevant uncertainty
        reduction. Uses the mean priority as the random-selection expectation
        rather than simulating draws, which is exact and keeps this deterministic.
        """
        ranked = self.rank(candidates)
        scores = [score for _, score in ranked]
        if not scores or budget <= 0:
            return {"prioritized": 0.0, "uniform_expected": 0.0, "ratio": 1.0}
        take = min(budget, len(scores))
        prioritized = sum(scores[:take])
        uniform_expected = (sum(scores) / len(scores)) * take
        ratio = prioritized / uniform_expected if uniform_expected > 0 else 1.0
        return {"prioritized": prioritized, "uniform_expected": uniform_expected, "ratio": ratio}
