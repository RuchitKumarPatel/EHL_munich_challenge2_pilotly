from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UncertaintySignal:
    """The two quantities that decide how much to explore in a given context.

    interval_width : upper - lower of the calibrated conformal interval on the
                     leading model. How unsure the estimate is.
    decision_margin: gap between the best and runner-up model's calibrated lower
                     bound. How close the decision was.
    """
    interval_width: float
    decision_margin: float


@dataclass
class UncertaintyGate:
    """method5's one new mechanism: a CONTEXT-DEPENDENT exploration rate.

    method4 explores at a flat rate everywhere. That is wasteful in both
    directions: it spends exploration budget in contexts where the conformal bound
    is already tight and the winner is obvious (pure quality tax, zero information
    gained), while under-exploring the contexts where the estimate is genuinely
    uncertain and a different choice might be better. Measured on dataset2, that
    flat tax is what kept method4 below `strongest` on quality.

    The fix is to make the exploration rate a function of uncertainty we ALREADY
    compute, rather than a hand-set constant:

        raw   = normalized(interval_width) / (softening + decision_margin)
        eps(x) = floor + (ceiling - floor) * squash(raw)

    High width and a near-tie -> explore. Tight certified bound and a clear winner
    -> act almost deterministically. This is the practical form of
    information-directed sampling (Russo & Van Roy): spend exploration where the
    ratio of information gained to regret incurred is favorable, instead of
    uniformly. Uncertainty-driven adaptive exploration (ADEU/VDBE) reports the same
    result empirically — concentrating exploration in high-uncertainty states
    reaches good policies far faster than uniform schemes.

    `floor` is NOT optional and must stay strictly positive. It is what preserves
    positivity: every action keeps non-zero probability in every context, so the
    logs this policy produces remain valid for evaluating its own successor. A
    zero floor would recover method3's failure mode, where 72% of (context, model)
    cells became permanently unobservable. The gate reallocates exploration; it
    never abolishes it.

    The same `UncertaintySignal` drives label triage (see labeling/priority.py), so
    one estimated quantity serves both purposes and there is nothing extra to fit.
    """
    floor: float = 0.02          # never explore less than this — positivity guarantee
    ceiling: float = 0.30        # never explore more than this — bounded quality tax
    softening: float = 0.05      # keeps a zero decision margin from exploding the ratio
    # Scale that the raw ratio is compared against. SELF-CALIBRATED by `fit` to the
    # median raw ratio observed on held-out data, so the gate always operates in its
    # discriminating range instead of saturating.
    #
    # This replaced a fixed absolute constant, which was a real defect rather than a
    # tuning nit: measured on dataset2 the raw ratios ranged 12-60 against a
    # hard-coded scale of 0.10, so the squash sat at 0.92-0.98 for every context and
    # epsilon varied only between 0.094 and 0.099. The gate was effectively FLAT —
    # exactly the behavior it exists to replace — and the correlation between
    # exploration and uncertainty came out NEGATIVE. Normalizing by the observed
    # median puts the squash at 0.5 at the median context by construction, and
    # removes a dataset-specific magic number at the same time.
    reference_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.floor <= self.ceiling < 1.0:
            raise ValueError("require 0 < floor <= ceiling < 1")
        if self.softening <= 0:
            raise ValueError("softening must be positive")
        if self.reference_ratio <= 0:
            raise ValueError("reference_ratio must be positive")

    def raw_ratio(self, signal: UncertaintySignal) -> float:
        """Unsquashed information-to-regret proxy. Higher = more worth exploring.
        Scale-free: the reference is applied in `epsilon`, so this stays a pure
        comparison of interval width against decision margin and can be shared
        directly with label triage."""
        width = max(0.0, signal.interval_width)
        margin = max(0.0, signal.decision_margin)
        return width / (self.softening + margin)

    def fit(self, signals: list[UncertaintySignal]) -> "UncertaintyGate":
        """Calibrate `reference_ratio` to the MEDIAN raw ratio of `signals`.

        Must be fit on held-out (calibration) data, never on test. The median is
        used rather than the mean because raw ratios are heavy-tailed — a single
        near-zero decision margin produces an enormous ratio that would drag a mean
        far above the bulk of the distribution and re-introduce saturation.
        """
        ratios = sorted(self.raw_ratio(s) for s in signals)
        positive = [r for r in ratios if r > 0]
        if positive:
            self.reference_ratio = positive[len(positive) // 2]
        return self

    def epsilon(self, signal: UncertaintySignal) -> float:
        """Exploration rate for this context, in [floor, ceiling]. Equals the
        midpoint of that range at a context whose raw ratio matches the reference."""
        raw = self.raw_ratio(signal)
        squashed = raw / (self.reference_ratio + raw)
        return self.floor + (self.ceiling - self.floor) * squashed

    def to_dict(self) -> dict[str, float]:
        return {"floor": self.floor, "ceiling": self.ceiling,
                "softening": self.softening, "reference_ratio": self.reference_ratio}

    @classmethod
    def from_dict(cls, payload: dict) -> "UncertaintyGate":
        return cls(
            floor=float(payload["floor"]), ceiling=float(payload["ceiling"]),
            softening=float(payload["softening"]),
            reference_ratio=float(payload.get("reference_ratio", 1.0)),
        )


def signal_from_bounds(
    lower_bounds: dict[str, float],
    upper_bounds: dict[str, float],
    eligible: set[str] | None = None,
) -> UncertaintySignal:
    """Build the signal from per-model calibrated bounds.

    `interval_width` is taken from the LEADING eligible model (the one the policy
    would otherwise commit to) — that is the estimate the decision actually rests
    on. `decision_margin` is the gap to the runner-up among eligible models; with
    fewer than two eligible models there is no decision to be uncertain about, so
    the margin is treated as maximal (1.0) and exploration falls toward the floor.
    """
    pool = [m for m in lower_bounds if eligible is None or m in eligible] or list(lower_bounds)
    if not pool:
        return UncertaintySignal(interval_width=0.0, decision_margin=1.0)
    ordered = sorted(pool, key=lambda m: (-lower_bounds[m], m))
    leader = ordered[0]
    width = max(0.0, upper_bounds.get(leader, lower_bounds[leader]) - lower_bounds[leader])
    margin = (lower_bounds[leader] - lower_bounds[ordered[1]]) if len(ordered) > 1 else 1.0
    return UncertaintySignal(interval_width=width, decision_margin=max(0.0, margin))
