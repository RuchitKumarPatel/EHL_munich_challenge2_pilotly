from __future__ import annotations

from method4.bridge import OpeningContext, extract_opening_features
from method4.policy.base import ActionDistribution
from method4.policy.marginal_value import MarginalValueRule
from method4.policy.probability_matching import make_distribution
from method4.policy.response_threshold import ResponseThresholdTable


def context_bucket(opening: OpeningContext) -> str:
    """A deterministic context label available BEFORE the trajectory runs.

    Response thresholds are per (model, task), but the true task type is a manifest
    label the router may not see at decision time — using it would be exactly the
    leakage method3 was built to prevent. So specialization is learned over an
    observable proxy: the DECLARED TOOL SET plus a prompt-size tercile.

    The tool set is the right signal here and is entirely legitimate — tools are
    part of the request the caller sends, available before any model runs, and
    method3's opening features already read `tool_count`/`distinct_tool_names`. It
    is also far more discriminative than the lexical hash: an initial version
    bucketed on the dominant hashed-bag-of-words bucket and produced only THREE
    buckets across 454 training trajectories, because repeated filler text
    dominated every hash. Tool signature separates task families cleanly.

    Still only a proxy, deliberately: if it correlates with task type,
    specialization emerges; if it does not, thresholds stay flat and the router
    degrades gracefully to pure probability matching.
    """
    tool_names = sorted(
        str(tool.get("name", tool.get("type", "?")))
        for tool in opening.tools if isinstance(tool, dict)
    )
    signature = "+".join(tool_names) if tool_names else "no_tools"
    tokens = extract_opening_features(opening).get("prompt_tokens", 0.0)
    size = "s" if tokens < 400 else ("m" if tokens < 1200 else "l")
    return f"{signature}#{size}"


def difficulty_stimulus(opening: OpeningContext) -> float:
    """Task 'stimulus' in [0, 1] for the response-threshold function, proxied by
    opening size. Bigger, wordier openings are harder on average — the same
    difficulty proxy method1/method2/method3 all rely on, kept explicit here."""
    features = extract_opening_features(opening)
    tokens = features.get("prompt_tokens", 0.0)
    # Saturating map: 0 tokens -> 0, ~2000 tokens -> ~0.9.
    return min(1.0, tokens / 2200.0)


class ForagingRouter:
    """method4's routing policy: MVT-bounded, probability-matched, threshold-modulated.

    Decision pipeline for one opening context:

      1. Calibrated quality lower bound per candidate model (method3's KNN reward
         model + split-conformal calibrator — reused unchanged via the bridge).
      2. MVT ladder (`MarginalValueRule`) sets the ESCALATION CEILING: the costliest
         model whose every rung up the price ladder paid for itself against the
         measured habitat rate. Models above that ceiling are not eligible.
      3. Probability matching over the eligible region on a Kelly (log
         quality-per-dollar) score, so mass concentrates on the best value model
         while remaining a distribution rather than an argmax.
      4. An exploration floor spread over EVERY candidate — including ineligible and
         zero-support ones — which is what preserves positivity in the logs this
         policy generates.
      5. Optionally, response thresholds nudge scores toward models that have
         historically succeeded on similar context buckets.

    Conforms to `StochasticPolicy`: sees an OpeningContext only, never a Trajectory.
    """

    def __init__(
        self,
        reward_model,
        calibrators: dict,
        pricing: dict[str, dict[str, float]],
        marginal_value: MarginalValueRule,
        thresholds: ResponseThresholdTable | None = None,
        temperature: float = 0.05,
        exploration_floor: float = 0.05,
        # NOTE: deliberately LOW. The MVT ladder has already decided how much it is
        # worth paying (the escalation ceiling), so applying a full price penalty
        # again inside the eligible band double-counts cost. Measured: costs span
        # 50x while quality spans ~1.8x, so risk_aversion=1.0 let price dominate the
        # score roughly 6:1 and collapsed 89/159 decisions onto the single cheapest
        # model, costing ~0.05 quality. These defaults were selected on the
        # CALIBRATION split by scripts/tune_method4.py, never on test.
        risk_aversion: float = 0.15,
        minimum_support: float = 3.0,
        threshold_weight: float = 0.35,
    ) -> None:
        self.reward_model = reward_model
        self.calibrators = calibrators
        self.pricing = pricing
        self.marginal_value = marginal_value
        self.thresholds = thresholds
        self.temperature = temperature
        self.exploration_floor = exploration_floor
        self.risk_aversion = risk_aversion
        self.minimum_support = minimum_support
        self.threshold_weight = threshold_weight

    def _price(self, model: str) -> float:
        return self.pricing[model]["input"]

    def _lower_bound(self, opening: OpeningContext, model: str) -> tuple[float, float]:
        estimate = self.reward_model.predict(opening, model)
        calibrator = self.calibrators.get(model)
        global_calibrator = self.calibrators.get("__global__")
        if calibrator is not None and calibrator.coverage_guaranteed:
            lower = calibrator.lower_bound(estimate.mean)
        elif global_calibrator is not None and global_calibrator.coverage_guaranteed:
            lower = global_calibrator.lower_bound(estimate.mean)
        else:
            lower = 0.0
        return lower, estimate.support

    def action_distribution(self, opening: OpeningContext, candidate_models: list[str]) -> ActionDistribution:
        if not candidate_models:
            raise ValueError("no candidate models")

        lower_bounds: dict[str, float] = {}
        supports: dict[str, float] = {}
        prices: dict[str, float] = {}
        for model in candidate_models:
            lower, support = self._lower_bound(opening, model)
            lower_bounds[model] = lower
            supports[model] = support
            prices[model] = self._price(model)

        supported = {m for m in candidate_models if supports[m] >= self.minimum_support}
        ladder_pool = supported or set(candidate_models)

        ceiling_model = self.marginal_value.best_escalation(
            {m: (lower_bounds[m], prices[m]) for m in ladder_pool}
        )
        ceiling_price = prices[ceiling_model]
        eligible = {m for m in ladder_pool if prices[m] <= ceiling_price}

        scoring_lower = dict(lower_bounds)
        if self.thresholds is not None:
            # Response-threshold modulation: a model that has historically engaged
            # successfully with this context bucket gets a boost, one that has not
            # gets damped. Multiplicative on the quality bound so it cannot flip a
            # model's ordering by more than threshold_weight.
            bucket = context_bucket(opening)
            stimulus = difficulty_stimulus(opening)
            for model in candidate_models:
                engagement = self.thresholds.engagement_probability(model, bucket, stimulus)
                factor = 1.0 + self.threshold_weight * (engagement - 0.5)
                scoring_lower[model] = max(1e-6, scoring_lower[model] * factor)

        distribution = make_distribution(
            quality_lower=scoring_lower,
            price=prices,
            eligible=eligible,
            temperature=self.temperature,
            exploration_floor=self.exploration_floor,
            risk_aversion=self.risk_aversion,
            reason=f"foraging:ceiling={ceiling_model}",
        )
        distribution.diagnostics.update({
            "ceiling_price": ceiling_price,
            "n_supported": float(len(supported)),
            "n_eligible": float(len(eligible)),
        })
        return distribution
