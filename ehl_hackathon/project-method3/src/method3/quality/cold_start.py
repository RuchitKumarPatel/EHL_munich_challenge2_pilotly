from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelDescriptor:
    """Public, pre-inference facts about a model — the kind of thing available for a
    model the historical log has never used."""
    name: str
    price_per_mtok: float
    family: str          # e.g. "claude", "gpt"
    tier: str            # "cheap" | "mid" | "frontier"


def infer_descriptor(name: str, price_per_mtok: float) -> ModelDescriptor:
    family = name.split("-")[0]
    if price_per_mtok <= 0.5:
        tier = "cheap"
    elif price_per_mtok <= 5.0:
        tier = "mid"
    else:
        tier = "frontier"
    return ModelDescriptor(name=name, price_per_mtok=price_per_mtok, family=family, tier=tier)


@dataclass
class ColdStartPrior:
    """A descriptor-based prior for models with little or no direct evidence.

    A KNN reward model fit per-model has nothing to say about a model that never
    appears in the training log — it can only return a neutral guess with zero
    support, and a support-gated router will then never select that model. That is
    safe but permanently blind: a newly released model can never be adopted, no
    matter how good it is. dataset2 makes this concrete by introducing
    `gpt-5.7-nova` only in its final era.

    The routing literature's answer is to condition on public descriptors rather
    than on direct history — UniRoute (arXiv:2502.08773) represents each model by a
    feature vector so routing generalizes to unseen models without retraining;
    Router-R1 (arXiv:2506.09033) and ReCal (arXiv:2606.12479) append textual
    descriptors (pricing, tier, example performance); RouteProfile
    (arXiv:2605.00180) profiles a newly released model from public signals.

    This is the small, honest version of that idea: pool the observed quality of
    models sharing a descriptor bucket (tier, then family, then global) and use it
    as a prior, blended with whatever direct evidence exists in proportion to how
    much of it there is. It is explicitly a PRIOR, not a measurement — `support`
    stays low so a support-gated router still treats the estimate as weak evidence,
    and `is_prior` marks every estimate that leaned on it.
    """
    tier_quality: dict[str, float] = field(default_factory=dict)
    family_quality: dict[str, float] = field(default_factory=dict)
    global_quality: float = 0.5
    descriptors: dict[str, ModelDescriptor] = field(default_factory=dict)

    def fit(self, descriptors: dict[str, ModelDescriptor], observed_quality: dict[str, list[float]]) -> "ColdStartPrior":
        self.descriptors = dict(descriptors)
        tier_scores: dict[str, list[float]] = {}
        family_scores: dict[str, list[float]] = {}
        everything: list[float] = []
        for model, scores in observed_quality.items():
            if not scores:
                continue
            descriptor = descriptors.get(model)
            everything.extend(scores)
            if descriptor is None:
                continue
            tier_scores.setdefault(descriptor.tier, []).extend(scores)
            family_scores.setdefault(descriptor.family, []).extend(scores)
        self.tier_quality = {tier: sum(v) / len(v) for tier, v in tier_scores.items() if v}
        self.family_quality = {family: sum(v) / len(v) for family, v in family_scores.items() if v}
        self.global_quality = (sum(everything) / len(everything)) if everything else 0.5
        return self

    def prior_for(self, model: str) -> float:
        """Most specific descriptor bucket with evidence wins: tier, then family,
        then the global mean."""
        descriptor = self.descriptors.get(model)
        if descriptor is not None:
            if descriptor.tier in self.tier_quality:
                return self.tier_quality[descriptor.tier]
            if descriptor.family in self.family_quality:
                return self.family_quality[descriptor.family]
        return self.global_quality

    def blend(self, model: str, direct_mean: float, direct_support: float, prior_strength: float = 4.0) -> float:
        """Shrink a low-support direct estimate toward the descriptor prior. With
        support >> prior_strength this is ~the direct estimate; with no support at
        all it is exactly the prior."""
        if direct_support <= 0.0:
            return self.prior_for(model)
        weight = direct_support / (direct_support + prior_strength)
        return weight * direct_mean + (1.0 - weight) * self.prior_for(model)

    def to_dict(self) -> dict[str, object]:
        return {
            "tier_quality": self.tier_quality,
            "family_quality": self.family_quality,
            "global_quality": self.global_quality,
            "descriptors": {name: {"name": d.name, "price_per_mtok": d.price_per_mtok, "family": d.family, "tier": d.tier}
                            for name, d in self.descriptors.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ColdStartPrior":
        instance = cls()
        instance.tier_quality = {k: float(v) for k, v in payload.get("tier_quality", {}).items()}
        instance.family_quality = {k: float(v) for k, v in payload.get("family_quality", {}).items()}
        instance.global_quality = float(payload.get("global_quality", 0.5))
        instance.descriptors = {
            name: ModelDescriptor(d["name"], float(d["price_per_mtok"]), d["family"], d["tier"])
            for name, d in payload.get("descriptors", {}).items()
        }
        return instance
