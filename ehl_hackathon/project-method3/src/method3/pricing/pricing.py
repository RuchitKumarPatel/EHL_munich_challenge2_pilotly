from __future__ import annotations

import json
from pathlib import Path

# $/token (not $/million — matches project-method1's convention; cost_model.py
# divides by 1000, same scaling as project-method1's pricing/cache_model.py).
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-fable-5": {"input": 0.00000030, "cached_input": 0.00000003},
    "claude-sonnet-5": {"input": 0.00000300, "cached_input": 0.00000030},
    "claude-opus-4-8": {"input": 0.00001500, "cached_input": 0.00000150},
    "claude-opus-5": {"input": 0.00001500, "cached_input": 0.00000150},
    "gpt-5.6-sol": {"input": 0.00000030, "cached_input": 0.00000003},
    "gpt-5.6-luna": {"input": 0.00000100, "cached_input": 0.00000010},
    "gpt-5.6-terra": {"input": 0.00000200, "cached_input": 0.00000020},
}


def load_pricing(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    if path is None:
        return {key: dict(value) for key, value in DEFAULT_PRICING.items()}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return {key: {"input": float(value["input"]), "cached_input": float(value.get("cached_input", value["input"]))} for key, value in loaded.items()}


def load_dataset_pricing(source: str | Path) -> dict[str, dict[str, float]]:
    """Read per-model prices declared by the dataset itself, if it declares any.

    dataset2's scenario_manifest.json records `effective_price_per_mtok` (prices
    genuinely move across its eras). Reading them is not a convenience — guessing
    instead is a correctness bug: `gpt-5.7-nova` matches none of the name rules in
    `ensure_models`, so it silently fell through to the CHEAPEST tier ($0.30/Mtok)
    when the dataset prices it at $4.00/Mtok. The router then believed a mid-priced
    model was the cheapest available and routed 100% of traffic to it, and every
    reported cost for that model was wrong by ~13x. Declared prices remove the guess.

    Where a model's price varies across eras, the MAXIMUM is taken: a router that
    is wrong about price should be wrong in the conservative direction (over- not
    under-estimating spend). Returns {} when the dataset declares nothing.
    """
    path = Path(source)
    for candidate in (path / "scenario_manifest.json", path.parent / "scenario_manifest.json"):
        if candidate.exists():
            rows = json.loads(candidate.read_text(encoding="utf-8"))
            per_mtok: dict[str, float] = {}
            for row in rows:
                for model, price in (row.get("effective_price_per_mtok") or {}).items():
                    per_mtok[model] = max(per_mtok.get(model, 0.0), float(price))
            # $/Mtok -> $/token, matching DEFAULT_PRICING's units.
            return {
                model: {"input": price / 1_000_000.0, "cached_input": price / 1_000_000.0 * 0.1}
                for model, price in per_mtok.items()
            }
    return {}


def ensure_models(
    pricing: dict[str, dict[str, float]],
    models: list[str],
    strict: bool = False,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Guarantee every model that could ever be looked up (all candidate models AND
    every real per-call model in the dataset, not just non-mixed logged_model) has a
    pricing entry — the gap that previously let a bare KeyError escape.

    Returns (pricing, guessed_models). Any model whose price had to be GUESSED from
    its name is reported back so the caller can surface it, because a silently
    wrong price corrupts the entire cost axis and every routing decision that
    compares prices. Set `strict=True` to refuse to guess at all.
    """
    output = {key: dict(value) for key, value in pricing.items()}
    guessed: list[str] = []
    for model in models:
        if model in output:
            continue
        guessed.append(model)
        if strict:
            continue
        if "opus" in model or "terra" in model:
            rate = 0.000015
        elif "sonnet" in model or "luna" in model:
            rate = 0.000003
        else:
            rate = 0.00000030
        output[model] = {"input": rate, "cached_input": rate * 0.1}
    if strict and guessed:
        raise KeyError(f"no declared price for model(s) {sorted(guessed)} — refusing to guess (strict=True)")
    return output, guessed


def resolve_pricing(source: str | Path, models: list[str]) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Preferred entry point: dataset-declared prices first, then the built-in
    table, then (reported) name-based guesses."""
    pricing = dict(DEFAULT_PRICING)
    pricing.update(load_dataset_pricing(source))
    return ensure_models(pricing, models)
