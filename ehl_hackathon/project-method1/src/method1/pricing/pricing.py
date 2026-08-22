from __future__ import annotations

import json
from pathlib import Path


DEFAULT_PRICING = {
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


def ensure_models(pricing: dict[str, dict[str, float]], models: list[str]) -> dict[str, dict[str, float]]:
    output = {key: dict(value) for key, value in pricing.items()}
    for model in models:
        if model in output:
            continue
        if "opus" in model or "terra" in model:
            rate = 0.000015
        elif "sonnet" in model or "luna" in model:
            rate = 0.000003
        else:
            rate = 0.00000030
        output[model] = {"input": rate, "cached_input": rate * 0.1}
    return output
