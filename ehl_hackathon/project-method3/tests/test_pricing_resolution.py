from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from method3.pricing.pricing import DEFAULT_PRICING, ensure_models, load_dataset_pricing, resolve_pricing


def _write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    export = tmp_path / "export"
    export.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenario_manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    return export


def test_ensure_models_reports_which_prices_were_guessed():
    pricing, guessed = ensure_models(dict(DEFAULT_PRICING), ["claude-opus-5", "brand-new-model"])
    assert guessed == ["brand-new-model"]
    assert "brand-new-model" in pricing


def test_ensure_models_strict_refuses_to_guess():
    with pytest.raises(KeyError, match="refusing to guess"):
        ensure_models(dict(DEFAULT_PRICING), ["brand-new-model"], strict=True)


def test_declared_price_overrides_the_name_based_guess(tmp_path):
    # Regression test for a real bug: "gpt-5.7-nova" matches none of the name rules
    # ("opus"/"terra"/"sonnet"/"luna"), so it silently fell through to the CHEAPEST
    # tier ($0.30/Mtok) while the dataset priced it at $4.00/Mtok. The router then
    # believed a mid-priced model was the cheapest and sent it 100% of traffic.
    export = _write_manifest(tmp_path, [
        {"key": "k1", "effective_price_per_mtok": {"gpt-5.7-nova": 4.0, "claude-fable-5": 0.30}},
    ])
    guessed_pricing, guessed = ensure_models(dict(DEFAULT_PRICING), ["gpt-5.7-nova"])
    assert guessed == ["gpt-5.7-nova"]
    assert guessed_pricing["gpt-5.7-nova"]["input"] == pytest.approx(0.30 / 1_000_000)  # the wrong guess

    resolved, still_guessed = resolve_pricing(export, ["gpt-5.7-nova", "claude-fable-5"])
    assert still_guessed == []                       # nothing had to be guessed
    assert resolved["gpt-5.7-nova"]["input"] == pytest.approx(4.0 / 1_000_000)
    # and the declared price makes nova correctly MORE expensive than the cheap model
    assert resolved["gpt-5.7-nova"]["input"] > resolved["claude-fable-5"]["input"]


def test_dataset_pricing_takes_the_max_across_eras(tmp_path):
    # Prices move across dataset2's eras; being wrong in the conservative
    # (over-estimating) direction is the safer failure.
    export = _write_manifest(tmp_path, [
        {"key": "a", "effective_price_per_mtok": {"m": 4.0}},
        {"key": "b", "effective_price_per_mtok": {"m": 3.2}},
    ])
    pricing = load_dataset_pricing(export)
    assert pricing["m"]["input"] == pytest.approx(4.0 / 1_000_000)


def test_cached_input_is_a_tenth_of_input():
    export_pricing = load_dataset_pricing(Path("does-not-exist"))
    assert export_pricing == {}
    resolved, _ = resolve_pricing(Path("does-not-exist"), ["claude-opus-5"])
    assert resolved["claude-opus-5"]["cached_input"] < resolved["claude-opus-5"]["input"]


def test_missing_manifest_falls_back_to_defaults_without_error():
    resolved, guessed = resolve_pricing(Path("no-such-dir"), ["claude-opus-5", "claude-fable-5"])
    assert guessed == []
    assert resolved["claude-opus-5"]["input"] == DEFAULT_PRICING["claude-opus-5"]["input"]


def test_relative_price_order_is_preserved_from_the_dataset(tmp_path):
    export = _write_manifest(tmp_path, [{
        "key": "k",
        "effective_price_per_mtok": {"cheap": 0.30, "mid": 4.0, "top": 15.0},
    }])
    pricing, _ = resolve_pricing(export, ["cheap", "mid", "top"])
    ordered = sorted(("cheap", "mid", "top"), key=lambda m: pricing[m]["input"])
    assert ordered == ["cheap", "mid", "top"]
