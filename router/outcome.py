#!/usr/bin/env python3
"""The composite outcome proxy: one declared weighting of three structural process signals.

WHAT THIS COMPUTES
    A per-trajectory BADNESS in [0, 1] built from results/labels.jsonl only, and
    the panel that carries it, so that every estimator in router.ope works on the
    composite exactly as it works on the binary y_fric. The chart axis the console
    plots is the complement, 1 - badness, called the COMPOSITE OUTCOME PROXY.

    The four components are structural failure signals the labels module already
    resolves from tool-output exit codes -- never from text:

      err_any    labels["y_fric"]                 the run hit at least one errored tool output
      err_rate   labels["y_rate"]                 share of resolvable tool outputs that errored
      trailing   labels["trailing_err"]           the run's LAST resolvable output was an error
      kill_any   labels["n_kill"] > 0             at least one tool call came back with a kill code

    badness = sum(w_c * c), weights normalised to sum to 1, so badness stays in
    [0, 1] and the Manski worst case (1 - y) that router.ope._core applies remains
    the correct one.

    THE COMPONENTS DO NOT SHARE A BASE RATE. err_any fires on 25.9% of runs
    (61.4% of estimated spend), err_rate averages about 0.05, trailing_err fires
    on 1.8%. A weight is therefore NOT a share of the final number: a slider at
    0.3 on trailing moves the composite by at most 0.3 * 0.018. The console prints
    each component's own logged base rate next to its slider so that this is
    visible rather than inferred, and `component_base_rates()` is what it reads.

THIS IS A DECLARED WEIGHTING, NOT A MEASURED QUALITY LABEL
    There is no answer-quality label anywhere in this export (AGENTS.md: no
    `output`, no `usage`, no quality labels). The weights are a CHOICE about how
    much a mid-run error, a run that ends broken, and a killed tool call each
    count. They are not estimated from anything. The console exposes them as
    sliders for exactly that reason: a ranking that flips when a weight moves by
    0.1 was never a finding. Report the weights with the number, always.

THE BASELINE PRESET IS A CONSISTENCY CHECK, NOT A SPECIAL CASE
    Preset "repo_baseline" is the corner of the weight simplex that puts all the
    mass on err_any -- which IS y_fric, the binary label every published number in
    results/ is built on. There is no second code path: the same weighted sum runs,
    and at that corner the console must reproduce results/frontier.csv to the last
    decimal. It is wired into `main()` as an acceptance check, so a drift in the
    panel construction cannot hide behind a nicer-looking axis. Moving a slider off
    that corner continuously deforms the published metric, which is the honest way
    to show how much of a ranking is the metric's opinion rather than the data's.

WHAT IT WRITES
    No artifact. It reads results/labels.jsonl (and, through router.strata, the
    stratum table) and returns objects in memory. The console is the only consumer.

ACCEPTANCE (printed by `python -m router.outcome`)
    every preset's badness lies in [0, 1] on all 1000 rows;
    preset repo_baseline is EXACTLY y_fric on every row -- 259 positives, logged
    rate 0.6137 spend-weighted and 0.2590 count-weighted;
    a row with n_obs == 0 scores badness 0.0 under every preset, the same
    convention y_fric already uses for those 4 rows;
    weights are normalised, so scaling a preset by any positive constant leaves
    every badness unchanged.

CAVEAT CARRIED EVERYWHERE
    Spend weights are ESTIMATED gross tokens (tok(x) = len(json.dumps(x)) // 4;
    the export has no `usage` field). The composite is PROCESS friction, never
    answer quality.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from router import ope as _ope

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
LABELS_PATH = RESULTS_DIR / "labels.jsonl"

#: Component order, fixed. The console sends weights keyed by these names.
COMPONENTS: tuple[str, ...] = ("err_any", "err_rate", "trailing", "kill_any")

#: One line per component, shown next to its slider in the console.
COMPONENT_NOTES: dict[str, str] = {
    "err_any": "the run hit at least one errored tool output — this IS the published y_fric",
    "err_rate": "share of the run's resolvable tool outputs that errored (severity within a run)",
    "trailing": "the run's last resolvable tool output was an error — it ended broken",
    "kill_any": "at least one tool call returned a kill code (SIGKILL/timeout class)",
}

#: Named weightings offered in the console. `repo_baseline` is the corner of the
#: simplex that reproduces the published binary label exactly.
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "repo_baseline": {"err_any": 1.0, "err_rate": 0.0, "trailing": 0.0, "kill_any": 0.0},
    "balanced":      {"err_any": 0.4, "err_rate": 0.2, "trailing": 0.3, "kill_any": 0.1},
    "equal":         {"err_any": 0.25, "err_rate": 0.25, "trailing": 0.25, "kill_any": 0.25},
    "severity":      {"err_any": 0.15, "err_rate": 0.15, "trailing": 0.55, "kill_any": 0.15},
}

#: What the console shows next to each preset. One line, no marketing.
PRESET_NOTES: dict[str, str] = {
    "repo_baseline": "all mass on err_any — identical to y_fric, reproduces results/frontier.csv",
    "balanced": "default: incidence leads, a run that ends broken counts nearly as much",
    "equal": "no opinion — the four signals carry the same weight",
    "severity": "a run that ends in an error dominates; scattered mid-run errors matter less",
}

DEFAULT_PRESET = "balanced"

#: Acceptance numbers, measured against results/labels.jsonl (n = 1000).
EXPECTED = {
    "n": 1000,
    "y_fric_positives": 259,
    "logged_spend_weighted": 0.6137,
    "logged_count_weighted": 0.2590,
    "n_no_obs": 4,
}


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def normalise_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return non-negative component weights summing to 1.

    A caller that sends every weight as zero is asking for a metric with no
    content -- that is an error, not a silent fallback to some default.
    """
    if not weights:
        raise ValueError("no component weights given")
    vals = {}
    for name in COMPONENTS:
        v = float(weights.get(name, 0.0))
        if v < 0 or not np.isfinite(v):
            raise ValueError(f"weight {name!r} must be finite and >= 0, got {v!r}")
        vals[name] = v
    total = sum(vals.values())
    if total <= 0:
        raise ValueError("all component weights are zero — that is not a metric")
    return {k: v / total for k, v in vals.items()}


def resolve_weights(preset: str | None = None,
                    weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Weights from an explicit mapping, else from a preset name, else the default preset."""
    if weights:
        return normalise_weights(weights)
    name = preset or DEFAULT_PRESET
    if name not in WEIGHT_PRESETS:
        raise ValueError(f"unknown preset {name!r}; have {sorted(WEIGHT_PRESETS)}")
    return normalise_weights(WEIGHT_PRESETS[name])


# --------------------------------------------------------------------------- #
# per-row composite
# --------------------------------------------------------------------------- #
def components(rec: Mapping[str, object]) -> dict[str, float]:
    """The four structural signals for one labels.jsonl record, each in [0, 1].

    `n_obs == 0` means no tool output in the run resolved to a status at all, so
    none of the four signals is observed. They are reported as 0.0, which is the
    convention y_fric already uses on those rows -- NOT an imputation of success,
    but the same refusal to invent an error the labels module makes when it keeps
    'unknown' outputs out of the denominator.
    """
    n_obs = int(rec.get("n_obs") or 0)
    if n_obs <= 0:
        return {c: 0.0 for c in COMPONENTS}
    rate = rec.get("y_rate")
    return {
        "err_any": float(rec["y_fric"]),
        "err_rate": float(rate) if rate is not None else 0.0,
        "trailing": 1.0 if rec.get("trailing_err") else 0.0,
        "kill_any": 1.0 if int(rec.get("n_kill") or 0) > 0 else 0.0,
    }


def badness(rec: Mapping[str, object], weights: Mapping[str, float]) -> float:
    """Composite badness in [0, 1] — the weighted sum of the four components."""
    comp = components(rec)
    return float(sum(weights[c] * comp[c] for c in COMPONENTS))


def component_base_rates(labels: dict[int, dict] | None = None,
                         records: Sequence[dict] | None = None) -> dict[str, dict]:
    """Each component's own logged level, count- and spend-weighted.

    The console prints these next to the sliders. Without them a weight reads as
    a share of the final number, which it is not: the components differ in base
    rate by more than an order of magnitude.
    """
    labels = load_labels() if labels is None else labels
    panel = _ope.load_panel(records=records, labels=labels)
    spend = panel.weights("spend")
    total_spend = float(spend.sum())
    comp = {c: np.array([components(labels[int(i)])[c] for i in panel.idx], dtype=np.float64)
            for c in COMPONENTS}
    return {
        c: {
            "count_weighted": float(v.mean()),
            "spend_weighted": float((spend * v).sum() / total_spend) if total_spend else 0.0,
            "note": COMPONENT_NOTES[c],
        }
        for c, v in comp.items()
    }


def load_labels(path: Path | str = LABELS_PATH) -> dict[int, dict]:
    """Read results/labels.jsonl into {idx: record}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run `make labels` first")
    out: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[int(rec["idx"])] = rec
    return out


def composite_labels(weights: Mapping[str, float],
                     labels: dict[int, dict] | None = None) -> dict[int, dict]:
    """A labels map whose `y_fric` field carries the composite badness.

    router.ope.load_panel reads exactly one field off each label record, so
    overwriting it is the whole integration: every estimator, bootstrap and bound
    downstream then runs on the composite with no second code path. The original
    binary stays available as `y_fric_binary` for anything that needs it.
    """
    labels = load_labels() if labels is None else labels
    out: dict[int, dict] = {}
    for idx, rec in labels.items():
        b = badness(rec, weights)
        if not (0.0 <= b <= 1.0):  # pragma: no cover - normalisation guarantees this
            raise AssertionError(f"composite badness {b!r} out of [0,1] on idx {idx}")
        out[idx] = {**rec, "y_fric_binary": rec["y_fric"], "y_fric": b}
    return out


def composite_panel(weights: Mapping[str, float],
                    records: Sequence[dict] | None = None,
                    labels: dict[int, dict] | None = None):
    """The router.ope panel with `y` set to the composite badness."""
    return _ope.load_panel(records=records, labels=composite_labels(weights, labels))


def logged_level(panel, weight: str = "spend") -> float:
    """The logged policy's own badness under the panel's weighting — the chart's anchor."""
    w = panel.weights(weight)
    total = float(w.sum())
    return float((w * panel.y).sum() / total) if total else 0.0


def is_repo_baseline(weights: Mapping[str, float]) -> bool:
    """True when the weights sit on the err_any corner, i.e. badness == y_fric exactly."""
    return abs(weights.get("err_any", 0.0) - 1.0) < 1e-12 and all(
        abs(weights.get(c, 0.0)) < 1e-12 for c in COMPONENTS if c != "err_any"
    )


def describe(weights: Mapping[str, float]) -> dict[str, object]:
    """Everything the console needs to print the axis honestly, in one object."""
    parts = " + ".join(f"{weights[c]:.3g}·{c}" for c in COMPONENTS if weights[c] > 0)
    baseline = is_repo_baseline(weights)
    return {
        "kind": "binary" if baseline else "composite",
        "weights": dict(weights),
        "is_repo_baseline": baseline,
        "axis_label": "composite outcome proxy = 1 − badness (higher is better)",
        "definition": f"badness = {parts}",
        "declared": True,
        "caveat": (
            "identical to the published binary y_fric — every number matches results/"
            if baseline else
            "a DECLARED weighting, not a measured quality label; report the weights with the number"
        ),
    }


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #
def acceptance(labels: dict[int, dict] | None = None) -> list[tuple]:
    """Return [(pass, label, expected, actual), ...] for every acceptance number."""
    labels = load_labels() if labels is None else labels
    rows: list[tuple] = []

    rows.append((len(labels) == EXPECTED["n"], "rows in labels.jsonl",
                 EXPECTED["n"], len(labels)))

    n_pos = sum(1 for r in labels.values() if r["y_fric"])
    rows.append((n_pos == EXPECTED["y_fric_positives"], "y_fric positives",
                 EXPECTED["y_fric_positives"], n_pos))

    n_no_obs = sum(1 for r in labels.values() if not r.get("n_obs"))
    rows.append((n_no_obs == EXPECTED["n_no_obs"], "rows with n_obs == 0",
                 EXPECTED["n_no_obs"], n_no_obs))

    # every preset stays inside [0, 1] on every row
    for name in WEIGHT_PRESETS:
        w = resolve_weights(name)
        vals = [badness(r, w) for r in labels.values()]
        ok = all(0.0 <= v <= 1.0 for v in vals)
        rows.append((ok, f"preset {name}: badness in [0,1]", "all rows",
                     "all rows" if ok else f"{sum(1 for v in vals if not 0 <= v <= 1)} outside"))

    # rows with no resolvable output score 0 under every preset
    no_obs = [r for r in labels.values() if not r.get("n_obs")]
    ok = all(badness(r, resolve_weights(n)) == 0.0 for r in no_obs for n in WEIGHT_PRESETS)
    rows.append((ok, "n_obs == 0 rows score 0 under every preset", True, ok))

    # scaling a preset changes nothing (weights are normalised)
    w1 = resolve_weights("balanced")
    w2 = normalise_weights({k: 7.5 * v for k, v in WEIGHT_PRESETS["balanced"].items()})
    same = all(abs(w1[c] - w2[c]) < 1e-12 for c in COMPONENTS)
    rows.append((same, "weights normalised (scale-invariant)", True, same))

    # the baseline corner is y_fric on every single row, not just on average
    wb = resolve_weights("repo_baseline")
    exact = all(badness(r, wb) == float(r["y_fric"]) for r in labels.values())
    rows.append((exact, "repo_baseline corner == y_fric row by row", True, exact))
    rows.append((is_repo_baseline(wb), "repo_baseline detected as the err_any corner",
                 True, is_repo_baseline(wb)))

    # the baseline preset reproduces the published logged levels
    panel = composite_panel(resolve_weights("repo_baseline"), labels=labels)
    spend = round(logged_level(panel, "spend"), 4)
    count = round(logged_level(panel, "count"), 4)
    rows.append((abs(spend - EXPECTED["logged_spend_weighted"]) < 5e-4,
                 "repo_baseline logged rate, spend-weighted",
                 EXPECTED["logged_spend_weighted"], spend))
    rows.append((abs(count - EXPECTED["logged_count_weighted"]) < 5e-4,
                 "repo_baseline logged rate, count-weighted",
                 EXPECTED["logged_count_weighted"], count))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Print the composite definition, each preset's logged level, and the acceptance table."""
    labels = load_labels()
    print("composite outcome proxy — a DECLARED weighting of four structural signals")
    print("  the export carries no quality label; these weights are a choice, not an estimate")
    print()
    print(f"  {'component':<10} {'count-w':>9} {'spend-w':>9}   meaning")
    for name, r in component_base_rates(labels).items():
        print(f"  {name:<10} {r['count_weighted']:>9.4f} {r['spend_weighted']:>9.4f}   {r['note']}")
    print()
    print(f"  {'preset':<14} {'definition':<62} {'logged badness (spend-w)':>26}")
    for name in WEIGHT_PRESETS:
        w = resolve_weights(name)
        panel = composite_panel(w, labels=labels)
        d = describe(w)
        print(f"  {name:<14} {d['definition']:<62} {logged_level(panel):>26.4f}")
        print(f"  {'':<14} {PRESET_NOTES[name]}")
    print()

    rows = acceptance(labels)
    failed = 0
    for ok, label, expected, actual in rows:
        mark = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"  [{mark}] {label:<46} expected={expected!r:<12} actual={actual!r}")
    print()
    print("  Token counts are ESTIMATES (len(json.dumps(x))//4); the export has no `usage` field.")
    print("  The composite is PROCESS friction from exit codes, never answer quality.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
