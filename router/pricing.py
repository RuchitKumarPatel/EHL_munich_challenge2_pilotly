#!/usr/bin/env python3
"""Named, ASSUMED price sheets and sign-stability tests for the router.

WHAT THIS COMPUTES
    Per-1M-token input/output rates for an anonymized model id under a *named*
    price sheet (longest-prefix match), and `sign_stable(src, dst, sheets)` —
    True only when `dst` is cheaper than `src` under EVERY sheet supplied.

WHAT IT WRITES
    Nothing. This module is pure data + pure functions. Dollar figures are
    produced by `router.costs`; this module supplies the rates and the naming
    discipline that makes an unlabelled dollar figure impossible to print.

THE PRICING CAVEAT (read before quoting any number)
    The model ids in the export are ANONYMIZED per AGENTS.md ("real names
    hidden, tier order not published"). NO PUBLIC PRICE SHEET APPLIES to
    `claude-opus-5`, `gpt-5.6-terra`, `claude-fable-5`, and the rest. Every
    sheet in this module is an ASSUMPTION, not a quote: SHEET_ASSUMED_DEFAULT
    mirrors the organizers' own placeholder in scripts/cost_model.py, and the
    two ALT_* sheets exist precisely so the team can sweep the assumption
    instead of believing it.

    Consequence, stated plainly: THE SIGN OF ANY DOLLAR SAVING IS CURRENTLY
    UNDETERMINED. Under SHEET_ASSUMED_DEFAULT `claude-fable` is the cheapest
    arm; under SHEET_ALT_COMPRESSED the `gpt-5.6` family undercuts it and the
    same reroute changes sign. We therefore never report a point saving from a
    single sheet — we report which reroutes survive `sign_stable()` across all
    sheets, and we flag the rest as sheet-dependent. WHICH SHEET IS REAL IS AN
    OPEN QUESTION FOR THE ORGANIZERS; if they post one, register it with
    `register_sheet()` and re-run — nothing else needs to change.

    Separately: all token counts feeding these rates are ESTIMATES (chars/4;
    the export has no `usage` field), and output tokens are unbillable here
    because the export has no `output` field. See `router.costs`.

ACCEPTANCE (checked by `python -m router.pricing`)
    - SHEET_ASSUMED_DEFAULT mirrors scripts/cost_model.py DEFAULT_PRICING on
      the input rate: claude-opus 15.0, claude-sonnet 3.0, claude-fable 0.8,
      gpt-5.6 2.0, _default 2.0.
    - Longest-prefix match: every one of the 9 arms observed in the export
      (opus-5 331, sonnet-5 281, terra 113, sol 112, fable-5 71, opus-4-8 69,
      luna 20, opus-4-6 2, sonnet-4-6 1) resolves without falling through to
      "_default" under SHEET_ASSUMED_DEFAULT.
    - Sign instability is real and is demonstrated, not asserted:
      sign_stable("claude-opus-5", "claude-fable-5") is True over all three
      sheets, while sign_stable("gpt-5.6-terra", "claude-fable-5") is False.
    - An unregistered sheet dict cannot be named, so `format_usd` refuses it.
"""
from __future__ import annotations

# Rates are USD per 1,000,000 ESTIMATED tokens. Keys are model-id PREFIXES;
# the longest matching prefix wins, "_default" is the fallback.

#: ASSUMED. Mirrors scripts/cost_model.py DEFAULT_PRICING (input/output legs).
#: The organizers' own placeholder — a starting point, not a quote.
SHEET_ASSUMED_DEFAULT: dict[str, dict[str, float]] = {
    "claude-opus":   {"in": 15.00, "out": 75.00},
    "claude-sonnet": {"in":  3.00, "out": 15.00},
    "claude-fable":  {"in":  0.80, "out":  4.00},
    "gpt-5.6":       {"in":  2.00, "out":  8.00},
    "_default":      {"in":  2.00, "out":  8.00},
}

#: ALTERNATIVE / ASSUMED. "The tier spread is much narrower than the default
#: sheet assumes." Frontier and cheap tiers converge; notably `claude-fable`
#: is NO LONGER the cheapest arm — the `gpt-5.6` family undercuts it. Any
#: reroute onto fable that looked like a saving under the default sheet is
#: sheet-dependent, which is exactly what `sign_stable()` is for.
SHEET_ALT_COMPRESSED: dict[str, dict[str, float]] = {
    "claude-opus":   {"in":  6.00, "out": 30.00},
    "claude-sonnet": {"in":  3.00, "out": 15.00},
    "claude-fable":  {"in":  2.50, "out": 12.50},
    "gpt-5.6":       {"in":  2.00, "out":  8.00},
    "_default":      {"in":  2.50, "out": 12.50},
}

#: ALTERNATIVE / ASSUMED. "Generation matters as much as tier": older
#: generations are discounted and the gpt siblings are not one flat rate.
#: Exercises longest-prefix matching (claude-opus-4 beats claude-opus;
#: gpt-5.6-luna beats gpt-5.6).
SHEET_ALT_GENERATIONAL: dict[str, dict[str, float]] = {
    "claude-opus":    {"in": 15.00, "out": 75.00},
    "claude-opus-4":  {"in":  9.00, "out": 45.00},
    "claude-sonnet":  {"in":  3.00, "out": 15.00},
    "claude-sonnet-4": {"in": 1.80, "out":  9.00},
    "claude-fable":   {"in":  0.80, "out":  4.00},
    "gpt-5.6":        {"in":  2.50, "out": 10.00},
    "gpt-5.6-luna":   {"in":  1.20, "out":  4.80},
    "_default":       {"in":  2.50, "out": 10.00},
}

#: The registry. A sheet must live here to be nameable, and it must be
#: nameable to appear next to a dollar figure. Register organizer-supplied
#: sheets with `register_sheet()`.
SHEETS: dict[str, dict[str, dict[str, float]]] = {
    "assumed_default": SHEET_ASSUMED_DEFAULT,
    "alt_compressed": SHEET_ALT_COMPRESSED,
    "alt_generational": SHEET_ALT_GENERATIONAL,
}

#: Default sweep set for `sign_stable`: a reroute must win under all of these.
ALL_SHEETS: tuple[str, ...] = ("assumed_default", "alt_compressed", "alt_generational")


class UnnamedSheetError(ValueError):
    """Raised when a dollar figure would be printed without a nameable sheet."""


def sheet_names() -> list[str]:
    """Names of every registered price sheet, in registration order."""
    return list(SHEETS)


def register_sheet(name: str, rates: dict[str, dict[str, float]]) -> str:
    """Register a new named sheet (e.g. one the organizers post); returns the name."""
    if not name or not isinstance(name, str):
        raise UnnamedSheetError("a price sheet needs a non-empty string name")
    if "_default" not in rates:
        raise ValueError(f"sheet {name!r} has no '_default' fallback rate")
    for prefix, r in rates.items():
        if not {"in", "out"} <= set(r):
            raise ValueError(f"sheet {name!r} entry {prefix!r} needs both 'in' and 'out'")
    SHEETS[name] = rates
    return name


def resolve_sheet(sheet: str | dict) -> tuple[str, dict[str, dict[str, float]]]:
    """Return (name, rates) for a sheet given by name or by a registered dict."""
    if isinstance(sheet, str):
        if sheet not in SHEETS:
            raise UnnamedSheetError(
                f"unknown price sheet {sheet!r}; registered: {sheet_names()}"
            )
        return sheet, SHEETS[sheet]
    if isinstance(sheet, dict):
        for name, rates in SHEETS.items():
            if rates is sheet or rates == sheet:
                return name, rates
        raise UnnamedSheetError(
            "this price-sheet dict is not registered, so no dollar figure derived "
            "from it can be labelled with its sheet. Call register_sheet(name, rates) first."
        )
    raise UnnamedSheetError(f"sheet must be a name or a registered dict, got {type(sheet).__name__}")


def sheet_name(sheet: str | dict) -> str:
    """Name of a sheet given by name or registered dict; raises if unnameable."""
    return resolve_sheet(sheet)[0]


def price_of(model: str, sheet: str | dict) -> dict[str, float]:
    """Rates {'in','out'} per 1M est. tokens for `model` under `sheet` (longest prefix wins)."""
    _, rates = resolve_sheet(sheet)
    if model in rates:
        return rates[model]
    best, best_len = None, -1
    for prefix, r in rates.items():
        if prefix != "_default" and model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = r, len(prefix)
    return best if best is not None else rates["_default"]


def rate_in(model: str, sheet: str | dict) -> float:
    """Input rate (USD per 1M est. tokens) for `model` under `sheet`."""
    return float(price_of(model, sheet)["in"])


def is_default_rate(model: str, sheet: str | dict) -> bool:
    """True if `model` fell through to the sheet's '_default' rate (no family matched)."""
    _, rates = resolve_sheet(sheet)
    if model in rates:
        return model == "_default"
    return not any(p != "_default" and model.startswith(p) for p in rates)


def sign_stable(src: str, dst: str, sheets=ALL_SHEETS, key: str = "in") -> bool:
    """True only if `dst` is strictly cheaper than `src` under EVERY sheet supplied."""
    sheets = list(sheets)
    if not sheets:
        return False  # no evidence is not evidence of a saving
    return all(price_of(dst, s)[key] < price_of(src, s)[key] for s in sheets)


def sign_stable_targets(src: str, candidates, sheets=ALL_SHEETS, key: str = "in") -> list[str]:
    """The candidates that are cheaper than `src` under every sheet (order preserved)."""
    return [c for c in candidates if sign_stable(src, c, sheets, key)]


def format_usd(usd: float, sheet: str | dict, *, note: str = "est. tokens",
               compact: bool = False) -> str:
    """Format a dollar figure with its sheet name attached; raises if the sheet is unnameable."""
    name = sheet_name(sheet)
    if compact:
        return f"${usd:,.2f} [{name}, ASSUMED]"
    return f"${usd:,.2f} [sheet={name}, ASSUMED; {note}]"


# ---------------------------------------------------------------- acceptance

#: The 9 arms observed in the export, with their run counts (measured).
OBSERVED_ARMS = {
    "claude-opus-5": 331, "claude-sonnet-5": 281, "gpt-5.6-terra": 113,
    "gpt-5.6-sol": 112, "claude-fable-5": 71, "claude-opus-4-8": 69,
    "gpt-5.6-luna": 20, "claude-opus-4-6": 2, "claude-sonnet-4-6": 1,
}


def main() -> int:
    """Print the acceptance checks for the price sheets."""
    ok = True

    def check(label, expected, actual):
        nonlocal ok
        good = expected == actual
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}\n         expected={expected!r}\n         actual  ={actual!r}")

    print("router.pricing — ALL SHEETS ARE ASSUMPTIONS (ids anonymized per AGENTS.md;")
    print("no public price sheet applies; the SIGN of any saving is undetermined).\n")

    print(f"registered sheets: {sheet_names()}\n")

    print("input rates (USD / 1M est. tokens) by arm:")
    hdr = f"  {'arm':<18}{'runs':>6}" + "".join(f"{n:>20}" for n in ALL_SHEETS)
    print(hdr)
    for arm, n in OBSERVED_ARMS.items():
        row = f"  {arm:<18}{n:>6}" + "".join(f"{rate_in(arm, s):>20.2f}" for s in ALL_SHEETS)
        print(row)
    print()

    print("acceptance:")
    check("SHEET_ASSUMED_DEFAULT mirrors scripts/cost_model.py DEFAULT_PRICING (input leg)",
          {"claude-opus": 15.0, "claude-sonnet": 3.0, "claude-fable": 0.8,
           "gpt-5.6": 2.0, "_default": 2.0},
          {k: v["in"] for k, v in SHEET_ASSUMED_DEFAULT.items()})
    check("all 9 observed arms resolve without hitting '_default' (assumed_default)",
          0, sum(is_default_rate(a, "assumed_default") for a in OBSERVED_ARMS))
    check("longest-prefix match: claude-opus-4-8 under alt_generational",
          9.0, rate_in("claude-opus-4-8", "alt_generational"))
    check("longest-prefix match: gpt-5.6-luna under alt_generational",
          1.2, rate_in("gpt-5.6-luna", "alt_generational"))
    check("sign_stable(claude-opus-5 -> claude-fable-5) over all 3 sheets",
          True, sign_stable("claude-opus-5", "claude-fable-5"))
    check("sign_stable(gpt-5.6-terra -> claude-fable-5) over all 3 sheets  [sheet-dependent]",
          False, sign_stable("gpt-5.6-terra", "claude-fable-5"))
    check("sign_stable is vacuously False on an empty sheet set",
          False, sign_stable("claude-opus-5", "claude-fable-5", sheets=()))

    try:
        format_usd(1.0, {"_default": {"in": 1.0, "out": 1.0}})
        got = "no error"
    except UnnamedSheetError:
        got = "UnnamedSheetError"
    check("format_usd refuses an unregistered (unnameable) sheet dict", "UnnamedSheetError", got)
    check("format_usd labels the sheet", "$12.34 [sheet=assumed_default, ASSUMED; est. tokens]",
          format_usd(12.34, "assumed_default"))

    print(f"\nRESULT: {'all checks pass' if ok else 'CHECKS FAILED'}")
    print("Open question for the organizers: which sheet, if any, is real?")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
