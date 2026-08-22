#!/usr/bin/env python3
"""Cache-aware billing on top of results/recon.jsonl.

WHAT THIS COMPUTES
    A trajectory is not one API call: it is N turns, each re-sending the grown
    prefix. `router.recon` already split every line into turn prefixes and
    reported, per line, `cache_read_tok` (every prefix but the last) and
    `cache_write_tok` (the final, largest prefix). This module turns those into
    *effective* (billable-equivalent) tokens and then into dollars:

        effective = rho * (read_mult*cache_read + write_mult*cache_write)
                  + (1 - rho) * (cache_read + cache_write)

    `rho` is the REALISED cache hit rate: rho=1.0 means every cacheable prefix
    actually hit, rho=0.0 means the cache never helped and you pay the fully
    uncached bill. Prefixes shorter than `min_cacheable_prefix` are never
    cacheable and are billed at 1.0x regardless of rho (that carve-out needs
    `prefix_tokens`; without it the record's totals are used as-is).

    Dollars = effective tokens x the INPUT rate of the arm under a NAMED price
    sheet. Output tokens are NOT billed: the export has no `output` field, so
    output volume is unknowable. Every token here is an ESTIMATE (chars/4; the
    export has no `usage` field) — `chars_per_token` rescales that constant.

WHAT IT WRITES
    Nothing. Pure functions over recon records, for router.policy /
    router.claims to call. Reads results/recon.jsonl in `main()` only.

THE SHEET DISCIPLINE
    `line_cost` and `total_cost` take `sheet` as a REQUIRED argument, and every
    dollar string in this module is produced by `router.pricing.format_usd`,
    which refuses any sheet it cannot name. Model ids are anonymized per
    AGENTS.md and no public price sheet applies: every sheet is an ASSUMPTION
    and the SIGN of a saving can flip between sheets. Never quote a dollar
    figure from here without its sheet name.

ACCEPTANCE (checked by `python -m router.costs [recon.jsonl]`)
    Corpus totals from recon: gross 334,729,910 est. tok = cache_read
    308,074,571 (92.0%) + cache_write 26,655,339 (8.0%).
    Effective multiplier vs a fully-uncached bill, at read 0.10x / write 1.25x:
        rho=1.00 -> 0.192   rho=0.90 -> 0.272
        rho=0.83 -> 0.329   rho=0.55 -> 0.555
    And: percentages (cost ratios, shares) are INVARIANT to chars_per_token
    over {3,4,5}; absolute dollars are not (they scale as 4/chars_per_token).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from router.pricing import ALL_SHEETS, format_usd, price_of, sheet_name

#: recon.jsonl token counts were produced with tok(x) = len(json.dumps(x)) // 4.
RECON_CHARS_PER_TOKEN = 4

#: Default billing parameters. All are ASSUMPTIONS about the provider's cache.
DEFAULTS = {
    "cache_read_mult": 0.10,
    "cache_write_mult": 1.25,
    "rho": 1.0,
    "min_cacheable_prefix": 1024,
    "chars_per_token": 4,
}

#: Published corpus totals from results/recon.jsonl (measured, see module doc).
ACCEPTANCE_TOTALS = {
    "gross_tok": 334_729_910,
    "cache_read_tok": 308_074_571,
    "cache_write_tok": 26_655_339,
    "n_turns": 10_845,
}

#: Effective multiplier vs fully-uncached at read 0.10x / write 1.25x.
ACCEPTANCE_RHO_SWEEP = {1.00: 0.192, 0.90: 0.272, 0.83: 0.329, 0.55: 0.555}


def default_recon_path() -> Path:
    """Path this module reads when no recon file is named on the command line."""
    return Path(__file__).resolve().parent.parent / "results" / "recon.jsonl"


def load_recon(path=None) -> list[dict]:
    """Load results/recon.jsonl (or `path`) as a list of records in file order."""
    p = Path(path) if path else default_recon_path()
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _params(**kw) -> dict:
    """Merge caller overrides onto DEFAULTS, rejecting unknown parameter names."""
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown billing parameter(s): {sorted(unknown)}")
    p = dict(DEFAULTS)
    p.update({k: v for k, v in kw.items() if v is not None})
    if not 0.0 <= p["rho"] <= 1.0:
        raise ValueError(f"rho must be in [0,1], got {p['rho']}")
    if p["chars_per_token"] <= 0:
        raise ValueError("chars_per_token must be positive")
    return p


def split_cacheable(rec: dict, min_cacheable_prefix: int = 1024) -> tuple[int, int, int]:
    """Split a recon record into (cacheable_read, cacheable_write, never_cacheable) est. tokens."""
    prefixes = rec.get("prefix_tokens")
    if not prefixes:  # totals-only record: trust the recon split, no carve-out
        return rec["cache_read_tok"], rec["cache_write_tok"], 0
    reads, write = prefixes[:-1], prefixes[-1]
    small = sum(p for p in reads if p < min_cacheable_prefix)
    big_read = sum(p for p in reads if p >= min_cacheable_prefix)
    if write < min_cacheable_prefix:
        return big_read, 0, small + write
    return big_read, write, small


def effective_tokens(rec: dict, **kw) -> float:
    """Billable-equivalent est. tokens for one recon record under the cache model."""
    p = _params(**kw)
    read, write, uncacheable = split_cacheable(rec, p["min_cacheable_prefix"])
    cached_leg = p["cache_read_mult"] * read + p["cache_write_mult"] * write
    uncached_leg = read + write
    eff = p["rho"] * cached_leg + (1.0 - p["rho"]) * uncached_leg + uncacheable
    return eff * (RECON_CHARS_PER_TOKEN / p["chars_per_token"])


def gross_tokens(rec: dict, chars_per_token: int = 4) -> float:
    """Fully-uncached est. tokens for one recon record (the denominator of the multiplier)."""
    return rec["gross_tok"] * (RECON_CHARS_PER_TOKEN / chars_per_token)


def line_cost(recon_record: dict, arm: str, sheet, **kw) -> float:
    """USD for one recon record if `arm` had served it, under the NAMED `sheet`."""
    sheet_name(sheet)  # refuse an unnameable sheet before any dollar exists
    return effective_tokens(recon_record, **kw) * price_of(arm, sheet)["in"] / 1e6


def resolve_route(records: list[dict], route) -> list[str]:
    """Normalise a route (None=logged / str / list / {idx: arm} / callable) to one arm per record."""
    if route is None:
        return [r["model"] for r in records]
    if isinstance(route, str):
        return [route] * len(records)
    if isinstance(route, dict):
        return [route.get(r["idx"], r["model"]) for r in records]
    if callable(route):
        return [route(r) for r in records]
    route = list(route)
    if len(route) != len(records):
        raise ValueError(f"route has {len(route)} entries for {len(records)} records")
    return route


def total_cost(records, route, sheet, **kw) -> float:
    """Total USD over `records` under `route` and the NAMED `sheet`."""
    records = list(records)
    arms = resolve_route(records, route)
    return sum(line_cost(r, a, sheet, **kw) for r, a in zip(records, arms))


def summarize(records, route, sheet, **kw) -> dict:
    """Cost summary for a route: sheet name always included, so no figure travels unlabelled."""
    records = list(records)
    p = _params(**kw)
    eff = sum(effective_tokens(r, **kw) for r in records)
    gross = sum(gross_tokens(r, p["chars_per_token"]) for r in records)
    return {
        "sheet": sheet_name(sheet),
        "sheet_status": "ASSUMED — ids anonymized, no public sheet applies",
        "n_records": len(records),
        "usd": total_cost(records, route, sheet, **kw),
        "effective_tokens_est": eff,
        "gross_tokens_est": gross,
        "effective_multiplier": eff / gross if gross else 0.0,
        "params": p,
        "token_basis": "ESTIMATED (chars/token constant; export has no usage field)",
        "output_tokens_billed": False,
    }


def rho_sweep(records, rhos=(1.00, 0.90, 0.83, 0.55), **kw) -> dict[float, float]:
    """Effective multiplier vs a fully-uncached bill at each rho (dimensionless, sheet-free)."""
    records = list(records)
    gross = sum(r["gross_tok"] for r in records)
    out = {}
    for rho in rhos:
        eff = sum(effective_tokens(r, **{**kw, "rho": rho, "chars_per_token": RECON_CHARS_PER_TOKEN})
                  for r in records)
        out[rho] = eff / gross if gross else 0.0
    return out


# ---------------------------------------------------------------- acceptance

def main(argv=None) -> int:
    """Print the rho sweep and chars_per_token sweep with their acceptance checks."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else default_recon_path()
    if not path.exists():
        print(f"router.costs: no recon file at {path}")
        print("Build it first (router.recon writes results/recon.jsonl), or pass a path:")
        print("    python -m router.costs path/to/recon.jsonl")
        return 2

    records = load_recon(path)
    ok = True

    def check(label, expected, actual):
        nonlocal ok
        good = expected == actual
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: expected={expected!r} actual={actual!r}")

    print(f"router.costs — {len(records)} recon records from {path}")
    print("All token counts are ESTIMATES (export has no usage field). Output tokens are")
    print("NOT billed (export has no output field). Every price sheet is an ASSUMPTION.\n")

    gross = sum(r["gross_tok"] for r in records)
    read = sum(r["cache_read_tok"] for r in records)
    write = sum(r["cache_write_tok"] for r in records)
    turns = sum(r["n_turns"] for r in records)
    print("corpus totals (est. tokens):")
    print(f"  n_turns      {turns:>15,}")
    print(f"  gross        {gross:>15,}")
    print(f"  cache_read   {read:>15,}   ({read / gross:.1%})")
    print(f"  cache_write  {write:>15,}   ({write / gross:.1%})\n")

    print("acceptance — corpus totals:")
    check("n_turns", ACCEPTANCE_TOTALS["n_turns"], turns)
    check("gross_tok", ACCEPTANCE_TOTALS["gross_tok"], gross)
    check("cache_read_tok", ACCEPTANCE_TOTALS["cache_read_tok"], read)
    check("cache_write_tok", ACCEPTANCE_TOTALS["cache_write_tok"], write)

    # --- rho sweep -------------------------------------------------------
    print(f"\nrho sweep — effective multiplier vs a fully-uncached bill")
    print(f"  (cache_read_mult={DEFAULTS['cache_read_mult']}, "
          f"cache_write_mult={DEFAULTS['cache_write_mult']}, "
          f"min_cacheable_prefix={DEFAULTS['min_cacheable_prefix']})")
    sweep = rho_sweep(records)
    print(f"  {'rho':>6}  {'multiplier':>11}  {'eff. tokens':>16}")
    for rho, mult in sweep.items():
        print(f"  {rho:>6.2f}  {mult:>11.3f}  {mult * gross:>16,.0f}")

    print("\nacceptance — rho sweep (multiplier rounded to 3dp):")
    for rho, expected in ACCEPTANCE_RHO_SWEEP.items():
        check(f"rho={rho:.2f}", expected, round(sweep[rho], 3))

    below = sum(1 for r in records for p in r.get("prefix_tokens", [])
                if p < DEFAULTS["min_cacheable_prefix"])
    check("turn prefixes below min_cacheable_prefix (carve-out is inert on this corpus)", 0, below)

    # --- chars_per_token sweep -------------------------------------------
    sheet = "assumed_default"
    print(f"\nchars_per_token sweep — sheet={sheet} (ASSUMED), rho=1.00")
    print("  Percentages are INVARIANT to the constant; absolute dollars are NOT.")
    print(f"  {'c/tok':>6}  {'logged route':>34}  {'all-fable route':>34}"
          f"  {'ratio':>7}  {'read share':>11}")
    ratios, shares, dollars = set(), set(), {}
    for cpt in (3, 4, 5):
        logged = total_cost(records, None, sheet, chars_per_token=cpt)
        fable = total_cost(records, "claude-fable-5", sheet, chars_per_token=cpt)
        eff_read = sum(split_cacheable(r)[0] for r in records)
        share = eff_read / gross
        ratios.add(round(fable / logged, 6))
        shares.add(round(share, 6))
        dollars[cpt] = logged
        print(f"  {cpt:>6}  {format_usd(logged, sheet, compact=True):>34}"
              f"  {format_usd(fable, sheet, compact=True):>34}"
              f"  {fable / logged:>7.4f}  {share:>11.4%}")

    print("\nacceptance — chars_per_token invariance:")
    check("cost RATIO (all-fable / logged) is identical across c/tok in {3,4,5}", 1, len(ratios))
    check("cache-read SHARE is identical across c/tok in {3,4,5}", 1, len(shares))
    check("absolute dollars DIFFER across c/tok in {3,4,5}", 3, len({round(v, 2) for v in dollars.values()}))
    check("dollars scale as 4/c_per_tok:  usd(3)/usd(4)", round(4 / 3, 6), round(dollars[3] / dollars[4], 6))
    check("dollars scale as 4/c_per_tok:  usd(5)/usd(4)", round(4 / 5, 6), round(dollars[5] / dollars[4], 6))

    # --- the same bill under every sheet ---------------------------------
    print("\nthe logged bill under each ASSUMED sheet (rho=1.00, c/tok=4) —")
    print("the spread IS the uncertainty; no single figure here is quotable on its own:")
    for s in ALL_SHEETS:
        print(f"  {format_usd(total_cost(records, None, s), s)}")

    print(f"\nRESULT: {'all checks pass' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
