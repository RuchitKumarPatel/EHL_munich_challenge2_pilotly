#!/usr/bin/env python3
"""Prose-to-claims gate: `python -m router.verify`.

WHAT THIS COMPUTES
    Nothing new. It enforces a contract that was already written down and never
    checked: docs/CONTRACTS.md and ADR-005 both say that any number quoted
    anywhere must exist in results/claims.json under a stable key. This module
    reads the git-tracked user-facing artifacts, pulls every numeral out of
    their PROSE, and reports the ones no claim stands behind.

WHAT IT DOES NOT DO
    It does not scan results/NUMBERS.md. That file is safe by construction:
    every numeral in it is interpolated through router/report.py's `_Q.__call__`,
    which raises KeyError on a key that is not in claims.json. Scanning it again
    would buy nothing.

    It does not claim that a bound numeral is the RIGHT claim. 28 claims carry
    small-int values 0..12, so a structural "1" or "2" will resolve by
    coincidence. Value matching can only find ORPHANS -- a number with no claim
    anywhere behind it -- and that is exactly what this gate is for.

    It does not read <script> or <style> bodies. Those hold chart geometry and
    row data (183 / 278 / 60,405 literals in console.html / site/index.html /
    site/data.js); every one of them that "matched" a claim matched a
    coincidental 0, 1 or 2. Same for fenced code blocks and inline code spans in
    markdown: a shell command is not a claim.

WHAT "PROSE" MEANS, PRECISELY
    Everything the stripper removes is replaced with blanks, never deleted, so
    a reported line number still points at the real line. Three rules in there
    are not obvious and each was a defect first (ADR-020):

      * A tag ends at its `>`, EXCEPT a `>` inside a quoted attribute value.
        `<[^>]+>` closes on that one and reads the rest of the tag as prose.
      * A code fence with no closing partner is blanked to end of file, the
        same fallback the HTML path already had for an unclosed <script>.
        Because that can also HIDE a numeral written after the bad fence, the
        gate prints a WARNING naming the line where it stopped reading.
      * The `%` that licenses the fraction reading is found by scanning to the
        next non-space, not by slicing a fixed two characters -- but only
        whitespace may sit between, or the tolerance starts binding numerals
        that are not shares.

TWO MORE THINGS THE PARSER HAS TO GET RIGHT
    LOCALE. Locale is DECLARED per artifact in ARTIFACTS rather than inferred
    from the digits, so "11,4" is 11.4 in a `de` file and "10,845" is 10845
    in an `en` file, with no guessing either way. The router console is English.

    PERCENT. Shares live in claims.json as fractions and are quoted as percents:
    "41.0% of est. tokens" is policy.refused.gross_share = 0.4098. A literal
    followed by `%` may therefore also bind to claim/100. Only when followed by
    `%` -- otherwise the tolerance would be wide enough to bind anything.

    A literal binds when it is a correct ROUNDING of the claim at the precision
    it was written with: |literal - claim| <= 0.5 * 10^-decimals.

WHAT IT WRITES
    No artifact. It prints the bindings and any orphans. Exit code is the gate:
    0 clean, 1 on any orphan, 2 BLOCKED when claims.json is missing OR when a
    required artifact is absent -- the same contract as `python -m router.gates`.
    An absent artifact blocks rather than being skipped into a PASS line: the
    deck carries most of the bound numerals and only arrived at the repo root
    at turn 5, so any later rename would otherwise have disabled the prose gate
    for the whole deck while the banner still read clean (ADR-018).

ACCEPTANCE (printed by `python -m router.verify`)
    On the shipped tree: 0 orphans across README.md, router/console.html,
    presentation.html, templates/presentation.html and site/index.html. The
    negative control lives in tests/test_verify.py: perturb one numeral in a
    scratch copy and the run must exit 1 naming file, line and literal.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS_NAME = os.path.join("results", "claims.json")

#: The git-tracked user-facing artifacts, each with its markup kind, the locale
#: its numerals are written in, and whether the gate REQUIRES it to be present.
#: Locale is declared, never sniffed. `required` is declared too, and for the
#: same reason: an artifact that disappears must be a decision someone made in
#: this tuple, not a file the scan quietly stopped covering. All five are
#: git-tracked and shipped, so all five are required -- tests/test_verify.py
#: pins that, so marking one optional to silence a red gate breaks a test.
ARTIFACTS: tuple[tuple[str, str, str, bool], ...] = (
    ("README.md", "markdown", "en", True),
    ("router/console.html", "html", "en", True),
    ("presentation.html", "html", "en", True),
    ("templates/presentation.html", "html", "en", True),
    ("site/index.html", "html", "en", True),
)

ARTIFACT_PATHS: tuple[str, ...] = tuple(rel for rel, _, _, _ in ARTIFACTS)

#: Numerals that are structure rather than measurement, each with the reason it
#: is not a claim. Entries are CONTEXT patterns, not bare values: "5" is a claim
#: (model.cv.folds) and "Quick start (5 minutes)" is not, and only the
#: surrounding words tell them apart. Matches are blanked before extraction.
#: tests/test_verify.py caps this dict and requires every entry to still match.
NOT_A_CLAIM: dict[str, str] = {
    r"\d\d\s*[-‒-―]\s*\d\d\s+Aug\s+\d{4}":
        "the hackathon dates on README.md's first line -- a calendar fact about "
        "the event, measured from nothing in the export",
    r"Quick start \(\d+ minutes\)":
        "a reading-time estimate for the README's own instructions, not a "
        "quantity derived from any trajectory",
    r"Python \d+\.\d+\+":
        "the interpreter version floor the repo needs to run; a dependency "
        "requirement, not a finding",
    r"(?m)^\s*0\d\s*[-‒-―]\s":
        "panel numbers on the console and the deck (\"01 - Frontier\"); they "
        "order the slides and carry no measurement",
    r"Gate\s*[1-9]\b":
        "ordinals naming which link of the five-gate chain is meant; the chain "
        "length is a design choice, the ordinal is a label",
    r"\d+\s*(?:%[-\s]?(?:Obergrenze|upper bound|confidence|CI\b)"
    r"|(?:st|nd|rd|th)\s+percentile)":
        "the level every interval and permutation cut-off in this project is "
        "reported at -- a convention we chose, so no claims key backs it. Matching "
        "it by value would bind it to family.without_heartbeat.n_runs by accident",
}

_NOT_A_CLAIM_RE = {p: re.compile(p) for p in NOT_A_CLAIM}

#: Locale-specific literal grammars. Each alternative is anchored so a grouped
#: number is consumed whole rather than split at its separator.
_NUMERAL_RE = {
    "en": re.compile(r"[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-−]?\d+(?:\.\d+)?"),
    "de": re.compile(r"[-−]?\d{1,3}(?:\.\d{3})+(?:,\d+)?|[-−]?\d+(?:,\d+)?"),
}


@dataclass(frozen=True)
class Binding:
    """One prose numeral and every claims key whose value it could be."""

    path: str
    line: int
    literal: str
    value: float
    keys: tuple[str, ...]
    via_percent: bool


@dataclass(frozen=True)
class Orphan:
    """One prose numeral that no claim in claims.json stands behind."""

    path: str
    line: int
    literal: str
    value: float


@dataclass
class Report:
    """The outcome of one verification pass over the tracked artifacts."""

    bindings: list = field(default_factory=list)
    orphans: list = field(default_factory=list)
    scanned: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    unused_allowlist: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the whole declared surface was read and nothing is orphaned.

        `missing` counts as much as `orphans`: a required artifact that is not
        on disk was not checked, and an unchecked artifact has not passed. The
        two are still reported separately because they need different fixes.
        """
        return not self.orphans and not self.missing


# ------------------------------------------------------------------ stripping

_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_SCRIPTISH_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_OPEN_SCRIPTISH_RE = re.compile(r"<(script|style)\b[^>]*>.*", re.S | re.I)
#: A tag runs to its `>`, but a `>` INSIDE a quoted attribute value is not the
#: end of the tag -- `<[^>]+>` closes there and the rest of the tag is then read
#: as prose. The three alternatives are disjoint on their first character, so
#: this stays linear: no ambiguity for the engine to backtrack over.
_TAG_RE = re.compile(r"""<(?:[^>"']|"[^"]*"|'[^']*')*>""")
_FENCE_RE = re.compile(r"^```.*?^```", re.S | re.M)
#: A fence with no closing partner, mirroring `_OPEN_SCRIPTISH_RE` on the HTML
#: side. Without it a dropped ``` spills a shell command into the scanned prose.
_OPEN_FENCE_RE = re.compile(r"^```.*", re.S | re.M)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _blank(match) -> str:
    """Replace a match with spaces and newlines so line numbers survive."""
    return re.sub(r"[^\n]", " ", match.group(0))


def prose(text: str, kind: str) -> str:
    """Reduce an artifact to the text a reader actually sees, line numbers intact.

    HTML loses comments, <script>/<style> bodies and tags; markdown loses fenced
    blocks and inline code spans. Everything removed is replaced with blanks
    rather than deleted, so a reported line number points at the real line.
    """
    if kind == "html":
        text = _COMMENT_RE.sub(_blank, text)
        text = _SCRIPTISH_RE.sub(_blank, text)
        text = _OPEN_SCRIPTISH_RE.sub(_blank, text)  # an unclosed <script> at EOF
        return _TAG_RE.sub(_blank, text)
    if kind == "markdown":
        text = _FENCE_RE.sub(_blank, text)
        text = _OPEN_FENCE_RE.sub(_blank, text)  # an unclosed ``` at EOF
        return _INLINE_CODE_RE.sub(_blank, text)
    raise ValueError("unknown artifact kind %r" % (kind,))


def unclosed_fences(text: str, kind: str) -> list:
    """Line numbers of code fences with no closing partner, 1-indexed.

    Blanking a dangling fence to end of file is the conservative read of a
    malformed document -- but it can also HIDE a real numeral written after
    the bad fence, turning a fail into a quiet pass. So the gate says out loud
    where it stopped reading. HTML has the same shape in `_OPEN_SCRIPTISH_RE`
    and is not reported, because an unclosed <script> is a browser-visible
    breakage a reader cannot miss; a dropped ``` still renders.
    """
    if kind != "markdown":
        return []
    balanced = _FENCE_RE.sub(_blank, text)
    return [i + 1 for i, line in enumerate(balanced.split("\n"))
            if line.startswith("```")]


def apply_allowlist(text: str) -> tuple[str, set]:
    """Blank every NOT_A_CLAIM context; return the text and the patterns that hit."""
    used = set()
    for pattern, rx in _NOT_A_CLAIM_RE.items():
        text, n = rx.subn(_blank, text)
        if n:
            used.add(pattern)
    return text, used


# -------------------------------------------------------------------- parsing

def parse_literal(literal: str, locale: str) -> tuple[float, int]:
    """Return (value, decimals) for one literal under a DECLARED locale.

    `en`: comma groups thousands, dot is the decimal point -- "10,845" is 10845.
    `de`: dot groups thousands, comma is the decimal point -- "11,4" is 11.4.
    No sniffing: the locale comes from ARTIFACTS, so the same digits never mean
    two things depending on what else is in the file.
    """
    body = literal.replace("−", "-")
    sign = -1.0 if body.startswith("-") else 1.0
    body = body.lstrip("-")
    if locale == "de":
        body = body.replace(".", "").replace(",", ".")
    elif locale == "en":
        body = body.replace(",", "")
    else:
        raise ValueError("unknown locale %r" % (locale,))
    decimals = len(body.split(".")[1]) if "." in body else 0
    return sign * float(body), decimals


def _numeric_claims(claims: dict) -> dict:
    """The claims a prose numeral could possibly be: the numbers, not the strings."""
    return {k: float(v) for k, v in claims.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def resolve(value: float, decimals: int, numeric: dict, percent: bool) -> tuple:
    """Every claims key `value` is a correct rounding of, at its own precision.

    A literal binds when |literal - claim| <= 0.5 * 10^-decimals, i.e. when
    printing the claim at the precision the prose used would reproduce the
    literal. When the numeral is followed by `%` the claim may also be the
    corresponding fraction, since shares live in claims.json as fractions.
    """
    tol = 0.5 * (10.0 ** -decimals)
    hits = [k for k, v in numeric.items() if abs(v - value) <= tol]
    if percent:
        hits += [k for k, v in numeric.items()
                 if abs(v * 100.0 - value) <= tol and k not in hits]
    return tuple(sorted(hits))


# ---------------------------------------------------------------------- pass

def load_claims(path: str) -> dict:
    """Load the claims table; raise FileNotFoundError when it has not been built."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def scan_artifact(text: str, rel: str, kind: str, locale: str,
                  numeric: dict) -> tuple[list, list, set]:
    """Bind or orphan every prose numeral in one artifact."""
    stripped, used = apply_allowlist(prose(text, kind))
    bindings, orphans = [], []
    for lineno, line in enumerate(stripped.split("\n"), 1):
        for m in _NUMERAL_RE[locale].finditer(line):
            literal = m.group(0)
            value, decimals = parse_literal(literal, locale)
            # Scan to the next non-space, never a fixed slice: two spaces
            # before the sign used to disable the fraction path silently. Only
            # WHITESPACE may sit between -- anything else and the tolerance
            # would start binding numerals that are not shares at all.
            percent = line[m.end():].lstrip(" \t").startswith("%")
            keys = resolve(value, decimals, numeric, percent)
            if keys:
                bindings.append(Binding(rel, lineno, literal, value, keys, percent))
            else:
                orphans.append(Orphan(rel, lineno, literal, value))
    return bindings, orphans, used


def run(root: str = REPO_ROOT, claims_path: str | None = None) -> Report:
    """Verify every tracked user-facing artifact under `root` against the claims."""
    claims_path = claims_path or os.path.join(root, CLAIMS_NAME)
    numeric = _numeric_claims(load_claims(claims_path))
    report = Report()
    used: set = set()
    for rel, kind, locale, required in ARTIFACTS:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            (report.missing if required else report.skipped).append(rel)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        for ln in unclosed_fences(text, kind):
            report.warnings.append(
                (rel, ln, "unclosed ``` fence -- everything after it was ignored"))
        bindings, orphans, hit = scan_artifact(text, rel, kind, locale, numeric)
        report.bindings.extend(bindings)
        report.orphans.extend(orphans)
        report.scanned.append(rel)
        used |= hit
    report.unused_allowlist = sorted(set(NOT_A_CLAIM) - used)
    return report


# -------------------------------------------------------------------- banner

def print_report(report: Report) -> None:
    """Print every binding, then every orphan, in the order they were read."""
    print("router.verify - every numeral quoted in a user-facing artifact, against")
    print("the claims table. Token counts behind these claims are ESTIMATES and")
    print("every dollar is an ASSUMPTION; this gate checks provenance, not truth.\n")
    for rel in report.scanned:
        bound = [b for b in report.bindings if b.path == rel]
        orph = [o for o in report.orphans if o.path == rel]
        print("%s - %d bound, %d orphan" % (rel, len(bound), len(orph)))
        for b in bound:
            shown = b.keys[0] + (" (+%d more)" % (len(b.keys) - 1) if len(b.keys) > 1 else "")
            print("    :%-4d %-10s -> %s%s"
                  % (b.line, b.literal, shown, "  [as %]" if b.via_percent else ""))
        for o in orph:
            print("    :%-4d %-10s -> ORPHAN (no claim has this value)" % (o.line, o.literal))
    for rel in report.skipped:
        print("%s - declared optional and absent, skipped" % rel)
    for rel in report.missing:
        print("%s - REQUIRED and absent from this tree" % rel)
    print()
    if report.missing:
        print("BLOCKED - %d required artifact(s) could not be read: %s."
              % (len(report.missing), ", ".join(report.missing)))
        print("The gate covers a surface declared in router/verify.ARTIFACTS, so a")
        print("file that moved narrows that surface without saying so -- and a")
        print("narrowed gate reads exactly like a clean one. Restore the path, or")
        print("update ARTIFACTS and say in docs/DECISIONS.md why it moved.")
    if report.orphans:
        print("FAIL - %d orphan numeral(s). Every number in a user-facing artifact"
              % len(report.orphans))
        print("must exist in the claims table under a stable key (docs/CONTRACTS.md,")
        print("ADR-005). Add it through router/report.py, or -- if it is structural")
        print("rather than measured -- add a NOT_A_CLAIM entry with its reason.")
    if not report.orphans and not report.missing:
        print("PASS - %d numeral(s) across %d artifact(s), all accounted for."
              % (len(report.bindings), len(report.scanned)))
    for rel, ln, why in report.warnings:
        print("WARNING - %s:%d %s" % (rel, ln, why))
    if report.unused_allowlist:
        print("\nNOTE - NOT_A_CLAIM entries that matched nothing: %s"
              % ", ".join(report.unused_allowlist))


def main(argv=None) -> int:
    """Run the prose gate; return 0 clean, 1 on an orphan, 2 when claims are missing."""
    argv = list(sys.argv[1:] if argv is None else argv)
    root = REPO_ROOT
    claims_path = None
    for flag, setter in (("--root", "root"), ("--claims", "claims")):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                print("%s needs a path" % flag, file=sys.stderr)
                return 2
            if setter == "root":
                root = argv[i + 1]
            else:
                claims_path = argv[i + 1]
    try:
        report = run(root=root, claims_path=claims_path)
    except FileNotFoundError as exc:
        print("BLOCKED - %s" % exc, file=sys.stderr)
        print("Run `python -m router.report` to build the claims table first.",
              file=sys.stderr)
        return 2
    print_report(report)
    if report.missing:
        # 2 and not 1, mirroring the absent claims table: the run did not cover
        # what it declares it covers, so its orphan list is partial and no
        # verdict about the prose follows from it. Orphans found on the part it
        # DID read are still printed above -- blocking must not swallow them.
        return 2
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
