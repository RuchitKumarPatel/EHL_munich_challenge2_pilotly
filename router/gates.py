#!/usr/bin/env python3
"""Publication gate entry point: `python -m router.gates`.

WHAT THIS COMPUTES
    Nothing of its own. This module is the NAME the pipeline agreed on (the
    Makefile's `gates` target, `router.cli`'s `gates` stage and
    `skills/90-leak-audit/SKILL.md`'s `router.leakaudit`) bound to the ONE
    implementation that exists: `router.controls`, which runs the six negative
    controls (permutation null, random feature, no post-treatment column, SMD
    balance, weight concentration, growth-ratio out-of-sample).

    It exists because three owners named the same gate three different things.
    Rather than duplicate the checks — which would let two copies drift and
    disagree about whether a claim is licensed — this file re-exports
    `router.controls` and delegates `main()` to it verbatim.

WHAT IT WRITES
    No artifact. It reads results/ and prints a banner. Exit code is the gate:
    0 green (PASS/WARN), 1 FAIL, 2 BLOCKED (an artifact it needs is missing).

ACCEPTANCE (printed by `python -m router.gates`)
    Identical to `python -m router.controls` — same six controls, same banner,
    same exit code. On the shipped artifacts: 4 PASS / 2 WARN / 0 FAIL, gate
    GREEN. All token counts feeding these controls are ESTIMATES
    (tok(x) = len(json.dumps(x)) // 4); the export has no `usage` field.
"""

from __future__ import annotations

from router.controls import (  # noqa: F401  (re-exported on purpose)
    CHECKS,
    load_bundle,
    print_banner,
    run_all,
    selftest,
)
from router.controls import main as _controls_main


def main(argv=None) -> int:
    """Run the publication gate; return 0 green, 1 FAIL, 2 BLOCKED."""
    return _controls_main(argv)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
