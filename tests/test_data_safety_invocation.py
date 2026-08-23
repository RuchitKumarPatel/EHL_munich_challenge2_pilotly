"""A data-safety verdict may not depend on how the suite was started.

WHY THIS FILE EXISTS
    tests/test_data_safety.py ships a `__main__` guard, so a human debugging on
    defense morning will run it the way the file invites:

        python tests/test_data_safety.py

    That form puts `tests/` on sys.path and NOT the repo root, so `router` is
    unimportable. Two places absorb that differently and both are wrong:

      * `RepoCarriesNoOpaqueExportToken._exempt_arm_ids` wrapped
        `from router.pricing import OBSERVED_ARMS` in `except Exception:
        return frozenset()`. The exemption table is a CORRECTNESS input to that
        check -- every arm id occurs in the export by construction -- so an
        empty table turns nine legitimate identifiers into nine reported leaks.
      * `ResultsCarryNoRawToolOutput._shingles` imports `router.io` unguarded,
        so its setUpClass errors and the whole class silently stops running.

    Measured on the tree this file was written against: the `-m` form ran 21
    tests and was green, the script form ran 19 and reported one failure plus
    two errors. Two invocations of one file, two different answers about
    whether the repo is safe to push.

WHAT IS TESTED
    1. PARITY. Both invocations are run as subprocesses and must agree on how
       many tests ran and how many failed, and both must succeed. This is the
       behavioural statement; everything else is a control on it.
    2. The arm-id exemption is not silently empty. Asserted against the pricing
       table itself, so it tracks the arm list rather than a copied count.
    3. The degrade path is GONE, not merely unreachable: with `router.pricing`
       made unimportable, `_exempt_arm_ids` must raise rather than hand back an
       empty exemption. Without this control, restoring the `except Exception`
       would leave tests 1 and 2 green -- they only exercise the happy path.

WHY PARITY AND NOT "THE SCRIPT FORM WORKS"
    The failure mode is disagreement. A future change that breaks BOTH forms in
    the same way is a different bug and other files catch it; a change that
    breaks one form is this bug returning, and only a comparison sees it.

RUNTIME
    ~7 s: two full subprocess runs of the data-safety suite at ~3 s each.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_REL = os.path.join("tests", "test_data_safety.py")

#: `unittest`'s own summary, printed by both the plain runner and any wrapper
#: that drives a TextTestRunner. Parsing this rather than a project-specific
#: receipt line keeps the comparison valid across the gate runner as well.
RAN_RE = re.compile(r"^Ran (\d+) tests? in ", re.M)
FAILED_RE = re.compile(r"^FAILED \((.*)\)\s*$", re.M)
COUNT_RE = re.compile(r"(failures|errors)=(\d+)")


def _run(argv):
    """Run one invocation of the data-safety suite from the repo root."""
    proc = subprocess.run(
        [sys.executable] + argv,
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        # A caller's SCAN_REF would make the two forms scan different trees for
        # reasons that have nothing to do with this bug.
        env={k: v for k, v in os.environ.items() if k != "DATA_SAFETY_SCAN_REF"})
    return proc


def _summary(proc):
    """(tests_run, n_failed) out of a unittest run's stderr."""
    out = proc.stderr + proc.stdout
    ran = RAN_RE.search(out)
    if ran is None:
        raise AssertionError(
            "no 'Ran N tests' line in the output of this invocation; exit=%d\n"
            "last 400 chars:\n%s" % (proc.returncode, out[-400:]))
    failed_block = FAILED_RE.search(out)
    n_failed = 0
    if failed_block:
        n_failed = sum(int(n) for _kind, n in COUNT_RE.findall(failed_block.group(1)))
    return int(ran.group(1)), n_failed


class TheSuiteAnswersTheSameHoweverItIsStarted(unittest.TestCase):
    """`python tests/test_data_safety.py` and `python -m tests.test_data_safety`."""

    @classmethod
    def setUpClass(cls):
        cls.module = _run(["-m", "tests.test_data_safety"])
        cls.script = _run([SUITE_REL])

    def test_both_invocations_run_the_same_number_of_tests(self):
        m_ran, _ = _summary(self.module)
        s_ran, _ = _summary(self.script)
        self.assertEqual(
            m_ran, s_ran,
            "the two invocations ran different test counts (-m: %d, script: %d). A "
            "class whose setUpClass errored does not run its tests, so a count gap "
            "means one form silently stopped checking something." % (m_ran, s_ran))

    def test_both_invocations_report_the_same_failure_count(self):
        _, m_failed = _summary(self.module)
        _, s_failed = _summary(self.script)
        self.assertEqual(
            m_failed, s_failed,
            "the two invocations disagree about whether this repo is safe "
            "(-m: %d failed, script: %d failed). Script-form output tail:\n%s"
            % (m_failed, s_failed, (self.script.stderr + self.script.stdout)[-1200:]))

    def test_both_invocations_exit_zero(self):
        self.assertEqual(
            (self.module.returncode, self.script.returncode), (0, 0),
            "data-safety suite is not green: -m exit=%d, script exit=%d"
            % (self.module.returncode, self.script.returncode))


class TheArmIdExemptionIsNeverSilentlyEmpty(unittest.TestCase):
    """The exemption is a correctness input, so losing it must be loud."""

    def test_the_exemption_matches_the_pricing_table(self):
        from router.pricing import OBSERVED_ARMS
        from tests.test_data_safety import RepoCarriesNoOpaqueExportToken as K

        self.assertEqual(K._exempt_arm_ids(), frozenset(OBSERVED_ARMS))
        self.assertTrue(OBSERVED_ARMS, "the pricing table lists no arms")

    def test_an_unimportable_pricing_table_raises_instead_of_emptying(self):
        """Negative control on the degrade path itself."""
        from tests.test_data_safety import RepoCarriesNoOpaqueExportToken as K

        saved = sys.modules.get("router.pricing")
        # None in sys.modules makes `from router.pricing import ...` raise
        # ImportError, which is exactly what the old `except Exception` ate.
        sys.modules["router.pricing"] = None
        try:
            with self.assertRaises(ImportError):
                K._exempt_arm_ids()
        finally:
            if saved is None:
                sys.modules.pop("router.pricing", None)
            else:
                sys.modules["router.pricing"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
