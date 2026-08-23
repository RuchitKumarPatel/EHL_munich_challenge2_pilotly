"""The prose-to-claims contract, as assertions.

WHAT THIS CHECKS
    docs/CONTRACTS.md and ADR-005 both assert that any number quoted anywhere
    must exist in results/claims.json under a stable key. `router.verify` is
    what enforces that on the four git-tracked user-facing artifacts. This file
    checks the enforcer in both directions:

      * POSITIVE — the shipped tree verifies clean, and two specific bindings
        that the parser has to get right are actually made: router/console.html
        is GERMAN, so its "11,4 pp" is a decimal comma and must bind to
        refusal.mde_best_powered_arm_pair.pp, not to 114; and its "1000" must
        bind to corpus.n_trajectories.
      * NEGATIVE CONTROL — without this the check is unfalsifiable. A scratch
        copy of the tree with one number perturbed must exit non-zero and name
        the file, the line and the offending literal.

    Plus the allowlist discipline copied from tests/test_data_safety.py: the
    NOT_A_CLAIM escape hatch stays short and every entry carries a real
    justification, so prose is not normalised past the gate one exception at a
    time.

WHAT IT WRITES
    Nothing outside a temporary directory. When results/claims.json is absent
    the tests skip with a message naming the module that builds it, rather than
    failing for the wrong reason.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import unittest

from router import verify

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS = os.path.join(REPO_ROOT, "results", "claims.json")


def _quiet(fn, *args):
    """Call an entry point with its banner swallowed; `make test` stays readable."""
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        return fn(*args)


def _need_claims():
    """Skip with an actionable message when the claims table is missing."""
    if not os.path.exists(CLAIMS):
        raise unittest.SkipTest(
            "the claims table is missing; run `python -m router.report` first")


class TheShippedTreeVerifiesClean(unittest.TestCase):
    """Every numeral in every tracked user-facing artifact resolves to a claim."""

    def setUp(self):
        _need_claims()
        self.report = verify.run(root=REPO_ROOT)

    def test_no_orphan_numeral_in_any_artifact(self):
        orphans = [(o.path, o.line, o.literal) for o in self.report.orphans]
        self.assertEqual(
            orphans, [],
            "numerals quoted in a user-facing artifact with no claim behind them. "
            "Either add the number to claims.json through router/report.py, or -- if "
            "it is structural rather than a measurement -- add a NOT_A_CLAIM entry "
            "with its justification. Orphans: %s" % (orphans,))

    def test_the_entry_point_exits_zero(self):
        self.assertEqual(_quiet(verify.main, []), 0)

    def test_the_german_decimal_comma_binds_to_the_mde(self):
        # router/console.html:  "die MDE liegt bei 11,4 pp".  A parser that reads
        # the comma as an English thousands separator sees 114 and reports a
        # false orphan.  This is the one non-obvious case in the whole module.
        keys = self._keys_for("router/console.html", "11,4")
        self.assertIn("refusal.mde_best_powered_arm_pair.pp", keys)

    def test_the_corpus_size_binds_in_the_german_console(self):
        keys = self._keys_for("router/console.html", "1000")
        self.assertIn("corpus.n_trajectories", keys)

    def _keys_for(self, path, literal):
        found = [b for b in self.report.bindings
                 if b.path == path and b.literal == literal]
        self.assertTrue(found, "no %r binding recorded for %s" % (literal, path))
        return sorted({k for b in found for k in b.keys})


class PerturbingOneNumberIsCaught(unittest.TestCase):
    """The negative control: a number that drifts off its claim must fail loudly."""

    def setUp(self):
        _need_claims()
        self.tmp = tempfile.mkdtemp(prefix="router-verify-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for rel in verify.ARTIFACT_PATHS:
            src = os.path.join(REPO_ROOT, rel)
            dst = os.path.join(self.tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

    def _perturb(self, rel, old, new):
        path = os.path.join(self.tmp, rel)
        with open(path, "r", encoding="utf-8") as fh:
            blob = fh.read()
        self.assertIn(old, blob, "%s no longer contains %r" % (rel, old))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(blob.replace(old, new, 1))

    def test_a_perturbed_german_numeral_is_reported_with_file_and_line(self):
        # 11,9 and not 11,7: 11.7 is recon.turns.per_line.claude-fable-5, so a
        # one-tenth slip would have bound to an unrelated claim by coincidence.
        # Value matching finds orphans, not wrong keys -- see router/verify.py.
        self._perturb("router/console.html", "11,4 pp", "11,9 pp")
        report = verify.run(root=self.tmp, claims_path=CLAIMS)
        self.assertTrue(report.orphans, "perturbing the MDE was not caught")
        hit = [o for o in report.orphans if o.literal == "11,9"]
        self.assertTrue(hit, "orphans were reported but not the perturbed literal: %s"
                        % ([(o.path, o.literal) for o in report.orphans],))
        self.assertEqual(hit[0].path, "router/console.html")
        self.assertGreater(hit[0].line, 0)

    def test_a_perturbed_english_numeral_is_reported(self):
        self._perturb("presentation.html", "10,845", "10,846")
        report = verify.run(root=self.tmp, claims_path=CLAIMS)
        self.assertIn("10,846", [o.literal for o in report.orphans])

    def test_the_entry_point_exits_one_on_an_orphan(self):
        self._perturb("presentation.html", "412.95", "412.96")
        self.assertEqual(
            _quiet(verify.main, ["--root", self.tmp, "--claims", CLAIMS]), 1)


class MissingClaimsIsBlockedNotGreen(unittest.TestCase):
    """An absent claims table must block (exit 2), never pass by default."""

    def test_exit_two_when_the_claims_table_is_absent(self):
        tmp = tempfile.mkdtemp(prefix="router-verify-noclaims-")
        self.addCleanup(shutil.rmtree, tmp, True)
        absent = os.path.join(tmp, "claims.json")
        self.assertEqual(_quiet(verify.main, ["--claims", absent]), 2)


class TheEscapeHatchStaysHonest(unittest.TestCase):
    """NOT_A_CLAIM is an allowlist, and an allowlist has to stay readable."""

    def test_the_allowlist_stays_small(self):
        self.assertLessEqual(
            len(verify.NOT_A_CLAIM), 8,
            "the non-claim allowlist is growing; prose is being normalised past "
            "the gate one exception at a time")

    def test_every_entry_carries_a_real_justification(self):
        for pattern, reason in verify.NOT_A_CLAIM.items():
            self.assertGreater(
                len(reason), 20,
                "%s is allowlisted without a real justification" % (pattern,))

    def test_every_entry_still_matches_something(self):
        # A stale allowlist entry is worse than none: it reads as a live
        # exemption while covering nothing, and hides that the prose moved on.
        _need_claims()
        unused = verify.run(root=REPO_ROOT).unused_allowlist
        self.assertEqual(
            unused, [],
            "NOT_A_CLAIM entries that matched nothing in any artifact -- remove "
            "them, or the allowlist stops describing the tree: %s" % (unused,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
