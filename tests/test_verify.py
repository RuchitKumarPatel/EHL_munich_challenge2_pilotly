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
      * COVERAGE — a gate that silently narrows its own surface is worse than
        no gate. An artifact declared in ARTIFACTS and absent from the tree
        must BLOCK, not be skipped into a PASS line (ADR-018).

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
import subprocess
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


def _capture(fn, *args):
    """Call an entry point and return (exit code, everything it printed)."""
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = fn(*args)
    return rc, buf.getvalue() + err.getvalue()


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


class AMissingArtifactBlocksInsteadOfPassing(unittest.TestCase):
    """A declared artifact that is absent must block the gate, never be skipped.

    presentation.html carries 36 of the 44 bound numerals and only reached the
    repo root at turn 5. Before this, a rename or a relocation by any later
    deck turn silently removed the whole deck from the gate's surface and the
    run still printed PASS -- so the prose contract would have read as enforced
    on a tree where nothing checked the deck at all.
    """

    def setUp(self):
        _need_claims()
        self.tmp = tempfile.mkdtemp(prefix="router-verify-missing-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for rel in verify.ARTIFACT_PATHS:
            dst = os.path.join(self.tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(REPO_ROOT, rel), dst)

    def _drop(self, rel):
        os.remove(os.path.join(self.tmp, rel))

    def test_removing_the_deck_is_not_ok(self):
        self._drop("presentation.html")
        report = verify.run(root=self.tmp, claims_path=CLAIMS)
        self.assertIn("presentation.html", report.missing)
        self.assertFalse(
            report.ok,
            "the deck vanished from the scanned surface and the report still "
            "read clean; a gate that cannot see an artifact has not passed it")

    def test_the_entry_point_exits_non_zero_and_names_the_missing_path(self):
        self._drop("presentation.html")
        rc, out = _capture(verify.main, ["--root", self.tmp, "--claims", CLAIMS])
        self.assertNotEqual(rc, 0, "a missing artifact exited 0")
        self.assertEqual(rc, 2, "an artifact the gate cannot read is BLOCKED (2), "
                                "not an orphan (1) -- see router/verify.py")
        self.assertIn("presentation.html", out)
        self.assertNotIn("PASS", out)

    def test_orphans_do_not_hide_behind_a_missing_artifact(self):
        # Both wrong at once: the run must still name the orphan it did find,
        # so a blocked exit never swallows a real finding.
        self._drop("README.md")
        path = os.path.join(self.tmp, "presentation.html")
        with open(path, "r", encoding="utf-8") as fh:
            blob = fh.read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(blob.replace("10,845", "10,846", 1))
        rc, out = _capture(verify.main, ["--root", self.tmp, "--claims", CLAIMS])
        self.assertEqual(rc, 2)
        self.assertIn("README.md", out)
        self.assertIn("10,846", out)

    def test_every_artifact_absent_still_blocks(self):
        for rel in verify.ARTIFACT_PATHS:
            self._drop(rel)
        rc, out = _capture(verify.main, ["--root", self.tmp, "--claims", CLAIMS])
        self.assertEqual(rc, 2, "an empty surface printed a verdict")
        self.assertNotIn("PASS", out)


class TheDeclaredSurfaceMatchesTheTree(unittest.TestCase):
    """`required` is a declaration about the repo, so the repo must bear it out."""

    def test_every_required_artifact_exists_in_the_shipped_tree(self):
        absent = [rel for rel, _, _, req in verify.ARTIFACTS
                  if req and not os.path.isfile(os.path.join(REPO_ROOT, rel))]
        self.assertEqual(
            absent, [],
            "declared required but not in the tree -- either the file moved and "
            "ARTIFACTS did not follow, or it is genuinely optional and must say "
            "so with a reason: %s" % (absent,))

    def test_no_artifact_is_quietly_downgraded_to_optional(self):
        # The failure mode this pins is social, not mechanical: the cheapest way
        # to turn a red gate green is to mark the artifact it complains about
        # optional. Flipping a flag here has to break a test and be argued for.
        optional = [rel for rel, _, _, req in verify.ARTIFACTS if not req]
        self.assertEqual(
            optional, [],
            "every user-facing artifact is git-tracked and shipped, so none is "
            "optional; if that changed, record why in docs/DECISIONS.md: %s"
            % (optional,))

    def test_every_declared_artifact_is_git_tracked(self):
        # An untracked file is not user-facing shipped content, so it has no
        # business being a required part of the gate's surface.
        try:
            out = subprocess.run(
                ["git", "-C", REPO_ROOT, "ls-files", "-z", "--"]
                + list(verify.ARTIFACT_PATHS),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise unittest.SkipTest("git is unavailable here: %s" % (exc,))
        tracked = set(out.stdout.decode("utf-8").split("\0")) - {""}
        untracked = [rel for rel in verify.ARTIFACT_PATHS if rel not in tracked]
        self.assertEqual(untracked, [], "declared but not git-tracked: %s"
                         % (untracked,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
