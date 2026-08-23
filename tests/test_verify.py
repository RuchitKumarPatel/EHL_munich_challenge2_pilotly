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
import re
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


class TheProseStripperLeavesNothingAReaderCannotSee(unittest.TestCase):
    """What the gate scans must be what the page shows -- no more, no less.

    Three ways it was not, each reproduced against the pre-fix module before
    the fix was written, and each LATENT against the five shipped artifacts
    today (measured: the quote-aware tag rule changes 0 lines across all four
    HTML artifacts, and README.md carries 0 unclosed and 0 indented fences).
    That is the point of pinning them now -- these fire the first time a human
    edits the deck or the README, which happens tomorrow morning.
    """

    def test_an_unclosed_fence_is_blanked_to_end_of_file(self):
        # `_FENCE_RE` requires a CLOSING fence. The HTML path already has an
        # open-<script>-at-EOF fallback; markdown had none, so a dropped
        # closing fence spilled a shell command into the scanned prose.
        text = "prose line\n```bash\nrun-command --threshold 999\n"
        self.assertNotIn("999", verify.prose(text, "markdown"))

    def test_the_prose_before_an_unclosed_fence_still_survives(self):
        # The mirror control: blanking to EOF must not eat the document.
        text = "the AUPRC is 0.4160\n```bash\nrun-command --threshold 999\n"
        self.assertIn("0.4160", verify.prose(text, "markdown"))

    def test_an_unclosed_fence_keeps_the_line_numbering_intact(self):
        # Every stripper in this module blanks rather than deletes, or a
        # reported line number stops pointing at the real line.
        text = "a\n```bash\nb\nc\n"
        self.assertEqual(len(verify.prose(text, "markdown").split("\n")),
                         len(text.split("\n")))

    def test_a_closed_fence_does_not_swallow_what_follows_it(self):
        # Regression control on the fix itself: the open-fence fallback must
        # not fire on a document whose fences are balanced.
        text = "a\n```bash\nrun --x 999\n```\nthe AUPRC is 0.4160\n"
        out = verify.prose(text, "markdown")
        self.assertNotIn("999", out)
        self.assertIn("0.4160", out)

    def test_an_unclosed_fence_is_reported_rather_than_silently_obeyed(self):
        # Blanking to EOF is the conservative read of a malformed document,
        # but it can also HIDE a real numeral that sits after the bad fence.
        # So the gate says out loud that it stopped reading, naming the line.
        warnings = verify.unclosed_fences("a\nb\n```bash\nc\n", "markdown")
        self.assertEqual(warnings, [3])
        self.assertEqual(verify.unclosed_fences("a\n```\nb\n```\n", "markdown"), [])

    def test_a_gt_inside_a_quoted_attribute_does_not_leak_a_numeral(self):
        # `<[^>]+>` closes at the FIRST `>`, including one inside a quoted
        # attribute value, so the tail of the tag was scanned as prose.
        out = verify.prose('<div title="revenue 5 > 2 loss">only 7 remains</div>',
                           "html")
        self.assertNotIn("2", out)
        self.assertIn("7", out)

    def test_the_same_holds_for_a_single_quoted_attribute(self):
        out = verify.prose("<div title='a > 2 b'>only 7 remains</div>", "html")
        self.assertNotIn("2", out)
        self.assertIn("7", out)

    def test_ordinary_markup_still_strips(self):
        # The mirror control: a quote-aware tag rule that stopped matching
        # plain tags would leak every attribute in the deck instead.
        out = verify.prose('<p class="lead">the AUPRC is 0.4160</p>', "html")
        self.assertNotIn("lead", out)
        self.assertIn("0.4160", out)

    def test_the_quote_aware_rule_changes_nothing_on_the_shipped_artifacts(self):
        # The claim that this fix is HARDENING and not a behaviour change,
        # asserted rather than asserted-in-a-note.
        old = re.compile(r"<[^>]+>")
        for rel, kind, _locale in [a[:3] for a in verify.ARTIFACTS]:
            path = os.path.join(REPO_ROOT, rel)
            if kind != "html" or not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            text = verify._COMMENT_RE.sub(verify._blank, text)
            text = verify._SCRIPTISH_RE.sub(verify._blank, text)
            text = verify._OPEN_SCRIPTISH_RE.sub(verify._blank, text)
            self.assertEqual(old.sub(verify._blank, text),
                             verify._TAG_RE.sub(verify._blank, text),
                             "%s: the quote-aware tag rule changed what is "
                             "scanned; that is a real change, not hardening" % rel)


class ThePercentLookaheadScansRatherThanSlices(unittest.TestCase):
    """Shares live in claims.json as fractions and are quoted as percents.

    The binding is only offered when the numeral is FOLLOWED BY `%` -- widen
    that and the tolerance binds anything. The old test was `line[end:end+2]`,
    a fixed two-character slice, so two spaces before the sign silently
    disabled the fraction path and a legitimate percent reported as an orphan.
    """

    NUMERIC = {"policy.refused.gross_share": 0.4098}

    def _keys(self, line):
        bindings, orphans, _used = verify.scan_artifact(
            line, "scratch.md", "markdown", "en", self.NUMERIC)
        return [b.keys for b in bindings], [o.literal for o in orphans]

    def test_a_percent_written_tight_binds(self):
        bound, _orphans = self._keys("the share is 41.0%")
        self.assertEqual(bound, [("policy.refused.gross_share",)])

    def test_a_percent_written_with_several_spaces_still_binds(self):
        # This is the defect: 41.0 and the % separated by more than one space.
        bound, orphans = self._keys("the share is 41.0   %")
        self.assertEqual(orphans, [])
        self.assertEqual(bound, [("policy.refused.gross_share",)])

    def test_a_percent_written_with_a_tab_still_binds(self):
        bound, _orphans = self._keys("the share is 41.0\t%")
        self.assertEqual(bound, [("policy.refused.gross_share",)])

    def test_a_numeral_with_words_before_the_percent_sign_does_not_bind(self):
        # The control that keeps the widening honest: only WHITESPACE may sit
        # between the numeral and the sign, or "41.0 of the total 90%" would
        # bind 41.0 as a fraction and the gate would stop finding orphans.
        _bound, orphans = self._keys("the share is 41.0 of the total")
        self.assertEqual(orphans, ["41.0"])


class TheTwoNewGuardsCompose(unittest.TestCase):
    """ADR-018 and ADR-020 landed on separate branches; this is the seam.

    Neither branch could write this test, because neither carried both
    features. ADR-018 makes an unreadable required artifact BLOCK the run;
    ADR-020 blanks a dangling markdown fence and warns about where it stopped
    reading. They meet in `run()` and in `print_report`, and the ways they
    could have met badly are all silent: a BLOCKED run that swallows the
    warning, a warning that suppresses the BLOCKED banner, or a PASS line
    printed alongside either. All three would leave a narrowed gate reading
    like a clean one, which is the single failure mode both ADRs exist to
    refuse.
    """

    def setUp(self):
        _need_claims()
        self.tmp = tempfile.mkdtemp(prefix="router-verify-compose-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for rel in verify.ARTIFACT_PATHS:
            dst = os.path.join(self.tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(REPO_ROOT, rel), dst)
        # One required artifact vanishes; a surviving one loses its closing
        # fence. Both defects at once, which is the case neither branch saw.
        os.remove(os.path.join(self.tmp, "presentation.html"))
        with open(os.path.join(self.tmp, "README.md"), "a",
                  encoding="utf-8") as fh:
            fh.write("\n```bash\nrun --threshold 999999\n")
        self.rc, self.out = _capture(
            verify.main, ["--root", self.tmp, "--claims", CLAIMS])

    def test_the_missing_artifact_still_blocks(self):
        self.assertEqual(self.rc, 2, "a warning downgraded a BLOCKED run")
        self.assertIn("BLOCKED", self.out)
        self.assertIn("presentation.html", self.out)

    def test_the_warning_survives_the_block(self):
        self.assertIn("WARNING", self.out)
        self.assertIn("README.md", self.out)

    def test_no_pass_line_is_printed_alongside_either(self):
        self.assertNotIn("PASS -", self.out)

    def test_the_fenced_literal_is_still_stripped_rather_than_orphaned(self):
        # ADR-020's blanking must keep working while ADR-018 is failing the
        # run, or the BLOCKED report grows a fake orphan a reader would chase.
        self.assertNotIn("999999", self.out)

if __name__ == "__main__":
    unittest.main(verbosity=2)
