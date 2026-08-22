"""The design matrix must not be able to see anything the model produced.

WHY THIS IS THE TEST THAT MATTERS
    46.5% of the estimated tokens on this corpus (10,517,503 of 22,631,879 on
    the naive per-item count) sit AFTER the first user message. Every one of
    those items is model output: assistant turns, reasoning blocks, tool calls
    and tool results. A router chooses the arm BEFORE any of that exists, so a
    feature computed from it is a collider -- it would predict friction by
    reading the friction. `empty_final` is the worked example: all 238 empty
    final messages are gpt, so that one column encodes the treatment perfectly
    and would look like a superb predictor while being a serialization artifact.

WHAT THIS CHECKS
    1. Rebuilding the entire feature matrix from inputs TRUNCATED at the first
       user message produces a byte-identical (X, cols, idx).
    2. Rebuilding it from inputs whose post-cut items have been REPLACED with
       junk assistant/function_call items, and whose `model` field has been
       overwritten with "REDACTED", produces the same bytes again. Truncation
       alone would pass if a feature silently treated a missing tail as zero;
       poisoning catches that.
    3. `extract()` REFUSES a full item list rather than quietly reading past
       the cut, so the gate is physical rather than a convention.
    4. The 12 banned column names are disjoint from the emitted columns, in the
       artifact as well as in the module.
    5. The manifest agrees with the .npz: same column order, all pre_treatment.

WHAT IT WRITES
    Nothing. `router.features.build()` is called with an explicit request
    iterable and its return value is compared in memory; results/features.npz
    is read but never rewritten.

RUNTIME
    ~8 s: three full 1000-line builds at ~2.5 s each.
"""

from __future__ import annotations

import json
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
FEATURES = os.path.join(RESULTS, "features.npz")
MANIFEST = os.path.join(RESULTS, "feature_manifest.json")

#: The 12 column names the brief bans, each with the measurement that killed it.
#: Kept here as a literal so a rename inside router/features.py cannot relax it.
BANNED_COLUMNS = frozenset({
    "est_tok",      # post-treatment: counts the whole trajectory
    "post_tok",     # post-treatment by definition
    "n_items",      # post-treatment: grows with the model's own turns
    "n_fc",         # function calls the model made
    "n_fco",        # function call outputs
    "n_asst",       # assistant items
    "n_reason",     # reasoning items
    "n_calls",      # tool calls -- this is the LABEL's input, not a feature
    "n_steps",      # same
    "final_len",    # length of the model's final message
    "empty_final",  # 238/238 empty finals are gpt: encodes family perfectly
    "sys_sha",      # 871 distinct over 1000 rows: an identifier, not a feature
})


#: Column names that trip the crude name heuristic below but are genuinely
#: pre-treatment, each with the reason it is exempt.
NAME_HEURISTIC_EXEMPT = {
    # "# === Expected output channel instructions ===" is a briefing block that
    # the harness pastes into the SYSTEM message. The flag says the block was
    # present before the run started; it says nothing about what the model wrote.
    "hdr_output_channel",
}


def _needs_export():
    """Skip when the export is not present (the tests are read-only over it)."""
    export = os.path.join(REPO_ROOT, "export")
    if not os.path.isdir(export):
        raise unittest.SkipTest("export/ is not present")


class TruncationEquivalence(unittest.TestCase):
    """Cutting or poisoning everything after the first user message changes nothing."""

    @classmethod
    def setUpClass(cls):
        _needs_export()
        from router import features

        cls.features = features
        cls.truncated_ok, cls.poisoned_ok = features.truncation_self_test()

    def test_truncated_input_gives_byte_identical_features(self):
        self.assertTrue(
            self.truncated_ok,
            "dropping every item after the first user message changed the design "
            "matrix, so at least one feature reads post-treatment data",
        )

    def test_poisoned_input_gives_byte_identical_features(self):
        self.assertTrue(
            self.poisoned_ok,
            "replacing the post-cut items with junk (and the arm id with REDACTED) "
            "changed the design matrix, so at least one feature reads past the cut",
        )

    def test_the_two_checks_are_not_the_same_check(self):
        # Truncation and poisoning differ: poisoning also redacts `model`, which
        # is the TREATMENT. If a feature ever read the arm, only this one fires.
        import inspect

        src = inspect.getsource(self.features._poisoned_requests)
        self.assertIn("REDACTED", src)
        self.assertIn('"model"', src)


class PhysicalGate(unittest.TestCase):
    """`pre_request` / `extract` refuse to be handed post-treatment data."""

    @classmethod
    def setUpClass(cls):
        _needs_export()
        from router import features
        from router.io import iter_lines

        cls.features = features
        cls.idx, cls.req = next(iter(iter_lines()))

    def test_pre_request_drops_the_treatment_and_ends_at_the_first_user_message(self):
        from router.io import first_user_index

        view = self.features.pre_request(self.req)
        self.assertNotIn("model", view, "pre_request must not carry the arm id")
        self.assertEqual(set(view), {"input", "tools"})
        items = view["input"]
        self.assertEqual(len(items), first_user_index(self.req["input"]) + 1)
        self.assertEqual(items[-1].get("role"), "user")

    def test_pre_request_is_shorter_than_the_full_trajectory(self):
        # Sanity: the slice must actually remove something on a real line.
        self.assertLess(len(self.features.pre_request(self.req)["input"]),
                        len(self.req["input"]))

    def test_extract_refuses_a_full_item_list(self):
        with self.assertRaises(ValueError):
            self.features.extract(self.idx, self.req, None)

    def test_extract_accepts_the_pre_request_view(self):
        row = self.features.extract(self.idx, self.features.pre_request(self.req), None)
        self.assertEqual(set(row), set(self.features.COLUMN_NAMES))


class BannedColumns(unittest.TestCase):
    """No emitted column may carry a banned name, in the module or in the artifact."""

    def test_module_columns_are_disjoint_from_the_banned_set(self):
        from router import features

        emitted = set(features.COLUMN_NAMES)
        self.assertEqual(emitted & BANNED_COLUMNS, set(),
                         "banned column names emitted by router.features")

    def test_the_modules_own_banned_set_covers_the_brief(self):
        from router import features

        self.assertTrue(
            BANNED_COLUMNS.issubset(set(features.BANNED)),
            "router.features.BANNED dropped a name the brief bans: %s"
            % sorted(BANNED_COLUMNS - set(features.BANNED)),
        )

    def test_artifact_columns_are_disjoint_from_the_banned_set(self):
        if not os.path.exists(FEATURES):
            raise unittest.SkipTest(
                "results/features.npz is missing; run `python -m router.features`")
        import numpy as np

        z = np.load(FEATURES, allow_pickle=False)
        cols = {str(c) for c in z["cols"]}
        self.assertEqual(cols & BANNED_COLUMNS, set(),
                         "banned column names present in results/features.npz")

    def test_no_column_name_smells_post_treatment(self):
        # A weaker, broader net than the exact ban list: nothing may be named
        # after an item the model produced.
        from router import features

        fragments = ("final", "assistant", "asst", "reason", "output", "post",
                     "sha", "hash", "n_call", "n_step")
        offenders = [c for c in features.COLUMN_NAMES
                     if any(f in c.lower() for f in fragments)
                     and c not in NAME_HEURISTIC_EXEMPT]
        self.assertEqual(offenders, [],
                         "column name suggests post-treatment content: %s" % offenders)


class ManifestAgreement(unittest.TestCase):
    """The manifest is the contract downstream reads; it must match the matrix."""

    @classmethod
    def setUpClass(cls):
        if not (os.path.exists(FEATURES) and os.path.exists(MANIFEST)):
            raise unittest.SkipTest(
                "results/features.npz or feature_manifest.json is missing; "
                "run `python -m router.features`")
        import numpy as np

        cls.z = np.load(FEATURES, allow_pickle=False)
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            cls.man = json.load(fh)

    def test_column_order_matches(self):
        self.assertEqual([c["name"] for c in self.man["cols"]],
                         [str(c) for c in self.z["cols"]])

    def test_every_manifest_entry_has_exactly_the_contract_fields(self):
        for entry in self.man["cols"]:
            self.assertEqual(set(entry), {"name", "pre_treatment", "block"},
                             "manifest entry %r" % entry.get("name"))
            self.assertIn(entry["block"], ("A", "B", "C"))

    def test_every_column_is_declared_pre_treatment(self):
        bad = [c["name"] for c in self.man["cols"] if not c["pre_treatment"]]
        self.assertEqual(bad, [], "post-treatment columns in the design matrix: %s" % bad)

    def test_shapes_agree(self):
        X, cols, idx = self.z["X"], self.z["cols"], self.z["idx"]
        self.assertEqual(X.shape, (idx.size, cols.size))
        self.assertEqual(idx.size, 1000)
        self.assertEqual(sorted(int(i) for i in idx), list(range(1000)))

    def test_the_outcome_derived_columns_stay_documented(self):
        """job_fric_eb / job_turns_eb summarise OTHER runs' outcomes; say so out loud.

        They pass the truncation test -- row i's own outcome never enters row i --
        but they are still built from y_fric and n_turns of same-job peers, which
        crosses a job-blocked fold boundary. router.strata already drops them for
        exactly that reason. This test does not judge the design; it fails only if
        the manifest stops warning about it, so the next reader cannot miss it.
        """
        names = {c["name"] for c in self.man["cols"]}
        eb = names & {"job_fric_eb", "job_turns_eb"}
        if not eb:
            self.skipTest("no empirical-Bayes columns in this manifest")
        notes = " ".join(self.man.get("notes", []))
        for col in sorted(eb):
            self.assertIn(col, notes,
                          "%s is emitted but the manifest does not explain how it is "
                          "built from other runs' outcomes" % col)
        self.assertIn("LEAVE-CURRENT-RUN-OUT", notes.upper())

    def test_the_matrix_is_finite(self):
        import numpy as np

        self.assertTrue(np.isfinite(self.z["X"]).all(), "non-finite value in the design matrix")


class PreTreatmentTokenSplit(unittest.TestCase):
    """The 53.5 / 46.5 split is the reason this whole file exists."""

    def test_split(self):
        _needs_export()
        from router import features

        pre, post = features.token_split()
        self.assertEqual(pre, 12_114_376)
        self.assertEqual(post, 10_517_503)
        self.assertEqual(pre + post, 22_631_879)
        self.assertAlmostEqual(100.0 * pre / (pre + post), 53.5, places=1)
        self.assertAlmostEqual(100.0 * post / (pre + post), 46.5, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
