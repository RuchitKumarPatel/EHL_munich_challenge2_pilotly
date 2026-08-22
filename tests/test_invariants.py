"""The pinned acceptance numbers, as assertions.

WHAT THIS CHECKS
    Every number in docs/CONTRACTS.md that a downstream slide or claim leans on,
    re-measured from the artifacts in results/ rather than copied from a module
    constant. A module is allowed to print PASS against its own EXPECTED dict;
    this file deliberately does not use those dicts, so a module that quietly
    edits its own expectations still fails here.

    Plus two behavioural invariants the brief calls out:
      * cost is monotone in rho (a higher cache-hit rate is never more expensive);
      * a fixed-seed bootstrap reproduces bit for bit.

WHAT IT WRITES
    Nothing. Read-only over results/. If an artifact is missing the relevant
    tests skip with a message naming the module that builds it, rather than
    failing for the wrong reason.

ONE DOCUMENTED DIVERGENCE FROM THE BRIEF
    The brief lists label base rates "9,633 / 474 / 259". 9,633 and 474 cannot
    both hold. Pairing tool calls to tool outputs is a bijection on this corpus
    (all 10,422 outputs consumed exactly once), so the status counts are a
    property of the output ITEMS: 9,161 ok + 474 err + 787 unknown = 10,422,
    i.e. 9,635 resolvable. 9,633 is reproducible only under a last-wins dict
    pairing, which mispairs calls to outputs AND breaks the 474 error count.
    This file asserts 9,635 and 474, and additionally asserts the arithmetic
    identity 9,635 + 787 == 10,422 so the divergence can never be hidden by
    moving a number. See router/labels.py's docstring.

CAVEAT
    All token counts asserted here are ESTIMATES: tok(x) = len(json.dumps(x))//4.
    The export has no `usage` field, so no token count in this project is measured.
"""

from __future__ import annotations

import collections
import json
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")


def _jsonl(name):
    """Load results/<name> as a list of records, or None when it is absent."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _need(records, name, builder):
    """Skip the test with an actionable message when an artifact is missing."""
    if records is None:
        raise unittest.SkipTest("results/%s is missing; run `%s` first" % (name, builder))
    return records


class ReconInvariants(unittest.TestCase):
    """The turn reconstruction: 10,845 turns and a 334.7M est.-token bill."""

    @classmethod
    def setUpClass(cls):
        cls.recon = _jsonl("recon.jsonl")

    def setUp(self):
        self.r = _need(self.recon, "recon.jsonl", "python -m router.recon")

    def test_one_row_per_trajectory(self):
        self.assertEqual(len(self.r), 1000)
        self.assertEqual(sorted(x["idx"] for x in self.r), list(range(1000)))

    def test_turn_total(self):
        self.assertEqual(sum(x["n_turns"] for x in self.r), 10_845)

    def test_every_line_has_at_least_one_turn(self):
        self.assertEqual(min(x["n_turns"] for x in self.r), 1)

    def test_gross_estimated_tokens(self):
        self.assertEqual(sum(x["gross_tok"] for x in self.r), 334_729_910)

    def test_gross_excluding_the_tools_block(self):
        # The tools array is re-sent on EVERY turn: 44,720,796 est. tok, 13.4%.
        tools = sum(x["tools_tok"] * x["n_turns"] for x in self.r)
        self.assertEqual(tools, 44_720_796)
        self.assertEqual(sum(x["gross_tok"] for x in self.r) - tools, 290_009_114)

    def test_cache_split(self):
        read = sum(x["cache_read_tok"] for x in self.r)
        write = sum(x["cache_write_tok"] for x in self.r)
        self.assertEqual(read, 308_074_571)
        self.assertEqual(write, 26_655_339)
        self.assertEqual(read + write, 334_729_910)
        self.assertAlmostEqual(100.0 * read / (read + write), 92.0, places=1)
        self.assertAlmostEqual(100.0 * write / (read + write), 8.0, places=1)

    def test_cache_split_holds_line_by_line(self):
        for x in self.r:
            self.assertEqual(x["cache_read_tok"] + x["cache_write_tok"], x["gross_tok"],
                             "line %d" % x["idx"])

    def test_naive_total_reproduces_the_starter_kit(self):
        self.assertEqual(sum(x["naive_tok"] for x in self.r), 22_631_879)

    def test_reconstruction_is_the_headline_correction(self):
        gross = sum(x["gross_tok"] for x in self.r)
        naive = sum(x["naive_tok"] for x in self.r)
        self.assertAlmostEqual(gross / naive, 14.79, places=2)

    def test_pre_treatment_prefix_never_exceeds_the_smallest_billed_prefix(self):
        for x in self.r:
            self.assertLessEqual(x["pre_tok"], min(x["prefix_tokens"]), "line %d" % x["idx"])

    def test_family_purity(self):
        fam = collections.Counter(x["family"] for x in self.r)
        self.assertEqual(fam["gpt"], 245)
        self.assertEqual(fam["claude"], 755)
        self.assertEqual(sum(fam.values()), 1000)

    def test_family_matches_the_model_id_prefix(self):
        # Perfect separation, 0 exceptions: the tools block and the arm id agree.
        for x in self.r:
            self.assertEqual(x["family"], "gpt" if x["model"].startswith("gpt") else "claude",
                             "line %d" % x["idx"])

    def test_turns_per_line_by_arm(self):
        by_arm = collections.defaultdict(list)
        for x in self.r:
            by_arm[x["model"]].append(x["n_turns"])
        expected = {
            "claude-opus-5": (331, 10.36), "claude-sonnet-5": (281, 11.83),
            "gpt-5.6-sol": (112, 10.85), "gpt-5.6-terra": (113, 8.51),
            "claude-fable-5": (71, 11.72), "claude-opus-4-8": (69, 10.29),
            "gpt-5.6-luna": (20, 17.30),
        }
        for arm, (n, mean) in expected.items():
            self.assertEqual(len(by_arm[arm]), n, arm)
            self.assertAlmostEqual(sum(by_arm[arm]) / n, mean, places=2, msg=arm)

    def test_spend_concentration(self):
        gross = sorted((x["gross_tok"] for x in self.r), reverse=True)
        total = sum(gross)
        for k, share in ((10, 17.7), (50, 40.5), (100, 54.7), (500, 90.2)):
            self.assertAlmostEqual(100.0 * sum(gross[:k]) / total, share, places=1,
                                   msg="top %d" % k)


class LabelInvariants(unittest.TestCase):
    """Outcome labels: 10,422 tool outputs, 474 errors, 259 frictional trajectories."""

    @classmethod
    def setUpClass(cls):
        cls.labels = _jsonl("labels.jsonl")

    def setUp(self):
        self.l = _need(self.labels, "labels.jsonl", "python -m router.labels")

    def test_one_row_per_trajectory(self):
        self.assertEqual(len(self.l), 1000)
        self.assertEqual(sorted(x["idx"] for x in self.l), list(range(1000)))

    def test_tool_output_totals(self):
        resolvable = sum(x["n_obs"] for x in self.l)
        unknown = sum(x["n_unknown"] for x in self.l)
        # See this module's docstring: 9,635, not the brief's 9,633.
        self.assertEqual(resolvable, 9_635)
        self.assertEqual(unknown, 787)
        self.assertEqual(resolvable + unknown, 10_422)
        self.assertAlmostEqual(100.0 * resolvable / 10_422, 92.4, places=1)

    def test_error_count(self):
        self.assertEqual(sum(x["n_err"] for x in self.l), 474)

    def test_killed_calls(self):
        self.assertEqual(sum(x["n_kill"] for x in self.l), 84)

    def test_friction_positives(self):
        self.assertEqual(sum(x["y_fric"] for x in self.l), 259)
        self.assertTrue(all(x["y_fric"] in (0, 1) for x in self.l))

    def test_y_proc_distribution(self):
        counts = collections.Counter(x["y_proc"] for x in self.l)
        self.assertEqual(counts[0], 737)
        self.assertEqual(counts[1], 241)
        self.assertEqual(counts[2], 18)
        self.assertEqual(counts[None], 4)
        self.assertEqual(sum(counts.values()), 1000)

    def test_errors_never_exceed_resolvable_outputs(self):
        for x in self.l:
            self.assertLessEqual(x["n_err"], x["n_obs"], "line %d" % x["idx"])


class JobKeyInvariants(unittest.TestCase):
    """The job key is the LITERAL cron path; a PII_ placeholder is never a join key."""

    @classmethod
    def setUpClass(cls):
        cls.jk = _jsonl("jobkey.jsonl")
        cls.recon = _jsonl("recon.jsonl")

    def setUp(self):
        self.j = _need(self.jk, "jobkey.jsonl", "python -m router.jobkey")

    def test_runs_carrying_a_cron_path(self):
        self.assertEqual(sum(1 for x in self.j if x["cron_path"]), 784)
        self.assertEqual(len({x["cron_path"] for x in self.j if x["cron_path"]}), 354)

    def test_bucket_split(self):
        buckets = collections.Counter(x["bucket"] for x in self.j)
        self.assertEqual(buckets["literal"], 374)
        self.assertEqual(buckets["pii"], 410)
        self.assertEqual(buckets["none"], 216)
        self.assertEqual(sum(buckets.values()), 1000)

    def test_literal_flag_agrees_with_the_bucket(self):
        for x in self.j:
            self.assertEqual(bool(x["literal"]), x["bucket"] == "literal", "idx %d" % x["idx"])

    def test_no_pii_placeholder_is_ever_marked_literal(self):
        # A PII_ token is renumbered per request, so it cannot key two rows together.
        for x in self.j:
            if x["literal"]:
                self.assertNotIn("PII_", x["cron_path"] or "", "idx %d" % x["idx"])

    def test_clean_jobs(self):
        clean = collections.defaultdict(list)
        for x in self.j:
            if x["literal"]:
                clean[x["cron_path"]].append(x["idx"])
        self.assertEqual(len(clean), 152)
        self.assertEqual(sum(len(v) for v in clean.values()), 374)

    def test_multi_arm_and_cross_family_jobs(self):
        recon = _need(self.recon, "recon.jsonl", "python -m router.recon")
        arm = {x["idx"]: x["model"] for x in recon}
        fam = {x["idx"]: x["family"] for x in recon}
        clean = collections.defaultdict(list)
        for x in self.j:
            if x["literal"]:
                clean[x["cron_path"]].append(x["idx"])
        multi = {p: v for p, v in clean.items() if len({arm[i] for i in v}) >= 2}
        cross = {p: v for p, v in clean.items() if len({fam[i] for i in v}) == 2}
        self.assertEqual(len(multi), 28)
        self.assertEqual(sum(len(v) for v in multi.values()), 245)
        self.assertEqual(len(cross), 23)
        self.assertEqual(sum(len(v) for v in cross.values()), 229)

    def test_the_pooled_cross_family_contrast_is_mostly_one_job(self):
        recon = _need(self.recon, "recon.jsonl", "python -m router.recon")
        fam = {x["idx"]: x["family"] for x in recon}
        clean = collections.defaultdict(list)
        for x in self.j:
            if x["literal"]:
                clean[x["cron_path"]].append(x["idx"])
        cross = {p: v for p, v in clean.items() if len({fam[i] for i in v}) == 2}
        biggest = max(cross.items(), key=lambda kv: len(kv[1]))
        self.assertEqual(biggest[0], "crons/heartbeat/")
        self.assertEqual(len(biggest[1]), 134)


class PolicyInvariants(unittest.TestCase):
    """The routable set: 675 trajectories carrying 59.0% of the estimated bill."""

    @classmethod
    def setUpClass(cls):
        cls.recon = _jsonl("recon.jsonl")
        cls.routes = _jsonl("routes.jsonl")

    def test_routable_set(self):
        recon = _need(self.recon, "recon.jsonl", "python -m router.recon")
        import numpy as np                                              # noqa: F401
        from router import policy

        features = os.path.join(RESULTS, "features.npz")
        if not os.path.exists(features):
            raise unittest.SkipTest(
                "results/features.npz is missing; run `python -m router.features`")
        z = np.load(features, allow_pickle=False)
        cols = [str(c) for c in z["cols"]]
        img = {int(i): int(v) for i, v in zip(z["idx"], z["X"][:, cols.index("img_pre")])}

        counts = policy.arm_counts(recon)
        mask = policy.routable_mask(recon, img, counts)
        self.assertEqual(int(mask.sum()), 675)
        gross = sum(r["gross_tok"] for r, keep in zip(recon, mask) if keep)
        self.assertEqual(gross, 197_551_096)
        self.assertAlmostEqual(100.0 * gross / 334_729_910, 59.0, places=1)

    def test_the_nine_pre_treatment_image_rows(self):
        import numpy as np

        features = os.path.join(RESULTS, "features.npz")
        if not os.path.exists(features):
            raise unittest.SkipTest(
                "results/features.npz is missing; run `python -m router.features`")
        z = np.load(features, allow_pickle=False)
        cols = [str(c) for c in z["cols"]]
        img = z["X"][:, cols.index("img_pre")]
        hit = sorted(int(i) for i, v in zip(z["idx"], img) if v > 0)
        self.assertEqual(hit, [32, 62, 82, 274, 378, 412, 584, 711, 796])

    def test_routes_artifact_matches_the_contract(self):
        rows = _need(self.routes, "routes.jsonl", "python -m router.policy")
        self.assertEqual(len(rows), 1000)
        required = {"idx", "logged", "route", "changed", "p_fric", "gates",
                    "cost_logged_usd", "cost_routed_usd", "supported"}
        for row in rows[:5] + rows[-5:]:
            self.assertTrue(required.issubset(row), "missing fields on idx %d" % row["idx"])
            self.assertIsInstance(row["route"], str, "route must be a scalar arm id")
            self.assertEqual(row["changed"], row["route"] != row["logged"])

    def test_reroutes_and_their_support(self):
        rows = _need(self.routes, "routes.jsonl", "python -m router.policy")
        changed = [r for r in rows if r["changed"]]
        self.assertEqual(len(changed), 119)
        self.assertEqual(sum(1 for r in changed if r["supported"]), 6)

    def test_every_reroute_is_in_lane_and_cheaper(self):
        rows = _need(self.routes, "routes.jsonl", "python -m router.policy")
        from router.pricing import ALL_SHEETS, rate_in

        for r in rows:
            if not r["changed"]:
                continue
            self.assertEqual(r["logged"].split("-")[0], r["route"].split("-")[0],
                             "cross-family reroute on idx %d" % r["idx"])
            for sheet in ALL_SHEETS:
                self.assertLess(rate_in(r["route"], sheet), rate_in(r["logged"], sheet),
                                "idx %d not cheaper under %s" % (r["idx"], sheet))


class CostInvariants(unittest.TestCase):
    """The cache-aware price model: multipliers, and monotonicity in rho."""

    @classmethod
    def setUpClass(cls):
        cls.recon = _jsonl("recon.jsonl")

    def setUp(self):
        self.r = _need(self.recon, "recon.jsonl", "python -m router.recon")

    def test_effective_multiplier_sweep(self):
        from router import costs

        sweep = costs.rho_sweep(self.r)
        for rho, expected in ((1.00, 0.192), (0.90, 0.272), (0.83, 0.329), (0.55, 0.555)):
            self.assertAlmostEqual(sweep[rho], expected, places=3, msg="rho=%.2f" % rho)

    def test_cost_is_monotone_in_rho(self):
        # rho is the share of the cacheable prefix that actually hits cache, so a
        # higher rho can never cost more. Strictly decreasing on this corpus.
        from router import costs

        grid = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.83, 0.9, 0.95, 1.0]
        totals = [sum(costs.effective_tokens(rec, rho=rho) for rec in self.r) for rho in grid]
        for lo, hi in zip(totals, totals[1:]):
            self.assertGreater(lo, hi, "cost must strictly decrease as rho rises")
        self.assertAlmostEqual(totals[0], 334_729_910.0, places=0,
                               msg="rho=0 must reduce to the fully-uncached bill")

    def test_dollar_figures_require_a_named_sheet(self):
        from router.pricing import UnnamedSheetError, format_usd

        with self.assertRaises(UnnamedSheetError):
            format_usd(1.23, {"claude-opus-5": {"in": 1.0, "out": 1.0}})

    def test_the_sheet_spread_is_the_uncertainty(self):
        from router import costs

        bills = {s: costs.total_cost(self.r, None, s) for s in
                 ("assumed_default", "alt_compressed", "alt_generational")}
        self.assertAlmostEqual(bills["assumed_default"], 412.95, places=1)
        self.assertAlmostEqual(bills["alt_compressed"], 231.89, places=1)
        self.assertAlmostEqual(bills["alt_generational"], 396.26, places=1)
        # ~1.8x apart: quote the spread, never one figure.
        self.assertGreater(max(bills.values()) / min(bills.values()), 1.5)


class SeededResamplingInvariants(unittest.TestCase):
    """A fixed seed reproduces; a different seed moves only the interval, not the point."""

    def _runs(self):
        for name, builder in (("jobkey.jsonl", "router.jobkey"),
                              ("labels.jsonl", "router.labels"),
                              ("recon.jsonl", "router.recon")):
            if not os.path.exists(os.path.join(RESULTS, name)):
                raise unittest.SkipTest("results/%s is missing; run `python -m %s`"
                                        % (name, builder))
        from router import family_contrast

        return family_contrast, family_contrast.build_runs()

    def test_fixed_seed_bootstrap_reproduces(self):
        fc, runs = self._runs()
        a = fc.pooled_contrast(runs, n_boot=300, seed=12345)[:3]
        b = fc.pooled_contrast(runs, n_boot=300, seed=12345)[:3]
        self.assertEqual(a, b, "a fixed-seed bootstrap must reproduce bit for bit")

    def test_the_point_estimate_does_not_depend_on_the_seed(self):
        fc, runs = self._runs()
        a = fc.pooled_contrast(runs, n_boot=300, seed=12345)
        b = fc.pooled_contrast(runs, n_boot=300, seed=999)
        self.assertAlmostEqual(a[0], b[0], places=12,
                               msg="only the interval may move with the seed")

    def test_the_cross_family_gap_is_not_established(self):
        fc, runs = self._runs()
        est, lo, hi = fc.pooled_contrast(runs, n_boot=300, seed=12345)[:3]
        # 21/72 - 33/157 = +8.15pp, and the interval straddles nothing useful:
        # 134 of the 229 runs are one job. Report bounds, never a point.
        self.assertAlmostEqual(100.0 * est, 8.15, places=2)
        self.assertLess(lo, est)
        self.assertGreater(hi, est)
        self.assertGreater(100.0 * hi, 18.0, "the data cannot exclude a large gpt penalty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
