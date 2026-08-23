"""Invariants for router.tooling — the tool-block interventions.

WHAT THIS CHECKS
    The savings this module reports are differences against `router.recon`, so
    the first and most important test is that the do-nothing policy reproduces
    recon EXACTLY. If the baseline drifts, every delta downstream is measured
    against the wrong ruler and the sign of a saving is not trustworthy.

    That check caught a real bug: `tok(tools_array)` is not `sum(tok(tool))`
    (json.dumps adds brackets and commas, and tok() floor-divides once per item
    rather than once per array), so the naive reconstruction was 105,101 est.
    tokens light corpus-wide. `frame` carries that difference.

    Beyond the baseline, four behavioural invariants:
      * a tighter description cap never costs more (monotonicity);
      * the oracle policy misses nothing, by definition;
      * job history is leak-free — a run's own tools never appear in its own
        history, only tools from strictly earlier runs of the same cron_path;
      * a deferred tool is absent from the prefix before its first call and
        present from that turn onward.

WHAT IT WRITES
    Nothing. Read-only over results/ and export/. Skips with a message naming
    the module that builds a missing artifact, rather than failing for the
    wrong reason.

CAVEAT
    All token counts here are ESTIMATES: tok(x) = len(json.dumps(x)) // 4.
"""

from __future__ import annotations

import json
import os
import unittest

from router import tooling

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
EXPORT = os.path.join(REPO, "export")


def _have(*names: str) -> bool:
    return all(os.path.exists(os.path.join(RESULTS, n)) for n in names)


class ToolingBaseline(unittest.TestCase):
    """The do-nothing policy must reproduce recon exactly, or every delta is wrong."""

    @classmethod
    def setUpClass(cls):
        if not _have("recon.jsonl"):
            raise unittest.SkipTest("results/recon.jsonl missing — run `make recon`")
        if not os.path.isdir(EXPORT):
            raise unittest.SkipTest("export/ missing — see skills/setup")
        cls.recon = tooling.load_recon()
        cls.records = tooling.load_tool_records(EXPORT)
        cls.keep_all = staticmethod(tooling.flat(lambda rec: set(rec["sizes"])))

    def test_all_tools_policy_reproduces_recon_gross(self):
        """Sum of prefixes under 'change nothing' == recon's gross_tok, to the token."""
        got = tooling.gross_tokens(self.records, self.recon, self.keep_all)
        want = sum(r["gross_tok"] for r in self.recon.values())
        self.assertEqual(want, got)

    def test_all_tools_policy_reproduces_recon_cost(self):
        """And the same for the cache-weighted bill built from recon's own split."""
        got = tooling.bill(self.records, self.recon, self.keep_all)
        want = (
            tooling.CACHE_READ_MULT * sum(r["cache_read_tok"] for r in self.recon.values())
            + tooling.CACHE_WRITE_MULT * sum(r["cache_write_tok"] for r in self.recon.values())
        )
        self.assertAlmostEqual(want, got, places=6)

    def test_frame_accounts_for_the_json_envelope(self):
        """sizes + frame == recon's tools_tok on every trajectory, and frame is never negative."""
        for rec in self.records:
            with self.subTest(idx=rec["idx"]):
                self.assertEqual(
                    self.recon[rec["idx"]]["tools_tok"],
                    sum(rec["sizes"].values()) + rec["frame"],
                )
                self.assertGreaterEqual(rec["frame"], 0)

    def test_tool_block_share_matches_the_contract(self):
        """docs/CONTRACTS.md: tool block re-billed across turns = 44,720,796."""
        total = sum(
            (sum(r["sizes"].values()) + r["frame"]) * self.recon[r["idx"]]["n_turns"]
            for r in self.records
        )
        self.assertEqual(44_720_796, total)


class ToolingBehaviour(unittest.TestCase):
    """Behavioural invariants that hold whatever the corpus happens to contain."""

    @classmethod
    def setUpClass(cls):
        if not _have("recon.jsonl"):
            raise unittest.SkipTest("results/recon.jsonl missing — run `make recon`")
        if not os.path.isdir(EXPORT):
            raise unittest.SkipTest("export/ missing — see skills/setup")
        cls.recon = tooling.load_recon()
        cls.records = tooling.load_tool_records(EXPORT)

    def test_tighter_cap_never_costs_more(self):
        """Truncation is monotone: a smaller budget cannot produce a larger bill."""
        def keep_all(rec):
            return set(rec["sizes"])

        costs = [
            tooling.bill(self.records, self.recon, tooling.flat(keep_all, cap))
            for cap in (1200, 800, 600, 400, 300, 200, 150)
        ]
        for tighter, looser in zip(costs[1:], costs[:-1]):
            self.assertLessEqual(tighter, looser)

    def test_truncation_never_drops_a_tool(self):
        """Capping changes size, never membership — miss rate is 0 by construction."""
        def keep_all(rec):
            return set(rec["sizes"])

        n_traj, n_pairs = tooling.misses(self.records, keep_all)
        self.assertEqual(0, n_traj)
        self.assertEqual(0, n_pairs)

    def test_oracle_misses_nothing(self):
        """Keeping exactly what was called cannot be missing anything."""
        n_traj, _ = tooling.misses(self.records, lambda rec: rec["called"])
        self.assertEqual(0, n_traj)

    def test_omitting_is_never_free_of_risk_on_this_corpus(self):
        """The leak-free omit policy does miss tools — the trade-off is real, not hypothetical."""
        if not _have("jobkey.jsonl"):
            self.skipTest("results/jobkey.jsonl missing — run `make jobkey`")
        history = tooling.prior_run_tools(self.records, tooling.load_jobkeys())
        n_traj, _ = tooling.misses(
            self.records,
            lambda rec: history.get(rec["idx"], set(rec["sizes"])) & set(rec["sizes"]),
        )
        self.assertGreater(n_traj, 0)

    def test_job_history_is_leak_free(self):
        """A run's history contains only tools seen in STRICTLY earlier runs of that job."""
        if not _have("jobkey.jsonl"):
            self.skipTest("results/jobkey.jsonl missing — run `make jobkey`")
        jobkeys = tooling.load_jobkeys()
        history = tooling.prior_run_tools(self.records, jobkeys)
        called = {rec["idx"]: rec["called"] for rec in self.records}

        by_job: dict[str, list[int]] = {}
        for rec in self.records:
            key = jobkeys.get(rec["idx"])
            if key and key.get("literal"):
                by_job.setdefault(key["cron_path"], []).append(rec["idx"])

        for path, indices in by_job.items():
            indices = sorted(indices)
            for position, idx in enumerate(indices):
                earlier = set().union(*(called[j] for j in indices[:position])) if position else set()
                with self.subTest(job=path, idx=idx):
                    self.assertEqual(earlier, history[idx])
        # the first run of every job starts with nothing
        for indices in by_job.values():
            self.assertEqual(set(), history[min(indices)])

    def test_deferred_tool_enters_the_prefix_at_its_first_call(self):
        """Intervention C: absent before the first call, present from that turn onward."""
        target = None
        for rec in self.records:
            extras = {n for n in rec["called"] if rec["first_turn"].get(n, 0) > 0}
            if extras and self.recon[rec["idx"]]["n_turns"] > 2:
                target = (rec, min(extras))
                break
        if target is None:
            self.skipTest("no trajectory calls a tool after its first turn")
        rec, name = target
        at_turn = tooling.deferred(lambda _r: set(), search_tok=0)
        first = rec["first_turn"][name]
        before = at_turn(rec, first - 1)
        after = at_turn(rec, first)
        self.assertLess(before, after)
        self.assertGreaterEqual(after - before, rec["sizes"][name])


class ToolingArtifact(unittest.TestCase):
    """results/tooling.json, once router.tooling has written it."""

    @classmethod
    def setUpClass(cls):
        if not _have("tooling.json"):
            raise unittest.SkipTest("results/tooling.json missing — run `make tooling`")
        with open(os.path.join(RESULTS, "tooling.json"), encoding="utf-8") as handle:
            cls.art = json.load(handle)

    def test_every_row_carries_its_miss_accounting(self):
        keys = {"policy", "cost_units", "cost_delta", "gross_tok", "gross_delta",
                "miss_trajectories", "miss_rate", "miss_pairs"}
        rows = (self.art["omit"] + self.art["truncate"] + self.art["defer"]
                + self.art["combined"] + [self.art["oracle"]])
        for row in rows:
            with self.subTest(policy=row.get("policy")):
                self.assertTrue(keys <= set(row))

    def test_the_leak_warning_survives_serialisation(self):
        """The upper-bound caveat must travel with the numbers, not live only in a docstring."""
        self.assertIn("UPPER BOUND", self.art["leak_warning"])
        self.assertIn("ESTIMATED", self.art["basis"])

    def test_truncation_rows_report_no_misses(self):
        for row in self.art["truncate"]:
            with self.subTest(policy=row["policy"]):
                self.assertEqual(0, row["miss_trajectories"])


if __name__ == "__main__":
    unittest.main()
