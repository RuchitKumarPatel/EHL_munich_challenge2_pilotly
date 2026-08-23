"""Invariants for loop/backlog.py::plan_turn — the turn scheduler.

WHAT THIS CHECKS
    The scheduler is a pure function of (queue, turn, deck_every, review_every),
    which makes it cheap to test exhaustively over a whole simulated night. Two
    properties matter enough to pin:

      * REVIEW IS NEVER STARVED. The original implementation returned `deck`
        before it tested the review cadence, so on any turn matching both
        cadences the deck won and review was pushed to its next multiple --
        which, whenever deck_every divides review_every, is also a deck turn.
        Measured over turns 1..40 on the old code: (5,4) fired review, but
        (5,10), (4,4), (5,5) and (3,6) never fired it ONCE. The shipped config
        was (5,4), so the loop's periodic security pass was working by luck.
        A stale deck costs one turn; a skipped review costs the night, so the
        tie now goes to review.

      * BOTH CADENCES DISABLE THE SAME WAY. review_every had a `> 0` guard and
        deck_every did not, so deck_every=0 raised ZeroDivisionError inside a
        command substitution the supervisor does not exit-check: run.sh got an
        empty TYPE and logged "unknown turn type", burying the real cause.

WHAT IT WRITES
    Nothing. plan_turn is pure; every case here passes its queue in by value and
    never touches loop/state/.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_backlog_module():
    """Import loop/backlog.py by path -- `loop` is a directory, not a package."""
    path = os.path.join(REPO_ROOT, "loop", "backlog.py")
    spec = importlib.util.spec_from_file_location("loop_backlog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backlog = _load_backlog_module()
plan_turn = backlog.plan_turn

# The cadence pairs the review turn measured as starved, plus the shipped one.
# (deck_every, review_every).
CADENCE_PAIRS = ((5, 4), (5, 10), (4, 4), (5, 5), (3, 6))

# A queue that is never dry: without it the planner would fall through to
# `council` and the cadence question would not even be asked.
BUSY_QUEUE = [
    {"id": "i0001", "status": "accepted"},
    {"id": "i0002", "status": "proposed"},
    {"id": "i0003", "status": "implemented"},
]


class ReviewIsNeverStarved(unittest.TestCase):
    """The periodic security pass must fire on a schedule, not by luck."""

    def test_review_fires_at_least_once_per_two_review_periods(self):
        # Two full review periods is the loosest bound that still detects
        # starvation: one period could legitimately be lost to a tie, never two.
        for deck_every, review_every in CADENCE_PAIRS:
            with self.subTest(deck_every=deck_every, review_every=review_every):
                window = range(1, review_every * 2 + 1)
                types = [plan_turn(BUSY_QUEUE, t, deck_every, review_every) for t in window]
                self.assertIn(
                    "review", types,
                    "review never fires in turns 1..%d for deck_every=%d review_every=%d; "
                    "got %r" % (review_every * 2, deck_every, review_every, types),
                )

    def test_review_fires_on_every_multiple_over_a_full_night(self):
        # LOOP_MAX_TURNS is 40. Over a whole night every review multiple should
        # be a review turn -- a tie must not silently move it.
        for deck_every, review_every in CADENCE_PAIRS:
            with self.subTest(deck_every=deck_every, review_every=review_every):
                for turn in range(1, 41):
                    if turn % review_every == 0:
                        self.assertEqual(
                            "review", plan_turn(BUSY_QUEUE, turn, deck_every, review_every),
                            "turn %d is a review multiple (review_every=%d) but planned as "
                            "something else with deck_every=%d" % (turn, review_every, deck_every),
                        )

    def test_deck_still_fires_on_its_own_multiples(self):
        # Letting review win the tie must not cost the deck its own cadence:
        # every deck multiple that is NOT also a review multiple stays a deck.
        for deck_every, review_every in CADENCE_PAIRS:
            with self.subTest(deck_every=deck_every, review_every=review_every):
                for turn in range(1, 41):
                    if turn % deck_every == 0 and turn % review_every != 0:
                        self.assertEqual(
                            "deck", plan_turn(BUSY_QUEUE, turn, deck_every, review_every),
                            "turn %d is a deck multiple (deck_every=%d) but planned as "
                            "something else" % (turn, deck_every),
                        )

    def test_the_deck_is_deferred_by_a_lost_tie_not_dropped(self):
        # Swapping the two checks and stopping there would just move the
        # starvation onto the deck: with deck_every == review_every EVERY deck
        # multiple is a tie, so the deck would never fire once. Measured on the
        # naive swap: (4,4) and (5,5) produced 0 deck turns over turns 1..40.
        for deck_every, review_every in CADENCE_PAIRS:
            with self.subTest(deck_every=deck_every, review_every=review_every):
                types = [plan_turn(BUSY_QUEUE, t, deck_every, review_every)
                         for t in range(1, 41)]
                self.assertGreater(types.count("deck"), 0,
                                   "deck never fires over turns 1..40")

    def test_a_deck_that_loses_a_tie_runs_on_the_very_next_turn(self):
        # The deferral is what makes the bound below hold. deck_every == 4 ==
        # review_every: turn 4 is a tie, so review takes it and the deck runs at 5.
        self.assertEqual("review", plan_turn(BUSY_QUEUE, 4, 4, 4))
        self.assertEqual("deck", plan_turn(BUSY_QUEUE, 5, 4, 4))

    def test_deck_staleness_is_bounded_by_deck_every_plus_one(self):
        # The cost of giving review the tie, stated as a bound rather than left
        # to be discovered. Holds for every (deck_every, review_every) pair in
        # 1..12 x 2..12; review_every=1 is excluded because "review every turn"
        # means the operator has asked for nothing else to run.
        for deck_every in range(1, 13):
            for review_every in range(2, 13):
                with self.subTest(deck_every=deck_every, review_every=review_every):
                    last_deck = 0
                    for turn in range(1, 41):
                        if plan_turn(BUSY_QUEUE, turn, deck_every, review_every) == "deck":
                            last_deck = turn
                        self.assertLessEqual(
                            turn - last_deck, deck_every + 1,
                            "deck went %d turns stale at turn %d (deck_every=%d, "
                            "review_every=%d)" % (turn - last_deck, turn, deck_every,
                                                  review_every),
                        )

    def test_no_review_multiple_is_missed_anywhere_in_the_grid(self):
        # The starvation bug in one line, generalised past the five pairs the
        # review turn happened to try.
        for deck_every in range(1, 13):
            for review_every in range(1, 13):
                for turn in range(1, 41):
                    if turn % review_every == 0:
                        self.assertEqual(
                            "review", plan_turn(BUSY_QUEUE, turn, deck_every, review_every),
                            "missed review at turn %d for (%d, %d)"
                            % (turn, deck_every, review_every))


class ZeroDisablesACadence(unittest.TestCase):
    """0 must mean "off" for both cadences, not a traceback for one of them."""

    def test_deck_every_zero_disables_rather_than_raises(self):
        for turn in range(1, 41):
            planned = plan_turn(BUSY_QUEUE, turn, 0, 4)
            self.assertNotEqual("deck", planned,
                                "deck_every=0 must disable the deck cadence")

    def test_review_every_zero_disables(self):
        for turn in range(1, 41):
            self.assertNotEqual("review", plan_turn(BUSY_QUEUE, turn, 5, 0),
                                "review_every=0 must disable the review cadence")

    def test_both_zero_falls_through_to_the_queue(self):
        # With both cadences off the planner is queue-driven only.
        self.assertEqual("merge", plan_turn(BUSY_QUEUE, 20, 0, 0))

    def test_negative_is_treated_as_disabled_not_as_a_modulus(self):
        # A typo'd -1 in config.env must not schedule every single turn.
        for turn in range(1, 41):
            self.assertNotEqual("deck", plan_turn(BUSY_QUEUE, turn, -1, 4))
            self.assertNotEqual("review", plan_turn(BUSY_QUEUE, turn, 5, -1))


class QueueOrderIsUnchanged(unittest.TestCase):
    """The cadence fix must not disturb the queue precedence below it."""

    def test_merge_outranks_implement_outranks_evaluate(self):
        self.assertEqual("merge", plan_turn(
            [{"status": "implemented"}, {"status": "accepted"}, {"status": "proposed"}], 1, 5, 4))
        self.assertEqual("implement", plan_turn(
            [{"status": "accepted"}, {"status": "proposed"}], 1, 5, 4))
        self.assertEqual("evaluate", plan_turn([{"status": "proposed"}], 1, 5, 4))

    def test_a_dry_queue_calls_a_council(self):
        self.assertEqual("council", plan_turn([{"status": "merged"}], 1, 5, 4))
        self.assertEqual("council", plan_turn([], 1, 5, 4))

    def test_turn_zero_is_never_a_cadence_turn(self):
        # Turn 0 is "before the first turn"; 0 % anything == 0 would otherwise
        # make it both a deck and a review turn.
        self.assertEqual("merge", plan_turn(BUSY_QUEUE, 0, 5, 4))


if __name__ == "__main__":
    unittest.main()
