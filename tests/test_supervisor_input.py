"""The supervisor must treat loop/state/state.json as untrusted input.

WHY THIS FILE EXISTS
    `loop/run.sh` is the one process in this project that runs OUTSIDE every
    tool-permission boundary: it is the thing that launches the sandboxed turn,
    so nothing sandboxes it. Its budget check used to build Python source text
    by interpolation --

        "$PY" -c "import sys;sys.exit(0 if float('$spent') >= ...)"

    -- where `$spent` is the `cost_usd` field printed straight out of
    loop/state/state.json. That file is rewritten by every turn agent, which
    runs with bypassPermissions. So a turn that wrote a crafted string into its
    own cost field would have that string executed as Python by the supervisor,
    at the top of every subsequent loop iteration, forever. Not a hypothetical
    reachability argument: `stop_reason` is the first thing the loop body calls.

WHAT IS TESTED
    1. A payload in `cost_usd` does not execute. The same payload is first run
       through the OLD interpolated form to prove it is live -- without that
       control, "no side effect" would also be true of an inert string.
    2. The loop stops CLEANLY on a state file it cannot read a number out of,
       rather than sailing past a guard that errored. Fail-closed: an unreadable
       cost field means the supervisor cannot know what it has spent.
    3. The ordinary paths still work -- over the ceiling stops, under it does
       not, and a turn counter arriving as a JSON string still trips max-turns
       instead of erroring out of the `[ -ge ]` comparison.
    4. A textual invariant over loop/run.sh: no `python -c` argument is written
       in double quotes anywhere in the file. That is the property that made the
       bug possible, and it is the only form of this test that keeps holding
       when someone adds a seventh helper next week.

HOW THE PAYLOAD STAYS SAFE
    It writes one sentinel file into the test's own TemporaryDirectory and does
    nothing else. It contains no `$`, no backtick and no double quote, because
    the old form interpolated it inside a double-quoted bash string and the
    point is to reproduce that path exactly.

RUNTIME
    ~2 s. Sources loop/run.sh with LOOP_RUN_SOURCED=1, which returns before the
    driver loop, and drives `stop_reason` against a state file in a temp dir.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SH = os.path.join(REPO_ROOT, "loop", "run.sh")


def _payload(sentinel: str) -> str:
    """A cost_usd value that executes iff it is interpolated into Python source.

    Assembled rather than written out so the shape is legible: it closes the
    `float('` the old code opened, runs its own expression, and reopens one so
    the rest of the line still parses.
    """
    return ("0') or __import__('pathlib').Path('%s').write_text('x') or float('0"
            % sentinel)


def _statefile(tmp: str, **fields) -> str:
    path = os.path.join(tmp, "state.json")
    base = {"turn": 3, "cost_usd": 1.0, "last_turn_type": "implement"}
    base.update(fields)
    with open(path, "w") as fh:
        json.dump(base, fh)
    return path


def _stop_reason(statefile: str, tmp: str, *, budget="400", max_turns="40") -> tuple:
    """Source run.sh, point it at a throwaway state file, call stop_reason.

    Returns (rc, reason, stderr). LOOP_MAX_SEVEN_DAY_PCT is pushed past 100 so a
    real ~/.claude.json cannot decide the outcome of these tests.
    """
    script = """
set -uo pipefail
export LOOP_RUN_SOURCED=1
source %(run_sh)s
STATEFILE=%(statefile)s
STOPFILE=%(tmp)s/no-such-stopfile
LOOP_BUDGET_USD=%(budget)s
LOOP_MAX_TURNS=%(max_turns)s
LOOP_MAX_SEVEN_DAY_PCT=101
DEADLINE_EPOCH=$(( $(date +%%s) + 3600 ))
reason=$(stop_reason); rc=$?
printf 'RC=%%s\\n' "$rc"
printf 'REASON=%%s\\n' "$reason"
""" % {"run_sh": RUN_SH, "statefile": statefile, "tmp": tmp,
       "budget": budget, "max_turns": max_turns}
    proc = subprocess.run(["bash", "-c", script], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    rc, reason = None, ""
    for line in proc.stdout.splitlines():
        if line.startswith("RC="):
            rc = int(line[3:])
        elif line.startswith("REASON="):
            reason = line[7:]
    return rc, reason, proc.stderr


class StateFileIsUntrustedInput(unittest.TestCase):
    """A crafted cost_usd must be data to the supervisor, never source."""

    def test_the_payload_executes_under_the_old_interpolated_form(self):
        # NEGATIVE CONTROL. Without this the next test passes just as happily on
        # a payload that never did anything, and proves nothing about the fix.
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = os.path.join(tmp, "sentinel")
            old = ("import sys;sys.exit(0 if float('%s') >= float('400') else 1)"
                   % _payload(sentinel))
            subprocess.run([os.path.join(REPO_ROOT, ".venv", "bin", "python"),
                            "-c", old], capture_output=True)
            self.assertTrue(
                os.path.exists(sentinel),
                "the probe payload is inert, so test_the_payload_does_not_execute "
                "would prove nothing")

    def test_the_payload_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = os.path.join(tmp, "sentinel")
            state = _statefile(tmp, cost_usd=_payload(sentinel))
            rc, reason, stderr = _stop_reason(state, tmp)
            self.assertFalse(
                os.path.exists(sentinel),
                "cost_usd out of state.json reached a Python interpreter as source")
            self.assertEqual(rc, 0,
                             "an unreadable cost_usd must stop the loop, not be "
                             "skipped past (reason=%r, stderr=%r)" % (reason, stderr))
            self.assertIn("cost_usd", reason,
                          "the stop reason must name what it could not read")

    def test_a_garbage_cost_field_stops_the_loop_cleanly(self):
        # Same fail-closed contract for plain corruption rather than an attack:
        # a half-written state file is far likelier than a crafted one.
        with tempfile.TemporaryDirectory() as tmp:
            state = _statefile(tmp, cost_usd="not-a-number")
            rc, reason, stderr = _stop_reason(state, tmp)
            self.assertEqual(rc, 0, "reason=%r stderr=%r" % (reason, stderr))
            self.assertIn("cost_usd", reason)


class OrdinaryStoppingConditionsStillHold(unittest.TestCase):
    """The hardening must not cost the guards it is protecting."""

    def test_over_the_ceiling_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, reason, stderr = _stop_reason(_statefile(tmp, cost_usd=401.5), tmp)
            self.assertEqual(rc, 0, "reason=%r stderr=%r" % (reason, stderr))
            self.assertIn("effort ceiling", reason)

    def test_under_the_ceiling_does_not_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, reason, stderr = _stop_reason(_statefile(tmp, cost_usd=12.7), tmp)
            self.assertEqual(rc, 1,
                             "nothing should stop the loop here (reason=%r, "
                             "stderr=%r)" % (reason, stderr))

    def test_max_turns_survives_a_turn_counter_written_as_a_string(self):
        # `[ "$turn" -ge 40 ]` on a non-integer is a bash error, and a guard that
        # errors reads as "do not stop" -- the counter silently stops counting.
        with tempfile.TemporaryDirectory() as tmp:
            rc, reason, stderr = _stop_reason(_statefile(tmp, turn="41"), tmp)
            self.assertEqual(rc, 0, "reason=%r stderr=%r" % (reason, stderr))
            self.assertIn("max turns", reason)

    def test_a_non_numeric_turn_counter_raises_no_bash_error(self):
        # Matched on the "run.sh: line N:" prefix rather than on the message,
        # because bash localises the message and this machine is German.
        with tempfile.TemporaryDirectory() as tmp:
            rc, reason, stderr = _stop_reason(_statefile(tmp, turn="lots"), tmp)
            self.assertNotRegex(stderr, r"run\.sh: \S+ \d+:",
                                "stop_reason raised a shell diagnostic")
            self.assertEqual(rc, 1, "reason=%r stderr=%r" % (reason, stderr))


class NoHelperBuildsPythonSourceByInterpolation(unittest.TestCase):
    """The textual invariant, which is what keeps holding next week."""

    def test_no_python_c_argument_is_double_quoted(self):
        # A single-quoted -c argument cannot interpolate a shell variable, so
        # this one grep is the whole class of bug. Python string literals inside
        # those blocks are written with double quotes to make it possible.
        with open(RUN_SH) as fh:
            source = fh.read()
        offenders = [line.strip() for line in source.splitlines()
                     if re.search(r'-c\s+"', line)]
        self.assertEqual(
            offenders, [],
            "python -c arguments must be single-quoted and take their values "
            "from argv:\n  " + "\n  ".join(offenders))

    def test_no_python_block_reopens_shell_quoting_mid_argument(self):
        # A single-quoted block is inert -- `$x` inside it is three characters.
        # The one way back out is to close the quote early and splice, the
        # '...'"$x"'...' idiom, so the check is that every block's closing quote
        # ends the whole argument rather than being followed by more of it.
        with open(RUN_SH) as fh:
            source = fh.read()
        blocks = list(re.finditer(r"-c\s+'(.*?)'", source, re.S))
        self.assertGreaterEqual(len(blocks), 4, "expected the state readers here")
        for block in blocks:
            tail = source[block.end():block.end() + 1]
            self.assertIn(tail, (" ", "\n", ""),
                          "a python -c argument continues past its closing quote "
                          "(%r follows):\n%s" % (tail, block.group(1)))


if __name__ == "__main__":
    unittest.main()
