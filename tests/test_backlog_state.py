"""loop/state/*.json is untrusted input to the state store, and to the supervisor.

WHY THIS FILE EXISTS
    Every turn agent runs with bypassPermissions and rewrites both state files.
    `loop/backlog.py` used to read them with a bare `json.loads` and index
    `i["id"]` / `i["status"]` with no `.get`, so a half-written or hand-edited
    file left the CLI dying with a traceback -- and `loop/run.sh` captures the
    CLI's stdout by command substitution under `set -uo pipefail` (no `-e`),
    with no exit-code check. The traceback therefore became:

        TURN=""   TYPE=""   ->   "unknown turn type '' -- stopping"

    which is the wrong cause printed for the right failure. The night ends, the
    log blames the planner, and the corrupt file is never named. MEASURED on the
    pre-fix code: `turn --bump`, `plan` and `list` each exit 1 with EMPTY stdout
    on a truncated state file, on a truncated backlog file, and on a backlog
    item missing its `status` key.

    The fail-open that matters most is the tempting fix rather than the bug:
    making `load_state()` fall back to its defaults on a parse error would zero
    `cost_usd`, and `stop_reason`'s budget ceiling reads exactly that field. A
    corrupt state file would then buy the loop an unlimited budget. So an
    unreadable state file must be an ERROR, never a default, and it must never
    be rewritten from those defaults -- test_corrupt_state_is_not_overwritten.

WHAT IS TESTED
    1. Every read path exits 3 (the "state unreadable" code, distinct from
       argparse's 2) with nothing on stdout and the offending FILE named on
       stderr -- for a truncated file, a wrong top-level type, and an item
       missing or misspelling a required key.
    2. A corrupt state file is not silently replaced by defaults: the bytes on
       disk are unchanged after a failed `turn --bump`.
    3. `next_id` cannot return an id that already exists, whatever is in the
       file. It used to swallow a non-numeric id and restart at i0001.
    4. The healthy paths still work, so the guard is not just "always fail".
    5. The supervisor side: `backlog_scalar` in loop/run.sh returns non-zero and
       prints nothing on a broken CLI, and run.sh contains no raw
       `$("${BACKLOG[@]}" ...)` capture -- the textual invariant that keeps
       holding when someone adds a seventh call site.

HOW THE REAL STATE IS KEPT SAFE
    Nothing here touches loop/state/. Each test copies loop/backlog.py into a
    throwaway tree, so the module's ROOT-relative paths resolve inside the
    temporary directory and the running night's counters cannot be moved.

RUNTIME
    ~2 s.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
BACKLOG_PY = os.path.join(REPO_ROOT, "loop", "backlog.py")
RUN_SH = os.path.join(REPO_ROOT, "loop", "run.sh")

STATE_UNREADABLE = 3   # the exit code this module promises the supervisor

GOOD_STATE = {"turn": 3, "cost_usd": 12.5, "last_turn_type": "implement"}
GOOD_ITEM = {"id": "i0001", "title": "an idea", "kind": "code", "why": "because",
             "status": "proposed", "branch": None, "created_turn": 0, "notes": []}


class _Sandbox:
    """A throwaway copy of the state store, so ROOT-relative paths land in tmp."""

    def __init__(self, tmp: str, backlog, state):
        self.root = tmp
        self.state_dir = os.path.join(tmp, "loop", "state")
        os.makedirs(self.state_dir)
        shutil.copy2(BACKLOG_PY, os.path.join(tmp, "loop", "backlog.py"))
        self.cli = os.path.join(tmp, "loop", "backlog.py")
        self.backlog_path = os.path.join(self.state_dir, "backlog.json")
        self.state_path = os.path.join(self.state_dir, "state.json")
        self._write(self.backlog_path, backlog)
        self._write(self.state_path, state)

    @staticmethod
    def _write(path, payload):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload if isinstance(payload, str)
                     else json.dumps(payload, indent=2))

    def run(self, *args):
        return subprocess.run([PY, self.cli, *args], capture_output=True, text=True)


def _sandbox(tmp, backlog=None, state=None):
    return _Sandbox(tmp,
                    [GOOD_ITEM] if backlog is None else backlog,
                    GOOD_STATE if state is None else state)


class UnreadableStateIsAnErrorNotADefault(unittest.TestCase):
    """A file this CLI cannot parse must stop the caller, loudly and by name."""

    def _assert_refused(self, proc, *, names):
        self.assertEqual(proc.returncode, STATE_UNREADABLE,
                         f"expected exit {STATE_UNREADABLE}, got {proc.returncode}: "
                         f"{proc.stderr[-400:]}")
        self.assertEqual(proc.stdout.strip(), "",
                         "a refused read must print NOTHING to stdout -- the supervisor "
                         "captures stdout as a scalar and cannot tell empty from absent")
        self.assertNotIn("Traceback", proc.stderr,
                         "the cause must be a sentence, not a traceback")
        for name in names:
            self.assertIn(name, proc.stderr)

    def test_truncated_backlog_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog='[{"id": "i0001"')
            for args in (("list",), ("plan", "--deck-every", "5"), ("render",),
                         ("add", "--title", "t", "--why", "w")):
                self._assert_refused(sb.run(*args), names=["backlog.json"])

    def test_truncated_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, state='{"turn": 3, "cost_usd": 12.5')
            for args in (("turn", "--bump"), ("plan", "--deck-every", "5"),
                         ("cost", "--add", "1.0"), ("stats",)):
                self._assert_refused(sb.run(*args), names=["state.json"])

    def test_backlog_that_is_not_a_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog={"i0001": "an idea"})
            self._assert_refused(sb.run("list"), names=["backlog.json", "list"])

    def test_state_that_is_not_an_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, state=[1, 2, 3])
            self._assert_refused(sb.run("turn", "--bump"), names=["state.json"])

    def test_item_missing_status_is_named_by_id(self):
        # The exact shape the python-reviewer found: _by_status indexes i["status"].
        with tempfile.TemporaryDirectory() as tmp:
            item = {k: v for k, v in GOOD_ITEM.items() if k != "status"}
            sb = _sandbox(tmp, backlog=[item])
            self._assert_refused(sb.run("list"), names=["i0001", "status"])

    def test_item_with_an_unknown_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog=[dict(GOOD_ITEM, status="donezo")])
            self._assert_refused(sb.run("list"), names=["i0001", "donezo"])

    def test_item_that_is_not_an_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog=["i0001"])
            self._assert_refused(sb.run("list"), names=["backlog.json"])

    def test_item_missing_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = {k: v for k, v in GOOD_ITEM.items() if k != "id"}
            sb = _sandbox(tmp, backlog=[item])
            self._assert_refused(sb.run("list"), names=["id"])

    def test_corrupt_state_is_not_overwritten(self):
        """THE FAIL-OPEN THAT MATTERS.

        Defaulting a corrupt state file would set cost_usd to 0, and
        `stop_reason` reads that field to enforce the effort ceiling. A turn
        could then buy the loop an unlimited budget by corrupting one file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = '{"turn": 3, "cost_usd": 12.5'
            sb = _sandbox(tmp, state=corrupt)
            proc = sb.run("turn", "--bump")
            self.assertEqual(proc.returncode, STATE_UNREADABLE)
            with open(sb.state_path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), corrupt,
                                 "a state file the CLI could not read must be left "
                                 "exactly as found, never rewritten from defaults")

    def test_a_missing_state_file_still_defaults(self):
        # Absent is not corrupt: the first turn of a night has no state file.
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp)
            os.unlink(sb.state_path)
            proc = sb.run("turn", "--bump")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "1")


class NextIdCannotCollide(unittest.TestCase):
    """An id the file already carries must never be handed out again."""

    def test_non_numeric_id_does_not_restart_the_counter(self):
        # Pre-fix: int("OLD") raised, the loop `continue`d, and the next id was
        # i0001 -- which is exactly the id a mixed file is most likely to hold.
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog=[dict(GOOD_ITEM, id="iOLD")])
            proc = sb.run("add", "--title", "t", "--why", "w")
            self.assertEqual(proc.returncode, STATE_UNREADABLE, proc.stdout)
            self.assertIn("iOLD", proc.stderr)

    def test_the_next_id_follows_the_highest_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog=[dict(GOOD_ITEM, id="i0009"),
                                        dict(GOOD_ITEM, id="i0027")])
            proc = sb.run("add", "--title", "t", "--why", "w")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "i0028")


class TheHealthyPathsStillWork(unittest.TestCase):
    """A guard that refuses everything would pass every test above."""

    def test_list_plan_turn_set_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog=[dict(GOOD_ITEM, status="accepted")])
            self.assertEqual(sb.run("list").returncode, 0)
            plan = sb.run("plan", "--deck-every", "5", "--review-every", "4")
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertEqual(plan.stdout.strip(), "implement")
            self.assertEqual(sb.run("turn", "--bump").stdout.strip(), "4")
            self.assertEqual(sb.run("set", "--id", "i0001", "--status",
                                    "implemented", "--note", "n").returncode, 0)
            self.assertEqual(sb.run("render").returncode, 0)
            self.assertEqual(sb.run("cost", "--add", "0.5").stdout.strip(), "13.0000")
            self.assertEqual(sb.run("stats").returncode, 0)

    def test_the_shipped_state_files_parse(self):
        # The repo's own files must satisfy the validator they now go through.
        proc = subprocess.run([PY, BACKLOG_PY, "stats"], cwd=REPO_ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TheSupervisorChecksTheExitCode(unittest.TestCase):
    """run.sh must never turn a failed read into an empty scalar it trusts."""

    def _backlog_scalar(self, cli, *args):
        script = f"""
set -uo pipefail
export LOOP_RUN_SOURCED=1
source {RUN_SH}
BACKLOG=({PY} {cli})
out=$(backlog_scalar {' '.join(args)}); rc=$?
printf 'RC=%s\\n' "$rc"
printf 'OUT=%s\\n' "$out"
"""
        proc = subprocess.run(["bash", "-c", script], cwd=REPO_ROOT,
                              capture_output=True, text=True)
        rc, out = None, ""
        for line in proc.stdout.splitlines():
            if line.startswith("RC="):
                rc = int(line[3:])
            elif line.startswith("OUT="):
                out = line[4:]
        return rc, out, proc.stderr + proc.stdout

    def test_a_broken_read_returns_non_zero_and_no_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog='[{"id": "i0001"')
            rc, out, log = self._backlog_scalar(sb.cli, "plan", "--deck-every", "5")
            self.assertNotEqual(rc, 0, "a failed backlog read must not read as success")
            self.assertEqual(out, "", "no value may be handed back from a failed read")
            self.assertIn("backlog.json", log,
                          "the supervisor log must name the real cause, not 'unknown turn type'")

    def test_a_healthy_read_passes_the_value_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = _sandbox(tmp, backlog=[dict(GOOD_ITEM, status="accepted")])
            rc, out, log = self._backlog_scalar(sb.cli, "plan", "--deck-every", "5",
                                                "--review-every", "4")
            self.assertEqual(rc, 0, log)
            self.assertEqual(out, "implement")

    def test_an_empty_answer_is_refused_even_with_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "fake.py")
            with open(fake, "w", encoding="utf-8") as fh:
                fh.write("import sys\nsys.exit(0)\n")   # exits clean, prints nothing
            rc, out, _ = self._backlog_scalar(fake, "plan")
            self.assertNotEqual(rc, 0,
                                "an empty scalar is what the case statement mis-reported as "
                                "'unknown turn type' -- it must be refused here instead")
            self.assertEqual(out, "")

    def test_no_raw_backlog_capture_survives_in_run_sh(self):
        """The durable invariant: every capture goes through the guard.

        This is the form of the test that keeps holding when someone adds a
        seventh call site next week, which a behavioural test cannot.
        """
        offenders = []
        with open(RUN_SH, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not re.search(r'\$\(\s*"\$\{BACKLOG\[@\]\}"', line):
                    continue
                if line.lstrip().startswith("#"):
                    continue                       # the comment explaining the rule
                if '"${BACKLOG[@]}" "$@"' in line:
                    continue                       # the guards' own pass-through
                offenders.append(f"{n}: {line.strip()}")
        self.assertEqual(offenders, [],
                         "capture the backlog CLI through backlog_scalar, not directly -- "
                         "a raw $(...) discards the exit code")

    def test_the_guards_are_the_only_pass_through(self):
        # The exemption above is only safe while exactly the two guards use it.
        with open(RUN_SH, encoding="utf-8") as fh:
            body = fh.read()
        self.assertEqual(body.count('"${BACKLOG[@]}" "$@"'), 2)


if __name__ == "__main__":
    unittest.main()
