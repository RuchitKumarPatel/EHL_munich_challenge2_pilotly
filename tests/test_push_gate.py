"""The push gate must scan the tree it is about to push, not the one checked out.

WHY THIS FILE EXISTS
    `loop/run.sh` ran the data-safety suite once, against whatever `HEAD` was,
    and then pushed every ref under `refs/heads`. A branch that an earlier turn
    updated and moved off is therefore pushed having never been scanned -- which
    is precisely the incident the gate was written to stop: on the loop's first
    night `bootstrap.sh` pushed every branch before any check ran, and one of
    them carried raw identifiers out of the export to a shared repository.

WHAT IS TESTED
    1. The suite's repo-file enumeration honours DATA_SAFETY_SCAN_REF, so it can
       be pointed at an arbitrary git ref instead of the working tree. Both
       directions: a clean ref passes, a ref whose tree carries a concrete
       placeholder fails and names the offending path.
    2. `push_all_branches` in loop/push_gate.sh refuses the branch that fails its
       own scan and still pushes the clean ones. Driven against a throwaway
       repository with a real local remote, so the assertion is what actually
       landed on the remote, not what the function printed.

HOW THE PROBE STAYS SAFE
    The placeholder used as a probe is assembled at runtime from an invented
    category, so this file carries no concrete placeholder of its own and the
    probe string occurs zero times in the export. The dirty commit is written
    with `git commit-tree` and left DANGLING -- it is never given a name under
    refs/heads, so no push path can ever see it.

RUNTIME
    ~6 s: two real runs of the two repo-file checks, plus a throwaway repository
    whose scanner is a stub.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Running this file directly puts tests/ on sys.path, not the repo root, so
# `from tests import test_data_safety` below would fail where discovery succeeds.
# Direct execution has to collect and pass exactly what discovery does; see
# TheCollectedSuiteIsTheSameBothWays.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PUSH_GATE = os.path.join(REPO_ROOT, "loop", "push_gate.sh")

#: Assembled, never written literally: this file is itself scanned by the suite
#: it is testing. Invented category, zero occurrences in the export.
PROBE_PLACEHOLDER = "PII_" + "ZZPROBE" + "_" + "7"

#: The marker the stub scanner in the throwaway repository treats as a leak.
STUB_MARKER = "ZZ" + "LEAKMARKER"


def _git(*args, cwd=REPO_ROOT, **kw):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True, **kw)


def _have_git():
    return _git("rev-parse", "--git-dir").returncode == 0


class RepoScanIsRefAware(unittest.TestCase):
    """DATA_SAFETY_SCAN_REF points the repo-file checks at a ref's tree."""

    #: Only the classes that enumerate repo files. The generated-artifact and
    #: export shingle checks are working-tree concerns and cost ten seconds.
    TARGET = ("tests.test_data_safety.RepoSourceCarriesNoExportContent",
              "tests.test_data_safety.RepoCarriesNoPresignedSignature")

    @classmethod
    def setUpClass(cls):
        if not _have_git():
            raise unittest.SkipTest("git is unavailable")

    def _run(self, ref):
        env = dict(os.environ, DATA_SAFETY_SCAN_REF=ref)
        return subprocess.run([os.environ.get("PYTHON", ".venv/bin/python"),
                               "-m", "unittest", *self.TARGET],
                              cwd=REPO_ROOT, capture_output=True, text=True, env=env)

    @staticmethod
    def _dangling_dirty_commit(rel_path, body):
        """A commit off HEAD's tree with one extra file, reachable by sha only."""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
            subprocess.run(["git", "read-tree", "HEAD"], cwd=REPO_ROOT, check=True, env=env)
            blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=REPO_ROOT,
                                  input=body, capture_output=True, text=True,
                                  check=True).stdout.strip()
            subprocess.run(["git", "update-index", "--add",
                            "--cacheinfo", "100644,%s,%s" % (blob, rel_path)],
                           cwd=REPO_ROOT, check=True, env=env)
            tree = subprocess.run(["git", "write-tree"], cwd=REPO_ROOT, check=True,
                                  capture_output=True, text=True, env=env).stdout.strip()
        return subprocess.run(["git", "commit-tree", tree, "-m", "data-safety probe"],
                              cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()

    def test_a_clean_ref_passes(self):
        # Negative control. Without this, "the dirty ref fails" proves only that
        # something is broken, not that the ref is what is being read.
        proc = self._run("HEAD")
        self.assertEqual(proc.returncode, 0,
                         "scanning HEAD's tree should be clean:\n%s" % proc.stderr[-2000:])

    def test_a_dirty_ref_fails_and_names_the_path(self):
        sha = self._dangling_dirty_commit(
            "docs/SCRATCH-PROBE.md",
            "A probe file. Concrete placeholder: %s\n" % PROBE_PLACEHOLDER)
        proc = self._run(sha)
        self.assertNotEqual(proc.returncode, 0,
                            "a ref carrying a concrete placeholder must fail the scan")
        self.assertIn("docs/SCRATCH-PROBE.md", proc.stderr,
                      "the failure must name the offending path:\n%s" % proc.stderr[-2000:])

    def test_the_probe_never_becomes_a_branch(self):
        # The probe commit is deliberately dangling. If it ever acquired a name
        # under refs/heads, push_all_branches would offer it to the remote.
        sha = self._dangling_dirty_commit("docs/SCRATCH-PROBE.md", "probe\n")
        named = _git("for-each-ref", "--format=%(objectname)", "refs/heads").stdout.split()
        self.assertNotIn(sha, named)


class PushGateRefusesOnlyTheDirtyBranch(unittest.TestCase):
    """push_all_branches skips the branch that fails its scan and pushes the rest."""

    @classmethod
    def setUpClass(cls):
        if not _have_git():
            raise unittest.SkipTest("git is unavailable")
        if not os.path.isfile(PUSH_GATE):
            raise unittest.SkipTest("loop/push_gate.sh is not present")
        if shutil.which("bash") is None:
            raise unittest.SkipTest("bash is unavailable")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pushgate-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work = os.path.join(self.tmp, "work")
        self.remote = os.path.join(self.tmp, "remote.git")
        subprocess.run(["git", "init", "-q", "--bare", self.remote], check=True)
        subprocess.run(["git", "init", "-q", "-b", "main", self.work], check=True)
        for key, val in (("user.email", "loop@example.invalid"), ("user.name", "loop")):
            _git("config", key, val, cwd=self.work)
        os.makedirs(os.path.join(self.work, "loop"))
        shutil.copy(PUSH_GATE, os.path.join(self.work, "loop", "push_gate.sh"))
        self._commit("README.md", "clean\n", "main")
        # `dirty-b` carries the marker the stub scanner rejects; the others do not.
        for branch, name, body in (("clean-a", "a.md", "clean\n"),
                                   ("dirty-b", "b.md", STUB_MARKER + "\n"),
                                   ("clean-c", "c.md", "clean\n")):
            _git("checkout", "-q", "-b", branch, "main", cwd=self.work)
            self._commit(name, body, branch)
            _git("checkout", "-q", "main", cwd=self.work)
        _git("remote", "add", "origin", self.remote, cwd=self.work)

    def _commit(self, rel, body, msg):
        with open(os.path.join(self.work, rel), "w") as fh:
            fh.write(body)
        _git("stage", rel, cwd=self.work)
        _git("-c", "commit.gpgsign=false", "commit", "-q", "-m", msg, cwd=self.work)

    def _stub_scanner(self):
        """Stands in for the real suite: a ref whose tree carries the marker is red."""
        path = os.path.join(self.tmp, "stub-scan.sh")
        return self._write_stub(
            'ref="${DATA_SAFETY_SCAN_REF:-}"\n'
            'if [ -z "$ref" ]; then\n'
            '  echo "DATA-SAFETY-RECEIPT ref=worktree files=3 ran=9 skipped=0 failed=0"\n'
            "  exit 0\n"
            "fi\n"
            'if git grep -q %s "$ref" --; then\n'
            '  echo "DATA-SAFETY-RECEIPT ref=$ref files=3 ran=9 skipped=0 failed=1"\n'
            "  exit 1\n"
            "fi\n"
            'echo "DATA-SAFETY-RECEIPT ref=$ref files=3 ran=9 skipped=0 failed=0"\n'
            "exit 0\n" % STUB_MARKER)

    def _write_stub(self, body, name="stub-scan.sh"):
        """A stand-in for `$PY -m tests.test_data_safety`, receipt and all."""
        path = os.path.join(self.tmp, name)
        with open(path, "w") as fh:
            fh.write("#!/usr/bin/env bash\n" + body)
        os.chmod(path, 0o755)
        return path

    def _run_gate(self, scanner=None):
        script = ('set -uo pipefail\n'
                  'ROOT="$1"; source "$ROOT/loop/push_gate.sh"\n'
                  'push_all_branches\n')
        logdir = os.path.join(self.tmp, "logs")
        os.makedirs(logdir, exist_ok=True)
        env = dict(os.environ,
                   PY=scanner or self._stub_scanner(), REMOTE="origin", PUSH="1",
                   INTEGRATION_BRANCH="main", DIVERGED="0", LOGDIR=logdir)
        return subprocess.run(["bash", "-c", script, "gate", self.work],
                              cwd=self.work, capture_output=True, text=True, env=env)

    def _on_remote(self):
        out = _git("for-each-ref", "--format=%(refname:short)", "refs/heads",
                   cwd=self.remote).stdout.split()
        return set(out)

    def test_the_clean_branches_are_pushed(self):
        proc = self._run_gate()
        landed = self._on_remote()
        self.assertIn("clean-a", landed, proc.stdout + proc.stderr)
        self.assertIn("clean-c", landed, proc.stdout + proc.stderr)

    def test_the_dirty_branch_is_refused(self):
        proc = self._run_gate()
        self.assertNotIn("dirty-b", self._on_remote(),
                         "a branch that fails its own scan must not reach the remote\n%s"
                         % (proc.stdout + proc.stderr))

    def test_one_bad_branch_does_not_suppress_the_others(self):
        # The old gate was all-or-nothing on the working tree. Per-branch scanning
        # is only an improvement if one red branch still lets the clean ones out.
        self._run_gate()
        landed = self._on_remote()
        self.assertEqual(landed & {"clean-a", "clean-c", "dirty-b"}, {"clean-a", "clean-c"})

    def test_the_refusal_is_logged_by_name(self):
        proc = self._run_gate()
        self.assertIn("dirty-b", proc.stdout + proc.stderr,
                      "a silently skipped branch is indistinguishable from a pushed one")


class TheGateRefusesAScanThatDidNotRun(unittest.TestCase):
    """A scanner that skipped everything must not read as a clean branch.

    THE BUG THIS PINS. `python -m unittest` exits 0 when every test skips, and
    every content check in tests/test_data_safety.py skipped whenever it could
    not enumerate files. `DATA_SAFETY_SCAN_REF=refs/heads/no-such-branch` printed
    `OK (skipped=6)` and exited 0, so the gate -- which read only that exit code
    -- pushed the branch having scanned nothing. Any git error reached the same
    place: a deleted ref, a dropped object, a name git refuses to parse.

    These cases drive the REAL push_gate.sh against a real local remote with
    stub scanners standing in for the suite, and assert on what LANDED on the
    remote rather than on what the gate printed.
    """

    @classmethod
    def setUpClass(cls):
        if not _have_git():
            raise unittest.SkipTest("git is unavailable")
        if not os.path.isfile(PUSH_GATE):
            raise unittest.SkipTest("loop/push_gate.sh is not present")
        if shutil.which("bash") is None:
            raise unittest.SkipTest("bash is unavailable")

    setUp = PushGateRefusesOnlyTheDirtyBranch.setUp
    _commit = PushGateRefusesOnlyTheDirtyBranch._commit
    _stub_scanner = PushGateRefusesOnlyTheDirtyBranch._stub_scanner
    _write_stub = PushGateRefusesOnlyTheDirtyBranch._write_stub
    _run_gate = PushGateRefusesOnlyTheDirtyBranch._run_gate
    _on_remote = PushGateRefusesOnlyTheDirtyBranch._on_remote

    #: Exit 0 and no receipt at all -- the shape of `unittest` skipping the lot.
    SILENT_GREEN = "exit 0\n"

    #: Exit 0 with a receipt admitting it enumerated nothing.
    ZERO_FILES = ('ref="${DATA_SAFETY_SCAN_REF:-worktree}"\n'
                  'echo "DATA-SAFETY-RECEIPT ref=$ref files=0 ran=6 skipped=6 failed=0"\n'
                  "exit 0\n")

    #: Exit 0 with a receipt admitting it ran nothing.
    ZERO_TESTS = ('ref="${DATA_SAFETY_SCAN_REF:-worktree}"\n'
                  'echo "DATA-SAFETY-RECEIPT ref=$ref files=126 ran=0 skipped=0 failed=0"\n'
                  "exit 0\n")

    def test_a_scanner_that_prints_no_receipt_pushes_nothing(self):
        proc = self._run_gate(self._write_stub(self.SILENT_GREEN))
        self.assertEqual(self._on_remote(), set(),
                         "a scan with no receipt must not let any branch out\n%s"
                         % (proc.stdout + proc.stderr))

    def test_a_receipt_reporting_zero_files_pushes_nothing(self):
        proc = self._run_gate(self._write_stub(self.ZERO_FILES))
        self.assertEqual(self._on_remote(), set(),
                         "enumerating zero files is not the same as finding nothing\n%s"
                         % (proc.stdout + proc.stderr))

    def test_a_receipt_reporting_zero_tests_pushes_nothing(self):
        proc = self._run_gate(self._write_stub(self.ZERO_TESTS))
        self.assertEqual(self._on_remote(), set(),
                         "running zero tests is not the same as passing them\n%s"
                         % (proc.stdout + proc.stderr))

    def test_the_honest_scanner_still_pushes_the_clean_branches(self):
        # Negative control. Without it the three cases above are satisfied by a
        # gate that refuses everything, which would be useless rather than safe.
        self._run_gate()
        self.assertEqual(self._on_remote() & {"clean-a", "clean-c", "dirty-b"},
                         {"clean-a", "clean-c"})

    def test_the_refusal_names_the_log(self):
        proc = self._run_gate(self._write_stub(self.SILENT_GREEN))
        self.assertIn("data-safety", proc.stdout + proc.stderr,
                      "a refusal the operator cannot trace is only half a refusal")


class TheSuiteFailsClosedWhenItCannotRun(unittest.TestCase):
    """The real suite, driven the way the gate drives it.

    The class above proves the GATE refuses a scan that did not run. This one
    proves the SUITE tells it so -- that an unresolvable ref, and a missing
    export, produce a non-zero exit and an honest receipt instead of a green
    skip. Both halves are needed: either one alone leaves the fail-open intact.
    """

    #: Resolvable by nothing. Deliberately not a sha, so `git ls-tree` errors.
    DEAD_REF = "refs/heads/zz-no-such-branch-for-the-data-safety-gate"

    @classmethod
    def setUpClass(cls):
        if not _have_git():
            raise unittest.SkipTest("git is unavailable")

    def _run(self, env_extra):
        env = dict(os.environ, **env_extra)
        env.pop("DATA_SAFETY_SCAN_REF", None)
        env.pop("DATA_SAFETY_STRICT", None)
        env.pop("DATA_SAFETY_EXPORT_DIR", None)
        env.update(env_extra)
        return subprocess.run([os.environ.get("PYTHON", ".venv/bin/python"),
                               "-m", "tests.test_data_safety"],
                              cwd=REPO_ROOT, capture_output=True, text=True, env=env)

    @staticmethod
    def _receipt(proc):
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.startswith("DATA-SAFETY-RECEIPT "):
                return dict(part.split("=", 1) for part in line.split()[1:])
        return None

    def test_an_unresolvable_ref_exits_non_zero(self):
        proc = self._run({"DATA_SAFETY_SCAN_REF": self.DEAD_REF})
        self.assertNotEqual(proc.returncode, 0,
                            "a ref the scan cannot read must be RED, not OK (skipped=n)"
                            "\n%s" % (proc.stdout + proc.stderr)[-2000:])

    def test_an_unresolvable_ref_admits_it_enumerated_nothing(self):
        proc = self._run({"DATA_SAFETY_SCAN_REF": self.DEAD_REF})
        receipt = self._receipt(proc)
        self.assertIsNotNone(receipt, "the gate runner must always print a receipt")
        self.assertEqual(receipt["files"], "0")
        self.assertNotEqual(receipt["failed"], "0")

    def test_a_real_ref_is_green_and_says_what_it_looked_at(self):
        # Negative control for both cases above.
        proc = self._run({"DATA_SAFETY_SCAN_REF": "HEAD"})
        self.assertEqual(proc.returncode, 0,
                         "HEAD's own tree must scan clean:\n%s"
                         % (proc.stdout + proc.stderr)[-2000:])
        receipt = self._receipt(proc)
        self.assertIsNotNone(receipt)
        self.assertGreater(int(receipt["files"]), 0)
        self.assertGreater(int(receipt["ran"]), 0)
        self.assertEqual(receipt["skipped"], "0",
                         "nothing may skip under a ref scan; a skip is the fail-open")

    def test_a_missing_export_is_red_under_strict(self):
        empty = tempfile.mkdtemp(prefix="no-export-")
        self.addCleanup(shutil.rmtree, empty, True)
        proc = self._run({"DATA_SAFETY_STRICT": "1", "DATA_SAFETY_EXPORT_DIR": empty})
        self.assertNotEqual(proc.returncode, 0,
                            "an inert export-backed detector is indistinguishable from "
                            "a clean repo, so under the gate it must be RED\n%s"
                            % (proc.stdout + proc.stderr)[-2000:])

    def test_a_missing_export_still_only_skips_outside_the_gate(self):
        # A developer laptop without the 101 MB export must still get a green
        # `make test`. STRICT is what separates the two, and it is set by the
        # gate and by nothing else.
        empty = tempfile.mkdtemp(prefix="no-export-")
        self.addCleanup(shutil.rmtree, empty, True)
        proc = self._run({"DATA_SAFETY_EXPORT_DIR": empty})
        self.assertEqual(proc.returncode, 0,
                         "outside the gate a missing export is a skip, not a failure\n%s"
                         % (proc.stdout + proc.stderr)[-2000:])
        receipt = self._receipt(proc)
        self.assertGreater(int(receipt["skipped"]), 0)


class EverySupervisorPushPathUsesTheGate(unittest.TestCase):
    """bootstrap.sh had no gate at all; re-running it reproduced the night-one leak.

    These are structural assertions on purpose. The behaviour they protect is
    tested above against a real remote; what can still go wrong is a second push
    path growing next to the gated one, which is exactly what happened when the
    gate was added to run.sh and not backported.
    """

    SCRIPTS = ("loop/run.sh", "loop/bootstrap.sh")

    @staticmethod
    def _read(rel):
        with open(os.path.join(REPO_ROOT, rel)) as fh:
            return fh.read()

    def test_both_supervisors_source_the_shared_gate(self):
        for rel in self.SCRIPTS:
            self.assertIn("loop/push_gate.sh", self._read(rel),
                          "%s must go through the shared gate" % rel)

    def test_neither_supervisor_pushes_on_its_own(self):
        for rel in self.SCRIPTS:
            body = self._read(rel)
            self.assertNotIn("git push", body,
                             "%s pushes without the gate; that is the night-one incident" % rel)

    def test_bootstrap_builds_the_artifacts_before_it_tests_them(self):
        # Two classes in tests/test_data_safety.py scan the generated artifacts and
        # SkipTest when that directory is missing. Testing before building makes
        # them green for the wrong reason.
        body = self._read("loop/bootstrap.sh").splitlines()
        build = next(i for i, ln in enumerate(body) if ln.startswith("make all"))
        check = next(i for i, ln in enumerate(body) if ln.startswith("make test"))
        self.assertLess(build, check, "bootstrap must run `make all` before `make test`")

    def test_the_gate_refuses_rather_than_warns(self):
        body = self._read("loop/bootstrap.sh")
        self.assertIn("if push_all_branches; then", body)
        self.assertRegex(body, r"die \"the data-safety gate refused")


class TheCollectedSuiteIsTheSameBothWays(unittest.TestCase):
    """`python tests/test_push_gate.py` must run what `-m unittest` runs.

    The `__main__` guard used to sit in the MIDDLE of this file, above
    OpaqueTokenScanIsRefAware, so direct execution exited before that class was
    defined and reported OK having never run the cases that pin the ref-aware
    fix. Discovery collected them, so nothing was ever red.

    Checked against the parse tree rather than by searching the text: the
    obvious `source.index('if __name__ ...')` matches the literal in this very
    method and quietly measures the wrong thing.
    """

    @staticmethod
    def _tree():
        import ast

        return ast.parse(open(os.path.abspath(__file__)).read())

    def test_the_main_guard_is_below_every_test_class(self):
        import ast

        tree = self._tree()
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        guards = [n for n in tree.body
                  if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)]
        self.assertEqual(len(guards), 1, "expected exactly one __main__ guard")
        self.assertGreater(
            guards[0].lineno, max(c.lineno for c in classes),
            "the __main__ guard must be below every test class, or direct "
            "execution collects only the part of the file above it")

    def test_every_class_in_the_file_is_actually_collected(self):
        import ast

        named = {n.name for n in self._tree().body if isinstance(n, ast.ClassDef)}
        suite = unittest.TestLoader().loadTestsFromName("tests.test_push_gate")
        collected = {type(t).__name__ for g in suite for t in g}
        self.assertEqual(named - collected, set(),
                         "classes defined but never collected: %s" % (named - collected))


class OpaqueTokenScanIsRefAware(unittest.TestCase):
    """The opaque-token detector must read the ref's blobs, not the working tree.

    WHY THIS CLASS EXISTS
        `RepoCarriesNoOpaqueExportToken` and the SCAN_REF plumbing were built on
        two different branches and first met when those branches were integrated.
        The enumerator was ref-aware (`_repo_files` shells out to `git ls-tree`
        under SCAN_REF); the reader was not (`_candidates` opened the path in the
        working tree). Under SCAN_REF the two disagree, and they disagree
        SILENTLY: a path that exists in the ref but not in the working tree is
        dropped by an `os.path.isfile` guard, and a path that exists in both is
        read at the WRONG content. Scanning a branch would therefore have
        examined `main`'s files and reported the branch clean.

    WHAT IS TESTED
        Both directions of the read path, using an invented token rather than a
        real one -- the bug is that the wrong bytes are read, which needs no leak
        to demonstrate. The probe is deliberately NOT in the export, so this file
        stays clean under the very detector it is testing.
    """

    #: 16+ chars of mixed case and digits, so OPAQUE_RE matches it. Invented:
    #: assembled at runtime and never written as one literal, for the same
    #: reason PROBE_PLACEHOLDER is.
    PROBE_TOKEN = "ZZ" + "PROBEOPAQUE" + "4" + "TOKEN" + "9"

    @classmethod
    def setUpClass(cls):
        if not _have_git():
            raise unittest.SkipTest("git is unavailable")
        from tests import test_data_safety
        if not os.path.isdir(test_data_safety.EXPORT):
            raise unittest.SkipTest("export/ is not present")
        cls.mod = test_data_safety

    def _candidates_under(self, ref):
        """token -> files, as the detector sees them with SCAN_REF set to `ref`."""
        mod = self.mod
        klass = mod.RepoCarriesNoOpaqueExportToken
        previous = mod.SCAN_REF
        mod.SCAN_REF = ref
        try:
            paths = mod.RepoCarriesNoPresignedSignature._repo_files()
            self.assertIsNotNone(paths, "the ref's tree should enumerate")
            return klass._candidates(paths, klass._exempt_arm_ids())
        finally:
            mod.SCAN_REF = previous

    def test_a_token_only_in_the_ref_is_extracted(self):
        sha = RepoScanIsRefAware._dangling_dirty_commit(
            "docs/SCRATCH-OPAQUE-PROBE.md",
            "A probe file. Opaque token: %s\n" % self.PROBE_TOKEN)
        found = self._candidates_under(sha)
        self.assertIn(
            self.PROBE_TOKEN, found,
            "the detector enumerated the ref's tree but read the working tree, so "
            "a file that exists only in the ref contributed nothing. Under the push "
            "gate this reports every branch as clean.")
        self.assertIn("docs/SCRATCH-OPAQUE-PROBE.md", found[self.PROBE_TOKEN])

    def test_the_working_tree_does_not_carry_the_probe(self):
        # Negative control. Without it, the assertion above would also pass if
        # the probe were somehow present in the checkout.
        found = self._candidates_under("")
        self.assertNotIn(self.PROBE_TOKEN, found,
                         "the probe must exist only inside the dangling commit")

    def test_the_probe_is_not_in_the_export(self):
        # This file is itself scanned by RepoCarriesNoOpaqueExportToken. The
        # probe is safe to write literally only because the export never saw it.
        klass = self.mod.RepoCarriesNoOpaqueExportToken
        self.assertEqual(klass._which_occur_in_export({self.PROBE_TOKEN}), set())


# Last in the file on purpose. It used to sit above OpaqueTokenScanIsRefAware, so
# `python tests/test_push_gate.py` exited before that class was even defined and
# reported OK having run none of it. Discovery collected it, so nothing was red.
if __name__ == "__main__":
    unittest.main(verbosity=2)
