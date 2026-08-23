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
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        with open(path, "w") as fh:
            fh.write("#!/usr/bin/env bash\n"
                     'ref="${DATA_SAFETY_SCAN_REF:-}"\n'
                     '[ -z "$ref" ] && exit 0\n'
                     'git grep -q %s "$ref" -- && exit 1\n'
                     "exit 0\n" % STUB_MARKER)
        os.chmod(path, 0o755)
        return path

    def _run_gate(self):
        script = ('set -uo pipefail\n'
                  'ROOT="$1"; source "$ROOT/loop/push_gate.sh"\n'
                  'push_all_branches\n')
        logdir = os.path.join(self.tmp, "logs")
        os.makedirs(logdir, exist_ok=True)
        env = dict(os.environ,
                   PY=self._stub_scanner(), REMOTE="origin", PUSH="1",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)


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
