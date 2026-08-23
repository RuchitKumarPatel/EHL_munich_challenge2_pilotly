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


class BranchNamesAreValidatedBeforeTheyReachGit(unittest.TestCase):
    """A ref name is attacker-influenced input, and `git push` obeys option shapes.

    WHY THIS CLASS EXISTS
        `push_all_branches` took every name out of `git for-each-ref` and handed
        it to `git rev-parse` and `git push` as a bare argument, with no `--`
        separator anywhere. `git branch` and `git checkout -b` refuse an
        option-shaped name, but `git update-ref` does not, and a turn agent runs
        with unrestricted bash. Measured on git 2.55 while writing this file:

          * `git update-ref refs/heads/--receive-pack=/tmp/x/pwn.sh HEAD` is
            accepted, and the ref shows up in `for-each-ref`.
          * `git rev-parse` echoes that name back with exit 0, so the gate's
            `|| continue` does not catch it.
          * `git push -q origin '--receive-pack=/tmp/x/pwn.sh'` parses it as an
            option and SPAWNS the named program locally, as the operator's user.
            The push then does whatever `push.default` says, so it is not even
            a no-op.
          * The same name makes `git ls-tree` exit 129, which the per-ref scan
            reads as an infrastructure error rather than as a leak.

        Two of the four shapes the review asked for -- a name containing a space
        and a name containing a glob character -- turn out to be unreachable end
        to end: `update-ref` refuses them and `for-each-ref` warns and ignores a
        ref file written by hand. They are still exercised against the validator
        directly, because the validator is what must not depend on that.

    WHAT IS TESTED
        The validator on all six shapes plus a negative control of real branch
        names; the end-to-end refusal against a real remote; a POSITIVE control
        that the payload does fire when the guard is bypassed, without which the
        refusal test would pass on a repo where nothing works; and that the push
        is pinned to the sha that was scanned rather than re-resolving the name.
    """

    #: An option-shaped ref name whose payload is a script we control. Never a
    #: real program name: the test asserts on a sentinel file, not on side effects.
    HOSTILE = "--receive-pack="

    @classmethod
    def setUpClass(cls):
        if not _have_git():
            raise unittest.SkipTest("git is unavailable")
        if not os.path.isfile(PUSH_GATE):
            raise unittest.SkipTest("loop/push_gate.sh is not present")
        if shutil.which("bash") is None:
            raise unittest.SkipTest("bash is unavailable")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="refname-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work = os.path.join(self.tmp, "work")
        self.remote = os.path.join(self.tmp, "remote.git")
        self.sentinel = os.path.join(self.tmp, "PAYLOAD-RAN")
        subprocess.run(["git", "init", "-q", "--bare", self.remote], check=True)
        subprocess.run(["git", "init", "-q", "-b", "main", self.work], check=True)
        for key, val in (("user.email", "loop@example.invalid"), ("user.name", "loop")):
            _git("config", key, val, cwd=self.work)
        os.makedirs(os.path.join(self.work, "loop"))
        shutil.copy(PUSH_GATE, os.path.join(self.work, "loop", "push_gate.sh"))
        self._commit("README.md", "clean\n")
        _git("checkout", "-q", "-b", "clean-a", "main", cwd=self.work)
        self._commit("a.md", "clean\n")
        _git("checkout", "-q", "main", cwd=self.work)
        _git("remote", "add", "origin", self.remote, cwd=self.work)
        # `main` has an upstream here exactly as it does in the real repo. That
        # is what makes the hostile push do work instead of bailing early.
        _git("push", "-q", "origin", "main:refs/heads/main", cwd=self.work)
        _git("branch", "--set-upstream-to=origin/main", "main", cwd=self.work)

    def _commit(self, rel, body):
        with open(os.path.join(self.work, rel), "w") as fh:
            fh.write(body)
        _git("stage", rel, cwd=self.work)
        _git("-c", "commit.gpgsign=false", "commit", "-q", "-m", rel, cwd=self.work)

    def _payload(self):
        """A stand-in for `git-receive-pack` that records that it was run."""
        path = os.path.join(self.tmp, "payload.sh")
        with open(path, "w") as fh:
            fh.write("#!/usr/bin/env bash\n"
                     "printf 'ran\\n' > %s\n"
                     "exec git-receive-pack \"$@\"\n" % self.sentinel)
        os.chmod(path, 0o755)
        return path

    def _hostile_name(self):
        return self.HOSTILE + self._payload()

    def _make_hostile_ref(self):
        """update-ref accepts what `git branch` refuses. Returns the ref's name."""
        name = self._hostile_name()
        proc = _git("update-ref", "refs/heads/" + name, "main", cwd=self.work)
        if proc.returncode != 0:
            self.skipTest("this git refuses to create the ref at all: %s" % proc.stderr)
        return name

    def _permissive_scanner(self):
        """Every ref is clean. The subject under test is the name, not the tree."""
        path = os.path.join(self.tmp, "stub-scan.sh")
        with open(path, "w") as fh:
            fh.write("#!/usr/bin/env bash\nexit 0\n")
        os.chmod(path, 0o755)
        return path

    def _run_gate(self, scanner=None):
        script = ('set -uo pipefail\n'
                  'ROOT="$1"; source "$ROOT/loop/push_gate.sh"\n'
                  'push_all_branches\n')
        logdir = os.path.join(self.tmp, "logs")
        os.makedirs(logdir, exist_ok=True)
        env = dict(os.environ,
                   PY=scanner or self._permissive_scanner(), REMOTE="origin", PUSH="1",
                   INTEGRATION_BRANCH="main", DIVERGED="0", LOGDIR=logdir)
        return subprocess.run(["bash", "-c", script, "gate", self.work],
                              cwd=self.work, capture_output=True, text=True, env=env)

    def _validator(self, name):
        """`ds_ref_name_ok` alone, so the six shapes can be checked one by one."""
        script = ('set -uo pipefail\n'
                  'ROOT="$1"; source "$ROOT/loop/push_gate.sh"\n'
                  'if ds_ref_name_ok "$2"; then echo OK; else echo REFUSED; fi\n')
        proc = subprocess.run(["bash", "-c", script, "gate", self.work, name],
                              cwd=self.work, capture_output=True, text=True,
                              env=dict(os.environ, ROOT=self.work))
        return proc.stdout.strip()

    def _on_remote(self):
        out = _git("for-each-ref", "--format=%(refname:short)", "refs/heads",
                   cwd=self.remote).stdout.split()
        return set(out)

    # -- the validator ----------------------------------------------------

    def test_the_validator_refuses_every_hostile_shape(self):
        for name in ("-dashlead",
                     "--receive-pack=/tmp/x/pwn.sh",
                     "--upload-pack=/tmp/x/pwn.sh",
                     "has space",
                     "glob*star",
                     "tilde~1",
                     ""):
            with self.subTest(name=name):
                self.assertEqual(self._validator(name), "REFUSED",
                                 "%r must never reach a git argument list" % name)

    def test_the_validator_accepts_the_names_this_repo_actually_uses(self):
        # Negative control. Without it, a validator that refuses everything --
        # and so silently stops backing the night's work up -- would pass above.
        for name in ("main", "idea/ref-name-injection", "text-classifier-finetune",
                     "frontend-console", "entire/415c277-e3b0c4", "v1.2_rc"):
            with self.subTest(name=name):
                self.assertEqual(self._validator(name), "OK",
                                 "%r is an ordinary branch name and must push" % name)

    # -- end to end -------------------------------------------------------

    def test_the_payload_fires_when_the_name_is_a_bare_argument(self):
        # POSITIVE CONTROL, and the whole reason this is a security finding
        # rather than a lint. This is the exact call the gate used to make.
        name = self._make_hostile_ref()
        _git("push", "-q", "origin", name, cwd=self.work)
        self.assertTrue(os.path.exists(self.sentinel),
                        "git push did not treat %r as an option on this git, so the "
                        "refusal test below would prove nothing" % name)

    def test_the_gate_does_not_run_the_payload(self):
        self._make_hostile_ref()
        proc = self._run_gate()
        self.assertFalse(os.path.exists(self.sentinel),
                         "push_all_branches executed a program named by a ref\n%s"
                         % (proc.stdout + proc.stderr))

    def test_the_hostile_ref_is_refused_out_loud(self):
        self._make_hostile_ref()
        proc = self._run_gate()
        out = proc.stdout + proc.stderr
        self.assertIn("unsafe name", out,
                      "a silently skipped branch is indistinguishable from a pushed "
                      "one -- and this one is the reason the gate exists\n%s" % out)
        self.assertNotEqual(proc.returncode, 0,
                            "refusing a branch must be reported to the caller")

    def test_the_clean_branches_still_push_in_the_same_cycle(self):
        self._make_hostile_ref()
        self._run_gate()
        self.assertIn("clean-a", self._on_remote(),
                      "one unsafe name must not suppress the night's backups")

    # -- time of check, time of use ---------------------------------------

    def test_the_push_lands_the_sha_that_was_scanned(self):
        # The gate scanned a ref by NAME and then pushed it by NAME, so anything
        # that moved the branch in between shipped an unscanned tree. Here the
        # scanner itself moves it, which is the same window made deterministic.
        mover = os.path.join(self.tmp, "moving-scan.sh")
        with open(mover, "w") as fh:
            fh.write("#!/usr/bin/env bash\n"
                     '[ -n "${DATA_SAFETY_SCAN_REF:-}" ] || exit 0\n'
                     '[ -f %(flag)s ] && exit 0\n'
                     "touch %(flag)s\n"
                     'cd %(work)s\n'
                     'git checkout -q clean-a\n'
                     'printf "moved after the scan\\n" > late.md\n'
                     'git stage late.md\n'
                     'git -c commit.gpgsign=false commit -q -m late\n'
                     'git checkout -q main\n'
                     "exit 0\n" % {"flag": os.path.join(self.tmp, "moved.flag"),
                                   "work": self.work})
        os.chmod(mover, 0o755)
        scanned = _git("rev-parse", "clean-a", cwd=self.work).stdout.strip()
        self._run_gate(scanner=mover)
        landed = _git("rev-parse", "clean-a", cwd=self.remote).stdout.strip()
        self.assertEqual(landed, scanned,
                         "the remote received a commit that no scan ever saw")


class ScanRefIsNotHandedToGitAsAnOption(unittest.TestCase):
    """The other half of i0023: the ref reaches the SUITE as an argument too.

    `push_gate.sh` now validates the branch name before it resolves it, so what
    the suite receives in DATA_SAFETY_SCAN_REF is a sha. That is the fix; this is
    the defence in depth behind it, because tests/test_data_safety.py is also run
    by hand, and its git calls were `git ls-tree -r --name-only -z $SCAN_REF` and
    `git cat-file -s $SCAN_REF:$path` with no separator of any kind.

    An option-shaped value there does not execute anything -- neither ls-tree nor
    cat-file has a `--receive-pack`-style escape -- but it makes both exit 129,
    and an enumeration that errors is exactly the fail-open path i0022 records.
    So the module refuses such a value outright rather than scanning nothing.
    """

    HOSTILE = ("--receive-pack=/tmp/x/pwn.sh", "-o/tmp/x", "--output=/tmp/x")

    @classmethod
    def setUpClass(cls):
        if not _have_git():
            raise unittest.SkipTest("git is unavailable")

    def _import_under(self, ref):
        """Exit code of importing the suite with that SCAN_REF, in a fresh process."""
        env = dict(os.environ, DATA_SAFETY_SCAN_REF=ref)
        return subprocess.run([os.environ.get("PYTHON", ".venv/bin/python"),
                               "-c", "import tests.test_data_safety"],
                              cwd=REPO_ROOT, capture_output=True, text=True, env=env)

    def test_an_option_shaped_ref_is_refused_rather_than_scanned(self):
        for ref in self.HOSTILE:
            with self.subTest(ref=ref):
                proc = self._import_under(ref)
                self.assertNotEqual(proc.returncode, 0,
                                    "a scan that cannot enumerate must be RED, not empty")
                self.assertIn("DATA_SAFETY_SCAN_REF", proc.stderr)

    def test_an_ordinary_ref_still_imports(self):
        # Negative control: refusing everything would disable the per-ref scan.
        for ref in ("", "HEAD", "main", "refs/heads/main",
                    "0123456789abcdef0123456789abcdef01234567"):
            with self.subTest(ref=ref):
                self.assertEqual(self._import_under(ref).returncode, 0)

    def test_every_git_call_that_takes_the_ref_ends_its_options(self):
        # Durable textual guard, in the idiom of EverySupervisorPushPathUsesTheGate.
        # The validator above is the real control; this is what stops a fourth
        # git call growing next to the two that were hardened.
        with open(os.path.join(REPO_ROOT, "tests", "test_data_safety.py")) as fh:
            body = fh.read()
        for call in ('"git", "ls-tree"', '"git", "cat-file"'):
            self.assertIn(call, body, "the hardened call was renamed away")
        for line in body.splitlines():
            if '"git", "cat-file"' in line:
                self.assertIn("--end-of-options", line, line.strip())
            if '"git", "ls-tree"' in line:
                self.assertIn('"--"', line, line.strip())
