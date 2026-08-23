"""The textclf reproduce path, as assertions.

WHAT THIS CHECKS
    docs/POSTMORTEM-textclf.md lands a measured NEGATIVE result on main, and a
    negative result nobody can re-run is a claim rather than evidence. The four
    modules under router/textclf/ are the thing a reader would run. This file
    pins the four properties that decide whether they can be run at all:

      * NO MACHINE-SPECIFIC ABSOLUTE PATH. Every module used to hardcode one
        laptop's repo root and one dead session scratch directory under /tmp.
        Both resolve to nothing on any other machine, and the /tmp one stops
        resolving on this one at the next reboot.
      * FAIL FAST AND SAY WHAT IS MISSING. With the scratch directory
        unconfigured the modules must stop with a message naming the
        environment variable -- before the ~10 s transformers import, not after
        it, and never with a FileNotFoundError against a path the reader has no
        way to interpret.
      * THE REPO'S OWN MODULES WIN. probe.py and finetune.py put the scratch
        directory on sys.path to reach the modules that are not in the repo.
        That directory also holds STALE COPIES of the repo's own files --
        including the pre-fix finetune.py whose degenerate arm handling is the
        headline trap in router/textclf/README.md -- so the scratch entries must
        sit AFTER the repo's, never before.
      * THE README NAMES WHAT THE CODE READS. The scratch inputs are derived
        from the licensed export and cannot be committed, so the reproduce
        instructions are the only place they can be described. This file
        extracts them from the source and fails when the README has drifted.

    Plus the isolation criterion i0004 was accepted on: nothing outside
    router/textclf/ may reference it, so `make all` and `make test` never import
    torch or transformers.

WHAT IT WRITES
    Nothing. Read-only over the repo, plus subprocesses that are expected to
    exit non-zero before they open anything.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTCLF = os.path.join(REPO_ROOT, "router", "textclf")
MODULES = ("encode.py", "evalpool.py", "finetune.py", "probe.py")
HELPER = "_scratch.py"
SCRATCH_ENV = "TEXTCLF_SCRATCH"
# The four entry points resolve the directory through the helper rather than
# naming the variable each: one spelling, one error message, one place to change.
RESOLVE = "scratch_dir()"

# The import machinery is what makes a stale sibling dangerous, so the heavy
# third-party imports are named here rather than inferred.
HEAVY = ("torch", "transformers")


def _src(name):
    with open(os.path.join(TEXTCLF, name), "r", encoding="utf-8") as fh:
        return fh.read()


def _readme():
    with open(os.path.join(TEXTCLF, "README.md"), "r", encoding="utf-8") as fh:
        return fh.read()


class NoModuleCarriesAMachineSpecificPath(unittest.TestCase):
    """An absolute path outside the repo is dead everywhere except one laptop."""

    def test_no_absolute_path_literal_in_any_module(self):
        every = sorted(f for f in os.listdir(TEXTCLF) if f.endswith(".py"))
        self.assertIn(HELPER, every, "the scratch helper has gone missing")
        # Any absolute path in a single- or double-quoted literal. Catches both
        # the /tmp scratch constant and the hardcoded /home repo root.
        pat = re.compile(r"""['"](/(?:tmp|home|Users|var|opt)/[^'"]*)['"]""")
        found = []
        for name in every:
            for line_no, line in enumerate(_src(name).splitlines(), 1):
                for m in pat.finditer(line):
                    found.append("%s:%d %s" % (name, line_no, m.group(1)))
        self.assertEqual(
            found, [],
            "router/textclf carries absolute paths that exist on one machine only. "
            "Resolve the repo root from __file__ and the scratch directory from the "
            "%s environment variable.\n  %s" % (SCRATCH_ENV, "\n  ".join(found)))


class TheScratchDirectoryIsConfiguredNotGuessed(unittest.TestCase):
    """Unconfigured must stop with an actionable message, and stop early."""

    def test_the_helper_is_the_only_place_the_variable_is_named(self):
        self.assertIn(SCRATCH_ENV, _src(HELPER))
        for name in MODULES:
            with self.subTest(module=name):
                self.assertIn(
                    RESOLVE, _src(name),
                    "%s does not resolve its scratch directory through "
                    "%s, so it is still implicit" % (name, HELPER))

    def test_the_scratch_check_precedes_the_heavy_imports(self):
        """Fail-fast is the whole point: a 10 s import before a config error is a
        reader giving up, and it makes the subprocess check below slow enough to
        be skipped by whoever next edits this file."""
        for name in MODULES:
            src = _src(name)
            if not any(re.search(r"^\s*(?:import|from)\s+%s\b" % h, src, re.M)
                       for h in HEAVY):
                continue
            with self.subTest(module=name):
                lines = src.splitlines()
                guard = next(i for i, l in enumerate(lines) if RESOLVE in l)
                heavy = min(
                    i for i, l in enumerate(lines)
                    if any(re.match(r"\s*(?:import|from)\s+%s\b" % h, l) for h in HEAVY))
                self.assertLess(
                    guard, heavy,
                    "%s imports %s at line %d before resolving %s at line %d; the "
                    "config error must come first" % (
                        name, HEAVY, heavy + 1, RESOLVE, guard + 1))

    def test_running_unconfigured_exits_non_zero_naming_the_variable(self):
        env = {k: v for k, v in os.environ.items() if k != SCRATCH_ENV}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in MODULES:
            with self.subTest(module=name):
                try:
                    proc = subprocess.run(
                        [sys.executable, os.path.join(TEXTCLF, name)],
                        capture_output=True, text=True, timeout=60,
                        cwd=REPO_ROOT, env=env)
                except subprocess.TimeoutExpired:
                    self.fail(
                        "%s did not stop within 60 s with no scratch directory "
                        "configured. On the machine that produced the postmortem "
                        "the hardcoded /tmp path still resolves, so the module "
                        "starts a real multi-hour run instead of reporting that "
                        "it was never told where to look." % name)
                self.assertNotEqual(
                    proc.returncode, 0,
                    "%s ran to completion with no scratch directory configured" % name)
                self.assertIn(
                    SCRATCH_ENV, proc.stderr + proc.stdout,
                    "%s failed without naming %s, so the reader cannot tell what "
                    "to set. stderr tail:\n%s" % (
                        name, SCRATCH_ENV, proc.stderr[-800:]))


class TheReposOwnModulesWinOverStaleScratchCopies(unittest.TestCase):
    """The scratch directory holds older copies of these same file names."""

    def test_scratch_is_appended_to_sys_path_never_inserted(self):
        offenders = []
        for name in MODULES:
            for line_no, line in enumerate(_src(name).splitlines(), 1):
                if "sys.path.insert" in line:
                    offenders.append("%s:%d %s" % (name, line_no, line.strip()))
        self.assertEqual(
            offenders, [],
            "sys.path.insert puts the scratch directory AHEAD of the repo, so a "
            "stale sibling of the same name shadows the shipped module -- and the "
            "stale finetune.py there is the degenerate revision README.md warns "
            "about. Use sys.path.append.\n  %s" % "\n  ".join(offenders))


class TheReadmeNamesEveryScratchInput(unittest.TestCase):
    """The scratch inputs cannot be committed, so the README is the only record."""

    def _scratch_reads(self):
        """Every `{SP}/...` path literal and every non-repo module import."""
        paths, mods = set(), set()
        known = {"sys", "os", "json", "time", "re", "math", "numpy", "np",
                 "torch", "transformers", "router"}
        for name in MODULES:
            src = _src(name)
            for m in re.finditer(r"\{SP\}/([A-Za-z0-9_./{}-]+)", src):
                # Strip an f-string interpolation such as L{L} back to its stem.
                paths.add(re.sub(r"\{[^}]*\}", "*", m.group(1)))
            for m in re.finditer(r"^\s*from\s+([A-Za-z_]\w*)\s+import", src, re.M):
                mod = m.group(1)
                if mod.split(".")[0] not in known and mod not in {
                        n[:-3] for n in MODULES + (HELPER,)}:
                    mods.add(mod + ".py")
        return paths, mods

    def test_every_scratch_path_and_module_appears_in_the_readme(self):
        readme = _readme()
        paths, mods = self._scratch_reads()
        self.assertTrue(paths, "extracted no scratch paths; the scanner has rotted")
        # A path built with an f-string interpolation -- emb_modernbert_L{L}.npy
        # -- is documented once the README names any concrete spelling of it, so
        # the `*` the scanner leaves behind matches a run of non-space.
        def documented(p):
            for cand in (p, os.path.basename(p)):
                rx = "".join(r"[^\s|`]*" if part == "*" else re.escape(part)
                             for part in re.split(r"(\*)", cand))
                if re.search(rx, readme):
                    return True
            return False

        missing = sorted(p for p in paths | mods if not documented(p))
        self.assertEqual(
            missing, [],
            "router/textclf/README.md does not name every artifact the code reads "
            "out of the scratch directory, so a reader cannot assemble it: %s"
            % missing)

    def test_the_readme_names_the_env_var(self):
        self.assertIn(
            SCRATCH_ENV, _readme(),
            "the README documents a reproduce command without saying how the "
            "scratch directory is located")


class TextclfStaysOutOfThePipeline(unittest.TestCase):
    """i0004's acceptance criterion: `make all` must not be able to reach torch."""

    def test_nothing_outside_textclf_references_it(self):
        proc = subprocess.run(
            ["git", "grep", "-l", "textclf", "--",
             ".", ":!docs/", ":!loop/", ":!tests/", ":!router/textclf"],
            capture_output=True, text=True, cwd=REPO_ROOT)
        hits = [l for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(
            hits, [],
            "router/textclf is imported from outside itself, so the pipeline can "
            "now pull in torch: %s" % hits)

    def test_the_makefile_has_no_textclf_target(self):
        with open(os.path.join(REPO_ROOT, "Makefile"), "r", encoding="utf-8") as fh:
            self.assertNotIn("textclf", fh.read())


if __name__ == "__main__":
    unittest.main()
