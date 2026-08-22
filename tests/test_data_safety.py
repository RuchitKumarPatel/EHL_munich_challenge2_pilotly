"""The export is proprietary; nothing we generate may carry it back out.

WHAT THIS CHECKS
    1. No file under results/ contains a `PII_` placeholder or an `<ENTITY_>`
       marker, with ONE licensed exception written into the artifact contract:
       results/jobkey.jsonl stores the placeholder cron path for rows whose
       bucket is "pii", precisely so the report can say how many runs have no
       usable job key. Those rows are additionally required to be `literal:
       false`, i.e. the placeholder is reported but never used as a join key.
       Any other PII_ occurrence anywhere in results/ is a leak.
    2. No file under results/ contains a verbatim run of raw tool-output text
       from the export. Checked by sampling 64-character shingles out of the
       function_call_output / custom_tool_call_output items and searching for
       them in every results/ file, binaries included (decoded latin-1, so an
       embedded string inside an .npz or a .png is still found).
    3. No file the repo tracks -- or would track, i.e. untracked and not
       gitignored -- matches an `X-Amz-Signature` pattern. Presigned S3 URLs
       are the classic way a credential escapes through a pasted log line.

WHY THE PII_ RULE IS NOT COSMETIC
    PII_ tokens are renumbered PER REQUEST. `PII_URL_3` in one line and
    `PII_URL_3` in another are different things. So a PII_ string is neither a
    safe identifier to publish nor a valid join key; the brief bans it as a key
    for the second reason and this file enforces the first.

WHAT IT WRITES
    Nothing. It reads results/, the export and the repo's own source files, and
    prints only counts and file paths -- never the matched text.

RUNTIME
    ~10 s, dominated by the shingle scan over the largest results/ file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
EXPORT = os.path.join(REPO_ROOT, "export")

PII_RE = re.compile(r"PII_[A-Z][A-Z0-9_]*")
ENTITY_RE = re.compile(r"<ENTITY_[^>]*>")
#: Presigned-URL signature parameter. The sibling parameters are listed so a
#: reader knows what else to look for; only the signature itself is asserted.
AMZ_SIG_RE = re.compile(r"X-Amz-Signature", re.I)
AMZ_SIBLINGS = ("X-Amz-Credential", "X-Amz-Security-Token", "AWS4-HMAC-SHA256")

#: The single licensed PII_ carve-out, from the results/jobkey.jsonl contract.
PII_ALLOWED_FILE = "jobkey.jsonl"

SHINGLE = 64
MAX_SHINGLES = 1200
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")


def _results_files():
    """Every regular file under results/, sorted; empty when results/ does not exist."""
    if not os.path.isdir(RESULTS):
        return []
    out = []
    for root, _dirs, names in os.walk(RESULTS):
        for name in sorted(names):
            out.append(os.path.join(root, name))
    return sorted(out)


def _text(path):
    """Read any file as text, latin-1 decoded so binaries still yield their strings."""
    with open(path, "rb") as fh:
        return fh.read().decode("latin-1")


def _rel(path):
    """Repo-relative path, for readable failure messages."""
    return os.path.relpath(path, REPO_ROOT)


class ResultsCarryNoPlaceholders(unittest.TestCase):
    """PII_ and <ENTITY_> markers may live in exactly one artifact, by contract."""

    @classmethod
    def setUpClass(cls):
        cls.files = _results_files()
        if not cls.files:
            raise unittest.SkipTest("results/ is empty; run `python -m router.cli all`")
        cls.blobs = {p: _text(p) for p in cls.files}

    def test_no_entity_marker_anywhere(self):
        offenders = {_rel(p): len(ENTITY_RE.findall(b)) for p, b in self.blobs.items()
                     if ENTITY_RE.search(b)}
        self.assertEqual(offenders, {},
                         "<ENTITY_> markers in results/: %s" % offenders)

    def test_pii_placeholders_only_in_the_licensed_artifact(self):
        offenders = {}
        for path, blob in self.blobs.items():
            if os.path.basename(path) == PII_ALLOWED_FILE:
                continue
            hits = PII_RE.findall(blob)
            if hits:
                offenders[_rel(path)] = {"occurrences": len(hits),
                                         "distinct": len(set(hits))}
        self.assertEqual(
            offenders, {},
            "PII_ placeholders outside results/%s. Only that artifact is licensed "
            "to carry them (bucket == 'pii' rows keep the placeholder path so the "
            "report can count them). Offenders: %s" % (PII_ALLOWED_FILE, offenders))

    def test_the_licensed_placeholders_are_never_used_as_a_key(self):
        path = os.path.join(RESULTS, PII_ALLOWED_FILE)
        if not os.path.exists(path):
            raise unittest.SkipTest("results/%s is missing" % PII_ALLOWED_FILE)
        with open(path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        n_pii = 0
        for row in rows:
            if PII_RE.search(row.get("cron_path") or ""):
                n_pii += 1
                self.assertEqual(row["bucket"], "pii", "idx %d" % row["idx"])
                self.assertFalse(row["literal"],
                                 "idx %d: a PII_ path must never be marked literal"
                                 % row["idx"])
            # The placeholder may appear ONLY in cron_path, never in another field.
            for field, value in row.items():
                if field == "cron_path":
                    continue
                self.assertIsNone(PII_RE.search(str(value)),
                                  "idx %d: PII_ in field %r" % (row["idx"], field))
        self.assertEqual(n_pii, 410, "expected 410 PII-bucket rows")


class ResultsCarryNoRawToolOutput(unittest.TestCase):
    """No verbatim run of tool-output text may survive into a generated artifact."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(EXPORT):
            raise unittest.SkipTest("export/ is not present")
        cls.files = _results_files()
        if not cls.files:
            raise unittest.SkipTest("results/ is empty; run `python -m router.cli all`")
        cls.shingles = cls._shingles()

    @staticmethod
    def _shingles():
        """Deterministic 64-char samples of raw tool-output text from the export."""
        from router.io import iter_lines

        out = []
        seen = set()
        for idx, req in iter_lines():
            if idx % 13:                       # deterministic 1-in-13 sample
                continue
            for item in req.get("input", []):
                if not isinstance(item, dict) or item.get("type") not in OUTPUT_TYPES:
                    continue
                raw = item.get("output")
                text = raw if isinstance(raw, str) else json.dumps(raw)
                # One shingle from the middle of each output: the head is often a
                # generic '{"success":true' prefix that proves nothing.
                start = max(0, (len(text) - SHINGLE) // 2)
                shingle = text[start:start + SHINGLE]
                if len(shingle) == SHINGLE and shingle not in seen:
                    seen.add(shingle)
                    out.append(shingle)
                if len(out) >= MAX_SHINGLES:
                    return out
        return out

    def test_the_probe_set_is_non_trivial(self):
        self.assertGreaterEqual(len(self.shingles), 200,
                                "too few tool-output shingles to make this test meaningful")

    def test_no_results_file_contains_raw_tool_output(self):
        offenders = {}
        for path in self.files:
            blob = _text(path)
            hits = sum(1 for s in self.shingles if s in blob)
            if hits:
                offenders[_rel(path)] = hits
        self.assertEqual(
            offenders, {},
            "verbatim tool-output text found in results/ (file -> matching probes "
            "out of %d). The export is proprietary; a generated artifact must carry "
            "aggregates, not transcript. Offenders: %s" % (len(self.shingles), offenders))


class RepoCarriesNoPresignedSignature(unittest.TestCase):
    """No tracked (or about-to-be-tracked) file may contain an X-Amz-Signature."""

    @classmethod
    def setUpClass(cls):
        cls.paths = cls._repo_files()

    @staticmethod
    def _repo_files():
        """Tracked files plus untracked-and-not-ignored ones (read-only git calls)."""
        paths = []
        for args in (["git", "ls-files", "-z"],
                     ["git", "ls-files", "--others", "--exclude-standard", "-z"]):
            try:
                raw = subprocess.run(args, cwd=REPO_ROOT, capture_output=True,
                                     check=True).stdout
            except (OSError, subprocess.CalledProcessError):
                return None
            paths.extend(n.decode("utf-8", "replace") for n in raw.split(b"\0") if n)
        return sorted(set(paths))

    def test_git_listing_is_available(self):
        if self.paths is None:
            raise unittest.SkipTest("git is unavailable; cannot enumerate repo files")
        self.assertGreater(len(self.paths), 0)

    def test_no_presigned_signature_in_any_repo_file(self):
        if self.paths is None:
            raise unittest.SkipTest("git is unavailable; cannot enumerate repo files")
        offenders = []
        for rel in self.paths:
            path = os.path.join(REPO_ROOT, rel)
            if not os.path.isfile(path) or os.path.getsize(path) > 64 * 1024 * 1024:
                continue
            try:
                blob = _text(path)
            except OSError:
                continue
            if AMZ_SIG_RE.search(blob):
                # This file names the pattern in its own source; that is not a hit.
                if os.path.abspath(path) == os.path.abspath(__file__):
                    continue
                offenders.append(rel)
        self.assertEqual(offenders, [],
                         "X-Amz-Signature found in repo files: %s (also check for %s)"
                         % (offenders, ", ".join(AMZ_SIBLINGS)))

    def test_the_export_is_not_tracked(self):
        if self.paths is None:
            raise unittest.SkipTest("git is unavailable; cannot enumerate repo files")
        tracked_export = [p for p in self.paths if p.startswith("export/")]
        self.assertEqual(tracked_export, [],
                         "the proprietary export must never be tracked: %s" % tracked_export)

    def test_results_is_not_tracked(self):
        if self.paths is None:
            raise unittest.SkipTest("git is unavailable; cannot enumerate repo files")
        tracked = [p for p in self.paths if p.startswith("results/")]
        self.assertEqual(tracked, [],
                         "results/ is generated output and is gitignored: %s" % tracked)


class RepoSourceCarriesNoExportContent(unittest.TestCase):
    """Source, docs and skills must not have transcript pasted into them.

    A file may NAME the pattern -- `PII_RE`, "PII_URL_N", a regex literal -- and
    it may quote the handful of placeholders that are FINDINGS rather than data
    (PII_URL_3 is the measurement that killed the banned job key; PII_SERVICE_1
    is the one trigger value router/features.py had to classify). What it may not
    do is carry an unexplained concrete placeholder, which would mean a slice of
    a real trajectory was pasted in.
    """

    #: A concrete placeholder is one ending in a per-request serial number.
    CONCRETE_PII = re.compile(r"PII_[A-Z]+(?:_[A-Z]+)*_\d+")
    CONCRETE_ENTITY = re.compile(r"<ENTITY_[A-Z]+(?:_[A-Z]+)*_\d+>")

    #: Each allowed token, with the reason it is documentary rather than data.
    DOCUMENTARY = {
        "PII_URL_3":      "appears 306x across 18 values; the measurement that "
                          "killed '# === Cron memory:' as a job key",
        "PII_SERVICE_1":  "the one 'Triggered by:' value router/features.py had "
                          "to classify (it is `other`, never `human`)",
        "PII_PROJECT_12": "synthetic probe string in router/demo.py's redaction "
                          "selftest; invented here, not read from the export",
        "<ENTITY_ORG_3>": "synthetic probe string in router/demo.py's redaction "
                          "selftest; invented here, not read from the export",
    }

    @classmethod
    def setUpClass(cls):
        cls.paths = RepoCarriesNoPresignedSignature._repo_files()

    def test_no_undocumented_placeholder_in_any_repo_file(self):
        if self.paths is None:
            raise unittest.SkipTest("git is unavailable; cannot enumerate repo files")
        offenders = {}
        for rel in self.paths:
            path = os.path.join(REPO_ROOT, rel)
            if not os.path.isfile(path) or os.path.getsize(path) > 16 * 1024 * 1024:
                continue
            try:
                blob = _text(path)
            except OSError:
                continue
            hits = set(self.CONCRETE_PII.findall(blob))
            hits |= set(self.CONCRETE_ENTITY.findall(blob))
            unexplained = sorted(hits - set(self.DOCUMENTARY))
            if unexplained:
                offenders[rel] = unexplained
        self.assertEqual(
            offenders, {},
            "undocumented concrete placeholders in repo files -- each is a slice of "
            "a real trajectory unless it is a stated finding. Add it to DOCUMENTARY "
            "with its justification, or remove it. Offenders: %s" % offenders)

    def test_the_allowlist_stays_small(self):
        # The point of an allowlist is that it is short enough to read. If it
        # grows, transcript is being normalised into the repo one exception at a
        # time -- which is exactly the failure mode this file exists to stop.
        self.assertLessEqual(len(self.DOCUMENTARY), 8,
                             "the documentary-placeholder allowlist is growing")
        for token, reason in self.DOCUMENTARY.items():
            self.assertGreater(len(reason), 20,
                               "%s is allowlisted without a real justification" % token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
