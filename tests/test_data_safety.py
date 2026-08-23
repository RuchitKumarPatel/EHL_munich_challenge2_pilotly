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
    4. No tracked file carries a concrete redaction serial that is not on a
       short, justified allowlist. The serial does not have to be attached with
       an underscore -- see RepoSourceCarriesNoExportContent, and
       ConcretePlaceholderShapes, which tests the pattern rather than the repo.
    5. No tracked file carries an opaque-looking identifier that also occurs in
       the export. This is the shape-free check: checks 1 and 4 both match a
       KNOWN placeholder shape, and a raw un-redacted vendor id has no shape.
       See RepoCarriesNoOpaqueExportToken.

WHY THE PII_ RULE IS NOT COSMETIC
    PII_ tokens are renumbered PER REQUEST. `PII_URL_3` in one line and
    `PII_URL_3` in another are different things. So a PII_ string is neither a
    safe identifier to publish nor a valid join key; the brief bans it as a key
    for the second reason and this file enforces the first.

WHAT IT WRITES
    Nothing. It reads results/, the export and the repo's own source files, and
    prints only counts and file paths -- never the matched text.

RUNTIME
    ~13 s: the shingle scan over the largest results/ file, plus one streamed
    pass over the export for the opaque-token check.
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

#: When set to a git ref, the two repo-file checks read that ref's tree instead
#: of the working tree. The loop's push gate uses this to scan each branch it is
#: about to push: it used to scan whatever happened to be checked out and then
#: push every branch, so a branch an earlier turn moved off went out unscanned.
#: The generated-artifact and export-shingle checks are unaffected -- those are
#: working-tree concerns and neither directory is tracked.
SCAN_REF = os.environ.get("DATA_SAFETY_SCAN_REF", "").strip()

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


def _repo_blob(rel, max_bytes):
    """Text of one repo file, from the working tree or from SCAN_REF's tree.

    Returns None for anything that is not a readable regular file, or that is
    larger than max_bytes -- a size guard, not a carve-out: the only files that
    hit it are the inlined figure payloads, which are base64 of our own PNGs.
    """
    if SCAN_REF:
        spec = "%s:%s" % (SCAN_REF, rel)
        size = subprocess.run(["git", "cat-file", "-s", spec], cwd=REPO_ROOT,
                              capture_output=True)
        if size.returncode != 0:
            return None            # a submodule, a symlink target, or a gone path
        try:
            if int(size.stdout.strip()) > max_bytes:
                return None
        except ValueError:
            return None
        blob = subprocess.run(["git", "cat-file", "blob", spec], cwd=REPO_ROOT,
                              capture_output=True)
        if blob.returncode != 0:
            return None
        return blob.stdout.decode("latin-1")
    path = os.path.join(REPO_ROOT, rel)
    if not os.path.isfile(path) or os.path.getsize(path) > max_bytes:
        return None
    try:
        return _text(path)
    except OSError:
        return None


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
        """Tracked files plus untracked-and-not-ignored ones (read-only git calls).

        Under SCAN_REF this is the ref's tree instead. There is no untracked half
        there and that is correct: an unpushed working-tree file cannot reach the
        remote through a ref, and the working-tree scan still covers it.
        """
        paths = []
        if SCAN_REF:
            listings = [["git", "ls-tree", "-r", "--name-only", "-z", SCAN_REF]]
        else:
            listings = [["git", "ls-files", "-z"],
                        ["git", "ls-files", "--others", "--exclude-standard", "-z"]]
        for args in listings:
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
            blob = _repo_blob(rel, 64 * 1024 * 1024)
            if blob is None:
                continue
            if AMZ_SIG_RE.search(blob):
                # This file names the pattern in its own source; that is not a hit.
                # Compared by repo-relative path so it holds under SCAN_REF too,
                # where the scanned file has no path on disk at all.
                if rel == _rel(os.path.abspath(__file__)):
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

    THE SERIAL IS NOT ALWAYS ATTACHED WITH AN UNDERSCORE
        The first version of this pattern required a trailing `_<digits>`, so a
        serial written in any other shorthand walked straight past it.
        loop/JOURNAL.md carried exactly that -- the prefix, a slash, then the
        digits -- from commit 5082319 until c05e078, surviving the turn-1
        cleanup that docs/DATA-SAFETY-DEBT.md records as complete. The value it
        encoded occurs in the export in the thousands.

        So the pattern now accepts up to two non-alphanumeric characters between
        the prefix and the serial, and NORMALISES what it finds back to the
        underscore form before comparing it to DOCUMENTARY. One allowlist entry
        therefore covers every spelling of the same placeholder, which is what
        keeps the allowlist inside its cap of 8.
    """

    #: A concrete placeholder is a prefix, then up to two non-alphanumeric
    #: characters, then a per-request serial. The separator class excludes
    #: WHITESPACE on purpose, and that exclusion was measured rather than
    #: guessed: with whitespace allowed, this very file failed itself, because a
    #: docstring sentence of the form "<prefix>, 3 of them" reads as a serial.
    #: A serial is written ATTACHED to its prefix -- `_12`, `/12`, `-12`, `:12`,
    #: `.12` -- so requiring attachment costs no real coverage and drops the
    #: whole prose false-positive class.
    SERIAL_SEP = r"[^\sA-Za-z0-9]{0,2}"
    CONCRETE_PII = re.compile(r"(PII_[A-Z]+(?:_[A-Z]+)*)" + SERIAL_SEP + r"(\d+)")
    CONCRETE_ENTITY = re.compile(
        r"(<ENTITY_[A-Z]+(?:_[A-Z]+)*)" + SERIAL_SEP + r"(\d+)>?")

    @staticmethod
    def _normalise(match, closing=""):
        """`prefix<any separator>serial` -> the canonical `prefix_serial` form."""
        prefix, serial = match
        return "%s_%s%s" % (prefix, serial, closing)

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
            blob = _repo_blob(rel, 16 * 1024 * 1024)
            if blob is None:
                continue
            hits = {self._normalise(m) for m in self.CONCRETE_PII.findall(blob)}
            hits |= {self._normalise(m, ">") for m in self.CONCRETE_ENTITY.findall(blob)}
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



class ConcretePlaceholderShapes(unittest.TestCase):
    """Unit tests for the placeholder PATTERN, not for the repo's contents.

    RepoSourceCarriesNoExportContent above is a corpus test: it passes when the
    repo happens to be clean, which it also does when the pattern is broken.
    That is how the slash shorthand survived a cleanup that was recorded as
    complete. These tests exercise the pattern directly, so a regression in it
    fails here whatever the repo looks like.

    Every probe is assembled at runtime from a prefix and a serial that are
    written separately in this file, so the file itself never carries a concrete
    placeholder for the corpus test above to find. The prefix is invented: it
    occurs 0 times in the export (measured), which is what makes it a probe
    rather than a sample.
    """

    CLS = RepoSourceCarriesNoExportContent
    PROBE = "PII_PROBE"
    ENTITY_PROBE = "<ENTITY_PROBE"

    #: The shorthands a human actually writes. The slash is the one that reached
    #: main in commit 5082319 and was removed in c05e078.
    ATTACHED_SEPARATORS = ("_", "/", "-", ":", ".", "#", "/-", "::")

    def _pii(self, text):
        return {self.CLS._normalise(m) for m in self.CLS.CONCRETE_PII.findall(text)}

    def _entity(self, text):
        return {self.CLS._normalise(m, ">")
                for m in self.CLS.CONCRETE_ENTITY.findall(text)}

    def test_the_underscore_form_still_matches(self):
        self.assertEqual(self._pii("see %s%s here" % (self.PROBE, "_7")),
                         {self.PROBE + "_7"})

    def test_a_serial_attached_by_any_separator_is_detected(self):
        """The i0011 regression: `_<digits>` was not the only spelling in use."""
        missed = [sep for sep in self.ATTACHED_SEPARATORS
                  if not self._pii("prefix %s%s%s tail" % (self.PROBE, sep, "7"))]
        self.assertEqual(missed, [],
                         "these separators hide a serial from the detector: %s" % missed)

    def test_normalisation_collapses_every_separator_to_one_token(self):
        """One DOCUMENTARY entry must cover every spelling, or the cap of 8 breaks."""
        found = set()
        for sep in self.ATTACHED_SEPARATORS:
            found |= self._pii("%s%s%s" % (self.PROBE, sep, "7"))
        self.assertEqual(found, {self.PROBE + "_7"},
                         "separators normalise to more than one token: %s"
                         % sorted(len(t) for t in found))

    def test_a_pattern_name_without_a_serial_is_not_a_placeholder(self):
        """`PII_URL_N` names the shape; it carries no per-request value."""
        self.assertEqual(self._pii("%s_N and %s_M are renumbered per request"
                                   % (self.PROBE, self.PROBE)), set())

    def test_a_number_separated_by_whitespace_is_prose_not_a_serial(self):
        """Measured carve-out: without it this file failed on its own docstring."""
        for text in ("%s, 3 of them", "%s appears 306 times", "%s and 12 others"):
            self.assertEqual(self._pii(text % self.PROBE), set(), text)

    def test_the_entity_shorthand_matches_and_keeps_its_bracket(self):
        for sep in ("_", "/"):
            self.assertEqual(self._entity("%s%s%s>" % (self.ENTITY_PROBE, sep, "7")),
                             {self.ENTITY_PROBE + "_7>"})

    def test_every_documentary_entry_is_written_in_normalised_form(self):
        """An allowlist entry the detector cannot reproduce would never match."""
        for token in self.CLS.DOCUMENTARY:
            found = self._entity(token) if token.startswith("<") else self._pii(token)
            self.assertEqual(found, {token},
                             "%s does not round-trip through the detector" % token)


class RepoCarriesNoOpaqueExportToken(unittest.TestCase):
    """No opaque-looking identifier in a repo file may occur in the export.

    WHY THIS EXISTS
        Every other placeholder check in this file matches a SHAPE -- `PII_...`
        or `<ENTITY_...>`. A raw, un-redacted identifier copied straight out of
        a trajectory has no shape to match, and that is exactly how three of
        them reached a shared repository on night one. This test needs no shape.
        It pulls every opaque-looking token out of every repo file and asks the
        export whether it knows it. A token the export knows came from the
        export.

    WHAT COUNTS AS OPAQUE -- TWO WINDOWS, AND WHY NOT ONE
        LONG: a maximal run of [A-Za-z0-9_-], 16 to 64 characters, mixing
        letters and digits. That is the shape of a key, a uuid or a session id.

        COMPACT: a maximal run of [A-Za-z0-9] with NO separator, 8 to 15
        characters, mixing letters and digits, minus two structural rejections
        below. The three raw identifiers this class cites are 11 characters, so
        the long window alone matched none of them and the control was inert
        (backlog i0018). `OpaqueDetectorCatchesTheIncidentItWasWrittenFor`
        replays that commit and is the test that keeps this honest.

        Simply lowering the long window to 8 was measured and REJECTED: it takes
        the repo from 27 candidates to 88 and 18 of those occur in the export by
        coincidence -- model ids, feature-column names, dates -- so the corpus
        test would be permanently and uninformatively red. Every one of the 18
        carries a `-` or a `_`. Excluding separators below 16 costs nothing real
        (a raw identifier is not hyphenated) and drops the false positives to 1.

        The two structural rejections in the compact window, both measured:
        `word1024` / `1024word` is a name with a size on it, not an identifier;
        and `<digits>x<digits>` is a pixel dimension -- `1920x1080` appears in
        loop/JOURNAL.md and in the export by pure coincidence, and it was the
        last false positive left. Both are shape rules, so neither can grow the
        way an allowlist can.

        Maximal matters: `data:...;base64,` payloads are stripped first, because
        presentation.html inlines a 577 KB matplotlib PNG whose `+` and `/`
        bytes otherwise shatter it into ~9000 fragments of exactly this shape.
        With the strip and both windows in place the whole repo yields 36
        candidates over 126 files, 0 of which the export knows -- small enough
        to scan the 101 MB export against directly, in ~3 s either way.

    WHAT IS STRUCTURALLY EXEMPT
        The arm identifiers in router.pricing.OBSERVED_ARMS. They are the
        study's unit of analysis, they are pinned in docs/CONTRACTS.md, and they
        are already anonymized per AGENTS.md -- one of them is 17 characters and
        occurs in the export by construction. This tracks the arm table rather
        than being an allowlist, so it cannot silently grow.

    WHAT IT PRINTS ON FAILURE
        The repo path, how many tokens matched, and a MASKED shape (letters ->
        a/A, digits -> #). Never the token. The token is the leak.

    RUNTIME
        ~3 s: one streamed pass over the export per 8 MB chunk.
    """

    OPAQUE_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{16,64}(?![A-Za-z0-9_-])")
    COMPACT_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9]{8,15}(?![A-Za-z0-9_-])")
    #: A name with a size attached, not an identifier.
    WORD_THEN_DIGITS_RE = re.compile(r"\A[A-Za-z]+[0-9]+\Z|\A[0-9]+[A-Za-z]+\Z")
    #: A pixel dimension. The one false positive the compact window otherwise has.
    DIMENSION_RE = re.compile(r"\A[0-9]+[xX][0-9]+\Z")
    DATA_URI_RE = re.compile(r"data:[A-Za-z0-9.+/-]*;base64,[A-Za-z0-9+/=\s]+")
    CHUNK = 8 * 1024 * 1024
    MAX_FILE = 16 * 1024 * 1024

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(EXPORT):
            raise unittest.SkipTest("export/ is not present")
        cls.paths = RepoCarriesNoPresignedSignature._repo_files()
        if cls.paths is None:
            raise unittest.SkipTest("git is unavailable; cannot enumerate repo files")
        cls.exempt = cls._exempt_arm_ids()
        cls.candidates = cls._candidates(cls.paths, cls.exempt)
        cls.known = cls._which_occur_in_export(set(cls.candidates))

    @staticmethod
    def _exempt_arm_ids():
        """The observed arm identifiers, read from the pricing table, not copied."""
        try:
            from router.pricing import OBSERVED_ARMS
        except Exception:                       # pragma: no cover - import guard
            return frozenset()
        return frozenset(OBSERVED_ARMS)

    @staticmethod
    def _mixed(token):
        return (any(c.isdigit() for c in token)
                and any(c.isalpha() for c in token))

    @classmethod
    def _opaque(cls, blob):
        blob = cls.DATA_URI_RE.sub(" ", blob)
        found = {t for t in cls.OPAQUE_RE.findall(blob) if cls._mixed(t)}
        found |= {t for t in cls.COMPACT_RE.findall(blob)
                  if cls._mixed(t)
                  and not cls.WORD_THEN_DIGITS_RE.match(t)
                  and not cls.DIMENSION_RE.match(t)}
        return found

    @classmethod
    def _candidates(cls, paths, exempt):
        """token -> the repo files carrying it.

        Reads through `_repo_blob`, so that under DATA_SAFETY_SCAN_REF this
        reads the REF's blobs. Opening the working-tree path instead would
        silently skip any file the ref has and the checkout does not, and would
        read the wrong bytes for every file they share -- which is the whole
        point of scanning a branch you are not standing on.
        """
        out = {}
        for rel in paths:
            blob = _repo_blob(rel, cls.MAX_FILE)
            if blob is None:
                continue
            for token in cls._opaque(blob) - exempt:
                out.setdefault(token, set()).add(rel)
        return out

    @classmethod
    def _which_occur_in_export(cls, tokens):
        """Stream every export file once; return the subset the export contains."""
        found = set()
        pending = set(tokens)
        if not pending:
            return found
        overlap = 64
        for root, _dirs, names in os.walk(EXPORT):
            for name in sorted(names):
                with open(os.path.join(root, name), "rb") as fh:
                    tail = ""
                    while pending:
                        raw = fh.read(cls.CHUNK)
                        if not raw:
                            break
                        window = tail + raw.decode("latin-1")
                        for token in list(pending):
                            if token in window:
                                found.add(token)
                                pending.discard(token)
                        tail = window[-overlap:]
        return found

    @staticmethod
    def _mask(token):
        """Shape only: letters collapse to a/A, digits to #. Safe to print."""
        return "".join("#" if c.isdigit() else "a" if c.islower()
                       else "A" if c.isupper() else c for c in token)

    def test_the_candidate_set_is_non_trivial(self):
        """A detector that finds nothing to check is not evidence of anything."""
        self.assertGreaterEqual(
            len(self.candidates), 5,
            "only %d opaque tokens extracted from %d repo files; the extractor is "
            "probably broken, not the repo clean" % (len(self.candidates), len(self.paths)))

    def test_the_export_scan_finds_a_token_that_is_really_there(self):
        """Negative control. Without it a scanner that always returns nothing passes."""
        from router.io import iter_lines

        probe = None
        for _idx, req in iter_lines():
            for token in sorted(self._opaque(json.dumps(req))):
                if token not in self.exempt:
                    probe = token
                    break
            if probe:
                break
        if probe is None:
            raise unittest.SkipTest("no opaque token found in the export to probe with")
        self.assertEqual(self._which_occur_in_export({probe}), {probe},
                         "the export scan missed a token taken straight out of the "
                         "export (shape %s)" % self._mask(probe))

    def test_no_repo_file_carries_a_token_the_export_knows(self):
        offenders = {}
        for token in sorted(self.known):
            for rel in sorted(self.candidates[token]):
                offenders.setdefault(rel, []).append(self._mask(token))
        self.assertEqual(
            offenders, {},
            "opaque tokens in repo files that also occur in the export -- shapes "
            "only, never the value. Each is a raw identifier copied out of a "
            "trajectory unless it is an arm id, and arm ids are already exempt. "
            "Offenders: %s" % offenders)


class OpaqueDetectorCatchesTheIncidentItWasWrittenFor(unittest.TestCase):
    """The detector above must catch the tokens whose removal it cites.

    WHY THIS EXISTS
        `RepoCarriesNoOpaqueExportToken` is a corpus test: it is green when the
        repo is clean and equally green when its extractor matches nothing. Its
        docstring cites the three raw identifiers removed in commit af4e78b as
        the incident it prevents -- and it matched none of them, because they
        are 11 characters long and its window started at 16. A control that
        cannot detect its own stated class defeats itself (backlog i0018).

        So this class replays the incident. It reads both sides of that commit
        out of history, takes every token the commit removed, keeps the ones the
        export actually knows, and asserts each is caught by one of this file's
        detectors. No token is written into this file: the fixture is the
        commit itself, and failures print masked shapes only.

    WHICH DETECTOR CATCHES WHICH
        Two of the five removed tokens are placeholder-shaped and belong to
        `RepoSourceCarriesNoExportContent`; three are raw identifiers with no
        shape at all and belong to the opaque detector. The assertion is on the
        UNION of the two, because between them they are the whole control.
    """

    FIXTURE_COMMIT = "af4e78b"
    FIXTURE_PATH = "docs/POSTMORTEM-textclf.md"

    #: Deliberately wider than either detector: this is the ground truth they
    #: are measured against, not a third detector.
    WIDE_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{6,64}(?![A-Za-z0-9_-])")

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(EXPORT):
            raise unittest.SkipTest("the licensed corpus is not present")
        before = cls._at(cls.FIXTURE_COMMIT + "^")
        after = cls._at(cls.FIXTURE_COMMIT)
        if before is None or after is None:
            raise unittest.SkipTest("commit %s is not reachable in this checkout"
                                    % cls.FIXTURE_COMMIT)
        removed = cls._wide(before) - cls._wide(after)
        cls.known = RepoCarriesNoOpaqueExportToken._which_occur_in_export(removed)

    @classmethod
    def _at(cls, rev):
        try:
            out = subprocess.run(["git", "show", "%s:%s" % (rev, cls.FIXTURE_PATH)],
                                 cwd=REPO_ROOT, capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.decode("utf-8", "replace")

    @classmethod
    def _wide(cls, blob):
        return {t for t in cls.WIDE_RE.findall(blob)
                if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)}

    @staticmethod
    def _caught_by_a_detector(token):
        """True if any detector in this file matches the token."""
        placeholder = RepoSourceCarriesNoExportContent
        if placeholder.CONCRETE_PII.findall(token):
            return True
        if placeholder.CONCRETE_ENTITY.findall(token):
            return True
        return token in RepoCarriesNoOpaqueExportToken._opaque(token)

    def test_the_fixture_commit_really_removed_corpus_content(self):
        """Control. Without it the assertion below is vacuous on an empty set."""
        self.assertGreaterEqual(
            len(self.known), 3,
            "commit %s removed only %d token(s) the corpus knows; the fixture no "
            "longer reproduces the incident and this class proves nothing"
            % (self.FIXTURE_COMMIT, len(self.known)))

    def test_every_removed_token_the_corpus_knows_is_caught(self):
        missed = sorted(RepoCarriesNoOpaqueExportToken._mask(t)
                        for t in self.known if not self._caught_by_a_detector(t))
        self.assertEqual(
            missed, [],
            "these tokens were removed from %s BECAUSE they are corpus content, "
            "and no detector in this file matches them -- shapes only, never the "
            "value: %s" % (self.FIXTURE_PATH, missed))


class CompactOpaqueTokenShapes(unittest.TestCase):
    """Unit tests for the compact half of the opaque pattern, not for the repo.

    Same reasoning as `ConcretePlaceholderShapes`: a corpus test is green when
    the extractor is blind, so the extractor needs tests of its own. Every probe
    here is synthesised. The two that are real strings -- a screen resolution
    and an embedding width -- are structural shapes, not corpus content.
    """

    CLS = RepoCarriesNoOpaqueExportToken

    def _opaque(self, text):
        return self.CLS._opaque(text)

    def test_an_eleven_character_identifier_is_caught(self):
        """The i0018 regression: the 16-char floor missed exactly this length."""
        for probe in ("Q7A4B2CD3EF", "R28XYZ4LMN5", "T591M8KK7PQ"):
            self.assertEqual(self._opaque("id %s here" % probe), {probe}, probe)

    def test_the_floor_is_eight_characters(self):
        self.assertEqual(self._opaque("A1B2C3D"), set())          # 7 -- below
        self.assertEqual(self._opaque("A1B2C3D4"), {"A1B2C3D4"})  # 8 -- at

    def test_the_long_window_still_takes_separator_bearing_tokens(self):
        probe = "sess-4f2a-11ee-9c0b-2b7d3a"
        self.assertEqual(self._opaque("path/%s/x" % probe), {probe})

    def test_a_compact_token_carrying_a_separator_is_not_compact(self):
        """8..15 is alnum-only on purpose. Allowing `-` and `_` down there costs
        18 false positives against the corpus (measured on this tree): model
        ids, field names and dates that collide by coincidence. The
        separator-bearing window therefore still starts at 16.
        """
        self.assertEqual(self._opaque("claude-5-a1"), set())

    def test_a_word_with_an_attached_serial_is_a_name_not_an_identifier(self):
        for probe in ("modernbert1024", "embeddings768", "1024channels"):
            self.assertEqual(self._opaque(probe), set(), probe)

    def test_the_rejection_is_only_for_a_single_word_digit_boundary(self):
        """`L1024tokens` alternates and stays a candidate. The rejection above is
        deliberately narrow: widening it to any letter/digit mixture would throw
        away the very shape the compact window exists to catch.
        """
        self.assertEqual(self._opaque("L1024tokens"), {"L1024tokens"})

    def test_a_pixel_dimension_is_not_an_identifier(self):
        """Measured carve-out: `1920x1080` sits in loop/JOURNAL.md AND in the
        corpus, by pure coincidence. Narrow, documented, and it cannot grow.
        """
        self.assertEqual(self._opaque("rendered at 1920x1080 ok"), set())

    def test_a_base64_payload_still_contributes_nothing(self):
        blob = "data:image/png;base64," + ("iVBORw0KGgoAAAANSUhEUg" * 40)
        self.assertEqual(self._opaque(blob), set())

    def test_letters_only_and_digits_only_are_never_opaque(self):
        self.assertEqual(self._opaque("abcdefghijkl 123456789012"), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
