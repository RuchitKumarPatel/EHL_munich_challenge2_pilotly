"""Where the scratch inputs live, and where the repo lives, resolved at run time.

Every module in this directory used to open with two absolute paths baked in: one
laptop's repo root, and the scratchpad of the single session that produced the
negative result in docs/POSTMORTEM-textclf.md. Neither survives being read by
anyone else, and the second one stops resolving on the original machine at the
next reboot -- so the reproduce command in README.md was documented but not
runnable.

The repo root is now derived from this file. The scratch directory is named by
the TEXTCLF_SCRATCH environment variable, and its absence is a loud stop rather
than a FileNotFoundError against a path the reader cannot interpret.

The scratch inputs are derived from the licensed export and so cannot be
committed; README.md lists what the directory has to contain.
"""

from __future__ import annotations

import os
import sys

ENV = "TEXTCLF_SCRATCH"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HELP = """
router/textclf needs a scratch directory and %(env)s is %(what)s.

    export %(env)s=/path/to/scratchpad
    %(env)s=... .venv/bin/python router/textclf/<module>.py

It holds the inputs that are derived from the licensed export and therefore are
NOT in this repository. router/textclf/README.md lists every file it must
contain. These modules read that directory; they do not build it.
""".strip()


def scratch_dir():
    """The configured scratch directory, or a stop that says what to set.

    Called before the transformers import in every entry point: a ten-second
    model import in front of a configuration error is how a reader gives up.
    """
    sp = os.environ.get(ENV, "").strip()
    if not sp:
        raise SystemExit(_HELP % {"env": ENV, "what": "not set"})
    if not os.path.isdir(sp):
        raise SystemExit(
            (_HELP % {"env": ENV, "what": "set to %r, which is not a directory" % sp}))
    return os.path.abspath(sp)


def on_path(*extra):
    """Put the repo root and the scratch subdirectories on sys.path, APPENDED.

    Appended and never inserted, on purpose. The scratch directory holds older
    copies of this directory's own file names -- among them the pre-fix
    finetune.py whose degenerate arm handling is the trap README.md ends on. An
    insert at position 0 makes those stale siblings shadow the shipped modules;
    an append leaves the script's own directory, which Python puts first, in
    charge of every name the repo actually ships.
    """
    for p in (REPO_ROOT,) + tuple(extra):
        if p not in sys.path:
            sys.path.append(p)
