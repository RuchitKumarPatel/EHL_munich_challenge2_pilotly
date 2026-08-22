"""Test package marker.

Present only because `unittest discover -s tests -t .` (what the Makefile's
`test:` target runs, since pytest is not installed in the venv) refuses a start
directory without an `__init__.py`. The file is empty of code on purpose: it
must not import anything, so a broken module in `router/` fails inside the test
that touches it rather than at collection time.

The tests themselves are read-only over `results/` and `export/`. They write no
file, run no git command and make no network call.
"""
